"""Track B, B7: a pure-PyTorch state-space backbone (bidirectional S4D) for the pair scorer.

`mamba-ssm` has no Windows / Python 3.13 wheel and this machine has neither nvcc nor MSVC,
so the selective-scan CUDA kernel cannot be built.  This is the diagonal state-space layer of
Gu et al. (S4D-Lin initialisation), computed as a long convolution via FFT -- no custom
kernel, exportable-in-principle, and fast on the GPU.  Not input-selective (that is Mamba's
addition); it tests the "long linear recurrence" half of the idea.

Shape contract identical to `neural.models.TCN`: x [N,9,T] (1 s raster) -> [N, 3c].
    stem   conv k7 s2 + BN + GELU + maxpool 2      (4 s steps, as the TCN)
    local  one residual conv block (k5)             (pulse shapes, sub-cycle detail)
    nblk x [ LayerNorm -> bidirectional S4D -> GELU -> GLU(Linear c->2c) ] + residual
    pool   AttnPool (attention + mean + max)

Importing this module registers "s4d" in `neural.models.BACKBONES` (so `infer2.load_model`
and `PairNet` build it) and in `neural.trackb_models.BACKBONES_B` (so `PairNetB` trains it).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from neural import models as M
from neural import trackb_models as TM


class S4DKernel(nn.Module):
    """Per-channel diagonal SSM kernels K[h, l], two directions, S4D-Lin init."""

    def __init__(self, h: int, n: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        n2 = n // 2
        log_dt = torch.rand(2, h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)                                  # [2,h]
        self.log_a_re = nn.Parameter(torch.log(0.5 * torch.ones(2, h, n2)))
        self.a_im = nn.Parameter(math.pi * torch.arange(n2).float().expand(2, h, n2).clone())
        self.c = nn.Parameter(torch.randn(2, h, n2, 2) * (0.5 ** 0.5))    # complex C

    def forward(self, L: int) -> torch.Tensor:                             # -> [2,h,L] fp32
        dt = self.log_dt.float().exp().unsqueeze(-1)                       # [2,h,1]
        a = torch.complex(-self.log_a_re.float().exp(), self.a_im.float())  # [2,h,n2]
        dta = a * dt
        c = torch.view_as_complex(self.c.float().contiguous()) * (torch.exp(dta) - 1.0) / a
        pos = torch.arange(L, device=a.device, dtype=torch.float32)
        vand = torch.exp(dta.unsqueeze(-1) * pos)                          # [2,h,n2,L]
        return 2.0 * torch.einsum("dhn,dhnl->dhl", c, vand).real


class S4DLayer(nn.Module):
    """Bidirectional S4D: y = causal(K_f) * u + anticausal(K_b) * u + D u, one FFT pair."""

    def __init__(self, h: int, n: int = 64, dropout: float = 0.0):
        super().__init__()
        self.kernel = S4DKernel(h, n)
        self.d = nn.Parameter(torch.randn(h))
        self.norm = nn.LayerNorm(h)
        self.out = nn.Linear(h, 2 * h)
        self.drop = nn.Dropout(dropout)

    def _mix(self, x: torch.Tensor) -> torch.Tensor:                       # x [N,L,h]
        L = x.shape[1]
        u = self.norm(x).float().transpose(1, 2)                           # [N,h,L]
        with torch.autocast("cuda", enabled=False):
            k = self.kernel(L)                                             # [2,h,L]
            kf = torch.fft.rfft(k, n=2 * L)                                # [2,h,L+1]
            uf = torch.fft.rfft(u, n=2 * L)                                # [N,h,L+1]
            # causal conv with K_f; anticausal conv with K_b = correlation = conj spectrum
            y = torch.fft.irfft(uf * (kf[0] + kf[1].conj()), n=2 * L)[..., :L]
            y = y - k[1, :, :1] * u                                        # lag 0 counted twice
            y = y + self.d.float().unsqueeze(-1) * u
        y = self.drop(F.gelu(y.transpose(1, 2)))                           # [N,L,h]
        return x + F.glu(self.out(y.to(x.dtype)), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.requires_grad:
            return checkpoint(self._mix, x, use_reentrant=False)
        return self._mix(x)


class S4DNet(nn.Module):
    """Pair-raster backbone: TCN stem + one local conv block + nblk bidirectional S4D layers."""

    def __init__(self, cin: int = 9, c: int = 128, nblk: int = 4, n: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, c, 7, stride=2, padding=3),
                                  nn.BatchNorm1d(c), nn.GELU(), nn.MaxPool1d(2))
        self.local = TM.TCNBlockD(c, 5, 1, dropout)
        self.layers = nn.ModuleList([S4DLayer(c, n, dropout) for _ in range(nblk)])
        self.norm = nn.LayerNorm(c)
        self.pool = M.AttnPool(c)
        self.out_dim = self.pool.out_dim

    def forward(self, x):                                                  # x [N,9,T]
        h = self.local(self.stem(x)).transpose(1, 2)                       # [N,L,c]
        for layer in self.layers:
            h = layer(h)
        return self.pool(self.norm(h))


M.BACKBONES.setdefault("s4d", S4DNet)
TM.BACKBONES_B.setdefault("s4d", S4DNet)


if __name__ == "__main__":                                                 # smoke test
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = TM.PairNetB("s4d").to(dev)
    print("params", TM.n_params(net))
    x = torch.randn(512, 9, 1800, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        s, _, _ = net(x)
    s.float().sum().backward()
    print("ok", s.shape, torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else "")
    # the anticausal part: a unit impulse must spread both ways
    net.eval()
    lay = net.backbone.layers[0]
    u = torch.zeros(1, 50, 128, device=dev)
    u[0, 25] = 1.0
    with torch.no_grad():
        y = lay(u) - u
    print("left", float(y[0, :25].abs().sum()), "right", float(y[0, 26:].abs().sum()))
    pn = M.PairNet("s4d").to(dev)
    pn.load_state_dict(net.state_dict())
    print("PairNet load ok")
