"""Track B, B5: sibling-detector context INSIDE the network.

The TCN pair scorer is unchanged up to its 128-d pair embedding z(d, c) and its own
logit l1(d, c).  B5 adds one permutation-equivariant set-attention layer across the
detectors of the same signal-window: for every candidate c, pair (d, c) attends over the
pairs (d', c) of every OTHER detector d' of that signal-window, with an attention bias
learned from the two detectors' trace similarity (zero-lag and +-10 s lagged
cross-correlation of occupancy, 30-s count correlation, relative activity).  A small MLP
turns [z, context] into a residual delta(d, c); the phase score is l1 + delta, softmaxed
over the signal's candidates as before.  The delta head is zero-initialised, so the
starting point is exactly the plain TCN.

Phase-anonymous: the layer never sees a phase or channel number -- detectors are an
unordered set (attention is equivariant over d and d'), candidates an unordered set
(the same weights for every c), and siblings are identified by behaviour only.

Two variants, fixed before launch:
  b5a  embedding attention only (keys/values from z(d', c) + similarity features);
  b5b  b5a + the siblings' first-pass candidate distribution p1(d', .) at c (p, log p,
       is-argmax) in keys and values, own p1 in the head, plus an auxiliary 0.5 x phase
       loss on l1 so the first pass stays a proper scorer ("the joint decoder inside the
       network").

`forward_batch` / `phase_loss` have the `train2` signatures, so `trackb_train.run` and
`infer2.score_windows` drive this model unchanged once patched (see b5_train / b5_infer).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from neural import train2 as T2  # noqa: E402
from neural.raster import assemble_flat  # noqa: E402
from neural.trackb_models import PairNetB  # noqa: E402

NSIM = 6
MAXLAG = 5          # 2-s bins -> +-10 s
AUX_W = 0.5


def sim_features(det: torch.Tensor, nact: torch.Tensor) -> torch.Tensor:
    """det [B,D,2,T] (occupancy, onsets), nact [B,D] -> S [B,D,D,NSIM] float32.

    S[b,d,j] describes sibling j as seen from d: zero-lag occupancy correlation, max
    correlation with j LAGGING d by 2..10 s, with j LEADING d by 2..10 s, 30-s onset-count
    correlation, relative activity log1p(n_j) - log1p(n_d) (/3), j active."""
    B, D, _, T = det.shape
    occ = det[:, :, 0, :].float().reshape(B * D, 1, T)
    a = F.avg_pool1d(occ, 2).reshape(B, D, -1)
    Tb = a.shape[-1]
    a = a - a.mean(-1, keepdim=True)
    a = a / (a.pow(2).mean(-1, keepdim=True).sqrt() + 1e-3)
    cs = []
    for lag in range(-MAXLAG, MAXLAG + 1):
        if lag >= 0:
            c = torch.matmul(a[:, :, :Tb - lag], a[:, :, lag:].transpose(1, 2))
        else:
            c = torch.matmul(a[:, :, -lag:], a[:, :, :Tb + lag].transpose(1, 2))
        cs.append(c / Tb)
    C = torch.stack(cs, -1)                                     # [B,D,D,2L+1]
    c0 = C[..., MAXLAG]
    lagj = C[..., MAXLAG + 1:].amax(-1)                         # j after d
    leadj = C[..., :MAXLAG].amax(-1)                            # j before d
    on = det[:, :, 1, :].float().reshape(B * D, 1, T)
    o = F.avg_pool1d(on, 30).reshape(B, D, -1)
    o = o - o.mean(-1, keepdim=True)
    o = o / (o.pow(2).mean(-1, keepdim=True).sqrt() + 1e-3)
    co = torch.matmul(o, o.transpose(1, 2)) / o.shape[-1]
    ln = torch.log1p(nact.float())
    rel = (ln[:, None, :] - ln[:, :, None]) / 3.0
    actj = (nact > 0).float()[:, None, :].expand(B, D, D)
    return torch.stack([c0, lagj, leadj, co, rel, actj], -1)


class SiblingLayer(nn.Module):
    def __init__(self, e: int = 128, heads: int = 4, use_p1: bool = False):
        super().__init__()
        self.h, self.dh, self.use_p1 = heads, e // heads, use_p1
        px = 3 if use_p1 else 0
        self.q = nn.Linear(e + px, e)
        self.k = nn.Linear(e + px, e)
        self.v = nn.Linear(e + px + NSIM, e)
        self.bias = nn.Sequential(nn.Linear(NSIM, 32), nn.GELU(), nn.Linear(32, heads))
        self.null_logit = nn.Parameter(torch.zeros(heads))
        self.null_v = nn.Parameter(torch.zeros(heads, e // heads))
        self.out = nn.Sequential(nn.LayerNorm(2 * e + px), nn.Linear(2 * e + px, e),
                                 nn.GELU(), nn.Linear(e, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, Z, l1, valid, S):
        """Z [B,D,K,E], l1 [B,D,K] (masked -1e4), valid [B,D,K] bool, S [B,D,D,NSIM].
        -> delta [B,D,K]."""
        B, D, K, E = Z.shape
        H, dh = self.h, self.dh
        if self.use_p1:
            p1 = torch.softmax(l1.float(), dim=2)
            px = torch.stack([p1, torch.log(p1.clamp_min(1e-6)) / 7.0,
                              (l1 >= l1.amax(2, keepdim=True)).float()], -1)
            px = px * valid[..., None]
            Zx = torch.cat([Z, px.to(Z.dtype)], -1)
        else:
            Zx = Z
        q = self.q(Zx).view(B, D, K, H, dh)
        k = self.k(Zx).view(B, D, K, H, dh)
        # value of sibling j seen from d depends on the pair (d, j) through S
        vz = self.v.weight[:, :Zx.shape[-1]]
        vs = self.v.weight[:, Zx.shape[-1]:]
        v_j = F.linear(Zx, vz, self.v.bias).view(B, 1, D, K, H, dh)          # by j
        v_s = F.linear(S.to(Zx.dtype), vs).view(B, D, D, 1, H, dh)          # by (d, j)
        att = torch.einsum("bdkhe,bjkhe->bkhdj", q, k) / math.sqrt(dh)       # [B,K,H,D,J]
        att = att.float() + self.bias(S.to(Zx.dtype)).float().permute(0, 3, 1, 2)[:, None]
        ar = torch.arange(D, device=Z.device)
        eye = ar[:, None] == ar[None, :]                                  # ONNX-friendly
        jm = valid.permute(0, 2, 1)[:, :, None, None, :]                    # [B,K,1,1,J]
        mask = jm & ~eye[None, None, None]
        att = att.masked_fill(~mask, -1e4)
        nl = self.null_logit.float().view(1, 1, H, 1, 1).expand(B, K, H, D, 1)
        w = torch.softmax(torch.cat([att, nl], -1), -1)                     # [B,K,H,D,J+1]
        wj = w[..., :D].to(Zx.dtype)
        ctx = torch.einsum("bkhdj,bjkhe->bdkhe", wj, v_j[:, 0]) \
            + torch.einsum("bkhdj,bdjhe->bdkhe", wj, v_s[:, :, :, 0]) \
            + w[..., D:].to(Zx.dtype).permute(0, 3, 1, 2, 4) * self.null_v.to(Zx.dtype)
        ctx = ctx.reshape(B, D, K, E)
        feats = [Z, ctx] + ([px.to(Z.dtype)] if self.use_p1 else [])
        return self.out(torch.cat(feats, -1)).squeeze(-1)


class SibNet(nn.Module):
    """TCN pair scorer + sibling set-attention.  `forward_grid` is the whole model."""

    def __init__(self, arch: str = "tcn", variant: str = "b5a", dropout: float = 0.0):
        super().__init__()
        self.arch, self.variant = arch, variant
        self.pair = PairNetB(arch, dropout=dropout)
        self.sib = SiblingLayer(128, 4, use_p1=(variant == "b5b"))
        self.aux_logit = None

    def forward_grid(self, x, bi, di, ki, B, D, K, det, nact, chunk: int = 0):
        if chunk and x.shape[0] > chunk:
            outs = [self.pair(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)]
            l1 = torch.cat([o[0] for o in outs])
            z = torch.cat([o[2] for o in outs])
        else:
            l1, _, z = self.pair(x)
        E = z.shape[-1]
        Z = torch.zeros(B, D, K, E, device=x.device, dtype=z.dtype)
        Z[bi, di, ki] = z
        L1 = torch.full((B, D, K), -1e4, device=x.device, dtype=torch.float32)
        L1[bi, di, ki] = l1.float()
        valid = torch.zeros(B, D, K, dtype=torch.bool, device=x.device)
        valid[bi, di, ki] = True
        with torch.autocast("cuda", enabled=False):
            S = sim_features(det, nact)
        delta = self.sib(Z, L1, valid, S).float()
        logit = torch.where(valid, L1 + delta, torch.full_like(L1, -1e4))
        self.aux_logit = L1 if self.variant == "b5b" else None
        return logit


def forward_batch(model, batch, dev, chunk: int = 0, chdrop: float = 0.0):
    """`train2.forward_batch` signature; `chdrop` accepted and ignored (B3 not kept)."""
    det = batch["det"].to(dev, non_blocking=True)
    ph = batch["ph"].to(dev, non_blocking=True)
    sig = batch["sig"].to(dev, non_blocking=True)
    nact = det[:, :, 1, :].float().sum(-1) * 2.0     # as the export sees it (clipped)
    ncand = batch["ncand"].to(dev)
    dmask = batch["dmask"].to(dev)
    bi, di, ki, B, D, K = T2.pair_index(dmask, ncand)
    x = assemble_flat(det, ph, sig, ncand, bi, di, ki)
    return model.forward_grid(x, bi, di, ki, B, D, K, det, nact, chunk=chunk)


_phase_loss = T2.phase_loss
_CURRENT: dict = {}          # b5_train registers the model here for the auxiliary loss


def phase_loss(logit, batch, dev):
    loss = _phase_loss(logit, batch, dev)
    m = _CURRENT.get("model")
    if loss is not None and m is not None and m.training and m.aux_logit is not None:
        aux = _phase_loss(m.aux_logit, batch, dev)
        if aux is not None:
            loss = loss + AUX_W * aux
    return loss


class SibExport(nn.Module):
    """ONNX-shaped wrapper: ONE signal-window at a time.
    x [D,K,9,T] (the ordinary pair rasters, every detector x every candidate) -> score
    [D,K] logits.  The detector block is read back from channels 0-1 of x."""

    def __init__(self, net: SibNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        D, K, C, T = x.shape
        xf = x.reshape(D * K, C, T)
        l1, _, z = self.net.pair(xf)
        E = z.shape[-1]
        Z = z.reshape(1, D, K, E)
        L1 = l1.reshape(1, D, K)
        det = x[:, 0, 0:2, :].reshape(1, D, 2, T)
        nact = (det[:, :, 1, :] * 2.0).sum(-1)
        valid = torch.ones(1, D, K, dtype=torch.bool, device=x.device)
        S = sim_features(det, nact)
        return (L1 + self.net.sib(Z, L1, valid, S)).reshape(D, K)
