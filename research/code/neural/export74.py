"""Note 74: ONNX export of a tcn69 function net (siba / sibm / no-sibling; TCN backbone) for the CPU package.

One graph per call = one signal x one window piece (sibling context needs every detector of the signal at once):
  inputs   xp   [P, C, T] float32   the KEPT (detector, candidate) pairs, 15 ad_all channels (tcn53.assemble53 order)
           pd   [P] int64           detector index of each pair (0 .. D-1)
           pk   [P] int64           candidate index of each pair (0 .. K-1)
           act  [D] float32         1 if the detector actuated in the piece (nact > 0), else 0
           dk   [D, K] float32      zeros; only its shape is used (D detectors, K candidates)
  outputs  flogp [D, 7] function log-probabilities (C7 order: Advance, Presence, Count, Yellow_Red, Mid, Bike, Other)
           logit [D, K] phase logits (-1e4 for pairs not supplied)
Pairs left out (candidate filter: tree probability < .01) behave exactly as in `tcn69_func.forward69` with a keep mask.
The research pipeline averages flogp over pieces (K = 4 past 2 h) and renormalises (tcn69_func.cmd_infer).

    python export74.py --ckpt PATH.pt --out PATH.onnx [--check]    (--check: parity vs forward69 on real batches, CPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcn69_func as T  # noqa: E402


def sib_attn1(sa: T.SibAttn, Z, w):
    """T.SibAttn.forward for B = 1 (Z [D,K,E], w [D,K]); the diagonal mask is built from arange (ONNX Runtime has no
    bool EyeLike kernel). Same math, same parameters."""
    D, K, E = Z.shape
    X = Z.permute(1, 0, 2)                                                   # [K,D,E]
    W = w.permute(1, 0)                                                      # [K,D]
    q = sa.q(X).view(K, D, sa.h, sa.dh).transpose(1, 2)
    k = sa.k(X).view(K, D, sa.h, sa.dh).transpose(1, 2)
    v = sa.v(X).view(K, D, sa.h, sa.dh).transpose(1, 2)
    s = q @ k.transpose(-1, -2) / (sa.dh ** 0.5)                             # [K,h,D,D]
    bias = torch.log(W.clamp(min=1e-6))[:, None, None, :] + torch.zeros_like(s)
    bias = bias.masked_fill((W <= 0)[:, None, None, :], -1e4)
    ar = torch.arange(D)
    bias = bias.masked_fill((ar[:, None] == ar[None, :])[None, None], -1e4)
    sn = (q * sa.nk[None, :, None, :]).sum(-1, keepdim=True) / (sa.dh ** 0.5)
    a = torch.softmax(torch.cat([s + bias, sn], -1), -1)[..., :D]
    o = (a @ v).transpose(1, 2).reshape(K, D, E)
    return sa.o(o).permute(1, 0, 2)


class Sib1(nn.Module):
    """forward69 for one sample (B = 1) on an explicit pair list; same math, no autocast."""

    def __init__(self, net: T.Net69):
        super().__init__()
        self.net = net
        self.sib = net.cfg["sib"]

    def forward(self, xp, pd, pk, act, dk):
        D, K = dk.shape[0], dk.shape[1]
        s, z = self.net.pair(xp)
        E = z.shape[-1]
        logit = torch.full_like(dk, -1e4).index_put((pd, pk), s.float())
        Z = torch.zeros(D, K, E, dtype=z.dtype).index_put((pd, pk), z.float())
        p = torch.softmax(logit, dim=1)
        valid = (logit > -1e3)
        zmax = Z.masked_fill(~valid[..., None], -1e4).amax(1)
        parts = [(p[..., None] * Z).sum(1), zmax]
        if self.sib != "none":
            w = p * valid.float() * act[:, None]                        # [D,K]
            ns = (w.sum(0, keepdim=True) - w).clamp(min=0)
            if self.sib == "mean":
                S = torch.einsum("dk,dke->ke", w, Z)[None]
                Sm = (S - w[..., None] * Z) / (ns[..., None] + 1e-3)
            else:
                Sm = sib_attn1(self.net.sib, Z, w)
            parts += [(p[..., None] * Sm).sum(1), torch.log1p((p * ns).sum(1, keepdim=True))]
        flog = self.net.fhead(torch.cat(parts, -1))
        return torch.log_softmax(flog, dim=-1), logit


def load(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = T.Net69(ck["cfg"]).eval()
    net.load_state_dict(ck["state"])
    assert ck["cfg"].get("arch", "tcn") == "tcn", "GRU backbone: export not covered here"
    return net, ck


def export(net, out):
    m = Sib1(net).eval()
    D, K, C, Tn = 5, 3, len(net.cfg["chans"]), 600
    pd = torch.arange(D).repeat_interleave(K)
    pk = torch.arange(K).repeat(D)
    ex = (torch.randn(D * K, C, Tn), pd, pk, torch.ones(D), torch.zeros(D, K))
    kw = dict(input_names=["xp", "pd", "pk", "act", "dk"], output_names=["flogp", "logit"], opset_version=17,
              dynamic_axes={"xp": {0: "P", 2: "T"}, "pd": {0: "P"}, "pk": {0: "P"}, "act": {0: "D"},
                            "dk": {0: "D", 1: "K"}, "flogp": {0: "D"}, "logit": {0: "D", 1: "K"}})
    with torch.no_grad():
        try:
            torch.onnx.export(m, ex, str(out), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(m, ex, str(out), **kw)
    return m


@torch.no_grad()
def check(net, m, out, n_batches=6):
    """parity on real fold-0 held-out pieces (CPU): forward69 (B = 1, optional keep mask) vs Sib1 torch vs ONNX."""
    import onnxruntime as ort
    from torch.utils.data import DataLoader
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    table, sig = T.F53.func_table(False)
    f0 = set((sig.period + "|" + sig.DeviceId)[sig.fold == 0])
    keys = sorted(k for k in table if k in f0 and table[k]["period"] == "stg")[:n_batches]
    start, secs = next((s, n) for w, s, n in T.I2.WINDOWS["stg"] if w == "m30_a")
    pcs = T.I2.pieces("stg", start, secs, 32)
    plan = [(k, pcs[0][0], (pcs[0][1] - pcs[0][0]) // 1000, "m30_a") for k in keys]
    dl = DataLoader(T.F53.DSF(table, plan, net.cfg["bw"]), batch_size=1, collate_fn=T.F53.collateF)
    res = {"max_abs_flogp_torch": 0.0, "max_abs_flogp_onnx": 0.0, "max_abs_logit_onnx": 0.0, "pieces": 0}
    rng = np.random.default_rng(0)
    for i, b in enumerate(dl):
        if i % 2:                                    # every other piece: random candidate filter (keep >= 1 per det)
            D, K = b["dmask"].shape[1], int(b["ncand"].max())
            keep = torch.from_numpy(rng.random((1, D, K)) < 0.5)
            keep[0, torch.arange(D), torch.from_numpy(rng.integers(0, K, D))] = True
            b["keep"] = keep
        logit0, flog0 = T.forward69(net, b, "cpu")
        bi, di, ki, B, D, K = T.T2.pair_index(b["dmask"], b["ncand"])
        if "keep" in b:
            mm = b["keep"][bi, di, ki]
            bi, di, ki = bi[mm], di[mm], ki[mm]
        xp = T.M.assemble53(b, bi, di, ki, net.cfg["chans"])
        act = (b["nact"][0] > 0).float()
        f1, l1 = m(xp, di, ki, act, torch.zeros(D, K))
        f2, l2 = sess.run(None, {"xp": xp.numpy(), "pd": di.numpy(), "pk": ki.numpy(), "act": act.numpy(),
                                 "dk": np.zeros((D, K), np.float32)})
        ref = torch.log_softmax(flog0[0], -1).numpy()
        res["max_abs_flogp_torch"] = max(res["max_abs_flogp_torch"], float(np.abs(f1.numpy() - ref).max()))
        res["max_abs_flogp_onnx"] = max(res["max_abs_flogp_onnx"], float(np.abs(f2 - ref).max()))
        ok = logit0[0].numpy() > -1e3
        res["max_abs_logit_onnx"] = max(res["max_abs_logit_onnx"], float(np.abs(l2[ok] - logit0[0].numpy()[ok]).max()))
        res["pieces"] += 1
    res["pass_1e-4"] = res["max_abs_flogp_onnx"] < 1e-4 and res["max_abs_logit_onnx"] < 1e-4
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    net, ck = load(a.ckpt)
    m = export(net, a.out)
    print(f"exported {a.out} ({Path(a.out).stat().st_size / 2**20:.1f} MB); cfg sib {net.cfg['sib']}, epoch {ck['epoch']}")
    if a.check:
        r = check(net, m, a.out)
        print(json.dumps(r))
        json.dump(r, open(Path(a.out).with_suffix(".parity.json"), "w"))


if __name__ == "__main__":
    main()
