"""Note 83: the production function network for candidate v4 = the FULL-DATA siba trained on v4l
(`%DC_WORK%/tcn53/models/x74_sibafull4l_full.pt`, q74 FULL job, snap74b code = research neural/ code, identical files).

Exported as the package's two graphs (export75.build_graphs), with ONE change to the head: the per-detector max over
candidates of the pair embedding ignores candidates the pair net did not run (logit <= -1e3), exactly as
`tcn69_func.forward69` with a keep mask.  With every candidate run (the default) the head is the same function as
note 75's.  This makes the optional siba candidate filter (tree p >= .01, note 74; OFF by default) exact.

    python export83.py export          -> package weights/funcnet/{tag}_pair.onnx, {tag}_head.onnx, manifest.json
    python export83.py parity          -> torch forward69 (fp32, CPU) vs the package FuncNet on the bench extracts
                                          (typical / busiest x 30 min / 3 h), every piece: unfiltered AND with a
                                          candidate filter -> %DC_WORK%/final_v3_work/v3fit83/siba_parity.json
CPU only (CUDA hidden), 4 threads.  locked_v2 asserted absent from every signal used.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PKG = W / "final_v3_candidate_v4"
CKPT = W / "tcn53" / "models" / "x74_sibafull4l_full.pt"
TAG = "x74_sibafull4l"
OUT = W / "final_v3_work" / "v3fit83"
B71 = W / "bench71"
THREADS = 4


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())


def build_net():
    import torch
    sys.path.insert(0, str(CODE))
    import rpath  # noqa: F401
    from neural import tcn69_func as T69
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert cfg["sib"] == "attn" and cfg["fstem"] == 1 and cfg["bw"] == 1000 and cfg.get("arch", "tcn") == "tcn", cfg
    net = T69.Net69(cfg).eval()
    net.load_state_dict(ck["state"])
    return net, cfg, T69


def cmd_export():
    import math
    import torch
    torch.set_num_threads(THREADS)
    net, cfg, T69 = build_net()

    class PairG(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            s, z = self.n.pair(x)
            return s, z

    class HeadG(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def sib(self, Z, w):
            sa = self.n.sib
            D, K, E = Z.shape
            X = Z.permute(1, 0, 2)
            Wt = w.permute(1, 0)
            q = sa.q(X).reshape(K, D, sa.h, sa.dh).transpose(1, 2)
            k = sa.k(X).reshape(K, D, sa.h, sa.dh).transpose(1, 2)
            v = sa.v(X).reshape(K, D, sa.h, sa.dh).transpose(1, 2)
            s = q @ k.transpose(-1, -2) / math.sqrt(sa.dh)
            bias = torch.log(Wt.clamp(min=1e-6))[:, None, None, :] + torch.zeros_like(s)
            bias = bias.masked_fill((Wt <= 0)[:, None, None, :], -1e4)
            idx = torch.arange(D, device=Z.device)
            eye = idx[:, None] == idx[None, :]
            bias = bias.masked_fill(eye[None, None], -1e4)
            sn = (q * sa.nk[None, :, None, :]).sum(-1, keepdim=True) / math.sqrt(sa.dh)
            a = torch.softmax(torch.cat([s + bias, sn], -1), -1)[..., :D]
            o = (a @ v).transpose(1, 2).reshape(K, D, E)
            return sa.o(o).permute(1, 0, 2)

        def forward(self, logit, Z, act):                     # [D,K], [D,K,E] (0 where not run), [D]
            p = torch.softmax(logit, dim=1)
            valid = logit > -1e3                               # note 83: = forward69's keep mask
            zmax = Z.masked_fill(~valid[..., None], -1e4).amax(1)
            w = p * valid.float() * act[:, None]
            Wk = w.sum(0, keepdim=True)
            ns = (Wk - w).clamp(min=0)
            Sm = self.sib(Z, w)
            parts = [(p[..., None] * Z).sum(1), zmax, (p[..., None] * Sm).sum(1),
                     torch.log1p((p * ns).sum(1, keepdim=True))]
            return torch.log_softmax(self.n.fhead(torch.cat(parts, -1)), dim=-1)

    outdir = PKG / "weights" / "funcnet"
    for f in outdir.glob("*.onnx"):
        f.unlink()
    fp, fh = outdir / f"{TAG}_pair.onnx", outdir / f"{TAG}_head.onnx"
    kw = dict(opset_version=17, dynamo=False)
    torch.onnx.export(PairG(net).eval(), (torch.randn(6, 15, 600),), str(fp), input_names=["x"],
                      output_names=["s", "z"], dynamic_axes={"x": {0: "N", 2: "T"}, "s": {0: "N"}, "z": {0: "N"}}, **kw)
    torch.onnx.export(HeadG(net).eval(), (torch.randn(5, 4), torch.randn(5, 4, 128), torch.ones(5)), str(fh),
                      input_names=["logit", "Z", "act"], output_names=["flogp"],
                      dynamic_axes={"logit": {0: "D", 1: "K"}, "Z": {0: "D", 1: "K"}, "act": {0: "D"},
                                    "flogp": {0: "D"}}, **kw)
    json.dump({"members": [{"tag": TAG, "pair": fp.name, "head": fh.name}],
               "averaging": "one member (full-data fit); several members would be averaged in probability space",
               "pieces": "the GRU's pieces (gru_input.split_range, K = 4 of a 32 grid above 120 min); mean log-prob",
               "channels": T69.ADALL, "raster_s": 1, "classes": T69.C7,
               "trained": "note 74 FULL x74_sibafull4l: all non-locked training signals, v4l labels "
                          "(tcn53/func_rows_v4l.parquet), 43 epochs on the six-fold-picked schedule (s74/full/"
                          "lr_siba_full.json), research tcn53 / tcn69_func (snap74b, content partner tie-break)",
               "candidate_filter": "optional (predict.SIBA_FILTER, OFF by default): the pair net runs only on (detector, "
                                   "candidate) pairs with tree p >= .01; the head masks the others (logit -1e4)",
               "note": "note 83 production net (candidate v4)"},
              open(outdir / "manifest.json", "w"), indent=1)
    log(f"exported {fp.name}, {fh.name}")


def cmd_parity():
    import torch
    torch.set_num_threads(THREADS)
    net, cfg, T69 = build_net()
    from neural import tcn53 as M
    pkg_mods = {f.stem for f in PKG.glob("*.py")}
    for name in list(sys.modules):                 # research modules with a package twin (common, ...) -> package's
        if name in pkg_mods:
            del sys.modules[name]
    sys.path.insert(0, str(PKG))
    import gru_input as gi
    import gru_blend as gb
    import funcnet as FN
    import predict as P
    assert Path(FN.__file__).resolve().parent == PKG.resolve()
    model = FN.FuncNet(PKG / "weights" / "funcnet", threads=THREADS)
    assert len(model.members) == 1
    lk = locked()
    res = {"ckpt": str(CKPT), "pieces": [], "max_abs_flogp": 0.0, "max_abs_prob": 0.0, "max_abs_logit": 0.0,
           "filt_max_abs_flogp": 0.0, "filt_max_abs_prob": 0.0, "filt_pairs_kept_share": []}
    cfg_g = gb.config(PKG / "weights")
    rng = np.random.default_rng(83)
    for sig in ("typical_r8", "typical_r11", "busiest_ch", "busiest_ev"):
        for L in ("m30", "h3"):
            f = B71 / f"ev_{sig}_{L}.parquet"
            if not f.exists():
                continue
            con = P._connect(THREADS)
            w0, w1, _ = P.load_events(con, str(f))
            assert not set(con.sql("select distinct DeviceId from ev").df().DeviceId.str.lower()) & lk
            t0, t1 = int(round(w0 * 1000)), int(round(w1 * 1000))
            plan = gb.piece_plan(cfg_g, round((w1 - w0) / 60))
            pieces = gi.split_range(t0, t1, gi.CHUNK_MS, **plan)
            z = next(iter(gi.build_streams(con, t0, t1, windows=pieces).values()))
            con.close()
            dets = [int(c) for c in z["det_ch"]]
            z["_chpos"] = {int(c): i for i, c in enumerate(z["det_ch"])}
            cand = [int(c) for c in z["cand"]]
            K = len(cand)
            # a filter: every detector keeps a random subset of its candidates (>= 1), some detectors keep all
            keep = {}
            for d in dets:
                if rng.random() < 0.2:
                    continue                                  # absent = keep every candidate
                m = rng.random(K) < 0.4
                m[rng.integers(K)] = True
                keep[d] = {c for c, x in zip(cand, m) if x}
            km = np.array([[c in keep.get(d, set(cand)) for c in cand] for d in dets], bool)
            res["filt_pairs_kept_share"].append(round(float(km.mean()), 3))
            pr = [(pa - t0, pb - t0) for pa, pb in pieces]
            P_on, P_onf = model.probs(z, dets, pr), model.probs(z, dets, pr, keep=keep)
            for filt in (False, True):
                acc = None
                for pa, pb in pieces:
                    a_, T = pa - t0, max(int((pb - pa) // 1000), 8)
                    det, ph, sg, nact, partner = M.render53(z, a_, dets, T, 1000)
                    b = dict(det=torch.from_numpy(det[None]), ph=torch.from_numpy(ph[None]),
                             sig=torch.from_numpy(sg[None]), ncand=torch.tensor([K]),
                             dmask=torch.ones(1, len(dets), dtype=torch.bool), partner=torch.from_numpy(partner[None]),
                             nact=torch.from_numpy(nact[None]))
                    if filt:
                        b["keep"] = torch.from_numpy(km[None])
                    with torch.no_grad():
                        lg_t, fl_t = T69.forward69(net, b, "cpu", chunk=1024)
                    lf_t = torch.log_softmax(fl_t.float(), dim=2)[0].numpy()
                    lf_o, lg_o = model.piece_logprobs(model.members[0], z, dets, a_, T, km if filt else None)
                    d1 = float(np.abs(lf_t - lf_o).max())
                    if filt:
                        res["filt_max_abs_flogp"] = max(res["filt_max_abs_flogp"], d1)
                    else:
                        d2 = float(np.abs(lg_t[0].numpy() - lg_o).max())
                        res["max_abs_flogp"] = max(res["max_abs_flogp"], d1)
                        res["max_abs_logit"] = max(res["max_abs_logit"], d2)
                    res["pieces"].append(dict(sig=sig, L=L, D=len(dets), K=K, T=T, filt=filt, flogp=d1))
                    acc = lf_t if acc is None else acc + lf_t
                pt = np.exp(acc / len(pieces))
                pt /= pt.sum(1, keepdims=True)
                k_ = "filt_max_abs_prob" if filt else "max_abs_prob"
                res[k_] = max(res[k_], float(np.abs(pt - (P_onf if filt else P_on)).max()))
            log(f"{sig} {L}: D {len(dets)} K {K} pieces {len(pieces)}; |dlogp| {res['max_abs_flogp']:.2e} "
                f"|dprob| {res['max_abs_prob']:.2e}; filtered |dlogp| {res['filt_max_abs_flogp']:.2e} "
                f"|dprob| {res['filt_max_abs_prob']:.2e}")
    res["pass_1e-4"] = bool(max(res["max_abs_flogp"], res["max_abs_prob"], res["filt_max_abs_flogp"],
                                res["filt_max_abs_prob"]) <= 1e-4)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "siba_parity.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in res.items() if k != "pieces"}))


if __name__ == "__main__":
    {"export": cmd_export, "parity": cmd_parity}[sys.argv[1]]()
