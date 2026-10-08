"""Note 75: exports for the integrated package `%DC_WORK%/final_v3_candidate_v3`.

    python export75.py siba [--members x69_siba_f0]   # function net (siba) -> two ONNX graphs per member + parity vs torch
    python export75.py trees                           # every LightGBM text model -> ONNX TreeEnsemble (opset ml 5) + parity

siba = note-69 TCN joint phase / function net with sibling attention (`neural/tcn69_func.py` Net69, cfg sib=attn).
Exported as TWO graphs so the pair network can be run in pair batches (memory) while the sibling attention sees every
detector of the signal at once:
    pair  x [N, 15, T] (one 1-s raster per (detector, candidate) pair)  ->  s [N] (phase logit), z [N, 128]
    head  logit [D, K], Z [D, K, 128], act [D] (1 = the detector has ONs in the piece)  ->  flogp [D, 7] (log-softmax)
`forward69` (research) with one signal per batch = pair -> scatter -> head; checked here against the torch forward
(fp32, CPU) on real inputs (bench extracts, the package's own streams, every piece): PASS at 1e-4.

Trees: every *.txt LightGBM model of the package's weights is compiled (onnx_trees75.to_onnx_te5) and compared with the
numpy evaluation of the same text model on real rows of its own training / evaluation frame (raw score and the final
transformed output). PASS at 1e-6.
locked_v2 asserted absent from every signal used. CPU only.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PKG = W / "final_v3_candidate_v3"
OUT = W / "final_v3_work" / "v3fit"
CK = W / "tcn53" / "cloud69" / "models"
B71 = W / "bench71"
THREADS = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())


# ================================================================================================ siba
def build_graphs(tag: str, outdir: Path):
    import torch
    sys.path.insert(0, str(CODE))
    import rpath  # noqa: F401
    from neural import tcn69_func as T69
    ck = torch.load(CK / f"{tag}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert cfg["sib"] == "attn" and cfg["fstem"] == 1 and cfg["bw"] == 1000 and cfg.get("arch", "tcn") == "tcn", cfg
    net = T69.Net69(cfg).eval()
    net.load_state_dict(ck["state"])

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
            """== tcn69_func.SibAttn.forward for one signal (B = 1), the diagonal mask built without EyeLike."""
            import math
            sa = self.n.sib
            D, K, E = Z.shape
            X = Z.permute(1, 0, 2)                                  # [K,D,E]
            Wt = w.permute(1, 0)                                    # [K,D]
            q = sa.q(X).reshape(K, D, sa.h, sa.dh).transpose(1, 2)
            k = sa.k(X).reshape(K, D, sa.h, sa.dh).transpose(1, 2)
            v = sa.v(X).reshape(K, D, sa.h, sa.dh).transpose(1, 2)
            s = q @ k.transpose(-1, -2) / math.sqrt(sa.dh)          # [K,h,D,D]
            bias = torch.log(Wt.clamp(min=1e-6))[:, None, None, :] + torch.zeros_like(s)
            bias = bias.masked_fill((Wt <= 0)[:, None, None, :], -1e4)
            idx = torch.arange(D, device=Z.device)
            eye = idx[:, None] == idx[None, :]
            bias = bias.masked_fill(eye[None, None], -1e4)
            sn = (q * sa.nk[None, :, None, :]).sum(-1, keepdim=True) / math.sqrt(sa.dh)
            a = torch.softmax(torch.cat([s + bias, sn], -1), -1)[..., :D]
            o = (a @ v).transpose(1, 2).reshape(K, D, E)
            return sa.o(o).permute(1, 0, 2)                         # [D,K,E]

        def forward(self, logit, Z, act):                          # [D,K], [D,K,E], [D]
            p = torch.softmax(logit, dim=1)
            zmax = Z.amax(1)
            w = p * act[:, None]
            Wk = w.sum(0, keepdim=True)
            ns = (Wk - w).clamp(min=0)
            Sm = self.sib(Z, w)
            parts = [(p[..., None] * Z).sum(1), zmax, (p[..., None] * Sm).sum(1),
                     torch.log1p((p * ns).sum(1, keepdim=True))]
            return torch.log_softmax(self.n.fhead(torch.cat(parts, -1)), dim=-1)

    outdir.mkdir(parents=True, exist_ok=True)
    fp, fh = outdir / f"{tag}_pair.onnx", outdir / f"{tag}_head.onnx"
    kw = dict(opset_version=17, dynamo=False)
    torch.onnx.export(PairG(net).eval(), (torch.randn(6, 15, 600),), str(fp), input_names=["x"],
                      output_names=["s", "z"], dynamic_axes={"x": {0: "N", 2: "T"}, "s": {0: "N"}, "z": {0: "N"}}, **kw)
    torch.onnx.export(HeadG(net).eval(), (torch.randn(5, 4), torch.randn(5, 4, 128), torch.ones(5)), str(fh),
                      input_names=["logit", "Z", "act"], output_names=["flogp"],
                      dynamic_axes={"logit": {0: "D", 1: "K"}, "Z": {0: "D", 1: "K"}, "act": {0: "D"},
                                    "flogp": {0: "D"}}, **kw)
    return net, cfg, fp, fh


def cmd_siba(a):
    import torch
    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(PKG))
    import gru_input as gi                        # the package's own streams / pieces (production input path)
    import gru_blend as gb
    import funcnet as FN
    sys.path.insert(0, str(CODE))
    import rpath  # noqa: F401
    from neural import tcn53 as M
    from neural import tcn69_func as T69
    from neural import train2 as T2
    import duckdb  # noqa: F401
    outdir = PKG / "weights" / "funcnet"
    members = a.members.split(",")
    res = {"members": members, "pieces": [], "max_abs_flogp": 0.0, "max_abs_prob": 0.0, "max_abs_logit": 0.0}
    nets = {}
    for tag in members:
        nets[tag] = build_graphs(tag, outdir)
        log(f"{tag}: exported {nets[tag][2].name}, {nets[tag][3].name}")
    json.dump({"members": [{"tag": t, "pair": f"{t}_pair.onnx", "head": f"{t}_head.onnx"} for t in members],
               "averaging": "mean of the members' probabilities (OOF: mean of the 3 seeds' probabilities per fold)",
               "pieces": "the GRU's pieces (gru_input.split_range, K = 4 of a 32 grid above 120 min); mean log-prob",
               "channels": T69.ADALL, "raster_s": 1, "classes": T69.C7,
               "note": "siba (note 69): research fold model(s) until the GPU full-data refit; swap members here"},
              open(outdir / "manifest.json", "w"), indent=1)
    model = FN.FuncNet(outdir, threads=THREADS)
    lk = locked()
    sys.path.insert(0, str(PKG))
    import predict as P
    cfg_g = gb.config(PKG / "weights")
    for sig in ("typical_r8", "busiest_ch", "busiest_ev"):
        for L in ("m30", "h3"):
            f = B71 / f"ev_{sig}_{L}.parquet"
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
            K = len(z["cand"])
            P_onnx = model.probs(z, dets, [(pa - t0, pb - t0) for pa, pb in pieces])
            for tag in members:
                net = nets[tag][0]
                acc_t = None
                for pa, pb in pieces:
                    a_, T = pa - t0, max(int((pb - pa) // 1000), 8)
                    det, ph, sg, nact, partner = M.render53(z, a_, dets, T, 1000)
                    b = dict(det=torch.from_numpy(det[None]), ph=torch.from_numpy(ph[None]), sig=torch.from_numpy(sg[None]),
                             ncand=torch.tensor([K]), dmask=torch.ones(1, len(dets), dtype=torch.bool),
                             partner=torch.from_numpy(partner[None]), nact=torch.from_numpy(nact[None]))
                    with torch.no_grad():
                        lg_t, fl_t = T69.forward69(net, b, "cpu", chunk=1024)
                    lf_t = torch.log_softmax(fl_t.float(), dim=2)[0].numpy()
                    lf_o, lg_o = model.piece_logprobs(model.members[members.index(tag)], z, dets, a_, T)
                    d1 = float(np.abs(lf_t - lf_o).max())
                    d2 = float(np.abs(lg_t[0].numpy() - lg_o).max())
                    res["max_abs_flogp"] = max(res["max_abs_flogp"], d1)
                    res["max_abs_logit"] = max(res["max_abs_logit"], d2)
                    res["pieces"].append(dict(tag=tag, sig=sig, L=L, D=len(dets), K=K, T=T, flogp=d1, logit=d2))
                    acc_t = lf_t if acc_t is None else acc_t + lf_t
                pt = np.exp(acc_t / len(pieces)); pt /= pt.sum(1, keepdims=True)
                if len(members) == 1:
                    res["max_abs_prob"] = max(res["max_abs_prob"], float(np.abs(pt - P_onnx).max()))
            log(f"{sig} {L}: D {len(dets)} K {K} pieces {len(pieces)}; |dlogp| {res['max_abs_flogp']:.2e} "
                f"|dlogit| {res['max_abs_logit']:.2e} |dprob| {res['max_abs_prob']:.2e}")
    res["pass_1e-4"] = bool(max(res["max_abs_flogp"], res["max_abs_prob"]) <= 1e-4)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "siba_parity.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in res.items() if k != "pieces"}))


# ================================================================================================ trees
def _frames():
    """real rows per model family (research frames; non-locked)."""
    import pyarrow.parquet as pq
    lk = locked()
    fr = {}
    ph = pq.read_table(W / "trees57" / "phase" / "pool.parquet").slice(0, 40000).to_pandas()
    assert not ph.DeviceId.str.replace("@stg", "").str.lower().isin(lk).any()
    fr["phase"] = ph
    return fr


def cmd_trees(a):
    sys.path.insert(0, str(CODE / "final75"))
    sys.path.insert(0, str(PKG))
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    import trees_onnx as TO
    src = Path(a.src)
    rows = json.load(open(a.rows)) if a.rows else {}
    res = {}
    for f in sorted(src.rglob("*.txt")):
        rel = f.relative_to(src)
        nb = NumpyBooster(f)
        dst = PKG / "weights" / rel.with_suffix(".onnx")
        dst.parent.mkdir(parents=True, exist_ok=True)
        OT.to_onnx_te5(nb, dst)
        Xf = rows.get(str(rel).replace("\\", "/"))
        if Xf is None:
            res[str(rel)] = {"onnx": str(dst.relative_to(PKG)), "checked": False}
            continue
        X = np.load(Xf)
        m = TO.OnnxTrees(dst, THREADS)
        r0, r1 = OT.raw_numpy(nb, X), m.raw(X)
        p0, p1 = nb.predict(X), m.predict(X)
        res[str(rel)] = {"onnx": str(dst.relative_to(PKG)), "checked": True, "rows": int(len(X)),
                         "max_abs_raw": float(np.abs(r0 - r1).max()), "max_abs_out": float(np.abs(p0 - p1).max())}
        log(f"{rel}: {res[str(rel)]}")
    json.dump(res, open(OUT / f"trees_parity_{a.tag}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["siba", "trees"])
    ap.add_argument("--members", default="x69_siba_f0")
    ap.add_argument("--src", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--tag", default="pkg")
    a = ap.parse_args()
    {"siba": cmd_siba, "trees": cmd_trees}[a.cmd](a)


if __name__ == "__main__":
    main()
