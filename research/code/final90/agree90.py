"""Note 90: agreement between the full-data siba members, v4o trio (x86_sibafull4o seeds 0/1/2 [+ 3 if present]) vs the v4l
trio of v4e (x74_sibafull4l seeds 0/1/2) -- is one member an outlier (orchestrator: seed 2 trained to a higher loss)?

Inputs, unfiltered, package FuncNet graphs one member at a time:
  bench   the 4 bench signals x 30 min / 3 h / 24 h (package streams from raw events, package piece plan)
  pool    ~60 non-locked pool signals (evenly spaced over the Dec period list) x windows m30_b and h3_a (research store
          bundles, tcn53 pieces) -- training signals of the full-data members, so agreement only
Per pair of members: argmax agreement and mean |dprob| (mean over detectors of sum_c |p_i - p_j| / 2); per member: mean
agreement with the other two.
    python agree90.py [extra_tag ...]  -> %DC_WORK%/final_v3_work/f90/agree90.json
CPU, 2 threads.  locked_v2 asserted absent.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

V4F = DC_WORK / "final_v3_candidate_v4f"
V4E = DC_WORK / "final_v3_candidate_v4e"
TMPD = DC_WORK / "final_v3_work" / "f90" / "agree_tmp"
B71 = DC_WORK / "bench71"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    extra = sys.argv[1:]
    sys.path.insert(0, str(V4F))
    import funcnet as FN
    import gru_blend as gb
    import predict as P
    assert Path(FN.__file__).resolve().parent == V4F.resolve()
    sets = {"v4o": [(V4F / "weights" / "funcnet", t) for t in os.environ.get("AGREE90_V4O", "x86_sibafull4o,x86_sibafull4o_s1,x86_sibafull4o_s3").split(",")],
            "v4l": [(V4E / "weights" / "funcnet", t) for t in ("x74_sibafull4l", "x74_sibafull4l_s1", "x74_sibafull4l_s2")]}
    for t in extra:                                # extra members exported to TMPD (siba90-style graphs)
        sets["v4o"].append((TMPD, t))
    members = {}
    for nm, lst in sets.items():
        for d, t in lst:
            m = FN.FuncNet.__new__(FN.FuncNet)
            m.members = [(FN._session(d / f"{t}_pair.onnx", 2), FN._session(d / f"{t}_head.onnx", 2))]
            m.pair_batch, m.member_tags = 64, [t]
            members[t] = m
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    probs = {t: [] for t in members}
    src = []
    cfg = gb.config(V4F / "weights")
    for sig in ("typical_r8", "typical_r11", "busiest_ev", "busiest_ch"):
        for L in ("m30", "h3", "h24"):
            con = P._connect(2)
            w0, w1, _ = P.load_events(con, str(B71 / f"ev_{sig}_{L}.parquet"))
            assert not set(con.sql("select distinct DeviceId from ev").df().DeviceId.str.lower()) & lk
            streams, rel = gb.streams_for(con, int(round(w0 * 1000)), int(round(w1 * 1000)),
                                          **gb.piece_plan(cfg, round((w1 - w0) / 60)))
            con.close()
            z = next(iter(streams.values()))
            dets = [int(c) for c in z["det_ch"]]
            for t, m in members.items():
                probs[t].append(m.probs(z, dets, rel))
            src += ["bench"] * len(dets)
    log(f"bench: {len(src)} detector-windows")
    from neural import data2 as D2
    from neural import phase_v3_net as P3N  # noqa: F401
    from neural import infer2 as I2
    from neural import tcn53 as T
    sigs = D2.training_signals()
    table = D2.load_table(sigs, labelled_only=False)
    keys = [k for k in sigs.key if k in table and table[k]["period"] == "dec"]
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, 60).astype(int)]
    assert not any(k.split("|", 1)[1].lower() in lk for k in keys)
    stores = D2.Stores()
    for name, start, secs in I2.WINDOWS["dec"]:
        if name not in ("m30_b", "h3_a"):
            continue
        pcs = I2.pieces("dec", start, secs, T.GRID)
        if secs > T.LONG_S:
            from pieces_infer import subset
            pcs = [pcs[i] for i in subset(len(pcs), T.K_LONG)]
        for k in keys:
            rec = table[k]
            z = stores.get(rec["period"]).get(rec["dev"])
            dets = [int(d) for d in rec["dets"]]
            if not dets or len(z["cand"]) < 1:
                continue
            for t, m in members.items():
                probs[t].append(m.probs(z, dets, pcs))
            src += ["pool"] * len(dets)
    log(f"total: {len(src)} detector-windows")
    src = np.array(src)
    Pm = {t: np.vstack(v) for t, v in probs.items()}
    ok = np.logical_and.reduce([np.isfinite(v).all(1) for v in Pm.values()])
    res = {"n": {s: int((ok & (src == s)).sum()) for s in ("bench", "pool")}, "pairs": {}, "member_vs_others": {}}
    for nm, lst in sets.items():
        tags = [t for _, t in lst]
        for a, b in itertools.combinations(tags, 2):
            r = {}
            for s in ("bench", "pool", "all"):
                m = ok & ((src == s) if s != "all" else True)
                r[s] = {"argmax_agree": round(float((Pm[a][m].argmax(1) == Pm[b][m].argmax(1)).mean()), 4),
                        "mean_tv": round(float(np.abs(Pm[a][m] - Pm[b][m]).sum(1).mean() / 2), 4)}
            res["pairs"][f"{a} ~ {b}"] = r
        for a in tags:
            others = [b for b in tags if b != a and not (b.endswith("_s3") or a.endswith("_s3")) or b == a]
            others = [b for b in tags if b != a]
            m = ok
            res["member_vs_others"][a] = {
                "argmax_agree_mean": round(float(np.mean([(Pm[a][m].argmax(1) == Pm[b][m].argmax(1)).mean() for b in others])), 4),
                "mean_tv_mean": round(float(np.mean([np.abs(Pm[a][m] - Pm[b][m]).sum(1).mean() / 2 for b in others])), 4),
                "class_share": {c: round(float((Pm[a][m].argmax(1) == i).mean()), 4) for i, c in enumerate(FN.C7)}}
    json.dump(res, open(DC_WORK / "final_v3_work" / "f90" / "agree90.json", "w"), indent=1)
    for k, v in res["pairs"].items():
        log(f"{k}: {v}")
    for k, v in res["member_vs_others"].items():
        log(f"{k}: {v}")


if __name__ == "__main__":
    main()
