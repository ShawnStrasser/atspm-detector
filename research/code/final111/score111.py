"""Note 111: OOF headline of package v6 (= note-109 design C) and of its fast setting le2h, vs v5c and v5c's le2h, from
saved six-fold OOF only (note 95 / 100b / 109 setup: 2026 pool, folds_v4, timing truth / v4q truth, Sept-2026 rows).

    phase     v5c = q95 p2_p95_ad; v5c_le2h = q95 p2_le2h; v6 = q109 p2_w32m3_rs0 (ranker s0 + siba head x3 -> decoder);
              v6_le2h = v6 on windows <= 2 h (m5 / m10 / m30 / h1), else q111 p2_trees_s0 (ranker s0 -> trees decoder)
    function  v5c = s100/b P_w32m3; v5c_le2h = it <= 2 h, else P_v5_nonet; v6 = s109/func P_m3_t0_s0 (trees s0 + net x3,
              stacker s0); v6_le2h = v6 <= 2 h, else s111/func P_nonet_t0_s0; gate .9 lane decode
    set F76_ARM=c & python score111.py      -> %DC_WORK%/s111/score111.json
locked_v2 asserted absent by the loaders.  CPU, 6 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
for _d in ("final95", "final109", "final103", "final100"):
    sys.path.insert(0, str(CODE / _d))
sys.path.insert(0, str(CODE))
import rpath  # noqa: E402,F401

LE = ["m5", "m10", "m30", "h1"]
FAMS = {"m5": ["m5"], "m10": ["m10"], "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h24": ["h24"],
        "ge30": ["m30", "h1", "h3", "h6", "h24", "full"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def phase(res):
    import phase95 as P
    import p109_phase as P109
    C = P.C
    comb, _ = P.pool(P.KEY4 + ["fold"])
    q = pd.read_parquet(P.P95 / "q95.parquet")
    q9 = pd.read_parquet(P109.OUT / "q109.parquet")
    q11 = pd.read_parquet(C.DC_WORK / "s111" / "phase" / "q111.parquet")
    for t in (q, q9, q11):
        for c in P.KEY4:
            assert (t[c].to_numpy() == comb[c].to_numpy()).all()
    pair = {"v5c": q.p2_p95_ad.to_numpy(), "v5c_le2h": q.p2_le2h.to_numpy(), "v6": q9.p2_w32m3_rs0.to_numpy(),
            "trees_s0": q11.p2_trees_s0.to_numpy()}
    rows = P.rows_()
    alt = rows.alt.to_numpy()
    sig = rows.sig.to_numpy()
    DET = P.DET
    pred = {}
    for nm, p in pair.items():
        d = comb[P.KEY4].copy()
        d["p"] = p
        d = d.dropna(subset=["p"]).sort_values(DET + ["p", "cand_phase"], ascending=[True, True, True, False, True])
        t = d.groupby(DET, sort=False).first().reset_index()
        pred[nm] = rows[DET].merge(t[DET + ["cand_phase"]], on=DET, how="left").cand_phase.to_numpy()
    le = rows.fam.isin(LE).to_numpy()
    pred["v6_le2h"] = np.where(le, pred["v6"], pred["trees_s0"])
    ok = {}
    for nm, pr in pred.items():
        okE = (pr == rows.Phase.to_numpy()).astype(float)
        okR = np.maximum(okE, np.array([p_ in s for p_, s in zip(pr, alt)], float))
        ok[nm] = {"E": okE, "R": okR}
    out = {"n": {}, "acc": {}, "delta_vs_v5c": {}, "v6_le2h_vs_v5c_le2h": {}}
    for sname, pre in (("everything", "E"), ("realistic", "R")):
        mset = rows[sname].to_numpy()
        for pn, fams in FAMS.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            out["n"].setdefault(pre, {})[pn] = [int(m.sum()), int(len(set(sig[m])))]
            for nm in ok:
                out["acc"].setdefault(nm, {}).setdefault(pre, {})[pn] = C.acc_ci(ok[nm][pre][m], sig[m])
                if nm != "v5c":
                    out["delta_vs_v5c"].setdefault(nm, {}).setdefault(pre, {})[pn] = C.delta_ci(
                        ok["v5c"][pre][m], ok[nm][pre][m], sig[m])
            out["v6_le2h_vs_v5c_le2h"].setdefault(pre, {})[pn] = C.delta_ci(ok["v5c_le2h"][pre][m],
                                                                             ok["v6_le2h"][pre][m], sig[m])
    res["phase"] = out
    for nm in ok:
        log(f"phase {nm:10s} E " + " ".join(f"{p} {out['acc'][nm]['E'][p][0]}" for p in FAMS))


def function(res):
    import cand64 as C
    import func95 as G
    import of77
    import f109_func as F109
    F, F4 = G.fsetup()
    s74, S, E, Xs, y, trm, _ = F.stacker_inputs()
    fr = E["fr"]
    assert not fr.DeviceId.str.lower().isin(C.locked()).any()
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    st = (fr.period == "stg").to_numpy()
    le = fr.wgroup.isin(LE).to_numpy()[:, None]
    full = np.load(F109.B100 / "P_w32m3.npy").astype(float)
    nonet5 = np.load(G.OOF95 / "P_v5_nonet.npy").astype(float)
    v6 = np.load(F109.OUT / "P_m3_t0_s0.npy").astype(float)
    nonet6 = np.load(G.DC_WORK / "s111" / "func" / "P_nonet_t0_s0.npy").astype(float)
    P = {"v5c": full, "v5c_le2h": np.where(le, full, nonet5), "v6": v6, "v6_le2h": np.where(le, v6, nonet6),
         "v6_nonet": nonet6}
    ok = {k: s74.gate_ok(E, v, lc) for k, v in P.items()}
    out = {"n": {}, "acc": {}, "delta_vs_v5c": {}, "v6_le2h_vs_v5c_le2h": {}}
    for stn in ("E", "R"):
        for pool, fams in FAMS.items():
            sc = st & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k][stn]) for k in ok])
            out["n"].setdefault(stn, {})[pool] = [int(sc.sum()), int(len(set(sig[sc])))]
            for k in ok:
                out["acc"].setdefault(k, {}).setdefault(stn, {})[pool] = C.acc_ci(ok[k][stn][sc], sig[sc])
                if k != "v5c":
                    out["delta_vs_v5c"].setdefault(k, {}).setdefault(stn, {})[pool] = C.delta_ci(
                        ok["v5c"][stn][sc], ok[k][stn][sc], sig[sc])
            out["v6_le2h_vs_v5c_le2h"].setdefault(stn, {})[pool] = C.delta_ci(
                ok["v5c_le2h"][stn][sc], ok["v6_le2h"][stn][sc], sig[sc])
    res["function"] = out
    for k in ok:
        log(f"function {k:10s} E " + " ".join(f"{p} {out['acc'][k]['E'][p][0]}" for p in FAMS))


if __name__ == "__main__":
    res = {}
    phase(res)
    function(res)
    o = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work") / "s111" / "score111.json"
    json.dump(res, open(o, "w"), indent=1)
    for t in ("phase", "function"):
        for k, v in res[t]["delta_vs_v5c"].items():
            log(f"{t} {k} dE vs v5c " + " ".join(f"{p} {v['E'][p]}" for p in FAMS) + f" | dR ge30 {v['R']['ge30']}")
        log(f"{t} v6_le2h vs v5c_le2h dE " + " ".join(f"{p} {res[t]['v6_le2h_vs_v5c_le2h']['E'][p]}" for p in FAMS))
