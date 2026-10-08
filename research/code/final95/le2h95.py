"""Note 95b: accuracy of the fast profile le2h (note 98: both networks only on samples <= 2 h) on the 2026-only v5 OOF.
Phase: trees bag 0.5 / TCN p95_ad 0.5 (tree p >= .01) on families m5-h1, trees alone above; ONE decoder re-fitted six-fold
on that mixed input (note-98 recipe). Function: v5 mean3 stacker OOF (P_v5) on families <= h1, a 'nonet' stacker OOF (net
columns = trees' probabilities, Sept-2026 rows) above. Gate .9; v4q truth; Sept-2026 rows.
    python le2h95.py phase | func   -> %DC_WORK%/final_v3_work/f95/oof/le2h95_<part>.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("F76_ARM", "c")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

LE = ["m5", "m10", "m30", "h1"]


def phase():
    import phase95 as P
    import p90_phase as P90
    import pandas as pd
    P90.C.NJ = 6
    comb, simc = P.pool()
    t = P.trees_bag(comb)
    nn = P.tcn_probs(comb, "p95_ad")
    fam = comb.win.map(lambda w: "full" if w.startswith("full") else w.split("_")[0]).to_numpy()
    nn = np.where(np.isin(fam, LE), nn, np.nan)
    p2, its = P90.decode_iters(comb, simc, P.blend(comb, t.p0_bag.to_numpy(), nn))
    q = pd.read_parquet(P.P95 / "q95.parquet")
    q["p2_le2h"] = p2
    q.to_parquet(P.P95 / "q95.parquet", index=False)
    print("le2h decoded", its)


def func():
    import func95 as FN
    import lightgbm as lgb
    F, F4 = FN.fsetup()
    import cand64 as C
    import of77
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr, Pt = E["fr"], E["Pt"]
    st = (fr.period == "stg").to_numpy()
    X = F4.net_X(s74, S, E, np.full_like(Pt, np.nan))
    tr0 = trm & st
    fo = fr.fold.to_numpy()
    R = np.zeros((len(y), 7))
    for seed in (0, 1, 2):
        for k in range(6):
            m = lgb.train(dict(F.PRM0, num_threads=6, seed=seed), lgb.Dataset(X[tr0 & (fo != k)], y[tr0 & (fo != k)]),
                          num_boost_round=150)
            R[fo == k] += m.predict(X[fo == k]) / 3
    np.save(FN.OOF95 / "P_v5_nonet.npy", R.astype(np.float32))
    P5 = np.load(FN.OOF95 / "P_v5.npy").astype(float)
    le = fr.wgroup.isin(LE).to_numpy()
    Pl = np.where(le[:, None], P5, R)
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    ok = {"v5": s74.gate_ok(E, P5, lc), "le2h": s74.gate_ok(E, Pl, lc), "nonet": s74.gate_ok(E, R, lc)}
    sig = fr.DeviceId.to_numpy()
    out = {}
    for stn in ("E", "R"):
        for pool, fams in {**C.POOLS, "gt2h": ["h3", "h6", "h24", "full"]}.items():
            sc = st & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k][stn]) for k in ok])
            r = {k: C.acc_ci(ok[k][stn][sc], sig[sc]) for k in ok}
            r["le2h - v5"] = C.delta_ci(ok["v5"][stn][sc], ok["le2h"][stn][sc], sig[sc])
            r["n"] = int(sc.sum())
            out[f"{stn}_{pool}"] = r
    json.dump(out, open(FN.OOF95 / "le2h95_func.json", "w"), indent=1)
    for k, v in out.items():
        print(k, v, flush=True)


if __name__ == "__main__":
    {"phase": phase, "func": func}[sys.argv[1]]()
