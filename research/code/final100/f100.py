"""Note 100b: the v5 function recipe (note 95: 2026 trees OOF + siba fold OOF -> context stacker, 3 LightGBM seeds, Sept-2026
training rows -> gate .9 decode, v4q truth) with a different network OOF, scored against v5 itself.

    set F76_ARM=c & python f100.py stack NAME NETSPEC     NETSPEC as s74.net_probs: PREFIX (3-seed mean, _s1/_s2 files)
                                                          or PREFIX:0 / PREFIX:0+1 (chosen seeds)  -> s100/b/P_NAME.npy
    set F76_ARM=c & python f100.py score NAME [NAME ...]  each vs v5 (f95/oof/P_v5.npy), paired signal bootstrap, Sept-2026
                                                          rows, E / R, >= 30 / 10 / 5 min, by class, by fold -> s100/b/f100.json
Same code path as func95.stage_stack / stage_score (imported); only the network input and the output folder change.
locked_v2 asserted absent by the loaders (func_table / frame).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
import func95 as G  # noqa: E402

OUT = G.DC_WORK / "s100" / "b"


def stack(name, spec):
    import lightgbm as lgb
    F, F4 = G.fsetup()
    OUT.mkdir(parents=True, exist_ok=True)
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    Pf = s74.net_probs(fr, spec)
    st = (fr.period == "stg").to_numpy()
    cov = float(np.mean(~np.isnan(Pf[st, 0])))
    F.log(f"{spec}: coverage stg rows {cov:.4f}")
    X = F4.net_X(s74, S, E, Pf)
    trm = trm & st
    fo = fr.fold.to_numpy()
    R = np.zeros((len(y), 7))
    for seed in (0, 1, 2):
        for k in range(6):
            tr, te = trm & (fo != k), fo == k
            m = lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=seed), lgb.Dataset(X[tr], y[tr]), num_boost_round=150)
            R[te] += m.predict(X[te]) / 3
        F.log(f"  stacker seed {seed} done")
    np.save(OUT / f"P_{name}.npy", R.astype(np.float32))
    json.dump({"spec": spec, "coverage_stg": cov, "train_rows": int(trm.sum())}, open(OUT / f"P_{name}.json", "w"))


def score(names):
    import cand64 as C
    import of77
    F, F4 = G.fsetup()
    s74, S, E, Xs, y, trm, _ = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    P = {"v5": np.load(G.OOF95 / "P_v5.npy").astype(float)}
    for n in names:
        P[n] = np.load(OUT / f"P_{n}.npy").astype(float)
    ok = {k: s74.gate_ok(E, v, lc) for k, v in P.items()}
    st = (fr.period == "stg").to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    fo = fr.fold.to_numpy()
    res = {}
    for stn in ("E", "R"):
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            pm = st & fr.wgroup.isin(fams).to_numpy()
            sc = pm & np.logical_and.reduce([~np.isnan(ok[k][stn]) for k in ok])
            r = {"n": int(sc.sum()), "signals": int(len(set(sig[sc])))}
            for k in ok:
                r[k] = C.acc_ci(ok[k][stn][sc], sig[sc])
            for k in names:
                r[f"{k} - v5"] = C.delta_ci(ok["v5"][stn][sc], ok[k][stn][sc], sig[sc])
            res[f"{stn}_{pool}"] = r
        m = st & fr.wgroup.isin(C.GE30).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k][stn]) for k in ok])
        res[f"{stn}_ge30_by_class"] = {k: {c: C.delta_ci(ok["v5"][stn][m & (cl == c)], ok[k][stn][m & (cl == c)],
                                                           sig[m & (cl == c)]) for c in
                                           ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]} for k in names}
        res[f"{stn}_ge30_by_fold_minus_v5"] = {k: [round(100 * float(ok[k][stn][m & (fo == f)].mean()
                                                                     - ok["v5"][stn][m & (fo == f)].mean()), 3)
                                                   for f in range(6)] for k in names}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "f100.json"
    old = json.load(open(f)) if f.exists() else {}
    old[",".join(names)] = res
    json.dump(old, open(f, "w"), indent=1, default=str)
    for k, v in res.items():
        if "ge30" in k or "_m10" in k or "_m5" in k:
            F.log(f"{k}: {v}")


def le2h(name):
    """fast profile le2h (note 98 / 95b): stacker OOF NAME on families <= h1, the saved v5 'nonet' OOF above; vs v5 full
    and v5 le2h (same nonet), E / R per pool."""
    import cand64 as C
    import of77
    F, F4 = G.fsetup()
    s74, S, E, Xs, y, trm, _ = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    st = (fr.period == "stg").to_numpy()
    le = fr.wgroup.isin(["m5", "m10", "m30", "h1"]).to_numpy()[:, None]
    R = np.load(G.OOF95 / "P_v5_nonet.npy").astype(float)
    P5 = np.load(G.OOF95 / "P_v5.npy").astype(float)
    Pn = np.load(OUT / f"P_{name}.npy").astype(float)
    P = {"v5": P5, "v5_le2h": np.where(le, P5, R), name: Pn, f"{name}_le2h": np.where(le, Pn, R)}
    ok = {k: s74.gate_ok(E, v, lc) for k, v in P.items()}
    out = {}
    for stn in ("E", "R"):
        for pool, fams in {**C.POOLS, "gt2h": ["h3", "h6", "h24", "full"]}.items():
            sc = st & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k][stn]) for k in ok])
            r = {k: C.acc_ci(ok[k][stn][sc], sig[sc]) for k in ok}
            r[f"{name}_le2h - v5"] = C.delta_ci(ok["v5"][stn][sc], ok[f"{name}_le2h"][stn][sc], sig[sc])
            r[f"{name}_le2h - v5_le2h"] = C.delta_ci(ok["v5_le2h"][stn][sc], ok[f"{name}_le2h"][stn][sc], sig[sc])
            r["n"] = int(sc.sum())
            out[f"{stn}_{pool}"] = r
    json.dump(out, open(OUT / f"le2h_{name}.json", "w"), indent=1)
    for k, v in out.items():
        F.log(f"{k}: {v}")


if __name__ == "__main__":
    if sys.argv[1] == "stack":
        stack(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "le2h":
        le2h(sys.argv[2])
    else:
        score(sys.argv[2:])
