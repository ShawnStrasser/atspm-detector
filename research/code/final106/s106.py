"""Note 106: vehicle paths / lane structure for the FUNCTION model (user idea: the function model does not "see" which
detectors are upstream / downstream of which, travel times, lane groups, number of lanes).  2026 six-fold OOF set-up of
v5b (note 95 / 100c: 2026 trees mean3 + siba w32 mean3 -> context stacker (47 cols, PRM0 150 rounds) -> gate .9 lane
decode -> twin decode; v4q truth; Sept-2026 rows only).

    set F76_ARM=c & python s106.py cache                  frame keys, base stacker matrix X0 (47), y, trm, lane ctx -> s106/
    set F76_ARM=c & python s106.py stack NAME COLS SEEDS  stacker OOF on X0 + extra column block(s) -> s106/P_NAME.npy
    set F76_ARM=c & python s106.py score NAME [NAME ...]  each vs REF (gate .9 decode), E / R, pools, by class / fold
COLS = '+'-joined blocks from s106/feat_*.parquet (frame-aligned in `cache` order) or 'none'; suffix '~shuf' permutes
the block's rows inside window groups (control).  SEEDS e.g. 0 or 0,1,2.
locked_v2 asserted absent by the loaders.  CPU <= 6 threads.
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
sys.path.insert(0, str(CODE / "final95"))
import func95 as G  # noqa: E402

OUT = G.DC_WORK / "s106"
NET = "x100_w32"
KEY = ["DeviceId", "Detector", "period", "win"]
THREADS = int(os.environ.get("T106", "4"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


_E = {}


def load():
    if not _E:
        F, F4 = G.fsetup()
        F.THREADS = THREADS
        s74, S, E, Xs, y, trm, names = F.stacker_inputs()
        _E.update(F=F, F4=F4, s74=s74, S=S, E=E, y=y, trm=trm, names=names)
    return _E


def stage_cache():
    import of77
    t0 = time.time()
    d = load()
    E, s74, S, F4 = d["E"], d["s74"], d["S"], d["F4"]
    fr = E["fr"]
    log(f"loaded frame {len(fr):,} rows in {time.time()-t0:.0f}s")
    Pf = s74.net_probs(fr, NET)
    X0 = F4.net_X(s74, S, E, Pf)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "X0.npy", X0.astype(np.float32))
    np.save(OUT / "Pt.npy", E["Pt"].astype(np.float32))
    np.save(OUT / "Pn.npy", np.where(np.isnan(Pf[:, :1]), E["Pt"], Pf).astype(np.float32))
    np.save(OUT / "y.npy", d["y"])
    np.save(OUT / "trm.npy", d["trm"] & (fr.period == "stg").to_numpy())
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")
    np.save(OUT / "lane_ctx.npy", lc)
    cols = KEY + ["wgroup", "fold", "pred_phase", "lanes5g", "det_n_on", "truth_v3s"]
    fr[cols].astype({"lanes5g": object}).to_parquet(OUT / "keys.parquet", index=False)
    json.dump(d["names"], open(OUT / "names0.json", "w"))
    log(f"X0 {X0.shape}; trm {int(np.load(OUT / 'trm.npy').sum()):,}")


def block(name):
    """frame-aligned feature block; name '~shuf' suffix = permuted inside (period, win) groups."""
    shuf = name.endswith("~shuf")
    nm = name.replace("~shuf", "")
    B = pd.read_parquet(OUT / f"feat_{nm}.parquet")
    k = pd.read_parquet(OUT / "keys.parquet", columns=KEY)
    assert len(B) == len(k) and (B.DeviceId.to_numpy() == k.DeviceId.to_numpy()).all() and \
        (B.win.to_numpy() == k.win.to_numpy()).all() and (B.Detector.to_numpy() == k.Detector.to_numpy()).all()
    cols = [c for c in B.columns if c not in KEY]
    Z = B[cols].to_numpy(np.float32)
    if shuf:
        rng = np.random.default_rng(106)
        g = k.groupby(["period", "win"]).indices
        for ix in g.values():
            Z[ix] = Z[rng.permutation(ix)]
    return Z, cols


def stack(name, spec, seeds, X0=None, extra=None):
    import lightgbm as lgb
    F = G.fsetup()[0]
    X = np.load(OUT / "X0.npy") if X0 is None else X0
    cols = []
    if spec != "none":
        for b in spec.split("+"):
            Z, c = block(b)
            X = np.hstack([X, Z])
            cols += c
    if extra is not None:
        X = np.hstack([X, extra])
    y, trm = np.load(OUT / "y.npy"), np.load(OUT / "trm.npy")
    fo = pd.read_parquet(OUT / "keys.parquet", columns=["fold"]).fold.to_numpy()
    R = np.zeros((len(y), 7))
    gain = np.zeros(X.shape[1])
    for seed in seeds:
        for k in range(6):
            tr, te = trm & (fo != k), fo == k
            m = lgb.train(dict(F.PRM0, num_threads=THREADS, seed=seed), lgb.Dataset(X[tr], y[tr]), num_boost_round=150)
            R[te] += m.predict(X[te]) / len(seeds)
            gain += m.feature_importance("gain")
        log(f"  {name}: stacker seed {seed} done")
    np.save(OUT / f"P_{name}.npy", R.astype(np.float32))
    names = json.load(open(OUT / "names0.json")) + cols
    gi = pd.Series(gain[:len(names)] / gain.sum(), index=names[:len(gain)]).sort_values(ascending=False)
    json.dump({"spec": spec, "seeds": list(seeds), "ncol": int(X.shape[1]), "new_gain_share": float(gi[cols].sum()) if cols else 0.0,
               "top": gi.head(20).round(4).to_dict(), "new": gi[cols].round(4).to_dict() if cols else {}},
              open(OUT / f"P_{name}.json", "w"), indent=1)
    return R


def ok_arrays(P, lconf=None, lanes=None):
    """gate .9 decode + stack-aware credit; optional replacement lane_conf / lanes5g columns."""
    d = load()
    E, s74 = d["E"], d["s74"]
    fr = E["fr"]
    lc = np.load(OUT / "lane_ctx.npy")[:, 2] if lconf is None else lconf
    if lanes is not None:
        old = fr.lanes5g.copy()
        fr["lanes5g"] = lanes
        try:
            return s74.gate_ok(E, P, lc)
        finally:
            fr["lanes5g"] = old
    return s74.gate_ok(E, P, lc)


def score_many(P: dict, ref: str, out_name: str, extra_masks: dict | None = None):
    """P: name -> probs (or ok dict).  Paired vs ref: E / R per pool, by class, by fold."""
    import cand64 as C
    d = load()
    E, S = d["E"], d["S"]
    fr = E["fr"]
    sig = fr.DeviceId.to_numpy()
    ok = {k: (v if isinstance(v, dict) else ok_arrays(v)) for k, v in P.items()}
    st = (fr.period == "stg").to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    fo = fr.fold.to_numpy()
    pools = {**C.POOLS, "all": C.FAMS}
    for w in ["m30", "h1", "h3", "h6", "h24", "full"]:
        pools.setdefault(w, [w])
    res = {}
    names = [k for k in ok if k != ref]
    for stn in ("E", "R"):
        for pool, fams in pools.items():
            sc = st & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k][stn]) for k in ok])
            r = {"n": int(sc.sum()), "signals": int(len(set(sig[sc])))}
            for k in ok:
                r[k] = C.acc_ci(ok[k][stn][sc], sig[sc])
            for k in names:
                r[f"{k} - {ref}"] = C.delta_ci(ok[ref][stn][sc], ok[k][stn][sc], sig[sc])
            res[f"{stn}_{pool}"] = r
        m = st & fr.wgroup.isin(C.GE30).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k][stn]) for k in ok])
        res[f"{stn}_ge30_by_class"] = {k: {c: C.delta_ci(ok[ref][stn][m & (cl == c)], ok[k][stn][m & (cl == c)],
                                                           sig[m & (cl == c)]) for c in
                                           ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]} for k in names}
        res[f"{stn}_ge30_by_fold"] = {k: [round(100 * float(ok[k][stn][m & (fo == f)].mean()
                                                            - ok[ref][stn][m & (fo == f)].mean()), 3)
                                          for f in range(6)] for k in names}
        for mn, mm in (extra_masks or {}).items():
            sc = m & mm
            res[f"{stn}_ge30_{mn}"] = {"n": int(sc.sum()), **{k: C.acc_ci(ok[k][stn][sc], sig[sc]) for k in ok},
                                       **{f"{k} - {ref}": C.delta_ci(ok[ref][stn][sc], ok[k][stn][sc], sig[sc])
                                          for k in names}}
    f = OUT / "scores.json"
    old = json.load(open(f)) if f.exists() else {}
    old[out_name] = res
    json.dump(old, open(f, "w"), indent=1, default=str)
    for k, v in res.items():
        if k in ("E_ge30", "R_ge30", "E_m5", "E_m10") or k.startswith("E_ge30_") and "by" not in k:
            log(f"{k}: " + ", ".join(f"{a}={b}" for a, b in v.items() if a not in ("signals",)))
    return res, ok


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "cache":
        stage_cache()
    elif a[0] == "stack":
        stack(a[1], a[2], [int(s) for s in a[3].split(",")])
    elif a[0] == "score":
        ref = os.environ.get("REF106", "base_s0")
        P = {n: np.load(OUT / f"P_{n}.npy").astype(float) for n in a[1:]}
        score_many(P, ref, ",".join(a[1:]))
