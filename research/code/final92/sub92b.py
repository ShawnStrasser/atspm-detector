"""Note 92b: 2-fold screen (folds 0 and 3, seed 0, v4o) of a siba net with an 11-class function head (Other split into
note 92's subclasses), inside note 92's K-class chain (11-class trees -> stacker -> kway / sum decode), vs v4f.

Only two folds of the new net exist, so EVERY arm's stacker is re-fitted the same way ("cross-2-fold"): the stacker that
scores fold 0 is trained on fold 3 rows only and vice versa (3 LightGBM seeds, PRM0 150 rounds, note-92 inputs). Arms:
  v4f2   v4f recipe: v4o trees (7) + x86_siba4l filtered 3-seed mean (7), 47 columns, K = 7
  sub2   note-92 sub: 11-class trees (7-view + 4 subclass probs) + x86_siba4l (7), K = 11
  net11  sub2's trees + the NEW 11-class net (x92_siba11, filtered, seed 0): its 7-view (subclasses summed into Other) in
         the 47 columns + its 4 subclass probabilities, K = 11
References on the same rows: v4f (note 90 full six-fold OOF P_v4f) and note 92's PK_sub (full six-fold).
Score = note 92 (sub92.gate_pred, ATSPM-only stack-aware, v4l truth), restricted to folds 0 / 3, paired signal bootstrap.

    python sub92b.py rows                    -> %DC_WORK%/tcn53/func_rows_v4o_sub11.parquet (y Other -> subclass)
    python sub92b.py stack --arm v4f2|sub2|net11   -> %DC_WORK%/x92/oof2/P2_<arm>.npy (NaN outside folds 0 / 3)
    python sub92b.py score                   -> %DC_WORK%/x92/oof2/sub92b.json
CPU 4 threads; locked_v2 asserted absent (rows, stacker inputs).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sub92 as S92  # noqa: E402

F = S92.F
DCW = S92.DCW
OUT = S92.X92 / "oof2"
FOLDS = (0, 3)
NET11 = "x92_siba11"
ROWS11 = DCW / "tcn53" / "func_rows_v4o_sub11.parquet"
log = S92.log


def stage_rows(a):
    r = pd.read_parquet(DCW / "tcn53" / "func_rows_v4o.parquet")
    L = S92.sub_lookup("sub")
    k = r[["DeviceId", "Detector"]].assign(DeviceId=r.DeviceId.str.lower())
    assert not k.DeviceId.isin(S92.locked()).any()
    s = k.merge(L.astype({"Detector": k.Detector.dtype})[["DeviceId", "Detector", "sub"]].drop_duplicates(
        ["DeviceId", "Detector"]), on=["DeviceId", "Detector"], how="left")["sub"].to_numpy(object)
    assert len(s) == len(r)
    m = (r.y.to_numpy(object) == "Other") & pd.notna(s)
    r["y"] = np.where(m, s, r.y.to_numpy(object))
    r.to_parquet(ROWS11, index=False)
    log(f"{ROWS11.name}: {int(m.sum()):,} Other rows -> subclass ({int((m & r.ok).sum()):,} usable); "
        f"usable classes {r[r.ok].y.value_counts().to_dict()}")


def net11_probs(fr, cls):
    """frame-aligned 11-class probabilities of the new net (folds 0 / 3; NaN elsewhere)."""
    PC = [f"P_{c}" for c in cls]
    P = np.full((len(fr), len(cls)), np.nan)
    for k in FOLDS:
        n = pd.read_parquet(DCW / "tcn53" / "fpreds" / f"{NET11}_f{k}.parquet")
        assert list(n.columns[4:]) == PC, n.columns
        n["DeviceId"] = n.DeviceId.str.lower()
        key = ["DeviceId", "Detector", "period", "win"]
        f = fr[key].assign(DeviceId=fr.DeviceId.str.lower()).merge(
            n.astype({"Detector": fr.Detector.dtype}), on=key, how="left")[PC].to_numpy(float)
        fk = fr.fold.to_numpy() == k
        P[fk] = f[fk]
    return P


def stage_stack(a):
    import lightgbm as lgb
    import fit84 as F4
    OUT.mkdir(parents=True, exist_ok=True)
    s74, S, E, X, y, trm, names = S92._stack_setup("v4f" if a.arm == "v4f2" else "sub")
    fr = E["fr"]
    fo = fr.fold.to_numpy()
    if a.arm == "v4f2":
        Xk, yk, K = X, y, 7
    else:
        cls = json.load(open(S92.X92 / "trees" / "sub" / "classes.json"))["classes"]
        sub = S92.keyed(fr, S92.sub_lookup("sub"), "sub")
        tr = fr.truth_v3s.to_numpy(object)
        yk = y.copy()
        m = (tr == "Other") & pd.notna(sub)
        yk[m] = [cls.index(s) for s in sub[m]]
        K = len(cls)
        if a.arm == "sub2":
            Xk = np.hstack([X, S92._SUBP["P"]])
        else:
            Pn = net11_probs(fr, cls)
            cov = ~np.isnan(Pn[:, 0])
            log(f"net11 coverage on folds {FOLDS}: {cov[np.isin(fo, FOLDS)].mean():.4f}")
            P7 = np.concatenate([Pn[:, :6], Pn[:, 6:7] + Pn[:, 7:].sum(1, keepdims=True)], 1)
            Xn = F4.net_X(s74, S, E, P7)                 # NaN rows -> trees (fit84.net_X), as every arm
            Psub = np.where(np.isnan(Pn[:, 7:]), np.nan, Pn[:, 7:])
            Xk = np.hstack([Xn, S92._SUBP["P"], Psub])
    log(f"{a.arm}: X {Xk.shape}, K {K}, train rows on folds {FOLDS}: {int((trm & np.isin(fo, FOLDS)).sum()):,}")
    R = np.full((len(y), K), np.nan)
    for k in FOLDS:
        R[fo == k] = 0.0
    for seed in (0, 1, 2):
        for k in FOLDS:
            j = [f for f in FOLDS if f != k][0]
            tr_, te = trm & (fo == j), fo == k
            mm = lgb.train(dict(F.PRM0, num_class=K, num_threads=4, seed=seed), lgb.Dataset(Xk[tr_], yk[tr_]),
                           num_boost_round=150)
            R[te] += mm.predict(Xk[te]) / 3
        log(f"  seed {seed} done")
    np.save(OUT / f"P2_{a.arm}.npy", R.astype(np.float32))


def stage_score(a):
    import cand64 as C
    import of77
    s74, S, E, X, y, trm, names = S92._stack_setup("v4f")
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    fo = fr.fold.to_numpy()
    cls = S92.classes()
    C7 = S92.C7
    Pv4f = np.load(S92.F90 / "P_v4f.npy").astype(float)
    pad = np.hstack([Pv4f, np.zeros((len(fr), len(cls) - 7))])

    def fill(P, ref):                                   # rows outside the screen folds: v4f (never scored)
        return np.where(np.isnan(P[:, :1]), ref, P)
    ok, pred = {}, {}

    def kway_sum(name, PK):
        o, p = S92.gate_pred(s74, S, E, fill(PK, pad), lc, cls)
        ok[f"{name}_kway"], pred[f"{name}_kway"] = o, np.array(cls, object)[p]
        PKf = fill(PK, pad)
        P7 = np.concatenate([PKf[:, :6], PKf[:, 6:7] + PKf[:, 7:].sum(1, keepdims=True)], 1)
        o, p = S92.gate_pred(s74, S, E, P7, lc, C7)
        lab = np.array(C7, object)[p]
        best = np.array(["Other"] + cls[7:], object)[PKf[:, 6:].argmax(1)]
        ok[f"{name}_sum"], pred[f"{name}_sum"] = o, np.where(lab == "Other", best, lab)
    o, p = S92.gate_pred(s74, S, E, Pv4f, lc, C7)
    ok["v4f"], pred["v4f"] = o, np.array(C7, object)[p]
    o, p = S92.gate_pred(s74, S, E, fill(np.load(OUT / "P2_v4f2.npy").astype(float), Pv4f), lc, C7)
    ok["v4f2"], pred["v4f2"] = o, np.array(C7, object)[p]
    kway_sum("sub6f", np.load(S92.OOF / "PK_sub.npy").astype(float))       # note 92 full six-fold stacker
    for arm in ("sub2", "net11"):
        if (OUT / f"P2_{arm}.npy").exists():
            kway_sum(arm, np.load(OUT / f"P2_{arm}.npy").astype(float))
    arms = list(ok)
    contrasts = [("v4f2", k) for k in arms if k not in ("v4f", "v4f2")] + [("v4f", k) for k in arms if k != "v4f"]
    contrasts += [(f"sub2_{d}", f"net11_{d}") for d in ("kway", "sum") if f"net11_{d}" in ok]
    tsub = np.where(tr == "Other", S92.keyed(fr, S92.sub_lookup("sub"), "sub"), None)
    res = {"folds": list(FOLDS), "arms": arms, "classes": cls}
    scr = np.isin(fo, FOLDS)
    for stt in ("E", "R"):
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            sc = scr & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k][stt]) for k in arms])
            r = {"n": int(sc.sum()), "signals": int(len(set(sig[sc])))}
            for k in arms:
                r[k] = C.acc_ci(ok[k][stt][sc], sig[sc])
            for b0, b1 in contrasts:
                r[f"{b1} - {b0}"] = C.delta_ci(ok[b0][stt][sc], ok[b1][stt][sc], sig[sc])
            res[f"{stt}_{pool}"] = r
        sc = scr & fr.wgroup.isin(C.GE30).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k][stt]) for k in arms])
        cl = np.where(np.isin(tr, S92.ATS), tr, "nonATSPM")
        res[f"{stt}_ge30_by_class_vs_v4f2"] = {c: {k: [int((sc & (cl == c)).sum())] + C.delta_ci(
            ok["v4f2"][stt][sc & (cl == c)], ok[k][stt][sc & (cl == c)], sig[sc & (cl == c)])
            for k in arms if k not in ("v4f", "v4f2")} for c in S92.ATS + ["nonATSPM"]}
        up = sc & (tsub == "Upstream_presence")
        res[f"{stt}_ge30_upstream_called_advance"] = {
            "rows": int(up.sum()), "detectors": int(len(set(zip(sig[up], fr.Detector.to_numpy()[up])))),
            **{k: {"rows_called_Advance": int((pred[k][up] == "Advance").sum()),
                   "rows_called_ATSPM": int(np.isin(pred[k][up], S92.ATS).sum()),
                   "exact_Upstream": round(float(np.mean(pred[k][up] == "Upstream_presence")), 4)} for k in arms}}
        for c in cls[7:]:
            mm = sc & (tsub == c)
            res.setdefault(f"{stt}_ge30_subclass_exact", {})[c] = {
                "rows": int(mm.sum()), **{k: round(float(np.mean(pred[k][mm] == c)), 4) for k in arms}}
    json.dump(res, open(OUT / "sub92b.json", "w"), indent=1, default=str)
    for k, v in res.items():
        log(f"{k}: {json.dumps(v, default=str)[:1800]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rows", "stack", "score"])
    ap.add_argument("--arm", default="net11", choices=["v4f2", "sub2", "net11"])
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
