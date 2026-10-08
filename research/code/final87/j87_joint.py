"""Note 87b B: ONE joint decider for phase and function -- each second-level tree sees the other task's probabilities,
iterated once, nested -- vs the current separate deciders (saved OOF only, CPU 4 threads).

Round 1 (current, unchanged):
  phase     p1(d, k)  = joint decoder on trees bag + GRU p3 (.01 filter)            (p87 'gru' arm = champion .9818)
  function  f1(d, c)  = mean3 context stacker on mean of the 3 filtered siba nets    (oof84 mean3_flt3 = champion .9192)
Round 2 (the joint decider, one pass):
  phase'    the same decoder recipe + 13 function columns per (detector, candidate): own f1 (7 classes) and the
            candidate's soft class composition S_c(k) = sum over the OTHER detectors d' of p1(d', k) * f1(d', c) for c in
            Advance / Presence / Count / Yellow_Red / non-ATSPM, plus S_n(k) = sum p1(d', k) (expected phase-mates)
  function' the same stacker recipe + 11 phase columns per detector: p1 top / margin / entropy / n candidates, frame phase
            == p1 top, and the expected phase-mate class mass M_c = sum_k p1(d, k) S_c(k) (5 + n)
Nesting: for held-out fold k, every round-1 input on the TRAINING folds is re-computed without fold k (inner OOF: the
decoder / stacker for training fold j fitted on folds not in {k, j}); fold k's inputs are the standard round-1 OOF.  So
fold k's labels never reach its round-2 model, directly or through a round-1 input.  (The base learners -- phase trees,
GRU, function trees, siba -- are standard OOF as in notes 64-84; not re-nested.)
Without the extra columns round 2 reproduces round 1 exactly (same seeds), so the delta is the joint information only.

    set F76_ARM=c & python j87_joint.py inner      -> %DC_WORK%/s87/joint/{Pin,Fin}_k*.npy
    set F76_ARM=c & python j87_joint.py joint [--shuffle]  -> P2 / F2 arrays
    set F76_ARM=c & python j87_joint.py score      -> %DC_WORK%/s87/joint/j87.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "final84"))
sys.path.insert(0, str(HERE.parent / "final83"))
sys.path.insert(0, str(HERE))
import fit84 as F4  # noqa: E402
import fit83 as F  # noqa: E402
import cand64 as C  # noqa: E402
import p87_phase as P87  # noqa: E402

NJ = 4
OUT = C.DC_WORK / "s87" / "joint"
F84 = C.DC_WORK / "final_v3_work" / "f84" / "oof"
KEY4, DET = C.KEY4, C.DET
G5 = ["A", "P", "C", "Y", "O"]
_E = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------------ data
def func_data():
    """frame E, stacker matrices Xm (train, unfiltered mean3) / Xf (apply, filtered mean3), y, trm, round-1 F1."""
    if "f" in _E:
        return _E["f"]
    s74, S, E, Xs, Xm, y, trm, names = F4.mean3_inputs()
    fr = E["fr"]
    fl = [s74.net_probs(fr, f"x74_sibaflt:{s}") for s in range(3)]
    Xf = F4.net_X(s74, S, E, (fl[0] + fl[1] + fl[2]) / 3)
    F1 = np.load(F84 / "P_mean3_flt3.npy").astype(float)
    assert F1.shape == (len(fr), 7)
    _E["f"] = (s74, S, E, Xm, Xf, y, trm, F1)
    return _E["f"]


def phase_data():
    if "p" in _E:
        return _E["p"]
    need = KEY4 + ["fold", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(P87.F76P / "pool.parquet", columns=need)
    simc = pd.read_parquet(P87.F76P / "pool_sim.parquet")
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    q = P87.base()
    for c in KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all(), c
    p1 = pd.read_parquet(C.DC_WORK / "s87" / "p87_q.parquet", columns=["p2_gru"]).p2_gru.to_numpy()
    p0 = P87.blend(q, q.p0t.to_numpy(), q.p_nn.to_numpy())
    import decode_train as dec
    pr = comb[KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[KEY4 + ["Phase", "fold", "y"]], on=KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    pos = comb[KEY4].reset_index().merge(X[KEY4].reset_index().rename(columns={"index": "xi"}), on=KEY4, how="left")
    xi = pos.sort_values("index").xi.to_numpy()          # comb row i -> X row xi[i]
    _E["p"] = (comb, X, xi, p1, dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS)
    return _E["p"]


def dec_fit_predict(X, cols, train, valid, pred):
    import lightgbm as lgb
    import train_official as T
    P = dict(T.BIN_PARAMS, n_jobs=NJ, bagging_seed=0, feature_fraction_seed=100, seed=0)
    n = P.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n, **P)
    m.fit(X.loc[train, cols], X.y[train], eval_set=[(X.loc[valid, cols], X.y[valid])], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    return m.predict_proba(X.loc[pred, cols])[:, 1]


def dec_round(X, cols, k_out):
    """decoder OOF predictions for fold k_out only (train folds != k_out, early stop on (k+1) % 6) -> normalised,
    X order; NaN elsewhere.  Same recipe as cand64.decode for that fold."""
    import train_official as T
    fx = X.fold.to_numpy()
    lab = X.Phase.notna().to_numpy()
    te = fx == k_out
    inner = (k_out + 1) % 6
    s = np.zeros(len(X))
    s[te] = dec_fit_predict(X, cols, (~te) & lab & (fx != inner), (~te) & lab & (fx == inner), te)
    return np.where(te, T.norm_prob(X, s), np.nan)


def dec_inner(X, cols, k, j):
    """decoder for training fold j of outer fold k: fitted on folds not in {k, j}; X order, normalised on fold j."""
    import train_official as T
    fx = X.fold.to_numpy()
    lab = X.Phase.notna().to_numpy()
    rest = [f for f in range(6) if f not in (k, j)]
    inner = next(f for f in [(j + i) % 6 for i in range(1, 6)] if f in rest)
    te = fx == j
    s = np.zeros(len(X))
    tr = lab & np.isin(fx, rest)
    s[te] = dec_fit_predict(X, cols, tr & (fx != inner), tr & (fx == inner), te)
    return np.where(te, T.norm_prob(X, s), np.nan)


def stack_fit_predict(Xtr, ytr, Xte):
    import lightgbm as lgb
    out = 0
    for seed in (0, 1, 2):
        m = lgb.train(dict(F.PRM0, num_threads=NJ, seed=seed), lgb.Dataset(Xtr, ytr), num_boost_round=150)
        out = out + m.predict(Xte) / 3
    return out


# ------------------------------------------------------------------------------------------------ inner OOF
def stage_inner(a):
    OUT.mkdir(parents=True, exist_ok=True)
    s74, S, E, Xm, Xf, y, trm, F1 = func_data()
    fo = E["fr"].fold.to_numpy()
    comb, X, xi, p1, cols = phase_data()
    fx = X.fold.to_numpy()
    chk = OUT / "checks.json"
    if not chk.exists():                     # the recipes reproduce round 1 on fold 0
        te = fo == 0
        tr = trm & ~te
        d_f = float(np.abs(stack_fit_predict(Xm[tr], y[tr], Xf[te]) - F1[te]).max())
        v = dec_round(X, cols, 0)
        d_p = float(np.nanmax(np.abs(v[xi] - p1)[comb.fold.to_numpy() == 0]))
        json.dump({"stacker_fold0_max_abs_vs_P_mean3_flt3": d_f, "decoder_fold0_max_abs_vs_p2_gru": d_p}, open(chk, "w"))
        log(f"checks: stacker {d_f:.2e}, decoder {d_p:.2e}")
    for k in range(6):
        fF, fP = OUT / f"Fin_k{k}.npy", OUT / f"Pin_k{k}.npy"
        if not fF.exists():
            t0 = time.time()
            Fin = F1.copy()
            for j in range(6):
                if j == k:
                    continue
                tr, te = trm & (fo != k) & (fo != j), fo == j
                Fin[te] = stack_fit_predict(Xm[tr], y[tr], Xf[te])
            np.save(fF, Fin.astype(np.float32))
            log(f"Fin k{k} ({time.time() - t0:.0f}s)")
        if not fP.exists():
            t0 = time.time()
            px = p1[np.argsort(xi)]                              # X order
            assert np.allclose(px[xi], p1)
            Pin = px.copy()
            for j in range(6):
                if j == k:
                    continue
                v = dec_inner(X, cols, k, j)
                Pin[fx == j] = v[fx == j]
            np.save(fP, Pin[xi].astype(np.float64))          # back to comb order
            log(f"Pin k{k} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ joint features
def link(comb, fr):
    """comb row -> frame row (or -1) and frame row -> its comb rows."""
    pid = fr.DeviceId.str.lower() + np.where(fr.period.to_numpy() == "stg", "@stg", "")
    fk = pd.DataFrame({"DeviceId": pid, "Detector": fr.Detector.astype(comb.Detector.dtype).to_numpy(),
                       "win": fr.win.to_numpy(), "fi": np.arange(len(fr))})
    assert not fk.duplicated(DET).any()
    m = comb[DET].merge(fk, on=DET, how="left")
    return m.fi.fillna(-1).astype(int).to_numpy()


def group5(F7, C7):
    i = {c: C7.index(c) for c in C7}
    o = F7[:, i["Mid"]] + F7[:, i["Bike"]] + F7[:, i["Other"]]
    return np.column_stack([F7[:, i["Advance"]], F7[:, i["Presence"]], F7[:, i["Count"]], F7[:, i["Yellow_Red"]], o])


def joint_feats(comb, fr, fi, p1, F7, C7, pred_phase):
    """phase extra (comb rows, 13 cols) and function extra (frame rows, 11 cols) from round-1 p1 / F7."""
    has = fi >= 0
    f5 = np.where(has[:, None], group5(F7, C7)[np.clip(fi, 0, None)], 0.0)
    f7 = np.where(has[:, None], F7[np.clip(fi, 0, None)], np.nan)
    g = comb.groupby(["DeviceId", "win", "cand_phase"], sort=False).ngroup().to_numpy()
    contrib = p1[:, None] * f5
    S = np.column_stack([np.bincount(g, contrib[:, c])[g] - contrib[:, c] for c in range(5)])
    Sn = np.bincount(g, p1)[g] - p1
    ph = np.column_stack([f7, S, Sn])                                                     # 7 + 5 + 1
    # function side: per detector-window of the pool
    dg = comb.groupby(DET, sort=False).ngroup().to_numpy()
    nd = dg.max() + 1
    M = np.column_stack([np.bincount(dg, p1 * S[:, c], nd) for c in range(5)] + [np.bincount(dg, p1 * Sn, nd)])
    d = pd.DataFrame({"dg": dg, "p": p1, "c": comb.cand_phase.to_numpy()})
    d = d.sort_values(["dg", "p"], ascending=[True, False])
    first = d.groupby("dg").head(1).set_index("dg")
    second = d.groupby("dg").nth(1).set_index("dg").p.reindex(range(nd)).fillna(0).to_numpy()
    top = first.p.reindex(range(nd)).to_numpy()
    topc = first.c.reindex(range(nd)).to_numpy()
    ent = np.bincount(dg, -p1 * np.log(np.clip(p1, 1e-12, 1)), nd)
    ncand = np.bincount(dg, minlength=nd).astype(float)
    dfi = np.full(nd, -1)
    dfi[dg] = fi
    fx = np.full((len(fr), 11), np.nan)
    ok = dfi >= 0
    agree = (topc == pred_phase[np.clip(dfi, 0, None)]).astype(float)
    fx[dfi[ok]] = np.column_stack([top, top - second, ent, ncand, agree, M])[ok]
    return ph.astype(np.float32), fx.astype(np.float32)


PH_COLS = [f"j_f_{c}" for c in ("A", "P", "C", "Y", "M", "B", "O")] + [f"j_S_{c}" for c in G5] + ["j_Sn"]


def stage_joint(a):
    tag = "shuf" if a.shuffle else "main"
    s74, S, E, Xm, Xf, y, trm, F1 = func_data()
    fr, C7 = E["fr"], E["C7"]
    assert C7 == ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"], C7
    fo = fr.fold.to_numpy()
    comb, X, xi, p1, cols = phase_data()
    fi = link(comb, fr)
    assert (comb.fold.to_numpy()[fi >= 0] == fo[fi[fi >= 0]]).all()
    log(f"pool rows with a function row {np.mean(fi >= 0):.3f}; frame rows with pool rows "
        f"{len(np.unique(fi[fi >= 0])) / len(fr):.3f}")
    pp = fr.pred_phase.to_numpy(float)
    P2 = np.full(len(comb), np.nan)
    F2 = np.zeros((len(fr), 7))
    rng = np.random.default_rng(87)
    wg = fr.wgroup.to_numpy()
    for k in range(6):
        t0 = time.time()
        Pin = np.load(OUT / f"Pin_k{k}.npy")
        Fin = np.load(OUT / f"Fin_k{k}.npy").astype(float)
        ph, fx = joint_feats(comb, fr, fi, Pin, Fin, C7, pp)
        if a.shuffle:                       # control: the joint columns permuted among rows of the same window family
            fam = comb.win.map(C.fam_of).to_numpy()
            for f in np.unique(fam):
                ix = np.flatnonzero(fam == f)
                ph[ix] = ph[rng.permutation(ix)]
            for f in np.unique(wg):
                ix = np.flatnonzero(wg == f)
                fx[ix] = fx[rng.permutation(ix)]
        Xk = X.copy()
        Xk[PH_COLS] = ph[np.argsort(xi)]
        assert np.array_equal(Xk[PH_COLS].to_numpy()[xi], ph, equal_nan=True)
        v = dec_round(Xk, cols + PH_COLS, k)
        te = comb.fold.to_numpy() == k
        P2[te] = v[xi][te]
        tr, tf = trm & (fo != k), fo == k
        F2[tf] = stack_fit_predict(np.hstack([Xm, fx])[tr], y[tr], np.hstack([Xf, fx])[tf])
        log(f"[{tag}] outer fold {k} done ({time.time() - t0:.0f}s)")
    np.save(OUT / f"P2_{tag}.npy", P2)
    np.save(OUT / f"F2_{tag}.npy", F2.astype(np.float32))


# ------------------------------------------------------------------------------------------------ score
def stage_score(a):
    import of77
    res = json.load(open(OUT / "j87.json")) if (OUT / "j87.json").exists() else {}
    tags = [t for t in ("main", "shuf") if (OUT / f"P2_{t}.npy").exists()]
    # phase
    q = P87.base()
    done = pd.read_parquet(C.DC_WORK / "s87" / "p87_q.parquet", columns=["p2_gru"])
    q["p1"] = done.p2_gru.to_numpy()
    for t in tags:
        q[f"p2_{t}"] = np.load(OUT / f"P2_{t}.npy")
        assert q[f"p2_{t}"].notna().all()
    rows = C.phase_rows()
    rows = rows[[c for c in rows.columns if not c.startswith(("pred_", "okE_", "okR_", "p_"))]].copy()
    import t57_phase as T57
    alt = T57.score_rows().set_index(DET).alt.reindex(rows.set_index(DET).index).to_numpy()
    sig = rows.dev_plain.to_numpy()
    arms = ["p1"] + [f"p2_{t}" for t in tags]
    for nm in arms:
        pred, _ = C.top1(q, nm, rows)
        rows[f"okE_{nm}"] = (pred == rows.Phase.to_numpy()).astype(float)
        rows[f"okR_{nm}"] = (rows[f"okE_{nm}"].astype(bool) | np.array([p in s_ for p, s_ in zip(pred, alt)])).astype(float)
    ph = {}
    pools = dict(C.POOLS, **{f: [f] for f in P87.FAMS})
    for st, okp in (("E", "okE"), ("R", "okR")):
        mset = rows["everything" if st == "E" else "realistic"].to_numpy()
        for pool, fams in pools.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            r = {"n": int(m.sum())}
            for nm in arms:
                r[nm] = C.acc_ci(rows[f"{okp}_{nm}"].to_numpy()[m], sig[m])
                if nm != "p1":
                    r[f"{nm} - p1"] = C.delta_ci(rows[f"{okp}_p1"].to_numpy()[m], rows[f"{okp}_{nm}"].to_numpy()[m], sig[m])
            ph[f"{st}_{pool}"] = r
        m30 = mset & rows.fam.isin(C.GE30).to_numpy()
        ph[f"{st}_ge30_by_fold"] = {k: {nm: round(100 * float(rows[f"{okp}_{nm}"].to_numpy()[m30 & (rows.fold == k).to_numpy()].mean()
                                                            - rows[f"{okp}_p1"].to_numpy()[m30 & (rows.fold == k).to_numpy()].mean()), 3)
                                        for nm in arms if nm != "p1"} for k in range(6)}
    res["phase"] = ph
    # function
    s74, S, E, Xm, Xf, y, trm, F1 = func_data()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    champ = np.load(C.DC_WORK / "lab80" / "ok_v4l_champ.npz")
    ok = {"f1": s74.gate_ok(E, F1, lc)}
    for t in tags:
        ok[f"f2_{t}"] = s74.gate_ok(E, np.load(OUT / f"F2_{t}.npy").astype(float), lc)
    fsig = fr.DeviceId.to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    fu = {}
    for st in ("E", "R"):
        for pool, fams in pools.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            sc = pm & np.logical_and.reduce([~np.isnan(ok[x][st]) for x in ok])
            r = {"n": int(sc.sum())}
            for x in ok:
                r[x] = C.acc_ci(ok[x][st][sc], fsig[sc])
                if x != "f1":
                    r[f"{x} - f1"] = C.delta_ci(ok["f1"][st][sc], ok[x][st][sc], fsig[sc])
            fu[f"{st}_{pool}"] = r
        m = fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok["f1"][st])
        if "f2_main" in ok:
            fu[f"{st}_ge30_by_class"] = {c: [int((m & (cl == c)).sum())] + C.delta_ci(
                ok["f1"][st][m & (cl == c)], ok["f2_main"][st][m & (cl == c)], fsig[m & (cl == c)])
                for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
            fu[f"{st}_ge30_by_fold"] = {k: round(100 * float(np.nanmean(ok["f2_main"][st][m & (fr.fold == k).to_numpy()])
                                                         - np.nanmean(ok["f1"][st][m & (fr.fold == k).to_numpy()])), 3)
                                        for k in range(6)}
    fu["f1_vs_note84_champion_ge30_E"] = float(np.nanmean(ok["f1"]["E"][fr.wgroup.isin(C.GE30).to_numpy()]))
    res["function"] = fu
    json.dump(res, open(OUT / "j87.json", "w"), indent=1, default=str)
    for k, v in res["phase"].items():
        if k.split("_")[1] in C.POOLS or "fold" in k:
            log(f"phase {k}: {v}")
    for k, v in res["function"].items():
        if (len(k.split("_")) > 1 and k.split("_")[1] in C.POOLS) or "class" in k or "fold" in k:
            log(f"function {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inner", "joint", "score"])
    ap.add_argument("--shuffle", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
