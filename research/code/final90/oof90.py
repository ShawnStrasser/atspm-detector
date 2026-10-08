"""Note 90: function side of the FINAL package v4f -- every function model refitted on the final training labels v4o
(research/labels/function_labels_v4o.parquet; restored training values in <col>_train, plain columns = v4l truth).

OOF (six folds, v4l truth, gate .9 decode, paired signal bootstrap; same 47 stacker columns / PRM0 150 rounds / stacker
seeds 0,1,2 as note 86):
  v4e   note 86 P_n86 = v4m trees OOF, mean3 stacker trained + applied on the filtered 3-seed-mean x86_siba4l OOF (loaded)
  v4f   the same recipe on the v4o trees OOF (f76/function_c_v4o = note 89b's 3-seed not_checkable arm)   <- candidate
  v4d / champion   references (note 88 P_v4m; note-81 champion)

    set F76_ARM=c & python oof90.py stack       -> %DC_WORK%/final_v3_work/f90/oof/P_v4f.npy
    set F76_ARM=c & python oof90.py score       -> .../oof90.json
    set F76_ARM=c & python oof90.py pkgfunc | pkglanes | pkgsbfeats | pkgsetback | pkgsingle   (oof88 recipes, PKG=v4o)
    set F76_ARM=c & python oof90.py pkgstacker  (note-86 mean3 recipe on all training rows, v4o trees)
Package fits go to %DC_WORK%/final_v3_work/v3fit90 (working copies f90/ln8, f90/sb7); v3fit88 / f88 untouched.
CPU, 4 threads.  locked_v2 asserted absent (fit83.stacker_inputs, oof88.check_truth_same).  One stage per process.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
os.environ["DC_PKG_LABELS"] = "v4o"
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
for _d in ("final84", "final83", "final88", "final86"):
    sys.path.insert(0, str(CODE / _d))
import fit84 as F4  # noqa: E402
import fit83 as F  # noqa: E402
import oof88 as O88  # noqa: E402

assert O88.PKG == "v4o"
DCW = F.DC_WORK
F90 = DCW / "final_v3_work" / "f90"
OOF = F90 / "oof"
O88.FIT = DCW / "final_v3_work" / "v3fit90"
O88.F88 = F90                     # ln8 / sb7 working copies for the package stages
F86 = DCW / "final_v3_work" / "f86" / "oof"
F88 = DCW / "final_v3_work" / "f88" / "oof"
LAB80 = DCW / "lab80"
NET = "x86_siba4l"


def use_v4o():
    import v3_retrain as V
    import atspm_score as AS
    O88.check_truth_same()
    V.LABEL_SETS["v3s"] = V.LABEL_SETS["v4o"]
    AS.V3S = O88.LAB["v4o"]
    F.LAB_V4L = O88.LAB["v4o"]
    O88.use_trees("v4o")


def inputs():
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    assert Path(F.CFGD).resolve() == Path(O88.TREES["v4o"] / O88.CFG_SUB).resolve()
    Pf = s74.net_probs(E["fr"], NET)                      # 3-seed mean, filtered (tree p >= .01), as note 86
    F.log(f"{NET}: coverage {np.mean(~np.isnan(Pf[:, 0])):.4f}")
    return s74, S, E, F4.net_X(s74, S, E, Pf), y, trm, names


def stage_stack(a):
    import lightgbm as lgb
    OOF.mkdir(parents=True, exist_ok=True)
    s74, S, E, X, y, trm, names = inputs()
    fo = E["fr"].fold.to_numpy()
    R = np.zeros((len(y), 7))
    t0 = time.time()
    for seed in (0, 1, 2):
        for k in range(6):
            tr, te = trm & (fo != k), fo == k
            m = lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=seed), lgb.Dataset(X[tr], y[tr]), num_boost_round=150)
            R[te] += m.predict(X[te]) / 3
        F.log(f"  stacker seed {seed} done ({time.time() - t0:.0f}s)")
    np.save(OOF / "P_v4f.npy", R.astype(np.float32))
    np.save(OOF / "Pt_v4o.npy", E["Pt"].astype(np.float32))
    json.dump({"train_rows": int(trm.sum()), "rows": int(len(y))}, open(OOF / "stack_rows.json", "w"))


def stage_score(a):
    import cand64 as C
    import of77
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    champ = np.load(LAB80 / "ok_v4l_champ.npz")
    assert (champ["sig"] == fr.DeviceId.to_numpy(str)).all() and (champ["win"] == fr.win.to_numpy(str)).all()
    P = {"v4d": np.load(F88 / "P_v4m.npy").astype(float), "v4e": np.load(F86 / "P_n86.npy").astype(float),
         "v4f": np.load(OOF / "P_v4f.npy").astype(float)}
    ok = {k: s74.gate_ok(E, v, lc) for k, v in P.items()}
    ok["champion"] = {st: champ[st].astype(float) for st in ("E", "R")}
    arms = list(ok)
    contrasts = [("v4e", "v4f"), ("v4d", "v4f"), ("champion", "v4f"), ("v4d", "v4e")]
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    lab = np.array([t in E["C7"] for t in tr]) & fr.wgroup.isin(C.GE30).to_numpy()
    Pt = {"v4o": np.load(OOF / "Pt_v4o.npy")}
    if (F88 / "Pt_v4m.npy").exists():
        Pt["v4m"] = np.load(F88 / "Pt_v4m.npy")
    res = {"arms": arms, "trees_alone_ge30_labelled_acc7": {
        t: round(float(np.mean(np.array(E["C7"])[v[lab].argmax(1)] == tr[lab])), 4) for t, v in Pt.items()}}
    for st in ("E", "R"):
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            sc = pm & np.logical_and.reduce([~np.isnan(ok[k][st]) for k in arms])
            r = {"n": int(sc.sum()), "signals": int(len(set(sig[sc])))}
            for k in arms:
                r[k] = C.acc_ci(ok[k][st][sc], sig[sc])
            for b0, b1 in contrasts:
                r[f"{b1} - {b0}"] = C.delta_ci(ok[b0][st][sc], ok[b1][st][sc], sig[sc])
            res[f"{st}_{pool}"] = r
        m = fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok["v4f"][st]) & ~np.isnan(ok["v4e"][st])
        res[f"{st}_ge30_by_class_v4f_minus_v4e"] = {
            c: [int((m & (cl == c)).sum())] + C.delta_ci(ok["v4e"][st][m & (cl == c)], ok["v4f"][st][m & (cl == c)],
                                                        sig[m & (cl == c)])
            for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        res[f"{st}_ge30_by_fold_v4f_minus_v4e"] = {
            int(k): round(100 * float(np.nanmean(ok["v4f"][st][m & (fr.fold.to_numpy() == k)])
                                      - np.nanmean(ok["v4e"][st][m & (fr.fold.to_numpy() == k)])), 3) for k in range(6)}
    json.dump(res, open(OOF / "oof90.json", "w"), indent=1, default=str)
    for k, v in res.items():
        F.log(f"{k}: {v}")


def stage_tcnphase(a):
    """orchestrator cleanup: re-score the v4f function OOF with the frame's phase input taken from the v4f PHASE pipeline
    (top of the TCN-blend decode p2_tcn_ad76, s90/p87_q.parquet) instead of the frame's GRU-era pred_phase, where the
    phase pool covers the row (62 %; elsewhere the frame phase stays).  Changed rows lose their (old-phase) lane string and
    lane context.  Re-done: stacker context (phase-mates, ranks, lanes), the six-fold mean3 stacker (same recipe), the gate
    .9 decode (grouped by phase).  Held at the frame phase: the 229 tree features / Pt, pick inputs, twin tokens.
    Control arm 'gru': the same replacement with the GRU-pipeline top (should reproduce v4f)."""
    import lightgbm as lgb
    import pandas as pd
    import cand64 as C
    import of77
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    sig = fr.DeviceId.to_numpy()
    lc0 = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    Pf = s74.net_probs(fr, NET)
    P0 = np.load(OOF / "P_v4f.npy").astype(float)
    ok = {"v4f": s74.gate_ok(E, P0, lc0)}
    q = pd.read_parquet(DCW / "s90" / "p87_q.parquet", columns=["DeviceId", "Detector", "win", "cand_phase", "p2_gru",
                                                                 "p2_tcn_ad76"])
    k = ["DeviceId", "Detector", "win"]
    tops = {}
    for c in ("p2_gru", "p2_tcn_ad76"):
        tops[c] = q.sort_values(k + [c], ascending=[True, True, True, False]).groupby(k, sort=False).first().cand_phase
    t = pd.DataFrame(tops).reset_index()
    t["period"] = np.where(t.DeviceId.str.endswith("@stg"), "stg", "dec")
    t["DeviceId"] = t.DeviceId.str.replace("@stg", "", regex=False).str.lower()
    key = fr[["DeviceId", "period", "Detector", "win"]].copy()
    key["DeviceId"] = key.DeviceId.str.lower()
    m = key.merge(t.astype({"Detector": fr.Detector.dtype}), on=["DeviceId", "period", "Detector", "win"], how="left")
    assert len(m) == len(fr)
    pp0, lanes0, LN0 = fr.pred_phase.copy(), fr.lanes5g.copy(), S._CTX["ln"].copy()
    fo = fr.fold.to_numpy()
    res = {"rows": int(len(fr)), "covered": round(float(m.p2_gru.notna().mean()), 4)}
    ge30 = fr.wgroup.isin(C.GE30).to_numpy()
    for arm, col in (("gru", "p2_gru"), ("tcn", "p2_tcn_ad76")):
        new = np.where(m[col].notna(), m[col].to_numpy(float), pp0.to_numpy(float))
        ch = pp0.notna().to_numpy() & (new != pp0.to_numpy(float))
        fr["pred_phase"] = new
        fr["lanes5g"] = lanes0.where(~ch, None)
        S._CTX["ln"] = np.where(ch[:, None], np.nan, LN0)
        X = F4.net_X(s74, S, E, Pf)
        R = np.zeros((len(y), 7))
        for seed in (0, 1, 2):
            for kf in range(6):
                tr, te = trm & (fo != kf), fo == kf
                mm = lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=seed), lgb.Dataset(X[tr], y[tr]),
                               num_boost_round=150)
                R[te] += mm.predict(X[te]) / 3
        np.save(OOF / f"P_v4f_{arm}phase.npy", R.astype(np.float32))
        ok[arm] = s74.gate_ok(E, R, np.where(ch, np.nan, lc0))
        res[f"changed_{arm}"] = {"all": int(ch.sum()), "ge30": int((ch & ge30).sum())}
        F.log(f"{arm}: phase input changed on {res[f'changed_{arm}']}")
        fr["pred_phase"], fr["lanes5g"], S._CTX["ln"] = pp0, lanes0, LN0
    for st in ("E", "R"):
        for pool, fams in C.POOLS.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            sc = pm & np.logical_and.reduce([~np.isnan(v[st]) for v in ok.values()])
            r = {"n": int(sc.sum())}
            for kk in ok:
                r[kk] = C.acc_ci(ok[kk][st][sc], sig[sc])
            for kk in ("gru", "tcn"):
                r[f"{kk} - v4f"] = C.delta_ci(ok["v4f"][st][sc], ok[kk][st][sc], sig[sc])
            res[f"{st}_{pool}"] = r
    json.dump(res, open(OOF / "tcnphase90.json", "w"), indent=1, default=str)
    for kk, v in res.items():
        F.log(f"{kk}: {v}")


# ------------------------------------------------------------------------------------------- package fits (v4o)
def stage_pkgstacker(a):
    """mean3 stacker = note-86 recipe (filtered 3-seed-mean x86_siba4l OOF) on ALL training rows, v4o trees OOF."""
    import lightgbm as lgb
    O88._pkg_out()
    d = O88.FIT / "stacker"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("stacker_mean3*"):
        f.unlink()
    s74, S, E, X, y, trm, names = inputs()
    F.log(f"stacker mean3 (note 90: v4o trees OOF, filtered x86_siba4l mean): X {X[trm].shape}")
    np.save(d / "X_check_mean3.npy", X[:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_mean3_s{s}.txt"
        lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=s), lgb.Dataset(X[trm], y[trm]),
                  num_boost_round=150).save_model(str(f))
        files.append(f.name)
        F.log(f"  seed {s} done")
    fr = E["fr"]
    json.dump({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files,
               "params": F.PRM0, "num_boost_round": 150,
               "variant": "mean3 (note 90): trained on the 3-seed MEAN of the v4l siba OOF fold models (x86_siba4l), "
                          "filtered (tree p >= .01) OOF",
               "net": "siba x86_siba4l 3-seed-mean OOF fold models (v4l-trained); production = 3 full-data siba members",
               "labels": "truth = v4l (= v4o plain columns); trees = v4o OOF (f76/function_c_v4o, note 89b)",
               "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique())}},
              open(d / "stacker_mean3.json", "w"), indent=1)


def _delegate(name):
    def f(a):
        getattr(O88, f"stage_{name}")(a)
    return f


for _n in ("pkgfunc", "pkglanes", "pkgsbfeats", "pkgsetback", "pkgsingle"):
    globals()[f"stage_{_n}"] = _delegate(_n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stack", "score", "tcnphase", "pkgfunc", "pkglanes", "pkgsbfeats", "pkgsetback", "pkgsingle",
                                      "pkgstacker"])
    a = ap.parse_args()
    F90.mkdir(parents=True, exist_ok=True)
    if a.stage in ("stack", "score", "tcnphase", "pkgstacker", "pkgsingle"):
        use_v4o()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    F.log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
