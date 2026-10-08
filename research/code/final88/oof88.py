"""Note 88: refit the function chain on labels whose cleansing no longer reads detector fault events 83-88, and score it
against candidate v4b's recipe (notes 84 / 84b: mean3 stacker + siba filter, .9192 E / .9291 R at >= 30 min).

Labels: function_labels_v4m.parquet (dq88.py labels: v4l with dq_suspect recomputed fault-free) + v3_retrain's
nc_high_mask card rule now actuation-only.  Scoring truth stays v4l (truth columns identical in v4m, asserted).

    python oof88.py fit                 229 function trees, 3 seeds x 6 folds (note-57 recipe, t57 setup, frame v6e arm c)
                                        on v4m -> %DC_WORK%/final_v3_work/f76/function_c_v4m/
    python oof88.py stack --trees v4m   mean3 context stacker, six-fold OOF exactly as oof84 mean3_flt3: trained on the
    python oof88.py stack --trees v4l   unfiltered x69_siba 3-seed mean, applied to mean(x74_sibaflt 0,1,2); stacker
                                        seeds 0/1/2; v4l = reproduction check of f84/oof/P_mean3_flt3.npy
                                        -> %DC_WORK%/final_v3_work/f88/oof/P_<trees>.npy
    python oof88.py score               v4c (v4m trees) vs v4b (f84 P_mean3_flt3) and the champion: gate .9 decode, paired
                                        signal bootstrap, >= 30 / 10 / 5 min, E and R, by class, by fold -> f88/oof/oof88.json
    python oof88.py pkgfunc             package: 229 trees full data on v4m (fit83 func recipe) -> v3fit88/function
    python oof88.py pkglanes            package: lanes D on the v4m OOF function block (fit83 lanes recipe) -> v3fit88/lanes
    python oof88.py pkgsbfeats / pkgsetback   setback (fit83 recipe) on the v4m OOF function -> f88/sb7, v3fit88/setback
    python oof88.py pkgsingle           package: 'single' fallback stacker (fit83 recipe), run before pkgstacker
    python oof88.py pkgstacker          package: mean3 stacker (fit84 recipe) on the v4m OOF trees -> v3fit88/stacker
CPU, 4 threads.  F76_ARM=c.  locked_v2 asserted absent (loaders, fit83.stacker_inputs).  One stage per process.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final84"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final83"))
import fit83 as F  # noqa: E402  (copies v3fit77 -> v3fit83 only if missing; sets paths)
import fit84 as F4  # noqa: E402

DCW = F.DC_WORK
F76D = DCW / "final_v3_work" / "f76"
CFG_SUB = Path("v6e") / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx"
TREES = {"v4l": F76D / "function_c_v4l", "v4m": F76D / "function_c_v4m",
         "v4n": F76D / "function_c_v4n",      # note 89: v4m + eight exclusion groups restored (= x89 combined arm)
         "v4o": F76D / "function_c_v4o",      # note 89: v4n + not_checkable restored (= x89/v4n not_checkable arm)
         "v4l81": F76D / "function_c_v4l81"}   # control: the CURRENT v4l table (note-81 R5 / R6 applied after the
#                                              note-80 fit that v4b uses), same code / card rule -> isolates the dq fix
REPO = F.rpath.REPO
LAB = {"v4l": REPO / "research" / "labels" / "function_labels_v4l.parquet",
       "v4m": REPO / "research" / "labels" / "function_labels_v4m.parquet",
       "v4n": REPO / "research" / "labels" / "function_labels_v4n.parquet",
       "v4o": REPO / "research" / "labels" / "function_labels_v4o.parquet",
       "v4q": REPO / "research" / "labels" / "function_labels_v4q.parquet"}   # note 95
# note 89 (orchestrator 2026-10-04): the package stages (pkg*) build on this label table / its OOF trees; v4n = final
PKG = os.environ.get("DC_PKG_LABELS", "v4q")      # note 95: default v4q (was v4o)
assert PKG in ("v4m", "v4n", "v4o", "v4q"), PKG
F88 = DCW / "final_v3_work" / "f88"
OOF = F88 / "oof"
FIT = DCW / "final_v3_work" / "v3fit88"
F84 = DCW / "final_v3_work" / "f84" / "oof"
LAB80 = DCW / "lab80"
log = F.log


def check_truth_same():
    cols = ["DeviceId", "detector", "truth_v3", "truth_v3s", "exclude_train_score", "validated", "stack_group",
            "stack_role", "yr_count_identical", "dec_fp_truth", "label_print_first", "train_use", "train_use_validated"]
    a = pd.read_parquet(LAB["v4l"], columns=cols)
    for t in sorted({"v4m", PKG}):
        b = pd.read_parquet(LAB[t], columns=cols)
        for c in cols:
            assert a[c].astype(str).equals(b[c].astype(str)), f"{t} differs from v4l in {c}"
    lk = F.F75.locked()
    assert not b.DeviceId.str.lower().isin(lk).any()


def use_trees(tag: str):
    """point fit83 / s59 at a trees OOF folder (the stacker's tree input)."""
    F.TREES_V4L = TREES[tag]
    F.CFGD = TREES[tag] / CFG_SUB
    assert (F.CFGD / "timing.jsonl").exists(), F.CFGD


# ============================================================================================ OOF trees
def stage_fit(a):
    import t57_function as T57F
    import f76_function as FF
    import v3_retrain as V
    check_truth_same()
    tag = a.trees
    assert tag in ("v4m", "v4l81"), tag
    V.LABEL_SETS["v3s"] = V.LABEL_SETS["v4m"] if tag == "v4m" else (LAB["v4l"], V.LABEL_SETS["v4m"][1])
    T57F.OUT = TREES[tag]
    T57F.setup("v6e")
    T57F.BASE_RUN = V.run_dir("exclude").name
    TREES[tag].mkdir(parents=True, exist_ok=True)
    T57F.stage_fit(argparse.Namespace(frame="v6e", cfg=FF.CFG, seeds="0,1,2", threads=F.THREADS))
    old = json.load(open(TREES["v4l"] / CFG_SUB / "cols.json"))["cols"]
    new = json.load(open(TREES[tag] / CFG_SUB / "cols.json"))["cols"]
    if tag != "v4m":
        return
    assert old == new, "feature list differs from the v4l 229-feature arm"
    # training-row accounting vs v4l (same frame rows)
    fr, cols = V.load_feats()
    lab = V.load_labels(fr, "exclude")
    _, ok_m = V.variant_target(lab, fr, T57F.VAR)
    V.LABELS_V3 = LAB["v4l"]                     # same Dec-role file; card rule = the new actuation-only one for both
    lab_l = V.load_labels(fr, "exclude")
    _, ok_l = V.variant_target(lab_l, fr, T57F.VAR)
    k = fr[V.KEY].assign(ok_l=ok_l, ok_m=ok_m)
    r = {"train_rows_v4l": int(ok_l.sum()), "train_rows_v4m": int(ok_m.sum()), "added": int((ok_m & ~ok_l).sum()),
         "removed": int((ok_l & ~ok_m).sum()),
         "detectors_added": int(k[k.ok_m & ~k.ok_l][["DeviceId", "Detector"]].drop_duplicates().shape[0])}
    json.dump(r, open(F88 / "train_rows.json", "w"), indent=1)
    log(f"training rows: {r}")


# ============================================================================================ OOF stacker
def stage_stack(a):
    import lightgbm as lgb
    use_trees(a.trees)
    OOF.mkdir(parents=True, exist_ok=True)
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    Xm = F4.net_X(s74, S, E, s74.net_probs(fr, "x69_siba"))                    # unfiltered 3-seed mean (train)
    flt = [s74.net_probs(fr, f"x74_sibaflt:{s}") for s in range(3)]
    Xf = F4.net_X(s74, S, E, (flt[0] + flt[1] + flt[2]) / 3)                     # all three filtered (apply)
    fo = fr.fold.to_numpy()
    R = np.zeros((len(y), 7))
    t0 = time.time()
    for seed in (0, 1, 2):
        for k in range(6):
            tr, te = trm & (fo != k), fo == k
            mm = lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=seed), lgb.Dataset(Xm[tr], y[tr]),
                           num_boost_round=150)
            R[te] += mm.predict(Xf[te]) / 3
        log(f"  stacker seed {seed} done ({time.time() - t0:.0f}s)")
    np.save(OOF / f"P_{a.trees}.npy", R.astype(np.float32))
    np.save(OOF / f"Pt_{a.trees}.npy", E["Pt"].astype(np.float32))
    if a.trees == "v4l":
        old = np.load(F84 / "P_mean3_flt3.npy")
        log(f"reproduction: v4l trees vs f84 P_mean3_flt3 max |diff| {np.abs(R - old).max():.2e}")


def stage_score(a):
    import cand64 as C
    import of77
    use_trees("v4m")
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    champ = np.load(LAB80 / "ok_v4l_champ.npz")
    assert (champ["sig"] == fr.DeviceId.to_numpy(str)).all() and (champ["win"] == fr.win.to_numpy(str)).all()
    P = {"v4b": np.load(F84 / "P_mean3_flt3.npy").astype(float), "v4c": np.load(OOF / "P_v4m.npy").astype(float)}
    if (OOF / "P_v4l.npy").exists():
        P["v4b_repro"] = np.load(OOF / "P_v4l.npy").astype(float)
    if (OOF / "P_v4l81.npy").exists():
        P["ctrl81"] = np.load(OOF / "P_v4l81.npy").astype(float)
    ok = {k: s74.gate_ok(E, v, lc) for k, v in P.items()}
    ok["champion"] = {st: champ[st].astype(float) for st in ("E", "R")}
    arms = list(ok)
    contrasts = [("v4b", "v4c"), ("champion", "v4c"), ("champion", "v4b")]
    if "ctrl81" in ok:
        contrasts += [("ctrl81", "v4c"), ("v4b", "ctrl81")]
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    # trees alone (OOF argmax, labelled >= 30 min rows): descriptive
    lab = np.array([t in E["C7"] for t in tr]) & fr.wgroup.isin(C.GE30).to_numpy()
    Pt = {t: np.load(OOF / f"Pt_{t}.npy") for t in ("v4l", "v4m", "v4l81") if (OOF / f"Pt_{t}.npy").exists()}
    res = {"arms": arms, "trees_alone_ge30_labelled_acc7": {
        t: round(float(np.mean(np.array(E["C7"])[v[lab].argmax(1)] == tr[lab])), 4) for t, v in Pt.items()}}
    # rows whose detector was re-admitted to training (fault-only dq) -- scored rows of those detectors
    chg = pd.read_csv(F88 / "dq_changes.csv")
    keys = set(zip(chg.DeviceId.str.lower(), chg.detector.astype(int)))
    readm = np.fromiter(((d, int(x)) in keys for d, x in zip(fr.DeviceId.str.lower(), fr.Detector)), bool, len(fr))
    for st in ("E", "R"):
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            sc = pm & np.logical_and.reduce([~np.isnan(ok[k][st]) for k in arms])
            r = {"n": int(sc.sum())}
            for k in arms:
                r[k] = C.acc_ci(ok[k][st][sc], sig[sc])
            for b0, b1 in contrasts:
                r[f"{b1} - {b0}"] = C.delta_ci(ok[b0][st][sc], ok[b1][st][sc], sig[sc])
            res[f"{st}_{pool}"] = r
        m = fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok["v4c"][st]) & ~np.isnan(ok["v4b"][st])
        res[f"{st}_ge30_by_class_v4c_minus_v4b"] = {
            c: [int((m & (cl == c)).sum())] + C.delta_ci(ok["v4b"][st][m & (cl == c)], ok["v4c"][st][m & (cl == c)],
                                                        sig[m & (cl == c)])
            for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        res[f"{st}_ge30_readmitted_detectors_v4c_minus_v4b"] = [int((m & readm).sum())] + C.delta_ci(
            ok["v4b"][st][m & readm], ok["v4c"][st][m & readm], sig[m & readm])
        if "ctrl81" in ok:
            mc = m & ~np.isnan(ok["ctrl81"][st])
            res[f"{st}_ge30_by_class_v4c_minus_ctrl81"] = {
                c: [int((mc & (cl == c)).sum())] + C.delta_ci(ok["ctrl81"][st][mc & (cl == c)],
                                                             ok["v4c"][st][mc & (cl == c)], sig[mc & (cl == c)])
                for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
            res[f"{st}_ge30_readmitted_detectors_v4c_minus_ctrl81"] = [int((mc & readm).sum())] + C.delta_ci(
                ok["ctrl81"][st][mc & readm], ok["v4c"][st][mc & readm], sig[mc & readm])
        res[f"{st}_ge30_by_fold_v4c_minus_v4b"] = {
            int(k): round(100 * float(np.nanmean(ok["v4c"][st][m & (fr.fold.to_numpy() == k)])
                                      - np.nanmean(ok["v4b"][st][m & (fr.fold.to_numpy() == k)])), 3) for k in range(6)}
    json.dump(res, open(OOF / "oof88.json", "w"), indent=1, default=str)
    for k, v in res.items():
        log(f"{k}: {v}")


# ============================================================================================ package fits
def _pkg_out():
    if not FIT.exists():
        shutil.copytree(DCW / "final_v3_work" / "v3fit83", FIT)
    F.OUT = FIT
    F.F75.OUT = FIT
    F.F76.OUT = FIT


def _v4m_everywhere():
    """function label table = PKG (v4n by default; was v4m) for trainers (LABEL_SETS['v3s']) and atspm_score.V3S (truth /
    scoring columns identical to v4l, asserted; v4n's restored training values live in <col>_train)."""
    import v3_retrain as V
    import atspm_score as AS
    check_truth_same()
    V.LABEL_SETS["v3s"] = V.LABEL_SETS[PKG]
    AS.V3S = LAB[PKG]
    F.LAB_V4L = LAB[PKG]
    use_trees(PKG)


def stage_pkgfunc(a):
    _pkg_out()
    for f in (FIT / "function").glob("function229_s*.txt"):
        f.unlink()
    _v4m_everywhere()
    F.stage_func(a)
    m = json.load(open(FIT / "function" / "function229.json"))
    m["recipe"] = m["recipe"].replace("v4l labels (note 80/81)", "v4m labels (note 88: v4l, dq_suspect without fault "
                                                                 "events; card rule actuation-only)" if PKG == "v4m" else
                                      f"{PKG} labels (note 89: v4m + exclusion groups restored to training)")
    m["n_estimators_rule"] = (f"mean best iteration of the 18 {PKG} six-fold fits (3 seeds); no early stopping")
    F.F75.save_json(m, FIT / "function" / "function229.json")


def stage_pkglanes(a):
    _pkg_out()
    _v4m_everywhere()
    F.F83 = F88                       # ln8 working copy (truth = print lanes, same for v4m: lane columns unchanged)
    src = DCW / "final_v3_work" / "f83" / "ln8"
    (F88 / "ln8").mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.parquet"):
        if not (F88 / "ln8" / f.name).exists():
            shutil.copy(f, F88 / "ln8" / f.name)
    for f in src.glob("*.json"):
        if not (F88 / "ln8" / f.name).exists():
            shutil.copy(f, F88 / "ln8" / f.name)
    F.stage_lanes(a)


def stage_pkgsbfeats(a):
    _pkg_out()
    _v4m_everywhere()
    F.F83 = F88
    F.stage_sbfeats(a)


def stage_pkgsetback(a):
    _pkg_out()
    F.F83 = F88
    F.stage_setback(a)


def stage_pkgsingle(a):
    """the 'single' fallback stacker (fit83 recipe) on the v4m OOF trees; run BEFORE pkgstacker (it clears stacker*)."""
    _pkg_out()
    use_trees(PKG)
    F.stage_stacker(a)


def stage_pkgstacker(a):
    import lightgbm as lgb
    _pkg_out()
    use_trees(PKG)
    d = FIT / "stacker"
    for f in d.glob("stacker_mean3*"):
        f.unlink()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    Xm = F4.net_X(s74, S, E, s74.net_probs(E["fr"], "x69_siba"))
    X, yy = Xm[trm], y[trm]
    log(f"stacker mean3 (v4m trees OOF, no health): X {X.shape}")
    np.save(d / "X_check_mean3.npy", Xm[:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_mean3_s{s}.txt"
        lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=s), lgb.Dataset(X, yy), num_boost_round=150).save_model(str(f))
        files.append(f.name)
        log(f"  seed {s} done")
    fr = E["fr"]
    json.dump({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files,
               "params": F.PRM0, "num_boost_round": 150,
               "variant": "mean3: trained on the 3-seed MEAN of the siba OOF fold models (1 x rows, notes 67 / 69 / 84)",
               "net": "siba x69_siba 3-seed-mean OOF fold models (v3s-trained); production = 3 full-data siba v4l "
                      "members averaged",
               "labels": "v4l truth (= v4m truth); trees = note-88 v4m OOF (dq without fault events)",
               "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique())}},
              open(d / "stacker_mean3.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "stack", "score", "pkgfunc", "pkglanes", "pkgsbfeats", "pkgsetback",
                                      "pkgsingle", "pkgstacker"])
    ap.add_argument("--trees", default="v4m", choices=["v4l", "v4m", "v4l81"])
    a = ap.parse_args()
    F88.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
