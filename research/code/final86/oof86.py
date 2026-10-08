"""Note 86: make the mean3 context stacker's training input consistent with production.

Before (candidate v4b, notes 84 / 84b): the stacker 'mean3' learned on the 3-seed-mean OOF of the x69_siba fold nets,
which were trained on the OLD v3s labels (old pre-note-76 inputs, cloud), while the production members are full-data
v4l refits.  Here the fold nets are re-trained on v4l (x86_siba4l, six folds x seeds 0/1/2, local A1000, snapshot
snap74b = post-note-76 inputs, as the production refits), their OOF inference is FILTERED (pair net only on candidates
with OOF tree p >= .01, as the package), and the mean3 stacker is re-fitted on that OOF.

Tree input (orchestrator, after note 88): the v4m trees OOF (f76/function_c_v4m; truth v4m = v4l, asserted); --trees v4l
reproduces the original set-up.  Packages are built on v4d (-> v4e).
Arms (same 47 columns / v4m-v4l target / PRM0 150 rounds / stacker seeds 0,1,2 / fold k's stacker never sees fold k):
  v4b     note-84b mean3_flt3 = v4l trees; stacker on unfiltered x69_siba mean, applied to mean(x74_sibaflt 0,1,2) (loaded)
  v4d     note-88 P_v4m = the same recipe on the v4m trees (= package v4d)                                     (loaded)
  n86     stacker trained on mean(NET filtered seeds 0,1,2), applied to the same                <- the candidate
  n86u    stacker trained on mean(NETnf, unfiltered), applied to mean(NET filtered) (package-style; only if the
          optional unfiltered inferences exist)
  champion  note-81 champion (ok_v4l_champ.npz)
NET defaults to x86_siba4l; `--net x74_sibaflt` is the dry run on the old nets (= "train the stacker on filtered OOF").

    set F76_ARM=c & python oof86.py stack [--net TAG]   -> %DC_WORK%/final_v3_work/f86/oof[_TAG][_v4l]/P_<arm>.npy
    set F76_ARM=c & python oof86.py score [--net TAG]   -> .../oof86.json
    set F76_ARM=c & python oof86.py fit   [--net TAG]   -> %DC_WORK%/final_v3_work/v3fit86/stacker/stacker_mean3_s{0,1,2}.txt
CPU, LightGBM 4 threads.  locked_v2 asserted absent (fit83.stacker_inputs).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final84"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final83"))
import fit84 as F4  # noqa: E402
import fit83 as F  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final88"))
import oof88 as O88  # noqa: E402  (note 88: v4m trees OOF folders, truth check)

F84 = F.DC_WORK / "final_v3_work" / "f84" / "oof"
F88 = F.DC_WORK / "final_v3_work" / "f88" / "oof"
LAB80 = F.DC_WORK / "lab80"
FIT = F.DC_WORK / "final_v3_work" / "v3fit86"
TREES_TAG = "v4m"


def out_dir(net, trees="v4m"):
    d = "oof" if net == "x86_siba4l" else "oof_" + net.replace(":", "_s")
    return F.DC_WORK / "final_v3_work" / "f86" / (d if trees == "v4m" else f"{d}_{trees}")


def use_trees(trees):
    """note 88 (orchestrator 2026-10-04): the stacker's tree input = v4m trees OOF (f76/function_c_v4m); truth v4m = v4l
    (asserted).  'v4l' = the original note-86 set-up (reference only)."""
    import v3_retrain as V
    import atspm_score as AS
    O88.check_truth_same()
    V.LABEL_SETS["v3s"] = V.LABEL_SETS[trees]       # pointers were flipped to v4o by note 89b; this scoring is v4m
    AS.V3S = O88.LAB[trees]
    F.LAB_V4L = O88.LAB[trees]
    O88.use_trees(trees)


def nf_tag(net):
    return {"x86_siba4l": "x86_siba4lnf", "x74_sibaflt": "x69_siba"}.get(net, "none")


def has_all(C, net):
    return net != "none" and all((C.FPRED / f"{net}{s}_f{k}.parquet").exists() for s in ("", "_s1", "_s2")
                                 for k in range(6))


def inputs(net):
    import cand64 as C
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    assert Path(F.CFGD).resolve() == Path(O88.TREES[TREES_TAG] / O88.CFG_SUB).resolve()
    fr = E["fr"]
    Pf = s74.net_probs(fr, net)                           # 3-seed mean, filtered
    F.log(f"{net}: coverage {np.mean(~np.isnan(Pf[:, 0])):.4f}")
    X = {"n86": F4.net_X(s74, S, E, Pf)}
    nf = nf_tag(net)
    if has_all(C, nf):
        Pu = s74.net_probs(fr, nf)
        F.log(f"{nf}: coverage {np.mean(~np.isnan(Pu[:, 0])):.4f}")
        X["n86u_train"] = F4.net_X(s74, S, E, Pu)
    return s74, S, E, X, y, trm, names, Pf


def stage_stack(a):
    import lightgbm as lgb
    out = out_dir(a.net, a.trees)
    out.mkdir(parents=True, exist_ok=True)
    s74, S, E, X, y, trm, names, Pf = inputs(a.net)
    fo = E["fr"].fold.to_numpy()
    arms = {"n86": ("n86", "n86")}
    if "n86u_train" in X:
        arms["n86u"] = ("n86u_train", "n86")              # train on unfiltered, apply to filtered (as the package)
    R = {k: np.zeros((len(y), 7)) for k in arms}
    t0 = time.time()
    for seed in (0, 1, 2):
        for k in range(6):
            tr, te = trm & (fo != k), fo == k
            prm = dict(F.PRM0, num_threads=F.THREADS, seed=seed)
            for arm, (xtr, xte) in arms.items():
                m = lgb.train(prm, lgb.Dataset(X[xtr][tr], y[tr]), num_boost_round=150)
                R[arm][te] += m.predict(X[xte][te]) / 3
        F.log(f"  stacker seed {seed} done ({time.time() - t0:.0f}s)")
    for arm, v in R.items():
        if (out / f"P_{arm}.npy").exists():
            F.log(f"{arm} vs previous run max |diff| {np.abs(np.load(out / f'P_{arm}.npy') - v).max():.2e}")
        np.save(out / f"P_{arm}.npy", v.astype(np.float32))
    np.save(out / "Pnet_mean.npy", Pf.astype(np.float32))


def stage_score(a):
    import cand64 as C
    import of77
    out = out_dir(a.net, a.trees)
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    champ = np.load(LAB80 / "ok_v4l_champ.npz")
    assert (champ["sig"] == fr.DeviceId.to_numpy(str)).all() and (champ["win"] == fr.win.to_numpy(str)).all()
    P = {"v4b": np.load(F84 / "P_mean3_flt3.npy").astype(float),     # v4l trees, v4b package recipe
         "v4d": np.load(F88 / "P_v4m.npy").astype(float)}             # v4m trees, same recipe (note 88 'v4c' = package v4d)
    for arm in ("n86", "n86u"):
        if (out / f"P_{arm}.npy").exists():
            P[arm] = np.load(out / f"P_{arm}.npy").astype(float)
    ok = {k: s74.gate_ok(E, v, lc) for k, v in P.items()}
    ok["champion"] = {st: champ[st].astype(float) for st in ("E", "R")}
    arms = list(ok)
    cand = [k for k in ("n86", "n86u") if k in ok]
    contrasts = [(r, c) for c in cand for r in ("v4d", "v4b", "champion")] + [("v4b", "v4d")] + ([("n86u", "n86")] if "n86u" in ok else [])
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    # the net alone (argmax of the 3-seed filtered mean vs v4l truth; labelled rows, >= 30 min) -- descriptive
    Pn = {"net_new": np.load(out / "Pnet_mean.npy").astype(float), "net_old_flt": s74.net_probs(fr, "x74_sibaflt" + (a.net[a.net.index(":"):] if ":" in a.net else ""))}
    lab = np.array([t in E["C7"] for t in tr]) & fr.wgroup.isin(C.GE30).to_numpy()
    res = {"net": a.net, "arms": arms,
           "net_alone_ge30_labelled": {k: float(np.mean(np.array(E["C7"])[np.nanargmax(np.nan_to_num(v[lab], nan=-1), 1)]
                                                        == tr[lab])) for k, v in Pn.items()}}
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
        for c1 in cand:
            for ref in ("v4d", "v4b", "champion"):
                m = fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok[c1][st]) & ~np.isnan(ok[ref][st])
                res[f"{st}_ge30_by_class_{c1}_minus_{ref}"] = {
                    c: [int((m & (cl == c)).sum())] + C.delta_ci(ok[ref][st][m & (cl == c)], ok[c1][st][m & (cl == c)],
                                                                sig[m & (cl == c)])
                    for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        m = fr.wgroup.isin(C.GE30).to_numpy()
        for c1 in cand:
            res[f"{st}_ge30_by_fold_{c1}_minus_v4d"] = {
                int(k): round(100 * float(np.nanmean(ok[c1][st][m & (fr.fold.to_numpy() == k)])
                                          - np.nanmean(ok["v4d"][st][m & (fr.fold.to_numpy() == k)])), 3)
                for k in range(6)}
    json.dump(res, open(out / "oof86.json", "w"), indent=1, default=str)
    for k, v in res.items():
        F.log(f"{k}: {v}")


def stage_fit(a):
    """the package stacker: mean3 recipe trained on ALL training rows of the filtered 3-seed-mean v4l OOF."""
    import lightgbm as lgb
    assert a.net == "x86_siba4l" and a.trees == "v4m"
    d = FIT / "stacker"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("stacker_mean3*"):
        f.unlink()
    s74, S, E, X, y, trm, names, Pf = inputs(a.net)
    Xm = X[a.arm_x]
    F.log(f"stacker mean3 (note 86, train input {a.arm_x}): X {Xm[trm].shape}")
    np.save(d / "X_check_mean3.npy", Xm[:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_mean3_s{s}.txt"
        lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=s), lgb.Dataset(Xm[trm], y[trm]),
                  num_boost_round=150).save_model(str(f))
        files.append(f.name)
        F.log(f"  seed {s} done")
    fr = E["fr"]
    json.dump({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files,
               "params": F.PRM0, "num_boost_round": 150,
               "variant": "mean3 (note 86): trained on the 3-seed MEAN of the v4l siba OOF fold models (x86_siba4l), "
                          + ("filtered (tree p >= .01) OOF" if a.arm_x == "n86" else "unfiltered OOF"),
               "net": "siba x86_siba4l 3-seed-mean OOF fold models (v4l-trained, post-note-76 inputs); production = "
                      "3 full-data siba v4l members averaged",
               "labels": "v4m (note 88; truth = v4l); trees = note-88 v4m OOF (f76/function_c_v4m)",
               "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique())}},
              open(d / "stacker_mean3.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stack", "score", "fit"])
    ap.add_argument("--net", default="x86_siba4l")
    ap.add_argument("--trees", default="v4m", choices=["v4m", "v4l"], help="stacker tree input (note 88: v4m)")
    ap.add_argument("--arm_x", default="n86", help="fit: n86 (filtered) or n86u_train (unfiltered)")
    a = ap.parse_args()
    TREES_TAG = a.trees
    use_trees(a.trees)
    t0 = time.time()
    {"stack": stage_stack, "score": stage_score, "fit": stage_fit}[a.stage](a)
    F.log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
