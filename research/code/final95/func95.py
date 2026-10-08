"""Note 95: the FUNCTION side on 2026-only data (training rows = Sept-2026 period only; Dec-2024 rows never trained on
and never scored; labels v4q).

    python func95.py fit        229 function trees x 3 seeds, six folds (folds_v4 via the frame), oof88 / note-89 recipe,
                                training mask AND period == 'stg' -> f76/function_c_v4q26 (OOF for every frame row; only stg
                                rows are used downstream)
    python func95.py stack      context stacker mean3 (note-86 recipe: 47 cols, PRM0 150 rounds, seeds 0/1/2) on the 2026 trees OOF +
                                the filtered 3-seed-mean 2026 siba fold OOF (s95_siba*), trained on Sept-2026 rows only
                                -> final_v3_work/f95/oof/P_v5.npy
    python func95.py score      gate .9 decode, Sept-2026 rows only, v4q truth, E / R, >= 30 / 10 / 5 min, CIs, with and
                                without n1_model_decided rows; v5 vs v4f (f90 P_v4f) paired -> f95/oof/score95.json
    python func95.py rows       training-row / signal / detector / label accounting, 2026-only vs before (v4q) -> s95/rows95.json
Run with F76_ARM=c.  CPU 6 threads.  locked_v2 asserted absent by the loaders.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
from common import DC_WORK  # noqa: E402

TREES95 = DC_WORK / "final_v3_work" / "f76" / "function_c_v4q26"
CFG = "drop:pp_xcand+pp_pdiff+pp_v2+yr+ratio+phctx"
S95 = DC_WORK / "s95"


def patch_2026(V):
    """variant_target AND period == 'stg' (looked up at call time by every trainer)."""
    if getattr(V, "_p95", False):
        return
    orig = V.variant_target

    def vt(lab, fr, v):
        y, ok = orig(lab, fr, v)
        return y, ok & (fr.period.to_numpy() == "stg")
    V.variant_target = vt
    V._p95 = True


def stage_fit(a):
    import t57_function as T57F
    import f76_function  # noqa: F401  (resets T57F.OUT on import: import first, override after)
    import v3_retrain as V
    assert Path(V.LABEL_SETS["v3s"][0]).name == "function_labels_v4q.parquet"
    patch_2026(V)
    T57F.OUT = TREES95
    T57F.setup("v6e")
    T57F.BASE_RUN = V.run_dir("exclude").name
    TREES95.mkdir(parents=True, exist_ok=True)
    T57F.stage_fit(argparse.Namespace(frame="v6e", cfg=CFG, seeds="0,1,2", threads=6))


CFG_SUB = Path("v6e") / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx"
F95 = DC_WORK / "final_v3_work" / "f95"
OOF95 = F95 / "oof"
NET95 = os.environ.get("NET95", "s95_siba")
LAB_Q = rpath.REPO / "research" / "labels" / "function_labels_v4q.parquet"


def fsetup():
    """fit83 / oof88 paths on the 2026 trees and v4q labels (truth = v4q plain columns)."""
    code = rpath.CODE
    for d in ("final84", "final83", "final88", "final86", "final90"):
        sys.path.insert(0, str(code / d))
    import fit84 as F4
    import fit83 as F
    import v3_retrain as V
    import atspm_score as AS
    V.LABEL_SETS["v3s"] = V.LABEL_SETS["v4q"]
    AS.V3S = LAB_Q
    F.LAB_V4L = LAB_Q
    F.TREES_V4L = TREES95
    F.CFGD = TREES95 / CFG_SUB
    assert (F.CFGD / "timing.jsonl").exists()
    return F, F4


def stage_stack(a):
    import lightgbm as lgb
    F, F4 = fsetup()
    OOF95.mkdir(parents=True, exist_ok=True)
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    Pf = s74.net_probs(fr, NET95)
    st = (fr.period == "stg").to_numpy()
    F.log(f"{NET95}: coverage stg rows {np.mean(~np.isnan(Pf[st, 0])):.4f}, dec rows {np.mean(~np.isnan(Pf[~st, 0])):.4f}")
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
    np.save(OOF95 / "P_v5.npy", R.astype(np.float32))
    np.save(OOF95 / "Pt_v4q26.npy", E["Pt"].astype(np.float32))
    json.dump({"train_rows": int(trm.sum()), "rows": int(len(y)), "net": NET95}, open(OOF95 / "stack_rows.json", "w"))


def stage_score(a):
    import cand64 as C
    import of77
    F, F4 = fsetup()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    P = {"v5": np.load(OOF95 / "P_v5.npy").astype(float),
         "v4f": np.load(DC_WORK / "final_v3_work" / "f90" / "oof" / "P_v4f.npy").astype(float)}
    assert all(len(v) == len(fr) for v in P.values())
    ok = {k: s74.gate_ok(E, v, lc) for k, v in P.items()}
    L = pd.read_parquet(LAB_Q, columns=["DeviceId", "detector", "n1_model_decided"])
    L["DeviceId"] = L.DeviceId.str.lower()
    L = L.rename(columns={"detector": "Detector"}).drop_duplicates(["DeviceId", "Detector"])
    L["Detector"] = L.Detector.astype(fr.Detector.dtype)
    n1 = fr[["DeviceId", "Detector"]].assign(DeviceId=fr.DeviceId.str.lower()).merge(L, on=["DeviceId", "Detector"],
                                                                                      how="left").n1_model_decided
    n1 = n1.fillna(False).to_numpy(bool)
    st = (fr.period == "stg").to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    res = {"net": NET95}
    for subset, msub in (("all", st), ("without_n1", st & ~n1), ("n1_only", st & n1)):
        for stn in ("E", "R"):
            for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
                pm = msub & fr.wgroup.isin(fams).to_numpy()
                sc = pm & ~np.isnan(ok["v5"][stn]) & ~np.isnan(ok["v4f"][stn])
                r = {"n": int(sc.sum()), "signals": int(len(set(sig[sc])))}
                for k in ok:
                    r[k] = C.acc_ci(ok[k][stn][sc], sig[sc])
                r["v5 - v4f"] = C.delta_ci(ok["v4f"][stn][sc], ok["v5"][stn][sc], sig[sc])
                res[f"{subset}|{stn}_{pool}"] = r
    for stn in ("E", "R"):
        m = st & fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok["v5"][stn]) & ~np.isnan(ok["v4f"][stn])
        res[f"{stn}_ge30_by_class"] = {c: {"n": int((m & (cl == c)).sum()), "v5": round(float(np.mean(ok["v5"][stn][m & (cl == c)])), 4),
                                           "v4f": round(float(np.mean(ok["v4f"][stn][m & (cl == c)])), 4),
                                           "d": C.delta_ci(ok["v4f"][stn][m & (cl == c)], ok["v5"][stn][m & (cl == c)], sig[m & (cl == c)])}
                                       for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        res[f"{stn}_ge30_by_fold_v5_minus_v4f"] = {int(k): round(100 * float(np.nanmean(ok["v5"][stn][m & (fr.fold.to_numpy() == k)])
                                                                         - np.nanmean(ok["v4f"][stn][m & (fr.fold.to_numpy() == k)])), 3)
                                                   for k in range(6)}
    lab = np.array([t in E["C7"] for t in tr]) & fr.wgroup.isin(C.GE30).to_numpy() & st
    Pt = E["Pt"]
    res["trees_alone_ge30_labelled_acc7_stg"] = round(float(np.mean(np.array(E["C7"])[Pt[lab].argmax(1)] == tr[lab])), 4)
    json.dump(res, open(OOF95 / f"score95_{NET95}.json", "w"), indent=1, default=str)
    for k, v in res.items():
        if "ge30" in k or "_m10" in k or "_m5" in k:
            F.log(f"{k}: {v}")


def stage_rows(a):
    import t57_function as T57F
    import v3_retrain as V
    T57F.OUT = S95 / "t57_dummy"
    T57F.setup("v6e")
    fr, _ = V.load_feats()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, T57F.VAR)
    L = pd.read_parquet(rpath.LABELS_CURRENT)
    t = L[["DeviceId", "detector", "truth_v3s"]].assign(DeviceId=L.DeviceId.str.lower()).rename(columns={"detector": "Detector"})
    k0 = fr[["DeviceId", "Detector"]].assign(DeviceId=fr.DeviceId.str.lower())
    t["Detector"] = t.Detector.astype(k0.Detector.dtype)
    tr = k0.merge(t.drop_duplicates(["DeviceId", "Detector"]), on=["DeviceId", "Detector"], how="left").truth_v3s.to_numpy(object)
    tr = np.where((fr.det_n_on >= 5).to_numpy(), tr, None)
    res = {}
    for nm, m in (("before_all_periods", np.ones(len(fr), bool)), ("2026_only", (fr.period == "stg").to_numpy())):
        k = fr[m]
        t = ok & m
        sc = m & pd.notna(tr)
        res[nm] = dict(frame_rows=int(m.sum()), signals=int(k.DeviceId.nunique()),
                       detectors=int(k[["DeviceId", "Detector"]].drop_duplicates().shape[0]),
                       training_rows=int(t.sum()), training_signals=int(fr.DeviceId[t].nunique()),
                       training_detectors=int(fr[t][["DeviceId", "Detector"]].drop_duplicates().shape[0]),
                       truth_rows=int(sc.sum()), truth_detectors=int(fr[sc][["DeviceId", "Detector"]].drop_duplicates().shape[0]),
                       truth_signals=int(fr.DeviceId[sc].nunique()),
                       windows_by_group=fr[m].wgroup.value_counts().to_dict())
    res["label_table"] = dict(rows=len(L), truth_rows=int(L.truth_v3s.notna().sum()),
                              truth_signals=int(L[L.truth_v3s.notna()].DeviceId.nunique()))
    S95.mkdir(exist_ok=True)
    json.dump(res, open(S95 / "rows95.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "rows", "stack", "score"])
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
