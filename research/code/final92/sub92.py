"""Note 92: split the non-ATSPM class Other into data-supported subclasses (training labels only; user rule: non-ATSPM
labels may be added, ATSPM labels never touched).

Subclass = print/config subtype of a detector whose TRAINING label (v4o label_print_first_train) / truth is Other.
A subclass needs >= MIN_DET trained detectors, else it stays Other (decided by `census`, before any score is seen).
Recipe = v4f's function chain (note 90): 229-feature trees arm c x 3 seeds (v4o labels) -> mean3 context stacker
(47 columns, x86_siba4l filtered 3-seed mean, PRM0 150 rounds, stacker seeds 0/1/2) -> gate-.9 lane decode.
K-class arm: trees K classes; the stacker sees the 47 columns built on the trees' 7-class view (subclasses summed into
Other) + the trees' subclass probabilities, and predicts K classes (target = truth_v3s, Other -> subclass); siba net
unchanged (7 classes). Decode:
  kway  (primary, pre-registered) every subclass is a full non-ATSPM class in the per-lane decode (as Mid / Bike)
  sum   (secondary) subclasses summed into Other for the decode; a detector decoded Other takes its best Other subclass
Score = ATSPM-only stack-aware (subclasses score as non-ATSPM), v4l truth, paired signal bootstrap vs v4f (P_v4f).
Arms: sub (real subclasses), shuf (control: the subclass labels permuted over the Other detectors, detector level,
same class sizes; trees AND stacker).

    set F76_ARM=c
    python sub92.py census                 -> %DC_WORK%/x92/census.json (+ decides the classes)
    python sub92.py repro                  seed 0 fold 0 base trees == f76/function_c_v4o (harness check)
    python sub92.py fit   --arm sub|shuf   trees -> %DC_WORK%/x92/trees/<arm>/
    python sub92.py stack --arm sub|shuf   -> %DC_WORK%/x92/oof/PK_<arm>.npy
    python sub92.py score                  -> %DC_WORK%/x92/oof/sub92.json
CPU <= 6 threads (4 trees). locked_v2 asserted absent. One stage per process.
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

CODE = Path(__file__).resolve().parents[1]
for _d in ("final90", "final84", "final83", "final88", "final86"):
    sys.path.insert(0, str(CODE / _d))
for _d in ("", "evaluation", "final76", "final77", "trackA", "lanes"):
    sys.path.insert(0, str(CODE / _d) if _d else str(CODE))
import fit83 as F  # noqa: E402
import oof88 as O88  # noqa: E402

DCW = F.DC_WORK
X92 = DCW / "x92"
OOF = X92 / "oof"
LAB = O88.LAB["v4o"]
CFG_SUB = O88.CFG_SUB
F90 = DCW / "final_v3_work" / "f90" / "oof"
MIN_DET = 50
THREADS = int(os.environ.get("X92_THREADS", "4"))
NET = "x86_siba4l"
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
ATS = C7[:4]
# pre-registered subtype groups (print_subtype of detectors labelled Other); 'explained' (whole-intersection channels not
# on the print: dummies, FYA logic, extension ...) and every rarer subtype stay Other
SUB = {"Upstream_presence": {"advance_presence", "eta", "radar_advance_presence", "advance_presence_long",
                             "advance_presence_trucks", "advance_long_presence"},
       "Long_zone": {"long_zone", "long_presence_20_75ft", "long_presence_0_75", "radar_long_presence",
                     "presence_setback_long", "long_phase_call_zone", "long_phase_zone"},
       "Setback_presence": {"presence_20_75", "setback", "setback_presence", "presence_setback", "presence_setback_75"},
       "Superseded_loop": {"superseded_by_radar", "old_call_loop", "abandoned_loop"},
       "Departure": {"departure", "count_loop_past_stopbar"}}
# definitional bucket of notes 70 / 78 (Other subtypes acting like a PM class)
DEFN = {"advance_presence", "long_zone", "presence_20_75", "superseded_by_radar", "departure", "eta"}
log = F.log


def locked() -> set:
    return set(pd.read_csv(DCW / "official/locked_v2.csv").DeviceId.astype(str).str.lower())


def subtype_table() -> pd.DataFrame:
    L = pd.read_parquet(LAB, columns=["DeviceId", "detector", "print_subtype", "label_print_first_train", "truth_v3s"])
    L["DeviceId"] = L.DeviceId.str.lower()
    assert not L.DeviceId.isin(locked()).any()
    inv = {s: c for c, ss in SUB.items() for s in ss}
    L["sub"] = L.print_subtype.map(inv)
    return L.rename(columns={"detector": "Detector"})


def classes() -> list[str]:
    f = X92 / "census.json"
    assert f.exists(), "run census first"
    return C7 + json.load(open(f))["classes"]


def keyed(fr: pd.DataFrame, L: pd.DataFrame, col: str) -> np.ndarray:
    k = fr[["DeviceId", "Detector"]].assign(DeviceId=fr.DeviceId.str.lower())
    x = k.merge(L.astype({"Detector": k.Detector.dtype})[["DeviceId", "Detector", col]], on=["DeviceId", "Detector"],
                how="left")
    assert len(x) == len(fr)
    return x[col].to_numpy(object)


def shuffle_map(L: pd.DataFrame, pool_mask: np.ndarray, cls: list[str]) -> pd.DataFrame:
    """control: the subclass labels permuted over all Other detectors (same sizes, random membership)."""
    L = L.copy()
    P = L[pool_mask]
    v = P["sub"].where(P["sub"].isin(cls), None).to_numpy(object)
    rng = np.random.default_rng(92)
    L.loc[P.index, "sub"] = v[rng.permutation(len(v))]
    return L


def sub_lookup(arm: str) -> pd.DataFrame:
    """per detector: training subclass (Other-labelled detectors only), for an arm."""
    L = subtype_table()
    cls = classes()[7:]
    L["sub"] = L["sub"].where(L["sub"].isin(cls), None)
    if arm == "shuf":
        L = shuffle_map(L, (L.label_print_first_train == "Other").to_numpy(), cls)
    return L


# ============================================================================================ trees
def _tree_setup(out: Path):
    import f76_function as FF
    import t57_function as T57F
    import v3_retrain as V
    O88.check_truth_same()
    V.LABEL_SETS["v3s"] = V.LABEL_SETS["v4o"]
    T57F.OUT = out
    T57F.setup("v6e")
    T57F.BASE_RUN = V.run_dir("exclude").name
    return FF, T57F, V


def stage_census(a):
    FF, T57F, V = _tree_setup(X92 / "trees" / "census")
    fr, cols = V.load_feats()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, T57F.VAR)
    L = subtype_table()
    sub = keyed(fr, L, "sub")
    st = keyed(fr, L, "print_subtype")
    m = ok & (y == "Other")
    d = fr.loc[m, ["DeviceId", "Detector"]].assign(sub=sub[m], st=st[m])
    det = d.drop_duplicates(["DeviceId", "Detector"])
    res = {"MIN_DET": MIN_DET, "other_training_rows": int(m.sum()), "other_training_detectors": int(len(det)),
           "by_group": {}, "by_subtype_detectors": det.st.fillna("NA").value_counts().head(40).to_dict()}
    for c in SUB:
        mm = d["sub"].to_numpy(object) == c
        res["by_group"][c] = {"rows": int(mm.sum()), "detectors": int((det["sub"] == c).sum()),
                              "signals": int(det[det["sub"] == c].DeviceId.nunique())}
    res["classes"] = [c for c in SUB if res["by_group"][c]["detectors"] >= MIN_DET]
    res["left_in_other"] = [c for c in SUB if c not in res["classes"]]
    X92.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(X92 / "census.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def _remap(V, arm):
    cls = classes()
    V.C7[:] = cls                      # in place: t57 / variant_target read V.C7
    Ls = sub_lookup(arm)
    orig = V.variant_target

    def vt(lab, fr, v):
        y, ok = orig(lab, fr, v)
        y = y.copy()
        s = keyed(lab, Ls, "sub")
        m = (y == "Other") & pd.notna(s)
        y[m] = s[m]
        log(f"{arm}: Other -> subclass on {int((m & ok).sum()):,} training rows; classes "
            f"{pd.Series(y[ok]).value_counts().to_dict()}")
        return y, ok
    V.variant_target = vt


def stage_repro(a):
    """harness check: base 7-class tree, seed 0 fold 0, must equal f76/function_c_v4o."""
    import shutil
    FF, T57F, V = _tree_setup(X92 / "trees" / "repro")
    d = X92 / "trees" / "repro" / CFG_SUB
    d.mkdir(parents=True, exist_ok=True)
    for k in range(1, 6):              # only fold 0 is fitted: the other folds' files exist, so stage_fit skips them
        shutil.copy(O88.TREES["v4o"] / CFG_SUB / f"P_first.all.wi_s0_f{k}.npy", d)
    T57F.stage_fit(argparse.Namespace(frame="v6e", cfg=FF.CFG, seeds="0", threads=THREADS))
    p = "P_first.all.wi_s0_f0.npy"
    new = np.load(X92 / "trees" / "repro" / CFG_SUB / p)
    ref = np.load(O88.TREES["v4o"] / CFG_SUB / p)
    print("repro max |dP|", float(np.abs(new - ref).max()), new.shape, ref.shape)


def stage_fit(a):
    FF, T57F, V = _tree_setup(X92 / "trees" / a.arm)
    _remap(V, a.arm)
    T57F.stage_fit(argparse.Namespace(frame="v6e", cfg=FF.CFG, seeds=a.seeds, threads=THREADS))
    json.dump({"classes": list(V.C7), "arm": a.arm}, open(X92 / "trees" / a.arm / "classes.json", "w"))


# ============================================================================================ stacker
_SUBP = {}


def _stack_setup(arm: str):
    import v3_retrain as V
    import atspm_score as AS
    O88.check_truth_same()
    V.LABEL_SETS["v3s"] = V.LABEL_SETS["v4o"]
    AS.V3S = LAB
    F.LAB_V4L = LAB
    if arm == "v4f":
        O88.use_trees("v4o")
    else:
        F.TREES_V4L = X92 / "trees" / arm
        F.CFGD = F.TREES_V4L / CFG_SUB
        assert (F.CFGD / "timing.jsonl").exists(), F.CFGD
        import s59_step6 as S59
        orig = S59.spec_probs

        def spec_probs(spec, fr):          # K-class trees: 7-class view (subclasses summed into Other) + subclass block
            P, seeds = orig(spec, fr)
            if P.shape[1] > 7:
                _SUBP["P"] = P[:, 7:].copy()
                P = np.concatenate([P[:, :6], P[:, 6:7] + P[:, 7:].sum(1, keepdims=True)], 1)
            return P, seeds
        S59.spec_probs = spec_probs
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    Pf = s74.net_probs(E["fr"], NET)
    import fit84 as F4
    X = F4.net_X(s74, S, E, Pf)
    return s74, S, E, X, y, trm, names


def stage_stack(a):
    import lightgbm as lgb
    OOF.mkdir(parents=True, exist_ok=True)
    s74, S, E, X, y, trm, names = _stack_setup(a.arm)
    fr = E["fr"]
    cls = json.load(open(X92 / "trees" / a.arm / "classes.json"))["classes"]
    Ls = sub_lookup(a.arm)
    sub = keyed(fr, Ls, "sub")
    tr = fr.truth_v3s.to_numpy(object)
    yk = y.copy()
    m = (tr == "Other") & pd.notna(sub)
    yk[m] = [cls.index(s) for s in sub[m]]
    Xk = np.hstack([X, _SUBP["P"]])
    log(f"{a.arm}: stacker X {Xk.shape}, train rows {int(trm.sum()):,}, K {len(cls)}; "
        f"train classes {pd.Series(np.array(cls, object)[yk[trm]]).value_counts().to_dict()}")
    fo = fr.fold.to_numpy()
    R = np.zeros((len(y), len(cls)))
    t0 = time.time()
    for seed in (0, 1, 2):
        for k in range(6):
            tr_, te = trm & (fo != k), fo == k
            mm = lgb.train(dict(F.PRM0, num_class=len(cls), num_threads=THREADS, seed=seed),
                           lgb.Dataset(Xk[tr_], yk[tr_]), num_boost_round=150)
            R[te] += mm.predict(Xk[te]) / 3
        log(f"  stacker seed {seed} done ({time.time() - t0:.0f}s)")
    np.save(OOF / f"PK_{a.arm}.npy", R.astype(np.float32))
    np.save(OOF / f"PtK_{a.arm}.npy", np.hstack([E["Pt"], _SUBP["P"]]).astype(np.float32))


# ============================================================================================ score
def gate_pred(s74, S, E, P, lc, cls):
    """s74.gate_ok with the decode on len(cls) classes (ATSPM = the first four) -> (ok arrays, frame-aligned pred)."""
    import v3_retrain as V
    import atspm_score as AS
    import atspm_decode as AD
    keep = (list(V.C7), AS.C7, AD.C7, AD.ATSPM, AD.A_IDX, AD.N_IDX, E["C7"])
    V.C7[:] = cls
    AS.C7 = np.array(cls, object)
    AD.C7 = list(cls)
    AD.ATSPM = np.array([c in ATS for c in cls])
    AD.A_IDX, AD.N_IDX = np.flatnonzero(AD.ATSPM), np.flatnonzero(~AD.ATSPM)
    E["C7"] = list(cls)
    try:
        fr = E["fr"]
        lanes0 = fr.lanes5g.copy()
        fr["lanes5g"] = lanes0.where(~(lc < s74.GATE), None)
        pr, cr = S.run_decode(E, P)
        fr["lanes5g"] = lanes0
        return S.ok_cols(E, cr), pr
    finally:
        V.C7[:] = keep[0]
        AS.C7, AD.C7, AD.ATSPM, AD.A_IDX, AD.N_IDX, E["C7"] = keep[1:]


def stage_score(a):
    import cand64 as C
    import of77
    s74, S, E, X, y, trm, names = _stack_setup("v4f")
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    L = subtype_table()
    st = keyed(fr, L, "print_subtype")
    cls = classes()
    nk = len(cls) - 7
    ok, pred = {}, {}
    ok["v4f"], p = gate_pred(s74, S, E, np.load(F90 / "P_v4f.npy").astype(float), lc, C7)
    pred["v4f"] = np.array(C7, object)[p]
    for arm in a.arms.split(","):
        PK = np.load(OOF / f"PK_{arm}.npy").astype(float)
        assert PK.shape[1] == len(cls)
        o, p = gate_pred(s74, S, E, PK, lc, cls)
        ok[f"{arm}_kway"], pred[f"{arm}_kway"] = o, np.array(cls, object)[p]
        P7 = np.concatenate([PK[:, :6], PK[:, 6:7] + PK[:, 7:].sum(1, keepdims=True)], 1)
        o, p = gate_pred(s74, S, E, P7, lc, C7)
        lab = np.array(C7, object)[p]
        best = np.array(["Other"] + cls[7:], object)[PK[:, 6:].argmax(1)]
        ok[f"{arm}_sum"], pred[f"{arm}_sum"] = o, np.where(lab == "Other", best, lab)
    arms = list(ok)
    res = {"classes": cls, "arms": arms}
    ge30 = fr.wgroup.isin(C.GE30).to_numpy()
    Lsub = sub_lookup("sub")
    tsub = np.where(tr == "Other", keyed(fr, Lsub, "sub"), None)
    defn = (tr == "Other") & np.isin(st.astype(str), list(DEFN))
    for stt in ("E", "R"):
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            sc = pm & np.logical_and.reduce([~np.isnan(ok[k][stt]) for k in arms])
            r = {"n": int(sc.sum()), "signals": int(len(set(sig[sc])))}
            for k in arms:
                r[k] = C.acc_ci(ok[k][stt][sc], sig[sc])
                if k != "v4f":
                    r[f"{k} - v4f"] = C.delta_ci(ok["v4f"][stt][sc], ok[k][stt][sc], sig[sc])
            if "shuf_kway" in ok and "sub_kway" in ok:
                r["sub_kway - shuf_kway"] = C.delta_ci(ok["shuf_kway"][stt][sc], ok["sub_kway"][stt][sc], sig[sc])
            res[f"{stt}_{pool}"] = r
        sc = ge30 & np.logical_and.reduce([~np.isnan(ok[k][stt]) for k in arms])
        cl = np.where(np.isin(tr, ATS), tr, "nonATSPM")
        res[f"{stt}_ge30_by_class"] = {c: {k: [int((sc & (cl == c)).sum())] + C.delta_ci(
            ok["v4f"][stt][sc & (cl == c)], ok[k][stt][sc & (cl == c)], sig[sc & (cl == c)]) for k in arms if k != "v4f"}
            for c in ATS + ["nonATSPM"]}
        # definitional bucket: Other-truth rows of the note-70/78 subtypes, error = called ATSPM (pt of all rows)
        dm = sc & defn
        res[f"{stt}_ge30_definitional"] = {
            "rows": int(dm.sum()), "detectors": int(len(set(zip(sig[dm], fr.Detector.to_numpy()[dm])))),
            **{k: {"err_rows": int((ok[k][stt][dm] == 0).sum()),
                   "err_pt_of_all": round(100 * float((ok[k][stt][dm] == 0).sum()) / int(sc.sum()), 3),
                   "err_rate_in_bucket": round(float((ok[k][stt][dm] == 0).mean()), 4),
                   "by_subtype_err_rows": pd.Series(st[dm][ok[k][stt][dm] == 0]).value_counts().to_dict()}
               for k in arms}}
        # each subclass's own accuracy (reported, not optimised): exact subclass vs non-ATSPM vs ATSPM
        res[f"{stt}_ge30_subclass_acc"] = {}
        for c in cls[7:] + ["Other_rest"]:
            mm = sc & ((tsub == c) if c != "Other_rest" else ((tr == "Other") & pd.isna(tsub)))
            cc = "Other" if c == "Other_rest" else c
            res[f"{stt}_ge30_subclass_acc"][c] = {"rows": int(mm.sum()), **{
                k: {"exact": round(float(np.mean(pred[k][mm] == cc)), 4) if mm.any() else None,
                    "nonATSPM": round(float(np.mean(~np.isin(pred[k][mm], ATS))), 4) if mm.any() else None,
                    "called": pd.Series(pred[k][mm]).value_counts().head(6).to_dict()} for k in arms}}
        res[f"{stt}_ge30_by_fold_sub_kway_minus_v4f"] = {
            int(f): round(100 * float(np.nanmean(ok["sub_kway"][stt][sc & (fr.fold.to_numpy() == f)])
                                      - np.nanmean(ok["v4f"][stt][sc & (fr.fold.to_numpy() == f)])), 3)
            for f in range(6)} if "sub_kway" in ok else None
    json.dump(res, open(OOF / "sub92.json", "w"), indent=1, default=str)
    np.savez_compressed(OOF / "ok92.npz", **{f"{k}_{s}": v[s] for k, v in ok.items() for s in ("E", "R")},
                        sig=sig.astype(str), win=fr.win.to_numpy(str))
    for k, v in res.items():
        log(f"{k}: {json.dumps(v, default=str)[:1500]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["census", "repro", "fit", "stack", "score"])
    ap.add_argument("--arm", default="sub", choices=["sub", "shuf"])
    ap.add_argument("--arms", default="sub,shuf")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    X92.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
