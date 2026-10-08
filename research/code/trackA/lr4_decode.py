"""Lanes redo, task 3 -- constrained function decode with the new lane grouping.

Lane similarity = P(same lane) from a logistic model on the lr2 pair cues + the
predicted-function pair type, trained on the lane-text signals (RL/CL/LL truth) and
applied out-of-fold: a signal in fold f is scored by a model that never saw fold f.
Lanes = average-linkage clustering on that probability, cut at `th`, then (rule: 1-3
lanes per phase) optionally merged down to 3.  Roles = A3's exact assignment (<= 1
Advance / Presence / Count per lane, Yellow_Red free, rest Other), or its `cap` form
(<= n_lanes of each per phase).

Full window only (the cues need a day or more of data).  Threshold chosen on folds 1-5;
fold 0 reported separately.  Lane-count checks are VALIDATION, never inputs: the label
lower bound max(#A, #P, #C), the RL/CL/LL token count, and -- when it exists -- a
per-(signal, phase) n_lanes table (lr_common.NLANES_LABELS).

Needs `a3_lanes.py` (the A3 stage: `cluster`, `assign`) next to it.

    python lr4_decode.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import lr_common as C
import a3_lanes as A3

CL5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
FEATS = ["hp_off", "hp_all", "lead_exc_q", "lead_sharp_all", "c_off", "z0_sharp_all",
         "bal"] + [f"pt_{c}" for c in CL5]


def pair_type(P, pf):
    for c in CL5:
        P[f"pt_{c}"] = ((P.pf_a == c).astype(float) + (P.pf_b == c).astype(float))
    return P


def fit(T):
    X = T[FEATS].to_numpy(float)
    med = np.nanmedian(X, 0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, T.same_lane)
    return lambda D: m.predict_proba(
        (np.where(np.isnan(D[FEATS].to_numpy(float)), med, D[FEATS].to_numpy(float))
         - mu) / sd)[:, 1]


def merge_to(lab, S, kmax):
    """Keep merging the closest pair of clusters (average linkage) until <= kmax."""
    lab = lab.copy()
    while len(np.unique(lab)) > kmax:
        u = np.unique(lab)
        best, bi, bj = -np.inf, None, None
        for a in range(len(u)):
            for b in range(a + 1, len(u)):
                v = S[np.ix_(lab == u[a], lab == u[b])].mean()
                if v > best:
                    best, bi, bj = v, u[a], u[b]
        lab[lab == bj] = bi
    return pd.factorize(lab)[0]


def main():
    keys = pd.read_parquet(C.WORK / "a2_oof_keys.parquet")
    Pr = np.load(C.WORK / "a2_oof_base_expert_mean.npy")
    full = (keys.wgroup == "full").to_numpy()
    K = keys[full].reset_index(drop=True)
    PK = Pr[full]
    K["DeviceId"] = K.DeviceId.str.lower()
    K = K[~K.DeviceId.isin(C.locked_ids())]
    PK = PK[K.index.to_numpy()]
    K = K.reset_index(drop=True)
    K["pf"] = np.array(CL5)[PK.argmax(1)]
    K["Detector"] = K.Detector.astype("int16")

    T = pd.read_parquet(C.WORK / "lr3_pairs_scored.parquet")       # truth pairs
    A = pd.read_parquet(C.WORK / "lr2_pairs_all.parquet")          # decode pairs
    kk = K[["DeviceId", "period", "Detector", "pf", "fold"]]
    for s in ("a", "b"):
        A = A.merge(kk.rename(columns={"Detector": f"d{s}", "pf": f"pf_{s}",
                                       "fold": f"fold_{s}"}),
                    on=["DeviceId", "period", f"d{s}"], how="left")
    A = pair_type(A, None)
    A["fold"] = A.fold_a
    A["p_same"] = np.nan
    for f in sorted(K.fold.unique()):
        m = fit(T[T.fold != f])
        sel = (A.fold == f).to_numpy()
        A.loc[sel, "p_same"] = m(A[sel])
    C.log(f"{len(A)} decode pairs scored; mean p_same {A.p_same.mean():.3f}")

    lk = {(d, p): dict(zip(zip(g.da, g.db), g.p_same))
          for (d, p), g in A.groupby(["DeviceId", "period"])}
    groups = K.groupby(["DeviceId", "period", "pred_phase"]).indices
    Smat = {}
    for key, rows in groups.items():
        dets = K.Detector.to_numpy()[rows]
        sub = lk.get(key[:2], {})
        n = len(dets)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a, b = min(dets[i], dets[j]), max(dets[i], dets[j])
                S[i, j] = S[j, i] = sub.get((a, b), 0.0)
        Smat[key] = (rows, S)

    def decode(th, mode, cap3):
        out = PK.argmax(1).copy()
        nl = np.ones(len(K), int)
        lid = np.zeros(len(K), int)
        for key, (rows, S) in Smat.items():
            if len(rows) == 1:
                continue
            lab = A3.cluster(np.arange(len(rows)), S, th)
            if cap3:
                lab = merge_to(lab, S, 3)
            out[rows] = A3.assign(PK[rows], lab, mode)
            nl[rows] = len(np.unique(lab))
            lid[rows] = lab
        return out, nl, lid

    yt = K.func5.to_numpy()
    apc = np.isin(yt, C.APC)

    def sc(pred, m):
        p = np.array(CL5)[pred]
        return {"acc5": round(float((yt[m] == p[m]).mean()), 4),
                "accAPC": round(float((yt[m & apc] == p[m & apc]).mean()), 4),
                "other_recall": round(float((p[m & (yt == "Other")] == "Other").mean()), 4)}

    dev = (K.fold != 0).to_numpy()
    f0 = ~dev
    allm = np.ones(len(K), bool)
    res = {"n_rows_full": int(len(K)), "n_phases": int(len(groups)),
           "argmax": {"folds1_5": sc(PK.argmax(1), dev), "fold0": sc(PK.argmax(1), f0),
                      "all": sc(PK.argmax(1), allm)}}
    C.log(f"argmax {res['argmax']}")
    tune = {}
    for mode in ("lane", "cap"):
        for cap3 in (False, True):
            for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                pred, nl, _ = decode(th, mode, cap3)
                k = f"{mode}/cap3={cap3}/th={th}"
                ph = pd.Series(nl).groupby([K.DeviceId, K.period, K.pred_phase]).first()
                tune[k] = {"folds1_5": sc(pred, dev),
                           "mean_n_lanes": round(float(ph.mean()), 3),
                           "share_gt3": round(float((ph > 3).mean()), 4)}
                C.log(f"{k}: {tune[k]}")
    res["tune"] = tune
    best = max(tune, key=lambda k: tune[k]["folds1_5"]["acc5"])
    mode, cap3, th = best.split("/")
    cap3 = cap3.endswith("True")
    th = float(th.split("=")[1])
    pred, nl, lid = decode(th, mode, cap3)
    res["chosen"] = {"key": best, "folds1_5": sc(pred, dev), "fold0": sc(pred, f0),
                     "all": sc(pred, allm)}
    C.log(f"chosen {res['chosen']}")
    # the physically-motivated setting regardless of score: lane mode, <= 3 lanes,
    # threshold 0.5 on the calibrated P(same lane)
    pp, nlp, lidp = decode(0.5, "lane", True)
    res["physical_lane_cap3_th0.5"] = {"folds1_5": sc(pp, dev), "fold0": sc(pp, f0),
                                       "all": sc(pp, allm)}

    # ---------------- lane-count validation (never an input)
    K["n_lanes"] = nlp
    K["lane"] = lidp
    lab = C.labelled_detectors()[["DeviceId", "Detector", "lane", "unit"]].rename(
        columns={"lane": "lane_tok"})
    lab["Detector"] = lab.Detector.astype("int16")
    K2 = K.merge(lab, on=["DeviceId", "Detector"], how="left")
    ph = K2.groupby(["DeviceId", "period", "pred_phase"]).agg(
        n_lanes=("n_lanes", "first"), n_det=("Detector", "size"),
        nA=("func5", lambda s: int((s == "Advance").sum())),
        nP=("func5", lambda s: int((s == "Presence").sum())),
        nC=("func5", lambda s: int((s == "Count").sum())),
        n_tok=("lane_tok", "nunique"), n_tagged=("lane_tok", lambda s: int(s.notna().sum())),
        n_unit=("unit", "nunique"))
    ph["L_min"] = ph[["nA", "nP", "nC"]].max(axis=1)
    ok = ph[ph.L_min > 0]
    plaus = (ok.n_lanes.between(1, 3)) & (ok.n_lanes >= ok.L_min)
    # lane-text phases: every A/P/C detector tagged, one sensor unit or none
    tt = ph[(ph.n_tagged >= 2) & (ph.n_tagged == ph.n_det) & (ph.n_unit <= 1)]
    res["n_lanes_check"] = {
        "n_phases": int(len(ok)),
        "dist": {str(k): int(v) for k, v in ok.n_lanes.value_counts().sort_index().items()},
        "L_min_dist": {str(k): int(v) for k, v in
                       ok.L_min.clip(upper=4).value_counts().sort_index().items()},
        "plausible_1to3_and_ge_Lmin": round(float(plaus.mean()), 4),
        "ge_Lmin": round(float((ok.n_lanes >= ok.L_min).mean()), 4),
        "exact_eq_Lmin": round(float((ok.n_lanes == ok.L_min).mean()), 4),
        "lane_text_phases_fully_tagged": int(len(tt)),
        "exact_eq_tokens": round(float((tt.n_lanes == tt.n_tok).mean()), 4) if len(tt) else None,
        "tokens_dist": {str(k): int(v) for k, v in tt.n_tok.value_counts().sort_index().items()},
        "inferred_on_those": {str(k): int(v) for k, v in
                              tt.n_lanes.value_counts().sort_index().items()}}
    # pairwise agreement of the inferred lanes with the RL/CL/LL truth pairs
    Tm = T[["DeviceId", "period", "da", "db", "same_lane"]]
    L1 = K[["DeviceId", "period", "Detector", "pred_phase", "lane"]]
    Tm = (Tm.merge(L1.rename(columns={"Detector": "da", "lane": "la", "pred_phase": "pa"}),
                   on=["DeviceId", "period", "da"])
            .merge(L1.rename(columns={"Detector": "db", "lane": "lb", "pred_phase": "pb"}),
                   on=["DeviceId", "period", "db"]))
    Tm = Tm[Tm.pa == Tm.pb]
    same_pred = Tm.la == Tm.lb
    res["pair_agreement_lane_text"] = {
        "n_pairs": int(len(Tm)),
        "accuracy": round(float((same_pred == Tm.same_lane).mean()), 4),
        "same_lane_recall": round(float(same_pred[Tm.same_lane].mean()), 4),
        "diff_lane_recall": round(float((~same_pred[~Tm.same_lane]).mean()), 4)}
    ext = C.load_nlanes_labels()
    if ext is not None:                      # future cabinet-print lane counts
        e = ph.reset_index()
        e["target"] = "P" + e.pred_phase.astype(int).astype(str)
        e = e.merge(ext, on=["DeviceId", "target"], how="inner", suffixes=("", "_true"))
        res["vs_external_nlanes"] = {
            "n": int(len(e)),
            "exact": round(float((e.n_lanes == e.n_lanes_true).mean()), 4)}
    ph.reset_index().to_parquet(C.WORK / "lr4_nlanes.parquet", index=False)
    C.log(json.dumps({k: res[k] for k in ("physical_lane_cap3_th0.5", "n_lanes_check",
                                          "pair_agreement_lane_text")}, indent=1))
    json.dump(res, open(C.WORK / "lr4_decode.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
