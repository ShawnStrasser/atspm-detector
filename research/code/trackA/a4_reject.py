"""A4 -- "Other" as a rejection, not a class.

Variant T (today): a 5-class head with Other trained as a class.
Variant R (this):  a 4-class head trained ONLY on Advance / Presence / Count /
                   Yellow_Red.  A detector is called Other when the head is not
                   confident OR when its feature vector is far from the training
                   distribution (isolation forest).  No Other label is used to fit the
                   head, and the rejection thresholds are set from the *rate* at which
                   real-class rows are rejected in the training folds, so no Other label
                   is used to calibrate them either.

The property being tested is generalisation to Other SUBTYPES the model never saw:
leave one raw Other string out of training and see whether it still comes out as Other.

    python a4_reject.py --stage oof      # 6-fold OOF for variant R, sweep the rule
    python a4_reject.py --stage loso     # leave-one-Other-subtype-out, fold 0
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

import a2_model as A2

WORK = A2.WORK
CLASSES4 = ["Advance", "Presence", "Count", "Yellow_Red"]
CLASSES5 = A2.CLASSES5
CLASSES3 = A2.CLASSES3
N_FOLDS = A2.N_FOLDS


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def novelty(fr, cols, trm, tem, seed=0):
    """Isolation-forest novelty score (higher = stranger), fitted on training rows."""
    from sklearn.ensemble import IsolationForest
    X = fr[cols].to_numpy(dtype=np.float32)
    med = np.nanmedian(X[trm], axis=0)
    X = np.where(np.isnan(X), med, X)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(trm)
    if len(idx) > 60000:
        idx = rng.choice(idx, 60000, replace=False)
    iso = IsolationForest(n_estimators=200, max_samples=256, random_state=seed,
                          n_jobs=12).fit(X[idx])
    return -iso.score_samples(X), X, idx


def reject_rule(P4, nov, conf_th, nov_th):
    pred = np.array(CLASSES4)[P4.argmax(1)]
    rej = (P4.max(1) < conf_th) | (nov > nov_th)
    return np.where(rej, "Other", pred)


def evaluate(fr, pred, mask=None) -> dict:
    m = np.ones(len(fr), bool) if mask is None else mask
    yt = fr.func5.to_numpy()[m]
    pr = pred[m]
    wg = fr.wgroup.to_numpy()[m]
    apc = np.isin(yt, CLASSES3)
    d = {"n": int(m.sum()), "acc5_allwin": round(float((yt == pr).mean()), 4),
         "accAPC_allwin": round(float((yt[apc] == pr[apc]).mean()), 4),
         "other_recall": round(float((pr[yt == "Other"] == "Other").mean()), 4),
         "other_precision": (round(float((yt[pr == "Other"] == "Other").mean()), 4)
                             if (pr == "Other").any() else None)}
    d["acc5_by_duration"] = {g: round(float((yt[wg == g] == pr[wg == g]).mean()), 4)
                             for g in A2.DUR_GROUPS if (wg == g).any()}
    d["accAPC_by_duration"] = {
        g: round(float((yt[(wg == g) & apc] == pr[(wg == g) & apc]).mean()), 4)
        for g in A2.DUR_GROUPS if ((wg == g) & apc).any()}
    full = wg == "full"
    per = {}
    for c in CLASSES5:
        tp = int(((yt[full] == c) & (pr[full] == c)).sum())
        fp = int(((yt[full] != c) & (pr[full] == c)).sum())
        fn = int(((yt[full] == c) & (pr[full] != c)).sum())
        per[c] = {"n": tp + fn,
                  "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                  "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    d["per_class_full"] = per
    return d


def stage_oof(a) -> None:
    fr = A2.load_frame()
    cols = A2.feat_cols(fr, A2.ALL_FAM)
    y5 = fr.func5.to_numpy()
    real = np.isin(y5, CLASSES4)
    y4 = pd.Series(np.where(real, y5, CLASSES4[0])).map(
        {c: i for i, c in enumerate(CLASSES4)}).to_numpy()
    folds = fr.fold.to_numpy()
    P4 = np.zeros((len(fr), 4))
    NOV = np.zeros(len(fr))
    CONF_Q, NOV_Q = {}, {}
    for k in range(N_FOLDS):
        te = folds == k
        trm = real & (folds != k)
        m = A2.fit_fold(fr, y4, cols, k, seed=0, train_mask=real, classes=CLASSES4)
        P4[te] = m.predict_proba(fr.loc[te, cols])
        nov, X, idx = novelty(fr, cols, trm, te)
        NOV[te] = nov[te]
        # thresholds come from the inner validation fold's real-class rows: never the
        # test fold, never an Other label, and not in-sample either
        cal = real & (folds == (k + 1) % N_FOLDS)
        ptr = m.predict_proba(fr.loc[cal, cols]).max(1)
        CONF_Q[k] = (ptr, nov[cal])
        log(f"  fold {k}: {m.best_iteration_} trees")
    res = {"n": int(len(fr)), "n_features": len(cols)}

    # variant T, the trained-Other head from A2 (same rows, same features)
    PT = np.load(WORK / "a2_oof_base_expert_mean.npy")
    predT = np.array(CLASSES5)[PT.argmax(1)]
    res["T_trained_other"] = evaluate(fr, predT)

    sweep = []
    for cq in (0.0, 0.10, 0.20, 0.30, 0.40):
        for nq in (1.0, 0.99, 0.97, 0.95, 0.90):
            pred = np.empty(len(fr), object)
            for k in range(N_FOLDS):
                te = folds == k
                ptr, ntr = CONF_Q[k]
                ct = np.quantile(ptr, cq) if cq > 0 else -1.0
                nt = np.quantile(ntr, nq) if nq < 1.0 else np.inf
                pred[te] = reject_rule(P4[te], NOV[te], ct, nt)
            e = evaluate(fr, pred)
            sweep.append({"conf_q": cq, "nov_q": nq, "acc5": e["acc5_allwin"],
                          "accAPC": e["accAPC_allwin"],
                          "other_R": e["other_recall"],
                          "other_P": e["other_precision"],
                          "acc5_full": e["acc5_by_duration"].get("full")})
            log(f"  cq={cq} nq={nq}: acc5 {e['acc5_allwin']:.4f} "
                f"APC {e['accAPC_allwin']:.4f} otherR {e['other_recall']:.3f}")
    res["sweep"] = sweep
    best = max(sweep, key=lambda r: r["acc5"])
    res["best_rule"] = best
    pred = np.empty(len(fr), object)
    for k in range(N_FOLDS):
        te = folds == k
        ptr, ntr = CONF_Q[k]
        ct = np.quantile(ptr, best["conf_q"]) if best["conf_q"] > 0 else -1.0
        nt = np.quantile(ntr, best["nov_q"]) if best["nov_q"] < 1.0 else np.inf
        pred[te] = reject_rule(P4[te], NOV[te], ct, nt)
    res["R_rejection"] = evaluate(fr, pred)
    np.save(WORK / "a4_P4.npy", P4)
    np.save(WORK / "a4_nov.npy", NOV)
    out = fr[["DeviceId", "Detector", "period", "win", "wgroup", "fold", "func5",
              "Function"]].copy()
    out["pred_R"] = pred
    out["pred_T"] = predT
    out.to_parquet(WORK / "a4_pred.parquet", index=False)
    json.dump(res, open(WORK / "a4_oof.json", "w"), indent=1, default=str)
    log(json.dumps({"T": res["T_trained_other"]["acc5_by_duration"],
                    "R": res["R_rejection"]["acc5_by_duration"],
                    "best": best}, indent=1))


def stage_loso(a) -> None:
    """Leave one raw Other string out of TRAINING, fold 0 as the test fold."""
    fr = A2.load_frame()
    cols = A2.feat_cols(fr, A2.ALL_FAM)
    y5 = fr.func5.to_numpy()
    raw = fr.Function.astype(str).str.strip().str.lower().to_numpy()
    real = np.isin(y5, CLASSES4)
    folds = fr.fold.to_numpy()
    te = folds == 0
    subs = (pd.Series(raw[(y5 == "Other")]).value_counts().head(6).index.tolist())
    log(f"Other subtypes tested: {subs}")

    # variant R: never trains on Other at all -> one fit, reused for every subtype
    y4 = pd.Series(np.where(real, y5, CLASSES4[0])).map(
        {c: i for i, c in enumerate(CLASSES4)}).to_numpy()
    mR = A2.fit_fold(fr, y4, cols, 0, seed=0, train_mask=real, classes=CLASSES4)
    P4 = np.zeros((len(fr), 4)); P4[te] = mR.predict_proba(fr.loc[te, cols])
    trm = real & (folds != 0)
    nov, _, _ = novelty(fr, cols, trm, te)
    cal = real & (folds == 1)
    ptr = mR.predict_proba(fr.loc[cal, cols]).max(1)
    ct = float(np.quantile(ptr, a.conf_q)) if a.conf_q > 0 else -1.0
    nt = float(np.quantile(nov[cal], a.nov_q)) if a.nov_q < 1.0 else np.inf
    predR = np.empty(len(fr), object)
    predR[te] = reject_rule(P4[te], nov[te], ct, nt)

    res = {"conf_q": a.conf_q, "nov_q": a.nov_q, "subtypes": {}}
    y5i = pd.Series(y5).map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    for s in subs:
        held = (y5 == "Other") & (raw == s)
        keep = ~held                               # rows allowed into training
        mT = A2.fit_fold(fr, y5i, cols, 0, seed=0, train_mask=keep, classes=CLASSES5)
        pT = np.array(CLASSES5)[mT.predict_proba(fr.loc[te, cols]).argmax(1)]
        hi = held[te]
        res["subtypes"][s] = {
            "n_rows_held": int(held.sum()), "n_test_rows": int(hi.sum()),
            "T_recall_on_unseen_subtype": round(float((pT[hi] == "Other").mean()), 4)
            if hi.any() else None,
            "R_recall_on_unseen_subtype": round(float((predR[te][hi] == "Other").mean()), 4)
            if hi.any() else None,
            "T_acc5_fold0_overall": round(float((y5[te] == pT).mean()), 4),
        }
        log(f"  {s}: n_test {int(hi.sum())} T {res['subtypes'][s]['T_recall_on_unseen_subtype']} "
            f"R {res['subtypes'][s]['R_recall_on_unseen_subtype']}")
    # a fully-trained T for reference (sees every subtype)
    mT = A2.fit_fold(fr, y5i, cols, 0, seed=0, classes=CLASSES5)
    pT = np.array(CLASSES5)[mT.predict_proba(fr.loc[te, cols]).argmax(1)]
    rec = {}
    for s in subs:
        hi = ((y5 == "Other") & (raw == s))[te]
        rec[s] = round(float((pT[hi] == "Other").mean()), 4) if hi.any() else None
    res["T_full_training"] = {"acc5_fold0": round(float((y5[te] == pT).mean()), 4),
                              "recall_by_subtype": rec}
    res["R_fold0"] = evaluate(fr, predR, te)
    json.dump(res, open(WORK / "a4_loso.json", "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["oof", "loso"])
    ap.add_argument("--conf_q", type=float, default=0.2)
    ap.add_argument("--nov_q", type=float, default=0.95)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
