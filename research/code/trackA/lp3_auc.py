"""Lanes from cabinet prints (note 30), step 3 -- same-lane cues and a supervised pair
model on the print truth.

Truth pairs: lane_pairs_labels.parquet (lp1) joined to the cues (lp2; both detectors
>= 20 actuations in the staging window).  Folds: %DC_WORK%/folds_v3.csv (signal-grouped).
Pair type from the PRINT function (for reporting only): adv-stop, stop-stop, adv-adv,
other (Mid / Other involved).  "multi" = pairs on phases the print gives >= 2 lanes: pairs
on a 1-lane phase are same-lane by construction, so multi is the discriminating set.

  1. univariate AUC per cue (fixed sign, from note 17), all / multi / multi by type.
  2. note-17 model transferred: logistic on note 17's cues, trained on the RL/CL/LL text
     pairs (lr3_pairs_scored.parquet) of the other folds, scored on the print pairs.
  3. supervised on the prints: logistic and gradient boosting, out-of-fold; feature sets
     cues / cues + A3 cue / + detector volumes and phase size / + predicted function
     (note-25 b7 OOF, only where it exists).
  4. the print-trained model scored on the text pairs (other folds only).
Writes lp3_auc.json and lp3_pairs_oof.parquet (OOF P(same lane) for EVERY lp2 pair, both
the note-17 model and the best print model -> input to lp4_nlanes.py).

    python lp3_auc.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lr_common as C
import lp1_labels as LP1

CL5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CUES = {
    "a_zero_lag": ["z0_ratio_all", "z0_sharp_all", "z0_exc_all",
                   "z0_ratio_q", "z0_sharp_q", "z0_exc_q"],
    "b_lead": ["lead_ratio_all", "lead_sharp_all", "lead_exc_all", "lead_asym_abs",
               "lead_ratio_q", "lead_sharp_q", "lead_exc_q"],
    "c_divergence": ["c_all", "c_off", "c_peak", "c_div", "r_shift"],
    "d_highpass": ["hp_all", "hp_off", "hp_peak"],
    "balance": ["bal"],
}
ALLC = sum(CUES.values(), [])
N17 = ["hp_off", "hp_all", "lead_exc_q", "lead_sharp_all", "c_off", "z0_sharp_all", "bal"]
PT = [f"pt_{c}" for c in CL5]
CTX = ["log_na", "log_nb", "log_ratio", "n_det_phase"]
HEAD = ["hp_off", "lead_exc_q", "lead_ratio_q", "c_off", "c_div", "z0_sharp_all", "a3_ex"]


def derive(P):
    P["c_div"] = P.c_off - P.c_peak
    P["r_shift"] = -np.abs(np.log(P.r_peak.clip(lower=1e-3) / P.r_off.clip(lower=1e-3)))
    P["lead_asym_abs"] = P.lead_asym_all.abs()
    return P


def a3_cue(P):
    fs = [C.DCW / "function_v4" / "feat_stg" / "det_lag_stg.parquet",
          C.WORK / "v5" / "det_lag_stg_new.parquet"]
    lag = []
    for f in fs:
        d = pd.read_parquet(f, columns=["DeviceId", "Detector", "other", "win",
                                        "lag_peak_excess"])
        lag.append(d[d.win.str.startswith("full")])
    lag = pd.concat(lag, ignore_index=True)
    lag["DeviceId"] = lag.DeviceId.str.lower()
    lag["da"] = np.minimum(lag.Detector, lag.other).astype("int16")
    lag["db"] = np.maximum(lag.Detector, lag.other).astype("int16")
    lag = lag.groupby(["DeviceId", "da", "db"]).lag_peak_excess.max().rename("a3_ex")
    return P.merge(lag.reset_index(), on=["DeviceId", "da", "db"], how="left")


def pred_type(P):
    v = pd.read_parquet(LP1.V3, columns=["DeviceId", "detector", "oof_pred"])
    v["DeviceId"] = v.DeviceId.str.lower()
    v["pf"] = v.oof_pred.replace({"Mid": "Other", "Bike": "Other"})
    v = v.rename(columns={"detector": "det"})[["DeviceId", "det", "pf"]]
    v["det"] = v.det.astype("int16")
    for s in ("a", "b"):
        P = P.merge(v.rename(columns={"det": f"d{s}", "pf": f"pf_{s}"}),
                    on=["DeviceId", f"d{s}"], how="left")
    for c in CL5:
        P[f"pt_{c}"] = np.where(P.pf_a.notna() & P.pf_b.notna(),
                                (P.pf_a == c).astype(float) + (P.pf_b == c).astype(float),
                                np.nan)
    return P


def all_pairs():
    """Every lp2 pair with cues, context, fold."""
    P = pd.read_parquet(C.WORK / "lp2_pairs.parquet")
    P["da"] = P.da.astype("int16")
    P["db"] = P.db.astype("int16")
    P = derive(a3_cue(P))
    P["log_na"] = np.log1p(P.n_a)
    P["log_nb"] = np.log1p(P.n_b)
    P["log_ratio"] = np.abs(P.log_na - P.log_nb)
    nd = pd.concat([P[["DeviceId", "target", "da"]].rename(columns={"da": "d"}),
                    P[["DeviceId", "target", "db"]].rename(columns={"db": "d"})]
                   ).drop_duplicates().groupby(["DeviceId", "target"]).size()
    P = P.merge(nd.rename("n_det_phase").reset_index(), on=["DeviceId", "target"])
    f = pd.read_csv(C.DCW / "folds_v3.csv")
    f["DeviceId"] = f.DeviceId.str.lower()
    P = P.merge(f[["DeviceId", "fold"]], on="DeviceId", how="left")
    P = pred_type(P)
    assert not P.DeviceId.isin(C.locked_ids()).any()
    return P


def truth(P):
    L = pd.read_parquet(C.WORK / "lane_pairs_labels.parquet")
    L["da"] = L.da.astype("int16")
    L["db"] = L.db.astype("int16")
    nl = pd.read_parquet(C.WORK / "nlanes_labels.parquet")[["DeviceId", "target", "n_lanes"]]
    L = L.merge(nl, on=["DeviceId", "target"])
    T = P.merge(L.drop(columns="target"), on=["DeviceId", "da", "db"], how="inner")
    stop = {"Presence", "Count", "Yellow_Red"}
    T["ptype"] = np.select(
        [T.f_a.isin(stop) & T.f_b.isin(stop),
         (T.f_a == "Advance") & T.f_b.isin(stop) | (T.f_b == "Advance") & T.f_a.isin(stop),
         (T.f_a == "Advance") & (T.f_b == "Advance")],
        ["stop-stop", "adv-stop", "adv-adv"], "other")
    T["multi"] = T.n_lanes >= 2
    T["spanning"] = (T.span_a > 1) | (T.span_b > 1)
    return T[T.fold.notna()].reset_index(drop=True)


def auc(y, s, m=None):
    m = np.ones(len(y), bool) if m is None else m
    ok = m & ~np.isnan(s)
    if ok.sum() < 20 or y[ok].min() == y[ok].max():
        return None
    return round(float(roc_auc_score(y[ok], s[ok])), 4)


def fit_predict(Xtr, ytr, Xte, model, seed=0):
    if model == "lr":
        med = np.nanmedian(Xtr, 0)
        med = np.where(np.isnan(med), 0, med)
        A = np.where(np.isnan(Xtr), med, Xtr)
        B = np.where(np.isnan(Xte), med, Xte)
        mu, sd = A.mean(0), A.std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=3000).fit((A - mu) / sd, ytr)
        return m.predict_proba((B - mu) / sd)[:, 1]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=20, random_state=seed)
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def oof(T, cols, model="lr", seed=0, target=None):
    """OOF on T; if `target` (a frame with a fold column) is given, also predict it."""
    X, y = T[cols].to_numpy(float), T.same_lane.to_numpy()
    out = np.full(len(T), np.nan)
    tout = None if target is None else np.full(len(target), np.nan)
    for f in sorted(T.fold.unique()):
        te = (T.fold == f).to_numpy()
        out[te] = fit_predict(X[~te], y[~te], X[te], model, seed)
        if target is not None:
            sel = (target.fold == f).to_numpy()
            if sel.any():
                tout[sel] = fit_predict(X[~te], y[~te], target.loc[sel, cols].to_numpy(float),
                                        model, seed)
    return out, tout


def summary(T, p):
    y = T.same_lane.to_numpy()
    mu = T.multi.to_numpy()
    r = {"all": auc(y, p), "multi": auc(y, p, mu)}
    for t in ("adv-stop", "stop-stop", "adv-adv", "other"):
        r[f"multi_{t}"] = auc(y, p, mu & (T.ptype == t).to_numpy())
    r["multi_nonspanning"] = auc(y, p, mu & ~T.spanning.to_numpy())
    return r


def main():
    P = all_pairs()
    T = truth(P)
    y = T.same_lane.to_numpy()
    res = {"n_pairs": int(len(T)), "n_signals": int(T.DeviceId.nunique()),
           "n_phases": int(T.groupby(["DeviceId", "target"]).ngroups),
           "share_same": round(float(y.mean()), 4),
           "multi_pairs": int(T.multi.sum()),
           "multi_share_same": round(float(y[T.multi].mean()), 4),
           "multi_by_type_n": T[T.multi].ptype.value_counts().to_dict(),
           "multi_by_type_same": T[T.multi].groupby("ptype").same_lane.mean().round(3).to_dict(),
           "multi_spanning_pairs": int((T.multi & T.spanning).sum()),
           "pred_type_coverage": round(float(T.pt_Advance.notna().mean()), 3)}
    C.log(json.dumps(res))
    uni = {}
    for c in ALLC + ["a3_ex"]:
        v = T[c].to_numpy(float)
        r = summary(T, v)
        r["coverage"] = round(float(np.isfinite(v).mean()), 3)
        uni[c] = r
        C.log(f"  {c:16s} {r}")
    res["univariate"] = uni

    # ---- note-17 model, trained on the RL/CL/LL text pairs of the other folds
    TX = pd.read_parquet(C.WORK / "lr3_pairs_scored.parquet")
    TX = TX[TX.fold.notna()].reset_index(drop=True)
    TX["same_lane"] = TX.same_lane.astype(int)
    tr17 = {}
    for name, cols in (("n17_compact_cues", N17), ("n17_all_cues+a3", ALLC + ["a3_ex"])):
        _, tp = oof(TX, cols, "lr", target=T)
        tr17[name] = summary(T, tp)
        C.log(f"  transfer {name}: {tr17[name]}")
        if name == "n17_compact_cues":
            T["p_n17"] = tp
            # the same for every pair (n_lanes input)
            _, pa = oof(TX, cols, "lr", target=P)
            P["p_n17"] = pa
    res["note17_model_on_prints"] = tr17

    # ---- supervised on the prints
    sets = {"cues": ALLC, "cues+a3": ALLC + ["a3_ex"], "cues+a3+ctx": ALLC + ["a3_ex"] + CTX}
    sup = {}
    for name, cols in sets.items():
        pl, _ = oof(T, cols, "lr")
        sup[f"lr/{name}"] = summary(T, pl)
        gs = [oof(T, cols, "gbm", s)[0] for s in (0, 1, 2)]
        sup[f"gbm/{name}"] = summary(T, np.mean(gs, 0))
        sup[f"gbm/{name}"]["seed_aucs_multi"] = [auc(y, g, T.multi.to_numpy()) for g in gs]
        C.log(f"  {name}: lr {sup[f'lr/{name}']} | gbm {sup[f'gbm/{name}']}")
    # predicted function pair type, where the note-25 OOF exists for both detectors
    cov = T.pt_Advance.notna().to_numpy()
    Tc = T[cov].reset_index(drop=True)
    for name, cols in (("cov/cues+a3+ctx", ALLC + ["a3_ex"] + CTX),
                       ("cov/cues+a3+ctx+pt", ALLC + ["a3_ex"] + CTX + PT)):
        g = np.mean([oof(Tc, cols, "gbm", s)[0] for s in (0, 1, 2)], 0)
        sup[f"gbm/{name}"] = summary(Tc, g)
        C.log(f"  {name}: {sup[f'gbm/{name}']}")
    sup["cov/n17_transfer"] = summary(Tc, T.p_n17.to_numpy()[cov])
    res["supervised_on_prints"] = sup

    # noise control: a pure-noise column added to the best set must not move it
    best_cols = sets["cues+a3+ctx"]
    rng = np.random.default_rng(0)
    T["noise"] = rng.normal(size=len(T))
    g = np.mean([oof(T, best_cols + ["noise"], "gbm", s)[0] for s in (0, 1, 2)], 0)
    res["noise_control_gbm_best+noise"] = summary(T, g)
    Ts = T.copy()
    Ts["same_lane"] = Ts.groupby("fold").same_lane.transform(
        lambda s: s.sample(frac=1, random_state=0).to_numpy())
    res["shuffled_label_control"] = summary(T, oof(Ts, best_cols, "gbm", 0)[0])
    C.log(f"  controls {res['noise_control_gbm_best+noise']} {res['shuffled_label_control']}")

    # ---- best print model: OOF on the truth pairs and on every pair; also on the text set
    ps, pa, px = [], [], []
    for s in (0, 1, 2):
        a, b = oof(T, best_cols, "gbm", s, target=P)
        ps.append(a)
        pa.append(b)
        px.append(oof(T, sets["cues+a3"], "gbm", s, target=TX)[1])  # no ctx on the text set
    T["p_print"] = np.mean(ps, 0)
    P["p_print"] = np.mean(pa, 0)
    TX["p_print"] = np.mean(px, 0)
    ytx = TX.same_lane.to_numpy()
    res["print_model_on_text_set"] = {
        "n": int(np.isfinite(TX.p_print).sum()),
        "auc": auc(ytx, TX.p_print.to_numpy()),
        "note17_oof_on_text_set": auc(ytx, TX.p_same.to_numpy())}
    # same comparison restricted to the text pairs whose features exist in both
    for t in ("stop-stop", "adv-stop"):
        m = (TX.ptype == t).to_numpy()
        res["print_model_on_text_set"][t] = [auc(ytx, TX.p_print.to_numpy(), m),
                                             auc(ytx, TX.p_same.to_numpy(), m)]
    C.log(f"  on text set {res['print_model_on_text_set']}")
    # importance proxy: univariate AUC of the OOF is above; store the pair table
    keep = ["DeviceId", "target", "da", "db", "fold", "n_a", "n_b", "p_n17", "p_print"]
    P[keep].to_parquet(C.WORK / "lp3_pairs_oof.parquet", index=False)
    T.drop(columns=["noise"]).to_parquet(C.WORK / "lp3_truth_scored.parquet", index=False)
    json.dump(res, open(C.WORK / "lp3_auc.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
