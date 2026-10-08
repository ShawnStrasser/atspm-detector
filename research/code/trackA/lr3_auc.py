"""Lanes redo, task 2b -- how well do the pair cues separate same-lane from
different-lane pairs on the lane-text signals?

Truth: RL / CL / LL in the channel text.  Same lane = same token AND the same sensor
unit ("Rad A", "Cam C") when both name one; a pair whose token matches but whose unit
is unknown on a phase served by two or more units is ambiguous and dropped (a phase can
serve two approaches, each with its own RL).

Univariate AUC per cue (oriented on all data, so it is an upper bound), then a logistic
regression on standardised cues, out-of-fold over the six signal-grouped folds, with and
without the predicted-function pair type (OOF argmax of the A2 function model, never
the label).  The A3 cue (`lag_peak_excess`, AUC .736 on its own pair set) is
recomputed on this pair set for a like-for-like reference.

    python lr3_auc.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lr_common as C

CUES = {
    "a_zero_lag": ["z0_ratio_all", "z0_sharp_all", "z0_exc_all",
                   "z0_ratio_q", "z0_sharp_q", "z0_exc_q"],
    "b_lead": ["lead_ratio_all", "lead_sharp_all", "lead_exc_all", "lead_asym_abs",
               "lead_ratio_q", "lead_sharp_q", "lead_exc_q"],
    "c_divergence": ["c_all", "c_off", "c_peak", "c_div", "r_shift"],
    "d_highpass": ["hp_all", "hp_off", "hp_peak"],
    "balance": ["bal"],
}


def truth_pairs() -> pd.DataFrame:
    lab = C.labelled_detectors()
    P = pd.read_parquet(C.WORK / "lr2_pairs.parquet")
    L = lab[["DeviceId", "Detector", "target", "lane", "unit", "func5"]].copy()
    L["Detector"] = L.Detector.astype("int16")
    for s in ("a", "b"):
        P = P.merge(L.rename(columns={"Detector": f"d{s}", "lane": f"lane_{s}",
                                      "unit": f"unit_{s}", "func5": f"f_{s}",
                                      "target": f"tg_{s}"}),
                    on=["DeviceId", f"d{s}"], how="left")
    P = P[P.lane_a.notna() & P.lane_b.notna()].copy()
    nunit = lab.groupby(["DeviceId", "target"]).unit.nunique().rename("n_units")
    P = P.merge(nunit, left_on=["DeviceId", "tg_a"], right_index=True, how="left")
    both = P.unit_a.notna() & P.unit_b.notna()
    same_tok = P.lane_a == P.lane_b
    P["same_lane"] = np.where(both, same_tok & (P.unit_a == P.unit_b), same_tok)
    ambiguous = same_tok & ~both & (P.n_units >= 2)
    P = P[~ambiguous].copy()
    P["same_lane"] = P.same_lane.astype(bool)
    stop = {"Presence", "Count", "Yellow_Red"}
    P["ptype"] = np.select(
        [P.f_a.isin(stop) & P.f_b.isin(stop),
         (P.f_a == "Advance") & P.f_b.isin(stop) | (P.f_b == "Advance") & P.f_a.isin(stop),
         (P.f_a == "Advance") & (P.f_b == "Advance")],
        ["stop-stop", "adv-stop", "adv-adv"], "other")
    P["c_div"] = P.c_off - P.c_peak
    P["r_shift"] = -np.abs(np.log(P.r_peak.clip(lower=1e-3) / P.r_off.clip(lower=1e-3)))
    P["lead_asym_abs"] = P.lead_asym_all.abs()
    return P


def add_a3_cue(P: pd.DataFrame) -> pd.DataFrame:
    lag = []
    for per, fs in (("dec", [C.DCW / "features" / "det_lag.parquet",
                             C.DCW / "features" / "det_lag_B.parquet"]),
                    ("stg", [C.DCW / "function_v4" / "feat_stg" / "det_lag_stg.parquet"])):
        for f in fs:
            if f.exists():
                d = pd.read_parquet(f, columns=["DeviceId", "Detector", "other", "win",
                                                "lag_peak_excess"])
                d = d[d.win.str.startswith("full")]
                d["period"] = per
                lag.append(d)
    lag = pd.concat(lag, ignore_index=True)
    lag["DeviceId"] = lag.DeviceId.str.lower()
    lag["da"] = np.minimum(lag.Detector, lag.other).astype("int16")
    lag["db"] = np.maximum(lag.Detector, lag.other).astype("int16")
    lag = lag.groupby(["DeviceId", "period", "da", "db"]).lag_peak_excess.max()
    return P.merge(lag.rename("a3_ex").reset_index(), on=["DeviceId", "period", "da", "db"],
                   how="left")


def pred_type(P: pd.DataFrame) -> pd.DataFrame:
    keys = pd.read_parquet(C.WORK / "a2_oof_keys.parquet")
    Pr = np.load(C.WORK / "a2_oof_base_expert_mean.npy")
    keys["pf"] = np.array(["Advance", "Presence", "Count", "Yellow_Red", "Other"])[
        Pr.argmax(1)]
    k = keys[keys.wgroup == "full"][["DeviceId", "period", "Detector", "pf", "fold"]]
    k["DeviceId"] = k.DeviceId.str.lower()
    k["Detector"] = k.Detector.astype("int16")
    for s in ("a", "b"):
        P = P.merge(k.rename(columns={"Detector": f"d{s}", "pf": f"pf_{s}",
                                      "fold": f"fold_{s}"}),
                    on=["DeviceId", "period", f"d{s}"], how="left")
    P["fold"] = P.fold_a.fillna(P.fold_b)
    return P


def oof(P, cols, model="lr", seed=0):
    X = P[cols].to_numpy(float)
    y = P.same_lane.to_numpy()
    out = np.full(len(P), np.nan)
    for f in sorted(P.fold.dropna().unique()):
        te = (P.fold == f).to_numpy()
        tr = ~te & P.fold.notna().to_numpy()
        med = np.nanmedian(X[tr], 0)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        Xte = np.where(np.isnan(X[te]), med, X[te])
        if model == "lr":
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, y[tr])
            out[te] = m.predict_proba((Xte - mu) / sd)[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               max_leaf_nodes=15, min_samples_leaf=20,
                                               random_state=seed).fit(X[tr], y[tr])
            out[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(out)
    return float(roc_auc_score(y[ok], out[ok])), out


def main():
    P = pred_type(add_a3_cue(truth_pairs()))
    P = P[P.fold.notna()].reset_index(drop=True)
    C.log(f"{len(P)} pairs, {P.DeviceId.nunique()} signals, same-lane share "
          f"{P.same_lane.mean():.3f}; by type {P.ptype.value_counts().to_dict()}")
    res = {"n_pairs": int(len(P)), "n_signals": int(P.DeviceId.nunique()),
           "share_same": round(float(P.same_lane.mean()), 4),
           "by_type_n": P.ptype.value_counts().to_dict(),
           "by_type_same": P.groupby("ptype").same_lane.mean().round(3).to_dict()}
    y = P.same_lane.to_numpy()
    uni = {}
    for fam, cols in list(CUES.items()) + [("a3_reference", ["a3_ex"])]:
        for c in cols:
            v = P[c].to_numpy(float)
            ok = ~np.isnan(v)
            a = roc_auc_score(y[ok], v[ok])
            row = {"auc": round(a, 4), "coverage": round(float(ok.mean()), 3)}
            for t in ("stop-stop", "adv-stop", "adv-adv"):
                m = ok & (P.ptype == t).to_numpy()
                if m.sum() > 20 and 0 < y[m].mean() < 1:
                    row[t] = round(roc_auc_score(y[m], v[m]), 4)
            uni[c] = row
            C.log(f"  {c:16s} {json.dumps(row)}")
    res["univariate"] = uni

    tcols = []
    for s in ("a", "b"):
        for c in ("Advance", "Presence", "Count", "Yellow_Red", "Other"):
            P[f"pf_{s}_{c}"] = (P[f"pf_{s}"] == c).astype(float)
    # symmetric pair-type one-hot from predicted function
    for c in ("Advance", "Presence", "Count", "Yellow_Red", "Other"):
        P[f"pt_{c}"] = P[f"pf_a_{c}"] + P[f"pf_b_{c}"]
        tcols.append(f"pt_{c}")
    allc = sum(CUES.values(), [])
    combos = {f"{k}": v for k, v in CUES.items()}
    combos["a+b"] = CUES["a_zero_lag"] + CUES["b_lead"]
    combos["all_cues"] = allc
    combos["all_cues+a3"] = allc + ["a3_ex"]
    combos["all_cues+a3+pred_type"] = allc + ["a3_ex"] + tcols
    comb = {}
    for name, cols in combos.items():
        a, _ = oof(P, cols)
        comb[name] = {"lr": round(a, 4)}
        if name.startswith("all_cues"):
            comb[name]["gbm"] = round(np.mean([oof(P, cols, "gbm", s)[0]
                                               for s in (0, 1, 2)]), 4)
        C.log(f"  OOF {name:24s} {comb[name]}")
    a, pr = oof(P, allc + ["a3_ex"] + tcols)
    P["p_same"] = pr
    comb["best_lr_by_type"] = {
        t: round(roc_auc_score(y[(P.ptype == t).to_numpy()],
                               pr[(P.ptype == t).to_numpy()]), 4)
        for t in ("stop-stop", "adv-stop", "adv-adv")
        if 0 < y[(P.ptype == t).to_numpy()].mean() < 1}
    C.log(f"  by type {comb['best_lr_by_type']}")
    res["oof_combined"] = comb
    P.to_parquet(C.WORK / "lr3_pairs_scored.parquet", index=False)
    json.dump(res, open(C.WORK / "lr3_auc.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
