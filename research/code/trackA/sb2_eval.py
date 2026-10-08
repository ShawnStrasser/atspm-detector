"""Setback distance, first look (note 31) -- score the estimators built by sb1_setback.py.

Truth: printed single-number distance of Advance detectors (Mid reported apart).  Every
calibration constant and the LightGBM model are fitted on the other folds (folds_v3.csv,
signal-grouped), so every number is out-of-fold.  A detector an estimator cannot score falls
back to the training-fold median distance; coverage is reported.

    python sb2_eval.py [--feat sb1_feat_h66_s0.parquet ...]
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "trackA" / "setback"
V40 = 58.7
L_EFF = 22.0
FEATS = ["tau", "pk_ratio", "pk_n", "same_lane", "tau_any", "pk_ratio_any", "pk_n_any",
         "same_lane_any", "dur_med", "dur_q25", "dur_ff", "pulse_share", "nq_med", "nq_mean",
         "nq_q75", "ny_med", "ny_q75", "ny_max", "wq_med", "wq_q75", "tstop_med", "q_share",
         "on_per_cyc", "red_mean", "green_mean", "n_on_w", "n_red"]
PARTNER = ["tau", "pk_ratio", "pk_n", "same_lane", "tau_any", "pk_ratio_any", "pk_n_any",
           "same_lane_any"]
FEATSETS = {"c_lgbm": FEATS,
            "c_lgbm_seeds345": FEATS,
            "c_lgbm_no_partner": [f for f in FEATS if f not in PARTNER],
            "c_lgbm_travel_only": ["tau", "pk_ratio", "pk_n", "same_lane", "dur_ff"],
            "c_lgbm_no_volume": [f for f in FEATS if f not in
                                 ("on_per_cyc", "n_on_w", "n_red", "red_mean", "green_mean")]}
IMP = []


def metrics(y, p):
    e = np.abs(p - y)
    return {"n": len(y), "medAE_ft": round(float(np.median(e)), 0),
            "within25pct": round(float(np.mean(e <= 0.25 * y)) * 100, 1),
            "within50ft": round(float(np.mean(e <= 50)) * 100, 1)}


def fit_pow(x, y):
    """log y = a + b log x (robust-ish: plain least squares on logs)."""
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    return a, b


def run(F: pd.DataFrame, train_funcs=("Advance",), seeds=(0, 1, 2)) -> pd.DataFrame:
    F = F.copy()
    P = pd.DataFrame(index=F.index)
    for f in sorted(F.fold.unique()):
        tr, te = F[(F.fold != f) & F.function.isin(train_funcs)], F[F.fold == f]
        med = tr.dist.median()
        P.loc[te.index, "median"] = med
        # (a) travel time to the stop-bar partner
        ok = te.tau.notna()
        P.loc[te.index, "a_40mph"] = np.where(ok, te.tau * V40, med)
        sp = (tr.dist / tr.tau).median()
        P.loc[te.index, "a_speedfit"] = np.where(ok, te.tau * sp, med)
        t2 = tr[tr.tau.notna() & (tr.pk_ratio > 1)]
        a, b = fit_pow(t2.tau, t2.dist)
        P.loc[te.index, "a_powfit"] = np.where(ok, np.exp(a) * te.tau ** b, med)
        okd = ok & (te.dur_ff > 0.15)
        P.loc[te.index, "a_dur_speed"] = np.where(okd, te.tau * L_EFF / te.dur_ff, med)
        # (b) queue spill-back: cars ahead in the lane when the loop is first held in red
        okq = te.ny_q75.notna()
        t3 = tr[tr.ny_q75.notna()]
        sp_q = (t3.dist / (t3.ny_q75 + 1)).median()
        P.loc[te.index, "b_queue"] = np.where(okq, sp_q * (te.ny_q75 + 1), med)
        # (c) LightGBM, log target, several feature sets
        for name, fs in FEATSETS.items():
            sd = (3, 4, 5) if name.endswith("seeds345") else seeds
            pr = np.zeros(len(te))
            for s in sd:
                m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                      min_child_samples=10, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, random_state=s, verbose=-1,
                                      importance_type="gain")
                m.fit(tr[fs].astype(float), np.log(tr.dist))
                pr += m.predict(te[fs].astype(float)) / len(sd)
                if name == "c_lgbm":
                    IMP.append(pd.Series(m.feature_importances_, index=fs))
            P.loc[te.index, name] = np.exp(pr)
        # shuffled-label control: same model, training targets permuted within the fold
        rng = np.random.default_rng(0)
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=10, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=0, verbose=-1)
        m.fit(tr[FEATS].astype(float), rng.permutation(np.log(tr.dist.to_numpy())))
        P.loc[te.index, "c_lgbm_shuffled"] = np.exp(m.predict(te[FEATS].astype(float)))
    return P


def table(F, P, mask, label):
    rows = []
    for c in P.columns:
        r = {"subset": label, "estimator": c, **metrics(F.dist[mask].to_numpy(),
                                                          P[c][mask].to_numpy())}
        rows.append(r)
    return rows


def main(feats):
    folds = pd.read_csv(DCW / "folds_v3.csv")
    folds["dev"] = folds.DeviceId.str.lower()
    allrows = []
    for fn in feats:
        F = pd.read_parquet(OUT / fn)
        F = F[F.dist_kind == "single"].merge(folds[["dev", "fold"]], on="dev", how="inner")
        F = F.reset_index(drop=True)
        tag = fn.replace("sb1_feat_", "").replace(".parquet", "")
        for tf, ttag in ((("Advance",), "adv"),):
            P = run(F, tf)
            adv = F.function == "Advance"
            rows = (table(F, P, adv, "Advance all")
                    + table(F, P, adv & F.tau.notna(), "Advance w/ stop-bar partner")
                    + table(F, P, adv & F.tau.notna() & (F.technology == "loop")
                            & (F.same_lane == 1), "Advance loop, same-lane partner")
                    + table(F, P, adv & (F.dist <= 100), "Advance <=100 ft")
                    + table(F, P, adv & (F.dist > 100), "Advance >100 ft")
                    + table(F, P, F.function == "Mid", "Mid"))
            for r in rows:
                r["window"], r["train"] = tag, ttag
            allrows += rows
            P.assign(dev=F.dev, detector=F.detector, dist=F.dist, function=F.function,
                     fold=F.fold).to_parquet(OUT / f"sb2_pred_{tag}_{ttag}.parquet")
    R = pd.DataFrame(allrows)
    R.to_csv(OUT / "sb2_results.csv", index=False)
    imp = pd.concat(IMP, axis=1).mean(1)
    print((imp / imp.sum()).sort_values(ascending=False).round(3).head(12).to_string())
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", nargs="+", default=["sb1_feat_h66_s0.parquet"])
    main(ap.parse_args().feat)
