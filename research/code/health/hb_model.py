"""Note 38: self-supervised health model = LightGBM on the per-detector-vs-siblings statistics,
trained on synthetic faults, six signal-grouped folds (folds_v4).

    python hb_model.py [--synth synth_s0.parquet] [--seed 0] [--shuffle]
      -> %DC_WORK%/health/model_oof_s{seed}.parquet   (synthetic rows, out-of-fold p_fault)
         %DC_WORK%/health/model_real_s{seed}.parquet  (real windows, fold model of the signal)
`--shuffle` = shuffled-label control.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

FEATS = ["n_on", "cov_h", "sib_on", "n_sib", "share", "rate_h", "max5", "occ_frac", "dur_max",
         "chat_frac", "n_fault", "drop_lam", "drop_frac", "drop_to_end", "dead_lam",
         "level_ratio", "level_llr", "corr", "corr_exp", "disp", "corr_gap", "zero15_exp",
         "max_dev_1h", "mean_on_s", "occ_max_bin", "night_day", "sig_night_day", "disp_z",
         "log_n_on", "log_share", "co_silent"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=6, verbose=-1)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    nb = (X.cov_h * H.BPH).clip(lower=1)
    X["drop_frac"] = (X.drop_b1 - X.drop_b0).clip(lower=0) / nb
    X["drop_to_end"] = X.drop_to_end.astype(float)
    X["log_n_on"] = np.log1p(X.n_on)
    X["log_share"] = np.log(X.share + 1e-4)
    return X[FEATS].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="synth_s0.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()
    S = pd.read_parquet(H.HB / a.synth)
    S = S[S.label.ne("unknown") & S.fold.notna()].reset_index(drop=True)
    y = S.label.ne("none").astype(int).to_numpy()
    if a.shuffle:
        y = np.random.default_rng(a.seed).permutation(y)
    X = prep(S)
    R = pd.read_parquet(H.HB / "real_stats.parquet")
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    fmap = dict(zip(folds.DeviceId.str.lower(), folds.fold))
    R["fold"] = R.DeviceId.map(fmap)
    R = R[R.fold.notna()].reset_index(drop=True)
    XR = prep(R)
    S["p_model"], R["p_model"] = np.nan, np.nan
    imp = []
    for k in range(6):
        tr, te = S.fold.ne(k).to_numpy(), S.fold.eq(k).to_numpy()
        m = lgb.train({**PARAMS, "seed": a.seed}, lgb.Dataset(X[tr], y[tr]), a.rounds)
        S.loc[te, "p_model"] = m.predict(X[te])
        # threshold: 2 % false alarms on the training folds' clean rows (not the test fold)
        pc = m.predict(X[tr & (S.label.eq("none").to_numpy())])
        S.loc[te, "thr"] = np.quantile(pc, 0.98)
        rk = R.fold.eq(k).to_numpy()
        R.loc[rk, "p_model"] = m.predict(XR[rk])
        R.loc[rk, "thr"] = np.quantile(pc, 0.98)
        imp.append(pd.Series(m.feature_importance("gain"), index=FEATS))
        print("fold", k, "done", flush=True)
    tag = f"s{a.seed}" + ("_shuf" if a.shuffle else "")
    S[["DeviceId", "period", "rep", "win_h", "detector", "label", "changed", "fold", "p_model",
       "thr"]].to_parquet(H.HB / f"model_oof_{tag}.parquet")
    R[["DeviceId", "period", "window", "detector", "p_model", "thr"]].to_parquet(
        H.HB / f"model_real_{tag}.parquet")
    print((sum(imp) / 6).sort_values(ascending=False).head(15).round(0).to_string())


if __name__ == "__main__":
    main()
