"""Permissive-phase SCORE per (signal, period, window, candidate phase), out-of-fold (note 50 experiment).

Inputs: the pm_ph_* phase features of pm_features.py (allowed codes only, phase-anonymous) + log window length.
Target (training only, never an input): the phase logs >= 5 FYA begin-permissive events (32) in that period, or the
timing says a detector of it also calls its through phase (label_check_pplt by_timing). Trained only at signals
that log event 32 in that period (elsewhere a missing event says nothing). Six signal-grouped folds (the frame's
folds_v4 fold), 3 seeds, small LightGBM; every phase row of fold k is scored by the fold-k model, so pm_score is
out-of-fold for every signal, including the ones without FYA logging.

    python pm_score.py        -> %DC_WORK%/trackA/pm/pm_score.parquet + AUC table on stdout
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pm_features as PF

DCW = PF.DCW
OUT = PF.OUT
FEATS_FR = DCW / "trackA" / "v3" / "frame_v6" / "feat_frame.parquet"
PPLT = DCW / "cabinet" / "label_check_pplt.parquet"
SECS = {n: s for per in PF.A2F.WINDOWS.values() for n, _, s in per}
LEFT = {1, 3, 5, 7}   # EVALUATION ONLY (agency convention), never a feature
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, n_estimators=300, verbose=-1, n_jobs=6)


def wgroup(w: str) -> str:
    return "full" if w.startswith("full") else w.split("_")[0]


def load() -> pd.DataFrame:
    P = pd.concat([pd.read_parquet(OUT / f"pm_ph_{p}.parquet") for p in ("dec", "stg")], ignore_index=True)
    P["p"] = P.p.astype(int)
    fold = pd.read_parquet(FEATS_FR, columns=["DeviceId", "fold"]).drop_duplicates("DeviceId")
    P = P.merge(fold, on="DeviceId", how="inner")
    L = pd.read_parquet(OUT / "pm_labels.parquet")
    logs = set(zip(L.DeviceId, L.period))
    P["logs32"] = [(d, per) in logs for d, per in zip(P.DeviceId, P.period)]
    P = P.merge(L[["DeviceId", "period", "p", "n32"]], on=["DeviceId", "period", "p"], how="left")
    pp = pd.read_parquet(PPLT)
    tim = set(zip(pp.loc[pp.by_timing, "sid"].str.lower(), pp.loc[pp.by_timing, "p"].astype(int)))
    beh = set(zip(pp.loc[pp.by_behaviour, "sid"].str.lower(), pp.loc[pp.by_behaviour, "p"].astype(int)))
    P["by_timing"] = [(d, p) in tim for d, p in zip(P.DeviceId, P.p)]
    P["by_beh"] = [(d, p) in beh for d, p in zip(P.DeviceId, P.p)]
    P["by_events"] = P.n32.fillna(0).ge(5)
    P["y"] = P.by_events | P.by_timing
    P["train"] = P.logs32
    P["left"] = P.p.isin(LEFT)
    P["log_secs"] = np.log(P.win.map(SECS))
    P["wg"] = P.win.map(wgroup)
    return P


def main() -> None:
    import lightgbm as lgb
    P = load()
    X = PF.PH_FEATS + ["log_secs"]
    oof = np.zeros(len(P))
    for s in (0, 1, 2):
        for k in range(6):
            tr = P.train & (P.fold != k)
            m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(P.loc[tr, X], P.loc[tr, "y"])
            te = (P.fold == k).to_numpy()
            oof[te] += m.predict_proba(P.loc[te, X])[:, 1] / 3
    P["pm_score"] = oof
    P[["DeviceId", "period", "win", "p", "pm_score"]].to_parquet(OUT / "pm_score.parquet", index=False)
    print(f"phase rows {len(P):,}; training rows (signals logging 32) {int(P.train.sum()):,}, positives "
          f"{int((P.train & P.y).sum()):,}; signals {P.DeviceId.nunique()}")
    rows = []
    sets = {"logging, left phases: events|timing vs none": P.train & P.left,
            "logging, all phases": P.train,
            "non-logging, left: timing vs none (not beh.)": ~P.train & P.left & (P.by_timing | ~P.by_beh)}
    for sn, m in sets.items():
        for g in ["all", "m5", "m30", "h6", "full"]:
            mm = m & (P.wg == g if g != "all" else True)
            y = P.y[mm]
            if y.nunique() < 2:
                continue
            r = {"set": sn, "wg": g, "n": int(mm.sum()), "pos": int(y.sum()),
                 "sig": P.DeviceId[mm].nunique(), "score": roc_auc_score(y, P.pm_score[mm])}
            for c in ("pm_ph_leave_sb", "pm_ph_leave_wmean", "pm_ph_drop43"):
                ok = mm & P[c].notna()
                r[c] = roc_auc_score(P.y[ok], P[c][ok]) if P.y[ok].nunique() == 2 else np.nan
                r[c + "_cov"] = ok.sum() / max(mm.sum(), 1)
            rows.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(float_format=lambda x: f"{x:.3f}"))
    # behaviour-only phases (note 44's label, non-logging signals): their score distribution
    bo = ~P.train & P.left & P.by_beh & ~P.by_timing
    ot = ~P.train & P.left & ~P.by_beh & ~P.by_timing
    print(f"non-logging left phases: behaviour-flagged median score {P.pm_score[bo].median():.3f} (n {bo.sum()}), "
          f"unflagged {P.pm_score[ot].median():.3f} (n {ot.sum()})")


if __name__ == "__main__":
    main()
