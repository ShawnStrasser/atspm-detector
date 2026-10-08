"""Setback distance, physics method (note 39), step 3 -- scoring.

Truth: printed single-number distance of Advance detectors (Mid apart), training/dev signals,
folds_v3 (signal-grouped).  Estimators (all from sb3_physics.py output):
  phys_user   the user's rule as written: speed from stop-bar zone ON time (partner > phase
              Count/YR > phase Presence > advance's own ON > 40 mph) x same-lane travel time.
  phys_adv    same, but the advance zone's own ON time first (it is in the lane, upstream, at
              free flow by construction), then the stop-bar hierarchy.
  phys_veh    per vehicle: (20 + 6 ft) / advance ON time x its own travel time, median over
              the matched vehicles (non-pulse advance zones); else phys_adv.
  phys_40     travel time x 40 mph (no ON-time speed).
Second step, reported apart (fitted on the other folds): cal_tech = travel time x a per-
technology speed; ml_resid = LightGBM on physics-only features (no green / red / volume /
phase size) predicting log(print / phys_adv); shuffled-label control.
No same-lane estimate -> "unknown"; in the all-detector table it falls back to the training-
fold median (as note 31).

    python sb4_eval.py
Writes %DC_WORK%/trackA/setback/sb4_results.csv.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import lightgbm as lgb
import lr_common as C

OUT = C.DCW / "trackA" / "setback"
V40 = 58.7
EDGE = {"Count": 0.0, "Yellow_Red": 0.0, "Presence": 20.0}
WINDOWS = ["h66_s0", "h24_s0", "h24_s40", "h6_s16", "h6_s60", "h2_s64", "h1_s64", "h1_s20"]
ML_FEATS = ["l_tau", "l_v_adv", "l_v_self", "l_v_pp", "l_v_pc", "l_dphys", "n_in", "pk",
            "adv_pulse", "adv_dur", "tech_c", "role_c"]


def metrics(y, p):
    e = np.abs(p - y)
    return {"n": len(y), "medAE_ft": round(float(np.median(e)), 0) if len(y) else np.nan,
            "within25pct": round(float(np.mean(e <= 0.25 * y)) * 100, 1) if len(y) else np.nan,
            "within50ft": round(float(np.mean(e <= 50)) * 100, 1) if len(y) else np.nan}


def clip0(d):
    return np.where(d < 15, 0.0, d)


def build(F: pd.DataFrame, pr: str = "model") -> pd.DataFrame:
    F = F.copy()
    tau, role = F[f"tau_{pr}"], F[f"role_{pr}"]
    F["cov"] = tau.notna() & (F[f"lane_{pr}"] == "same")
    F["cov_other"] = tau.notna() & (F[f"lane_{pr}"] == "other")
    edge = role.map(EDGE).fillna(0.0)
    F["edge"] = edge
    v_user = (F[f"v_self_{pr}"].fillna(F.v_phase_count).fillna(F.v_phase_pres)
              .fillna(F.v_adv).fillna(V40))
    v_adv = (F.v_adv.fillna(F[f"v_self_{pr}"]).fillna(F.v_phase_count)
             .fillna(F.v_phase_pres).fillna(V40))
    F["phys_user"] = clip0(tau * v_user + edge)
    F["phys_adv"] = clip0(tau * v_adv + edge)
    F["phys_40"] = clip0(tau * V40 + edge)
    F["phys_veh"] = F[f"d_veh_{pr}"].where(F[f"d_veh_{pr}"].notna(), F.phys_adv)
    F["tau"] = tau
    F["role"] = role
    F["v_used"] = v_adv
    F["partner_type"] = np.select([F["cov"] & (role == "Count"), F["cov"] & (role == "Yellow_Red"),
                                   F["cov"] & (role == "Presence"), F["cov_other"]],
                                  ["same-lane Count", "same-lane YR", "same-lane Presence",
                                   "other-lane only"], "none")
    lg = lambda s: np.log(s.astype(float))
    F["l_tau"], F["l_v_adv"] = lg(tau), lg(F.v_adv)
    F["l_v_self"], F["l_v_pp"], F["l_v_pc"] = lg(F[f"v_self_{pr}"]), lg(F.v_phase_pres), lg(F.v_phase_count)
    F["l_dphys"] = lg(F.phys_adv.clip(lower=15))
    F["n_in"], F["pk"] = F[f"n_in_{pr}"], F[f"pk_strength_{pr}"]
    F["tech_c"] = F.technology.map({"loop": 0, "radar": 1, "video": 2}).astype(float)
    F["role_c"] = role.map({"Count": 0, "Yellow_Red": 1, "Presence": 2}).astype(float)
    return F


def second_step(F: pd.DataFrame) -> pd.DataFrame:
    """Fold-fitted corrections on covered Advance rows only; out-of-fold."""
    F = F.copy()
    for c in ("median", "cal_tech", "ml_resid", "ml_shuffled"):
        F[c] = np.nan
    adv = F.function == "Advance"
    for k in sorted(F.fold.unique()):
        tr, te = adv & (F.fold != k), F.fold == k
        F.loc[te, "median"] = F.dist[tr].median()
        tc = tr & F["cov"]
        for t in ("loop", "radar", "video"):
            s = tc & (F.technology == t)
            sp = ((F.dist - F.edge) / F.tau)[s].median() if s.sum() >= 5 else \
                ((F.dist - F.edge) / F.tau)[tc].median()
            m = te & (F.technology == t)
            F.loc[m, "cal_tech"] = clip0(F.tau[m] * sp + F.edge[m])
        X, y = F.loc[tc, ML_FEATS].astype(float), np.log(F.dist[tc] / F.phys_adv[tc].clip(lower=15))
        pr = np.zeros(te.sum())
        for s in (0, 1, 2):
            m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7,
                                  min_child_samples=15, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=s, verbose=-1)
            m.fit(X, y)
            pr += m.predict(F.loc[te, ML_FEATS].astype(float)) / 3
        F.loc[te, "ml_resid"] = clip0(F.phys_adv[te].clip(lower=15) * np.exp(pr))
        rng = np.random.default_rng(k)
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7,
                              min_child_samples=15, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=0, verbose=-1)
        m.fit(X, rng.permutation(y.to_numpy()))
        F.loc[te, "ml_shuffled"] = clip0(F.phys_adv[te].clip(lower=15)
                                         * np.exp(m.predict(F.loc[te, ML_FEATS].astype(float))))
    for c in ("cal_tech", "ml_resid", "ml_shuffled"):
        F.loc[~F["cov"], c] = np.nan
    return F


EST = ["phys_user", "phys_adv", "phys_veh", "phys_40", "cal_tech", "ml_resid", "ml_shuffled"]


def tables(F: pd.DataFrame, win: str, pairing: str) -> list[dict]:
    rows = []
    adv = F.function == "Advance"
    def add(mask, label, est, fallback=False):
        p = F[est].where(F["cov"], F["median"]) if fallback else F[est]
        m = mask & (F["cov"] | fallback)
        r = metrics(F.dist[m].to_numpy(), p[m].to_numpy())
        rows.append({"window": win, "pairing": pairing, "subset": label, "estimator": est,
                     **r})
    add(adv, "Advance all (fallback fold median)", "median", True)
    for e in EST:
        add(adv, "Advance all (fallback fold median)", e, True)
        add(adv, "Advance covered (same-lane partner)", e)
        for t in ("loop", "radar", "video"):
            add(adv & (F.technology == t), f"covered {t}", e)
        for pt in ("same-lane Count", "same-lane Presence"):
            add(adv & (F.partner_type == pt), f"covered {pt}", e)
        add(adv & (F.partner_type == "same-lane Count") & (F.technology == "loop"),
            "covered loop, same-lane Count", e)
        add(adv & (F.technology == "loop"), "covered loop", e)
        add(adv & (F.dist <= 100), "covered <=100 ft", e)
        add(adv & (F.dist > 100), "covered >100 ft", e)
        add(F.function == "Mid", "Mid covered", e)
    # other-lane-only partners (not part of the method; shown for information)
    o = adv & F["cov_other"]
    for e in ("phys_adv", "phys_40"):
        r = metrics(F.dist[o].to_numpy(), F[e][o].to_numpy())
        rows.append({"window": win, "pairing": pairing, "subset": "other-lane partner only",
                     "estimator": e, **r})
    return rows


def main(suffix: str = ""):
    folds = pd.read_csv(C.DCW / "folds_v3.csv")
    folds["dev"] = folds.DeviceId.str.lower()
    allrows, cov = [], []
    for w in WINDOWS:
        fn = OUT / f"sb3_phys_{w}{suffix}.parquet"
        if not fn.exists():
            continue
        F0 = pd.read_parquet(fn)
        F0 = F0[F0.dist_kind == "single"].merge(folds[["dev", "fold"]], on="dev", how="inner")
        F0 = F0.reset_index(drop=True)
        for pr in ("model", "print"):
            F = second_step(build(F0, pr))
            allrows += tables(F, w, pr)
            a = F[F.function == "Advance"]
            cov.append({"window": w, "pairing": pr, "n_adv": len(a),
                        **a.partner_type.value_counts().to_dict(),
                        "lane_pick_correct": round(float(
                            (a[f"lane_truth_{pr}"][a["cov"]] == 1).mean()), 3)
                        if f"lane_truth_{pr}" in a else np.nan,
                        "lane_truth_known": int(a[f"lane_truth_{pr}"][a["cov"]].notna().sum())
                        if f"lane_truth_{pr}" in a else 0})
            if w == "h66_s0" and pr == "model":
                F.to_parquet(OUT / f"sb4_pred_h66_s0{suffix}.parquet", index=False)
    R = pd.DataFrame(allrows)
    R.to_csv(OUT / f"sb4_results{suffix}.csv", index=False)
    pd.DataFrame(cov).to_csv(OUT / f"sb4_coverage{suffix}.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(pd.DataFrame(cov).to_string(index=False))
    s = R[(R.window == "h66_s0")]
    print(s.pivot_table(index=["pairing", "subset"], columns="estimator",
                        values="medAE_ft", aggfunc="first").to_string())


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")
