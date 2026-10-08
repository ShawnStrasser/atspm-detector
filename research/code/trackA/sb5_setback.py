"""Setback distance for EVERY advance detector (note 41) -- core: features, estimators, and
`setback(events, predictions) -> table`.

Hi-res log only, phase-anonymous.  Inputs are the phase / function the classifier PREDICTED
(research: note-28 OOF; production: `predict.predict` output), never print labels.  Per detector
the classifier calls Advance or Mid:

  physics   travel time advance -> same-lane stop-bar zone (a detector predicted Presence / Count /
            Yellow_Red on the same predicted phase; same lane = note-30 pair model P >= .5), tau =
            note-39 mode-anchored median (`sb3_physics.robust_tau`), times a speed.  Speeds:
            `user` = note 39's hierarchy (partner ON time > phase Count > phase Presence > own ON
            (26 ft) > 40 mph); `lfit` = own ON time of isolated free-flow vehicles with an
            effective length fitted per fold, else `user`.  distance = v * tau + zone edge.
  learned   LightGBM quantile regression (P10 / P50 / P90 of log distance, 3 seeds) on behaviour
            features (phase green / red / cycle, own volume and ON times, green-arrival profile,
            queue spill-back and start-wave release, travel-time features, the physics estimates,
            the phase's predicted-function make-up).  Interval widened by split-conformal (CQR)
            scores from inner folds of the training folds (target 80 %).
  output    P50 = learned (physics is one of its inputs); < 15 ft -> 0.  Detectors predicted
            Presence / Count / Yellow_Red -> 0 ft.  Other / Bike -> no answer.

The same SQL runs on the research cache (sb5_eval.py) and on raw events (`setback`), so the
definitions exist once.  Time is seconds relative to the window start (w0).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import sb3_physics as S3  # noqa: E402  (robust_tau)
import lr2_pairs as L2  # noqa: E402    (pair cue SQL)
import lp3_auc as L3  # noqa: E402

V40 = 58.7
VMIN, VMAX = 22.0, 88.0
L_VEH = 20.0
ZONE = {"Count": 6.0, "Yellow_Red": 6.0, "Presence": 20.0}
EDGE = {"Count": 0.0, "Yellow_Red": 0.0, "Presence": 20.0}
ROLE_RANK = {"Count": 0, "Yellow_Red": 1, "Presence": 2}
STOPBAR = ("Presence", "Count", "Yellow_Red")
TARGET = ("Advance", "Mid")
CLIP0 = 15.0
PAIR_COLS = L3.ALLC + ["lph_a", "lph_b", "log_ratio", "n_det_phase"]

# --------------------------------------------------------------------------- SQL
# needs: iv(dev, det, t, dur, quiet) relative seconds; cyc(dev, phase, t_g, t_y, t_red, t_next);
#        ad(dev, det, phase, is_t, is_s)
SQL_ONC = """
CREATE OR REPLACE TEMP TABLE onc AS
WITH o AS (
  SELECT i.dev, i.det, ad.phase, i.t, i.dur, i.quiet, floor(i.t / 3600)::INT AS hr
  FROM iv i JOIN ad ON i.dev = ad.dev AND i.det = ad.det
), g AS (
  SELECT *, t - lag(t + dur) OVER w AS gap_prev, lead(t) OVER w - (t + dur) AS gap_next
  FROM o WINDOW w AS (PARTITION BY dev, det ORDER BY t)
), h AS (SELECT dev, det, hr, count(*) AS nh FROM o GROUP BY ALL),
hm AS (SELECT dev, det, median(nh) AS nh_med FROM h GROUP BY ALL)
SELECT g.*, (h.nh <= hm.nh_med) AS low, c.t_g, c.t_y, c.t_red, c.t_next,
       g.t - c.t_g AS since, c.t_y - g.t AS to_y,
       CASE WHEN g.t < c.t_y THEN 'G' WHEN g.t >= c.t_red THEN 'R' ELSE 'Y' END AS state
FROM g JOIN h USING (dev, det, hr) JOIN hm USING (dev, det)
ASOF JOIN cyc c ON g.dev = c.dev AND g.phase = c.phase AND g.t >= c.t_g
WHERE g.t < c.t_next
"""

SQL_SPEED = """
SELECT dev, det, count(*) AS n_iso, avg((dur <= 0.15)::INT) AS pulse,
       count(*) FILTER (low) AS n_low,
       median(dur) FILTER (dur > 0.15) AS dur50_all,
       median(dur) FILTER (dur > 0.15 AND low) AS dur50_low
FROM onc
WHERE to_y > 0 AND since >= 15 AND gap_prev >= 3 AND gap_next >= 3
GROUP BY ALL
"""

SQL_MATCH = """
WITH a AS (
  SELECT o.dev, o.det AS da, o.t, o.dur AS dur_a, o.low
  FROM onc o SEMI JOIN (SELECT dev, det FROM ad WHERE is_t) tg ON o.dev = tg.dev AND o.det = tg.det
  WHERE to_y >= 10 AND since >= 10 AND gap_prev >= 4 AND gap_next >= 2
), ap AS (SELECT a.*, p.db, a.t + 0.2 AS t2 FROM a JOIN ptab p ON a.dev = p.dev AND a.da = p.da),
b AS (SELECT dev, det, t FROM onc)
SELECT ap.dev, ap.da, ap.db, ap.low, ap.dur_a, b.t - ap.t AS dt
FROM ap ASOF JOIN b ON ap.dev = b.dev AND ap.db = b.det AND b.t >= ap.t2
WHERE b.t - ap.t <= 15
"""

SQL_BEH = """
WITH t AS (SELECT o.* FROM onc o SEMI JOIN (SELECT dev, det FROM ad WHERE is_t) x
           ON o.dev = x.dev AND o.det = x.det),
rel AS (   -- occupied across begin-green: the vehicle stopped on the zone leaves on the start wave
  SELECT dev, det, t_next, min(t + dur - t_next) AS rel
  FROM t WHERE state <> 'G' AND t + dur > t_next AND t + dur - t_next < 60 GROUP BY ALL
), fst AS (
  SELECT dev, det, t_g, min(since) AS first_lag FROM t WHERE state = 'G' GROUP BY ALL
), hold AS (   -- first ON in red that is a stopped car: time after red start
  SELECT dev, det, t_red, min(t - t_red) AS t_hold
  FROM t WHERE state = 'R' AND (dur >= 5 OR t + dur >= t_next) GROUP BY ALL
), b AS (
  SELECT dev, det, count(*) AS n_on_w, count(DISTINCT t_g) AS n_cyc_on,
         median(dur) AS dur_med, quantile_cont(dur, .25) AS dur_q25,
         quantile_cont(dur, .75) AS dur_q75, quantile_cont(dur, .9) AS dur_q90,
         avg((dur <= 0.15)::INT) AS pulse_share,
         avg((state = 'R')::INT) AS f_red, avg((state = 'Y')::INT) AS f_yel,
         avg((state = 'G' AND since < 5)::INT) AS f_g5,
         avg((state = 'G' AND since < 10)::INT) AS f_g10,
         avg((state = 'G' AND to_y < 10)::INT) AS f_glate,
         avg((state = 'R' AND dur >= 5)::INT) AS f_red_long,
         avg(dur) FILTER (state = 'R') AS dur_mean_red,
         avg(dur) FILTER (state = 'G') AS dur_mean_green,
         avg(quiet::INT) AS f_quiet
  FROM t GROUP BY ALL
)
SELECT b.*, r.n_rel, r.rel_med, r.rel_q25, f.first_lag_med, f.first_lag_q25, x.n_hold, x.hold_med
FROM b
LEFT JOIN (SELECT dev, det, count(*) AS n_rel, median(rel) AS rel_med,
                  quantile_cont(rel, .25) AS rel_q25 FROM rel GROUP BY ALL) r USING (dev, det)
LEFT JOIN (SELECT dev, det, median(first_lag) AS first_lag_med,
                  quantile_cont(first_lag, .25) AS first_lag_q25 FROM fst GROUP BY ALL) f
       USING (dev, det)
LEFT JOIN (SELECT dev, det, count(*) AS n_hold, median(t_hold) AS hold_med FROM hold
           GROUP BY ALL) x USING (dev, det)
"""

SQL_PHASE = """
SELECT dev, phase, count(*) AS n_cyc, avg(t_y - t_g) AS green_mean,
       quantile_cont(t_y - t_g, .9) AS green_q90, avg(t_next - t_red) AS red_mean,
       avg(t_next - t_g) AS cycle_mean,
       sum(t_y - t_g) / nullif(sum(t_next - t_g), 0) AS green_share
FROM cyc c SEMI JOIN (SELECT DISTINCT dev, phase FROM ad) a USING (dev, phase) GROUP BY ALL
"""


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- features
def _speed(row: dict | None, role: str) -> float:
    if row is None or pd.isna(row.get("dur")) or row["pulse"] >= 0.5:
        return np.nan
    v = (L_VEH + ZONE.get(role, 6.0)) / row["dur"]
    return v if VMIN <= v <= VMAX else np.nan


def features(con, dets: pd.DataFrame, hours: float, pair_model) -> pd.DataFrame:
    """con holds iv / cyc (relative seconds).  dets: dev, det, phase, fn (predicted), n_on,
    p_mid (optional).  Returns one row per target (fn in Advance / Mid, or is_t forced)."""
    d = dets.copy()
    d["det"] = d.det.astype("int16")
    if "is_t" not in d:
        d["is_t"] = d.fn.isin(TARGET)
    d["is_s"] = d.fn.isin(STOPBAR)
    nmin = 20 if hours >= 24 else 5
    act = d[d.n_on >= nmin]
    nd = act.groupby(["dev", "phase"]).size().rename("n_det_phase")
    comp = (d.assign(one=1).pivot_table(index=["dev", "phase"], columns="fn", values="one",
                                        aggfunc="sum", fill_value=0)
            .reindex(columns=["Advance", "Mid", "Presence", "Count", "Yellow_Red", "Other",
                              "Bike"], fill_value=0))
    comp.columns = [f"ph_n_{c}" for c in comp.columns]
    ad = d[d.is_t | d.is_s][["dev", "det", "phase", "is_t", "is_s"]]
    con.register("ad_df", ad)
    con.execute("CREATE OR REPLACE TEMP TABLE ad AS SELECT * FROM ad_df")
    con.execute(SQL_ONC)
    spd = con.sql(SQL_SPEED).df()
    spd["dur"] = np.where(spd.n_low >= 20, spd.dur50_low, spd.dur50_all)
    spd = spd.set_index(["dev", "det"])
    beh = con.sql(SQL_BEH).df()
    ph = con.sql(SQL_PHASE).df()
    # candidate stop-bar partners on the same predicted phase
    T = d[d.is_t][["dev", "det", "phase"]]
    S = d[d.is_s][["dev", "det", "phase", "fn"]]
    cand = T.rename(columns={"det": "da"}).merge(S.rename(columns={"det": "db"}),
                                                  on=["dev", "phase"])
    cand = cand[cand.da != cand.db]
    con.register("pt_df", cand[["dev", "da", "db"]])
    con.execute("CREATE OR REPLACE TEMP TABLE ptab AS SELECT * FROM pt_df")
    M = con.sql(SQL_MATCH).df()
    rows = []
    for (dev, da, db), g in M.groupby(["dev", "da", "db"]):
        lw = g.low.to_numpy()
        use = lw if lw.sum() >= 30 else np.ones(len(g), bool)
        r = S3.robust_tau(g.dt.to_numpy()[use], g.dur_a.to_numpy()[use])
        r.update(dev=dev, da=da, db=db)
        rows.append(r)
    TAU = pd.DataFrame(rows, columns=["dev", "da", "db", "n_match", "tau", "n_in",
                                      "pk_strength", "tau_med_raw", "mode", "d_veh", "v_veh"])
    # same-lane probability (note-30 pair model on note-17 cues, canonical da < db)
    ps = pair_probs(con, cand, act, nd, hours, pair_model)
    cand = cand.merge(TAU, on=["dev", "da", "db"], how="left").merge(
        ps, on=["dev", "da", "db"], how="left")
    cand["v_self"] = [_speed(spd.loc[(a, b)].to_dict() if (a, b) in spd.index else None, f)
                      for a, b, f in zip(cand.dev, cand.db, cand.fn)]
    Sv = S.copy()
    Sv["v"] = [_speed(spd.loc[(a, b)].to_dict() if (a, b) in spd.index else None, f)
               for a, b, f in zip(Sv.dev, Sv.det, Sv.fn)]
    vc = Sv[Sv.fn.isin(["Count", "Yellow_Red"])].groupby(["dev", "phase"]).v.median() \
        .rename("v_phase_count")
    vp = Sv[Sv.fn == "Presence"].groupby(["dev", "phase"]).v.median().rename("v_phase_pres")
    best = pd.DataFrame([_pick(g) for _, g in cand.groupby(["dev", "da"])])
    F = d[d.is_t].merge(beh, on=["dev", "det"], how="left")
    if len(best):
        F = F.merge(best.rename(columns={"da": "det"}), on=["dev", "det"], how="left")
    for c in ("tau", "role", "same", "p_same", "n_in", "pk_strength", "v_self", "tau_any",
              "pk_any", "p_same_any", "n_cand", "n_acc", "d_veh", "tau_raw_med"):
        if c not in F:
            F[c] = np.nan
    F = (F.merge(ph, on=["dev", "phase"], how="left")
         .merge(vc.reset_index(), on=["dev", "phase"], how="left")
         .merge(vp.reset_index(), on=["dev", "phase"], how="left")
         .merge(comp.reset_index(), on=["dev", "phase"], how="left")
         .merge(nd.reset_index(), on=["dev", "phase"], how="left"))
    adv = spd.reset_index()[["dev", "det", "pulse", "dur"]].rename(
        columns={"pulse": "adv_pulse", "dur": "adv_dur"})
    F = F.merge(adv, on=["dev", "det"], how="left")
    F["v_adv"] = np.where(F.adv_pulse < 0.5, (L_VEH + 6.0) / F.adv_dur, np.nan)
    F.loc[(F.v_adv < VMIN) | (F.v_adv > VMAX), "v_adv"] = np.nan
    # stop-bar volume on the phase, rank of this detector's travel time on its phase
    sbv = d[d.is_s].groupby(["dev", "phase"]).n_on.sum().rename("sb_on")
    F = F.merge(sbv.reset_index(), on=["dev", "phase"], how="left")
    F["on_ph"] = F.n_on_w / hours
    F["on_per_cyc"] = F.n_on_w / F.n_cyc
    F["vol_vs_sb"] = np.log((F.n_on_w + 1) / (F.sb_on.fillna(0) + 1))
    F["cyc_on_share"] = F.n_cyc_on / F.n_cyc
    F["rel_share"] = F.n_rel / F.n_cyc
    F["hold_share"] = F.n_hold / F.n_cyc
    F["tau_rank"] = F.tau_any / F.groupby(["dev", "phase"]).tau_any.transform("max")
    F["covered"] = F.tau.notna() & (F.same == 1)
    F["edge"] = F.role.map(EDGE).fillna(0.0).where(F.covered, 0.0)
    F["v_user"] = (F.v_self.fillna(F.v_phase_count).fillna(F.v_phase_pres)
                   .fillna(F.v_adv).fillna(V40))
    F["d_user"] = np.where(F.covered, F.tau * F.v_user + F.edge, np.nan)
    F["d_40"] = np.where(F.covered, F.tau * V40 + F.edge, np.nan)
    F["d_any40"] = F.tau_any * V40
    F["hours"] = hours
    return F


def _pick(g: pd.DataFrame) -> dict:
    """Same-lane first (P >= .5), Count > Yellow_Red > Presence, then P(same)."""
    g = g.assign(rank=g.fn.map(ROLE_RANK), ok=g.tau.notna(),
                 same=(g.p_same >= 0.5).fillna(False))
    acc = g[g.ok]
    r = {"dev": g.dev.iloc[0], "da": g.da.iloc[0], "n_cand": len(g), "n_acc": len(acc)}
    if acc.empty:
        return r
    # note 77: complete behavioural tie-breaks (stable sorts), never the partner's channel order
    b = acc.sort_values(["same", "rank", "p_same", "pk_strength", "n_in", "tau"],
                        ascending=[False, True, False, False, False, True], kind="stable").iloc[0]
    a = acc.sort_values(["pk_strength", "n_in", "p_same", "tau"], ascending=[False, False, False, True],
                        kind="stable").iloc[0]
    r.update(tau=b.tau, role=b.fn, same=float(b.same), p_same=b.p_same, n_in=b.n_in,
             pk_strength=b.pk_strength, v_self=b.v_self, d_veh=b.get("d_veh", np.nan),
             tau_raw_med=b.tau_med_raw, tau_any=a.tau, pk_any=a.pk_strength,
             p_same_any=a.p_same)
    return r


def pair_probs(con, cand, act, nd, hours, model) -> pd.DataFrame:
    """P(same lane) for the target -> partner pairs (note-17 cues, note-30 print model)."""
    out = pd.DataFrame(columns=["dev", "da", "db", "p_same"])
    if not len(cand) or model is None:
        return out
    a = act[["dev", "det"]]
    c = cand.merge(a.rename(columns={"det": "da"})).merge(a.rename(columns={"det": "db"}))
    if not len(c):
        return out
    c = c.assign(x=np.minimum(c.da, c.db), y=np.maximum(c.da, c.db))
    prs = c[["dev", "x", "y"]].drop_duplicates().rename(columns={"x": "da", "y": "db"})
    # note 77: score BOTH orientations and average -- the cues are not symmetric in a / b, so the channel order
    # (x = lower channel) must not decide which one the model sees
    prs = pd.concat([prs, prs.rename(columns={"da": "db", "db": "da"})], ignore_index=True)
    con.register("prs_df", prs)
    con.execute("CREATE OR REPLACE TEMP TABLE prs AS SELECT * FROM prs_df")
    con.execute("""CREATE OR REPLACE TEMP TABLE onw AS
        SELECT i.dev, i.det, i.t, i.quiet FROM iv i
        SEMI JOIN (SELECT dev, da AS det FROM prs UNION SELECT dev, db FROM prs) p
          ON i.dev = p.dev AND i.det = p.det""")
    nq = con.sql("SELECT dev, det, count(*) AS n, count(*) FILTER (quiet) AS nq "
                 "FROM onw GROUP BY ALL").df()
    corr = con.sql(L2.SQL_CORR).df()
    mm = con.sql(L2.SQL_MIN.format(nm=int(hours * 60))).df()
    if not len(corr):
        return out
    P = L2.cues(corr, nq).merge(mm, on=["dev", "da", "db"], how="left")
    P = P.replace([np.inf, -np.inf], np.nan)
    ph_ = c[["dev", "x", "y", "phase"]].drop_duplicates(["dev", "x", "y"])
    ph_ = pd.concat([ph_.rename(columns={"x": "da", "y": "db"}), ph_.rename(columns={"x": "db", "y": "da"})])
    P = P.merge(ph_, on=["dev", "da", "db"])
    P = P.merge(nd.reset_index(), on=["dev", "phase"], how="left")
    P = L3.derive(P)
    P["lph_a"] = np.log1p(P.n_a / hours)
    P["lph_b"] = np.log1p(P.n_b / hours)
    P["log_ratio"] = np.abs(np.log1p(P.n_a) - np.log1p(P.n_b))
    P["p_same"] = model.predict_proba(P[PAIR_COLS].to_numpy(float))[:, 1]
    P = P[["dev", "da", "db", "p_same"]].assign(x=lambda q: np.minimum(q.da, q.db), y=lambda q: np.maximum(q.da, q.db))
    P = P.groupby(["dev", "x", "y"], as_index=False).p_same.mean().rename(columns={"x": "da", "y": "db"})
    return pd.concat([P, P.rename(columns={"da": "db", "db": "da"})], ignore_index=True)


# --------------------------------------------------------------------------- estimators
FEATS = ["green_mean", "green_q90", "red_mean", "cycle_mean", "green_share", "n_cyc",
         "on_ph", "on_per_cyc", "vol_vs_sb", "cyc_on_share", "dur_med", "dur_q25", "dur_q75",
         "dur_q90", "pulse_share", "f_red", "f_yel", "f_g5", "f_g10", "f_glate", "f_red_long",
         "dur_mean_red", "dur_mean_green", "f_quiet", "rel_med", "rel_q25", "rel_share",
         "first_lag_med", "first_lag_q25", "hold_med", "hold_share", "adv_pulse", "adv_dur",
         "tau", "same", "p_same", "n_in", "pk_strength", "v_self", "tau_any", "pk_any",
         "p_same_any", "n_cand", "n_acc", "tau_rank", "v_phase_count", "v_phase_pres",
         "d_user", "d_lfit", "d_any_lfit", "role_c", "n_det_phase", "ph_n_Advance",
         "ph_n_Mid", "ph_n_Presence", "ph_n_Count", "ph_n_Yellow_Red", "ph_n_Other", "p_mid"]
STD_FEATS = ["green_mean", "green_q90", "red_mean", "cycle_mean", "green_share", "n_det_phase",
             "ph_n_Advance", "ph_n_Mid", "ph_n_Presence", "ph_n_Count", "ph_n_Yellow_Red",
             "p_mid"]          # "design standard": phase timing + phase make-up only
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbose=-1)


def add_physics(F: pd.DataFrame, L_fit: float) -> pd.DataFrame:
    F = F.copy()
    F["role_c"] = F.role.map({"Count": 0, "Yellow_Red": 1, "Presence": 2}).astype(float)
    v_l = np.where(F.adv_pulse < 0.5, L_fit / F.adv_dur, np.nan)
    v_l = np.where((v_l >= VMIN) & (v_l <= VMAX), v_l, np.nan)
    F["v_lfit"] = pd.Series(v_l, index=F.index).fillna(F.v_user)
    F["d_lfit"] = np.where(F.covered, F.tau * F.v_lfit + F.edge, np.nan)
    F["d_any_lfit"] = F.tau_any * F.v_lfit
    if "p_mid" not in F:
        F["p_mid"] = np.nan
    return F


def fit_L(F: pd.DataFrame) -> float:
    """Effective length (ft) for speed = L / own ON time, fitted on covered training rows."""
    m = F.covered & (F.adv_pulse < 0.5) & F.adv_dur.notna() & F.dist.notna() & (F.dist > 0)
    L = ((F.dist - F.edge) / F.tau * F.adv_dur)[m]
    return float(L.median()) if m.sum() >= 10 else 26.0


def _lg(y):
    return np.log(np.maximum(y, 10.0))


def fit_quant(X, y, seeds=(0, 1, 2), alphas=(0.1, 0.5, 0.9)):
    import lightgbm as lgb
    return {a: [lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=s, **LGB)
                .fit(X, y) for s in seeds] for a in alphas}


def pred_quant(ms, X) -> np.ndarray:
    P = np.column_stack([np.mean([m.predict(X) for m in ms[a]], 0) for a in sorted(ms)])
    return np.sort(P, axis=1)                    # no crossing


def fit_all(F: pd.DataFrame, seeds=(0, 1, 2), feats=None, inner=True) -> dict:
    """Everything the estimator needs, fitted on labelled rows F (dist, fold)."""
    feats = feats or FEATS
    L = fit_L(F)
    F = add_physics(F, L)
    y = _lg(F.dist.to_numpy())
    X = F[feats].astype(float)
    ms = fit_quant(X, y, seeds)
    q = 0.0
    if inner and F.fold.nunique() >= 3:          # CQR on inner folds (1 seed)
        sc = []
        for k in sorted(F.fold.unique()):
            tr, te = (F.fold != k).to_numpy(), (F.fold == k).to_numpy()
            P = pred_quant(fit_quant(X[tr], y[tr], (0,), (0.1, 0.9)), X[te])
            sc.append(np.maximum(P[:, 0] - y[te], y[te] - P[:, 1]))
        sc = np.concatenate(sc)
        q = float(np.quantile(sc, min(1.0, 0.8 * (1 + 1 / len(sc)))))
    return {"L_fit": L, "models": ms, "cqr": q, "feats": feats,
            "median": float(F.dist.median())}


def apply_fit(F: pd.DataFrame, fit: dict) -> pd.DataFrame:
    F = add_physics(F, fit["L_fit"])
    P = pred_quant(fit["models"], F[fit["feats"]].astype(float))
    out = pd.DataFrame(index=F.index)
    out["p10"] = np.exp(P[:, 0] - fit["cqr"])
    out["p50"] = np.exp(P[:, 1])
    out["p90"] = np.exp(P[:, 2] + fit["cqr"])
    for c in ("p10", "p50", "p90"):
        out[c] = np.where(out[c] < CLIP0, 0.0, out[c])
    out["d_lfit"] = np.where(F.d_lfit < CLIP0, 0.0, F.d_lfit)
    out["d_user"] = np.where(F.d_user < CLIP0, 0.0, F.d_user)
    out["covered"] = F.covered
    return out


# --------------------------------------------------------------------------- production-style
def setback(events, predictions: pd.DataFrame, model_path=None, threads: int = 4,
            memory: str = "4GB", internal: bool = False) -> pd.DataFrame:
    """Raw hi-res events + the classifier's per-detector output -> one row per detector:
    DeviceId, Detector, function, phase, distance_ft (ONE number: the P50; stop-bar = 0),
    setback_confidence (high / medium / low, from the internal P10-P90 width), method.
    internal=True also returns the P10 / P90 (not a user-facing output; user decision 2026-09-29).
    predictions: DeviceId, Detector and phase_guess / function_guess (predict.predict output) or
    phase / function; optional p_mid, n_actuations.  Advance / Mid -> estimate; Presence /
    Count / Yellow_Red -> 0; anything else -> NaN."""
    import predict as PR                                   # model/ (production loaders)
    pk = pickle.load(open(model_path or default_model_path(), "rb"))
    con = PR._connect(threads, memory)
    try:
        w0, w1, info = PR.load_events(con, events)
        if not info["n_events"]:
            return pd.DataFrame()
        PR.build_chunk_tables(con)
        hours = (w1 - w0) / 3600.0
        con.execute(f"""CREATE OR REPLACE TEMP TABLE iv AS
            SELECT m.DeviceId AS dev, o.det, o.t_on - {w0} AS t, o.dur::DOUBLE AS dur,
                   (floor((o.t_on % 86400) / 3600) >= 22 OR floor((o.t_on % 86400) / 3600) < 6) AS quiet  -- clock hour; hour(to_timestamp()) followed the session time zone (note 48)
            FROM onev_all o JOIN devmap m USING (dev)""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cyc AS
            SELECT m.DeviceId AS dev, c.p::INT AS phase, c.gs - {w0} AS t_g, c.ge - {w0} AS t_y,
                   c.rs - {w0} AS t_red, c.ng - {w0} AS t_next
            FROM cyc_all c JOIN devmap m USING (dev) WHERE c.ng IS NOT NULL""")
        p = predictions.copy()
        ph = "phase_guess" if "phase_guess" in p else "phase"
        fn = "function_guess" if "function_guess" in p else "function"
        p = p.rename(columns={"DeviceId": "dev", "Detector": "det", ph: "phase", fn: "fn"})
        p["fn"] = p.fn.astype(str).str.replace("yellow_red", "Yellow_Red").str.capitalize() \
            .str.replace("Yellow_red", "Yellow_Red")
        if "n_on" not in p:
            n = con.sql("SELECT dev, det, count(*) AS n_on FROM iv GROUP BY ALL").df()
            p = p.merge(n, on=["dev", "det"], how="left")
        p["n_on"] = p.n_on.fillna(0)
        p = p[p.phase.notna()].assign(phase=lambda x: x.phase.astype(int))
        F = features(con, p[["dev", "det", "phase", "fn", "n_on"] +
                            (["p_mid"] if "p_mid" in p else [])], hours, pk["pair_model"])
    finally:
        con.close()
    E = apply_fit(F, pk["fit"]) if len(F) else pd.DataFrame(columns=["p10", "p50", "p90"])
    F = pd.concat([F[["dev", "det", "fn", "phase"]], E[["p10", "p50", "p90", "covered"]]], axis=1)
    out = p[["dev", "det", "fn", "phase"]].merge(F, on=["dev", "det", "fn", "phase"], how="left")
    sb = out.fn.isin(STOPBAR)
    out.loc[sb, ["p10", "p50", "p90"]] = 0.0
    out["method"] = np.select([sb, out.p50.notna() & out.covered.eq(True), out.p50.notna()],
                              ["stop-bar zone", "learned + travel time", "learned"], "none")
    out["distance_ft"] = out.p50.round(0)
    out["setback_confidence"] = confidence(out.p10, out.p50, out.p90,
                                           pk["fit"].get("conf_cuts", CONF_CUTS))
    out.loc[sb, "setback_confidence"] = "high"
    out = out.rename(columns={"dev": "DeviceId", "det": "Detector", "fn": "function"})
    cols = ["DeviceId", "Detector", "function", "phase", "distance_ft", "setback_confidence",
            "method"]
    return out[cols + (["p10", "p90"] if internal else [])]


# relative P10-P90 width (log(P90 / P10)) cut points -> high / medium / low; set from the OOF
# width terciles in sb5_eval.py (stored in the fitted pickle as conf_cuts)
CONF_CUTS = (0.6, 1.0)


def confidence(p10, p50, p90, cuts=CONF_CUTS) -> pd.Series:
    w = np.log(np.maximum(np.asarray(p90, float), 15.0) / np.maximum(np.asarray(p10, float), 15.0))
    c = np.select([w <= cuts[0], w <= cuts[1]], ["high", "medium"], "low")
    return pd.Series(np.where(np.isnan(np.asarray(p50, float)), None, c), index=p50.index)


def default_model_path() -> Path:
    dcw = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
    return dcw / "trackA" / "setback" / "sb5_final.pkl"
