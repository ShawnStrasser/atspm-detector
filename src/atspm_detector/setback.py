"""Setback distance of every detector the function model calls Advance or Mid (note 41; packaged in note 48).

One number per detector, `distance_ft` = the model's median estimate of the distance from the stop bar
(feet).  Detectors the function model calls Presence / Count / Yellow_Red sit at the stop bar -> 0.
Other / Bike -> no answer.  Estimates under 15 ft -> 0.  `setback_confidence` = high / medium / low from
the width of the model's internal 10-90 % interval (terciles over all predicted Advance / Mid in training;
no truth used).  The interval itself is internal, not an output.

Hi-res log only, phase-anonymous: the predicted phase is only a grouping key (which stop-bar zones share
a phase with the advance loop, which cycles to measure against), never a feature.  Inputs:
  * behaviour of the detector against its predicted phase's cycles (volume, ON times, arrival shares in
    green / red, start-wave release, first-ON lag, held-in-red timing),
  * phase timing and the phase's make-up by predicted function (the agency's design standard shows here:
    the model must be re-fitted per agency),
  * travel time to a same-lane stop-bar zone on the same phase (note 39's physics, same-lane probability
    from the note-30 pair model) and the physics distance estimates.
Model: LightGBM quantile regression -- P50 of log distance, one model per sample-length group (m30 / h6 / h24 / full);
P10 / P90 band (3 seeds each), widened by split-conformal scores -- run as ONNX tree ensembles.  The same-lane pair
model is a scikit-learn HistGradientBoostingClassifier exported to JSON and evaluated here with numpy (`HGBNumpy`).
Research code: research/code/trackA/sb5_setback.py (definitions), sb6_export.py (export + parity).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from .common import read_json



V40 = 58.7                      # 40 mph in ft/s
VMIN, VMAX = 22.0, 88.0
L_VEH = 20.0
ZONE = {"Count": 6.0, "Yellow_Red": 6.0, "Presence": 20.0}
EDGE = {"Count": 0.0, "Yellow_Red": 0.0, "Presence": 20.0}
ROLE_RANK = {"Count": 0, "Yellow_Red": 1, "Presence": 2}
STOPBAR = ("Presence", "Count", "Yellow_Red")
TARGET = ("Advance", "Mid")
CLIP0 = 15.0
PAIR_COLS = ["z0_ratio_all", "z0_sharp_all", "z0_exc_all", "z0_ratio_q", "z0_sharp_q", "z0_exc_q",
             "lead_ratio_all", "lead_sharp_all", "lead_exc_all", "lead_asym_abs", "lead_ratio_q",
             "lead_sharp_q", "lead_exc_q", "c_all", "c_off", "c_peak", "c_div", "r_shift", "hp_all",
             "hp_off", "hp_peak", "bal", "lph_a", "lph_b", "log_ratio", "n_det_phase"]


# --------------------------------------------------------------------------- models (numpy only)
class HGBNumpy:
    """sklearn HistGradientBoostingClassifier (binary, numeric splits) from its exported node arrays."""

    def __init__(self, path):
        m = read_json(path)
        self.base = float(m["baseline"])
        self.n_features = int(m["n_features"])
        self.trees = [{k: np.asarray(v) for k, v in t.items()} for t in m["trees"]]

    def raw(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        out = np.full(X.shape[0], self.base)
        for t in self.trees:
            node = np.zeros(X.shape[0], dtype=np.int64)
            leaf = t["is_leaf"].astype(bool)
            act = np.flatnonzero(~leaf[node])
            while act.size:
                nd = node[act]
                v = X[act, t["feature_idx"][nd]]
                go_left = np.where(np.isnan(v), t["missing_go_to_left"][nd].astype(bool),
                                   v <= t["num_threshold"][nd])
                node[act] = np.where(go_left, t["left"][nd], t["right"][nd])
                act = act[~leaf[node[act]]]
            out += t["value"][node]
        return out

    def predict_proba(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw(X)))


class SetbackModel:
    """The distance = sb7 P50 (note 58), one LightGBM P50 model per sample-length group (m30 / h6 / h24 / full, one seed;
    the group by sample hours, setback_p50.json group_rule_hours); the confidence band = the note-41 P10 / P90 models
    (3 seeds each, split-conformal widened).  All run as ONNX tree ensembles, loaded on first use."""

    def __init__(self, model_dir):
        md = self.md = Path(model_dir)
        self.meta = read_json(md / "setback_model.json")
        self.p50 = read_json(md / "setback_p50.json")
        self.pair = HGBNumpy(md / self.meta["pair_model"])
        self.features = self.meta["features"]
        self._bags: dict = {}

    def _bag(self, key, files):
        """tree sessions are loaded on first use: one P50 length group per call, the band models only when some
        detector is called Advance / Mid."""
        if key not in self._bags:
            from . import trees_onnx
            self._bags[key] = trees_onnx.Bag([self.md / f for f in files])
        return self._bags[key]

    def band(self, a: float):
        return self._bag(("q", a), {float(k): v for k, v in self.meta["band_models"].items()}[a])

    def p50_model(self, g: str):
        return self._bag(("g", g), self.p50["groups"][g]["onnx_files"])

    def group(self, hours: float) -> str:
        for g, (lo, hi) in self.p50["group_rule_hours"].items():
            if lo <= hours < hi:
                return g
        return "full"

    def apply(self, F: pd.DataFrame, hours: float) -> pd.DataFrame:
        g = self.group(hours)
        Fg = add_physics(F, self.p50["groups"][g]["L_fit"])
        p50 = np.exp(self.p50_model(g).predict(Fg[self.p50["features"]].astype(float)))
        Fb = add_physics(F, self.meta["L_fit"])
        Xb = Fb[self.features].astype(float)
        q = self.meta["cqr"]
        out = pd.DataFrame(index=F.index)
        out["p10"] = np.minimum(np.exp(self.band(0.1).predict(Xb) - q), p50)
        out["p50"] = p50
        out["p90"] = np.maximum(np.exp(self.band(0.9).predict(Xb) + q), p50)
        for c in ("p10", "p50", "p90"):
            out[c] = np.where(out[c] < CLIP0, 0.0, out[c])
        out["covered"] = F.covered
        out["group"] = g
        return out


# --------------------------------------------------------------------------- SQL (sb5_setback / lr2_pairs)
SQL_ONC = """
CREATE OR REPLACE TEMP TABLE sb_onc AS
WITH o AS (
  SELECT i.dev, i.det, ad.phase, i.t, i.dur, i.quiet, floor(i.t / 3600)::INT AS hr
  FROM sb_iv i JOIN sb_ad ad ON i.dev = ad.dev AND i.det = ad.det
), g AS (
  SELECT *, t - lag(t + dur) OVER w AS gap_prev, lead(t) OVER w - (t + dur) AS gap_next
  FROM o WINDOW w AS (PARTITION BY dev, det ORDER BY t)
), h AS (SELECT dev, det, hr, count(*) AS nh FROM o GROUP BY ALL),
hm AS (SELECT dev, det, median(nh) AS nh_med FROM h GROUP BY ALL)
SELECT g.*, (h.nh <= hm.nh_med) AS low, c.t_g, c.t_y, c.t_red, c.t_next,
       g.t - c.t_g AS since, c.t_y - g.t AS to_y,
       CASE WHEN g.t < c.t_y THEN 'G' WHEN g.t >= c.t_red THEN 'R' ELSE 'Y' END AS state
FROM g JOIN h USING (dev, det, hr) JOIN hm USING (dev, det)
ASOF JOIN sb_cyc c ON g.dev = c.dev AND g.phase = c.phase AND g.t >= c.t_g
WHERE g.t < c.t_next
"""

SQL_SPEED = """
SELECT dev, det, count(*) AS n_iso, avg((dur <= 0.15)::INT) AS pulse,
       count(*) FILTER (low) AS n_low,
       median(dur) FILTER (dur > 0.15) AS dur50_all,
       median(dur) FILTER (dur > 0.15 AND low) AS dur50_low
FROM sb_onc
WHERE to_y > 0 AND since >= 15 AND gap_prev >= 3 AND gap_next >= 3
GROUP BY ALL
"""

SQL_MATCH = """
WITH a AS (
  SELECT o.dev, o.det AS da, o.t, o.dur AS dur_a, o.low
  FROM sb_onc o SEMI JOIN (SELECT dev, det FROM sb_ad WHERE is_t) tg ON o.dev = tg.dev AND o.det = tg.det
  WHERE to_y >= 10 AND since >= 10 AND gap_prev >= 4 AND gap_next >= 2
), ap AS (SELECT a.*, p.db, a.t + 0.2 AS t2 FROM a JOIN sb_ptab p ON a.dev = p.dev AND a.da = p.da),
b AS (SELECT dev, det, t FROM sb_onc)
SELECT ap.dev, ap.da, ap.db, ap.low, ap.dur_a, b.t - ap.t AS dt
FROM ap ASOF JOIN b ON ap.dev = b.dev AND ap.db = b.det AND b.t >= ap.t2
WHERE b.t - ap.t <= 15
"""

SQL_BEH = """
WITH t AS (SELECT o.* FROM sb_onc o SEMI JOIN (SELECT dev, det FROM sb_ad WHERE is_t) x
           ON o.dev = x.dev AND o.det = x.det),
rel AS (
  SELECT dev, det, t_next, min(t + dur - t_next) AS rel
  FROM t WHERE state <> 'G' AND t + dur > t_next AND t + dur - t_next < 60 GROUP BY ALL
), fst AS (
  SELECT dev, det, t_g, min(since) AS first_lag FROM t WHERE state = 'G' GROUP BY ALL
), hold AS (
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
FROM sb_cyc c SEMI JOIN (SELECT DISTINCT dev, phase FROM sb_ad) a USING (dev, phase) GROUP BY ALL
"""

SQL_CORR = """
WITH a AS (
  SELECT o.dev, o.det AS da, p.db, o.t, o.quiet, floor(o.t / 20)::INT AS bk
  FROM sb_onw o JOIN sb_prs p ON o.dev = p.dev AND o.det = p.da
), b AS (
  SELECT dev, det AS db, t,
         unnest([floor(t / 20)::INT - 1, floor(t / 20)::INT, floor(t / 20)::INT + 1]) AS bk
  FROM sb_onw
)
SELECT a.dev, a.da, a.db, floor((b.t - a.t) / 0.5)::INT AS lb, a.quiet, count(*) AS n
FROM a JOIN b ON a.dev = b.dev AND a.db = b.db AND a.bk = b.bk
WHERE abs(b.t - a.t) < 15
GROUP BY ALL
"""

SQL_MIN = """
WITH cnt AS (SELECT dev, det, floor(t / 60)::INT AS m, count(*) AS n FROM sb_onw GROUP BY ALL),
grid AS (SELECT d.dev, d.det, m.m FROM (SELECT DISTINCT dev, det FROM sb_onw) d
         CROSS JOIN (SELECT unnest(range(0, {nm})) AS m) m),
g AS (SELECT g.dev, g.det, g.m, g.m // 15 AS blk, coalesce(c.n, 0)::DOUBLE AS n
      FROM grid g LEFT JOIN cnt c ON g.dev = c.dev AND g.det = c.det AND g.m = c.m),
r AS (SELECT *, n - avg(n) OVER (PARTITION BY dev, det, blk) AS res FROM g),
tot AS (SELECT dev, blk, sum(n) AS tn FROM g GROUP BY ALL),
lvl AS (SELECT dev, blk, CASE WHEN rk >= 0.67 THEN 2 WHEN rk <= 0.33 THEN 0 ELSE 1 END AS lv
        FROM (SELECT dev, blk, percent_rank() OVER (PARTITION BY dev ORDER BY tn) AS rk
              FROM tot)),
r2 AS (SELECT r.*, lvl.lv FROM r JOIN lvl ON r.dev = lvl.dev AND r.blk = lvl.blk),
blk AS (SELECT dev, det, blk, any_value(lv) AS lv, sum(n) AS n FROM r2 GROUP BY ALL),
m AS (
  SELECT a.dev, a.det AS da, b.det AS db,
         corr(a.res, b.res) AS hp_all,
         corr(a.res, b.res) FILTER (a.lv = 0) AS hp_off,
         corr(a.res, b.res) FILTER (a.lv = 2) AS hp_peak
  FROM r2 a JOIN r2 b ON a.dev = b.dev AND a.m = b.m
  JOIN sb_prs p ON a.dev = p.dev AND a.det = p.da AND b.det = p.db
  GROUP BY ALL
), q AS (
  SELECT a.dev, a.det AS da, b.det AS db,
         corr(a.n, b.n) AS c_all,
         corr(a.n, b.n) FILTER (a.lv = 0) AS c_off,
         corr(a.n, b.n) FILTER (a.lv = 2) AS c_peak,
         sum(b.n) FILTER (a.lv = 2) / nullif(sum(a.n) FILTER (a.lv = 2), 0) AS r_peak,
         sum(b.n) FILTER (a.lv = 0) / nullif(sum(a.n) FILTER (a.lv = 0), 0) AS r_off
  FROM blk a JOIN blk b ON a.dev = b.dev AND a.blk = b.blk
  JOIN sb_prs p ON a.dev = p.dev AND a.det = p.da AND b.det = p.db
  GROUP BY ALL
)
SELECT * FROM m FULL JOIN q USING (dev, da, db)
"""


# --------------------------------------------------------------------------- helpers
def robust_tau(dt: np.ndarray, dur: np.ndarray | None = None) -> dict:
    """Mode-anchored median of advance -> stop-bar delays (0.2-15 s) (note 39)."""
    ok = (dt > 0.2) & (dt <= 15)
    dt = dt[ok]
    dur = None if dur is None else dur[ok]
    r = {"n_match": len(dt), "tau": np.nan, "n_in": 0, "pk_strength": np.nan,
         "tau_med_raw": float(np.median(dt)) if len(dt) else np.nan}
    if len(dt) < 5:
        return r
    h, e = np.histogram(dt, bins=np.arange(0.2, 15.2, 0.2))
    hs = np.convolve(h, np.ones(5) / 5, "same")
    k = hs.argmax()
    mode = (e[k] + e[k + 1]) / 2
    lo, hi = 0.7 * mode, 1.4 * mode
    inw = dt[(dt >= lo) & (dt <= hi)]
    bg = len(dt) / 14.8 * (hi - lo)
    r.update(n_in=len(inw), pk_strength=len(inw) / max(bg, 0.5), mode=mode)
    if len(inw) >= 5 and r["pk_strength"] >= 1.5:
        r["tau"] = float(np.median(inw))
        if dur is not None:
            w = (dt >= lo) & (dt <= hi) & (dur > 0.15)
            v = (L_VEH + 6.0) / dur[w]
            w2 = (v >= VMIN) & (v <= VMAX)
            if w2.sum() >= 5:
                r["d_veh"] = float(np.median(v[w2] * dt[w][w2]))
                r["v_veh"] = float(np.median(v[w2]))
    return r


def cues(corr: pd.DataFrame, nq: pd.DataFrame) -> pd.DataFrame:
    """Correlogram -> per-pair cues (research/code/trackA/lr2_pairs.cues)."""
    lags = np.arange(-30, 30)
    mid = lags * 0.5 + 0.25
    out = []
    for quiet_tag, sub in (("all", corr), ("q", corr[corr.quiet])):
        H = (sub.groupby(["dev", "da", "db", "lb"]).n.sum().unstack("lb")
             .reindex(columns=lags, fill_value=0).fillna(0))
        M = H.to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            base = M[:, np.abs(mid) >= 10].mean(1).clip(min=0.5)
            z = (np.abs(mid) < 0.5)
            fwd = (mid >= 1.5) & (mid <= 8.5)
            bwd = (mid <= -1.5) & (mid >= -8.5)
            near = (np.abs(mid) >= 1.0) & (np.abs(mid) < 3.0)
            d = pd.DataFrame(index=H.index)
            d[f"z0_ratio_{quiet_tag}"] = np.log(M[:, z].mean(1).clip(min=0.1) / base)
            d[f"z0_sharp_{quiet_tag}"] = np.log(M[:, z].mean(1).clip(min=0.1)
                                                / M[:, near].mean(1).clip(min=0.5))
            d[f"z0_exc_{quiet_tag}"] = (M[:, z] - base[:, None]).sum(1)
            pf, pb = M[:, fwd].max(1), M[:, bwd].max(1)
            d[f"lead_ratio_{quiet_tag}"] = np.log(np.maximum(pf, pb) / base)
            band = np.where(pf >= pb, 0, 1)
            Mf, Mb = M[:, fwd], M[:, bwd][:, ::-1]
            MM = np.where(band[:, None] == 0, Mf, Mb)
            run = MM[:, :-1] + MM[:, 1:]
            d[f"lead_sharp_{quiet_tag}"] = np.log(run.max(1).clip(min=0.5)
                                                  / (2 * np.median(MM, 1)).clip(min=0.5))
            d[f"lead_exc_{quiet_tag}"] = (run.max(1) - 2 * base)
            d[f"lead_lag_{quiet_tag}"] = np.where(band == 0, 1, -1) * (
                1.5 + 0.5 * run.argmax(1) + 0.5)
            d[f"lead_asym_{quiet_tag}"] = np.log((M[:, fwd].sum(1) + 1) / (M[:, bwd].sum(1) + 1))
        out.append(d)
    d = pd.concat(out, axis=1).reset_index()
    d = d.merge(nq.rename(columns={"det": "da", "n": "n_a", "nq": "nq_a"}), on=["dev", "da"])
    d = d.merge(nq.rename(columns={"det": "db", "n": "n_b", "nq": "nq_b"}), on=["dev", "db"])
    for t, na, nb in (("all", "n_a", "n_b"), ("q", "nq_a", "nq_b")):
        mn = np.minimum(d[na], d[nb]).clip(lower=1)
        d[f"z0_exc_{t}"] = d[f"z0_exc_{t}"] / mn
        d[f"lead_exc_{t}"] = d[f"lead_exc_{t}"] / mn
    d["bal"] = np.minimum(d.n_a, d.n_b) / np.maximum(d.n_a, d.n_b).clip(lower=1)
    return d


def _speed(row: dict | None, role: str) -> float:
    if row is None or pd.isna(row.get("dur")) or row["pulse"] >= 0.5:
        return np.nan
    v = (L_VEH + ZONE.get(role, 6.0)) / row["dur"]
    return v if VMIN <= v <= VMAX else np.nan


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


def pair_probs(con, cand, act, nd, hours, model: HGBNumpy) -> pd.DataFrame:
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
    con.register("sb_prs_df", prs)
    con.execute("CREATE OR REPLACE TEMP TABLE sb_prs AS SELECT * FROM sb_prs_df")
    con.unregister("sb_prs_df")
    con.execute("""CREATE OR REPLACE TEMP TABLE sb_onw AS
        SELECT i.dev, i.det, i.t, i.quiet FROM sb_iv i
        SEMI JOIN (SELECT dev, da AS det FROM sb_prs UNION SELECT dev, db FROM sb_prs) p
          ON i.dev = p.dev AND i.det = p.det""")
    nq = con.sql("SELECT dev, det, count(*) AS n, count(*) FILTER (quiet) AS nq "
                 "FROM sb_onw GROUP BY ALL").df()
    corr = con.sql(SQL_CORR).df()
    mm = con.sql(SQL_MIN.format(nm=int(hours * 60))).df()
    if not len(corr):
        return out
    P = cues(corr, nq).merge(mm, on=["dev", "da", "db"], how="left")
    P = P.replace([np.inf, -np.inf], np.nan)
    ph_ = c[["dev", "x", "y", "phase"]].drop_duplicates(["dev", "x", "y"])
    ph_ = pd.concat([ph_.rename(columns={"x": "da", "y": "db"}), ph_.rename(columns={"x": "db", "y": "da"})])
    P = P.merge(ph_, on=["dev", "da", "db"])
    P = P.merge(nd.reset_index(), on=["dev", "phase"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        P["c_div"] = P.c_off - P.c_peak
        P["r_shift"] = -np.abs(np.log(P.r_peak.clip(lower=1e-3) / P.r_off.clip(lower=1e-3)))
        P["lead_asym_abs"] = P.lead_asym_all.abs()
        P["lph_a"] = np.log1p(P.n_a / hours)
        P["lph_b"] = np.log1p(P.n_b / hours)
        P["log_ratio"] = np.abs(np.log1p(P.n_a) - np.log1p(P.n_b))
    P["p_same"] = model.predict_proba(P[PAIR_COLS].to_numpy(float))
    P = P[["dev", "da", "db", "p_same"]].assign(x=lambda q: np.minimum(q.da, q.db), y=lambda q: np.maximum(q.da, q.db))
    P = P.groupby(["dev", "x", "y"], as_index=False).p_same.mean().rename(columns={"x": "da", "y": "db"})
    return pd.concat([P, P.rename(columns={"da": "db", "db": "da"})], ignore_index=True)


def features(con, dets: pd.DataFrame, hours: float, pair_model: HGBNumpy) -> pd.DataFrame:
    """con holds sb_iv / sb_cyc (seconds relative to the window start).  dets: dev, det, phase, fn
    (predicted), n_on, p_mid.  One row per predicted Advance / Mid (sb5_setback.features)."""
    d = dets.copy()
    d["det"] = d.det.astype("int16")
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
    con.register("sb_ad_df", ad)
    con.execute("CREATE OR REPLACE TEMP TABLE sb_ad AS SELECT * FROM sb_ad_df")
    con.unregister("sb_ad_df")
    con.execute(SQL_ONC)
    spd = con.sql(SQL_SPEED).df()
    spd["dur"] = np.where(spd.n_low >= 20, spd.dur50_low, spd.dur50_all)
    spd = spd.set_index(["dev", "det"])
    beh = con.sql(SQL_BEH).df()
    ph = con.sql(SQL_PHASE).df()
    T = d[d.is_t][["dev", "det", "phase"]]
    S = d[d.is_s][["dev", "det", "phase", "fn"]]
    cand = T.rename(columns={"det": "da"}).merge(S.rename(columns={"det": "db"}),
                                                  on=["dev", "phase"])
    cand = cand[cand.da != cand.db]
    con.register("sb_pt_df", cand[["dev", "da", "db"]])
    con.execute("CREATE OR REPLACE TEMP TABLE sb_ptab AS SELECT * FROM sb_pt_df")
    con.unregister("sb_pt_df")
    M = con.sql(SQL_MATCH).df()
    rows = []
    for (dev, da, db), g in M.groupby(["dev", "da", "db"]):
        lw = g.low.to_numpy()
        use = lw if lw.sum() >= 30 else np.ones(len(g), bool)
        r = robust_tau(g.dt.to_numpy()[use], g.dur_a.to_numpy()[use])
        r.update(dev=dev, da=da, db=db)
        rows.append(r)
    TAU = pd.DataFrame(rows, columns=["dev", "da", "db", "n_match", "tau", "n_in",
                                      "pk_strength", "tau_med_raw", "mode", "d_veh", "v_veh"])
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
    F["hours"] = hours
    return F


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


def confidence(p10, p50, p90, cuts) -> pd.Series:
    w = np.log(np.maximum(np.asarray(p90, float), 15.0) / np.maximum(np.asarray(p10, float), 15.0))
    c = np.select([w <= cuts[0], w <= cuts[1]], ["high", "medium"], "low")
    return pd.Series(np.where(np.isnan(np.asarray(p50, float)), None, c), index=p50.index)


# --------------------------------------------------------------------------- entry point
def setback(con, w0: float, w1: float, dets: pd.DataFrame, model: SetbackModel) -> pd.DataFrame:
    """`con` = predict's open connection (onev_all, cyc_all, devmap built).  `dets`: DeviceId, Detector,
    phase (predicted), function (predicted, 7 classes), p_mid.  -> DeviceId, Detector, distance_ft,
    setback_confidence (+ internal p10 / p50 / p90)."""
    cols = ["DeviceId", "Detector", "distance_ft", "setback_confidence"]
    p = dets.rename(columns={"DeviceId": "dev", "Detector": "det", "function": "fn"}).copy()
    p = p[p.phase.notna() & p.fn.notna()].copy()
    if not len(p):
        return pd.DataFrame(columns=cols)
    p["phase"] = p.phase.astype(int)
    p["det"] = p.det.astype(int)
    hours = (w1 - w0) / 3600.0
    con.execute(f"""CREATE OR REPLACE TEMP TABLE sb_iv AS
        SELECT m.DeviceId AS dev, o.det, o.t_on - {w0} AS t, o.dur::DOUBLE AS dur,
               (floor((o.t_on % 86400) / 3600) >= 22 OR floor((o.t_on % 86400) / 3600) < 6) AS quiet
        FROM onev_all o JOIN devmap m USING (dev)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE sb_cyc AS
        SELECT m.DeviceId AS dev, c.p::INT AS phase, c.gs - {w0} AS t_g, c.ge - {w0} AS t_y,
               c.rs - {w0} AS t_red, c.ng - {w0} AS t_next
        FROM cyc_all c JOIN devmap m USING (dev) WHERE c.ng IS NOT NULL""")
    n = con.sql("SELECT dev, det::INT AS det, count(*) AS n_on FROM sb_iv GROUP BY ALL").df()
    p = p.merge(n, on=["dev", "det"], how="left")
    p["n_on"] = p.n_on.fillna(0)
    try:
        if p.fn.isin(TARGET).any():
            F = features(con, p[["dev", "det", "phase", "fn", "n_on", "p_mid"]], hours, model.pair)
        else:
            F = pd.DataFrame()
    finally:
        for t in ("sb_iv", "sb_cyc", "sb_ad", "sb_onc", "sb_ptab", "sb_prs", "sb_onw"):
            con.execute(f"DROP TABLE IF EXISTS {t}")
    if len(F):
        E = model.apply(F, hours)
        F = pd.concat([F[["dev", "det"]].astype({"det": int}), E[["p10", "p50", "p90"]]], axis=1)
    else:
        F = pd.DataFrame(columns=["dev", "det", "p10", "p50", "p90"])
    out = p[["dev", "det", "fn"]].merge(F, on=["dev", "det"], how="left")
    sb = out.fn.isin(STOPBAR)
    out.loc[sb, ["p10", "p50", "p90"]] = 0.0
    out["distance_ft"] = out.p50.astype(float).round(0)
    out["setback_confidence"] = confidence(out.p10, out.p50, out.p90, model.meta["conf_cuts"])
    out.loc[sb, "setback_confidence"] = "high"
    out = out.rename(columns={"dev": "DeviceId", "det": "Detector"})
    return out[cols + ["p10", "p50", "p90"]]
