"""Expert-shaped, phase-anonymous function features (the 54 `px_*` columns).

The traffic engineer's own definitions of each detector function, turned into numbers:

  pulse signature      Count is usually wired 'pulse' (every ON is one 0.1 s tick);
                       Presence is 'normal' (ON for the whole passage) -> the shape of the
                       ON-duration histogram, not its mean.
  first car of green   a Yellow_Red has a ~5 mph speed filter, so it misses vehicles
                       starting from a stop -> share of green ONs in the first seconds of
                       green, and the lag to the first ON of green.
  spill-back           the queue reaches an Advance loop only well into red -> occupancy of
                       red split into thirds, and its late / early ratio.
  saturation           a long zone plateaus during the day, a count zone does not ->
                       occupancy per actuation in the busiest time bins vs the quietest.
  arrival randomness   free-running arrivals are Poisson (index ~ 1), a coordinated platoon
                       is not -> index of dispersion of binned counts, split by the
                       controller's own coord / free state (event 131).
  lane structure       lane-by-lane loops and a loop spanning both lanes track each other
                       off-peak and diverge at peak -> binned-count correlation off-peak vs
                       peak, and the change in their count ratio.

Everything is a share, a ratio or a correlation, so it does not depend on the sample
length; channel and phase numbers are join keys only (the "sibling" detectors are the ones
the phase model put on the same predicted phase).  The SQL is the research definition
verbatim; only the input tables are built here from the tables `predict.build_chunk_tables` creates (`devmap`,
`onev_all`, `coordiv`) and the shared 1 / 8 / 10 / 11 cycle table `cyc5` (streams.ensure_cyc5).
Research builder: `research/code/trackA/a2_features.py` (note 14, A2); parity with the training
frame verified by `research/code/trackA/v3_candidate_verify.py` (note 32).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .streams import ensure_cyc5

# --------------------------------------------------------------- detector level
SQL_DET = """
WITH b AS (
  SELECT dev, det, t_on, t_off, dur,
         least(cast((t_on - {t0}) / {bs} AS INT), {nb} - 1) AS bk
  FROM onw
), agg AS (
  SELECT dev, det, count(*) AS n,
    avg(CASE WHEN dur <= 0.25 THEN 1 ELSE 0 END)                 AS px_pulse_frac,
    avg(CASE WHEN dur <= 0.35 THEN 1 ELSE 0 END)                 AS px_short_frac,
    avg(CASE WHEN dur > 0.35 AND dur <= 1.5 THEN 1 ELSE 0 END)   AS px_mid_frac,
    avg(CASE WHEN dur > 1.5 THEN 1 ELSE 0 END)                   AS px_long_frac,
    avg(CASE WHEN dur > 10 THEN 1 ELSE 0 END)                    AS px_vlong_frac,
    quantile_cont(dur, 0.10) AS q10, quantile_cont(dur, 0.25) AS q25,
    quantile_cont(dur, 0.50) AS q50, quantile_cont(dur, 0.75) AS q75,
    quantile_cont(dur, 0.95) AS q95,
    stddev_samp(ln(greatest(dur, 0.05)))                         AS px_dur_logsd,
    max(dur) AS dmax
  FROM b GROUP BY 1,2
), pb AS (
  SELECT dev, det, bk, count(*) AS n, sum(least(dur, {bs})) AS occ FROM b GROUP BY 1,2,3
), pstat AS (
  SELECT dev, det,
    var_samp(n) / nullif(avg(n), 0)                              AS px_disp_index,
    stddev_samp(n) / nullif(avg(n), 0)                           AS px_cnt_cv,
    quantile_cont(n, 0.9) / nullif(quantile_cont(n, 0.5), 0)     AS px_peak_over_med,
    max(occ) / {bs}                                              AS px_occ_bin_max,
    quantile_cont(occ, 0.9) / {bs}                               AS px_occ_bin_p90,
    count(*) AS nbin_seen
  FROM pb GROUP BY 1,2
), sat AS (
  -- occupancy per actuation in the busiest bins vs the quietest: a long zone
  -- saturates (ratio << 1), a stop-bar count zone does not (ratio ~ 1)
  SELECT dev, det,
    sum(occ) FILTER (rk >= 0.67) / nullif(sum(n) FILTER (rk >= 0.67), 0) AS occ_per_on_hi,
    sum(occ) FILTER (rk <= 0.33) / nullif(sum(n) FILTER (rk <= 0.33), 0) AS occ_per_on_lo
  FROM (SELECT *, percent_rank() OVER (PARTITION BY dev, det ORDER BY n) AS rk FROM pb)
  GROUP BY 1,2
), cd AS (
  SELECT b.dev, b.det, co.is_coord, count(*) AS n
  FROM b ASOF LEFT JOIN coordw co ON b.dev = co.dev AND b.t_on >= co.t0
  GROUP BY 1,2,3,  b.bk
), cdstat AS (
  SELECT dev, det,
    var_samp(n) FILTER (is_coord) / nullif(avg(n) FILTER (is_coord), 0)      AS px_disp_coord,
    var_samp(n) FILTER (NOT is_coord) / nullif(avg(n) FILTER (NOT is_coord), 0) AS px_disp_free
  FROM cd GROUP BY 1,2
)
SELECT a.dev, a.det, a.n AS px_n_on,
  a.px_pulse_frac, a.px_short_frac, a.px_mid_frac, a.px_long_frac, a.px_vlong_frac,
  a.q10 AS px_dur_q10, a.q25 AS px_dur_q25, a.q75 AS px_dur_q75, a.q95 AS px_dur_q95,
  (a.q75 - a.q25) / nullif(a.q50, 0)                           AS px_dur_iqr_over_med,
  a.q95 / nullif(a.q10, 0)                                     AS px_dur_q95_over_q10,
  a.px_dur_logsd,
  a.px_short_frac + a.px_long_frac - a.px_mid_frac              AS px_bimodality,
  a.dmax                                                        AS px_dur_max,
  p.px_disp_index, p.px_cnt_cv, p.px_peak_over_med, p.px_occ_bin_max, p.px_occ_bin_p90,
  s.occ_per_on_hi / nullif(s.occ_per_on_lo, 0)                  AS px_saturation,
  s.occ_per_on_hi                                               AS px_occ_per_on_hi,
  c.px_disp_coord, c.px_disp_free,
  c.px_disp_coord / nullif(c.px_disp_free, 0)                   AS px_disp_coord_over_free
FROM agg a
LEFT JOIN pstat p ON a.dev = p.dev AND a.det = p.det
LEFT JOIN sat   s ON a.dev = s.dev AND a.det = s.det
LEFT JOIN cdstat c ON a.dev = c.dev AND a.det = c.det
"""

# ------------------------------------------------- against the predicted phase's cycles
SQL_CYC = """
WITH j AS (
  SELECT o.dev, o.det, o.t_on, o.t_off, o.dur, c.cyc, c.gs, c.ge, c.ng,
         greatest(c.ng - c.ge, 0) AS red_secs
  FROM (SELECT o.* , t.p FROM onw o JOIN tgt t ON o.dev = t.dev AND o.det = t.det) o
  ASOF LEFT JOIN cycw c ON o.dev = c.dev AND o.p = c.p AND o.t_on >= c.gs
), k AS (
  SELECT *, (ge + red_secs/3.0) AS r1, (ge + 2*red_secs/3.0) AS r2,
    greatest(least(t_off, ge + red_secs/3.0) - greatest(t_on, ge), 0) AS ov1,
    greatest(least(t_off, ge + 2*red_secs/3.0) - greatest(t_on, ge + red_secs/3.0), 0) AS ov2,
    greatest(least(t_off, ng) - greatest(t_on, ge + 2*red_secs/3.0), 0) AS ov3,
    greatest(least(t_off, ng) - greatest(t_on, ge), 0) AS ovred,
    (t_on - gs) AS d_gs
  FROM j WHERE cyc IS NOT NULL
), g AS (
  SELECT dev, det, cyc, min(d_gs) FILTER (d_gs >= 0 AND t_on < ge) AS first_g
  FROM k GROUP BY 1,2,3
), gg AS (
  SELECT dev, det,
    median(first_g)                                              AS px_g_first_med,
    avg(CASE WHEN first_g <= 2 THEN 1 ELSE 0 END)                AS px_g_first_le2,
    avg(CASE WHEN first_g >= 6 THEN 1 ELSE 0 END)                AS px_g_first_ge6,
    count(*) FILTER (first_g IS NOT NULL)                        AS n_cyc_hit
  FROM g GROUP BY 1,2
), a AS (
  SELECT dev, det, count(*) AS n,
    sum(ov1) AS s1, sum(ov2) AS s2, sum(ov3) AS s3, sum(ovred) AS sred,
    count(DISTINCT cyc) AS ncyc_seen,
    avg(CASE WHEN t_off >= ng THEN 1 ELSE 0 END)                 AS px_span_to_green,
    count(*) FILTER (d_gs >= 0 AND d_gs < 5 AND t_on < ge)       AS n_g5,
    count(*) FILTER (t_on < ge)                                  AS n_green,
    count(DISTINCT cyc) FILTER (ovred > 0.5 * red_secs)          AS ncyc_hold,
    median(dur) FILTER (t_on >= ge)                              AS px_dur_med_red,
    median(dur) FILTER (t_on < ge)                               AS px_dur_med_green
  FROM k GROUP BY 1,2
), sec AS (
  SELECT dev, p, count(*) AS n_cyc, sum(greatest(ng - ge, 0)) AS red_secs
  FROM cycw GROUP BY 1,2
)
SELECT a.dev, a.det,
  a.s1 / nullif(s.red_secs/3.0, 0)                               AS px_occ_red_t1,
  a.s2 / nullif(s.red_secs/3.0, 0)                               AS px_occ_red_t2,
  a.s3 / nullif(s.red_secs/3.0, 0)                               AS px_occ_red_t3,
  (a.s3 + 1e-3) / (a.s1 + 1e-3)                                  AS px_red_late_over_early,
  a.sred / nullif(s.red_secs, 0)                                 AS px_occ_red_all,
  a.px_span_to_green,
  a.n_g5 / nullif(a.n_green, 0)::DOUBLE                          AS px_frac_on_first5_green,
  a.n_g5 / nullif(s.n_cyc, 0)::DOUBLE                            AS px_on_first5_per_cycle,
  a.ncyc_hold / nullif(s.n_cyc, 0)::DOUBLE                       AS px_hold_through_red,
  a.px_dur_med_red, a.px_dur_med_green,
  a.px_dur_med_red / nullif(a.px_dur_med_green, 0)               AS px_dur_red_over_green,
  gg.px_g_first_med, gg.px_g_first_le2, gg.px_g_first_ge6,
  gg.n_cyc_hit / nullif(s.n_cyc, 0)::DOUBLE                      AS px_cyc_hit_frac
FROM a
LEFT JOIN gg ON a.dev = gg.dev AND a.det = gg.det
LEFT JOIN tgt t ON a.dev = t.dev AND a.det = t.det
LEFT JOIN sec s ON a.dev = s.dev AND t.p = s.p
"""

# ------------------------------------------------------------- detector pairs
SQL_PAIR = """
WITH cnt AS (
  SELECT dev, det, least(cast((t_on - {t0}) / {bs} AS INT), {nb} - 1) AS bk,
         count(*) AS n
  FROM onw GROUP BY 1,2,3
), dets AS (
  SELECT dev, det FROM cnt GROUP BY 1,2 HAVING sum(n) >= 20
), bins AS (SELECT unnest(range(0, {nb})) AS bk)
, grid AS (SELECT d.dev, d.det, b.bk FROM dets d CROSS JOIN bins b)
, bc AS (
  SELECT g.dev, g.det, g.bk, coalesce(c.n, 0)::DOUBLE AS n
  FROM grid g LEFT JOIN cnt c ON g.dev = c.dev AND g.det = c.det AND g.bk = c.bk
), tot AS (
  SELECT dev, bk, sum(n) AS tn FROM bc GROUP BY 1,2
), lvl AS (
  SELECT dev, bk, CASE WHEN rk >= 0.67 THEN 2 WHEN rk <= 0.33 THEN 0 ELSE 1 END AS lv
  FROM (SELECT dev, bk, percent_rank() OVER (PARTITION BY dev ORDER BY tn) AS rk FROM tot)
), b2 AS (SELECT bc.*, lvl.lv FROM bc JOIN lvl ON bc.dev = lvl.dev AND bc.bk = lvl.bk)
, pr AS (
  SELECT a.dev, a.det, b.det AS oth,
    corr(a.n, b.n)                       AS c_all,
    corr(a.n, b.n) FILTER (a.lv = 0)     AS c_off,
    corr(a.n, b.n) FILTER (a.lv = 2)     AS c_peak,
    sum(b.n) FILTER (a.lv = 2) / nullif(sum(a.n) FILTER (a.lv = 2), 0) AS r_peak,
    sum(b.n) FILTER (a.lv = 0) / nullif(sum(a.n) FILTER (a.lv = 0), 0) AS r_off
  FROM b2 a JOIN b2 b ON a.dev = b.dev AND a.bk = b.bk AND a.det <> b.det
  GROUP BY 1,2,3
), pt AS (
  SELECT pr.*, ta.p AS pa, tb.p AS pb,
         coalesce(pr.c_off, pr.c_all) - coalesce(pr.c_peak, pr.c_all) AS c_div,
         pr.r_peak - pr.r_off AS r_div
  FROM pr LEFT JOIN tgt ta ON pr.dev = ta.dev AND pr.det = ta.det
          LEFT JOIN tgt tb ON pr.dev = tb.dev AND pr.oth = tb.det
), pt2 AS (
  -- note 76: the 'at best' columns average over every neighbour tied on c_all (float32 precision) instead of
  -- arg_max's arbitrary pick (row order -> channel / phase numbering)
  SELECT pt.*, c_all::FLOAT AS cf,
         max(c_all::FLOAT) OVER (PARTITION BY dev, det) AS mx_any,
         max(CASE WHEN pa = pb THEN c_all::FLOAT END) OVER (PARTITION BY dev, det) AS mx_sib
  FROM pt
)
SELECT dev, det,
  max(c_all)                                    AS px_corr_any_max,
  max(c_all) FILTER (pa = pb)                   AS px_corr_sib_max,
  max(c_off) FILTER (pa = pb)                   AS px_corroff_sib_max,
  count(*) FILTER (pa = pb AND c_all >= 0.8)    AS px_n_sib_corr08,
  count(*) FILTER (c_all >= 0.8)                AS px_n_any_corr08,
  avg(c_div) FILTER (pa = pb AND cf = mx_sib)   AS px_corrdiv_at_best_sib,
  avg(r_div) FILTER (pa = pb AND cf = mx_sib)   AS px_ratiodiv_at_best_sib,
  avg(r_peak) FILTER (pa = pb AND cf = mx_sib)  AS px_rpeak_at_best_sib,
  avg(r_off) FILTER (pa = pb AND cf = mx_sib)   AS px_roff_at_best_sib,
  avg(c_div) FILTER (cf = mx_any)               AS px_corrdiv_at_best_any,
  avg(r_div) FILTER (cf = mx_any)               AS px_ratiodiv_at_best_any,
  max(c_div) FILTER (pa = pb AND c_all >= 0.5)  AS px_corrdiv_max_sib,
  median(c_all) FILTER (pa = pb)                AS px_corr_sib_med,
  count(*) FILTER (pa = pb)                     AS px_n_sib_pairs
FROM pt2 GROUP BY 1,2
"""


def bins(secs: float) -> tuple[int, float]:
    """Number and width of the count bins: ~15 min each, 12 .. 288 bins."""
    nb = int(min(288, max(12, round(secs / 900.0))))
    return nb, secs / nb


def prepare_tables(con, w0: float, w1: float, tgt: pd.DataFrame) -> None:
    """The four input tables of the SQL above, from the package's own tables.

    onw    detector ON intervals with the ON inside [w0, w1), duration in double
           precision (whole milliseconds, exactly as the research cache stored it)
    cycw   color cycles of every phase: begin green (1), begin yellow (8), begin red
           clearance (10); green ends at the first of yellow / red clearance / next
           green; only cycles that start inside the window and have a next green
    coordw the coordination state changes (event 131)
    tgt    each detector's predicted phase (DeviceId, Detector, pred_phase)
    """
    con.execute(f"""CREATE OR REPLACE TEMP TABLE onw AS
        SELECT dev, det, t_on, t_off, round((t_off - t_on) * 1000.0) / 1000.0 AS dur
        FROM onev_all
        WHERE t_on >= {w0} AND t_on < {w1} AND det BETWEEN 1 AND 64 AND t_off IS NOT NULL""")
    ensure_cyc5(con)            # the 1 / 8 / 10 / 11 cycle table, shared with the network input (streams.py)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cycw AS
        SELECT n.dev, n.p, n.cyc::INT AS cyc,
               epoch_ms(n.green_start)/1000.0 AS gs,
               epoch_ms(coalesce(n.yellow_start, n.red_start, n.next_green))/1000.0 AS ge,
               epoch_ms(n.next_green)/1000.0 AS ng
        FROM cyc5 n
        WHERE n.next_green IS NOT NULL
          AND epoch_ms(n.green_start)/1000.0 >= {w0}
          AND epoch_ms(n.green_start)/1000.0 < {w1}""")
    con.execute("""CREATE OR REPLACE TEMP TABLE coordw AS
        SELECT dev, t0, coalesce(is_coord, false) AS is_coord FROM coordiv""")
    t = tgt[["DeviceId", "Detector", "pred_phase"]].copy()
    t["DeviceId"] = t.DeviceId.astype(str)
    t["Detector"] = t.Detector.astype("int16")
    t["pred_phase"] = t.pred_phase.astype("int16")
    con.register("tgt_df", t)
    con.execute("""CREATE OR REPLACE TEMP TABLE tgt AS
        SELECT m.dev, t.Detector AS det, t.pred_phase AS p
        FROM tgt_df t JOIN devmap m USING (DeviceId)""")
    con.unregister("tgt_df")


def compute(con, t0: float, secs: float) -> pd.DataFrame:
    """Run the three blocks over the prepared tables -> one row per (dev, det)."""
    nb, bs = bins(secs)
    d = con.sql(SQL_DET.format(t0=t0, bs=bs, nb=nb)).df()
    c = con.sql(SQL_CYC).df()
    p = con.sql(SQL_PAIR.format(t0=t0, bs=bs, nb=nb)).df()
    out = d.merge(c, on=["dev", "det"], how="outer").merge(p, on=["dev", "det"], how="outer")
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:          # the training tables were stored in float32
        if out[col].dtype == np.float64:
            out[col] = out[col].astype(np.float32)
    return out


def build(con, w0: float, w1: float, tgt: pd.DataFrame, devmap: pd.DataFrame) -> pd.DataFrame:
    """-> DeviceId, Detector, px_* for every detector of the sample."""
    prepare_tables(con, w0, w1, tgt)
    out = compute(con, w0, max(w1 - w0, 1.0))
    out = out.merge(devmap[["dev", "DeviceId"]], on="dev", how="left").rename(columns={"det": "Detector"})
    out = out.drop(columns=["dev", "px_n_on"])
    for tname in ("onw", "cycw", "coordw", "tgt"):
        con.execute(f"DROP TABLE IF EXISTS {tname}")
    return out
