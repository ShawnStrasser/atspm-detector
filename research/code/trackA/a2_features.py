"""A2 -- expert-shaped, phase-anonymous function features.

The engineer's own definitions, turned into numbers:

  pulse signature      Count is usually wired 'pulse' (every ON is one 0.1 s tick);
                       Presence is 'normal' (ON for the whole passage) -> duration
                       histogram shape, not its mean.
  first car of green   a Yellow_Red has a ~5 mph speed filter, so it misses vehicles
                       starting from a stop -> share of green ONs in the first seconds
                       of green, and the lag to the first ON of green.
  spill-back           queue reaches an Advance loop only well into red -> occupancy
                       of red split into thirds, and its late/early ratio.
  saturation           a long zone plateaus during the day, a count zone does not ->
                       occupancy per actuation in the busiest bins vs the quietest.
  arrival randomness   free-running arrivals are Poisson (index ~ 1), a coordinated
                       platoon is not -> index of dispersion of binned counts, split
                       by the controller's own coord / free state.
  lane structure       two lane-by-lane loops and one loop spanning both lanes track
                       each other off-peak and diverge at peak -> binned-count
                       correlation off-peak vs peak, and the change in count ratio.

Everything is a share, a ratio or a correlation, so it is duration invariant; channel
numbers and phase numbers are join keys only.

    python a2_features.py --period dec
    python a2_features.py --period stg
"""
from __future__ import annotations
import argparse
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
WORK = DCW / "trackA"
FRAME = DCW / "function_v4" / "funcframe_v4.parquet"

CACHE = {"dec": DCW / "cache", "stg": DCW / "official" / "stg" / "cache"}
FULLWIN = {"dec": "full72", "stg": "full66"}


def _w(name, start, secs):
    return (name, pd.Timestamp(start), float(secs))


WINDOWS = {
    "dec": [_w("m30_a", "2024-12-02 07:30:00", 1800),
            _w("m30_b", "2024-12-03 12:00:00", 1800),
            _w("m30_c", "2024-12-03 21:30:00", 1800),
            _w("m30_d", "2024-12-04 16:45:00", 1800),
            _w("h1_a", "2024-12-02 17:00:00", 3600),
            _w("h1_b", "2024-12-03 02:00:00", 3600),
            _w("h1_c", "2024-12-04 09:00:00", 3600),
            _w("h3_a", "2024-12-02 06:00:00", 3 * 3600),
            _w("h3_b", "2024-12-03 14:00:00", 3 * 3600),
            _w("h6_a", "2024-12-03 06:00:00", 6 * 3600),
            _w("h6_b", "2024-12-04 12:00:00", 6 * 3600),
            _w("h24_a", "2024-12-02 00:00:00", 24 * 3600),
            _w("h24_b", "2024-12-04 00:00:00", 24 * 3600),
            _w("full72", "2024-12-02 00:00:00", 72 * 3600),
            _w("m5_a", "2024-12-02 07:45:00", 300),
            _w("m5_b", "2024-12-03 12:20:00", 300),
            _w("m5_c", "2024-12-03 22:10:00", 300),
            _w("m5_d", "2024-12-04 17:05:00", 300),
            _w("m10_a", "2024-12-02 08:05:00", 600),
            _w("m10_b", "2024-12-03 13:00:00", 600),
            _w("m10_c", "2024-12-04 02:30:00", 600),
            _w("m10_d", "2024-12-04 17:20:00", 600)],
    "stg": [_w("m30_a", "2026-09-21 07:30:00", 1800),
            _w("m30_b", "2026-09-19 12:00:00", 1800),
            _w("m30_c", "2026-09-19 21:30:00", 1800),
            _w("m30_d", "2026-09-18 17:00:00", 1800),
            _w("h1_a", "2026-09-20 17:00:00", 3600),
            _w("h1_b", "2026-09-20 02:00:00", 3600),
            _w("h1_c", "2026-09-21 09:00:00", 3600),
            _w("h3_a", "2026-09-21 06:00:00", 3 * 3600),
            _w("h3_b", "2026-09-19 14:00:00", 3 * 3600),
            _w("h6_a", "2026-09-20 06:00:00", 6 * 3600),
            _w("h6_b", "2026-09-19 12:00:00", 6 * 3600),
            _w("h24_a", "2026-09-19 00:00:00", 24 * 3600),
            _w("h24_b", "2026-09-20 00:00:00", 24 * 3600),
            _w("full66", "2026-09-18 16:15:00", 66 * 3600),
            _w("m5_a", "2026-09-21 07:45:00", 300),
            _w("m5_b", "2026-09-19 12:20:00", 300),
            _w("m5_c", "2026-09-19 22:10:00", 300),
            _w("m5_d", "2026-09-18 17:05:00", 300),
            _w("m10_a", "2026-09-21 08:05:00", 600),
            _w("m10_b", "2026-09-19 13:00:00", 600),
            _w("m10_c", "2026-09-20 02:30:00", 600),
            _w("m10_d", "2026-09-18 17:20:00", 600)],
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=12")
    con.execute("SET preserve_insertion_order=false")
    (DCW / "tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


# --------------------------------------------------------------------- G1 + G2
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


# ------------------------------------------------------------------------- G3
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


# ------------------------------------------------------------------------- G4
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


def build(period: str, only: str | None = None) -> None:
    t0all = time.time()
    con = connect()
    cache = CACHE[period]
    di = (cache / "det_intervals.parquet").as_posix()
    pc = (cache / "phase_cycles.parquet").as_posix()
    cs = (cache / "coord_state.parquet").as_posix()

    fr = pd.read_parquet(FRAME, columns=["DeviceId", "Detector", "win", "period",
                                         "pred_phase"])
    fr = fr[fr.period == period]
    log(f"[{period}] targets {fr.shape}, {fr.DeviceId.nunique()} signals")

    det_parts, cyc_parts, pair_parts = [], [], []
    for name, t0, secs in WINDOWS[period]:
        if only and name != only:
            continue
        tw = time.time()
        t1 = t0 + pd.Timedelta(seconds=secs)
        e0 = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
        nb = int(min(288, max(12, round(secs / 900.0))))
        bs = secs / nb
        tgt = fr[fr.win == name][["DeviceId", "Detector", "pred_phase"]].rename(
            columns={"DeviceId": "dev", "Detector": "det", "pred_phase": "p"})
        tgt = tgt.astype({"det": "int16", "p": "int16"})
        con.register("tgt_df", tgt)
        con.execute("CREATE OR REPLACE TEMP TABLE tgt AS SELECT * FROM tgt_df")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE onw AS
            SELECT DeviceId AS dev, Detector::SMALLINT AS det,
                   epoch_ms(t_on)/1000.0 AS t_on, epoch_ms(t_off)/1000.0 AS t_off,
                   dur::DOUBLE AS dur
            FROM read_parquet('{di}')
            WHERE t_on >= TIMESTAMP '{t0}' AND t_on < TIMESTAMP '{t1}'
              AND Detector BETWEEN 1 AND 64 AND dur IS NOT NULL""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cycw AS
            SELECT DeviceId AS dev, Phase::SMALLINT AS p, cyc::INT AS cyc,
                   epoch_ms(green_start)/1000.0 AS gs,
                   epoch_ms(coalesce(yellow_start, red_start, next_green))/1000.0 AS ge,
                   epoch_ms(next_green)/1000.0 AS ng
            FROM read_parquet('{pc}')
            WHERE Phase BETWEEN 1 AND 16 AND next_green IS NOT NULL
              AND green_start >= TIMESTAMP '{t0}' AND green_start < TIMESTAMP '{t1}'""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE coordw AS
            SELECT DeviceId AS dev, epoch_ms(t_start)/1000.0 AS t0,
                   coalesce(is_coord, false) AS is_coord
            FROM read_parquet('{cs}')""")

        d = con.sql(SQL_DET.format(t0=e0, bs=bs, nb=nb)).df()
        c = con.sql(SQL_CYC).df()
        p = con.sql(SQL_PAIR.format(t0=e0, bs=bs, nb=nb)).df()
        for df, parts in ((d, det_parts), (c, cyc_parts), (p, pair_parts)):
            df["win"] = name
            parts.append(df)
        log(f"[{period}] {name}: det {d.shape} cyc {c.shape} pair {p.shape} "
            f"({time.time()-tw:.0f}s, nb={nb}, bs={bs:.0f}s)")

    out = {}
    for tag, parts in (("det", det_parts), ("cyc", cyc_parts), ("pair", pair_parts)):
        df = pd.concat(parts, ignore_index=True)
        df = df.rename(columns={"dev": "DeviceId", "det": "Detector"})
        df["period"] = period
        df["Detector"] = df.Detector.astype("int16")
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        suffix = f"_{only}" if only else ""
        dest = WORK / f"feat_expert_{tag}_{period}{suffix}.parquet"
        df.to_parquet(dest, index=False)
        out[tag] = df.shape
        log(f"wrote {dest} {df.shape}")
    log(f"[{period}] done in {time.time()-t0all:.0f}s  {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=["dec", "stg"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    build(a.period, a.only)
