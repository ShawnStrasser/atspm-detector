"""Detector-pair actuation lag, a function-model input (one row per (DeviceId, win, Detector, other)).

The signed lag from each ON of `Detector` to the ONs of `other` within +-8 s, as a 0.5-s correlogram: its chance-corrected
peak, the coincidence within 0.5 / 1 s, the share of the other's ONs that come after.  An upstream Advance loop fires
~2-6 s before the stop-bar Presence loop of the same approach; a per-lane Count loop sits at the same longitudinal
position as its neighbour (lag ~ 0) but sees different vehicles; a Yellow_Red loop trails the stop-bar loop.
Phase-anonymous: channel numbers are join keys only (top-K ties are kept, never decided by channel number).
"""
from __future__ import annotations

import pandas as pd

LAG_MAX = 8.0          # seconds either side
LAG_BUCKET = 8.0       # bucket width for the equi-join
MAX_ON_PER_DET = 4000  # sub-sample cap per detector per window (bounds the join)
TOPK_LAG = 12          # neighbours kept per detector


# cross-correlogram: lags binned at LAG_BIN s over [-LAG_MAX, +LAG_MAX], the peak of a
# 3-bin smoothed histogram, chance-corrected (uniform expectation = n_m / NBIN per bin).
LAG_BIN = 0.5
NBIN = int(2 * LAG_MAX / LAG_BIN)

SQL_LAG = f"""
WITH sub AS (
  SELECT dev, det, t_on FROM (
    SELECT dev, det, t_on,
           row_number() OVER (PARTITION BY dev, det ORDER BY t_on) AS rn,
           count(*) OVER (PARTITION BY dev, det) AS n
    FROM onev)
  WHERE n <= {MAX_ON_PER_DET}
     OR rn % ((n / {MAX_ON_PER_DET})::BIGINT + 1) = 0
), na AS (
  SELECT dev, det, count(*) AS n_a FROM sub GROUP BY 1,2
), a AS (
  SELECT dev, det, t_on,
         unnest([(t_on/{LAG_BUCKET})::BIGINT - 1, (t_on/{LAG_BUCKET})::BIGINT,
                 (t_on/{LAG_BUCKET})::BIGINT + 1]) AS bk
  FROM sub
), b AS (
  SELECT dev, det, t_on, (t_on/{LAG_BUCKET})::BIGINT AS bk FROM sub
), h AS (
  SELECT a.dev, a.det, b.det AS oth,
         floor((b.t_on - a.t_on) / {LAG_BIN})::INT AS lb, count(*) AS c
  FROM a JOIN b ON a.dev = b.dev AND a.bk = b.bk AND a.det <> b.det
  WHERE abs(b.t_on - a.t_on) <= {LAG_MAX}
  GROUP BY 1,2,3,4
), h3 AS (
  SELECT *, sum(c) OVER (PARTITION BY dev, det, oth ORDER BY lb
                         RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS c3
  FROM h
), tot AS (
  SELECT dev, det, oth, sum(c) AS n_m,
         sum(c) FILTER (lb IN (-1, 0))        AS n05,
         sum(c) FILTER (lb BETWEEN -2 AND 1)  AS n1,
         sum(c) FILTER (lb >= 0)              AS n_after
  FROM h GROUP BY 1,2,3
), pk AS (
  SELECT dev, det, oth, lb AS pk_lb, c3 AS pk_c3 FROM h3
  -- `lb` last: a perfectly symmetric correlogram must not pick its sign at random
  QUALIFY row_number() OVER (PARTITION BY dev, det, oth
                             ORDER BY c3 DESC, abs(lb), lb) = 1
)
SELECT t.dev, t.det, t.oth,
       (pk.pk_lb + 0.5) * {LAG_BIN}                                   AS lag_peak,
       (pk.pk_c3 - 3.0 * t.n_m / {NBIN}) / nullif(na.n_a, 0)::DOUBLE  AS lag_peak_excess,
       (pk.pk_c3 - 3.0 * t.n_m / {NBIN}) /
           nullif(sqrt(3.0 * t.n_m / {NBIN}), 0)                      AS lag_peak_z,
       (t.n05 - 2.0 * t.n_m / {NBIN}) / nullif(na.n_a, 0)::DOUBLE     AS coinc_05_excess,
       (t.n1  - 4.0 * t.n_m / {NBIN}) / nullif(na.n_a, 0)::DOUBLE     AS coinc_1_excess,
       t.n05 / nullif(na.n_a, 0)::DOUBLE                              AS coinc_05,
       t.n1  / nullif(na.n_a, 0)::DOUBLE                              AS coinc_1,
       t.n_after / nullif(t.n_m, 0)::DOUBLE                           AS frac_after,
       na.n_a, nb.n_a AS n_b, t.n_m
FROM tot t JOIN pk  ON t.dev = pk.dev AND t.det = pk.det AND t.oth = pk.oth
           JOIN na  ON t.dev = na.dev AND t.det = na.det
           JOIN na nb ON t.dev = nb.dev AND t.oth = nb.det
-- top-K by excess; a tie at the cut keeps every tied neighbour (never decided by channel number, note 76)
QUALIFY rank() OVER (PARTITION BY t.dev, t.det
                     ORDER BY ((pk.pk_c3 - 3.0 * t.n_m / {NBIN}) /
                               nullif(na.n_a, 0)::DOUBLE)::FLOAT DESC) <= {TOPK_LAG}
"""


def build(con, win: str, devmap: pd.DataFrame, sql: str) -> pd.DataFrame:
    df = con.sql(sql).df()
    if not len(df):
        return df
    df = df.merge(devmap, on="dev", how="left").drop(columns=["dev"])
    df["win"] = win
    return df
