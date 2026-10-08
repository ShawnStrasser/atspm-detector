"""A3 (part 1) -- the raw detector-pair table the lane grouper needs.

Same binned-count correlations as `a2_features.py`, but kept per PAIR instead of
aggregated to the detector: (DeviceId, win, Detector, other) -> correlation of the
binned counts over all bins / the quiet bins / the busy bins, and how the pair's
count ratio moves between them.  Two loops in the same lane track each other at every
demand level; a loop that spans two lanes tracks its neighbours off-peak and falls
behind at peak (two side-by-side vehicles = 2 actuations on the pair, 1 on the
spanning loop).
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

import a2_features as A2

WORK = A2.WORK

SQL_PAIR_RAW = """
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
), tot AS (SELECT dev, bk, sum(n) AS tn FROM bc GROUP BY 1,2)
, lvl AS (
  SELECT dev, bk, CASE WHEN rk >= 0.67 THEN 2 WHEN rk <= 0.33 THEN 0 ELSE 1 END AS lv
  FROM (SELECT dev, bk, percent_rank() OVER (PARTITION BY dev ORDER BY tn) AS rk FROM tot)
), b2 AS (SELECT bc.*, lvl.lv FROM bc JOIN lvl ON bc.dev = lvl.dev AND bc.bk = lvl.bk)
SELECT a.dev, a.det, b.det AS oth,
  corr(a.n, b.n)                       AS c_all,
  corr(a.n, b.n) FILTER (a.lv = 0)     AS c_off,
  corr(a.n, b.n) FILTER (a.lv = 2)     AS c_peak,
  sum(a.n)                             AS n_a,
  sum(b.n)                             AS n_b,
  sum(b.n) FILTER (a.lv = 2) / nullif(sum(a.n) FILTER (a.lv = 2), 0) AS r_peak,
  sum(b.n) FILTER (a.lv = 0) / nullif(sum(a.n) FILTER (a.lv = 0), 0) AS r_off
FROM b2 a JOIN b2 b ON a.dev = b.dev AND a.bk = b.bk AND a.det < b.det
GROUP BY 1,2,3
"""


def build(period: str) -> None:
    t0all = time.time()
    con = A2.connect()
    di = (A2.CACHE[period] / "det_intervals.parquet").as_posix()
    parts = []
    for name, t0, secs in A2.WINDOWS[period]:
        tw = time.time()
        t1 = t0 + pd.Timedelta(seconds=float(secs))
        e0 = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
        # capped at 96 bins: the lane graph only needs the shape of the co-variation,
        # and the 288-bin version of the same join costs 3x for no visible change
        nb = int(min(96, max(12, round(secs / 900.0))))
        bs = secs / nb
        con.execute(f"""CREATE OR REPLACE TEMP TABLE onw AS
            SELECT DeviceId AS dev, Detector::SMALLINT AS det,
                   epoch_ms(t_on)/1000.0 AS t_on
            FROM read_parquet('{di}')
            WHERE t_on >= TIMESTAMP '{t0}' AND t_on < TIMESTAMP '{t1}'
              AND Detector BETWEEN 1 AND 64""")
        d = con.sql(SQL_PAIR_RAW.format(t0=e0, bs=bs, nb=nb)).df()
        d["win"] = name
        parts.append(d)
        A2.log(f"[{period}] {name}: {d.shape} ({time.time()-tw:.0f}s)")
    df = pd.concat(parts, ignore_index=True).rename(
        columns={"dev": "DeviceId", "det": "Detector", "oth": "other"})
    df["period"] = period
    df["Detector"] = df.Detector.astype("int16")
    df["other"] = df.other.astype("int16")
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    dest = WORK / f"pairs_raw_{period}.parquet"
    df.to_parquet(dest, index=False)
    A2.log(f"wrote {dest} {df.shape} in {time.time()-t0all:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=["dec", "stg"])
    build(ap.parse_args().period)
