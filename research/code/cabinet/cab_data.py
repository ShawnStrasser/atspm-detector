"""Actuation count per detector channel from the hi-res logs (DuckDB aggregates only).

Detector On (82), de-duplicated, Parameter 1..64, per window:
  dec2024  = data/raw/Train_Dec_*_2024.parquet   (3 days)
  staging  = data/staging*/date=*/part_*.parquet (about 2.7 days, 2026-09)
plus the number of distinct clock-hours with at least one actuation and the max 5-minute count.
The full data-quality checks live in dq_*.py; this is only the "is the channel alive" count.

    python cab_data.py 01001 01005 ...   -> %DC_WORK%/cabinet/activity.parquet (rows for these signals replaced)
"""
from __future__ import annotations

import os
import sys

import duckdb
import pandas as pd

from cab_common import CAB, DC_WORK, device_id

WINDOWS = {
    "dec2024": [str(DC_WORK / "data/raw/Train_Dec_*_2024.parquet")],
    "staging": [str(DC_WORK / "data/staging/*/*.parquet"), str(DC_WORK / "data/staging_other/*/*.parquet")],
}


def activity(device_ids: list[str]) -> pd.DataFrame:
    con = duckdb.connect()
    mem, thr = os.environ.get("DC_DUCK_MEM", "10GB"), int(os.environ.get("DC_DUCK_THREADS", "6"))  # shared CPU caps
    con.execute(f"SET memory_limit='{mem}'; SET threads={thr}; SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    ids = ",".join(f"'{d}'" for d in device_ids)
    out = []
    for win, globs in WINDOWS.items():
        src = " UNION ALL ".join(
            f"SELECT DeviceId, Timestamp, CAST(EventId AS INT) EventId, CAST(Parameter AS INT) Parameter "
            f"FROM read_parquet('{g.replace(chr(92), '/')}') WHERE DeviceId IN ({ids}) AND EventId = 82 "
            f"AND Parameter BETWEEN 1 AND 64" for g in globs)
        q = f"""
        WITH ev AS (SELECT DISTINCT * FROM ({src})),
        b AS (SELECT DeviceId, Parameter, time_bucket(INTERVAL 5 MINUTE, Timestamp) t5, count(*) n FROM ev GROUP BY ALL)
        SELECT DeviceId, Parameter AS Detector, '{win}' AS window, sum(n) AS n_on,
               count(DISTINCT date_trunc('hour', t5)) AS active_hours, max(n) AS max_5min
        FROM b GROUP BY ALL"""
        out.append(con.execute(q).df())
        span = con.execute(f"SELECT min(Timestamp), max(Timestamp) FROM ({src})").fetchone()
        out[-1]["window_start"], out[-1]["window_end"] = span
    return pd.concat(out, ignore_index=True)


def update(dns: list[str]) -> pd.DataFrame:
    ids = {device_id(dn): dn for dn in dns}
    a = activity(list(ids))
    a.insert(0, "DeviceName", a.DeviceId.map(ids))
    f = CAB / "activity.parquet"
    if f.exists():
        old = pd.read_parquet(f)
        a = pd.concat([old[~old.DeviceId.isin(ids)], a], ignore_index=True)
    a.to_parquet(f, index=False)
    return a


if __name__ == "__main__":
    a = update(sys.argv[1:])
    print(a[a.DeviceName.isin(sys.argv[1:])].groupby(["DeviceName", "window"]).agg(ch=("Detector", "size"), n=("n_on", "sum")))
