"""Note 119 (LOCKED EXAM, user-authorised 2026-10-07): extract the 115 locked_v2 signals' Sept-2026 staging events from
the raw one-time pull (dc_work/data/staging), one parquet per signal, raw rows (the package de-duplicates and filters).
    python ex119.py   -> %DC_WORK%/s119/ev/<DeviceId>.parquet + s119/ev_summary.csv
"""
import os
import time
from pathlib import Path

import duckdb
import pandas as pd

W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
OUT = W / "s119"
t0 = time.time()
lk = pd.read_csv(W / "official" / "locked_v2.csv")
ids = sorted(lk.DeviceId.str.lower().unique())
assert len(ids) == 115
c = duckdb.connect()
c.execute(f"SET threads=4; SET memory_limit='10GB'; SET temp_directory='{(W / 'tmp').as_posix()}'")
idl = ",".join(f"'{d}'" for d in ids)
src = (W / "data" / "staging" / "**" / "*.parquet").as_posix()
c.execute(f"""CREATE TEMP TABLE ev AS
    SELECT lower(CAST(DeviceId AS VARCHAR)) AS DeviceId, CAST(Timestamp AS TIMESTAMP) AS Timestamp,
           CAST(EventId AS INTEGER) AS EventId, CAST(Parameter AS INTEGER) AS Parameter
    FROM read_parquet('{src}', hive_partitioning=false)
    WHERE lower(CAST(DeviceId AS VARCHAR)) IN ({idl})""")
rows = []
for d in ids:
    f = OUT / "ev" / f"{d}.parquet"
    c.execute(f"COPY (SELECT * FROM ev WHERE DeviceId = '{d}' ORDER BY Timestamp) TO '{f.as_posix()}' (FORMAT parquet)")
    n, a, b = c.sql(f"SELECT count(*), min(Timestamp), max(Timestamp) FROM ev WHERE DeviceId = '{d}'").fetchone()
    rows.append(dict(DeviceId=d, n_events=n, t0=a, t1=b))
s = pd.DataFrame(rows).merge(lk.assign(DeviceId=lk.DeviceId.str.lower())[["DeviceId", "DeviceName", "set"]], on="DeviceId")
s.to_csv(OUT / "ev_summary.csv", index=False)
print(s.describe(include="all").T.to_string())
print(f"signals with events {int((s.n_events > 0).sum())} / 115; extract {time.time() - t0:.0f} s")
