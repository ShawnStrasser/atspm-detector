"""Note 151 prep: the h24_b day (Mon 2026-09-28 00:00-24:00) of the 755 signals of notes 140-150 as ONE parquet,
sorted by DeviceId, Timestamp, allowed event codes only (predict drops the others anyway), locked_v2 asserted absent.
    python prep151.py  -> %DC_WORK%/s151/h24b_events.parquet, h24b_ids.csv
"""
import os
from pathlib import Path

import duckdb
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s151"
EVD = DCW / "health4" / "w40_events"
CODES = "1,7,8,9,10,11,43,44,81,82,131,150,173"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    sig = set(pd.read_parquet(DCW / "s140" / "parity_h24b.parquet", columns=["DeviceId"]).DeviceId.str.lower())
    assert not sig & locked
    dirs = sorted(p for p in EVD.iterdir() if p.name.split("=", 1)[1].lower() in sig)
    ids = [p.name.split("=", 1)[1] for p in dirs]
    pd.DataFrame({"DeviceId": ids}).to_csv(OUT / "h24b_ids.csv", index=False)
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='10GB'")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    files = ",".join(f"'{p.as_posix()}/*.parquet'" for p in dirs)
    con.execute(f"""COPY (SELECT DeviceId, Timestamp, EventId::INTEGER AS EventId, Parameter::INTEGER AS Parameter
        FROM read_parquet([{files}], hive_partitioning=true)
        WHERE Timestamp >= TIMESTAMP '2026-09-28 00:00:00' AND Timestamp < TIMESTAMP '2026-09-29 00:00:00'
          AND EventId IN ({CODES})
        ORDER BY DeviceId, Timestamp) TO '{(OUT / 'h24b_events.parquet').as_posix()}' (ROW_GROUP_SIZE 200000)""")
    print(con.sql(f"SELECT count(*), count(DISTINCT DeviceId) FROM '{(OUT / 'h24b_events.parquet').as_posix()}'").fetchall())


if __name__ == "__main__":
    main()
