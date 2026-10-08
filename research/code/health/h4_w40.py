"""Note 46: the late-September pull (dc_work/data/staging_2026_w40, Sat 26 Sep 16:15 - Mon 28 Sep 24:00) as an
independent period for health: per-signal event cache (training signals only, de-duplicated, channels <= 64 for
81/82, allowed codes only), then health v4 (final configuration) on the whole span (~56 h) with the same out-of-fold site inputs.

    python h4_w40.py cache   -> %DC_WORK%/health4/w40_events/DeviceId=*/
    python h4_w40.py health  -> %DC_WORK%/health4/health_w40.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h4_final as F  # noqa: E402
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
SRC = H.DCW / "data" / "staging_2026_w40"
EVC = H.DCW / "health4" / "w40_events"
SPAN = ("2026-09-26 16:15", "2026-09-29 00:00")


def devs():
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    return sorted({d.lower() for d in folds.DeviceId} - locked)


def cache(a):
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='10GB'; SET threads=8; SET temp_directory='{(H.DCW / 'tmp').as_posix()}'")
    con.register("keep", pd.DataFrame({"d": devs()}))
    codes = ",".join(str(c) for c in hc.ALLOWED)
    con.execute(f"""COPY (SELECT DISTINCT lower(DeviceId) AS DeviceId, Timestamp, EventId, Parameter
        FROM read_parquet('{SRC.as_posix()}/date=*/*.parquet', hive_partitioning=false)
        WHERE lower(DeviceId) IN (SELECT d FROM keep) AND EventId IN ({codes})
          AND NOT (EventId IN (81, 82) AND Parameter > 64))
        TO '{EVC.as_posix()}' (FORMAT parquet, PARTITION_BY (DeviceId), OVERWRITE_OR_IGNORE true)""")
    print(len(list(EVC.glob("DeviceId=*"))), "signals cached")


def one(dev):
    p = EVC / f"DeviceId={dev}"
    if not p.is_dir():
        return None
    ev = ds.dataset(p).to_table().to_pandas()
    ph, fn, ln, pc = F.inputs("stg", dev)
    h = F.run(ev, pd.Timestamp(SPAN[0]), pd.Timestamp(SPAN[1]), dev, "final", ph, fn, ln, pc, 0)
    x = F.slim(h)
    x["bad_periods"] = F.periods(h)
    x["reason"] = h.reason
    x.insert(0, "DeviceId", dev)
    return x


def health(a):
    t = time.time()
    with Pool(a.procs, initializer=F.init, initargs=(["final"],)) as p:
        res = [r for r in p.imap_unordered(one, devs(), chunksize=2) if r is not None]
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(H.DCW / "health4" / "health_w40.parquet", index=False)
    print(df.shape, df.status.value_counts().to_dict(), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "health"])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    globals()[a.stage](a)
