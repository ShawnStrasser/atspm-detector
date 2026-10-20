"""Note 153: health(con, ...) on an existing DuckDB connection with the events already in a table (no copy).

    python api153.py <job> [--threads 4]
jobs: h24_1, h24_10, h24_100 (the first 1 / 10 / 100 seed-151 h24_b signals, start / end = the day), multi (the three
4-5 day logs together).  The detectors table = predict()'s answers of that signal set (the run153 'new' outputs):
phase_guess, function_guess, the lane span (entries of `lanes`), class probabilities.  Loading the tables is not
timed.  Prints the health time, the peak working set and the differences against predict()'s health columns."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
H = ["health_status", "health_score", "health_reason", "health_bad_periods", "health_categories", "health_config",
     "health_watch", "health_signal_note"]
MULTI = ["ev_typical_r8_d5", "ev_busiest_ev_d5", "ev_asclog_d4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import duckdb
    import numpy as np
    import pandas as pd
    from run153 import order_ids, peak_mb
    from atspm_detector import health
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='10GB'")
    runs = DCW / "s153" / "runs"
    if a.job.startswith("h24_"):
        n = int(a.job[4:])
        ids = order_ids()[:n]
        day = str(DCW / "s151" / "h24b_events.parquet").replace("\\", "/")
        con.execute(f"CREATE TABLE events AS SELECT DeviceId, Timestamp, EventId, Parameter FROM read_parquet('{day}') "
                    f"WHERE DeviceId IN (SELECT unnest(?))", [ids])
        out = pd.read_parquet(runs / f"new_batch{n}_det.parquet")
        kw = dict(start="2026-09-28 00:00", end="2026-09-29 00:00")
    else:
        for i, m in enumerate(MULTI):
            p = str(DCW / "s141p" / "data" / f"{m}.parquet").replace("\\", "/")
            con.execute(f"{'CREATE TABLE events AS' if i == 0 else 'INSERT INTO events'} SELECT DeviceId, Timestamp, "
                        f"EventId, Parameter FROM read_parquet('{p}')")
        out = pd.read_parquet(runs / "new_multi_det.parquet")
        out = out[out["item"] == "multi_all"]
        kw = {}
    span = out.lanes.map(lambda s: float(len(s.split(","))) if isinstance(s, str) and s else np.nan)
    det = pd.DataFrame({"DeviceId": out.DeviceId, "Detector": out.Detector, "phase": out.phase_guess,
                        "function": out.function_guess, "lanes": span})
    for c in ["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other", "p_mid", "p_bike"]:
        det[c] = out[c]
    con.register("det_df", det)
    con.execute("CREATE TABLE detectors AS SELECT * FROM det_df")
    n_ev = con.execute("SELECT count(*) FROM events").fetchone()[0]
    base = peak_mb()
    t0 = time.perf_counter()
    got = health(con, "events", "detectors", **kw)
    dt = time.perf_counter() - t0
    m = out.dropna(subset=["health_status"])[["DeviceId", "Detector"] + H].merge(
        got, on=["DeviceId", "Detector"], how="outer", suffixes=("", "_sql"), indicator=True)
    diff = {}
    for c in H:
        x, y = m[c], m[c + "_sql"]
        if c == "health_score":
            eq = np.isclose(x.astype(float), y.astype(float), rtol=1e-6, atol=0) | (x.isna() & y.isna())
        else:
            eq = x.fillna("").astype(str) == y.fillna("").astype(str)
        diff[c] = int((~eq).sum())
    bad = m[m.health_reason.fillna("") != m.health_reason_sql.fillna("")]
    for _, r in bad.head(5).iterrows():
        print("  ", r.Detector, "|", r.health_reason, "|", r.health_reason_sql)
    print(f"{a.job}: {n_ev:,} events, {len(got)} detectors, health {dt:.2f} s, peak {peak_mb():.0f} MB "
          f"(before {base:.0f}), merge {m._merge.value_counts().to_dict()}, differences {diff}", flush=True)


if __name__ == "__main__":
    main()
