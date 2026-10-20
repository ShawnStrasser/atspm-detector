"""Note 153 timing: detector health ALONE, numpy (main d187b2d) vs SQL, on the first N seed-151 h24_b signals for
windows of 1 day (Mon 28 Sep), 2 days (27-28 Sep) or the whole pull ("3": Sat 26 Sep 16:15 - Tue 29 Sep 00:00, about
2.3 days; the pulls start on the 26th afternoon).  One (method, N, days) per process; peak = process peak working set.

    python tim153.py <method> <N> <days> --src <package parent>
methods: numpy    main's _health per part of 10 signals, as main's predict runs it (src = main d187b2d)
         sqlpred  the SQL _health per batch of 25 signals, as predict on the SQL health runs it (src = commit 90b979a)
         api      health(con, "events", "detectors", start, end) with the events already in a DuckDB table
The classifier's answers come from cap153.py (one-day predict of the 755 signals; reused for 2 / 3 days: the health
inputs are the same kind, only the window grows).  Events / answers loading is not timed.  Appends one CSV row to
%DC_WORK%/s153/tim/times.csv and saves the health output for the cross-method equality check."""
import argparse
import os
import sys
import time
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
WIN = {"1": ("2026-09-28 00:00", "2026-09-29 00:00"), "2": ("2026-09-27 00:00", "2026-09-29 00:00"),
       "3": ("2026-09-26 00:00", "2026-09-29 00:00")}
H = ["health_status", "health_score", "health_reason", "health_bad_periods", "health_categories", "health_config",
     "health_watch", "health_signal_note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("n", type=int)
    ap.add_argument("days")
    ap.add_argument("--src", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import duckdb
    import numpy as np
    import pandas as pd
    from run153 import order_ids, peak_mb
    from atspm_detector import pipeline as PP
    ids = order_ids()[:a.n]
    start, end = WIN[a.days]
    ev = str(DCW / "s153" / "d3_events.parquet")
    ans = pd.read_parquet(DCW / "s153" / "tim" / "answers_755.parquet")
    ans = ans[ans.DeviceId.isin(ids)].reset_index(drop=True)
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='10GB'; "
                f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    notes = {}
    if a.method in ("numpy", "sqlpred"):
        PP.load_events(con, ev, ids, start, end)
        W = PP._windows(con, start, end)
        n_ev = con.execute("SELECT count(*) FROM evd").fetchone()[0]
        span = float((W.b1 - W.b0).sum()) / 86400.0
        devs = sorted(W.DeviceId.astype(str))
        wdev = W.set_index(W.DeviceId.astype(str))
        base = peak_mb()
        t0 = time.perf_counter()
        if a.method == "numpy":
            dg = {k: g for k, g in ans.groupby("DeviceId", sort=True)}
            parts = []
            for i in range(0, len(devs), 10):
                part = devs[i:i + 10]
                res = PP._health(con, wdev.loc[part].reset_index(drop=True), dg, notes)
                for dev, (h, note) in res.items():
                    if len(h):
                        h = h.rename(columns={"detector": "Detector"})
                        h.insert(0, "DeviceId", dev)
                        h["health_signal_note"] = note
                        parts.append(h)
            out = pd.concat(parts, ignore_index=True)
        else:
            parts = []
            for i in range(0, len(devs), 25):
                part = devs[i:i + 25]
                parts.append(PP._health(con, wdev.loc[part].reset_index(drop=True), ans[ans.DeviceId.isin(part)], notes))
            out = pd.concat(parts, ignore_index=True)
        dt = time.perf_counter() - t0
    else:
        from atspm_detector.health_duckdb import health
        con.execute(f"CREATE TABLE events AS SELECT DeviceId, Timestamp, EventId, Parameter FROM read_parquet(?) "
                    f"WHERE DeviceId IN (SELECT unnest(?)) AND Timestamp >= TIMESTAMP '{start}' "
                    f"AND Timestamp < TIMESTAMP '{end}'", [ev, ids])
        fp = ans.function_pred.where(ans.function_pred.map(lambda v: isinstance(v, str)))
        ln = ans.n_lanes_spanned.astype(float).where(ans.n_lanes_spanned.astype(float) > 0)
        det = pd.DataFrame({"DeviceId": ans.DeviceId.astype(str), "Detector": ans.Detector.astype(int),
                            "phase": ans.phase_pred.astype(float), "function": fp.astype(object),
                            "lanes": np.where(fp.eq("Bike") & ln.notna(), 1.0, ln)})
        for c in ["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other", "p_mid", "p_bike"]:
            det[c] = ans[c].astype(float)
        con.register("det_df", det)
        con.execute("CREATE TABLE detectors AS SELECT * FROM det_df")
        n_ev = con.execute("SELECT count(*) FROM events").fetchone()[0]
        w = con.execute("SELECT DeviceId, min(Timestamp), max(Timestamp) FROM events GROUP BY 1").df()
        span = None
        base = peak_mb()
        t0 = time.perf_counter()
        out = health(con, "events", "detectors", start=start, end=end)
        dt = time.perf_counter() - t0
        t0s, t1s = pd.Timestamp(start), pd.Timestamp(end)
        f0 = pd.to_datetime(w.iloc[:, 1]); f1 = pd.to_datetime(w.iloc[:, 2]) + pd.Timedelta(seconds=1)
        b0 = np.where((f0 - t0s).dt.total_seconds() > 1800, f0, t0s)
        b1 = np.where((t1s - f1).dt.total_seconds() > 1800, f1, t1s)
        span = float(pd.Series(pd.to_datetime(b1) - pd.to_datetime(b0)).dt.total_seconds().sum()) / 86400
    pk = peak_mb()
    out = out[["DeviceId", "Detector"] + H].copy()
    out["health_score"] = out.health_score.astype(float)
    out.loc[out.health_status.eq("not_enough_data"), "health_score"] = np.nan
    out.to_parquet(DCW / "s153" / "tim" / f"out_{a.method}_{a.n}_{a.days}.parquet")
    row = dict(method=a.method, n=a.n, days=a.days, n_sig=len(set(out.DeviceId)), n_det=len(out), n_events=n_ev,
               signal_days=round(span, 2), health_s=round(dt, 2), ms_per_sigday=round(1000 * dt / span, 1),
               peak_mb=round(pk), peak_before_mb=round(base))
    f = DCW / "s153" / "tim" / "times.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
    print(row, flush=True)


if __name__ == "__main__":
    main()
