"""Health benchmark: about a minute on one machine, one JSON line appended to benchmarks/results.jsonl per run; then
python benchmarks/compare.py flags what got slower.

    python benchmarks/bench_health.py                       # 20 synthetic signals x 2 days (tests/synth.py, seed 0): batch
                                                            # health() and ped_health(), frozen oracle (tests/legacy) and
                                                            # package, interleaved (ABBA: the order flips every repeat,
                                                            # since a machine slows down over a run), vehicle outputs checked equal;
                                                            # then the log in 15-minute updates into a file store, vehicle and
                                                            # ped in the same pass, and a judge of both
    python benchmarks/bench_health.py --events E.parquet --detectors D.parquet   # a real sample instead
    python benchmarks/bench_health.py --impl package --jobs update      # one side, or some of the jobs
Records: batch times (median of --repeat runs, seconds) and their package / oracle ratio; update median / p95 / max, rows,
carry, judge time (vehicle and ped, plus the ped judge and the ped model on their own), store rows and size, outputs checked
equal to health(bin_origin="clock") and ped_health(start, end); peak memory.  Different outputs fail the run.  updates.peak_mb is the process's peak after each block, measured before any batch job runs; a batch job's own
peak needs a process of its own (--jobs health_w1 --repeat 1), since a peak only ever goes up.  The package under test is
the one on PYTHONPATH.  Run it on a quiet machine; compare a line only with lines of the same machine and DuckDB version.

Reference points: health() on 30 signals x 3 days 10.9 s, judging 6.4 s of it [study B]; whole network x 3 days 7.3 min in one
process on DuckDB 2.0, about 2 min with 12 workers [154ad]; chunked prototype 96 ms mean / 175 ms p95 per 30-signal chunk
[study B]; ped health on September (816 signals, 2.75 days) 19 s, plus 11 s for the ped movement model."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _impls(which):
    out = {}
    if which in ("oracle", "both"):
        from legacy import health_duckdb_v2 as H0, ped_health_duckdb_v3 as P0
        out["oracle"] = (H0, P0)
    if which in ("package", "both"):
        from atspm_detector import health_duckdb as H, ped_health_duckdb as P
        out["package"] = (H, P)
    return out


def _code_id() -> str:
    """a short hash of the package's health code (modules and templates): which code a line measured."""
    import atspm_detector
    base = Path(atspm_detector.__file__).parent
    h = hashlib.sha256()
    for f in sorted([*base.glob("health_duckdb.py"), *base.glob("incremental.py"), *base.glob("ped_*.py"), *base.glob("hq/**/*")]):
        if f.is_file() and f.suffix in (".py", ".sql"):
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        return None


def _peak_mb():
    """the process's peak memory so far (Windows: peak working set), None where psutil does not report it."""
    try:
        import psutil
        m = psutil.Process().memory_info()
        return getattr(m, "peak_wset", None) and m.peak_wset / 2 ** 20
    except ImportError:
        return None


def _updates(con, temp) -> dict:
    """the events in 15-minute updates into a file store, a judge of the whole log, against health(bin_origin="clock")."""
    import shutil
    import tempfile

    from atspm_detector import health_duckdb as H, incremental as I, ped_health_duckdb as P
    folder = tempfile.mkdtemp(dir=temp)
    a, z = con.execute("SELECT min(Timestamp), max(Timestamp) FROM events").fetchall()[0]
    a, T = pd.Timestamp(a).floor("15min"), pd.Timestamp(z).floor("15min") + pd.Timedelta(minutes=15)
    horizon = (T - a) / pd.Timedelta(days=1)
    peak = {"loaded": _peak_mb()}
    try:
        I.attach_store(con, f"{Path(folder).as_posix()}/store.duckdb", name="bench_store")
        secs, stats = [], []
        for c in pd.date_range(a + pd.Timedelta(minutes=15), T, freq="15min"):
            con.execute("CREATE OR REPLACE TEMP TABLE bench_chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?",
                        [c - pd.Timedelta(minutes=15), c])
            t = time.perf_counter()
            stats.append(I.update(con, "bench_chunk", "detectors", store="bench_store", horizon_days=max(horizon, 3)))
            secs.append(time.perf_counter() - t)
        con.execute("DROP TABLE bench_chunk")
        peak["updates"] = _peak_mb()
        t = time.perf_counter()
        got = I.judge_rows(con, "detectors", store="bench_store", end=T, horizon_days=horizon)
        judge_s = time.perf_counter() - t
        peak["judge"] = _peak_mb()
        log = "bench_store.health_ped_events"
        t = time.perf_counter()
        P.judge_log(con, log, a, T, True, store="bench_store.")
        ped_s = time.perf_counter() - t
        t = time.perf_counter()
        P.judge_log(con, log, a, T, True, assign=True, store="bench_store.")
        ped_model_s = time.perf_counter() - t
        t = time.perf_counter()
        ref = H.health(con, "events", "detectors", start=a, end=T, periods=True, bin_origin="clock")
        health_s = time.perf_counter() - t
        peak["health_clock"] = _peak_mb()
        same = all(_sorted(x).equals(_sorted(y)) for x, y in zip(got, ref))
        ped_ref = P.ped_health(con, "events", start=a, end=T, periods=True)
        ped_same = all(_sorted(x).equals(_sorted(y)) for x, y in zip((got.ped, got.ped_periods), ped_ref))
        rows = {n: con.execute(f"SELECT count(*) FROM bench_store.{n}").fetchall()[0][0]
                for n in (*I.FEATURES, *I.PED_TABLES, "health_carry", "health_detector_phase")}
        con.execute("CHECKPOINT bench_store")
        size = os.path.getsize(f"{folder}/store.duckdb") / 2 ** 20
        I.detach_store(con, "bench_store")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    s = pd.Series(secs)
    return {"chunks": len(secs), "update_s": {"median": round(s.median(), 3), "p95": round(s.quantile(.95), 3),
                                              "max": round(s.max(), 3), "sum": round(s.sum(), 1)},
            "events_per_chunk_median": int(pd.Series([x.events_in for x in stats]).median()),
            "rows_per_chunk_median": int(pd.Series([sum(x.rows_written.values()) for x in stats]).median()),
            "labelled_per_chunk_median": int(pd.Series([x.labelled_rows for x in stats]).median()),
            "ped_events_per_chunk_median": int(pd.Series([x.ped_events_in for x in stats]).median()),
            "ped_findings": len(got.ped), "ped_periods": len(got.ped_periods),
            "carry_rows": stats[-1].carry_rows, "store_rows": rows, "store_mb": round(size, 1), "judge_s": round(judge_s, 2),
            "judge_ped_s": round(ped_s, 2), "judge_ped_model_s": round(ped_model_s, 2),
            "health_clock_s": round(health_s, 2), "equal_to_health_clock": bool(same), "equal_to_ped_health": bool(ped_same),
            "peak_mb": {k: v and round(v, 1) for k, v in peak.items()}}


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
    return df.sort_values(keys, kind="stable").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=("oracle", "package", "both"), default="both")
    ap.add_argument("--events"), ap.add_argument("--detectors")
    ap.add_argument("--signals", type=int, default=20), ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--jobs", default="health_w1,health_w4,ped_health,update", help="comma-separated subset of the jobs")
    ap.add_argument("--threads", type=int, default=4), ap.add_argument("--memory", default="4GB")
    ap.add_argument("--temp", default=None, help="DuckDB temp_directory (absolute path)")
    ap.add_argument("--note", default=""), ap.add_argument("--out", default=str(Path(__file__).with_name("results.jsonl")))
    a = ap.parse_args(argv)
    con = duckdb.connect()
    con.execute(f"SET threads = {a.threads}; SET memory_limit = '{a.memory}'")
    if a.temp:
        con.execute(f"SET temp_directory = '{Path(a.temp).as_posix()}'")
    if a.events:
        con.execute(f"CREATE TABLE events AS SELECT * FROM read_parquet('{Path(a.events).as_posix()}') "
                    "ORDER BY DeviceId, Timestamp")
        con.execute(f"CREATE TABLE detectors AS SELECT * FROM read_parquet('{Path(a.detectors).as_posix()}')")
        inp = {"events": Path(a.events).name, "detectors": Path(a.detectors).name}
    else:
        import synth
        ev, det = synth.make(a.signals, a.days, seed=0)
        con.register("ev_df", ev), con.register("det_df", det)
        con.execute("CREATE TABLE events AS SELECT * FROM ev_df ORDER BY DeviceId, Timestamp; "
                    "CREATE TABLE detectors AS SELECT * FROM det_df")
        con.unregister("ev_df"), con.unregister("det_df")
        inp = {"synth": f"{a.signals} signals x {a.days} days, seed 0"}
    inp["rows"] = con.execute("SELECT count(*) FROM events").fetchone()[0]
    inp["signals"] = con.execute("SELECT count(DISTINCT DeviceId) FROM events").fetchone()[0]
    impls = _impls(a.impl)
    checks = {}
    want = a.jobs.split(",")
    updates = _updates(con, a.temp) if "update" in want else None     # first: its peak memory is then its own
    if updates and not updates["equal_to_health_clock"]:
        checks["updates"] = "DIFFERENT"
    if updates and not updates["equal_to_ped_health"]:
        checks["updates_ped"] = "DIFFERENT"
    jobs = {"health_w1": lambda H, P: H.health(con, "events", "detectors", periods=True),
            "health_w4": lambda H, P: H.health(con, "events", "detectors", periods=True, workers=4),
            "ped_health": lambda H, P: P.ped_health(con, "events", periods=True)}
    jobs = {j: jobs[j] for j in want if j != "update"}
    times = {k: {j: [] for j in jobs} for k in impls}
    outs = {}
    for rep in range(a.repeat):               # interleaved: each repeat runs every job of every implementation,
        for j, f in jobs.items():             # in the other order on every other repeat (ABBA)
            for k, (H, P) in (list(impls.items()) if rep % 2 == 0 else list(impls.items())[::-1]):
                t = time.perf_counter()
                r = f(H, P)
                times[k][j].append(time.perf_counter() - t)
                outs.setdefault((k, j), r)
    if len(impls) == 2:                       # the package must give the oracle's outputs at this size
        for j in jobs:
            x, y = outs[("oracle", j)], outs[("package", j)]
            same = all(u.equals(v) for u, v in zip(x, y))
            checks[j] = "same" if same else "DIFFERENT"
    med = {k: {j: round(statistics.median(v), 3) for j, v in t.items()} for k, t in times.items()}
    line = {"date": dt.datetime.now().isoformat(timespec="seconds"), "note": a.note, "impl": a.impl,
            "machine": platform.node(), "python": platform.python_version(), "duckdb": duckdb.__version__,
            "pandas": pd.__version__, "commit": _commit(), "threads": a.threads, "memory": a.memory, "code": _code_id(), "input": inp,
            "repeat": a.repeat, "seconds": med, "runs": {k: {j: [round(x, 3) for x in v] for j, v in t.items()}
                                                          for k, t in times.items()},
            "rss_mb": _rss_mb(), "peak_mb": _peak_mb()}
    if len(impls) == 2:
        line["ratio_package_oracle"] = {j: round(med["package"][j] / med["oracle"][j], 3) for j in jobs}
        line["equal_outputs"] = checks
    if updates:
        line["updates"] = updates
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line, indent=1))
    if "DIFFERENT" in checks.values():
        raise SystemExit(f"outputs differ: {checks}")


if __name__ == "__main__":
    main()
