"""Signal groups, the statement loop of one group, and worker processes (moved from health_duckdb, note 154ad).

This file also runs as a script, the worker of health(workers=N): `python -P run.py <job dir> <k>`.  It then imports
nothing of the package (standard library and duckdb only): the statements it runs come rendered in the job file."""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PREFIX = "hq_"


def groups(sig, max_events) -> list:
    """signals (key, events) in order -> groups of keys of at most max_events events (None: one group); a signal with
    more events than that is a group of its own."""
    out, cur, size = [], [], 0
    for d, n in sig:
        if cur and max_events is not None and size + n > max_events:
            out.append(cur)
            cur, size = [], 0
        cur.append(d)
        size += n
    if cur:
        out.append(cur)
    return out


def group(con, stmts: list, drops: list, nf: int) -> list:
    """run a script (Plan.script) for the signals in its window table; the last nf statements are the final queries:
    their results as DataFrames."""
    res = []
    for i, st in enumerate(stmts):
        if i < len(stmts) - nf:
            con.execute(st)
        else:
            res.append(con.execute(st).df())
        for t in drops[i]:
            con.execute(f"DROP TABLE IF EXISTS {t}")
    return res


def _bytes(s: str) -> float:
    """a DuckDB memory setting ('9.3 GiB', '512.0 MB', '1000 bytes') in bytes."""
    m = re.fullmatch(r"\s*([\d.]+)\s*(?:([KMGTP])(i?)B|B|bytes)?\s*", str(s), re.I)
    if not m:
        raise ValueError(f"cannot read the memory_limit {s!r}")
    unit = (1024 if m.group(3) else 1000) ** ("KMGTP".find(m.group(2).upper()) + 1) if m.group(2) else 1
    return float(m.group(1)) * unit


def parallel(con, grp: list, events_table: str | None, detectors_table: str, script: tuple, nf: int, workers: int,
             allowed: tuple = (), store: str = "", store_tables: tuple = ()) -> list:
    """health(workers=N): N worker processes (_worker) start while this connection copies the window's events (with
    their group), the windows and the detectors table once into a DuckDB file of the job directory (note 154ad: about
    half the time of a parquet copy partitioned by group); then each worker takes the groups in order, the next one
    nobody has taken.  script: (statements, drops) rendered for the job's tables (hq_events, hq_job.detectors; each worker puts its group's rows of hq_job.ev
    in place of hq_events).  A judge of a store (events_table None) copies the rows of store_tables of the store instead
    (a store file open for writing cannot be opened by another process); its script reads them as hq_job.<table>.  The
    results come back pickled (each the worker's own .df(), so the dtypes are those of a run in this process) and are
    returned in group order."""
    setting = dict(con.execute("SELECT name, value FROM duckdb_settings() WHERE name IN ('threads', 'memory_limit', "
                               "'temp_directory', 'preserve_insertion_order')").fetchall())
    workers = min(workers, len(grp))
    tdir = setting.get("temp_directory") or ""
    base = tdir if os.path.isabs(tdir) else tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    jd = tempfile.mkdtemp(prefix="atspm_health_", dir=base)
    gp = Path(jd).as_posix()
    procs, attached = [], False
    try:
        with open(f"{jd}/job.json", "w", encoding="utf-8") as f:
            json.dump({"stmts": script[0], "drops": script[1], "nf": nf, "groups": len(grp),
                       "threads": max(1, int(setting["threads"]) // workers),
                       "memory": f"{int(_bytes(setting['memory_limit']) / workers / 2 ** 20)}MiB",
                       "order": str(setting.get("preserve_insertion_order", "true")).lower()}, f)
        for k in range(workers):    # this file run as a script: no import of the whole package (about 0.7 s each)
            log = open(f"{jd}/worker{k}.log", "w", encoding="utf-8")
            procs.append((subprocess.Popen([sys.executable, "-P", str(Path(__file__).resolve()), jd, str(k)],
                                           stdin=subprocess.PIPE, stdout=log, stderr=log), log))
        con.execute(f"ATTACH '{gp}/job.duckdb' AS {PREFIX}job")
        attached = True
        con.execute(f"CREATE TABLE {PREFIX}job.wall AS SELECT w.DeviceId, w.s_ns, w.e_ns, g.g FROM {PREFIX}wall w "
                    "JOIN (SELECT unnest(?) AS d, unnest(?) AS g) g ON CAST(w.DeviceId AS VARCHAR) = g.d",
                    [[str(d) for g in grp for d in g], [i for i, g in enumerate(grp) for _ in g]])
        if events_table is None:
            for t in store_tables:
                con.execute(f"CREATE TABLE {PREFIX}job.{t} AS SELECT * FROM {store}.{t} "
                            f"WHERE DeviceId IN (SELECT DeviceId FROM {PREFIX}job.wall)")
        else:
            codes = ",".join(str(c) for c in allowed)
            # the rows step ev keeps (allowed codes inside the signal's window), each with its group
            con.execute(f"""CREATE TABLE {PREFIX}job.ev AS SELECT e.DeviceId, e.Timestamp, e.EventId, e.Parameter, w.g
                FROM {events_table} e JOIN {PREFIX}job.wall w ON e.DeviceId = w.DeviceId
                WHERE e.EventId IN ({codes}) AND (e.EventId NOT IN (81, 82) OR e.Parameter BETWEEN 1 AND 64)
                  AND e.Timestamp >= make_timestamp(w.s_ns // 1000) AND e.Timestamp <= make_timestamp(w.e_ns // 1000 + 1)
                  AND epoch_ns(e.Timestamp) >= w.s_ns AND epoch_ns(e.Timestamp) < w.e_ns""")
        con.execute(f"CREATE TABLE {PREFIX}job.detectors AS SELECT * FROM {detectors_table}")
        con.execute(f"DETACH {PREFIX}job")
        attached = False
        for p, _ in procs:          # go (a worker that already failed is reported below)
            try:
                p.stdin.write(b"1")
                p.stdin.close()
            except OSError:
                pass
        while any(p.poll() is None for p, _ in procs) and not any(p.poll() for p, _ in procs):
            time.sleep(0.2)
        bad = [k for k, (p, _) in enumerate(procs) if p.poll()]
        if bad:
            procs[bad[0]][1].close()
            with open(f"{jd}/worker{bad[0]}.log", encoding="utf-8", errors="replace") as f:
                raise RuntimeError(f"health worker {bad[0]} failed:\n{f.read()[-3000:]}")
        res = []
        for i in range(len(grp)):
            with open(f"{jd}/r{i}.pkl", "rb") as f:
                res.append(pickle.load(f))
        return res
    finally:
        for p, log in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
            log.close()
        if attached:
            con.execute(f"DETACH {PREFIX}job")
        shutil.rmtree(jd, ignore_errors=True)


def _worker(jd: str, k: int):
    """worker k of health(workers=N) on the job directory jd (parallel): once the data is ready (a byte on stdin; the
    pipe closing without it -- the caller gone -- ends the worker), takes the next free group until none is left."""
    import duckdb
    with open(f"{jd}/job.json", encoding="utf-8") as f:
        J = json.load(f)
    gp = Path(jd).as_posix()
    con = duckdb.connect()
    con.execute(f"SET threads = {J['threads']}")
    con.execute(f"SET memory_limit = '{J['memory']}'")
    con.execute(f"SET temp_directory = '{gp}/tmp{k}'")
    con.execute(f"SET preserve_insertion_order = {J['order']}")
    if sys.stdin.buffer.read(1) != b"1":
        return
    con.execute(f"ATTACH '{gp}/job.duckdb' AS {PREFIX}job (READ_ONLY)")
    for i in range(J["groups"]):
        try:
            os.close(os.open(f"{jd}/c{i}", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wpart AS SELECT DeviceId, s_ns, e_ns FROM {PREFIX}job.wall "
                    f"WHERE g = {i}")
        # the events source of the statements (hq_events) is this group's rows of the job file: a subquery, no view
        src = f"(SELECT DeviceId, Timestamp, EventId, Parameter FROM {PREFIX}job.ev WHERE g = {i})"
        stmts = [re.sub(rf"\b{PREFIX}events\b", src, st) for st in J["stmts"]]
        r = group(con, stmts, J["drops"], J["nf"])
        r = (r[0], r[1] if len(r) > 1 else None) + tuple(r[2:])
        with open(f"{jd}/r{i}.tmp", "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(f"{jd}/r{i}.tmp", f"{jd}/r{i}.pkl")
    con.close()


if __name__ == "__main__":      # worker k of health(workers=N): python -P run.py <job dir> <k>
    _worker(sys.argv[1], int(sys.argv[2]))
