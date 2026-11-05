"""Detector health kept up to date from chunks of events (design 4 and 7.2): each update() turns a chunk into small stored
rows, in a DuckDB store attached to the connection; judge() gives the tables of health() whenever they are wanted.

    con = duckdb.connect()
    attach_store(con, "health.duckdb")
    update(con, "events", "detectors")          # every 15 minutes, with whatever arrived (any order; late rows counted)
    r = judge(con, "detectors", summary=True)   # r.findings, r.status: vehicle detectors and ped buttons in one table

    trim(con)                                   # update does this itself; compact=True also merges the stored deltas
    rebuild(con, "raw", "detectors", devices=["1001"])   # again from raw events: late data, a detector's phase changed
    export_store(con, "folder"); import_store(con2, "folder")     # parquet: carries a store across DuckDB versions

The store holds what health() would extract from the same events; judged at T with horizon H, on events from T - H on, the
answers equal health(start=T - H, end=T, bin_origin="clock"), for any cutting of the events into updates (design 3.5).
judge_rows() gives the per-detector rows those tables are made from.  Vehicle and ped health come in the same pass; a store
attached with ped=False keeps the vehicle rows only.

judge(outputs=("tables", "bins")) adds r.bins (bins.py): rows are only ever added and past rows never change.  Each judge gives the bins since the store's last bins, whatever the
cutting of the events into updates, for the same judge times, and with a past TimeStamp any problem it is the first to find in its window's released bins: the store's
health_bins_released (kept for the horizon) knows what was released, so no bin, detector and label is written twice.  A button or detector the bins' judge rated stuck ON and still open stays in the carry (RATED)
for as long as it stays on, past the horizon and the trim, and keeps its bins; health() on a window knows no ON before it and stays as it was."""
from __future__ import annotations

import hashlib
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from . import bins as B
from . import findings as F
from . import health_duckdb as H
from . import ped_health_duckdb as PH
from .hq import plan as _plan
from .hq import render as _render
from .hq import run as _run

FORMAT_VERSION = "3"            # the store's tables (2: the ped log; 3: the active table)
EXTRACTOR_VERSION = "1"         # what the rows mean: a store written by another version is refused
GREEN_OVERLAP = {"all": None, "count": H.OVERLAP_FNS}
FEATURES = ("health_signal_5min", "health_detector_5min", "health_phase_5min", "health_phase_15min", "health_tick_counts",
            "health_intervals")
PED_TABLES = ("health_ped_events", "health_log_15min")
JUDGE_TABLES = FEATURES + ("health_carry",)              # what the vehicle judge reads
DEVICE_TABLES = FEATURES + ("health_carry", "health_detector_phase") + PED_TABLES      # one device's rows: what rebuild replaces
RELEASED = "health_bins_released"
STORE_TABLES = DEVICE_TABLES + ("health_active", RELEASED)        # all but health_meta
OUTPUTS = ("tables", "15min", "active", "bins")
RATED = (9223, 9232)            # carry rows written by a judge, an ON rated stuck and still open (hq/sql/findings/held.sql): not the extractor's to rewrite
TEMPS = ("uev", "uwall", "wpart", "uwp", "pwin")
P = H.PREFIX
DEFAULT_HORIZON_DAYS = 3
# a row stamped further than this (the longest window health() takes) plus the horizon before the store's latest time is a
# wrong clock: no window can hold it and, taken at its word, it would open a chain decades long (design 3.3)
STALE_MARGIN = pd.Timedelta(days=H.MAX_DAYS)


@dataclass
class UpdateStats:
    """what one update() did."""
    events_in: int = 0                                  # new rows taken (allowed codes; duplicates inside the chunk too)
    events_late: dict = field(default_factory=dict)     # signal -> rows at or before its watermark, not processed
    events_bad_stamp: dict = field(default_factory=dict)    # signal -> rows of a wrong clock (design 3.3), not processed
    events_after_upto: dict = field(default_factory=dict)   # signal -> rows stamped at or after upto, not processed by this update
    ped_events_in: int = 0                              # ped on: new rows of any code taken by the ped part
    ped_events_late: dict = field(default_factory=dict)     # signal -> rows of any code at or before its last processed one
    signals: int = 0                                    # signals with new rows
    rows_written: dict = field(default_factory=dict)    # store table -> rows
    carry_rows: int = 0                                 # rows of the carry after the update, the whole store
    open_chains: int = 0
    open_pieces: int = 0
    labelled_rows: int = 0                              # rows written that depend on something still open
    phase_changes: int = 0                              # detectors whose phase differs from the store's
    through: pd.Timestamp | None = None                 # the latest watermark
    seconds: dict = field(default_factory=dict)         # block -> s


@dataclass
class HealthResult:
    """what judge() gives.  findings: as health(); status: as health(summary=True), with summary=True; q15, active, bins: with
    "15min" / "active" / "bins" in outputs (findings.Q15_COLS, findings.ACTIVE_COLS, bins.BIN_COLS)."""
    findings: pd.DataFrame
    status: pd.DataFrame | None = None
    q15: pd.DataFrame | None = None
    active: pd.DataFrame | None = None
    bins: pd.DataFrame | None = None


@dataclass
class JudgeRows:
    """what judge_rows() gives: the per-detector rows of health_duckdb.health() and ped_health_duckdb.ped_health(), with the
    windows they cover; it unpacks as (detectors, periods), as health_duckdb.health() returns."""
    detectors: pd.DataFrame
    periods: pd.DataFrame | None = None
    ped: pd.DataFrame | None = None
    ped_periods: pd.DataFrame | None = None
    windows: pd.DataFrame | None = None
    ped_windows: pd.DataFrame | None = None
    limits: pd.DataFrame | None = None

    def __getitem__(self, i):
        return (self.detectors, self.periods)[i]


LIMITS_FINAL = """SELECT w.DeviceId, l.det, l.lim_stuck, l.bad_stuck, l.lim_dropout, l.bad_dropout, l.lim_rapid, l.bad_rapid, l.lim_volume, l.bad_volume,
       1.0 AS share
FROM limits l JOIN win w USING (dev)"""


def _check_name(store: str) -> str:
    if not re.fullmatch(r"[A-Za-z_]\w*", store):
        raise ValueError(f"store name {store!r}: letters, digits and _ only")
    return store


def _horizon(horizon_days) -> pd.Timedelta:
    """the horizon, a positive whole number of quarter-hours (design 4.6)."""
    h = pd.Timedelta(days=horizon_days)
    if h <= pd.Timedelta(0) or h % pd.Timedelta(minutes=15) != pd.Timedelta(0):
        raise ValueError(f"horizon_days={horizon_days}: must be a positive whole number of quarter-hours")
    return h


@lru_cache(maxsize=16)
def _ops(s: str, upto=None, cut="2000-01-01", keep_ns=0, ped=False) -> dict:
    """the named statements of store_ops.sql for store s, up to time upto, with the trim's cut; keep_ns: how far before the
    store's latest time a row is still believed; ped: the chunk's rows of every code are read."""
    txt = _render.render("store_ops.sql", s=f"{s}.", P=P, ALLOWED=H.ALLOWED, upto=upto, cut=cut, keep_ns=keep_ns, ped=ped)
    return {n: q for n, q, _ in _plan.steps(txt)}


def _extractor_context(overlap, device_type, ped) -> dict:
    ctx = H._context(H._options(None), False, True, None, None, False, False, overlap=overlap, phase_only=True,
                     fixed_origin=True)
    return {**ctx, "device_type": device_type, "ped": ped, "LOG_CODES": PH.LOG_CODES}


@lru_cache(maxsize=8)
def _update_plan(store: str, events: str, detectors: str, overlap, device_type: str, restart: bool = False, ped: bool = True) -> _plan.Plan:
    """restart: some signal's watermark is past believing, its carry is not read (design 3.3).  ped: the ped part too (design 6)."""
    carry = f"{store}.health_carry"
    if restart:
        carry = f"(SELECT * FROM {carry} WHERE DeviceId NOT IN (SELECT DeviceId FROM {P}uwall WHERE stale))"
    q = _plan.Plan(P)
    q.subs.update({"@EVENTS@": events, "@DETECTORS@": detectors, "@WINDOWS@": f"{P}wpart", "@DISTINCT@": "DISTINCT",
                   "@CARRY@": carry, "@PHASES@": f"{store}.health_detector_phase", "@PEDWIN@": f"{P}pwin"})
    q.render("update.sql", **_extractor_context(overlap, device_type, ped))
    return q


def fingerprint(overlap=None) -> str:
    """a hash of the extractor as rendered (T4.10: a change must be looked at; see EXTRACTOR_VERSION)."""
    q = _update_plan("s", "e", "d", overlap, "VARCHAR")
    return hashlib.sha256(repr([(n, _plan.tokens(s)) for n, s, _ in q.ctes]).encode()).hexdigest()[:16]


def _meta(con, store: str) -> dict:
    return dict(con.execute(f"SELECT Key, Value FROM {store}.health_meta").fetchall())


def _set_meta(con, store: str, **kv):
    con.execute(f"DELETE FROM {store}.health_meta WHERE Key IN (SELECT unnest(?))", [list(kv)])
    con.executemany(f"INSERT INTO {store}.health_meta VALUES (?, ?)", [[k, str(v)] for k, v in kv.items()])


@contextmanager
def _transaction(con):
    con.execute("BEGIN")
    try:
        yield
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def _ns_ts(ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(ns), unit="ns")


def attach_store(con, path: str | None = None, *, name: str = "health_store", horizon_days=None, options: dict | None = None,
                 green_overlap: str = "all", ped: bool | None = None) -> str:
    """ATTACH the DuckDB file at path (None: an in-memory store) as `name` and check it against this package: returns name.
    The tables are made by the first update, which knows the type of DeviceId.  green_overlap: "all" (every detector with
    a phase) or "count" (Count zones only: cheaper, a function change then needs a new store).  horizon_days: recorded in the
    store, the default of its update, judge and trim; None keeps what is recorded (a new store: 3).  ped: keep the ped log in the
    same pass, recorded in the store too (a new store: True; it cannot change later: the rows before it would be missing)."""
    _check_name(name)
    H._options(options)
    if horizon_days is not None:
        _horizon(horizon_days)
    if green_overlap not in GREEN_OVERLAP:
        raise ValueError(f"green_overlap must be one of {tuple(GREEN_OVERLAP)}")
    if not con.execute("SELECT 1 FROM duckdb_databases() WHERE database_name = ?", [name]).fetchall():
        where = ":memory:" if path is None else str(path).replace("'", "''")
        con.execute(f"ATTACH '{where}' AS {name}")
    con.execute(_ops(name)["meta"])
    m = _meta(con, name)
    if not m:
        with _transaction(con):
            _set_meta(con, name, format_version=FORMAT_VERSION, extractor_version=EXTRACTOR_VERSION,
                      extractor_fingerprint=fingerprint(GREEN_OVERLAP[green_overlap]), origin="epoch",
                      green_overlap=green_overlap, occ_strict=True, ped=True if ped is None else bool(ped),
                      horizon_days=DEFAULT_HORIZON_DAYS if horizon_days is None else horizon_days,
                      created_with=f"duckdb {duckdb.__version__}")
        return name
    if m.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"store {name}: format version {m.get('format_version')}, this package reads {FORMAT_VERSION}; "
                         "start a new store and rebuild it from raw events")
    if m.get("extractor_version") != EXTRACTOR_VERSION:
        raise ValueError(f"store {name}: written by extractor version {m.get('extractor_version')}, this package has "
                         f"{EXTRACTOR_VERSION}; start a new store")
    if m.get("green_overlap") != green_overlap:
        raise ValueError(f"store {name}: green_overlap is {m.get('green_overlap')!r}, not {green_overlap!r}")
    _store_ped(m, ped, name)
    if horizon_days is not None:
        with _transaction(con):
            _set_meta(con, name, horizon_days=horizon_days)
    return name


def detach_store(con, name: str = "health_store") -> None:
    con.execute(f"DETACH {_check_name(name)}")


def _store_ped(m: dict, ped, store: str) -> bool:
    """the store's ped setting; a value asked for must agree with it."""
    has = m.get("ped") == "True"
    if ped is not None and bool(ped) != has:
        raise ValueError(f"store {store} was made with ped={has}; a store's ped setting cannot change")
    return has


def _store_horizon(m: dict, horizon_days) -> pd.Timedelta:
    """the horizon asked for, else the one the store records."""
    return _horizon(float(m["horizon_days"]) if horizon_days is None else horizon_days)


def _device_type(con, table: str) -> str:
    return next(str(r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall() if r[0].lower() == "deviceid")


def update(con, events_table: str, detectors_table: str, *, store: str = "health_store", upto=None, horizon_days=None,
           options: dict | None = None, ped: bool | None = None) -> UpdateStats:
    """Take, per signal, the events after its watermark (and before upto, if given) and write their rows, the new carry, the
    bookkeeping and the trim to the store in ONE transaction: a failure leaves the store as it was, and running an update
    again with the same events changes nothing (they are late then, and counted).  A live run should pass upto, the end of
    the time it covers: an event stamped later (a controller's clock gone wrong) would otherwise become its signal's
    watermark and the store's latest time, and the trim would count back from it for every signal.  Rows held back by upto
    are counted per signal in events_after_upto: a later update takes them only if its table holds them again, so a clock
    running ahead shows there and nowhere else (design 3.3).

    A row stamped more than the horizon + 92 days before the store's latest time is a wrong clock on a signal the store has
    not heard from in that long: not processed, counted, and if that signal's own last event is that old its carry starts
    over (design 3.3).  A repeated hour of the fall clock change falls behind the watermark and is late like any other row;
    the hour is then not mixed in, as a batch run over the same rows would do (design 3.3, 3.5).
    horizon_days: None uses the store's; a value is recorded, and the next trim (this update's, if it has new rows) uses it.
    ped: None uses the store's setting.  The ped part has its own watermark, the signal's last event of any code (design 3.3).

    events_table      DeviceId, Timestamp, EventId, Parameter (any codes, any order); events of one instant must come in
                      one update
    detectors_table   DeviceId, Detector, phase: the phase a start's colour and the green overlap use (logged in
                      health_detector_phase when it changes; from then on the new one)"""
    t0 = time.perf_counter()
    H._options(options)                     # occ_strict=False is refused
    _check_name(store)
    m = _meta(con, store)
    with_ped = _store_ped(m, ped, store)
    horizon = _store_horizon(m, horizon_days)
    dup = con.execute(f"SELECT DeviceId, Detector FROM {detectors_table} GROUP BY DeviceId, Detector HAVING count(*) > 1 "
                      "LIMIT 1").fetchall()
    if dup:
        raise ValueError(f"detectors table: signal {dup[0][0]} detector {dup[0][1]} is listed more than once")
    ev_type = _device_type(con, events_table)
    dtype = m.get("device_type", ev_type)
    events = events_table if ev_type == dtype else (
        f"(SELECT CAST(DeviceId AS {dtype}) AS DeviceId, Timestamp, EventId, Parameter FROM {events_table})")
    overlap = GREEN_OVERLAP[m["green_overlap"]]
    if "device_type" not in m:              # the store's first update
        with _transaction(con):
            con.execute(_render.render("store.sql", device_type=dtype, s=f"{store}."))
            _set_meta(con, store, device_type=dtype)
    stats = UpdateStats()
    q = _update_plan(store, f"{P}uev", detectors_table, overlap, dtype, ped=with_ped)
    try:
        ops = _ops(store, upto=upto, keep_ns=(horizon + STALE_MARGIN).value, ped=with_ped)
        con.execute(ops["uev"].replace("@EVENTS@", events))
        left_out = con.execute(ops["unusable"].replace("@EVENTS@", events)).fetchall()
        stats.events_after_upto = {d: int(n) for d, _, n in left_out if n}
        bad = {d: int(n) for d, n, _ in left_out if n}
        con.execute(ops["uwall"])
        sig = con.execute(f"SELECT DeviceId, n_new, n_late, n_old, stale FROM {P}uwall").fetchall()
        stats.events_late = {d: int(n) for d, _, n, _, _ in sig if n}
        stats.events_in = sum(int(n) for _, n, _, _, _ in sig)
        for d, _, _, n, _ in sig:
            if n:
                bad[d] = bad.get(d, 0) + int(n)
        stats.events_bad_stamp = bad
        if with_ped:
            con.execute(ops["uwp"])
            sp = con.execute(f"SELECT DeviceId, n_new, n_late FROM {P}uwp").fetchall()
            stats.ped_events_in = sum(int(n) for _, n, _ in sp)
            stats.ped_events_late = {d: int(n) for d, _, n in sp if n}
            con.execute(f"CREATE OR REPLACE TEMP TABLE {P}pwin AS SELECT DeviceId, s_ns, e_ns, wm FROM {P}uwp WHERE n_new > 0")
        if not stats.events_in and not stats.ped_events_in:
            if horizon_days is not None:
                with _transaction(con):
                    _set_meta(con, store, horizon_days=horizon_days)
            stats.through = _through(con, store)
            return stats
        con.execute(f"CREATE OR REPLACE TEMP TABLE {P}wpart AS SELECT DeviceId, s_ns, e_ns FROM {P}uwall WHERE n_new > 0")
        if any(n and st for _, n, _, _, st in sig):
            q = _update_plan(store, f"{P}uev", detectors_table, overlap, dtype, restart=True, ped=with_ped)
        emits = tuple(q.emits)
        stmts, drops = q.script([], keep=emits + ("win",))
        t1 = time.perf_counter()
        for st, dr in zip(stmts, drops):
            con.execute(st)
            for t in dr:
                con.execute(f"DROP TABLE IF EXISTS {t}")
        t2 = time.perf_counter()
        stats.signals = con.execute(f"SELECT count(*) FROM {P}wpart").fetchall()[0][0]
        for step, t in q.emits.items():
            stats.rows_written[t] = stats.rows_written.get(t, 0) + con.execute(f"SELECT count(*) FROM {P}{step}").fetchall()[0][0]
        rows = stats.rows_written
        stats.labelled_rows = con.execute(
            f"SELECT (SELECT count(*) FROM {P}health_detector_5min WHERE Chain_Tag IS NOT NULL OR Piece_Tag IS NOT NULL)"
            f" + (SELECT count(*) FROM {P}health_phase_5min WHERE Piece_Tag IS NOT NULL)").fetchall()[0][0]
        stats.open_chains, stats.open_pieces = con.execute(
            f"SELECT count(*) FILTER (EventId = 9222), count(*) FILTER (EventId BETWEEN 9300 AND 9302) "
            f"FROM {P}health_carry").fetchall()[0]
        stats.phase_changes = rows["health_detector_phase"] and con.execute(
            f"SELECT count(*) FROM {P}health_detector_phase n SEMI JOIN {store}.health_detector_phase o "
            "ON o.DeviceId = n.DeviceId AND o.Detector = n.Detector").fetchall()[0][0]
        with _transaction(con):
            con.execute(f"DELETE FROM {store}.health_carry WHERE EventId <> 9401 AND EventId NOT IN {RATED} AND DeviceId IN (SELECT DeviceId FROM {P}wpart)")
            if with_ped:
                con.execute(f"DELETE FROM {store}.health_carry WHERE EventId = 9401 AND DeviceId IN (SELECT DeviceId FROM {P}pwin)")
            for step, t in q.emits.items():
                con.execute(f"INSERT INTO {store}.{t} BY NAME SELECT * FROM {P}{step}")
            through = _through(con, store)
            _set_meta(con, store, through=through.isoformat(), **({} if horizon_days is None else {"horizon_days": horizon_days}))
            _trim(con, store, horizon, through)
        t3 = time.perf_counter()
        stats.through = through
        stats.carry_rows = con.execute(f"SELECT count(*) FROM {store}.health_carry").fetchall()[0][0]
        stats.seconds = {"windows": t1 - t0, "extract": t2 - t1, "write": t3 - t2}
        return stats
    finally:
        for n, _, _ in q.ctes:
            con.execute(f"DROP TABLE IF EXISTS {P}{n}")
        for n in TEMPS:
            con.execute(f"DROP TABLE IF EXISTS {P}{n}")


def _through(con, store: str) -> pd.Timestamp | None:
    x = con.execute(f"SELECT epoch_ns(max(TimeStamp)) FROM {store}.health_carry WHERE EventId IN (9401, 9402)").fetchall()[0][0]
    return None if x is None else _ns_ts(x)


def _trim(con, store: str, horizon: pd.Timedelta, through: pd.Timestamp, again: bool = False) -> None:
    """the deletes of a trim at C = floor_15min(through) - horizon (design 4.6), on the caller's transaction.  again: rows may
    have come in before the earlier trim (rebuild): cut there at least."""
    cut = through.floor("15min") - horizon
    done = _meta(con, store).get("trimmed_before")
    if done is not None and pd.Timestamp(done) >= cut:
        if not again:
            return
        cut = pd.Timestamp(done)
    ops = _ops(store, cut=cut)
    for n, st in ops.items():
        if n.startswith("trim_"):
            con.execute(st)
    _set_meta(con, store, trimmed_before=cut.isoformat())


def trim(con, *, store: str = "health_store", horizon_days=None, compact: bool = False) -> None:
    """Trim the store at its latest watermark less the horizon, by default its own (update does this itself); compact=True
    also merges the delta rows per key, which changes no total (design 4.6)."""
    m = _meta(con, _check_name(store))
    horizon = _store_horizon(m, horizon_days)
    through = _through(con, store) if "device_type" in m else None
    if through is None:
        return
    ops = _ops(store)
    tables = {"compact_signal": "health_signal_5min", "compact_detector": "health_detector_5min",
              "compact_phase": "health_phase_5min", "compact_phase15": "health_phase_15min",
              "compact_ticks": "health_tick_counts", "compact_log": "health_log_15min"} if compact else {}
    try:
        with _transaction(con):
            _trim(con, store, horizon, through)
            for op, t in tables.items():
                con.execute(f"CREATE OR REPLACE TEMP TABLE {P}{op} AS {ops[op]}")
                con.execute(f"DELETE FROM {store}.{t}")
                con.execute(f"INSERT INTO {store}.{t} BY NAME SELECT * FROM {P}{op}")
    finally:
        for op in tables:
            con.execute(f"DROP TABLE IF EXISTS {P}{op}")


def _judge_window(con, store: str, m: dict, end, horizon_days) -> tuple:
    """the window (start, end) of a judge: end on a quarter-hour and after every processed event, default the first quarter-hour
    after them; start = end - horizon, not before the store's trim."""
    horizon = _store_horizon(m, horizon_days)
    through = _through(con, store) if "device_type" in m else None
    if through is None:
        raise ValueError(f"store {store} holds no events yet")
    end = through.floor("15min") + pd.Timedelta(minutes=15) if end is None else pd.Timestamp(end)
    if end != end.floor("15min"):
        raise ValueError(f"end {end}: a judge ends on a quarter-hour")
    if end <= through:
        raise ValueError(f"end {end}: the store already holds events up to {through}; end must be after them")
    start = end - horizon
    if "trimmed_before" in m and start < pd.Timestamp(m["trimmed_before"]):
        raise ValueError(f"window start {start} is before the store's trim at {m['trimmed_before']}; a longer horizon "
                         "fills in as data arrives")
    return start, end


def judge(con, detectors_table: str, *, store: str = "health_store", end=None, horizon_days=None, options: dict | None = None,
          max_events: int | None = 4_000_000, ped: bool | None = None, ped_options: dict | None = None, ped_assign: bool = False,
          ped_max_events: int | None = 3_000_000, summary: bool = False, outputs: tuple = ("tables",), workers: int = 1,
          bin_minutes: int = 15) -> HealthResult:
    """The health of the window [end - horizon, end) from the store, as health() gives it: r.findings, and with summary=True
    r.status.  end: on a quarter-hour and after every processed event; default the first quarter-hour after them (a window
    starting before the store's trim is refused).  horizon_days: None uses the store's.  ped: None follows the store; False
    leaves ped out; ped_options, ped_assign: as health().  detectors_table: as health().  workers > 1: the vehicle groups run
    in that many processes, on a copy of the store's rows.  Records the end as the store's last_judge_end.
    outputs: "tables" always; "15min" adds r.q15 (findings.q15, the quarter-hours since the previous judge); "active" adds
    r.active and keeps it in the store's health_active (off until reviewed; hq/sql/findings/active.sql has the states); "bins" adds r.bins
    (bins.py): the bins of bin_minutes since the last judge that gave bins, up to the last whole bin before end, and the problems it is the first to find in the window's released bins, with
    their past TimeStamp; no bin, detector and label twice."""
    unknown = set(outputs) - set(OUTPUTS)
    if unknown:
        raise ValueError(f"unknown outputs {sorted(unknown)}: choose from {OUTPUTS}")
    _check_name(store)
    m = _meta(con, store)
    start, end = _judge_window(con, store, m, end, horizon_days)
    r = judge_rows(con, detectors_table, store=store, end=end, horizon_days=horizon_days, options=options, max_events=max_events,
                   ped=ped, ped_options=ped_options, ped_assign=ped_assign, ped_max_events=ped_max_events, workers=workers)
    with_status = summary or len(set(outputs) & {"15min", "active", "bins"}) > 0
    tabs = F.tables(con, r.detectors, r.periods, r.ped, r.ped_periods, r.windows, r.ped_windows, summary=with_status)
    findings, status = tabs if with_status else (tabs, None)
    out = HealthResult(findings, status if summary else None)
    if "15min" in outputs:
        last = m.get("last_judge_end")
        out.q15 = F.q15(con, findings, status, start if last is None else max(start, pd.Timestamp(last)), end)
    if "active" in outputs:
        out.active = F.active_next(con, f"{store}.health_active", findings, status, end)
    add, drop, upto = [], [], None
    if "bins" in outputs:
        upto = B.bin_floor(con, end, bin_minutes)
        since = max(start, pd.Timestamp(m.get("bins_through", start)))
        # the store knows what it released from known on: a store from before the table knows nothing before its last bins, and a longer horizon cannot reach behind what was let go
        known = min(since, max(start, pd.Timestamp(m.get("bins_known_from", m.get("bins_through", start)))))
        con.execute(_render.render("store.sql", device_type=m["device_type"], s=f"{store}."))
        given = con.execute(f"SELECT TimeStamp, DeviceId, DetectorType, Detector, Label, Score FROM {store}.{RELEASED} WHERE TimeStamp >= ?", [known]).df()
        spans, add, drop = B.held(con, store, findings, status, r.ped, since, upto, end, m.get("ped") == "True")
        past = B.held(con, store, findings, status, r.ped, known, since, end, m.get("ped") == "True")[0] if known < since else None
        out.bins = B.release(con, findings, status, since, upto, known, given, bin_minutes, spans, past)
    with _transaction(con):
        _set_meta(con, store, last_judge_end=end.isoformat(), **({} if upto is None else {"bins_through": upto.isoformat(), "bins_known_from": known.isoformat()}))
        if upto is not None:
            con.execute(f"DELETE FROM {store}.{RELEASED} WHERE TimeStamp < ?", [known])
            if len(out.bins):
                con.register(f"{P}released_in", out.bins)
                con.execute(f"INSERT INTO {store}.{RELEASED} BY NAME SELECT * FROM {P}released_in")
                con.unregister(f"{P}released_in")
        for t, d, code, par in drop:
            con.execute(f"DELETE FROM {store}.health_carry WHERE TimeStamp = ? AND DeviceId = ? AND EventId = ? AND Parameter = ?", [t, d, code, par])
        for t, d, code, par in add:
            con.execute(f"INSERT INTO {store}.health_carry VALUES (?, ?, ?, ?, true)", [t, d, code, par])
        if out.active is not None:
            con.execute(f"DELETE FROM {store}.health_active")
            con.register(f"{P}active_in", out.active)
            con.execute(f"INSERT INTO {store}.health_active BY NAME SELECT * FROM {P}active_in")
            con.unregister(f"{P}active_in")
    return out


def judge_rows(con, detectors_table: str, *, store: str = "health_store", end=None, horizon_days=None, options: dict | None = None,
               periods: bool = True, max_events: int | None = 4_000_000, ped: bool | None = None, ped_options: dict | None = None,
               ped_assign: bool = False, ped_max_events: int | None = 3_000_000, workers: int = 1, limits: bool = False) -> JudgeRows:
    """The per-detector rows judge() makes its tables from: health_duckdb.health()'s (detectors, periods; with periods=False the
    detectors only) and with ped on ped_health_duckdb.ped_health()'s (ped, ped_periods), with the windows they cover."""
    _check_name(store)
    opt = H._options(options)
    m = _meta(con, store)
    with_ped = _store_ped(m, True, store) if ped else (m.get("ped") == "True" and ped is None)
    start, end = _judge_window(con, store, m, end, horizon_days)
    job = workers > 1
    q, groups = _judge_plan(con, detectors_table, store, start, end, opt, periods, max_events, job)
    try:
        windows = con.execute(f'SELECT DeviceId, make_timestamp(s_ns::BIGINT // 1000) AS "start", '
                              f'make_timestamp(e_ns::BIGINT // 1000) AS "end" FROM {P}wall').df()
        finals = ([H.FINAL, H.PERIODS_FINAL] if periods else [H.FINAL]) + ([LIMITS_FINAL] if limits else [])
        stmts, drops = q.script(finals)
        nf = len(finals)
        if job and len(groups) > 1:
            res = _run.parallel(con, groups, None, detectors_table, (stmts, drops), nf, workers, store=store, store_tables=JUDGE_TABLES)
        else:
            if job:     # one group: this process, on the store's own rows
                q, groups = _judge_plan(con, detectors_table, store, start, end, opt, periods, max_events)
                stmts, drops = q.script(finals)
            res = H._run_groups(con, q, stmts, drops, groups, nf)
    finally:
        con.execute(f"DROP TABLE IF EXISTS {P}wall")
    out = H._frames(res, periods)
    r = JudgeRows(*out, windows=windows) if periods else JudgeRows(out, windows=windows)
    if limits:
        parts = [x[-1] for x in res if len(x) > 2 and x[-1] is not None]
        r.limits = pd.concat(parts, ignore_index=True) if parts else None
    if with_ped:
        log = f"{store}.health_ped_events"
        p = PH.judge_log(con, log, start, end, periods, ped_max_events, ped_options, ped_assign, store=f"{store}.")
        r.ped, r.ped_periods = p if periods else (p, None)
        r.ped_windows = F.event_windows(con, log, start=start, end=end, ped=True)
    return r


def _judge_plan(con, detectors_table: str, store: str, start, end, opt: dict, periods: bool, max_events: int | None,
                job: bool = False) -> tuple:
    """the steps of a judge over [start, end) and its signal groups; the windows are in hq_wall (windows from the store,
    the same rules and groups as health(); a store's windows always start on the clock).  job: the steps read a copy of the
    store's rows and of the detectors table in the job file of health_duckdb's workers (run.parallel)."""
    cols = H._cols(con, detectors_table)
    probs, cfg = set(H.PROB.values()) <= cols, "configured_function" in cols
    sp = f"{store}."
    wq = _plan.Plan(P)
    wq.render("windows.sql", **H._context(H._options(None), False, False, start, end, probs, False, store=sp))
    wq.ctes = [c for c in wq.ctes if c[0] in ("wraw", "wbin", "wsel")]
    con.execute(f"CREATE OR REPLACE TEMP TABLE {P}wall AS " + wq.text(
        "SELECT w.DeviceId, w.s_ns, w.e_ns, r.n FROM wsel w JOIN wraw r USING (DeviceId)"))
    groups = H._wall_groups(con, max_events, "clock")
    q = _plan.Plan(P, opt)
    q.subs.update({"@DETECTORS@": f"{P}job.detectors" if job else detectors_table, "@WINDOWS@": f"{P}wpart"})
    q.render("judge_store.sql", **H._context(opt, periods, True, None, None, probs, cfg, "clock", store=f"{P}job." if job else sp,
                                             overlap=None))
    return q, groups


def rebuild(con, events_table: str, detectors_table: str, *, store: str = "health_store", devices=None, upto=None,
            options: dict | None = None) -> UpdateStats:
    """Replace what the store holds on these devices (None: every device of events_table) with a fresh extraction of their raw
    events (late data, a detector's phase changed), in ONE transaction; the result equals a store fed those events from the
    start, cut at this store's trim.  A device with no event in events_table stays as it was.  Other devices are not touched.
    upto: as update()."""
    _check_name(store)
    m = _meta(con, store)
    if "device_type" not in m:
        raise ValueError(f"store {store} holds no events yet; use update")
    events = events_table
    if devices is not None:
        names = ",".join("'" + str(d).replace("'", "''") + "'" for d in devices)
        events = f"(SELECT * FROM {events_table} WHERE CAST(DeviceId AS VARCHAR) IN ({names}))" if names else None
    if events is None:
        return UpdateStats(through=_through(con, store))
    fresh = f"{store}_rebuild"
    attach_store(con, None, name=fresh, horizon_days=float(m["horizon_days"]), green_overlap=m["green_overlap"],
                 ped=m.get("ped") == "True")
    try:
        with _transaction(con):
            con.execute(_render.render("store.sql", device_type=m["device_type"], s=f"{fresh}."))
            _set_meta(con, fresh, device_type=m["device_type"])
        stats = update(con, events, detectors_table, store=fresh, upto=upto, options=options)
        where = f"WHERE DeviceId IN (SELECT DeviceId FROM {fresh}.health_carry)"
        with _transaction(con):
            for t in DEVICE_TABLES:
                keep = f" AND EventId NOT IN {RATED}" if t == "health_carry" else ""     # what a judge rated stays: a fresh extraction knows no judge
                con.execute(f"DELETE FROM {store}.{t} {where}{keep}")
                con.execute(f"INSERT INTO {store}.{t} BY NAME SELECT * FROM {fresh}.{t}")
            through = _through(con, store)
            _set_meta(con, store, through=through.isoformat())
            _trim(con, store, _store_horizon(m, None), through, again=True)
        stats.through = through
        stats.carry_rows = con.execute(f"SELECT count(*) FROM {store}.health_carry").fetchall()[0][0]
        return stats
    finally:
        detach_store(con, fresh)


def export_store(con, folder, *, store: str = "health_store") -> None:
    """A parquet copy of the store, one file per table in `folder`: moves a store across DuckDB versions (import_store)."""
    _check_name(store)
    m = _meta(con, store)
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    if "device_type" in m:      # a store from before a table has it now, empty
        con.execute(_render.render("store.sql", device_type=m["device_type"], s=f"{store}."))
    for t in (STORE_TABLES if "device_type" in m else ()) + ("health_meta",):
        where = str(path / f"{t}.parquet").replace("'", "''")
        con.execute(f"COPY (SELECT * FROM {store}.{t}) TO '{where}' (FORMAT PARQUET)")


def import_store(con, folder, *, store: str = "health_store") -> None:
    """Fill a new, empty store (attach_store first) from an export_store folder, in one transaction.  The export must have this
    package's format and extractor versions; what the store records (horizon, ped, green overlap, its trim) comes with it."""
    _check_name(store)
    if "device_type" in _meta(con, store):
        raise ValueError(f"store {store} already holds data; import into a new store")
    path = Path(folder)

    def src(t):
        return "read_parquet('" + str(path / f"{t}.parquet").replace("'", "''") + "')"
    if not (path / "health_meta.parquet").exists():
        raise ValueError(f"{folder}: no export of a store here")
    m = dict(con.execute(f"SELECT Key, Value FROM {src('health_meta')}").fetchall())
    if m.get("format_version") != FORMAT_VERSION or m.get("extractor_version") != EXTRACTOR_VERSION:
        raise ValueError(f"{folder}: format version {m.get('format_version')}, extractor version {m.get('extractor_version')}; "
                         f"this package reads {FORMAT_VERSION}, {EXTRACTOR_VERSION}")
    with _transaction(con):
        if "device_type" in m:
            con.execute(_render.render("store.sql", device_type=m["device_type"], s=f"{store}."))
            for t in STORE_TABLES:
                if (path / f"{t}.parquet").exists():    # an export from before a table has no file of it
                    con.execute(f"INSERT INTO {store}.{t} BY NAME SELECT * FROM {src(t)}")
        con.execute(f"DELETE FROM {store}.health_meta")
        _set_meta(con, store, **m)
