"""Detector health for a run every 15 minutes, with state carried as tables, each signal on its own:

    r = live_run(con, "raw_data", "detectors", state=..., history=..., ped_history=..., now=None, complete_through=None)
        -> tables hl_final (rows of detector_health), hl_state, hl_rows, hl_ped; r.findings (each whole-day check's findings, with the
           check's midnight as TimeStamp), r.checked (the latest midnight checked), r.held_back (signal -> events left out by now)

hl_state goes to the next run as state.  hl_rows are added to history, which only a whole-day check reads; hl_ped to ped_history.

Every signal is tracked on its own, as the other performance measures are: nothing a signal gets depends on any other signal.
  state       per signal: the end of its last written bin (9602), the midnight of its last whole-day check (9603), the quarter-hour
              of its first event (9604), the runs whose rows it has in history (9605), its events past its last bin (9700), the
              version (9606), and the carry, limits and open ped ONs as before.  Every state row has a DeviceId.
  bins        a signal's bin is written by the run its own data reach the end of: its last event within a second of the bin's end,
              or complete_through (the caller: this signal's events are all in up to then).  Its events after its last bin wait in
              the state (9700), so each bin is judged whole, in one run, with the rows it would have had on time.  A bin with no
              event of the signal has no row, so bins are judged only where the signal has events.
  midnight    a signal's whole-day check runs at the first run its own data pass that midnight, or when the caller says the day is
              over for it: complete_through at or past it, or now at least clock_ahead_minutes past it.  It needs a whole day of
              the signal's data since its first event, and some of its data in the check's window.  A day closed by the caller
              before the signal's data reached it is closed for good: its later events still go into the store and the next
              checks, but its bins before that midnight are not written.
  clocks      now (the caller's time, on the controllers' clock): events stamped at or after now + clock_ahead_minutes are left
              out and counted per signal (a clock running ahead moves nothing on).  Without now each signal's stamps are believed;
              a wrong clock moves only its own signal.
  trim        each signal's rows are trimmed at its own watermark less the horizon (incremental per_device store).
  subsets     a run over some signals reads and writes only their state rows and history, so runs over separate subsets equal one run.
  ped history the 15-minute press counts and logged bins of each ped input (hl_ped: TimeStamp, DeviceId, Input, Presses; a row with
              no Input is a logged bin), kept ped_history_days before the check window for the ped sudden stop (option stop_history),
              which needs 14 logged days of the 28 before a silence.  Rows are read as the largest count per key, so a retried run's
              rows count once.

The steps of a long run are those of the same events cut into 15-minute runs, so any cutting gives the same rows (equal to one big run).
A state written before 2.2.0 (one set of markers with no DeviceId) is converted on reading: every signal in it takes those markers,
which is where each one was, and the runs it lists are listed for each signal, checked by run as before."""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

import pandas as pd

from . import bins as B
from . import findings as F
from . import incremental as I
from .hq import render as _render

STORE, CHECK = "health_store", "health_check"
ROW_TABLES = ("health_signal_5min", "health_detector_5min", "health_phase_5min", "health_phase_15min", "health_tick_counts", "health_intervals", "health_ped_events", "health_log_15min")
RELEASED = "released"
Q = pd.Timedelta(minutes=15)
DAY = pd.Timedelta(days=1)
SECOND = pd.Timedelta(seconds=1)
LIMIT_CODES = {"lim_stuck": 9501, "bad_stuck": 9502, "lim_rapid": 9505, "bad_rapid": 9506, "lim_volume": 9507, "bad_volume": 9508}
TRIMMED, BINS_THROUGH, LAST_CHECK, FIRST_DAY, RUN_ID, VERSION, PEND = 9601, 9602, 9603, 9604, 9605, 9606, 9700
MARKS = {BINS_THROUGH: "b", LAST_CHECK: "c", FIRST_DAY: "f"}
PON_WALKS, PON_GREENS, INPUT_PHASE, INPUT_FLAGS = 9233, 9234, 9520, 9521
PED_HISTORY_DAYS = 28           # ped_stop.HIST_DAYS: the look-back of the sudden stop against the button's own weeks
TS9, I4, I8, I2, F8, TXT = "TIMESTAMP_NS", "INTEGER", "BIGINT", "SMALLINT", "DOUBLE", "VARCHAR"
ROW_COLUMNS = {"Tbl": TXT, "DeviceId": None, "TimeStamp": TS9, "Events": I4, "First_Event": TS9, "Last_Event": TS9, "Detector": I8, "Chain_Tag": TS9, "Piece_Phase": I4, "Piece_Tag": TS9, "Piece_Kind": I2,
               "Starts": I4, "Fast_Starts": I4, "Chatter_Starts": I4, "Starts_GY": I4, "Fast_Starts_R2": I4, "Chatter_Starts_R2": I4, "Relogged_Ons": I4, "Stray_Offs": I4, "Closing_Offs": I4,
               "On_Time_ns": I8, "Unknown_Time_ns": I8, "On_Green5_ns": I8, "Clean_Ons": I4, "Max_On_s": F8, "Relog_Time_ns": I8, "Ons_5s": I4, "Ons_60s": I4, "Long_Max_s": F8, "Phase": I4, "Greens": I4,
               "Green_Yellow_ns": I8, "Green5_ns": I8, "Kind": TXT, "A": I4, "B": I2, "Count": I4, "Channel": I4, "StartTime": TS9, "EndTime": TS9, "Relogged": "BOOLEAN", "Value": I4, "EventId": I2,
               "Parameter": I4, "All_First": TS9, "All_Last": TS9, "DetectorType": TXT, "Label": TXT, "Score": F8, "Run": I8}
OUTPUT = "TimeStamp TIMESTAMP, DeviceId {d}, DetectorType VARCHAR, Detector BIGINT, Label VARCHAR, Score DOUBLE"
PED_OUTPUT = "TimeStamp TIMESTAMP, DeviceId {d}, Input INTEGER, Presses INTEGER"
NO_STATE = "detector_health state {!r} does not exist. Pass state '' only for a first run."


@dataclass
class LiveResult:
    """what live_run gives besides its tables; unpacks as (findings, checked)."""
    findings: pd.DataFrame | None = None        # every whole-day check's findings of the run, TimeStamp = the check's midnight
    checked: pd.Timestamp | None = None         # the latest midnight checked
    held_back: dict = field(default_factory=dict)   # signal -> events stamped at or after now + clock_ahead_minutes, left out

    def __iter__(self):
        return iter((self.findings, self.checked))


def _ts(t) -> str:
    return f"TIMESTAMP '{pd.Timestamp(t)}'"


def _conf(x, lo, hi):
    return f"(CASE WHEN ({x}) > 0 AND ({lo}) > 0 THEN 1.0 / (1.0 + exp(-ln(4.0) * ln(({x}) / ({lo})) / ln(greatest(({hi}) / ({lo}), 1.05)))) ELSE 0.0 END)"


def _read_path(con, path, name):
    """a view of the parquet or CSV files at a path, folder or glob, CSV as text so ids and stamps keep their form; None when no file matches."""
    if os.path.isdir(path):
        kinds = [k for k in ("parquet", "csv") if con.execute("SELECT count(*) FROM glob(?)", [os.path.join(path, f"*.{k}")]).fetchall()[0][0]]
        path = os.path.join(path, f"*.{kinds[0] if kinds else 'parquet'}")
    if not con.execute("SELECT count(*) FROM glob(?)", [path]).fetchall()[0][0]:
        return None
    text = path.replace("'", "''")
    reader = f"read_csv('{text}', union_by_name=true, all_varchar=true)" if path.lower().endswith(".csv") else f"read_parquet('{text}', union_by_name=true)"
    con.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {reader}")
    return name


def _source(con, src, name):
    """a state or history given as DataFrame, Arrow, path or function: a relation name, or None when there is nothing."""
    if callable(src):
        src = src()
    if src is None or (isinstance(src, str) and not src):
        return None
    if isinstance(src, str):
        return _read_path(con, src, name)
    if hasattr(src, "__len__") and len(src) == 0:
        return None
    con.register(name, src)
    return name


def _load_state(con, state, device_type):
    """the state into the store and the hl_ tables; returns each signal's markers and watermark (DataFrame DeviceId, b, c, f, wm).
    A state from before 2.2.0 (markers with no DeviceId) is converted: each of its signals takes the shared markers."""
    src = _source(con, state, "hl_state_in")
    if src is None and isinstance(state, str) and state:
        raise FileNotFoundError(NO_STATE.format(state))
    cols = f"TimeStamp TIMESTAMP, DeviceId {device_type}, EventId INTEGER, Parameter INTEGER, Value DOUBLE"
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_s ({cols})")
    if src:
        con.execute(f"INSERT INTO hl_s SELECT TimeStamp::TIMESTAMP, DeviceId::{device_type}, EventId::INTEGER, Parameter::INTEGER, Value::DOUBLE FROM {src}")
        version = con.execute(f"SELECT max(Value) FROM hl_s WHERE EventId = {VERSION}").fetchall()[0][0]
        if version is not None and version != float(I.EXTRACTOR_VERSION):
            raise ValueError(f"detector_health state was written by extractor version {version:g}, this package has {I.EXTRACTOR_VERSION}. Pass state '' to start again and delete the old rows.")
        # before 2.2.0: one set of markers and run list for all signals, with no DeviceId
        legacy = con.execute(f"SELECT count(*) FROM hl_s WHERE DeviceId IS NULL AND EventId IN ({BINS_THROUGH}, {LAST_CHECK}, {FIRST_DAY}, {RUN_ID})").fetchall()[0][0]
        if legacy:
            con.execute(f"""INSERT INTO hl_s SELECT g.TimeStamp, d.DeviceId, g.EventId, CASE WHEN g.EventId = {RUN_ID} THEN 1 ELSE 0 END, g.Value
                            FROM (SELECT * FROM hl_s WHERE DeviceId IS NULL AND EventId IN ({BINS_THROUGH}, {LAST_CHECK}, {FIRST_DAY}, {RUN_ID})) g
                            CROSS JOIN (SELECT DISTINCT DeviceId FROM hl_s WHERE DeviceId IS NOT NULL) d
                            WHERE g.EventId = {RUN_ID} OR d.DeviceId NOT IN (SELECT DeviceId FROM hl_s WHERE EventId = g.EventId AND DeviceId IS NOT NULL)""")
        con.execute("DELETE FROM hl_s WHERE DeviceId IS NULL")
    s = f"{STORE}."
    con.execute(f"INSERT INTO {s}health_carry SELECT TimeStamp, DeviceId, EventId::SMALLINT, Parameter, true FROM hl_s WHERE EventId BETWEEN 9200 AND 9232 OR EventId BETWEEN 9300 AND 9403")
    con.execute(f"INSERT INTO {s}health_detector_phase SELECT DeviceId, Parameter::SMALLINT, Value::INTEGER, TimeStamp FROM hl_s WHERE EventId = 9404")
    wide = ", ".join(f"max(Value) FILTER (EventId = {c}) AS {k}" for k, c in LIMIT_CODES.items())
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_limits AS SELECT DeviceId, Parameter AS det, {wide} FROM hl_s WHERE EventId BETWEEN 9501 AND 9508 GROUP BY DeviceId, Parameter")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_pon AS SELECT DeviceId, Parameter AS ch, TimeStamp AS t_on, max(Value) FILTER (EventId = {PON_WALKS}) AS walks, max(Value) FILTER (EventId = {PON_GREENS}) AS greens
                    FROM hl_s WHERE EventId IN ({PON_WALKS}, {PON_GREENS}) GROUP BY ALL""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_inputs AS SELECT DeviceId, Parameter AS ch, max(Value) FILTER (EventId = {INPUT_PHASE}) AS ph, max(Value) FILTER (EventId = {INPUT_FLAGS}) AS flags
                    FROM hl_s WHERE EventId IN ({INPUT_PHASE}, {INPUT_FLAGS}) GROUP BY ALL""")
    # the runs each signal has rows of in history; Legacy: listed by a state from before 2.2.0 for all its signals, checked by run alone
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_runs AS SELECT DISTINCT TimeStamp, DeviceId, Value::BIGINT AS Run, Parameter = 1 AS Legacy FROM hl_s WHERE EventId = {RUN_ID}")
    return con.execute(f"""SELECT DeviceId, max(TimeStamp) FILTER (EventId = {BINS_THROUGH}) AS b, max(TimeStamp) FILTER (EventId = {LAST_CHECK}) AS c,
                                  max(TimeStamp) FILTER (EventId = {FIRST_DAY}) AS f, max(TimeStamp) FILTER (EventId IN (9401, 9402)) AS wm
                           FROM hl_s GROUP BY DeviceId""").df()


def _live_step(con, T, detectors):
    """the rows of the bin ending at T of the signals in hl_due, from the rules that can answer now, and the open ped ONs for the next bin."""
    a = T - Q
    A, Z = _ts(a), _ts(T)
    s = f"{STORE}."
    has = con.execute("SELECT count(*) FROM hl_limits").fetchall()[0][0] > 0
    parts = []
    if has:
        # stuck on: a chain still open at the bin's end, or closed in it, held for its limit so far.  Not over a comms gap, and the signal logged in the bin.
        x = f"(epoch(least(h.f, {Z})) - epoch(h.s))"
        c = _conf(x, "l.lim_stuck", "l.bad_stuck")
        parts.append(f"""SELECT h.DeviceId, 'Vehicle' AS DetectorType, h.det::BIGINT AS Detector, 'stuck on' AS Label,
               least(1.0, sum((epoch(least(h.f, {Z})) - epoch(greatest(h.s, {A}))) / 900.0 * least(1.0, {c} / 0.65))) AS Score
            FROM (WITH hraw AS (SELECT DeviceId, Parameter AS det, TimeStamp AS t, EventId FROM {s}health_carry WHERE EventId IN (9201, 9202, 9222)),
                  hoff AS (SELECT DeviceId, det, max(t) FILTER (EventId = 9201) AS off FROM hraw GROUP BY ALL HAVING count(*) FILTER (EventId = 9222) = 1),
                  vopen AS (SELECT r.DeviceId, r.det, min(r.t) AS t0 FROM hraw r JOIN hoff o ON o.DeviceId = r.DeviceId AND o.det = r.det WHERE r.EventId IN (9202, 9222) AND (o.off IS NULL OR r.t > o.off) GROUP BY ALL),
                  lastev AS (SELECT DeviceId, max(TimeStamp) AS z FROM {s}health_carry WHERE EventId = 9402 GROUP BY ALL),
                  mk AS (SELECT DeviceId, max(TimeStamp) AS mk FROM {s}health_carry WHERE EventId = 9403 GROUP BY ALL)
                  SELECT v.DeviceId, v.det, v.t0::TIMESTAMP AS s, {Z} AS f FROM vopen v JOIN lastev l USING (DeviceId) LEFT JOIN mk USING (DeviceId)
                  WHERE epoch({Z}) - epoch(l.z) <= 120 AND NOT coalesce(epoch({Z}) - epoch(v.t0) > 60 AND mk.mk >= v.t0, false)
                  UNION ALL SELECT DeviceId, Channel, StartTime::TIMESTAMP, EndTime::TIMESTAMP FROM {s}health_intervals WHERE Kind = 'long_on' AND EndTime >= {A} AND EndTime < {Z}) h
            JOIN (SELECT DISTINCT DeviceId FROM {s}health_signal_5min WHERE TimeStamp >= {A} AND TimeStamp < {Z} AND Events > 0) USING (DeviceId)
            JOIN hl_limits l ON l.DeviceId = h.DeviceId AND l.det = h.det
            WHERE {x} >= l.lim_stuck AND least(h.f, {Z}) > greatest(h.s, {A}) GROUP BY ALL""")
        # too many: the bin's busiest 5 minutes against the limit
        parts.append(f"""SELECT DeviceId, 'Vehicle', det::BIGINT, 'too many for the traffic', count(*) / 3.0 FROM (
              SELECT r.DeviceId, r.det, r.b, l.lim_volume, CASE WHEN f.function = 'Advance' THEN r.nst * 12.0 / f.ln1 WHEN p.sgy >= 60 THEN r.ngy / p.sgy * 3600.0 / f.ln1 END AS q5
              FROM (SELECT DeviceId, Detector::INT AS det, TimeStamp::TIMESTAMP AS b, sum(coalesce(Starts, 0)) AS nst, sum(coalesce(Starts_GY, 0)) AS ngy FROM {s}health_detector_5min WHERE TimeStamp >= {A} AND TimeStamp < {Z} GROUP BY ALL) r
              JOIN (SELECT DeviceId, Detector, function, phase, greatest(coalesce(lanes, 1), 1) AS ln1 FROM {detectors}) f ON f.DeviceId = r.DeviceId AND f.Detector = r.det
              JOIN hl_limits l ON l.DeviceId = r.DeviceId AND l.det = r.det
              LEFT JOIN (SELECT DeviceId, Phase, TimeStamp::TIMESTAMP AS b, sum(Green_Yellow_ns) / 1e9 AS sgy FROM {s}health_phase_5min WHERE TimeStamp >= {A} AND TimeStamp < {Z} GROUP BY ALL) p ON p.DeviceId = r.DeviceId AND p.Phase = f.phase AND p.b = r.b)
            WHERE q5 > lim_volume GROUP BY ALL""")
    # ped: each ON with its walks and greens so far.  A signal with no ped event in the bin keeps its open ONs as they are.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_pev AS SELECT DeviceId, TimeStamp::TIMESTAMP AS t, EventId, Parameter FROM {s}health_ped_events WHERE TimeStamp >= {A} AND TimeStamp < {Z} AND EventId IN (1, 21, 89, 90)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_p AS
        WITH e AS (SELECT DeviceId, Parameter AS ch, t, EventId FROM hl_pev WHERE EventId IN (89, 90)
                   UNION ALL SELECT DeviceId, ch, t_on, 90 FROM hl_pon),
        x AS (SELECT *, lead(EventId) OVER w AS ne, lead(t) OVER w AS nt FROM e WINDOW w AS (PARTITION BY DeviceId, ch ORDER BY t, EventId))
        SELECT x.DeviceId, x.ch, x.t AS t_on, CASE WHEN x.ne = 89 THEN x.nt END AS t_off, x.ne IS NULL AS opn, i.ph, coalesce(i.flags, 0)::INT AS flags, coalesce(c.walks, 0) AS w0, coalesce(c.greens, 0) AS g0
        FROM x LEFT JOIN hl_inputs i ON i.DeviceId = x.DeviceId AND i.ch = x.ch LEFT JOIN hl_pon c ON c.DeviceId = x.DeviceId AND c.ch = x.ch AND c.t_on = x.t
        WHERE x.EventId = 90 AND (x.ne IS NULL OR x.ne = 89)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_pc AS
        SELECT p.*, coalesce(p.t_off, {Z}) AS as_of, p.w0 + coalesce(c.nw, 0) AS walks, p.g0 + coalesce(c.ng, 0) AS greens, coalesce(c.nw, 0) AS walks_bin
        FROM hl_p p LEFT JOIN (SELECT p.DeviceId, p.ch, p.t_on, count(*) FILTER (w.EventId = 21) AS nw, count(*) FILTER (w.EventId = 1) AS ng
                               FROM hl_p p JOIN hl_pev w ON w.DeviceId = p.DeviceId AND w.Parameter = p.ph AND w.EventId IN (1, 21) AND w.t > p.t_on AND w.t < coalesce(p.t_off, {Z})
                               GROUP BY ALL) c ON c.DeviceId = p.DeviceId AND c.ch = p.ch AND c.t_on = p.t_on""")
    if has:
        held = "(epoch(as_of) - epoch(t_on))"
        parts.append(f"""SELECT DeviceId, 'Ped', ch::BIGINT,
               CASE WHEN NOT opn AND (flags & 1) = 0 AND {held} >= 150 AND {held} < 180 AND walks >= 2 THEN 'long hold' ELSE 'stuck on' END AS lab,
               least(1.0, sum((epoch(as_of) - epoch(greatest(t_on, {A}))) / 900.0))
            FROM hl_pc WHERE ph IS NOT NULL AND as_of > {A} AND CASE WHEN NOT opn THEN CASE WHEN (flags & 1) = 1 THEN {held} >= 600 ELSE {held} >= 300 OR walks >= 3 OR ({held} >= 150 AND walks >= 2) END
                               ELSE (flags & 2) = 0 AND walks_bin > 0 AND walks >= 3 AND walks >= 0.8 * greens END GROUP BY ALL""")
    con.execute("CREATE OR REPLACE TEMP TABLE hl_pon AS SELECT DeviceId, ch, t_on, walks::DOUBLE AS walks, greens::DOUBLE AS greens FROM hl_pc WHERE opn")
    if not parts:
        return B.empty_bins()
    rows = con.execute("SELECT * FROM (" + " UNION ALL ".join(f"SELECT * FROM ({p})" for p in parts) + ") u WHERE u.DeviceId IN (SELECT DeviceId FROM hl_due)").df()
    rows.columns = ["DeviceId", "DetectorType", "Detector", "Label", "Score"]
    if rows.empty:
        return B.empty_bins()
    # a bin's rows add up to at most 1
    tot = rows.groupby(["DeviceId", "DetectorType", "Detector"]).Score.transform("sum")
    rows["Score"] = (rows.Score / tot.clip(lower=1)).round(6)
    rows["TimeStamp"] = a
    rows["Detector"] = rows.Detector.astype("Int64")
    return rows[B.BIN_COLS].sort_values(["DeviceId", "DetectorType", "Detector", "Label"]).reset_index(drop=True)


def _typed_history(con, history, cut, device_type):
    """the saved rows in the listed types, whatever they went through.  A run listed for a checked signal whose rows of that signal are not
    there raises: the check would write wrong rows that can never be fixed.  A run listed by a state from before 2.2.0 needs its rows of any signal."""
    hist = _source(con, history, "hl_hist")
    listed = con.execute(f"""SELECT DISTINCT r.DeviceId, r.Run, r.Legacy FROM hl_runs r SEMI JOIN hl_chk c USING (DeviceId) WHERE r.TimeStamp > {_ts(cut)}""").fetchall()
    found_dev, found_run = set(), set()
    if hist and listed:
        con.execute(f"CREATE OR REPLACE TEMP TABLE hl_found AS SELECT DISTINCT DeviceId::{device_type} AS DeviceId, Run::BIGINT AS Run FROM {hist}")
        found_dev = set(con.execute("SELECT DeviceId, Run FROM hl_found").fetchall())
        found_run = {r for _, r in found_dev}
    missing = sorted({r for d, r, leg in listed if ((r not in found_run) if leg else ((d, r) not in found_dev))})
    if missing:
        runs = {r for _, r, _ in listed}
        raise ValueError(f"detector_health history {history!r} lacks the rows of {len(missing)} of the {len(runs)} runs the state lists (first: run {missing[0]}). Restore them, or pass state '' to start again.")
    if not hist:
        return None
    have = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {hist}").fetchall()}
    cols = ", ".join((f'"{k}"::{t or device_type}' if k in have else f"NULL::{t or device_type}") + f' AS "{k}"' for k, t in ROW_COLUMNS.items())
    # only the checked signals, and only rows of runs their state lists
    con.execute(f"""CREATE OR REPLACE TEMP VIEW hl_hist_t AS SELECT * FROM (SELECT {cols} FROM {hist}) h
                    WHERE h.DeviceId IN (SELECT DeviceId FROM hl_chk)
                      AND (h.DeviceId, h.Run) IN (SELECT DeviceId, Run FROM hl_runs)""")
    return "hl_hist_t"


def _ped_history(con, ped_history, M, params, device_type):
    """(counts, logged) for the ped option stop_history: the saved ped rows and this run's, of the checked signals, in the ped_history_days
    before the check window, each key once (the largest count: a retried run's rows count once)."""
    days = params["ped_history_days"]
    if not params["ped"] or not days:
        return None
    lo, hi = M - pd.Timedelta(days=params["horizon_days"]) - pd.Timedelta(days=days), M - pd.Timedelta(days=params["horizon_days"])
    src = _source(con, ped_history, "hl_ped_hist")
    parts = ["SELECT TimeStamp, DeviceId, Input, Presses FROM hl_ped_out"]
    if src:
        parts.append(f"SELECT TimeStamp::TIMESTAMP, DeviceId::{device_type}, Input::INTEGER, Presses::INTEGER FROM {src}")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_ph AS SELECT TimeStamp, DeviceId, Input, max(Presses) AS Presses FROM ({' UNION ALL '.join(parts)}) p
                    WHERE TimeStamp >= {_ts(lo)} AND TimeStamp < {_ts(hi)} AND DeviceId IN (SELECT DeviceId FROM hl_chk) GROUP BY ALL""")
    con.execute("CREATE OR REPLACE TEMP VIEW hl_ph_counts AS SELECT DeviceId, Input, TimeStamp, Presses AS presses FROM hl_ph WHERE Input IS NOT NULL")
    con.execute("CREATE OR REPLACE TEMP VIEW hl_ph_logged AS SELECT DeviceId, TimeStamp FROM hl_ph WHERE Input IS NULL")
    return ("hl_ph_counts", "hl_ph_logged")


def _whole_day(con, M, detectors, history, ped_history, params, temp_dir, devices):
    """the whole-day check at midnight M of these signals: (late rows, findings with TimeStamp M)."""
    device_type = I._meta(con, STORE)["device_type"]
    con.register("hl_chk_in", pd.DataFrame({"DeviceId": list(devices)}))
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_chk AS SELECT DISTINCT DeviceId::{device_type} AS DeviceId FROM hl_chk_in")
    con.unregister("hl_chk_in")
    cut = M - pd.Timedelta(days=params["horizon_days"])
    hist = _typed_history(con, history, cut, device_type)
    ped_hist = _ped_history(con, ped_history, M, params, device_type)
    path = os.path.join(temp_dir, f"check_{os.getpid()}.duckdb")
    for f in (path, path + ".wal"):
        if os.path.exists(f):
            os.remove(f)
    I.attach_store(con, path, name=CHECK, horizon_days=params["horizon_days"], ped=params["ped"])
    c = f"{CHECK}."
    mine = "DeviceId IN (SELECT DeviceId FROM hl_chk)"
    try:
        con.execute(_render.render("store.sql", device_type=device_type, s=c))
        I._set_meta(con, CHECK, device_type=device_type)
        for t in ROW_TABLES:
            cols = [r[0] for r in con.execute(f"DESCRIBE {STORE}.{t}").fetchall()]
            tc = "EndTime" if t == "health_intervals" else "TimeStamp"
            sel = ", ".join(f'"{x}"' for x in cols)
            if hist:
                con.execute(f"INSERT INTO {c}{t} BY NAME SELECT {sel} FROM {hist} WHERE Tbl = '{t}' AND {tc} >= {_ts(cut)}")
            con.execute(f"INSERT INTO {c}{t} BY NAME SELECT {sel} FROM {STORE}.{t} WHERE {tc} >= {_ts(cut)} AND {mine}")
        con.execute(f"INSERT INTO {c}health_carry SELECT * FROM {STORE}.health_carry WHERE {mine}")
        con.execute(f"INSERT INTO {c}health_detector_phase SELECT * FROM {STORE}.health_detector_phase WHERE {mine}")
        given = B.empty_bins()
        known = M - DAY
        parts = [con.execute(f"SELECT TimeStamp, DeviceId, DetectorType, Detector, Label, Score FROM hl_released WHERE TimeStamp >= {_ts(known)} AND TimeStamp < {_ts(M)} AND {mine}").df()]
        if hist:
            parts.append(con.execute(f"SELECT TimeStamp, DeviceId, DetectorType, Detector, Label, Score FROM {hist} WHERE Tbl = '{RELEASED}' AND TimeStamp >= {_ts(known)} AND TimeStamp < {_ts(M)}").df())
        parts = [p for p in parts if len(p)]
        if parts:
            given = pd.concat(parts, ignore_index=True)[B.BIN_COLS]
        r = I.judge_rows(con, detectors, store=CHECK, end=M, options=params["options"], ped=params["ped"], ped_options=params["ped_options"], ped_assign=params["ped_assign"],
                         workers=params["workers"], limits=True, ped_history=ped_hist)
        f, s = F.tables(con, r.detectors, r.periods, r.ped, r.ped_periods, r.windows, r.ped_windows, summary=True)
        spans, add, drop = B.held(con, CHECK, f, s, r.ped, known, M, M, params["ped"])
        late = B.late_rows(B.bins(con, f, s, known, M, 15, spans), given, known)
        for t, d, code, par in drop:
            con.execute(f"DELETE FROM {STORE}.health_carry WHERE TimeStamp = ? AND DeviceId = ? AND EventId = ? AND Parameter = ?", [t, d, code, par])
        for t, d, code, par in add:
            con.execute(f"INSERT INTO {STORE}.health_carry VALUES (?, ?, ?, ?, true)", [t, d, code, par])
        # the limits and ped inputs of the checked signals are replaced; the others' stay
        if r.limits is not None and len(r.limits):
            con.register("hl_lim_in", r.limits)
            con.execute(f"DELETE FROM hl_limits WHERE {mine}")
            con.execute("INSERT INTO hl_limits SELECT DeviceId, det::INT, lim_stuck, bad_stuck, lim_rapid, bad_rapid, lim_volume, bad_volume FROM hl_lim_in")
            con.unregister("hl_lim_in")
        if r.ped is not None and len(r.ped):
            p = r.ped[["DeviceId", "Input", "Phase", "Type", "config"]].copy()
            con.register("hl_ped_in", p)
            con.execute(f"DELETE FROM hl_inputs WHERE {mine}")
            con.execute("""INSERT INTO hl_inputs SELECT DeviceId, Input::INT AS ch, Phase::DOUBLE AS ph,
                           ((Type = 'presence')::INT + 2 * (coalesce(config, '') LIKE '%on ped recall%')::INT)::DOUBLE AS flags FROM hl_ped_in""")
            con.unregister("hl_ped_in")
        f = f.copy()
        f.insert(0, "TimeStamp", M)
        return late, f
    finally:
        I.detach_store(con, CHECK)
        for g in (path, path + ".wal"):
            if os.path.exists(g):
                os.remove(g)


def live_run(con, raw, detectors, *, state=None, history=None, ped_history=None, horizon_days=3, ped=True, options=None, ped_options=None, ped_assign=False,
             workers=1, clock_ahead_minutes=60, now=None, complete_through=None, ped_history_days=PED_HISTORY_DAYS, temp_dir=None) -> LiveResult:
    """One run: tables hl_final, hl_state, hl_rows and hl_ped on con (module docstring).  raw: this run's events (DeviceId, TimeStamp, EventId,
    Parameter); detectors: as health().  state: the last run's hl_state (DataFrame, path or function; '' for a first run).  history: saved hl_rows
    (folder, glob, DataFrame or function); ped_history: saved hl_ped, the same way.  now: the caller's time on the controllers' clock (events at or
    after now + clock_ahead_minutes are left out; a midnight is over for every signal once now is clock_ahead_minutes past it).
    complete_through: a time, or a table (DeviceId, Through): each signal's events are all in up to then, so its bins and midnights up to then
    close without waiting for its next event.  ped_history_days: days of ped press counts kept before the check window (0: none)."""
    params = dict(horizon_days=horizon_days, ped=ped, options=options, ped_options=ped_options, ped_assign=ped_assign, workers=workers,
                  ped_history_days=ped_history_days if ped else 0)
    temp_dir = temp_dir or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    device_type = next(r[1] for r in con.execute(f"DESCRIBE SELECT DeviceId FROM {raw}").fetchall())
    try:
        return _live_run(con, raw, detectors, state, history, ped_history, params, clock_ahead_minutes, now, complete_through, temp_dir, device_type)
    finally:
        _clean(con)


def _clean(con):
    """nothing of a run but its four outputs stays on the connection, as save() writes every table it can see."""
    if con.execute("SELECT count(*) FROM duckdb_databases() WHERE database_name = ?", [STORE]).fetchall()[0][0]:
        I.detach_store(con, STORE)
    for kind, name in con.execute("""SELECT 'TABLE', table_name FROM duckdb_tables() WHERE starts_with(table_name, 'hl_')
                                     UNION ALL SELECT 'VIEW', view_name FROM duckdb_views() WHERE starts_with(view_name, 'hl_')""").fetchall():
        if name not in ("hl_final", "hl_state", "hl_rows", "hl_ped"):
            try:
                con.execute(f"DROP {kind} IF EXISTS {name}")
            except Exception:       # a frame given as state or history
                con.unregister(name)


def _complete(con, complete_through, device_type) -> dict:
    """signal -> the time its events are all in up to (complete_through as a time: '*' for every signal)."""
    if complete_through is None:
        return {}
    if isinstance(complete_through, (str, pd.Timestamp)) or hasattr(complete_through, "year"):
        return {"*": pd.Timestamp(complete_through)}
    src = _source(con, complete_through, "hl_complete_in")
    if src is None:
        return {}
    return {d: pd.Timestamp(t) for d, t in con.execute(f"SELECT DeviceId::{device_type}, max(Through::TIMESTAMP) FROM {src} GROUP BY 1").fetchall() if t is not None}


def _plan(devs, ebins, complete, now, clock_ahead, horizon):
    """per signal: the quarters to step (those with its events up to where it is processed to), the quarters to write, the midnights it
    crosses and which of them it is checked at, and where it is processed to (p) -- from its own data and markers alone."""
    ahead = pd.Timedelta(minutes=clock_ahead)
    plan = {}
    for d, m in devs.items():
        e = ebins.get(d, [])
        reach = (m["last"] + SECOND).floor("15min") if m["last"] is not None else None
        done = complete.get(d, complete.get("*"))
        if done is not None:
            cap = done.floor("15min") if now is None else min(done, now + ahead).floor("15min")
            reach = cap if reach is None else max(reach, cap)
        b, c = m["b"], m["c"]
        reach = b if reach is None else reach
        last_mid = reach.floor("D")
        if now is not None:
            last_mid = max(last_mid, (now - ahead).floor("D"))
        mids = list(pd.date_range(c + DAY, last_mid, freq="D")) if last_mid > c else []
        p = max([b, reach] + mids)
        steps = [t for t in e if t <= p]
        due = {t for t in steps if b < t <= reach}
        wm = m["wm"]
        checks = [M for M in mids if m["f"] <= M - DAY and ((wm is not None and wm >= M - horizon) or any(M - horizon < t <= M for t in e))]
        plan[d] = dict(steps=set(steps), due=due, mids=set(mids), checks=set(checks), p=p, c=max([c] + mids))
    return plan


def _live_run(con, raw, detectors, state, history, ped_history, params, clock_ahead_minutes, now, complete_through, temp_dir, device_type):
    horizon_days, ped, options = params["horizon_days"], params["ped"], params["options"]
    horizon = pd.Timedelta(days=horizon_days)
    I.attach_store(con, None, name=STORE, horizon_days=horizon_days, ped=ped)
    I._set_meta(con, STORE, device_type=device_type, per_device=True)
    con.execute(_render.render("store.sql", device_type=device_type, s=f"{STORE}."))
    marks = _load_state(con, state, device_type)
    now = None if now is None else pd.Timestamp(now)
    out = LiveResult()
    bound = "" if now is None else f" WHERE TimeStamp < {_ts(now + pd.Timedelta(minutes=clock_ahead_minutes))}"
    if now is not None:
        out.held_back = {d: int(n) for d, n in con.execute(
            f"SELECT DeviceId, count(*) FROM {raw} WHERE TimeStamp >= {_ts(now + pd.Timedelta(minutes=clock_ahead_minutes))} GROUP BY 1").fetchall()}
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_ev AS SELECT * FROM (SELECT DeviceId::{device_type} AS DeviceId, TimeStamp AS Timestamp, EventId, Parameter FROM {raw}
                    UNION ALL SELECT DeviceId, TimeStamp, Value::INT, Parameter FROM hl_s WHERE EventId = {PEND}){bound.replace('TimeStamp', 'Timestamp')}""")
    for name, cols in (("hl_released", OUTPUT), ("hl_out", OUTPUT), ("hl_ped_out", PED_OUTPUT)):
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} ({cols.format(d=device_type)})")
    # each signal: its markers, its first and last event and the quarters holding its events
    seen = con.execute("SELECT DeviceId, min(Timestamp), max(Timestamp) FROM hl_ev GROUP BY 1").fetchall()
    ebins = {}
    for d, t in con.execute("SELECT DISTINCT DeviceId, time_bucket(INTERVAL 15 MINUTE, Timestamp::TIMESTAMP) + INTERVAL 15 MINUTE FROM hl_ev").fetchall():
        ebins.setdefault(d, []).append(pd.Timestamp(t))
    devs = {}
    for r in marks.itertuples(index=False):
        if r.b is not None and not pd.isna(r.b):
            devs[r.DeviceId] = dict(b=pd.Timestamp(r.b), c=pd.Timestamp(r.c), f=pd.Timestamp(r.f), wm=None if pd.isna(r.wm) else pd.Timestamp(r.wm), last=None)
    for d, a, z in seen:
        if d not in devs:       # a signal seen for the first time starts at its first event
            a = pd.Timestamp(a)
            devs[d] = dict(b=a.floor("15min"), c=a.floor("D"), f=a.floor("15min"), wm=None, last=None)
        devs[d]["last"] = pd.Timestamp(z)
    plan = _plan(devs, ebins, _complete(con, complete_through, device_type), now, clock_ahead_minutes, horizon)
    run_id = max(int(time.time() * 1000), 1 + int(con.execute("SELECT coalesce(max(Run), 0) FROM hl_runs").fetchall()[0][0]))
    con.register("hl_lim_p", pd.DataFrame({"DeviceId": list(plan), "p": [v["p"] for v in plan.values()]}))
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_lim AS SELECT DeviceId::{device_type} AS DeviceId, p::TIMESTAMP AS p FROM hl_lim_p")
    con.unregister("hl_lim_p")
    after = None
    found = []

    def update(stop):
        """this signal's events before stop and before where it is processed to, from the last stop on; then their ped history rows."""
        lo = "" if after is None else f" AND e.Timestamp >= {_ts(after)}"
        hi = "" if stop is None else f" AND e.Timestamp < {_ts(stop)}"
        con.execute(f"CREATE OR REPLACE TEMP VIEW hl_part AS SELECT e.* FROM hl_ev e JOIN hl_lim l USING (DeviceId) WHERE e.Timestamp < l.p{hi}{lo}")
        I.update(con, "hl_part", detectors, store=STORE, upto=stop, options=options, ped=ped)
        if params["ped_history_days"]:
            w = " AND ".join(x for x in ("" if after is None else f"TimeStamp >= {_ts(after)}", "" if stop is None else f"TimeStamp < {_ts(stop)}") if x) or "true"
            con.execute(f"""INSERT INTO hl_ped_out
                SELECT time_bucket(INTERVAL 15 MINUTE, TimeStamp::TIMESTAMP), DeviceId, Parameter, count(DISTINCT TimeStamp) FROM {STORE}.health_ped_events WHERE EventId = 90 AND {w} GROUP BY ALL
                UNION ALL SELECT DISTINCT TimeStamp::TIMESTAMP, DeviceId, NULL, NULL FROM {STORE}.health_log_15min WHERE {w}""")

    def write(rows):
        if len(rows):
            con.register("hl_rows_in", rows)
            con.execute("INSERT INTO hl_out BY NAME SELECT * FROM hl_rows_in")
            con.execute("INSERT INTO hl_released BY NAME SELECT * FROM hl_rows_in")
            con.unregister("hl_rows_in")

    def due_table(ds):
        con.register("hl_due_in", pd.DataFrame({"DeviceId": ds}))
        con.execute(f"CREATE OR REPLACE TEMP TABLE hl_due AS SELECT DeviceId::{device_type} AS DeviceId FROM hl_due_in")
        con.unregister("hl_due_in")

    stops = sorted(set().union(*[v["steps"] | v["mids"] for v in plan.values()])) if plan else []
    for stop in stops:
        update(stop)
        after = stop
        stepping = [d for d, v in plan.items() if stop in v["steps"]]
        if stepping:        # every signal with events in the bin steps its open ped ONs; only those whose bin it is get rows
            due_table([d for d in stepping if stop in plan[d]["due"]])
            write(_live_step(con, stop, detectors))
        checked = [d for d, v in plan.items() if stop in v["checks"]]
        if checked:
            late, f = _whole_day(con, stop, detectors, history, ped_history, params, temp_dir, checked)
            write(late)
            found.append(f)
            out.checked = stop
    last_p = max((v["p"] for v in plan.values()), default=None)
    if last_p is not None and (after is None or after < last_p):       # events behind a signal's last bin: into the store, no bins
        update(last_p)
    if found:
        out.findings = pd.concat(found, ignore_index=True)
    s = f"{STORE}."
    union = " UNION ALL BY NAME ".join(f"SELECT '{t}' AS Tbl, * FROM {s}{t}" for t in ROW_TABLES)
    typed = ", ".join(f'"{k}"::{t or device_type} AS "{k}"' for k, t in ROW_COLUMNS.items() if k != "Run")
    con.execute(f"""CREATE OR REPLACE TABLE hl_rows AS SELECT {typed}, {run_id}::BIGINT AS "Run" FROM (
        {union} UNION ALL BY NAME SELECT '{RELEASED}' AS Tbl, TimeStamp, DeviceId, DetectorType, Detector, Label, Score FROM hl_released)""")
    # the markers, the run listed for each signal with rows in it (a run with no rows of a signal has nothing its check must find), and runs past need dropped
    mk = pd.DataFrame({"DeviceId": list(plan), "b": [v["p"] for v in plan.values()], "c": [v["c"] for v in plan.values()], "f": [devs[d]["f"] for d in plan]})
    con.register("hl_mk_in", mk)
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_mk AS SELECT DeviceId::{device_type} AS DeviceId, b::TIMESTAMP AS b, c::TIMESTAMP AS c, f::TIMESTAMP AS f FROM hl_mk_in")
    con.unregister("hl_mk_in")
    con.execute(f"INSERT INTO hl_runs SELECT m.b, m.DeviceId, {run_id}, false FROM hl_mk m WHERE m.DeviceId IN (SELECT DISTINCT DeviceId FROM hl_rows)")
    con.execute(f"DELETE FROM hl_runs USING hl_mk m WHERE hl_runs.DeviceId = m.DeviceId AND hl_runs.TimeStamp < m.b - INTERVAL ({horizon_days + 1}) DAY")
    con.execute("CREATE OR REPLACE TABLE hl_ped AS SELECT * FROM hl_ped_out ORDER BY TimeStamp, DeviceId, Input NULLS FIRST")
    long_lim = " UNION ALL ".join(f"SELECT NULL::TIMESTAMP, DeviceId, {c}, det, {k}::DOUBLE FROM hl_limits" for k, c in LIMIT_CODES.items())
    con.execute(f"""CREATE OR REPLACE TABLE hl_state AS
        SELECT TimeStamp::TIMESTAMP AS TimeStamp, DeviceId, EventId::INTEGER AS EventId, Parameter::INTEGER AS Parameter, NULL::DOUBLE AS Value FROM {s}health_carry
        UNION ALL SELECT ValidFrom::TIMESTAMP, DeviceId, 9404, Detector::INTEGER, Phase::DOUBLE FROM {s}health_detector_phase
        UNION ALL SELECT * FROM ({long_lim})
        UNION ALL SELECT t_on, DeviceId, {PON_WALKS}, ch, walks FROM hl_pon
        UNION ALL SELECT t_on, DeviceId, {PON_GREENS}, ch, greens FROM hl_pon
        UNION ALL SELECT NULL::TIMESTAMP, DeviceId, {INPUT_PHASE}, ch, ph FROM hl_inputs
        UNION ALL SELECT NULL::TIMESTAMP, DeviceId, {INPUT_FLAGS}, ch, flags FROM hl_inputs
        UNION ALL SELECT TimeStamp, DeviceId, {RUN_ID}, Legacy::INTEGER, Run::DOUBLE FROM hl_runs
        UNION ALL SELECT e.Timestamp::TIMESTAMP, e.DeviceId, {PEND}, e.Parameter, e.EventId::DOUBLE FROM hl_ev e JOIN hl_lim l USING (DeviceId) WHERE e.Timestamp >= l.p
        UNION ALL SELECT b, DeviceId, {BINS_THROUGH}, 0, NULL::DOUBLE FROM hl_mk
        UNION ALL SELECT c, DeviceId, {LAST_CHECK}, 0, NULL::DOUBLE FROM hl_mk
        UNION ALL SELECT f, DeviceId, {FIRST_DAY}, 0, NULL::DOUBLE FROM hl_mk
        UNION ALL SELECT NULL::TIMESTAMP, DeviceId, {VERSION}, 0, {float(I.EXTRACTOR_VERSION)}::DOUBLE FROM hl_mk""")
    con.execute("CREATE OR REPLACE TABLE hl_final AS SELECT * FROM hl_out ORDER BY TimeStamp, DeviceId, DetectorType DESC, Detector, Label")
    return out


def empty_run(con, state, raw="raw_data"):
    """a run with no events: tables hl_final, hl_rows and hl_ped with no rows, in all their columns, and hl_state as it was."""
    src = _source(con, state, "hl_state_in")
    if src is None and isinstance(state, str) and state:
        raise FileNotFoundError(NO_STATE.format(state))
    dtype = next(r[1] for r in con.execute(f"DESCRIBE SELECT DeviceId FROM {src or raw}").fetchall())
    if src:
        con.execute(f"CREATE OR REPLACE TABLE hl_state AS SELECT TimeStamp::TIMESTAMP AS TimeStamp, DeviceId, EventId::INTEGER AS EventId, Parameter::INTEGER AS Parameter, Value::DOUBLE AS Value FROM {src}")
    else:
        con.execute(f"CREATE OR REPLACE TABLE hl_state (TimeStamp TIMESTAMP, DeviceId {dtype}, EventId INTEGER, Parameter INTEGER, Value DOUBLE)")
    con.execute(f"CREATE OR REPLACE TABLE hl_final ({OUTPUT.format(d=dtype)})")
    con.execute(f"CREATE OR REPLACE TABLE hl_ped ({PED_OUTPUT.format(d=dtype)})")
    cols = ", ".join(f'"{k}" {t or dtype}' for k, t in ROW_COLUMNS.items())
    con.execute(f"CREATE OR REPLACE TABLE hl_rows ({cols})")
    _clean(con)
