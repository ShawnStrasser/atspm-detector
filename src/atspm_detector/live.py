"""Detector health for a run every 15 minutes, with state carried as two tables:

    live_run(con, "raw_data", "detectors", state=..., history=...)  -> tables hl_final (rows of detector_health), hl_state, hl_rows; returns (findings, time) of a whole-day check, or (None, None)

hl_state goes to the next run as state.  hl_rows are added to history, which only a whole-day check reads.  Each 15-minute bin the run reaches gets a set of rows from the rules that can answer at the bin's end, on what the bin brought and on the state (no history is read).
The first run past a midnight also runs the whole check on the last horizon_days and adds what it finds in the day that just ended: rows with past TimeStamps.
A midnight is past when 9 in 10 of the signals heard from in the day before it are past it, or once the run is clock_ahead_minutes past it; until then the run stops at that midnight.
A long run is the same steps in a row, so it writes what the same events cut into 15-minute runs write."""
from __future__ import annotations

import os
import tempfile
import time

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
LIMIT_CODES = {"lim_stuck": 9501, "bad_stuck": 9502, "lim_rapid": 9505, "bad_rapid": 9506, "lim_volume": 9507, "bad_volume": 9508, "fast_known": 9511}
BINS_THROUGH, LAST_CHECK, FIRST_DAY, TRIMMED, RUN_ID, VERSION, PEND = 9602, 9603, 9604, 9601, 9605, 9606, 9700
PON_WALKS, PON_GREENS, INPUT_PHASE, INPUT_FLAGS = 9233, 9234, 9520, 9521
TS9, I4, I8, I2, F8, TXT = "TIMESTAMP_NS", "INTEGER", "BIGINT", "SMALLINT", "DOUBLE", "VARCHAR"
ROW_COLUMNS = {"Tbl": TXT, "DeviceId": None, "TimeStamp": TS9, "Events": I4, "First_Event": TS9, "Last_Event": TS9, "Detector": I8, "Chain_Tag": TS9, "Piece_Phase": I4, "Piece_Tag": TS9, "Piece_Kind": I2,
               "Starts": I4, "Fast_Starts": I4, "Chatter_Starts": I4, "Starts_GY": I4, "Fast_Starts_R2": I4, "Chatter_Starts_R2": I4, "Relogged_Ons": I4, "Stray_Offs": I4, "Closing_Offs": I4,
               "On_Time_ns": I8, "Unknown_Time_ns": I8, "On_Green5_ns": I8, "Clean_Ons": I4, "Max_On_s": F8, "Relog_Time_ns": I8, "Ons_5s": I4, "Ons_60s": I4, "Long_Max_s": F8, "Phase": I4, "Greens": I4,
               "Green_Yellow_ns": I8, "Green5_ns": I8, "Kind": TXT, "A": I4, "B": I2, "Count": I4, "Channel": I4, "StartTime": TS9, "EndTime": TS9, "Relogged": "BOOLEAN", "Value": I4, "EventId": I2,
               "Parameter": I4, "All_First": TS9, "All_Last": TS9, "DetectorType": TXT, "Label": TXT, "Score": F8, "Run": I8}
OUTPUT = "TimeStamp TIMESTAMP, DeviceId {d}, DetectorType VARCHAR, Detector BIGINT, Label VARCHAR, Score DOUBLE"
NO_STATE = "detector_health state {!r} does not exist. Pass state '' only for a first run."


def _ts(t) -> str:
    return f"TIMESTAMP '{pd.Timestamp(t)}'"


def _conf(x, lo, hi):
    return f"(CASE WHEN ({x}) > 0 AND ({lo}) > 0 THEN 1.0 / (1.0 + exp(-ln(4.0) * ln(({x}) / ({lo})) / ln(greatest(({hi}) / ({lo}), 1.05)))) ELSE 0.0 END)"


def _sc(x, lo, hi):
    return f"(CASE WHEN ({x}) < ({lo}) THEN 0.0 ELSE least(0.35 + 0.40 * greatest((({x}) - ({lo})) / greatest(({hi}) - ({lo}), 1e-9), 0.0), 1.0) END)"


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
    s = f"{STORE}."
    con.execute(f"INSERT INTO {s}health_carry SELECT TimeStamp, DeviceId, EventId::SMALLINT, Parameter, true FROM hl_s WHERE EventId BETWEEN 9200 AND 9232 OR EventId BETWEEN 9300 AND 9403")
    con.execute(f"INSERT INTO {s}health_detector_phase SELECT DeviceId, Parameter::SMALLINT, Value::INTEGER, TimeStamp FROM hl_s WHERE EventId = 9404")
    wide = ", ".join(f"max(Value) FILTER (EventId = {c}) AS {k}" for k, c in LIMIT_CODES.items())
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_limits AS SELECT DeviceId, Parameter AS det, {wide} FROM hl_s WHERE EventId BETWEEN 9501 AND 9511 GROUP BY DeviceId, Parameter")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_pon AS SELECT DeviceId, Parameter AS ch, TimeStamp AS t_on, max(Value) FILTER (EventId = {PON_WALKS}) AS walks, max(Value) FILTER (EventId = {PON_GREENS}) AS greens
                    FROM hl_s WHERE EventId IN ({PON_WALKS}, {PON_GREENS}) GROUP BY ALL""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_inputs AS SELECT DeviceId, Parameter AS ch, max(Value) FILTER (EventId = {INPUT_PHASE}) AS ph, max(Value) FILTER (EventId = {INPUT_FLAGS}) AS flags
                    FROM hl_s WHERE EventId IN ({INPUT_PHASE}, {INPUT_FLAGS}) GROUP BY ALL""")
    m = {int(c): t for c, t in con.execute("SELECT EventId, TimeStamp FROM hl_s WHERE EventId IN (9601, 9602, 9603, 9604)").fetchall()}
    if TRIMMED in m:
        I._set_meta(con, STORE, trimmed_before=pd.Timestamp(m[TRIMMED]).isoformat())
    out = {k: pd.Timestamp(m[c]) if c in m else None for k, c in (("bins_through", BINS_THROUGH), ("last_check", LAST_CHECK), ("first_day", FIRST_DAY))}
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_runs AS SELECT TimeStamp, Value::BIGINT AS Run FROM hl_s WHERE EventId = {RUN_ID}")
    return out


def _live_step(con, T, detectors):
    """the rows of the bin ending at T, from the rules that can answer now, and the state for the next bin."""
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
        # too fast: a detector the last check rated too fast, at its limit in this bin
        vf = _sc("xr", "1.0", "bx")
        cf = _conf(vf, "0.35", "0.75")
        parts.append(f"""SELECT DeviceId, 'Vehicle', det::BIGINT, 'too-fast actuations', least(1.0, {cf} / 0.87) FROM (
              SELECT t.DeviceId, t.det, t.cf, 1000.0 * t.cf / greatest(o.ca, 1) / l.lim_rapid AS xr, l.bad_rapid / l.lim_rapid AS bx
              FROM (SELECT k.DeviceId, k.Detector::INT AS det, sum(k.Count) AS cf FROM {s}health_tick_counts k
                    JOIN (SELECT DeviceId, Detector, (CASE WHEN function IN ('Count', 'Presence', 'Yellow_Red') THEN 0.8 ELSE 0.6 END) / greatest(coalesce(lanes, 1), 1) AS gap_lim FROM {detectors}) f ON f.DeviceId = k.DeviceId AND f.Detector = k.Detector
                    WHERE k.Kind = 'gap_pair' AND k.TimeStamp = {A} AND k.A < f.gap_lim * 10 GROUP BY ALL) t
              JOIN (SELECT DeviceId, Detector::INT AS det, sum(coalesce(Starts, 0) + coalesce(Relogged_Ons, 0)) AS ca FROM {s}health_detector_5min WHERE TimeStamp >= {A} AND TimeStamp < {Z} GROUP BY ALL) o USING (DeviceId, det)
              JOIN hl_limits l USING (DeviceId, det) WHERE t.cf >= 3 AND l.fast_known = 1)
            WHERE {vf} >= 0.35""")
        # too many: the bin's busiest 5 minutes against the limit
        parts.append(f"""SELECT DeviceId, 'Vehicle', det::BIGINT, 'too many for the traffic', count(*) / 3.0 FROM (
              SELECT r.DeviceId, r.det, r.b, l.lim_volume, CASE WHEN f.function = 'Advance' THEN r.nst * 12.0 / f.ln1 WHEN p.sgy >= 60 THEN r.ngy / p.sgy * 3600.0 / f.ln1 END AS q5
              FROM (SELECT DeviceId, Detector::INT AS det, TimeStamp::TIMESTAMP AS b, sum(coalesce(Starts, 0)) AS nst, sum(coalesce(Starts_GY, 0)) AS ngy FROM {s}health_detector_5min WHERE TimeStamp >= {A} AND TimeStamp < {Z} GROUP BY ALL) r
              JOIN (SELECT DeviceId, Detector, function, phase, greatest(coalesce(lanes, 1), 1) AS ln1 FROM {detectors}) f ON f.DeviceId = r.DeviceId AND f.Detector = r.det
              JOIN hl_limits l ON l.DeviceId = r.DeviceId AND l.det = r.det
              LEFT JOIN (SELECT DeviceId, Phase, TimeStamp::TIMESTAMP AS b, sum(Green_Yellow_ns) / 1e9 AS sgy FROM {s}health_phase_5min WHERE TimeStamp >= {A} AND TimeStamp < {Z} GROUP BY ALL) p ON p.DeviceId = r.DeviceId AND p.Phase = f.phase AND p.b = r.b)
            WHERE q5 > lim_volume GROUP BY ALL""")
    # ped: each ON with its walks and greens so far
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
    rows = con.execute(" UNION ALL ".join(f"SELECT * FROM ({p})" for p in parts)).df()
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
    """the saved rows in the listed types, whatever they went through.  A run the state lists whose rows are not there raises: the check would write wrong rows that can never be fixed."""
    hist = _source(con, history, "hl_hist")
    listed = {r[0] for r in con.execute(f"SELECT Run FROM hl_runs WHERE TimeStamp > {_ts(cut)}").fetchall()}
    found = {r[0] for r in con.execute(f"SELECT DISTINCT Run::BIGINT FROM {hist}").fetchall()} if hist else set()
    missing = sorted(listed - found)
    if missing:
        raise ValueError(f"detector_health history {history!r} lacks the rows of {len(missing)} of the {len(listed)} runs the state lists (first: run {missing[0]}). Restore them, or pass state '' to start again.")
    if not hist:
        return None
    have = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {hist}").fetchall()}
    cols = ", ".join((f'"{k}"::{t or device_type}' if k in have else f"NULL::{t or device_type}") + f' AS "{k}"' for k, t in ROW_COLUMNS.items())
    con.execute(f"CREATE OR REPLACE TEMP VIEW hl_hist_t AS SELECT {cols} FROM {hist}")
    return "hl_hist_t"


def _whole_day(con, M, detectors, history, params, temp_dir):
    device_type = I._meta(con, STORE)["device_type"]
    cut = M - pd.Timedelta(days=params["horizon_days"])
    hist = _typed_history(con, history, cut, device_type)
    path = os.path.join(temp_dir, f"check_{os.getpid()}.duckdb")
    for f in (path, path + ".wal"):
        if os.path.exists(f):
            os.remove(f)
    I.attach_store(con, path, name=CHECK, horizon_days=params["horizon_days"], ped=params["ped"])
    c = f"{CHECK}."
    try:
        con.execute(_render.render("store.sql", device_type=device_type, s=c))
        I._set_meta(con, CHECK, device_type=device_type)
        for t in ROW_TABLES:
            cols = [r[0] for r in con.execute(f"DESCRIBE {STORE}.{t}").fetchall()]
            tc = "EndTime" if t == "health_intervals" else "TimeStamp"
            sel = ", ".join(f'"{x}"' for x in cols)
            if hist:
                con.execute(f"INSERT INTO {c}{t} BY NAME SELECT {sel} FROM {hist} WHERE Tbl = '{t}' AND {tc} >= {_ts(cut)} AND Run IN (SELECT Run FROM hl_runs)")
            con.execute(f"INSERT INTO {c}{t} BY NAME SELECT {sel} FROM {STORE}.{t} WHERE {tc} >= {_ts(cut)}")
        con.execute(f"INSERT INTO {c}health_carry SELECT * FROM {STORE}.health_carry")
        con.execute(f"INSERT INTO {c}health_detector_phase SELECT * FROM {STORE}.health_detector_phase")
        given = B.empty_bins()
        known = M - DAY
        parts = [con.execute(f"SELECT TimeStamp, DeviceId, DetectorType, Detector, Label, Score FROM hl_released WHERE TimeStamp >= {_ts(known)} AND TimeStamp < {_ts(M)}").df()]
        if hist:
            parts.append(con.execute(f"SELECT TimeStamp, DeviceId, DetectorType, Detector, Label, Score FROM {hist} WHERE Tbl = '{RELEASED}' AND TimeStamp >= {_ts(known)} AND TimeStamp < {_ts(M)} AND Run IN (SELECT Run FROM hl_runs)").df())
        parts = [p for p in parts if len(p)]
        if parts:
            given = pd.concat(parts, ignore_index=True)[B.BIN_COLS]
        r = I.judge_rows(con, detectors, store=CHECK, end=M, options=params["options"], ped=params["ped"], ped_options=params["ped_options"], ped_assign=params["ped_assign"], workers=params["workers"], limits=True)
        f, s = F.tables(con, r.detectors, r.periods, r.ped, r.ped_periods, r.windows, r.ped_windows, summary=True)
        spans, add, drop = B.held(con, CHECK, f, s, r.ped, known, M, M, params["ped"])
        late = B.late_rows(B.bins(con, f, s, known, M, 15, spans), given, known)
        for t, d, code, par in drop:
            con.execute(f"DELETE FROM {STORE}.health_carry WHERE TimeStamp = ? AND DeviceId = ? AND EventId = ? AND Parameter = ?", [t, d, code, par])
        for t, d, code, par in add:
            con.execute(f"INSERT INTO {STORE}.health_carry VALUES (?, ?, ?, ?, true)", [t, d, code, par])
        if r.limits is not None and len(r.limits):
            flag = s[s.Status.isin(["suspect", "bad"])][["DeviceId", "DetectorType", "Detector"]].drop_duplicates()
            fast = f[(f.Label == "too-fast actuations") & (f.DetectorType == "Vehicle") & (f.Confidence >= 0.5)].merge(flag, on=["DeviceId", "DetectorType", "Detector"])[["DeviceId", "Detector"]].drop_duplicates()
            fast["Detector"] = fast.Detector.astype("int64")
            con.register("hl_lim_in", r.limits)
            con.register("hl_fast_in", fast)
            con.execute("""CREATE OR REPLACE TEMP TABLE hl_limits AS SELECT l.DeviceId, l.det::INT AS det, l.lim_stuck, l.bad_stuck, l.lim_rapid, l.bad_rapid, l.lim_volume, l.bad_volume, (k.Detector IS NOT NULL)::INT::DOUBLE AS fast_known
                           FROM hl_lim_in l LEFT JOIN hl_fast_in k ON k.DeviceId = l.DeviceId AND k.Detector = l.det""")
            con.unregister("hl_lim_in"); con.unregister("hl_fast_in")
        if r.ped is not None and len(r.ped):
            p = r.ped[["DeviceId", "Input", "Phase", "Type", "config"]].copy()
            con.register("hl_ped_in", p)
            con.execute("""CREATE OR REPLACE TEMP TABLE hl_inputs AS SELECT DeviceId, Input::INT AS ch, Phase::DOUBLE AS ph,
                           ((Type = 'presence')::INT + 2 * (coalesce(config, '') LIKE '%on ped recall%')::INT)::DOUBLE AS flags FROM hl_ped_in""")
            con.unregister("hl_ped_in")
        return late, f
    finally:
        I.detach_store(con, CHECK)
        for g in (path, path + ".wal"):
            if os.path.exists(g):
                os.remove(g)


def live_run(con, raw, detectors, *, state=None, history=None, horizon_days=3, ped=True, options=None, ped_options=None, ped_assign=False, workers=1, clock_ahead_minutes=60, temp_dir=None):
    params = dict(horizon_days=horizon_days, ped=ped, options=options, ped_options=ped_options, ped_assign=ped_assign, workers=workers)
    temp_dir = temp_dir or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    device_type = next(r[1] for r in con.execute(f"DESCRIBE SELECT DeviceId FROM {raw}").fetchall())
    try:
        return _live_run(con, raw, detectors, state, history, horizon_days, params, clock_ahead_minutes, temp_dir, device_type, ped, options)
    finally:
        _clean(con)


def _clean(con):
    """nothing of a run but its three outputs stays on the connection, as save() writes every table it can see."""
    if con.execute("SELECT count(*) FROM duckdb_databases() WHERE database_name = ?", [STORE]).fetchall()[0][0]:
        I.detach_store(con, STORE)
    for kind, name in con.execute("""SELECT 'TABLE', table_name FROM duckdb_tables() WHERE starts_with(table_name, 'hl_')
                                     UNION ALL SELECT 'VIEW', view_name FROM duckdb_views() WHERE starts_with(view_name, 'hl_')""").fetchall():
        if name not in ("hl_final", "hl_state", "hl_rows"):
            try:
                con.execute(f"DROP {kind} IF EXISTS {name}")
            except Exception:       # a frame given as state or history
                con.unregister(name)


def _midnights_past(known, mids, last, clock_ahead_minutes):
    """which of the midnights a run may cross, in order: 9 in 10 of the signals heard from in the day before it are past it, or the run is clock_ahead_minutes past it.  known: each signal's last event."""
    kept = []
    for M in mids:
        heard = known[known >= M - DAY]
        if (len(heard) and (heard >= M).mean() >= 0.9) or last >= M + pd.Timedelta(minutes=clock_ahead_minutes):
            kept.append(M)
        else:
            break
    return kept


def _live_run(con, raw, detectors, state, history, horizon_days, params, clock_ahead_minutes, temp_dir, device_type, ped, options):
    I.attach_store(con, None, name=STORE, horizon_days=horizon_days, ped=ped)
    I._set_meta(con, STORE, device_type=device_type)
    con.execute(_render.render("store.sql", device_type=device_type, s=f"{STORE}."))
    meta = _load_state(con, state, device_type)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE hl_ev AS SELECT DeviceId, TimeStamp AS Timestamp, EventId, Parameter FROM {raw}
                    UNION ALL SELECT DeviceId, TimeStamp, Value::INT, Parameter FROM hl_s WHERE EventId = {PEND}""")
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_released ({OUTPUT.format(d=device_type)})")
    con.execute(f"CREATE OR REPLACE TEMP TABLE hl_out ({OUTPUT.format(d=device_type)})")
    # where the run has reached: only 1 in 10 of the signals this run brought have an event later, so a few clocks running ahead move nothing.
    # Only this run's events count: rows held back by the last run must not hold time back when most signals stop reporting.
    first, last = con.execute(f"""WITH l AS (SELECT DeviceId, min(TimeStamp) AS a, max(TimeStamp) AS z FROM {raw} GROUP BY DeviceId) SELECT quantile_disc(a, 0.5), quantile_disc(z, 0.9) FROM l""").fetchall()[0]
    found, checked = None, None
    cut = None
    run_id = None
    if last is not None:
        first, last = pd.Timestamp(first), pd.Timestamp(last)
        reach = (last + pd.Timedelta(seconds=1)).floor("15min")
        through = meta["bins_through"] if meta["bins_through"] is not None else first.floor("15min")
        run_id = max(int(time.time() * 1000), 1 + int(con.execute("SELECT coalesce(max(Run), 0) FROM hl_runs").fetchall()[0][0]))
        first_day = meta["first_day"] if meta["first_day"] is not None else first.floor("15min")
        last_check = meta["last_check"] if meta["last_check"] is not None else first.floor("D")
        quarters = list(pd.date_range(through + Q, reach, freq="15min")) if reach > through else []
        mids = list(pd.date_range(last_check + DAY, last.floor("D"), freq="D")) if last.floor("D") > last_check else []
        known = con.execute(f"""WITH r AS (SELECT DeviceId, max(Timestamp) AS z FROM hl_ev GROUP BY 1), s AS (SELECT DeviceId, TimeStamp AS z FROM hl_s WHERE EventId = 9401)
                                SELECT greatest(r.z, s.z) AS z FROM r FULL JOIN s USING (DeviceId)""").df().z
        past = _midnights_past(known, mids, last, clock_ahead_minutes)
        if len(past) < len(mids):       # the run stops at that midnight: later events stay in the state, later bins wait
            quarters = [q for q in quarters if q <= mids[len(past)]]
        mids = past
        after = None

        def part():
            con.execute("CREATE OR REPLACE TEMP VIEW hl_part AS SELECT * FROM hl_ev" + ("" if after is None else f" WHERE Timestamp >= {_ts(after)}"))
            return "hl_part"

        def write(rows):
            if len(rows):
                con.register("hl_rows_in", rows)
                con.execute("INSERT INTO hl_out BY NAME SELECT * FROM hl_rows_in")
                con.execute("INSERT INTO hl_released BY NAME SELECT * FROM hl_rows_in")
                con.unregister("hl_rows_in")

        for stop in sorted(set(quarters) | set(mids)):
            I.update(con, part(), detectors, store=STORE, upto=stop, options=options, ped=ped)
            after = stop
            if stop in quarters:
                write(_live_step(con, stop, detectors))
                through = stop
            if stop in mids:
                if first_day <= stop - DAY:
                    late, found = _whole_day(con, stop, detectors, history, params, temp_dir)
                    checked = stop
                    write(late)
                last_check = stop
        cut = through
        if after is None or after < cut:        # a run that reached nothing new: rows of signals behind the others
            I.update(con, part(), detectors, store=STORE, upto=cut, options=options, ped=ped)
        meta.update(bins_through=through, last_check=last_check, first_day=first_day)
    s = f"{STORE}."
    union = " UNION ALL BY NAME ".join(f"SELECT '{t}' AS Tbl, * FROM {s}{t}" for t in ROW_TABLES)
    run = str(run_id) if run_id is not None else "NULL::BIGINT"
    typed = ", ".join(f'"{k}"::{t or device_type} AS "{k}"' for k, t in ROW_COLUMNS.items() if k != "Run")
    con.execute(f"""CREATE OR REPLACE TABLE hl_rows AS SELECT {typed}, {run}::BIGINT AS "Run" FROM (
        {union} UNION ALL BY NAME SELECT '{RELEASED}' AS Tbl, TimeStamp, DeviceId, DetectorType, Detector, Label, Score FROM hl_released)""")
    if cut is not None:
        if con.execute("SELECT count(*) FROM hl_rows").fetchall()[0][0]:    # a run with no rows has nothing the check must find
            con.execute(f"INSERT INTO hl_runs VALUES ({_ts(cut)}, {run_id})")
        con.execute(f"DELETE FROM hl_runs WHERE TimeStamp < {_ts(cut - pd.Timedelta(days=horizon_days + 1))}")
    m = I._meta(con, STORE)
    long_lim = " UNION ALL ".join(f"SELECT NULL::TIMESTAMP, DeviceId, {c}, det, {k}::DOUBLE FROM hl_limits" for k, c in LIMIT_CODES.items())
    pending = (f"UNION ALL SELECT Timestamp::TIMESTAMP, DeviceId, {PEND}, Parameter, EventId::DOUBLE FROM hl_ev WHERE Timestamp >= {_ts(cut)} AND Timestamp < {_ts(cut + pd.Timedelta(minutes=max(15, clock_ahead_minutes)))}"
               if cut is not None else f"UNION ALL SELECT TimeStamp, DeviceId, EventId, Parameter, Value FROM hl_s WHERE EventId = {PEND}")
    version = f"SELECT NULL::TIMESTAMP, NULL::{device_type}, {VERSION}, 0, {float(I.EXTRACTOR_VERSION)}::DOUBLE"
    marks = [(BINS_THROUGH, meta["bins_through"]), (LAST_CHECK, meta["last_check"]), (FIRST_DAY, meta["first_day"]), (TRIMMED, m.get("trimmed_before"))]
    mark_sql = " UNION ALL ".join(f"SELECT TIMESTAMP '{pd.Timestamp(v)}', NULL::{device_type}, {c}, 0, NULL::DOUBLE" for c, v in marks if v is not None) or f"SELECT NULL::TIMESTAMP, NULL::{device_type}, 0, 0, NULL::DOUBLE WHERE false"
    con.execute(f"""CREATE OR REPLACE TABLE hl_state AS
        SELECT TimeStamp::TIMESTAMP AS TimeStamp, DeviceId, EventId::INTEGER AS EventId, Parameter::INTEGER AS Parameter, NULL::DOUBLE AS Value FROM {s}health_carry
        UNION ALL SELECT ValidFrom::TIMESTAMP, DeviceId, 9404, Detector::INTEGER, Phase::DOUBLE FROM {s}health_detector_phase
        UNION ALL SELECT * FROM ({long_lim})
        UNION ALL SELECT t_on, DeviceId, {PON_WALKS}, ch, walks FROM hl_pon
        UNION ALL SELECT t_on, DeviceId, {PON_GREENS}, ch, greens FROM hl_pon
        UNION ALL SELECT NULL::TIMESTAMP, DeviceId, {INPUT_PHASE}, ch, ph FROM hl_inputs
        UNION ALL SELECT NULL::TIMESTAMP, DeviceId, {INPUT_FLAGS}, ch, flags FROM hl_inputs
        UNION ALL SELECT TimeStamp, NULL::{device_type}, {RUN_ID}, 0, Run::DOUBLE FROM hl_runs
        {pending}
        UNION ALL SELECT * FROM ({mark_sql})
        UNION ALL {version}""")
    con.execute("CREATE OR REPLACE TABLE hl_final AS SELECT * FROM hl_out ORDER BY TimeStamp, DeviceId, DetectorType DESC, Detector, Label")
    return found, checked


def empty_run(con, state, raw="raw_data"):
    """a run with no events: tables hl_final and hl_rows with no rows, in all their columns, and hl_state as it was."""
    src = _source(con, state, "hl_state_in")
    if src is None and isinstance(state, str) and state:
        raise FileNotFoundError(NO_STATE.format(state))
    dtype = next(r[1] for r in con.execute(f"DESCRIBE SELECT DeviceId FROM {src or raw}").fetchall())
    if src:
        con.execute(f"CREATE OR REPLACE TABLE hl_state AS SELECT TimeStamp::TIMESTAMP AS TimeStamp, DeviceId, EventId::INTEGER AS EventId, Parameter::INTEGER AS Parameter, Value::DOUBLE AS Value FROM {src}")
    else:
        con.execute(f"CREATE OR REPLACE TABLE hl_state (TimeStamp TIMESTAMP, DeviceId {dtype}, EventId INTEGER, Parameter INTEGER, Value DOUBLE)")
    con.execute(f"CREATE OR REPLACE TABLE hl_final ({OUTPUT.format(d=dtype)})")
    cols = ", ".join(f'"{k}" {t or dtype}' for k, t in ROW_COLUMNS.items())
    con.execute(f"CREATE OR REPLACE TABLE hl_rows ({cols})")
    _clean(con)
