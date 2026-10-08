"""Detector phase + function inference: raw controller events in, one row per detector out.

Python
------
    from atspm_detector import predict
    out = predict("events.parquet")                       # DataFrame, parquet / csv path or glob
    out, phases = predict(df, start="2026-09-21 08:00", end="2026-09-21 11:00", return_phases=True)

Command line
------------
    atspm-detector --events events.parquet --out preds.csv [--out-phases phases.csv] [--out-atspm config.csv]
    python -m atspm_detector --events day.parquet --device-ids <guid> --start "..." --end "..." --out preds.csv

`events` has the columns DeviceId, Timestamp, EventId, Parameter (lowercase / underscore spellings accepted).  Any number
of signals and any duration from a few minutes to days.  Every signal is processed on its own (its own window and sample
length), so a signal's answers never depend on which other signals are in the same call; for memory, pass a few
signals at a time or `chunk_signals`.
Timestamps are read as the controller's wall-clock time.  A time-zone-aware timestamp keeps its own wall-clock time
(expressed in its own zone / offset, then the zone is dropped); it is never converted to this computer's zone.  A
parquet column stored as UTC without a recorded zone is read as UTC wall-clock time (with a warning).  `start` / `end`
are wall-clock times too (an offset on them is dropped); without them a signal's window runs from its first to its
last event.  A start / end more than 30 min outside a signal's own data is treated as not given for that signal
(same answers as leaving it out; a start years too early costs nothing).  `minutes_of_data` is the window length: a gap in the log is not subtracted (the health text
"N actuations in X h" counts only the hours with data).
Rows with a missing or unreadable value (or a Parameter outside 0..65535) are dropped and counted in one warning.

Pipeline (every model runs on the CPU with onnxruntime; no lightgbm / torch / scipy / sklearn)
  input     events 1, 7-11 (colour), 43 / 44 (calls), 81 / 82 (detector), 131 / 150 (coordination), 173 (flash); exact
            duplicates and detector channels outside 1..64 (0, dummies > 64) dropped; fault events 83-88 never read
  phase     pair ranker (LightGBM, 261 features) -> the network's pair phase head on the candidates the ranker gives
            >= .01, averaged 50 / 50 -> joint decoder (similarity + lead neighbours)
  function  function trees (229 features) + the network's function head (same pass) -> context stacker (46 columns:
            probabilities, stack size, span / track / co-location, lanes, phase- and lane-mates) -> per-lane decode
            (lane confidence >= .9); samples under 30 min: no lanes, Count / Yellow_Red twin decode instead
  side      lanes (from 30 min), setback distance, night speed: computed from the log and the answers above
  health    rule checks on the log (health_core.py, health_v4.py) against per-type
            limits (function x lane span x sample length, fitted once on healthy training detectors), computed AFTER
            classification and reported next to it; it never changes a phase or function answer (only the review
            flag / reason).  Whole-day checks (busy at night, unusual daily pattern) run on samples with a full day
One path at every sample length: the network always runs.  A `profile` argument, CLI --profile or env
DC_FAST_PROFILE is accepted for compatibility and ignored, with a FutureWarning.
No channel->phase table is used; phase and channel numbers are grouping keys only.

Refusals: `min_actuations` (default 5) -> "not enough data"; `min_prob` (default 0 = off) -> "not confident enough".
The raw opinion is always kept in phase_guess / function_guess.

Output columns: DeviceId, Detector, phase_pred, phase_prob, phase_2nd, phase_2nd_prob, function_pred, function_prob,
status, review_flag, n_actuations, minutes_of_data, phase_guess, phase_guess_prob, function_guess, function_guess_prob,
phase_margin, p_advance, p_presence, p_count, p_yellow_red, p_other, p_mid, p_bike, n_candidate_phases, lanes,
lane_conf, phase_n_lanes, phase_n_lanes_conf, distance_ft, setback_confidence, night_speed_mph, night_speed_vehicles,
health_status, health_score, health_reason, health_bad_periods, health_watch, health_categories, health_config,
health_signal_note, review_reason.
Health columns: health_status ok / watch / suspect / bad / not_enough_data (bad = two independent findings, or one
lasting 4 h or more of the day); health_score 0-1 (1 = healthy);
health_reason one plain line ("det 15: P5 Presence (covers 2 lanes) - Stuck on: held ON 3 times, 2 h in total");
health_bad_periods JSON list of {start, end, what}; health_categories "Stuck on (bad); Erratic counts (suspect)";
health_config setup notes (extension time on a count zone, set to pulse but holds ON); health_watch information only;
health_signal_note one line per signal when many of its detectors had an unusual day (a signal-wide event).
`return_phases=True` also returns the per-phase table (n_detectors, n_lanes, n_lanes_conf, lane volumes, night speed).
`to_atspm_config(out)` (or --out-atspm) turns the output into the detector configuration of the atspm package.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import blend as gb
from . import decode as dec
from . import features as f1
from . import features_expert as fx
from . import features_lag as f3
from . import features_partner as f2
from . import function as fn
from . import function_stage as fs
from . import health_core as hc
from . import health_v4 as hv4
from . import night_speed as ns_mod
from . import setback as sb_mod
from . import similarity as sim_mod
from . import trees_onnx
from .common import ALLOWED_EVENTS, MAX_DETECTOR_CHANNEL, read_json
from .common import timedelta as _td
from .health import flag_detectors, status_for_user

_HERE = Path(__file__).resolve().parent


def _weights_dir() -> Path:
    """The packaged weights; DC_WEIGHTS = another folder (absolute, or a folder name inside the package)."""
    name = os.environ.get("DC_WEIGHTS")
    if not name:
        return _HERE / "weights"
    p = Path(name)
    return p if p.is_absolute() else _HERE / name


DEFAULT_MODEL_DIR = _weights_dir()

EV_LIST = ",".join(str(e) for e in ALLOWED_EVENTS)
WIN = "infer"

ALIASES = {"deviceid": "DeviceId", "device_id": "DeviceId", "devid": "DeviceId",
           "timestamp": "Timestamp", "time_stamp": "Timestamp", "ts": "Timestamp",
           "eventid": "EventId", "event_id": "EventId", "eventcode": "EventId",
           "event_code": "EventId", "parameter": "Parameter", "param": "Parameter",
           "eventparam": "Parameter", "event_param": "Parameter"}
NEEDED = ["DeviceId", "Timestamp", "EventId", "Parameter"]

OUT_COLS = ["DeviceId", "Detector", "phase_pred", "phase_prob", "phase_2nd",
            "phase_2nd_prob", "function_pred", "function_prob", "status", "review_flag",
            "n_actuations", "minutes_of_data"]
EXTRA_COLS = ["phase_guess", "phase_guess_prob", "function_guess", "function_guess_prob",
              "phase_margin", "p_advance", "p_presence", "p_count", "p_yellow_red",
              "p_other", "p_mid", "p_bike", "n_candidate_phases",
              "lanes", "lane_conf", "phase_n_lanes", "phase_n_lanes_conf",
              "distance_ft", "setback_confidence", "night_speed_mph", "night_speed_vehicles",
              "health_status", "health_score", "health_reason", "health_bad_periods",
              "health_watch", "health_categories", "health_config", "health_signal_note", "review_reason"]
PHASE_COLS = ["DeviceId", "phase", "n_detectors", "n_lanes", "n_lanes_conf",
              "lane_volumes_per_hour", "night_speed_mph", "night_speed_vehicles"]
PHASE_STEM = "phase_lgbm_v5"
DECODE_STEM = "decode_v3"

MIN_ACTUATIONS = 5       # below this many detector ON events in the sample: no answer
MIN_PROB = 0.0           # below this top-phase probability: no answer (0 = off)
# The network's candidate filter: the pair graph runs only on (detector, candidate) pairs with ranker p >= .01 (ON by
# default; predict(siba_filter=False) / DC_SIBA_FILTER=0 runs the FUNCTION pass on every pair, a second network pass).
SIBA_FILTER = os.environ.get("DC_SIBA_FILTER", "1") != "0"
SIBA_KEEP_MIN_TREE_P = 0.01

# answered, but thin enough to flag for review
LOW_ACTUATIONS = 20
LOW_MINUTES = 10.0
LOW_CONF = 0.5

_VERBOSE = False


def _warn_profile(profile) -> None:
    """A `profile` argument or DC_FAST_PROFILE is accepted for compatibility and ignored (one path)."""
    src = (f"profile={profile!r}" if profile is not None
           else f"DC_FAST_PROFILE={os.environ['DC_FAST_PROFILE']!r}" if os.environ.get("DC_FAST_PROFILE") else None)
    if src:
        warnings.warn(f"{src} is ignored: atspm_detector has one path (the network always runs); "
                      "remove the argument", FutureWarning, stacklevel=3)


def _bag_files(model_dir: Path, stem: str, meta: dict) -> list[Path]:
    """The seed models of a stage, or the single unseeded file."""
    n = int(meta.get("n_models", 1))
    files = [model_dir / f"{stem}_s{i}.onnx" for i in range(n)]
    files = [f for f in files if f.exists()]
    if not files and (model_dir / f"{stem}.onnx").exists():
        files = [model_dir / f"{stem}.onnx"]
    if not files:
        raise FileNotFoundError(f"no {stem} model files in {model_dir}")
    return files


def log(m):
    if _VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ connection
def _q(s) -> str:
    """`s` as a SQL string literal (single quotes doubled), so paths / ids with an apostrophe are safe."""
    return "'" + str(s).replace("'", "''") + "'"


def _connect(threads: int = 4, memory: str = "4GB") -> duckdb.DuckDBPyConnection:
    """Small, self-contained in-memory DuckDB connection (system temp dir for spills, capped resources)."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit={_q(memory)}")
    con.execute(f"SET threads={max(1, int(threads))}")
    tmp = Path(tempfile.gettempdir()) / "duckdb_atspm_detector"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={_q(tmp.as_posix())}")
    con.execute("SET preserve_insertion_order=false")
    return con


# ------------------------------------------------------------- event ingestion
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if str(c) in NEEDED:
            continue
        if key in ALIASES:
            ren[c] = ALIASES[key]
    out = df.rename(columns=ren) if ren else df
    missing = [c for c in NEEDED if c not in out.columns]
    if missing:
        raise ValueError(
            f"events are missing the column(s) {missing}; need "
            "DeviceId, Timestamp, EventId, Parameter (lowercase/underscore spellings ok)")
    return out[NEEDED]


def _parquet_zone(con, path: str, col: str):
    """The time zone pandas recorded for column `col` in a parquet file's metadata (None if absent)."""
    try:
        rows = con.execute("SELECT value FROM parquet_kv_metadata(?) WHERE decode(key) = 'pandas' LIMIT 1",
                           [path]).fetchall()
        if not rows:
            return None
        v = rows[0][0]
        meta = json.loads(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v))
        for c in meta.get("columns", []):
            if str(c.get("name")) == col or str(c.get("field_name")) == col:
                tz = (c.get("metadata") or {}).get("timezone")
                return str(tz) if tz else None
    except Exception:                                              # pragma: no cover
        return None
    return None


_OFFSET = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2}):?(\d{2})?$")


def _tz_wallclock(con, zone) -> str:
    """SQL for a TIMESTAMP WITH TIME ZONE column `Timestamp` as the wall-clock time in its own zone `zone` (the zone
    the data were written with), the zone then dropped.  Unknown zone: UTC wall-clock time, with a warning."""
    if zone:
        m = _OFFSET.match(str(zone).strip())
        if m:
            mins = (int(m.group(2)) * 60 + int(m.group(3) or 0)) * (1 if m.group(1) == "+" else -1)
            return f"(timezone('UTC', Timestamp) + INTERVAL ({mins}) MINUTE)"
        try:
            con.execute(f"SELECT timezone({_q(zone)}, TIMESTAMPTZ '2000-01-01 00:00:00+00')").fetchall()
            return f"timezone({_q(zone)}, Timestamp)"
        except Exception:                                          # pragma: no cover
            pass
    warnings.warn("time-zone-aware timestamps without a known zone were read as UTC wall-clock time; pass local "
                  "(zone-free) timestamps, or a DataFrame with a tz-aware column, to keep the local time", stacklevel=4)
    return "timezone('UTC', Timestamp)"


def _source(con, events) -> tuple[str, str]:
    """(relation, timestamp SQL) for `events` (DataFrame, path or glob): a DuckDB relation with the columns DeviceId,
    Timestamp, EventId, Parameter, and the SQL of its timestamp as the controller's wall-clock time.  Time-zone-aware
    timestamps keep their own wall-clock time (each value expressed in its own zone / offset, then the zone dropped):
    they are never converted to this machine's zone."""
    zone = None
    if isinstance(events, pd.DataFrame):
        df = _normalise_columns(events)
        if isinstance(df["Timestamp"].dtype, pd.DatetimeTZDtype):
            df = df.assign(Timestamp=df["Timestamp"].dt.tz_localize(None))   # own wall-clock time
        con.register("src_events_df", df)
        rel = "src_events_df"
    else:
        p = str(events).replace("\\", "/")
        csv = p.lower().endswith((".csv", ".csv.gz", ".txt", ".tsv"))
        reader = (f"read_csv({_q(p)}, header=true, union_by_name=true" if csv
                  else f"read_parquet({_q(p)}, union_by_name=true")
        r0 = con.sql(f"SELECT * FROM {reader}) LIMIT 0")
        cols, types = list(r0.columns), [str(t) for t in r0.types]
        ren, have = {}, set(cols)
        for c in cols:
            key = str(c).strip().lower().replace(" ", "_")
            if c not in NEEDED and key in ALIASES and ALIASES[key] not in have:
                ren[c] = ALIASES[key]
        missing = [c for c in NEEDED if c not in have and c not in ren.values()]
        if missing:
            raise ValueError(f"{events}: missing column(s) {missing}")
        tcol = next(c for c in cols if ren.get(c, c) == "Timestamp")
        ts = None
        if csv:
            # the four columns are read as text and cast row by row below (a bad value drops that row instead of
            # failing the whole file); the timestamp keeps the layout the CSV sniffer found, and text with a zone
            # offset keeps its own wall-clock time (the offset is dropped by the cast)
            fmt = None
            try:
                first = con.execute("SELECT file FROM glob(?) ORDER BY file LIMIT 1", [p]).fetchone()
                if first:
                    fmt = con.execute("SELECT TimestampFormat FROM sniff_csv(?)", [first[0]]).fetchone()[0]
            except Exception:                                      # pragma: no cover
                fmt = None
            need = [c for c in cols if ren.get(c, c) in NEEDED]
            reader += ", types={" + ", ".join(f"{_q(c)}: 'VARCHAR'" for c in need) + "}"
            if fmt and types[cols.index(tcol)] == "TIMESTAMP":
                ts = f"TRY_STRPTIME(Timestamp, {_q(fmt)})"
        elif types[cols.index(tcol)] == "TIMESTAMP WITH TIME ZONE":
            zone = _parquet_zone(con, p, tcol)
        sel = ", ".join(f'"{c}" AS {ren.get(c, c)}' for c in cols if ren.get(c, c) in NEEDED)
        rel = f"(SELECT {sel} FROM {reader}))"
        if ts is not None:
            return rel, ts
    ttype = str(con.sql(f"SELECT Timestamp FROM {rel} LIMIT 0").types[0])
    ts = _tz_wallclock(con, zone) if ttype == "TIMESTAMP WITH TIME ZONE" else "Timestamp"
    return rel, ts


def _wall(x) -> str:
    """A start / end bound as a TIMESTAMP literal (wall-clock time; a zone or offset on it is dropped)."""
    t = pd.Timestamp(x)
    t = t.tz_localize(None) if t.tzinfo is not None else t
    return f"TIMESTAMP '{t.isoformat(sep=' ')}'"


def load_events(con, events, device_ids=None, start=None, end=None):
    """The de-duplicated, protocol-filtered events: TEMP TABLE `evd` (with an integer signal key `dev`), the signal map
    `devmap` (DeviceId -> dev) and the view `ev` (DeviceId, Timestamp, EventId, Parameter).  Returns (w0, w1, info).
    Rows of the allowed codes with a missing or unreadable value (or a Parameter outside 0..65535) are dropped and
    counted (info['n_invalid'], one warning); detector events keep channels 1..64 only."""
    rel, ts = _source(con, events)
    typed = (f"SELECT TRY_CAST(DeviceId AS VARCHAR) AS DeviceId, TRY_CAST({ts} AS TIMESTAMP) AS Timestamp, "
             f"TRY_CAST(EventId AS INTEGER) AS EventId, TRY_CAST(Parameter AS INTEGER) AS Parameter FROM {rel}")
    where = [f"(EventId IS NULL OR EventId IN ({EV_LIST}))"]
    if device_ids:
        where.append(f"DeviceId IN ({','.join(_q(d) for d in device_ids)})")
    if start:
        where.append(f"(Timestamp IS NULL OR Timestamp >= {_wall(start)})")
    if end:
        where.append(f"(Timestamp IS NULL OR Timestamp < {_wall(end)})")
    bad = ("(DeviceId IS NULL OR Timestamp IS NULL OR EventId IS NULL OR Parameter IS NULL "
           "OR Parameter < 0 OR Parameter > 65535)")
    keep = f"NOT {bad} AND NOT (EventId IN (81,82) AND (Parameter < 1 OR Parameter > {MAX_DETECTOR_CHANNEL}))"
    src = f"(SELECT *, {bad} AS bad, {keep} AS good FROM ({typed}) WHERE {' AND '.join(where)})"
    st = con.sql(f"SELECT DeviceId, count(*) FILTER (good) AS n_good, count(*) FILTER (bad) AS n_bad "
                 f"FROM {src} GROUP BY 1").fetchall()
    n_bad = int(sum(r[2] for r in st))
    if n_bad:
        warnings.warn(f"{n_bad:,} event row(s) with a missing or invalid value (DeviceId / Timestamp / EventId / "
                      f"Parameter) were dropped", stacklevel=3)
    con.execute("CREATE OR REPLACE TEMP TABLE devmap (DeviceId VARCHAR, dev SMALLINT)")
    ids = sorted(str(r[0]) for r in st if r[0] is not None and r[1])
    if ids:
        con.executemany("INSERT INTO devmap VALUES (?, ?)", [(d, i + 1) for i, d in enumerate(ids)])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE evd AS
        SELECT DISTINCT m.dev, e.DeviceId, e.Timestamp, CAST(e.EventId AS USMALLINT) AS EventId,
                        CAST(e.Parameter AS USMALLINT) AS Parameter
        FROM {src} e JOIN devmap m USING (DeviceId) WHERE e.good""")
    con.execute("CREATE OR REPLACE TEMP VIEW ev AS SELECT DeviceId, Timestamp, EventId, Parameter FROM evd")
    w0, w1, info = _window(con)
    info["n_invalid"] = n_bad
    return w0, w1, info


def _window(con):
    """(w0, w1, info) of the events in `evd`: first event and one second past the last, in epoch seconds."""
    n, nd, t0, t1 = con.sql("SELECT count(*), count(DISTINCT dev), min(Timestamp), max(Timestamp) FROM evd").fetchone()
    if not n:
        return None, None, {"n_events": 0}
    w0 = (pd.Timestamp(t0) - pd.Timestamp("1970-01-01")).total_seconds()
    w1 = (pd.Timestamp(t1) - pd.Timestamp("1970-01-01")).total_seconds() + 1.0
    log(f"{n:,} events, {nd} signals, span {pd.Timestamp(t0)} .. {pd.Timestamp(t1)}")
    return w0, w1, {"n_events": int(n), "n_signals": int(nd), "t0": t0, "t1": t1}


def _one_signal(con, device_id: str):
    """Restrict `evd` / `devmap` to ONE signal of `evd_all` (signal key 1, exactly as when it is passed alone), so a
    signal's answers never depend on the other signals of the call: its own window, sample length and tables."""
    con.execute("CREATE OR REPLACE TEMP TABLE evd AS SELECT 1::SMALLINT AS dev, DeviceId, Timestamp, EventId, "
                "Parameter FROM evd_all WHERE DeviceId = ?", [device_id])
    con.execute("CREATE OR REPLACE TEMP TABLE devmap (DeviceId VARCHAR, dev SMALLINT)")
    con.execute("INSERT INTO devmap VALUES (?, 1)", [device_id])
    con.execute("DROP TABLE IF EXISTS cyc5")      # built once per signal by streams.py
    return _window(con)


def build_chunk_tables(con) -> None:
    """Derived tables, all keyed by the integer signal key `dev`:

        onev_all   detector ON intervals (an 82 followed by an 81 on the channel; unpaired ONs dropped)
        cyc_all    colour cycles from 1 / 8 / 10 with 7 (green termination) and 9 (end yellow) as fall-backs, so
                   controllers that log only 1 + 7 still get a usable green / red split
        gs_all     the green-state bitmask over time (which phases are green)
        coordiv    coordination state changes (131; pattern 1..253 = coordinated)
        calls_all  phase calls 43 / 44;  cand: candidate phases (a Begin Green in the sample)
    """
    con.execute("""CREATE OR REPLACE TEMP TABLE onev_all AS
        WITH e AS (SELECT dev, Parameter::SMALLINT AS det, Timestamp AS ts, EventId FROM evd WHERE EventId IN (81,82)),
             d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev FROM e
                   WINDOW w AS (PARTITION BY dev, det ORDER BY ts, CASE WHEN EventId=82 THEN 0 ELSE 1 END))
        SELECT dev, det, epoch_ms(ts)/1000.0 AS t_on, epoch_ms(nts)/1000.0 AS t_off,
               (epoch_ms(nts - ts)/1000.0)::FLOAT AS dur
        FROM d WHERE EventId = 82 AND nev = 81 AND nts IS NOT NULL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cyc_raw AS
        WITH g AS (
          SELECT dev, Parameter::SMALLINT AS p, Timestamp AS t, EventId,
                 SUM(CASE WHEN EventId=1 THEN 1 ELSE 0 END) OVER (
                     PARTITION BY dev, Parameter
                     ORDER BY t, CASE EventId WHEN 1 THEN 0 WHEN 7 THEN 1 WHEN 8 THEN 2
                                              WHEN 9 THEN 3 WHEN 10 THEN 4 ELSE 5 END
                     ROWS UNBOUNDED PRECEDING) AS cyc
          FROM evd WHERE EventId IN (1,7,8,9,10,11)
        ), c AS (
          SELECT dev, p, cyc,
                 min(t) FILTER (EventId=1)  AS green_start,
                 min(t) FILTER (EventId=8)  AS yellow_ev,
                 min(t) FILTER (EventId=7)  AS green_term,
                 min(t) FILTER (EventId=10) AS red_ev,
                 min(t) FILTER (EventId=9)  AS yellow_end,
                 min(t) FILTER (EventId=11) AS redclr_ev
          FROM g WHERE cyc > 0 AND p BETWEEN 1 AND 16 GROUP BY 1,2,3
        )
        SELECT dev, p, cyc, green_start,
               coalesce(yellow_ev, green_term) AS yellow_start,
               coalesce(red_ev, yellow_end) AS red_start,
               redclr_ev AS redclr_end,
               LEAD(green_start) OVER (PARTITION BY dev, p ORDER BY green_start) AS next_green
        FROM c WHERE green_start IS NOT NULL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cyc_all AS
        SELECT dev, p, cyc::INT AS cyc,
               epoch_ms(green_start)/1000.0 AS gs,
               epoch_ms(coalesce(yellow_start, red_start, next_green))/1000.0 AS ge,
               epoch_ms(coalesce(red_start, yellow_start))/1000.0 AS rs,
               epoch_ms(next_green)/1000.0 AS ng,
               (epoch_ms(coalesce(yellow_start, red_start) - green_start)/1000.0)::FLOAT AS green_secs
        FROM cyc_raw""")
    con.execute("""CREATE OR REPLACE TEMP TABLE gs_all AS
        WITH iv AS (SELECT dev, p, green_start AS t0, coalesce(yellow_start, red_start, next_green) AS t1
                    FROM cyc_raw WHERE coalesce(yellow_start, red_start, next_green) IS NOT NULL),
             ch AS (SELECT dev, t0 AS t,  (1::BIGINT << (p-1)) AS d FROM iv
                    UNION ALL
                    SELECT dev, t1 AS t, -(1::BIGINT << (p-1)) AS d FROM iv),
             agg AS (SELECT dev, t, sum(d) AS d FROM ch GROUP BY 1,2),
             run AS (SELECT dev, t, sum(d) OVER (PARTITION BY dev ORDER BY t ROWS UNBOUNDED PRECEDING) AS mask
                     FROM agg)
        SELECT dev, epoch_ms(t)/1000.0 AS t0, epoch_ms(LEAD(t) OVER (PARTITION BY dev ORDER BY t))/1000.0 AS t1, mask
        FROM run""")
    con.execute("DELETE FROM gs_all WHERE t1 IS NULL")
    con.execute("""CREATE OR REPLACE TEMP TABLE coordiv AS
        SELECT dev, epoch_ms(Timestamp)/1000.0 AS t0, (Parameter BETWEEN 1 AND 253) AS is_coord
        FROM evd WHERE EventId = 131""")
    con.execute("""CREATE OR REPLACE TEMP TABLE calls_all AS
        SELECT dev, Parameter::SMALLINT AS p, EventId::SMALLINT AS ev, epoch_ms(Timestamp)/1000.0 AS t
        FROM evd WHERE EventId IN (43,44) AND Parameter BETWEEN 1 AND 16""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cand AS
        SELECT DISTINCT dev, Parameter::SMALLINT AS p FROM evd WHERE EventId = 1 AND Parameter BETWEEN 1 AND 16""")


# ------------------------------------------------------------- per-signal facts
def detector_universe(con) -> pd.DataFrame:
    """Every detector channel that appears on an 81/82 event, with its actuation count (82 events)."""
    return con.sql("""
        SELECT DeviceId, Parameter::INT AS Detector, count(*) FILTER (EventId = 82)::BIGINT AS n_actuations
        FROM evd WHERE EventId IN (81,82) GROUP BY 1,2 ORDER BY 1,2""").df()


def signal_facts(con) -> pd.DataFrame:
    """Per signal: span, candidate phases, whether colour-termination / call events exist."""
    return con.sql("""
        SELECT DeviceId,
               (epoch_ms(max(Timestamp) - min(Timestamp))/60000.0)::DOUBLE AS minutes_of_data,
               count(*) FILTER (EventId = 1 AND Parameter BETWEEN 1 AND 16) AS n_green,
               bit_count(bit_or(CASE WHEN EventId = 1 AND Parameter BETWEEN 1 AND 16
                                     THEN (1::INTEGER << (Parameter - 1)) ELSE 0 END))::BIGINT AS n_candidate_phases,
               count(*) FILTER (EventId IN (7,8,9,10)) AS n_green_end,
               count(*) FILTER (EventId IN (43,44)) AS n_calls
        FROM evd GROUP BY 1""").df()


def gate_frame(con, win_secs: float, w1: float | None = None) -> pd.DataFrame:
    """The classifiability gate (`health.py`): no actuations / stuck ON / chatter storm.  A channel with no completed
    actuation but an ON that never ends inside the sample (82s and no 81 after them) is stuck ON from its first such
    82 to the sample end `w1` (`open_on_s`): reported as stuck ON, not as 'no actuations'.  Channels with completed
    actuations are judged on those alone (an ON still open at the sample end is normal, e.g. a queue at red)."""
    days = max(win_secs / 86400.0, 1e-6)
    tail_end = float(w1) if w1 is not None else None
    tail = ("" if tail_end is None else f"""
        , e AS (SELECT dev, Parameter::SMALLINT AS det, Timestamp AS ts, EventId FROM evd WHERE EventId IN (81,82)),
        lo AS (SELECT dev, det, max(ts) FILTER (EventId = 81) AS last_off FROM e GROUP BY 1,2),
        tl AS (SELECT e.dev, e.det, greatest({tail_end} - epoch_ms(min(e.ts))/1000.0, 0) AS tail_s
               FROM e JOIN lo USING (dev, det)
               WHERE e.EventId = 82 AND (lo.last_off IS NULL OR e.ts > lo.last_off) GROUP BY 1,2)""")
    src = "a" if tail_end is None else "a FULL OUTER JOIN tl USING (dev, det)"
    tcol = "0" if tail_end is None else "(CASE WHEN a.n_on IS NULL THEN coalesce(tl.tail_s, 0) ELSE 0 END)"
    h = con.sql(f"""
        WITH a AS (
          SELECT dev, det, count(*) AS n_on, sum(dur) AS occ, max(dur) AS longest_on_s,
                 max(cnt) AS max_on_per_min
          FROM (SELECT dev, det, dur, count(*) OVER (PARTITION BY dev, det,
                       (t_on/60)::BIGINT) AS cnt FROM onev_all) GROUP BY 1,2
        ){tail}
        SELECT d.DeviceId, det::INT AS Detector, coalesce(a.n_on, 0) AS n_on,
               coalesce(a.n_on, 0) / {days} AS on_per_day,
               (coalesce(a.occ, 0) + {tcol}) / {max(win_secs, 1.0)} AS frac_time_on,
               greatest(coalesce(a.longest_on_s, 0), {tcol}) AS longest_on_s, coalesce(a.max_on_per_min, 0)
                   AS max_on_per_min, {tcol} AS open_on_s
        FROM {src} JOIN devmap d USING (dev)
    """).df()
    if not len(h):
        h = pd.DataFrame(columns=["DeviceId", "Detector", "n_on", "on_per_day",
                                  "frac_time_on", "longest_on_s", "max_on_per_min", "open_on_s"])
    return flag_detectors(h)


def on_table(con, b0: float, b1: float) -> dict:
    """{DeviceId: {det: (t_on, t_off)}} of the ON intervals starting inside [b0, b1), sorted by time; fetched once and
    shared by the function stage (pick / lanes / twins) and the night speed."""
    r = con.execute("SELECT m.DeviceId, o.det::INT AS det, o.t_on, o.t_off FROM onev_all o JOIN devmap m USING (dev) "
                    "WHERE o.t_on >= ? AND o.t_on < ? ORDER BY m.DeviceId, o.det, o.t_on", [b0, b1]).fetchnumpy()
    out: dict = {}
    n = len(r["det"])
    if not n:
        return out
    dv, dt = np.asarray(r["DeviceId"], dtype=object), np.asarray(r["det"], dtype=np.int64)
    ton, toff = np.asarray(r["t_on"], dtype=np.float64), np.asarray(r["t_off"], dtype=np.float64)
    brk = np.flatnonzero((dv[1:] != dv[:-1]) | (dt[1:] != dt[:-1])) + 1
    for a, b in zip(np.r_[0, brk], np.r_[brk, n]):
        out.setdefault(str(dv[a]), {})[int(dt[a])] = (ton[a:b], toff[a:b])
    return out


# ------------------------------------------------------------------- features
EMPTY_SIM = pd.DataFrame({"DeviceId": pd.Series(dtype=object),
                          "Detector": pd.Series(dtype="int16"),
                          "other": pd.Series(dtype="int16"),
                          "phi": pd.Series(dtype="float32"),
                          "n_common": pd.Series(dtype="int64"),
                          "win": pd.Series(dtype=object)})


def build_features(con, w0: float, w1: float, devmap: pd.DataFrame):
    """-> (pair features, similarity graph, lead-neighbour graph, detector-pair lag table)."""
    secs = max(w1 - w0, 1.0)
    f1.apply_window(con, w0, w1)
    parts = f1.shared_parts(con)
    base = f1.build_window(con, WIN, secs, parts, devmap)
    if not len(base):
        return None, None, None, None
    base = f1.finalise(base)
    extra = f2.build_window(con, WIN, secs, parts, devmap)
    del parts
    sim = sim_mod.build_window(con, WIN, secs, devmap)
    if sim is None or not len(sim):
        sim = EMPTY_SIM.copy()
    lead = sim_mod.build_lead_window(con, WIN, devmap)
    df = base
    if extra is not None and len(extra):
        df = base.merge(extra, on=["DeviceId", "Detector", "cand_phase", "win"], how="left")
    df = f2.add_partner_diffs(df, f2.PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                                    "f_on_green", "excl_diff_min",
                                                    "release_frac_long", "call43_fwd_lift"])
    lag = None
    try:
        lag = f3.build(con, WIN, devmap, f3.SQL_LAG).rename(columns={"det": "Detector", "oth": "other"})
        lag["Detector"] = lag.Detector.astype(df.Detector.dtype)
        lag["other"] = lag.other.astype(df.Detector.dtype)
    except Exception as exc:                                       # pragma: no cover
        log(f"lag features unavailable ({type(exc).__name__}: {exc})")
    return df, sim, lead, lag


# --------------------------------------------------------------------- scoring
def _det_groups(df: pd.DataFrame) -> list:
    return [df.DeviceId.to_numpy(), df.Detector.to_numpy()]


def _softmax_by_detector(df: pd.DataFrame, s: np.ndarray, T: float = 1.0) -> np.ndarray:
    g = _det_groups(df)
    v = pd.Series(np.asarray(s, dtype=np.float64) / T)
    e = np.exp(v - v.groupby(g, sort=False).transform("max"))
    return (e / e.groupby(g, sort=False).transform("sum")).to_numpy()


def _normalise_by_detector(df: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    v = pd.Series(np.clip(np.asarray(s, dtype=np.float64), 1e-9, None))
    return (v / v.groupby(_det_groups(df), sort=False).transform("sum")).to_numpy()


def siba_phase_probs(streams: dict, pieces_rel, net, keep: dict) -> pd.DataFrame:
    """The phase network = the network's pair phase head, on the KEPT (detector, candidate) pairs (`keep`
    {(DeviceId, det): set of candidate phases}) -> DeviceId, Detector, cand_phase, p_gru (per detector a softmax over its
    kept candidates, members averaged).  The pass is memoised on the streams and reused by the function stage."""
    rows = []
    for dev, z in streams.items():
        dets = [int(c) for c in z["det_ch"]]
        cand = np.asarray(z["cand"], dtype=int)
        if not dets or cand.size < 1:
            continue
        kd = {d: {int(c) for c in keep[(dev, d)]} for d in dets if (dev, d) in keep}
        if not kd:
            continue
        _, pp = net.both(z, dets, pieces_rel, keep=kd)
        km = np.array([[int(c) in kd.get(d, ()) for c in cand] for d in dets], bool)
        di, ki = np.nonzero(km)
        rows.append(pd.DataFrame({"DeviceId": dev, "Detector": np.asarray(dets, np.int64)[di],
                                  "cand_phase": cand[ki].astype(np.int64), "p_gru": pp[di, ki].astype(np.float64)}))
    if not rows:
        return pd.DataFrame({"DeviceId": pd.Series(dtype=object), "Detector": pd.Series(dtype="int64"),
                             "cand_phase": pd.Series(dtype="int64"), "p_gru": pd.Series(dtype="float64")})
    return pd.concat(rows, ignore_index=True)


def score(df: pd.DataFrame, sim: pd.DataFrame, model_dir: Path, con=None, w0: float | None = None,
          w1: float | None = None, blend_cfg: dict | None = None, streams=None, pieces_rel=None,
          lead: pd.DataFrame | None = None) -> pd.DataFrame:
    """Phase scoring.  Stage 1: the pair ranker, softmax per detector (p0; kept as p0_tree).  Stage 1b: the network's
    phase head on the candidates the ranker gives >= .01, blended 0.5 / 0.5 (a signal without streams -- not seen in
    practice, the streams are built from the same candidates and detectors -- keeps the ranker alone).  Stage 2: the
    joint decoder decode_v3, reading the similarity and lead-neighbour graphs; its binary output is normalised per
    detector (prob)."""
    meta = read_json(model_dir / f"{PHASE_STEM}.json")
    files = _bag_files(model_dir, PHASE_STEM, meta)
    miss = [c for c in meta["features"] if c not in df.columns]
    if miss:                                       # columns added in one step (no fragmented frame)
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=miss)], axis=1)
    df = df.sort_values(["DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    X = df[meta["features"]].to_numpy(np.float64, na_value=np.nan)
    ps = [_softmax_by_detector(df, np.asarray(trees_onnx.load(f).predict(X))) for f in files]
    p0 = np.mean(ps, axis=0)
    # p0_tree = ranker only (the network's candidate filter reads it); p0 is blended below
    df = pd.concat([df, pd.DataFrame({"p0": p0, "p0_tree": p0.copy()}, index=df.index)], axis=1)
    if blend_cfg is not None and streams:
        k = df[df.p0 >= gb.KEEP_MIN_TREE_P]
        keep: dict = {}
        for dv, d, c in zip(k.DeviceId.astype(str), k.Detector.astype(int), k.cand_phase.astype(int)):
            keep.setdefault((dv, d), set()).add(c)
        net = siba_phase_probs(streams, pieces_rel, fs.net_model(model_dir), keep)
        if len(net):
            df["p0"] = gb.mix(df, "p0", net, float(blend_cfg["weight_lightgbm"]))
        log(f"phase net on {len(net):,} of {len(df):,} pairs (ranker p >= {gb.KEEP_MIN_TREE_P})")
    dmeta = read_json(model_dir / f"{DECODE_STEM}.json")
    dbst = trees_onnx.load(model_dir / f"{DECODE_STEM}.onnx")
    key = ["DeviceId", "Detector", "win", "cand_phase"]
    Xd = dec.assemble(df[key + ["p0"]], pairs=df, sim=sim, lead=lead)
    for c in dmeta["features"]:
        if c not in Xd.columns:
            Xd[c] = np.nan
    sd = np.asarray(dbst.predict(Xd[dmeta["features"]].to_numpy(np.float64, na_value=np.nan)))
    Xd["prob"] = (_normalise_by_detector(Xd, sd) if dmeta.get("mode") == "binary"
                  else _softmax_by_detector(Xd, sd, dmeta.get("temperature", 1.0)))
    return df.merge(Xd[key + ["prob"]], on=key, how="left")


def _function_frame(df: pd.DataFrame, lag: pd.DataFrame | None) -> pd.DataFrame:
    """The function design matrix: the pair features of the detector's PREDICTED phase (argmax of the decoded phase
    probability), plus shape, sibling-relative and cross-detector lag aggregates.  Same code path as training."""
    pairs = df.drop(columns=["p0", "prob", "p0_tree"], errors="ignore")
    probs = df[["DeviceId", "Detector", "win", "cand_phase", "prob"]]
    p = pairs.merge(probs, on=["DeviceId", "Detector", "win", "cand_phase"], how="inner")
    i = p.groupby(["DeviceId", "Detector", "win"], sort=False)["prob"].idxmax()
    top = p.loc[i].copy().rename(columns={"cand_phase": "pred_phase", "prob": "top_prob"})
    top = fn.add_shape_features(top)
    top = fn.add_sibling_features(top)
    top = top.reset_index(drop=True)
    if lag is not None and len(lag):
        top = fn.add_lag_features(top, lag)
    # the training frame carried a flag for the full-span window group (>= ~66 h); at inference the sample's own
    # length decides it
    if "win_secs" in top.columns:
        top["is_full"] = (top.win_secs.astype(float) >= 48 * 3600.0)
    return top


def _bin_window(start, end, w0: float, w1: float) -> tuple[float, float]:
    """The sample window: the requested start / end where given (training samples were fixed clock windows), else the
    first event and one second past the last."""
    def sec(x):
        t = pd.Timestamp(x)
        t = t.tz_localize(None) if t.tzinfo is not None else t
        return (t - pd.Timestamp("1970-01-01")).total_seconds()
    b0 = sec(start) if start else w0
    b1 = sec(end) if end else w1
    return (b0, b1) if b1 > b0 else (w0, w1)


# A requested start / end more than this far outside the signal's own data is treated as not given (the
# window then runs from its first event / to one second past its last).  Inside it the requested clock window is kept
# (training samples were fixed clock windows; a signal can be quiet for up to ~18 min at night).
CLIP_TOL_S = 1800.0


def _clip_window(b0: float, b1: float, w0: float, w1: float) -> tuple[float, float]:
    """[b0, b1) clipped to the signal's data span [w0, w1) where it reaches more than CLIP_TOL_S beyond it, so a start
    far before the first event (or an end far after the last) costs nothing and answers exactly as if it were not
    given."""
    if w0 - b0 > CLIP_TOL_S:
        b0 = w0
    if b1 - w1 > CLIP_TOL_S:
        b1 = w1
    return b0, b1


def siba_keep(df: pd.DataFrame, thr: float = SIBA_KEEP_MIN_TREE_P) -> dict:
    """{DeviceId: {detector: candidate phases with ranker p >= thr}} (the network's candidate filter)."""
    k = df[df.p0_tree >= thr]
    out: dict = {}
    for dv, d, c in zip(k.DeviceId.astype(str), k.Detector.astype(int), k.cand_phase.astype(int)):
        out.setdefault(dv, {}).setdefault(int(d), set()).add(int(c))
    return out


def score_function(df: pd.DataFrame, model_dir: Path, con, b0: float, b1: float, streams=None, pieces_rel=None,
                   siba_filter: bool | None = None, lag=None, devmap=None,
                   on_iv: dict | None = None):
    """The function half (function_stage.py) on the design frame built at each detector's predicted phase, with the
    expert count bins laid on [b0, b1)."""
    use_f = SIBA_FILTER if siba_filter is None else bool(siba_filter)
    net_keep = siba_keep(df) if use_f and "p0_tree" in df.columns else None
    top = _function_frame(df, lag)
    if not len(top):
        return None, pd.DataFrame()
    ex = fx.build(con, b0, b1, top, devmap)
    ex["Detector"] = ex.Detector.astype(top.Detector.dtype)
    top = top.merge(ex, on=["DeviceId", "Detector"], how="left")
    return fs.run(con, top, model_dir, b0, b1, streams, pieces_rel, log, net_keep=net_keep, on_iv=on_iv)


# ----------------------------------------------------------------- assembly
PROB_COLS = ["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other", "p_mid",
             "p_bike"]


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=OUT_COLS + EXTRA_COLS)


def _top_phase(ph: pd.DataFrame | None) -> pd.DataFrame:
    """(DeviceId, Detector) -> best and second-best phase with their probabilities."""
    cols = ["DeviceId", "Detector", "phase_pred", "phase_prob", "phase_2nd", "phase_2nd_prob"]
    if ph is None or not len(ph):
        return pd.DataFrame(columns=cols)
    p = ph.sort_values(["DeviceId", "Detector", "prob"], ascending=[True, True, False])
    g = p.groupby(["DeviceId", "Detector"], sort=False)
    a = g.head(1).rename(columns={"cand_phase": "phase_pred", "prob": "phase_prob"})
    b = g.nth(1).rename(columns={"cand_phase": "phase_2nd", "prob": "phase_2nd_prob"})
    return a[cols[:4]].merge(b[["DeviceId", "Detector", "phase_2nd", "phase_2nd_prob"]],
                             on=["DeviceId", "Detector"], how="left")


# ------------------------------------------------------------ lanes, setback, health, night speed
FUNC_CLASSES = ["Advance", "Presence", "Count", "Yellow_Red", "Other", "Mid", "Bike"]   # PROB_COLS order
POST_COLS = ["lanes", "lane_conf", "phase_n_lanes", "phase_n_lanes_conf", "distance_ft",
             "setback_confidence", "night_speed_mph", "night_speed_vehicles", "health_status", "health_score",
             "health_reason", "health_bad_periods", "health_watch", "health_categories", "health_config",
             "health_signal_note"]
FN_COLS = ["function_pred", "function_prob"] + PROB_COLS     # what _assemble takes from the function stage


def _ts(sec: float) -> pd.Timestamp:
    return pd.Timestamp("1970-01-01") + _td(seconds=float(sec))


def _periods_json(L) -> str:
    if not isinstance(L, list) or not L:
        return ""
    return json.dumps([{"start": str(d["start"]), "end": str(d["end"]), "what": d["what"],
                        "recovered": d.get("recovered")} for d in L])


def _lane_tuple(s) -> tuple:
    if not isinstance(s, str):
        return ()
    return tuple(int(x) for x in s.split(",") if x.strip().isdigit())


def _health_events(con, dev: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> hc.Prep:
    """One signal's allowed events inside [start, end) as health_core's prepared arrays, straight from `evd` (already
    de-duplicated and channel-filtered): seconds from the window start (exact: integer nanoseconds / 1e9)."""
    s_ns, e_ns = int(start_ts.value), int(end_ts.value)
    codes = ",".join(str(c) for c in hc.ALLOWED)
    r = con.execute(f"SELECT (epoch_ns(e.Timestamp) - {s_ns})::DOUBLE / 1e9 AS t, e.EventId, e.Parameter "
                    f"FROM evd e JOIN devmap m USING (dev) WHERE m.DeviceId = ? AND e.EventId IN ({codes}) "
                    f"AND epoch_ns(e.Timestamp) >= {s_ns} AND epoch_ns(e.Timestamp) < {e_ns}", [dev]).fetchnumpy()
    return hc.prep_arrays(r["t"], r["EventId"], r["Parameter"])


def post_outputs(con, w0: float, w1: float, b0: float, b1: float, univ: pd.DataFrame,
                 top: pd.DataFrame, fn: pd.DataFrame | None, lph: pd.DataFrame | None, model_dir: Path,
                 on_iv: dict | None = None):
    """Lanes (from the function stage), setback, health and night speed for every detector, from the log still open on
    `con` and the model's own answers (the raw opinion, before any refusal).  [b0, b1) = the sample window.
    Returns (detector table with POST_COLS, phase table, note)."""
    key = ["DeviceId", "Detector"]
    d = univ[key].merge(top[key + ["phase_pred", "phase_prob"]], on=key, how="left")
    extra = ["lanes", "n_lanes_spanned", "lane_conf", "lane_phase"]
    if fn is not None and len(fn):
        d = d.merge(fn[key + ["function_pred"] + PROB_COLS + extra], on=key, how="left")
    else:
        d["function_pred"] = None
        for c in PROB_COLS + extra:
            d[c] = np.nan
    notes, h_parts, sp_det, sp_ph = [], [], [], []
    start_ts, end_ts = _ts(b0), _ts(b1)
    pt = lph.copy() if lph is not None and len(lph) else pd.DataFrame(columns=PHASE_COLS[:6])
    out = d[key].copy()
    lt = d[key + ["lanes", "lane_conf", "lane_phase"]].copy()
    lt["lanes"] = lt.lanes.where(lt.lanes.notna() & lt.lanes.ne("") & d.function_pred.ne("Bike"), "")
    if len(pt):
        pm = pt.set_index([pt.DeviceId, pt.phase.astype(float)])
        ix = pd.MultiIndex.from_arrays([lt.DeviceId, lt.lane_phase.astype(float)])
        lt["phase_n_lanes"] = pm.n_lanes.reindex(ix).to_numpy(float)
        lt["phase_n_lanes_conf"] = pm.n_lanes_conf.reindex(ix).to_numpy(float)
    else:
        lt["phase_n_lanes"] = np.nan
        lt["phase_n_lanes_conf"] = np.nan
    lt.loc[lt.lanes.eq(""), ["phase_n_lanes", "phase_n_lanes_conf", "lane_conf"]] = np.nan
    out = out.merge(lt[key + ["lanes", "lane_conf", "phase_n_lanes", "phase_n_lanes_conf"]], on=key, how="left")
    # setback: every detector with a predicted phase and function (the final, decoded function)
    try:
        sbm = fs._cached(("setback", str(model_dir)), lambda: sb_mod.SetbackModel(model_dir / "setback"))
        sd = d[key + ["phase_pred", "function_pred", "p_mid"]].rename(
            columns={"phase_pred": "phase", "function_pred": "function"})
        st = sb_mod.setback(con, b0, b1, sd, sbm)
        out = out.merge(st[key + ["distance_ft", "setback_confidence"]], on=key, how="left")
    except Exception as exc:                                       # pragma: no cover
        notes.append(f"setback unavailable ({type(exc).__name__}: {exc})")
        out["distance_ft"] = np.nan
    sbd = dict(zip(zip(out.DeviceId, out.Detector.astype(int)), out.distance_ft))
    # greens of every phase inside the window (night speed), one query for all signals
    cy = con.execute("SELECT m.DeviceId, c.p::INT AS p, c.gs, c.ge FROM cyc_all c JOIN devmap m USING (dev) "
                     "WHERE c.gs < ? AND c.ge > ? ORDER BY m.DeviceId, c.p, c.gs", [b1, b0]).df()
    cyd = {k: v for k, v in cy.groupby("DeviceId", sort=False)}
    on_iv = on_iv if on_iv is not None else on_table(con, b0, b1)
    for dev, g in d.groupby("DeviceId", sort=True):
        try:
            prep = _health_events(con, dev, start_ts, end_ts)
            phase = {int(k): float(v) for k, v in zip(g.Detector, g.phase_pred) if pd.notna(v)}
            pconf = {int(k): float(v) for k, v in zip(g.Detector, g.phase_prob) if pd.notna(v)}
            func = {int(r.Detector): {c: float(getattr(r, pc)) for c, pc in zip(FUNC_CLASSES, PROB_COLS)}
                    for r in g.itertuples() if pd.notna(r.p_advance)}
            span = {int(k): float(v) for k, v in zip(g.Detector, g.n_lanes_spanned) if pd.notna(v) and v > 0}
            flab = {int(k): v for k, v in zip(g.Detector, g.function_pred) if isinstance(v, str)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h, sig_note = hv4.assess(prep, start_ts, end_ts, phase, flab, func, span, pconf or None)
            if len(h):
                h = h.rename(columns={"detector": "Detector"})
                h.insert(0, "DeviceId", dev)
                h["health_signal_note"] = sig_note
                h_parts.append(h)
        except Exception as exc:                                   # pragma: no cover
            notes.append(f"health unavailable for {dev} ({type(exc).__name__})")
        # night-time approach speed: advance -> lane-mate stop-bar zone, isolated night vehicles
        try:
            iv = on_iv.get(dev, {})
            on = {k: v[0] for k, v in iv.items()}
            off = {k: v[1] for k, v in iv.items()}
            c = cyd.get(dev)
            greens = ({} if c is None else
                      {int(k): (np.maximum(x["gs"].to_numpy(float), b0), np.minimum(x["ge"].to_numpy(float), b1))
                       for k, x in c.groupby("p")})
            dl = []
            for r in g.itertuples():
                if pd.isna(r.phase_pred) or not isinstance(r.function_pred, str):
                    continue
                dl.append(dict(det=int(r.Detector), phase=int(r.phase_pred), function=r.function_pred,
                               lanes=_lane_tuple(r.lanes), setback_ft=float(sbd.get((dev, int(r.Detector)), np.nan))))
            dr, pr = ns_mod.night_speed(b0, b1, dl, on, off, greens)
            for x in dr:
                sp_det.append((dev, int(x["det"]), x["speed_mph"], x["n"] if np.isfinite(x["speed_mph"]) else np.nan))
            for x in pr:
                sp_ph.append((dev, int(x["phase"]), x["speed_mph"], x["n"] if np.isfinite(x["speed_mph"]) else np.nan))
        except Exception as exc:                                   # pragma: no cover
            notes.append(f"night speed unavailable for {dev} ({type(exc).__name__}: {exc})")
    if sp_det:
        sdf = pd.DataFrame(sp_det, columns=["DeviceId", "Detector", "night_speed_mph", "night_speed_vehicles"])
        sdf["night_speed_mph"] = sdf.night_speed_mph.astype(float).round(1)
        out = out.merge(sdf, on=key, how="left")
    if h_parts:
        h = pd.concat(h_parts, ignore_index=True)
        h["Detector"] = h.Detector.astype(int)
        h["health_bad_periods"] = h.health_bad_periods.map(_periods_json)
        # a detector too quiet to judge has no score, so filtering on health_score never keeps it by accident
        h.loc[h.health_status.eq("not_enough_data"), "health_score"] = np.nan
        out = out.merge(h[key + ["health_status", "health_score", "health_reason", "health_bad_periods",
                                 "health_watch", "health_categories", "health_config", "health_signal_note"]],
                        on=key, how="left")
    for c in POST_COLS:
        if c not in out.columns:
            out[c] = np.nan
    if len(pt):
        if sp_ph:
            spp = pd.DataFrame(sp_ph, columns=["DeviceId", "phase", "night_speed_mph", "night_speed_vehicles"])
            spp["night_speed_mph"] = spp.night_speed_mph.astype(float).round(1)
            pt = pt.merge(spp, on=["DeviceId", "phase"], how="left")
        for c in PHASE_COLS:
            if c not in pt.columns:
                pt[c] = np.nan
        pt = pt[PHASE_COLS].sort_values(["DeviceId", "phase"]).reset_index(drop=True)
    else:
        pt = pd.DataFrame(columns=PHASE_COLS)
    return out, pt, "; ".join(notes)


def _shown(v: float, d0: int) -> str:
    """`v` with d0 decimals, or as many more (up to 6) as it needs to print exactly."""
    for d in range(d0, 7):
        if float(f"{v:.{d}f}") == v:
            return f"{v:.{d}f}"
    return f"{v:.6f}"


def _below(x: float, lim: float, d0: int) -> str:
    """`x` (< lim) printed so that it reads below the printed limit (e.g. 0.986 under a 0.99 limit printed as
    "0.99 (need >= 0.99)", 9.998 min under 10 as "only 10 min"): rounded to d0 decimals when that is already below,
    else cut (not rounded) at the first decimal that is -- 0.986, 9.9."""
    shown = float(_shown(lim, d0))
    if float(f"{x:.{d0}f}") < shown:
        return f"{x:.{d0}f}"
    for d in range(d0 + 1, 7):
        v = np.floor(x * 10 ** d) / 10 ** d
        if v < shown:
            return f"{v:.{d}f}"
    return f"{np.floor(x * 1e6) / 1e6:.6f}"


def _assemble(univ, facts, gate, top, fn, post, model_note: str,
              min_actuations: int = MIN_ACTUATIONS,
              min_prob: float = MIN_PROB) -> pd.DataFrame:
    key = ["DeviceId", "Detector"]
    res = univ.merge(facts[["DeviceId", "minutes_of_data", "n_candidate_phases",
                            "n_green_end", "n_calls"]], on="DeviceId", how="left")
    res = res.merge(top, on=key, how="left")
    res["phase_margin"] = res.phase_prob.fillna(0) - res.phase_2nd_prob.fillna(0)

    if fn is not None and len(fn):
        res = res.merge(fn[key + FN_COLS], on=key, how="left")
    for c in PROB_COLS + ["function_prob"]:
        if c not in res.columns:
            res[c] = np.nan
    if "function_pred" not in res.columns:
        res["function_pred"] = pd.NA

    if gate is not None and len(gate):
        res = res.merge(gate[key + ["gate_flag", "gate_reason"]], on=key, how="left")
    else:
        res["gate_flag"] = pd.NA
        res["gate_reason"] = pd.NA
    res["gate_flag"] = res.gate_flag.fillna("failed")
    res["gate_reason"] = res.gate_reason.fillna("no_events")
    res.loc[res.n_actuations.fillna(0) == 0, ["gate_flag", "gate_reason"]] = ["failed", "no_events"]
    if post is not None and len(post):
        res = res.merge(post, on=key, how="left")
    for c in POST_COLS:
        if c not in res.columns:
            res[c] = np.nan

    # the model's raw opinion is always kept, even where we refuse to answer
    res["phase_guess"] = res.phase_pred
    res["phase_guess_prob"] = res.phase_prob
    res["function_guess"] = res.function_pred
    res["function_guess_prob"] = res.function_prob

    # ---- status ------------------------------------------------------------
    status, review, reason = [], [], []
    for r in res.itertuples():
        notes = []
        if (r.n_actuations or 0) == 0:
            st = "cannot classify: no actuations"
            rv, rs = True, "no actuations"
        elif r.gate_flag == "failed" and r.gate_reason != "near_zero_volume":
            st = status_for_user(r.gate_reason)
            rv, rs = True, r.gate_reason
        elif not (r.n_candidate_phases or 0):
            st = ("cannot classify: no phase begin-green (event 1) records in this window")
            rv, rs = True, "no phase events"
        elif pd.isna(r.phase_pred):
            st = "cannot classify: no usable detector/phase evidence in this window"
            rv, rs = True, "no prediction"
        elif (r.n_actuations or 0) < min_actuations:
            n = int(r.n_actuations)
            st = (f"not enough data: {n} actuation" + ("" if n == 1 else "s") +
                  f" in sample (need >= {min_actuations})")
            rv, rs = True, "not enough data"
        elif min_prob > 0 and (r.phase_prob or 0) < min_prob:
            st = (f"not confident enough: phase probability "
                  f"{_below(float(r.phase_prob or 0), min_prob, 2)} (need >= {_shown(min_prob, 2)})")
            rv, rs = True, "not confident enough"
        else:
            rs = ""
            if (r.n_actuations or 0) < LOW_ACTUATIONS:
                n = int(r.n_actuations)
                notes.append(f"ok - low evidence ({n} actuation" +
                             ("" if n == 1 else "s") + ")")
                rs = "few actuations"
            if (r.minutes_of_data or 0) < LOW_MINUTES:
                notes.append(f"low evidence: only {_below(r.minutes_of_data, LOW_MINUTES, 0)} min of data")
                rs = rs or "short sample"
            if not (r.n_green_end or 0):
                notes.append("reduced accuracy: no green-termination events (7/8/9/10)")
                rs = rs or "no colour-state events"
            if not (r.n_calls or 0):
                notes.append("reduced accuracy: no phase call events (43/44)")
                rs = rs or "no 43/44 events"
            if r.health_status in ("suspect", "bad"):
                notes.append(f"detector health {r.health_status} (see health_reason)")
                rs = rs or f"health {r.health_status}"
            if (r.phase_prob or 0) < LOW_CONF:
                notes.append("low confidence: phase probability below 0.5")
                rs = "low confidence phase"
            st = "; ".join(notes) if notes else "ok"
            rv = bool(notes)
        status.append(st)
        review.append(rv)
        reason.append(rs)
    res["status"] = status
    res["review_flag"] = review
    res["review_reason"] = reason
    # refused channels carry no answer -- only a guess (health is still reported)
    dead = (res.status.str.startswith("cannot classify") |
            res.status.str.startswith("not enough data") |
            res.status.str.startswith("not confident enough"))
    res.loc[dead, ["phase_pred", "phase_prob", "phase_2nd", "phase_2nd_prob",
                   "function_pred", "function_prob", "phase_margin"] + PROB_COLS +
            ["lanes", "lane_conf", "phase_n_lanes", "phase_n_lanes_conf", "distance_ft",
             "setback_confidence", "night_speed_mph", "night_speed_vehicles"]] = np.nan
    if model_note:
        res["status"] = res.status + "; " + model_note

    for c in ("phase_pred", "phase_2nd", "phase_guess", "phase_n_lanes", "night_speed_vehicles"):
        res[c] = pd.to_numeric(res[c], errors="coerce").astype("Int64")
    res["lanes"] = res.lanes.where(res.lanes.notna() & res.lanes.ne(""), None)
    res["n_actuations"] = res.n_actuations.fillna(0).astype("Int64")
    res["minutes_of_data"] = res.minutes_of_data.astype(float).round(2)
    return res[OUT_COLS + EXTRA_COLS].sort_values(["DeviceId", "Detector"]).reset_index(
        drop=True)


# ---------------------------------------------------------------- entry points
def predict(events, start=None, end=None,
            device_ids=None, model_dir=None, threads: int = 4, memory: str = "4GB",
            chunk_signals: int | None = None, verbose: bool = False,
            min_actuations: int = MIN_ACTUATIONS,
            min_prob: float = MIN_PROB, return_phases: bool = False, siba_filter: bool | None = None,
            profile: str | None = None):
    """Raw hi-res events -> one row per detector channel.  Never raises on thin data.

    Parameters
    ----------
    events         pandas DataFrame, or a path / glob to parquet or csv, with columns
                   DeviceId, Timestamp, EventId, Parameter (lowercase variants accepted).
    start, end     optional timestamp strings (wall-clock time); `end` is exclusive.
    device_ids     optional list of DeviceIds to keep.
    model_dir      weight folder (default: the packaged weights; env DC_WEIGHTS).
    threads, memory  DuckDB threads / memory limit (onnxruntime: env DC_TREE_THREADS / DC_NET_THREADS, default 4).
    chunk_signals  load the signals in groups of this many to bound peak memory (answers are the same either way).
    min_actuations below this many detector ON events no answer is given.
    min_prob       below this top-phase probability no answer is given (0 = off).
                   The model's raw opinion is kept in phase_guess / function_guess.
    return_phases  also return the per-phase table (n_lanes, lane volumes, night speed) as a second value.
    siba_filter    run the network's function pass only on candidates with ranker p >= .01 (None = default ON).
    profile        accepted for compatibility and ignored (one path); passing it gives a FutureWarning, nothing else.
    """
    global _VERBOSE
    _VERBOSE = verbose
    _warn_profile(profile)
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not (model_dir / f"{PHASE_STEM}.json").exists():
        raise FileNotFoundError(f"no models in {model_dir}")
    if gb.config(model_dir) is None:
        raise FileNotFoundError(f"no network ({gb.CONFIG_FILE} and its weights) in {model_dir}")
    empty = (_empty_result(), pd.DataFrame(columns=PHASE_COLS))

    if chunk_signals:
        ids = device_ids or list_signals(events, start, end, threads, memory)
        if len(ids) > chunk_signals:
            parts = [predict(events, start, end, ids[i:i + chunk_signals],
                             model_dir, threads, memory, None, verbose, min_actuations,
                             min_prob, True, siba_filter)
                     for i in range(0, len(ids), chunk_signals)]
            dets = [a for a, _ in parts if len(a)]
            phs = [b for _, b in parts if len(b)]
            out = (pd.concat(dets, ignore_index=True) if dets else empty[0],
                   pd.concat(phs, ignore_index=True) if phs else empty[1])
            return out if return_phases else out[0]

    con = _connect(threads, memory)
    try:
        w0, w1, info = load_events(con, events, device_ids, start, end)
        if not info["n_events"]:
            return empty if return_phases else empty[0]
        ids = [r[0] for r in con.sql("SELECT DeviceId FROM devmap ORDER BY dev").fetchall()]
        if len(ids) > 1:
            # every signal on its own (its own window, sample length and tables): batching never changes an answer
            con.execute("ALTER TABLE evd RENAME TO evd_all")
        dets, phs = [], []
        for dv in ids:
            if len(ids) > 1:
                w0, w1, info = _one_signal(con, dv)
            r = _predict_loaded(con, w0, w1, start, end, model_dir, siba_filter, min_actuations, min_prob)
            if r is not None:
                dets.append(r[0])
                if len(r[1]):
                    phs.append(r[1])
        if not dets:
            return empty if return_phases else empty[0]
        res = dets[0] if len(dets) == 1 else pd.concat(dets, ignore_index=True)
        phases = (phs[0] if len(phs) == 1 else pd.concat(phs, ignore_index=True)) if phs else empty[1]
        return (res, phases) if return_phases else res
    finally:
        con.close()


def _predict_loaded(con, w0: float, w1: float, start, end, model_dir: Path, siba_filter,
                    min_actuations: int, min_prob: float):
    """The whole pipeline on the ONE signal loaded in `evd` / `devmap` -> (detector table, phase table), or None."""
    build_chunk_tables(con)
    univ = detector_universe(con)
    facts = signal_facts(con)
    if not len(univ):
        return None
    devmap = con.sql("SELECT dev, DeviceId FROM devmap").df()
    gate = gate_frame(con, w1 - w0, w1)
    note = ""
    # ---- the network's input (one network serves both tasks; it runs at every sample length)
    cfg = gb.config(model_dir)
    minutes = round((w1 - w0) / 60.0)       # a request for exactly two hours stays on the all-pieces path
    streams, pieces_rel = gb.streams_for(con, int(round(w0 * 1000)), int(round(w1 * 1000)),
                                         **gb.piece_plan(cfg, minutes))
    try:
        df, sim, lead, lag = build_features(con, w0, w1, devmap)
    except Exception as exc:                                  # pragma: no cover
        df, sim, lead, lag = None, None, None, None
        note = f"feature build failed ({type(exc).__name__})"
    ph = fnr = lph = None
    b0, b1 = _clip_window(*_bin_window(start, end, w0, w1), w0, w1)
    on_iv = on_table(con, b0, b1)
    if df is not None and len(df):
        log(f"features {df.shape}, similarity {sim.shape}")
        df = score(df, sim, model_dir, con, w0, w1, cfg, streams, pieces_rel, lead)
        ph = df[["DeviceId", "Detector", "cand_phase", "prob"]].copy()
        s = ph.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
        ph["prob"] = ph.prob / s.replace(0, np.nan)
        ph = ph.dropna(subset=["prob"])
        try:
            fnr, lph = score_function(df, model_dir, con, b0, b1, streams, pieces_rel, siba_filter, lag, devmap,
                                      on_iv)
        except Exception as exc:                              # pragma: no cover
            note = (note + "; " if note else "") + \
                f"function model unavailable ({type(exc).__name__}: {exc})"
        del df, sim, lead, lag
    streams = None
    top = _top_phase(ph)
    post, phases, pnote = post_outputs(con, w0, w1, b0, b1, univ, top, fnr, lph, model_dir, on_iv)
    log("lanes, setback, health and night speed done")
    if pnote:
        note = (note + "; " if note else "") + pnote
    res = _assemble(univ, facts, gate, top, fnr, post, note, min_actuations, min_prob)
    return res, phases


def list_signals(events, start=None, end=None, threads: int = 2,
                 memory: str = "2GB") -> list[str]:
    """DeviceIds present in `events` (after the time filter), sorted."""
    con = _connect(threads, memory)
    try:
        rel, ts = _source(con, events)
        where = ["1=1"]
        if start:
            where.append(f"TRY_CAST({ts} AS TIMESTAMP) >= {_wall(start)}")
        if end:
            where.append(f"TRY_CAST({ts} AS TIMESTAMP) < {_wall(end)}")
        return [r[0] for r in con.sql(
            f"SELECT DISTINCT CAST(DeviceId AS VARCHAR) FROM {rel} "
            f"WHERE DeviceId IS NOT NULL AND {' AND '.join(where)} ORDER BY 1").fetchall()]
    finally:
        con.close()


def run(events: str, out: str, device_ids=None, start=None, end=None,
        model_dir=None, threads: int = 4,
        memory: str = "4GB", chunk_signals: int | None = None,
        min_actuations: int = MIN_ACTUATIONS,
        min_prob: float = MIN_PROB, out_phases: str | None = None,
        siba_filter: bool | None = None, profile: str | None = None, verbose: bool = True,
        out_atspm: str | None = None) -> pd.DataFrame:
    """CLI helper: predict and write a CSV (and, if asked, the per-phase lane table and the atspm detector
    configuration from `to_atspm_config` with its defaults)."""
    t0 = time.time()
    res, phases = predict(events, start, end, device_ids, model_dir, threads,
                          memory, chunk_signals, verbose, min_actuations, min_prob, True, siba_filter, profile)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    if out_phases:
        Path(out_phases).parent.mkdir(parents=True, exist_ok=True)
        phases.to_csv(out_phases, index=False)
    if out_atspm:
        from .atspm_export import to_atspm_config
        Path(out_atspm).parent.mkdir(parents=True, exist_ok=True)
        cfg = to_atspm_config(res)
        cfg.to_csv(out_atspm, index=False)
        print(f"wrote {out_atspm}: {len(cfg)} atspm detector_config rows", flush=True)
    print(f"wrote {out}: {len(res)} detectors, "
          f"{res.DeviceId.nunique() if len(res) else 0} signals, "
          f"{time.time()-t0:.1f}s", flush=True)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="atspm-detector", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, help="parquet/csv path or glob")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--out-phases", default=None,
                    help="optional CSV of the per-phase table (n_lanes per predicted phase)")
    ap.add_argument("--out-atspm", default=None,
                    help="optional CSV of the detector configuration for the atspm package (DeviceId, Phase, "
                         "Parameter, Function: Advance / Presence / Yellow_Red / 'Stopbar Count'; detectors with "
                         "health 'bad' left out, at most one detector per phase, function and lane)")
    ap.add_argument("--device-ids", default=None, help="comma separated DeviceId filter")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None, help="exclusive")
    ap.add_argument("--models", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", default="4GB")
    ap.add_argument("--chunk-signals", type=int, default=None,
                    help="process this many signals at a time to bound memory")
    ap.add_argument("--min-actuations", type=int, default=MIN_ACTUATIONS,
                    help="below this many detector ON events in the sample, report "
                         "'not enough data' instead of an answer (default %(default)s; "
                         "use 1 to always answer)")
    ap.add_argument("--min-prob", type=float, default=MIN_PROB,
                    help="below this top-phase probability, report 'not confident enough' "
                         "instead of an answer (default %(default)s = off)")
    ap.add_argument("--no-siba-filter", action="store_true",
                    help="run the network's function pass on every candidate (default: ranker p >= .01 only)")
    ap.add_argument("--profile", default=None, help=argparse.SUPPRESS)   # accepted and ignored, warns
    ap.add_argument("--quiet", action="store_true", help="no progress lines")
    a = ap.parse_args(argv)
    ids = [s.strip() for s in a.device_ids.split(",")] if a.device_ids else None
    run(a.events, a.out, ids, a.start, a.end, Path(a.models),
        a.threads, a.memory, a.chunk_signals, a.min_actuations, a.min_prob, a.out_phases,
        False if a.no_siba_filter else None, a.profile, not a.quiet, a.out_atspm)


if __name__ == "__main__":
    main()
