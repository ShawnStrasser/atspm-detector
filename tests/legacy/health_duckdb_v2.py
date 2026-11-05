# FROZEN ORACLE (stage 0 of the incremental-health design): src/atspm_detector/health_duckdb.py as of 2026-10-29 (ped round 5 + ped movement
# model merged).  Test-only; never edited.  Changes against the original: imports and weight paths only.
"""Detector health in DuckDB SQL (note 153): the six health checks of 1.1.0 as plain SQL (ported from the numpy modules
health_core / health_v4 / health_v4_stats / health_counts, removed in 1.1.1; names below refer to them), vectorised
across signals (the signal key in every GROUP BY / PARTITION BY).  predict() runs it on its loaded events.

    health_sql(events_table="events", detectors_table="detectors", start=None, end=None) -> str   the query text
    health(con, events_table="events", detectors_table="detectors", start=None, end=None) -> DataFrame
    health(con, ..., periods=True) -> (detectors, periods)    + one row per period (health_sql(..., periods=True):
                                                                the two query texts)

The query reads the caller's tables where they are (no copy):
  events     DeviceId, Timestamp, EventId, Parameter (the controller log, local time)
  detectors  DeviceId, Detector, phase, function, lanes [, p_advance, p_presence, p_count, p_yellow_red, p_other,
             p_mid, p_bike] [, configured_function] -- the classifier's answers (predicted phase, function, lane span,
             class probabilities) and, optional, the function the agency configured (only the pulse-mode check reads it:
             a zone configured as Count / Advance / Yellow_Red may pulse)
and returns one row per detector with actuations in the window: DeviceId, Detector, health_status, health_score,
health_reason, health_categories, health_config, health_watch, health_signal_note; with periods=True also one row per
period: DeviceId, Detector, start, end, condition, severity, limit_s, in_closure, whole_window (note 154aa).

Same checks, same limits (weights/health/*.json, injected as VALUES tables), same order of decisions as the numpy
version had; sums and medians are DuckDB's, so a measure may differ from it in the last bits (accepted, user 2026-10-20).
health_sql gives one WITH query (steps named hq_*); health() runs the same steps as temp tables (a step used once is
written into the step that uses it), which saves DuckDB's half second of planning the whole query.
Layout (each block a few steps):
  inputs     windows, events (allowed codes, de-duplicated), comms gaps, covered 5-min bins
  per ON     detector events in (time, ON before OFF) order with their neighbours (lag / lead); continuous ONs
  bins       5-min counts / time ON per detector (dense grid) with running sums for window sums
  stage 1    sudden silence, stuck-on, night count, level shift -> pass-1 status (twice: pass 2 without the
             detectors pass 1 calls bad as yardsticks)
  measures   15-min occupancy context, Count zone held ON, per-ON statistics, too fast / too many, hourly GEH,
             spiky bursts, holds with their queue context, busy at night
  verdict    limits, scores, resolver / severity (CASE logic), categories, reason text; the periods table
"""
from __future__ import annotations

import importlib.util
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

WDIR = Path(importlib.util.find_spec("atspm_detector").origin).parent / "weights" / "health"
ALLOWED = (1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173)
FNS = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")
PROB = {"Advance": "p_advance", "Presence": "p_presence", "Count": "p_count", "Yellow_Red": "p_yellow_red",
        "Other": "p_other", "Mid": "p_mid", "Bike": "p_bike"}
NS5 = 300_000_000_000                    # 5 min in ns
NSH = 3_600_000_000_000                  # 1 h in ns
NSQ = NSH // 4                           # 15 min in ns
# note 154 changes, each switchable (all off = 1.2.0 exactly); lane_pairs: partners of a lane closure besides the
# detector's own phase -- None (own phase only), "nema" (2-5, 1-6, 3-8, 4-7) or "any" (every phase of the signal)
OPTIONS = {"silent_limits": True, "silent_merge": True, "no_recovery_cap": True, "basic_periods": True,
           "signal_silent": True, "phase_drop": True, "drop_end": True,
           "drop_no_mates": True, "long_zone": True, "lane_closure": True, "lane_pairs": "nema", "plan_switch": True,
           "all_silences": True, "own_history": True,
           "no_daily_pattern": True, "busy_ease": True,   # approved 2026-10-22
           # approved 2026-10-28 (notes 154n-o, 154aa): hour-by-hour drops / rises against each phase mate
           "mate_drop": True, "count_rise": True, "mate_noise": True,
           # approved 2026-10-28 (notes 154x-y, 154aa): no "counts too high" on a day when the detector nearly stopped
           # counting on the window's other days (_mate_hours)
           "rise_dead_days": True,
           # approved 2026-10-29 (notes 154aa, 154ac): lane-closure candidates also from near-zero runs of >= 3
           # quarter-hours (any closure of an hour or more), not only of >= 2 judged hours; and no closure on a day when
           # the detector has faults of its own before and after it (LC_FAULT_DAY) (_closure)
           "closure_1h": True,
           # approved 2026-10-29 (notes 154aa, 154ac): a Presence detector whose calls are nearly all pulses (PULSE)
           # looks set to pulse mode -> suspect at least, category "Configuration"; count, advance and yellow-red zones
           # may pulse: one configured as such (configured_function, when given) is never flagged (_outputs)
           "pulse_mode": True,
           # approved 2026-10-28 (notes 154q, 154w): time ON with nothing imputed (_strict_ons) -- an ON then an ON = still
           # ON unless a signal-wide logging gap lies between (then unknown); OFF then OFF = missing data (the second OFF
           # ignored); a leading OFF dropped.  False: atspm-style imputation as 1.2.0
           "occ_strict": True,
           # approved 2026-10-28 (notes 154t, 154w, 154aa): detectors that do not look like traffic are nobody's yardstick
           # for the night checks (True) and the count checks (hourly yardstick gh, mate_drop / count_rise mates: "all")
           "clean_yardsticks": "all"}
OPTIONS_OFF = {k: (None if k == "lane_pairs" else False) for k in OPTIONS}
NEMA_PAIRS = ((2, 5), (1, 6), (3, 8), (4, 7))
# note 154n: hour-by-hour drop / rise against each phase mate (options mate_drop, count_rise)
MATE_HOURS = {"low": 0.5, "high": 2.0, "own_low": 0.75, "own_high": 1.5, "min_exp": 8.0, "min_extra": 15.0,
              "min_mate": 10.0, "high_share": 0.6, "hours": 3, "geh": (15.0, 20.0), "occ": 0.8, "z": 2.5, "phi": 1.5}
# note 154t: a detector that does not look like traffic is nobody's yardstick for the night checks (option
# clean_yardsticks): night (01-05) per hour >= own x its day (07-19) per hour, while its phase mates
# (their day level < rest_min an hour or no mates: the rest of the signal) keep <= rest x theirs (rest, own_min: the no-daily-
# pattern limits); it counts (>= own_min an hour by day), the mates carry traffic (>= rest_min an hour by day); >= 2
# covered hours of each
CLEAN_YARD = {"own": 0.5, "rest": 0.3, "own_min": 1.5, "rest_min": 20.0}
# note 154aa (option pulse_mode): a Presence detector with >= n complete calls in the window, >= share of them lasting
# s seconds or less (the log's 0.1-s ticks: a ~0.13-s pulse logs as 0.1 or 0.2 s) and a median call under s
# (2B033 det 11: every call ~0.13 s, ~0 % ON)
PULSE = {"n": 200, "share": 0.8, "s": 0.2}
# note 154ac (with closure_1h): a lane closure with faults of the detector's own on the same calendar day on both sides
# (sides 2; 1 = either side) is part of the fault; all: every closure (False: only the quarter-hour candidates)
LC_FAULT_DAY = {"sides": 2, "all": True}
MAX_DAYS = 92                           # longest window health() takes (a guard, not a dtype limit: bins INT32,
                                         # times BIGINT ns); memory ~15 MB per signal-day (note 154)


@lru_cache(maxsize=2)
def _json(name: str) -> dict:
    with open(WDIR / name, encoding="utf-8") as f:
        return json.load(f)


def _f(x) -> str:
    """a float as a DOUBLE literal."""
    return f"CAST({float(x)!r} AS DOUBLE)" if x == x else "CAST('nan' AS DOUBLE)"


def _s(x) -> str:
    return "'" + str(x).replace("'", "''") + "'"


def _values(cols: str, rows: list) -> str:
    """a VALUES table (cols: 'a, b, c'); rows of SQL literal text."""
    if not rows:
        n = len(cols.split(","))
        return f"SELECT * FROM (VALUES ({', '.join(['NULL'] * n)})) t({cols}) WHERE false"
    return f"SELECT * FROM (VALUES {', '.join('(' + ', '.join(r) + ')' for r in rows)}) t({cols})"


PREFIX = "hq_"


def _options(opt: dict | None) -> dict:
    o = {**OPTIONS, **(opt or {})}
    bad = set(o) - set(OPTIONS)
    if bad:
        raise ValueError(f"unknown health option(s): {sorted(bad)}")
    if o["lane_pairs"] not in (None, "nema", "any"):
        raise ValueError("lane_pairs must be None, 'nema' or 'any'")
    if not isinstance(o["occ_strict"], bool):
        raise ValueError("occ_strict must be True or False")
    if o["clean_yardsticks"] not in (False, True, "all"):
        raise ValueError("clean_yardsticks must be False, True or 'all'")
    return o


class _Q:
    """an ordered list of named steps (CTEs) -> one WITH query, or a script of temp tables.  Step names are prefixed
    with PREFIX when rendered, so they never meet a table of the caller's."""

    def __init__(self, opt: dict | None = None):
        self.ctes: list = []
        self.subs: dict = {}          # @EVENTS@ etc. -> the caller's table names, put in after the renaming
        self.opt: dict = _options(opt)
        self.periods: bool = False    # build the periods-table steps too (note 154aa)

    def add(self, name: str, sql: str, mat: bool = False):
        self.ctes.append((name, sql.strip(), mat))

    def _renamer(self):
        names = sorted((n for n, _, _ in self.ctes), key=len, reverse=True)
        rx = re.compile(r"\b(" + "|".join(names) + r")\b")

        def rn(sql):
            sql = rx.sub(lambda m: PREFIX + m.group(1), sql)
            for k, v in self.subs.items():
                sql = sql.replace(k, v)
            return sql
        return rn

    def text(self, final: str) -> str:
        rn = self._renamer()
        body = ",\n".join(f"{PREFIX}{n} AS {'MATERIALIZED ' if m else ''}(\n{rn(s)}\n)" for n, s, m in self.ctes)
        return f"WITH {body}\n{rn(final)}"

    def script(self, finals) -> tuple:
        """(statements, drops): a step used by more than one later step becomes a temp table, a step used once is
        written into the statement that uses it; drops[i] = the temp tables no longer needed after statement i.
        `finals`: the final query, or a list of them (the last statements, in that order)."""
        finals = [finals] if isinstance(finals, str) else list(finals)
        fk = [f"__final{i}" for i in range(len(finals))]
        rn = self._renamer()
        names = [n for n, _, _ in self.ctes]
        body = {n: s for n, s, _ in self.ctes}
        words = {n: set(re.findall(r"\w+", s)) for n, s in body.items()}
        for k, f in zip(fk, finals):
            words[k] = set(re.findall(r"\w+", f))
        uses = {n: [] for n in names}
        for n in names + fk:
            for o in names:
                if o != n and o in words[n]:
                    uses[o].append(n)
        table = {n for n in names if len(uses[n]) != 1}

        def deps(n, seen):
            out = []
            for o in names:
                if o in table or o in seen or o not in words[n]:
                    continue
                seen.add(o)
                out += deps(o, seen) + [o]
            return out
        stmts, made = [], []
        for n in names:
            if n not in table:
                continue
            inl = sorted(set(deps(n, set())), key=names.index)
            w = ("WITH " + ",\n".join(f"{PREFIX}{o} AS (\n{rn(body[o])}\n)" for o in inl) + "\n") if inl else ""
            stmts.append(f"CREATE OR REPLACE TEMP TABLE {PREFIX}{n} AS {w}{rn(body[n])}")
            made.append(n)
        for k, f in zip(fk, finals):
            inl = sorted(set(deps(k, set())), key=names.index)
            w = ("WITH " + ",\n".join(f"{PREFIX}{o} AS (\n{rn(body[o])}\n)" for o in inl) + "\n") if inl else ""
            stmts.append(w + rn(f))
        # last statement that reads each table (directly, or through a step written into it)
        last = {}
        for i, st in enumerate(stmts):
            w = set(re.findall(r"\w+", st.split(" AS ", 1)[1] if i < len(made) else st))
            for n in made:
                if PREFIX + n in w:
                    last[n] = i
        drops = [[PREFIX + n for n in made if last.get(n, made.index(n)) == i] for i in range(len(stmts))]
        return stmts, drops


# ============================================================================ inputs and per-ON layer
def _nn(x: str) -> str:
    """a DOUBLE with NaN as NULL."""
    return f"(CASE WHEN isnan(({x})::DOUBLE) THEN NULL ELSE ({x})::DOUBLE END)"


def _inputs(q: _Q, events: str, detectors: str, windows: str | None, start, end, probabilities: bool = True,
            distinct: bool = True, configured: bool = False):
    codes = ",".join(str(c) for c in ALLOWED)
    good = f"EventId IN ({codes}) AND (EventId NOT IN (81, 82) OR Parameter BETWEEN 1 AND 64)"
    q.subs.update({"@EVENTS@": events, "@DETECTORS@": detectors, "@DISTINCT@": "DISTINCT" if distinct else ""})
    events, detectors = "@EVENTS@", "@DETECTORS@"
    if windows is not None:
        q.subs["@WINDOWS@"] = windows
        windows = "@WINDOWS@"
    if windows is None:
        # the sample window of each signal: start / end where given, else its first event and one second past its
        # last; a bound more than 30 min outside the signal's data is treated as not given (pipeline._windows)
        b0 = "w0" if start is None else str(_ns(start))
        b1 = "w1" if end is None else str(_ns(end))
        rng = []
        if start is not None:
            rng.append(f"Timestamp >= {_ts_lit(start)}")
        if end is not None:
            rng.append(f"Timestamp < {_ts_lit(end)}")
        where = " AND ".join([good, "DeviceId IS NOT NULL", "Timestamp IS NOT NULL"] + rng)
        q.add("wraw", f"""
SELECT DeviceId, epoch_ns(min(Timestamp)) AS w0, epoch_ns(max(Timestamp)) + 1000000000 AS w1, count(*) AS n
FROM {events} WHERE {where} GROUP BY DeviceId""")
        q.add("wbin", f"""
SELECT DeviceId, w0, w1, CASE WHEN {b1} > {b0} THEN {b0} ELSE w0 END AS b0,
       CASE WHEN {b1} > {b0} THEN {b1} ELSE w1 END AS b1 FROM wraw""")
        q.add("wsel", """
SELECT DeviceId, CASE WHEN w0 - b0 > 1800000000000 THEN w0 ELSE b0 END AS s_ns,
       CASE WHEN b1 - w1 > 1800000000000 THEN w1 ELSE b1 END AS e_ns FROM wbin""")
        windows = "wsel"
    # one row per signal: its window [s_ns, e_ns) (epoch ns), length T (s), 5-min bins nb5, length class wg and the
    # occupancy bin size bs (15 min, 5 min for a sample under ~1.2 h)
    q.add("win", f"""
SELECT dev, DeviceId, s_ns, e_ns, T, T / 3600.0 AS hours, ceil(T / 300.0)::INT AS nb5, wg,
       CASE WHEN wg = 'm30' THEN 300 ELSE 900 END AS bs,
       ceil(T / CASE WHEN wg = 'm30' THEN 300.0 ELSE 900.0 END)::INT AS nbb,
       make_timestamp(s_ns // 1000) AS lo, make_timestamp(e_ns // 1000 + 1) AS hi,
       s_ns % 86400000000000 AS sod
FROM (SELECT (dense_rank() OVER (ORDER BY CAST(DeviceId AS VARCHAR)))::INT AS dev, DeviceId,
             s_ns::BIGINT AS s_ns, e_ns::BIGINT AS e_ns, (e_ns - s_ns) / 1e9 AS T,
             CASE WHEN (e_ns - s_ns) / 3.6e12 < sqrt(1.5) THEN 'm30'
                  WHEN (e_ns - s_ns) / 3.6e12 < sqrt(72.0) THEN 'h3' ELSE 'h24' END AS wg
      FROM {windows})""", mat=True)
    # allowed events inside the window, de-duplicated; dn = ns from the window start
    q.add("ev", f"""
SELECT @DISTINCT@ w.dev, epoch_ns(e.Timestamp) - w.s_ns AS dn, e.EventId::INT AS eid, e.Parameter::INT AS par
FROM {events} e JOIN win w ON e.DeviceId = w.DeviceId
WHERE e.{good.replace(' Parameter', ' e.Parameter').replace('(EventId', '(e.EventId')}
  AND e.Timestamp >= w.lo AND e.Timestamp <= w.hi
  AND epoch_ns(e.Timestamp) >= w.s_ns AND epoch_ns(e.Timestamp) < w.e_ns""", mat=True)
    # the classifier's answers per detector (phase, function, lane span, class probabilities)
    pc = ", ".join((_nn(f"d.{c}") if probabilities else "NULL::DOUBLE") + f" AS {c}" for c in PROB.values())
    # note 154ac: the configured function (optional column configured_function), read by the pulse-mode check only
    pc += ", " + ("NULLIF(trim(CAST(d.configured_function AS VARCHAR)), '')" if configured else "NULL::VARCHAR") + " AS cfn"
    q.add("dcl", f"""
SELECT w.dev, d.Detector::INT AS det, {_nn('d.phase')} AS ph, CAST(d.function AS VARCHAR) AS fn,
       {_nn('d.lanes')} AS lanes, {pc}
FROM {detectors} d JOIN win w ON CAST(d.DeviceId AS VARCHAR) = CAST(w.DeviceId AS VARCHAR)""", mat=True)
    # covered 5-min bins (any event) and comms gaps (no event of any code for > 120 s)
    q.add("evm", "SELECT dev, dn // 60000000000 AS m, min(dn) AS a, max(dn) AS z FROM ev GROUP BY dev, m", mat=True)
    q.add("cov5", "SELECT DISTINCT dev, (m // 5)::INT AS b FROM evm", mat=True)
    q.add("gaps", """
SELECT dev, t AS g0 FROM (SELECT dev, z / 1e9 AS t, lead(a) OVER (PARTITION BY dev ORDER BY m) / 1e9 AS nt FROM evm)
WHERE nt - t > 120.0""", mat=True)
    # detector events in (time, ON before OFF) order; isl = OFFs before the event: a continuous ON (an ON, further
    # ONs logged without an OFF, the OFF that ends it) is one island
    q.add("de", """
SELECT dev, par AS det, dn, eid, lag(eid) OVER w AS pe, lag(dn) OVER w AS pdn, lead(eid) OVER w AS ne,
       lead(dn) OVER w AS ndn, lag(eid, 2) OVER w AS ppe, lag(dn, 2) OVER w AS p2dn, lag(eid, 3) OVER w AS p3e,
       lead(eid, 2) OVER w AS nne, lead(dn, 2) OVER w AS n2dn
FROM ev WHERE eid IN (81, 82) WINDOW w AS (PARTITION BY dev, par ORDER BY dn, eid DESC)""", mat=True)
    q.add("dets", "SELECT DISTINCT dev, det FROM de", mat=True)
    # continuous ONs (an ON after a non-ON, further ONs logged without an OFF, the OFF that ends it): start s, end f
    # (the OFF, else the window end), open (no OFF), multi (ONs logged again inside it).  Single-ON ones straight from
    # the event order; the few with re-logged ONs are numbered among their own events.
    q.add("mst", "SELECT dev, det, dn FROM de WHERE eid = 82 AND pe IS DISTINCT FROM 82 AND ne = 82", mat=True)
    q.add("multis", """
SELECT s.dev, s.det, s.dn AS on_dn, e.dn AS off_dn
FROM mst s LEFT JOIN (SELECT e.dev, e.det, e.dn, m.dn AS sdn
                      FROM (SELECT dev, det, dn FROM de WHERE eid = 81 AND pe = 82 AND ppe = 82) e
                      ASOF JOIN mst m ON m.dev = e.dev AND m.det = e.det AND e.dn >= m.dn) e
  ON e.dev = s.dev AND e.det = s.det AND e.sdn = s.dn""")
    if q.opt["occ_strict"]:
        _strict_ons(q)
        return
    q.add("isl0", """
SELECT d.dev, d.det, d.dn / 1e9 AS s, coalesce(d.ndn / 1e9, w.T) AS f, d.ne IS NULL AS open_end, false AS multi
FROM de d JOIN win w USING (dev) WHERE d.eid = 82 AND d.pe IS DISTINCT FROM 82 AND d.ne IS DISTINCT FROM 82
UNION ALL
SELECT m.dev, m.det, m.on_dn / 1e9, coalesce(m.off_dn / 1e9, w.T), m.off_dn IS NULL, true
FROM multis m JOIN win w USING (dev)""")
    q.add("ons", """
SELECT c.dev, c.det, c.s, c.f, c.open_end, c.multi, c.f - c.s AS dur, false AS gap FROM isl0 c WHERE c.f - c.s <= 60
UNION ALL
SELECT c.dev, c.det, c.s, c.f, c.open_end, c.multi, c.f - c.s, coalesce(g.g0 < c.f, false)
FROM (SELECT * FROM isl0 WHERE f - s > 60) c ASOF LEFT JOIN gaps g ON c.dev = g.dev AND g.g0 >= c.s""", mat=True)
    # a channel whose first event is an OFF was ON when the window opened
    q.add("lead0", "SELECT dev, det, dn / 1e9 AS f FROM de WHERE eid = 81 AND pe IS NULL")


def _strict_ons(q: _Q):
    """note 154q (option occ_strict, the "held" form approved 2026-10-28): continuous ONs with nothing imputed.  A chain
    = an ON and the ONs re-logged after it without an OFF (the controller re-logs ON while an input is held); it ends at
    the OFF after it (a leading OFF and a second OFF add nothing) or at the window end when nothing follows.  Chains over
    a comms gap or running into the signal's own silence at the window end carry gap = true: their time ON is unknown,
    left out of time ON, holds and the per-ON statistics."""
    rel_p = "coalesce(pe = 82, false)"     # re-logged: the previous event was an ON
    rel_n = "coalesce(ne = 82, false)"     # the next event is an ON
    q.add("cst", f"SELECT dev, det, dn FROM de WHERE eid = 82 AND NOT {rel_p} AND {rel_n}", mat=True)
    q.add("isl0", f"""
SELECT d.dev, d.det, d.dn / 1e9 AS s, coalesce(d.ndn / 1e9, w.T) AS f, d.ne IS NULL AS open_end, false AS multi
FROM de d JOIN win w USING (dev) WHERE eid = 82 AND NOT {rel_p} AND NOT {rel_n}
UNION ALL
SELECT e.dev, e.det, m.dn / 1e9, coalesce(e.ndn / 1e9, w.T), e.ne IS NULL, true
FROM (SELECT dev, det, dn, ne, ndn FROM de WHERE eid = 82 AND {rel_p} AND NOT {rel_n}) e
ASOF JOIN cst m ON m.dev = e.dev AND m.det = e.det AND e.dn >= m.dn JOIN win w ON w.dev = e.dev""")
    # the signal's last event: an open chain is cut by the signal's own silence at the window end (> 120 s)
    q.add("isl1", """
SELECT c.*, c.open_end AND w.T - l.z > 120.0 AS lost
FROM isl0 c JOIN win w USING (dev) JOIN (SELECT dev, max(z) / 1e9 AS z FROM evm GROUP BY dev) l USING (dev)""")
    q.add("ons", """
SELECT c.dev, c.det, c.s, c.f, c.open_end, c.multi, c.f - c.s AS dur, c.lost AS gap FROM isl1 c WHERE c.f - c.s <= 60
UNION ALL
SELECT c.dev, c.det, c.s, c.f, c.open_end, c.multi, c.f - c.s, c.lost OR coalesce(g.g0 < c.f, false)
FROM (SELECT * FROM isl1 WHERE f - s > 60) c ASOF LEFT JOIN gaps g ON c.dev = g.dev AND g.g0 >= c.s""", mat=True)
    q.add("lead0", "SELECT dev, det, dn / 1e9 AS f FROM de WHERE false")


def _ns(x) -> int:
    t = pd.Timestamp(x)
    t = t.tz_localize(None) if t.tzinfo is not None else t
    return int(t.value)


def _ts_lit(x) -> str:
    t = pd.Timestamp(x)
    t = t.tz_localize(None) if t.tzinfo is not None else t
    return f"TIMESTAMP '{t.isoformat(sep=' ')}'"


# ============================================================================ 5-min bins
def _qocc(q: _Q, a: str = "") -> str:
    """the time ON a detector's quiet bin (no ON, < 1 s ON) is judged on; occ_strict: time of unknown state counts too
    (not quiet).  The signal-wide silence keeps the known time only: inputs lost while ON re-log an ON when they return."""
    return f"({a}occ + {a}unk)" if q.opt["occ_strict"] else f"{a}occ"


def _bins(q: _Q):
    # time ON split across 5-min bins: continuous ONs not over a comms gap + the ON at the window start
    split = """
WITH o AS (SELECT o.dev, o.det, o.s, o.f, floor(o.s / 300.0)::INT AS b0, least(floor(o.f / 300.0)::INT, w.nb5 - 1) AS b1
           FROM (@SRC@) o
           JOIN win w USING (dev))
SELECT dev, det, b, sum(occ) AS occ FROM (
  SELECT dev, det, b0 AS b, greatest(least(f, (b0 + 1) * 300.0) - s, 0.0) AS occ FROM o WHERE b1 = b0
  UNION ALL
  SELECT dev, det, b, greatest(least(f, (b + 1) * 300.0) - greatest(s, b * 300.0), 0.0)
  FROM (SELECT dev, det, s, f, unnest(range(b0, b1 + 1)) AS b FROM o WHERE b1 > b0))
GROUP BY dev, det, b"""
    q.add("occ5", split.replace("@SRC@", "SELECT dev, det, s, f FROM ons WHERE NOT gap UNION ALL SELECT dev, det, 0.0, f FROM lead0"))
    strict = q.opt["occ_strict"]
    if strict:
        # note 154q: time whose ON / OFF state is unknown (lost OFF, comms gap, signal stopped) -- neither ON nor quiet
        q.add("unk5", split.replace("@SRC@", "SELECT dev, det, s, f FROM ons WHERE gap"))
    q.add("n5", f"SELECT dev, det, (dn // {NS5})::INT AS b, count(*) AS n FROM de WHERE eid = 82 GROUP BY dev, det, b")
    q.add("bins5", f"""
SELECT b.dev, b.b, c.b IS NOT NULL AS cov, ((b.s_ns + b.b * {NS5}) // {NSH}) % 24 AS hr
FROM (SELECT dev, s_ns, unnest(range(nb5)) AS b FROM win) b LEFT JOIN cov5 c ON c.dev = b.dev AND c.b = b.b""",
          mat=True)
    # dense detector x 5-min grid: count n, time ON occ (s), covered
    uc = ", coalesce(u.occ, 0.0) AS unk" if strict else ""
    uj = " LEFT JOIN unk5 u ON u.dev = d.dev AND u.det = d.det AND u.b = b.b" if strict else ""
    q.add("g5", f"""
SELECT d.dev, d.det, b.b, b.cov, b.hr, coalesce(n.n, 0)::DOUBLE AS n, coalesce(o.occ, 0.0) AS occ{uc}
FROM dets d JOIN bins5 b USING (dev) LEFT JOIN n5 n ON n.dev = d.dev AND n.det = d.det AND n.b = b.b
LEFT JOIN occ5 o ON o.dev = d.dev AND o.det = d.det AND o.b = b.b{uj}""", mat=True)
    # running sums at the bin boundaries k = 0 .. nb5 (sum over bins < k): counts cn, quiet bins cq (no ON, < 1 s ON);
    # looked up one boundary per CTE (_at), so no join order can pair two lookups of the same detector
    q.add("k5", f"""
SELECT dev, det, b + 1 AS k, sum(n) OVER w AS cn, sum((n = 0 AND {_qocc(q)} < 1)::INT) OVER w AS cq
FROM g5 WINDOW w AS (PARTITION BY dev, det ORDER BY b ROWS UNBOUNDED PRECEDING)
UNION ALL SELECT dev, det, 0, 0.0, 0 FROM dets""", mat=True)
    # per detector: ONs on covered bins, active, live, day / night ONs
    q.add("dstat", """
SELECT g.dev, g.det, sum(g.n * g.cov::INT) AS n_on, (sum(g.n) > 0 OR sum(g.occ) > 0) AS active,
       sum(g.n * g.cov::INT) > 0 AS live,
       sum(g.n * (g.cov AND g.hr >= 7 AND g.hr < 19)::INT) AS dy, sum(g.n * (g.cov AND (g.hr >= 21 OR g.hr < 5))::INT) AS nt
FROM g5 g GROUP BY g.dev, g.det""", mat=True)
    q.add("sig0", """
SELECT s.dev, s.n_cov, s.last_cov, s.ok_night, a.n_active
FROM (SELECT dev, count(*) FILTER (cov) AS n_cov, coalesce(max(b) FILTER (cov), -1) AS last_cov,
             count(*) FILTER (cov AND (hr >= 21 OR hr < 5)) >= 24 AND count(*) FILTER (cov AND hr >= 7 AND hr < 19) >= 24
             AS ok_night FROM bins5 GROUP BY dev) s
JOIN (SELECT dev, sum(active::INT) AS n_active FROM dstat GROUP BY dev) a USING (dev)""", mat=True)
    # twins: two channels carrying (almost) the same 5-min counts = one zone wired to two inputs
    q.add("twin", """
SELECT a.dev, a.det AS i, b.det AS j
FROM dstat a JOIN dstat b ON a.dev = b.dev AND a.det <> b.det AND a.n_on > 0 AND b.n_on > 0
     AND abs(a.n_on - b.n_on) < 0.1 * greatest(a.n_on + b.n_on, 1)
JOIN g5 x ON x.dev = a.dev AND x.det = a.det JOIN g5 y ON y.dev = a.dev AND y.det = b.det AND y.b = x.b
GROUP BY a.dev, a.det, b.det, a.n_on, b.n_on
HAVING sum(abs(x.n * x.cov::INT - y.n * y.cov::INT)) / greatest(a.n_on + b.n_on, 1) < 0.1""", mat=True)
    # 15-min bins of whole 5-min triples: counts, all three covered (partner correlation, level shift)
    q.add("a15", """
SELECT g.dev, g.det, g.b // 3 AS k, sum(g.n) AS n15, bool_and(g.cov) AS c15
FROM g5 g JOIN win w USING (dev) WHERE g.b < w.nb5 // 3 * 3 GROUP BY g.dev, g.det, g.b // 3""", mat=True)
    # quiet runs (no ON, < 1 s ON, covered) of every detector with ONs: [a, z) in 5-min bins
    raw = f"""
SELECT dev, det, min(b) AS a, max(b) + 1 AS z FROM (
  SELECT g.dev, g.det, g.b, g.b - row_number() OVER (PARTITION BY g.dev, g.det ORDER BY g.b) AS grp
  FROM g5 g JOIN dstat s USING (dev, det) WHERE s.n_on > 0 AND g.n = 0 AND {_qocc(q, 'g.')} < 1 AND g.cov)
GROUP BY dev, det, grp"""
    if q.opt["silent_merge"]:
        # note 154: a quiet run broken by one daytime (05-21 h) 5-min bin of at most 2 actuations (a stray ON inside an
        # outage) is one run, each side at least 30 min, so the outage keeps its true start and end
        q.add("runs0", raw, mat=True)
        q.add("runs", """
SELECT dev, det, min(a) AS a, max(z) AS z FROM (
  SELECT *, sum(new::INT) OVER (PARTITION BY dev, det ORDER BY a) AS grp FROM (
    SELECT r.*, NOT coalesce(r.a - lag(r.z) OVER w = 1 AND lag(r.z - r.a) OVER w >= 6 AND r.z - r.a >= 6
                             AND g.n <= 2 AND g.cov AND g.hr >= 5 AND g.hr < 21, false) AS new
    FROM runs0 r LEFT JOIN g5 g ON g.dev = r.dev AND g.det = r.det AND g.b = r.a - 1
    WINDOW w AS (PARTITION BY r.dev, r.det ORDER BY r.a)))
GROUP BY dev, det, grp""", mat=True)
    else:
        q.add("runs", raw, mat=True)
    # level shift of the share of the signal's traffic (15-min, best single change point, Poisson LLR)
    q.add("lvl0", """
SELECT dev, det, k, n15, S15, c15,
       sum(n15) OVER w AS cx, sum(S15) OVER w AS cs, row_number() OVER w AS pos,
       count(*) OVER (PARTITION BY dev, det) AS nl, sum(n15) OVER (PARTITION BY dev, det) AS X,
       sum(S15) OVER (PARTITION BY dev, det) AS SS
FROM (SELECT dev, det, k, n15, sum(n15) OVER (PARTITION BY dev, k) - n15 AS S15, c15 FROM a15) WHERE c15
WINDOW w AS (PARTITION BY dev, det ORDER BY k ROWS UNBOUNDED PRECEDING)""")
    q.add("lvlshift", """
SELECT dev, det, first(llr ORDER BY llr DESC, pos) AS level_llr, first(ratio ORDER BY llr DESC, pos) AS level_ratio,
       first(pos * 3 ORDER BY llr DESC, pos) AS level_b
FROM (SELECT dev, det, pos,
        (CASE WHEN x1 > 0 THEN x1 * ln(greatest(p1, 1e-12)) ELSE 0 END - p1 * s1)
        + (CASE WHEN x2 > 0 THEN x2 * ln(greatest(p2, 1e-12)) ELSE 0 END - p2 * s2)
        - (X * ln(greatest(p0, 1e-12)) - p0 * SS) AS llr,
        ((x2 + 0.5) / greatest(s2, 1)) / ((x1 + 0.5) / greatest(s1, 1)) AS ratio
      FROM (SELECT *, cx AS x1, cs AS s1, X - cx AS x2, SS - cs AS s2,
                   CASE WHEN cs > 0 THEN cx / greatest(cs, 1e-9) ELSE 0 END AS p1,
                   CASE WHEN SS - cs > 0 THEN (X - cx) / greatest(SS - cs, 1e-9) ELSE 0 END AS p2,
                   X / greatest(SS, 1) AS p0
            FROM lvl0 WHERE nl >= 8 AND X >= 20 AND pos >= 2 AND pos <= nl - 2))
GROUP BY dev, det""")
    # continuous ONs held >= 5 min, not over a comms gap (stuck-on candidates); co = other detectors of the signal
    # held ON (>= 5 min) through at least half of it
    q.add("hold", """
SELECT h.*, row_number() OVER (ORDER BY h.dev, h.det, h.s) AS hid FROM (
  SELECT dev, det, s, f, open_end, dur FROM ons WHERE dur >= 300 AND NOT gap) h""", mat=True)
    q.add("hold_co", """
SELECT a.hid, count(DISTINCT b.det) FILTER (least(a.f, b.f) - greatest(a.s, b.s) >= 0.5 * a.dur) AS co
FROM hold a LEFT JOIN hold b ON a.dev = b.dev AND a.det <> b.det GROUP BY a.hid""", mat=True)


# ============================================================================ stage 1 (health_core.health)
def _sc1(x: str, lo: str, hi: str) -> str:
    """health_core._score: 0 below lo, .35 at lo rising to 1 at hi; NULL stays NULL."""
    return (f"(CASE WHEN ({x}) IS NULL OR isnan({x}) THEN NULL WHEN ({x}) < {lo} THEN 0.0 "
            f"ELSE 0.35 + 0.65 * least(greatest((({x}) - {lo}) / ({hi} - {lo}), 0.0), 1.0) END)")


def _at(q: _Q, name: str, src: str, who: str, at: tuple, col: str):
    """{name}_1 .. {name}_n: `src` with k5's running sum `col` of detector `who` at each boundary column of `at` added as
    {col}_{boundary} -- one lookup per CTE."""
    prev = src
    for i, b in enumerate(at, 1):
        q.add(f"{name}_{i}", f"""
SELECT x.*, k.{col} AS {col}_{b} FROM {prev} x JOIN k5 k ON k.dev = x.dev AND k.det = x.{who} AND k.k = x.{b}""",
              mat=True)
        prev = f"{name}_{i}"


def _stage1(q: _Q, p: str):
    """the four stage-1 rules for the detectors of {p}ph (dev, det, ph): references, silent runs, night count against
    the detector it tracks, stuck-on holds with their recovery -> {p}st (one row per detector)."""
    # reference candidates: the other live detectors of the signal, never a twin; same = on the same phase
    q.add(f"{p}rf", f"""
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same
FROM {p}ph a JOIN {p}ph b ON a.dev = b.dev AND a.det <> b.det
JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL""", mat=True)
    # yardstick kind: the other detectors on its phase when there are >= 2, else the rest of the signal
    q.add(f"{p}kd", f"""
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM {p}ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM {p}rf GROUP BY dev, det) r
USING (dev, det)""", mat=True)
    q.add(f"{p}pref", f"""
SELECT f.dev, f.det, f.r FROM {p}rf f JOIN {p}kd k USING (dev, det) WHERE k.isph AND f.same""", mat=True)
    # recovery partner candidates: every other live detector on the predicted phase, twins included
    q.add(f"{p}pc", f"""
SELECT a.dev, a.det, b.det AS c FROM {p}ph a JOIN {p}ph b ON a.dev = b.dev AND a.det <> b.det AND a.ph = b.ph
JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live""", mat=True)
    # reference members of every detector for window sums: phase refs, else the rest of the signal
    q.add(f"{p}rm", f"""
SELECT dev, det, r FROM {p}pref
UNION ALL SELECT k.dev, k.det, d.det FROM {p}kd k JOIN dets d ON d.dev = k.dev AND d.det <> k.det WHERE NOT k.isph""",
          mat=True)
    # silent runs: expected ONs lam = n_on / (reference ONs outside the run) x reference ONs inside it
    q.add(f"{p}rs0", f"""
SELECT u.dev, u.det, u.a, u.z, m.r, w.nb5 FROM runs u JOIN {p}rm m ON m.dev = u.dev AND m.det = u.det
JOIN win w ON w.dev = u.dev""", mat=True)
    _at(q, f"{p}rs", f"{p}rs0", "r", ("a", "z", "nb5"), "cn")
    q.add(f"{p}rs", f"""
SELECT dev, det, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM {p}rs_3 GROUP BY dev, det, a, z""")
    # the run with the most expected ONs; equal ones (to 6 decimals: the sums' last bits depend on how DuckDB splits
    # the work) -> the earliest (note 154ad)
    q.add(f"{p}dr", f"""
SELECT k.dev, k.det, coalesce(d.lam, 0.0) AS drop_lam, coalesce(d.b0, -1) AS drop_b0, coalesce(d.b1, -1) AS drop_b1
FROM {p}kd k LEFT JOIN (
  SELECT dev, det, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
  FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM {p}rs r JOIN dstat s USING (dev, det)
        {"JOIN win w ON w.dev = r.dev WHERE r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5" if q.opt["no_recovery_cap"] else ""})
  WHERE lam > 0 GROUP BY dev, det) d USING (dev, det)""", mat=True)
    if q.opt["all_silences"]:
        # note 154f: every silent run of >= 15 min with its expected count (a failure that comes back each day); the
        # ones over the detector's own suspect limit (a night one >= 50 expected) are listed next to the worst
        q.add(f"{p}dra", f"""
SELECT r.dev, r.det, r.a, r.z, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam
FROM {p}rs r JOIN dstat s USING (dev, det) JOIN win w ON w.dev = r.dev
WHERE r.z - r.a >= 3 AND r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5""", mat=True)
    q.add(f"{p}drx", f"""
SELECT d.*, s.n_on, (d.drop_b1 = w.nb5 OR (d.drop_b1 > 0 AND g.last_cov < d.drop_b1)) AS drop_to_end
FROM {p}dr d JOIN dstat s USING (dev, det) JOIN win w ON w.dev = d.dev JOIN sig0 g ON g.dev = d.dev""", mat=True)
    # partner requests: the whole sample (night), each stuck-on hold >= 15 min and each silent run (left out);
    # [xa, xb) = 15-min bins left out
    q.add(f"{p}rq", f"""
SELECT 'n' AS kind, dev * 128 + det AS id, dev, det, 0 AS xa, 0 AS xb FROM {p}kd
UNION ALL
SELECT 'e', h.hid, h.dev, h.det, floor(h.s / 300.0)::INT // 3, ceil(least(ceil(h.f / 300.0), w.nb5) / 3.0)::INT
FROM hold h JOIN {p}kd k USING (dev, det) JOIN win w ON w.dev = h.dev WHERE h.dur >= 900
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0 // 3, ceil(drop_b1 / 3.0)::INT FROM {p}drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0""", mat=True)
    q.add(f"{p}rc", f"""
SELECT r.kind, r.id, c.c FROM {p}rq r JOIN {p}pc c ON c.dev = r.dev AND c.det = r.det
WHERE r.kind <> 'e' OR NOT EXISTS (
  SELECT 1 FROM hold e JOIN hold o ON o.dev = e.dev AND o.det = c.c AND o.dur >= 900 AND o.s < e.f AND o.f > e.s
  WHERE e.hid = r.id)""")
    q.add(f"{p}pxy", f"""
SELECT r.kind, r.id, c.c, x.n15 AS x, y.n15 AS y, x.k + 1 AS w
FROM {p}rq r JOIN {p}rc c ON c.kind = r.kind AND c.id = r.id
JOIN a15 x ON x.dev = r.dev AND x.det = r.det AND x.c15 AND NOT (x.k >= r.xa AND x.k < r.xb)
JOIN a15 y ON y.dev = r.dev AND y.det = c.c AND y.k = x.k""")
    # the partner: best 15-min correlation (ties: its own counts, never its channel order), kept from 0.7
    q.add(f"{p}part", f"""
SELECT kind, id, p FROM (
  SELECT kind, id, first(c ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS p,
         first(cr ORDER BY round(cr, 12) DESC, ys DESC, yw DESC, c) AS pcr
  FROM (SELECT kind, id, c, corr(y, x) AS cr, sum(y) AS ys, sum(y * w) AS yw, count(*) AS m,
               min(x) < max(x) AS xv, min(y) < max(y) AS yv FROM {p}pxy GROUP BY kind, id, c)
  WHERE m >= 16 AND xv AND yv GROUP BY kind, id) WHERE pcr >= 0.7""", mat=True)
    # stuck-on holds >= 15 min and silent runs: share after / before (6 h each side, >= 3 h of data)
    q.add(f"{p}rr", f"""
SELECT 'e' AS kind, h.hid AS id, h.dev, h.det, floor(h.s / 300.0)::INT AS a, least(ceil(h.f / 300.0)::INT, w.nb5) AS b
FROM hold h JOIN {p}kd k USING (dev, det) JOIN win w ON w.dev = h.dev WHERE h.dur >= 900 AND NOT h.open_end
UNION ALL
SELECT 'd', dev * 128 + det, dev, det, drop_b0, drop_b1 FROM {p}drx
WHERE n_on > 0 AND drop_lam >= 30 AND drop_b1 > drop_b0 AND NOT drop_to_end""")
    q.add(f"{p}rr2", f"""
SELECT r.*, greatest(r.a - 72, 0) AS lo0, least(r.b + 72, w.nb5) AS hi1, pa.p,
       (r.a >= 36 AND w.nb5 - r.b >= 36) AS can
FROM {p}rr r JOIN win w ON w.dev = r.dev LEFT JOIN {p}part pa ON pa.kind = r.kind AND pa.id = r.id""", mat=True)
    q.add(f"{p}rec", f"""
SELECT kind, id, CASE WHEN can AND c0 >= 36 AND c1 >= 36 THEN ((x1 + 0.5) / greatest(s1, 1.0)) /
                      ((x0 + 0.5) / greatest(s0, 1.0)) END AS rec
FROM (
  SELECT r.kind, r.id, any_value(r.can) AS can,
         count(*) FILTER (mm.own AND g.cov AND g.b < r.a) AS c0, count(*) FILTER (mm.own AND g.cov AND g.b >= r.b) AS c1,
         coalesce(sum(g.n) FILTER (mm.own AND g.b < r.a), 0) AS x0, coalesce(sum(g.n) FILTER (mm.own AND g.b >= r.b), 0) AS x1,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b < r.a), 0) AS s0,
         coalesce(sum(g.n) FILTER (NOT mm.own AND g.b >= r.b), 0) AS s1
  FROM {p}rr2 r
  JOIN (SELECT r.kind, r.id, r.det AS m, true AS own FROM {p}rr2 r
        UNION ALL SELECT r.kind, r.id, r.p, false FROM {p}rr2 r WHERE r.p IS NOT NULL
        UNION ALL SELECT r.kind, r.id, m.r, false FROM {p}rr2 r JOIN {p}rm m ON m.dev = r.dev AND m.det = r.det
        WHERE r.p IS NULL) mm ON mm.kind = r.kind AND mm.id = r.id
  JOIN g5 g ON g.dev = r.dev AND g.det = mm.m AND ((g.b >= r.lo0 AND g.b < r.a) OR (g.b >= r.b AND g.b < r.hi1))
  GROUP BY r.kind, r.id)""", mat=True)
    # stuck-on per detector: the longest hold >= 15 min (first in time), its shared count and recovery
    q.add(f"{p}ep", f"""
SELECT h.dev, h.det, max(h.dur) AS ep_dur, first(h.s ORDER BY h.dur DESC, h.s) AS ep_t0,
       first(h.f ORDER BY h.dur DESC, h.s) AS ep_t1, first(c.co ORDER BY h.dur DESC, h.s) AS ep_co,
       first(r.rec ORDER BY h.dur DESC, h.s) AS ep_rec
FROM hold h JOIN {p}kd k USING (dev, det) JOIN hold_co c ON c.hid = h.hid
LEFT JOIN {p}rec r ON r.kind = 'e' AND r.id = h.hid WHERE h.dur >= 900 GROUP BY h.dev, h.det""")
    # silent run inside a stuck-on hold (>= half of it): not a silence
    q.add(f"{p}dis", f"""
SELECT d.dev, d.det, coalesce(sum(greatest(least(h.f, d.drop_b1 * 300.0) - greatest(h.s, d.drop_b0 * 300.0), 0.0))
         FILTER (h.hid IS NOT NULL), 0)
       >= 0.5 * (d.drop_b1 - d.drop_b0) * 300.0 AS drop_in_stuck
FROM {p}drx d LEFT JOIN hold h ON h.dev = d.dev AND h.det = d.det AND h.dur >= 900
WHERE d.n_on > 0 AND d.drop_lam >= 30 AND d.drop_b1 > d.drop_b0 GROUP BY d.dev, d.det, d.drop_b0, d.drop_b1""")
    # night count (21-05) against the detector it tracks, else its phase, else the rest of the signal
    kd, rm, pn = f"{p}kd", f"{p}rm", f"{p}part"
    if q.opt["clean_yardsticks"]:
        # note 154t: the night yardsticks without the detectors that do not look like traffic (phase mates while >= 2
        # are left, else the rest of the signal without them)
        kd, rm = f"{p}nkd", f"{p}nrm"
        # a partner that does not look like traffic is left out (the phase mates, else the rest of the signal, instead)
        pn = (f"(SELECT * FROM {p}part x WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = x.id // 128 "
              "AND t.det = x.p))")
        q.add(f"{p}nkd", f"""
SELECT a.dev, a.det, a.ph, coalesce(r.nsame, 0) >= 2 AS isph
FROM {p}ph a LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame FROM {p}rf f
                        WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = f.dev AND t.det = f.r) GROUP BY dev, det) r
USING (dev, det)""", mat=True)
        q.add(f"{p}nrm", f"""
SELECT f.dev, f.det, f.r FROM {p}rf f JOIN {p}nkd k USING (dev, det)
WHERE k.isph AND f.same AND NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = f.dev AND t.det = f.r)
UNION ALL SELECT k.dev, k.det, d.det FROM {p}nkd k JOIN dets d ON d.dev = k.dev AND d.det <> k.det
WHERE NOT k.isph AND NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = d.dev AND t.det = d.det)""", mat=True)
    q.add(f"{p}rsum", f"""
SELECT m.dev, m.det, sum(x.dy) AS rd, sum(x.nt) AS rn FROM {rm} m JOIN dstat x ON x.dev = m.dev AND x.det = m.r
GROUP BY m.dev, m.det""")
    q.add(f"{p}nt", f"""
SELECT dev, det, nt AS night_n, ex AS night_exp, (nt + 0.5) / (ex + 0.5) AS night_ratio, who FROM (
  SELECT k.dev, k.det, s.nt, s.dy / greatest(coalesce(ps.dy, r.rd, 0), 1) * coalesce(ps.nt, r.rn, 0) AS ex,
         CASE WHEN pa.p IS NOT NULL THEN 'd' || pa.p || ', which it tracks'
              WHEN k.isph THEN 'the other detectors on its phase' ELSE 'the rest of the signal' END AS who
  FROM {kd} k JOIN dstat s USING (dev, det) JOIN sig0 g ON g.dev = k.dev
  LEFT JOIN {pn} pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
  LEFT JOIN dstat ps ON ps.dev = k.dev AND ps.det = pa.p LEFT JOIN {p}rsum r ON r.dev = k.dev AND r.det = k.det
  WHERE g.ok_night AND s.dy >= 20) WHERE ex >= 30""")
    # the four rules -> pass status
    sd = _sc1("d.drop_lam", "30.0", "100.0")
    ss = _sc1("coalesce(e.ep_dur, 0)", "900.0", "3600.0")
    nc = q.opt["no_recovery_cap"]
    # note 154g: without the "came back" caps (_u): the detector's own rating; the capped status stays the yardstick's
    # (a hold the traffic came back from stays suspect under 1 h: 10005 det 19, 12 min = suspect; 2B334 det 23, 1 h 50 = bad)
    unc = ("""CASE WHEN n_on < 20 AND s_stuck_u = 0 THEN 'not_enough_data' WHEN hs_u < 0.25 THEN 'bad' WHEN hs_u < 0.70 THEN 'suspect'
               ELSE 'ok' END AS status_u, """ if nc else "")
    hsu = (""", (1 - coalesce(s_dropout_u, 0)) * (1 - s_stuck_u) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs_u""" if nc else "")
    colu = (f"""CASE WHEN d.n_on > 0 THEN CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                  ELSE {sd} END END
            AS s_dropout_u,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR (e.ep_rec BETWEEN 0.5 AND 2.0 AND e.ep_dur < 3600))
               THEN least({ss}, 0.35) ELSE {ss} END AS s_stuck_u,
          """ if nc else "")
    q.add(f"{p}st", f"""
SELECT *, {unc}CASE WHEN hs < 0.25 THEN 'bad' WHEN hs < 0.70 THEN 'suspect' WHEN n_on < 20 THEN 'not_enough_data'
               ELSE 'ok' END AS status
FROM (SELECT *, (1 - coalesce(s_dropout, 0)) * (1 - s_stuck) * (1 - coalesce(s_night_drop, 0))
                * (1 - coalesce(s_level, 0)) AS hs{hsu}
  FROM (SELECT d.dev, d.det, d.n_on, d.drop_lam, d.drop_b0, d.drop_b1, d.drop_to_end,
          coalesce(i.drop_in_stuck, false) AS drop_in_stuck, rd.rec AS drop_rec, e.ep_dur, e.ep_t0, e.ep_t1, e.ep_co,
          e.ep_rec, n.night_ratio, n.night_n, n.night_exp, n.who AS night_ref, l.level_ratio,
          coalesce(l.level_llr, 0.0) AS level_llr, l.level_b,
          CASE WHEN d.n_on > 0 THEN (
            CASE WHEN coalesce(i.drop_in_stuck, false) THEN 0.0
                 WHEN {sd} > 0 AND NOT d.drop_to_end AND rd.rec BETWEEN 0.5 AND 2.0 THEN least({sd}, 0.35)
                 ELSE {sd} END) END AS s_dropout,
          CASE WHEN coalesce(e.ep_dur, 0) >= 900 AND (e.ep_co >= 3 OR e.ep_rec BETWEEN 0.5 AND 2.0)
               THEN least({ss}, 0.35) ELSE {ss} END AS s_stuck,
          {colu}
          CASE WHEN n.night_ratio IS NOT NULL THEN
               {_sc1('-ln(greatest(n.night_ratio, 1e-6))', '-ln(0.19)', '-ln(0.063)')} END AS s_night_drop,
          CASE WHEN l.level_ratio IS NOT NULL THEN CASE WHEN l.level_llr <= 50 THEN 0.0
               ELSE {_sc1('-ln(l.level_ratio)', '-ln(0.15)', '-ln(0.05)')} END END AS s_level
        FROM {p}drx d LEFT JOIN {p}dis i USING (dev, det)
        LEFT JOIN {p}rec rd ON rd.kind = 'd' AND rd.id = d.dev * 128 + d.det
        LEFT JOIN {p}ep e USING (dev, det) LEFT JOIN {p}nt n USING (dev, det) LEFT JOIN lvlshift l USING (dev, det)))""",
          mat=True)


def _no_traffic(q: _Q):
    """note 154t (option clean_yardsticks): detectors that do not look like traffic (CLEAN_YARD) -> ntm (the levels,
    every detector), ntraf (dev, det).  From the 5-min counts only, so it is known before any check that uses a
    yardstick (the no-daily-pattern / busy-at-night measures are decided later, from the yardsticks)."""
    C = CLEAN_YARD
    q.add("ntm", f"""
WITH h AS (SELECT dev, count(*) FILTER (cov AND hr >= 7 AND hr < 19) / 12.0 AS hd,
                  count(*) FILTER (cov AND hr >= 1 AND hr < 5) / 12.0 AS hn FROM bins5 GROUP BY dev),
o AS (SELECT g.dev, g.det, coalesce(sum(g.n) FILTER (g.cov AND g.hr >= 7 AND g.hr < 19), 0) AS d,
             coalesce(sum(g.n) FILTER (g.cov AND g.hr >= 1 AND g.hr < 5), 0) AS n FROM g5 g GROUP BY g.dev, g.det),
r AS (SELECT a.dev, a.det, coalesce(sum(b.d) FILTER (a.ph = b.ph), 0) AS pd, coalesce(sum(b.n) FILTER (a.ph = b.ph), 0) AS pn,
             sum(b.d) AS sd, sum(b.n) AS sn
      FROM (SELECT o.*, p.ph FROM o JOIN p1ph p USING (dev, det)) a
      JOIN (SELECT o.*, p.ph FROM o JOIN p1ph p USING (dev, det)) b ON b.dev = a.dev AND b.det <> a.det
      GROUP BY a.dev, a.det)
SELECT o.dev, o.det, h.hd, h.hn, o.d / h.hd AS own_d, o.n / h.hn AS own_n, x.ph_rest,
       CASE WHEN x.ph_rest THEN r.pd ELSE r.sd END / h.hd AS rest_d, CASE WHEN x.ph_rest THEN r.pn ELSE r.sn END / h.hn AS rest_n
FROM o JOIN h USING (dev) JOIN r USING (dev, det)
CROSS JOIN LATERAL (SELECT r.pd / greatest(h.hd, 1e-9) >= {_f(C['rest_min'])} AS ph_rest) x
WHERE h.hd >= 2 AND h.hn >= 2""", mat=True)
    q.add("ntraf", f"""
SELECT dev, det FROM ntm WHERE own_d >= {_f(C['own_min'])} AND rest_d >= {_f(C['rest_min'])}
  AND own_n >= {_f(C['own'])} * own_d AND rest_n <= {_f(C['rest'])} * rest_d""", mat=True)


def _stage1_both(q: _Q):
    """pass 1 on every detector; pass 2 (signals with a detector pass 1 calls bad, and a predicted phase) without the
    bad detectors as anybody's phase mate; a bad detector keeps its pass-1 row."""
    q.add("p1ph", "SELECT d.dev, d.det, c.ph FROM dets d LEFT JOIN dcl c USING (dev, det)", mat=True)
    if q.opt["clean_yardsticks"]:
        _no_traffic(q)
    _stage1(q, "p1")
    q.add("p2sig", """
SELECT DISTINCT s.dev FROM p1st s WHERE s.status = 'bad'
  AND s.dev IN (SELECT dev FROM dcl WHERE ph IS NOT NULL AND NOT isnan(ph))""", mat=True)
    q.add("p2ph", """
SELECT p.dev, p.det, CASE WHEN s.status = 'bad' THEN NULL ELSE p.ph END AS ph
FROM p1ph p JOIN p1st s USING (dev, det) WHERE p.dev IN (SELECT dev FROM p2sig)""", mat=True)
    _stage1(q, "p2")
    q.add("stage1", """
SELECT x.*, a.status AS status1, a.hs AS score1 FROM (
  SELECT * FROM p1st WHERE dev NOT IN (SELECT dev FROM p2sig) OR status = 'bad'
  UNION ALL SELECT b.* FROM p2st b JOIN p1st a USING (dev, det) WHERE a.status <> 'bad') x
JOIN p1st a USING (dev, det)""", mat=True)
    if q.opt["all_silences"]:
        q.add("silall", """
SELECT a.* FROM p1dra a JOIN p1st s USING (dev, det) WHERE a.dev NOT IN (SELECT dev FROM p2sig) OR s.status = 'bad'
UNION ALL SELECT b.* FROM p2dra b JOIN p1st s USING (dev, det) WHERE s.status <> 'bad'""", mat=True)


# ============================================================================ measures (health_v4_stats, health_counts)
C7 = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")


def _in(xs) -> str:
    return "(" + ", ".join(_s(x) for x in xs) + ")"


def _sgeh(m: str, c: str) -> str:
    """signed GEH: + when m < c (under-counting)."""
    return (f"(CASE WHEN ({m}) < ({c}) THEN 1 ELSE -1 END * sqrt(2 * (({m}) - ({c})) ^ 2 / "
            f"greatest(({m}) + ({c}), 1e-9)))")


def _measures(q: _Q, LJ: dict, R: dict):
    """the per-detector measures the six checks read (the scored table X) and the holds (E)."""
    alts = ", ".join(f"CASE WHEN c.{PROB[f]} >= 0.15 THEN {_s(f)} END" for f in C7)
    top = "greatest(" + ", ".join(f"c.{PROB[f]}" for f in C7) + ")"
    # ---- the scored table: detectors with a predicted function the checks know
    q.add("xbase", f"""
SELECT s.*, c.fn, c.ph, c.lanes, w.hours, w.wg, w.bs, CASE WHEN c.lanes >= 2 THEN '2+' ELSE '1' END AS span,
       {top} AS f_top,
       CASE WHEN {top} < 0.7 THEN list_filter([{alts}], lambda x: x IS NOT NULL) ELSE []::VARCHAR[] END AS alts
FROM stage1 s JOIN dcl c USING (dev, det) JOIN win w USING (dev) WHERE c.fn IN {_in(FNS)}""", mat=True)
    # every detector with its answers (phase, function, traffic zone, lanes) and pass-1 status
    q.add("dz", f"""
SELECT d.dev, d.det, c.ph, c.fn, coalesce(c.fn IN {_in(('Advance', 'Count'))}, false) AS tr,
       greatest(coalesce(c.lanes, 1.0), 1.0) AS ln1, c.lanes, s.status1
FROM dets d LEFT JOIN dcl c USING (dev, det) LEFT JOIN stage1 s USING (dev, det)""", mat=True)
    # ---- occupancy bins of bs seconds (15 min; 5 min under ~1.2 h): count n, fraction ON occ (NULL = not covered)
    q.add("gb1", """
SELECT g.dev, g.det, g.k, CASE WHEN c.k IS NOT NULL THEN g.n END AS n,
       CASE WHEN c.k IS NOT NULL THEN g.occs / w.bs END AS occ, c.k IS NOT NULL AS cov
FROM (SELECT g.dev, g.det, g.b // (w.bs // 300) AS k, sum(g.n) AS n, sum(g.occ) AS occs
      FROM g5 g JOIN win w USING (dev) GROUP BY g.dev, g.det, g.b // (w.bs // 300)) g JOIN win w USING (dev)
LEFT JOIN (SELECT DISTINCT c.dev, c.b // (w.bs // 300) AS k FROM cov5 c JOIN win w USING (dev)) c
  ON c.dev = g.dev AND c.k = g.k""")
    # traffic yardstick of each detector: its phase's Advance / Count zones, else the signal's, else all others
    q.add("rmm", """
SELECT dev, det, r FROM (
  SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AND b.tr AS s1, b.tr AS s2,
         bool_or(coalesce(a.ph = b.ph, false) AND b.tr) OVER (PARTITION BY a.dev, a.det) AS h1,
         bool_or(b.tr) OVER (PARTITION BY a.dev, a.det) AS h2
  FROM dz a JOIN dz b ON a.dev = b.dev AND a.det <> b.det)
WHERE (h1 AND s1) OR (NOT h1 AND h2 AND s2) OR (NOT h1 AND NOT h2)""", mat=True)
    q.add("gb", """
SELECT g.*, CASE WHEN g.cov THEN r.ref END AS ref, CASE WHEN p.mocc > 0.01 THEN g.occ / p.mocc END AS rel, p.mocc,
       z.ph, z.fn, z.status1, (z.status1 IS DISTINCT FROM 'bad' AND g.occ IS NOT NULL) AS hl
FROM gb1 g JOIN (SELECT dev, det, avg(occ) AS mocc FROM gb1 GROUP BY dev, det) p USING (dev, det)
JOIN dz z USING (dev, det)
LEFT JOIN (SELECT m.dev, m.det, g.k, sum(coalesce(g.n, 0)) AS ref FROM rmm m JOIN gb1 g ON g.dev = m.dev AND g.det = m.r
           GROUP BY m.dev, m.det, g.k) r USING (dev, det, k)""", mat=True)
    # congestion: median of the other detectors' ON fraction relative to their own mean (signals with >= 4)
    q.add("gcong", """
SELECT dev, det, k, CASE WHEN m > 0 THEN (L[(CASE WHEN lo < p THEN lo ELSE lo + 1 END) + 1]
                                          + L[(CASE WHEN h < p THEN h ELSE h + 1 END) + 1]) / 2.0 END AS cong
FROM (SELECT *, m // 2 AS h, CASE WHEN m % 2 = 1 THEN m // 2 ELSE m // 2 - 1 END AS lo FROM (
  SELECT g.dev, g.det, g.k, l.L, CASE WHEN g.rel IS NULL THEN len(l.L) ELSE len(list_filter(l.L, lambda x: x < g.rel)) END AS p,
         len(l.L) - (g.rel IS NOT NULL)::INT AS m
  FROM gb g JOIN (SELECT dev, k, list_sort(list(rel) FILTER (rel IS NOT NULL)) AS L FROM gb GROUP BY dev, k) l
  USING (dev, k) WHERE g.dev IN (SELECT dev FROM dets GROUP BY dev HAVING count(*) >= 4)))""")
    # peer: mean ON fraction of the healthy phase mates; same: of the healthy phase mates of the same function
    q.add("gctx", """
SELECT g.*, c.cong,
  CASE WHEN g.ph IS NOT NULL THEN (a.so - coalesce(CASE WHEN g.hl THEN g.occ END, 0)) / nullif(a.sh - g.hl::INT, 0)
       END AS peer,
  CASE WHEN g.ph IS NOT NULL AND g.fn IS NOT NULL THEN (b.so - coalesce(CASE WHEN g.hl THEN g.occ END, 0))
       / nullif(b.sh - g.hl::INT, 0) END AS same
FROM gb g LEFT JOIN gcong c USING (dev, det, k)
LEFT JOIN (SELECT dev, ph, k, sum(occ) FILTER (hl) AS so, count(*) FILTER (hl) AS sh FROM gb WHERE ph IS NOT NULL
           GROUP BY dev, ph, k) a ON a.dev = g.dev AND a.ph = g.ph AND a.k = g.k
LEFT JOIN (SELECT dev, ph, fn, k, sum(occ) FILTER (hl) AS so, count(*) FILTER (hl) AS sh FROM gb
           WHERE ph IS NOT NULL AND fn IS NOT NULL GROUP BY dev, ph, fn, k) b
  ON b.dev = g.dev AND b.ph = g.ph AND b.fn = g.fn AND b.k = g.k""", mat=True)
    # count / time-ON elasticity vs the traffic yardstick on its busier half (queue pattern)
    q.add("slp", """
SELECT dev, det, CASE WHEN one >= 6 AND vx > 0.5 THEN bn END AS elhi_cnt,
       CASE WHEN one >= 6 AND vx > 0.5 THEN bo END AS elhi_occ
FROM (SELECT dev, det, count(*) AS one, regr_slope(lnn, lr) AS bn, regr_slope(lo, lr) AS bo,
             var_pop(lr) * count(*) AS vx
      FROM (SELECT g.dev, g.det, ln(g.ref + 1) AS lr, ln(g.n + 1) AS lnn, ln(g.occ + 0.005) AS lo
            FROM gb g JOIN win w USING (dev) JOIN (SELECT dev, det, median(ref) FILTER (n IS NOT NULL AND ref IS NOT NULL)
                                                   AS rmed FROM gb GROUP BY dev, det) m USING (dev, det)
            WHERE w.wg <> 'm30' AND g.n IS NOT NULL AND g.ref IS NOT NULL AND g.ref >= m.rmed)
      GROUP BY dev, det)""")
    q.add("lkc", """
SELECT dev, det, CASE WHEN c1 IS NOT NULL AND NOT isnan(c1) THEN c1 WHEN NOT isnan(c2) THEN c2 END AS c_occ_like
FROM (SELECT g.dev, g.det,
        CASE WHEN count(*) FILTER (g.occ IS NOT NULL AND g.same IS NOT NULL) >= 6
             THEN corr(g.same, g.occ) FILTER (g.occ IS NOT NULL AND g.same IS NOT NULL) END AS c1,
        CASE WHEN count(*) FILTER (g.occ IS NOT NULL AND g.peer IS NOT NULL) >= 6
             THEN corr(g.peer, g.occ) FILTER (g.occ IS NOT NULL AND g.peer IS NOT NULL) END AS c2
      FROM gctx g JOIN win w USING (dev) WHERE w.wg <> 'm30' GROUP BY g.dev, g.det)""")
    # ---- per-ON statistics: ONs logged again without an OFF, holds, the median clean ON (Count mode)
    q.add("act", """
SELECT o.dev, o.det, a.n_on_a, a.n_rep, a.n_rep / a.n_on_a AS rep_frac, count(*) AS n_start,
       max(CASE WHEN o.gap THEN 0.0 ELSE o.dur END) FILTER (NOT o.open_end) AS max_on_s,
       coalesce(sum(CASE WHEN o.gap THEN 0.0 ELSE o.dur END) FILTER (NOT o.open_end AND o.multi), 0.0) AS rep_time_s,
       CASE WHEN count(*) FILTER (NOT o.multi AND NOT o.open_end AND NOT o.gap) >= 5
            THEN median(o.dur) FILTER (NOT o.multi AND NOT o.open_end AND NOT o.gap) END AS dur_p50,
       count(*) FILTER (NOT o.gap AND o.dur >= 5) AS n_ge5, count(*) FILTER (NOT o.gap AND o.dur >= 60) AS n_ge60,
       max(o.dur) FILTER (NOT o.gap) AS long_max_s
FROM ons o JOIN (SELECT dev, det, count(*) AS n_on_a, count(*) FILTER (pe = 82) AS n_rep FROM de WHERE eid = 82
                 GROUP BY dev, det) a USING (dev, det)
GROUP BY o.dev, o.det, a.n_on_a, a.n_rep""", mat=True)
    # ---- too fast: every ON-to-ON gap under 0.8 s / lanes (stop-bar zones) or 0.6 s / lanes, in whole 0.1-s ticks
    tf = LJ["too_fast"]
    q.add("onx", """
WITH r AS (SELECT dev, det, dn, pe = 81 AND ppe = 81 AS rp, ne = 81 AND nne = 81 AS rn FROM de
           WHERE eid = 82 AND ((pe = 81 AND ppe = 81) OR (ne = 81 AND nne = 81))),
o AS (SELECT d.dev, d.det, d.dn FROM de d WHERE d.eid = 82 AND (d.dev, d.det) IN (SELECT (dev, det) FROM r))
SELECT r.dev, r.det, r.dn, CASE WHEN r.rp THEN p.dn END AS prv, CASE WHEN r.rn THEN n.dn END AS nxt
FROM r ASOF LEFT JOIN o p ON p.dev = r.dev AND p.det = r.det AND r.dn > p.dn
ASOF LEFT JOIN o n ON n.dev = r.dev AND n.det = r.det AND r.dn < n.dn""")
    q.add("tfo", f"""
SELECT d.dev, d.det, d.dn, z.lim,
       coalesce(round_even((d.dn - CASE WHEN d.pe = 82 THEN d.pdn WHEN d.pe = 81 AND d.ppe = 82 THEN d.p2dn
                                        ELSE x.prv END) / 1e8, 0) < z.lim * 10, false) AS f,
       CASE WHEN d.ne = 82 THEN d.ndn WHEN d.ne = 81 AND d.nne = 82 THEN d.n2dn ELSE x.nxt END IS NOT NULL AS nx,
       coalesce(round_even((CASE WHEN d.ne = 82 THEN d.ndn WHEN d.ne = 81 AND d.nne = 82 THEN d.n2dn ELSE x.nxt END
                            - d.dn) / 1e8, 0) < z.lim * 10, false) AS fnx
FROM de d JOIN (SELECT dev, det, (CASE WHEN fn IN {_in(tf['stopbar'])} THEN {_f(tf['gap_s_stopbar'])}
                                        ELSE {_f(tf['gap_s_other'])} END) / ln1 AS lim FROM dz) z USING (dev, det)
LEFT JOIN onx x ON x.dev = d.dev AND x.det = d.det AND x.dn = d.dn
WHERE d.eid = 82""", mat=True)
    q.add("tf", """
SELECT dev, det, count(*) AS N, count(*) FILTER (f) AS n_fast, 1000.0 * count(*) FILTER (f) / greatest(count(*), 1)
       AS fast_rate, any_value(lim) AS gap_lim,
       CASE WHEN count(*) FILTER (f AND nx) > 0 THEN count(*) FILTER (f AND nx AND NOT fnx) / count(*) FILTER (f AND nx)
       END AS pair_share
FROM tfo GROUP BY dev, det""")
    # ---- vehicle starts (an ON after a non-ON) with their colour state; phase timelines (>= 3 greens)
    q.add("st0r", """
WITH r AS (SELECT dev, det, dn FROM de WHERE eid = 82 AND pe = 81 AND ppe = 81),
o AS (SELECT d.dev, d.det, d.dn FROM de d WHERE d.eid = 82 AND d.pe IS DISTINCT FROM 82
      AND (d.dev, d.det) IN (SELECT (dev, det) FROM r))
SELECT r.dev, r.det, r.dn, p.dn AS prv FROM r ASOF LEFT JOIN o p ON p.dev = r.dev AND p.det = r.det AND r.dn > p.dn
UNION ALL
SELECT d.dev, d.det, d.dn, m.on_dn FROM de d JOIN multis m ON m.dev = d.dev AND m.det = d.det AND m.off_dn = d.pdn
WHERE d.eid = 82 AND d.pe = 81 AND d.ppe = 82 AND d.p3e = 82""")
    q.add("starts", f"""
SELECT d.dev, d.det, d.dn, d.dn / 1e9 AS x, (d.dn // {NS5})::INT AS b,
       coalesce(round_even((d.dn - CASE WHEN d.pe = 81 AND d.ppe = 82 AND d.p3e IS DISTINCT FROM 82 THEN d.p2dn
                                        ELSE r.prv END) / 1e8, 0) < 10, false) AS f1,
       coalesce(d.pe = 81 AND round_even((d.dn - d.pdn) / 1e8, 0) < 3, false) AS chat
FROM de d LEFT JOIN st0r r ON r.dev = d.dev AND r.det = d.det AND r.dn = d.dn
WHERE d.eid = 82 AND d.pe IS DISTINCT FROM 82""")
    q.add("seg", """
SELECT c.dev, c.ph, c.t AS a, coalesce(lead(c.t) OVER (PARTITION BY c.dev, c.ph ORDER BY c.t, c.eid), w.T) AS z, c.st
FROM (SELECT dev, par AS ph, dn / 1e9 AS t, eid, CASE eid WHEN 1 THEN 0 WHEN 8 THEN 1 ELSE 2 END AS st,
             lag(CASE eid WHEN 1 THEN 0 WHEN 8 THEN 1 ELSE 2 END) OVER (PARTITION BY dev, par ORDER BY dn, eid) AS pst
      FROM ev WHERE eid IN (1, 8, 9, 10)) c JOIN win w USING (dev)
WHERE c.pst IS DISTINCT FROM c.st""", mat=True)
    q.add("phs3", "SELECT dev, par AS ph FROM ev WHERE eid = 1 GROUP BY dev, par HAVING count(*) >= 3", mat=True)
    # pieces: green 0, yellow 1, red split into its first 2 s (2) and the rest (3); valid = no comms gap starts in it
    q.add("pcs", """
SELECT p.*, NOT coalesce(g.g0 < p.z, false) AS valid FROM (
  SELECT * FROM (SELECT dev, ph, a, z, st FROM seg WHERE st < 2
                 UNION ALL SELECT dev, ph, a, least(a + 2.0, z), 2 FROM seg WHERE st = 2
                 UNION ALL SELECT dev, ph, least(a + 2.0, z), z, 3 FROM seg WHERE st = 2) WHERE z > a) p
ASOF LEFT JOIN gaps g ON p.dev = g.dev AND g.g0 >= p.a""", mat=True)
    q.add("sst", """
SELECT s.*, CASE WHEN p.valid THEN p.st ELSE 4 END AS state
FROM (SELECT s.*, z.ph, z.ln1 FROM starts s JOIN dz z USING (dev, det)) s
ASOF LEFT JOIN (SELECT p.* FROM pcs p JOIN phs3 USING (dev, ph)) p ON s.dev = p.dev AND s.ph = p.ph AND s.x >= p.a""",
          mat=True)
    q.add("sgyb", """
SELECT dev, ph, b, sum(least(z, (b + 1) * 300.0) - greatest(a, b * 300.0)) AS sgy
FROM (SELECT p.dev, p.ph, p.a, p.z, unnest(range(floor(p.a / 300.0)::INT, least(floor(p.z / 300.0)::INT, w.nb5 - 1) + 1)) AS b
      FROM pcs p JOIN phs3 USING (dev, ph) JOIN win w USING (dev) WHERE p.valid AND p.st < 2)
GROUP BY dev, ph, b""", mat=True)
    # too many for the traffic: busiest 5 min in veh/h/lane, starts in green + yellow per green + yellow second
    # (>= 60 s of it in the bin) and all starts (in float32, as the research)
    q.add("q5b", """
SELECT s.dev, s.det, s.b, count(*) AS n, count(*) FILTER (s.state < 2) AS ngy, any_value(s.ln1) AS ln1
FROM sst s GROUP BY s.dev, s.det, s.b""", mat=True)
    q.add("q5gyb", """
SELECT z.dev, z.det, y.b, ((coalesce(q.ngy, 0)::FLOAT / y.sgy::FLOAT) * 3600::FLOAT)::DOUBLE / z.ln1 AS q5_gy
FROM dz z JOIN phs3 p ON p.dev = z.dev AND p.ph = z.ph JOIN sgyb y ON y.dev = z.dev AND y.ph = z.ph
LEFT JOIN q5b q ON q.dev = z.dev AND q.det = z.det AND q.b = y.b WHERE y.sgy >= 60""", mat=True)
    q.add("q5", """
SELECT a.dev, a.det, g.q5_gy, a.q5_all FROM (SELECT dev, det, max(((n::FLOAT / 300::FLOAT) * 3600::FLOAT)::DOUBLE / ln1)
                                                AS q5_all FROM q5b GROUP BY dev, det) a
LEFT JOIN (SELECT dev, det, max(q5_gy) AS q5_gy FROM q5gyb GROUP BY dev, det) g USING (dev, det)""")
    # rule F: are the fast starts the same share of the starts in every 15-min period?
    q.add("fe", """
SELECT dev, det, count(*) AS fe_n, any_value(p) AS fe_p,
       CASE WHEN count(*) >= 2 AND any_value(p) > 0 AND any_value(p) < 1
            THEN sum((x - p * n) ^ 2 / (p * (1 - p) * n)) / greatest(count(*) - 1, 1) END AS fe_phi
FROM (SELECT *, sum(x) OVER (PARTITION BY dev, det) / sum(n) OVER (PARTITION BY dev, det) AS p FROM (
        SELECT dev, det, b // 3 AS g, count(*) FILTER (f1 AND NOT chat AND state <> 2) AS x,
               count(*) - count(*) FILTER (chat AND state <> 2) AS n
        FROM sst GROUP BY dev, det, b // 3) WHERE n > 0)
GROUP BY dev, det""")
    _held_on(q, R)
    _hourly(q, LJ)
    if q.opt["drop_end"]:
        _low_runs(q)
    _spiky(q, LJ)
    _holds(q, LJ, R)
    if q.opt["mate_drop"] or q.opt["count_rise"]:
        _mate_hours(q)


def _held_on(q: _Q, R: dict):
    """Count zone held ON in light traffic (score_v4c.occ_hi_c + note 135 green test) -> ohi, hib (the bins)."""
    olp = _values("bs, tl, lp", [[_f(a), _f(b), _f(c)] for a, b, c in R["occ_lim_pulse"]])
    ctp = _values("bs, tp", [[_f(a), _f(b)] for a, b in R["count_t_pass"]])
    q.add("xm", """
SELECT x.*, CASE WHEN x.fn <> 'Count' THEN '' WHEN a.dur_p50 IS NULL THEN 'unknown' WHEN a.dur_p50 <= 0.25 THEN 'pulse'
                 ELSE 'normal' END AS cmode
FROM xbase x LEFT JOIN act a USING (dev, det)""", mat=True)
    q.add("ohr", f"""
SELECT h.*, CASE WHEN h.xfn = 'Count' AND h.occ > greatest(l.lp, h.n * t.tp / h.bs, 0.20) AND h.cong < 1.5
                  AND NOT coalesce(h.pocc >= 0.40, false) AND h.tl <= 1 THEN true ELSE false END AS hic
FROM (SELECT g.dev, g.det, g.k, g.n, g.occ, g.cong, x.ph AS xph, x.fn AS xfn, w.bs,
             CASE WHEN g.ph IS NOT NULL THEN (a.so - coalesce(g.occ, 0)) / CASE WHEN a.co > 1 THEN a.co - 1 END END AS pocc,
             CASE WHEN m.rmax > 0 THEN least(greatest(floor(5 * g.ref / m.rmax), 0), 4) END AS tl
      FROM gctx g JOIN xm x USING (dev, det) JOIN win w USING (dev)
      JOIN (SELECT dev, det, max(ref) AS rmax FROM gb GROUP BY dev, det) m USING (dev, det)
      LEFT JOIN (SELECT g.dev, g.ph, g.k, sum(g.occ) AS so, count(g.occ) AS co FROM gb g JOIN xm x USING (dev, det)
                 WHERE x.cmode <> 'unknown' AND g.ph IS NOT NULL GROUP BY g.dev, g.ph, g.k) a
        ON a.dev = g.dev AND a.ph = g.ph AND a.k = g.k
      WHERE x.cmode <> 'unknown') h
LEFT JOIN ({olp}) l ON l.bs = h.bs AND l.tl = h.tl LEFT JOIN ({ctp}) t ON t.bs = h.bs""", mat=True)
    # note 135: the zone ON for >= 20 % of its phase's green after the first 5 s (>= 60 s of such green in the bin)
    q.add("hicd", "SELECT DISTINCT dev, det, xph AS ph FROM ohr WHERE hic AND xph IS NOT NULL")
    q.add("grn", """
SELECT p.dev, p.ph, p.a + 5.0 AS a, p.z FROM pcs p JOIN (SELECT DISTINCT dev, ph FROM hicd) USING (dev, ph)
WHERE p.st = 0 AND p.valid AND p.z > p.a + 5.0""")
    q.add("gonb", """
SELECT dev, det, k, sum(least(z, (k + 1) * bs) - greatest(a, k * bs)) FILTER (gon) AS gon,
       sum(least(z, (k + 1) * bs) - greatest(a, k * bs)) FILTER (NOT gon) AS gav
FROM (SELECT i.*, w.bs, unnest(range(floor(i.a / w.bs)::INT, least(floor(i.z / w.bs)::INT, w.nbb - 1) + 1)) AS k
      FROM (SELECT h.dev, h.det, greatest(o.s, g.a) AS a, least(o.f, g.z) AS z, true AS gon
            FROM (SELECT h.dev, h.det, h.ph, o.s, o.f FROM hicd h
                  JOIN (SELECT dev, det, s, f FROM ons WHERE NOT gap UNION ALL SELECT dev, det, 0.0, f FROM lead0) o
                  ON o.dev = h.dev AND o.det = h.det) o
            JOIN grn g ON g.dev = o.dev AND g.ph = o.ph AND o.s < g.z AND o.f > g.a JOIN hicd h ON h.dev = o.dev
              AND h.det = o.det
            UNION ALL SELECT h.dev, h.det, g.a, g.z, false FROM hicd h JOIN grn g USING (dev, ph)) i
      JOIN win w USING (dev))
GROUP BY dev, det, k""")
    q.add("hib", """
SELECT h.dev, h.det, h.k, h.occ FROM ohr h JOIN gonb g USING (dev, det, k)
WHERE h.hic AND h.xph IS NOT NULL AND g.gav >= 60 AND coalesce(g.gon, 0) >= 0.20 * g.gav""", mat=True)
    q.add("ohi", """
SELECT o.dev, o.det, count(h.k) * any_value(o.bs) / 60.0 AS hi_min_c, max(h.occ) AS hi_occ_max
FROM (SELECT DISTINCT dev, det, bs FROM ohr) o LEFT JOIN hib h USING (dev, det) GROUP BY o.dev, o.det""")


def _hourly(q: _Q, LJ: dict):
    """hourly GEH against the yardstick (health_counts.hourly_geh): count drops and misses at night -> geh."""
    q.add("hix", f"""
SELECT dev, hid, hid - min(hid) OVER (PARTITION BY dev) AS j, hid % 24 AS hod, nb_ = 12 AND nc = 12 AS ch,
       count(*) OVER (PARTITION BY dev) AS H
FROM (SELECT b.dev, (w.s_ns + b.b * {NS5}) // {NSH} AS hid, count(*) AS nb_, count(*) FILTER (b.cov) AS nc
      FROM bins5 b JOIN win w USING (dev) WHERE w.hours >= {_f(LJ['min_hours_counts'])} GROUP BY b.dev, hid)""",
          mat=True)
    q.add("mh", f"""
SELECT g.dev, g.det, (w.s_ns + g.b * {NS5}) // {NSH} AS hid, sum(g.n) AS M, sum(g.occ) AS O
FROM g5 g JOIN win w USING (dev) WHERE w.hours >= {_f(LJ['min_hours_counts'])} GROUP BY g.dev, g.det, hid""",
          mat=True)
    # yardstick: the detector it tracks (partner), else its phase mates (>= 2), else the rest of the signal
    cya = q.opt["clean_yardsticks"] == "all"
    if cya:
        # note 154t ("all"): a detector that looks like traffic gets the yardstick without the ones that do not (its
        # partner too); one that does not keeps its own (two of them tracking each other stay each other's yardstick)
        q.add("yk", """
SELECT k.dev, k.det, pa.p, k.isph, false AS cl FROM p1kd k LEFT JOIN p1part pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
WHERE EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = k.dev AND t.det = k.det)
UNION ALL
SELECT k.dev, k.det, pa.p, k.isph, true FROM p1nkd k
LEFT JOIN (SELECT * FROM p1part x WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = x.id // 128 AND t.det = x.p)) pa
  ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det
WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = k.dev AND t.det = k.det)""", mat=True)
        q.add("gh", """
SELECT m.dev, m.det, x.j, x.hod, x.ch, x.H, m.M, m.O, y.p IS NOT NULL OR y.isph AS has_m,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN CASE WHEN y.cl THEN s.SMc ELSE s.SM END - m.M ELSE coalesce(r.RM, 0) END
       AS R, coalesce(r.RO, 0) AS Om
FROM mh m JOIN hix x USING (dev, hid) JOIN yk y USING (dev, det)
JOIN (SELECT m.dev, m.hid, sum(m.M) AS SM, sum(m.M) FILTER (t.det IS NULL) AS SMc FROM mh m LEFT JOIN ntraf t USING (dev, det)
      GROUP BY m.dev, m.hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM, sum(b.O) AS RO
           FROM (SELECT dev, det, p AS r FROM yk WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1pref f JOIN yk y USING (dev, det) WHERE y.p IS NULL AND NOT y.cl
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1nrm f JOIN yk y USING (dev, det)
                 WHERE y.p IS NULL AND y.cl AND y.isph) a
           JOIN mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)""",
              mat=True)
    else:
        q.add("yk", """
SELECT k.dev, k.det, pa.p, k.isph FROM p1kd k LEFT JOIN p1part pa ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det""",
              mat=True)
    if not cya:
        q.add("gh", """
SELECT m.dev, m.det, x.j, x.hod, x.ch, x.H, m.M, m.O, y.p IS NOT NULL OR y.isph AS has_m,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN s.SM - m.M ELSE coalesce(r.RM, 0) END AS R, coalesce(r.RO, 0) AS Om
FROM mh m JOIN hix x USING (dev, hid) JOIN yk y USING (dev, det)
JOIN (SELECT dev, hid, sum(M) AS SM FROM mh GROUP BY dev, hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM, sum(b.O) AS RO
           FROM (SELECT dev, det, p AS r FROM yk WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1pref f JOIN yk y USING (dev, det) WHERE y.p IS NULL) a
           JOIN mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)""",
          mat=True)

    def share(mask: str) -> str:
        return (f"CASE WHEN count(*) FILTER ({mask} AND R >= 30) >= 4 THEN median(M / R) FILTER ({mask} AND R >= 30) "
                f"ELSE coalesce(sum(M) FILTER ({mask}), 0) / greatest(coalesce(sum(R) FILTER ({mask}), 0), 1) END")
    day, nt = "ch AND hod >= 7 AND hod < 19", "ch AND (hod >= 21 OR hod < 5)"
    q.add("gh1", f"""
SELECT dev, det, coalesce(sum(M) FILTER ({day}), 0) >= 20 AND coalesce(sum(R) FILTER ({day}), 0) > 0 AS elig,
       {share(day)} AS s, count(*) FILTER ({nt}) AS nn
FROM gh GROUP BY dev, det""", mat=True)
    # note 154: running counts and expected counts, for the recovery test of a count drop
    cum = (",\n       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.M ELSE 0.0 END) OVER w AS cm, "
           "sum(CASE WHEN g.ch AND g.R >= 30 THEN g.R ELSE 0.0 END) OVER w AS cr,\n"
           "       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.M ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cm_t,\n"
           "       sum(CASE WHEN g.ch AND g.R >= 30 THEN g.R ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cr_t"
           if q.opt["drop_end"] else "")
    q.add("gh2", f"""
SELECT g.*, a.s, a.elig, CASE WHEN g.ch THEN {_sgeh('g.M', 'a.s * g.R')} ELSE 0.0 END AS sg,
       sum(CASE WHEN g.ch THEN {_sgeh('g.M', 'a.s * g.R')} ELSE 0.0 END) OVER w AS cs, sum(g.ch::INT) OVER w AS cc,
       sum(CASE WHEN g.ch THEN {_sgeh('g.M', 'a.s * g.R')} ELSE 0.0 END) OVER (PARTITION BY g.dev, g.det) AS cs_t,
       sum(g.ch::INT) OVER (PARTITION BY g.dev, g.det) AS cc_t{cum}
FROM gh g JOIN gh1 a USING (dev, det) WINDOW w AS (PARTITION BY g.dev, g.det ORDER BY g.j ROWS UNBOUNDED PRECEDING)""",
          mat=True)
    # count drops: the hour with the largest step up in under-counting (>= 3 covered hours each side)
    q.add("ghk", """
SELECT a.dev, a.det, coalesce(k.k, 1) AS k, a.elig AND k.k IS NOT NULL AS has_k FROM gh1 a LEFT JOIN (
  SELECT dev, det, first(j + 1 ORDER BY stat DESC, j) AS k FROM (
    SELECT dev, det, j, ((cs_t - cs) / (cc_t - cc) - cs / cc) * sqrt(cc * (cc_t - cc) / cc_t) AS stat
    FROM gh2 WHERE j <= H - 2 AND cc >= 3 AND cc_t - cc >= 3) GROUP BY dev, det) k USING (dev, det)""")
    if q.opt["drop_end"]:
        # note 154: the recovery -- the hour after the drop with the largest step back down in under-counting (>= 3
        # covered hours each side), kept when the hours after it under-count at most 0.35 x the dropped hours and
        # below half the suspect GEH, and count at least 60 % of what its share BEFORE the drop gives there (>= 50
        # expected: quiet night hours cannot show a recovery); the drop then ends there, not at the window end
        lo = _f(LJ["count_drops_geh"][0] / 2)
        q.add("ghk2", f"""
SELECT dev, det, first(j + 1 ORDER BY stat DESC, j) AS k2 FROM (
  SELECT g.dev, g.det, g.j, ((g.cs - b.cs) / (g.cc - b.cc) - (g.cs_t - g.cs) / (g.cc_t - g.cc))
         * sqrt((g.cc - b.cc) * (g.cc_t - g.cc) / (g.cc_t - b.cc)) AS stat,
         (g.cs - b.cs) / (g.cc - b.cc) AS m_in, (g.cs_t - g.cs) / (g.cc_t - g.cc) AS m_rec
  FROM gh2 g JOIN ghk k USING (dev, det) JOIN gh2 b ON b.dev = g.dev AND b.det = g.det AND b.j = k.k - 1
  WHERE k.has_k AND g.j >= k.k AND g.j <= g.H - 2 AND g.cc - b.cc >= 3 AND g.cc_t - g.cc >= 3 AND b.cr > 0
    AND (g.cr_t - g.cr) * b.cm / b.cr >= 50 AND g.cm_t - g.cm >= 0.6 * (g.cr_t - g.cr) * b.cm / b.cr)
WHERE m_in > 0 AND m_rec <= 0.35 * m_in AND m_rec < {lo} GROUP BY dev, det""")
        q.add("gh3", f"""
SELECT g.dev, g.det, k.k, k.has_k, r.k2, g.j < k.k AND g.ch AS pre,
       g.j >= k.k AND (r.k2 IS NULL OR g.j < r.k2) AND g.ch AS post, g.j >= k.k AND g.ch AS post0, g.M, g.R, g.O,
       g.Om, g.ch, g.sg, g.s, g.hod, g.H, g.has_m, g.elig
FROM gh2 g JOIN ghk k USING (dev, det) LEFT JOIN ghk2 r USING (dev, det)""")
    else:
        q.add("gh3", f"""
SELECT g.dev, g.det, k.k, k.has_k, NULL::BIGINT AS k2, g.j < k.k AND g.ch AS pre, g.j >= k.k AND g.ch AS post, g.M, g.R,
       g.O, g.Om, g.ch, g.sg, g.s, g.hod, g.H, g.has_m, g.elig
FROM gh2 g JOIN ghk k USING (dev, det)""")
    q.add("gh4", f"""
SELECT dev, det, any_value(k) AS k, any_value(has_k) AS has_k, any_value(k2) AS k2, any_value(H) AS H,
       any_value(has_m) AS has_m,
       any_value(elig) AS elig, any_value(s) AS s, {share('pre')} AS s2,
       count(*) FILTER (pre AND R >= 30) >= 3 AS re, greatest(count(*) FILTER (pre), 1) AS npre,
       greatest(count(*) FILTER (post), 1) AS npost, coalesce(sum(M) FILTER (pre), 0) AS Mpre,
       coalesce(sum(M) FILTER (post), 0) AS Mpost, coalesce(sum(R) FILTER (post), 0) AS Rpost,
       coalesce(sum(O) FILTER (pre), 0) AS Opre, coalesce(sum(O) FILTER (post), 0) AS Opost,
       coalesce(sum(Om) FILTER (pre), 0) AS Ompre, coalesce(sum(Om) FILTER (post), 0) AS Ompost,
       count(*) FILTER ({nt}) AS nn, coalesce(sum(M) FILTER ({nt}), 0) AS Mn, coalesce(sum(s * R) FILTER ({nt}), 0) AS Cn
FROM gh3 GROUP BY dev, det""")
    if q.opt["clean_yardsticks"] is True:   # "all": gh is clean already
        _night_clean(q, share, day, nt)
        ngeh = ("CASE WHEN n.det IS NOT NULL THEN CASE WHEN n.elig AND n.nn >= 4 THEN "
                f"{_sgeh('n.Mn / greatest(n.nn, 1)', 'n.Cn / greatest(n.nn, 1)')} END ELSE "
                f"CASE WHEN a.elig AND a.nn >= 4 THEN {_sgeh('a.Mn / greatest(a.nn, 1)', 'a.Cn / greatest(a.nn, 1)')} END "
                "END AS geh_night,\n       CASE WHEN n.det IS NOT NULL THEN n.Mn ELSE a.Mn END AS geh_night_n0, "
                "CASE WHEN n.det IS NOT NULL THEN n.Cn ELSE a.Cn END AS geh_night_exp0,")
        nj = "\nLEFT JOIN ghnn n USING (dev, det)"
    else:
        ngeh = (f"CASE WHEN a.elig AND a.nn >= 4 THEN {_sgeh('a.Mn / greatest(a.nn, 1)', 'a.Cn / greatest(a.nn, 1)')} END "
                "AS geh_night,\n       a.Mn AS geh_night_n0, a.Cn AS geh_night_exp0,")
        nj = ""
    q.add("geh", f"""
SELECT a.dev, a.det, a.elig AS geh_ok,
       {ngeh}
       CASE WHEN a.has_k AND b.med > 0 THEN b.med END AS geh_drop,
       CASE WHEN a.has_k THEN make_timestamp(((x.hid0 + least(a.k, a.H - 1)) * {NSH}) // 1000) END AS drop_t,
       CASE WHEN a.has_k AND a.k2 IS NOT NULL THEN make_timestamp(((x.hid0 + a.k2) * {NSH}) // 1000) END AS drop_t2,
       a.Opost / a.npost / 3600.0 AS occ_after,
       a.Mpre / a.npre AS own_before, a.Mpost / a.npost AS own_after,
       CASE WHEN a.re THEN a.s2 ELSE a.s END * a.Rpost / a.npost AS exp_after,
       CASE WHEN a.has_k THEN (a.Opost / a.npost) / greatest(a.Opre / a.npre, 1e-6) END AS occ_own,
       CASE WHEN a.has_k AND a.has_m THEN (a.Ompost / a.npost) / greatest(a.Ompre / a.npre, 1e-6) END AS occ_mates
FROM gh4 a JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) x USING (dev){nj}
-- note 154h: the severity (geh_drop) over every hour after the drop, as 1.2.0; its end bounds only the period / text
LEFT JOIN (SELECT g.dev, g.det, median(CASE WHEN a.re THEN CASE WHEN g.ch THEN {_sgeh('g.M', 'a.s2 * g.R')} ELSE 0.0 END
                                           ELSE g.sg END) FILTER ({"g.post0" if q.opt["drop_end"] else "g.post"}) AS med
           FROM gh3 g JOIN gh4 a USING (dev, det) GROUP BY g.dev, g.det) b USING (dev, det)""")


def _night_clean(q: _Q, share, day: str, nt: str):
    """note 154t (option clean_yardsticks): the hourly night GEH against a yardstick without the detectors that do not
    look like traffic (ntraf) -> ghnn (Mn, Cn, nn, elig), only for the detectors whose yardstick held one (the others
    keep theirs exactly)."""
    q.add("ykn", """
SELECT k.dev, k.det, pa.p, k.isph FROM p1nkd k
LEFT JOIN (SELECT * FROM p1part x WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = x.id // 128 AND t.det = x.p)) pa
  ON pa.kind = 'n' AND pa.id = k.dev * 128 + k.det""",
          mat=True)
    # the detectors whose night yardstick (partner, else phase mates, else the rest of the signal) held one
    q.add("ykaff", """
SELECT DISTINCT y.dev, y.det FROM yk y JOIN ntraf t ON t.dev = y.dev AND t.det <> y.det
WHERE t.det = y.p OR (y.p IS NULL AND NOT y.isph)
   OR (y.p IS NULL AND y.isph AND EXISTS (SELECT 1 FROM p1pref f WHERE f.dev = y.dev AND f.det = y.det AND f.r = t.det))""",
          mat=True)
    q.add("ghn", """
SELECT m.dev, m.det, x.hod, x.ch, m.M,
       CASE WHEN y.p IS NULL AND NOT y.isph THEN s.SM - CASE WHEN t.det IS NULL THEN m.M ELSE 0 END
            ELSE coalesce(r.RM, 0) END AS R
FROM mh m JOIN ykaff USING (dev, det) JOIN hix x USING (dev, hid) JOIN ykn y USING (dev, det)
LEFT JOIN ntraf t USING (dev, det)
JOIN (SELECT m.dev, m.hid, sum(m.M) FILTER (t.det IS NULL) AS SM FROM mh m LEFT JOIN ntraf t USING (dev, det)
      GROUP BY m.dev, m.hid) s USING (dev, hid)
LEFT JOIN (SELECT a.dev, a.det, b.hid, sum(b.M) AS RM
           FROM (SELECT dev, det, p AS r FROM ykn WHERE p IS NOT NULL
                 UNION ALL SELECT f.dev, f.det, f.r FROM p1nrm f JOIN ykn y USING (dev, det) WHERE y.p IS NULL AND y.isph) a
           JOIN mh b ON b.dev = a.dev AND b.det = a.r GROUP BY a.dev, a.det, b.hid) r USING (dev, det, hid)""", mat=True)
    q.add("ghnn", f"""
SELECT g.dev, g.det, a.elig, a.nn, coalesce(sum(g.M) FILTER ({nt}), 0) AS Mn, coalesce(sum(a.s * g.R) FILTER ({nt}), 0) AS Cn
FROM ghn g JOIN (SELECT dev, det, coalesce(sum(M) FILTER ({day}), 0) >= 20 AND coalesce(sum(R) FILTER ({day}), 0) > 0
                        AS elig, {share(day)} AS s, count(*) FILTER ({nt}) AS nn FROM ghn GROUP BY dev, det) a
  USING (dev, det)
GROUP BY g.dev, g.det, a.elig, a.nn""")


def _mate_hours(q: _Q):
    """note 154n: hour-by-hour count drop / rise against each phase mate (complete hours; Bike detectors neither judged
    nor used) -> mhrun (every run of low / high hours) and mhbest (per detector its strongest drop and rise run)."""
    M = MATE_HOURS
    q.add("mhg", """
SELECT m.dev, m.det, m.hid, x.hod, m.hid // 24 AS dy, m.M, m.O, z.ph
FROM mh m JOIN hix x USING (dev, hid) JOIN dz z USING (dev, det)
WHERE x.ch AND z.ph IS NOT NULL AND coalesce(z.fn, '') <> 'Bike'""", mat=True)
    # note 154t ("all"): a mate that does not look like traffic is left out (unless the detector is one itself)
    xm = ("""
WHERE NOT EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = b.dev AND t.det = b.det)
   OR EXISTS (SELECT 1 FROM ntraf t WHERE t.dev = a.dev AND t.det = a.det)""" if q.opt["clean_yardsticks"] == "all" else "")
    q.add("mhp", f"""
SELECT a.dev, a.det, b.det AS mate, a.hid, a.hod, a.dy, a.M, b.M AS Mm
FROM mhg a JOIN mhg b ON b.dev = a.dev AND b.hid = a.hid AND b.ph = a.ph AND b.det <> a.det{xm}""", mat=True)
    # its share of each mate over the day (07-19) or the night hours of every day
    q.add("mhqa", f"""
SELECT dev, det, mate, hod >= 7 AND hod < 19 AS dband, median(M / Mm) AS q_band
FROM mhp x WHERE Mm >= {_f(M['min_mate'])} GROUP BY ALL""", mat=True)
    # its share of each mate at the same time of day (+-1 h): >= 3 days there -> the median over all of them, else the
    # earlier day (amb: a 2-day yardstick)
    q.add("mhqh", f"""
SELECT k.dev, k.det, k.mate, k.dy, k.hod,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.q)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.q) FILTER (s.dy < k.dy) END AS q_hod,
       count(DISTINCT s.dy) < 3 AS amb
FROM (SELECT DISTINCT dev, det, mate, dy, hod FROM mhp) k
JOIN (SELECT dev, det, mate, dy, hod, M / Mm AS q FROM mhp x WHERE Mm >= {_f(M['min_mate'])}) s
  ON s.dev = k.dev AND s.det = k.det AND s.mate = k.mate AND (abs(s.hod - k.hod) <= 1 OR abs(s.hod - k.hod) = 23)
GROUP BY k.dev, k.det, k.mate, k.dy, k.hod""", mat=True)
    # its own count and time ON at the same time of day, by the same day rule
    q.add("mhoh", """
SELECT k.dev, k.det, k.dy, k.hod,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.M)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.M) FILTER (s.dy < k.dy) END AS own_ref,
       CASE WHEN count(DISTINCT s.dy) >= 3 THEN median(s.O)
            WHEN count(DISTINCT s.dy) = 2 AND min(s.dy) < k.dy THEN median(s.O) FILTER (s.dy < k.dy) END AS occ_ref
FROM (SELECT DISTINCT dev, det, dy, hod FROM mhg) k
JOIN mhg s ON s.dev = k.dev AND s.det = k.det AND (abs(s.hod - k.hod) <= 1 OR abs(s.hod - k.hod) = 23)
GROUP BY k.dev, k.det, k.dy, k.hod""", mat=True)
    lo, hi = _f(M["low"]), _f(M["high"])
    zc = _f(M["z"])
    if q.opt["mate_noise"]:
        # note 154o: counting noise with a mild over-dispersion (network-wide: hour-to-hour scatter about the time-of-day
        # yardstick is Poisson at low volume; the proportional part at high volume is the 0.5 x / 2 x ratio's job)
        ph = _f(M["phi"])
        zlo = f" AND (p.M - h.q_hod * p.Mm) / sqrt({ph} * h.q_hod * p.Mm + 1) <= -{zc}"
        zhi = f" AND (p.M - h.q_hod * p.Mm) / sqrt({ph} * h.q_hod * p.Mm + 1) >= {zc}"
        zolo = f" AND (g.M - o.own_ref) / sqrt({ph} * o.own_ref + 1) <= -{zc}"
        zohi = f" AND (g.M - o.own_ref) / sqrt({ph} * o.own_ref + 1) >= {zc}"
        zjoin = zojoin = ""
    else:
        zlo = zhi = zjoin = zolo = zohi = zojoin = ""
    q.add("mhv", f"""
SELECT p.dev, p.det, p.hid, count(*) AS nv, median(h.q_hod * p.Mm) AS E,
       sum(p.Mm) FILTER (p.M <= {lo} * h.q_hod * p.Mm{zlo}) / sum(p.Mm) AS wlo,
       count(*) FILTER (p.M >= {hi} * h.q_hod * p.Mm AND (NOT h.amb OR p.M >= {hi} * a.q_band * p.Mm)
                        AND p.M - h.q_hod * p.Mm >= {_f(M['min_extra'])}{zhi}) AS nhi
FROM mhp p JOIN mhqh h USING (dev, det, mate, dy, hod) {zjoin}
JOIN mhqa a ON a.dev = p.dev AND a.det = p.det AND a.mate = p.mate AND a.dband = (p.hod >= 7 AND p.hod < 19)
WHERE h.q_hod IS NOT NULL AND p.Mm >= 5 GROUP BY p.dev, p.det, p.hid""", mat=True)
    q.add("mhf", f"""
SELECT g.dev, g.det, g.hid, g.M, g.O, v.E, o.occ_ref,
       coalesce(v.nv >= 1 AND v.E >= {_f(M['min_exp'])}, false) AS jl, coalesce(v.nv >= 1, false) AS jh,
       coalesce(v.nv >= 1 AND v.E >= {_f(M['min_exp'])} AND v.wlo >= 0.5 AND g.M <= {_f(M['own_low'])} * o.own_ref{zolo},
                false) AS lo,
       coalesce(v.nv >= 1 AND v.nhi >= least(2, v.nv) AND v.nhi >= {_f(M['high_share'])} * v.nv
                AND g.M >= {_f(M['own_high'])} * o.own_ref + 10{zohi}, false) AS hi
FROM mhg g LEFT JOIN mhv v USING (dev, det, hid) LEFT JOIN mhoh o USING (dev, det, dy, hod) {zojoin}""", mat=True)
    # runs: hours not judged (too few expected) do not break a run
    dead = q.opt["rise_dead_days"]
    q.add("mhrun0" if dead else "mhrun", """
SELECT * , sqrt(2 * (M - E) ^ 2 / greatest(M + E, 1e-9)) AS rgeh FROM (
  SELECT dev, det, 'drop' AS kind, min(hid) FILTER (lo) AS a, max(hid) FILTER (lo) + 1 AS z, count(*) FILTER (lo) AS hours,
         sum(M) FILTER (lo) AS M, sum(E) FILTER (lo) AS E, sum(O) FILTER (lo) AS O, sum(occ_ref) FILTER (lo) AS Oref
  FROM (SELECT *, sum((jl AND NOT lo)::INT) OVER (PARTITION BY dev, det ORDER BY hid) AS grp FROM mhf)
  GROUP BY dev, det, grp HAVING count(*) FILTER (lo) >= 1
  UNION ALL
  SELECT dev, det, 'rise', min(hid) FILTER (hi), max(hid) FILTER (hi) + 1, count(*) FILTER (hi),
         sum(M) FILTER (hi), sum(E) FILTER (hi), sum(O) FILTER (hi), sum(occ_ref) FILTER (hi)
  FROM (SELECT *, sum((jh AND NOT hi)::INT) OVER (PARTITION BY dev, det ORDER BY hid) AS grp FROM mhf)
  GROUP BY dev, det, grp HAVING count(*) FILTER (hi) >= 1)""", mat=True)
    if dead:
        # note 154x: a day that is "too high" only because the detector nearly stopped on the window's other days (the
        # time-of-day yardstick, a median over the days, is then the dead level: 10035 det 2) is not a rise -- the
        # other days (<= 10 % of that day's 06-22 h count while its mates kept >= half their traffic) are the fault
        q.add("mhday", """
SELECT g.dev, g.det, g.dy, g.Md, m.Mmd
FROM (SELECT dev, det, dy, sum(M) AS Md FROM mhg WHERE hod >= 6 AND hod < 22 GROUP BY ALL) g
JOIN (SELECT dev, det, dy, sum(Mm) AS Mmd FROM mhp WHERE hod >= 6 AND hod < 22 GROUP BY ALL) m USING (dev, det, dy)""",
              mat=True)
        q.add("mhrun", """
SELECT * FROM mhrun0 u
WHERE u.kind <> 'rise' OR NOT EXISTS (
  SELECT 1 FROM mhday r JOIN mhday d ON d.dev = r.dev AND d.det = r.det AND d.dy <> r.dy
  WHERE r.dev = u.dev AND r.det = u.det AND r.dy = u.a // 24
    AND d.Md <= 0.1 * r.Md AND d.Mmd >= 0.5 * r.Mmd AND r.Mmd > 0)""", mat=True)
    # the strongest run of each kind; a drop counts only when its time ON fell too (a queue over the zone holds it ON
    # longer and counts fewer) or it nearly stopped counting
    q.add("mhbest", f"""
SELECT d.dev, d.det, true AS mh_ok, r.hours AS md_h, r.rgeh AS md_geh, r.M / greatest(r.E, 1e-9) AS md_r, r.a AS md_a,
       r.z AS md_z, r.M / (r.z - r.a) AS md_own, r.E / (r.z - r.a) AS md_exp,
       u.hours AS mr_h, u.rgeh AS mr_geh, u.M / greatest(u.E, 1e-9) AS mr_r, u.a AS mr_a, u.z AS mr_z
FROM (SELECT DISTINCT dev, det FROM mhv) d
LEFT JOIN (SELECT * FROM mhrun WHERE kind = 'drop' AND hours >= {M['hours']}
             AND (O < {_f(M['occ'])} * Oref OR M <= 0.1 * E)
           QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY rgeh DESC, a) = 1) r USING (dev, det)
LEFT JOIN (SELECT * FROM mhrun WHERE kind = 'rise' AND hours >= {M['hours']}
           QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY rgeh DESC, a) = 1) u USING (dev, det)""",
          mat=True)


def _low_runs(q: _Q):
    """note 154b: every run of >= 2 low hours of a detector (count <= half of what its pre-drop share of the yardstick
    gives, >= 10 expected; hours with less expected do not break a run) -> lvr: the periods of a count drop, so a drop
    that comes back each day is listed each day."""
    q.add("lvr", """
SELECT * FROM (
WITH h AS (SELECT g.dev, g.det, g.j, g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10 AS judged,
                  g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10
                  AND g.M <= 0.5 * CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R AS low
           FROM gh2 g JOIN gh4 a USING (dev, det)),
r AS (SELECT *, sum((judged AND NOT low)::INT) OVER (PARTITION BY dev, det ORDER BY j) AS grp FROM h)
SELECT dev, det, min(j) FILTER (low) AS ja, max(j) FILTER (low) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (low) >= 2)""", mat=True)


def _spiky(q: _Q, LJ: dict):
    """bursts in 10-min counts against its own level of the surrounding 2 h and the phase mates' own rise
    (health_counts.spiky) -> spk (per detector), spk_runs (the bursts)."""
    P = LJ["spiky"]
    k, wb = int(P["bin_min"] * 60 // 300), int(P["window_bins"])
    q.add("o10", f"""
SELECT g.dev, g.det, g.b // {k} AS t, sum(g.n) AS own, bool_and(g.cov) AS c
FROM g5 g JOIN win w USING (dev)
WHERE w.hours >= {_f(LJ['min_hours_counts'])} AND w.nb5 // {k} >= {2 * wb} AND g.b < w.nb5 // {k} * {k}
GROUP BY g.dev, g.det, g.b // {k}""", mat=True)
    lvl = ("CASE WHEN len(list_filter(coalesce({a}, []) || coalesce({b}, []), lambda x: x IS NOT NULL)) >= 3 THEN "
           "list_median(list_filter(coalesce({a}, []) || coalesce({b}, []), lambda x: x IS NOT NULL)) END")
    q.add("o10l", f"""
SELECT dev, det, t, own, c, m,
       list(CASE WHEN c THEN own END) OVER wl AS ol, list(CASE WHEN c THEN own END) OVER wr AS orr,
       list(CASE WHEN c THEN m END) OVER wl AS ml, list(CASE WHEN c THEN m END) OVER wr AS mr
FROM (SELECT o.*, coalesce(m.m, 0.0) AS m FROM o10 o LEFT JOIN (
        SELECT f.dev, f.det, o.t, sum(o.own) AS m FROM p1rf f JOIN o10 o ON o.dev = f.dev AND o.det = f.r
        WHERE f.same GROUP BY f.dev, f.det, o.t) m USING (dev, det, t))
WINDOW wl AS (PARTITION BY dev, det ORDER BY t ROWS BETWEEN {wb} PRECEDING AND 2 PRECEDING),
       wr AS (PARTITION BY dev, det ORDER BY t ROWS BETWEEN 2 FOLLOWING AND {wb} FOLLOWING)""")
    q.add("o10s", f"""
SELECT *, b IS NOT NULL AND c AND own / ex >= {_f(P['ratio'])} AND g >= {_f(P['geh'])} AND own > ex AS sp FROM (
  SELECT *, sqrt(2 * (own - ex) ^ 2 / greatest(own + ex, 1e-9)) AS g FROM (
    SELECT *, greatest(b, 1.0) * greatest(CASE WHEN m >= {_f(P['mates_min'])} AND bm IS NOT NULL
                                               THEN m / greatest(bm, 1.0) ELSE 1.0 END, 1.0) AS ex FROM (
      SELECT dev, det, t, own, c, m, {lvl.format(a='ol', b='orr')} AS b, {lvl.format(a='ml', b='mr')} AS bm FROM o10l)))""",
          mat=True)
    q.add("spk_runs", """
SELECT dev, det, min(t) AS a, max(t) + 1 AS z FROM (
  SELECT dev, det, t, t - row_number() OVER (PARTITION BY dev, det ORDER BY t) AS grp FROM o10s WHERE sp)
GROUP BY dev, det, grp""", mat=True)
    q.add("spk", f"""
SELECT s.*, CASE WHEN spk_daily < {_f(P['daily_none'])} THEN 0
                 WHEN spk_daily < {_f(P['daily_low'])} AND lv = 2 AND spk_gmax < {_f(P['geh_big'])} THEN 1 ELSE lv END
            AS spk_lvl
FROM (SELECT s.*, CASE WHEN spk_bursts >= {_f(P['bad'])} THEN 2 WHEN spk_bursts >= {_f(P['suspect'])} THEN 1 ELSE 0 END AS lv
      FROM (SELECT o.dev, o.det, coalesce(r.nr, 0) / greatest(w.hours / 24.0, 1.0) AS spk_bursts,
                   coalesce(max(o.g) FILTER (o.sp), 0.0) AS spk_gmax, sum(o.own) * 24.0 / greatest(w.hours, 1e-9) AS spk_daily
            FROM o10s o JOIN win w USING (dev)
            LEFT JOIN (SELECT dev, det, count(*) AS nr FROM spk_runs GROUP BY dev, det) r USING (dev, det)
            GROUP BY o.dev, o.det, r.nr, w.hours) s) s""")


def _holds(q: _Q, LJ: dict, R: dict):
    """holds (continuous ONs >= 5 min): held ON together with other detectors (common), and the queue context of each
    (health_v4_stats.episodes_ctx) -> ectx (one row per hold)."""
    tol = _f(LJ["common_hold_s"])
    q.add("hpair", f"""
SELECT a.hid, b.det AS dj, abs(b.s - a.s) <= {tol} AND abs(b.f - a.f) <= {tol} AS near
FROM hold a JOIN hold b ON a.dev = b.dev AND a.det <> b.det AND least(a.f, b.f) > greatest(a.s, b.s)
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det WHERE t.i IS NULL""")
    q.add("hcom", """
SELECT h.hid, coalesce(p.anyov, false) AS n_ov_any, coalesce(p.anyov AND p.allnear, false) AS common, p.mates
FROM hold h LEFT JOIN (SELECT hid, true AS anyov, bool_and(near) AS allnear, list(DISTINCT dj ORDER BY dj) AS mates
                       FROM hpair GROUP BY hid) p USING (hid)""", mat=True)
    qt = _values("type, qt", [[_s(a), _f(b)] for a, b in R["q_type"]])
    # the hold in bs-bins: [ia, iz) touched, full bins [fa, fz)
    q.add("hb", """
SELECT h.hid, h.dev, h.det, h.s, h.f, h.dur, x.ph AS p, w.bs, w.nbb, floor(h.s / w.bs)::INT AS ia,
       ceil(h.f / w.bs)::INT AS iz, ceil(h.s / w.bs)::INT AS fa, floor(h.f / w.bs)::INT AS fz
FROM hold h JOIN win w USING (dev) LEFT JOIN xbase x USING (dev, det)""", mat=True)
    # healthy phase mates not themselves held >= 99 % ON for >= 80 % of the hold
    q.add("hmate", """
SELECT h.hid, z.det AS r FROM hb h JOIN dz z ON z.dev = h.dev AND z.ph = h.p AND z.det <> h.det
WHERE z.status1 IS DISTINCT FROM 'bad' AND NOT (least(h.iz, h.nbb) > h.ia AND (
  SELECT count(*) FILTER (g.occ >= 0.99) FROM gb g WHERE g.dev = h.dev AND g.det = z.det
  AND g.k >= h.ia AND g.k < least(h.iz, h.nbb)) / (least(h.iz, h.nbb) - h.ia) >= 0.8)""", mat=True)
    q.add("hpm", f"""
SELECT m.hid, g.k, avg(g.occ) AS pm, bool_or(g.occ >= t.qt) AS hit
FROM hmate m JOIN hb h USING (hid) JOIN gb g ON g.dev = h.dev AND g.det = m.r
LEFT JOIN xm x ON x.dev = h.dev AND x.det = m.r LEFT JOIN ({qt}) t ON t.type = x.fn || ' ' || x.span
GROUP BY m.hid, g.k""")
    q.add("htr", """
SELECT h.hid, g.k, sum(coalesce(g.n, 0)) AS tot FROM hb h
JOIN (SELECT h.hid, z.det AS r, bool_or(z.tr) OVER (PARTITION BY h.hid) AS anytr, z.tr
      FROM hb h JOIN dz z ON z.dev = h.dev AND z.ph = h.p AND z.det <> h.det) m ON m.hid = h.hid AND (m.tr OR NOT m.anytr)
JOIN gb g ON g.dev = h.dev AND g.det = m.r GROUP BY h.hid, g.k""")
    q.add("hrh", """
SELECT h.hid, bool_or(g.occ >= 0.95) AS ref_held_e FROM hb h JOIN rmm m ON m.dev = h.dev AND m.det = h.det
JOIN gb g ON g.dev = h.dev AND g.det = m.r AND g.k >= h.ia AND g.k < least(h.iz, h.nbb) GROUP BY h.hid""")
    q.add("ectx", """
SELECT h.hid, h.dev, h.det, h.s, h.f, h.dur, h.p,
  coalesce((SELECT count(*) FROM hmate m WHERE m.hid = h.hid), 0) AS n_hpeer_e, b.phx_h_e, b.corr_h_e, b.cover_e,
  b.refx_e, b.light_e, b.busy_min_e, b.busy_occ_e, coalesce(r.ref_held_e, false) AS ref_held_e, t.trafx_e, c.co AS co5,
  m.n_ov_any, m.common, m.mates
FROM hb h JOIN hold_co c USING (hid) JOIN hcom m USING (hid) LEFT JOIN hrh r USING (hid)
LEFT JOIN (
  SELECT h.hid,
    CASE WHEN any_value(np.hid) IS NOT NULL AND count(*) FILTER (g.k >= h.ia AND g.k < h.iz) > 0 THEN
      avg(p.pm) FILTER (g.k >= h.ia AND g.k < h.iz) / greatest(avg(p.pm), 1e-9) END AS phx_h_e,
    CASE WHEN any_value(np.hid) IS NOT NULL AND count(*) FILTER (NOT (g.k >= h.ia AND g.k < h.iz) AND g.occ IS NOT NULL
         AND p.pm IS NOT NULL) >= 4 THEN nullif(corr(p.pm, g.occ) FILTER (NOT (g.k >= h.ia AND g.k < h.iz)
         AND g.occ IS NOT NULL AND p.pm IS NOT NULL), 'nan'::DOUBLE) END AS corr_h_e,
    CASE WHEN any_value(np.hid) IS NOT NULL AND count(*) FILTER (g.k >= h.ia AND g.k < h.iz) > 0 THEN
      count(*) FILTER (g.k >= h.ia AND g.k < h.iz AND p.hit) / count(*) FILTER (g.k >= h.ia AND g.k < h.iz) END AS cover_e,
    CASE WHEN count(*) FILTER (g.k >= h.ia AND g.k < h.iz) > 0 THEN
      avg(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) / greatest(avg(g.ref), 1e-9) END AS refx_e,
    count(*) FILTER (NOT (g.k >= h.ia - 1 AND g.k <= h.iz) AND g.occ >= 0.9 AND g.ref <= 0.3 * mr.rm
                     AND (mr.mpo IS NULL OR NOT coalesce(g.peer > mr.mpo, false))) AS light_e,
    CASE WHEN count(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) = 0 THEN 0.0
         ELSE count(*) FILTER ((g.k < h.ia - 1 OR g.k > h.iz) AND g.ref IS NOT NULL AND g.occ IS NOT NULL
                               AND g.ref >= mr.rh) * any_value(h.bs) / 60.0 END AS busy_min_e,
    CASE WHEN count(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) > 0 THEN
      median(g.occ) FILTER ((g.k < h.ia - 1 OR g.k > h.iz) AND g.ref IS NOT NULL AND g.occ IS NOT NULL AND g.ref >= mr.rh)
      END AS busy_occ_e
  FROM hb h JOIN gctx g ON g.dev = h.dev AND g.det = h.det
  JOIN (SELECT h.hid, avg(g.ref) AS rm, avg(g.peer) AS mpo, avg(g.ref) FILTER (g.k >= h.ia AND g.k < h.iz) AS rh
        FROM hb h JOIN gctx g ON g.dev = h.dev AND g.det = h.det GROUP BY h.hid) mr ON mr.hid = h.hid
  LEFT JOIN hpm p ON p.hid = h.hid AND p.k = g.k
  LEFT JOIN (SELECT DISTINCT hid FROM hmate) np ON np.hid = h.hid
  GROUP BY h.hid) b USING (hid)
LEFT JOIN (
  SELECT h.hid, CASE WHEN count(*) FILTER (t.k >= greatest(h.ja, 0) AND t.k < least(h.jz, h.nbb)) > 0 THEN
    avg(t.tot) FILTER (t.k >= greatest(h.ja, 0) AND t.k < least(h.jz, h.nbb)) / greatest(avg(t.tot), 1e-9) END AS trafx_e
  FROM (SELECT *, CASE WHEN fz > fa THEN fa ELSE ia END AS ja, CASE WHEN fz > fa THEN fz ELSE iz END AS jz FROM hb) h
  JOIN htr t USING (hid) GROUP BY h.hid) t USING (hid)""", mat=True)
# ============================================================================ limits, scores, verdict (health_v4)
KEYS = ("stuck", "count_on", "dropout", "level", "night_drop", "prof", "rapid", "volume", "spiky", "rise")
STAT = ("dropout", "rapid", "level", "night_drop", "prof", "spiky", "rise")
FAMILY = {"rapid": "fast", "dropout": "silent", "stuck": "stuck", "level": "level", "night_drop": "level",
          "volume": "shape", "prof": "shape", "count_on": "stuck", "spiky": "shape", "rise": "rise"}
EVID = {"stuck": "stuck", "count_on": "stuck", "dropout": "quiet", "level": "quiet", "night_drop": "quiet",
        "rapid": "fast", "volume": "fast", "prof": "prof", "spiky": "spiky", "rise": "rise"}
CATEGORY = {"stuck": "Stuck on", "count_on": "Stuck on", "dropout": "Counts fall", "level": "Counts fall",
            "night_drop": "Misses vehicles at night", "prof": "Busy at night", "rapid": "Too-fast actuations",
            "volume": "Too many for the traffic", "spiky": "Spiky counts", "rise": "Counts too high"}
NO_SHARED = ("Advance", "Count", "Yellow_Red")
DAYNS = 86_400_000_000_000


def _sc(x: str, lo: str, bad: str) -> str:
    """health_v4.score: 0 below the suspect limit, .35 at it, .75 at the bad limit (linear), 1 at most; NULL = not
    checked."""
    return (f"(CASE WHEN ({x}) IS NULL OR isnan({x}) OR ({lo}) IS NULL OR isnan({lo}) THEN NULL WHEN ({x}) < ({lo}) "
            f"THEN 0.0 ELSE least(0.35 + 0.40 * greatest((({x}) - ({lo})) / greatest(({bad}) - ({lo}), 1e-9), 0.0), 1.0) END)")


def _limits(q: _Q, LJ: dict, R: dict):
    """per-detector [suspect, bad] limits of every check (health_v4.limits); the least strict plausible function when
    the model is unsure."""
    h, cur, sh = LJ["h24"], LJ["current"], LJ["short"]
    rows = []
    for t in ("stuck_s", "silent_exp"):
        for k, (a, b) in h[t].items():
            rows.append([_s(t), _s(k), _f(a), _f(b)])
    for k, v in h["busy_z"].items():
        rows.append(["'busy_z'", _s(k), _f(v), "NULL"])
    for k, v in LJ["too_fast"]["per_1000"].items():
        rows.append(["'per_1000'", _s(k), _f(v), "NULL"])
    q.add("ltab", _values("tab, key, lo, hi", rows), mat=True)
    lv = []
    for tab, levels in (("stuck_x", R["cell"]["v4c"]["stuck_x"]), ("n_ep", R["cell"]["v4c"]["n_ep"]),
                        ("q5_gy", R["fast118c"]["q5_gy"]), ("q5_all", R["fast118c"]["q5_all"])):
        for i, lev in enumerate(levels):
            for r in lev["rows"]:
                v = r[-1]
                if v is None or v != v:
                    continue
                key = dict(zip(lev["keys"], (str(x) for x in r[:-1])))
                lv.append([_s(tab), str(i)] + [_s(key[c]) if c in key else "NULL"
                                                for c in ("fn", "span", "cmode", "wg", "band")] + [_f(v)])
    q.add("lref", _values("tab, lvl, fn, span, cmode, wg, band, v", lv), mat=True)
    # the functions whose limits apply: its own, and when the model is unsure (top < .7) every plausible one
    q.add("lcand", """
SELECT x.dev, x.det, x.fn AS fa, true AS own, x.span, x.cmode, x.wg, x.band,
       CASE WHEN x.fn = 'Count' THEN '|' || CASE WHEN x.cmode = 'normal' THEN 'normal' ELSE 'pulse' END ELSE '' END AS cms,
       x.cmode AS cmr
FROM xx0 x
UNION ALL
SELECT x.dev, x.det, a.fa, false, x.span, x.cmode, x.wg, x.band,
       CASE WHEN a.fa = 'Count' THEN '|' || CASE WHEN x.dur_p50 <= 0.25 THEN 'pulse' ELSE 'normal' END ELSE '' END,
       CASE WHEN a.fa <> 'Count' THEN '' WHEN x.dur_p50 IS NULL THEN 'unknown' WHEN x.dur_p50 <= 0.25 THEN 'pulse'
            ELSE 'normal' END
FROM xx0 x, (SELECT unnest(x2.alts) AS fa, x2.dev, x2.det FROM xx0 x2) a
WHERE a.dev = x.dev AND a.det = x.det AND a.fa <> x.fn""", mat=True)

    def tab(t, col="lo", split=False):
        key = "c.fa || '|' || c.span" + (" || c.cms" if split else "")
        return f"(SELECT max(l.{col}) FROM ltab l WHERE l.tab = '{t}' AND l.key = {key})"

    def ref(t, cm_fn=False):
        cm = "c.cmr" if cm_fn else "c.cmode"
        return (f"(SELECT first(l.v ORDER BY l.lvl) FROM lref l WHERE l.tab = '{t}' AND (l.fn IS NULL OR l.fn = c.fa) "
                f"AND (l.span IS NULL OR l.span = c.span) AND (l.cmode IS NULL OR l.cmode = {cm}) "
                f"AND (l.wg IS NULL OR l.wg = c.wg) AND (l.band IS NULL OR l.band = c.band))")
    q.add("lval", f"""
SELECT c.dev, c.det, c.own, {ref('n_ep')} AS n_ep, {tab('per_1000')} AS rapid,
       {tab('stuck_s', 'lo', True)} AS st_lo, {tab('stuck_s', 'hi', True)} AS st_hi,
       {tab('silent_exp', 'lo')} AS si_lo, {tab('silent_exp', 'hi')} AS si_hi,
       {ref('stuck_x', True)} AS stuck_x, {ref('q5_gy')} AS q5gy, {ref('q5_all')} AS q5all
FROM lcand c""")
    tf, a, b, c = LJ["too_fast"], *sh["stuck_bad"]
    q.add("limits", f"""
SELECT x.dev, x.det,
  v.n_ep + 1 AS lim_n_ep, (v.n_ep + 1) * {_f(sh['n_holds_bad_x'])} AS bad_n_ep,
  v.rapid AS lim_rapid, v.rapid * {_f(tf['bad_x'])} AS bad_rapid,
  {_f(LJ['count_drops_geh'][0])} AS lim_level, {_f(LJ['count_drops_geh'][1])} AS bad_level,
  {_f(LJ['night_geh'][0])} AS lim_night_drop, {_f(LJ['night_geh'][1])} AS bad_night_drop,
  {_f(LJ['count_held_on_min'][0])} AS lim_count_on, {_f(LJ['count_held_on_min'][1])} AS bad_count_on,
  CASE WHEN x.wg = 'h24' THEN greatest(v.st_lo, {_f(LJ['hold_floor_s'])})
       ELSE least(greatest(v.sx, 300.0), 900.0) END AS lim_stuck,
  CASE WHEN x.wg = 'h24' THEN greatest(v.st_hi, v.st_lo, {_f(LJ['hold_floor_s'])})
       ELSE greatest({_f(a)} + {_f(b)} * least(greatest(v.sx, 300.0), 900.0), {_f(c)} * least(greatest(v.sx, 300.0), 900.0))
       END AS bad_stuck,
  CASE WHEN x.wg = 'h24' THEN v.si_lo ELSE {_f(cur['silent_exp'])} END AS lim_dropout,
  CASE WHEN x.wg = 'h24' THEN {"greatest(v.si_hi, v.si_lo * " + _f(sh['silent_bad_x']) + ")"
                               if q.opt['silent_limits'] else "v.si_hi"}
       ELSE {_f(cur['silent_exp'] * sh['silent_bad_x'])} END AS bad_dropout,
  CASE WHEN x.wg = 'h24' THEN (SELECT max(l.lo) FROM ltab l WHERE l.tab = 'busy_z' AND l.key = x.fn || '|' || x.span)
       ELSE {_f(cur['busy_z'])} END AS lim_prof,
  CASE WHEN x.wg = 'h24' THEN {_f(LJ['too_many_h24_veh_h_lane'][0])}
       ELSE greatest(CASE WHEN x.fn = 'Advance' THEN v.q5all ELSE v.q5gy END, {_f(cur['too_many_veh_h_lane'])}) END
       AS lim_volume,
  CASE WHEN x.wg = 'h24' THEN {_f(LJ['too_many_h24_veh_h_lane'][1])}
       ELSE greatest(CASE WHEN x.fn = 'Advance' THEN v.q5all ELSE v.q5gy END, {_f(cur['too_many_veh_h_lane'])})
            * {_f(sh['too_many_bad_x'])} END AS bad_volume
FROM xx0 x JOIN (
  SELECT dev, det, max(n_ep) AS n_ep, max(rapid) AS rapid, max(st_lo) AS st_lo, max(st_hi) AS st_hi,
         max(si_lo) AS si_lo, max(si_hi) AS si_hi, max(q5gy) AS q5gy, max(q5all) AS q5all,
         CASE WHEN bool_or(own AND stuck_x IS NOT NULL) THEN max(stuck_x) END AS sx
  FROM lval GROUP BY dev, det) v USING (dev, det)""", mat=True)


def _tod(q: _Q, LJ: dict, R: dict):
    """busy at night (24-h samples, h116): per whole day the night level against its type's model and its phase mates;
    the day with the strongest finding is kept; a signal note when many detectors had an unusual night."""
    g = R["tod"]["groups"]
    rows = [[_s(k), _f(v["bc"][0]), _f(v["bc"][1]), _f(v["bc"][2]), _f(v["sc"]), "true" if v["occ"] else "false",
             _f(v["bo"][0]) if v["occ"] else "NULL", _f(v["bo"][1]) if v["occ"] else "NULL",
             _f(v["so"]) if v["occ"] else "NULL", _f(v["cal_med"]), _f(v["cal_mad"])] for k, v in g.items()]
    q.add("todg", _values("key, b0, b1, b2, sc, occ, o0, o1, so, med, mad", rows), mat=True)
    P = LJ["busy"]
    # the whole days (00:00-24:00) of each 24-h window; one not starting at midnight is scored by clock hour as one day
    q.add("tdays", f"""
WITH w AS (SELECT dev, s_ns, e_ns, nbb, T, s_ns - sod + CASE WHEN sod > 0 THEN {DAYNS} ELSE 0 END AS f0
           FROM win WHERE wg = 'h24' AND bs = 900),
d AS (SELECT dev, s_ns, nbb, f0 + unnest(range((e_ns - f0) // {DAYNS})) * {DAYNS} AS d0 FROM w
      WHERE e_ns >= f0 + {DAYNS}
      UNION ALL SELECT dev, s_ns, nbb, s_ns FROM w WHERE e_ns < f0 + {DAYNS} AND T >= 86400)
SELECT dev, s_ns, nbb, d0, round_even((d0 - s_ns) / 9e11, 0)::INT AS j0,
       ((d0 % {DAYNS}) // 1000000000 // 900) % 96 AS sh
FROM d WHERE round_even((d0 - s_ns) / 9e11, 0) >= 0 AND round_even((d0 - s_ns) / 9e11, 0) + 96 <= nbb""", mat=True)
    q.add("tdh", """
SELECT dev, det, d0, hr, CASE WHEN count(H) = 2 THEN sum(H) END AS N, avg(Q) AS O FROM (
  SELECT g.dev, g.det, t.d0, ((g.k - t.j0 + t.sh) % 96) // 2 AS hh, ((g.k - t.j0 + t.sh) % 96) // 4 AS hr,
         CASE WHEN count(g.n) > 0 THEN sum(g.n) END AS H, avg(g.occ) AS Q
  FROM tdays t JOIN gb g ON g.dev = t.dev AND g.k >= t.j0 AND g.k < t.j0 + 96
  JOIN xbase x ON x.dev = g.dev AND x.det = g.det GROUP BY g.dev, g.det, t.d0, hh, hr)
GROUP BY dev, det, d0, hr""")
    q.add("tdf", """
SELECT a.*, sqrt((a.nn + 0.5) / (b.b4n + 0.5)) AS r, asin(sqrt(a.onm)) AS onv,
       asin(sqrt(least(greatest(b.b4o, 0.0), 1.0))) AS od, log10(1 + a.tot) AS lv,
       CASE WHEN a.tot / 24.0 < 20 THEN 'low' WHEN a.tot / 24.0 < 100 THEN 'medium' ELSE 'high' END AS band
FROM (SELECT dev, det, d0, sum(N) AS tot, avg(N) FILTER (hr >= 1 AND hr < 5) AS nn, sum(N) FILTER (hr >= 1 AND hr < 5)
             AS n_night, avg(least(greatest(O, 0.0), 1.0)) FILTER (hr >= 1 AND hr < 5) AS onm
      FROM tdh GROUP BY dev, det, d0 HAVING count(N) = 24 AND count(O) = 24 AND sum(N) >= 50) a
JOIN (SELECT a.dev, a.det, a.d0, max(s4n) / 4 AS b4n, max(s4o) / 4 AS b4o FROM (
        SELECT a.dev, a.det, a.d0, a.hr, sum(b.N) AS s4n, sum(least(greatest(b.O, 0.0), 1.0)) AS s4o
        FROM tdh a JOIN tdh b ON b.dev = a.dev AND b.det = a.det AND b.d0 = a.d0 AND (b.hr - a.hr + 24) % 24 < 4
        GROUP BY a.dev, a.det, a.d0, a.hr) a GROUP BY a.dev, a.det, a.d0) b USING (dev, det, d0)""")
    # z of the detector's own type, and the smallest over the plausible functions when the model is unsure
    q.add("tdz", """
SELECT f.dev, f.det, f.d0, c.own, ((greatest((f.r - (g.b0 + g.b1 * f.od + g.b2 * f.lv)) / g.sc,
       CASE WHEN g.occ THEN (f.onv - (g.o0 + g.o1 * f.lv)) / g.so END)) - g.med) / g.mad AS z
FROM tdf f JOIN (SELECT dev, det, fa, own FROM lcand) c USING (dev, det) JOIN xbase x USING (dev, det)
JOIN (SELECT f.dev, f.det, f.d0, c.fa, first(g.key ORDER BY o.o) AS key
      FROM tdf f JOIN lcand c USING (dev, det) JOIN xbase x USING (dev, det),
      (SELECT 1 AS o UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) o, todg g
      WHERE g.key = CASE o.o WHEN 1 THEN c.fa || '|' || x.span || '|' || f.band WHEN 2 THEN c.fa || '|*|' || f.band
                             WHEN 3 THEN c.fa || '|' || x.span || '|*' ELSE c.fa || '|*|*' END
      GROUP BY f.dev, f.det, f.d0, c.fa) k ON k.dev = f.dev AND k.det = f.det AND k.d0 = f.d0 AND k.fa = c.fa
JOIN todg g ON g.key = k.key""")
    q.add("tdd", f"""
SELECT f.*, z.z_own AS tod_z0, CASE WHEN z.z_own IS NOT NULL THEN least(z.z_own, z.z_alt) END AS tod_z,
       f.r - m.mate_r AS tod_dph, m.mate_r AS tod_mate_r, l.lim_prof
FROM tdf f JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS z_own, min(z) FILTER (NOT own) AS z_alt FROM tdz
                 GROUP BY dev, det, d0) z USING (dev, det, d0)
JOIN xbase x USING (dev, det) JOIN limits l USING (dev, det)
LEFT JOIN (SELECT a.dev, a.det, a.d0, median(b.r) AS mate_r FROM tdf a JOIN xbase xa USING (dev, det)
           JOIN xbase xb2 ON xb2.dev = a.dev AND xb2.ph = xa.ph AND xb2.det <> a.det
           JOIN tdf b ON b.dev = a.dev AND b.det = xb2.det AND b.d0 = a.d0
           JOIN (SELECT dev, det, d0, max(z) FILTER (own) AS zo FROM tdz GROUP BY dev, det, d0) bz
             ON bz.dev = b.dev AND bz.det = b.det AND bz.d0 = b.d0
           WHERE bz.zo IS NOT NULL AND b.tot >= 200 GROUP BY a.dev, a.det, a.d0) m USING (dev, det, d0)""")
    q.add("tdl", f"""
SELECT *, CASE WHEN tod_z IS NULL THEN NULL
               WHEN n_night >= {_f(P['min_night'])} AND coalesce(tod_dph, -1) >= {_f(P['mates_delta'])}
                    AND tod_z >= lim_prof * {_f(P['z_x'])} AND tod_z >= lim_prof THEN 0.8
               WHEN n_night >= {_f(P['min_night'])} AND coalesce(tod_dph, -1) >= {_f(P['mates_delta'])}
                    AND tod_z >= lim_prof * {_f(P['z_x'])} THEN 0.5 ELSE 0.0 END AS lvl
FROM tdd""")
    q.add("tod", """
SELECT dev, det, first(tod_z ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_z,
       first(tod_dph ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_dph,
       first(n_night ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_n_night,
       first(r ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_r, first(tod_mate_r ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_mate_r,
       first(d0 ORDER BY coalesce(lvl, -1) DESC, d0) AS tod_d0
FROM tdl GROUP BY dev, det""")
    q.add("todnote", """
SELECT dev, first(d0 ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS d0, first(n_odd ORDER BY n_odd / greatest(n_ok, 1)
       DESC, d0) AS n_odd, first(n_ok ORDER BY n_odd / greatest(n_ok, 1) DESC, d0) AS n_ok
FROM (SELECT dev, d0, count(*) FILTER (tod_z >= 3.5) AS n_odd, count(*) AS n_ok FROM tdd GROUP BY dev, d0)
WHERE n_odd >= 5 AND n_odd >= 0.20 * n_ok GROUP BY dev""")


def _verdict(q: _Q, p: str, src: str, LJ: dict, lc: bool = False):
    """the check scores and the resolver / severity (health_v4.scores + status) for the rows of `src` -> {p}V; lc: `src`
    carries lc_d / lc_l (a silence / count drop explained by a lane closure: not scored)."""
    P = LJ["busy"]
    sd = _sc("x.drop_lam", "l.lim_dropout", "l.bad_dropout")
    g0, g1 = (_f(v) for v in MATE_HOURS["geh"])
    # note 154n: a run of low / high hours against the phase mates (a lane closure explains a drop)
    md = (f"CASE WHEN x.fn = 'Bike' THEN NULL {'WHEN x.lc_l OR x.lc_md THEN 0.0' if lc else ''} WHEN x.md_h IS NOT NULL "
          f"THEN {_sc('x.md_geh', g0, g1)} WHEN x.mh_ok THEN 0.0 END" if q.opt["mate_drop"] else "NULL::DOUBLE")
    rise = (f"CASE WHEN x.fn = 'Bike' THEN NULL {'WHEN x.lc_mr THEN 0.0' if lc else ''} "
            f"WHEN x.mr_h IS NOT NULL THEN {_sc('x.mr_geh', g0, g1)} "
            f"WHEN x.mh_ok THEN 0.0 END" if q.opt["count_rise"] else "NULL::DOUBLE")
    lcd = "WHEN x.lc_d THEN 0.0" if lc else ""
    q.add(f"{p}S", f"""
SELECT x.dev, x.det, s.s_stuck_v AS v_stuck,
  CASE WHEN x.count_on_ok THEN {_sc('coalesce(x.hi_min_c, 0)', 'l.lim_count_on', 'l.bad_count_on')} END AS v_count_on,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.silent THEN 0.0 {lcd}
       {"WHEN x.drop_b1 - x.drop_b0 < 3 THEN least(x.sd, 0.5)" if q.opt['silent_limits'] else ""} ELSE x.sd END AS v_dropout,
  CASE WHEN x.fn = 'Bike' OR NOT x.geh_ok THEN NULL {"WHEN x.lc_l THEN 0.0" if lc else ""}
       {"WHEN x.geh_drop IS NOT NULL AND x.occ_own >= 3.0 AND x.occ_own >= 3.0 * coalesce(x.occ_mates, 1.0) THEN " + _sc('x.geh_drop', 'l.lim_level', 'l.bad_level')
        + " WHEN x.occ_own >= 1.0 AND coalesce(x.occ_after, 0) >= 0.25 THEN 0.0" if q.opt['long_zone'] else ""}
       WHEN x.geh_drop IS NOT NULL AND ((x.occ_mates IS NOT NULL AND x.occ_own / greatest(x.occ_mates, 1e-9) < 0.8)
            {"OR (x.occ_mates IS NULL AND x.occ_own < 0.8) OR x.own_after < 0.1 * x.exp_after"
             if q.opt['drop_no_mates'] else ""})
       THEN {_sc('x.geh_drop', 'l.lim_level', 'l.bad_level')} ELSE 0.0 END AS v_lvl_old,
  {md} AS v_lvl_md,
  CASE WHEN v_lvl_old IS NULL AND v_lvl_md IS NULL THEN NULL
       ELSE greatest(coalesce(v_lvl_old, 0.0), coalesce(v_lvl_md, 0.0)) END AS v_level,
  {rise} AS v_rise,
  CASE WHEN x.fn = 'Bike' THEN NULL {"WHEN x.moved AND x.lc_d THEN 0.0" if lc else ""}
       {"WHEN x.moved AND coalesce(x.drop_lam, 0) < 50 THEN 0.0" if q.opt["silent_limits"] else ""}
       WHEN x.moved AND coalesce(x.drop_lam, 0) < 100 THEN least(x.sd, 0.5)
       WHEN x.moved THEN x.sd ELSE x.nd END AS v_night_drop,
  CASE WHEN x.fn <> 'Bike' AND coalesce(CASE WHEN x.silent THEN 0.0 ELSE x.sd END, 0)
            >= {"0.42" if q.opt["no_recovery_cap"] else "0.35"}
       THEN CASE WHEN x.pv IS NOT NULL THEN 0.0 END ELSE x.pv END AS v_prof,
  CASE WHEN x.N >= 50 THEN {_sc('x.xr + 1e-9 * (x.xr > 0)::INT', '1.0', 'l.bad_rapid / l.lim_rapid')} END AS v_rapid,
  CASE WHEN x.q5_use IS NOT NULL THEN {_sc('x.q5_use / l.lim_volume', '1.0', 'l.bad_volume / l.lim_volume')} END
       AS v_volume,
  CASE WHEN x.spk_lvl >= 2 THEN 1.0 WHEN x.spk_lvl >= 1 THEN 0.5 WHEN x.spk_lvl IS NOT NULL THEN 0.0 END AS v_spiky,
  x.silent AS silent_night
FROM (SELECT x.*, coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AS silent,
             coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AND coalesce(x.sd0, 0) > coalesce(x.nd, 0) + 1e-9 AS moved,
             x.sd0 AS sd
      FROM (SELECT x.*, CASE WHEN x.n_on > 0 THEN CASE WHEN x.drop_in_stuck THEN 0.0
                 {"" if q.opt["no_recovery_cap"] else
                  f"WHEN x.s_dropout = 0.35 AND x.drop_lam > 30.001 AND {sd} > 0.35 THEN 0.35"} ELSE {sd} END END AS sd0,
                   {_sc('x.geh_night', 'l.lim_night_drop', 'l.bad_night_drop')} AS nd,
                   CASE {"WHEN x.flat THEN 0.8 WHEN x.qzone THEN 0.0" if q.opt["no_daily_pattern"] else ""} WHEN x.tod_z IS NULL THEN NULL
                        WHEN x.tod_n_night >= {_f(P['min_night'])} AND coalesce(x.tod_dph, -1) >= {_f(P['mates_delta'])}
                             AND x.tod_z >= l.lim_prof * {_f(P['z_x'])} AND x.tod_z >= l.lim_prof
                             {"AND x.tod_z >= 1.5 * l.lim_prof" if q.opt["busy_ease"] else ""} THEN 0.8
                        WHEN x.tod_n_night >= {_f(P['min_night'])} AND coalesce(x.tod_dph, -1) >= {_f(P['mates_delta'])}
                             AND x.tod_z >= l.lim_prof * {_f(P['z_x'])} THEN 0.5 ELSE 0.0 END AS pv,
                   CASE WHEN x.n_fast >= 3 THEN x.fast_rate / l.lim_rapid ELSE 0.0 END AS xr,
                   CASE WHEN x.drop_b1 > x.drop_b0 THEN (SELECT count(*) FILTER (b.hr >= 21 OR b.hr < 5) FROM bins5 b
                        WHERE b.dev = x.dev AND b.b >= x.drop_b0 AND b.b < x.drop_b1) / (x.drop_b1 - x.drop_b0)
                        ELSE 0.0 END AS nf
            FROM {src} x JOIN limits l USING (dev, det)) x) x
JOIN limits l USING (dev, det) JOIN stk s USING (dev, det)""",
          mat=True)
    fk =", ".join(f"coalesce(v_{k} >= 0.35, false) AS f_{k}" for k in KEYS)
    q.add(f"{p}F0", f"""
SELECT s.*, {fk}, x.queue_pat8, x.c_occ_like, x.hours, x.status1, x.status AS status2, x.n_hmates, x.n_on,
       x.drop_b0, x.drop_b1, x.fn, x.c1_pre, x.hi_min_c, x.fe_n, x.fe_phi{", x.drop_lam" if q.opt["no_recovery_cap"] else ""}
FROM {p}S s JOIN {src} x USING (dev, det)""")
    # Q2 (queue pattern explains a count drop), R6 (silence in a sample under 1 h: watch), counts fall = one finding
    q.add(f"{p}F1", f"""
SELECT * REPLACE (
    CASE WHEN f_dropout AND f_level AND NOT (v_dropout >= v_level) THEN false ELSE f_dropout END AS f_dropout,
    CASE WHEN f_dropout AND f_level AND v_dropout >= v_level THEN false ELSE f_level END AS f_level)
FROM (SELECT * REPLACE (f_level AND NOT (queue_pat8 AND coalesce(c_occ_like >= 0.8, false)
                                         AND NOT coalesce(v_lvl_md > v_lvl_old, false)) AS f_level,
                        f_dropout AND NOT (hours < 1) AS f_dropout),
             f_dropout AND hours < 1 AS w_dropout_short,
             status2 <> status1 AND n_hmates = 0 AND status1 IN ('suspect', 'bad') AS w_no_yardstick
      FROM {p}F0)""")
    stat_border = " OR ".join(f"(f_{k} AND v_{k} < 0.42)" for k in STAT)
    nf = " + ".join(f"f_{k}::INT" for k in KEYS)
    q.add(f"{p}F2", f"""
SELECT *, ({nf}) = 1 AND ({stat_border}) AS r7 FROM {p}F1""")
    keep = ", ".join(f"f_{k} AND NOT r7 AS f_{k}" for k in KEYS)
    bord = ", ".join(f"f_{k} AND r7 AS b_{k}" for k in KEYS)
    q.add(f"{p}F3", f"SELECT * REPLACE ({keep}), {bord} FROM {p}F2")
    prod = " * ".join(f"(1 - CASE WHEN f_{k} THEN v_{k} ELSE 0 END)" for k in KEYS)
    anyfin = " OR ".join(f"v_{k} IS NOT NULL" for k in KEYS)

    def real(k):
        if k == "dropout" and q.opt["no_recovery_cap"]:   # note 154f: a short or small silence is not a 2nd finding
            return "(f_dropout AND NOT coalesce(v_dropout = 0.35, false) AND drop_b1 - drop_b0 >= 3 AND drop_lam >= 50)"
        return f"(f_{k} AND NOT coalesce(v_{k} = 0.35, false))" if k in ("stuck", "dropout") else f"f_{k}"
    fams = sorted(set(FAMILY.values()))
    nfam = " + ".join("(" + " OR ".join(real(k) for k in KEYS if FAMILY[k] == f) + ")::INT" for f in fams)
    evs = sorted(set(EVID.values()))
    nev = " + ".join("(" + " OR ".join(real(k) for k in KEYS if EVID[k] == e) + ")::INT" for e in evs)
    q.add(f"{p}F4", f"""
SELECT *, CASE WHEN {anyfin} THEN {prod} END AS score, {nev} AS n_ev, {nfam} AS n_fam,
       (drop_b1 - drop_b0) * 300 / 3600.0 AS silent_h,
       CASE WHEN {prod} < 0.25 AND ({anyfin}) THEN 'bad' WHEN {prod} < 0.70 AND ({anyfin}) THEN 'suspect'
            WHEN r7 OR w_dropout_short OR w_no_yardstick THEN 'watch' WHEN n_on < 20 THEN 'not_enough_data'
            ELSE 'ok' END AS st0
FROM {p}F3""")
    q.add(f"{p}V", f"""
SELECT * EXCLUDE (st0, st1, st2, st3),
       CASE WHEN st3 = 'bad' AND f_rapid AND ({nf}) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'suspect' ELSE st3 END AS health_st,
       CASE WHEN st3 = 'bad' AND f_rapid AND ({nf}) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'F'
            WHEN st3 <> st2 THEN 'C' WHEN st2 <> st1 THEN 'P' WHEN st1 <> st0 AND n_fam < 2 THEN 'A' ELSE '' END AS sev_rule
FROM (SELECT *, CASE WHEN fn = 'Count' AND c1_pre AND hi_min_c >= 30 AND st2 IN ('ok', 'watch') THEN 'suspect'
                     ELSE st2 END AS st3
      FROM (SELECT *, CASE WHEN st1 = 'suspect' AND (f_prof OR (f_dropout AND silent_h >= 4.0)) THEN 'bad' ELSE st1 END
                   AS st2
            FROM (SELECT *, CASE WHEN st0 = 'suspect' AND (n_fam >= 2 OR n_ev >= 2) THEN 'bad' ELSE st0 END AS st1
                  FROM {p}F4)))""", mat=True)


def _assemble(q: _Q, LJ: dict, R: dict):
    """the scored table X, holds with their flags, limits, the pre-pass and the clean silence yardstick -> xf, vf."""
    mh_on = q.opt["mate_drop"] or q.opt["count_rise"]
    mhc = (", coalesce(mb.mh_ok, false) AS mh_ok, mb.md_h, mb.md_geh, mb.md_r, mb.md_a, mb.md_z, mb.md_own, mb.md_exp, "
           "mb.mr_h, mb.mr_geh, mb.mr_r, mb.mr_a, mb.mr_z" if mh_on else "")
    mhj = "\nLEFT JOIN mhbest mb USING (dev, det)" if mh_on else ""
    q.add("xx0", f"""
SELECT x.*, a.rep_frac, a.max_on_s, a.rep_time_s, a.dur_p50, a.n_on_a, a.n_rep, a.n_start, a.n_ge5, a.n_ge60,
       a.long_max_s, s.elhi_cnt, s.elhi_occ, k.c_occ_like, q.q5_gy, q.q5_all, f.fe_n, f.fe_p, f.fe_phi, t.N, t.n_fast,
       t.fast_rate, t.gap_lim, t.pair_share, x.n_on / x.hours AS rate,
       CASE WHEN x.n_on / x.hours < 20 THEN 'low' WHEN x.n_on / x.hours < 100 THEN 'medium' ELSE 'high' END AS band,
       x.fn || ' ' || x.span AS type, greatest(x.ep_dur, a.max_on_s) AS stuck_x,
       coalesce(s.elhi_occ > 0 AND s.elhi_occ > s.elhi_cnt, false) AS queue_pat8,
       CASE WHEN x.ph IS NOT NULL THEN hm.nh - (x.status1 <> 'bad')::INT END AS n_hmates,
       o.hi_min_c, o.hi_occ_max, coalesce(g.geh_ok, false) AS geh_ok, g.geh_night,
       CASE WHEN g.geh_night IS NOT NULL THEN g.geh_night_n0 END AS geh_night_n,
       CASE WHEN g.geh_night IS NOT NULL THEN g.geh_night_exp0 END AS geh_night_exp, g.geh_drop, g.drop_t, g.drop_t2,
       g.occ_after,
       g.own_before, g.own_after, g.exp_after, g.occ_own, g.occ_mates, p.spk_bursts, p.spk_gmax, p.spk_daily, p.spk_lvl,
       m.mocc AS occ{mhc}
FROM xm x LEFT JOIN act a USING (dev, det) LEFT JOIN slp s USING (dev, det) LEFT JOIN lkc k USING (dev, det)
LEFT JOIN q5 q USING (dev, det) LEFT JOIN fe f USING (dev, det) LEFT JOIN tf t USING (dev, det)
LEFT JOIN ohi o USING (dev, det) LEFT JOIN geh g USING (dev, det) LEFT JOIN spk p USING (dev, det)
LEFT JOIN (SELECT DISTINCT dev, det, mocc FROM gb) m USING (dev, det){mhj}
LEFT JOIN (SELECT dev, ph, sum((status1 <> 'bad')::INT) AS nh FROM xm GROUP BY dev, ph) hm
  ON hm.dev = x.dev AND hm.ph = x.ph""", mat=True)
    _limits(q, LJ, R)
    _tod(q, LJ, R)
    if q.opt["no_daily_pattern"]:
        # note 154m: day (07-19) and night (01-05) levels of the detector and of its phase mates (no mates: the rest
        # of the signal), and how much its hour-of-day profile follows theirs.  No daily pattern = the mates clearly
        # follow the day (>= 100 an hour by day, night <= 30 % of it), the detector counts (>= 1.5 an hour by day; less
        # is a nearly dead zone) and either keeps >= 80 % of its daytime level at night with a profile unlike theirs,
        # or is a low-volume zone (< 30 an hour) keeping >= 30 % (and 2.5 x the mates' share) of it at night; a zone
        # ON >= 25 % of the daytime hours and >= 3 x its night share while it counts LESS by day (< 0.8 x night) is a
        # queue zone: vehicles sit on it -- neither noise nor busy at night (user: 2B345 det 11).  own / ref per hour of
        # day are averages of whole counts (exact); their means over the hours are taken in hour order, so the text's
        # "~26 an hour" never depends on the order DuckDB's threads add them up in (note 154ad)
        q.add("flatm", """
SELECT *, coalesce(occ_d >= 900 AND occ_d >= 3 * occ_n AND own_d < 0.8 * own_n, false) AS qzone FROM (
WITH t AS (SELECT dev, hid, sum(M) AS S FROM mh GROUP BY dev, hid),
pm AS (SELECT m.dev, z.det, m.hid, sum(m.M) AS P FROM mh m JOIN dz zm USING (dev, det)
       JOIN dz z ON z.dev = m.dev AND z.ph = zm.ph AND z.det <> m.det WHERE z.ph IS NOT NULL GROUP BY m.dev, z.det, m.hid),
r AS (SELECT m.dev, m.det, x.hod, avg(m.M) AS own, avg(m.O) AS occ, avg(coalesce(p.P, t.S - m.M)) AS ref
      FROM mh m JOIN hix x USING (dev, hid) JOIN t USING (dev, hid) LEFT JOIN pm p USING (dev, det, hid)
      WHERE x.ch GROUP BY m.dev, m.det, x.hod)
SELECT dev, det, list_avg(list(own ORDER BY hod) FILTER (hod >= 7 AND hod < 19)) AS own_d,
       list_avg(list(own ORDER BY hod) FILTER (hod >= 1 AND hod < 5)) AS own_n,
       list_avg(list(ref ORDER BY hod) FILTER (hod >= 7 AND hod < 19)) AS rest_d,
       list_avg(list(ref ORDER BY hod) FILTER (hod >= 1 AND hod < 5)) AS rest_n,
       avg(occ) FILTER (hod >= 7 AND hod < 19) AS occ_d, avg(occ) FILTER (hod >= 1 AND hod < 5) AS occ_n,
       corr(own, ref) AS prof_corr, count(*) AS n_hod
FROM r GROUP BY dev, det)""", mat=True)
        q.add("flat", """
SELECT * FROM flatm
WHERE n_hod >= 20 AND rest_d >= 100 AND rest_n <= 0.3 * rest_d AND own_d >= 1.5 AND NOT qzone
  AND ((own_n >= 0.8 * own_d AND coalesce(prof_corr, 0) < 0.6)
       OR (own_d < 30 AND own_n >= greatest(0.3, 2.5 * rest_n / rest_d) * own_d AND coalesce(prof_corr, 0) < 0.8))""",
              mat=True)
    # Count zone held ON / extension; the queue excuse is never for a counting zone
    q.add("xx", f"""
SELECT x.*, t.tod_z, t.tod_dph, t.tod_n_night, t.tod_r, t.tod_mate_r, t.tod_d0,
       {"f.det IS NOT NULL AS flat, f.own_d AS flat_d, f.own_n AS flat_n, f.rest_d AS flat_rd, f.rest_n AS flat_rn, "
        "coalesce(fm.qzone, false) AS qzone,"
        if q.opt["no_daily_pattern"] else "false AS flat, NULL::DOUBLE AS flat_d, NULL::DOUBLE AS flat_n, "
        "NULL::DOUBLE AS flat_rd, NULL::DOUBLE AS flat_rn, false AS qzone,"}
       x.fn = 'Count' AND (len(x.alts) = 0 OR list_has_all(['Count'], x.alts)) AND NOT x.c1 AS count_on_ok,
       x.c1 AS c1_pre, CASE WHEN x.c1 THEN 'C1' WHEN x.c2 THEN 'C2' ELSE '' END AS cfg,
       x.fn IN {_in(NO_SHARED)} AND NOT list_has_any(x.alts, ['Presence', 'Other', 'Mid']) AS noq,
       CASE WHEN x.fn = 'Advance' THEN x.q5_all ELSE x.q5_gy END AS q5_use
FROM (SELECT x.*, coalesce(x.fn = 'Count' AND ((x.rep_time_s >= 600 AND x.rep_time_s >= 0.70 * x.occ * x.hours * 3600)
                           OR (x.n_rep >= 20 AND x.n_rep >= 0.05 * x.n_on)), false) AS c1,
             coalesce(x.fn = 'Count' AND x.cmode = 'pulse' AND (x.n_ge60 >= 1 OR x.n_ge5 >= 3), false) AS c2
      FROM xx0 x) x
LEFT JOIN tod t USING (dev, det)
{"LEFT JOIN flat f USING (dev, det) LEFT JOIN flatm fm USING (dev, det)" if q.opt["no_daily_pattern"] else ""}""",
          mat=True)
    # holds with the queue / traffic test (episode_flags)
    q.add("ef", f"""
SELECT e.*, e.q_ok AND e.dur < 3600 AND NOT e.common AS q1,
       e.q_ok AND e.dur >= 3600 AND coalesce(e.cover >= 0.75, false) AND NOT e.lifted0 AND NOT e.common AS q1b,
       e.q_ok AND e.lifted0 AS lifted,
       CASE WHEN e.fn IN {_in(NO_SHARED)} THEN 0 ELSE e.co5 END >= 3 AND NOT e.common AS shared
FROM (SELECT e.*, coalesce(e.q_ok0 AND e.dur >= 3600 AND e.cover >= 0.75 AND e.busy_min_e >= 60 AND e.busy_occ_e < 0.5
                           AND NOT e.ref_held_e, false) AS lifted0, coalesce(e.q_ok0, false) AS q_ok
      FROM (SELECT e.*, x.fn, coalesce(CASE WHEN x.noq THEN 0 ELSE e.n_hpeer_e END, 0) > 0 AND e.phx >= 1.5
                   AND e.corr_h_e >= 0.70 AND e.refx_e >= 0.5 AND coalesce(e.light_e, 1) = 0 AND e.trafx_e >= 0.5 AS q_ok0
            FROM (SELECT e.*, e.cover_e AS cover, CASE WHEN coalesce(e.cover_e, 0) >= 0.75
                         THEN greatest(coalesce(e.phx_h_e, 0), 1.5) ELSE e.phx_h_e END AS phx FROM ectx e) e
            LEFT JOIN xx x USING (dev, det)) e) e""", mat=True)
    q.add("efl", """
SELECT e.*, l.lim_stuck, CASE WHEN e.common THEN 'together' WHEN e.q1 OR e.q1b THEN 'queue'
       WHEN e.shared THEN 'shared' WHEN e.lifted THEN 'not_queue' ELSE 'counted' END AS label,
       NOT e.q1 AND NOT e.q1b AND NOT e.shared AS counted
FROM ef e JOIN limits l USING (dev, det) WHERE e.dur >= l.lim_stuck""", mat=True)
    # stuck on: counted holds over the limit; total time and number against the limits; caps (health_v4._stuck)
    q.add("stk", f"""
SELECT x.dev, x.det, CASE WHEN coalesce(e.n_common, 0) > 0 THEN 1.0
  WHEN coalesce(e.n_ep, 0) > 0 AND e.n_ep = e.n_lift AND sv > {_f(0.35)} THEN 0.35
  WHEN coalesce(e.n_over, 0) > 0 AND coalesce(e.n_ep, 0) = 0 AND coalesce(e.n_sh, 0) = 0 AND e.n_q > 0 THEN 0.0
  WHEN coalesce(e.n_ep, 0) = 0 AND coalesce(e.n_sh, 0) > 0 THEN 0.35 ELSE sv END AS s_stuck_v,
  coalesce(e.n_ep, 0) AS st_n_ep, coalesce(e.tot, 0) AS st_tot_s
FROM (SELECT x.dev, x.det, CASE WHEN x.{"s_stuck_u" if q.opt["no_recovery_cap"] else "s_stuck"} = 0.35
                                    AND coalesce(e.n_ep, 0) <= 1 AND s0 > 0.35 THEN 0.35 ELSE s0 END AS sv
      FROM (SELECT x.*, CASE WHEN st IS NULL THEN coalesce(sn, 0) ELSE greatest(st, coalesce(sn, 0)) END AS s0 FROM (
              SELECT x.dev, x.det, x.s_stuck, {"x.s_stuck_u," if q.opt["no_recovery_cap"] else ""}
                     {_sc('coalesce(e.tot, 0.0)', 'l.lim_stuck', 'l.bad_stuck')} AS st,
                     {_sc('coalesce(e.n_ep, 0)::DOUBLE', 'l.lim_n_ep', 'l.bad_n_ep')} AS sn
              FROM xx x JOIN limits l USING (dev, det) LEFT JOIN (
                SELECT dev, det, count(*) FILTER (counted) AS n_ep, sum(dur) FILTER (counted) AS tot FROM efl
                GROUP BY dev, det) e USING (dev, det)) x) x
      LEFT JOIN (SELECT dev, det, count(*) FILTER (counted) AS n_ep FROM efl GROUP BY dev, det) e USING (dev, det)) x
LEFT JOIN (SELECT dev, det, count(*) AS n_over, count(*) FILTER (counted) AS n_ep, sum(dur) FILTER (counted) AS tot,
                  count(*) FILTER (q1 OR q1b) AS n_q, count(*) FILTER (shared AND NOT q1 AND NOT q1b) AS n_sh,
                  count(*) FILTER (counted AND lifted) AS n_lift, count(*) FILTER (common) AS n_common
           FROM efl GROUP BY dev, det) e USING (dev, det)""", mat=True)
    # pre-pass: phase mates with a finding of their own (not a silence) leave the silence yardstick
    _verdict(q, "pre", "xx", LJ)
    q.add("flag", """
SELECT dev, det, f_stuck OR f_count_on OR f_prof OR f_rapid OR f_volume OR f_spiky AS flagged FROM preV""")
    # goes silent, clean yardstick (h118c_dropout): references by the phase of the scored detectors only
    q.add("dcph", "SELECT d.dev, d.det, x.ph FROM dets d LEFT JOIN xbase x USING (dev, det)", mat=True)
    q.add("dcrf", """
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same, coalesce(f.flagged, false) AS fl
FROM dcph a JOIN dcph b ON a.dev = b.dev AND a.det <> b.det JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det LEFT JOIN flag f ON f.dev = b.dev AND f.det = b.det
WHERE t.i IS NULL""")
    q.add("dccand", """
SELECT x.dev, x.det, coalesce(k.nsame, 0) >= 2 AS isph, coalesce(k.nkeep, 0) >= 2 AS keep2
FROM xx x LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame, count(*) FILTER (same AND NOT fl) AS nkeep
                     FROM dcrf GROUP BY dev, det) k USING (dev, det)
WHERE x.drop_lam >= 30 AND x.n_on > 0""", mat=True)
    # members of S0 (yardstick as in stage 1) and S1 (without the flagged detectors); dropped = flagged ones left out
    q.add("dcm", """
SELECT c.dev, c.det, 0 AS v, f.r FROM dccand c JOIN dcrf f USING (dev, det) WHERE c.isph AND f.same
UNION ALL SELECT c.dev, c.det, 0, d.det FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det WHERE NOT c.isph
UNION ALL SELECT c.dev, c.det, 1, f.r FROM dccand c JOIN dcrf f USING (dev, det) WHERE c.isph AND c.keep2 AND f.same
          AND NOT f.fl
UNION ALL SELECT c.dev, c.det, 1, d.det FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det
          LEFT JOIN flag f ON f.dev = d.dev AND f.det = d.det WHERE NOT (c.isph AND c.keep2)
          AND NOT coalesce(f.flagged, false)""")
    q.add("dcnd", """
SELECT c.dev, c.det, CASE WHEN c.isph AND c.keep2 THEN count(*) FILTER (f.same AND f.fl)
                          ELSE count(*) FILTER (fl2) END AS n_dropped
FROM dccand c JOIN (SELECT a.dev, a.det, a.same, a.fl, coalesce(g.flagged, false) AS fl2 FROM
                      (SELECT d.dev, c.det, d.det AS r, coalesce(f.same, false) AS same, coalesce(f.fl, false) AS fl
                       FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det
                       LEFT JOIN dcrf f ON f.dev = c.dev AND f.det = c.det AND f.r = d.det) a
                    LEFT JOIN flag g ON g.dev = a.dev AND g.det = a.r) f USING (dev, det)
GROUP BY c.dev, c.det, c.isph, c.keep2""")
    q.add("dcrs0", """
SELECT u.dev, u.det, m.v, u.a, u.z, m.r, w.nb5 FROM runs u JOIN dcm m ON m.dev = u.dev AND m.det = u.det
JOIN win w ON w.dev = u.dev""", mat=True)
    _at(q, "dcrs", "dcrs0", "r", ("a", "z", "nb5"), "cn")
    q.add("dcrs", """
SELECT dev, det, v, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM dcrs_3 GROUP BY dev, det, v, a, z""")
    q.add("dc", f"""
SELECT c.dev, c.det, coalesce(r0.lam, 0.0) AS drop_lam_chk, coalesce(r1.lam, 0.0) AS drop_lam_c,
       coalesce(r1.b0, -1) AS drop_b0_c, coalesce(r1.b1, -1) AS drop_b1_c, n.n_dropped
FROM dccand c JOIN dcnd n USING (dev, det)
LEFT JOIN (SELECT dev, det, v, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
           FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM dcrs r JOIN dstat s USING (dev, det)
                 {"JOIN win w ON w.dev = r.dev WHERE r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5" if q.opt["no_recovery_cap"] else ""})
           WHERE lam > 0 GROUP BY dev, det, v) r0 ON r0.dev = c.dev AND r0.det = c.det AND r0.v = 0
LEFT JOIN (SELECT dev, det, v, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
           FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM dcrs r JOIN dstat s USING (dev, det)
                 {"JOIN win w ON w.dev = r.dev WHERE r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5" if q.opt["no_recovery_cap"] else ""})
           WHERE lam > 0 GROUP BY dev, det, v) r1 ON r1.dev = c.dev AND r1.det = c.det AND r1.v = 1""")
    chain = [n for n in ("signal_silent", "phase_drop") if q.opt[n]]
    own = q.opt["own_history"]
    q.add("xfa0" if own else "xfa" if chain else "xf", """
SELECT x.* REPLACE (
    CASE WHEN m THEN x.drop_lam * c.drop_lam_c / c.drop_lam_chk ELSE x.drop_lam END AS drop_lam,
    CASE WHEN m THEN c.drop_b0_c ELSE x.drop_b0 END AS drop_b0,
    CASE WHEN m THEN c.drop_b1_c ELSE x.drop_b1 END AS drop_b1,
    CASE WHEN m THEN CASE WHEN x.drop_in_stuck THEN 0.0 WHEN x.drop_lam * c.drop_lam_c / c.drop_lam_chk < 30 THEN 0.0
                          ELSE 0.35 + 0.65 * least((x.drop_lam * c.drop_lam_c / c.drop_lam_chk - 30) / 70, 1) END
         ELSE x.s_dropout END AS s_dropout)
FROM (SELECT x.*, coalesce(c.n_dropped > 0 AND c.drop_lam_chk > 0, false) AS m FROM xx x LEFT JOIN dc c USING (dev, det)) x
LEFT JOIN dc c USING (dev, det)""", mat=True)
    if own:
        # note 154j: the detector's own count at the same clock times on the other days, outside its silences that
        # rate on their own (>= 30 expected; a quiet night is a reference, a repeating outage is not)
        q.add("ownh", f"""
SELECT * FROM (
WITH r AS (SELECT dev, det, drop_b0, drop_b1 FROM xfa0 WHERE drop_b1 > drop_b0 AND drop_lam > 0),
g AS (SELECT g.dev, g.det, g.b, g.n, g.cov, (w.s_ns // {NS5} + g.b) % 288 AS bod, r.drop_b0, r.drop_b1
      FROM g5 g JOIN r USING (dev, det) JOIN win w USING (dev)),
ref AS (SELECT dev, det, bod, avg(n) AS mn FROM g
        WHERE cov AND NOT (b >= drop_b0 AND b < drop_b1)
          AND NOT EXISTS (SELECT 1 FROM {"silall" if q.opt["all_silences"] else "runs"} u
                          WHERE u.dev = g.dev AND u.det = g.det {"AND u.lam >= 30" if q.opt["all_silences"] else ""}
                          AND g.b >= u.a AND g.b < u.z)
        GROUP BY dev, det, bod)
SELECT g.dev, g.det, sum(f.mn) AS own_typ, count(f.mn) AS nref, count(*) AS nb
FROM g LEFT JOIN ref f USING (dev, det, bod) WHERE g.b >= g.drop_b0 AND g.b < g.drop_b1 GROUP BY g.dev, g.det)""",
              mat=True)
        q.add("xfa" if chain else "xf", """
SELECT x.* REPLACE (CASE WHEN o.nref >= 0.8 * o.nb AND x.drop_lam > 3 * o.own_typ + 10 THEN 3 * o.own_typ + 10
                         ELSE x.drop_lam END AS drop_lam)
FROM xfa0 x LEFT JOIN ownh o USING (dev, det)""", mat=True)
    src = "xfa"
    for i, n in enumerate(chain):
        out = "xf" if i == len(chain) - 1 else "xfs"
        (_signal_silent if n == "signal_silent" else _phase_drop)(q, src, out)
        src = out
    if q.opt["lane_closure"]:
        _closure(q, LJ)
    if q.opt["lane_closure"]:
        _verdict(q, "fin", "xfc", LJ, lc=True)
    else:
        _verdict(q, "fin", "xf", LJ)
    _outputs(q)


def _signal_silent(q: _Q, src: str = "xfa", out: str = "xf"):
    """note 154: every detector of the signal silent (no ON, < 1 s ON) for >= 1 h of daytime (07-19 h) while the signal
    keeps serving greens -- all inputs lost at once (detector rack, cabinet), which phase-mate yardsticks cannot see
    since the mates are silent too.  Each detector's expected count: its daytime ONs per green outside the run x the
    greens inside it; where that beats its own silence, the silence becomes this run -> xf (xf0 + sig_silent)."""
    q.add("sgb", f"""
SELECT b.dev, b.b, b.cov, b.hr, coalesce(n.n, 0) AS n, coalesce(n.occ, 0) AS occ, coalesce(g.k, 0) AS greens
FROM bins5 b LEFT JOIN (SELECT dev, b, sum(n) AS n, sum(occ) AS occ FROM g5 GROUP BY dev, b) n USING (dev, b)
LEFT JOIN (SELECT dev, (dn // {NS5})::BIGINT AS b, count(*) AS k FROM ev WHERE eid = 1 GROUP BY ALL) g USING (dev, b)""",
          mat=True)
    q.add("sgr", """
SELECT dev, a, z, nday FROM (
  SELECT dev, min(b) AS a, max(b) + 1 AS z, count(*) FILTER (hr >= 7 AND hr < 19) AS nday,
         row_number() OVER (PARTITION BY dev ORDER BY count(*) FILTER (hr >= 7 AND hr < 19) DESC, min(b)) AS rk
  FROM (SELECT *, b - row_number() OVER (PARTITION BY dev ORDER BY b) AS grp FROM sgb
        WHERE cov AND n = 0 AND occ < 1 AND greens > 0)
  GROUP BY dev, grp)
WHERE rk = 1 AND nday >= 12""", mat=True)
    q.add("sgl", """
SELECT g.dev, g.det, r.a, r.z,
       sum(g.n) FILTER (NOT (g.b >= r.a AND g.b < r.z)) / greatest(sum(s.greens) FILTER (NOT (g.b >= r.a AND g.b < r.z)), 1)
       * sum(s.greens) FILTER (g.b >= r.a AND g.b < r.z) AS lam_sig
FROM g5 g JOIN sgr r USING (dev) JOIN sgb s ON s.dev = g.dev AND s.b = g.b
WHERE g.cov AND g.hr >= 7 AND g.hr < 19
GROUP BY g.dev, g.det, r.a, r.z
HAVING sum(g.n) FILTER (NOT (g.b >= r.a AND g.b < r.z)) >= 20""", mat=True)
    q.add(out, f"""
SELECT x.* EXCLUDE (msg) REPLACE (CASE WHEN msg THEN l.lam_sig ELSE x.drop_lam END AS drop_lam,
         CASE WHEN msg THEN l.a ELSE x.drop_b0 END AS drop_b0, CASE WHEN msg THEN l.z ELSE x.drop_b1 END AS drop_b1,
         CASE WHEN msg THEN 0.35 + 0.65 * least((l.lam_sig - 30) / 70, 1) ELSE x.s_dropout END AS s_dropout),
       coalesce(msg, false) AS sig_silent
FROM (SELECT x.*, l.lam_sig >= 30 AND l.lam_sig > coalesce(x.drop_lam, 0) AS msg FROM {src} x LEFT JOIN sgl l USING (dev, det)) x
LEFT JOIN sgl l USING (dev, det)""", mat=True)


def _phase_drop(q: _Q, src: str = "xfa", out: str = "xf"):
    """note 154b: a whole phase drops together -- its total count <= 0.2 x its usual share of the rest of the signal (the
    same clock hour on other days of the same type) for >= 2 hours (06-21), >= 200 vehicles expected, while the phase
    keeps getting greens (reference hours: the rest of the signal carried 0.5-2 x what it carries now).  The phase-mate
    yardstick moves with it, so the per-detector checks cannot see it.  Each of its detectors whose own count in the run
    is <= 25 % of its share of the
    rest of the signal gets the run as its finding -> out (src + ph_drop, ph_k, ph_m, ph_obs, ph_exp)."""
    q.add("pdm", """
SELECT m.dev, p.ph, m.hid, x.j, x.hod, x.ch, sum(m.M) FILTER (z.ph = p.ph) AS mp, sum(m.M) FILTER (z.ph <> p.ph) AS rr
FROM mh m JOIN dz z USING (dev, det) JOIN hix x USING (dev, hid)
JOIN (SELECT dev, ph FROM dz WHERE ph IS NOT NULL AND fn IS DISTINCT FROM 'Bike' GROUP BY dev, ph HAVING count(*) >= 2) p
  ON p.dev = m.dev
WHERE z.ph IS NOT NULL AND z.fn IS DISTINCT FROM 'Bike'
GROUP BY m.dev, p.ph, m.hid, x.j, x.hod, x.ch""", mat=True)
    # usual share: the same clock hour on OTHER days of the same type (weekday / weekend) -- a minor phase's share of
    # the signal moves with the hour and the day, so an all-day median turns every morning peak into a "drop"
    q.add("pdr", f"""
SELECT * FROM (
WITH p AS (SELECT *, hid // 24 AS day, isodow(make_timestamp(hid * {NSH} // 1000)) >= 6 AS wk FROM pdm),
h AS (SELECT a.dev, a.ph, a.hid, a.j, a.mp, a.rr, median(b.mp / b.rr) AS sp
      FROM p a JOIN p b ON b.dev = a.dev AND b.ph = a.ph AND b.hod = a.hod AND b.wk = a.wk AND b.day <> a.day
                        AND b.ch AND b.rr >= 30 AND b.rr BETWEEN 0.5 * a.rr AND 2 * a.rr
      WHERE a.ch AND a.hod >= 6 AND a.hod < 21 AND a.rr >= 30 GROUP BY a.dev, a.ph, a.hid, a.j, a.mp, a.rr),
l AS (SELECT *, j - row_number() OVER (PARTITION BY dev, ph ORDER BY j) AS g FROM h
      WHERE sp > 0 AND sp * rr >= 30 AND mp <= 0.2 * sp * rr)
SELECT dev, ph, min(hid) AS ha, max(hid) + 1 AS hz FROM l GROUP BY dev, ph, g
HAVING count(*) >= 2 AND sum(sp * rr) >= 200)""", mat=True)
    q.add("pdg", f"""
SELECT r.dev, r.ph, r.ha, r.hz FROM pdr r JOIN win w USING (dev)
WHERE EXISTS (SELECT 1 FROM ev e WHERE e.dev = r.dev AND e.eid = 1 AND e.par = r.ph
              AND (w.s_ns + e.dn) // {NSH} >= r.ha AND (w.s_ns + e.dn) // {NSH} < r.hz)""", mat=True)
    q.add("pdd", """
SELECT z.dev, z.det, g.ph, g.ha, g.hz,
       sum(m.M) FILTER (m.hid >= g.ha AND m.hid < g.hz) AS obs,
       sum(m.M) FILTER (NOT (m.hid >= g.ha AND m.hid < g.hz) AND x.ch AND x.hod >= 7 AND x.hod < 19 AND p.rr >= 30)
       / nullif(sum(p.rr) FILTER (NOT (m.hid >= g.ha AND m.hid < g.hz) AND x.ch AND x.hod >= 7 AND x.hod < 19
                                 AND p.rr >= 30), 0)
       * sum(p.rr) FILTER (m.hid >= g.ha AND m.hid < g.hz) AS ex
FROM pdg g JOIN dz z ON z.dev = g.dev AND z.ph = g.ph AND z.fn IS DISTINCT FROM 'Bike'
JOIN mh m ON m.dev = z.dev AND m.det = z.det JOIN hix x ON x.dev = m.dev AND x.hid = m.hid
JOIN pdm p ON p.dev = m.dev AND p.ph = g.ph AND p.hid = m.hid
GROUP BY z.dev, z.det, g.ph, g.ha, g.hz""", mat=True)
    q.add("pdk", """
SELECT * FROM (
  SELECT d.*, count(*) FILTER (d.ex >= 30 AND d.obs <= 0.25 * d.ex) OVER (PARTITION BY d.dev, d.ph, d.ha) AS k,
         count(*) OVER (PARTITION BY d.dev, d.ph, d.ha) AS m FROM pdd d)
QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY ex - obs DESC, ph, ha) = 1""", mat=True)
    # lockstep: >= 4 detectors of a phase (mirrored copies counted once) logging the same hourly count (within 1 + 1 %,
    # >= 20 an hour) for >= 6 hours in a row (note 154f: 3 h of low traffic can match by chance) -- traffic never does that, not even three zones of one lane; the inputs log the same pulses (card in fail-safe, a call placed on them).  Needs
    # no history, so it is seen on any day of the failure, not only its first.
    q.add("lkh", """
SELECT m.dev, z.ph, m.hid, x.j, x.ch, m.det, m.M, median(m.M) OVER (PARTITION BY m.dev, z.ph, m.hid) AS md
FROM mh m JOIN dz z USING (dev, det) JOIN hix x USING (dev, hid) JOIN dstat s USING (dev, det)
WHERE z.ph IS NOT NULL AND z.fn IS DISTINCT FROM 'Bike' AND s.live
  AND NOT EXISTS (SELECT 1 FROM twin t WHERE t.dev = m.dev AND t.i = m.det AND t.j < m.det)""", mat=True)
    q.add("lkr", """
SELECT dev, ph, min(hid) AS ha, max(hid) + 1 AS hz, median(md) AS md FROM (
  SELECT *, j - row_number() OVER (PARTITION BY dev, ph ORDER BY j) AS g FROM (
    SELECT dev, ph, hid, j, any_value(md) AS md FROM lkh WHERE ch AND md >= 20
    GROUP BY dev, ph, hid, j HAVING count(*) FILTER (abs(M - md) <= 0.01 * md + 1) >= 4))
GROUP BY dev, ph, g HAVING count(*) >= 6""", mat=True)
    q.add("lkd", """
SELECT * FROM (
  SELECT h.dev, h.det, r.ph, r.ha, r.hz, r.md, count(*) OVER (PARTITION BY r.dev, r.ph, r.ha) AS nk FROM lkr r
  JOIN lkh h ON h.dev = r.dev AND h.ph = r.ph AND h.hid >= r.ha AND h.hid < r.hz
  GROUP BY h.dev, h.det, r.dev, r.ph, r.ha, r.hz, r.md HAVING count(*) FILTER (abs(h.M - h.md) <= 0.01 * h.md + 1) >= 3)
QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY hz - ha DESC, ph, ha) = 1""", mat=True)
    q.add(out, f"""
SELECT x.* EXCLUDE (mpd, mlk) REPLACE (
         CASE WHEN mlk THEN 1e6 WHEN mpd THEN k.ex - k.obs ELSE x.drop_lam END AS drop_lam,
         CASE WHEN mlk THEN (l.ha * {NSH} - w.s_ns) // {NS5} WHEN mpd THEN (k.ha * {NSH} - w.s_ns) // {NS5}
              ELSE x.drop_b0 END AS drop_b0,
         CASE WHEN mlk THEN least((l.hz * {NSH} - w.s_ns) // {NS5}, w.nb5)
              WHEN mpd THEN least((k.hz * {NSH} - w.s_ns) // {NS5}, w.nb5) ELSE x.drop_b1 END AS drop_b1,
         CASE WHEN mlk THEN 1.0 WHEN mpd THEN 0.35 + 0.65 * least((k.ex - k.obs - 30) / 70, 1) ELSE x.s_dropout END
           AS s_dropout),
       mpd AND NOT mlk AS ph_drop, mlk AS ph_lock, k.k AS ph_k, k.m AS ph_m, k.obs AS ph_obs, k.ex AS ph_exp,
       l.nk AS lk_n, l.md AS lk_md
FROM (SELECT x.*, coalesce(k.ex >= 30 AND k.obs <= 0.25 * k.ex AND k.ex - k.obs >= 30
                           AND k.ex - k.obs > coalesce(x.drop_lam, 0), false) AS mpd, l.det IS NOT NULL AS mlk
      FROM {src} x LEFT JOIN pdk k USING (dev, det) LEFT JOIN lkd l USING (dev, det)) x
LEFT JOIN pdk k USING (dev, det) LEFT JOIN lkd l USING (dev, det) JOIN win w USING (dev)""", mat=True)


# lane-closure partner classes: like with like (Other and Bike zones never take part)
LC_CLASS = {"Advance": "up", "Mid": "up", "Count": "bar", "Presence": "bar", "Yellow_Red": "bar"}


def _closure(q: _Q, LJ: dict):
    """note 154b (rules from the user's review 2026-10-22): a lane closure = the lane goes to NEAR ZERO (<= 15 % of its
    expected count, ON < 2 % of the time) and a detector of the same kind on its phase (or the NEMA-paired phase) takes
    the lost vehicles back (>= half on one detector; 0.5-1.5 x in all), judged against the same clock hours on days of
    the same type (weekday / weekend).  Candidates: every near-zero run of >= 2 judged hours, and the silent run;
    option closure_1h (note 154aa): also every near-zero run of >= 3 judged quarter-hours (a closure of an hour or more
    always holds 3 whole quarter-hours).  A silence that starts AND ends with a coordination pattern change (event 131)
    is a timing-plan switch.
    -> xfc (xf + lc_d, lc_l), lcres (one row per closure / switch)."""
    cls = _values("fn, cl", [[_s(k), _s(v)] for k, v in LC_CLASS.items()])
    ex = ("AND NOT x.sig_silent " if q.opt["signal_silent"] else "") +         ("AND NOT x.ph_drop AND NOT x.ph_lock " if q.opt["phase_drop"] else "")
    ex += "AND (x.dev, x.det) NOT IN (SELECT (dev, det) FROM flag WHERE flagged)"
    # near-zero hours: judged (covered, >= 10 expected), <= 15 % of what its share gives -- of its own yardstick before
    # the drop, or of the other phases' traffic (a closure leaves those alone; a same-lane yardstick closes with it);
    # unjudged hours do not break a run, a normal hour does
    q.add("lcz", """
SELECT * FROM (
WITH o AS (SELECT m.dev, m.det, m.hid, m.M, sum(m2.M) AS ro FROM mh m JOIN dz z USING (dev, det)
           JOIN mh m2 ON m2.dev = m.dev AND m2.hid = m.hid JOIN dz z2 ON z2.dev = m2.dev AND z2.det = m2.det
           WHERE z.ph IS NOT NULL AND z2.ph IS NOT NULL AND z2.ph <> z.ph GROUP BY m.dev, m.det, m.hid, m.M),
os AS (SELECT o.*, x.j, x.ch, median(o.M / o.ro) FILTER (x.ch AND o.ro >= 30) OVER (PARTITION BY o.dev, o.det) AS so
       FROM o JOIN hix x USING (dev, hid)),
h AS (SELECT g.dev, g.det, g.j,
             (g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10) OR coalesce(b.ch AND b.so * b.ro >= 10, false)
             AS judged,
             (g.ch AND CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R >= 10
              AND g.M <= 0.15 * CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R)
             OR coalesce(b.ch AND b.so * b.ro >= 10 AND b.M <= 0.15 * b.so * b.ro, false) AS z
      FROM gh2 g JOIN gh4 a USING (dev, det) LEFT JOIN os b ON b.dev = g.dev AND b.det = g.det AND b.j = g.j),
r AS (SELECT *, sum((judged AND NOT z)::INT) OVER (PARTITION BY dev, det ORDER BY j) AS grp FROM h)
SELECT dev, det, min(j) FILTER (z) AS ja, max(j) FILTER (z) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (z) >= 2)""", mat=True)
    q1h = q.opt["closure_1h"]
    if q1h:
        # note 154aa (closure_1h): the same near-zero test on quarter-hours (covered, >= 2.5 expected, <= 15 %): the
        # yardstick's hour spread over its quarters as the other phases' traffic is, or the other phases' traffic of
        # the quarter itself at the detector's hourly share of it; runs of >= 3 near-zero quarters
        q.add("lcq", f"""
SELECT * FROM (
WITH m AS (SELECT g.dev, g.det, (w.s_ns + g.b * {NS5}) // {NSQ} AS qid, sum(g.n) AS M, count(*) FILTER (g.cov) = 3 AS cq
           FROM g5 g JOIN win w USING (dev) JOIN dz z ON z.dev = g.dev AND z.det = g.det
           WHERE w.hours >= {_f(LJ['min_hours_counts'])} AND z.ph IS NOT NULL GROUP BY g.dev, g.det, qid),
o AS (SELECT m.dev, m.det, m.qid, m.qid // 4 AS hid, m.M, m.cq, coalesce(sum(m2.M), 0) AS ro
      FROM m JOIN dz z USING (dev, det) LEFT JOIN (m m2 JOIN dz z2 ON z2.dev = m2.dev AND z2.det = m2.det)
        ON m2.dev = m.dev AND m2.qid = m.qid AND z2.ph IS NOT NULL AND z2.ph <> z.ph
      GROUP BY m.dev, m.det, m.qid, m.M, m.cq),
oh AS (SELECT o.*, sum(o.ro) OVER (PARTITION BY o.dev, o.det, o.hid) AS roh,
              sum(o.M) OVER (PARTITION BY o.dev, o.det, o.hid) AS Mh,
              bool_and(o.cq) OVER (PARTITION BY o.dev, o.det, o.hid) AND count(*) OVER (PARTITION BY o.dev, o.det, o.hid) = 4
              AS chq FROM o),
os AS (SELECT oh.*, median(Mh / roh) FILTER (chq AND roh >= 30) OVER (PARTITION BY dev, det) AS so FROM oh),
e AS (SELECT s.dev, s.det, s.qid, s.M, s.cq,
             CASE WHEN g.ch THEN CASE WHEN a.re THEN a.s2 ELSE a.s END * g.R
                  * CASE WHEN s.roh > 0 THEN s.ro / s.roh ELSE 0.25 END END AS ey,
             s.so * s.ro AS eo
      FROM os s JOIN hix x ON x.dev = s.dev AND x.hid = s.hid
      LEFT JOIN gh2 g ON g.dev = s.dev AND g.det = s.det AND g.j = x.j LEFT JOIN gh4 a ON a.dev = s.dev AND a.det = s.det),
h AS (SELECT dev, det, qid, coalesce(cq AND ey >= 2.5, false) OR coalesce(cq AND eo >= 2.5, false) AS judged,
             coalesce(cq AND ey >= 2.5 AND M <= 0.15 * ey, false) OR coalesce(cq AND eo >= 2.5 AND M <= 0.15 * eo, false) AS z
      FROM e),
r AS (SELECT *, sum((judged AND NOT z)::INT) OVER (PARTITION BY dev, det ORDER BY qid) AS grp FROM h)
SELECT dev, det, min(qid) FILTER (z) AS ja, max(qid) FILTER (z) + 1 AS jz FROM r
GROUP BY dev, det, grp HAVING count(*) FILTER (z) >= 3)""", mat=True)
    lq = (f"\nUNION ALL\nSELECT z.dev, z.det, greatest((z.ja * {NSQ} - w.s_ns) // {NS5}, 0), "
          f"least((z.jz * {NSQ} - w.s_ns) // {NS5}, w.nb5), true FROM lcq z JOIN win w USING (dev)" if q1h else "")
    # closure_1h: quarter runs flagged (qr), so a silence inside one stays a candidate of its own (plan switch test)
    qr, qrf = (", false AS qr", ", false") if q1h else ("", "")
    q.add("lcan", f"""
SELECT z.dev, z.det, greatest(((h.hid0 + z.ja) * {NSH} - w.s_ns) // {NS5}, 0) AS b0,
       least(((h.hid0 + z.jz) * {NSH} - w.s_ns) // {NS5}, w.nb5) AS b1{qr}
FROM lcz z JOIN win w USING (dev) JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) h USING (dev){lq}
UNION ALL
SELECT dev, det, drop_b0, drop_b1{qrf} FROM xf WHERE drop_b1 - drop_b0 >= 3 AND drop_lam >= 30""", mat=True)
    lqc = (f"\nUNION ALL\nSELECT x.dev, x.det, x.ph, x.fn, 'q' || z.ja, greatest((z.ja * {NSQ} - w.s_ns) // {NS5}, 0), "
           f"least((z.jz * {NSQ} - w.s_ns) // {NS5}, w.nb5)\nFROM lcq z JOIN xf x USING (dev, det) JOIN win w USING (dev)\n"
           f"WHERE x.ph IS NOT NULL {ex}" if q1h else "")
    q.add("lcc", f"""
SELECT x.dev, x.det, x.ph, x.fn, 'd' AS kind, x.drop_b0::BIGINT AS b0, x.drop_b1::BIGINT AS b1
FROM xf x JOIN limits l USING (dev, det)
WHERE x.n_on > 0 AND x.ph IS NOT NULL AND x.drop_b1 - x.drop_b0 >= 3 AND x.drop_lam >= l.lim_dropout {ex}
  AND NOT EXISTS (SELECT 1 FROM lcan a JOIN lcz z ON z.dev = a.dev AND z.det = a.det
                  WHERE a.dev = x.dev AND a.det = x.det AND a.b0 <= x.drop_b0 AND a.b1 >= x.drop_b1
                  AND a.b1 - a.b0 > x.drop_b1 - x.drop_b0{" AND NOT a.qr" if q1h else ""})
UNION ALL
SELECT x.dev, x.det, x.ph, x.fn, 'z' || z.ja, greatest(((h.hid0 + z.ja) * {NSH} - w.s_ns) // {NS5}, 0),
       least(((h.hid0 + z.jz) * {NSH} - w.s_ns) // {NS5}, w.nb5)
FROM lcz z JOIN xf x USING (dev, det) JOIN win w USING (dev)
JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) h USING (dev)
WHERE x.ph IS NOT NULL {ex}{lqc}""", mat=True)
    pairs = q.opt["lane_pairs"]
    if pairs == "nema":
        allow = " OR ".join(f"(least(c.ph, p.ph) = {a} AND greatest(c.ph, p.ph) = {b})" for a, b in NEMA_PAIRS)
    elif pairs == "any":
        allow = "true"
    else:
        allow = "false"
    # partner groups: its own phase (gq -1), and its own phase + one allowed other phase gq
    q.add("lcg", f"""
SELECT dev, det, kind, b0, b1, ph, fn, -1.0 AS gq FROM lcc
UNION ALL
SELECT c.dev, c.det, c.kind, c.b0, c.b1, c.ph, c.fn, p.ph FROM lcc c
JOIN (SELECT DISTINCT dev, ph FROM dz WHERE ph IS NOT NULL) p ON p.dev = c.dev AND p.ph <> c.ph AND ({allow})""",
          mat=True)
    # closure_1h: the quarter-hour candidates only add candidates; the references stay those of the hour runs
    nqk = " AND o.kind NOT LIKE 'q%'" if q1h else ""
    # members: the candidate (d), its partners of the same kind (p), the reference (x: every live detector on the
    # other phases without a problem of its own, and not of a phase + kind with a candidate: its partners gain)
    q.add("lcm", f"""
SELECT g.dev, g.det, g.kind, g.gq, g.det AS m, 'd' AS role FROM lcg g JOIN ({cls}) a ON a.fn = g.fn
UNION ALL
SELECT g.dev, g.det, g.kind, g.gq, z.det, 'p'
FROM lcg g JOIN dz z ON z.dev = g.dev AND z.det <> g.det AND (z.ph = g.ph OR z.ph = g.gq)
JOIN ({cls}) a ON a.fn = g.fn JOIN ({cls}) b ON b.fn = z.fn AND b.cl = a.cl
JOIN dstat s ON s.dev = z.dev AND s.det = z.det AND s.live
LEFT JOIN twin t ON t.dev = g.dev AND t.i = g.det AND t.j = z.det
WHERE t.i IS NULL AND NOT EXISTS (SELECT 1 FROM twin u JOIN dz w ON w.dev = u.dev AND w.det = u.j
                                   WHERE u.dev = z.dev AND u.i = z.det AND u.j < z.det AND w.fn = z.fn)
UNION ALL
SELECT g.dev, g.det, g.kind, g.gq, z.det, 'x'
FROM lcg g JOIN dz z ON z.dev = g.dev AND z.ph IS NOT NULL AND z.ph <> g.ph AND z.ph <> g.gq
  AND z.status1 IS DISTINCT FROM 'bad'
JOIN dstat s ON s.dev = z.dev AND s.det = z.det AND s.live
LEFT JOIN ({cls}) k ON k.fn = z.fn
WHERE NOT EXISTS (SELECT 1 FROM lcc o LEFT JOIN ({cls}) ko ON ko.fn = o.fn
                  WHERE o.dev = z.dev AND (o.det = z.det OR (o.ph = z.ph AND ko.cl = k.cl)){nqk})""", mat=True)
    # sums in the period P and, outside every candidate period of the detector and the member's own anomalies, with
    # the other phases' traffic at those same bins: the same clock hours on days of the same type (t), the 2 h either
    # side (ad; for an hour run past the hour next to it, which may hold the start), anywhere (a); the time ON in P
    wk = f"isodow(make_timestamp((w.s_ns + b.b * {NS5}) // 1000)) >= 6"
    # closure_1h: a member's own quarter-hour runs (often quiet night quarters) stay in its reference
    nq = " AND NOT a.qr" if q1h else ""
    q.add("lcrows", f"""
SELECT * FROM (
WITH bw AS (SELECT b.dev, b.b, b.hr, {wk} AS wk FROM bins5 b JOIN win w USING (dev))
SELECT m.dev, m.det, m.kind, m.gq, m.m, m.role, c.b0, c.b1, n.b, n.n, n.occ, n.hr, bw.wk,
       n.b >= c.b0 AND n.b < c.b1 AS inp,
       NOT EXISTS (SELECT 1 FROM lcan a WHERE a.dev = m.dev AND a.det = m.m AND n.b >= a.b0 AND n.b < a.b1{nq})
       AND NOT EXISTS (SELECT 1 FROM lcc o WHERE o.dev = m.dev AND o.det = m.det AND n.b >= o.b0 AND n.b < o.b1{nqk}) AS ref
FROM lcm m JOIN lcc c ON c.dev = m.dev AND c.det = m.det AND c.kind = m.kind
JOIN g5 n ON n.dev = m.dev AND n.det = m.m AND n.cov JOIN bw ON bw.dev = n.dev AND bw.b = n.b)""", mat=True)
    q.add("lcs", """
SELECT * FROM (
WITH hp AS (SELECT DISTINCT dev, det, kind, hr, wk FROM lcrows WHERE role = 'd' AND inp),
xb AS (SELECT dev, det, kind, gq, b, sum(n) AS xn FROM lcrows WHERE role = 'x' AND ref GROUP BY ALL),
r AS (SELECT r.*, h.hr IS NOT NULL AS th, xb.xn,
             CASE WHEN r.kind = 'd' THEN 0 WHEN r.kind LIKE 'q%' THEN 3 ELSE 12 END AS g
      FROM lcrows r LEFT JOIN hp h ON h.dev = r.dev AND h.det = r.det AND h.kind = r.kind AND h.hr = r.hr AND h.wk = r.wk
      LEFT JOIN xb ON xb.dev = r.dev AND xb.det = r.det AND xb.kind = r.kind AND xb.gq = r.gq AND xb.b = r.b)
SELECT dev, det, kind, gq, m, role,
       sum(n) FILTER (inp) AS sp, sum(occ) FILTER (inp) AS op, count(*) FILTER (inp) AS np,
       sum(n) FILTER (ref AND NOT inp) AS sra, sum(xn) FILTER (ref AND NOT inp) AS xra,
       sum(n) FILTER (ref AND NOT inp AND th) AS srt, sum(xn) FILTER (ref AND NOT inp AND th) AS xrt,
       count(*) FILTER (ref AND NOT inp AND th) AS nrt,
       sum(n) FILTER (ref AND (b >= b0 - g - 24 AND b < b0 - g OR b >= b1 + g AND b < b1 + g + 24)) AS sad,
       sum(xn) FILTER (ref AND (b >= b0 - g - 24 AND b < b0 - g OR b >= b1 + g AND b < b1 + g + 24)) AS xad,
       sum(n) FILTER (ref AND b >= b0 - g - 12 AND b < b0 - g) AS pre,
       sum(xn) FILTER (ref AND b >= b0 - g - 12 AND b < b0 - g) AS xpre,
       sum(n) FILTER (ref AND b >= b1 + g AND b < b1 + g + 12) AS post,
       sum(xn) FILTER (ref AND b >= b1 + g AND b < b1 + g + 12) AS xpost
FROM r GROUP BY dev, det, kind, gq, m, role)""")
    q.add("lch", """
SELECT dev, det, kind, gq, m, role, (b - b0) // 12 AS hh, sum(n) AS n FROM lcrows WHERE inp GROUP BY ALL""", mat=True)
    sg = "sqrt(2 * ({a} - {b}) ^ 2 / greatest({a} + {b}, 1e-9))"
    q.add("lce", """
SELECT * FROM (
WITH t AS (SELECT *, bool_and(nrt >= 0.5 * np) OVER (PARTITION BY dev, det, kind, gq) AS tod FROM lcs),
s AS (SELECT dev, det, kind, gq, role, m, sp, op, np, pre, post, xpre, xpost, tod,
             CASE WHEN tod THEN srt WHEN xad >= 100 THEN sad ELSE sra END AS sr,
             CASE WHEN tod THEN xrt WHEN xad >= 100 THEN xad ELSE xra END AS xr FROM t),
x AS (SELECT dev, det, kind, gq, sum(sp) AS xp FROM s WHERE role = 'x' GROUP BY ALL),
e AS (SELECT s.dev, s.det, s.kind, s.gq, s.m, s.role, s.tod, s.op / greatest(s.np * 300.0, 1) AS occ_p, s.sp AS obs,
             s.sr / s.xr * x.xp AS ex, s.sr / s.xr AS rr,
             (s.xpre >= 30 AND s.sp / x.xp >= 1.25 * coalesce(s.pre, 0) / s.xpre)
             OR (s.xpost >= 30 AND s.sp / x.xp >= 1.25 * coalesce(s.post, 0) / s.xpost)
             OR (coalesce(s.xpre, 0) < 30 AND coalesce(s.xpost, 0) < 30) AS step
      FROM s JOIN x USING (dev, det, kind, gq) WHERE s.role <> 'x' AND s.xr >= 100 AND x.xp >= 30),
xh AS (SELECT dev, det, kind, gq, hh, sum(n) AS xn FROM lch WHERE role = 'x' GROUP BY ALL),
fr AS (SELECT p.dev, p.det, p.kind, p.gq, p.m,
              count(*) FILTER (p.n > e.rr * xh.xn AND e.rr * xh.xn >= 5) / nullif(count(*) FILTER (e.rr * xh.xn >= 5), 0)
              AS frac
       FROM lch p JOIN xh USING (dev, det, kind, gq, hh) JOIN e ON e.dev = p.dev AND e.det = p.det AND e.kind = p.kind
            AND e.gq = p.gq AND e.m = p.m
       WHERE p.role = 'p' GROUP BY p.dev, p.det, p.kind, p.gq, p.m)
SELECT d.dev, d.det, d.kind, d.gq, d.obs AS obs_d, d.ex AS exp_d, d.ex - d.obs AS loss, d.tod, d.occ_p AS occ_d,
       max(e.obs - e.ex) AS top_gain, arg_max(f.frac, e.obs - e.ex) AS top_frac, arg_max(e.step, e.obs - e.ex) AS top_step, sum(e.obs) AS obs_g, sum(e.ex) AS exp_g, sum(e.obs - e.ex) AS gain,
       list(e.m ORDER BY e.obs - e.ex DESC, e.m) FILTER (e.obs - e.ex >= 0.15 * (d.ex - d.obs)) AS gainers,
       string_agg('det ' || e.m || CASE WHEN z.ph IS NOT NULL THEN ' (P' || trunc(z.ph)::BIGINT || ')' ELSE '' END, ', '
                  ORDER BY e.obs - e.ex DESC, e.m) FILTER (e.obs - e.ex >= 0.15 * (d.ex - d.obs)) AS glab
FROM e d JOIN e ON e.dev = d.dev AND e.det = d.det AND e.kind = d.kind AND e.gq = d.gq AND e.role = 'p'
LEFT JOIN dz z ON z.dev = e.dev AND z.det = e.m
LEFT JOIN fr f ON f.dev = e.dev AND f.det = e.det AND f.kind = e.kind AND f.gq = e.gq AND f.m = e.m
WHERE d.role = 'd'
GROUP BY d.dev, d.det, d.kind, d.gq, d.obs, d.ex, d.tod, d.occ_p)""")
    # a closure: near zero (<= 15 % of expected, ON < 2 % of the time), a clear loss (>= 100 vehicles, GEH >= 8), one
    # partner of the same kind takes back >= 55 % of it, all of them 0.5-2 x of it, clearly (GEH >= 3), and that
    # partner gains in >= 2/3 of the hours and steps up (>= 1.25 x) against the hour before or after the period (a
    # spike or a normal ramp is not a transfer); its own phase first, else the paired phase taking back the most
    q.add("lcr", f"""
SELECT * FROM (
  SELECT e.*, row_number() OVER (PARTITION BY dev, det, kind ORDER BY gq >= 0, gain / loss DESC, gq) AS rk
  FROM lce e WHERE obs_d <= 0.15 * exp_d AND occ_d < 0.02 AND loss >= 100 AND {sg.format(a='exp_d', b='obs_d')} >= 8
       AND top_gain >= 0.55 * loss AND gain >= 0.5 * loss AND gain <= 2 * loss AND {sg.format(a='obs_g', b='exp_g')} >= 3
       AND coalesce(top_frac, 0) >= 0.67 AND coalesce(top_step, false))
WHERE rk = 1""", mat=True)
    tol = 1  # 5-min bins either side (note 154f: 5 min, user)
    if q.opt["plan_switch"]:
        q.add("lcsw", f"""
SELECT c.dev, c.det, c.kind, first(p.par ORDER BY abs(p.dn // {NS5} - c.b0), p.dn, p.par) AS pat
FROM lcc c JOIN ev p ON p.dev = c.dev AND p.eid = 131 AND abs(p.dn // {NS5} - c.b0) <= {tol}
WHERE c.kind = 'd' AND EXISTS (SELECT 1 FROM ev e WHERE e.dev = c.dev AND e.eid = 131 AND abs(e.dn // {NS5} - c.b1) <= {tol})
GROUP BY c.dev, c.det, c.kind""")
    else:
        q.add("lcsw", "SELECT NULL::INT AS dev, NULL::INT AS det, NULL::VARCHAR AS kind, NULL::INT AS pat WHERE false")
    # one row per closure / switch; a run inside a longer run of the same detector and type is the same event
    a0 = """
  SELECT c.dev, c.det, c.kind, r.gq, r.loss, r.gain, r.gainers, r.glab, c.b0, c.b1, s.pat IS NOT NULL AS switch, s.pat
  FROM lcc c LEFT JOIN lcr r USING (dev, det, kind) LEFT JOIN lcsw s USING (dev, det, kind)
  WHERE r.dev IS NOT NULL OR s.dev IS NOT NULL"""
    if q1h:
        # note 154ac (user, review v13 row 5: 10055 det 9): a closure on a day when the detector has faults of its own
        # (LC_FAULT_DAY) before AND after it on the same calendar day is part of the fault, not a closure.  Its faults:
        # silences over its suspect limit, count drops against its phase mates (>= 2 h, >= the suspect GEH), counted
        # stuck-on holds -- each not overlapping any closure / switch candidate of the detector (those are explained)
        F, M = LC_FAULT_DAY, MATE_HOURS
        sil = (f"SELECT u.dev, u.det, w.s_ns + u.a * {NS5} AS fa, w.s_ns + u.z * {NS5} AS fz FROM silall u "
               "JOIN limits l USING (dev, det) JOIN win w USING (dev) WHERE u.lam >= l.lim_dropout\nUNION ALL\n"
               if q.opt["all_silences"] else "")
        mdr = (f"\nUNION ALL\nSELECT dev, det, a * {NSH}, z * {NSH} FROM mhrun WHERE kind = 'drop' AND hours >= 2 "
               f"AND rgeh >= {_f(M['geh'][0])}" if q.opt["mate_drop"] else "")
        q.add("lcfp", f"""
{sil}SELECT x.dev, x.det, w.s_ns + x.drop_b0 * {NS5}, w.s_ns + x.drop_b1 * {NS5} FROM xf x JOIN limits l USING (dev, det)
JOIN win w USING (dev) WHERE x.drop_b1 > x.drop_b0 AND x.drop_lam >= l.lim_dropout
UNION ALL
SELECT e.dev, e.det, w.s_ns + round(e.s * 1e9)::BIGINT, w.s_ns + round(e.f * 1e9)::BIGINT FROM efl e JOIN win w USING (dev)
WHERE e.counted{mdr}""", mat=True)
        dy = 24 * NSH
        who = "" if F["all"] else "c.kind LIKE 'q%' AND "
        bf = (f"EXISTS (SELECT 1 FROM f WHERE f.dev = c.dev AND f.det = c.det AND f.fz <= c.a_ns "
              f"AND (f.fz - 1) // {dy} = c.a_ns // {dy})")
        af = (f"EXISTS (SELECT 1 FROM f WHERE f.dev = c.dev AND f.det = c.det AND f.fa >= c.z_ns "
              f"AND f.fa // {dy} = (c.z_ns - 1) // {dy})")
        a0 = f"""
WITH c AS (SELECT c.*, w.s_ns + c.b0 * {NS5} AS a_ns, least(w.s_ns + c.b1 * {NS5}, w.e_ns) AS z_ns
           FROM ({a0}) c JOIN win w USING (dev)),
f AS (SELECT * FROM lcfp p WHERE NOT EXISTS (SELECT 1 FROM c WHERE c.dev = p.dev AND c.det = p.det
                                              AND c.a_ns < p.fz AND c.z_ns > p.fa))
SELECT c.* EXCLUDE (a_ns, z_ns) FROM c
WHERE c.switch OR NOT ({who}{bf} {"AND" if F["sides"] == 2 else "OR"} {af})"""
    q.add("lcres", f"""
SELECT * FROM (
WITH a AS ({a0})
SELECT a.*, w.s_ns + a.b0 * {NS5} AS a_ns, least(w.s_ns + a.b1 * {NS5}, w.e_ns) AS z_ns
FROM a JOIN win w USING (dev)
WHERE NOT EXISTS (SELECT 1 FROM a o WHERE o.dev = a.dev AND o.det = a.det AND o.kind <> a.kind AND o.switch = a.switch
                  AND {"o.b0 < a.b1 AND o.b1 > a.b0" if q1h else "o.b0 <= a.b0 AND o.b1 >= a.b1"}
                  AND (o.b1 - o.b0 > a.b1 - a.b0 OR (o.b1 - o.b0 = a.b1 - a.b0 AND o.kind < a.kind))))""", mat=True)
    # the silence / the count drop are explained when closures cover half of it
    mhx = ""
    if q.opt["mate_drop"] or q.opt["count_rise"]:   # note 154n: the runs against the phase mates too
        # note 154o: any overlap with a closure explains a run (the closure is its near-zero core: 03055 det 2), and a
        # detector with a timing-plan switch counts by the plan (03038 det 56): its runs are not faults
        ov = (f"(SELECT coalesce(sum(greatest(least(r.z_ns, x.{{p}}_z * {NSH}) - greatest(r.a_ns, x.{{p}}_a * {NSH}), 0)), 0) "
              f"FROM lcres r WHERE r.dev = x.dev AND {{who}}) > 0 OR EXISTS (SELECT 1 FROM lcres r WHERE r.dev = x.dev "
              f"AND r.det = x.det AND r.switch)")
        mhx = (",\n  coalesce(" + ov.format(p="md", who="r.det = x.det") + ", false) AS lc_md,\n  coalesce("
               + ov.format(p="mr", who="(r.det = x.det OR list_contains(r.gainers, x.det))") + ", false) AS lc_mr")
    q.add("xfc", f"""
SELECT x.* EXCLUDE (l0, l1),
  EXISTS (SELECT 1 FROM lcres r WHERE r.dev = x.dev AND r.det = x.det AND x.drop_b1 > x.drop_b0
          AND least(r.b1, x.drop_b1) - greatest(r.b0, x.drop_b0) >= 0.5 * (x.drop_b1 - x.drop_b0)) AS lc_d,
  coalesce(x.l1 > x.l0 AND (SELECT sum(greatest(least(r.b1, x.l1) - greatest(r.b0, x.l0), 0)) FROM lcres r
                            WHERE r.dev = x.dev AND r.det = x.det AND NOT r.switch) >= 0.5 * (x.l1 - x.l0), false) AS lc_l{mhx}
FROM (SELECT x.*, (epoch_ns(x.drop_t) - w.s_ns) // {NS5} AS l0,
             coalesce((epoch_ns(x.drop_t2) - w.s_ns) // {NS5}, w.nb5) AS l1 FROM xf x JOIN win w USING (dev)) x""", mat=True)


def _dur(x: str) -> str:
    """health_v4._dur: '45 s', '12 min', '2 h 5 min'."""
    m = f"round_even(({x}) / 60.0, 0)::BIGINT"
    return (f"(CASE WHEN ({x}) < 300 THEN format('{{:.0f}} s', ({x})) WHEN {m} < 60 THEN {m} || ' min' "
            f"ELSE ({m} // 60) || ' h' || CASE WHEN {m} % 60 > 0 THEN ' ' || ({m} % 60) || ' min' ELSE '' END END)")


def _hm(ns: str, fmt: str = "%a %H:%M") -> str:
    return f"strftime(make_timestamp(({ns}) // 1000), '{fmt}')"


def _outputs(q: _Q):
    """categories with severity, the plain reason, configuration notes, watch notes, the periods table, signal notes."""
    if q.opt["pulse_mode"]:
        # note 154aa: complete calls (one ON to its OFF, nothing lost or re-logged) of the Presence detectors; note 154ac
        # (user: count, yellow-red and advance zones may pulse, presence may not): a zone configured as Count / Advance
        # (Advance_Presence too) / Yellow_Red, where the detectors table gives the configured function, is never flagged
        P = PULSE
        q.add("pmode", f"""
SELECT * FROM (
  SELECT o.dev, o.det, count(*) AS np, count(*) FILTER (round(o.dur, 3) <= {_f(P['s'])}) / count(*) AS ps, median(o.dur) AS pmed
  FROM ons o JOIN xf x USING (dev, det) JOIN dcl c USING (dev, det)
  WHERE x.fn = 'Presence' AND NOT coalesce(lower(c.cfn) IN ('count', 'yellow_red', 'yellow-red')
                                          OR lower(c.cfn) LIKE 'advance%', false) AND NOT o.multi AND NOT o.open_end AND NOT o.gap GROUP BY o.dev, o.det)
WHERE np >= {P['n']} AND ps >= {_f(P['share'])} AND round(pmed, 3) < {_f(P['s'])}""", mat=True)
    kv = _values("key, ki, cat", [[_s(k), str(i), _s(CATEGORY[k])] for i, k in enumerate(KEYS)])
    fsel = " ".join(f"WHEN '{k}' THEN v.f_{k}" for k in KEYS)
    vsel = " ".join(f"WHEN '{k}' THEN v.v_{k}" for k in KEYS)
    # the reported findings: the kept ones, plus the Count zone held ON of rule C
    q.add("fnd", f"""
SELECT v.dev, v.det, k.key, k.ki, k.cat, coalesce(CASE k.key {vsel} END, 0.0) AS sc,
       CASE WHEN (v.sev_rule = 'C' AND k.key = 'count_on') OR (v.sev_rule = 'F' AND k.key = 'rapid') THEN 'suspect'
            WHEN v.sev_rule = 'P' AND (k.key = 'prof' OR (k.key = 'dropout' AND v.silent_h >= 4.0)) THEN 'bad'
            WHEN coalesce(CASE k.key {vsel} END, 0.0) >= 0.75 THEN 'bad' ELSE 'suspect' END AS sev
FROM finV v, ({kv}) k WHERE CASE k.key {fsel} END OR (k.key = 'count_on' AND v.sev_rule = 'C')""", mat=True)
    # stuck-on holds over the detector's limit, labelled (health_v4._stuck_eps)
    q.add("eps", """
SELECT e.dev, e.det, count(*) FILTER (e.label <> 'queue') AS n_cnt, sum(e.dur) FILTER (e.label <> 'queue') AS tot,
       min(e.s) FILTER (e.label <> 'queue') AS t0, count(*) FILTER (e.label = 'not_queue') AS n_nq,
       min(e.busy_occ_e) FILTER (e.label = 'not_queue') AS nq_occ, count(*) FILTER (e.label = 'together') AS n_tog,
       sum(e.dur) FILTER (e.label = 'together') AS tog_tot, min(e.s) FILTER (e.label = 'together') AS tog_t0,
       arg_min(e.dur, e.s) FILTER (e.label = 'together') AS tog_dur,
       list_sort(list_distinct(flatten(list(e.mates) FILTER (e.label = 'together')))) AS tog_mates
FROM efl e GROUP BY e.dev, e.det""")
    ns = "w.s_ns + round(({t}) * 1e9)::BIGINT"
    why = ("CASE WHEN e.n_nq > 0 AND e.n_nq = e.n_cnt THEN format('; not a queue: at the same traffic it was ON {:.0f} % "
           "of the time', 100 * e.nq_occ) ELSE '' END")
    stuck = f"""CASE WHEN e.n_tog > 0 THEN 'held ON ' || CASE WHEN e.n_tog = 1
        THEN {_dur('e.tog_dur')} || ' from ' || {_hm(ns.format(t='e.tog_t0'))}
        ELSE e.n_tog || ' times from ' || {_hm(ns.format(t='e.tog_t0'))} || ', ' || {_dur('e.tog_tot')} || ' in total' END
        || ' together with det ' || array_to_string(e.tog_mates, ', ') || ' (same start and end: a common failure)'
      WHEN e.n_cnt > 1 THEN 'held ON ' || e.n_cnt || ' times, ' || {_dur('e.tot')} || ' in total' || {why}
      WHEN e.n_cnt = 1 THEN 'held ON ' || {_dur('e.tot')} || ' from ' || {_hm(ns.format(t='e.t0'))} || {why}
      ELSE 'held ON up to ' || {_dur('x.stuck_x')} END"""
    if q.opt["silent_merge"]:   # note 154: a window over a day names the day too
        t0, t1 = (f"CASE WHEN w.hours > 24.5 THEN {_hm(f'w.s_ns + x.drop_b{i} * {NS5}')} "
                  f"ELSE {_hm(f'w.s_ns + x.drop_b{i} * {NS5}', '%H:%M')} END" for i in (0, 1))
        silent = f"'no actuations ' || {t0} || '-' || {t1} || format('; ~{{:,.0f}} expected', x.drop_lam)"
    else:
        silent = (f"'no actuations ' || {_hm('w.s_ns + x.drop_b0 * ' + str(NS5), '%H:%M')} || '-' || "
                  f"{_hm('w.s_ns + x.drop_b1 * ' + str(NS5), '%H:%M')} || format('; ~{{:,.0f}} expected', x.drop_lam)")
    if q.opt["signal_silent"]:
        silent += (" || CASE WHEN x.sig_silent THEN '; every detector at the signal was silent while it kept serving "
                   "greens (all inputs lost: detector rack / cabinet)' ELSE '' END")
    if q.opt["phase_drop"]:   # note 154b: the whole phase fell together
        silent = (f"CASE WHEN x.ph_lock THEN 'logs the same count as ' || (x.lk_n - 1) || ' other detectors on P' "
                  f"|| trunc(x.ph)::BIGINT || ' every hour ' || {t0} || '-' || {t1} || format(' (~{{:,.0f}} an hour each): "
                  f"the inputs no longer see vehicles and log the same pulses (detector card in fail-safe, or a call placed "
                  f"on them)', x.lk_md) "
                  f"WHEN x.ph_drop THEN 'whole phase dropped ' || {t0} || '-' || {t1} "
                  f"|| format(': {{:,.0f}} actuations where ~{{:,.0f}} expected; ', x.ph_obs, x.ph_exp) || x.ph_k || ' of ' "
                  f"|| x.ph_m || ' detectors on P' || trunc(x.ph)::BIGINT || ' fell together while the rest of the signal "
                  f"counted normally (approach closed, or a shared card / cable)' ELSE {silent} END")
    if q.opt["drop_end"]:   # note 154b: plain words
        level = """CASE WHEN x.own_after < 0.1 * x.exp_after THEN 'nearly stopped counting'
       WHEN x.occ_own >= 3.0 AND x.occ_own >= 3.0 * coalesce(x.occ_mates, 1.0) THEN format('counts fell while its time ON rose {:.0f} x (holds ON longer: stuck, or a changed '
                                         || 'setting)', x.occ_own)
       ELSE 'counts and occupancy both fell' END || CASE WHEN x.drop_t IS NOT NULL THEN ' from ' || strftime(x.drop_t, '%a %H:%M') ELSE '' END
       || CASE WHEN x.drop_t2 IS NOT NULL THEN ' to ' || strftime(x.drop_t2, '%a %H:%M') ELSE '' END
       || format(': {:.0f} per hour where its phase mates'' traffic gives ~{:.0f} (GEH {:.1f})', x.own_after,
                 x.exp_after, x.geh_drop)"""
    else:
        level = """'counts dropped' || CASE WHEN x.drop_t IS NOT NULL THEN ' from ' || strftime(x.drop_t, '%a %H:%M')
       ELSE '' END || format(': {:.0f} per hour where its phase mates'' traffic gives ~{:.0f} (GEH {:.1f}); '
                             || 'its occupancy fell too', x.own_after, x.exp_after, x.geh_drop)"""
    if q.opt["all_silences"]:
        lcx_ = (f"AND NOT EXISTS (SELECT 1 FROM lcres c WHERE c.dev = u.dev AND c.det = u.det "
                f"AND least(c.z_ns, w.s_ns + u.z * {NS5}) - greatest(c.a_ns, w.s_ns + u.a * {NS5}) >= "
                f"0.5 * (u.z - u.a) * {NS5})" if q.opt["lane_closure"] else "")
        q.add("silmore", f"""
SELECT u.dev, u.det, u.a, u.z, u.lam FROM silall u JOIN xf x USING (dev, det) JOIN limits l USING (dev, det)
JOIN win w USING (dev)
WHERE u.lam >= l.lim_dropout AND NOT (u.a < x.drop_b1 AND u.z > x.drop_b0) {lcx_}
  AND (u.lam >= 50 OR (SELECT count(*) FILTER (b.hr >= 21 OR b.hr < 5) FROM bins5 b
                       WHERE b.dev = u.dev AND b.b >= u.a AND b.b < u.z) < 0.8 * (u.z - u.a))""", mat=True)
        silent += (" || coalesce((SELECT '; also ' || string_agg(" + _hm(f"w.s_ns + u.a * {NS5}") + " || '-' || "
                   + _hm(f"w.s_ns + u.z * {NS5}") + " || format(' (~{:,.0f})', u.lam), ', ' ORDER BY u.a) "
                   "FILTER (u.rk <= 3) || CASE WHEN count(*) > 3 THEN format(' and {} more', count(*) - 3) ELSE '' END "
                   "FROM (SELECT *, row_number() OVER (ORDER BY round(lam, 6) DESC, a) AS rk FROM silmore "
                   "WHERE dev = x.dev AND det = x.det) u), '')")
    if q.opt["mate_drop"]:   # note 154n: the run against the phase mates, when it rates the drop
        hh = "make_timestamp(x.md_{} * 3600000000)"
        level = (f"CASE WHEN coalesce(v.v_lvl_md, 0) > coalesce(v.v_lvl_old, 0) THEN format('counts fell to {{:.0f}} % of "
                 f"what its phase mates gave at that time of day on the other days: ', 100 * x.md_r) || "
                 f"strftime({hh.format('a')}, '%a %H:%M') || '-' || strftime({hh.format('z')}, '%H:%M') || "
                 f"format(' ({{}} h, ~{{:.0f}} an hour where ~{{:.0f}} expected; GEH {{:.0f}})', x.md_h, x.md_own, "
                 f"x.md_exp, x.md_geh) ELSE {level} END")
    rise = ("format('counts rose to {:.1f} x what its phase mates gave at that time of day on the other days: ', x.mr_r) "
            "|| strftime(make_timestamp(x.mr_a * 3600000000), '%a %H:%M') || '-' || "
            "strftime(make_timestamp(x.mr_z * 3600000000), '%H:%M') || format(' ({} h, GEH {:.0f}): extra calls', "
            "x.mr_h, x.mr_geh)" if q.opt["count_rise"] else "''")
    q.add("evid", f"""
SELECT f.dev, f.det, f.key, f.ki,
       CASE WHEN f.key = 'prof' AND x.flat THEN 'No daily pattern' ELSE f.cat END AS cat, f.sc, f.sev, CASE f.key
  WHEN 'stuck' THEN {stuck}
  WHEN 'count_on' THEN format('count zone ON up to {{:.0f}} % in light traffic for {{:.0f}} min', 100 * x.hi_occ_max,
                              x.hi_min_c)
  WHEN 'dropout' THEN {silent}
  WHEN 'night_drop' THEN CASE WHEN v.silent_night THEN {silent} WHEN x.geh_night_exp IS NOT NULL
       THEN format('night {{:,.0f}} actuations vs ~{{:,.0f}} expected (GEH {{:.1f}})', x.geh_night_n, x.geh_night_exp,
                   x.geh_night) ELSE 'fewer actuations at night' END
  WHEN 'level' THEN {level}
  WHEN 'rapid' THEN format('{{:,}} of {{:,}} ONs follow the previous ON within {{:.2g}} s ({{:.0f}} per 1,000; '
                           || 'limit {{:.0f}})', x.n_fast, x.N, x.gap_lim, x.fast_rate, l.lim_rapid)
       || CASE WHEN x.pair_share >= 0.9 THEN '; counts each vehicle twice' ELSE '' END
       || CASE WHEN v.sev_rule = 'F' THEN ', spread evenly with traffic - flickers as vehicles pass, check zone setup'
               ELSE '' END
  WHEN 'spiky' THEN format('{{:.0f}} bursts{{}} in 10-min counts at 3 x its usual level or more (largest GEH {{:.0f}}); '
                           || '~{{:,.0f}} vehicles a day', x.spk_bursts, CASE WHEN x.hours > 24.5 THEN ' a day' ELSE '' END,
                           x.spk_gmax, x.spk_daily)
  WHEN 'volume' THEN format('busiest 5 min {{:,.0f}} veh/h/lane vs limit {{:,.0f}}', x.q5_use, l.lim_volume)
  WHEN 'rise' THEN {rise}
  WHEN 'prof' THEN CASE WHEN x.flat THEN format('counts do not follow the day: ~{{:.0f}} an hour by day, ~{{:.0f}} at night, while '
       || 'the rest of the signal drops from ~{{:,.0f}} to ~{{:,.0f}} (noise, not traffic)',
       x.flat_d, x.flat_n, x.flat_rd, x.flat_rn) ELSE
       format('night level {{:.0f}} % of its busiest hours; phase mates ', 100 * x.tod_r ^ 2)
       || CASE WHEN x.tod_mate_r IS NOT NULL THEN format('{{:.0f}} %', 100 * x.tod_mate_r ^ 2) ELSE 'none' END END
  END AS txt
FROM fnd f JOIN xf x USING (dev, det) JOIN finV v USING (dev, det) JOIN limits l USING (dev, det) JOIN win w USING (dev)
LEFT JOIN eps e USING (dev, det)""")
    lc = q.opt["lane_closure"]
    if lc:
        # note 154: lane closures and timing-plan switches, reported (severity 'info') next to the faults
        rng = f"{_hm('r.a_ns')} || '-' || {_hm('r.z_ns')}"
        q.add("lctx", f"""
SELECT r.dev, r.det, 'closure' AS key, 99 AS ki, CASE WHEN r.switch THEN 'Timing plan' ELSE 'Lane closure' END
       AS cat, -1.0 AS sc, 'info' AS sev,
       CASE WHEN r.switch THEN {rng} || CASE WHEN r.kind = 'd' THEN ': silent' ELSE ': counts low' END
                               || ' while coordination pattern ' || r.pat || ' is in effect'
                               || coalesce('; its traffic moves to ' || r.glab, '')
            ELSE {rng} || format(': ~{{:,.0f}} of its vehicles moved to ', r.loss) || r.glab END AS txt, r.a_ns AS o
FROM lcres r""")
        q.add("lcx", "SELECT dev, det, bool_and(switch) AS all_switch FROM lcres GROUP BY dev, det", mat=True)
    ev_src = ("(SELECT dev, det, key, ki, cat, sc, sev, txt, 0::BIGINT AS o FROM evid UNION ALL SELECT * FROM lctx)" if lc
              else "(SELECT *, 0::BIGINT AS o FROM evid)")
    q.add("cats", f"""
SELECT dev, det, string_agg(cat || ' (' || sev || ')', '; ' ORDER BY s DESC, ki0, cat) AS health_categories,
       string_agg(cat || ': ' || txt, '; ' ORDER BY s DESC, ki0, cat) AS parts, bool_or(lane) AS lane_cat
FROM (SELECT dev, det, cat, max(sc) AS s, min(ki) AS ki0,
             CASE WHEN bool_or(sev = 'bad') THEN 'bad' WHEN bool_or(sev = 'info') THEN 'info' ELSE 'suspect' END AS sev,
             string_agg(txt, '; ' ORDER BY ki, o, txt) AS txt, bool_or(key IN ('stuck', 'rapid', 'volume', 'prof', 'spiky')) AS lane
      FROM {ev_src} GROUP BY dev, det, cat)
GROUP BY dev, det""")
    if q.periods:
        _periods(q, lc)
    _outputs_rest(q, lc)


WHOLE = ("busy at night", "no daily pattern", "misses vehicles at night")   # findings with no real time of their own


def _conf(x: str, lo: str, hi: str) -> str:
    """note 154o: a period's severity on its check's own scale: 0.5 at the suspect limit lo, 0.8 at the bad limit hi,
    on log strength; 0 when the check has a limit but no strength."""
    return (f"(CASE WHEN ({x}) > 0 AND ({lo}) > 0 THEN 1.0 / (1.0 + exp(-ln(4.0) * ln(({x}) / ({lo})) "
            f"/ ln(greatest(({hi}) / ({lo}), 1.05)))) WHEN ({lo}) IS NOT NULL THEN 0.0 END)")


def _periods(q: _Q, lc: bool):
    """note 154aa: the periods table (health(..., periods=True)) -> pall (dev, det, o, a, z ns, what, sev, lim, lc, ww):
    where each finding the rules kept saw the problem (suspect / bad detectors only), with its severity on the check's
    own suspect (lo) -> bad (hi) scale (x = its strength) and lim = the type's bad limit of an event check (s), plus
    every lane closure / timing-plan switch; a finding without a period of its own spans the window (ww)."""
    M, P = MATE_HOURS, _json("health_limits.json")["spiky"]
    g0, g1 = _f(M["geh"][0]), _f(M["geh"][1])
    sc = ", r.v_{}, 0.35, 0.75, NULL"
    if q.opt["drop_end"]:   # note 154b: every run of low hours, not only the first drop
        lvp = (f"UNION ALL SELECT r.dev, r.det, 4, (h.hid0 + u.ja) * {NSH}, least((h.hid0 + u.jz) * {NSH}, w.e_ns), "
               f"'counts dropped'{sc.format('level')} FROM r JOIN lvr u USING (dev, det) JOIN win w USING (dev) "
               "JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) h USING (dev) WHERE r.f_level"
               + (" AND NOT coalesce(r.v_lvl_md > r.v_lvl_old, false)" if q.opt["mate_drop"] else ""))
    else:
        lvp = ("UNION ALL SELECT r.dev, r.det, 4, epoch_ns(x.drop_t), coalesce(epoch_ns(x.drop_t2), w.e_ns), "
               f"'counts dropped'{sc.format('level')} FROM r JOIN xf x USING (dev, det) JOIN win w USING (dev) "
               "WHERE r.f_level AND x.drop_t IS NOT NULL")
    # note 154n: the runs of low / high hours against the phase mates (>= the suspect GEH, >= 2 h)
    mhp = ""
    if q.opt["mate_drop"]:
        mhp += (f"UNION ALL SELECT r.dev, r.det, 4, u.a * {NSH}, least(u.z * {NSH}, w.e_ns), 'counts dropped', u.rgeh, "
                f"{g0}, {g1}, NULL FROM r JOIN mhrun u USING (dev, det) JOIN win w USING (dev) WHERE r.f_level "
                f"AND coalesce(r.v_lvl_md > r.v_lvl_old, false) AND u.kind = 'drop' AND u.hours >= 2 AND u.rgeh >= {g0}\n  ")
    if q.opt["count_rise"]:
        mhp += (f"UNION ALL SELECT r.dev, r.det, 10, u.a * {NSH}, least(u.z * {NSH}, w.e_ns), 'counts too high', u.rgeh, "
                f"{g0}, {g1}, NULL FROM r JOIN mhrun u USING (dev, det) JOIN win w USING (dev) WHERE r.f_rise "
                f"AND u.kind = 'rise' AND u.hours >= 2 AND u.rgeh >= {g0}")
    sil = ("x.drop_lam, l.lim_dropout, l.bad_dropout, NULL FROM r JOIN xf x USING (dev, det) "
           "JOIN limits l USING (dev, det) JOIN win w USING (dev)")
    spl = _f(P["bad"] * P["bin_min"] * 60)          # spiky: its bad number of 10-min bursts, in seconds
    sm = ("UNION ALL SELECT r.dev, r.det, 2, w.s_ns + u.a * " + str(NS5) + ", w.s_ns + u.z * " + str(NS5) + ", 'silent', "
          "u.lam, l.lim_dropout, l.bad_dropout, NULL FROM r JOIN silmore u USING (dev, det) JOIN win w USING (dev) "
          "JOIN limits l USING (dev, det) WHERE r.f_dropout OR r.f_night_drop" if q.opt["all_silences"] else "")
    lcp = ("UNION ALL SELECT dev, det, 9, a_ns, z_ns, CASE WHEN switch THEN 'timing plan' ELSE 'lane closure' END, "
           "NULL, NULL, NULL, NULL FROM lcres" if lc else "")
    q.add("pkept", f"""
WITH r AS (SELECT * FROM finV WHERE health_st IN ('suspect', 'bad')),
runs_hi AS (SELECT dev, det, min(k) AS a, max(k) + 1 AS z FROM (SELECT dev, det, k, k - row_number() OVER
            (PARTITION BY dev, det ORDER BY k) AS g FROM hib) GROUP BY dev, det, g),
fh AS (SELECT dev, det, dn // {NSH} AS h, count(*) FILTER (f) AS cf, count(*) AS ca FROM tfo GROUP BY dev, det, h),
runs_fast AS (SELECT dev, det, min(h) AS a, max(h) + 1 AS z FROM (
              SELECT f.dev, f.det, f.h, f.h - row_number() OVER (PARTITION BY f.dev, f.det ORDER BY f.h) AS g
              FROM fh f JOIN limits l USING (dev, det)
              WHERE f.cf >= 3 AND 1000.0 * f.cf / greatest(f.ca, 1) >= l.lim_rapid) GROUP BY dev, det, g),
qv AS (SELECT q.dev, q.det, q.b FROM q5b q JOIN xf x USING (dev, det) JOIN limits l USING (dev, det)
       WHERE x.fn = 'Advance' AND ((q.n::FLOAT / 300::FLOAT) * 3600::FLOAT)::DOUBLE / q.ln1 > l.lim_volume
       UNION ALL SELECT q.dev, q.det, q.b FROM q5gyb q JOIN xf x USING (dev, det) JOIN limits l USING (dev, det)
       WHERE x.fn <> 'Advance' AND q.q5_gy > l.lim_volume),
runs_vol AS (SELECT dev, det, min(b) AS a, max(b) + 1 AS z FROM (SELECT dev, det, b, b - row_number() OVER
             (PARTITION BY dev, det ORDER BY b) AS g FROM qv) GROUP BY dev, det, g)
  SELECT r.dev, r.det, 0 AS o, w.s_ns + round(e.s * 1e9)::BIGINT AS a, w.s_ns + round(e.f * 1e9)::BIGINT AS z,
         'stuck on' AS what, e.dur AS x, l.lim_stuck AS lo, l.bad_stuck AS hi, l.bad_stuck AS lim
         FROM r JOIN efl e USING (dev, det) JOIN win w USING (dev) JOIN limits l USING (dev, det)
         WHERE r.f_stuck AND e.label <> 'queue'
  UNION ALL SELECT r.dev, r.det, 1, w.s_ns + u.a * w.bs * 1000000000, least(w.s_ns + u.z * w.bs * 1000000000, w.e_ns),
         'count zone held ON', (u.z - u.a) * w.bs / 60.0, l.lim_count_on, l.bad_count_on, 60 * l.bad_count_on
         FROM r JOIN runs_hi u USING (dev, det) JOIN win w USING (dev) JOIN limits l USING (dev, det)
         WHERE r.f_count_on OR r.sev_rule = 'C'
  UNION ALL SELECT r.dev, r.det, 2, w.s_ns + r.drop_b0 * {NS5}, w.s_ns + r.drop_b1 * {NS5}, 'silent', {sil}
         WHERE r.f_dropout
  UNION ALL SELECT r.dev, r.det, 3, w.s_ns + r.drop_b0 * {NS5}, w.s_ns + r.drop_b1 * {NS5}, 'silent at night', {sil}
         WHERE r.f_night_drop AND r.silent_night
  {sm}
  {lvp}
  UNION ALL SELECT r.dev, r.det, 5, w.s_ns + u.a * {NSH}, least(w.s_ns + u.z * {NSH}, w.e_ns), 'too-fast actuations'
         {sc.format('rapid')} FROM r JOIN runs_fast u USING (dev, det) JOIN win w USING (dev) WHERE r.f_rapid
  UNION ALL SELECT r.dev, r.det, 6, w.s_ns + u.a * {NS5}, least(w.s_ns + u.z * {NS5}, w.e_ns), 'too many for the traffic'
         {sc.format('volume')} FROM r JOIN runs_vol u USING (dev, det) JOIN win w USING (dev) WHERE r.f_volume
  UNION ALL SELECT r.dev, r.det, 7, x.tod_d0 + {NSH}, x.tod_d0 + 5 * {NSH}, 'busy at night'{sc.format('prof')}
         FROM r JOIN xf x USING (dev, det) WHERE r.f_prof AND x.tod_d0 IS NOT NULL AND NOT coalesce(x.flat, false)
  UNION ALL SELECT r.dev, r.det, 7, w.s_ns, w.e_ns, 'no daily pattern'{sc.format('prof')}
         FROM r JOIN xf x USING (dev, det) JOIN win w USING (dev) WHERE r.f_prof AND coalesce(x.flat, false)
  UNION ALL SELECT r.dev, r.det, 8, w.s_ns + u.a * 600000000000, least(w.s_ns + u.z * 600000000000, w.e_ns),
         'spiky counts',
         (SELECT max(o.g) FROM o10s o WHERE o.dev = u.dev AND o.det = u.det AND o.t >= u.a AND o.t < u.z AND o.sp),
         {_f(P['geh'])}, {_f(P['geh_big'])}, {spl}
         FROM r JOIN spk_runs u USING (dev, det) JOIN win w USING (dev) WHERE r.f_spiky
  {mhp}
  {lcp}""", mat=True)
    # a finding without a period of its own: one row over the window, at the finding's own score (0.35 suspect ->
    # 0.5, 0.75 bad -> 0.8; >= 0.8 when the rules rate the finding bad)
    v = "CASE f.key " + " ".join(f"WHEN '{k}' THEN v.v_{k}" for k in KEYS) + " END"
    what = """CASE f.key WHEN 'stuck' THEN 'stuck on' WHEN 'count_on' THEN 'count zone held ON' WHEN 'dropout' THEN 'silent'
         WHEN 'night_drop' THEN CASE WHEN v.silent_night THEN 'silent at night' ELSE 'misses vehicles at night' END
         WHEN 'level' THEN 'counts dropped' WHEN 'rapid' THEN 'too-fast actuations'
         WHEN 'volume' THEN 'too many for the traffic' WHEN 'spiky' THEN 'spiky counts' WHEN 'rise' THEN 'counts too high'
         WHEN 'prof' THEN CASE WHEN coalesce(x.flat, false) THEN 'no daily pattern' ELSE 'busy at night' END END"""
    q.add("pwhole", f"""
SELECT f.dev, f.det, 20 + f.ki AS o, w.s_ns AS a, w.e_ns AS z, {what} AS what,
       CASE WHEN f.sev = 'bad' THEN greatest({_conf(v, '0.35', '0.75')}, 0.8) ELSE {_conf(v, '0.35', '0.75')} END AS conf,
       CASE f.key WHEN 'stuck' THEN l.bad_stuck WHEN 'count_on' THEN 60 * l.bad_count_on WHEN 'spiky' THEN {spl} END AS lim
FROM fnd f JOIN finV v USING (dev, det) JOIN xf x USING (dev, det) JOIN win w USING (dev) JOIN limits l USING (dev, det)
WHERE v.health_st IN ('suspect', 'bad')""")
    # in a lane closure / plan switch of its own, or within 30 min of one (the lane reopening)
    pm = ("\nUNION ALL\nSELECT p.dev, p.det, 30, w.s_ns, w.e_ns, 'pulse mode', 0.5, NULL, false, true FROM pmode p "
          "JOIN win w USING (dev)" if q.opt["pulse_mode"] else "")
    lcj = ("(SELECT count(*) > 0 FROM lcres c WHERE c.dev = p.dev AND c.det = p.det AND c.a_ns < p.z + 1800000000000 "
           "AND c.z_ns > p.a - 1800000000000)" if lc else "false")
    q.add("pall", f"""
SELECT p.dev, p.det, p.o, p.a, p.z, p.what,
       CASE WHEN p.o IN (2, 3) AND p.z - p.a >= 4 * {NSH} AND p.x >= p.lo THEN greatest(p.conf, 0.8) ELSE p.conf END AS sev,
       p.lim, {lcj} AS lc, p.what IN {_in(WHOLE)} AS ww
FROM (SELECT p.*, {_conf('p.x', 'p.lo', 'p.hi')} AS conf FROM pkept p) p
UNION ALL
SELECT w.dev, w.det, w.o, w.a, w.z, w.what, w.conf, w.lim, false, true FROM pwhole w
WHERE NOT EXISTS (SELECT 1 FROM pkept p WHERE p.dev = w.dev AND p.det = w.det AND p.what = w.what){pm}""")


def _outputs_rest(q: _Q, lc: bool):
    lab = ("'det ' || x.det || ': ' || CASE WHEN x.ph IS NOT NULL THEN 'P' || trunc(x.ph)::BIGINT || ' ' ELSE '' END "
           "|| replace(x.fn, 'Yellow_Red', 'Yellow-red')")
    nl = "greatest(round_even(x.lanes, 0)::BIGINT, 1)"
    border = " ".join(f"WHEN v.b_{k} THEN {_s('borderline ' + CATEGORY[k].lower())}" for k in KEYS)
    st = ("CASE WHEN k.dev IS NOT NULL AND v.health_st IN ('ok', 'watch', 'not_enough_data') THEN CASE WHEN k.all_switch "
          "THEN 'timing_plan' ELSE 'lane_closure' END ELSE v.health_st END" if lc else "v.health_st")
    pmo = q.opt["pulse_mode"]
    q.add("outx0" if pmo else "outx", f"""
SELECT v.dev, v.det, {st} AS health_status,
       CASE WHEN v.health_st <> 'not_enough_data' AND NOT isnan(v.score) THEN v.score END AS health_score,
       CASE WHEN v.health_st IN ('suspect', 'bad') {"OR k.dev IS NOT NULL" if lc else ""}
            THEN b.lb || ' - ' || coalesce(c.parts, v.health_st)
            WHEN v.health_st = 'watch' THEN b.lb || ' - watch' || CASE WHEN v.wt <> '' THEN ': ' || v.wt ELSE '' END
            WHEN v.health_st = 'not_enough_data' THEN b.lb || ' - too few actuations to judge'
            ELSE b.lb || ' - nothing unusual' END AS health_reason,
       coalesce(c.health_categories, '') AS health_categories,
       CASE x.cfg WHEN 'C1' THEN format('Extension time on a count zone: {{:,}} of {{:,}} ONs logged again without an OFF',
                                        x.n_rep, trunc(x.n_on)::BIGINT)
                  WHEN 'C2' THEN 'Set to pulse but holds ON: ' || CASE WHEN x.long_max_s IS NOT NULL
                       THEN x.n_ge5 || ' ONs over 5 s, longest ' || {_dur('x.long_max_s')}
                       ELSE 'holds ON although set to pulse' END ELSE '' END AS health_config,
       v.wt AS health_watch
FROM (SELECT v.*, array_to_string(list_filter([
        CASE WHEN v.w_no_yardstick THEN 'no healthy phase mate to compare with' END,
        CASE {border} END,
        CASE WHEN v.w_dropout_short AND NOT (v.b_dropout OR v.b_level) THEN 'borderline counts fall' END],
        lambda x: x IS NOT NULL), '; ') AS wt FROM finV v) v
JOIN xf x USING (dev, det) LEFT JOIN cats c USING (dev, det)
{"LEFT JOIN lcx k USING (dev, det)" if lc else ""}
JOIN (SELECT x.dev, x.det, {lab} || CASE WHEN x.lanes IS NOT NULL AND coalesce(c.lane_cat, false) THEN ' (covers '
             || {nl} || ' lane' || CASE WHEN {nl} > 1 THEN 's' ELSE '' END || ')' ELSE '' END AS lb
      FROM xf x LEFT JOIN cats c USING (dev, det)) b USING (dev, det)""")
    if pmo:
        # note 154aa: looks set to pulse mode -- suspect at least (a worse rating stays), the note added
        txt = ("format('Configuration: looks set to pulse mode: {:.0f} % of its {:,} calls last 0.2 s or less; a presence "
               "detector should keep its call ON while a vehicle is on it', 100 * p.ps, p.np)")
        up = "p.dev IS NOT NULL AND o.health_status IN ('ok', 'watch', 'lane_closure', 'timing_plan')"
        q.add("outx", f"""
SELECT o.dev, o.det, CASE WHEN {up} THEN 'suspect' ELSE o.health_status END AS health_status,
       CASE WHEN {up} THEN least(coalesce(o.health_score, 1.0), 0.69) ELSE o.health_score END AS health_score,
       CASE WHEN p.dev IS NULL THEN o.health_reason
            WHEN o.health_status IN ('ok', 'watch') THEN split_part(o.health_reason, ' - ', 1) || ' - ' || {txt}
            ELSE o.health_reason || '; ' || {txt} END AS health_reason,
       CASE WHEN p.dev IS NULL THEN o.health_categories WHEN o.health_categories = '' THEN 'Configuration (suspect)'
            ELSE o.health_categories || '; Configuration (suspect)' END AS health_categories,
       o.health_config, o.health_watch
FROM outx0 o LEFT JOIN pmode p USING (dev, det)""")
    # detectors without a known predicted function: the stage-1 checks as they are (pass-2 status)
    s1 = ("(SELECT * REPLACE (status_u AS status, hs_u AS hs, s_dropout_u AS s_dropout, s_stuck_u AS s_stuck) FROM stage1)"
          if q.opt["no_recovery_cap"] else "stage1")
    q.add("outf", f"""
SELECT s.dev, s.det, s.status AS health_status, CASE WHEN s.status <> 'not_enough_data' THEN s.hs END AS health_score,
       'det ' || s.det || ': ' || CASE s.status
         WHEN 'ok' THEN format('{{:,.0f}} actuations in {{:.1f}} h, nothing unusual', s.n_on, g.n_cov * 300 / 3600.0)
         WHEN 'not_enough_data' THEN format('only {{:.0f}} actuations in {{:.1f}} h - too few to judge', s.n_on,
                                            g.n_cov * 300 / 3600.0)
         ELSE array_to_string(list_filter([CASE WHEN s.s_stuck > 0 THEN 'stayed ON continuously for '
                                                  || round(s.ep_dur / 60)::BIGINT || ' min' END,
                                             CASE WHEN s.s_dropout > 0 THEN 'went silent' END,
                                             CASE WHEN s.s_night_drop > 0 THEN 'misses vehicles at night' END,
                                             CASE WHEN s.s_level > 0 THEN 'its share of the traffic dropped' END],
                                            lambda x: x IS NOT NULL), '; ')
         END AS health_reason, '' AS health_categories, '' AS health_config, '' AS health_watch
FROM {s1} s
JOIN sig0 g USING (dev) JOIN win w USING (dev) ANTI JOIN xbase x USING (dev, det)""")
    if q.periods:
        _periods_basic(q, s1)
    # signal notes: many detectors with an unusual night; holds shared by several detectors (common failure)
    q.add("cgrp", """
SELECT dev, g, min(s) AS s0, first(dur ORDER BY s, dur) AS dur0, count(DISTINCT round_even(s / 60, 0)) AS k
FROM (SELECT e.dev, e.s, e.dur, list_sort(list_distinct(list_append(coalesce(e.mates, []), e.det))) AS g
      FROM efl e WHERE e.common)
GROUP BY dev, g""")
    if q.opt["signal_silent"]:
        sgn = (", CASE WHEN sr.dev IS NOT NULL THEN 'no detector logged an actuation ' || "
               + _hm(f"w.s_ns + sr.a * {NS5}") + " || '-' || " + _hm(f"w.s_ns + sr.z * {NS5}")
               + " || ' while the signal kept serving greens: all detector inputs lost (detector rack / cabinet)' END")
        sgj = " LEFT JOIN sgr sr USING (dev)"
    else:
        sgn, sgj = "", ""
    q.add("snote", f"""
SELECT w.dev, array_to_string(list_filter([
  CASE WHEN t.dev IS NOT NULL THEN t.n_odd || ' of ' || t.n_ok || ' detectors at this signal had an unusual night on '
       || {_hm('t.d0', '%a %d %b')} || ': a signal-wide event or an input change, not one bad detector' END,
  c.txt{sgn}], lambda x: x IS NOT NULL), '; ') AS health_signal_note
FROM win w LEFT JOIN todnote t USING (dev){sgj}
LEFT JOIN (SELECT g.dev, string_agg(len(g.g) || ' detectors (det ' || array_to_string(g.g, ', ') || ') held ON together '
                  || CASE WHEN g.k = 1 THEN 'from ' || {_hm('w.s_ns + round(g.s0 * 1e9)::BIGINT')} || ' for '
                          || {_dur('g.dur0')} ELSE g.k || ' times from ' || {_hm('w.s_ns + round(g.s0 * 1e9)::BIGINT')} END
                  || ' (same start and end): one common failure, check the cabinet first', '; ' ORDER BY g.s0, g.g) AS txt
           FROM cgrp g JOIN win w USING (dev) GROUP BY g.dev) c USING (dev)""")


def _periods_basic(q: _Q, s1: str):
    """note 154aa: the periods of the detectors without a known function (outf, suspect / bad): the stage-1 silence and
    longest hold (option basic_periods), every other finding over the window; severity from the finding's stage-1
    score (0.35 suspect -> 0.5, 0.75 -> 0.8); no type, so no limit -> pbasic (columns as pall)."""
    c = lambda v: _conf(v, "0.35", "0.75")   # noqa: E731
    bp = q.opt["basic_periods"]
    hold = "s.ep_t0 IS NOT NULL" if bp else "false"
    gap = "s.drop_b1 > s.drop_b0" if bp else "false"
    per = (f"""
UNION ALL SELECT s.dev, s.det, 0, w.s_ns + round(s.ep_t0 * 1e9)::BIGINT, w.s_ns + round(s.ep_t1 * 1e9)::BIGINT, 'stuck on',
       {c('s.s_stuck')}, NULL, false, false FROM s JOIN win w USING (dev) WHERE s.s_stuck > 0 AND {hold}
UNION ALL SELECT s.dev, s.det, 2, w.s_ns + s.drop_b0 * {NS5}, least(w.s_ns + s.drop_b1 * {NS5}, w.e_ns), 'silent',
       {c('s.s_dropout')}, NULL, false, false FROM s JOIN win w USING (dev) WHERE s.s_dropout > 0 AND {gap}""" if bp else "")
    q.add("pbasic", f"""
WITH s AS (SELECT s.* FROM {s1} s ANTI JOIN xbase x USING (dev, det) WHERE s.status IN ('suspect', 'bad'))
SELECT s.dev, s.det, 20 AS o, w.s_ns AS a, w.e_ns AS z, 'stuck on' AS what, {c('s.s_stuck')} AS sev, NULL::DOUBLE AS lim,
       false AS lc, true AS ww FROM s JOIN win w USING (dev) WHERE s.s_stuck > 0 AND NOT {hold}
UNION ALL SELECT s.dev, s.det, 22, w.s_ns, w.e_ns, 'silent', {c('s.s_dropout')}, NULL, false, true
       FROM s JOIN win w USING (dev) WHERE s.s_dropout > 0 AND NOT {gap}
UNION ALL SELECT s.dev, s.det, 23, w.s_ns, w.e_ns, 'misses vehicles at night', {c('s.s_night_drop')}, NULL, false, true
       FROM s JOIN win w USING (dev) WHERE s.s_night_drop > 0
UNION ALL SELECT s.dev, s.det, 24, w.s_ns, w.e_ns, 'counts dropped', {c('s.s_level')}, NULL, false, true
       FROM s JOIN win w USING (dev) WHERE s.s_level > 0{per}""")


FINAL = """
SELECT w.DeviceId, o.det AS Detector, o.health_status, o.health_score, o.health_reason, o.health_categories,
       o.health_config, o.health_watch, n.health_signal_note
FROM (SELECT * FROM outx UNION ALL BY NAME SELECT * FROM outf) o JOIN win w USING (dev) JOIN snote n USING (dev)
ORDER BY CAST(w.DeviceId AS VARCHAR), o.det"""
OUT_COLS = ["DeviceId", "Detector", "health_status", "health_score", "health_reason", "health_categories",
            "health_config", "health_watch", "health_signal_note"]
# note 154aa: one row per period (health(..., periods=True))
PERIODS_FINAL = """
SELECT w.DeviceId, p.det AS Detector, make_timestamp(p.a // 1000) AS "start", make_timestamp(p.z // 1000) AS "end",
       p.what AS condition, round(p.sev, 3) AS severity, p.lim::DOUBLE AS limit_s, coalesce(p.lc, false) AS in_closure,
       p.ww AS whole_window
FROM (SELECT * FROM pall UNION ALL SELECT * FROM pbasic) p JOIN win w USING (dev)
ORDER BY CAST(w.DeviceId AS VARCHAR), p.det, p.a, p.o, p.z"""
PERIOD_COLS = ["DeviceId", "Detector", "start", "end", "condition", "severity", "limit_s", "in_closure", "whole_window"]


def build(events: str = "events", detectors: str = "detectors", windows: str | None = None, start=None, end=None,
          upto: str | None = None, probabilities: bool = True, distinct: bool = True, options: dict | None = None,
          periods: bool = False, configured: bool = False) -> _Q:
    """the health steps; `windows` (optional): a table DeviceId, s_ns, e_ns (epoch ns) of the sample windows (predict
    passes its own); `upto`: stop after that step (testing); `options`: OPTIONS overrides; `periods`: also the steps
    of the periods table (PERIODS_FINAL); `configured`: the detectors table has a configured_function column."""
    q = _Q(options)
    q.periods = periods
    _inputs(q, events, detectors, windows, start, end, probabilities, distinct, configured)
    _bins(q)
    _stage1_both(q)
    LJ, R = _json("health_limits.json"), _json("health_v4_refs.json")
    _measures(q, LJ, R)
    _assemble(q, LJ, R)
    if upto is not None:
        q.ctes = q.ctes[:[n for n, _, _ in q.ctes].index(upto) + 1]
    return q


def health_sql(events_table: str = "events", detectors_table: str = "detectors", start=None, end=None,
               probabilities: bool = True, options: dict | None = None, periods: bool = False, configured: bool = False):
    """The detector health of every signal in `events_table` as ONE DuckDB query (text), to run on the connection that
    holds the tables -- nothing is copied out.

    events_table     DeviceId, Timestamp, EventId, Parameter (the hi-res controller log, local time)
    detectors_table  DeviceId, Detector, phase, function, lanes [, p_advance, p_presence, p_count, p_yellow_red,
                     p_other, p_mid, p_bike] -- each detector's predicted phase, function, lane span and (optional,
                     probabilities=True) class probabilities, e.g. the columns of predict()'s output
                     (phase_pred, function_pred, ...) renamed; with configured=True also configured_function (the
                     function the agency configured, e.g. Presence / Count / Advance / Yellow_Red; NULL = not known):
                     the pulse-mode check then never flags a zone configured as Count / Advance / Yellow_Red
    start / end      the sample window (end exclusive); default: each signal's first event to one second past its last
    periods          False: the detector query (str); True: (detector query, periods query), each complete on its own
    Detector query: one row per detector with actuations in the window (OUT_COLS): DeviceId, Detector, health_status,
    health_score, health_reason, health_categories, health_config, health_watch, health_signal_note.
    Periods query: one row per period (PERIOD_COLS), as health(..., periods=True).
    Table names are inserted as given (quote them yourself if they need it)."""
    q = build(events_table, detectors_table, None, start, end, probabilities=probabilities, options=options,
              periods=periods, configured=configured)
    return (q.text(FINAL), q.text(PERIODS_FINAL)) if periods else q.text(FINAL)


@lru_cache(maxsize=8)
def _script(events_table: str, detectors_table: str, probs: bool, distinct: bool, opt: tuple = (), periods: bool = False,
            configured: bool = False):
    q = build(events_table, detectors_table, f"{PREFIX}wpart", probabilities=probs, distinct=distinct,
              options=dict(opt), periods=periods, configured=configured)
    stmts, drops = q.script([FINAL, PERIODS_FINAL] if periods else [FINAL])
    return q, stmts, drops


def _cols(con, table: str) -> set:
    return {str(r[0]).lower() for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()}


def health(con, events_table: str = "events", detectors_table: str = "detectors", start=None, end=None,
           windows_table: str | None = None, max_events: int | None = 4_000_000,
           distinct: bool = True, options: dict | None = None, periods: bool = False, workers: int = 1):
    """Run the detector health on an existing DuckDB connection (the tables stay where they are) and return a DataFrame
    (columns OUT_COLS, one row per detector), or with periods=True the tuple (detectors, periods): periods has one row
    per period (PERIOD_COLS) -- DeviceId, Detector, start, end (local), condition ('stuck on', 'silent', 'counts
    dropped', ..., 'lane closure', 'timing plan'), severity (0-1: 0.5 at the check's suspect limit, 0.8 at its bad
    limit; NULL for lane closures / timing plans), limit_s (the type's bad limit of an event check: stuck on, count zone
    held ON, spiky counts; else NULL), in_closure (inside a lane closure / timing-plan switch of the detector or within
    30 min of one), whole_window (the finding has no time of its own: the row spans the window).  Every finding of a
    suspect / bad detector has at least one row.
    Signals are processed in groups of at most `max_events` events (memory); the steps run as temp tables (prefix hq_),
    each dropped as soon as it is no longer needed.  The events are read once per group, so store them sorted by
    DeviceId, then Timestamp.  A detectors table with a configured_function column (the agency's configured function:
    Presence, Count, Advance, Yellow_Red, ...; NULL = not known): pulse mode never flags a zone configured as Count,
    Advance or Yellow_Red.
    `workers` > 1: the groups run in that many Python processes at once (note 154ad: one DuckDB process uses only 2-3
    cores on this work, whatever its threads), each with its share of this connection's threads and memory_limit
    (threads = workers gives each one thread, the most work per core) and a temp directory of its own.  The window's
    events are first copied once into a DuckDB file under this connection's temp_directory (the system temp folder when
    that is not an absolute path), so their order and place (a network share) no longer matter; the workers start
    meanwhile.  Same result as workers=1.
    `windows_table` (optional): DeviceId, s_ns, e_ns (epoch ns) of each signal's window, in place of start / end
    (max_events None: no grouping); `distinct` False when the events hold no duplicate rows (skips the
    de-duplication); `options`: OPTIONS overrides.  A window may run from 30 minutes (shorter ones are judged as
    30-minute samples) to MAX_DAYS days."""
    opt = tuple(sorted(_options(options).items()))
    if start is not None and end is not None and pd.Timestamp(end) - pd.Timestamp(start) > pd.Timedelta(days=MAX_DAYS):
        raise ValueError(f"window longer than {MAX_DAYS} days; split it")
    cols = _cols(con, detectors_table)
    probs, cfg = set(PROB.values()) <= cols, "configured_function" in cols
    if windows_table is None:
        q = _Q()
        _inputs(q, events_table, detectors_table, None, start, end, probs)
        wq = _Q()
        wq.subs = q.subs
        wq.ctes = [c for c in q.ctes if c[0] in ("wraw", "wbin", "wsel")]
        con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wall AS " + wq.text(
            "SELECT w.DeviceId, w.s_ns, w.e_ns, r.n FROM wsel w JOIN wraw r USING (DeviceId)"))
    elif max_events is None:
        con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wall AS SELECT DeviceId, s_ns, e_ns, 0 AS n FROM {windows_table}")
    else:
        codes = ",".join(str(c) for c in ALLOWED)
        con.execute(f"""CREATE OR REPLACE TEMP TABLE {PREFIX}wall AS
            SELECT w.DeviceId, w.s_ns, w.e_ns, count(e.EventId) AS n FROM {windows_table} w LEFT JOIN {events_table} e
            ON e.DeviceId = w.DeviceId AND e.EventId IN ({codes}) AND epoch_ns(e.Timestamp) >= w.s_ns
            AND epoch_ns(e.Timestamp) < w.e_ns GROUP BY w.DeviceId, w.s_ns, w.e_ns""")
    sig = con.execute(f"SELECT DeviceId, n FROM {PREFIX}wall ORDER BY CAST(DeviceId AS VARCHAR)").fetchall()
    groups, cur, size = [], [], 0
    for d, n in sig:
        if cur and max_events is not None and size + n > max_events:
            groups.append(cur)
            cur, size = [], 0
        cur.append(d)
        size += n
    if cur:
        groups.append(cur)
    long = con.execute(f"SELECT DeviceId FROM {PREFIX}wall WHERE e_ns - s_ns > {MAX_DAYS} * {DAYNS} LIMIT 1").fetchall()
    if long:
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wall")
        raise ValueError(f"signal {long[0][0]}: window longer than {MAX_DAYS} days; split it")
    job = (probs, distinct, opt, periods, cfg)
    try:
        if workers > 1 and len(groups) > 1:
            res = _parallel(con, groups, events_table, detectors_table, job, workers)
        else:
            q, stmts, drops = _script(events_table, detectors_table, *job)
            res = []
            try:
                for g in groups:
                    con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wpart AS SELECT DeviceId, s_ns, e_ns FROM "
                                f"{PREFIX}wall WHERE DeviceId IN (SELECT unnest(?))", [list(g)])
                    res.append(_group(con, stmts, drops, periods))
            finally:
                for n, _, _ in q.ctes:
                    con.execute(f"DROP TABLE IF EXISTS {PREFIX}{n}")
                con.execute(f"DROP TABLE IF EXISTS {PREFIX}wpart")
    finally:
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wall")
    out = [r[0] for r in res]
    det = pd.concat(out, ignore_index=True)[OUT_COLS] if out else pd.DataFrame(columns=OUT_COLS)
    if not periods:
        return det
    pout = [r[1] for r in res]
    per = pd.concat(pout, ignore_index=True)[PERIOD_COLS] if pout else pd.DataFrame(columns=PERIOD_COLS)
    return det, per


def _group(con, stmts: list, drops: list, periods: bool) -> tuple:
    """the steps for the signals in hq_wpart -> (detectors, periods or None)."""
    nf = 2 if periods else 1
    res = []
    for i, st in enumerate(stmts):
        if i < len(stmts) - nf:
            con.execute(st)
        else:
            res.append(con.execute(st).df())
        for t in drops[i]:
            con.execute(f"DROP TABLE IF EXISTS {t}")
    return res[0], (res[1] if periods else None)


def _bytes(s: str) -> float:
    """a DuckDB memory setting ('9.3 GiB', '512.0 MB', '1000 bytes') in bytes."""
    m = re.fullmatch(r"\s*([\d.]+)\s*(?:([KMGTP])(i?)B|B|bytes)?\s*", str(s), re.I)
    if not m:
        raise ValueError(f"cannot read the memory_limit {s!r}")
    unit = (1024 if m.group(3) else 1000) ** ("KMGTP".find(m.group(2).upper()) + 1) if m.group(2) else 1
    return float(m.group(1)) * unit


def _parallel(con, groups: list, events_table: str, detectors_table: str, job: tuple, workers: int) -> list:
    """health(workers=N): N worker processes (_worker) start while this connection copies the window's events (with
    their group), the windows and the detectors table once into a DuckDB file of the job directory (note 154ad: about
    half the time of a parquet copy partitioned by group); then each worker takes the groups in order, the next one
    nobody has taken.  The results come back pickled (each the worker's own .df(), so the dtypes are those of a run in
    this process) and are returned in group order."""
    import os
    import pickle
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    setting = dict(con.execute("SELECT name, value FROM duckdb_settings() WHERE name IN ('threads', 'memory_limit', "
                               "'temp_directory', 'preserve_insertion_order')").fetchall())
    workers = min(workers, len(groups))
    tdir = setting.get("temp_directory") or ""
    base = tdir if os.path.isabs(tdir) else tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    jd = tempfile.mkdtemp(prefix="atspm_health_", dir=base)
    gp = Path(jd).as_posix()
    procs, attached = [], False
    try:
        with open(f"{jd}/job.json", "w", encoding="utf-8") as f:
            json.dump({"job": [job[0], job[1], [list(x) for x in job[2]], job[3], job[4]], "groups": len(groups),
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
                    [[str(d) for g in groups for d in g], [i for i, g in enumerate(groups) for _ in g]])
        codes = ",".join(str(c) for c in ALLOWED)
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
        for i in range(len(groups)):
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
    """worker k of health(workers=N) on the job directory jd (_parallel): once the data is ready (a byte on stdin; the
    pipe closing without it -- the caller gone -- ends the worker), takes the next free group until none is left."""
    import os
    import pickle
    import sys

    import duckdb
    with open(f"{jd}/job.json", encoding="utf-8") as f:
        J = json.load(f)
    probs, distinct, opt, periods, cfg = J["job"]
    gp = Path(jd).as_posix()
    _, stmts, drops = _script(f"{PREFIX}events", f"{PREFIX}job.detectors", probs, distinct,
                              tuple(tuple(x) for x in opt), periods, cfg)
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
        con.execute(f"CREATE OR REPLACE TEMP VIEW {PREFIX}events AS SELECT DeviceId, Timestamp, EventId, Parameter "
                    f"FROM {PREFIX}job.ev WHERE g = {i}")
        con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wpart AS SELECT DeviceId, s_ns, e_ns FROM {PREFIX}job.wall "
                    f"WHERE g = {i}")
        r = _group(con, stmts, drops, periods)
        with open(f"{jd}/r{i}.tmp", "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(f"{jd}/r{i}.tmp", f"{jd}/r{i}.pkl")
    con.close()


if __name__ == "__main__":      # worker k of health(workers=N): python -P health_duckdb.py <job dir> <k>
    import sys
    _worker(sys.argv[1], int(sys.argv[2]))
