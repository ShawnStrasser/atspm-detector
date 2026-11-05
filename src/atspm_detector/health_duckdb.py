"""The rules under atspm_detector.health() (findings.py makes its tables from the rows given here): one row per detector and
one per period.  Detector health in DuckDB SQL (note 153): the six health checks of 1.1.0 as plain SQL (ported from the numpy modules
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
The SQL lives in Jinja templates (hq/sql, rendered by hq/plan.py; this module chooses the options and limits):
  windows.sql, detectors.sql   each signal's window (bin_origin "window": from its start; "clock": from the quarter-hour
                            at or before it), the detectors table
  extract/*.sql             the extractor (incremental-health design 3-4): each signal's window of events, read once,
                            as small rows -- per 5-min bin and detector / phase / signal: counts, starts with their
                            colour, time ON / unknown / green (integer ns), the chains' facts; holds, comms gaps, plan
                            changes, too-fast tick counts -- written to the store tables (store.sql; temp tables hq_health_*
                            here), and what is still open at the window's last event
  store_read.sql            the store's rows read back with the run's keys
  finalize.sql              the open chains and colour pieces closed at the window end, the rows settled (d5, p5)
  adapters.sql              today's step names (occ5, hold, act, tf, q5b, ...) rebuilt from those rows
  judge/bins.sql            5-min counts / time ON per detector (dense grid) with running sums for window sums
  judge/stage1.sql          sudden silence, stuck-on, night count, level shift -> pass-1 status (twice: pass 2 without
                            the detectors pass 1 calls bad as yardsticks)
  judge/measures.sql ...    15-min occupancy context, Count zone held ON, per-ON statistics, too fast / too many, hourly
                            GEH, spiky bursts, holds with their queue context, busy at night
  judge/verdict.sql ...     limits, scores, resolver / severity (CASE logic), categories, reason text; the periods table
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .hq import plan as _plan
from .hq import render as _render
from .hq import run as _run

WDIR = Path(__file__).resolve().parent / "weights" / "health"
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
           # counting on the window's other days (judge/mate_hours.sql)
           "rise_dead_days": True,
           # approved 2026-10-29 (notes 154aa, 154ac): lane-closure candidates also from near-zero runs of >= 3
           # quarter-hours (any closure of an hour or more), not only of >= 2 judged hours; and no closure on a day when
           # the detector has faults of its own before and after it (LC_FAULT_DAY) (judge/closure.sql)
           "closure_1h": True,
           # approved 2026-10-29 (notes 154aa, 154ac): a Presence detector whose calls are nearly all pulses (PULSE)
           # looks set to pulse mode -> suspect at least, category "Configuration"; count, advance and yellow-red zones
           # may pulse: one configured as such (configured_function, when given) is never flagged (judge/outputs.sql)
           "pulse_mode": True,
           # approved 2026-10-28 (notes 154q, 154w): time ON with nothing imputed (extract/detector.sql) -- an ON then
           # an ON = still ON unless a signal-wide logging gap lies between (then unknown); OFF then OFF = missing data
           # (the second OFF ignored); a leading OFF dropped.  Always on: 2.0.0 removed False (1.2.0's atspm-style
           # imputation) -- the extractor that feeds batch and incremental health alike implements this one only
           "occ_strict": True,
           # approved 2026-10-28 (notes 154t, 154w, 154aa): detectors that do not look like traffic are nobody's yardstick
           # for the night checks (True) and the count checks (hourly yardstick gh, mate_drop / count_rise mates: "all")
           "clean_yardsticks": "all"}
OPTIONS_OFF = {k: (None if k == "lane_pairs" else k == "occ_strict") for k in OPTIONS}   # 1.2.0, time ON not imputed
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
# (every call ~0.13 s, ~0 % ON)
PULSE = {"n": 200, "share": 0.8, "s": 0.2}
# note 154ac (with closure_1h): a lane closure with faults of the detector's own on the same calendar day on both sides
# (sides 2; 1 = either side) is part of the fault; all: every closure (False: only the quarter-hour candidates)
LC_FAULT_DAY = {"sides": 2, "all": True}
MAX_DAYS = 92                           # longest window health() takes (a guard, not a dtype limit: bins INT32,
                                         # times BIGINT ns); memory ~15 MB per signal-day (note 154)
PREFIX = "hq_"
# design 3.4: "window" = each signal's window start (today's origin: health()'s answers as before); "clock" = the
# quarter-hour at or before it (bins, quarter-hours and hours on the clock: the incremental store's origin)
BIN_ORIGINS = ("window", "clock")
# the functions whose green overlap the judge reads (held_on: Count zones held ON); batch health() extracts it for these
# only, an incremental store for every detector with a phase (design decision 6) -- the answers are the same
OVERLAP_FNS = ("Count",)
C7 = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")
# limits, scores, verdict (health_v4)
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
# lane-closure partner classes: like with like (Other and Bike zones never take part)
LC_CLASS = {"Advance": "up", "Mid": "up", "Count": "bar", "Presence": "bar", "Yellow_Red": "bar"}
WHOLE = ("busy at night", "no daily pattern", "misses vehicles at night")   # findings with no real time of their own
OUT_COLS = ["DeviceId", "Detector", "health_status", "health_score", "health_reason", "health_categories",
            "health_config", "health_watch", "health_signal_note"]
# note 154aa: one row per period (health(..., periods=True))
PERIOD_COLS = ["DeviceId", "Detector", "start", "end", "condition", "severity", "limit_s", "in_closure", "whole_window"]


@lru_cache(maxsize=2)
def _json(name: str) -> dict:
    with open(WDIR / name, encoding="utf-8") as f:
        return json.load(f)


def _options(opt: dict | None) -> dict:
    o = {**OPTIONS, **(opt or {})}
    bad = set(o) - set(OPTIONS)
    if bad:
        raise ValueError(f"unknown health option(s): {sorted(bad)}")
    if o["lane_pairs"] not in (None, "nema", "any"):
        raise ValueError("lane_pairs must be None, 'nema' or 'any'")
    if o["occ_strict"] is not True:
        raise ValueError("occ_strict=False (1.2.0's imputed time ON) was removed in 2.0.0; occ_strict must be True")
    if o["clean_yardsticks"] not in (False, True, "all"):
        raise ValueError("clean_yardsticks must be False, True or 'all'")
    return o


def _tables(LJ: dict, R: dict) -> dict:
    """the limit tables the templates inject as VALUES (rows of Python values; T in the templates)."""
    h = LJ["h24"]
    ltab = [[t, k, float(a), float(b)] for t in ("stuck_s", "silent_exp") for k, (a, b) in h[t].items()]
    ltab += [["busy_z", k, float(v), None] for k, v in h["busy_z"].items()]
    ltab += [["per_1000", k, float(v), None] for k, v in LJ["too_fast"]["per_1000"].items()]
    lref = []
    for tab, levels in (("stuck_x", R["cell"]["v4c"]["stuck_x"]), ("n_ep", R["cell"]["v4c"]["n_ep"]),
                        ("q5_gy", R["fast118c"]["q5_gy"]), ("q5_all", R["fast118c"]["q5_all"])):
        for i, lev in enumerate(levels):
            for r in lev["rows"]:
                v = r[-1]
                if v is None or v != v:
                    continue
                key = dict(zip(lev["keys"], (str(x) for x in r[:-1])))
                lref.append([tab, i] + [key.get(c) for c in ("fn", "span", "cmode", "wg", "band")] + [float(v)])
    todg = [[k, float(v["bc"][0]), float(v["bc"][1]), float(v["bc"][2]), float(v["sc"]), bool(v["occ"]),
             float(v["bo"][0]) if v["occ"] else None, float(v["bo"][1]) if v["occ"] else None,
             float(v["so"]) if v["occ"] else None, float(v["cal_med"]), float(v["cal_mad"])]
            for k, v in R["tod"]["groups"].items()]
    return {"ltab": ltab, "lref": lref, "todg": todg,
            "olp": [[float(a), float(b), float(c)] for a, b, c in R["occ_lim_pulse"]],
            "ctp": [[float(a), float(b)] for a, b in R["count_t_pass"]],
            "qt": [[str(a), float(b)] for a, b in R["q_type"]],
            "kv": [[k, i, CATEGORY[k]] for i, k in enumerate(KEYS)],
            "cls": [[k, v] for k, v in LC_CLASS.items()]}


def _context(opt: dict, periods: bool, windows: bool, start, end, probabilities: bool, configured: bool,
             bin_origin: str = "window", store: str = "", overlap: tuple | None = OVERLAP_FNS, phase_only: bool = False,
             fixed_origin: bool | None = None) -> dict:
    """everything the health templates read: the options, the inputs' shape, the bin origin, the limits and the
    constants.  store: the store's name and a dot, for steps reading a store (empty: a batch run's temp tables); overlap:
    the functions whose green overlap the extractor keeps (OVFN; None: every detector with a phase); phase_only: the
    detectors table is read for each detector's phase only (an update); fixed_origin: the extractor's times count from
    the epoch (ORG) rather than each signal's window start, as with bin_origin "clock" and in every store (design 3.4), so
    a double expression on a time gives the same bits for any window and any chunking; TE: the window end on that axis."""
    if bin_origin not in BIN_ORIGINS:
        raise ValueError(f"bin_origin must be one of {BIN_ORIGINS}")
    LJ, R = _json("health_limits.json"), _json("health_v4_refs.json")
    tf = LJ["too_fast"]
    P = LJ["spiky"]
    D = {"drop_geh_half": LJ["count_drops_geh"][0] / 2,                         # half the suspect GEH of a count drop
         "silent_bad": LJ["current"]["silent_exp"] * LJ["short"]["silent_bad_x"],   # bad silence, short samples
         "spk_k": int(P["bin_min"] * 60 // 300), "spk_wb": int(P["window_bins"]),   # spiky: 5-min bins per bin,
         "spk_2wb": 2 * int(P["window_bins"]),                                      # window bins (twice)
         "spk_lim": P["bad"] * P["bin_min"] * 60}                                   # its bad bursts, in seconds
    groups = (lambda m: [[k for k in KEYS if m[k] == g] for g in sorted(set(m.values()))])
    fixed = bin_origin == "clock" if fixed_origin is None else fixed_origin
    return dict(opt=opt, periods=periods, windows=windows, clock=bin_origin == "clock", start=start, end=end,
                probabilities=probabilities, configured=configured, OVFN=overlap, store=store, phase_only=phase_only,
                ORG="0" if fixed else "w.s_ns", TE="(w.e_ns / 1e9)" if fixed else "w.T",
                # too fast: the 0.1-s ticks the extractor keeps (the largest limit, 0.8 s at one lane, in ticks)
                TFC=math.ceil(max(tf["gap_s_stopbar"], tf["gap_s_other"]) * 10), LJ=LJ, T=_tables(LJ, R), D=D, NS5=NS5, NSH=NSH, NSQ=NSQ, DAYNS=DAYNS,
                ALLOWED=ALLOWED, FNS=FNS, PROB=PROB, PROBCOLS=list(PROB.values()), C7=C7, KEYS=KEYS, STAT=STAT,
                FAMS=groups(FAMILY), EVS=groups(EVID), CATEGORY=CATEGORY, NO_SHARED=NO_SHARED, WHOLE=WHOLE,
                NEMA_PAIRS=NEMA_PAIRS, M=MATE_HOURS, CY=CLEAN_YARD, PULSE=PULSE, LCF=LC_FAULT_DAY)


FINAL, PERIODS_FINAL = (s for _, s, _ in _plan.steps(_render.render("outputs/tables.sql")))


# the carry of a batch run: nothing carried in (a store binds its table here, design 2.2)
NO_CARRY = (f"(SELECT DeviceId, NULL::TIMESTAMP_NS AS TimeStamp, NULL::INTEGER AS EventId, NULL::INTEGER AS Parameter "
            f"FROM {PREFIX}win WHERE false)")


def _subs(events: str, detectors: str, windows: str | None, distinct: bool = True) -> dict:
    s = {"@EVENTS@": events, "@DETECTORS@": detectors, "@DISTINCT@": "DISTINCT" if distinct else "", "@CARRY@": NO_CARRY}
    if windows is not None:
        s["@WINDOWS@"] = windows
    return s


def build(events: str = "events", detectors: str = "detectors", windows: str | None = None, start=None, end=None,
          upto: str | None = None, probabilities: bool = True, distinct: bool = True, options: dict | None = None,
          periods: bool = False, configured: bool = False, bin_origin: str = "window") -> _plan.Plan:
    """the health steps; `windows` (optional): a table DeviceId, s_ns, e_ns (epoch ns) of the sample windows (predict
    passes its own); `upto`: stop after that step (testing); `options`: OPTIONS overrides; `periods`: also the steps
    of the periods table (PERIODS_FINAL); `configured`: the detectors table has a configured_function column;
    `bin_origin`: "window" (each signal's window start, default) or "clock" (the quarter-hour at or before it)."""
    q = _plan.Plan(PREFIX, _options(options))
    q.subs.update(_subs(events, detectors, windows, distinct))
    q.render("health.sql", **_context(q.opt, periods, windows is not None, start, end, probabilities, configured,
                                      bin_origin))
    if upto is not None:
        q.ctes = q.ctes[:[n for n, _, _ in q.ctes].index(upto) + 1]
    return q


def health_sql(events_table: str = "events", detectors_table: str = "detectors", start=None, end=None,
               probabilities: bool = True, options: dict | None = None, periods: bool = False, configured: bool = False,
               bin_origin: str = "window"):
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
    bin_origin       "window" (default): bins from each signal's window start; "clock": from the quarter-hour at or
                     before it (the window then starts there), as the incremental store bins
    Detector query: one row per detector with actuations in the window (OUT_COLS): DeviceId, Detector, health_status,
    health_score, health_reason, health_categories, health_config, health_watch, health_signal_note.
    Periods query: one row per period (PERIOD_COLS), as health(..., periods=True).
    Table names are inserted as given (quote them yourself if they need it)."""
    q = build(events_table, detectors_table, None, start, end, probabilities=probabilities, options=options,
              periods=periods, configured=configured, bin_origin=bin_origin)
    return (q.text(FINAL), q.text(PERIODS_FINAL)) if periods else q.text(FINAL)


@lru_cache(maxsize=8)
def _script(events_table: str, detectors_table: str, probs: bool, distinct: bool, opt: tuple = (), periods: bool = False,
            configured: bool = False, bin_origin: str = "window"):
    q = build(events_table, detectors_table, f"{PREFIX}wpart", probabilities=probs, distinct=distinct,
              options=dict(opt), periods=periods, configured=configured, bin_origin=bin_origin)
    stmts, drops = q.script([FINAL, PERIODS_FINAL] if periods else [FINAL])
    return q, stmts, drops


def _cols(con, table: str) -> set:
    return {str(r[0]).lower() for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()}


def health(con, events_table: str = "events", detectors_table: str = "detectors", start=None, end=None,
           windows_table: str | None = None, max_events: int | None = 4_000_000,
           distinct: bool = True, options: dict | None = None, periods: bool = False, workers: int = 1,
           bin_origin: str = "window"):
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
    30-minute samples) to MAX_DAYS days.
    `bin_origin`: "window" (default) bins each signal's window from its start, as before 2.0.0; "clock" starts the
    window on the quarter-hour at or before it, so bins, quarter-hours and hours lie on the clock, and times count from
    the epoch (the bins and the time axis of the incremental store: its answers equal this run's)."""
    opt = tuple(sorted(_options(options).items()))
    if bin_origin not in BIN_ORIGINS:
        raise ValueError(f"bin_origin must be one of {BIN_ORIGINS}")
    if start is not None and end is not None and pd.Timestamp(end) - pd.Timestamp(start) > pd.Timedelta(days=MAX_DAYS):
        raise ValueError(f"window longer than {MAX_DAYS} days; split it")
    cols = _cols(con, detectors_table)
    probs, cfg = set(PROB.values()) <= cols, "configured_function" in cols
    if windows_table is None:
        wq = _plan.Plan(PREFIX)
        wq.subs.update(_subs(events_table, detectors_table, None))
        wq.render("windows.sql", **_context(_options(None), False, False, start, end, probs, False))
        wq.ctes = [c for c in wq.ctes if c[0] in ("wraw", "wbin", "wsel")]
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
    groups = _wall_groups(con, max_events, bin_origin)
    job = (probs, distinct, opt, periods, cfg, bin_origin)
    nf = 2 if periods else 1
    try:
        if workers > 1 and len(groups) > 1:
            _, stmts, drops = _script(f"{PREFIX}events", f"{PREFIX}job.detectors", *job)
            res = _run.parallel(con, groups, events_table, detectors_table, (stmts, drops), nf, workers, ALLOWED)
        else:
            q, stmts, drops = _script(events_table, detectors_table, *job)
            res = _run_groups(con, q, stmts, drops, groups, nf)
    finally:
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wall")
    return _frames(res, periods)


def _wall_groups(con, max_events: int | None, bin_origin: str) -> list:
    """the signal groups of the windows table hq_wall (DeviceId, s_ns, e_ns, n), its windows checked, and in clock mode
    started on the quarter-hour."""
    sig = con.execute(f"SELECT DeviceId, n FROM {PREFIX}wall ORDER BY CAST(DeviceId AS VARCHAR)").fetchall()
    groups = _run.groups(sig, max_events)
    long = con.execute(f"SELECT DeviceId FROM {PREFIX}wall WHERE e_ns - s_ns > {MAX_DAYS} * {DAYNS} LIMIT 1").fetchall()
    if long:
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wall")
        raise ValueError(f"signal {long[0][0]}: window longer than {MAX_DAYS} days; split it")
    if bin_origin == "clock":   # as the step win does: what workers copy of the events starts there too
        con.execute(f"UPDATE {PREFIX}wall SET s_ns = s_ns::BIGINT - s_ns::BIGINT % {NSQ}")
    return groups


def _run_groups(con, q, stmts: list, drops: list, groups: list, nf: int) -> list:
    """the script of plan q for each signal group, in this process: [(detectors, periods or None), ...]."""
    res = []
    try:
        for g in groups:
            con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wpart AS SELECT DeviceId, s_ns, e_ns FROM "
                        f"{PREFIX}wall WHERE DeviceId IN (SELECT unnest(?))", [list(g)])
            r = _run.group(con, stmts, drops, nf)
            res.append((r[0], r[1] if nf > 1 else None) + tuple(r[2:]))
    finally:
        for n, _, _ in q.ctes:
            con.execute(f"DROP TABLE IF EXISTS {PREFIX}{n}")
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wpart")
    return res


def _frames(res: list, periods: bool):
    """the groups' results as one table, or two with periods."""
    out = [r[0] for r in res]
    det = pd.concat(out, ignore_index=True)[OUT_COLS] if out else pd.DataFrame(columns=OUT_COLS)
    if not periods:
        return det
    pout = [r[1] for r in res]
    per = pd.concat(pout, ignore_index=True)[PERIOD_COLS] if pout else pd.DataFrame(columns=PERIOD_COLS)
    return det, per
