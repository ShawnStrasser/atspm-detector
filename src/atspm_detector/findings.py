"""Detector health as tables: vehicle detectors and ped buttons together.  No rule is judged here; the rules are in
health_duckdb (vehicle, per detector) and ped_health_duckdb (ped, per input), the SQL of this layer in hq/sql/findings.

    f = health(con, "events", "detectors")                     # the findings table
    f, s = health(con, "events", "detectors", summary=True)    # + the status table
    f = health(con, "events", vehicle=False)                   # ped buttons only (ped=False: vehicle only)
    f = ped_health(con, "events")                              # the same as vehicle=False
    sql = health_sql("events", "detectors")                    # the vehicle part as one query to run yourself
    f, s = tables(con, detectors, periods, ped_inputs, ped_periods, windows, ped_windows, summary=True)
                                                               # from the per-detector rows, already computed

Findings: one row per finding, no row = no finding (FINDING_COLS).
  DeviceId, DetectorType ("Vehicle" or "Ped"; numbers can overlap), Detector (the ped input for a button; empty on a
  signal note), Start, End, Label, Confidence, Detail (the evidence text of that finding).
  - Each period of the health rules is a row: Label = its condition, Confidence = its severity (0-1: 0.5 at the check's
    suspect limit, 0.8 at its bad limit; empty for lane closures and timing plans; 0 for the ped notes "omitted during
    preemption" and "call dropped").  A period that shares a check with other periods shares its Detail.
  - A finding with no time of its own gets one row per calendar day of the window, Start and End being that day cut to
    the window.
  - So do the rows the rules give no severity for, with a fixed Confidence: borderline (watch) findings WATCH_CONFIDENCE
    (their Label is the note: "borderline stuck on", ...); config notes, signal notes and not_enough_data NOTE_CONFIDENCE.
Status (summary=True): one row per judged detector or button and calendar day of the window (STATUS_COLS): DeviceId,
  DetectorType, Detector, Date, Status.  The rules judge a window as a whole, so each day of the window carries the
  window's status.  check_status() enforces one row per (DeviceId, DetectorType, Detector, Date).
"""
from __future__ import annotations

import pandas as pd

from . import health_duckdb as _hq
from . import ped_health_duckdb as _ped
from .hq import plan as _plan
from .hq import render as _render
from .hq import run as _run

VEHICLE, PED = "Vehicle", "Ped"
FINDING_COLS = ["DeviceId", "DetectorType", "Detector", "Start", "End", "Label", "Confidence", "Detail"]
STATUS_COLS = ["DeviceId", "DetectorType", "Detector", "Date", "Status"]
STATUS_KEY = ["DeviceId", "DetectorType", "Detector", "Date"]
Q15_COLS = ["TimeStamp", "DeviceId", "DetectorType", "Detector", "Label", "Confidence", "Status"]
ACTIVE_COLS = ["DeviceId", "DetectorType", "Detector", "Label", "Start", "End", "Confidence", "State", "FirstSeen", "LastSeen",
               "Missed"]
WINDOW_COLS = ["DeviceId", "start", "end"]
PREFIX = "fd_"
NOTE_CONFIDENCE = 0.0       # as the ped notes: not a fault
WATCH_CONFIDENCE = 0.4      # below 0.5, the severity at a check's suspect limit: they stayed under the rule that flags
TOO_FEW = "not_enough_data"

# what each period reads in health_reason: category, what the piece starts with ("": the category's other piece), kind.
# Kind = the period's condition; a detector without a known function has no categories (category "")
_SILENT = "^(no actuations|logs the same count|whole phase dropped)"
PIECES = (("Stuck on", "^held ON", "stuck on"), ("Stuck on", "^count zone ON", "count zone held ON"),
          ("Counts fall", _SILENT, "silent"), ("Counts fall", "", "counts dropped"),
          ("Misses vehicles at night", _SILENT, "silent at night"),
          ("Misses vehicles at night", "", "misses vehicles at night"),
          ("Too-fast actuations", "", "too-fast actuations"), ("Too many for the traffic", "", "too many for the traffic"),
          ("Busy at night", "", "busy at night"), ("No daily pattern", "", "no daily pattern"),
          ("Spiky counts", "", "spiky counts"), ("Counts too high", "", "counts too high"),
          ("Lane closure", "", "lane closure"), ("Timing plan", "", "timing plan"), ("Configuration", "", "pulse mode"),
          ("", "^stayed ON", "stuck on"), ("", "^went silent", "silent"),
          ("", "^misses vehicles at night", "misses vehicles at night"), ("", "^its share", "counts dropped"))
CLOSURES = ("Lane closure", "Timing plan")
CLOSURE_KINDS = ("lane closure", "timing plan")
# condition -> kinds of piece in order of preference: a silence of a detector rated only for its nights is told there
KINDS = [[k, k, 1] for k in dict.fromkeys(p[2] for p in PIECES)] + [
    ["silent", "silent at night", 2], ["silent", "misses vehicles at night", 3]]
_CATS = "|".join(dict.fromkeys(p[0] for p in PIECES if p[0]))
_HEADS = "|".join(("count zone ON", "no actuations", "logs the same count", "whole phase dropped", "nearly stopped counting",
                   "counts fell", "counts and occupancy both fell", "counts dropped", "stayed ON", "went silent",
                   "misses vehicles at night", "its share", "[A-Z][a-z]{2} [0-9]{2}:[0-9]{2}-"))

CONTEXT = dict(
    CAT_HEAD=f"^({_CATS}): ", CAT_SPLIT=f"; ({_CATS}): ", HEAD_SPLIT=f"; ({_HEADS})", PIECES=PIECES, KINDS=KINDS,
    CLOSURES=CLOSURES, CLOSURE_KINDS=CLOSURE_KINDS,
    LABEL_RX=(r"^(P[0-9]+ )?(" + "|".join(f.replace("Yellow_Red", "Yellow-red") for f in _hq.FNS)
              + r")( \(covers [0-9]+ lanes?\))? - "),
    VEH_CONFIG=(("^Extension time on a count zone", "extension time on a count zone"),
                ("^Set to pulse", "set to pulse but holds ON")),
    SIGNAL_NOTES=(("unusual night", "unusual night at many detectors"), ("held ON together", "detectors held ON together"),
                  ("^no detector logged", "all detector inputs lost")),
    PED_PIECES=(("^stuck ON", "stuck on"), ("^held ON", "long hold"), ("^presses every", "machine-regular presses"),
                ("^pressed ", "never calls a walk"), ("^no presses for", "sudden stop"), ("^walks jumped with no presses", "walks without presses"), ("^sustained pressing: ", "sustained pressing")),
    PED_SPLIT="; (stuck ON: |held ON |presses every |pressed |no presses for |walks jumped with no presses |sustained pressing: )",
    PED_NOTES=(("^omitted during preemption", _ped.NOTE), ("^possible configuration issue|ped calls for P[0-9]+ waited", "call dropped"),
               ("^never calls a walk not checked", "never calls a walk not checked"),
               ("^serves ped overlap", "serves ped overlap"), ("^calls P[0-9]+ and P[0-9]+ at once|^calls two ped phases", "calls two ped phases"),
               ("on ped recall", "on ped recall"), ("^same button as ped", "one button wired to two inputs"),
               ("^switches ON together", "switches ON together with another input"), ("^ped in recall", "ped in recall"),
               ("^ped movement not checked", "ped movement not checked"), ("^its presses call no phase", "no ped movement")),
    PED_WAITS=(_ped.NOTE, "call dropped"),
    CONFIG_LABEL="config note", SIGNAL_LABEL="signal note", TOO_FEW=TOO_FEW,
    NOTE_CONF=NOTE_CONFIDENCE, WATCH_CONF=WATCH_CONFIDENCE)

FINDINGS_FINAL, STATUS_FINAL = (s for _, s, _ in _plan.steps(_render.render("findings/final.sql")))
_SOURCES = {"@WIN@": "fdi_win", "@VDET@": "fdi_vdet", "@VPER@": "fdi_vper", "@PDET@": "fdi_pdet", "@PPER@": "fdi_pper"}


def build(vehicle: bool = True, ped: bool = True) -> _plan.Plan:
    """the steps of the two tables; the sources are bound by placeholder (fdi_*, the frames tables() registers)."""
    q = _plan.Plan(PREFIX)
    q.subs.update(_SOURCES)
    q.render("findings/tables.sql", vehicle=vehicle, ped=ped, **CONTEXT)
    return q


def check_status(status: pd.DataFrame) -> pd.DataFrame:
    """the status table, or ValueError when a detector or button has two rows for one date."""
    dup = status[status.duplicated(STATUS_KEY, keep=False)]
    if len(dup):
        r = dup.iloc[0]
        raise ValueError(f"status table: {len(dup)} rows break one row per (DeviceId, DetectorType, Detector, Date), "
                         f"first {r.DeviceId}, {r.DetectorType}, {r.Detector}, {r.Date}")
    return status


def empty_findings() -> pd.DataFrame:
    return pd.DataFrame({"DeviceId": pd.Series(dtype=object), "DetectorType": pd.Series(dtype=object),
                         "Detector": pd.Series(dtype="Int64"), "Start": pd.Series(dtype="datetime64[us]"),
                         "End": pd.Series(dtype="datetime64[us]"), "Label": pd.Series(dtype=object),
                         "Confidence": pd.Series(dtype="float64"), "Detail": pd.Series(dtype=object)})[FINDING_COLS]


def _empty_status() -> pd.DataFrame:
    return pd.DataFrame({"DeviceId": pd.Series(dtype=object), "DetectorType": pd.Series(dtype=object),
                         "Detector": pd.Series(dtype="Int64"), "Date": pd.Series(dtype="datetime64[us]"),
                         "Status": pd.Series(dtype=object)})[STATUS_COLS]


def _has(frame) -> bool:
    return frame is not None and len(frame) > 0


def _periods(periods, inputs: pd.DataFrame) -> pd.DataFrame:
    """a periods table for the SQL; empty: typed as a table with rows would be (an empty frame may hold any dtypes)."""
    if _has(periods):
        return periods[["DeviceId", "Detector", "start", "end", "condition", "severity", "whole_window"]]
    return pd.DataFrame({"DeviceId": pd.Series(dtype=inputs["DeviceId"].dtype), "Detector": pd.Series(dtype="int64"),
                         "start": pd.Series(dtype="datetime64[us]"), "end": pd.Series(dtype="datetime64[us]"),
                         "condition": pd.Series(dtype=object), "severity": pd.Series(dtype="float64"),
                         "whole_window": pd.Series(dtype=bool)})


def _windows(frames: list) -> pd.DataFrame:
    """[(detector type, its output frame, its windows), ...] -> one windows table; every signal of an output has one."""
    parts = []
    for kind, frame, win in frames:
        if win is None:
            raise ValueError(f"{kind} rows need their windows (DeviceId, start, end)")
        win = pd.DataFrame(win)[WINDOW_COLS]
        if win.DeviceId.duplicated().any():
            raise ValueError(f"{kind} windows: signal {win.DeviceId[win.DeviceId.duplicated()].iloc[0]} has two windows")
        missing = set(frame.DeviceId) - set(win.DeviceId)
        if missing:
            raise ValueError(f"{kind} windows: no window for signal {sorted(map(str, missing))[0]}")
        parts.append(win.assign(DetectorType=kind))
    return pd.concat(parts, ignore_index=True)


def tables(con, detectors: pd.DataFrame | None = None, periods: pd.DataFrame | None = None,
           ped_inputs: pd.DataFrame | None = None, ped_periods: pd.DataFrame | None = None,
           windows: pd.DataFrame | None = None, ped_windows: pd.DataFrame | None = None, summary: bool = False):
    """The findings table (FINDING_COLS), or with summary=True (findings, status), from the per-detector rows on connection
    `con`.  detectors / periods: health_duckdb.health(..., periods=True); ped_inputs / ped_periods:
    ped_health_duckdb.ped_health(..., periods=True); leave a pair out when that side did not run.  windows: DeviceId, start,
    end (end exclusive) of each judged signal; ped_windows: the same for the ped side, when it differs."""
    vehicle, ped = _has(detectors), _has(ped_inputs)
    if not vehicle and not ped:
        return (empty_findings(), _empty_status()) if summary else empty_findings()
    frames = ([(VEHICLE, detectors, windows)] if vehicle else []) + \
             ([(PED, ped_inputs, ped_windows if ped_windows is not None else windows)] if ped else [])
    src = {"fdi_win": _windows(frames)}
    if vehicle:
        src.update(fdi_vdet=detectors, fdi_vper=_periods(periods, detectors))
    if ped:
        src.update(fdi_pdet=ped_inputs, fdi_pper=_periods(ped_periods, ped_inputs))
    q = build(vehicle, ped)
    stmts, drops = q.script([FINDINGS_FINAL, STATUS_FINAL] if summary else [FINDINGS_FINAL])
    try:
        for name, frame in src.items():
            con.register(name, frame)
        res = _run.group(con, stmts, drops, 2 if summary else 1)
    finally:
        for n, _, _ in q.ctes:
            con.execute(f"DROP TABLE IF EXISTS {PREFIX}{n}")
        for name in src:
            con.unregister(name)
    out = _finish(res[0], FINDING_COLS)
    return (out, check_status(_finish(res[1], STATUS_COLS))) if summary else out


def _finish(frame: pd.DataFrame, cols: list) -> pd.DataFrame:
    frame = frame[cols].copy()
    frame["Detector"] = frame["Detector"].astype("Int64")
    return frame


def event_windows(con, events_table: str = "events", detectors_table: str = "detectors", start=None, end=None,
                  windows_table: str | None = None, bin_origin: str = "window", ped: bool = False) -> pd.DataFrame:
    """Each signal's judged window (WINDOW_COLS), by the rules of the vehicle health (ped=True: of the ped health) on the same
    arguments.  ped=True gives only the signals with ped input events."""
    if ped:
        sql = _plan.steps(_render.render("ped/windows.sql", E=events_table, start=start, end=end, CODES=_ped.CODES, carry=False))[0][1]
        return con.execute(f'SELECT DeviceId, make_timestamp((s * 1e6)::BIGINT) AS "start", make_timestamp((e * 1e6)::BIGINT) AS "end" '
                           f"FROM ({sql})").df()
    if windows_table is None:
        wq = _plan.Plan(_hq.PREFIX)
        wq.subs.update(_hq._subs(events_table, detectors_table, None))
        wq.render("windows.sql", **_hq._context(_hq._options(None), False, False, start, end, False, False))
        wq.ctes = [c for c in wq.ctes if c[0] in ("wraw", "wbin", "wsel")]
        src = "(" + wq.text("SELECT DeviceId, s_ns, e_ns FROM wsel") + ")"
    else:
        src = windows_table
    floor = f"s_ns::BIGINT - s_ns::BIGINT % {_hq.NSQ}" if bin_origin == "clock" else "s_ns::BIGINT"
    return con.execute(f'SELECT DeviceId, make_timestamp(({floor}) // 1000) AS "start", '
                       f'make_timestamp(e_ns::BIGINT // 1000) AS "end" FROM {src}').df()


def health(con, events_table: str = "events", detectors_table: str = "detectors", start=None, end=None, *,
           vehicle: bool = True, ped: bool | None = None, summary: bool = False,
           windows_table: str | None = None, max_events: int | None = 4_000_000,
           ped_max_events: int | None = 3_000_000, distinct: bool = True, options: dict | None = None,
           ped_options: dict | None = None, ped_assign: bool = False, workers: int = 1, bin_origin: str = "window"):
    """The health of the vehicle detectors and ped buttons of `events_table` on an existing DuckDB connection (nothing is
    copied out): the findings table (FINDING_COLS), or with summary=True (findings, status).

    events_table      DeviceId, Timestamp, EventId, Parameter (the controller log, local time; any codes)
    detectors_table   DeviceId, Detector, phase, function, lanes [, p_advance, ... p_bike] [, configured_function]: the
                      classifier's answers or your own (health_duckdb)
    start / end       the window (end exclusive); default: each signal's first event to one second past its last
    vehicle           False leaves the vehicle checks out (detectors_table is then not read)
    ped               None runs the ped checks when the log holds ped input events, True always, False never
    summary           also return the status table
    windows_table     DeviceId, s_ns, e_ns (epoch ns) of each signal's window, in place of start / end; no ped side, so
                      ped must not be True
    max_events        vehicle signals are processed in groups of at most this many events (memory); ped_max_events: ped
    distinct          False when the events hold no duplicate rows (skips the de-duplication)
    options           vehicle rule overrides (health_duckdb.OPTIONS); ped_options: ped rule overrides (ped_health_duckdb.OPT)
    ped_assign        also run the ped movement model (ped_assign; needs onnxruntime)
    workers           > 1: the vehicle groups run in that many Python processes at once; same result
    bin_origin        "window" (default): bins from each signal's window start; "clock": from the quarter-hour at or
                      before it, as an incremental store bins (its judge then gives this run's answers)"""
    if not vehicle and ped is False:
        raise ValueError("vehicle=False and ped=False leave nothing to judge")
    if windows_table is not None:
        if ped:
            raise ValueError("windows_table has no ped side; give start / end, or ped=False")
        ped = False
    det = per = win = pin = ped_in = ped_per = None
    if ped is not False:
        pin = event_windows(con, events_table, start=start, end=end, ped=True)
        if len(pin):
            ped_in, ped_per = _ped.ped_health(con, events_table, start, end, periods=True, max_events=ped_max_events,
                                              options=ped_options, assign=ped_assign)
    if vehicle:
        det, per = _hq.health(con, events_table, detectors_table, start, end, windows_table=windows_table,
                              max_events=max_events, distinct=distinct, options=options, periods=True, workers=workers,
                              bin_origin=bin_origin)
        win = event_windows(con, events_table, detectors_table, start, end, windows_table, bin_origin)
    return tables(con, det, per, ped_in, ped_per, win, pin, summary=summary)


def ped_health(con, events_table: str = "events", start=None, end=None, *, summary: bool = False,
               max_events: int | None = 3_000_000, options: dict | None = None, assign: bool = False):
    """The ped buttons only: health(con, events_table, start=start, end=end, vehicle=False, ped=True, ...)."""
    return health(con, events_table, start=start, end=end, vehicle=False, ped=True, summary=summary, ped_max_events=max_events,
                  ped_options=options, ped_assign=assign)


def health_sql(events_table: str = "events", detectors_table: str = "detectors", start=None, end=None,
               probabilities: bool = True, options: dict | None = None, summary: bool = False, configured: bool = False,
               bin_origin: str = "window"):
    """The vehicle findings as ONE DuckDB query (text), to run on the connection that holds the tables: the rows of
    health(..., ped=False).  summary=True: (findings query, status query), each complete on its own.  probabilities: the
    detectors table has the p_* columns; configured: it has configured_function.  Table names are inserted as given (quote
    them yourself if they need it).  The ped rules are partly in Python, so they have no query of this kind."""
    q = _hq.build(events_table, detectors_table, None, start, end, probabilities=probabilities, options=options, periods=True,
                  configured=configured, bin_origin=bin_origin)
    q.add("fdi_vdet", _hq.FINAL)
    q.add("fdi_vper", _hq.PERIODS_FINAL)
    q.add("fdi_win", f"""SELECT DeviceId, '{VEHICLE}' AS DetectorType, make_timestamp(s_ns // 1000) AS "start",
                         make_timestamp(e_ns // 1000) AS "end" FROM win""")
    f = build(True, False)
    f.subs.update({"@WIN@": f"{_hq.PREFIX}fdi_win", "@VDET@": f"{_hq.PREFIX}fdi_vdet", "@VPER@": f"{_hq.PREFIX}fdi_vper"})
    return (f.text(FINDINGS_FINAL, q), f.text(STATUS_FINAL, q)) if summary else f.text(FINDINGS_FINAL, q)


def q15(con, findings: pd.DataFrame, status: pd.DataFrame, since, end) -> pd.DataFrame:
    """The optional 15-minute table (Q15_COLS): each finding of a suspect or bad detector or button, one row per quarter-hour
    of [since, end) it touches (the highest Confidence of that label in that quarter-hour)."""
    q = _plan.Plan("fq_")
    q.subs.update({"@FINDINGS@": "fdi_f", "@STATUS@": "fdi_s"})
    q.render("findings/q15.sql", since=since, end=end)
    try:
        con.register("fdi_f", findings)
        con.register("fdi_s", status)
        return _finish(con.execute(q.text("SELECT * FROM q15 ORDER BY TimeStamp, CAST(DeviceId AS VARCHAR), DetectorType DESC, "
                                          "Detector, Label")).df(), Q15_COLS)
    finally:
        con.unregister("fdi_f")
        con.unregister("fdi_s")


def active_next(con, active_table: str, findings: pd.DataFrame, status: pd.DataFrame, end) -> pd.DataFrame:
    """The next state of the active table (ACTIVE_COLS; hq/sql/findings/active.sql has the rules) after a judge ending at end."""
    q = _plan.Plan("fa_")
    q.subs.update({"@FINDINGS@": "fdi_f", "@STATUS@": "fdi_s", "@ACTIVE@": active_table})
    q.render("findings/active.sql", end=end)
    try:
        con.register("fdi_f", findings)
        con.register("fdi_s", status)
        return _finish(con.execute(q.text("SELECT * FROM anext ORDER BY CAST(DeviceId AS VARCHAR), DetectorType DESC, Detector, "
                                          'Label, "Start"')).df(), ACTIVE_COLS)
    finally:
        con.unregister("fdi_f")
        con.unregister("fdi_s")
