"""Detector health in time bins: what a dashboard reads.  No rule is judged here; the rules are in health_duckdb and ped_health_duckdb, the
findings table in findings.py, the SQL of this layer in hq/sql/findings/bins.sql and held.sql.

    b = bins(con, findings, status, since, end)      # from a judge's two tables, the bins of [since, end)
    b = health_bins(con, "events", "detectors", start, end)    # the same with no store: a health() of each judge's window
    r = incremental.judge(con, "detectors", outputs=("tables", "bins"))    # r.bins: from a store, each bin once

One row per bin per problem (BIN_COLS); no row = no problem found; the latest bin is the live status.  Rows are only ever added: a judge gives the bins since the one before and, with a past
  TimeStamp, any problem it is the first to find in a bin of its window already released, unless a row of that bin, detector and label exists (late_rows; a store keeps what it released in
  health_bins_released for the horizon).
  Score is how much of the detector was failing in that bin: a complete failure over the whole bin is about 1, half a bin of it is 0.5.  Problems
  overlapping at a moment share that moment's worst value, so the rows of a bin add up to the detector's score, at most 1.  Sum of Score / 4 =
  hours of complete failure (15-minute bins).
  Score is not Confidence.  Confidence is a rule's severity (0.5 at its suspect limit, 0.8 at its bad limit; untouched); Score =
  min(1, Confidence / SCORE_FULL[Label]).
  Borderline findings, notes and set-up notes (ped recall, two ped phases, a timing plan, a lane closure, ...) are not problems: no rows.  Not
  enough data: one row a day, Score empty.
"""
from __future__ import annotations

import pandas as pd

from . import findings as F
from .hq import plan as _plan
from .ped_health_duckdb import LIM

BIN_COLS = ["TimeStamp", "DeviceId", "DetectorType", "Detector", "Label", "Score"]
# Score = min(1, Confidence / SCORE_FULL[Label]): the severity at which a rule's period counts as a complete failure.  From the user's labelled faults and the daily
# 3-day windows of real data.  A rule whose severity stops short of 1 reaches its full score where its scale stops.
SCORE_FULL = {
    "stuck on": 0.65, "count zone held ON": 0.65, "silent": 0.6, "silent at night": 0.7, "spiky counts": 0.65,
    "counts dropped": 0.87, "misses vehicles at night": 0.87, "too-fast actuations": 0.87, "too many for the traffic": 0.87,
    "busy at night": 0.82, "no daily pattern": 0.82, "counts too high": 0.95,
    "long hold": 0.8, "machine-regular presses": 0.8, "never calls a walk": 0.8, "sudden stop": 0.8, "walks without presses": 0.79, "sustained pressing": 0.79,
}
# while an ON is held stuck these are its consequences, not more problems
EXPLAINED = ("silent", "silent at night", "misses vehicles at night", "counts dropped", "sudden stop", "walks without presses")
HOLD_LABELS = ("stuck on", "count zone held ON")
DAILY, HOURLY = "D", "h"
KEY = ["TimeStamp", "DeviceId", "DetectorType", "Detector"]


def score(label: str, confidence: float) -> float:
    """the Score of a rule's period at full coverage; a label that is not a problem has none."""
    full = SCORE_FULL.get(label)
    return None if full is None or not confidence or confidence != confidence else min(1.0, confidence / full)


def empty_bins() -> pd.DataFrame:
    return pd.DataFrame({"TimeStamp": pd.Series(dtype="datetime64[us]"), "DeviceId": pd.Series(dtype=object), "DetectorType": pd.Series(dtype=object),
                         "Detector": pd.Series(dtype="Int64"), "Label": pd.Series(dtype=object), "Score": pd.Series(dtype="float64")})[BIN_COLS]


def bin_floor(con, t, bin_minutes: int = 15) -> pd.Timestamp:
    """the start of the bin holding t, as bins are cut."""
    return pd.Timestamp(con.execute(f"SELECT time_bucket(INTERVAL ({int(bin_minutes)}) MINUTE, ?::TIMESTAMP)", [t]).fetchall()[0][0])


def bins(con, findings: pd.DataFrame, status: pd.DataFrame, since, end, bin_minutes: int = 15, held: pd.DataFrame | None = None) -> pd.DataFrame:
    """The rows of the bins of [since, end) (BIN_COLS) from a judge's findings and status tables.  held: DeviceId, DetectorType, Detector, Start, End:
    periods of an ON held stuck, from a store (incremental.judge)."""
    if pd.Timestamp(since) >= pd.Timestamp(end) or findings is None or (not len(findings) and (held is None or not len(held))):
        return empty_bins()
    with_held = held is not None and len(held) > 0
    q = _plan.Plan("fb_")
    q.subs.update({"@FINDINGS@": "fbi_f", "@STATUS@": "fbi_s", "@HELD@": "fbi_h"})
    q.render("findings/bins.sql", since=since, end=end, B=int(bin_minutes), FULL=[[k, float(v)] for k, v in SCORE_FULL.items()], TOO_FEW=F.TOO_FEW,
             EXPLAINED=EXPLAINED, held=with_held)
    try:
        con.register("fbi_f", findings)
        con.register("fbi_s", status)
        if with_held:
            con.register("fbi_h", held)
        out = con.execute(q.text("SELECT * FROM brows ORDER BY TimeStamp, CAST(DeviceId AS VARCHAR), DetectorType DESC, Detector, Label")).df()
    finally:
        for name in ("fbi_f", "fbi_s") + (("fbi_h",) if with_held else ()):
            con.unregister(name)
    out = out[BIN_COLS].copy()
    out["Detector"] = out["Detector"].astype("Int64")
    return out


def late_rows(found: pd.DataFrame, released: pd.DataFrame, since) -> pd.DataFrame:
    """Of the rows a later judge finds in bins before since, already released: those to release now, as new rows with their past TimeStamp.  released: the rows given so far (BIN_COLS).
    A row of that bin, detector and label exists: nothing, existing rows never change.  So that a bin's rows still add up to at most 1, new rows of a bin share what its existing rows leave;
    a bin they already fill gets the row with Score 0.  Not enough data is not a problem found late: only for the day since falls in."""
    rows = found[(found.Label != F.TOO_FEW) | (found.TimeStamp >= pd.Timestamp(since).normalize())]
    if rows.empty:
        return empty_bins()
    if released.empty:
        return rows[BIN_COLS].reset_index(drop=True)
    rows, had = _keyed(rows), _keyed(released)
    rows = rows.merge(had[KEY + ["Label"]].assign(had=True), how="left", on=KEY + ["Label"])
    rows = rows[rows.had.isna()].drop(columns="had")
    if rows.empty:
        return empty_bins()
    rows = rows.merge(had.groupby(KEY, as_index=False).Score.sum().rename(columns={"Score": "used"}), how="left", on=KEY)
    room = (1 - rows.used.fillna(0)).clip(lower=0)
    rows["Score"] = (rows.Score * (room / rows.groupby(KEY).Score.transform("sum")).clip(upper=1)).round(6)
    return rows[BIN_COLS].reset_index(drop=True)


def _keyed(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.astype({"TimeStamp": "datetime64[us]", "Detector": "Int64", "Score": "float64"})


def release(con, findings: pd.DataFrame, status: pd.DataFrame, since, upto, known_from, released: pd.DataFrame, bin_minutes: int = 15, spans: pd.DataFrame | None = None,
            past_spans: pd.DataFrame | None = None) -> pd.DataFrame:
    """What a judge releases: the late rows it finds in [known_from, since), then the rows of the bins of [since, upto).  released: the rows given so far from known_from on; spans, past_spans: the
    periods of ONs held stuck in those two stretches (held())."""
    new = bins(con, findings, status, since, upto, bin_minutes, spans)
    if pd.Timestamp(known_from) >= pd.Timestamp(since):
        return new
    late = late_rows(bins(con, findings, status, known_from, since, bin_minutes, past_spans), released, since)
    return pd.concat([late, new], ignore_index=True) if len(late) and len(new) else (late if len(late) else new)


def held(con, store: str, findings: pd.DataFrame, status: pd.DataFrame, ped_rows: pd.DataFrame | None, since, end, judged, ped: bool) -> tuple:
    """From a store and its judge's tables: (the periods of ONs held stuck in [since, end), for bins(); the carry rows to add; the carry rows to drop).
    Carry rows: TimeStamp, DeviceId, EventId, Parameter."""
    q = _plan.Plan("fh_")
    q.subs.update({"@FINDINGS@": "fhi_f", "@STATUS@": "fhi_s", "@PEDROWS@": "fhi_p"})
    q.render("findings/held.sql", s=f"{store}.", since=since, end=end, judged=judged, ped=ped, vehicle_labels=HOLD_LABELS, hold_gap_s=LIM["hold_gap_s"])
    rows = ped_rows if ped_rows is not None and len(ped_rows) else pd.DataFrame(
        {"DeviceId": findings["DeviceId"].iloc[:0], "Input": pd.Series(dtype="int64"), "Phase": pd.Series(dtype="float64")})
    try:
        con.register("fhi_f", findings)
        con.register("fhi_s", status)
        con.register("fhi_p", rows[["DeviceId", "Input", "Phase"]])
        spans = con.execute(q.text("SELECT * FROM hheld")).df()
        add = con.execute(q.text("SELECT * FROM hinsert")).fetchall()
        drop = con.execute(q.text("SELECT * FROM hdelete")).fetchall()
    finally:
        for name in ("fhi_f", "fhi_s", "fhi_p"):
            con.unregister(name)
    return spans, add, drop


def judge_ends(start, end, every: str = DAILY) -> list:
    """The clock's judge times in (start, end]: the same whatever the runs that carry the events (every: DAILY, HOURLY)."""
    unit = pd.Timedelta(days=1) if every == DAILY else pd.Timedelta(hours=1)
    first = pd.Timestamp(start).floor(every) + unit
    return list(pd.date_range(first, pd.Timestamp(end), freq=unit)) if first <= pd.Timestamp(end) else []


def health_bins(con, events_table: str = "events", detectors_table: str = "detectors", start=None, end=None, *, every: str = DAILY, horizon_days=3,
                bin_minutes: int = 15, **health_options) -> pd.DataFrame:
    """The bins of [start, end) with no store: health() of the window of each judge time of the clock (every: DAILY or HOURLY), and at end, each giving the
    bins since the judge before and its late rows (late_rows).  Equals a store's judges at those times, but for a stuck ON from before a window: only a store carries that."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    h = pd.Timedelta(days=horizon_days)
    parts, since, given = [], start, empty_bins()
    for t in dict.fromkeys(judge_ends(start, end, every) + [end]):
        w0 = max(start, t - h)
        found, status = F.health(con, events_table, detectors_table, w0, t, summary=True, bin_origin="clock", **health_options)
        upto = bin_floor(con, t, bin_minutes)
        b = release(con, found, status, since, upto, min(w0, since), given, bin_minutes)
        if len(b):
            parts.append(b)
            given = b if given.empty else pd.concat([given, b], ignore_index=True)
        given = given[given.TimeStamp >= w0]
        since = max(since, upto)
    return pd.concat(parts, ignore_index=True) if parts else empty_bins()
