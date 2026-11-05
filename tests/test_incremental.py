"""Stage 4 of the incremental-health design: chunked updates, the carry, finalize and retention.  Any cutting of the events
into updates must give the same settled store rows and the same answers as one update, and those answers must equal
health(start=T - H, end=T, bin_origin="clock", periods=True) (design 3.5).

T4.1   chunk invariance: one chunk, 15-, 5- and 60-minute chunks, 3 seeds of random us cuts, a cut on an event and 1 us after
T4.2   batch identity, 3 days in 15-minute chunks; a window ending on an odd quarter-hour
T4.3   edge scenes (a-m), each crossing several cuts, each with a check that it reaches its branch; the rare branches of
       test_extract in chunks; random logs of the detector and colour logic
T4.4   carry closure: carrying twice equals carrying once, row for row
T4.5   bounded state over 14 days
T4.6   retention; what straddles the cut; the horizon a store records
T4.7   late, re-delivered and backlog data; wrong clocks, old, ahead and out of range, each row in one count; a signal starting
       over beside live carry; the clock changes
T4.8   atomicity: a failure before COMMIT, a process killed before COMMIT
T4.9   guards
T4.10  the extractor's fingerprint is pinned
T4.11  warm store against a cold health(start=S)
plus the ownership test (design 3.2 rule 2) on chunks: at each cut the store holds exactly the facts of the events so far."""
import os
import subprocess
import sys
import textwrap

import duckdb
import numpy as np
import pandas as pd
import pytest

import make_golden as G
import synth
import test_extract as X
from atspm_detector import health_duckdb as H
from atspm_detector import incremental as I

ST = "health_store"
DAY = pd.Timedelta(days=1)
FEATURES = I.FEATURES
# steps of a judge that must not depend on the chunking: the settled rows, the open objects and their verdicts
SETTLED = {"win": "SELECT * FROM win",
           "d5": "SELECT * FROM d5",
           "p5": "SELECT * FROM p5",
           "tkg": "SELECT dev, det, q, ta, tb, sum(n)::BIGINT AS n FROM tkg GROUP BY ALL",
           "rs5": "SELECT * FROM rs5",
           "rgq": "SELECT * FROM rgq",
           "rtko": "SELECT dev, det, a, sum(n)::BIGINT AS n FROM rtko GROUP BY ALL",
           "rhold": "SELECT * FROM rhold",
           "rplan": "SELECT * FROM rplan",
           "oxl": "SELECT * FROM oxl",
           "oxt": "SELECT * FROM oxt",
           "fch": "SELECT * FROM fch",
           "fpc": "SELECT * FROM fpc"}
# counts: a key without a row and a row with 0 are the same fact (a labelled row does not carry the chain's counts)
COUNTS = ("nev", "non", "nrel", "nst", "nfst", "nchat", "nst_gy", "nfst_r2", "nchat_r2", "nclean", "n5s", "n60s", "ngreen")


def connect(ev, det):
    con = G.connect(ev, det)
    con.execute("SET memory_limit = '2GB'; SET max_temp_directory_size = '4GB'")
    return con


def end_of(ev) -> pd.Timestamp:
    """the first midnight after the log"""
    return ev.Timestamp.max().normalize() + DAY


def feed(con, cuts, store=ST, horizon_days=3, **kw):
    I.attach_store(con, name=store, **kw)
    return [I.update(con, "events", "detectors", store=store, upto=c, horizon_days=horizon_days) for c in cuts]


def grid(a, b, step):
    x = list(pd.date_range(a + pd.Timedelta(step), b, freq=step))
    return x + [b] if not x or x[-1] != b else x


def few_then_slow(seeds, n):
    """the first n seeds run by default, the rest under the slow mark (pytest -m slow)"""
    return [s if i < n else pytest.param(s, marks=pytest.mark.slow) for i, s in enumerate(seeds)]


def random_cuts(a, b, n, seed):
    """n cuts at random microseconds in [a, b), then b"""
    rng = np.random.default_rng(seed)
    us = np.sort(rng.integers(a.value // 1000, b.value // 1000, n))
    return [pd.Timestamp(int(x) * 1000) for x in us] + [b]


def _settled_rows(con, q):
    stmts, drops = q.script(list(SETTLED.values()))
    out, first = {}, len(stmts) - len(SETTLED)
    names = list(SETTLED)
    try:
        for i, st in enumerate(stmts):
            r = con.execute(st)
            if i >= first:
                df = r.df()
                for c in df.columns:
                    if c in COUNTS:
                        df[c] = df[c].fillna(0)
                out[names[i - first]] = df.sort_values(list(df.columns), kind="stable").reset_index(drop=True)
            for t in drops[i]:
                con.execute(f"DROP TABLE IF EXISTS {t}")
        return out
    finally:
        for n, _, _ in q.ctes:
            con.execute(f"DROP TABLE IF EXISTS {H.PREFIX}{n}")


def settled(con, store, end, horizon_days=3):
    """the judge's settled rows, open objects and verdicts: {step: rows, sorted}, a key without a row counting as a count
    of 0."""
    q, _ = I._judge_plan(con, "detectors", store, end - pd.Timedelta(days=horizon_days), end, H._options(None), True, None)
    try:
        con.execute(f"CREATE OR REPLACE TEMP TABLE {H.PREFIX}wpart AS SELECT DeviceId, s_ns, e_ns FROM {H.PREFIX}wall")
        return _settled_rows(con, q)
    finally:
        con.execute(f"DROP TABLE IF EXISTS {H.PREFIX}wall")
        con.execute(f"DROP TABLE IF EXISTS {H.PREFIX}wpart")


def same_as_batch(con, store, end, horizon_days, what=""):
    """the store's settled rows against the same steps of health(start=T - H, end=T, bin_origin="clock"), rebuilt from the raw
    events; the batch's green overlap is kept for Count zones only (OVERLAP_FNS), so the store's time ON in green is compared
    on those."""
    q = H.build("events", "detectors", start=end - pd.Timedelta(days=horizon_days), end=end, periods=True, bin_origin="clock")
    a, b = settled(con, store, end, horizon_days), _settled_rows(con, q)
    win, zones = a["win"], con.execute("SELECT CAST(DeviceId AS VARCHAR), Detector::INT FROM detectors WHERE function IN ('Count')").fetchall()
    dev = dict(zip(win.DeviceId.astype(str), win.dev))
    keep = {(dev[d], z) for d, z in zones if d in dev}
    for x in (a, b):
        d5 = x["d5"]
        d5["ong5_ns"] = [v if (i, j) in keep else pd.NA for i, j, v in zip(d5.dev, d5.det, d5.ong5_ns)]
        x["d5"] = d5.sort_values(list(d5.columns.drop("ong5_ns")), kind="stable").reset_index(drop=True)
    same_settled(a, b, what or "store against batch")
    return a


def same_settled(a, b, what=""):
    assert set(a) == set(b)
    for k in a:
        assert list(a[k].columns) == list(b[k].columns), (what, k)
        va, vb = G.values(a[k]), G.values(b[k])
        for c in va:
            assert va[c] == vb[c], f"{what} {k}.{c}: {_first_diff(va[c], vb[c])} (rows {len(va[c])}, {len(vb[c])})"


def _first_diff(x, y):
    d = [i for i, (p, q) in enumerate(zip(x, y)) if p != q]
    return [(i, x[i], y[i]) for i in d[:3]]


def same_outputs(a, b, what=""):
    """two (detectors, periods): the same rows in any order, every column exact."""
    for x, y, name in zip(a, b, ("detectors", "periods")):
        assert list(x.columns) == list(y.columns)
        keys = [c for c in x.columns if not pd.api.types.is_float_dtype(x[c])]
        x = x.sort_values(keys, kind="stable").reset_index(drop=True)
        y = y.sort_values(keys, kind="stable").reset_index(drop=True)
        assert len(x) == len(y), (what, name, len(x), len(y))
        vx, vy = G.values(x), G.values(y)
        for c in vx:
            assert vx[c] == vy[c], f"{what} {name}.{c}: {_first_diff(vx[c], vy[c])}"


def carry_rows(con, store):
    return con.execute(f"SELECT TimeStamp, DeviceId, EventId, Parameter FROM {store}.health_carry ORDER BY ALL").fetchall()


# --- T4.1, T4.2 ---------------------------------------------------------------------------------------------------------
_EDGE = {}


def edge():
    """the edge set of test_extract, one update (the reference of T4.1)."""
    if not _EDGE:
        ev, det = synth.edges(*synth.make(3, 2, seed=5))
        con = connect(ev, det)
        T = end_of(ev)
        feed(con, [T], store="one")
        _EDGE.update(con=con, T=T, ev=ev, ref=settled(con, "one", T, 2), out=I.judge_rows(con, "detectors", store="one", end=T, horizon_days=2))
    return _EDGE


def _cuts(kind):
    E = edge()
    a, T, ev = E["ev"].Timestamp.min().floor("15min"), E["T"], E["ev"]
    if kind in ("15min", "5min", "60min"):
        return grid(a, T, kind)
    if kind.startswith("seed"):
        return random_cuts(a, T, 80, int(kind[4:]))
    # cuts exactly on events and one microsecond after them, whatever kind of event it is
    t = ev.sample(20, random_state=7).Timestamp.sort_values()
    return sorted(set(t)) + [T] if kind == "on" else sorted(set(t + pd.Timedelta(microseconds=1))) + [T]




def test_one_update_equals_batch():
    E = edge()
    con, T = E["con"], E["T"]
    same_outputs(E["out"], H.health(con, "events", "detectors", start=T - 2 * DAY, end=T, periods=True, bin_origin="clock"))
    same_as_batch(con, "one", T, 2)
    # a store keeps the green overlap of every detector with a phase (design decision 6), a batch run of the Count zones
    assert con.execute("SELECT count(DISTINCT Detector) FROM one.health_detector_5min WHERE On_Green5_ns IS NOT NULL").fetchall()[0][0] > 2


def test_batch_identity_hourly_chunks():
    """a window ending on an odd quarter-hour: the store's rows of the hour its start falls in begin at that start"""
    ev, det = synth.make(2, 2, seed=11)
    T = ev.Timestamp.min().normalize() + DAY + pd.Timedelta(hours=8, minutes=15)
    con = connect(ev, det)
    S = T - DAY
    I.attach_store(con)
    # each update sees only the rows of its chunk, as atspm's runs do
    for c in grid(S, T, "60min"):
        con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?",
                    [c - pd.Timedelta(minutes=60), c])
        I.update(con, "chunk", "detectors")
    got = I.judge_rows(con, "detectors", end=T, horizon_days=1)
    same_outputs(got, H.health(con, "events", "detectors", start=S, end=T, periods=True, bin_origin="clock"))
    assert len(got[0]) and len(got[1])




def test_compaction_changes_no_total():
    ev, det = synth.make(2, 1, seed=4)
    con = connect(ev, det)
    T = end_of(ev)
    feed(con, grid(ev.Timestamp.min().floor("15min"), T, "3h"))
    before = con.execute(f"SELECT count(*) FROM {ST}.health_detector_5min").fetchall()[0][0]
    ref = settled(con, ST, T, 1)
    out = I.judge_rows(con, "detectors", end=T, horizon_days=1)
    I.trim(con, horizon_days=3, compact=True)
    assert con.execute(f"SELECT count(*) FROM {ST}.health_detector_5min").fetchall()[0][0] < before
    same_settled(settled(con, ST, T, 1), ref)
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), out)




# --- ownership (design 3.2 rule 2) on chunks ------------------------------------------------------------------------------


# --- T4.3 edge scenes -----------------------------------------------------------------------------------------------------
_BASE = []


def base():
    if not _BASE:
        _BASE.extend(synth.make(2, 1, seed=3, faults=False))
    return _BASE


def at(sec):
    return base()[0].Timestamp.min().normalize() + pd.Timedelta(seconds=sec)


def scene_log(rows, drop=None):
    ev, det = base()
    if drop is not None:
        ev = ev[~drop(ev)]
    if rows:
        add = pd.DataFrame(rows, columns=ev.columns).astype({"EventId": "int32", "Parameter": "int32"})
        ev = pd.concat([ev, add], ignore_index=True)
    return ev, det


def dets(ev, chans, a, b, dev="S00"):
    return (ev.DeviceId == dev) & ev.EventId.isin([81, 82]) & ev.Parameter.isin(chans) & (ev.Timestamp >= a) & (ev.Timestamp < b)


def colour(ev, ph, a, b, dev="S00"):
    return (ev.DeviceId == dev) & ev.EventId.isin(X.COLOUR) & (ev.Parameter == ph) & (ev.Timestamp >= a) & (ev.Timestamp < b)


def silence(ev, a, b, dev="S00"):
    return (ev.DeviceId == dev) & (ev.Timestamp >= a) & (ev.Timestamp < b)


def intervals(con, kind, store=ST):
    return con.execute(f"SELECT * FROM {store}.health_intervals WHERE Kind = ?", [kind]).df()


def run_scene(rows, drop, cuts, reach=None):
    """the log in the given cuts (seconds of the day) and in one update: the settled rows, the carry and the answers equal;
    the answers equal the batch's.  reach(con, stats) checks the log reached its branch."""
    ev, det = scene_log(rows, drop)
    T = end_of(ev)
    con = connect(ev, det)
    stats = feed(con, [at(c) for c in sorted(cuts)] + [T])
    same_as_batch(con, ST, T, 1)
    got = I.judge_rows(con, "detectors", end=T, horizon_days=1)
    same_outputs(got, H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))
    if reach is not None:
        reach(con, stats, T)
    return con


def S00(t, e, p):
    return ("S00", at(t), e, p)




def test_b1_stuck_on_still_open_at_the_end():
    t0 = 80000.0

    def reach(con, stats, T):
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        f = x["fch"]
        f = f[(f.dev == w) & (f.det == 1)]
        assert len(f) == 1 and not f.gap.iloc[0] and f.dur.iloc[0] > 300
    run_scene([S00(t0 - 5, 81, 1), S00(t0, 82, 1), S00(t0 + 400, 82, 1)], lambda e: dets(e, [1], at(t0 - 10), at(90000)),
              [t0 - 1, t0 + 1, t0 + 399, t0 + 401, 82000, 84000, 85000], reach)






































# the rare branches of test_extract, in chunks: random cuts and a cut after each of the last events


# --- random logs of the detector and colour logic -----------------------------------------------------------------------------
def fuzz_log(seed):
    """3 signals x 6 h of random detector strings (fast pulses, holds, re-logged ONs, stray OFFs, ON and OFF at one instant),
    colour strings (short and long pieces, repeats, changes at one instant), plans and comms gaps."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-10-05")
    rows, dets_ = [], []
    tick = lambda x: np.round(x, 1)
    for s in range(3):
        dev = f"F{s}"
        for ph in (1, 2, 3, 4, 16):
            t = rng.uniform(0, 60)
            while t < 21000:
                rows.append((dev, tick(t), int(rng.choice([1, 1, 8, 9, 10, 7, 11])), ph))
                t += float(rng.choice([0.0, 0.1, 1.0, 1.9, 2.0, 2.1, 4.9, 5.0, 5.1, 30.0, 300.0, 2000.0], p=[.03, .03, .1, .05, .05, .05, .05, .05, .05, .4, .1, .04]))
        for det in range(1, 21):
            ph = int(rng.choice([1, 2, 3, 4, 16]))
            fn = str(rng.choice(["Count", "Presence", "Advance"]))
            dets_.append({"DeviceId": dev, "Detector": det, "phase": float(ph), "function": fn, "lanes": 1.0, **{c: (0.9 if f == fn else 0.02) for f, c in synth.PROBS.items()}})
            t = rng.uniform(0, 200)
            while t < 21000:
                e = int(rng.choice([82, 81], p=[.55, .45]))
                rows.append((dev, tick(t), e, det))
                if rng.random() < .05:              # ON and OFF at one instant
                    rows.append((dev, tick(t), 81 if e == 82 else 82, det))
                t += float(rng.choice([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 3.0, 20.0, 70.0, 400.0, 2500.0], p=[.15, .1, .1, .1, .1, .1, .1, .1, .08, .05, .02]))
        rows += [(dev, tick(t), 131, int(rng.integers(1, 5))) for t in rng.uniform(0, 21000, 6)]
    df = pd.DataFrame(rows, columns=["DeviceId", "t", "EventId", "Parameter"])
    keep = np.ones(len(df), bool)
    for s in range(3):                                      # comms gaps
        for g in rng.uniform(0, 20000, 4):
            keep &= ~((df.DeviceId == f"F{s}") & (df.t >= g) & (df.t < g + rng.uniform(130, 900)))
    df = df[keep]
    ts = t0 + pd.to_timedelta((df.t * 10).round().astype("int64") * 100, unit="ms")
    ev = pd.DataFrame({"DeviceId": df.DeviceId.to_numpy(), "Timestamp": ts.to_numpy(), "EventId": df.EventId.astype("int32").to_numpy(), "Parameter": df.Parameter.astype("int32").to_numpy()})
    return ev, pd.DataFrame(dets_)




# --- T4.4 ---------------------------------------------------------------------------------------------------------------------


# --- T4.5 ---------------------------------------------------------------------------------------------------------------------


# --- T4.6 ---------------------------------------------------------------------------------------------------------------------


def at_from(a, sec):
    return (a + pd.Timedelta(seconds=sec)).value






def add_rows(ev, rows):
    return pd.concat([ev, pd.DataFrame(rows, columns=ev.columns).astype({"EventId": "int32", "Parameter": "int32"})], ignore_index=True)










def test_the_store_keeps_its_horizon():
    """attach_store's horizon holds for update, judge and trim unless one of them is given another"""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    t = pd.date_range(a, a + 9 * DAY, freq="15min", inclusive="left")
    pulses = pd.concat([pd.DataFrame({"DeviceId": "S00", "Timestamp": t + pd.Timedelta(seconds=s), "EventId": e, "Parameter": 1}) for s, e in ((0, 82), (1, 81))])
    pulses = pulses.astype({"EventId": "int32", "Parameter": "int32"})
    con = connect(pulses, det)
    T = a + 9 * DAY
    horizon = lambda st=ST: I._meta(con, st)["horizon_days"]
    I.attach_store(con, horizon_days=7)
    for c in grid(a, T, "1D"):
        through = I.update(con, "events", "detectors", upto=c).through
    assert horizon() == "7" and pd.Timestamp(I._meta(con, ST)["trimmed_before"]) == through.floor("15min") - 7 * DAY
    got = I.judge_rows(con, "detectors")                         # a 7-day window, not refused
    assert len(got[0]) == 1
    same_outputs(got, I.judge_rows(con, "detectors", horizon_days=7))
    with pytest.raises(ValueError, match="before the store's trim"):
        I.judge_rows(con, "detectors", horizon_days=8)
    I.trim(con)
    I.attach_store(con)                                     # attaching again changes nothing
    assert horizon() == "7" and pd.Timestamp(I._meta(con, ST)["trimmed_before"]) == through.floor("15min") - 7 * DAY
    # a horizon given to attach_store is recorded and the next update trims by it
    I.attach_store(con, horizon_days=2)
    con.execute("CREATE TABLE next_day AS SELECT * REPLACE (Timestamp + INTERVAL 1 DAY AS Timestamp) FROM events WHERE Timestamp >= ?", [T - DAY])
    through = I.update(con, "next_day", "detectors").through
    assert horizon() == "2" and pd.Timestamp(I._meta(con, ST)["trimmed_before"]) == through.floor("15min") - 2 * DAY
    same_outputs(I.judge_rows(con, "detectors"), I.judge_rows(con, "detectors", horizon_days=2))
    # a horizon given to an update with nothing new is recorded too, and the next update's trim uses it
    I.update(con, "next_day", "detectors", horizon_days=1.5)
    assert horizon() == "1.5"
    con.execute("CREATE TABLE day_after AS SELECT * REPLACE (Timestamp + INTERVAL 1 DAY AS Timestamp) FROM next_day")
    through = I.update(con, "day_after", "detectors").through
    assert pd.Timestamp(I._meta(con, ST)["trimmed_before"]) == through.floor("15min") - 1.5 * DAY
    # a new store: 3; a horizon given to update is recorded
    I.attach_store(con, name="other")
    assert horizon("other") == "3"
    I.update(con, "events", "detectors", store="other", horizon_days=1)
    assert horizon("other") == "1"
    same_outputs(I.judge_rows(con, "detectors", store="other"), I.judge_rows(con, "detectors", store="other", horizon_days=1))
    with pytest.raises(ValueError, match="before the store's trim"):
        I.judge_rows(con, "detectors", store="other", horizon_days=2)


# --- T4.7 ---------------------------------------------------------------------------------------------------------------------
def store_rows(con, store):
    return {t: sorted(map(repr, con.execute(f"SELECT * FROM {store}.{t}").fetchall())) for t in FEATURES + ("health_carry", "health_detector_phase")}


def test_rerun_is_idempotent_and_late_rows_are_counted():
    ev, det = base()
    con = connect(ev, det)
    a = ev.Timestamp.min().normalize()
    mid = a + pd.Timedelta(hours=8)
    feed(con, [mid])
    before = store_rows(con, ST)
    s = I.update(con, "events", "detectors", upto=mid)       # the same update again
    assert s.events_in == 0 and sum(s.events_late.values()) > 0 and s.rows_written == {}
    assert store_rows(con, ST) == before
    # an event before the watermark, and a duplicate of a processed one, in a later chunk: dropped and counted
    n0 = len(I.judge_rows(con, "detectors", horizon_days=1).detectors)
    con.execute("CREATE TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?", [mid, mid + pd.Timedelta(hours=1)])
    con.execute("INSERT INTO chunk VALUES ('S00', ?, 82, 1), ('S00', ?, 81, 1)", [mid - pd.Timedelta(hours=1), mid - pd.Timedelta(minutes=59)])
    con.execute("INSERT INTO chunk SELECT * FROM events WHERE DeviceId = 'S01' AND Timestamp >= ? AND Timestamp < ? LIMIT 5", [mid - pd.Timedelta(hours=1), mid])
    s = I.update(con, "chunk", "detectors")
    assert s.events_late == {"S00": 2, "S01": 5}
    feed_clean = connect(ev, det)
    feed(feed_clean, [mid + pd.Timedelta(hours=1)])
    assert store_rows(con, ST)["health_carry"] == store_rows(feed_clean, ST)["health_carry"]
    T = (mid + pd.Timedelta(hours=1, minutes=15)).floor("15min")
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), I.judge_rows(feed_clean, "detectors", end=T, horizon_days=1))
    assert n0 > 0




def test_chunk_with_no_new_rows_and_a_signal_with_nothing_new():
    ev, det = base()
    con = connect(ev, det)
    a = ev.Timestamp.min().normalize()
    feed(con, [a + pd.Timedelta(hours=1)])
    con.execute("CREATE TABLE empty_chunk AS SELECT * FROM events WHERE false")
    s = I.update(con, "empty_chunk", "detectors")
    assert s.events_in == 0 and s.signals == 0
    con.execute("CREATE TABLE s0 AS SELECT * FROM events WHERE DeviceId = 'S00' AND Timestamp >= ? AND Timestamp < ?", [a + pd.Timedelta(hours=1), a + pd.Timedelta(hours=2)])
    before = con.execute(f"SELECT * FROM {ST}.health_carry WHERE DeviceId = 'S01' ORDER BY ALL").fetchall()
    s = I.update(con, "s0", "detectors")
    assert s.signals == 1 and con.execute(f"SELECT * FROM {ST}.health_carry WHERE DeviceId = 'S01' ORDER BY ALL").fetchall() == before




def test_upto_keeps_a_wrong_clock_out_of_the_store():
    """an event stamped a year ahead, in a chunk: with upto it is not processed, moves no watermark and trims nothing"""
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    con = connect(ev, det)
    mid = a + pd.Timedelta(hours=12)
    feed(con, [mid], horizon_days=.25)
    con.execute("CREATE TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?", [mid, T])
    con.execute("INSERT INTO chunk VALUES ('S00', ?, 82, 1)", [T + 365 * DAY])
    s = I.update(con, "chunk", "detectors", upto=T, horizon_days=.25)
    assert s.through < T and s.events_after_upto == {"S00": 1} and not s.events_bad_stamp
    clean = connect(ev, det)
    feed(clean, [mid, T], horizon_days=.25)
    assert store_rows(con, ST) == store_rows(clean, ST)


def feature_rows(con, store):
    """what the judge reads: all but the log of phases, whose first row takes the time of the signal's first update"""
    x = store_rows(con, store)
    del x["health_detector_phase"]
    return x


def bad_clock_table(con, name, rows):
    con.execute(f"CREATE TABLE {name} (DeviceId VARCHAR, Timestamp TIMESTAMP, EventId INT, Parameter INT)")
    con.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)










USABLE = f"EventId IN ({','.join(map(str, H.ALLOWED))}) AND (EventId NOT IN (81, 82) OR Parameter BETWEEN 1 AND 64)"
















def fall_back_log():
    """the fall clock change on zone-free local stamps: 01:00-02:00 comes twice.  second: a second pass with its own events
    (the first pass's, 3 s later), all before the first pass's last event of each signal."""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    h1, h2 = a + pd.Timedelta(hours=1), a + pd.Timedelta(hours=2)
    second = ev[(ev.Timestamp >= h1) & (ev.Timestamp < h2 - pd.Timedelta(minutes=5))].assign(Timestamp=lambda x: x.Timestamp + pd.Timedelta(seconds=3))
    return ev, second, det, a, h1, h2








# --- T4.8 ---------------------------------------------------------------------------------------------------------------------
class FailBefore:
    """a connection that fails at the first statement starting with `word` (after `skip` of them)"""

    def __init__(self, con, word, skip=0):
        self.con, self.word, self.skip = con, word, skip

    def _check(self, sql):
        if sql.lstrip().upper().startswith(self.word):
            if self.skip == 0:
                self.skip = -1
                raise RuntimeError("injected")
            self.skip -= 1

    def execute(self, sql, *a):
        self._check(sql)
        return self.con.execute(sql, *a)

    def executemany(self, sql, *a):
        self._check(sql)
        return self.con.executemany(sql, *a)

    def __getattr__(self, n):
        return getattr(self.con, n)










# --- T4.9, T4.10 ------------------------------------------------------------------------------------------------------------
def test_guards():
    ev, det = base()
    con = connect(ev, det)
    a = ev.Timestamp.min().normalize()
    with pytest.raises(ValueError, match="occ_strict"):
        I.attach_store(con, options={"occ_strict": False})
    I.attach_store(con)
    assert I.trim(con) is None                              # nothing to trim yet
    with pytest.raises(ValueError, match="holds no events"):
        I.judge_rows(con, "detectors")
    with pytest.raises(ValueError, match="occ_strict"):
        I.update(con, "events", "detectors", options={"occ_strict": False})
    with pytest.raises(ValueError, match="quarter-hours"):
        I.update(con, "events", "detectors", horizon_days=0.001)
    with pytest.raises(ValueError, match="quarter-hours"):
        I.update(con, "events", "detectors", horizon_days=-1)
    with pytest.raises(ValueError, match="letters"):
        I.attach_store(con, name="a; DROP TABLE events")
    with pytest.raises(ValueError, match="green_overlap"):
        I.attach_store(con, green_overlap="some")
    con.execute("CREATE TABLE dup AS SELECT * FROM detectors UNION ALL SELECT * FROM detectors LIMIT 100")
    with pytest.raises(ValueError, match="listed more than once"):
        I.update(con, "events", "dup")
    mid = a + pd.Timedelta(hours=8, minutes=7)
    I.update(con, "events", "detectors", upto=mid)
    th = I.update(con, "events", "detectors", upto=mid).through
    with pytest.raises(ValueError, match="after them"):
        I.judge_rows(con, "detectors", end=th.floor("15min"))
    with pytest.raises(ValueError, match="quarter-hour"):
        I.judge_rows(con, "detectors", end=a + pd.Timedelta(hours=9, minutes=1))
    # an event exactly on the quarter-hour: an end there is at the watermark, refused too
    q = th.floor("15min") + pd.Timedelta(minutes=15)
    I.attach_store(con, name="exact")
    I.update(con, "events", "detectors", store="exact", upto=mid)
    con.execute("CREATE TABLE on_q (DeviceId VARCHAR, Timestamp TIMESTAMP, EventId INT, Parameter INT)")
    con.execute("INSERT INTO on_q VALUES ('S00', ?, 131, 1)", [q])
    assert I.update(con, "on_q", "detectors", store="exact").through == q
    with pytest.raises(ValueError, match="after them"):
        I.judge_rows(con, "detectors", store="exact", end=q)
    I.detach_store(con, "exact")
    with pytest.raises(ValueError, match="quarter-hours"):
        I.judge_rows(con, "detectors", horizon_days=0.01)
    # the default end: the first quarter-hour after the last event
    d, p = I.judge_rows(con, "detectors", horizon_days=1)
    same_outputs((d, p), I.judge_rows(con, "detectors", horizon_days=1, end=th.floor("15min") + pd.Timedelta(minutes=15)))
    with pytest.raises(ValueError, match="green_overlap"):
        I.attach_store(con, green_overlap="count")
    con.execute(f"UPDATE {ST}.health_meta SET Value = '0' WHERE Key = 'extractor_version'")
    with pytest.raises(ValueError, match="extractor version"):
        I.attach_store(con)
    con.execute(f"UPDATE {ST}.health_meta SET Value = '0' WHERE Key = 'format_version'")
    with pytest.raises(ValueError, match="format version"):
        I.attach_store(con)
    assert I.detach_store(con) is None




FINGERPRINT = "037dfcd80910dcfa"


def test_extractor_fingerprint_is_pinned():
    """a change of the rendered extractor must be looked at: update this value, and decide whether EXTRACTOR_VERSION goes
    up (an old store's rows would then be refused)."""
    assert I.fingerprint() == FINGERPRINT
    assert I.EXTRACTOR_VERSION == "1"




# --- plan lint, bind ------------------------------------------------------------------------------------------------------------
def test_update_and_judge_plans_lint_and_bind_on_empty_inputs():
    con = duckdb.connect()
    con.execute("CREATE TABLE events (DeviceId VARCHAR, Timestamp TIMESTAMP, EventId INTEGER, Parameter INTEGER)")
    con.execute("CREATE TABLE detectors (DeviceId VARCHAR, Detector BIGINT, phase DOUBLE, function VARCHAR, lanes DOUBLE, configured_function VARCHAR, "
                + ", ".join(f"{c} DOUBLE" for c in H.PROB.values()) + ")")
    I.attach_store(con)
    con.execute(I._render.render("store.sql", device_type="VARCHAR", s=f"{ST}."))
    u = I._update_plan(ST, "events", "detectors", None, "VARCHAR")
    assert u.lint(allow_alias=X.ALIAS) == []
    assert set(u.emits.values()) == set(FEATURES) | set(I.PED_TABLES) | {"health_carry", "health_detector_phase"}
    con.execute(f"CREATE TEMP TABLE {H.PREFIX}wpart AS SELECT 'S1' AS DeviceId, 0::BIGINT AS s_ns, 3600000000000::BIGINT AS e_ns")
    con.execute(f"CREATE TEMP TABLE {H.PREFIX}pwin AS SELECT 'S1' AS DeviceId, 0::BIGINT AS s_ns, 3600000000000::BIGINT AS e_ns, NULL::BIGINT AS wm")
    stmts, drops = u.script([], keep=tuple(u.emits))
    for st, dr in zip(stmts, drops):
        con.execute(st)
        for t in dr:
            con.execute(f"DROP TABLE IF EXISTS {t}")
    ddl = duckdb.connect()
    ddl.execute(I._render.render("store.sql", device_type="VARCHAR"))
    for step, t in u.emits.items():       # what an update writes has the layout of store.sql
        got = [(d[0], str(d[1])) for d in con.execute(f"SELECT * FROM {H.PREFIX}{step} LIMIT 0").description]
        assert got == [(r[0], r[1]) for r in ddl.execute(f"DESCRIBE {t}").fetchall()], t
    con.execute(f"INSERT INTO {ST}.health_signal_5min VALUES ('S1', TIMESTAMP_NS '2026-10-05', 1, TIMESTAMP_NS '2026-10-05', TIMESTAMP_NS '2026-10-05')")
    q, _ = I._judge_plan(con, "detectors", ST, pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06"), H._options(None), True, None)
    assert q.lint(allow_alias=X.ALIAS) == []
    con.execute(f"CREATE OR REPLACE TEMP TABLE {H.PREFIX}wpart AS SELECT DeviceId, s_ns, e_ns FROM {H.PREFIX}wall")
    stmts, drops = q.script([H.FINAL, H.PERIODS_FINAL])
    for st, dr in zip(stmts, drops):
        con.execute(st)
        for t in dr:
            con.execute(f"DROP TABLE IF EXISTS {t}")


def test_the_extractor_reads_no_store_table_but_the_carry():
    u = I._update_plan(ST, "events", "detectors", None, "VARCHAR")
    for n, s, _ in u.ctes:
        for t in FEATURES:
            assert f"{ST}.{t}" not in s, (n, t)
    text = "\n".join(s for _, s, _ in u.ctes)
    assert text.count("@CARRY@") == 1 and text.count("@EVENTS@") == 3    # xev, xpb, health_ped_events: an update binds them to uev


# --- T4.11 --------------------------------------------------------------------------------------------------------------------


