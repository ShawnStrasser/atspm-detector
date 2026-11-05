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
    t = ev.sample(40, random_state=7).Timestamp.sort_values()
    return sorted(set(t)) + [T] if kind == "on" else sorted(set(t + pd.Timedelta(microseconds=1))) + [T]


@pytest.mark.parametrize("kind", few_then_slow(["15min", "on", "after", "60min", "seed1", "5min", "seed2", "seed3"], 3))
def test_chunk_invariance(kind):
    E = edge()
    con, T = E["con"], E["T"]
    st = f"s_{kind}"
    feed(con, _cuts(kind), store=st)
    same_settled(settled(con, st, T, 2), E["ref"], kind)
    same_outputs(I.judge_rows(con, "detectors", store=st, end=T, horizon_days=2), E["out"], kind)
    assert carry_rows(con, st) == carry_rows(con, "one")
    con.execute(f"DETACH {st}")


def test_one_update_equals_batch():
    E = edge()
    con, T = E["con"], E["T"]
    same_outputs(E["out"], H.health(con, "events", "detectors", start=T - 2 * DAY, end=T, periods=True, bin_origin="clock"))
    same_as_batch(con, "one", T, 2)
    # a store keeps the green overlap of every detector with a phase (design decision 6), a batch run of the Count zones
    assert con.execute("SELECT count(DISTINCT Detector) FROM one.health_detector_5min WHERE On_Green5_ns IS NOT NULL").fetchall()[0][0] > 2


def test_batch_identity_three_days():
    """a window ending on an odd quarter-hour: the store's rows of the hour its start falls in begin at that start"""
    ev, det = synth.make(2, 4, seed=11)
    T = ev.Timestamp.min().normalize() + 3 * DAY + pd.Timedelta(hours=8, minutes=15)
    con = connect(ev, det)
    S = T - 3 * DAY
    I.attach_store(con)
    # each update sees only the rows of its chunk, as atspm's runs do
    for c in grid(S, T, "15min"):
        con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?",
                    [c - pd.Timedelta(minutes=15), c])
        I.update(con, "chunk", "detectors")
    got = I.judge_rows(con, "detectors", end=T, horizon_days=3)
    same_outputs(got, H.health(con, "events", "detectors", start=S, end=T, periods=True, bin_origin="clock"))
    assert len(got[0]) and len(got[1])


def test_groups_equal_batch_groups():
    E = edge()
    con, T = E["con"], E["T"]
    a = I.judge_rows(con, "detectors", store="one", end=T, horizon_days=2, max_events=300_000)
    assert a[0].DeviceId.nunique() == 3
    b = H.health(con, "events", "detectors", start=T - 2 * DAY, end=T, periods=True, bin_origin="clock", max_events=300_000)
    same_outputs(a, b, "groups")


def test_compaction_changes_no_total():
    ev, det = synth.make(2, 1, seed=4)
    con = connect(ev, det)
    T = end_of(ev)
    feed(con, grid(ev.Timestamp.min().floor("15min"), T, "60min"))
    before = con.execute(f"SELECT count(*) FROM {ST}.health_detector_5min").fetchall()[0][0]
    ref = settled(con, ST, T, 1)
    out = I.judge_rows(con, "detectors", end=T, horizon_days=1)
    I.trim(con, horizon_days=3, compact=True)
    assert con.execute(f"SELECT count(*) FROM {ST}.health_detector_5min").fetchall()[0][0] < before
    same_settled(settled(con, ST, T, 1), ref)
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), out)


def test_compaction_keeps_out_what_the_same_call_trims():
    ev, det = base()
    con = connect(ev, det)
    T = end_of(ev)
    feed(con, grid(ev.Timestamp.min().floor("15min"), T, "60min"))
    I.trim(con, horizon_days=.25, compact=True)
    cut = pd.Timestamp(I._meta(con, ST)["trimmed_before"])
    assert cut == ev.Timestamp.max().floor("15min") - pd.Timedelta(hours=6)
    for t in FEATURES[:5]:
        mn = con.execute(f"SELECT min(TimeStamp) FROM {ST}.{t}").fetchall()[0][0]
        assert mn >= cut, t
    keep = connect(ev, det)
    feed(keep, [T], horizon_days=.25)
    same_settled(settled(con, ST, T, .25), settled(keep, ST, T, .25))


# --- ownership (design 3.2 rule 2) on chunks ------------------------------------------------------------------------------
def test_each_cut_holds_exactly_the_facts_of_the_events_so_far():
    ev, det = synth.edges(*synth.make(3, 2, seed=5))
    con = connect(ev, det)
    a = ev.Timestamp.min().floor("15min")
    I.attach_store(con)
    cuts = grid(a, end_of(ev), "15min")
    for i, c in enumerate(cuts):
        I.update(con, "events", "detectors", upto=c)
        if i % 37 == 12 or i == len(cuts) - 1:
            # the raw recomputation of test_extract over the events before this cut, on the store's origin
            con.execute(f"CREATE OR REPLACE TABLE wtab AS SELECT DeviceId, 0::BIGINT AS s_ns, {c.value}::BIGINT AS e_ns FROM events GROUP BY DeviceId")
            raw = X._raw(con)
            rows = {t: con.execute(f"SELECT * FROM {ST}.{t}").df() for t in FEATURES}
            X.check_owners(raw, rows, open_labels=False)


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
    feed(con, [T], store="one")
    same_settled(settled(con, ST, T, 1), settled(con, "one", T, 1))
    assert carry_rows(con, ST) == carry_rows(con, "one")
    same_as_batch(con, ST, T, 1)
    got = I.judge_rows(con, "detectors", end=T, horizon_days=1)
    same_outputs(got, I.judge_rows(con, "detectors", store="one", end=T, horizon_days=1))
    same_outputs(got, H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))
    if reach is not None:
        reach(con, stats, T)
    return con


def S00(t, e, p):
    return ("S00", at(t), e, p)


def test_a_stuck_on_across_many_chunks_closing_later_with_relogged_ons():
    t0, t1 = 20000.0, 24000.0
    rows = [S00(t0 - 5, 81, 1), S00(t0, 82, 1), S00(t0 + 500, 82, 1), S00(t0 + 1700.5, 82, 1), S00(t1 - 100, 82, 1), S00(t1, 81, 1)]

    def reach(con, stats, T):
        h = intervals(con, "long_on")
        h = h[(h.DeviceId == "S00") & (h.Channel == 1)]
        assert len(h) == 1 and h.Relogged.iloc[0] and h.StartTime.iloc[0] == at(t0) and h.EndTime.iloc[0] == at(t1)
        lab = con.execute(f"SELECT count(*), sum(On_Time_ns) FROM {ST}.health_detector_5min WHERE DeviceId = 'S00' AND Detector = 1 AND Chain_Tag IS NOT NULL").fetchall()[0]
        assert lab[0] >= 8 and lab[1] > 0
    run_scene(rows, lambda e: dets(e, [1], at(t0 - 10), at(t1 + 10)),
              [t0 - 3, t0, t0 + 0.05, t0 + 100, t0 + 500, t0 + 501, t0 + 1500, t0 + 1700.5, t0 + 1700.6, t0 + 3000, t1 - 100, t1, t1 + 1], reach)


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


def test_b2_stuck_on_lost_when_the_signal_went_silent_before_the_end():
    t0, last = 80000.0, 85800.0

    def reach(con, stats, T):
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        f = x["fch"]
        f = f[(f.dev == w) & (f.det == 1)]
        assert len(f) == 1 and f.gap.iloc[0]                # 86400 - 85800 s > 120 s: lost
        d = x["d5"]
        d = d[(d.dev == w) & (d.det == 1)]
        assert d.unk_ns.sum() > 0
    run_scene([S00(t0 - 5, 81, 1), S00(t0, 82, 1), S00(t0 + 400, 82, 1)],
              lambda e: dets(e, [1], at(t0 - 10), at(90000)) | silence(e, at(last), at(90000)),
              [t0 - 1, t0 + 1, 82000, 84000, 85000, last + 60], reach)


def test_c_chain_over_a_comms_gap_found_in_a_later_chunk():
    t0, g0, g1, t1 = 30000.0, 30400.0, 30900.0, 31300.0

    def reach(con, stats, T):
        b = intervals(con, "bad_chain")
        b = b[(b.DeviceId == "S00") & (b.Channel == 1)]
        assert b.StartTime.tolist() == [at(t0)] and b.EndTime.tolist() == [at(t1)]
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        d = x["d5"]
        d = d[(d.dev == w) & (d.det == 1) & (d.b >= t0 // 300) & (d.b <= t1 // 300)]
        assert d.unk_ns.sum() == (t1 - t0) * 10 ** 9 and d[d.b < t1 // 300].on_ns.isna().all()
    run_scene([S00(t0 - 5, 81, 1), S00(t0, 82, 1), S00(t1, 81, 1)],
              lambda e: dets(e, [1], at(t0 - 10), at(t1 + 10)) | silence(e, at(g0), at(g1)),
              [t0 + 1, t0 + 200, g0 + 50, g1 - 50, g1 + 50, t1 - 1, t1 + 1], reach)


def test_d_off_off_and_a_leading_off_right_after_a_cut():
    t = 40000.0
    seq = [(0.0, 82), (0.5, 81), (0.9, 81), (1.3, 82), (1.4, 81), (1.5, 81), (1.6, 82), (1.7, 81), (2.0, 81), (2.1, 81), (2.3, 82), (2.4, 81)]
    rows = [S00(t - 4, 81, 5)] + [S00(t + x, e, 5) for x, e in seq] + [S00(t + 3, 81, 30), S00(t + 300, 81, 30)]

    def reach(con, stats, T):
        n = con.execute(f"SELECT Detector, sum(Stray_Offs) FROM {ST}.health_detector_5min WHERE DeviceId = 'S00' AND Detector IN (5, 30) AND TimeStamp >= ? GROUP BY 1 ORDER BY 1",
                        [at(t).floor("5min")]).fetchall()
        assert n[1] == (30, 2) and n[0][1] >= 4
    # a cut after every event of the run
    run_scene(rows, lambda e: dets(e, [5], at(t - 5), at(t + 5)), [t + x + 0.05 for x, _ in seq] + [t + 3.05], reach)


def test_e_too_fast_pair_straddling_a_cut_and_the_last_on_pending():
    t, last = 50000.0 + 0.3, 86390.0
    fast = [(0.0, 82), (0.1, 81), (0.3, 82), (0.4, 81), (0.6, 82), (0.7, 81), (1.2, 82), (1.3, 81)]
    rows = [S00(t - 2, 81, 2), S00(last - 2, 81, 2)] + [S00(t + x, e, 2) for x, e in fast] + [S00(last, 82, 2), S00(last + 0.1, 81, 2), S00(last + 0.3, 82, 2), S00(last + 0.4, 81, 2)]

    def reach(con, stats, T):
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        o = x["oxt"]
        assert ((o.dev == w) & (o.det == 2) & (o.ta == 3)).any()
        p = con.execute(f"SELECT sum(Count) FROM {ST}.health_tick_counts WHERE DeviceId = 'S00' AND Detector = 2 AND Kind = 'gap_pair' AND A = 3 AND B = 3").fetchall()[0][0]
        assert p >= 1
    drop = lambda e: dets(e, [2], at(t - 3), at(t + 3)) | dets(e, [2], at(last - 3), at(86400))
    run_scene(rows, drop, [t + 0.05, t + 0.2, t + 0.35, t + 0.5, t + 0.9, last + 0.2, last + 0.35], reach)


@pytest.mark.parametrize("gap_at,valid", [(5.0, True), (1.0, False)], ids=["gap after 2 s", "gap inside 2 s"])
def test_f_start_in_the_first_2_s_of_red_across_a_cut(gap_at, valid):
    # phase 2: yellow, red at t, then fast and chatter starts on det 1 in the first 2 s, cuts among them, then a comms gap
    # starting inside or after those 2 s, then green again
    t, g1 = 60000.0, 60000.0 + gap_at + 300
    rows = [S00(t - 4, 8, 2), S00(t, 9, 2), S00(t - 5, 81, 1)]
    rows += [S00(t + x, e, 1) for x, e in ((0.1, 82), (0.2, 81), (0.7, 82), (0.8, 81), (0.9, 82), (1.0, 81))]
    rows += [S00(t + gap_at, 131, 7), S00(g1, 131, 1), S00(g1 + 1, 1, 2)]
    drop = lambda e: silence(e, at(t - 4), at(g1 + 1)) | dets(e, [1], at(t - 100), at(g1 + 1)) | colour(e, 2, at(g1 + 1), at(g1 + 20))

    def reach(con, stats, T):
        lab = con.execute(f"SELECT sum(Fast_Starts_R2), sum(Chatter_Starts_R2) FROM {ST}.health_detector_5min WHERE DeviceId = 'S00' AND Detector = 1 AND Piece_Kind = 2").fetchall()[0]
        assert lab[0] >= 1 and lab[1] >= 1
        b = intervals(con, "bad_piece")
        b = b[(b.DeviceId == "S00") & (b.Channel == 2) & (b.Value == 2) & (b.StartTime == at(t))]
        assert len(b) == (0 if valid else 1)
    run_scene(rows, drop, [t + 0.15, t + 0.75, t + 0.85, t + 0.95, t + 1.5, t + 50, g1 + 0.5, g1 + 5], reach)


def test_f2_red_ending_inside_its_first_2_s_with_the_cut_inside():
    t = 62000.0
    rows = [S00(t - 2, 81, 1), S00(t - 4, 8, 2), S00(t, 9, 2), S00(t + 0.3, 82, 1), S00(t + 0.4, 81, 1), S00(t + 1.0, 1, 2)]
    drop = lambda e: colour(e, 2, at(t - 4), at(t + 20)) | dets(e, [1], at(t - 3), at(t + 3))
    run_scene(rows, drop, [t + 0.35, t + 0.5, t + 1.5])


def test_g_phase_resting_in_green_for_hours_then_a_gap_inside_the_open_piece():
    t, g0, g1, t1 = 20000.0, 30000.0, 30300.0, 33000.0
    rows = [S00(t, 1, 2), S00(t1, 8, 2), S00(t1 + 4, 9, 2)]
    drop = lambda e: colour(e, 2, at(t - 30), at(t1 + 20)) | silence(e, at(g0), at(g1))

    def reach(con, stats, T):
        b = intervals(con, "bad_piece")
        b = b[(b.DeviceId == "S00") & (b.Channel == 2) & (b.Value == 0) & (b.StartTime == at(t))]
        assert len(b) == 1 and b.EndTime.iloc[0] == at(t1)
        lab = con.execute(f"SELECT sum(Starts_GY), sum(On_Green5_ns) FROM {ST}.health_detector_5min WHERE DeviceId = 'S00' AND Piece_Phase = 2 AND Piece_Tag = ?", [at(t)]).fetchall()[0]
        assert lab[0] > 0 and lab[1] > 0
        tm = con.execute(f"SELECT sum(Green_Yellow_ns), sum(Green5_ns) FROM {ST}.health_phase_5min WHERE DeviceId = 'S00' AND Phase = 2 AND Piece_Tag = ?", [at(t)]).fetchall()[0]
        assert tm[0] > 0 and tm[1] > 0
    run_scene(rows, drop, [t + 1, t + 3, t + 6, 23000, 27000, g0 - 10, g0 + 100, g1 + 100, 32000, t1 + 1, t1 + 5], reach)


def test_g2_resting_green_with_no_gap_is_time_and_starts():
    t, t1 = 20000.0, 33000.0
    rows = [S00(t, 1, 2), S00(t1, 8, 2), S00(t1 + 4, 9, 2)]
    drop = lambda e: colour(e, 2, at(t - 30), at(t1 + 20))

    def reach(con, stats, T):
        assert not len(intervals(con, "bad_piece"))
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        p = x["p5"]
        p = p[(p.dev == w) & (p.ph == 2) & (p.b >= t // 300) & (p.b <= t1 // 300)]
        assert p.gy_ns.sum() >= (t1 + 4 - t) * 10 ** 9
    run_scene(rows, drop, [t + 1, t + 3, t + 6, 23000, 27000, 31000, 32000, t1 + 1, t1 + 5], reach)


def test_g3_resting_green_still_open_at_the_end_with_a_gap_inside():
    t, g0, g1 = 70000.0, 80000.0, 80300.0
    rows = [S00(t, 1, 2)]
    drop = lambda e: colour(e, 2, at(t - 30), at(90000)) | silence(e, at(g0), at(g1))

    def reach(con, stats, T):
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        f = x["fpc"]
        assert f[(f.dev == w) & (f.ph == 2)].valid.tolist() == [False]
        p = x["p5"]
        assert p[(p.dev == w) & (p.ph == 2) & (p.b > t // 300)].gy_ns.isna().all()
    run_scene(rows, drop, [t + 1, t + 6, 75000, g0 - 10, g0 + 100, g1 + 100, 82000, 85000], reach)


def test_g4_piece_starting_on_the_cut_with_a_comms_gap_right_after_it():
    """a green opens on the signal's last event before a cut, with a start at that instant, and a comms gap starts right there:
    the piece closes bad in a later chunk, and the start labelled in the earlier chunk is settled as bad"""
    t, g1 = 60000.0, 60300.0
    rows = [S00(t - 30, 9, 2), S00(t - 3, 81, 1), S00(t, 1, 2), S00(t, 82, 1), S00(t, 81, 1), S00(g1, 8, 2), S00(g1 + 4, 9, 2)]
    drop = lambda e: silence(e, at(t), at(g1)) | colour(e, 2, at(t - 60), at(g1 + 20)) | dets(e, [1], at(t - 10), at(t + 10))

    def reach(con, stats, T):
        l0 = con.execute(f"SELECT sum(Starts_GY) FROM {ST}.health_detector_5min WHERE DeviceId = 'S00' AND Detector = 1 AND Piece_Tag = ?", [at(t)]).fetchall()[0][0]
        assert l0 == 1
        b = intervals(con, "bad_piece")
        assert len(b[(b.DeviceId == "S00") & (b.Channel == 2) & (b.StartTime == at(t))]) == 1
    run_scene(rows, drop, [t + 0.05, t + 100, g1 + 0.5, g1 + 5], reach)


def test_g5_open_piece_starting_at_a_comms_gap_is_invalid_at_the_end():
    t, g1 = 70000.0, 70300.0
    rows = [S00(t, 1, 2)]
    drop = lambda e: silence(e, at(t), at(g1)) | colour(e, 2, at(t - 60), at(90000))

    def reach(con, stats, T):
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        f = x["fpc"]
        assert f[(f.dev == w) & (f.ph == 2)].valid.tolist() == [False]
    run_scene(rows, drop, [t + 0.05, t + 100, g1 + 1, 80000], reach)


@pytest.mark.parametrize("gap_at,valid", [(5.0, True), (1.0, False)], ids=["gap after 2 s", "gap inside 2 s"])
def test_f3_red_still_open_at_the_end(gap_at, valid):
    t, g1 = 60000.0, 60000.0 + gap_at + 300
    rows = [S00(t - 4, 8, 2), S00(t, 9, 2), S00(t - 5, 81, 1)]
    rows += [S00(t + x, e, 1) for x, e in ((0.1, 82), (0.2, 81), (0.7, 82), (0.8, 81), (0.9, 82), (1.0, 81))]
    rows += [S00(t + gap_at, 131, 7), S00(g1, 131, 1)]
    drop = lambda e: silence(e, at(t - 4), at(g1 + 1)) | dets(e, [1], at(t - 100), at(g1 + 1)) | colour(e, 2, at(t - 4), at(90000))

    def reach(con, stats, T):
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        f = x["fpc"]
        assert f[(f.dev == w) & (f.ph == 2)].valid.tolist() == [valid]
    run_scene(rows, drop, [t + 0.15, t + 0.75, t + 0.85, t + 1.5, t + 50, g1 + 0.5, g1 + 500], reach)


def test_h_signal_silent_for_several_chunks():
    g0, g1 = 41000.0, 41000.0 + 2500
    cuts = list(np.arange(40500.0, 45000.0, 900.0))

    def reach(con, stats, T):
        b = intervals(con, "comms_gap")
        b = b[b.DeviceId == "S00"]
        assert len(b) == 1 and (b.EndTime.iloc[0] - b.StartTime.iloc[0]).total_seconds() > 2400
    run_scene([], lambda e: silence(e, at(g0), at(g1)), cuts, reach)


def test_i_signal_silent_at_the_end_the_lost_test_and_the_30_minute_end_rule():
    last = 80000.0

    def reach(con, stats, T):
        w = settled(con, ST, T, 1)["win"].set_index("DeviceId")
        assert w.e_ns["S00"] < T.value - 1800 * 10 ** 9 and w.e_ns["S01"] == T.value
    run_scene([S00(last - 503, 81, 1), S00(last - 500, 82, 1)], lambda e: silence(e, at(last), at(90000)) | dets(e, [1], at(last - 505), at(90000)),
              [last - 600, last - 400, last + 300, 82000, 83000], reach)


def test_j_plan_change_at_a_cut_beside_a_silence():
    g0, g1 = 50000.0, 50400.0
    rows = [S00(g0 - 1, 131, 4), S00(g1, 131, 5), S00(g1 + 900, 131, 6)]

    def reach(con, stats, T):
        p = intervals(con, "plan")
        p = p[p.DeviceId == "S00"]
        assert {at(g0 - 1), at(g1), at(g1 + 900)} <= set(p.StartTime)
        assert len(intervals(con, "comms_gap")) >= 1
    run_scene(rows, lambda e: silence(e, at(g0), at(g1)), [g0 - 1, g0 - 0.999999, g1, g1 + 0.000001, g1 + 900], reach)


def test_k_phase_with_fewer_than_3_greens():
    t = 30000.0
    rows = [S00(t, 1, 16), S00(t + 20, 8, 16), S00(t + 24, 9, 16), S00(t + 3600, 1, 16), S00(t + 3620, 8, 16)]

    def reach(con, stats, T):
        x = settled(con, ST, T, 1)
        p = x["p5"]
        w = x["win"].set_index("DeviceId").dev["S00"]
        assert p[(p.dev == w) & (p.ph == 16)].ngreen.sum() == 2
    run_scene(rows, None, [t + 1, t + 21, t + 25, t + 3601, t + 3621], reach)


def test_l_signal_whose_data_starts_late_starts_on_its_quarter_hour():
    a = 25000.0

    def reach(con, stats, T):
        w = settled(con, ST, T, 1)["win"].set_index("DeviceId")
        assert w.s_ns["S00"] == at(a).floor("15min").value and w.s_ns["S01"] == T.value - DAY.value
    run_scene([], lambda e: silence(e, at(0), at(a)), [a - 100, a + 1, a + 900, a + 1800], reach)


def test_m_phase_change_between_updates_counts_from_that_update_and_is_logged():
    ev, det = base()
    T = end_of(ev)
    a = ev.Timestamp.min().normalize()
    con = connect(ev, det)
    I.attach_store(con)
    mid = a + pd.Timedelta(hours=12)
    s0 = I.update(con, "events", "detectors", upto=mid)
    assert s0.phase_changes == 0 and s0.rows_written["health_detector_phase"] == len(det)
    con.execute("CREATE TABLE det2 AS SELECT * FROM detectors")
    con.execute("UPDATE det2 SET phase = 4 WHERE DeviceId = 'S00' AND Detector = 1")
    s1 = I.update(con, "events", "det2")
    assert s1.phase_changes == 1
    log = con.execute(f"SELECT Phase, ValidFrom FROM {ST}.health_detector_phase WHERE DeviceId = 'S00' AND Detector = 1 ORDER BY ValidFrom").df()
    assert log.Phase.tolist() == [2, 4] and log.ValidFrom.iloc[1] >= mid
    # the store's start colours: phase 2's until the change, phase 4's after (documented result; rebuild for exactness)
    I.attach_store(con, name="old")
    I.update(con, "events", "detectors", store="old")
    I.attach_store(con, name="new")
    I.update(con, "events", "det2", store="new")
    def gy(st, cond):
        return con.execute(f"SELECT coalesce(sum(Starts_GY), 0) FROM {st}.health_detector_5min WHERE DeviceId = 'S00' AND Detector = 1 AND TimeStamp {cond} ?", [mid.floor('5min')]).fetchall()[0][0]
    assert gy(ST, "<") == gy("old", "<") and gy(ST, ">") == gy("new", ">") and gy("old", ">") != gy("new", ">")


# the rare branches of test_extract, in chunks: random cuts and a cut after each of the last events
@pytest.mark.parametrize("name", X.SCENES)
def test_rare_branches_in_chunks(name):
    con, _ = X.SCENES[name]()
    ev = con.execute("SELECT Timestamp FROM events WHERE DeviceId = 'S00' ORDER BY Timestamp").df().Timestamp
    T = ev.max().floor("15min") + pd.Timedelta(minutes=15)
    a = ev.min().floor("15min")
    span = (T - a)
    hd = span / DAY
    near = list(ev.iloc[-16:] + pd.Timedelta(microseconds=1))
    # the rows these scenes add are mostly in the middle of the log: a dense stretch of cuts over every third hour too
    cuts = sorted(set(random_cuts(a, T, 24, 5) + near + [T]))
    feed(con, cuts)
    feed(con, [T], store="one")
    same_settled(settled(con, ST, T, hd), settled(con, "one", T, hd), name)
    same_as_batch(con, ST, T, hd, name)
    assert carry_rows(con, ST) == carry_rows(con, "one")
    got = I.judge_rows(con, "detectors", end=T, horizon_days=hd)
    same_outputs(got, H.health(con, "events", "detectors", start=a, end=T, periods=True, bin_origin="clock"), name)


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


@pytest.mark.parametrize("seed", few_then_slow(range(8), 2))
def test_random_logs_in_random_chunks(seed):
    ev, det = fuzz_log(seed)
    con = connect(ev, det)
    a, T = ev.Timestamp.min().floor("15min"), pd.Timestamp("2026-10-05 06:00")
    cuts = random_cuts(a, T, 50, seed + 100)
    feed(con, cuts)
    feed(con, [T], store="one")
    same_settled(settled(con, ST, T, .25), settled(con, "one", T, .25), f"seed {seed}")
    same_as_batch(con, ST, T, .25, f"seed {seed}")
    assert carry_rows(con, ST) == carry_rows(con, "one")
    # both kinds of exception row and open objects do occur in these logs, and the carry stays small
    n = {k: len(intervals(con, k)) for k in ("bad_chain", "bad_piece", "long_on", "comms_gap")}
    assert n["bad_chain"] + n["bad_piece"] > 0 and n["comms_gap"] > 0, n
    per_det = con.execute(f"SELECT max(c) FROM (SELECT count(*) AS c FROM {ST}.health_carry WHERE EventId IN (9201, 9202, 9221, 9222) GROUP BY DeviceId, Parameter)").fetchall()[0][0]
    assert per_det <= 4
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=.25),
                 H.health(con, "events", "detectors", start=a, end=T, periods=True, bin_origin="clock"), f"seed {seed}")


# --- T4.4 ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", few_then_slow([1, 2, 3], 1))
def test_carry_closure(seed):
    ev, det = fuzz_log(seed + 50)
    con = connect(ev, det)
    rng = np.random.default_rng(seed)
    a, T = ev.Timestamp.min(), pd.Timestamp("2026-10-05 06:00")
    feed(con, [T], store="whole")
    for k in range(6):
        cut = a + pd.Timedelta(seconds=float(rng.uniform(100, 20000)))
        st = f"split{k}"
        feed(con, [cut, T], store=st)
        assert carry_rows(con, st) == carry_rows(con, "whole"), cut
        con.execute(f"DETACH {st}")


# --- T4.5 ---------------------------------------------------------------------------------------------------------------------
def test_bounded_state():
    """14 days: detectors stuck ON the whole time, phases resting: the carry stays the same size, labelled rows stay flat."""
    ev, det = synth.make(1, 14, seed=2, faults=False)
    a = ev.Timestamp.min().normalize()
    # stuck: det 1 ON from the first minute, re-logged every hour; phase 3 rests from the second hour on
    keep = ~(dets(ev, [1], a, a + 15 * DAY) | colour(ev, 3, a + pd.Timedelta(hours=2), a + 15 * DAY))
    rel = [("S00", a + pd.Timedelta(minutes=1, hours=h), 82, 1) for h in range(14 * 24)]
    add = pd.DataFrame(rel, columns=ev.columns).astype({"EventId": "int32", "Parameter": "int32"})
    ev = pd.concat([ev[keep], add], ignore_index=True)
    con = connect(ev, det)
    I.attach_store(con)
    hist = []
    for c in grid(a, a + 14 * DAY, "60min"):
        con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?", [c - pd.Timedelta(minutes=60), c])
        s = I.update(con, "chunk", "detectors", horizon_days=1)
        lab = con.execute(f"SELECT count(*) FROM {ST}.health_detector_5min WHERE Chain_Tag IS NOT NULL OR Piece_Tag IS NOT NULL").fetchall()[0][0]
        lab += con.execute(f"SELECT count(*) FROM {ST}.health_phase_5min WHERE Piece_Tag IS NOT NULL").fetchall()[0][0]
        rows = {t: con.execute(f"SELECT count(*) FROM {ST}.{t}").fetchall()[0][0] for t in FEATURES}
        hist.append((s.carry_rows, lab, sum(rows.values())))
    h = np.array(hist)
    # from day 3 on (the horizon full)
    carry, lab, total = h[48:, 0], h[48:, 1], h[48:, 2]
    assert carry.max() - carry.min() <= 8 and carry.max() < 200, (carry.min(), carry.max())
    assert lab.max() <= 1.25 * lab.min() and lab.min() > 0, (lab.min(), lab.max())
    assert total.max() <= 1.25 * total.min(), (total.min(), total.max())


# --- T4.6 ---------------------------------------------------------------------------------------------------------------------
def test_retention():
    ev, det = synth.make(2, 3, seed=6)
    a = ev.Timestamp.min().normalize()
    # a chain open from day 0 to the end, re-logged on day 0
    keep = ~dets(ev, [1], a + pd.Timedelta(hours=2), a + 4 * DAY)
    add = pd.DataFrame([("S00", a + pd.Timedelta(hours=2), 82, 1), ("S00", a + pd.Timedelta(hours=3), 82, 1), ("S00", a + pd.Timedelta(hours=4), 82, 1)], columns=ev.columns).astype({"EventId": "int32", "Parameter": "int32"})
    ev = pd.concat([ev[keep], add], ignore_index=True)
    con = connect(ev, det)
    T = end_of(ev)
    feed(con, grid(a, T, "60min"), horizon_days=1)
    feed(con, [T], store="one", horizon_days=1)
    cut = pd.Timestamp(I._meta(con, ST)["trimmed_before"])
    assert cut == ev.Timestamp.max().floor("15min") - DAY
    for t in ("health_signal_5min", "health_detector_5min", "health_phase_5min", "health_phase_15min", "health_tick_counts"):
        assert con.execute(f"SELECT min(TimeStamp) FROM {ST}.{t}").fetchall()[0][0] >= cut
    assert con.execute(f"SELECT min(EndTime) FROM {ST}.health_intervals").fetchall()[0][0] >= cut
    # the chain from day 0 survives, with its true start and hours held
    assert con.execute(f"SELECT count(*) FROM {ST}.health_carry WHERE DeviceId = 'S00' AND Parameter = 1 AND EventId = 9222").fetchall()[0][0] == 1
    x = settled(con, ST, T, 1)
    w = x["win"].set_index("DeviceId")
    f = x["fch"]
    f = f[(f.dev == w.dev["S00"]) & (f.det == 1)]
    assert len(f) == 1 and f.s.iloc[0] == at_from(a, 2 * 3600) - w.s_ns["S00"] and f.dur.iloc[0] > 2.5 * 86400
    assert x["d5"].b.min() >= 0 and x["fch"].s.min() < 0        # no key before the window, though the chain began before it
    same_settled(x, settled(con, "one", T, 1))
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), I.judge_rows(con, "detectors", store="one", end=T, horizon_days=1))
    # before the trim: refused
    with pytest.raises(ValueError, match="before the store's trim"):
        I.judge_rows(con, "detectors", end=T, horizon_days=2)
    # a pruned detector's history: gone once all of it is older than the cut, unless a chain is open
    assert con.execute(f"SELECT count(*) FROM {ST}.health_carry WHERE EventId IN (9201, 9202, 9221, 9222) AND TimeStamp < ? AND Parameter <> 1", [cut - pd.Timedelta(minutes=2)]).fetchall()[0][0] == 0


def at_from(a, sec):
    return (a + pd.Timedelta(seconds=sec)).value


def test_pruned_history_changes_nothing_at_or_after_the_cut():
    """a detector silent for longer than the horizon: its carried history is dropped; its next event, after the cut, gives the
    same rows as with the history kept."""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    con = connect(ev[(ev.Timestamp < a + pd.Timedelta(hours=8)) | ~dets(ev, [2], a, a + DAY)], det)
    T = end_of(ev)
    feed(con, grid(a, T, "60min"), horizon_days=.25)
    feed(con, [T], store="keep", horizon_days=1)
    n = con.execute(f"SELECT count(*) FROM {ST}.health_carry WHERE DeviceId = 'S00' AND Parameter = 2 AND EventId < 9300").fetchall()[0][0]
    assert n == 0 or con.execute(f"SELECT max(TimeStamp) FROM {ST}.health_carry WHERE DeviceId = 'S00' AND Parameter = 2 AND EventId < 9300").fetchall()[0][0] >= T - DAY
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=.25), I.judge_rows(con, "detectors", store="keep", end=T, horizon_days=.25))


def test_trim_keeps_the_history_a_lagging_signal_may_still_need():
    """S00 stops logging just before the cut, with a detector's ON and OFF 0.3 s and 0.2 s before it; S01 goes on until the
    trim's cut reaches that time; S00's backlog goes on 0.1 s after the cut.  The too-fast pair and the start's neighbours
    across the cut come out as in a store that trims nothing."""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    C = a + pd.Timedelta(hours=10)
    sec = lambda x: pd.Timedelta(seconds=x)
    keep = ~dets(ev, [1], C - sec(40), C + sec(40))
    add = pd.DataFrame([("S00", C - sec(30), 81, 1), ("S00", C - sec(.3), 82, 1), ("S00", C - sec(.2), 81, 1), ("S00", C + sec(.1), 82, 1), ("S00", C + sec(.2), 81, 1)],
                       columns=ev.columns).astype({"EventId": "int32", "Parameter": "int32"})
    ev = pd.concat([ev[keep], add], ignore_index=True)
    con = connect(ev, det)
    con.execute("CREATE TABLE c1 AS SELECT * FROM events WHERE Timestamp < ?", [C])
    con.execute("CREATE TABLE c2 AS SELECT * FROM events WHERE DeviceId = 'S01' AND Timestamp >= ? AND Timestamp < ?", [C, C + pd.Timedelta(hours=6, minutes=1)])
    con.execute("CREATE TABLE c3 AS SELECT * FROM events WHERE (DeviceId = 'S00' AND Timestamp >= ? OR Timestamp >= ?) AND Timestamp < ?", [C, C + pd.Timedelta(hours=6, minutes=1), C + pd.Timedelta(hours=6, minutes=5)])
    I.attach_store(con, name="trims")
    I.attach_store(con, name="keeps")
    for c in ("c1", "c2", "c3"):
        I.update(con, c, "detectors", store="trims", horizon_days=.25)
        I.update(con, c, "detectors", store="keeps", horizon_days=2)
    assert I._meta(con, "trims")["trimmed_before"] == C.isoformat()
    n = con.execute(f"SELECT count(*) FROM trims.health_tick_counts WHERE DeviceId = 'S00' AND Detector = 1 AND Kind = 'gap_pair' AND TimeStamp = ? AND A = 4", [C]).fetchall()[0][0]
    assert n == 1
    for t in ("health_detector_5min", "health_tick_counts", "health_signal_5min", "health_phase_5min"):
        q = f"SELECT * FROM {{}}.{t} WHERE TimeStamp >= ? ORDER BY ALL"
        assert con.execute(q.format("trims"), [C]).fetchall() == con.execute(q.format("keeps"), [C]).fetchall(), t


def add_rows(ev, rows):
    return pd.concat([ev, pd.DataFrame(rows, columns=ev.columns).astype({"EventId": "int32", "Parameter": "int32"})], ignore_index=True)


def test_what_straddles_the_cut_survives_the_trim():
    """the trim's cut C falls inside a comms gap, a hold, a chain over that gap and a colour piece over it; the last three reach
    into the judge's window, which starts 15 minutes after C.  A hold ending exactly at C stays, one ending just before it goes."""
    ev, det = base()
    C = ev.Timestamp.max().floor("15min") - pd.Timedelta(hours=6)
    S = C + pd.Timedelta(minutes=15)
    sec = lambda x: pd.Timedelta(seconds=x)
    T = end_of(ev)
    assert T - S == pd.Timedelta(hours=6)
    keep = ~(silence(ev, C - sec(100), C + sec(100))
             | dets(ev, [2], C - sec(310), S + sec(310)) | colour(ev, 2, C - sec(400), S + sec(320))
             | dets(ev, [1, 2, 9], C - sec(410), S + sec(210), dev="S01"))
    ev2 = add_rows(ev[keep], [("S00", C - sec(300), 82, 2), ("S00", S + sec(300), 81, 2),
                              ("S00", C - sec(300), 1, 2), ("S00", S + sec(300), 8, 2), ("S00", S + sec(304), 9, 2),
                              ("S01", C - sec(200), 82, 1), ("S01", S + sec(200), 81, 1),
                              ("S01", C - sec(400), 82, 2), ("S01", C, 81, 2),
                              ("S01", C - sec(400), 82, 9), ("S01", C - sec(.1), 81, 9)])
    con = connect(ev2, det)
    a = ev.Timestamp.min().normalize()
    cuts = sorted(set(grid(a, T, "60min")) | {C - sec(x) for x in (350, 250, 50)} | {C + sec(150), C + sec(500), S, S + sec(100), S + sec(250), S + sec(350)})
    feed(con, cuts, store="trims", horizon_days=.25)
    feed(con, cuts, store="keeps", horizon_days=2)
    assert pd.Timestamp(I._meta(con, "trims")["trimmed_before"]) == C
    hd = .25
    same_settled(settled(con, "trims", T, hd), settled(con, "keeps", T, hd), "straddling")
    same_outputs(I.judge_rows(con, "detectors", store="trims", end=T, horizon_days=hd), I.judge_rows(con, "detectors", store="keeps", end=T, horizon_days=hd))
    q = "SELECT Kind, DeviceId, Channel FROM trims.health_intervals WHERE StartTime < ? AND EndTime >= ? AND Kind <> 'plan' ORDER BY ALL"
    got = con.execute(q, [C, C]).fetchall()
    assert ("long_on", "S01", 1) in got and ("long_on", "S01", 2) in got and ("bad_chain", "S00", 2) in got and ("bad_piece", "S00", 2) in got
    assert any(k == "comms_gap" and d == "S00" for k, d, _ in got)
    assert ("long_on", "S01", 9) not in got and con.execute("SELECT count(*) FROM keeps.health_intervals WHERE Kind = 'long_on' AND DeviceId = 'S01' AND Channel = 9 AND EndTime = ?", [C - sec(.1)]).fetchall()[0][0] == 1
    assert not con.execute("SELECT 1 FROM trims.health_intervals WHERE EndTime < ?", [C]).fetchall()
    for t in FEATURES[:5]:
        assert con.execute(f"SELECT min(TimeStamp) FROM trims.{t}").fetchall()[0][0] == C, t


def test_a_pruned_detector_leaves_the_phase_of_its_number_alone():
    """detector 5 goes silent long before the cut and its history is pruned; phase 5 rests in a green all that time (carry rows
    with the same number): the piece stays, and starts on phase 5's detector 8 still count in that green"""
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    t0 = a + pd.Timedelta(hours=6)
    ev2 = add_rows(ev[~(dets(ev, [5], t0, T) | colour(ev, 5, t0 - pd.Timedelta(seconds=60), T))], [("S00", t0, 1, 5)])
    con = connect(ev2, det)
    cuts = grid(a, T, "60min")
    feed(con, cuts, store="trims", horizon_days=.25)
    feed(con, cuts, store="keeps", horizon_days=2)
    q = "SELECT count(*) FROM {}.health_carry WHERE DeviceId = 'S00' AND Parameter = 5 AND EventId {}"
    assert con.execute(q.format("trims", "IN (9201, 9202, 9221, 9222)")).fetchall()[0][0] == 0
    assert con.execute(q.format("keeps", "IN (9201, 9202, 9221, 9222)")).fetchall()[0][0] > 0
    assert con.execute(q.format("trims", "= 9300")).fetchall()[0][0] == 1
    same_settled(settled(con, "trims", T, .25), settled(con, "keeps", T, .25), "pruned")
    same_outputs(I.judge_rows(con, "detectors", store="trims", end=T, horizon_days=.25), I.judge_rows(con, "detectors", store="keeps", end=T, horizon_days=.25))


def test_trim_keeps_each_detector_s_current_phase():
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    con = connect(ev, det)
    con.execute("CREATE TABLE det2 AS SELECT * FROM detectors")
    con.execute("UPDATE det2 SET phase = 4 WHERE DeviceId = 'S00' AND Detector = 1")
    I.attach_store(con)
    mid = a + pd.Timedelta(hours=8)
    log = lambda: con.execute(f"SELECT Phase, ValidFrom FROM {ST}.health_detector_phase WHERE DeviceId = 'S00' AND Detector = 1 ORDER BY ValidFrom").fetchall()
    first = I.update(con, "events", "detectors", upto=mid, horizon_days=.5)
    v0 = log()[0][1]
    assert first.rows_written["health_detector_phase"] == len(det)
    I.update(con, "events", "det2", upto=mid + pd.Timedelta(hours=1), horizon_days=.5)
    assert [p for p, _ in log()] == [2, 4]                  # the change is logged, the old row still inside the horizon
    v1 = log()[1][1]
    for c in grid(mid + pd.Timedelta(hours=1), T, "60min"):
        s = I.update(con, "events", "det2", upto=c, horizon_days=.25)
        assert s.phase_changes == 0 and s.rows_written["health_detector_phase"] == 0    # no current phase went missing
    assert log() == [(4, v1)] and v0 < pd.Timestamp(I._meta(con, ST)["trimmed_before"])
    assert con.execute(f"SELECT count(*) FROM {ST}.health_detector_phase").fetchall()[0][0] == len(det)


@pytest.mark.parametrize("seed", few_then_slow([20, 21, 22], 1))
def test_random_logs_trimmed_on_the_way(seed):
    """random logs, random cuts, a short horizon and compactions on the way, against a store that trims nothing, at the horizon
    and at a shorter window"""
    ev, det = fuzz_log(seed)
    con = connect(ev, det)
    rng = np.random.default_rng(seed)
    a, T = ev.Timestamp.min().floor("15min"), pd.Timestamp("2026-10-05 06:00")
    I.attach_store(con, name="trims")
    for c in random_cuts(a, T, 50, seed + 7):
        I.update(con, "events", "detectors", store="trims", upto=c, horizon_days=.0625)
        if rng.random() < .1:
            I.trim(con, store="trims", horizon_days=.0625, compact=True)
    feed(con, [T], store="keeps")
    for hd in (.0625, .03125):
        same_settled(settled(con, "trims", T, hd), settled(con, "keeps", T, hd), f"seed {seed} {hd}")
        same_outputs(I.judge_rows(con, "detectors", store="trims", end=T, horizon_days=hd), I.judge_rows(con, "detectors", store="keeps", end=T, horizon_days=hd))


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


def test_a_backlog_from_a_silent_signal_is_accepted():
    ev, det = base()
    con = connect(ev, det)
    a = ev.Timestamp.min().normalize()
    T = end_of(ev)
    I.attach_store(con)
    gap_a, gap_b = a + pd.Timedelta(hours=6), a + pd.Timedelta(hours=14)
    # S01 reports nothing from 06:00 to 14:00; S00 goes on; S01's backlog arrives in the 14:00 update
    for c in grid(a, T, "60min"):
        lo = c - pd.Timedelta(minutes=60)
        if lo < gap_a or c > gap_b:
            lo2 = gap_a if c == gap_b + pd.Timedelta(minutes=60) else lo
            con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE Timestamp < ? AND ((DeviceId = 'S00' AND Timestamp >= ?) OR (DeviceId = 'S01' AND Timestamp >= ?))", [c, lo, lo2])
        else:
            con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE DeviceId = 'S00' AND Timestamp >= ? AND Timestamp < ?", [lo, c])
        s = I.update(con, "chunk", "detectors")
        assert not s.events_late
    w = con.execute(f"SELECT DeviceId, epoch_ns(TimeStamp) FROM {ST}.health_carry WHERE EventId = 9402 ORDER BY 1").fetchall()
    assert len(w) == 2
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1),
                 H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))


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


def test_integer_device_ids_and_a_later_input_of_another_type():
    ev, det = base()
    ev = ev.assign(DeviceId=ev.DeviceId.str[1:].astype(int) + 100)
    det = det.assign(DeviceId=det.DeviceId.str[1:].astype(int) + 100)
    con = connect(ev, det)
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    mid = a + pd.Timedelta(hours=12)
    con.execute("CREATE TABLE c1 AS SELECT * FROM events WHERE Timestamp < ?", [mid])
    con.execute("CREATE TABLE c2 AS SELECT CAST(DeviceId AS VARCHAR) AS DeviceId, Timestamp, EventId, Parameter FROM events WHERE Timestamp >= ?", [mid])
    I.attach_store(con)
    I.update(con, "c1", "detectors")
    I.update(con, "c2", "detectors")
    assert I._meta(con, ST)["device_type"] in ("BIGINT", "INTEGER")
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1),
                 H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))


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


@pytest.mark.parametrize("stamp,code,par", [("2000-01-01", 82, 1), ("2015-01-01", 82, 1), ("1970-01-01", 82, 1), ("1701-06-01", 82, 1),
                                            ("2000-01-01", 81, 1), ("2000-01-01", 1, 2), ("2000-01-01", 131, 2)],
                         ids=["ON 2000", "ON 2015", "ON 1970", "ON 1701", "OFF", "green", "plan"])
def test_a_wrong_old_clock_alone_in_the_first_chunk_blocks_nothing(stamp, code, par):
    """the store has no other time to doubt it by, so the row is taken; the next update knows better: that signal starts over.
    In the end nothing differs from a store that never saw the row."""
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    con = connect(ev, det)
    bad_clock_table(con, "c0", [("S00", pd.Timestamp(stamp), code, par)])
    I.attach_store(con)
    s = I.update(con, "c0", "detectors", upto=T)
    assert s.events_in == 1 and s.through == pd.Timestamp(stamp)
    cuts = grid(a, T, "60min")
    for c in cuts:
        s = I.update(con, "events", "detectors", upto=c)
        assert s.signals == 2 and not s.events_bad_stamp
    feed(con, cuts, store="clean")
    assert feature_rows(con, ST) == feature_rows(con, "clean")
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))


@pytest.mark.parametrize("stamp", ["2000-01-01", "1970-01-01", "1701-06-01"])
def test_a_wrong_old_clock_on_a_signal_never_heard_from_again(stamp):
    """the row stays in that signal's carry, an open chain decades long: no other signal's update reads it and no judge does"""
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    con = connect(ev[ev.DeviceId == "S01"], det)
    bad_clock_table(con, "c0", [("S00", pd.Timestamp(stamp), 82, 1)])
    I.attach_store(con)
    I.update(con, "c0", "detectors", upto=T)
    feed(con, grid(a, T, "60min"))
    assert con.execute(f"SELECT count(*) FROM {ST}.health_carry WHERE DeviceId = 'S00' AND EventId = 9222").fetchall()[0][0] == 1
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))


def test_a_wrong_old_clock_in_a_chunk_is_held_back_and_counted():
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    mid = a + pd.Timedelta(hours=12)
    con = connect(ev, det)
    bad = pd.Timestamp("2015-01-01")
    con.execute("CREATE TABLE c1 AS SELECT * FROM events WHERE Timestamp < ?", [mid])
    con.execute("INSERT INTO c1 VALUES ('S00', ?, 82, 1), ('S00', ?, 1, 2)", [bad, bad])
    I.attach_store(con)
    s = I.update(con, "c1", "detectors")
    assert s.events_bad_stamp == {"S00": 2} and not s.events_late
    feed(con, [mid], store="clean")
    assert store_rows(con, ST) == store_rows(con, "clean")
    # the same chunk again: the row is still no use, the rest is late; nothing changes
    s = I.update(con, "c1", "detectors")
    assert s.events_in == 0 and s.events_late["S00"] > 0 and s.rows_written == {}
    assert store_rows(con, ST) == store_rows(con, "clean")
    # on a signal the store has heard from, such a row is late; a signal new to the store is held back the same way
    bad_clock_table(con, "c2", [("S01", bad, 82, 1), ("S02", bad, 82, 1), ("S02", bad + DAY, 81, 1)])
    s = I.update(con, "c2", "detectors")
    assert s.events_in == 0 and s.events_late == {"S01": 1} and s.events_bad_stamp == {"S02": 2}
    assert store_rows(con, ST) == store_rows(con, "clean")
    I.update(con, "events", "detectors", upto=T)
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))


def test_stamps_no_nanosecond_time_can_hold_are_counted():
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    mid = a + pd.Timedelta(hours=2)
    con = connect(ev, det)
    con.execute("CREATE TABLE c1 AS SELECT * REPLACE (Timestamp::TIMESTAMP AS Timestamp) FROM events WHERE Timestamp < ?", [mid])
    con.execute("INSERT INTO c1 VALUES ('S00', TIMESTAMP '0001-01-01', 82, 1), ('S01', TIMESTAMP '2500-01-01', 82, 1), ('S01', TIMESTAMP '2500-01-02', 81, 1)")
    I.attach_store(con)
    s = I.update(con, "c1", "detectors")
    assert s.events_bad_stamp == {"S00": 1, "S01": 2}
    feed(con, [mid], store="clean")
    assert store_rows(con, ST) == store_rows(con, "clean")
    s = I.update(con, "c1", "detectors", upto=mid)          # such a stamp is bad whatever upto says, not a row for later
    assert s.events_bad_stamp == {"S00": 1, "S01": 2} and not s.events_after_upto


USABLE = f"EventId IN ({','.join(map(str, H.ALLOWED))}) AND (EventId NOT IN (81, 82) OR Parameter BETWEEN 1 AND 64)"


def test_every_row_of_a_chunk_lands_in_exactly_one_count():
    """with upto, a chunk with a row of each sort: new, late, on the watermark, past upto, exactly on upto, out of range on either
    side, no stamp, older than the limit, no signal, another code.  Each allowed row of a named signal is in one count; no row
    but the new ones reaches the store."""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    mid, end = a + pd.Timedelta(hours=2), a + pd.Timedelta(hours=3)
    con = connect(ev, det)
    cols = "SELECT DeviceId, Timestamp::TIMESTAMP AS Timestamp, EventId, Parameter FROM events"
    I.attach_store(con)
    I.attach_store(con, name="clean")
    I.update(con, "events", "detectors", upto=mid)
    I.update(con, "events", "detectors", store="clean", upto=mid)
    w = I._through(con, ST)
    con.execute(f"CREATE TABLE ch AS {cols} WHERE Timestamp >= ? AND Timestamp < ?", [mid, end])
    allowed = f"{USABLE} AND DeviceId IS NOT NULL"
    n_new = con.execute(f"SELECT count(*) FROM ch WHERE {allowed}").fetchall()[0][0]
    wm_s00 = con.execute(f"SELECT TimeStamp FROM {ST}.health_carry WHERE DeviceId = 'S00' AND EventId = 9402").fetchall()[0][0]
    on_wm = con.execute(f"SELECT EventId, Parameter FROM events WHERE DeviceId = 'S00' AND Timestamp = ? AND {USABLE} LIMIT 1", [wm_s00]).fetchall()[0]
    far = "TIMESTAMP '2500-01-01'"
    con.execute(f"""INSERT INTO ch VALUES
        ('S00', ?, 82, 1),                      -- late
        ('S00', ?, ?, ?),                       -- on the watermark: late
        ('S00', ?, 82, 1),                      -- a year past upto
        ('S00', ?, 81, 1),                      -- exactly on upto
        ('S01', ?, 82, 2),                      -- past upto
        ('S00', {far}, 82, 1),                  -- out of range, past upto too
        ('S00', TIMESTAMP '1650-01-01', 82, 1), -- out of range, before upto
        ('S01', NULL, 82, 1),                   -- no stamp
        ('S02', TIMESTAMP '2015-01-01', 82, 1), -- older than the limit, on a signal the store has not met
        ('S02', {far}, 81, 1),                  -- the same signal, out of range
        ('S02', ?, 82, 1),                      -- the same signal, past upto
        (NULL, ?, 82, 1),                       -- no signal
        (NULL, {far}, 82, 1),
        (NULL, ?, 82, 1),
        ('S00', ?, 999, 1),                     -- not a code health reads
        ('S00', {far}, 999, 1),
        ('S00', ?, 82, 65)                      -- not a detector
        """, [mid - pd.Timedelta(hours=1), wm_s00, on_wm[0], on_wm[1], end + 365 * DAY, end, end + DAY, end, mid, end, mid, mid])
    n_all = con.execute(f"SELECT count(*) FROM ch WHERE {allowed}").fetchall()[0][0]
    assert n_all == n_new + 11
    s = I.update(con, "ch", "detectors", upto=end)
    assert s.events_in == n_new
    assert s.events_late == {"S00": 2}
    assert s.events_bad_stamp == {"S00": 2, "S01": 1, "S02": 2}
    assert s.events_after_upto == {"S00": 2, "S01": 1, "S02": 1}
    assert s.events_in + sum(x for d in (s.events_late, s.events_bad_stamp, s.events_after_upto) for x in d.values()) == n_all
    assert s.through == I._through(con, ST) and s.through > w
    I.update(con, "events", "detectors", store="clean", upto=end)
    assert store_rows(con, ST) == store_rows(con, "clean")
    # the same chunk again: only the new rows change their count, they are late now
    s2 = I.update(con, "ch", "detectors", upto=end)
    assert s2.events_in == 0 and sum(s2.events_late.values()) == n_new + 2
    assert s2.events_bad_stamp == s.events_bad_stamp and s2.events_after_upto == s.events_after_upto
    # no upto: a good stamp is no business of that count
    con.execute("DELETE FROM ch WHERE Timestamp >= ? AND Timestamp < TIMESTAMP '2262-01-01'", [end])
    s3 = I.update(con, "ch", "detectors")
    assert not s3.events_after_upto and s3.events_bad_stamp == s.events_bad_stamp


def test_rows_past_upto_are_counted_and_a_later_update_can_still_take_them():
    """a table that holds more than the time covered: what lies past upto is counted, not lost; the next update takes it"""
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    mid = a + pd.Timedelta(hours=12)
    con = connect(ev, det)
    s = feed(con, [mid])[0]
    expect = dict(con.execute(f"SELECT DeviceId, count(*) FROM events WHERE Timestamp >= ? AND {USABLE} GROUP BY 1", [mid]).fetchall())
    assert s.events_after_upto == expect and set(expect) == {"S00", "S01"}
    s = I.update(con, "events", "detectors", upto=T)
    assert s.events_in == sum(expect.values()) and not s.events_after_upto
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))


@pytest.mark.parametrize("ahead", [pd.Timedelta(hours=1), pd.Timedelta(seconds=5)], ids=["1 hour", "5 s"])
def test_a_clock_running_ahead_shows_in_the_count_when_upto_holds_its_rows_back(ahead):
    """S01's stamps run ahead of the time it reports them; each update, with upto = the end of the time it covers, gets what arrived
    in those 15 minutes.  The rows stamped past upto are not processed and never come again: lost, and that count is the only sign."""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    con = connect(ev, det)
    I.attach_store(con)
    held, took = 0, 0
    for c in grid(a, a + pd.Timedelta(hours=3), "15min"):
        con.execute(f"CREATE OR REPLACE TEMP TABLE ch AS SELECT * REPLACE (CASE WHEN DeviceId = 'S01' THEN Timestamp + INTERVAL {int(ahead.total_seconds())} SECOND ELSE Timestamp END AS Timestamp) FROM events WHERE Timestamp >= ? AND Timestamp < ?",
                    [c - pd.Timedelta(minutes=15), c])
        past, before = con.execute(f"SELECT count(*) FILTER (Timestamp >= ?), count(*) FILTER (Timestamp < ?) FROM ch WHERE DeviceId = 'S01' AND {USABLE}", [c, c]).fetchall()[0]
        s = I.update(con, "ch", "detectors", upto=c)
        assert s.events_after_upto == ({"S01": past} if past else {})
        assert not s.events_bad_stamp and not s.events_late
        held += s.events_after_upto.get("S01", 0)
        took += before
    assert held > 0
    wm = con.execute(f"SELECT TimeStamp FROM {ST}.health_carry WHERE DeviceId = 'S01' AND EventId = 9402").fetchall()
    if ahead >= pd.Timedelta(minutes=15):
        assert not wm and took == 0                         # the signal looks silent
    else:
        assert wm and took > held                           # a few seconds' rows lost at the end of each update


def test_a_signal_starting_over_leaves_the_other_signals_carry_alone():
    """A has been silent past the limit and comes back in an update where the other signals hold live carry: an open chain, open
    colour pieces, a last ON.  Only A starts over: the store equals one that never saw A's old rows, and the batch."""
    ev, det = synth.make(3, 2, seed=808, faults=True)
    a = ev.Timestamp.min().normalize()
    T = end_of(ev)
    devs = sorted(ev.DeviceId.unique())
    A, B = devs[0], devs[1]
    ret = a + pd.Timedelta(hours=9, minutes=7)
    sec = lambda x: pd.Timedelta(seconds=x)
    # B's detector 1 held ON across the update that brings A back
    held = dets(ev, [1], ret - sec(1200), ret + sec(1200), dev=B)
    ev = add_rows(ev[~held], [(B, ret - sec(900), 82, 1), (B, ret + sec(900), 81, 1)])
    old = ev[(ev.DeviceId == A) & (ev.Timestamp < a + pd.Timedelta(hours=6))].assign(Timestamp=lambda x: x.Timestamp - 150 * DAY)
    live = ev[(ev.DeviceId != A) | (ev.Timestamp >= ret)]
    con = connect(pd.concat([old, live], ignore_index=True), det)
    hd = (T - a) / DAY
    I.attach_store(con, name="m", horizon_days=hd)
    I.attach_store(con, name="clean", horizon_days=hd)
    con.execute("CREATE TABLE c0 AS SELECT * FROM events WHERE Timestamp < ?", [a])
    I.update(con, "c0", "detectors", store="m")
    restarts, lo = 0, a
    for c in sorted(set(grid(a, T, "60min")) | {ret + sec(60)}):
        con.execute("CREATE OR REPLACE TEMP TABLE ch AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?", [lo, c])
        lo = c
        stale_before = con.execute("SELECT count(*) FROM m.health_carry WHERE DeviceId = ? AND EventId = 9402 AND TimeStamp < ?", [A, a]).fetchall()[0][0]
        live_carry = {k: con.execute(f"SELECT count(*) FROM m.health_carry WHERE DeviceId = ? AND EventId {q}", [B]).fetchall()[0][0]
                      for k, q in (("chain", "= 9222"), ("piece", "BETWEEN 9300 AND 9302"), ("last on", "= 9221"))}
        s = I.update(con, "ch", "detectors", store="m")
        I.update(con, "ch", "detectors", store="clean")
        stale_after = con.execute("SELECT count(*) FROM m.health_carry WHERE DeviceId = ? AND EventId = 9402 AND TimeStamp < ?", [A, a]).fetchall()[0][0]
        if stale_before and not stale_after:                # A's update
            restarts += 1
            assert s.signals == 3 and all(live_carry.values()), live_carry
            assert con.execute("SELECT count(*) FROM m.health_carry WHERE DeviceId = ? AND EventId = 9222 AND Parameter = 1 AND TimeStamp = ?", [B, ret - sec(900)]).fetchall()[0][0] == 1
    assert restarts == 1
    same_settled(settled(con, "m", T, hd), settled(con, "clean", T, hd), "restart against clean")
    assert carry_rows(con, "m") == carry_rows(con, "clean")
    same_outputs(I.judge_rows(con, "detectors", store="m", end=T), H.health(con, "events", "detectors", start=a, end=T, periods=True, bin_origin="clock"), "restart against batch")


def test_the_limit_of_belief_is_exact():
    """lo = the latest stamp known - horizon - 92 days.  A watermark exactly at lo holds: its open chain goes on, and a row repeated on
    it is late, the store as if it had not come.  A watermark 1 us older starts over.  A row exactly at lo is believed, one 1 us older
    is not."""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    us = pd.Timedelta(microseconds=1)
    limit = 95 * DAY                                        # the recorded horizon, 3, + 92
    columns = "(DeviceId VARCHAR, Timestamp TIMESTAMP, EventId INT, Parameter INT)"
    for later, holds in ((pd.Timedelta(0), True), (us, False)):
        con = connect(ev, det)
        con.execute("CREATE TABLE c0 AS SELECT DeviceId, Timestamp::TIMESTAMP AS Timestamp, EventId, Parameter FROM events WHERE Timestamp < ?", [a + pd.Timedelta(hours=2)])
        W = pd.Timestamp(con.execute("SELECT max(Timestamp) FROM c0").fetchall()[0][0]) + pd.Timedelta(seconds=1)
        con.execute("INSERT INTO c0 VALUES ('S00', ?, 82, 1)", [W])
        R = W + limit + later
        rows = [("S01", R, 1, 2), ("S00", R - pd.Timedelta(seconds=1), 1, 2)]
        con.execute(f"CREATE TABLE c1 {columns}")
        con.executemany("INSERT INTO c1 VALUES (?, ?, ?, ?)", rows)
        con.execute(f"CREATE TABLE c1_repeat {columns}")
        con.executemany("INSERT INTO c1_repeat VALUES (?, ?, ?, ?)", rows + [("S00", W, 82, 1)])
        I.attach_store(con)
        I.attach_store(con, name="clean")
        I.update(con, "c0", "detectors")
        I.update(con, "c0", "detectors", store="clean")
        assert I._through(con, ST) == W
        s = I.update(con, "c1_repeat", "detectors")
        I.update(con, "c1", "detectors", store="clean")
        assert s.events_late == {"S00": 1} and not s.events_bad_stamp and s.events_in == 2
        chain = con.execute(f"SELECT count(*) FROM {ST}.health_carry WHERE DeviceId = 'S00' AND EventId = 9222 AND Parameter = 1").fetchall()[0][0]
        assert chain == (1 if holds else 0)
        assert feature_rows(con, ST) == feature_rows(con, "clean")
        # rows of signals new to the store
        lo = R - limit
        con.execute(f"CREATE TABLE c2 {columns}")
        con.executemany("INSERT INTO c2 VALUES (?, ?, ?, ?)", [("S02", lo, 82, 1), ("S03", lo - us, 82, 1)])
        s = I.update(con, "c2", "detectors")
        assert s.events_in == 1 and s.events_bad_stamp == {"S03": 1} and not s.events_late
        wm = con.execute(f"SELECT DeviceId, TimeStamp FROM {ST}.health_carry WHERE EventId = 9402 AND DeviceId IN ('S02', 'S03')").fetchall()
        assert [(d, pd.Timestamp(t)) for d, t in wm] == [("S02", lo)]
        con.close()


def test_every_kind_of_interval_goes_at_the_trim():
    """a comms gap with a chain and a colour piece over it, all ended well before the cut: every kind of interval goes"""
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    G0 = a + pd.Timedelta(hours=3)
    sec = lambda x: pd.Timedelta(seconds=x)
    keep = ~(silence(ev, G0 - sec(100), G0 + sec(100)) | dets(ev, [2], G0 - sec(310), G0 + sec(310)) | colour(ev, 2, G0 - sec(400), G0 + sec(320)) | dets(ev, [1], G0 - sec(10), G0 + sec(410), dev="S01"))
    ev2 = add_rows(ev[keep], [("S00", G0 - sec(300), 82, 2), ("S00", G0 + sec(300), 81, 2), ("S00", G0 - sec(300), 1, 2), ("S00", G0 + sec(300), 8, 2), ("S00", G0 + sec(304), 9, 2), ("S01", G0, 82, 1), ("S01", G0 + sec(400), 81, 1)])
    con = connect(ev2, det)
    cuts = sorted(set(grid(a, T, "60min")) | {G0 - sec(50), G0 + sec(150)})
    feed(con, cuts, store="trims", horizon_days=.25)
    feed(con, cuts, store="keeps", horizon_days=2)
    C = pd.Timestamp(I._meta(con, "trims")["trimmed_before"])
    kinds = {r[0] for r in con.execute("SELECT DISTINCT Kind FROM keeps.health_intervals WHERE EndTime < ?", [C]).fetchall()}
    assert {"comms_gap", "long_on", "bad_chain", "bad_piece"} <= kinds
    assert not con.execute("SELECT Kind FROM trims.health_intervals WHERE EndTime < ?", [C]).fetchall()


def test_a_signal_silent_past_the_limit_starts_over_and_one_inside_it_does_not():
    """a chain open at the end of day 0.  Back after 100 days, more than the horizon + 92: what the carry says is past believing,
    the signal starts cold, as health(start=S) does.  Back after 60 days: the carry holds, the chain has its true start."""
    ev, det = base()
    last = ev.Timestamp.max()
    ev0 = add_rows(ev[~dets(ev, [1], last - pd.Timedelta(hours=1), last + DAY)], [("S00", last - pd.Timedelta(minutes=30), 82, 1)])
    d1 = dets(ev, [1], ev.Timestamp.min(), last + DAY, dev="S00")   # no event of detector 1 after the gap: the chain stays open
    for days, hd, starts_over in ((100, 1, True), (60, 1, False), (100, 30, False)):     # 30: its horizon reaches further back
        ev1 = ev[~d1].assign(Timestamp=lambda x: x.Timestamp + days * DAY)
        T = end_of(ev1)
        con = connect(ev1, det)
        con.register("ev0_df", ev0)
        con.execute("CREATE TABLE c0 AS SELECT * FROM ev0_df")
        con.unregister("ev0_df")
        I.attach_store(con)
        I.update(con, "c0", "detectors", horizon_days=hd)
        before = store_rows(con, ST)
        with pytest.raises(RuntimeError, match="injected"):
            I.update(FailBefore(con, "COMMIT"), "events", "detectors", horizon_days=hd)
        assert store_rows(con, ST) == before                # the failure leaves the old carry, and all the rest
        s = I.update(con, "events", "detectors", horizon_days=hd)
        assert s.signals == 2 and not s.events_bad_stamp
        x = settled(con, ST, T, 1)
        w = x["win"].set_index("DeviceId").dev["S00"]
        f = x["fch"]
        f = f[(f.dev == w) & (f.det == 1)]
        gaps = intervals(con, "comms_gap")
        if starts_over:
            assert not len(f) and not len(gaps[gaps.EndTime - gaps.StartTime > 50 * DAY])
            same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))
        else:
            assert len(f) == 1 and f.s.iloc[0] < 0 and len(gaps[gaps.EndTime - gaps.StartTime > 50 * DAY]) == 2
        con.close()


def fall_back_log():
    """the fall clock change on zone-free local stamps: 01:00-02:00 comes twice.  second: a second pass with its own events
    (the first pass's, 3 s later), all before the first pass's last event of each signal."""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    h1, h2 = a + pd.Timedelta(hours=1), a + pd.Timedelta(hours=2)
    second = ev[(ev.Timestamp >= h1) & (ev.Timestamp < h2 - pd.Timedelta(minutes=5))].assign(Timestamp=lambda x: x.Timestamp + pd.Timedelta(seconds=3))
    return ev, second, det, a, h1, h2


def test_the_repeated_hour_of_the_fall_clock_change_is_late_and_not_mixed_in():
    """fed as it arrives, in 15-minute updates, the second pass falls behind the watermark: counted late, the store as if it had
    never come; the answers equal a batch run without it, and differ from a batch run that mixes both passes (design 3.5)."""
    first, second, det, a, h1, h2 = fall_back_log()
    T = end_of(first)
    con = connect(pd.concat([first, second], ignore_index=True), det)
    con.register("f_df", first)
    con.register("s_df", second)
    con.execute("CREATE TABLE first_pass AS SELECT * FROM f_df; CREATE TABLE second_pass AS SELECT * FROM s_df")
    con.unregister("f_df"), con.unregister("s_df")
    n_second = con.execute(f"SELECT count(*) FROM second_pass WHERE EventId IN ({','.join(map(str, H.ALLOWED))}) AND (EventId NOT IN (81, 82) OR Parameter BETWEEN 1 AND 64)").fetchall()[0][0]
    I.attach_store(con)
    I.attach_store(con, name="clean")
    prev, late = a, 0
    for c in grid(a, h2, "15min"):
        for st in (ST, "clean"):
            I.update(con, "first_pass", "detectors", store=st, upto=c)
    for c in grid(h1, h2, "15min"):
        con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM second_pass WHERE Timestamp >= ? AND Timestamp < ?", [c - pd.Timedelta(minutes=15), c])
        s = I.update(con, "chunk", "detectors")
        assert s.events_in == 0 and not s.events_bad_stamp
        late += sum(s.events_late.values())
    assert late == n_second > 1000
    for st in (ST, "clean"):
        I.update(con, "first_pass", "detectors", store=st, upto=T)
    assert store_rows(con, ST) == store_rows(con, "clean")
    got = I.judge_rows(con, "detectors", end=T, horizon_days=1)
    same_outputs(got, H.health(con, "first_pass", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))
    with pytest.raises(AssertionError):                     # "events" holds both passes
        same_as_batch(con, ST, T, 1)


def test_both_passes_of_the_repeated_hour_in_one_update_are_mixed_as_in_a_batch():
    """a chunk that holds both passes, from before them: its rows are all after the watermark, sorted in together"""
    first, second, det, a, h1, h2 = fall_back_log()
    T = end_of(first)
    con = connect(pd.concat([first, second], ignore_index=True), det)
    feed(con, [h1])
    con.execute("CREATE TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ?", [h1])
    assert not I.update(con, "chunk", "detectors").events_late
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))
    same_as_batch(con, ST, T, 1)


def test_the_skipped_hour_of_the_spring_clock_change_is_a_comms_gap_in_both():
    g0, g1 = 2 * 3600.0, 3 * 3600.0

    def reach(con, stats, T):
        b = intervals(con, "comms_gap")
        b = b[b.EndTime - b.StartTime > pd.Timedelta(minutes=55)]
        assert sorted(b.DeviceId) == ["S00", "S01"]
    run_scene([], lambda e: (e.Timestamp >= at(g0)) & (e.Timestamp < at(g1)), [g0 - 300, g0 + 300, g1 - 300, g1 + 300], reach)


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


@pytest.mark.parametrize("word,skip", [("COMMIT", 0), ("INSERT INTO HEALTH_STORE.HEALTH_DETECTOR_5MIN", 0), ("INSERT INTO HEALTH_STORE.HEALTH_INTERVALS", 0),
                                       ("DELETE FROM HEALTH_STORE.HEALTH_TICK_COUNTS", 0), ("CREATE OR REPLACE TEMP TABLE HQ_XDE", 0)])
def test_a_failure_leaves_the_store_unchanged_and_the_rerun_equals_a_clean_run(word, skip):
    ev, det = base()
    a, T = ev.Timestamp.min().normalize(), end_of(ev)
    con = connect(ev, det)
    c0, c1 = a + pd.Timedelta(hours=10), a + pd.Timedelta(hours=11)
    feed(con, [c0], horizon_days=.25)
    before = store_rows(con, ST)
    meta = I._meta(con, ST)
    with pytest.raises(RuntimeError, match="injected"):
        I.update(FailBefore(con, word, skip), "events", "detectors", upto=c1, horizon_days=.25)
    assert store_rows(con, ST) == before and I._meta(con, ST) == meta
    assert not con.execute("SELECT table_name FROM duckdb_tables() WHERE table_name LIKE 'hq_%'").fetchall()
    I.update(con, "events", "detectors", upto=c1, horizon_days=.25)
    I.update(con, "events", "detectors", upto=T, horizon_days=.25)
    clean = connect(ev, det)
    feed(clean, [c0, c1, T], horizon_days=.25)
    assert store_rows(con, ST) == store_rows(clean, ST)


def test_a_failure_while_recording_the_horizon_loses_nothing():
    """a store cannot run without its recorded horizon: a failure between the old row's delete and the new row's insert keeps it"""
    ev, det = base()
    con = connect(ev, det)
    a = ev.Timestamp.min().normalize()
    I.attach_store(con, horizon_days=7)
    with pytest.raises(RuntimeError, match="injected"):
        I.attach_store(FailBefore(con, f"INSERT INTO {ST.upper()}.HEALTH_META"), horizon_days=2)
    assert I._meta(con, ST)["horizon_days"] == "7"
    assert I.update(con, "events", "detectors", upto=a + pd.Timedelta(hours=1)).events_in > 0
    with pytest.raises(RuntimeError, match="injected"):
        I.update(FailBefore(con, f"INSERT INTO {ST.upper()}.HEALTH_META"), "events", "detectors", upto=a, horizon_days=2)
    assert I._meta(con, ST)["horizon_days"] == "7"
    # a new store's first bookkeeping: all of it or none of it, and attaching again works
    class FailHalfway(FailBefore):
        def executemany(self, sql, rows):
            self.con.execute(sql, rows[0])
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        I.attach_store(FailHalfway(con, "NOTHING"), name="other", horizon_days=2)
    assert I._meta(con, "other") == {}
    I.attach_store(con, name="other", horizon_days=2)
    assert I._meta(con, "other")["horizon_days"] == "2"


def test_a_process_killed_after_commit_keeps_the_whole_update(tmp_path):
    """the update reached the log but the file was never checkpointed: on the next attach the store holds all of it, and
    running the same update again changes nothing"""
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    ep, dp, sp = (tmp_path / "ev.parquet").as_posix(), (tmp_path / "det.parquet").as_posix(), (tmp_path / "s.duckdb").as_posix()
    G.to_pq(ev[ev.Timestamp < a + pd.Timedelta(hours=4)], ep)
    G.to_pq(det, dp)
    c0, c1 = a + pd.Timedelta(hours=2), a + pd.Timedelta(hours=3)
    code = f"""
        import os, duckdb, pandas as pd
        from atspm_detector import incremental as I
        con = duckdb.connect()
        con.execute("CREATE TABLE events AS SELECT * FROM read_parquet('{ep}')")
        con.execute("CREATE TABLE detectors AS SELECT * FROM read_parquet('{dp}')")
        I.attach_store(con, '{sp}')
        I.update(con, "events", "detectors", upto=pd.Timestamp('{c0}'))
        I.update(con, "events", "detectors", upto=pd.Timestamp('{c1}'))
        os._exit(9)
        """
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=dict(os.environ), capture_output=True, text=True)
    assert r.returncode == 9, r.stderr[-1500:]
    con = duckdb.connect()
    con.execute(f"CREATE TABLE events AS SELECT * FROM read_parquet('{ep}')")
    con.execute(f"CREATE TABLE detectors AS SELECT * FROM read_parquet('{dp}')")
    I.attach_store(con, sp)
    clean = connect(ev[ev.Timestamp < a + pd.Timedelta(hours=4)], det)
    feed(clean, [c0, c1])
    assert store_rows(con, ST) == store_rows(clean, ST)
    s = I.update(con, "events", "detectors", upto=c1)
    assert s.events_in == 0 and store_rows(con, ST) == store_rows(clean, ST)
    con.close()


def test_a_process_killed_before_commit_leaves_the_file_store_consistent(tmp_path):
    ev, det = base()
    a = ev.Timestamp.min().normalize()
    ep, dp, sp = (tmp_path / "ev.parquet").as_posix(), (tmp_path / "det.parquet").as_posix(), (tmp_path / "s.duckdb").as_posix()
    G.to_pq(ev[ev.Timestamp < a + pd.Timedelta(hours=4)], ep)
    G.to_pq(det, dp)
    c0, c1 = a + pd.Timedelta(hours=2), a + pd.Timedelta(hours=3)
    code = f"""
        import os, duckdb, pandas as pd
        from atspm_detector import incremental as I
        con = duckdb.connect()
        con.execute("CREATE TABLE events AS SELECT * FROM read_parquet('{ep}')")
        con.execute("CREATE TABLE detectors AS SELECT * FROM read_parquet('{dp}')")
        I.attach_store(con, '{sp}')
        I.update(con, "events", "detectors", upto=pd.Timestamp('{c0}'))
        con.execute("CHECKPOINT health_store")
        class Kill:
            def __init__(self, c): self.c = c
            def execute(self, s, *a):
                if s.lstrip().startswith("COMMIT"): os._exit(9)
                return self.c.execute(s, *a)
            def __getattr__(self, n): return getattr(self.c, n)
        I.update(Kill(con), "events", "detectors", upto=pd.Timestamp('{c1}'))
        """
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=dict(os.environ), capture_output=True, text=True)
    assert r.returncode == 9, r.stderr[-1500:]
    con = duckdb.connect()
    con.execute(f"CREATE TABLE events AS SELECT * FROM read_parquet('{ep}')")
    con.execute(f"CREATE TABLE detectors AS SELECT * FROM read_parquet('{dp}')")
    I.attach_store(con, sp)
    clean = connect(ev[ev.Timestamp < a + pd.Timedelta(hours=4)], det)
    feed(clean, [c0])
    assert store_rows(con, ST) == store_rows(clean, ST)     # the killed update left nothing
    for x in (con, clean):                                  # and the rerun goes on from there
        I.update(x, "events", "detectors", upto=c1)
    assert store_rows(con, ST) == store_rows(clean, ST)
    con.close()


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


def test_store_with_overlap_of_count_zones_only_gives_the_same_answers():
    ev, det = base()
    con = connect(ev, det)
    T = end_of(ev)
    feed(con, grid(ev.Timestamp.min().floor("15min"), T, "60min"), green_overlap="count")
    assert con.execute(f"SELECT count(DISTINCT Detector) FROM {ST}.health_detector_5min WHERE On_Green5_ns IS NOT NULL").fetchall()[0][0] == 2
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1),
                 H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))


FINGERPRINT = "037dfcd80910dcfa"


def test_extractor_fingerprint_is_pinned():
    """a change of the rendered extractor must be looked at: update this value, and decide whether EXTRACTOR_VERSION goes
    up (an old store's rows would then be refused)."""
    assert I.fingerprint() == FINGERPRINT
    assert I.EXTRACTOR_VERSION == "1"


def test_exact_ties_decide_alike_in_a_store_and_a_clock_batch():
    """chains of exactly 5.0 s, 60.0 s and 300.0 s at many times.  On a window's origin today's double expression (f / 1e9 -
    s / 1e9) lands a tiny bit above or below the limit, depending on the times and the origin (design 3.5); from the epoch,
    where all these times have the same binary exponent, it lands exactly on it.  A store and health(bin_origin="clock") use
    that one origin, whatever the window and the chunks, and every chain's fact comes out the same."""
    rows = []
    t = 3000.1
    for k in range(240):
        d = (5.0, 60.0, 300.0)[k % 3]
        rows += [S00(t, 82, 1), S00(t + d, 81, 1)]
        t += d + 13.7
    # and across each power of two of seconds from the window start, where the noise on a window's origin comes from
    for edge_s in (4096, 8192, 16384, 32768, 65536):
        for x in (1.3, 2.7, 3.9):
            for c, d in ((2, 5.0), (3, 60.0), (4, 300.0)):
                rows += [S00(edge_s - x * c, 82, c), S00(edge_s - x * c + d, 81, c)]
    ev, det = scene_log(rows, lambda e: dets(e, [1], at(2900), at(t + 100)) | dets(e, [2, 3, 4], at(0), at(86400)))
    T = end_of(ev)
    con = connect(ev, det)

    def off_limit(origin):
        return con.execute(f"""SELECT count(*) FROM (
            SELECT (f - {origin}) / 1e9 - (s - {origin}) / 1e9 AS d, (f - s) / 1e9 AS l FROM (
                SELECT epoch_ns(Timestamp) AS s, lead(epoch_ns(Timestamp)) OVER (PARTITION BY Parameter ORDER BY Timestamp) AS f, EventId AS e FROM events
                WHERE DeviceId = 'S00' AND Parameter BETWEEN 1 AND 4 AND EventId IN (81, 82) AND Timestamp BETWEEN ? AND ?) WHERE e = 82)
            WHERE l IN (5, 60, 300) AND d <> l""", [at(2950), at(70000)]).fetchall()[0][0]
    assert off_limit(0) == 0 and off_limit((T - DAY).value) > 0     # so this test sees a change of origin
    feed(con, random_cuts(at(0), T, 40, 9))
    q = H.build("events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock")
    b = con.execute(q.text("SELECT dev, det, b, nst, n5s, n60s, max_on, lmax FROM d5")).df().sort_values(["dev", "det", "b"]).reset_index(drop=True)
    a = settled(con, ST, T, 1)["d5"][["dev", "det", "b", "nst", "n5s", "n60s", "max_on", "lmax"]].sort_values(["dev", "det", "b"]).reset_index(drop=True)
    for c in ("n5s", "n60s", "nst"):
        a[c], b[c] = a[c].fillna(0).astype("int64"), b[c].fillna(0).astype("int64")
    va, vb = G.values(a), G.values(b)
    for c in va:
        assert va[c] == vb[c], (c, _first_diff(va[c], vb[c]))
    same_outputs(I.judge_rows(con, "detectors", end=T, horizon_days=1), H.health(con, "events", "detectors", start=T - DAY, end=T, periods=True, bin_origin="clock"))


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
def test_warm_store_against_cold_batch():
    """history before S: only what crosses S differs -- a chain's time and a colour shown at S, the first ON's previous ON
    and the first start's previous start.  Every later bin is the same."""
    ev, det = synth.make(2, 3, seed=8, faults=False)
    con = connect(ev, det)
    a = ev.Timestamp.min().normalize()
    T = a + 3 * DAY
    S = T - DAY
    I.attach_store(con)
    I.update(con, "events", "detectors", upto=T)
    warm = settled(con, ST, T, 1)
    I.attach_store(con, name="cold")
    con.execute("CREATE TABLE after_s AS SELECT * FROM events WHERE Timestamp >= ?", [S])
    I.update(con, "after_s", "detectors", store="cold")
    cold = settled(con, "cold", T, 1)
    # nothing keyed before the window: the store's older rows do not reach these
    for k in ("rs5", "rgq", "rtko", "rhold", "rplan"):
        same_settled({k: warm[k]}, {k: cold[k]}, "keyed inside the window")
    assert len(warm["rtko"]) and len(warm["rplan"])
    # bins after the first half hour (no chain here lasts that long, every phase has shown its colour)
    for k in ("d5", "p5"):
        same_settled({k: warm[k][warm[k].b >= 6].reset_index(drop=True)}, {k: cold[k][cold[k].b >= 6].reset_index(drop=True)}, "after 30 min")
    # the cold store equals a batch run from S, warm or not
    same_outputs(I.judge_rows(con, "detectors", store="cold", end=T, horizon_days=1),
                 H.health(con, "events", "detectors", start=S, end=T, periods=True, bin_origin="clock"))
    # the first bins differ only by what crosses S
    w, c = warm["d5"], cold["d5"]
    assert w[w.b < 6].on_ns.sum() >= c[c.b < 6].on_ns.sum()
    a_, b_ = I.judge_rows(con, "detectors", end=T, horizon_days=1), I.judge_rows(con, "detectors", store="cold", end=T, horizon_days=1)
    assert a_[0].health_status.tolist() == b_[0].health_status.tolist()


def test_warm_store_a_hold_crossing_the_window_start_keeps_its_true_start():
    """a chain from 2 h before S to 1 h after it (closed), and one from 3 h before S still open at T, re-logged inside the
    window (a detector with no event at all in the window has no row, as in health()): the warm store knows both starts; a
    cold run sees neither."""
    ev, det = synth.make(2, 2, seed=8, faults=False)
    a = ev.Timestamp.min().normalize()
    T = a + 2 * DAY
    S = T - DAY
    hr = pd.Timedelta(hours=1)
    # and det 5's last ONs, a too-fast pair, 1 h before S: its pending pair is not the window's
    # det 2's hold ended before S: not the window's
    keep = ~(dets(ev, [1], S - 3 * hr, S + 2 * hr) | dets(ev, [3], S - 4 * hr, T) | dets(ev, [5], S - 2 * hr, T) | dets(ev, [2], S - 7 * hr, S - 3 * hr))
    add = pd.DataFrame([("S00", S - 2.5 * hr, 81, 1), ("S00", S - 2 * hr, 82, 1), ("S00", S + hr, 81, 1), ("S00", S - 3.5 * hr, 81, 3),
                        ("S00", S - 3 * hr, 82, 3), ("S00", S + hr, 82, 3), ("S00", S - 1.5 * hr, 81, 5), ("S00", S - hr, 82, 5),
                        ("S00", S - hr + pd.Timedelta(seconds=.1), 81, 5), ("S00", S - hr + pd.Timedelta(seconds=.3), 82, 5),
                        ("S00", S - hr + pd.Timedelta(seconds=.4), 81, 5), ("S00", S - 6.5 * hr, 81, 2), ("S00", S - 6 * hr, 82, 2), ("S00", S - 4 * hr, 81, 2)],
                       columns=ev.columns).astype({"EventId": "int32", "Parameter": "int32"})
    ev = pd.concat([ev[keep], add], ignore_index=True)
    con = connect(ev, det)
    I.attach_store(con)
    I.update(con, "events", "detectors", upto=T)
    d, p = I.judge_rows(con, "detectors", end=T, horizon_days=1)
    x = settled(con, ST, T, 1)
    w = x["win"].set_index("DeviceId")
    h = x["rhold"]
    h = h[(h.dev == w.dev["S00"]) & (h.det == 1)]
    assert h.s.tolist() == [(S - 2 * hr).value - w.s_ns["S00"]] and h.f.tolist() == [(S + hr).value - w.s_ns["S00"]]
    f = x["fch"]
    f = f[(f.dev == w.dev["S00"]) & (f.det == 3)]
    assert f.s.tolist() == [(S - 3 * hr).value - w.s_ns["S00"]] and f.dur.iloc[0] == 27 * 3600 and not f.gap.iloc[0]
    assert x["d5"].b.min() >= 0 and x["tkg"].q.min() >= 0
    assert not len(x["rhold"][(x["rhold"].dev == w.dev["S00"]) & (x["rhold"].det == 2)])
    assert len(intervals(con, "long_on").query("DeviceId == 'S00' and Channel == 2")) == 1
    s = p[(p.DeviceId == "S00") & (p.condition == "stuck on")]
    assert (s.Detector == 3).any() and pd.Timestamp(s[s.Detector == 3].start.min()) == S - 3 * hr, s
    # the bins of the window hold the time of the held chains from S on, none before it
    d5 = x["d5"]
    assert d5[(d5.dev == w.dev["S00"]) & (d5.det == 3)].on_ns.sum() == DAY.value
    assert d.health_status.notna().all()
