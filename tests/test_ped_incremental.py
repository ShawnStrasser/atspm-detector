"""Stage 5 of the incremental-health design: ped health in the same pass.  The store's ped judge must give, for any cutting of the events into
updates, exactly ped_health(start=T - H, end=T, periods=True) on the raw events (design 3.5, 6).

T5.1  ped_health() through the one ped judge equals the frozen oracle v3 (the final rules on the raw log): defaults, all off, each option,
      periods, the movement model, on the golden sets
T5.2  ped_assignments() over the store's log equals it over the raw events
T5.3  ped chunk invariance: 15-minute chunks, random microsecond cuts, a cut on an event and 1 us after
T5.4  the store against ped_health(start=S, end=T): with and without history before S; through trims
T5.5  ped scenes across cuts, each with a check that it reaches its branch
T5.6  one read of the chunk per update
plus: the vehicle answers do not change with ped on; a store with ped off; late, re-delivered and wrongly stamped rows; one transaction."""
import duckdb
import numpy as np
import pandas as pd
import pytest

import make_golden as G
import synth
import test_incremental as TI
from atspm_detector import health_duckdb as H
from atspm_detector import incremental as I
from atspm_detector import ped_assign as A
from atspm_detector import ped_health_duckdb as P
from legacy import ped_health_duckdb_v3 as P0
from test_incremental import FailBefore, few_then_slow, grid, random_cuts, same_outputs

ST = "health_store"
T0 = pd.Timestamp("2026-09-28 06:00")
DAY = pd.Timedelta(days=1)
H1 = 1 / 96     # a horizon of 15 minutes


def con_of(rows, dev="S"):
    """a log of (seconds from T0, code, parameter), a detectors table with no detector"""
    ev = pd.DataFrame(rows, columns=["t", "EventId", "Parameter"])
    ev = pd.DataFrame({"DeviceId": dev, "Timestamp": T0 + pd.to_timedelta(ev.t, unit="s"), "EventId": ev.EventId, "Parameter": ev.Parameter})
    con = duckdb.connect()
    con.execute("SET threads = 4; SET memory_limit = '2GB'")
    con.register("e_df", ev)
    con.execute("CREATE TABLE events AS SELECT * FROM e_df")
    con.unregister("e_df")
    empty_detectors(con)
    return con


def empty_detectors(con):
    con.execute("CREATE TABLE detectors (DeviceId VARCHAR, Detector BIGINT, phase DOUBLE, function VARCHAR, lanes DOUBLE, "
                + ", ".join(f"{c} DOUBLE" for c in H.PROB.values()) + ")")


def fed(con, cuts, store=ST, horizon_days=30, **kw):
    I.attach_store(con, name=store, horizon_days=horizon_days, **kw)
    return [I.update(con, "events", "detectors", store=store, upto=c) for c in cuts]


def ped_of(con, T, hd, store=ST, **kw):
    r = I.judge_rows(con, "detectors", store=store, end=T, horizon_days=hd, **kw)
    return r.ped, r.ped_periods


def batch(con, T, hd, **kw):
    return P.ped_health(con, start=T - pd.Timedelta(days=hd), end=T, periods=True, **kw)


def same_ped(a, b, what=""):
    """two (inputs, periods), the order of the rows too"""
    same_outputs(a, b, what)
    assert list(a[0].DeviceId.astype(str) + a[0].Input.astype(str)) == list(b[0].DeviceId.astype(str) + b[0].Input.astype(str)), what
    assert G.dtypes(a[0]) == G.dtypes(b[0]) and G.dtypes(a[1]) == G.dtypes(b[1]), what


# --- T5.1 ---------------------------------------------------------------------------------------------------------------------
_RAW = {}


def raw(name):
    if name not in _RAW:
        ev, det = G.inputs(name)
        _RAW[name] = G.connect(ev, det)
    return _RAW[name]


OPTS = [("defaults", {}), ("all off", {k: False for k in P.OPT})] + [(f"{k} off", {k: False}) for k in P.OPT]


def _exact(a, b, what):
    for k in range(2):
        assert list(a[k].columns) == list(b[k].columns), what
        va, vb = G.values(a[k]), G.values(b[k])
        for c in va:
            assert va[c] == vb[c], (what, k, c)
        assert G.dtypes(a[k]) == G.dtypes(b[k]), what


@pytest.mark.parametrize("name,opt", OPTS, ids=[n for n, _ in OPTS])
@pytest.mark.parametrize("inp", ["sample", "synth_a", "synth_b"])
def test_ped_health_equals_the_oracle(inp, name, opt):
    con = raw(inp)
    _exact(P.ped_health(con, periods=True, options=opt), P0.ped_health(con, periods=True, options=opt), (inp, name))
    d = P.ped_health(con, options=opt)
    assert d.equals(P0.ped_health(con, options=opt))


def test_ped_health_equals_the_oracle_on_a_window_and_with_the_model():
    con = raw("synth_a")
    kw = dict(start="2026-10-05 05:17", end="2026-10-06 21:01", periods=True)
    _exact(P.ped_health(con, **kw), P0.ped_health(con, **kw), "window")
    a, b = P.ped_health(con, assign=True, periods=True), P0.ped_health(con, assign=True, periods=True)
    assert a[0].astype(str).equals(b[0].astype(str))
    assert a[1].equals(b[1])


def test_the_steps_for_the_log_run_with_the_final_rules_only():
    """the hourly dead rule and what only it read are gone (no dead code)"""
    names = [n for n, _ in P._steps("events")]
    assert not {"f_dead", "alive", "hours", "mins"} & set(names) and {"f_stop", "f_hold", "sil", "logq"} <= set(names)
    assert "dead_h" not in P.LIM and "dead_waits" not in P.LIM


# --- shared data: synthetic signals with faults, in one set --------------------------------------------------------------------
_SYN = {}


def syn():
    """3 signals x 2 days with faults (lost walks, dropped calls, a preemption with no end, duplicates, an out-of-order block)"""
    if not _SYN:
        ev, det = synth.make(3, 2, seed=0)
        con = TI.connect(ev, det)
        T = TI.end_of(ev)
        a = ev.Timestamp.min().floor("15min")
        I.attach_store(con, name="one", horizon_days=3)
        I.update(con, "events", "detectors", store="one", upto=T)
        _SYN.update(con=con, T=T, a=a, ev=ev, det=det)
    return _SYN


# --- T5.2 ---------------------------------------------------------------------------------------------------------------------
def test_ped_assignments_over_the_log_equal_the_raw_events():
    E = syn()
    con, T = E["con"], E["T"]
    S = T - 2 * DAY
    a, ca = A.ped_assignments(con, "events", S, T, return_candidates=True)
    b, cb = A.ped_assignments(con, "one.health_ped_events", S, T, return_candidates=True)
    assert len(a) and a.astype(str).equals(b.astype(str))
    assert len(ca) and ca.astype(str).equals(cb.astype(str))


def test_the_log_keeps_every_code_the_ped_modules_read():
    assert set(P.CODES) | set(A.CODES) == set(P.LOG_CODES)


# --- T5.3, T5.4 ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cut", ["15 min", *few_then_slow(["random", "hour"], 1)])
def test_chunk_invariance_and_batch_identity(cut):
    E = syn()
    ev, det, T, a = E["ev"], E["det"], E["T"], E["a"]
    con = TI.connect(ev, det)
    if cut == "15 min":
        cuts = grid(a, T, "15min")
    elif cut == "hour":
        cuts = grid(a, T, "60min")
    else:
        cuts = random_cuts(a, T, 40, 3)
    stats = fed(con, cuts, horizon_days=3)
    ref = batch(con, T, 2)
    assert len(ref[0]) >= 6 and {"bad", "ok"} & set(ref[0].status)
    got = ped_of(con, T, 2)
    same_ped(got, ref, cut)
    same_ped(got, ped_of(E["con"], T, 2, store="one"), "one chunk")
    assert sum(s.rows_written.get("health_ped_events", 0) for s in stats) > 1000
    assert con.execute(f"SELECT count(*), count(DISTINCT DeviceId) FROM {ST}.health_carry WHERE EventId = 9401").fetchall() == [(3, 3)]


def test_the_rows_kept_do_not_depend_on_the_chunking():
    E = syn()
    con = TI.connect(E["ev"], E["det"])
    fed(con, random_cuts(E["a"], E["T"], 40, 5), horizon_days=3)
    for t, cols in (("health_ped_events", "*"),
                    ("health_log_15min", "DeviceId, TimeStamp, min(All_First), max(All_Last)"),
                    ("health_intervals", "*")):
        where = " WHERE Kind = 'silence'" if t == "health_intervals" else ""
        group = " GROUP BY DeviceId, TimeStamp" if "min(" in cols else ""
        q = "SELECT {c} FROM {s}.{t}{w}{g} ORDER BY ALL"
        x = con.execute(q.format(c=cols, s=ST, t=t, w=where, g=group)).fetchall()
        y = E["con"].execute(q.format(c=cols, s="one", t=t, w=where, g=group)).fetchall()
        assert x == y and len(x), t


@pytest.mark.parametrize("hd", [1, 0.5])
def test_history_before_the_window_changes_nothing(hd):
    """a warm store judges only its window, as ped_health(start=S, end=T): nothing crosses S (design 3.5)"""
    E = syn()
    same_ped(ped_of(E["con"], E["T"], hd, store="one"), batch(E["con"], E["T"], hd), f"horizon {hd}")


def test_a_store_fed_from_the_window_start():
    E = syn()
    T, S = E["T"], E["T"] - DAY
    con = TI.connect(E["ev"], E["det"])
    con.execute("DELETE FROM events WHERE Timestamp < ?", [S])
    fed(con, grid(S, T, "60min"), horizon_days=1)
    same_ped(ped_of(con, T, 1), batch(con, T, 1), "from S")


def test_trims_and_compaction_change_nothing():
    E = syn()
    ev, det, T, a = E["ev"], E["det"], E["T"], E["a"]
    con = TI.connect(ev, det)
    I.attach_store(con, horizon_days=1)
    for k, c in enumerate(grid(a, T, "30min")):
        I.update(con, "events", "detectors", upto=c)
        if k % 7 == 3:
            I.trim(con, compact=True)
    same_ped(ped_of(con, T, 1), batch(con, T, 1), "trimmed")
    cut = pd.Timestamp(I._meta(con, ST)["trimmed_before"])
    for t, c in (("health_ped_events", "TimeStamp"), ("health_log_15min", "TimeStamp"), ("health_intervals", "EndTime")):
        assert con.execute(f"SELECT count(*) FROM {ST}.{t} WHERE {c} < ?", [cut]).fetchall()[0][0] == 0, t
    with pytest.raises(ValueError, match="trim"):
        I.judge_rows(con, "detectors", end=T, horizon_days=1.5)


def test_the_movement_model_through_the_judge():
    E = syn()
    con, T = E["con"], E["T"]
    a = ped_of(con, T, 2, store="one", ped_assign=True)
    b = batch(con, T, 2, assign=True)
    assert "assign_confidence" in a[0].columns and a[0].astype(str).equals(b[0].astype(str))
    assert a[1].equals(b[1])


def test_ped_options_reach_the_judge():
    E = syn()
    opt = {k: False for k in P.OPT}
    same_ped(ped_of(E["con"], E["T"], 2, store="one", ped_options=opt), batch(E["con"], E["T"], 2, options=opt), "all off")


# --- T5.5 ---------------------------------------------------------------------------------------------------------------------
def press(ch, t, dur=0.2):
    return [(t, 90, ch), (round(t + dur, 1), 89, ch)]


def walk(ph, t):
    return [(t - 1, 1, ph), (t, 21, ph), (t + 7, 22, ph), (t + 20, 23, ph)]


def beat(a, b, step=10):
    return [(t, 82, 1) for t in range(a, b, step)]


def waits(ch, a, n=10, step=300):
    """normal presses, each with its call and walk: the input is typed and judged"""
    return [x for k in range(n) for x in press(ch, a + step * k + ch - 0.1) + [(a + step * k + ch, 45, ch)] + walk(ch, a + step * k + ch + 5)]


def run_scene(rows, cut_s=(), seed=1, **judge_kw):
    """the scene fed in random chunks and at the given seconds (on an event, and 1 us after), each against the batch."""
    con = con_of(rows)
    a = T0.floor("15min")
    z = con.execute("SELECT max(Timestamp) FROM events").fetchall()[0][0]
    T = pd.Timestamp(z).floor("15min") + pd.Timedelta(minutes=15)
    hd = (T - a) / DAY
    ref = batch(con, T, hd, **{k: v for k, v in judge_kw.items() if k == "options"})
    kw = {"ped_options": judge_kw["options"]} if "options" in judge_kw else {}
    named = [T0 + pd.Timedelta(seconds=s) for s in cut_s]
    named = sorted(set(named + [x + pd.Timedelta(microseconds=1) for x in named]))
    for name, cuts in {"random": random_cuts(a, T, 12, seed), "named": named + [T]}.items():
        c = con_of(rows)
        fed(c, cuts, horizon_days=hd + 1)
        same_ped(ped_of(c, T, hd, **kw), ref, name)
    return ref[0].set_index("Input"), ref[1]


def test_a_press_held_across_chunks():
    r = beat(0, 12000)
    r += [x for t in range(3000, 3360, 90) for x in walk(4, t)]
    r += press(4, 2990, 185) + [(2990.1, 45, 4)]                         # 3 min through 2 walks
    r += press(6, 4990, 400) + [(4990.1, 45, 6)]                         # by length, 0 walks of its own
    r += press(8, 6990, 100) + [(6990.1, 45, 8)] + [x for t in (7000, 7030, 7060) for x in walk(8, t)]    # by walks
    r += press(10, 8000, 170) + [(8000.1, 45, 10)] + walk(10, 8010) + walk(10, 8100)      # 150-179 s, 2 walks: suspect
    for ch in (4, 6, 8, 10):
        r += waits(ch, 9000)
    o, per = run_scene(r, cut_s=(2990, 3000, 3090, 4990, 5000, 7000.0, 8000))
    assert o.loc[4, "status"] == "bad" and "held 185 s through" in o.loc[4, "reason"]
    assert o.loc[6, "status"] == "bad"
    assert o.loc[8, "status"] == "bad" and "through 3 walks" in o.loc[8, "reason"]
    assert o.loc[10, "status"] == "suspect" and "long hold" in set(per.condition)


def test_a_press_still_on_at_the_end_and_an_off_with_no_on():
    r = beat(0, 7200)
    r += [x for t in range(100, 7200, 100) for x in walk(2, t)]          # P2 walks every cycle from t = 100
    r += press(2, 50) + [(50.1, 45, 2)] + [(90, 90, 2), (90.1, 45, 2)]   # ON at 90, never OFF
    o, per = run_scene(r, cut_s=(90, 100, 3600))
    assert o.loc[2, "status"] == "bad" and "still ON when the log ends" in o.loc[2, "reason"] and "every cycle" in o.loc[2, "reason"]
    r = beat(0, 7200) + [x for t in range(100, 3600, 100) for x in walk(2, t)] + [(3650, 89, 2)] + press(2, 4000) + [(4000.1, 45, 2)]
    r += [(t, 1, 2) for t in range(4100, 7200, 100)]
    o, per = run_scene(r, cut_s=(3600, 3650, 4000))
    assert o.loc[2, "status"] == "bad" and "already ON when the log starts" in o.loc[2, "reason"]


def test_a_call_waiting_through_greens_across_chunks_is_a_note():
    r = beat(0, 6000) + press(8, 999.9) + [(1000, 45, 8), (1100, 1, 8), (1200, 1, 8), (1300, 1, 8)] + walk(8, 1400) + waits(8, 3000)
    o, per = run_scene(r, cut_s=(1000, 1100, 1200, 1300))
    assert o.loc[8, "status"] == "ok" and "low confidence" in o.loc[8, "config"] and "call dropped" in set(per.condition)


def test_a_preemption_with_no_logged_end_crosses_chunks():
    r = beat(0, 30000) + press(8, 16999.9) + [(17000, 45, 8), (17100, 1, 8), (17200, 1, 8), (17300, 1, 8), (15000, 102, 7)] + walk(8, 17400)
    r += waits(8, 20000)
    o, per = run_scene(r, cut_s=(15000, 17000, 17200))
    assert "omitted during preemption" in o.loc[8, "config"] and "omitted during preemption" in set(per.condition)


def test_a_hole_and_a_log_gap_across_a_cut():
    r = beat(0, 1030) + beat(1065, 3000) + beat(3400, 9000)              # silent 1020-1065 and 2990-3400
    r += press(4, 999.9) + [(1000, 45, 4), (1070, 1, 4), (1140, 1, 4), (1210, 1, 4)] + walk(4, 1300)     # the wait ends at the hole
    r += press(6, 2900, 700) + [(2900.1, 45, 6)] + walk(6, 3650)         # ON across the gap: its length is not known
    r += waits(4, 5000) + waits(6, 5000)
    o, per = run_scene(r, cut_s=(1020, 1030, 1064, 1065, 2990, 3000, 3399, 3400))
    assert o.loc[4, "status"] == "ok" and "waited" not in o.loc[4, "config"] and o.loc[6, "status"] == "ok"
    off = run_scene(r, cut_s=(1030, 3000), options={"log_gaps": False})[0]
    assert "waited through" in off.loc[4, "config"] and off.loc[6, "status"] == "bad"


def test_an_orphan_clearance_and_a_split_walk():
    r = beat(0, 9000) + press(8, 999.9) + [(1000, 45, 8), (1053, 1, 8), (1060, 22, 8), (1080, 23, 8)]    # the 21 lost, its green in the chunk before
    r += [(t, 1, 8) for t in range(1200, 4800, 100)] + press(8, 4899.9) + [(4900, 45, 8)] + walk(8, 4920) + waits(8, 6000)
    o, per = run_scene(r, cut_s=(1053, 1060))
    assert o.loc[8, "status"] == "ok" and "waited" not in o.loc[8, "config"]
    r = beat(0, 4000) + press(2, 100) + [(100.1, 45, 2)] + walk(2, 120) + waits(2, 1000)     # cut between the 21 and the 22 23
    o, per = run_scene(r, cut_s=(121, 127.5, 140))
    assert o.loc[2, "status"] == "ok"


def test_a_log_without_overlap_walks_or_preempt_codes():
    r = beat(0, 3000) + [x for t in (100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300) for x in press(6, t)]
    o, _ = run_scene(r, cut_s=(1000,))
    assert o.loc[6, "status"] == "ok" and "never calls a walk not checked" in o.loc[6, "config"]
    o, _ = run_scene(r + [(2500, 67, 9)], cut_s=(1000,))
    assert o.loc[6, "status"] == "config_issue"


def test_a_sudden_stop_across_chunks():
    r = [(t, 1, 2) for t in range(0, 3 * 86400 - 6 * 3600, 30)]
    for day in (0, 1, 2):
        stop = 19 if day == 0 else 10 if day == 1 else 7
        for h in range(7, stop):
            for m in range(5):
                r += press(6, day * 86400 + (h - 6) * 3600 + m * 720 + (h * 37 + m * 91) % 200)
    o, per = run_scene(r, cut_s=(100000, 130000))
    assert o.loc[6, "status"] in ("suspect", "bad") and "sudden stop" in set(per.condition)


def test_events_of_codes_nothing_reads_still_count_as_logging():
    """code 200 is in no list: it is not kept, but it keeps the signal from a hole, as in a batch run; a signal with no allowed event at all"""
    r = [(t, 200, 1) for t in range(0, 6000, 10)] + press(4, 999.9) + [(1000, 45, 4), (1070, 1, 4), (1140, 1, 4), (1210, 1, 4)] + walk(4, 1300) + waits(4, 3000)
    o, _ = run_scene(r, cut_s=(1000, 1070))
    assert "waited through" in o.loc[4, "config"]
    c = con_of([(t, 200, 1) for t in range(0, 6000, 10)] + press(4, 100) + [(100.1, 45, 4), (1000, 21, 4), (1007, 22, 4), (1020, 23, 4)])
    s = fed(c, [T0 + pd.Timedelta(minutes=30), T0 + pd.Timedelta(hours=2)])
    assert s[0].events_in == 0 and s[0].ped_events_in > 0 and I._through(c, ST) is not None
    assert c.execute(f"SELECT count(*) FROM {ST}.health_ped_events WHERE EventId = 200").fetchall()[0][0] == 0
    r2 = I.judge_rows(c, "detectors", end=T0.floor("15min") + pd.Timedelta(hours=2), horizon_days=1)
    assert r2.detectors.empty and len(r2.ped)


# --- T5.6 ---------------------------------------------------------------------------------------------------------------------
def test_the_chunk_is_read_once():
    """the ped part reads the update's temp table of the chunk, not the chunk; the only other read of the chunk counts the unusable stamps"""
    ops = I._ops(ST, ped=True)
    assert [n for n, s in ops.items() if "@EVENTS@" in s] == ["uev", "unusable"]
    assert "EventId IN" not in ops["uev"]
    u = I._update_plan(ST, "uev_table", "detectors", None, "VARCHAR")
    assert all("@PEDWIN@" not in s or "health_carry" not in s for _, s, _ in u.ctes)


def test_every_statement_runs_on_an_empty_store():
    con = con_of(beat(0, 100))
    q = P._steps("x", None, False, "s.")
    con.execute("ATTACH ':memory:' AS s")
    con.execute(I._render.render("store.sql", device_type="VARCHAR", s="s."))
    con.execute(f"CREATE TEMP TABLE pq_wpart AS SELECT 1 AS dev, 'S' AS DeviceId, 0.0::DOUBLE AS s, 1.0::DOUBLE AS e, TIMESTAMP '2026-01-01' AS lo, TIMESTAMP '2026-01-02' AS hi")
    con.execute("CREATE TEMP TABLE x AS SELECT TIMESTAMP '2026-01-01' AS Timestamp, 'S' AS DeviceId, 1 AS EventId, 1 AS Parameter WHERE false")
    for n, s in q:
        con.execute(f"CREATE OR REPLACE TEMP TABLE pq_{n} AS {s}")


# --- the vehicle answers, a store with ped off ------------------------------------------------------------------------------
def test_ped_on_leaves_the_vehicle_answers_as_they_were():
    E = syn()
    con = TI.connect(E["ev"], E["det"])
    fed(con, grid(E["a"], E["T"], "60min"), store="p0", horizon_days=3, ped=False)
    fed(con, grid(E["a"], E["T"], "60min"), store="p1", horizon_days=3)
    T = E["T"]
    r0, r1 = I.judge_rows(con, "detectors", store="p0", end=T, horizon_days=2), I.judge_rows(con, "detectors", store="p1", end=T, horizon_days=2)
    same_outputs(r0, r1, "ped on or off")
    same_outputs(r1, H.health(con, "events", "detectors", start=T - 2 * DAY, end=T, periods=True, bin_origin="clock"), "against the batch")
    for t in I.FEATURES:
        q = "SELECT * FROM {}.{} WHERE {} ORDER BY ALL".format("{}", t, "Kind <> 'silence'" if t == "health_intervals" else "true")
        assert con.execute(q.format("p0")).fetchall() == con.execute(q.format("p1")).fetchall(), t
    assert r0.ped is None and r0.ped_periods is None and r1.ped is not None
    assert con.execute("SELECT count(*) FROM p0.health_ped_events").fetchall()[0][0] == 0
    assert con.execute("SELECT count(*) FROM p0.health_carry WHERE EventId = 9401").fetchall()[0][0] == 0
    assert I.judge_rows(con, "detectors", store="p1", end=T, horizon_days=2, ped=False).ped is None
    with pytest.raises(ValueError, match="ped"):
        I.judge_rows(con, "detectors", store="p0", end=T, horizon_days=2, ped=True)
    with pytest.raises(ValueError, match="ped"):
        I.update(con, "events", "detectors", store="p0", ped=True)
    with pytest.raises(ValueError, match="ped"):
        I.attach_store(con, name="p1", ped=False)


# --- late, re-delivered, wrongly stamped, one transaction --------------------------------------------------------------------
def scene_rows():
    return beat(0, 6000) + press(8, 999.9) + [(1000, 45, 8), (1100, 1, 8), (1200, 1, 8)] + walk(8, 1400) + waits(8, 3000)


def test_rows_delivered_twice_are_late_and_counted():
    c = con_of(scene_rows())
    a, T = T0.floor("15min"), T0 + pd.Timedelta(hours=2)
    fed(c, [T0 + pd.Timedelta(minutes=30), T])
    before = c.execute(f"SELECT * FROM {ST}.health_ped_events ORDER BY ALL").fetchall()
    s = I.update(c, "events", "detectors", upto=T)
    assert s.ped_events_in == 0 and s.ped_events_late["S"] > 0 and s.events_in == 0 and not s.rows_written
    assert c.execute(f"SELECT * FROM {ST}.health_ped_events ORDER BY ALL").fetchall() == before
    same_ped(ped_of(c, T.ceil("15min"), 1), batch(c, T.ceil("15min"), 1), "twice")


def test_a_wrong_old_stamp_on_a_ped_row_is_held_back():
    c = con_of(scene_rows())
    T = T0 + pd.Timedelta(hours=2)
    fed(c, [T0 + pd.Timedelta(minutes=30)], horizon_days=1)
    c.execute("INSERT INTO events VALUES ('S', TIMESTAMP '2015-01-01', 90, 3)")
    I.update(c, "events", "detectors", upto=T)
    assert c.execute(f"SELECT count(*) FROM {ST}.health_ped_events WHERE year(TimeStamp) = 2015").fetchall()[0][0] == 0
    same_ped(ped_of(c, T, 1), batch(c, T, 1), "old stamp")


def test_a_signal_starting_over_leaves_no_silence_across_the_years():
    """its ped watermark is past believing, as a vehicle one: the carry's last event is not read, no silence runs from it"""
    c = con_of(scene_rows())
    c.execute("INSERT INTO events SELECT 'R', TIMESTAMP '2019-06-01 00:00:00' + INTERVAL (i) MINUTE, 82, 1 FROM range(3) t(i)")
    I.attach_store(c, horizon_days=1)
    c.execute("CREATE TABLE a0 AS SELECT * FROM events WHERE DeviceId = 'R'")
    I.update(c, "a0", "detectors")
    c.execute("CREATE TABLE a1 AS SELECT * FROM events WHERE DeviceId = 'S' UNION ALL SELECT 'R', T + INTERVAL 5 SECOND, 82, 1 FROM (SELECT ?::TIMESTAMP AS T)", [T0])
    I.update(c, "a1", "detectors")
    got = c.execute(f"SELECT StartTime FROM {ST}.health_intervals WHERE Kind = 'silence' AND DeviceId = 'R' AND year(StartTime) = 2019 AND year(EndTime) = 2026").fetchall()
    assert not got
    assert c.execute(f"SELECT year(TimeStamp) FROM {ST}.health_carry WHERE EventId = 9401 AND DeviceId = 'R'").fetchall() == [(2026,)]


def test_a_failure_before_commit_loses_nothing():
    rows = scene_rows()
    c = con_of(rows)
    a, b = T0 + pd.Timedelta(minutes=30), T0 + pd.Timedelta(hours=2)
    fed(c, [a])
    q = f"SELECT * FROM {ST}.health_ped_events ORDER BY ALL"
    before, carry = c.execute(q).fetchall(), TI.carry_rows(c, ST)
    with pytest.raises(RuntimeError, match="injected"):
        I.update(FailBefore(c, "COMMIT"), "events", "detectors", upto=b)
    assert c.execute(q).fetchall() == before and TI.carry_rows(c, ST) == carry
    I.update(c, "events", "detectors", upto=b)
    clean = con_of(rows)
    fed(clean, [a, b])
    assert c.execute(q).fetchall() == clean.execute(q).fetchall() and TI.carry_rows(c, ST) == TI.carry_rows(clean, ST)
