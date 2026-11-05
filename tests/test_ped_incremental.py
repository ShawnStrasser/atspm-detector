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


def test_the_log_keeps_every_code_the_ped_modules_read():
    assert set(P.CODES) | set(A.CODES) == set(P.LOG_CODES)


# --- T5.3, T5.4 ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cut", few_then_slow(["hour", "random", "15 min"], 1))
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


# --- late, re-delivered, wrongly stamped, one transaction --------------------------------------------------------------------
def scene_rows():
    return beat(0, 6000) + press(8, 999.9) + [(1000, 45, 8), (1100, 1, 8), (1200, 1, 8)] + walk(8, 1400) + waits(8, 3000)








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
