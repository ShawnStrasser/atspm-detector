"""Option preempt_holds (on by default, notes 162 / 163): time a detector is held ON inside a preemption is not stuck-on time (a
queue behind a train); a batch run, a store fed in any cutting (its ped log keeps the preempt events) and the live 15-minute
rule agree."""
import duckdb
import pandas as pd
import pytest

import make_golden as G
from atspm_detector import health_duckdb as hq
from atspm_detector import incremental as I

from synth import make
from test_incremental import grid, same_outputs


def _log():
    """one synthetic day; det 5 (P4 Presence) held ON 01:00-02:30 inside preempt 3 (01:00-02:29), det 6 (P8 Presence)
    held ON 03:00-04:30 with preempt 3 only 03:00-03:10 (a stuck detector that a short preemption must not clear)."""
    ev, det = make(n_signals=1, days=1, seed=3, faults=False)
    t0 = ev.Timestamp.min().normalize()
    S = lambda h, m=0: t0 + pd.Timedelta(hours=h, minutes=m)
    keep = ~((ev.EventId.isin([81, 82]) & (((ev.Parameter == 5) & (ev.Timestamp >= S(0, 59)) & (ev.Timestamp < S(2, 31)))
                                          | ((ev.Parameter == 6) & (ev.Timestamp >= S(2, 59)) & (ev.Timestamp < S(4, 31)))))
             | ev.EventId.between(102, 111))
    rows = [(S(1), 82, 5), (S(2, 30), 81, 5), (S(3), 82, 6), (S(4, 30), 81, 6),
            (S(1), 102, 3), (S(1), 105, 3), (S(2, 29), 104, 3), (S(3), 102, 3), (S(3), 105, 3), (S(3, 10), 104, 3)]
    add = pd.DataFrame(rows, columns=["Timestamp", "EventId", "Parameter"]).assign(DeviceId="S00")
    ev = pd.concat([ev[keep], add], ignore_index=True).sort_values(["Timestamp", "EventId"], kind="stable")
    return ev.reset_index(drop=True), det


def _run(ev, det, on):
    con = duckdb.connect()
    con.register("e", ev)
    con.register("d", det)
    out, per = hq.health(con, "e", "d", options={"preempt_holds": on}, periods=True)
    return out.set_index("Detector"), per


def test_hold_inside_a_preemption_is_not_stuck():
    ev, det = _log()
    off, poff = _run(ev, det, False)
    on, pon = _run(ev, det, True)
    stuck = lambda p, d: ((p.Detector == d) & (p.condition == "stuck on")).any()
    assert stuck(poff, 5) and stuck(poff, 6)              # without the option both holds are stuck on
    assert not stuck(pon, 5)                              # held only while the preemption ran: a queue, not a fault
    assert on.loc[5, "health_status"] == "ok"
    assert stuck(pon, 6)                                  # 80 of its 90 min outside the preemption: still stuck on
    assert not ((pon.Detector == 5) & (pon.condition == "silent")).any()   # the hold is still a hold, not a silence
    # nothing else moves
    rest = [c for c in off.index if c != 5]
    assert off.loc[rest, "health_status"].equals(on.loc[rest, "health_status"])


def test_no_preempt_events_changes_nothing():
    ev, det = make(n_signals=1, days=1, seed=4, faults=True)
    ev = ev[~ev.EventId.between(102, 111)]
    a, pa = _run(ev, det, False)
    b, pb = _run(ev, det, True)
    assert a.equals(b) and pa.reset_index(drop=True).equals(pb.reset_index(drop=True))


# --- a store (incremental): the preemptions come from its ped log ------------------------------------------------------------

def _store_log():
    """_log() (and a copy of it, signal S01) plus a preemption still running at the day's end (23:00, no end logged) with det 5 held ON from 23:05 to the end"""
    ev, det = _log()
    t0 = ev.Timestamp.min().normalize()
    S = lambda h, m=0: t0 + pd.Timedelta(hours=h, minutes=m)
    ev = ev[~(ev.EventId.isin([81, 82]) & (ev.Parameter == 5) & (ev.Timestamp >= S(23, 4)))]
    add = pd.DataFrame([(S(23), 102, 2), (S(23, 5), 82, 5)], columns=["Timestamp", "EventId", "Parameter"]).assign(DeviceId="S00")
    ev = pd.concat([ev, add], ignore_index=True)
    ev = pd.concat([ev, ev.assign(DeviceId="S01")], ignore_index=True)        # a second signal, for the worker processes
    det = pd.concat([det, det.assign(DeviceId="S01")], ignore_index=True)
    return ev.sort_values(["DeviceId", "Timestamp", "EventId"], kind="stable").reset_index(drop=True), det, t0 + pd.Timedelta(days=1)


def _fed(ev, det, cuts, ped=True):
    con = G.connect(ev, det)
    con.execute("SET memory_limit = '2GB'")
    I.attach_store(con, ped=ped)
    for c in cuts:
        I.update(con, "events", "detectors", upto=c, horizon_days=1)
    return con


def _fed_day():
    ev, det, T = _store_log()
    t0 = T - pd.Timedelta(days=1)
    cuts = grid(t0, T, "37min") + [t0 + pd.Timedelta(hours=1, minutes=30)]       # a cut inside a preemption too
    return _fed(ev, det, sorted(set(cuts))), T


def test_a_store_judges_as_a_batch_run():
    con, T = _fed_day()
    hd = 1
    got = I.judge_rows(con, "detectors", end=T, horizon_days=hd)
    want = hq.health(con, "events", "detectors", start=T - pd.Timedelta(days=hd), end=T, periods=True, bin_origin="clock")
    same_outputs(got, want)
    per = got[1]
    stuck = lambda d: ((per.DeviceId == "S00") & (per.Detector == d) & (per.condition == "stuck on")).any()
    assert not stuck(5) and stuck(6)      # det 5: held only during preemptions (one still running at the end); det 6: stuck
    # judged by 2 worker processes, a signal each: the ped log goes with them
    par = I.judge_rows(con, "detectors", end=T, horizon_days=hd, workers=2, max_events=1)
    same_outputs(par, got)


def test_a_store_without_its_ped_log_judges_as_before():
    ev, det, T = _store_log()
    con = _fed(ev, det, [T], ped=False)
    got = I.judge_rows(con, "detectors", end=T, horizon_days=1)
    same_outputs(got, hq.health(con, "events", "detectors", start=T - pd.Timedelta(days=1), end=T, periods=True, bin_origin="clock",
                                options={"preempt_holds": False}))


def test_a_store_knows_a_preemption_from_before_its_window():
    """the window starts at 01:30, inside preempt 3 (01:00-02:29) and det 5's hold (01:00-02:30): the store carries the ON from 01:00
    (a batch run knows no ON before its window) and reads the preemption's start from its ped log, so the hold is still no fault"""
    con, T = _fed_day()
    for on, stuck in ((False, True), (True, False)):
        per = I.judge_rows(con, "detectors", end=T, horizon_days=22.5 / 24, options={"preempt_holds": on})[1]
        held = per[(per.DeviceId == "S00") & (per.Detector == 5) & (per.condition == "stuck on")]
        assert (held.start == T - pd.Timedelta(hours=23)).any() == stuck


# --- live (15-minute runs) ---------------------------------------------------------------------------------------------------

def test_live_stuck_on_counts_only_the_time_outside_a_preemption(tmp_path):
    """det 9 held ON from BASE, limit 300 s; a preemption BASE to BASE + 40 min (+ 2 min recovery): no row until its time outside
    passes 300 s (the 4th bin: 45 - 42 = 3 min outside at the 3rd bin's end), every bin without the preemption"""
    import test_live as TL
    held = TL.with_events(TL.beat("S1", TL.BASE, TL.BASE + 4 * TL.Q), "S1", (TL.BASE, 82, 9))
    train = TL.with_events(held, "S1", (TL.BASE, 102, 1), (TL.BASE, 105, 1), (TL.BASE + pd.Timedelta(minutes=40), 104, 1))
    count = lambda ev, d: [len(TL.of(r, "Vehicle", 9, "stuck on")) for r in TL.bins_in_turn(ev, TL.marks(), d, 4)]
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    assert count(held, tmp_path / "a") == [1, 1, 1, 1]
    assert count(train, tmp_path / "b") == [0, 0, 0, 1]
