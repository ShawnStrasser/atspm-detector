"""Pedestrian push-button health (ped_health_duckdb, note 155) on small synthetic logs."""
import re

import duckdb
import pandas as pd

from atspm_detector.ped_health_duckdb import ped_health

T0 = pd.Timestamp("2026-09-28 06:00")


def _con(rows, dev="S"):
    ev = pd.DataFrame(rows, columns=["t", "EventId", "Parameter"])
    ev = pd.DataFrame({"DeviceId": dev, "Timestamp": T0 + pd.to_timedelta(ev.t, unit="s"), "EventId": ev.EventId,
                       "Parameter": ev.Parameter})
    con = duckdb.connect()
    con.register("e_df", ev)
    con.execute("CREATE TABLE events AS SELECT * FROM e_df ORDER BY DeviceId, Timestamp")
    return con


def _press(ch, t, dur=0.2):
    return [(t, 90, ch), (round(t + dur, 1), 89, ch)]


def _walk(ph, t):
    return [(t - 1, 1, ph), (t, 21, ph), (t + 7, 22, ph), (t + 20, 23, ph)]


def _beat(t0, t1, step=10):
    """the signal logging something else (a vehicle detector event) every `step` s: no log hole (>= 30 s silent)"""
    return [(t, 82, 1) for t in range(t0, t1, step)]


def _log():
    r = _beat(0, 10000)
    for k in range(12):                       # ped 2: a normal button, each press placed a P2 call, walk 20 s later
        t = 100.0 * k
        r += _press(2, t) + [(t + 0.1, 45, 2)] + _walk(2, t + 20)
    r += _press(4, 10, 400)                   # ped 4: held 400 s
    r += _press(8, 999.9) + [(1000, 45, 8), (1100, 1, 8), (1200, 1, 8), (1300, 1, 8)]   # P8 call stands 3 greens
    r += _press(8, 1399.9) + [(1400, 45, 8), (1420, 21, 8), (1427, 22, 8), (1440, 23, 8)]
    for k in range(26):                       # ped 3: a press every 60.0 s, never a call
        r += _press(3, 3000 + 60.0 * k)
    for t in (5000, 5003, 5004, 5010, 5011, 5013, 5030, 5031, 5032, 5040, 5050, 5051):   # ped 6: no call, no walk
        r += _press(6, t)
    for k in range(12):                       # ped 9: no ped call; ped overlap 10 walks 30 s later
        t = 6000 + 200.0 * k
        r += _press(9, t) + [(t + 30, 67, 10)]
    for t in (8500, 8600, 8700):              # ped 11: three presses
        r += _press(11, t)
    for k in range(12):                       # ped 12 and ped 13: the same 1.5-s ONs
        r += _press(12, 9000 + 50.0 * k, 1.5) + _press(13, 9000 + 50.0 * k, 1.5) + [(9000.1 + 50.0 * k, 45, 4)]
    return r


def _by_input(out):
    return out.set_index("Input")


def test_checks_on_a_small_log():
    out, per = ped_health(_con(_log()), periods=True)
    o = _by_input(out)
    assert list(out.columns) == ["DeviceId", "Input", "Phase", "Type", "presses", "status", "reason", "config"]
    assert o.loc[2, "status"] == "ok" and o.loc[2, "Phase"] == 2 and o.loc[2, "Type"] == "unknown"
    assert o.loc[4, "status"] == "bad" and "stuck ON: held 7 min" in o.loc[4, "reason"]
    # a call that waited is a note, never a status; a single wait reads low confidence (user review 2026-11-02)
    assert o.loc[8, "status"] == "not_enough_data" and "possible configuration issue (low confidence)" in o.loc[8, "config"]
    assert "waited through 3 greens" in o.loc[8, "config"]
    assert "preemption not checked: the log holds no preempt events" in o.loc[8, "config"]   # no preempt event at all
    assert o.loc[3, "status"] == "bad" and "every 60.0 s" in o.loc[3, "reason"]   # machine-regular: bad, no call
    assert "never" not in o.loc[3, "reason"]          # machine-regular, not also "never calls"
    assert o.loc[6, "status"] == "config_issue" and "no ped call or walk" in o.loc[6, "reason"]
    assert o.loc[9, "status"] == "ok" and "ped overlap 10" in o.loc[9, "config"]
    assert o.loc[11, "status"] == "not_enough_data"
    assert "same button as ped 13" in o.loc[12, "config"] and o.loc[12, "status"] == "ok"
    assert list(per.columns) == ["DeviceId", "Detector", "start", "end", "condition", "severity", "limit_s",
                                 "in_closure", "whole_window"]
    s = per[per.Detector == 4].iloc[0]
    assert s.condition == "stuck on" and s.start == T0 + pd.Timedelta(seconds=10) and s.limit_s == 300
    assert s.severity >= 0.8
    d = per[per.Detector == 8].iloc[0]
    assert d.condition == "call dropped" and d.start == T0 + pd.Timedelta(seconds=1000)
    assert d.end == T0 + pd.Timedelta(seconds=1400)
    assert set(per.condition) == {"stuck on", "call dropped", "machine-regular presses", "never calls a walk"}
    assert not per.in_closure.any() and not per.whole_window.any()


def test_stuck_at_the_end_of_the_log_and_by_walks():
    r = _beat(0, 2000) + _press(2, 0) + [(0.1, 45, 2)] + _walk(2, 20) + [(10, 90, 5), (10.1, 45, 2)]  # ped 5 never OFF
    r += [x for t in range(100, 2000, 100) for x in _walk(2, t)]       # ... while P2 walks every 100 s
    r += [(30, 90, 6), (30.1, 45, 6)] + [(t, 1, 6) for t in range(100, 2000, 100)]   # ped 6: no OFF, no walks
    for t in (1030, 1060, 1150):                                       # ped 7: ON 250 s while P2 walks
        r += _walk(2, t)
    r += _press(7, 990, 250) + [(990.1, 45, 2)]
    o = _by_input(ped_health(_con(r)))
    assert o.loc[5, "status"] == "bad" and "still ON when the log ends" in o.loc[5, "reason"]
    assert o.loc[6, "status"] != "bad"                                # a lost OFF: no walks while "ON"
    assert o.loc[7, "status"] == "bad" and "held 250 s through 6 walks" in o.loc[7, "reason"]


def test_sudden_stop_against_its_own_level():
    """the hourly dead rule is gone (user: 4-7 h gaps are too short to flag): a button stops against what its own presses
    at those times of day expect"""
    r = [(t, 1, 2) for t in range(0, 3 * 86400 - 6 * 3600, 30)]        # logging all the time (from 06:00)
    for day in (0, 1, 2):
        stop = 19 if day == 0 else 10 if day == 1 else 7               # stops on day 1 at 10:00, for good
        for h in range(7, stop):
            for m in range(5):                                         # 5 separate presses an hour
                r += _press(6, day * 86400 + (h - 6) * 3600 + m * 720 + (h * 37 + m * 91) % 200)
    out, per = ped_health(_con(r), periods=True)
    stop = per[per.condition == "sudden stop"]
    assert len(stop) == 1 and stop.iloc[0].start == pd.Timestamp("2026-09-29 10:00")
    assert out.iloc[0].status in ("suspect", "bad") and "no presses for" in out.iloc[0].reason
    assert "the end of the log" in out.iloc[0].reason
    off = ped_health(_con(r), options={"stop_test": False})
    assert off.iloc[0].status == "ok"


def test_a_stop_flagged_only_by_the_simple_rule_sits_at_the_suspect_limit():
    """24 logged hours at 30+ presses a day flag whatever the chance says: its severity is 0.5 at least, as for every suspect limit"""
    r = [(t, 1, 2) for t in range(0, 3 * 86400 - 6 * 3600, 30)]
    for h in range(7, 19):                                             # 4 presses an hour on day 0, silent from then on
        for m in range(4):
            r += _press(6, (h - 6) * 3600 + m * 900 + (h * 37 + m * 91) % 200)
    for h in range(7, 19):                                             # and again on the last day: the run has an end
        for m in range(4):
            r += _press(6, 2 * 86400 + (h - 6) * 3600 + m * 900 + (h * 37 + m * 91) % 200)
    out, per = ped_health(_con(r), periods=True)
    stop = per[per.condition == "sudden stop"]
    assert len(stop) == 1 and out.iloc[0].status == "suspect"
    assert 1 / 1000 < 1 / float(re.search(r"chance about 1 in ([\d,]+)", out.iloc[0].reason).group(1).replace(",", ""))
    assert stop.iloc[0].severity == 0.5


def test_a_long_hold_sits_at_the_suspect_limit():
    r = _beat(0, 1000) + _press(4, 100, 166) + [(100.1, 45, 4)] + _walk(4, 130) + _walk(4, 200)
    out, per = ped_health(_con(r), periods=True)
    hold = per[per.condition == "long hold"]
    assert len(hold) == 1 and out.iloc[0].status == "suspect" and hold.iloc[0].severity == 0.5


def test_a_short_silence_of_a_busy_button_is_not_a_fault():
    r = [(t, 1, 2) for t in range(0, 2 * 86400 - 6 * 3600, 30)]
    for day in (0, 1):
        for h in range(7, 19):
            if day == 1 and 10 <= h < 14:                              # 4 h without a press
                continue
            for m in range(5):
                r += _press(6, day * 86400 + (h - 6) * 3600 + m * 720 + (h * 37 + m * 91) % 200)
    assert ped_health(_con(r)).iloc[0].status == "ok"


def test_stuck_on_with_no_walks_is_bad_with_a_configuration_note():
    r = _beat(0, 3000) + _press(4, 100, 400) + _press(4, 1000) + _press(6, 1500, 400) + [(1500.1, 45, 6)] + _walk(6, 1600) + _walk(6, 1700)
    o = _by_input(ped_health(_con(r)))
    assert o.loc[4, "status"] == "bad" and "stuck ON through 0 walks" in o.loc[4, "config"]
    assert o.loc[6, "status"] == "bad" and "0 walks" not in o.loc[6, "config"]


def test_groups_and_window():
    rows = _log()
    a = ped_health(_con(rows))
    con = _con(rows)
    con.execute("INSERT INTO events SELECT 'T' AS DeviceId, Timestamp, EventId, Parameter FROM events")
    b = ped_health(con, max_events=50)                       # one signal per group
    assert len(b) == 2 * len(a)
    pd.testing.assert_frame_equal(b[b.DeviceId == "S"].reset_index(drop=True), a)
    c = ped_health(_con(rows), start=T0 + pd.Timedelta(seconds=2900), end=T0 + pd.Timedelta(seconds=5500))
    assert set(c.Input) == {3, 6}


def test_first_pass_rules_when_switched_off():
    o = _by_input(ped_health(_con(_log()), options={"regular_bad": False, "config_issue": False,
                                                    "preempt_wait": False, "stuck_2walks": False,
                                                    "orphan_clearance": False, "log_gaps": False}))
    assert o.loc[3, "status"] == "suspect" and o.loc[6, "status"] == "suspect"
    assert "configuration" not in o.loc[6, "reason"] and "preemption" not in o.loc[8, "config"]


def _wait(ph, tc, greens, t_end, pre=()):
    """a ped call for `ph` at tc (placed by a press of input ph), green starts at `greens`, then the walk at t_end;
    pre: preempt events (t, code, preempt number)"""
    return (_press(ph, tc - 0.1) + [(tc, 45, ph)] + [(t, 1, ph) for t in greens]
            + [(t_end - 1, 1, ph), (t_end, 21, ph), (t_end + 7, 22, ph), (t_end + 20, 23, ph)] + list(pre))


def test_a_preemption_excuses_the_whole_wait():
    r = _beat(0, 66000)
    # ped 2: the call waits 3 greens, a preemption (102 -> 104) starts before the first of them: a note, not a fault
    r += _wait(2, 1000, (1100, 1200, 1300), 1400, [(1050, 102, 1), (1150, 104, 1)])
    # ped 4: 3 greens, then a preemption begins before the walk: any preemption during the wait excuses all of it
    r += _wait(4, 3000, (3050, 3100, 3150, 3300), 3400, [(3200, 102, 2), (3250, 104, 2)])
    # ped 6: only preempt entry / exit (105 / 111) logged, the call comes during its 120-s recovery: a note
    r += _wait(6, 5010, (5050, 5100), 5400, [(4900, 105, 3), (5000, 111, 3)])
    # ped 8: 3 greens, no preemption at all: call dropped (a note)
    r += _wait(8, 7000, (7100, 7200, 7300), 7400)
    # ped 10: 3 greens, a preemption that ended (+120 s recovery) before the call: does not excuse it
    r += _wait(10, 9500, (9600, 9700, 9800), 9900, [(9000, 102, 1), (9100, 104, 1)])
    # ped 12: a preemption at the start of a long wait (call 11000, preemption 10990-11010), then 5 greens without a
    # walk: the whole wait is excused, with the label
    r += _wait(12, 11000, (11200, 11300, 11400, 11500, 11600), 11700, [(10990, 102, 1), (11010, 104, 1)])
    # ped 14: preempt 4 on 12990 - off 13030 with its entry, but its exit (111) is missing: the next 111 comes 1.7 h
    # later.  The preemption ends at its input off (+ 120 s recovery = 13150); a call at 13300 is after it
    r += _wait(14, 13300, (13400, 13500, 13600), 13700, [(12990, 102, 4), (12995, 105, 4), (13030, 104, 4),
                                                         (19000, 111, 4)])
    # ped 16: preempt 7 call on at 15000 and no end logged (off and exit missing): still on, for at most 6 h (to
    # 36600 + 120 s); a call at 17600 waits inside it: excused
    r += _wait(16, 17600, (17700, 17800, 17900), 18000, [(15000, 102, 7)])
    # ped 18: the same with preempt 8 on at 21000 (entry 105 logged too, no 104 / 111): a call 30 min later, excused
    r += _wait(18, 22800, (22900, 23000), 23100, [(21000, 102, 8), (21000, 105, 8)])
    # ped 20: preempt 9 entry at 40000, no end logged: still on to 6 h later (61600 + 120 s); a call at 62000 is after
    # it: call dropped
    r += _wait(20, 62000, (62100, 62200, 62300), 62400, [(40000, 105, 9)])
    for ch in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20):               # enough normal presses to be typed / judged
        for k in range(10):
            t = 30000 + 300 * k + ch
            r += _press(ch, t - 0.1) + [(t, 45, ch), (t + 5, 1, ch), (t + 5, 21, ch), (t + 12, 22, ch), (t + 25, 23, ch)]
    out, per = ped_health(_con(r), periods=True)
    o = _by_input(out)
    for ch, g in ((2, 3), (4, 4), (6, 2), (12, 5), (16, 3), (18, 2)):
        assert o.loc[ch, "status"] == "ok" and o.loc[ch, "config"] == (
            f"omitted during preemption: 1 ped call for P{ch} waited up to {g} greens while a preemption ran "
            f"(not a fault)")
    assert o.loc[8, "status"] == "ok" and "with no preemption during the wait" in o.loc[8, "config"]
    for ch in (10, 14, 20):
        assert o.loc[ch, "status"] == "ok" and "waited through 3 greens" in o.loc[ch, "config"]
    n = per[per.condition == "omitted during preemption"]
    assert sorted(n.Detector) == [2, 4, 6, 12, 16, 18] and (n.severity == 0).all()
    assert n[n.Detector == 12].iloc[0].end == T0 + pd.Timedelta(seconds=11700)    # the whole long wait, labelled
    assert sorted(per[per.condition == "call dropped"].Detector) == [8, 10, 14, 20]
    off = _by_input(ped_health(_con(r), options={"preempt_wait": False}))
    assert all("low confidence" in off.loc[ch, "config"] and off.loc[ch, "status"] == "ok" for ch in (2, 4, 6, 12, 16, 18))


def test_stuck_on_3_min_through_2_walks():
    r = _beat(0, 12000)
    r += [x for k in range(4) for x in _walk(4, 3000 + 90 * k)]        # P4 walks every 90 s
    r += _press(4, 2990, 185) + [(2990.1, 45, 4)]                      # ped 4: ON 185 s through 2 walks (3000, 3090)
    r += _press(6, 4990, 170) + [(4990.1, 45, 6)] + _walk(6, 5000) + _walk(6, 5100)   # ped 6: 170 s, 2 walks: suspect
    r += _press(8, 6990, 200) + [(6990.1, 45, 8)] + _walk(8, 7000)    # ped 8: 200 s, 1 walk
    for ch in (4, 6, 8):
        for k in range(10):
            t = 9000 + 300 * k + ch
            r += _press(ch, t - 0.1) + [(t, 45, ch)] + _walk(ch, t + 5)
    o = _by_input(ped_health(_con(r)))
    assert o.loc[4, "status"] == "bad" and "held 185 s through 2 walks" in o.loc[4, "reason"]
    assert "limit 5 min, 3 walks, or 3 min through 2 walks" in o.loc[4, "reason"]
    assert o.loc[6, "status"] == "suspect" and "held ON 170 s through 2 walks" in o.loc[6, "reason"]
    assert o.loc[8, "status"] == "ok"
    off = _by_input(ped_health(_con(r), options={"stuck_2walks": False}))
    assert off.loc[6, "status"] == "ok"
    assert off.loc[4, "status"] == "ok" and "limit" not in off.loc[4, "reason"]


def test_a_lost_walk_start_is_a_walk():
    # ped 8: the call at 1000 is served by a walk whose 21 is missing from the log (only 22 / 23 at 1060 / 1080);
    # ped 6: the call at 1010 is served by a walk whose 21 and 22 are both missing (only the 23 at 1090);
    # then P8 / P6 green without a ped call for an hour, until the next press
    r = _beat(0, 9000) + _press(8, 999.9) + [(1000, 45, 8), (1053, 1, 8), (1060, 22, 8), (1080, 23, 8)]
    r += _press(6, 1009.9) + [(1010, 45, 6), (1053, 1, 6), (1090, 23, 6)]
    r += [(t, 1, ph) for t in range(1200, 4800, 100) for ph in (6, 8)]
    for ph in (6, 8):
        r += _press(ph, 4899.9) + [(4900, 45, ph), (4919, 1, ph), (4920, 21, ph), (4927, 22, ph), (4940, 23, ph)]
        for k in range(10):
            t = 6000 + 300 * k + ph
            r += _press(ph, t - 0.1) + [(t, 45, ph), (t + 5, 1, ph), (t + 5, 21, ph), (t + 12, 22, ph), (t + 25, 23, ph)]
    o = _by_input(ped_health(_con(r)))                          # default (user ruling): lost data is never a finding
    assert o.loc[8, "status"] == "ok" and o.loc[6, "status"] == "ok"
    off = _by_input(ped_health(_con(r), options={"orphan_clearance": False}))
    assert "waited through" in off.loc[8, "config"] and "waited through" in off.loc[6, "config"]
    assert "waited" not in o.loc[8, "config"] and "waited" not in o.loc[6, "config"]


def test_a_log_gap_is_never_a_finding():
    r = _beat(0, 3000) + _beat(3300, 6000) + _beat(7000, 12000)          # the log is lost 3000-3300 and 6000-7000
    # ped 4: ON at 2900, OFF at 3400: 500 s, but the log was lost for 300 s of it (no walk logged): not stuck
    r += _press(4, 2900, 500) + [(2900.1, 45, 4)] + _walk(4, 3420)
    # ped 6: a call at 5900, the log is lost 6000-7000 (its walk in there), then 3 P6 greens with no call standing
    # until the next press: the wait ends at the gap, no finding
    r += _press(6, 5899.9) + [(5900, 45, 6)] + [(t, 1, 6) for t in (7100, 7200, 7300)]
    r += _press(6, 7399.9) + [(7400, 45, 6)] + _walk(6, 7420)
    for ch in (4, 6):
        for k in range(10):
            t = 9000 + 300 * k + ch
            r += _press(ch, t - 0.1) + [(t, 45, ch)] + _walk(ch, t + 5)
    o = _by_input(ped_health(_con(r)))
    assert o.loc[4, "status"] == "ok" and o.loc[6, "status"] == "ok"
    off = _by_input(ped_health(_con(r), options={"log_gaps": False}))
    assert off.loc[4, "status"] == "bad" and "waited through" in off.loc[6, "config"]
    assert "waited" not in o.loc[6, "config"]


def test_a_short_log_hole_is_never_a_finding():
    """a hole of 30 s or more without any event of the signal (a feed can lose 1-3 min blocks)"""
    r = _beat(0, 1030) + _beat(1065, 1500) + _beat(2000, 2045) + _beat(2065, 2500) + _beat(3000, 6000)
    # ped 4: call at 1000, the log silent 1020-1065 (45 s), then 3 P4 greens without a walk, walk at 1300: the wait
    # ends at the hole (its walk may be in it): no finding
    r += _wait(4, 1000, (1070, 1140, 1210), 1300)
    # ped 6: the same with a 25-s silence (2040-2065, common at a quiet signal): call dropped
    r += _wait(6, 2000, (2070, 2140, 2210), 2300)
    for ch in (4, 6):
        for k in range(10):
            t = 3000 + 300 * k + ch
            r += _press(ch, t - 0.1) + [(t, 45, ch)] + _walk(ch, t + 5)
    o = _by_input(ped_health(_con(r)))
    assert o.loc[4, "status"] == "ok" and "waited" not in o.loc[4, "config"] and "waited through 3 greens" in o.loc[6, "config"]
    off = _by_input(ped_health(_con(r), options={"log_gaps": False}))
    assert "waited through" in off.loc[4, "config"]


def test_a_lost_green_and_walk_start():
    """a block lost from the log takes the phase's green start, walk and clearance; only its don't walk (23)
    with the yellow is logged.  The walk is not placed at the phase's previous (already ended) green, before the call"""
    r = _beat(0, 6000)
    # P4: green 2000, yellow 2008; ped call 2030; (lost: green 2050, walk 2051, clearance 2058); don't walk + yellow
    # 2080; then greens every 100 s without a call standing, until the next press
    r += [(2000, 1, 4), (2008, 8, 4)] + _press(4, 2029.9) + [(2030, 45, 4), (2080, 8, 4), (2080, 23, 4)]
    r += [x for t in range(2150, 2550, 100) for x in ((t, 1, 4), (t + 8, 8, 4))]
    for k in range(10):
        t = 3000 + 300 * k
        r += _press(4, t - 0.1) + [(t, 45, 4)] + _walk(4, t + 5)
    o = _by_input(ped_health(_con(r)))
    assert o.loc[4, "status"] == "ok" and "waited" not in o.loc[4, "config"]
    off = _by_input(ped_health(_con(r), options={"orphan_clearance": False}))
    assert "waited through" in off.loc[4, "config"]


def test_late_call_option():
    """a call placed after the yellow that leads into its phase's green: with late_call that green is not owed"""
    r = _beat(0, 6000)
    # ped 2: call at 1000, a P4 yellow at 995 and the P2 green at 1003 (no walk), the next P2 green 1100 (no walk),
    # walk with the green at 1200: 2 greens without a walk, the first right after a call placed during the clearance
    r += _press(2, 999.9) + [(1000, 45, 2), (995, 8, 4), (1003, 1, 2), (1095, 8, 4), (1100, 1, 2)] + _walk(2, 1200)
    # ped 6: the same but the call (2000) came before the yellow (2005): that green was owed -> call dropped
    r += _press(6, 1999.9) + [(2000, 45, 6), (2005, 8, 4), (2010, 1, 6), (2095, 8, 4), (2100, 1, 6)] + _walk(6, 2200)
    for ch in (2, 6):
        for k in range(10):
            t = 3000 + 250 * k + ch
            r += _press(ch, t - 0.1) + [(t, 45, ch)] + _walk(ch, t + 5)
    o = _by_input(ped_health(_con(r)))                                   # default: on
    assert "waited" not in o.loc[2, "config"] and "waited through 2 greens" in o.loc[6, "config"]
    off = _by_input(ped_health(_con(r), options={"late_call": False}))
    assert "waited through 2 greens" in off.loc[2, "config"] and "waited through 2 greens" in off.loc[6, "config"]
    assert (o.status == "ok").all()                                      # never a status


def test_a_call_during_its_own_running_walk_is_served():
    """03064 ped 2: the call and the walk's clearance had the same timestamp; no wait, no finding"""
    r = _beat(0, 6000)
    # the 45 lands on the very tick the walk ends (the 22), then two greens with no call standing, then the next press
    r += [(1000, 1, 2), (1001, 21, 2), (1008, 22, 2), (1008, 45, 2), (1020, 23, 2), (1100, 1, 2), (1200, 1, 2)]
    for k in range(10):
        t = 3000 + 250 * k
        r += _press(2, t - 0.1) + [(t, 45, 2)] + _walk(2, t + 5)
    o = _by_input(ped_health(_con(r)))
    assert o.loc[2, "status"] == "ok" and o.loc[2, "config"] == ""


def test_preemption_wording_by_what_the_log_holds():
    """a call dropped at a signal without preempt events: 'none logged at this signal' when another signal of the log
    has some, 'not checked' when the log has none at all"""
    a = _beat(0, 5100) + _wait(8, 1000, (1100, 1200, 1300), 1400) + [(5000, 102, 1), (5020, 104, 1)]   # S: later
    b = _beat(0, 5100) + _wait(8, 1000, (1100, 1200, 1300), 1400)                     # R: no preempt event
    con = _con(a)
    ev = _con(b, dev="R").execute("SELECT * FROM events").df()
    con.register("r_df", ev)
    con.execute("INSERT INTO events SELECT * FROM r_df")
    o = ped_health(con).set_index(["DeviceId", "Input"])
    assert "with no preemption during the wait" in o.loc[("S", 8), "config"]
    assert "with no preempt event logged at this signal" in o.loc[("R", 8), "config"]
    only_r = _by_input(ped_health(_con(b, dev="R")))
    assert only_r.loc[8, "status"] == "not_enough_data"
    assert "(preemption not checked: the log holds no preempt events)" in only_r.loc[8, "config"]


def test_never_calls_not_judged_without_any_overlap_walk_in_the_log():
    """a download without the ped overlap walk code (67) cannot tell a button served by an overlap apart (Oct 2026):
    never calls a walk is a config note then, not a finding; with a 67 anywhere in the log it is config_issue"""
    r = _beat(0, 3000)
    for t in (100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300):   # ped 6: no call, no walk
        r += _press(6, t)
    o = _by_input(ped_health(_con(r)))
    assert o.loc[6, "status"] == "ok" and "never calls a walk not checked" in o.loc[6, "config"]
    off = _by_input(ped_health(_con(r), options={"log_gaps": False}))
    assert off.loc[6, "status"] == "config_issue"
    w = _by_input(ped_health(_con(r + [(2500, 67, 9)])))
    assert w.loc[6, "status"] == "config_issue" and "no ped call or walk" in w.loc[6, "reason"]


def test_periods_without_a_limited_finding():
    """periods=True when no finding of a kind with a limit (stuck on) was made, or none at all."""
    r = [x for k in range(12) for x in _press(2, 100.0 * k) + [(100.0 * k + 0.1, 45, 2)] + _walk(2, 100.0 * k + 20)]
    out, per = ped_health(_con(r), periods=True)
    assert per.empty and list(per.columns)[:2] == ["DeviceId", "Detector"] and out.status.eq("ok").all()
    for k in range(26):                       # ped 3: a press every 60.0 s, never a call
        r += _press(3, 3000 + 60.0 * k)
    out, per = ped_health(_con(r), periods=True)
    assert per.condition.tolist() == ["machine-regular presses"] and per.limit_s.isna().all()
