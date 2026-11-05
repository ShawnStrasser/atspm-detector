"""A brief release inside a stuck ON: stuck-on periods of an input close together read as one hold (ped option hold_join)."""
import duckdb
import pandas as pd

from atspm_detector.ped_health_duckdb import LIM, ped_health

T0 = pd.Timestamp("2026-10-30 06:00")


def _con(rows):
    ev = pd.DataFrame(rows, columns=["t", "EventId", "Parameter"])
    ev = pd.DataFrame({"DeviceId": "S", "Timestamp": T0 + pd.to_timedelta(ev.t, unit="s"), "EventId": ev.EventId,
                       "Parameter": ev.Parameter})
    con = duckdb.connect()
    con.register("e_df", ev)
    con.execute("CREATE TABLE events AS SELECT * FROM e_df ORDER BY DeviceId, Timestamp")
    return con


def _press(ch, t, dur=0.2):
    return [(t, 90, ch), (round(t + dur, 1), 89, ch)]


def _beat(t0, t1, step=10):
    return [(t, 82, 1) for t in range(t0, t1, step)]


def _judge(rows, **options):
    return ped_health(_con(rows), periods=True, options=options or None)


def _stuck(per):
    return per[per.condition == "stuck on"].reset_index(drop=True)


def _reason(inputs, ch):
    return inputs.set_index("Input").loc[ch, "reason"]


def _walks(ph, t0, t1, step=90):
    """a walk at every green of the phase, as a held button brings"""
    r = []
    for t in range(t0, t1, step):
        r += [(t - 1, 1, ph), (t, 21, ph), (t + 7, 22, ph), (t + 20, 23, ph)]
    return r


def test_a_tenth_of_a_second_off_inside_a_hold_is_one_period_of_its_full_length():
    r = _beat(0, 9000) + _press(5, 100, 2000) + _press(5, 2100.1, 3000)
    i, p = _judge(r)
    s = _stuck(p)
    assert len(s) == 1
    assert s.start[0] == T0 + pd.Timedelta(seconds=100) and s.end[0] == T0 + pd.Timedelta(seconds=5100.1)
    assert "held 1 h 23 min" in _reason(i, 5)


def test_switching_it_off_gives_the_pieces_back():
    r = _beat(0, 9000) + _press(5, 100, 2000) + _press(5, 2100.1, 3000)
    i, p = _judge(r, hold_join=False)
    assert len(_stuck(p)) == 2 and "held 50 min" in _reason(i, 5)


def test_chatter_between_two_holds_is_inside_the_one_period():
    chatter = [x for k in range(8) for x in _press(5, 2101 + k * 0.4, 0.2)]
    r = _beat(0, 9000) + _press(5, 100, 2000) + chatter + _press(5, 2110, 3000)
    assert len(_stuck(_judge(r)[1])) == 1


def test_brief_presses_between_two_holds_stay_in_every_count():
    chatter = [x for k in range(8) for x in _press(5, 2101 + k * 0.4, 0.2)]
    r = _beat(0, 9000) + _press(5, 100, 2000) + chatter + _press(5, 2110, 3000)
    joined, apart = _judge(r), _judge(r, hold_join=False)
    assert len(_stuck(joined[1])) == 1 and len(_stuck(apart[1])) == 2
    assert joined[0].set_index("Input").loc[5, "presses"] == apart[0].set_index("Input").loc[5, "presses"] == 10


def test_a_real_release_stays_two_periods():
    gap = LIM["hold_gap_s"] + 111                      # the regular 171-s OFF of a button that stops walking in it
    r = _beat(0, 9000) + _press(5, 100, 2000) + _press(5, 2100 + gap, 3000)
    assert len(_stuck(_judge(r)[1])) == 2


def test_who_is_flagged_does_not_change():
    r = _beat(0, 9000) + _press(5, 100, 2000) + _press(5, 2100.1, 3000)    # stuck
    r += _press(6, 100, 100) + _press(6, 200.1, 100)                       # 200 s in two pieces, neither stuck
    for a, b in ((_judge(r)[0], _judge(r, hold_join=False)[0]),):
        assert a.set_index("Input").status.to_dict() == b.set_index("Input").status.to_dict()
    assert 6 not in set(_stuck(_judge(r)[1]).Detector)


def test_presses_of_a_chattering_button_are_not_a_hold():
    # OFF under a second again and again, never one long ON: no stuck-on period, however close the presses
    r = _beat(0, 3000) + [x for k in range(800) for x in _press(5, 100 + k * 1.5, 0.5)]
    assert _stuck(_judge(r)[1]).empty


def test_walks_of_the_pieces_add_up_and_the_button_stays_bad():
    r = _beat(0, 9000) + _walks(8, 200, 4000) + _press(8, 100, 2000) + [(100.1, 45, 8)] + _press(8, 2100.3, 2000)
    i, p = _judge(r)
    s = _stuck(p)
    assert len(s) == 1 and i.set_index("Input").loc[8, "status"] == "bad"
    assert "through 43 walks" in _reason(i, 8)


def test_a_long_hold_touching_a_hold_is_part_of_it():
    r = _beat(0, 9000) + _walks(8, 200, 500) + [(100.1, 45, 8)] + _press(8, 100, 400) + _press(8, 500.5, 170)
    r += _walks(8, 560, 660)
    p = _judge(r)[1]
    assert "long hold" not in set(p.condition)
    assert "long hold" in set(_judge(r, hold_join=False)[1].condition)
