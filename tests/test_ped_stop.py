"""The sudden-stop test on synthetic counts; stuck ON with no OFF; the carry of an open ON across chunks."""
import duckdb
import numpy as np
import pandas as pd
from atspm_detector.ped_health_duckdb import ped_health
from atspm_detector.ped_stop import open_ons, stop_scores

T0 = pd.Timestamp("2026-10-01")


def _run(per_bin, stop_bin=None, days=3, seed=1):
    """one input with presses at a flat daytime level; zero from stop_bin on."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(T0, periods=96 * days, freq="15min")
    day = (t.hour >= 7) & (t.hour < 19)
    n = np.where(day, rng.poisson(per_bin, len(t)), 0)
    if stop_bin is not None:
        n[stop_bin:] = 0
    con = duckdb.connect()
    con.register("c", pd.DataFrame({"dev": "a", "ch": 2, "b": t, "n": n}))
    con.register("l", pd.DataFrame({"dev": "a", "b": t}))
    df = stop_scores(con, "c", "l")
    return df.iloc[0] if len(df) else pd.Series({"rating": None})


def test_busy_button_that_stops_flags():
    r = _run(5, stop_bin=96 + 40)
    assert r.rating == "bad" and pd.isna(r.next_b)


def test_low_volume_button_never_flags():
    assert _run(0.02, stop_bin=96 + 40).rating is None


def test_steady_button_does_not_flag():
    assert _run(5).rating is None


def test_a_period_that_ended_has_the_next_press_as_its_end():
    t = pd.date_range(T0, periods=96 * 3, freq="15min")
    n = np.where((t.hour >= 7) & (t.hour < 19), 5, 0)
    n[96 + 40:96 + 136] = 0
    keep = np.ones(len(t), bool)
    keep[96 + 130:96 + 150] = False   # a data gap after the silence begins
    con = duckdb.connect()
    con.register("c", pd.DataFrame({"dev": "a", "ch": 2, "b": t[keep], "n": n[keep]}))
    con.register("l", pd.DataFrame({"dev": "a", "b": t[keep]}))
    r = stop_scores(con, "c", "l").iloc[0]
    assert r.rating in ("bad", "suspect")
    assert pd.Timestamp(r.next_b) in set(t[keep]) and n[list(t).index(pd.Timestamp(r.next_b))] > 0


def _log(on_at, walks_from=None, off_at=None, walks_until=None, hours=8, walk_before=False, extra_press=True):
    """a signal with phase 2 greens every 100 s; a walk at each green from walks_from on; a normal press early; an ON at on_at."""
    rows = []
    for s in range(0, hours * 3600, 100):
        t = T0 + pd.Timedelta(seconds=s)
        rows.append((t, 1, 2))
        rows.append((t + pd.Timedelta(seconds=30), 8, 2))
        walking = (walks_from is not None and s >= walks_from and (walks_until is None or s < walks_until)) or walk_before
        if walking:
            rows += [(t + pd.Timedelta(seconds=1), 21, 2), (t + pd.Timedelta(seconds=8), 22, 2), (t + pd.Timedelta(seconds=20), 23, 2)]
    if extra_press:
        rows += [(T0 + pd.Timedelta(seconds=50), 90, 2), (T0 + pd.Timedelta(seconds=50.05), 45, 2), (T0 + pd.Timedelta(seconds=50.3), 89, 2)]
    if on_at is not None:
        rows += [(T0 + pd.Timedelta(seconds=on_at), 90, 2), (T0 + pd.Timedelta(seconds=on_at + 0.05), 45, 2)]
    if off_at is not None:
        rows.append((T0 + pd.Timedelta(seconds=off_at), 89, 2))
    df = pd.DataFrame(rows, columns=["Timestamp", "EventId", "Parameter"]).assign(DeviceId="S")
    con = duckdb.connect()
    con.register("d", df)
    con.execute('CREATE TABLE events AS SELECT DeviceId, "Timestamp", EventId::INT AS EventId, Parameter::INT AS Parameter FROM d')
    return con


def _status(con, **kw):
    o = ped_health(con, **kw)
    return o[o.Input == 2].status.iloc[0], o[o.Input == 2].reason.iloc[0]


def test_on_with_no_off_and_every_cycle_walks_is_stuck():
    st, why = _status(_log(on_at=7200, walks_from=7200))
    assert st == "bad" and "walking every cycle" in why


def test_on_with_no_off_and_no_walks_is_lost_data():
    assert "stuck" not in _status(_log(on_at=7200))[1]


def test_recall_already_before_the_on_is_not_stuck():
    assert "stuck" not in _status(_log(on_at=7200, walk_before=True))[1]


def test_a_carried_on_keeps_its_real_start():
    con = _log(on_at=7200, walks_from=7200)
    cut = T0 + pd.Timedelta(hours=4)
    ons = open_ons(con, "events", None, cut)
    assert len(ons) == 1
    o, p = ped_health(con, start=cut, end=T0 + pd.Timedelta(hours=8), periods=True, carry=ons)
    s = p[p.condition == "stuck on"]
    assert len(s) == 1 and s.start.iloc[0] == T0 + pd.Timedelta(seconds=7200)


def test_an_off_with_no_on_in_the_window_counts_from_the_window_start():
    con = _log(on_at=None, walks_from=0, walks_until=7200, off_at=7200, extra_press=False)
    assert _status(con)[0] == "bad"
