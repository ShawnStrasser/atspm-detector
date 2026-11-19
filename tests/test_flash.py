"""No data from a signal in flash is used (flash.py), and a saved ped configuration (ped_health config=)."""
import duckdb
import pandas as pd

from atspm_detector.flash import drop_flash, without_flash
from atspm_detector.ped_health_duckdb import ped_health

T0 = pd.Timestamp("2026-11-17 00:00")


def _con(rows):
    ev = pd.DataFrame(rows, columns=["dev", "t", "EventId", "Parameter"])
    ev = pd.DataFrame({"DeviceId": ev.dev, "Timestamp": T0 + pd.to_timedelta(ev.t, unit="s"), "EventId": ev.EventId,
                       "Parameter": ev.Parameter})
    con = duckdb.connect()
    con.register("e_df", ev)
    con.execute("CREATE TABLE events AS SELECT * FROM e_df ORDER BY DeviceId, Timestamp")
    return con


def _kept(con, name):
    return {(d, round((pd.Timestamp(t) - T0).total_seconds())) for d, t in
            con.execute(f"SELECT DeviceId, Timestamp FROM {name} WHERE EventId = 90").fetchall()}


def test_flash_periods_are_dropped():
    h = 3600
    rows = [("A", 21 * h, 1, 2), ("A", 21 * h + 60, 90, 2),           # A: running, then a night flash 22:00 - 06:00
            ("A", 22 * h, 173, 3), ("A", 22 * h + 420, 90, 2),         # a press in flash
            ("A", 24 * h - 0.1, 173, 3),                                # the state logged again: still in flash
            ("A", 30 * h, 173, 2), ("A", 30 * h + 6, 1, 2), ("A", 30 * h + 60, 90, 2),
            ("B", 600, 90, 4), ("B", 6 * h, 173, 2), ("B", 6 * h + 6, 1, 4),   # B: the log opens in flash
            ("B", 7 * h, 90, 4),
            ("C", 60, 1, 2), ("C", 120, 90, 6), ("C", 5 * h, 173, 2),   # C: running; a 2 logged again is no flash
            ("D", 60, 1, 2), ("D", 2 * h, 131, 255), ("D", 2 * h + 30, 90, 8),   # D: flash by its coordination plan
            ("D", 3 * h, 131, 1), ("D", 3 * h + 30, 90, 8)]
    con = _con(rows)
    v = without_flash(con, "events")
    assert v != "events"
    assert _kept(con, v) == {("A", 21 * h + 60), ("A", 30 * h + 60), ("B", 7 * h), ("C", 120), ("D", 3 * h + 30)}
    assert drop_flash(con, "events") > 0
    assert _kept(con, "events") == _kept(con, v)


def test_no_flash_leaves_the_table():
    con = _con([("A", 60, 1, 2), ("A", 120, 90, 2), ("A", 3600, 173, 2)])
    assert without_flash(con, "events") == "events"
    assert drop_flash(con, "events") == 0


def _ped_log():
    """ped 6 at a signal whose phases walk every cycle with no ped call (ped recall); 12 presses served, then a burst of
    12 presses in a night flash: from the window alone it calls nothing.  Overlap 10 walks so 'never calls' is checked."""
    r = [("S", t, 82, 1) for t in range(0, 4000, 10)]
    for k in range(30):
        t = 100.0 * k
        r += [("S", t - 1, 1, 6), ("S", t, 21, 6), ("S", t + 7, 22, 6), ("S", t + 20, 23, 6)]
    r += [("S", 50.0 * k + 30, 90, 6) for k in range(12)] + [("S", 50.0 * k + 30.2, 89, 6) for k in range(12)]
    r += [("S", 3100, 173, 3)] + [("S", 3200 + k, 90, 6) for k in range(12)] + [("S", 3200.2 + k, 89, 6) for k in range(12)]
    r += [("S", 3300, 67, 10)]
    return r


def test_flash_presses_never_make_a_finding():
    out = ped_health(_con(_ped_log())).set_index("Input")
    assert out.loc[6, "status"] == "ok" and out.loc[6, "presses"] == 12


def test_saved_ped_config():
    con = _con(_ped_log())
    con.execute("CREATE TABLE cfg (DeviceId VARCHAR, Detector INT, phase INT, function VARCHAR, in_recall BOOLEAN)")
    con.execute("INSERT INTO cfg VALUES ('S', 6, NULL, 'Ped', true)")
    import atspm_detector.ped_health_duckdb as ph
    real = ph.without_flash
    try:   # with the flash presses kept, the window alone cannot place ped 6: only the saved row keeps it ok
        ph.without_flash = lambda con, E, name="x": E
        bare = ped_health(con).set_index("Input")
        saved = ped_health(con, config="cfg").set_index("Input")
        con.execute("UPDATE cfg SET phase = 6, in_recall = false")
        phase = ped_health(con, config="cfg").set_index("Input")
    finally:
        ph.without_flash = real
    assert bare.loc[6, "status"] == "config_issue"
    assert saved.loc[6, "status"] == "ok" and "ped in recall (saved configuration)" in saved.loc[6, "config"]
    assert phase.loc[6, "status"] == "ok" and phase.loc[6, "Phase"] == 6


def test_predict_loads_no_flash():
    """predict()'s loader: the flash rows go, and a signal with nothing left drops out of the signal map"""
    from atspm_detector.pipeline import load_events
    h = 3600
    rows = [("A", 60, 1, 2), ("A", 120, 82, 3), ("A", h, 173, 3), ("A", h + 60, 82, 3), ("A", 2 * h, 173, 2),
            ("A", 2 * h + 6, 1, 2), ("A", 2 * h + 60, 82, 3),
            ("B", 60, 173, 3), ("B", 120, 82, 5)]                      # B: in flash for all of its log
    ev = pd.DataFrame(rows, columns=["DeviceId", "t", "EventId", "Parameter"])
    ev = ev.assign(Timestamp=T0 + pd.to_timedelta(ev.t, unit="s")).drop(columns="t")
    con = duckdb.connect()
    w0, w1, info = load_events(con, ev)
    kept = sorted(round((pd.Timestamp(t) - T0).total_seconds()) for d, t in
                  con.execute("SELECT DeviceId, Timestamp FROM evd WHERE EventId = 82").fetchall())
    assert kept == [120, 2 * h + 60]
    assert con.execute("SELECT DeviceId, dev FROM devmap").fetchall() == [("A", 1)]
    assert con.execute("SELECT DISTINCT dev FROM evd").fetchall() == [(1,)]
