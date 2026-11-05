"""Ped movement assignment (ped_assign, note 156) on small synthetic logs."""
import duckdb
import numpy as np
import pandas as pd

from atspm_detector.ped_assign import COLS, FEATURES, ped_assignments, ped_detectors

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


def _served(ch, ph, t):
    """a press on input ch while phase ph is idle: ped call 0.1 s later, walk 20 s later, then don't walk"""
    return _press(ch, t) + [(t + 0.1, 45, ph), (t + 20, 21, ph), (t + 27, 22, ph), (t + 40, 23, ph)]


def _times(seed, n=40):
    """n press times, 60-240 s apart (seeded: independent buttons)"""
    rng = np.random.default_rng(seed)
    return np.round(100 + np.cumsum(rng.uniform(60, 240, n)), 1)


def _log():
    r = [(1, 23, ph) for ph in (2, 4, 6, 8)]
    for ch, ph, seed in ((2, 2, 1), (4, 4, 2), (6, 8, 3), (8, 6, 4)):   # input 6 -> phase 8, 8 -> 6 (crossed wiring)
        for t in _times(seed):
            r += _served(ch, ph, t)
    for t in _times(5):                       # input 3: presses, no call, no walk
        r += _press(3, t)
    for t in _times(6):                       # input 9 -> ped overlap 10 (no ped call): walk 30 s after the press
        r += _press(9, t) + [(t + 30, 67, 10), (t + 37, 68, 10)]
    return r


def test_assignments():
    out = ped_assignments(_con(_log()))
    assert list(out.columns) == COLS
    got = {r.Input: (r.Phases, r.Overlaps, r.no_movement) for r in out.itertuples()}
    assert got[2] == ([2], [], False)
    assert got[4] == ([4], [], False)
    assert got[6] == ([8], [], False)
    assert got[8] == ([6], [], False)
    assert got[9] == ([], [10], False)
    assert got[3] == ([], [], True)
    assert (out.support == 40).all() and not out.low_support.any() and not out.in_recall.any()
    assert (out.set_index("Input").loc[[2, 4, 6, 8, 9], "top_p"] > 0.9).all()


def test_candidates_and_window():
    con = _con(_log())
    out, cand = ped_assignments(con, return_candidates=True)
    assert {"DeviceId", "Input", "kind", "movement", "p", *FEATURES} <= set(cand.columns)
    assert set(cand.kind) == {"phase", "overlap"}
    assert cand[cand.kind == "overlap"].call1.isna().all()          # overlaps have no ped-call code
    # a window that holds 5 presses per input: low support, temp tables cleaned up
    few = ped_assignments(con, start=T0, end=T0 + pd.Timedelta(seconds=float(min(_times(s)[4] for s in range(1, 7))) + 1))
    assert few.support.between(1, 9).all() and few.low_support.all()
    assert not con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name LIKE 'pa_%'").fetchone()[0]


def test_no_presses():
    out = ped_assignments(_con([(t, 1, 2) for t in range(10)]))
    assert out.empty and list(out.columns) == COLS


def test_add_to_ped_health():
    from atspm_detector.ped_health_duckdb import ped_health
    from atspm_detector.ped_assign import NO_MOVEMENT_NOTE, add_assignments
    con = _con(_log())
    h = add_assignments(con, ped_health(con))
    r = h.set_index("Input")
    assert r.loc[6, "Phases"] == [8] and r.loc[9, "Overlaps"] == [10]
    assert NO_MOVEMENT_NOTE in r.loc[3, "config"] and NO_MOVEMENT_NOTE not in str(r.loc[2, "config"])


def test_ped_health_assign_flag():
    """assign=True: the model's columns; an input whose presses call nothing is a config_issue (user ruling)"""
    from atspm_detector.ped_health_duckdb import ped_health
    from atspm_detector.ped_assign import NO_MOVEMENT_NOTE, NO_MOVEMENT_REASON
    h, per = ped_health(_con(_log()), assign=True, periods=True)
    assert {"Phases", "Overlaps", "assign_confidence"} <= set(h.columns)
    r = h.set_index("Input")
    assert r.loc[3, "status"] == "config_issue" and r.loc[3, "reason"] == NO_MOVEMENT_REASON
    assert NO_MOVEMENT_NOTE in r.loc[3, "config"]
    assert (r.drop(index=3).status == "ok").all()
    # option config_issue off: the note only
    h = ped_health(_con(_log()), assign=True, options={"config_issue": False}).set_index("Input")
    assert h.loc[3, "status"] == "ok" and NO_MOVEMENT_NOTE in h.loc[3, "config"]


def test_no_overlap_events_not_checked():
    """a log without any ped overlap walk (67) cannot tell an overlap button apart: a note, never a finding"""
    from atspm_detector.ped_health_duckdb import ped_health
    from atspm_detector.ped_assign import NOT_CHECKED_NOTE
    rows = [r for r in _log() if r[1] not in (67, 68)]
    h = ped_health(_con(rows), assign=True).set_index("Input")
    assert h.loc[3, "status"] != "config_issue" and NOT_CHECKED_NOTE in h.loc[3, "config"]
    d = ped_detectors(_con(rows), health=False).set_index("Detector")
    assert d.loc[3, "status"].startswith("cannot classify") and d.loc[3, "review_flag"]


def test_signal_without_ped_movement():
    """a signal with presses but no ped call / walk / overlap walk in the window: its inputs still get an answer
    (calls nothing) in ped_assignments and a config issue row in ped_detectors; a window of only such signals too"""
    from atspm_detector.ped_health_duckdb import ped_health
    from atspm_detector.ped_assign import NO_MOVEMENT_NOTE
    lone = [(t + 0.5, ev, 5) for t in _times(7) for ev in (90, 89)]
    both = pd.concat([_con(_log()).execute("SELECT * FROM events").df(),
                      _con(lone, dev="T").execute("SELECT * FROM events").df()], ignore_index=True)
    con = duckdb.connect()
    con.register("b_df", both)
    con.execute("CREATE TABLE events AS SELECT * FROM b_df ORDER BY DeviceId, Timestamp")
    a = ped_assignments(con).set_index(["DeviceId", "Input"])
    t = a.loc[("T", 5)]
    assert t.Phases == [] and t.Overlaps == [] and t.no_movement and t.support == 40 and pd.isna(t.top_movement)
    assert a.loc[("S", 2), "Phases"] == [2]
    d = ped_detectors(con).set_index(["DeviceId", "Detector"])
    t = d.loc[("T", 5)]
    assert t.status.startswith("config issue") and pd.isna(t.phase_pred) and pd.isna(t.phase_guess)
    assert t.n_candidate_phases == 0 and t.n_actuations == 40 and t.review_flag
    h = ped_health(con, assign=True).set_index(["DeviceId", "Input"])
    assert h.loc[("T", 5), "status"] == "config_issue" and NO_MOVEMENT_NOTE in h.loc[("T", 5), "config"]
    assert t.health_status == "config_issue"
    # only such a signal: rows, not an empty table (no overlap walks anywhere: not checked, never a finding)
    only = _con(lone, dev="T")
    assert ped_assignments(only).no_movement.all()
    d = ped_detectors(only)
    assert len(d) == 1 and d.status.iloc[0].startswith("cannot classify")


def test_ped_detectors_schema():
    """ped_detectors: predict()'s columns and dtypes, one row per (ped input, movement)"""
    import importlib.resources
    import warnings

    from atspm_detector import EXTRA_COLS, OUT_COLS, predict
    sample = str(importlib.resources.files("atspm_detector") / "data" / "sample_events.parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        veh = predict(sample)
    d = ped_detectors(_con(_log()))
    assert list(d.columns) == OUT_COLS + EXTRA_COLS
    assert d.dtypes.drop("DeviceId").to_dict() == veh.dtypes.drop("DeviceId").to_dict()
    r = d.set_index("Detector")
    assert (r.loc[2, "phase_pred"], r.loc[2, "function_pred"], r.loc[2, "status"]) == (2, "Ped", "ok")
    assert (r.loc[6, "phase_pred"], r.loc[8, "phase_pred"]) == (8, 6)
    assert (r.loc[9, "phase_pred"], r.loc[9, "function_pred"]) == (10, "Ped_Overlap")
    assert pd.isna(r.loc[3, "phase_pred"]) and r.loc[3, "status"].startswith("config issue") and r.loc[3, "review_flag"]
    assert r.loc[3, "health_status"] == "config_issue" and (r.drop(index=3).health_status == "ok").all()
    assert (d.n_actuations == 40).all() and (d.function_prob.dropna() == 1).all()
    assert (r.drop(index=3).phase_prob > 0.9).all()
    both = pd.concat([veh, d], ignore_index=True)       # one detector table
    assert list(both.columns) == OUT_COLS + EXTRA_COLS
    # few presses: one row, no answer, the best movement kept as a guess
    few = ped_detectors(_con(_log()), end=T0 + pd.Timedelta(seconds=float(_times(1)[4]) + 1), health=False)
    assert few.status.str.startswith("not enough data").all() and few.phase_pred.isna().all()
    assert few.phase_guess.notna().all()
    assert ped_detectors(_con([(t, 1, 2) for t in range(10)])).empty
