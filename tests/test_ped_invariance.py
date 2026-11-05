"""The ped movement model must not know any number (note 156; same rule as the vehicle model's check.py renumbering):
random renumberings of the phases, ped overlaps and ped inputs in the raw log leave every answer the same, up to that
relabelling.  Features, probabilities and answers are compared bit for bit."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from atspm_detector.ped_health_duckdb import ped_health
from atspm_detector.ped_assign import FEATURES, RECALL_NOTE, ped_assignments, ped_detectors, ped_features

T0 = pd.Timestamp("2026-09-28 06:00")
PHASE_CODES = (21, 22, 23, 45)
OVERLAP_CODES = (67, 68)
INPUT_CODES = (89, 90)


def _press(ch, t, dur=0.2):
    return [(t, 90, ch), (round(t + dur, 1), 89, ch)]


def _served(ch, ph, t):
    return _press(ch, t) + [(t + 0.1, 45, ph), (t + 20, 21, ph), (t + 27, 22, ph), (t + 40, 23, ph)]


def _times(seed, n=40):
    rng = np.random.default_rng(seed)
    return np.round(100 + np.cumsum(rng.uniform(60, 240, n)), 1)


def _log():
    """every case in one signal: straight and crossed wiring, one button on two phases, two phases that time together
    (exact ties), an overlap, a press during the walk (no walk after it), a button that calls nothing"""
    r = [(1, 23, ph) for ph in (2, 4, 6, 8, 3, 7)]
    for ch, ph, seed in ((2, 2, 1), (4, 4, 2), (6, 8, 3), (8, 6, 4)):
        for t in _times(seed):
            r += _served(ch, ph, t)
    for t in _times(5):                                     # input 5 -> phases 3 and 7, logged together: exact ties
        r += _served(5, 3, t) + _served(5, 7, t)[2:]
    for t in _times(6):                                     # input 9 -> overlap 10
        r += _press(9, t) + [(t + 30, 67, 10), (t + 37, 68, 10)]
    for t in _times(7):                                     # input 12: only pressed 5 s into a walk of phase 2
        r += [(t, 21, 2), (t + 5.2, 45, 2)] + _press(12, t + 5.2) + [(t + 8, 22, 2), (t + 20, 23, 2)]
    for t in _times(8):                                     # input 3: presses, nothing else
        r += _press(3, t)
    return r


def _recall_log():
    """phases 2 and 4 walk every minute with nobody pushing; buttons 2 and 4 place no calls; button 6 calls phase 6"""
    r = [(1, 23, 6)]
    for t in np.arange(100, 12000, 60.0):
        r += [(t, 21, 2), (t, 21, 4), (t + 7, 22, 2), (t + 7, 22, 4), (t + 20, 23, 2), (t + 20, 23, 4)]
    for ch, seed in ((2, 11), (4, 12)):
        for t in _times(seed):
            r += _press(ch, t)
    for t in _times(13):
        r += _served(6, 6, t)
    return r


def _con(rows, dev="S"):
    ev = pd.DataFrame(rows, columns=["t", "EventId", "Parameter"])
    ev = pd.DataFrame({"DeviceId": dev, "Timestamp": T0 + pd.to_timedelta(ev.t, unit="s"), "EventId": ev.EventId,
                       "Parameter": ev.Parameter})
    con = duckdb.connect()
    con.register("e_df", ev)
    con.execute("CREATE TABLE events AS SELECT * FROM e_df ORDER BY DeviceId, Timestamp")
    return con


def _bijection(rng, old, pool):
    return {int(a): int(b) for a, b in zip(sorted(old), rng.choice(pool, size=len(old), replace=False))}


def _scramble(rows, seed):
    """random one-to-one maps: phases onto 1-16, overlaps onto 1-32, inputs onto 1-72; the rows shuffled too"""
    rng = np.random.default_rng(seed)
    ph = _bijection(rng, {p for _, e, p in rows if e in PHASE_CODES}, np.arange(1, 17))
    ov = _bijection(rng, {p for _, e, p in rows if e in OVERLAP_CODES}, np.arange(1, 33))
    inp = _bijection(rng, {p for _, e, p in rows if e in INPUT_CODES}, np.arange(1, 73))
    out = [(t, e, ph[p] if e in PHASE_CODES else ov[p] if e in OVERLAP_CODES else inp[p]) for t, e, p in rows]
    return [out[i] for i in rng.permutation(len(out))], ph, ov, inp


def _key(k, m, ph, ov):
    return (k, ph[m] if k == "phase" else ov[m])


def _same_float(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


SEEDS = list(range(12))


@pytest.mark.parametrize("log", [_log, _recall_log], ids=["wiring", "recall"])
@pytest.mark.parametrize("seed", SEEDS)
def test_features_follow(log, seed):
    rows = log()
    new, ph, ov, inp = _scramble(rows, seed)
    a, b = ped_features(_con(rows)), ped_features(_con(new))
    a["key"] = [(inp[i], *_key(k, m, ph, ov)) for i, k, m in zip(a.Input, a.kind, a.movement)]
    b["key"] = [(i, k, m) for i, k, m in zip(b.Input, b.kind, b.movement)]
    a, b = a.sort_values("key").reset_index(drop=True), b.sort_values("key").reset_index(drop=True)
    assert list(a.key) == list(b.key)
    for c in FEATURES + ["walk_nocall"]:
        assert _same_float(a[c].astype(float), b[c].astype(float)), c


@pytest.mark.parametrize("log", [_log, _recall_log], ids=["wiring", "recall"])
@pytest.mark.parametrize("seed", SEEDS)
def test_answers_follow(log, seed):
    rows = log()
    new, ph, ov, inp = _scramble(rows, seed)
    a, ca = ped_assignments(_con(rows), return_candidates=True)
    b, cb = ped_assignments(_con(new), return_candidates=True)
    b = b.set_index("Input")
    for r in a.itertuples():
        x = b.loc[inp[r.Input]]
        assert x.Phases == sorted(ph[m] for m in r.Phases)
        assert x.Overlaps == sorted(ov[m] for m in r.Overlaps)
        assert x.no_movement == r.no_movement and x.in_recall == r.in_recall and x.support == r.support
        assert _same_float([x.confidence, x.top_p], [r.confidence, r.top_p])
        if pd.isna(r.top_movement):
            assert pd.isna(x.top_movement)
        else:
            m = int(r.top_movement[2:] if r.top_movement.startswith("ph") else r.top_movement[3:])
            want = f"ph{ph[m]}" if r.top_movement.startswith("ph") else f"ovl{ov[m]}"
            assert x.top_movement == want
    pa = {(inp[i], *_key(k, m, ph, ov)): (p, g) for i, k, m, p, g in zip(ca.Input, ca.kind, ca.movement, ca.p, ca.gated)}
    pb = {(i, k, m): (p, g) for i, k, m, p, g in zip(cb.Input, cb.kind, cb.movement, cb.p, cb.gated)}
    assert pa.keys() == pb.keys()
    for k, (p, g) in pa.items():
        assert g == pb[k][1] and _same_float([p], [pb[k][0]]), k


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_detector_rows_follow(seed):
    """ped_detectors: no number breaks a tie for the 2nd movement or the guess"""
    rows = _log()
    new, ph, ov, inp = _scramble(rows, seed)
    a, b = ped_detectors(_con(rows), health=False), ped_detectors(_con(new), health=False)

    def mv(fn, v):
        return None if pd.isna(v) else (ov[int(v)] if fn == "Ped_Overlap" else ph[int(v)])

    got = {(int(r.Detector), None if pd.isna(r.phase_pred) else int(r.phase_pred)): r for r in b.itertuples()}
    assert len(a) == len(b)
    for r in a.itertuples():
        x = got[(inp[r.Detector], mv(r.function_pred, r.phase_pred))]
        assert mv(r.function_pred, r.phase_2nd) == (None if pd.isna(x.phase_2nd) else int(x.phase_2nd))
        assert _same_float([r.phase_prob, r.phase_2nd_prob], [x.phase_prob, x.phase_2nd_prob])
        assert r.status == x.status and (r.function_pred == x.function_pred or pd.isna(r.function_pred) and pd.isna(x.function_pred))


def test_no_number_feature():
    for f in FEATURES:
        assert "number" not in f and f not in ("ped_detector", "movement", "Input")
    assert "same_number" not in FEATURES


def test_tied_phases_both_kept_and_no_guess_by_number():
    """two phases with the same log get the same probability: both or neither, and no 'best' picked by number"""
    a, c = ped_assignments(_con(_log()), return_candidates=True)
    r = a.set_index("Input").loc[5]
    x = c[(c.Input == 5) & c.movement.isin([3, 7])]
    assert x.p.iloc[0] == x.p.iloc[1]
    assert r.Phases == [3, 7] and pd.isna(r.top_movement)


def test_press_during_the_walk_is_served():
    """a button only pressed while its phase already walks: no walk follows, but it calls that phase (03028)"""
    r = ped_assignments(_con(_log())).set_index("Input").loc[12]
    assert r.Phases == [2]


def test_recall_no_guess_no_config_issue():
    """user ruling: a phase in recall cannot be tied to a button.  No answer, no guess, a note, never a config issue;
    the same whichever number the button has (button 2 next to phase 2 gets no credit for it)"""
    con = _con(_recall_log())
    a = ped_assignments(con).set_index("Input")
    for i in (2, 4):
        r = a.loc[i]
        assert r.in_recall and not r.no_movement and r.Phases == [] and pd.isna(r.top_movement) and np.isnan(r.top_p)
    assert a.loc[6, "Phases"] == [6] and not a.loc[6, "in_recall"]
    h = ped_health(con, assign=True).set_index("Input")
    assert h.loc[2, "status"] != "config_issue" and RECALL_NOTE in h.loc[2, "config"]
    d = ped_detectors(con, health=False).set_index("Detector")
    assert d.loc[2, "status"].startswith(RECALL_NOTE) and pd.isna(d.loc[2, "phase_pred"]) and pd.isna(d.loc[2, "phase_guess"])
    assert not d.loc[2, "review_flag"]


def _busy_recall_log():
    """the same recall phases with a press on some button before every walk: no ped call, so still in recall"""
    r = _recall_log()
    for t in np.arange(100, 12000, 60.0):
        r += _press(2, t - 10)
    return r


def test_recall_does_not_depend_on_press_traffic():
    """the first rule counted presses on any button: heavy press traffic took a real recall phase out of recall"""
    con = _con(_busy_recall_log())
    a, c = ped_assignments(con, return_candidates=True)
    a = a.set_index("Input")
    assert (c[c.movement.isin([2, 4]) & (c.kind == "phase")].walk_nocall == 1).all()
    assert a.loc[2, "in_recall"] and a.loc[4, "in_recall"]
    assert ped_health(con, assign=True).set_index("Input").loc[2, "status"] != "config_issue"


def _thin_log(walks):
    """phase 2 walks `walks` times with no ped call; button 2 has presses and no calls; button 6 calls phase 6"""
    r = [(1, 23, 6)]
    for t in np.arange(walks) * 300.0 + 100:
        r += [(t, 21, 2), (t + 7, 22, 2), (t + 20, 23, 2)]
    r += _press(2, 1000.0) * 1
    for t in np.arange(100, 3000, 100.0):
        r += _press(2, t + 3)
    for t in _times(13, 20):
        r += _served(6, 6, t)
    return r


@pytest.mark.parametrize("walks", [1, 2, 4])
def test_too_few_walks_is_not_a_config_issue(walks):
    con = _con(_thin_log(walks))
    a = ped_assignments(con).set_index("Input").loc[2]
    assert a.Phases == [] and not a.in_recall and a.low_support and a.support >= 10
    assert ped_health(con, assign=True).set_index("Input").loc[2, "status"] != "config_issue"
    d = ped_detectors(con, health=False).set_index("Detector").loc[2]
    assert d.status.startswith("not enough data") and pd.isna(d.phase_pred)


def test_five_walks_without_calls_is_recall():
    a = ped_assignments(_con(_thin_log(5))).set_index("Input").loc[2]
    assert a.in_recall and not a.no_movement and not a.low_support


def test_overlap_is_never_in_recall():
    """an overlap has no call code; walks with no press must not make it recall"""
    r = [(1, 23, 6)]
    for t in np.arange(100, 12000, 60.0):
        r += [(t, 67, 10), (t + 7, 68, 10)]
    for t in _times(13):
        r += _served(6, 6, t)
    for t in _times(11):
        r += _press(2, t)
    a, c = ped_assignments(_con(r), return_candidates=True)
    assert not c[c.kind == "overlap"].gated.any() and not a.in_recall.any()
