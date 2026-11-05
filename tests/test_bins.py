"""The health bins (bins.py): the scoring rules on small tables, then a store's bins on synthetic events: each bin once whatever the cuts, a stuck ON past the window."""
import numpy as np
import duckdb
import pandas as pd
import pytest
import synth

from atspm_detector import bins as B
from atspm_detector import findings as F
from atspm_detector import incremental as I
from atspm_detector import ped_health_duckdb as PH

T = pd.Timestamp("2026-10-05 10:00:00")
BIN = pd.Timedelta(minutes=15)
DAY = pd.Timedelta(days=1)
ST = "health_store"
LATE_KEY = ["TimeStamp", "DeviceId", "DetectorType", "Detector", "Label"]


def frames(*rows, status="bad"):
    """findings from (detector, start minutes, end minutes, label, confidence), a status for each detector"""
    f = pd.DataFrame([{"DeviceId": "S1", "DetectorType": "Vehicle", "Detector": d, "Start": T + pd.Timedelta(minutes=a), "End": T + pd.Timedelta(minutes=z),
                       "Label": lab, "Confidence": c, "Detail": ""} for d, a, z, lab, c in rows])[F.FINDING_COLS]
    s = pd.DataFrame({"DeviceId": "S1", "DetectorType": "Vehicle", "Detector": sorted(set(f.Detector)), "Date": T.normalize(), "Status": status})[F.STATUS_COLS]
    return f, s


def run_bins(*rows, status="bad", since=T, end=T + 8 * BIN, bin_minutes=15):
    f, s = frames(*rows, status=status)
    return B.bins(duckdb.connect(), f, s, since, end, bin_minutes)


def test_half_a_bin_of_complete_failure_scores_half():
    b = run_bins((1, 0, 7.5, "silent", 0.95))
    assert b.TimeStamp.tolist() == [T] and b.Score.tolist() == [0.5] and b.Label.tolist() == ["silent"]
    assert list(b.columns) == B.BIN_COLS


def test_a_failure_scores_about_one_per_bin_and_hours_are_the_sum_over_four():
    b = run_bins((1, 0, 180, "stuck on", 0.9), end=T + 12 * BIN)
    assert len(b) == 12 and (b.Score == 1).all() and b.Score.sum() / 4 == 3


def test_overlapping_problems_share_the_worst_value_and_add_up_to_the_bin():
    # the first five minutes alone, then both for ten minutes
    b = run_bins((1, 0, 15, "silent", 0.8), (1, 5, 15, "spiky counts", 0.4))
    r = b.set_index("Label").Score
    assert r["silent"] == pytest.approx(1 / 3 + 1 / 3)
    assert r["spiky counts"] == pytest.approx(1 / 3)
    assert b.Score.sum() == pytest.approx(1.0, abs=1e-5)


def test_rows_of_a_bin_never_pass_one():
    labels = ["silent", "counts dropped", "too-fast actuations", "spiky counts", "busy at night"]
    b = run_bins(*[(1, k, 60, lab, 0.99) for k, lab in enumerate(labels)], end=T + 4 * BIN)
    assert b.groupby("TimeStamp").Score.sum().max() <= 1 + 1e-5


@pytest.mark.parametrize("label, conf", [("borderline stuck on", 0.4), ("config note", 0.0), ("on ped recall", 0.0), ("calls two ped phases", 0.0), ("lane closure", None),
                                         ("timing plan", None), ("pulse mode", 0.5), ("signal note", 0.0), ("omitted during preemption", 0.0), ("call dropped", 0.5)])
def test_borderline_and_notes_are_not_problems(label, conf):
    assert run_bins((1, 0, 60, label, conf)).empty


def test_not_enough_data_is_one_row_a_day_with_no_score():
    since = pd.Timestamp("2026-10-05")
    f, s = frames((1, 0, 60, F.TOO_FEW, 0.0), status="not_enough_data")
    days = pd.DataFrame({"DeviceId": "S1", "DetectorType": "Vehicle", "Detector": 1, "Start": [since, since + DAY], "End": [since + DAY, since + 2 * DAY],
                         "Label": F.TOO_FEW, "Confidence": 0.0, "Detail": ""})[F.FINDING_COLS]
    b = B.bins(duckdb.connect(), days, s, since, since + 2 * DAY)
    assert b.TimeStamp.tolist() == [since, since + DAY] and b.Score.isna().all() and (b.Label == F.TOO_FEW).all()


def test_a_detector_rated_ok_has_no_rows():
    assert run_bins((1, 0, 60, "silent", 0.9), status="ok").empty


def test_rows_stay_inside_since_and_end():
    b = run_bins((1, -60, 120, "silent", 0.9), since=T, end=T + 2 * BIN)
    assert b.TimeStamp.tolist() == [T, T + BIN] and (b.Score == 1).all()


def test_other_bin_sizes():
    b = run_bins((1, 0, 30, "silent", 0.9), since=T, end=T + pd.Timedelta(hours=2), bin_minutes=60)
    assert b.TimeStamp.tolist() == [T] and b.Score.tolist() == [0.5]


def test_every_rule_reaches_one_at_or_before_a_confidence_of_one_and_a_note_has_no_score():
    for label, full in B.SCORE_FULL.items():
        assert 0 < full <= 1
        assert B.score(label, 1.0) == 1.0 and B.score(label, full) == 1.0
    assert B.score("borderline stuck on", 0.4) is None and B.score("silent", None) is None


def test_every_rule_of_the_findings_layer_has_a_score():
    kinds = {p[2] for p in F.CONTEXT["PIECES"]} | {p[1] for p in F.CONTEXT["PED_PIECES"]}
    assert kinds - set(B.SCORE_FULL) == {"lane closure", "timing plan", "pulse mode"}


# --- a store: a stuck ON past the window ----------------------------------------------------------------------------------------------------------------

_S = {}


def stuck_events():
    """one signal; from `on` to `off` detector 1 and ped input 2 ON, P2 walking at every green.  1-day horizon: the ON leaves the window after a day"""
    if _S:
        return _S
    ev, det = synth.make(1, 7, seed=3, faults=False)
    dev = ev.DeviceId.iloc[0]
    day0 = ev.Timestamp.min().floor("D")
    on, off = day0 + pd.Timedelta(days=1, hours=12), day0 + pd.Timedelta(days=5, hours=6, minutes=7, seconds=30)
    margin = pd.Timedelta(minutes=5)
    keep = ~(ev.EventId.isin([81, 82]) & (ev.Parameter == 1) & (ev.Timestamp >= on - margin) & (ev.Timestamp <= off + margin))
    keep &= ~(ev.EventId.isin([89, 90]) & (ev.Parameter == 2) & (ev.Timestamp >= on - margin) & (ev.Timestamp <= off + margin))
    keep &= ~(ev.EventId.isin([21, 22, 23]) & (ev.Parameter == 2) & (ev.Timestamp >= on) & (ev.Timestamp <= off + 2 * margin))
    greens = ev[(ev.EventId == 1) & (ev.Parameter == 2) & (ev.Timestamp > on) & (ev.Timestamp < off)].Timestamp
    walks = pd.concat([pd.DataFrame({"Timestamp": greens + pd.Timedelta(seconds=s), "EventId": e}) for s, e in ((0.1, 21), (7, 22), (15, 23))]).assign(DeviceId=dev, Parameter=2)
    add = pd.DataFrame({"DeviceId": dev, "Timestamp": [on, off, on, on + pd.Timedelta(seconds=0.1), off], "EventId": [82, 81, 90, 45, 89], "Parameter": [1, 1, 2, 2, 2]})
    ev = pd.concat([ev[keep], add, walks], ignore_index=True).astype({"EventId": "int32", "Parameter": "int32"})
    _S.update(ev=ev, det=det, dev=dev, day0=day0, on=on, off=off)
    return _S


def connect(ev, det):
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.register("e", ev)
    con.execute("CREATE TABLE events AS SELECT * FROM e")
    con.unregister("e")
    con.register("d", det)
    con.execute("CREATE TABLE detectors AS SELECT * FROM d")
    con.unregister("d")
    return con


def run_cuts(con, cuts, judges, horizon_days=1, store=ST):
    """updates cut at cuts; a judge at each of the judge times, which are cuts too: {judge time: result}"""
    I.attach_store(con, None, name=store, horizon_days=horizon_days)
    a, out = pd.Timestamp("2000-01-01"), {}
    for b in sorted(set(cuts) | set(judges)):
        con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?", [a, b])
        I.update(con, "chunk", "detectors", store=store, upto=b)
        if b in judges:
            out[b] = I.judge(con, "detectors", store=store, end=b, outputs=("tables", "bins"))
        a = b
    return out


def carry(con, code, store=ST):
    return con.execute(f"SELECT TimeStamp, Parameter FROM {store}.health_carry WHERE EventId = {code} ORDER BY 1, 2").fetchall()


@pytest.fixture(scope="module")
def stuck():
    s = stuck_events()
    con = connect(s["ev"], s["det"])
    days = [s["day0"] + k * DAY for k in range(1, 8)]
    runs = run_cuts(con, days, days)
    return dict(s, con=con, runs=runs, days=days, b=pd.concat([r.bins for r in runs.values()], ignore_index=True))


@pytest.mark.parametrize("kind, det", [("Vehicle", 1), ("Ped", 2)])
def test_a_stuck_on_scores_every_bin_from_its_on_to_its_off_past_the_window(stuck, kind, det):
    b = stuck["b"]
    x = b[(b.DetectorType == kind) & (b.Detector == det) & (b.Label == "stuck on")].sort_values("TimeStamp")
    assert x.TimeStamp.iloc[0] == stuck["on"] and x.TimeStamp.iloc[-1] == stuck["off"].floor("15min")
    assert x.TimeStamp.diff().dropna().eq(BIN).all()
    assert (x.Score.iloc[:-1] == 1).all() and x.Score.iloc[-1] == pytest.approx(0.5)
    assert x.Score.sum() / 4 == pytest.approx((stuck["off"] - stuck["on"]) / pd.Timedelta(hours=1))
    assert not b[(b.DetectorType == kind) & (b.Detector == det)].TimeStamp.lt(stuck["on"]).any()


def test_a_stuck_on_out_of_the_window_has_no_finding_only_its_bins(stuck):
    day = stuck["on"].normalize() + 4 * DAY     # the ON is days out of the 1-day window
    f = stuck["runs"][day].findings
    assert f[(f.Label == "stuck on") & (f.Detector.isin([1, 2]))].empty
    b = stuck["runs"][day].bins
    assert set(b[b.Label == "stuck on"].Detector) == {1, 2}


def test_the_stuck_detector_is_not_also_called_silent(stuck):
    b = stuck["b"]
    x = b[(b.Detector == 1) & (b.DetectorType == "Vehicle") & (b.TimeStamp >= stuck["on"]) & (b.TimeStamp < stuck["off"].floor("15min"))]
    assert set(x.Label) == {"stuck on"}


def test_each_bin_once_and_each_detectors_bin_adds_up_to_at_most_one(stuck):
    b = stuck["b"]
    assert not b.duplicated(["TimeStamp", "DeviceId", "DetectorType", "Detector", "Label"]).any()
    assert b.groupby(["TimeStamp", "DeviceId", "DetectorType", "Detector"]).Score.sum().max() <= 1 + 1e-5


def test_the_carry_holds_the_on_while_it_is_on_and_lets_it_go_after_its_off():
    s = stuck_events()
    con = connect(s["ev"], s["det"])
    I.attach_store(con, None, horizon_days=1)
    for k in range(1, 8):
        b = s["day0"] + k * DAY
        con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?", [b - DAY, b])
        I.update(con, "chunk", "detectors", upto=b)
        I.judge(con, "detectors", end=b, outputs=("tables", "bins"))
        held = s["on"] < b <= s["off"]
        assert [t for t, _ in carry(con, 9223)] == ([s["on"].to_pydatetime()] if held else []), b
        assert [(t, p // 1000) for t, p in carry(con, 9232)] == ([(s["on"].to_pydatetime(), 2)] if held else []), b


def test_a_new_store_from_export_and_rebuild_keep_the_on():
    s = stuck_events()
    con = connect(s["ev"], s["det"])
    days = [s["day0"] + k * DAY for k in range(1, 5)]
    run_cuts(con, days, days)
    before = carry(con, 9232)
    assert before
    I.rebuild(con, "chunk", "detectors", upto=days[-1])
    assert carry(con, 9232) == before


def test_nothing_is_rated_without_bins_and_health_on_a_window_is_unchanged():
    s = stuck_events()
    con = connect(s["ev"], s["det"])
    days = [s["day0"] + k * DAY for k in range(1, 5)]
    I.attach_store(con, None, horizon_days=1)
    for b in days:
        con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?", [b - DAY, b])
        I.update(con, "chunk", "detectors", upto=b)
        r = I.judge(con, "detectors", end=b)
    assert r.bins is None and not carry(con, 9223) and not carry(con, 9232)
    f = F.health(con, "events", "detectors", days[-1] - DAY, days[-1], bin_origin="clock")
    assert f[(f.Label == "stuck on") & (f.Detector.isin([1, 2]))].empty


def test_no_row_while_the_signal_is_silent_and_on_again_when_it_returns():
    s = stuck_events()
    gone_a, gone_b = s["on"] + pd.Timedelta(days=2), s["on"] + pd.Timedelta(days=2, hours=5)
    ev = s["ev"][~((s["ev"].Timestamp >= gone_a) & (s["ev"].Timestamp < gone_b))]
    con = connect(ev, s["det"])
    days = [s["day0"] + k * DAY for k in range(1, 8)]
    b = pd.concat([r.bins for r in run_cuts(con, days, days).values()], ignore_index=True)
    for kind, det in (("Vehicle", 1), ("Ped", 2)):
        x = b[(b.DetectorType == kind) & (b.Detector == det) & (b.Label == "stuck on")]
        assert x.TimeStamp.max() == s["off"].floor("15min")
        assert not x.TimeStamp.between(gone_a, gone_b - BIN).any()
        assert x.TimeStamp.between(gone_b, gone_b + 8 * BIN).sum() >= 7


def test_a_button_whose_phase_stops_walking_stops_scoring():
    s = stuck_events()
    stop = s["on"] + pd.Timedelta(days=2)
    ev = s["ev"][~((s["ev"].EventId.isin([21, 22, 23])) & (s["ev"].Parameter == 2) & (s["ev"].Timestamp >= stop))]
    con = connect(ev, s["det"])
    days = [s["day0"] + k * DAY for k in range(1, 7)]
    b = pd.concat([r.bins for r in run_cuts(con, days, days).values()], ignore_index=True)
    x = b[(b.DetectorType == "Ped") & (b.Detector == 2) & (b.Label == "stuck on")]
    assert x.TimeStamp.max() == stop.floor("15min") - BIN or x.TimeStamp.max() == stop.floor("15min")


# --- each bin once, whatever the cuts ---------------------------------------------------------------------------------------------------------------------

def ordered(b):
    return b.sort_values(B.BIN_COLS).reset_index(drop=True)


def faults():
    return synth.make(2, 7, seed=5, faults=True)


def judged_bins(con, judges, cuts=(), horizon_days=3):
    runs = run_cuts(con, list(cuts) + list(judges), judges, horizon_days=horizon_days)
    return ordered(pd.concat([r.bins for r in runs.values()], ignore_index=True))


@pytest.mark.parametrize("seed", [1, 2])
def test_bins_do_not_depend_on_the_cuts(seed):
    ev, det = faults()
    day0 = ev.Timestamp.min().floor("D")
    judges = [day0 + k * DAY for k in range(2, 7)]
    whole = judged_bins(connect(ev, det), judges)
    assert len(whole) > 0
    rng = np.random.default_rng(seed)
    cuts = [day0 + pd.Timedelta(seconds=int(x)) for x in rng.integers(3600, 6 * 86400, 12)]
    pd.testing.assert_frame_equal(judged_bins(connect(ev, det), judges, cuts), whole)


def test_nothing_is_written_twice_and_a_bin_never_passes_one():
    ev, det = faults()
    day0 = ev.Timestamp.min().floor("D")
    judges = [day0 + k * DAY for k in range(2, 7)]
    con = connect(ev, det)
    runs = run_cuts(con, judges, judges, horizon_days=3)
    for t, r in runs.items():
        if len(r.bins):
            assert r.bins.TimeStamp.min() >= t - 3 * DAY and r.bins.TimeStamp.max() < t
    b = pd.concat([r.bins for r in runs.values()], ignore_index=True)
    assert not b.duplicated(LATE_KEY).any()
    assert b.groupby(LATE_KEY[:-1]).Score.sum().max() <= 1 + 1e-5
    kept = con.execute(f"SELECT TimeStamp, DeviceId, DetectorType, Detector, Label, Score FROM {ST}.health_bins_released").df()
    want = b[b.TimeStamp >= judges[-1] - 3 * DAY]
    pd.testing.assert_frame_equal(ordered(kept), ordered(want), check_dtype=False)


def test_the_store_equals_a_health_of_each_window_with_no_store():
    ev, det = faults()
    day0 = ev.Timestamp.min().floor("D")
    start, end = day0 + DAY, day0 + 6 * DAY
    con = connect(ev[ev.Timestamp >= start], det)
    batch = ordered(B.health_bins(con, "events", "detectors", start, end, horizon_days=3))
    store = judged_bins(connect(ev[ev.Timestamp >= start], det), [start + k * DAY for k in range(1, 6)])
    assert len(batch) > 0
    pd.testing.assert_frame_equal(store, batch, check_dtype=False)


# --- late finds: a later judge's first find in a bin already released ---------------------------------------------------------------------------------------

def row(t, label, score, det=1):
    return {"TimeStamp": t, "DeviceId": "S1", "DetectorType": "Vehicle", "Detector": det, "Label": label, "Score": score}


def table(*rows):
    return pd.DataFrame(list(rows))[B.BIN_COLS]


def test_a_late_row_is_a_new_row_and_an_existing_one_is_never_written_again():
    got = B.late_rows(table(row(T, "silent", 0.4), row(T + BIN, "silent", 0.5)), table(row(T, "silent", 0.2)), T + 2 * BIN)
    assert got.TimeStamp.tolist() == [T + BIN] and got.Score.tolist() == [0.5]


def test_a_late_row_of_another_label_or_detector_is_new():
    got = B.late_rows(table(row(T, "spiky counts", 0.2), row(T, "silent", 0.3, det=2)), table(row(T, "silent", 0.2)), T + BIN)
    assert sorted(zip(got.Label, got.Detector)) == [("silent", 2), ("spiky counts", 1)]


def test_late_rows_share_what_the_bin_has_left_and_a_full_bin_gets_a_row_with_score_zero():
    got = B.late_rows(table(row(T, "spiky counts", 0.5)), table(row(T, "silent", 0.7)), T + BIN)
    assert got.Score.tolist() == [pytest.approx(0.3)]
    two = B.late_rows(table(row(T, "spiky counts", 0.5), row(T, "counts dropped", 0.5)), table(row(T, "silent", 0.5)), T + BIN)
    assert two.Score.tolist() == [pytest.approx(0.25), pytest.approx(0.25)]
    assert B.late_rows(table(row(T, "spiky counts", 0.5)), table(row(T, "silent", 1.0)), T + BIN).Score.tolist() == [0]


def test_not_enough_data_is_not_a_late_find_but_for_the_day_being_finished():
    day = T.normalize()
    found = table(row(day - DAY, F.TOO_FEW, None), row(day, F.TOO_FEW, None))
    assert B.late_rows(found.iloc[:1], table(row(day - DAY + BIN, "silent", 0.2)), day).empty
    got = B.late_rows(found, table(row(day - DAY + BIN, "silent", 0.2)), day + pd.Timedelta(hours=14))
    assert got.TimeStamp.tolist() == [day] and got.Score.isna().all()


class Scripted:
    """a store over real rows of one synthetic signal, whose judges find what the test says: a judge at day k of the verdict rows given for k, (detector, start, end, label, confidence)"""

    def __init__(self, monkeypatch, script, horizon_days=3):
        self.s, self.script, self.k = stuck_events(), script, None
        self.con = connect(self.s["ev"], self.s["det"])
        self.dev, self.day0 = self.s["dev"], self.s["day0"]
        I.attach_store(self.con, None, horizon_days=horizon_days)
        self.upto = pd.Timestamp("2000-01-01")
        monkeypatch.setattr(F, "tables", lambda con, *a, **k: self.told())

    def told(self):
        rows = self.script.get(self.k) or []
        cols = F.FINDING_COLS
        f = pd.DataFrame([{"DeviceId": self.dev, "DetectorType": "Vehicle", "Detector": d, "Start": a, "End": z, "Label": lab, "Confidence": c, "Detail": ""} for d, a, z, lab, c in rows], columns=cols)
        if not rows:
            f = f.astype({"DeviceId": object, "DetectorType": object, "Detector": "int64", "Start": "datetime64[us]", "End": "datetime64[us]", "Label": object, "Confidence": "float64", "Detail": object})
        s = pd.DataFrame({"DeviceId": self.dev, "DetectorType": "Vehicle", "Detector": sorted({r[0] for r in rows}) or [99], "Date": self.day0, "Status": "bad"})[F.STATUS_COLS]
        return f, s

    def at(self, t):
        return pd.Timestamp(t) if not isinstance(t, (int, float)) else self.day0 + t * DAY

    def judge(self, k, con=None, **kw):
        con, t = con or self.con, self.at(k)
        self.k = k
        con.execute("CREATE OR REPLACE TEMP TABLE chunk AS SELECT * FROM events WHERE Timestamp >= ? AND Timestamp < ?", [self.upto, t])
        I.update(con, "chunk", "detectors", upto=t)
        self.upto = t
        return I.judge(con, "detectors", end=t, outputs=("tables", "bins"), **kw).bins

    def released(self):
        return self.con.execute(f"SELECT count(*) FROM {ST}.health_bins_released").fetchall()[0][0]


def _hours(day0, k, a, z=None):
    return day0 + k * DAY + pd.Timedelta(hours=a), day0 + k * DAY + pd.Timedelta(hours=a + 1 if z is None else z)


def test_a_later_judge_releases_what_it_finds_in_released_bins_once_and_old_rows_stay(monkeypatch):
    day0 = stuck_events()["day0"]
    first = (1, *_hours(day0, 2, 10), "silent", 0.9)
    found_late = (2, *_hours(day0, 2, 5), "stuck on", 0.9)
    s = Scripted(monkeypatch, {3: [first], 4: [first, found_late], 5: [first, found_late], 6: [first, found_late]})
    a = s.judge(3)
    assert len(a) == 4 and set(a.Detector) == {1}
    b = s.judge(4)
    assert set(b.Detector) == {2} and b.TimeStamp.min() == day0 + 2 * DAY + pd.Timedelta(hours=5) and b.TimeStamp.max() < day0 + 3 * DAY
    assert len(b) == 4 and (b.Score == 1).all()
    assert s.judge(5).empty and s.judge(6).empty


def test_judging_the_same_time_again_releases_nothing(monkeypatch):
    day0 = stuck_events()["day0"]
    s = Scripted(monkeypatch, {3: [(1, *_hours(day0, 2, 10), "silent", 0.9)]})
    assert len(s.judge(3)) == 4
    assert s.judge(3).empty and s.released() == 4


def test_the_store_keeps_what_it_released_only_for_the_window(monkeypatch):
    day0 = stuck_events()["day0"]
    s = Scripted(monkeypatch, {3: [(1, *_hours(day0, 2, 10), "silent", 0.9)], 4: [], 5: [], 6: []})
    s.judge(3), s.judge(4), s.judge(5)
    assert s.released() == 4
    s.judge(6)
    assert s.released() == 0


def test_a_longer_horizon_does_not_write_again_what_the_store_let_go(monkeypatch):
    day0 = stuck_events()["day0"]
    old = (1, *_hours(day0, 2, 10), "silent", 0.9)
    s = Scripted(monkeypatch, {3: [old], 4: [old], 5: [old], 6: [old]})
    assert len(s.judge(3, horizon_days=1)) == 4
    assert s.judge(4, horizon_days=1).empty and s.judge(5, horizon_days=1).empty
    assert s.released() == 0
    assert s.judge(6, horizon_days=3).empty


def test_a_store_from_before_the_table_adds_no_late_row_before_its_last_bins(monkeypatch):
    day0 = stuck_events()["day0"]
    late = (2, *_hours(day0, 2, 5), "stuck on", 0.9)
    newer = (2, *_hours(day0, 3, 5), "stuck on", 0.9)
    s = Scripted(monkeypatch, {3: [], 4: [late, newer]})
    s.judge(3)
    s.con.execute(f"DROP TABLE {ST}.health_bins_released")
    s.con.execute(f"DELETE FROM {ST}.health_meta WHERE Key = 'bins_known_from'")
    b = s.judge(4)
    assert b.TimeStamp.min() == newer[1] and len(b) == 4 and s.released() == 4


def test_what_was_released_comes_along_in_an_export(monkeypatch, tmp_path):
    day0 = stuck_events()["day0"]
    old = (1, *_hours(day0, 2, 10), "silent", 0.9)
    s = Scripted(monkeypatch, {3: [old], 4: [old]})
    assert len(s.judge(3)) == 4
    I.export_store(s.con, tmp_path)
    (tmp_path / "x").mkdir()
    other = connect(s.s["ev"], s.s["det"])
    I.attach_store(other, None)
    I.import_store(other, tmp_path)
    assert s.judge(4, con=other).empty


def test_each_daily_verdict_releases_the_whole_past_day(stuck):
    days = stuck["days"]
    on, off = stuck["on"], stuck["off"]
    whole = [t for t in days if on <= t - DAY and t <= off]
    assert whole
    for t in whole:
        b = stuck["runs"][t].bins
        x = b[(b.DetectorType == "Vehicle") & (b.Detector == 1) & (b.Label == "stuck on")].sort_values("TimeStamp")
        assert x.TimeStamp.tolist() == list(pd.date_range(t - DAY, t - BIN, freq=BIN)), t
        assert (x.Score == 1).all()


def test_a_verdict_after_a_judge_in_the_day_still_gives_the_whole_day(monkeypatch):
    day0 = stuck_events()["day0"]
    morning = (1, *_hours(day0, 3, 8, 22), "silent", 0.9)
    s = Scripted(monkeypatch, {3: [], day0 + 3 * DAY + pd.Timedelta(hours=14): [], 4: [morning]})
    s.judge(3)
    assert s.judge(day0 + 3 * DAY + pd.Timedelta(hours=14)).empty
    b = s.judge(4)
    assert b.TimeStamp.tolist() == list(pd.date_range(morning[1], morning[2] - BIN, freq=BIN))


def test_a_new_rating_of_an_on_covers_bins_released_before_it():
    # an ON from 23:50: too young to call at midnight, rated the next day; its first ten minutes are in a bin already released
    s = stuck_events()
    on = s["day0"] + 2 * DAY - pd.Timedelta(minutes=10)
    off = s["day0"] + 4 * DAY
    dev, ev = s["dev"], s["ev"]
    margin = pd.Timedelta(minutes=5)
    keep = ~(ev.EventId.isin([81, 82]) & (ev.Parameter == 7) & (ev.Timestamp >= on - margin) & (ev.Timestamp <= off + margin))
    add = pd.DataFrame({"DeviceId": dev, "Timestamp": [on, off], "EventId": [82, 81], "Parameter": [7, 7]})
    e = pd.concat([ev[keep], add], ignore_index=True).astype({"EventId": "int32", "Parameter": "int32"})
    days = [s["day0"] + k * DAY for k in range(1, 6)]
    runs = run_cuts(connect(e, s["det"]), days, days, horizon_days=3)
    b = pd.concat([r.bins for r in runs.values()], ignore_index=True)
    assert not b.duplicated(LATE_KEY).any()
    found = runs[s["day0"] + 3 * DAY].bins
    x = found[(found.DetectorType == "Vehicle") & (found.Detector == 7) & (found.Label == "stuck on")]
    assert x.TimeStamp.min() == on.floor("15min") and x.Score.iloc[0] == pytest.approx(10 / 15, abs=1e-4)
    assert (b[(b.TimeStamp < s["day0"] + 2 * DAY) & (b.Detector == 7) & (b.DetectorType == "Vehicle") & (b.Label == "stuck on")]).TimeStamp.min() == on.floor("15min")


def test_a_brief_press_inside_a_stuck_on_past_the_window_leaves_its_bins_unbroken():
    # a one-second OFF, days after the ON left the 1-day window: the button goes on stuck and the press stays in the log
    s = stuck_events()
    blip = s["on"] + pd.Timedelta(days=3, hours=1, minutes=3)
    add = pd.DataFrame({"DeviceId": s["dev"], "Timestamp": [blip, blip + pd.Timedelta(seconds=1), blip + pd.Timedelta(seconds=1.1)], "EventId": [89, 90, 45], "Parameter": [2, 2, 2]})
    ev = pd.concat([s["ev"], add], ignore_index=True).astype({"EventId": "int32", "Parameter": "int32"})
    con = connect(ev, s["det"])
    days = [s["day0"] + k * DAY for k in range(1, 8)]
    b = pd.concat([r.bins for r in run_cuts(con, days, days).values()], ignore_index=True)
    x = b[(b.DetectorType == "Ped") & (b.Detector == 2) & (b.Label == "stuck on")].sort_values("TimeStamp")
    assert not x.duplicated(LATE_KEY).any()
    assert x.TimeStamp.iloc[0] == s["on"] and x.TimeStamp.iloc[-1] == s["off"].floor("15min")
    assert x.TimeStamp.diff().dropna().eq(BIN).all()
    assert x.Score.sum() / 4 == pytest.approx((s["off"] - s["on"]) / pd.Timedelta(hours=1), abs=0.01)
    with_blip = PH.ped_health(con, "events", start=blip - DAY, end=blip + DAY).set_index("Input").loc[2, "presses"]
    con.execute("DELETE FROM events WHERE EventId IN (89, 90) AND Timestamp BETWEEN ? AND ?", [blip, blip + pd.Timedelta(seconds=1)])
    assert with_blip == PH.ped_health(con, "events", start=blip - DAY, end=blip + DAY).set_index("Input").loc[2, "presses"] + 1
