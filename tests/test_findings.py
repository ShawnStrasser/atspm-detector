"""The findings and status tables (findings.py): today's outputs of health() and ped_health() as one findings table and an
optional status table, with one status row per detector or button and date."""
import duckdb
import pandas as pd
import pytest
import synth

from atspm_detector import findings as F
from atspm_detector import health_duckdb as hq
from atspm_detector import ped_health_duckdb as pq

T = pd.Timestamp
START, END = T("2026-10-05 06:30"), T("2026-10-08 00:00")     # three calendar days, the first cut, ending at midnight
WIN = pd.DataFrame({"DeviceId": ["S"], "start": [START], "end": [END]})


def veh(*rows):
    """today's detector table: rows of (Detector, status, reason text after the label, categories, watch, config, note)."""
    out = [dict(DeviceId="S", Detector=d, health_status=st, health_score=0.5, health_reason=f"det {d}: P2 Presence (covers 2 lanes) - {why}",
                health_categories=cats, health_watch=wt, health_config=cfg, health_signal_note=note)
           for d, st, why, cats, wt, cfg, note in rows]
    return pd.DataFrame(out)[hq.OUT_COLS]


def per(*rows):
    """today's periods table: rows of (Detector, start, end, condition, severity, whole_window)."""
    out = [dict(DeviceId="S", Detector=d, start=T(a), end=T(z), condition=c, severity=s, limit_s=None, in_closure=False, whole_window=w)
           for d, a, z, c, s, w in rows]
    return pd.DataFrame(out, columns=hq.PERIOD_COLS)


def ped(*rows):
    """today's ped table: rows of (Input, status, reason, config)."""
    out = [dict(DeviceId="S", Input=i, Phase=2, Type="pulse", presses=50, status=st, reason=why, config=cfg) for i, st, why, cfg in rows]
    return pd.DataFrame(out)[pq.COLS]


OK = (1, "ok", "nothing unusual", "", "", "", "")
QUIET_PED = (1, "ok", "no problem found", "")


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def test_no_finding_no_row_and_summary_is_optional(con):
    f = F.tables(con, veh(OK), per(), windows=WIN)
    assert isinstance(f, pd.DataFrame) and f.empty and list(f.columns) == F.FINDING_COLS
    f, s = F.tables(con, veh(OK), per(), windows=WIN, summary=True)
    assert f.empty and list(s.columns) == F.STATUS_COLS
    assert s.Date.dt.strftime("%m-%d").tolist() == ["10-05", "10-06", "10-07"]
    assert s.Status.tolist() == ["ok"] * 3 and s.DetectorType.tolist() == ["Vehicle"] * 3


def test_each_finding_reads_its_own_evidence(con):
    why = ("Stuck on: held ON 6 min from Tue 01:32; not a queue: at the same traffic it was ON 3 % of the time; count zone ON up to 28 % "
           "in light traffic for 30 min; Counts fall: no actuations Mon 03:10-Mon 05:55; ~50 expected; also Tue 04:20-Tue 05:55 (~44); "
           "Too-fast actuations: 9 of 90 ONs follow the previous ON within 0.8 s (100 per 1,000; limit 50); counts each vehicle twice")
    d = veh((3, "bad", why, "Stuck on (bad); Counts fall (bad); Too-fast actuations (suspect)", "", "", ""))
    p = per((3, "2026-10-06 01:32", "2026-10-06 01:38", "stuck on", 0.6, False),
            (3, "2026-10-06 02:00", "2026-10-06 02:15", "count zone held ON", 0.55, False),
            (3, "2026-10-05 03:10", "2026-10-05 05:55", "silent", 0.7, False),
            (3, "2026-10-06 04:20", "2026-10-06 05:55", "silent", 0.51, False),
            (3, "2026-10-06 06:00", "2026-10-06 07:00", "too-fast actuations", 0.8, False))
    f = F.tables(con, d, p, windows=WIN)
    assert list(f.columns) == F.FINDING_COLS and len(f) == 5
    got = dict(zip(zip(f.Label, f.Start.dt.strftime("%d %H:%M")), f.Detail))
    assert got["stuck on", "06 01:32"] == "held ON 6 min from Tue 01:32; not a queue: at the same traffic it was ON 3 % of the time"
    assert got["count zone held ON", "06 02:00"] == "count zone ON up to 28 % in light traffic for 30 min"
    silent = "no actuations Mon 03:10-Mon 05:55; ~50 expected; also Tue 04:20-Tue 05:55 (~44)"
    assert got["silent", "05 03:10"] == silent and got["silent", "06 04:20"] == silent
    assert got["too-fast actuations", "06 06:00"].endswith("limit 50); counts each vehicle twice")
    r = f[f.Label == "stuck on"].iloc[0]
    assert (r.DeviceId, r.DetectorType, r.Detector, r.Start, r.End, r.Confidence) == ("S", "Vehicle", 3, T("2026-10-06 01:32"), T("2026-10-06 01:38"), 0.6)


def test_no_time_of_its_own_is_one_row_per_calendar_day(con):
    d = veh((3, "bad", "Busy at night: night level 41 % of its busiest hours; phase mates 7 %", "Busy at night (bad)", "", "", ""))
    f = F.tables(con, d, per((3, START, END, "busy at night", 0.8, True)), windows=WIN)
    assert f.Start.tolist() == [START, T("2026-10-06"), T("2026-10-07")]
    assert f.End.tolist() == [T("2026-10-06"), T("2026-10-07"), T("2026-10-08")]         # end at midnight: no fourth day
    assert set(f.Label) == {"busy at night"} and set(f.Confidence) == {0.8} and f.Detail.nunique() == 1


def test_the_second_past_the_last_event_is_not_a_new_day(con):
    # a default window ends one second past the last event: a log ending at 23:59:59.9 still has this day as its last
    w = pd.DataFrame({"DeviceId": ["S"], "start": [T("2026-10-05")], "end": [T("2026-10-07 00:00:00.9")]})
    f, s = F.tables(con, veh((1, "not_enough_data", "too few actuations to judge", "", "", "", "")), per(), windows=w, summary=True)
    assert s.Date.tolist() == [T("2026-10-05"), T("2026-10-06")]
    assert f.End.tolist() == [T("2026-10-06"), T("2026-10-07")]


def test_a_time_of_its_own_stays_one_row_over_many_days(con):
    d = veh((3, "bad", "Stuck on: held ON 2 days from Mon 07:00", "Stuck on (bad)", "", "", ""))
    f = F.tables(con, d, per((3, "2026-10-05 07:00", "2026-10-07 07:00", "stuck on", 1.0, False)), windows=WIN)
    assert len(f) == 1 and f.Start[0] == T("2026-10-05 07:00") and f.End[0] == T("2026-10-07 07:00")


def test_a_window_inside_one_day_has_one_day(con):
    w = pd.DataFrame({"DeviceId": ["S"], "start": [T("2026-10-05 06:30")], "end": [T("2026-10-05 07:00")]})
    f, s = F.tables(con, veh((1, "not_enough_data", "too few actuations to judge", "", "", "", "")), per(), windows=w, summary=True)
    assert len(s) == 1 and len(f) == 1 and (f.Start[0], f.End[0]) == (w.start[0], w.end[0])


def test_a_closure_is_a_label_with_its_own_text(con):
    why = ("Lane closure: Mon 10:00-Mon 12:00: ~400 of its vehicles moved to det 2 (P6); Tue 10:00-Tue 11:00: ~90 of its vehicles moved to det 2 (P6); "
           "Timing plan: Mon 18:00-Mon 19:00: silent while coordination pattern 254 is in effect; its traffic moves to det 5")
    d = veh((3, "lane_closure", why, "Lane closure (info); Timing plan (info)", "", "", ""))
    p = per((3, "2026-10-06 10:00", "2026-10-06 11:00", "lane closure", None, False),
            (3, "2026-10-05 10:00", "2026-10-05 12:00", "lane closure", None, False),
            (3, "2026-10-05 18:00", "2026-10-05 19:00", "timing plan", None, False))
    f = F.tables(con, d, p, windows=WIN)
    assert f.Label.tolist() == ["lane closure", "timing plan", "lane closure"] and f.Confidence.isna().all()
    assert not {"limit_s", "in_closure", "whole_window"} & set(f.columns)
    assert f.Detail[0].startswith("Mon 10:00-Mon 12:00: ~400") and f.Detail[0].endswith("(P6)")
    assert f.Detail[2].startswith("Tue 10:00-Tue 11:00: ~90") and f.Detail[2].endswith("(P6)")
    assert f.Detail[1] == "Mon 18:00-Mon 19:00: silent while coordination pattern 254 is in effect; its traffic moves to det 5"


def test_notes_become_labels_with_a_confidence(con):
    d = veh((1, "watch", "watch: borderline stuck on; no healthy phase mate to compare with", "", "borderline stuck on; no healthy phase mate to compare with", "", "note A"),
            (2, "ok", "nothing unusual", "", "", "Extension time on a count zone: 3 of 90 ONs logged again without an OFF", "note A"),
            (4, "ok", "nothing unusual", "", "", "Set to pulse but holds ON: 3 ONs over 5 s, longest 9 s", "note A"),
            (5, "not_enough_data", "too few actuations to judge", "", "", "", "note A"))
    d["health_signal_note"] = "5 of 6 detectors at this signal had an unusual night on Tue 06 Oct: a signal-wide event or an input change, not one bad detector; 3 detectors (det 1, 2, 4) held ON together from Mon 01:27 for 13 min (same start and end): one common failure, check the cabinet first"
    f = F.tables(con, d, per(), windows=WIN)
    assert len(f) == 3 * 7                                  # no time of their own: three days each
    one = f[f.Start == START].set_index("Label")
    assert one.Confidence["borderline stuck on"] == F.WATCH_CONFIDENCE
    assert one.Confidence["no healthy phase mate to compare with"] == F.WATCH_CONFIDENCE
    assert one.Detail["extension time on a count zone"].startswith("Extension time") and one.Detector["extension time on a count zone"] == 2
    assert one.Detail["set to pulse but holds ON"].startswith("Set to pulse")
    assert one.Detail["not_enough_data"] == "too few actuations to judge"
    for lab in ("extension time on a count zone", "set to pulse but holds ON", "not_enough_data",
                "unusual night at many detectors", "detectors held ON together"):
        assert one.Confidence[lab] == F.NOTE_CONFIDENCE, lab
    # a signal note speaks of the signal: one row a day, no detector, however many detectors repeat it
    assert one.Detector["unusual night at many detectors"] is pd.NA and one.Detector["detectors held ON together"] is pd.NA
    assert f.Label.isin(["config note", "signal note"]).sum() == 0


def test_a_detector_with_no_known_function(con):
    d = veh((7, "bad", "x", "", "", "", ""))
    d["health_reason"] = "det 7: stayed ON continuously for 20 min; went silent; its share of the traffic dropped"
    p = per((7, "2026-10-06 01:00", "2026-10-06 01:20", "stuck on", 0.6, False), (7, "2026-10-06 02:00", "2026-10-06 03:00", "silent", 0.6, False),
            (7, START, END, "counts dropped", 0.5, True))
    f = F.tables(con, d, p, windows=WIN).set_index("Label")
    assert f.Detail["stuck on"] == "stayed ON continuously for 20 min" and f.Detail["silent"] == "went silent"
    assert set(f.Detail["counts dropped"]) == {"its share of the traffic dropped"} and len(f.loc["counts dropped"]) == 3


def test_ped_rows(con):
    i = ped((1, "bad", "stuck ON: held 2 h through 3 walks (limit 5 min, 3 walks, or 3 min through 2 walks); no presses for 25 logged hours, "
                       "Sun 10-04 16:45 to Mon 10-05 17:45; its own level expects ~45 (chance about 1 in 60 allowing for clumps); last button event is an OFF, so not held on",
             "serves ped overlap 1: its presses bring that overlap's walk (no ped call is logged for an overlap); P2 on ped recall: it walks without presses; "
             "2 ped calls for P2 waited, up to 3 greens without a walk with no preemption during the wait (not a fault); "
             "omitted during preemption: 1 ped call for P2 waited up to 4 greens while a preemption ran (not a fault)"),
            (2, "not_enough_data", "only 3 presses: too few to judge", ""), (3, "ok", "no problem found", ""))
    p = per((1, "2026-10-05 07:00", "2026-10-05 09:00", "stuck on", 0.8, False), (1, "2026-10-05 16:45", "2026-10-06 17:45", "sudden stop", 0.5, False),
            (1, "2026-10-06 08:00", "2026-10-06 08:05", "call dropped", 0.0, False), (1, "2026-10-06 09:00", "2026-10-06 09:05", "call dropped", 0.0, False),
            (1, "2026-10-06 10:00", "2026-10-06 10:05", "omitted during preemption", 0.0, False))
    f, s = F.tables(con, ped_inputs=i, ped_periods=p, windows=WIN, summary=True)
    assert set(f.DetectorType) == {"Ped"} and set(s.DetectorType) == {"Ped"}
    get = lambda lab: f[f.Label == lab]                                                  # noqa: E731
    assert get("stuck on").Detail.iloc[0].startswith("stuck ON: held 2 h") and "no presses" not in get("stuck on").Detail.iloc[0]
    assert get("sudden stop").Detail.iloc[0].startswith("no presses for 25") and get("sudden stop").Detail.iloc[0].endswith("not held on")
    assert len(get("call dropped")) == 2 and get("call dropped").Detail.iloc[0].startswith("2 ped calls for P2 waited")
    assert len(get("omitted during preemption")) == 1 and get("omitted during preemption").Detail.iloc[0].startswith("omitted during preemption: 1 ped call")
    # the other notes and too few to judge: a row per day; a wait's note only on the wait's own rows
    assert len(get("serves ped overlap")) == 3 and len(get("on ped recall")) == 3 and len(get("not_enough_data")) == 3
    assert get("not_enough_data").Detector.iloc[0] == 2 and get("not_enough_data").Detail.iloc[0] == "only 3 presses: too few to judge"
    assert len(f) == 2 + 2 + 1 + 9
    assert 3 not in set(f.Detector)                                                      # no finding, no row
    assert s[s.Detector == 3].Status.tolist() == ["ok"] * 3 and s[s.Detector == 1].Status.tolist() == ["bad"] * 3


def test_vehicle_and_ped_numbers_can_overlap(con):
    i, v = ped((2, "suspect", "no presses for 9 logged hours", "")), veh((2, "bad", "Stuck on: held ON 9 h from Mon 07:00", "Stuck on (bad)", "", "", ""))
    f, s = F.tables(con, v, per((2, "2026-10-05 07:00", "2026-10-05 16:00", "stuck on", 0.9, False)), i,
                    per((2, "2026-10-06 07:00", "2026-10-06 16:00", "sudden stop", 0.5, False)), WIN, summary=True)
    assert f.DetectorType.tolist() == ["Vehicle", "Ped"] and f.Detector.tolist() == [2, 2]
    assert len(s) == 6 and not s.duplicated(F.STATUS_KEY).any()
    assert s[s.DetectorType == "Ped"].Status.tolist() == ["suspect"] * 3 and s[s.DetectorType == "Vehicle"].Status.tolist() == ["bad"] * 3


def test_vehicle_and_ped_windows_may_differ(con):
    pw = pd.DataFrame({"DeviceId": ["S"], "start": [T("2026-10-06 12:00")], "end": [T("2026-10-07 00:00")]})
    _, s = F.tables(con, veh(OK), per(), ped(QUIET_PED), per(), WIN, pw, summary=True)
    assert s.DetectorType.value_counts().to_dict() == {"Vehicle": 3, "Ped": 1}


def test_status_has_exactly_one_row_per_detector_and_date(con):
    two = pd.concat([veh(OK), veh((2, "bad", "Stuck on: held ON 9 h from Mon 07:00", "Stuck on (bad)", "", "", ""))])
    _, s = F.tables(con, two, per(), windows=WIN, summary=True)
    assert len(s) == 6 and s.groupby(F.STATUS_KEY, dropna=False).size().eq(1).all()
    # an exact repeat of a detector is one row; two verdicts for one detector and date are refused
    _, s = F.tables(con, pd.concat([veh(OK), veh(OK)]), per(), windows=WIN, summary=True)
    assert len(s) == 3
    clash = pd.concat([veh(OK), veh((1, "bad", "x", "", "", "", ""))])
    with pytest.raises(ValueError, match="one row per"):
        F.tables(con, clash, per(), windows=WIN, summary=True)
    with pytest.raises(ValueError, match="one row per"):
        F.check_status(pd.DataFrame({"DeviceId": ["S", "S"], "DetectorType": ["Ped", "Ped"], "Detector": [1, 1],
                                     "Date": [T("2026-10-05")] * 2, "Status": ["ok", "bad"]}))
    ok = pd.DataFrame({"DeviceId": ["S", "S"], "DetectorType": ["Ped", "Vehicle"], "Detector": [1, 1], "Date": [T("2026-10-05")] * 2, "Status": ["ok", "bad"]})
    assert F.check_status(ok) is ok


def test_windows_are_required_and_single(con):
    with pytest.raises(ValueError, match="need their windows"):
        F.tables(con, veh(OK), per())
    with pytest.raises(ValueError, match="no window for signal S"):
        F.tables(con, veh(OK), per(), windows=WIN.assign(DeviceId="other"))
    with pytest.raises(ValueError, match="has two windows"):
        F.tables(con, veh(OK), per(), windows=pd.concat([WIN, WIN], ignore_index=True))


def test_nothing_ran_gives_empty_typed_tables(con):
    f, s = F.tables(con, summary=True)
    assert f.empty and s.empty and list(f.columns) == F.FINDING_COLS and list(s.columns) == F.STATUS_COLS
    assert f.Start.dtype.kind == "M" and f.Confidence.dtype.kind == "f" and s.Date.dtype.kind == "M"
    assert F.tables(con, veh(OK).iloc[:0], per(), windows=WIN).empty


def test_device_ids_may_be_numbers(con):
    d, w = veh(OK).assign(DeviceId=7), WIN.assign(DeviceId=7)
    _, s = F.tables(con, d, None, ped(QUIET_PED).assign(DeviceId=7), None, w, summary=True)
    assert set(s.DeviceId) == {7} and len(s) == 6


def test_the_plan_keeps_the_sql_rules():
    for v, p in ((True, True), (True, False), (False, True)):
        assert F.build(v, p).lint() == []


def test_every_category_has_a_piece_and_every_condition_a_kind():
    pieces = {p[0] for p in F.PIECES}
    assert set(hq.CATEGORY.values()) <= pieces
    kinds = {k for k, _, _ in F.KINDS}
    assert {"stuck on", "count zone held ON", "silent", "silent at night", "counts dropped", "too-fast actuations", "too many for the traffic",
            "busy at night", "no daily pattern", "spiky counts", "counts too high", "misses vehicles at night", "lane closure", "timing plan",
            "pulse mode"} <= kinds


# ---- on real runs of today's health and ped health, over a synthetic log


@pytest.fixture(scope="module")
def log():
    ev, det = synth.make(2, 2, seed=3)
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.register("e_df", ev)
    c.register("d_df", det)
    c.execute("CREATE TABLE events AS SELECT * FROM e_df ORDER BY DeviceId, Timestamp; CREATE TABLE detectors AS SELECT * FROM d_df")
    c.unregister("e_df"), c.unregister("d_df")
    h, hp = hq.health(c, "events", "detectors", periods=True)
    i, ip = pq.ped_health(c, "events", periods=True)
    yield c, h, hp, i, ip, F.health(c, "events", "detectors", summary=True)
    c.close()


def test_entry_point_keeps_every_finding_and_status(log):
    c, h, hp, i, ip, (f, s) = log
    assert len(ip) and len(hp)
    assert not s.duplicated(F.STATUS_KEY).any()
    v, p = f[f.DetectorType == "Vehicle"], f[f.DetectorType == "Ped"]
    # each period with a time of its own is a row with its severity; a whole-window one a row on each day of its window
    for side, table, name in ((v, hp, "vehicle"), (p, ip, "ped")):
        own = table[~table.whole_window]
        m = own.assign(start=own.start.astype("datetime64[us]"), end=own.end.astype("datetime64[us]")).merge(
            side, left_on=["DeviceId", "Detector", "start", "end", "condition"], right_on=["DeviceId", "Detector", "Start", "End", "Label"], how="left")
        assert m.Label.notna().all(), name
        assert (m.severity.fillna(-1).round(6) == m.Confidence.fillna(-1).round(6)).all(), name
        assert m.Detail.ne("").all(), name
    # every judged detector and button has its status on every day, equal to today's
    sv = s[s.DetectorType == "Vehicle"].groupby(["DeviceId", "Detector"]).agg(st=("Status", "first"), n=("Status", "size"), u=("Status", "nunique")).reset_index()
    m = h.merge(sv, on=["DeviceId", "Detector"], how="outer")
    assert (m.health_status == m.st).all() and (m.n == 2).all() and (m.u == 1).all()
    sp = s[s.DetectorType == "Ped"].groupby(["DeviceId", "Detector"]).agg(st=("Status", "first"), n=("Status", "size")).reset_index()
    m = i.merge(sp, left_on=["DeviceId", "Input"], right_on=["DeviceId", "Detector"], how="outer")
    assert (m.status == m.st).all() and (m.n == 2).all()


def test_options_run_one_side_only(log):
    c, _, _, _, _, (f, s) = log
    only = F.health(c, "events", "detectors", ped=False)
    assert only.equals(f[f.DetectorType == "Vehicle"].reset_index(drop=True))
    only = F.health(c, "events", vehicle=False)
    assert only.equals(f[f.DetectorType == "Ped"].reset_index(drop=True))
    with pytest.raises(ValueError, match="nothing to judge"):
        F.health(c, "events", vehicle=False, ped=False)


def test_ped_runs_only_when_the_log_has_ped_events(log):
    c, _, _, _, _, (f, _) = log
    c.execute("CREATE TEMP TABLE no_ped AS SELECT * FROM events WHERE EventId NOT IN (89, 90)")
    try:
        got = F.health(c, "no_ped", "detectors")
        assert set(got.DetectorType) == {"Vehicle"}
        with pytest.raises(ValueError, match="no ped side"):
            F.health(c, "no_ped", "detectors", windows_table="w", ped=True)
    finally:
        c.execute("DROP TABLE no_ped")


def test_event_windows_follow_each_side(log):
    c, *_ = log
    w = F.event_windows(c, "events", "detectors", "2026-10-05 06:07", "2026-10-06 06:07")
    assert set(w.start) == {T("2026-10-05 06:07")} and set(w.end) == {T("2026-10-06 06:07")}
    w = F.event_windows(c, "events", "detectors", "2026-10-05 06:07", "2026-10-06 06:07", bin_origin="clock")
    assert set(w.start) == {T("2026-10-05 06:00")}
    w = F.event_windows(c, "events", start="2026-10-05 06:07", end="2026-10-06 06:07", ped=True)
    assert set(w.start) == {T("2026-10-05 06:07")} and len(w) == 2
    c.execute(f"CREATE TEMP TABLE wtab AS SELECT DISTINCT DeviceId, {T('2026-10-05').value} AS s_ns, {T('2026-10-06').value} AS e_ns FROM events")
    try:
        w = F.event_windows(c, windows_table="wtab")
        assert set(w.start) == {T("2026-10-05")} and set(w.end) == {T("2026-10-06")}
        f = F.health(c, "events", "detectors", windows_table="wtab")
        assert set(f.DetectorType) <= {"Vehicle"}
    finally:
        c.execute("DROP TABLE wtab")
