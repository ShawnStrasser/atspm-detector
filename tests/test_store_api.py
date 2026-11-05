"""Stage 6 of the incremental-health design: the public entry points in the findings layout, judge's tables, output layers, rebuild,
export / import and the guards.

T6.1  judge(...).findings / .status equal health(start=T - H, end=T, bin_origin="clock", summary=True), over any cutting of the events
T6.2  the 15-minute table and the active table's states from fixed findings
T6.3  rebuild() of one device equals a store fed the right events from the start
T6.4  export_store then import_store on another connection: the next judge is identical
T6.5  an old format version, a window before the trim: a clear message
plus: the same findings across the old and the new layout (frozen oracle -> tables() == health()); health_sql; workers; one status row per key."""
import duckdb
import pandas as pd
import pytest

import atspm_detector as ad
import synth
from atspm_detector import findings as F
from atspm_detector import health_duckdb as H
from atspm_detector import incremental as I
from atspm_detector import ped_health_duckdb as P
from legacy import health_duckdb_v2 as H0
from legacy import ped_health_duckdb_v3 as P0

ST = "health_store"


@pytest.fixture(scope="module")
def log():
    ev, det = synth.make(3, 2, seed=3)
    c = duckdb.connect()
    c.execute("SET threads = 4; SET memory_limit = '3GB'")
    c.register("e_df", ev)
    c.register("d_df", det)
    c.execute("CREATE TABLE events AS SELECT * FROM e_df ORDER BY DeviceId, Timestamp; CREATE TABLE detectors AS SELECT * FROM d_df")
    c.unregister("e_df"), c.unregister("d_df")
    last = c.execute("SELECT max(Timestamp) FROM events").fetchall()[0][0]
    end = pd.Timestamp(last).floor("15min") + pd.Timedelta(minutes=15)
    yield c, end
    c.close()


def fed(c, cuts, store=ST, horizon_days=1, **kw):
    I.attach_store(c, name=store, horizon_days=horizon_days, **kw)
    for cut in cuts:
        I.update(c, "events", "detectors", store=store, upto=cut)


def cuts_of(end, step_h=6):
    start = end - pd.Timedelta(days=2)
    return [start + pd.Timedelta(hours=step_h * k) for k in range(1, int(48 / step_h) + 1)]


def batch(c, end, days=1, **kw):
    return ad.health(c, "events", "detectors", end - pd.Timedelta(days=days), end, bin_origin="clock", **kw)


def same(a, b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


# ---- the public entry points
def test_health_gives_the_two_tables_only_with_the_agreed_columns(log):
    c, end = log
    f, s = batch(c, end, summary=True)
    assert list(f.columns) == ["DeviceId", "DetectorType", "Detector", "Start", "End", "Label", "Confidence", "Detail"]
    assert list(s.columns) == ["DeviceId", "DetectorType", "Detector", "Date", "Status"]
    assert set(f.DetectorType) == {"Vehicle", "Ped"} and set(s.DetectorType) == {"Vehicle", "Ped"}
    assert not s.duplicated(["DeviceId", "DetectorType", "Detector", "Date"]).any()
    same(batch(c, end), f)


def test_a_status_table_with_two_rows_for_one_date_is_refused():
    s = pd.DataFrame({"DeviceId": ["A", "A"], "DetectorType": ["Ped", "Ped"], "Detector": [1, 1], "Date": ["2026-10-05"] * 2,
                      "Status": ["ok", "bad"]})
    with pytest.raises(ValueError, match="one row per"):
        F.check_status(s)
    F.check_status(s.assign(DetectorType=["Ped", "Vehicle"]))   # the numbers of the two types can overlap


def test_ped_health_is_the_ped_side_of_health(log):
    c, end = log
    f, s = ad.health(c, "events", vehicle=False, summary=True)
    g, t = ad.ped_health(c, "events", summary=True)
    same(f, g), same(s, t)
    assert set(f.DetectorType) == {"Ped"}


def test_health_sql_is_one_query_with_the_rows_of_health(log):
    c, end = log
    kw = dict(start=end - pd.Timedelta(days=1), end=end, bin_origin="clock", probabilities=True)
    f, s = F.health_sql("events", "detectors", summary=True, **kw)
    want_f, want_s = batch(c, end, ped=False, summary=True)
    assert len(c.extract_statements(f)) == 1
    same(F._finish(c.execute(f).df(), F.FINDING_COLS), want_f)
    same(F._finish(c.execute(s).df(), F.STATUS_COLS), want_s)
    assert ad.health_sql("events", "detectors", **kw) == f


def test_the_same_findings_in_the_old_and_the_new_layout(log):
    """the frozen oracle's rows (old layout: limit_s, in_closure, whole_window, the reason strings), put through tables(), are health()'s."""
    c, end = log
    a = end - pd.Timedelta(days=1)
    d0, p0 = H0.health(c, "events", "detectors", a, end, periods=True)
    i0, q0 = P0.ped_health(c, "events", a, end, periods=True)
    win = F.event_windows(c, "events", "detectors", a, end)
    pin = F.event_windows(c, "events", start=a, end=end, ped=True)
    f, s = ad.health(c, "events", "detectors", a, end, summary=True)
    f0, s0 = F.tables(c, d0, p0, i0, q0, win, pin, summary=True)
    same(f, f0), same(s, s0)
    assert len(f) and {"Vehicle", "Ped"} == set(f.DetectorType)


# ---- T6.1 judge's tables
def test_t61_judge_equals_batch_for_any_cutting(log):
    c, end = log
    want_f, want_s = batch(c, end, summary=True)
    for store, step in (("cut6", 6), ("cut48", 48)):
        fed(c, cuts_of(end, step), store=store)
        r = I.judge(c, "detectors", store=store, end=end, summary=True)
        same(r.findings, want_f), same(r.status, want_s)
        assert r.q15 is None and r.active is None
        same(I.judge(c, "detectors", store=store, end=end).findings, want_f)
        assert I._meta(c, store)["last_judge_end"] == end.isoformat()
        I.detach_store(c, store)


def test_judge_can_leave_ped_out_and_workers_change_nothing(log):
    c, end = log
    fed(c, cuts_of(end, 12), store="w")
    one = I.judge(c, "detectors", store="w", end=end, ped=False, summary=True)
    same(one.findings, batch(c, end, ped=False))
    r = I.judge(c, "detectors", store="w", end=end, ped=False, summary=True, workers=2, max_events=1)
    same(r.findings, one.findings), same(r.status, one.status)
    I.detach_store(c, "w")


# ---- T6.2 output layers
W0 = pd.Timestamp("2026-10-05 00:00")


def finding(det, a, z, label="stuck on", conf=0.9, kind="Vehicle"):
    return dict(DeviceId="S", DetectorType=kind, Detector=det, Start=pd.Timestamp(a), End=pd.Timestamp(z), Label=label, Confidence=conf, Detail="d")


def table(rows, cols=F.FINDING_COLS):
    return F.empty_findings() if not rows else pd.DataFrame(rows)[cols].astype({"Detector": "Int64"})


def stat(**by_det):
    return pd.DataFrame({"DeviceId": "S", "DetectorType": "Vehicle", "Detector": [int(k[1:]) for k in by_det], "Date": W0, "Status": list(by_det.values())}
                        ).astype({"Detector": "Int64"})


def test_t62_quarter_hours_since_the_last_judge():
    c = duckdb.connect()
    f = table([finding(1, "2026-10-05 10:05", "2026-10-05 10:35"), finding(1, "2026-10-05 10:20", "2026-10-05 10:25", conf=0.95),
               finding(1, "2026-10-05 10:05", "2026-10-05 10:35", "config note", 0.0), finding(1, "2026-10-05 10:00", "2026-10-05 11:00", "lane closure", None),
               finding(2, "2026-10-05 10:05", "2026-10-05 10:35"), finding(3, "2026-10-05 10:05", "2026-10-05 10:35", conf=0.4)])
    q = F.q15(c, f, stat(d1="bad", d2="ok", d3="suspect"), pd.Timestamp("2026-10-05 10:15"), pd.Timestamp("2026-10-05 12:00"))
    assert list(q.columns) == F.Q15_COLS
    assert q.TimeStamp.astype(str).tolist() == ["2026-10-05 10:15:00", "2026-10-05 10:30:00"]
    assert q.Detector.tolist() == [1, 1] and q.Label.tolist() == ["stuck on"] * 2
    assert q.Confidence.tolist() == [0.95, 0.9] and q.Status.tolist() == ["bad", "bad"]


def active_store():
    c = duckdb.connect()
    I.attach_store(c, name=ST)
    c.execute(I._render.render("store.sql", device_type="VARCHAR", s=f"{ST}."))
    return c


def step(c, rows, status, end):
    n = F.active_next(c, f"{ST}.health_active", table(rows), status, pd.Timestamp(end))
    c.execute(f"DELETE FROM {ST}.health_active")
    c.register("n", n)
    c.execute(f"INSERT INTO {ST}.health_active BY NAME SELECT * FROM n")
    c.unregister("n")
    return n


def test_t62_active_states():
    c = active_store()
    e1, e2, e3, e4 = (f"2026-10-0{d} 00:00" for d in (6, 7, 8, 9))
    # new, reaching the window end: provisional; new and over already, on a flagged detector: confirmed; an ok detector: nothing
    n = step(c, [finding(1, "2026-10-05 20:00", e1), finding(2, "2026-10-05 08:00", "2026-10-05 09:00"), finding(3, "2026-10-05 08:00", "2026-10-05 09:00")],
             stat(d1="bad", d2="suspect", d3="ok"), e1)
    assert dict(zip(n.Detector, n.State)) == {1: "provisional", 2: "confirmed"}
    # seen again: confirmed, the times grown; missed once: stays
    n = step(c, [finding(1, e1, e2)], stat(d1="bad", d2="suspect"), e2)
    assert dict(zip(n.Detector, n.State)) == {1: "confirmed", 2: "confirmed"}
    r = n[n.Detector == 1].iloc[0]
    assert r.Start == pd.Timestamp("2026-10-05 20:00") and r.End == pd.Timestamp(e2) and r.Missed == 0
    assert r.FirstSeen == pd.Timestamp(e1) and r.LastSeen == pd.Timestamp(e2)
    # detector 2 missed twice running, detector 1 back to ok: both cleared, shown once
    n = step(c, [], stat(d1="ok", d2="suspect"), e3)
    assert dict(zip(n.Detector, n.State)) == {1: "cleared", 2: "cleared"}
    assert step(c, [], stat(d1="ok", d2="ok"), e4).empty


def test_judge_outputs_are_off_by_default_and_the_quarter_hours_do_not_overlap(log):
    c, end = log
    fed(c, cuts_of(end, 24)[:-1], store="o")
    t1 = end - pd.Timedelta(days=1)
    I.judge(c, "detectors", store="o", end=t1)
    assert c.execute("SELECT count(*) FROM o.health_active").fetchall()[0][0] == 0
    I.update(c, "events", "detectors", store="o", upto=end)
    r = I.judge(c, "detectors", store="o", end=end, outputs=("tables", "15min", "active"))
    assert r.status is None
    if len(r.q15):
        assert r.q15.TimeStamp.min() >= t1 and r.q15.TimeStamp.max() < end
    assert c.execute("SELECT count(*) FROM o.health_active").fetchall()[0][0] == len(r.active)
    with pytest.raises(ValueError, match="unknown outputs"):
        I.judge(c, "detectors", store="o", end=end, outputs=("15minutes",))
    I.detach_store(c, "o")


# ---- T6.3 rebuild
def test_t63_rebuild_of_one_device_equals_a_store_fed_the_right_events(log):
    c, end = log
    dev = c.execute("SELECT min(DeviceId) FROM events").fetchall()[0][0]
    other = c.execute("SELECT max(DeviceId) FROM events").fetchall()[0][0]
    hole = (end - pd.Timedelta(hours=20), end - pd.Timedelta(hours=8))
    c.execute(f"""CREATE OR REPLACE TEMP TABLE holed AS SELECT * FROM events
                  WHERE NOT (DeviceId = '{dev}' AND Timestamp >= '{hole[0]}' AND Timestamp < '{hole[1]}')""")
    I.attach_store(c, name="rb", horizon_days=1)
    for cut in cuts_of(end, 6):
        I.update(c, "holed", "detectors", store="rb", upto=cut)
    before = I.judge(c, "detectors", store="rb", end=end, summary=True)
    want_f, want_s = batch(c, end, summary=True)
    assert not before.status.equals(want_s) or not before.findings.equals(want_f)
    keep = c.execute(f"SELECT * FROM rb.health_signal_5min WHERE DeviceId = '{other}' ORDER BY ALL").df()
    stats = I.rebuild(c, "events", "detectors", store="rb", devices=[dev])
    assert stats.signals == 1
    same(c.execute(f"SELECT * FROM rb.health_signal_5min WHERE DeviceId = '{other}' ORDER BY ALL").df(), keep)
    r = I.judge(c, "detectors", store="rb", end=end, summary=True)
    same(r.findings, want_f), same(r.status, want_s)
    assert "rb_rebuild" not in {x[0] for x in c.execute("SELECT database_name FROM duckdb_databases()").fetchall()}
    # the store is left whole: updates go on, and a device with no events stays as it was
    I.rebuild(c, "events", "detectors", store="rb", devices=["no such device"])
    same(I.judge(c, "detectors", store="rb", end=end, summary=True).findings, want_f)
    I.detach_store(c, "rb")
    c.execute("DROP TABLE holed")


# ---- T6.4 export / import
def test_t64_export_then_import_gives_an_identical_next_judge(log, tmp_path):
    c, end = log
    fed(c, cuts_of(end, 12), store="ex")
    want = I.judge(c, "detectors", store="ex", end=end, summary=True)
    I.export_store(c, tmp_path / "copy", store="ex")
    d = duckdb.connect()
    d.execute("SET threads = 4; SET memory_limit = '3GB'")
    for t in ("events", "detectors"):
        d.register("x", c.execute(f"SELECT * FROM {t}").df())
        d.execute(f"CREATE TABLE {t} AS SELECT * FROM x")
        d.unregister("x")
    I.attach_store(d, str(tmp_path / "new.duckdb"), name="im")
    I.import_store(d, tmp_path / "copy", store="im")
    got = I.judge(d, "detectors", store="im", end=end, summary=True)
    same(got.findings, want.findings), same(got.status, want.status)
    for k in ("horizon_days", "ped", "through", "trimmed_before", "green_overlap"):
        assert I._meta(d, "im")[k] == I._meta(c, "ex")[k], k
    with pytest.raises(ValueError, match="already holds data"):
        I.import_store(d, tmp_path / "copy", store="im")
    I.detach_store(d, "im")
    d.close()
    I.detach_store(c, "ex")


# ---- T6.5 guards
def test_t65_an_export_or_store_of_another_version_is_refused(log, tmp_path):
    c, end = log
    fed(c, cuts_of(end, 48), store="g")
    I.export_store(c, tmp_path / "old", store="g")
    d = duckdb.connect()
    d.execute(f"""COPY (SELECT Key, CASE Key WHEN 'format_version' THEN '1' ELSE Value END AS Value
                  FROM read_parquet('{(tmp_path / "old" / "health_meta.parquet").as_posix()}'))
                  TO '{(tmp_path / "meta.parquet").as_posix()}' (FORMAT PARQUET)""")
    (tmp_path / "old" / "health_meta.parquet").unlink()
    (tmp_path / "meta.parquet").rename(tmp_path / "old" / "health_meta.parquet")
    I.attach_store(d, name="n")
    with pytest.raises(ValueError, match="format version 1"):
        I.import_store(d, tmp_path / "old", store="n")
    with pytest.raises(ValueError, match="no export"):
        I.import_store(d, tmp_path / "nothing", store="n")
    d.close()
    c.execute("UPDATE g.health_meta SET Value = '2' WHERE Key = 'format_version'")
    with pytest.raises(ValueError, match="format version 2.*start a new store"):
        I.attach_store(c, name="g")
    c.execute(f"UPDATE g.health_meta SET Value = '{I.FORMAT_VERSION}' WHERE Key = 'format_version'")
    with pytest.raises(ValueError, match="before the store's trim"):
        I.judge(c, "detectors", store="g", end=end, horizon_days=30)
    with pytest.raises(ValueError, match="quarter-hour"):
        I.judge(c, "detectors", store="g", end=end + pd.Timedelta(minutes=1))
    I.detach_store(c, "g")


def test_a_rebuild_of_an_empty_store_is_refused(log):
    c, _ = log
    I.attach_store(c, name="empty")
    with pytest.raises(ValueError, match="no events yet"):
        I.rebuild(c, "events", "detectors", store="empty")
    I.detach_store(c, "empty")


def test_the_old_engines_stay_under_the_tables(log):
    c, end = log
    a = end - pd.Timedelta(days=1)
    d, p = H.health(c, "events", "detectors", a, end, periods=True)
    i, q = P.ped_health(c, "events", a, end, periods=True)
    f = F.tables(c, d, p, i, q, F.event_windows(c, "events", "detectors", a, end), F.event_windows(c, "events", start=a, end=end, ped=True))
    same(f, ad.health(c, "events", "detectors", a, end))
