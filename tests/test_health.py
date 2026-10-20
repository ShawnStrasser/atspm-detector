"""Health (DuckDB SQL, note 153): the checks, their limits and measures on small logs, batch == alone."""
import json

import duckdb
import numpy as np
import pandas as pd

from atspm_detector import health
from atspm_detector import health_duckdb as hq

CHECKS = {"Stuck on", "Counts fall", "Misses vehicles at night", "Busy at night", "Too-fast actuations",
          "Too many for the traffic", "Spiky counts"}
REMOVED = ("Chattering", "Erratic time ON", "Unusual daily pattern", "ON longer than its kind", "Goes silent",
           "Count drops", "Count zone held ON", "undercounts vs partner", "Erratic counts")
T0 = pd.Timestamp("2026-09-28 00:00")


def test_categories_are_the_checks():
    assert set(hq.CATEGORY.values()) == CHECKS
    sql = hq.health_sql("e", "d")
    assert "Extension time on a count zone" in sql and "Set to pulse but holds ON" in sql


def test_limits_are_the_reviewed_cut_offs():
    lim = hq._json("health_limits.json")
    h = lim["h24"]
    for t in ("stuck_s", "silent_exp"):
        for key, (s, b) in h[t].items():
            assert 0 < s <= b, (t, key)
    # stuck on: bad starts at the user's minutes (sure types)
    for key, mins in (("Bike|1", 10), ("Count|1|pulse", 20), ("Yellow_Red|1", 20), ("Advance|1", 30), ("Mid|1", 35),
                      ("Presence|1", 60), ("Other|2+", 60)):
        assert abs(h["stuck_s"][key][1] - 60 * mins) < 1e-6, key
    tf = lim["too_fast"]["per_1000"]
    assert all(tf[f"{f}|2+"] == tf[f"{f}|1"] for f in hq.C7)            # 2+ lanes: the 1-lane limit
    assert lim["too_many_h24_veh_h_lane"][0] == 2100.0


def _con(rows, dets, end="2026-09-28 00:30"):
    """a connection with events (t seconds, EventId, Parameter) of one signal 'S' and its detectors table."""
    ev = pd.DataFrame(rows, columns=["t", "EventId", "Parameter"])
    ev = pd.DataFrame({"DeviceId": "S", "Timestamp": T0 + pd.to_timedelta(ev.t, unit="s"), "EventId": ev.EventId,
                       "Parameter": ev.Parameter})
    de = pd.DataFrame([dict(DeviceId="S", Detector=d, phase=p, function=f, lanes=ln) for d, p, f, ln in dets])
    con = duckdb.connect()
    con.register("e_df", ev)
    con.register("d_df", de)
    con.execute("CREATE TABLE events AS SELECT * FROM e_df; CREATE TABLE detectors AS SELECT * FROM d_df")
    con.execute(f"CREATE TABLE windows AS SELECT 'S' AS DeviceId, {T0.value} AS s_ns, "
                f"{pd.Timestamp(end).value} AS e_ns")
    return con


def _step(con, name):
    q = hq.build("events", "detectors", "windows", upto=name, probabilities=False)
    return con.sql(q.text(f"SELECT * FROM {name}")).df()


def test_too_fast_counts_every_on_to_on_gap():
    # channel 5 (Presence, 1 lane: gap limit 0.8 s): ONs at 0, 0.5 (fast), 0.7 (fast, logged again without an OFF),
    # 10.0, 10.8 (not fast: 8 ticks); channel 6 (Advance, 2 lanes: 0.3 s): 0, 0.2 (fast), 5, 5.3 (not fast)
    rows = [(0.0, 82, 5), (0.3, 81, 5), (0.5, 82, 5), (0.7, 82, 5), (1.0, 81, 5), (10.0, 82, 5), (10.4, 81, 5),
            (10.8, 82, 5), (11.0, 81, 5),
            (0.0, 82, 6), (0.1, 81, 6), (0.2, 82, 6), (0.25, 81, 6), (5.0, 82, 6), (5.1, 81, 6), (5.3, 82, 6),
            (5.4, 81, 6)]
    r = _step(_con(rows, [(5, 2.0, "Presence", 1.0), (6, 2.0, "Advance", 2.0)]), "tf").set_index("det")
    assert r.loc[5, "N"] == 5 and r.loc[5, "n_fast"] == 2
    assert r.loc[6, "N"] == 4 and r.loc[6, "n_fast"] == 1
    assert abs(r.loc[5, "fast_rate"] - 400.0) < 1e-9
    # pairs: channel 6: its short gap is followed by a normal one (a pair); channel 5: one of its two short gaps
    assert r.loc[6, "pair_share"] == 1.0 and r.loc[5, "pair_share"] == 0.5


def test_common_holds():
    # det 1 and 2 held ON together (start and end within 5 s): common; det 3 overlaps det 4 but ends 10 min later
    holds = [(100.0, 2000.0, 1), (103.0, 2004.0, 2), (5000.0, 6000.0, 3), (5100.0, 6600.0, 4), (9000.0, 9900.0, 5)]
    rows = [r for s, f, c in holds for r in ((s, 82, c), (f, 81, c))]
    rows += [(float(t), 1, 2) for t in range(0, 10800, 60)]                 # a log without comms gaps
    rows += [r for k in range(5) for r in ((7200.0 + 10 * k, 82, 1), (7201.0 + 10 * k, 81, 1))]   # 1 and 2: not twins
    con = _con(rows, [(c, 2.0, "Presence", 1.0) for c in range(1, 6)], end="2026-09-28 03:00")
    h = _step(con, "hold").merge(_step(con, "hcom"), on="hid").sort_values("s")
    assert h.n_ov_any.tolist() == [True, True, True, True, False]
    assert h.common.tolist() == [True, True, False, False, False]
    assert list(h.mates.iloc[0]) == [2]


def _three():
    """the bundled sample as three signals: as it is, without two detector channels, and with its phases renumbered;
    the detectors table from predict's answers."""
    from atspm_detector import predict
    from atspm_detector.check import SAMPLE
    ev = pd.read_parquet(SAMPLE)
    ev["Timestamp"] = pd.to_datetime(ev.Timestamp)
    out = predict(ev, min_actuations=1)
    o = out.dropna(subset=["phase_guess"])
    det = pd.DataFrame({"DeviceId": "A", "Detector": o.Detector, "phase": o.phase_guess, "function": o.function_guess,
                        "lanes": 1.0})
    for c in hq.PROB.values():
        det[c] = o[c].to_numpy()
    drop = sorted(det.Detector)[:2]
    ev_b = ev[~(ev.EventId.isin([81, 82]) & ev.Parameter.isin(drop))].assign(DeviceId="B")
    ev_c = ev.assign(DeviceId="C")
    col = ev_c.EventId.isin([1, 8, 9, 10])
    ev_c.loc[col, "Parameter"] = ev_c.loc[col, "Parameter"] + 20
    evs = [ev.assign(DeviceId="A"), ev_b, ev_c]
    dets = [det, det[~det.Detector.isin(drop)].assign(DeviceId="B"), det.assign(DeviceId="C", phase=det.phase + 20)]
    return evs, dets


def test_health_batch_equals_alone_and_sample_reports_only_the_checks():
    evs, dets = _three()
    con = duckdb.connect()

    def run(e, d):
        con.register("e_df", pd.concat(e, ignore_index=True))
        con.register("d_df", pd.concat(d, ignore_index=True))
        con.execute("CREATE OR REPLACE TABLE events AS SELECT * FROM e_df")
        con.execute("CREATE OR REPLACE TABLE detectors AS SELECT * FROM d_df")
        return health(con, "events", "detectors", start="2024-12-03 12:00", end="2024-12-03 12:31")
    batch = run(evs, dets)
    for e, d in zip(evs, dets):
        alone = run([e], [d]).reset_index(drop=True)
        b = batch[batch.DeviceId == alone.DeviceId.iat[0]].reset_index(drop=True)
        pd.testing.assert_frame_equal(alone, b)
    # the renumbered phases change only the "P<n>" labels
    a, c = batch[batch.DeviceId == "A"], batch[batch.DeviceId == "C"]
    assert a.health_status.tolist() == c.health_status.tolist()
    for cat in batch.health_categories.fillna(""):
        for part in [x for x in cat.split("; ") if x]:
            assert part.rsplit(" (", 1)[0] in CHECKS, part
    text = " ".join(batch.health_reason.fillna("")) + " " + " ".join(batch.health_bad_periods.fillna(""))
    for w in REMOVED:
        assert w not in text, w
    json.loads(batch.health_bad_periods.dropna().iloc[0] or "[]")
    assert np.isfinite(batch.health_score.dropna()).all()
