"""live.py, a run every 15 minutes with state carried as tables, each signal on its own: one big run equals 15-minute runs whatever the other signals do
(late, hours behind, out of order, in subsets), each run's own bin, late finds inside the previous day, midnight per signal, the clock guard, the ped history,
a state from before 2.2.0, the per-run rules."""
import os
import subprocess
import sys
import uuid

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synth

from atspm_detector import incremental as I
from atspm_detector import live as L

Q = pd.Timedelta(minutes=15)
DAY = pd.Timedelta(days=1)
KEY = ["TimeStamp", "DeviceId", "DetectorType", "Detector", "Label"]
T0 = pd.Timestamp("2026-10-05")
BASE = pd.Timestamp("2026-10-06 08:00")


def one_run(raw, det, state, history, rows_folder=None, name=None, ped_folder=None, result=None, **kw):
    """One run on its own connection, as a process of its own would be: (detector_health, health_state, health_rows, findings).
    ped_folder: hl_ped is saved there and read back as ped_history.  result: a list the LiveResult is added to."""
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.register("raw_in", raw)
    con.execute("CREATE TABLE raw_data AS SELECT DeviceId, Timestamp AS TimeStamp, EventId, Parameter FROM raw_in")
    con.unregister("raw_in")
    con.register("det_in", det)
    con.execute("CREATE TEMP TABLE dets AS SELECT * FROM det_in")
    con.unregister("det_in")
    if ped_folder is not None:
        kw["ped_history"] = str(ped_folder / "*.parquet")
    if len(raw) or kw.get("now") is not None or kw.get("complete_through") is not None:
        r = L.live_run(con, "raw_data", "dets", state=state, history=history, horizon_days=kw.pop("horizon_days", 2), **kw)
        found = r.findings
        if result is not None:
            result.append(r)
    else:
        found = L.empty_run(con, state)
    out, st, rows = (con.execute(f"SELECT * FROM {t}").df() for t in ("hl_final", "hl_state", "hl_rows"))
    if rows_folder is not None and len(rows):
        con.execute(f"COPY hl_rows TO '{rows_folder}/{name}.parquet' (FORMAT PARQUET)")
    if ped_folder is not None and con.execute("SELECT count(*) FROM hl_ped").fetchall()[0][0]:
        con.execute(f"COPY hl_ped TO '{ped_folder}/{name or uuid.uuid4().hex}.parquet' (FORMAT PARQUET)")
    leftover = con.execute("SELECT table_name FROM duckdb_tables() WHERE starts_with(table_name, 'hl_') AND table_name NOT IN ('hl_final', 'hl_state', 'hl_rows', 'hl_ped')").fetchall()
    assert not leftover
    con.close()
    return out, st, rows, found


def chain(ev, det, edges, folder, through_csv=False, **kw):
    """Runs cut at edges, carrying only saved tables.  through_csv: state and rows through CSV files.  Returns the rows of each run and the last state."""
    rows_dir = folder / "rows"
    rows_dir.mkdir(parents=True)
    state, outs = "", []
    state_path = folder / "state.csv"
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = ev[(ev.Timestamp >= a) & (ev.Timestamp < b)]
        history = str(rows_dir / ("*.csv" if through_csv else "*.parquet"))
        out, st, rows, _ = one_run(chunk, det, state, history, None if through_csv else rows_dir, f"r{i:05d}", **kw)
        if through_csv:
            st.to_csv(state_path, index=False)
            if len(rows):
                rows.to_csv(rows_dir / f"r{i:05d}.csv", index=False)
            state = str(state_path)
        else:
            state = st
        outs.append(out)
    return outs, st


def ordered(f):
    return f.sort_values(KEY).reset_index(drop=True)


def same_rows(a, b):
    pd.testing.assert_frame_equal(ordered(a), ordered(b), check_dtype=False)


START, END = T0, T0 + DAY + 3 * Q
# one long run, then 15-minute runs across midnight
EDGES = [START] + list(pd.date_range(T0 + DAY - 3 * Q, END, freq="15min"))


@pytest.fixture(scope="module")
def log():
    ev, det = synth.make(3, 2, seed=7)
    return ev[(ev.Timestamp >= START) & (ev.Timestamp < END)], det


@pytest.fixture(scope="module")
def big(log, tmp_path_factory):
    folder = tmp_path_factory.mktemp("big")
    (folder / "rows").mkdir()
    return one_run(log[0], log[1], "", str(folder / "rows" / "*.parquet"))


@pytest.fixture(scope="module")
def cut(log, tmp_path_factory):
    return chain(log[0], log[1], EDGES, tmp_path_factory.mktemp("cut")), EDGES


def state_rows(st):
    keep = st[~st.EventId.isin([L.TRIMMED, L.RUN_ID])]
    cols = ["EventId", "DeviceId", "Parameter", "TimeStamp", "Value"]
    return keep[cols].sort_values(cols).reset_index(drop=True)


def test_one_big_run_equals_15_minute_runs(big, cut):
    (outs, st), _ = cut
    assert len(big[0]) > 100
    same_rows(pd.concat(outs, ignore_index=True), big[0])
    pd.testing.assert_frame_equal(state_rows(st), state_rows(big[1]), check_dtype=False)




def test_an_ordinary_run_writes_only_its_own_bin(cut):
    (outs, _), edges = cut
    ordinary = [(a, o) for a, o in zip(edges[:-1], outs) if a.normalize() != a]
    assert sum(len(o) for _, o in ordinary) > 0
    for a, o in ordinary:
        assert o.TimeStamp.isin([a - Q, a]).all() and (o.TimeStamp.max() >= a - Q if len(o) else True)


def test_the_first_run_past_midnight_adds_rows_only_inside_the_previous_day(cut):
    (outs, _), edges = cut
    late = 0
    for a, b, o in zip(edges[1:-1], edges[2:], outs[1:]):
        old = o[o.TimeStamp < b - 2 * Q]
        if len(old):
            M = b.floor("D")
            assert (old.TimeStamp >= M - DAY).all() and (old.TimeStamp < M).all() and a - M < 2 * Q
            late += len(old)
    assert late > 0


def test_no_key_is_written_twice(cut):
    (outs, _), _ = cut
    assert not pd.concat(outs, ignore_index=True).duplicated(KEY).any()








def test_a_missing_state_path_raises_and_empty_text_is_a_first_run(log, tmp_path):
    ev, det = log
    with pytest.raises(FileNotFoundError, match="does not exist"):
        one_run(ev.head(50), det, str(tmp_path / "nope.parquet"), str(tmp_path / "*.parquet"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        one_run(ev.head(0), det, str(tmp_path / "nope.parquet"), str(tmp_path / "*.parquet"))
    one_run(ev.head(50), det, "", str(tmp_path / "*.parquet"))


def test_a_state_of_another_extractor_version_raises(log, tmp_path):
    ev, det = log
    _, st, _, _ = one_run(ev.head(5000), det, "", str(tmp_path / "*.parquet"))
    st.loc[st.EventId == L.VERSION, "Value"] = float(I.EXTRACTOR_VERSION) + 1
    with pytest.raises(ValueError, match="extractor version"):
        one_run(ev.iloc[5000:5100], det, st, str(tmp_path / "*.parquet"))




def test_a_run_with_no_rows_is_not_listed(tmp_path):
    """Every event past where the run reached, and no bin closed: no row file, and the next check must not miss one."""
    a = BASE
    raw = pd.DataFrame({"DeviceId": ["S1"], "Timestamp": [a + pd.Timedelta(minutes=1)], "EventId": [1], "Parameter": [2]})
    det = pd.DataFrame({"DeviceId": ["S1"], "Detector": [9], "phase": [2.0], "function": ["Advance"], "lanes": [1.0]})
    _, st, rows, _ = one_run(raw, det, marks(), str(tmp_path / "*.parquet"))
    assert len(rows) == 0 or (st.EventId == L.RUN_ID).sum() == 1
    if len(rows) == 0:
        assert (st.EventId == L.RUN_ID).sum() == 0


def test_digit_string_ids_keep_their_form_through_a_csv_state(tmp_path):
    det = pd.DataFrame({"DeviceId": ["01000"], "Detector": [9], "phase": [2.0], "function": ["Advance"], "lanes": [1.0]})
    raw = beat("01000", BASE, BASE + Q)
    _, st, _, _ = one_run(raw, det, "", str(tmp_path / "*.parquet"))
    st.to_csv(tmp_path / "state.csv", index=False)
    next_raw = beat("01000", BASE + Q, BASE + 2 * Q)
    _, st2, _, _ = one_run(next_raw, det, str(tmp_path / "state.csv"), str(tmp_path / "*.parquet"))
    ids = set(st2.DeviceId.dropna())
    assert ids == {"01000"}




# ---- the per-run rules, on hand-made states

def marks(through=BASE, last_check=None, first_day=None, ids=("S1",), det=9, stuck=(300.0, 600.0), rapid=(1.0, 2.0), volume=(100000.0, 200000.0), ped=None):
    """A state as a check leaves it: bins written through `through`, and the limits given."""
    last_check = through.normalize() if last_check is None else last_check
    first_day = through - 2 * DAY if first_day is None else first_day
    rows = [(through, None, L.BINS_THROUGH, 0, None), (last_check, None, L.LAST_CHECK, 0, None), (first_day, None, L.FIRST_DAY, 0, None)]
    for d in ids:
        for code, v in ((9501, stuck[0]), (9502, stuck[1]), (9505, rapid[0]), (9506, rapid[1]), (9507, volume[0]), (9508, volume[1])):
            rows.append((None, d, code, det, v))
        if ped:
            rows += [(None, d, L.INPUT_PHASE, ped[0], 2.0), (None, d, L.INPUT_FLAGS, ped[0], float(ped[1]))]
    return pd.DataFrame(rows, columns=["TimeStamp", "DeviceId", "EventId", "Parameter", "Value"]).astype({"TimeStamp": "datetime64[us]", "Value": "float64"})


def beat(device, a, b, walks=False, pulses=True):
    """A live signal: phase 2 cycling each minute, detector 2 pulsing, and an event at the very end of each quarter hour and of the stretch."""
    rows = []
    for t in pd.date_range(a, b, freq="60s", inclusive="left"):
        s = pd.Timedelta(seconds=1)
        rows += [(t, 1, 2), (t + 30 * s, 8, 2), (t + 34 * s, 10, 2)]
        if walks:
            rows.append((t + s, 21, 2))
        if pulses:
            rows += [(t + 40 * s, 82, 2), (t + 41 * s, 81, 2)]
        if (t + 60 * s).minute % 15 == 0 and (t + 60 * s).second == 0:     # each run's last event at its bin's end
            rows.append((t + pd.Timedelta(seconds=59.5), 81, 2))
    rows.append((b - pd.Timedelta(seconds=0.5), 81, 2))
    f = pd.DataFrame(rows, columns=["Timestamp", "EventId", "Parameter"]).assign(DeviceId=device)
    return f[["DeviceId", "Timestamp", "EventId", "Parameter"]].astype({"EventId": "int32", "Parameter": "int32"})


def with_events(frame, device="S1", *rows):
    extra = pd.DataFrame(rows, columns=["Timestamp", "EventId", "Parameter"]).assign(DeviceId=device)
    return pd.concat([frame, extra[frame.columns].astype(frame.dtypes.to_dict())], ignore_index=True)


DET = pd.DataFrame({"DeviceId": ["S1", "S1"], "Detector": [2, 9], "phase": [2.0, 2.0], "function": ["Presence", "Advance"], "lanes": [1.0, 1.0]})


def bins_in_turn(events, state, tmp_path, n, start=BASE):
    """n 15-minute runs, the state carried: the rows of each."""
    out = []
    for i in range(n):
        a, b = start + i * Q, start + (i + 1) * Q
        got, state, _, _ = one_run(events[(events.Timestamp >= a) & (events.Timestamp < b)], DET, state, str(tmp_path / "*.parquet"))
        out.append(got)
    return out


def of(rows, kind, detector, label):
    return rows[(rows.DetectorType == kind) & (rows.Detector == detector) & (rows.Label == label)]


def test_vehicle_stuck_on_each_bin_from_its_limit_has_its_row(tmp_path):
    ev = with_events(beat("S1", BASE, BASE + 4 * Q), "S1", (BASE, 82, 9))
    runs = bins_in_turn(ev, marks(), tmp_path, 4)
    rows = [of(r, "Vehicle", 9, "stuck on") for r in runs]
    assert [len(r) for r in rows] == [1, 1, 1, 1]        # past its 300 s limit in the first bin, still open at each run's end
    assert rows[2].TimeStamp.iloc[0] == BASE + 2 * Q and rows[2].Score.iloc[0] > 0


def test_stuck_on_below_the_limit_and_when_no_check_has_run_has_no_row(tmp_path):
    ev = with_events(beat("S1", BASE, BASE + 3 * Q), "S1", (BASE + pd.Timedelta(seconds=5), 82, 9))
    runs = bins_in_turn(ev, marks(stuck=(3600.0, 7200.0)), tmp_path, 3)
    assert not any(len(of(r, "Vehicle", 9, "stuck on")) for r in runs)
    first = marks().iloc[:3]
    assert all(len(r) == 0 for r in bins_in_turn(ev, first, tmp_path, 3))


def fast_pulses(a, b, gap=0.5):
    rows = []
    for t in pd.date_range(a, b, freq=pd.Timedelta(seconds=gap), inclusive="left"):
        rows += [(t, 82, 9), (t + pd.Timedelta(seconds=0.1), 81, 9)]
    return rows


def test_too_fast_has_no_per_run_rule(tmp_path):
    ev = with_events(beat("S1", BASE, BASE + Q), "S1", *fast_pulses(BASE + pd.Timedelta(minutes=1), BASE + pd.Timedelta(minutes=2)))
    runs = bins_in_turn(ev, marks(), tmp_path, 1)
    assert len(of(runs[0], "Vehicle", 9, "too-fast actuations")) == 0


def test_too_many_is_the_bins_busiest_5_minutes_against_the_limit(tmp_path):
    pulses = [(t, c, 9) for t0 in pd.date_range(BASE, BASE + pd.Timedelta(minutes=5), freq="20s", inclusive="left") for t, c in ((t0, 82), (t0 + pd.Timedelta(seconds=1), 81))]
    ev = with_events(beat("S1", BASE, BASE + Q), "S1", *pulses)
    rows = of(bins_in_turn(ev, marks(volume=(100.0, 200.0)), tmp_path, 1)[0], "Vehicle", 9, "too many for the traffic")
    assert rows.Score.tolist() == [pytest.approx(1 / 3)]
    assert len(of(bins_in_turn(ev, marks(volume=(100000.0, 200000.0)), tmp_path, 1)[0], "Vehicle", 9, "too many for the traffic")) == 0


def test_ped_stuck_on_closed_past_the_limit(tmp_path):
    ev = with_events(beat("S1", BASE, BASE + Q), "S1", (BASE + pd.Timedelta(seconds=10), 90, 3), (BASE + pd.Timedelta(seconds=410), 89, 3))
    rows = of(bins_in_turn(ev, marks(ped=(3, 0)), tmp_path, 1)[0], "Ped", 3, "stuck on")
    assert rows.Score.tolist() == [pytest.approx(400 / 900, abs=1e-4)]


def test_ped_long_hold_needs_2_walks(tmp_path):
    hold = ((BASE + pd.Timedelta(seconds=10), 90, 3), (BASE + pd.Timedelta(seconds=170), 89, 3))
    ev = with_events(beat("S1", BASE, BASE + Q), "S1", *hold)
    assert len(bins_in_turn(ev, marks(ped=(3, 0)), tmp_path, 1)[0].query("DetectorType == 'Ped'")) == 0
    walks = with_events(ev, "S1", (BASE + pd.Timedelta(seconds=61), 21, 2), (BASE + pd.Timedelta(seconds=121), 21, 2))
    assert len(of(bins_in_turn(walks, marks(ped=(3, 0)), tmp_path, 1)[0], "Ped", 3, "long hold")) == 1


def test_ped_open_on_with_walks_at_most_greens_has_a_row_and_the_next_bin_its_own(tmp_path):
    ev = with_events(beat("S1", BASE, BASE + 2 * Q, walks=True), "S1", (BASE + pd.Timedelta(seconds=10), 90, 3))
    runs = bins_in_turn(ev, marks(ped=(3, 0)), tmp_path, 2)
    rows = [of(r, "Ped", 3, "stuck on") for r in runs]
    assert [len(r) for r in rows] == [1, 1] and rows[1].TimeStamp.iloc[0] == BASE + Q
    assert rows[1].Score.iloc[0] == pytest.approx(1.0)


def test_ped_on_recall_or_few_walks_open_has_no_row(tmp_path):
    ev = with_events(beat("S1", BASE, BASE + Q, walks=True), "S1", (BASE + pd.Timedelta(seconds=10), 90, 3))
    assert len(bins_in_turn(ev, marks(ped=(3, 2)), tmp_path, 1)[0].query("DetectorType == 'Ped'")) == 0
    few = with_events(beat("S1", BASE, BASE + Q), "S1", (BASE + pd.Timedelta(seconds=10), 90, 3))
    assert len(bins_in_turn(few, marks(ped=(3, 0)), tmp_path, 1)[0].query("DetectorType == 'Ped'")) == 0


def test_one_row_per_bin_per_ped_input(tmp_path):
    two = ((BASE + pd.Timedelta(seconds=10), 90, 3), (BASE + pd.Timedelta(seconds=320), 89, 3), (BASE + pd.Timedelta(seconds=330), 90, 3), (BASE + pd.Timedelta(seconds=640), 89, 3))
    ev = with_events(beat("S1", BASE, BASE + Q), "S1", *two)
    rows = of(bins_in_turn(ev, marks(ped=(3, 0)), tmp_path, 1)[0], "Ped", 3, "stuck on")
    assert len(rows) == 1 and rows.Score.iloc[0] == pytest.approx(620 / 900, abs=1e-4)


# ---- each signal on its own: late, hours behind, out of order, in subsets, alone

def delivered(ev, det, edges, folder, through, **kw):
    """Runs at edges where signal d's events are in up to through(d, b) at the run ending b (on time: b).  Each run carries only its saved tables.
    Returns (rows of all runs, last state, the runs' LiveResults, the ped history saved)."""
    rows_dir, ped_dir = folder / "rows", folder / "ped"
    rows_dir.mkdir(parents=True)
    ped_dir.mkdir()
    state, outs, results = "", [], []
    devices = sorted(ev.DeviceId.unique())
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = pd.concat([ev[(ev.DeviceId == d) & (ev.Timestamp >= through(d, a)) & (ev.Timestamp < through(d, b))] for d in devices])
        out, state, _, _ = one_run(chunk, det, state, str(rows_dir / "*.parquet"), rows_dir, f"r{i:05d}", ped_folder=ped_dir, result=results, **kw)
        outs.append(out)
    return pd.concat(outs, ignore_index=True), state, results, saved_ped(ped_dir)


def saved_ped(folder):
    files = list(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["TimeStamp", "DeviceId", "Input", "Presses"])
    f = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True)
    return f.groupby(["TimeStamp", "DeviceId", "Input"], dropna=False, as_index=False).Presses.max().sort_values(["TimeStamp", "DeviceId", "Input"]).reset_index(drop=True)


def findings_of(results):
    parts = [r.findings for r in results if r.findings is not None and len(r.findings)]
    return pd.concat(parts, ignore_index=True) if parts else None


def same_findings(a, b):
    key = ["TimeStamp", "DeviceId", "DetectorType", "Detector", "Start", "Label", "Detail"]
    pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True), check_dtype=False)


@pytest.fixture(scope="module")
def big_all(log, tmp_path_factory):
    """one run over the whole log, with its LiveResult and ped history"""
    folder = tmp_path_factory.mktemp("bigall")
    (folder / "rows").mkdir()
    (folder / "ped").mkdir()
    res = []
    out, st, _, _ = one_run(log[0], log[1], "", str(folder / "rows" / "*.parquet"), ped_folder=folder / "ped", result=res)
    return out, st, res, saved_ped(folder / "ped")


LAG = pd.Timedelta(hours=3)
HOUR = pd.Timedelta(hours=1)


def late_and_bursty(d, b):
    """S01 three hours behind the others, S02 in bursts on the hour, S00 on time"""
    return {"S01": b - LAG, "S02": b.floor("h")}.get(d, b)


def test_a_signal_hours_behind_or_in_bursts_gets_the_rows_of_one_big_run(log, big_all, tmp_path):
    """Each signal's bins are written when its own events reach them, and its whole-day check when its own events pass midnight:
    the rows, the findings, the ped history and the state are those of one big run, for the signal 3 hours behind (its midnight
    comes 3 hours after the others') and the one reporting in hourly bursts."""
    ev, det = log
    edges = [START, T0 + DAY - 2 * Q] + list(pd.date_range(T0 + DAY - Q, END + LAG + Q, freq="15min"))
    rows, st, results, ped = delivered(ev, det, edges, tmp_path, late_and_bursty)
    same_rows(rows, big_all[0])
    assert not rows.duplicated(KEY).any()
    same_findings(findings_of(results), findings_of(big_all[2]))
    pd.testing.assert_frame_equal(state_rows(st), state_rows(big_all[1]), check_dtype=False)
    pd.testing.assert_frame_equal(ped, big_all[3], check_dtype=False)
    alone = {r.checked for r in results if r.findings is not None and set(r.findings.DeviceId) == {"S01"}}
    assert alone == {T0 + DAY}        # S01 checked on its own, in a later run than the others


def test_a_signal_more_than_a_horizon_behind_keeps_its_rows(log, tmp_path):
    """horizon 1 day, S01 26 hours behind: its rows are trimmed at its own time, not the others'."""
    ev, det = log
    lag = pd.Timedelta(hours=26)
    edges = list(pd.date_range(START, END + lag, freq="2h")) + [END + lag + 2 * Q]
    rows, st, results, _ = delivered(ev, det, edges, tmp_path, lambda d, b: b - lag if d == "S01" else b, horizon_days=1)
    (tmp_path / "b" / "rows").mkdir(parents=True)
    res = []
    out, bst, _, _ = one_run(ev, det, "", str(tmp_path / "b" / "rows" / "*.parquet"), result=res, horizon_days=1)
    same_rows(rows, out)
    same_findings(findings_of(results), findings_of(res))
    pd.testing.assert_frame_equal(state_rows(st), state_rows(bst), check_dtype=False)


def test_a_signal_alone_gets_what_it_gets_among_others(log, big_all, tmp_path):
    ev, det = log
    (tmp_path / "rows").mkdir()
    res = []
    out, st, _, _ = one_run(ev[ev.DeviceId == "S01"], det[det.DeviceId == "S01"], "", str(tmp_path / "rows" / "*.parquet"), result=res)
    same_rows(out, big_all[0][big_all[0].DeviceId == "S01"])
    f = findings_of(big_all[2])
    same_findings(findings_of(res), f[f.DeviceId == "S01"])


def test_subsets_run_apart_with_their_own_cuts_equal_one_big_run(log, big_all, tmp_path):
    """S00 and S01 in 15-minute runs over midnight, S02 in hourly runs; each run reads and replaces only its signals' state rows, history shared."""
    ev, det = log
    rows_dir, ped_dir = tmp_path / "rows", tmp_path / "ped"
    rows_dir.mkdir()
    ped_dir.mkdir()
    groups = {("S00", "S01"): [START, T0 + DAY - 2 * Q] + list(pd.date_range(T0 + DAY - Q, END, freq="15min")),
              ("S02",): [START] + list(pd.date_range(T0 + 20 * HOUR, END + 3 * Q, freq="h"))}
    runs = sorted((b, g, a) for g, e in groups.items() for a, b in zip(e[:-1], e[1:]))
    state, outs, results = pd.DataFrame(), [], []
    for i, (b, g, a) in enumerate(runs):
        mine = state[state.DeviceId.isin(g)] if len(state) else state
        chunk = ev[ev.DeviceId.isin(g) & (ev.Timestamp >= a) & (ev.Timestamp < b)]
        out, st, _, _ = one_run(chunk, det[det.DeviceId.isin(g)], mine if len(mine) else "", str(rows_dir / "*.parquet"), rows_dir, f"r{i:05d}",
                                ped_folder=ped_dir, result=results)
        assert st.DeviceId.notna().all() and set(st.DeviceId) <= set(g)
        state = pd.concat([state[~state.DeviceId.isin(g)] if len(state) else state, st], ignore_index=True)
        outs.append(out)
    same_rows(pd.concat(outs, ignore_index=True), big_all[0])
    same_findings(findings_of(results), findings_of(big_all[2]))
    pd.testing.assert_frame_equal(state_rows(state), state_rows(big_all[1]), check_dtype=False)
    pd.testing.assert_frame_equal(saved_ped(ped_dir), big_all[3], check_dtype=False)


def test_a_clock_running_ahead_is_left_out_and_moves_nothing(log, big_all, tmp_path):
    """now = each run's end: S02's events from 20:00 on, stamped two days ahead, are left out and counted; its bins stay where its good events are,
    and the others get what they get without it."""
    ev, det = log
    jump = T0 + 20 * HOUR
    wrong = ev[(ev.DeviceId == "S02") & (ev.Timestamp >= jump)].assign(Timestamp=lambda f: f.Timestamp + 2 * DAY)
    edges = [START, jump] + list(pd.date_range(T0 + DAY - Q, END, freq="15min"))
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    state, outs, results = "", [], []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        good = ev[(ev.DeviceId != "S02") | (ev.Timestamp < jump)]
        chunk = pd.concat([good[(good.Timestamp >= a) & (good.Timestamp < b)], wrong[(wrong.Timestamp >= a + 2 * DAY) & (wrong.Timestamp < b + 2 * DAY)]])
        out, state, _, _ = one_run(chunk, det, state, str(rows_dir / "*.parquet"), rows_dir, f"r{i:05d}", result=results, now=b)
        outs.append(out)
    rows = pd.concat(outs, ignore_index=True)
    same_rows(rows[rows.DeviceId != "S02"], big_all[0][big_all[0].DeviceId != "S02"])
    assert sum(r.held_back.get("S02", 0) for r in results) == len(wrong) > 0
    assert rows[rows.DeviceId == "S02"].TimeStamp.lt(jump).all()
    assert state[(state.DeviceId == "S02") & (state.EventId == L.BINS_THROUGH)].TimeStamp.iloc[0] <= END


def test_midnight_comes_from_a_signals_own_events_or_the_caller(log, tmp_path):
    """S01 stops logging at 22:00.  Its whole-day check waits for its own next event; with complete_through (its events are all in up to the run's
    end) it comes at the midnight run like the others', with findings from its own data only."""
    ev, det = log
    quiet = ev[(ev.DeviceId != "S01") | (ev.Timestamp < T0 + 22 * HOUR)]
    edges = [START, T0 + DAY - 2 * Q] + list(pd.date_range(T0 + DAY - Q, T0 + DAY + 2 * Q, freq="15min"))
    for given in (False, True):
        folder = tmp_path / str(given)
        (folder / "rows").mkdir(parents=True)
        state, results = "", []
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            chunk = quiet[(quiet.Timestamp >= a) & (quiet.Timestamp < b)]
            _, state, _, _ = one_run(chunk, det, state, str(folder / "rows" / "*.parquet"), folder / "rows", f"r{i:05d}", result=results,
                                     **({"complete_through": b} if given else {}))
        f = findings_of(results)
        assert set(f[f.DeviceId != "S01"].TimeStamp) == {T0 + DAY}
        checked = state[(state.DeviceId == "S01") & (state.EventId == L.LAST_CHECK)].TimeStamp.iloc[0]
        assert (checked == T0 + DAY) == given and (f.DeviceId == "S01").any() == given


def legacy(st):
    """a state as 2.1.0 wrote it: one set of markers, the run list and the version with no DeviceId"""
    marks = st[st.EventId.isin([L.BINS_THROUGH, L.LAST_CHECK, L.FIRST_DAY])]
    assert (marks.groupby("EventId").TimeStamp.nunique() == 1).all()
    shared = marks.drop_duplicates("EventId").assign(DeviceId=None)
    runs = st[st.EventId == L.RUN_ID].drop_duplicates("Value").assign(DeviceId=None, Parameter=0)
    version = st[st.EventId == L.VERSION].head(1).assign(DeviceId=None)
    rest = st[~st.EventId.isin([L.BINS_THROUGH, L.LAST_CHECK, L.FIRST_DAY, L.RUN_ID, L.VERSION])]
    return pd.concat([rest, shared, runs, version], ignore_index=True)


def test_a_state_from_before_2_2_is_converted_and_carries_on(log, big_all, tmp_path):
    """Cut at 18:00, the state rewritten in the 2.1.0 form: the runs after it write what an unbroken chain writes."""
    ev, det = log
    edges = [START, T0 + 18 * HOUR] + list(pd.date_range(T0 + DAY - 2 * Q, END, freq="15min"))
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    state, outs = "", []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        # the first run closes 18:00 for every signal (complete_through), so all stand at one marker, as 2.1.0 kept them
        out, state, _, _ = one_run(ev[(ev.Timestamp >= a) & (ev.Timestamp < b)], det, legacy(state) if i == 1 else state, str(rows_dir / "*.parquet"), rows_dir, f"r{i:05d}",
                                   **({"complete_through": b} if i == 0 else {}))
        outs.append(out)
    same_rows(pd.concat(outs, ignore_index=True), big_all[0])
    assert state.DeviceId.notna().all()


def ped_log(days=17, silent_from=15):
    """One signal: phases 2 and 4 green every minute; button 2 pressed every 30 min until day silent_from, button 4 every 30 min throughout,
    each press walked at its phase's next green."""
    t0 = T0 - 15 * DAY
    s = pd.Timedelta(seconds=1)
    minutes = pd.date_range(t0, t0 + days * DAY, freq="60s", inclusive="left")
    parts = [pd.DataFrame({"Timestamp": minutes + k * s, "EventId": e, "Parameter": p})
             for k, e, p in ((0, 1, 2), (20, 8, 2), (24, 10, 2), (30, 1, 4), (50, 8, 4), (54, 10, 4))]
    half = minutes[minutes.minute % 30 == 0]
    for ch, off, g in ((2, 5, 60), (4, 15, 75)):
        t = half[half < t0 + silent_from * DAY] if ch == 2 else half
        parts += [pd.DataFrame({"Timestamp": t + k * s, "EventId": e, "Parameter": ch})
                  for k, e in ((off, 90), (off + 0.1, 45), (off + 0.3, 89), (g + 1, 21), (g + 8, 22), (g + 15, 23))]
    f = pd.concat(parts, ignore_index=True).assign(DeviceId="P1").sort_values("Timestamp", kind="stable")
    return f[["DeviceId", "Timestamp", "EventId", "Parameter"]].astype({"EventId": "int32", "Parameter": "int32"}).reset_index(drop=True), t0


def ped_counts(ev, lo, hi):
    """the ped history rows a caller would have saved from earlier runs: 15-minute presses and logged bins"""
    e = ev[(ev.Timestamp >= lo) & (ev.Timestamp < hi)]
    b = e.Timestamp.dt.floor("15min")
    logged = pd.DataFrame({"TimeStamp": b.unique(), "DeviceId": "P1", "Input": None, "Presses": None})
    p = e[e.EventId == 90].assign(TimeStamp=b).groupby(["TimeStamp", "Parameter"], as_index=False).size()
    presses = pd.DataFrame({"TimeStamp": p.TimeStamp, "DeviceId": "P1", "Input": p.Parameter, "Presses": p["size"]})
    return pd.concat([logged, presses], ignore_index=True).astype({"Input": "Int32", "Presses": "Int32", "TimeStamp": "datetime64[us]"})


def test_a_ped_sudden_stop_needs_only_the_ped_history_with_a_3_day_horizon(tmp_path):
    """A button silent for two days after two weeks of presses: with 3 days of horizon the sudden stop against its own weeks is found only with the
    ped history (presses per 15 minutes and logged bins, not the whole log), and a run's own ped rows are those a caller saves."""
    ev, t0 = ped_log()
    run_from = t0 + 15 * DAY
    det = pd.DataFrame({"DeviceId": ["P1"], "Detector": [99], "phase": [2.0], "function": ["Presence"], "lanes": [1.0]})
    history = ped_counts(ev, t0, run_from)
    found = {}
    for days in (0, 28):
        folder = tmp_path / str(days)
        (folder / "rows").mkdir(parents=True)
        (folder / "ped").mkdir()
        if days:
            history.to_parquet(folder / "ped" / "before.parquet")
        res = []
        one_run(ev[ev.Timestamp >= run_from], det, "", str(folder / "rows" / "*.parquet"), ped_folder=folder / "ped", result=res, horizon_days=3,
                ped_history_days=days, name="run", complete_through=t0 + 17 * DAY)
        f = findings_of(res)
        found[days] = f[(f.DetectorType == "Ped") & (f.Detector == 2) & (f.Label == "sudden stop")]
        if days:
            mine = pd.read_parquet(folder / "ped" / "run.parquet")
            want = ped_counts(ev, run_from, t0 + 17 * DAY).sort_values(["TimeStamp", "Input"]).reset_index(drop=True)
            got = mine.sort_values(["TimeStamp", "Input"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(got[["TimeStamp", "Input", "Presses"]], want[["TimeStamp", "Input", "Presses"]], check_dtype=False)
    assert len(found[0]) == 0
    assert len(found[28]) >= 1 and (found[28].TimeStamp == t0 + 17 * DAY).any()


def tz_rows(folder):
    ev, det = synth.make(1, 2, seed=3)
    ev = ev[ev.Timestamp < T0 + DAY + 6 * Q]
    edges = [T0, T0 + DAY + 6 * Q]     # one run across midnight
    from pathlib import Path
    outs, _ = chain(ev, det, edges, Path(folder))
    return pd.concat(outs, ignore_index=True)




if __name__ == "__main__":
    tz_rows(sys.argv[1]).to_parquet(sys.argv[2])
