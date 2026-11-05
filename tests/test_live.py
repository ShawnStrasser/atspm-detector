"""live.py, a run every 15 minutes with state carried as two tables: one big run equals 15-minute runs, each run's own bin, late finds inside the previous day, the midnight rule, the per-run rules."""
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


def one_run(raw, det, state, history, rows_folder=None, name=None, **kw):
    """One run on its own connection, as a process of its own would be: (detector_health, health_state, health_rows, findings)."""
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.register("raw_in", raw)
    con.execute("CREATE TABLE raw_data AS SELECT DeviceId, Timestamp AS TimeStamp, EventId, Parameter FROM raw_in")
    con.unregister("raw_in")
    con.register("det_in", det)
    con.execute("CREATE TEMP TABLE dets AS SELECT * FROM det_in")
    con.unregister("det_in")
    if len(raw):
        found, _ = L.live_run(con, "raw_data", "dets", state=state, history=history, horizon_days=kw.pop("horizon_days", 2), **kw)
    else:
        found = L.empty_run(con, state)
    out, st, rows = (con.execute(f"SELECT * FROM {t}").df() for t in ("hl_final", "hl_state", "hl_rows"))
    if rows_folder is not None and len(rows):
        con.execute(f"COPY hl_rows TO '{rows_folder}/{name}.parquet' (FORMAT PARQUET)")
    leftover = con.execute("SELECT table_name FROM duckdb_tables() WHERE starts_with(table_name, 'hl_') AND table_name NOT IN ('hl_final', 'hl_state', 'hl_rows')").fetchall()
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


@pytest.fixture(scope="module")
def log():
    ev, det = synth.make(3, 3, seed=7)
    return ev[ev.Timestamp < T0 + 2 * DAY + 3 * Q], det


@pytest.fixture(scope="module")
def big(log, tmp_path_factory):
    folder = tmp_path_factory.mktemp("big")
    (folder / "rows").mkdir()
    return one_run(log[0], log[1], "", str(folder / "rows" / "*.parquet"))


@pytest.fixture(scope="module")
def cut(log, tmp_path_factory):
    edges = list(pd.date_range(T0, T0 + 2 * DAY + 3 * Q, freq="15min"))
    return chain(log[0], log[1], edges, tmp_path_factory.mktemp("cut")), edges


def state_rows(st):
    keep = st[~st.EventId.isin([L.TRIMMED, L.RUN_ID])]
    cols = ["EventId", "DeviceId", "Parameter", "TimeStamp", "Value"]
    return keep[cols].sort_values(cols).reset_index(drop=True)


def test_one_big_run_equals_15_minute_runs(big, cut):
    (outs, st), _ = cut
    assert len(big[0]) > 100
    same_rows(pd.concat(outs, ignore_index=True), big[0])
    pd.testing.assert_frame_equal(state_rows(st), state_rows(big[1]), check_dtype=False)


def test_state_and_rows_through_csv_change_nothing(log, big, tmp_path):
    edges = list(pd.date_range(T0, T0 + 2 * DAY + 3 * Q, freq="60min")) + [T0 + 2 * DAY + 3 * Q]
    outs, _ = chain(log[0], log[1], edges, tmp_path, through_csv=True)
    got, want = ordered(pd.concat(outs, ignore_index=True)), ordered(big[0])
    pd.testing.assert_frame_equal(got, want, check_dtype=False, rtol=1e-6)


def test_an_ordinary_run_writes_only_its_own_bin(cut):
    (outs, _), edges = cut
    ordinary = [(a, o) for a, o in zip(edges[:-1], outs) if a.normalize() != a]
    assert sum(len(o) for _, o in ordinary) > 0
    for a, o in ordinary:
        assert o.TimeStamp.isin([a - Q, a]).all() and (o.TimeStamp.max() >= a - Q if len(o) else True)


def test_the_first_run_past_midnight_adds_rows_only_inside_the_previous_day(cut):
    (outs, _), edges = cut
    late = 0
    for a, b, o in zip(edges[:-1], edges[1:], outs):
        old = o[o.TimeStamp < b - 2 * Q]
        if len(old):
            M = b.floor("D")
            assert (old.TimeStamp >= M - DAY).all() and (old.TimeStamp < M).all() and a - M < 2 * Q
            late += len(old)
    assert late > 0


def test_no_key_is_written_twice(cut):
    (outs, _), _ = cut
    assert not pd.concat(outs, ignore_index=True).duplicated(KEY).any()


def test_a_run_straddling_midnight_and_a_catch_up_equal_the_plain_chain(log, cut, tmp_path):
    (outs, _), _ = cut
    M = T0 + DAY
    edges = list(pd.date_range(T0, M - Q, freq="6h")) + [M - pd.Timedelta(minutes=10), M + pd.Timedelta(minutes=10), T0 + 2 * DAY + 3 * Q]
    got, _ = chain(log[0], log[1], edges, tmp_path)
    same_rows(pd.concat(got, ignore_index=True), pd.concat(outs, ignore_index=True))


def test_rows_of_a_failed_attempt_are_ignored(log, cut, tmp_path):
    (outs, _), edges = cut
    stop = T0 + DAY - 2 * Q
    ev, det = log
    rows = tmp_path / "rows"
    rows.mkdir()
    state = ""
    pieces = [(T0, stop), (stop, T0 + 2 * DAY + 3 * Q)]
    got = []
    for i, (a, b) in enumerate(pieces):
        chunk = ev[(ev.Timestamp >= a) & (ev.Timestamp < b)]
        if i == 1:      # an attempt whose rows were saved and whose state never was
            _, _, failed, _ = one_run(chunk[chunk.Timestamp < a + Q], det, state, str(rows / "*.parquet"))
            failed.assign(Run=failed.Run + 12345).to_parquet(rows / "failed.parquet")
        out, state, _, _ = one_run(chunk, det, state, str(rows / "*.parquet"), rows, f"p{i}")
        got.append(out)
    same_rows(pd.concat(got, ignore_index=True), pd.concat(outs, ignore_index=True))


def test_a_listed_run_missing_from_history_raises_before_the_check(log, cut, tmp_path):
    ev, det = log
    M = T0 + DAY
    rows = tmp_path / "rows"
    rows.mkdir()
    _, state, _, _ = one_run(ev[ev.Timestamp < M - Q], det, "", str(rows / "*.parquet"), rows, "first")
    (rows / "first.parquet").unlink()
    with pytest.raises(ValueError, match="lacks the rows of 1 of the 1 runs"):
        one_run(ev[(ev.Timestamp >= M - Q) & (ev.Timestamp < M + Q)], det, state, str(rows / "*.parquet"))


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


def test_an_empty_run_keeps_the_state_and_writes_all_columns(log, big, tmp_path):
    ev, det = log
    a_state, a_rows = big[1], big[2]
    out, st, rows, _ = one_run(ev.head(0), det, a_state, str(tmp_path / "*.parquet"))
    assert len(out) == 0 and len(rows) == 0
    assert list(rows.columns) == list(a_rows.columns) == list(L.ROW_COLUMNS) and len(L.ROW_COLUMNS) == 50
    pd.testing.assert_frame_equal(state_rows(st), state_rows(a_state), check_dtype=False)


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


def test_a_3_hour_outage_for_most_signals_loses_no_events(tmp_path):
    ev, det = synth.make(10, 1, seed=11, faults=False)
    gone = (ev.DeviceId != "S00") & (ev.Timestamp >= T0 + pd.Timedelta(hours=8)) & (ev.Timestamp < T0 + pd.Timedelta(hours=11))
    ev = ev[~gone & (ev.Timestamp < T0 + pd.Timedelta(hours=12))]
    (tmp_path / "big").mkdir()
    (tmp_path / "cut").mkdir()
    _, _, big_rows, _ = one_run(ev, det, "", str(tmp_path / "big" / "*.parquet"))
    edges = list(pd.date_range(T0 + pd.Timedelta(hours=7), T0 + pd.Timedelta(hours=12), freq="15min"))
    first = ev[ev.Timestamp < edges[0]]
    rows_dir = tmp_path / "cut" / "rows"
    rows_dir.mkdir()
    _, state, _, _ = one_run(first, det, "", str(rows_dir / "*.parquet"), rows_dir, "first")
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = ev[(ev.Timestamp >= a) & (ev.Timestamp < b)]
        if len(chunk):
            _, state, _, _ = one_run(chunk, det, state, str(rows_dir / "*.parquet"), rows_dir, f"r{i:03d}")
    con = duckdb.connect()
    sum_events = "SELECT sum(Events) FROM read_parquet('{}') WHERE Tbl = 'health_signal_5min'"
    got = con.execute(sum_events.format(str(rows_dir / "*.parquet"))).fetchall()[0][0]
    assert got == big_rows[big_rows.Tbl == "health_signal_5min"].Events.sum()


# ---- the midnight rule

M0 = pd.Timestamp("2026-10-07")


def past(n_past, n_before=None, last=None, n_old=0):
    n_before = 100 - n_past if n_before is None else n_before
    stamps = [M0 + pd.Timedelta(minutes=5)] * n_past + [M0 - pd.Timedelta(minutes=30)] * n_before + [M0 - 2 * DAY] * n_old
    return L._midnights_past(pd.Series(stamps), [M0], last or M0 + pd.Timedelta(minutes=5), 60)


def test_midnight_needs_9_in_10_of_the_signals_heard_from_in_the_day_before():
    assert past(89) == []
    assert past(90) == [M0]
    assert past(9, 1, n_old=50) == [M0]            # a signal silent the whole day before does not count
    assert past(10, 90, last=M0 + pd.Timedelta(minutes=59)) == []


def test_midnight_also_fires_once_the_run_is_clock_ahead_minutes_past():
    assert past(10, 90, last=M0 + pd.Timedelta(minutes=60)) == [M0]


def test_a_run_stops_at_midnight_until_it_fires_then_loses_nothing(tmp_path):
    """8 of 10 signals past midnight: the run stops there and holds its later events; the next run takes them."""
    M = BASE.normalize() + DAY
    ids = [f"S{i}" for i in range(10)]
    det = pd.DataFrame({"DeviceId": ids, "Detector": 2, "phase": 2.0, "function": "Presence", "lanes": 1.0})
    state = marks(through=M - Q, last_check=M - DAY, first_day=M - Q, ids=ids)
    ahead = pd.concat([beat(d, M - Q, M + Q) for d in ids[:8]] + [beat(d, M - Q, M - pd.Timedelta(minutes=2)) for d in ids[8:]])
    _, st, _, _ = one_run(ahead, det, state, str(tmp_path / "*.parquet"))
    code = lambda s, c: s[s.EventId == c].TimeStamp.iloc[0]
    assert code(st, L.BINS_THROUGH) == M and code(st, L.LAST_CHECK) == M - DAY
    held = st[st.EventId == L.PEND]
    assert len(held) > 0 and held.TimeStamp.min() >= M
    rest = pd.concat([beat(d, M - pd.Timedelta(minutes=2), M + Q) for d in ids[8:]])
    _, st2, _, _ = one_run(rest, det, st, str(tmp_path / "*.parquet"))
    assert code(st2, L.LAST_CHECK) == M and code(st2, L.BINS_THROUGH) == M + Q


# ---- the per-run rules, on hand-made states

def marks(through=BASE, last_check=None, first_day=None, ids=("S1",), det=9, stuck=(300.0, 600.0), rapid=(1.0, 2.0), volume=(100000.0, 200000.0), fast=0.0, ped=None):
    """A state as a check leaves it: bins written through `through`, and the limits given."""
    last_check = through.normalize() if last_check is None else last_check
    first_day = through - 2 * DAY if first_day is None else first_day
    rows = [(through, None, L.BINS_THROUGH, 0, None), (last_check, None, L.LAST_CHECK, 0, None), (first_day, None, L.FIRST_DAY, 0, None)]
    for d in ids:
        for code, v in ((9501, stuck[0]), (9502, stuck[1]), (9505, rapid[0]), (9506, rapid[1]), (9507, volume[0]), (9508, volume[1]), (9511, fast)):
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


def test_vehicle_stuck_on_waits_for_30_minutes_then_each_bin_has_its_row(tmp_path):
    ev = with_events(beat("S1", BASE, BASE + 4 * Q), "S1", (BASE, 82, 9))
    runs = bins_in_turn(ev, marks(), tmp_path, 4)
    rows = [of(r, "Vehicle", 9, "stuck on") for r in runs]
    assert [len(r) for r in rows] == [0, 1, 1, 1]        # held 30 minutes at the end of the second bin, still open at each run's end
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


@pytest.mark.parametrize("known, n", [(1.0, 1), (0.0, 0)])
def test_too_fast_only_for_a_detector_the_check_rated_too_fast(tmp_path, known, n):
    ev = with_events(beat("S1", BASE, BASE + Q), "S1", *fast_pulses(BASE + pd.Timedelta(minutes=1), BASE + pd.Timedelta(minutes=2)))
    runs = bins_in_turn(ev, marks(fast=known), tmp_path, 1)
    assert len(of(runs[0], "Vehicle", 9, "too-fast actuations")) == n


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


def tz_rows(folder):
    ev, det = synth.make(1, 2, seed=3)
    edges = list(pd.date_range(T0, T0 + 2 * DAY, freq="3h"))
    from pathlib import Path
    outs, _ = chain(ev, det, edges, Path(folder))
    return pd.concat(outs, ignore_index=True)


def test_rows_do_not_change_with_the_machines_time_zone(tmp_path):
    here = tz_rows(tmp_path / "here")
    assert len(here)
    path = tmp_path / "tokyo.parquet"
    env = {**os.environ, "TZ": "Asia/Tokyo"}
    subprocess.run([sys.executable, __file__, str(tmp_path / "there"), str(path)], check=True, env=env)
    same_rows(pd.read_parquet(path), here)


if __name__ == "__main__":
    tz_rows(sys.argv[1]).to_parquet(sys.argv[2])
