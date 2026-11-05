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


def tz_rows(folder):
    ev, det = synth.make(1, 2, seed=3)
    ev = ev[ev.Timestamp < T0 + DAY + 6 * Q]
    edges = [T0, T0 + DAY + 6 * Q]     # one run across midnight
    from pathlib import Path
    outs, _ = chain(ev, det, edges, Path(folder))
    return pd.concat(outs, ignore_index=True)




if __name__ == "__main__":
    tz_rows(sys.argv[1]).to_parquet(sys.argv[2])
