"""Stage 3 of the incremental-health design: the judge reads the extractor's rows through the adapters, and the new
health() equals the frozen oracle (tests/legacy) exactly.

T3.1  detectors and periods tables equal the oracle's, every column and value (floats bit for bit -- measured; the design
      allows 1e-9 on scores, not needed so far), on the edge patterns of synth.edges for the defaults, all off and each
      option switched alone, odd windows and a windows table (the golden cases are in test_golden)
T3.2  health_sql() is one WITH query and gives health()'s rows
T3.3  workers=3 (small groups) equals workers=1, in both bin origins
plus: bin_origin="clock" equals a window-origin run over the same windows started on the quarter-hour; occ_strict=False
(the 1.2.0 imputation) is refused; the listed last-bit cases (design 3.5): a value exactly on a limit, which today's double
sums and medians put on either side by rounding noise and the integers put exactly on it, reproduced here."""
import duckdb
import pandas as pd
import pytest

import make_golden as G
import synth
from atspm_detector import health_duckdb as H
from legacy import health_duckdb_v2 as H0

_CON = []


def _con():
    if not _CON:
        con = G.connect(*synth.edges(*synth.make(3, 2, seed=5)))
        con.execute("SET memory_limit = '2GB'; SET max_temp_directory_size = '4GB'")
        _CON.append(con)
    return _CON[0]


def _equal(a, b):
    """two outputs equal: columns, dtypes, rows in order, every value (NaN = NaN)."""
    assert list(a.columns) == list(b.columns)
    assert G.dtypes(a) == G.dtypes(b)
    assert G.values(a) == G.values(b)


def _flip(k, v):
    return {"lane_pairs": [None, "any"], "clean_yardsticks": [False, True], "occ_strict": []}.get(k, [not v])


OPTS = [("defaults", {}), ("all off", dict(H.OPTIONS_OFF))]
OPTS += [(f"{k}={x}", {k: x}) for k, v in H.OPTIONS.items() for x in _flip(k, v)]
# today's code fails there too: silent_merge off -- phase_drop's text needs silent_merge's times (found in stage 1);
# all_silences off -- with closure_1h on, the closure's fault-day list (lcfp) loses its column names fa / fz (stage 3)
KNOWN_FAIL = {"silent_merge=False", "all_silences=False"}


@pytest.mark.parametrize("name,opt", OPTS, ids=[n for n, _ in OPTS])
def test_options_equal_oracle(name, opt):
    con = _con()
    if name in KNOWN_FAIL:
        for M in (H0, H):
            with pytest.raises(Exception):
                M.health(con, "events", "detectors", periods=True, options=opt)
        return
    o0 = dict(opt, occ_strict=True) if name == "all off" else opt     # the oracle's OPTIONS_OFF still imputes
    for x, y in zip(H0.health(con, "events", "detectors", periods=True, options=o0),
                    H.health(con, "events", "detectors", periods=True, options=opt)):
        _equal(x, y)


@pytest.mark.parametrize("kw", [dict(start="2026-10-05 01:07:13", end="2026-10-06 19:41:05.5"),
                                dict(start="2026-10-05 22:00"), dict(end="2026-10-05 07:31")],
                         ids=["odd", "start", "end"])
def test_windows_equal_oracle(kw):
    con = _con()
    for x, y in zip(H0.health(con, "events", "detectors", periods=True, **kw),
                    H.health(con, "events", "detectors", periods=True, **kw)):
        _equal(x, y)


def test_windows_table_equal_oracle():
    con = _con()
    con.execute("""CREATE OR REPLACE TABLE w30 AS SELECT DeviceId,
                   epoch_ns(min(Timestamp)) + 3333300000000 + (hash(DeviceId) % 7) * 5000000000000 AS s_ns,
                   epoch_ns(min(Timestamp)) + 3333300000000 + (hash(DeviceId) % 7) * 5000000000000 + 1800000000000
                   AS e_ns FROM events GROUP BY DeviceId""")
    kw = dict(windows_table="w30", max_events=None, distinct=False)
    _equal(H0.health(con, "events", "detectors", **kw), H.health(con, "events", "detectors", **kw))


def test_health_sql_is_one_query_equal_to_health():
    con = G.connect(*G.inputs("sample"))
    d, p = H.health_sql("events", "detectors", periods=True)
    assert d.startswith("WITH ") and p.startswith("WITH ")
    a, b = H.health(con, "events", "detectors", periods=True)
    key = ["DeviceId", "Detector"]
    x = con.execute(d).df().sort_values(key, kind="stable").reset_index(drop=True)
    _equal(a.sort_values(key, kind="stable").reset_index(drop=True), x[a.columns])
    pk = ["DeviceId", "Detector", "start", "end", "condition"]
    _equal(b.sort_values(pk, kind="stable").reset_index(drop=True),
           con.execute(p).df()[b.columns].sort_values(pk, kind="stable").reset_index(drop=True))


def test_workers_equal_one_process():
    con = _con()
    for x, y in zip(H.health(con, "events", "detectors", periods=True, max_events=300_000),
                    H.health(con, "events", "detectors", periods=True, max_events=300_000, workers=3)):
        _equal(x, y)


@pytest.mark.parametrize("kw", [dict(start="2026-10-05 01:07:13", end="2026-10-06 19:41:05"), dict(windows_table="wq2")],
                         ids=["start and end", "windows table"])
def test_workers_equal_one_process_clock_origin(kw):
    """the workers' copy of the events starts on the quarter-hour too: events just before an odd window start count."""
    con = _con()
    con.execute(f"""CREATE OR REPLACE TABLE wq2 AS SELECT DeviceId, {pd.Timestamp('2026-10-05 01:07:13').value} AS s_ns,
                    {pd.Timestamp('2026-10-06 19:41:05').value} AS e_ns FROM events GROUP BY DeviceId""")
    k = dict(kw, periods=True, max_events=300_000, bin_origin="clock")
    for x, y in zip(H.health(con, "events", "detectors", **k), H.health(con, "events", "detectors", workers=3, **k)):
        _equal(x, y)
    assert not con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name LIKE 'hq_%'").fetchone()[0]


def test_clock_origin():
    """bin_origin="clock": each window starts on the quarter-hour at or before its start -- the same answers as a window-
    origin run over windows started there; a window already on the quarter-hour gives the window-origin answers."""
    con = _con()
    s, e = "2026-10-05 01:07:13", "2026-10-06 19:41:05"
    a = H.health(con, "events", "detectors", start=s, end=e, periods=True, bin_origin="clock")
    con.execute(f"""CREATE OR REPLACE TABLE wq AS SELECT DeviceId, {pd.Timestamp('2026-10-05 01:00').value} AS s_ns,
                    {pd.Timestamp(e).value} AS e_ns FROM events GROUP BY DeviceId""")
    b = H.health(con, "events", "detectors", windows_table="wq", max_events=None, periods=True)
    for x, y in zip(a, b):
        _equal(x, y)
    s = "2026-10-05 01:15"
    for x, y in zip(H.health(con, "events", "detectors", start=s, end=e, periods=True, bin_origin="clock"),
                    H.health(con, "events", "detectors", start=s, end=e, periods=True)):
        _equal(x, y)
    with pytest.raises(ValueError, match="bin_origin"):
        H.health(con, "events", "detectors", bin_origin="hour")


def test_occ_strict_false_refused():
    with pytest.raises(ValueError, match="occ_strict"):
        H.health(duckdb.connect(), options={"occ_strict": False})
    assert H.OPTIONS_OFF["occ_strict"] is True


def _replace(ev, det_id, a, b, rows, colour=None):
    """the synthetic log with detector det_id's events (and the colours of phase `colour`) in [a, b) replaced by rows."""
    m = (ev.DeviceId == "S00") & (ev.Timestamp >= a) & (ev.Timestamp < b) & (
        (ev.EventId.isin([81, 82]) & (ev.Parameter == det_id))
        | (ev.EventId.isin([1, 7, 8, 9, 10, 11]) & (ev.Parameter == colour)))
    add = pd.DataFrame(rows, columns=["DeviceId", "Timestamp", "EventId", "Parameter"])
    out = pd.concat([ev[~m], add], ignore_index=True)
    out["EventId"], out["Parameter"] = out.EventId.astype("int32"), out.Parameter.astype("int32")
    return out.sort_values(["DeviceId", "Timestamp"], kind="stable").reset_index(drop=True)


def _whole(con):
    """one thread (today's double sums in one order) and each signal's whole log as its window."""
    con.execute("SET threads = 1; SET memory_limit = '2GB'; SET max_temp_directory_size = '4GB'")
    con.execute("""CREATE OR REPLACE TABLE wtab AS SELECT DeviceId, epoch_ns(min(Timestamp)) AS s_ns,
                   epoch_ns(max(Timestamp)) + 1000000000 AS e_ns FROM events GROUP BY DeviceId""")


def test_listed_tie_green_and_yellow_exactly_60_s():
    """a 5-min bin with exactly 60.0 s of green + yellow: "too many for the traffic" judges a bin of 60 s or more.  Today's
    sum of double seconds falls just under 60 here (on other pieces just over), so that bin is left out and the detector
    is fine on this check; the integer sum is exactly 60, the bin counts, and its 120 ONs make the detector bad."""
    ev, det = synth.make(2, 1, seed=3)
    t0 = ev.Timestamp.min()
    S = lambda tenths: t0 + pd.Timedelta(hours=3) + pd.Timedelta(milliseconds=100 * tenths)
    rows, k = [("S00", S(-1200), 9, 2), ("S00", S(4200), 1, 2)], 0
    for s, g in ((61, 172), (312, 197), (602, 126)):      # 3 greens + 3 x 3.5 s of yellow = 60.0 s, in tenths of a second
        rows += [("S00", S(s), 1, 2), ("S00", S(s + g), 8, 2), ("S00", S(s + g + 35), 9, 2)]
        for i in range(g // 3 - 1):
            if k < 120:
                rows += [("S00", S(s + 1 + i * 3), 82, 2), ("S00", S(s + 2 + i * 3), 81, 2)]
                k += 1
    assert k == 120
    con = G.connect(_replace(ev, 2, S(-1200), S(4200), rows, 2), det)
    _whole(con)
    a, b = G.steps(H0, con, ["sgyb"])["sgyb"][0], G.steps(H, con, ["sgyb"])["sgyb"][0]
    pick = lambda x: x[(x.dev == 1) & (x.ph == 2) & (x.b == 36)].sgy.iloc[0]
    assert pick(a) < 60 and 60 - pick(a) < 1e-9 and pick(b) == 60
    A, B = H0.health(con, "events", "detectors", periods=True), H.health(con, "events", "detectors", periods=True)
    mine = lambda x: x[(x.DeviceId == "S00") & (x.Detector == 2)]
    assert "Too many for the traffic" not in mine(A[0]).health_reason.iloc[0]
    assert "Too many for the traffic" in mine(B[0]).health_reason.iloc[0]
    assert "too many for the traffic" not in set(mine(A[1]).condition) and "too many for the traffic" in set(mine(B[1]).condition)
    assert mine(A[0]).health_status.iloc[0] == "ok" and mine(B[0]).health_status.iloc[0] == "bad"
    # det 2 is then no yardstick for its signal's other detectors (their expected counts move, their statuses do not)
    same = lambda x: x[~((x.DeviceId == "S00") & (x.Detector == 2))].health_status.tolist()
    assert same(A[0]) == same(B[0])
    other = lambda x: x[x.DeviceId != "S00"].reset_index(drop=True)                             # other signals: no change
    _equal(other(A[0]), other(B[0]))
    _equal(other(A[1]), other(B[1]))


def test_listed_tie_median_on_exactly_a_quarter_second():
    """half the ONs of a Count zone 0.2 s, half 0.3 s: the median is exactly 0.25 s, and a Count zone at or under it is a
    "pulse" zone.  Today's median of double seconds lands just over 0.25 here ("normal"); the integers give exactly 0.25."""
    ev, det = synth.make(2, 1, seed=3)
    t0 = ev.Timestamp.min()
    S = lambda tenths: t0 + pd.Timedelta(milliseconds=100 * tenths)
    rows = []
    for i, d in enumerate((2, 3, 2, 3, 2, 3)):
        s = 72004 + 600 * i
        rows += [("S00", S(s), 82, 17), ("S00", S(s + d), 81, 17)]
    con = G.connect(_replace(ev, 17, t0, ev.Timestamp.max() + pd.Timedelta(seconds=1), rows), det)
    _whole(con)
    a, b = G.steps(H0, con, ["act", "xm"]), G.steps(H, con, ["act", "xm"])
    pick = lambda x, c: x[0][(x[0].dev == 1) & (x[0].det == 17)][c].iloc[0]
    assert 0.25 < pick(a["act"], "dur_p50") < 0.25 + 1e-9 and pick(b["act"], "dur_p50") == 0.25
    assert pick(a["xm"], "cmode") == "normal" and pick(b["xm"], "cmode") == "pulse"
