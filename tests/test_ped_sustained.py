"""The sustained pressing check (ped option sustained_press, on): limits, the period, its rating, the findings table, the store."""
import math

import pandas as pd
import pytest

import test_incremental as TI  # noqa: F401  (test_ped_incremental's helpers need it importable)
from atspm_detector import ped_health_duckdb as P
from atspm_detector.findings import health
from test_ped_incremental import T0, batch, con_of, fed, ped_of, same_ped

DAY = 86400
BIN = 900
FIRST = 18 * 3600                                       # 09-29 00:00, in seconds from T0
START = T0 + pd.Timedelta(seconds=FIRST)
END = START + pd.Timedelta(days=3)
HOT_FROM = FIRST + 2 * DAY + 10 * 3600                  # 10:00 on the last day
ON = {"sustained_press": True}


def _presses(t0, span, n, ch=2):
    """n presses spread over the span, gaps uneven (even ones would read as machine-regular)"""
    rows = []
    for k in range(n):
        t = t0 + (k + 0.5 + 0.4 * math.sin(k * k)) * span / n
        rows += [(t, 90, ch), (t + 0.2, 89, ch), (t + 0.05, 45, 2)]
    return rows


def _walks(t0, span, n, ph=2):
    rows = []
    for k in range(n):
        t = t0 + (k + 0.25) * span / n
        rows += [(t, 21, ph), (t + 7, 22, ph), (t + 20, 23, ph)]
    return rows


def _log(usual=100, day=1200, hot=6, per_hot=40, walks_hot=2):
    """three logged days: `usual` presses on the first two, spread over 07:00-19:00 with a walk an hour; on the last, `hot` quarter hours in a row from 10:00 with per_hot presses and
    walks_hot walks each, and the rest of `day` spread over 00:00-10:00."""
    rows = [(float(t), 1, 4) for t in range(FIRST, FIRST + 3 * DAY, 100)]       # the signal keeps logging
    for d in range(3):
        base = FIRST + d * DAY
        if d < 2:
            rows += _walks(base + 7 * 3600, 12 * 3600, 12) + _presses(base + 7 * 3600, 12 * 3600, usual)
    for k in range(hot):
        rows += _presses(HOT_FROM + k * BIN, BIN, per_hot) + _walks(HOT_FROM + k * BIN, BIN, walks_hot)
    rows += _presses(FIRST + 2 * DAY, 10 * 3600, max(day - hot * per_hot, 0))
    return rows


def _found(rows, options=ON):
    i, p = P.ped_health(con_of(rows), start=START, end=END, periods=True, options=options)
    return i[i.Input == 2].iloc[0], p[p.condition == "sustained pressing"]


def test_on_by_default_and_it_can_be_switched_off():
    assert P.OPT["sustained_press"] is True
    assert len(_found(_log(), options=None)[1]) == 1
    i, p = _found(_log(), options={"sustained_press": False})
    assert p.empty and "sustained" not in i.reason


def test_it_flags_the_day_as_suspect_from_the_first_busy_quarter_hour_to_the_last():
    i, p = _found(_log())
    assert len(p) == 1 and i.status == "suspect"
    assert p.start.iloc[0] == T0 + pd.Timedelta(seconds=HOT_FROM)
    assert p.end.iloc[0] == T0 + pd.Timedelta(seconds=HOT_FROM + 6 * BIN)
    assert i.reason.startswith("sustained pressing: 1,200 presses on Thu 10-01, 12.0 x its usual ~100 a day, with 6 15-minute periods")
    assert 0.5 <= p.severity.iloc[0] < 0.8


def test_it_stays_suspect_however_strong_and_is_at_least_a_half_at_its_limit():
    assert _found(_log(usual=20, day=6000))[1].severity.iloc[0] == 0.79
    assert _found(_log(usual=200, day=1000))[1].severity.iloc[0] == pytest.approx(0.5)


@pytest.mark.parametrize("kw", [
    dict(day=999),                       # under 1,000 presses
    dict(usual=201, day=1000),           # under 5 x usual
    dict(usual=19, day=2000),            # usual under 20
    dict(hot=4),                         # 4 busy quarter hours, not 5
    dict(per_hot=29),                    # under 30 presses in a quarter hour
    dict(walks_hot=5),                   # 8 presses a walk: people waiting at a busy crossing
])
def test_one_limit_short_is_no_finding(kw):
    assert _found(_log(**kw))[1].empty


@pytest.mark.parametrize("kw", [dict(usual=200, day=1000), dict(usual=20, day=1000), dict(hot=5), dict(per_hot=30), dict(walks_hot=4)])
def test_at_each_limit_it_flags(kw):
    assert len(_found(_log(**kw))[1]) == 1


def test_a_quarter_hour_without_a_walk_counts_as_one_walk():
    assert len(_found(_log(walks_hot=0))[1]) == 1


def test_a_silent_day_counts_towards_usual():
    # one busy day and two logged days with no press at all: no usual level to stand against
    assert _found(_log(usual=0))[1].empty


def test_the_findings_table_reads_the_check():
    con = con_of(_log())
    f = health(con, "events", start=START, end=END, vehicle=False, ped=True, ped_options=ON)
    q = f[(f.Detector == 2) & (f.Label == "sustained pressing")]
    assert len(q) == 1 and q.Detail.iloc[0].startswith("sustained pressing: 1,200 presses") and 0.5 <= q.Confidence.iloc[0] < 0.8


def test_the_store_gives_the_same_answer_for_any_cutting():
    con = con_of(_log())
    fed(con, [START + pd.Timedelta(hours=h) for h in (7, 31, 60, 72)], horizon_days=3)
    got, want = ped_of(con, END, 3, ped_options=ON), batch(con, END, 3, options=ON)
    assert len(want[1][want[1].condition == "sustained pressing"]) == 1
    same_ped(got, want)
