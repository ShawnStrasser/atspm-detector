"""The walks-without-presses check (on since the review of 2026-11-03), unique presses, the Confidence of a suspect flag, and a very long presence ON over a log hole."""
import numpy as np
import pandas as pd
import pytest

import test_incremental as TI  # noqa: F401  (test_ped_incremental's helpers need it importable)
from atspm_detector import health_duckdb as H
from atspm_detector import ped_health_duckdb as P
from atspm_detector import ped_stop as S
from atspm_detector.findings import health
from test_ped_incremental import batch, con_of, fed, ped_of, same_ped

T0 = pd.Timestamp("2026-09-28 06:00")
DAY = 86400
FIRE = 2 * DAY + 3 * 3600    # 09:00 on day 3


def _unique(rows, ch=2):
    """the presses the judge's step uniq keeps"""
    con = con_of(rows)
    t = pd.Series([r[0] for r in rows])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE pq_wpart AS SELECT 0 AS dev, 'S' AS DeviceId, {T0.timestamp() - 1.0} AS s, {T0.timestamp() + t.max() + 2.0} AS e,
                    make_timestamp(({T0.timestamp() - 1.0} * 1e6)::BIGINT) AS lo, make_timestamp(({T0.timestamp() + t.max() + 2.0} * 1e6)::BIGINT) AS hi""")
    out = P._run_group(con, P._steps("events"), {"uniq": "dev, ch, t_on"})["uniq"]
    return len(out[out.ch == ch])


def _press(t, ch=2, call_ph=2, off=0.3):
    rows = [(t, 90, ch), (t + off, 89, ch)]
    if call_ph is not None:
        rows.append((t + 0.05, 45, call_ph))
    return rows


def test_one_person_pressing_many_times_counts_once():
    assert _unique([r for i in range(8) for r in _press(i * 1.0)]) == 1


def test_a_button_that_keeps_repeating_keeps_counting():
    assert _unique([r for i in range(360) for r in _press(i * 10.0)]) == 240    # every 10 s for an hour: 3,600 s of one run


def test_two_people_16_s_apart_count_twice_and_a_gap_of_exactly_15_s_stays_in_the_run():
    assert _unique(_press(0.0) + _press(16.0)) == 2
    assert _unique(_press(0.0) + _press(15.0)) == 2         # 15 s after the run's start: the next slot of the same run
    assert _unique(_press(0.0) + _press(14.9)) == 1


def _log(days=3.0, persons_day=40, presses_each=1, walk_every_from=None, walk_every_until=None, stop_at=None, ch=2, ph=2, seed=1, recall=False, second_input=None, walk_every_n=1):
    """a signal with greens of its phase every 100 s.  People come 07:00-19:00: each brings presses_each presses 1 s apart, a ped call, and a walk at the next green.  stop_at (s): no
    press from then on.  walk_every_from (s): a walk at every walk_every_n-th green from then on (a button held ON with its ON lost, or recall).  second_input: another input calling
    the phase, pressed then."""
    rng = np.random.default_rng(seed)
    rows, walks = [], set()
    n = int(days * DAY)
    day0 = T0.normalize()
    for d in range(int(days) + 2):
        base = (day0 + pd.Timedelta(days=d) - T0).total_seconds()
        for t in np.sort(rng.uniform(base + 7 * 3600, base + 19 * 3600, persons_day)):
            if t < 0 or t >= n or (stop_at is not None and t >= stop_at):
                continue
            for k in range(presses_each):
                rows += _press(float(t) + k, ch, ph)
            walks.add(int(t // 100) + 1)
    if second_input is not None:
        for t in np.arange(second_input[0], second_input[1], 110.0):
            rows += _press(float(t), 5, ph)
            walks.add(int(t // 100) + 1)
    for g in range(0, n // 100):
        s = g * 100.0
        rows.append((s, 1, ph))
        rows.append((s + 40, 8, ph))
        rows.append((s + 50, 1, ph + 2))     # the signal keeps logging: another phase's green
        every = recall or (walk_every_from is not None and s >= walk_every_from and (walk_every_until is None or s < walk_every_until) and g % walk_every_n == 0)
        if g in walks or every:
            rows += [(s + 0.5, 21, ph), (s + 8, 22, ph), (s + 20, 23, ph)]
    return rows


def _periods(rows, ch=2, days=3, **kw):
    i, p = P.ped_health(con_of(rows), start=T0, end=T0 + pd.Timedelta(days=days), periods=True, **kw)
    return i[i.Input == ch].iloc[0], p[p.Detector == ch]


def _walks(p):
    return p[p.condition == "walks without presses"]


def test_the_check_is_on_and_can_be_switched_off():
    assert P.OPT["walks_no_press"] is True
    rows = _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE)
    assert len(_walks(_periods(rows)[1])) == 1
    assert _walks(_periods(rows, options={"walks_no_press": False})[1]).empty
    assert not {"uniq", "pbin", "f_nop"} & {n for n, _ in P._steps("events", {"walks_no_press": False})}


def test_walks_jump_while_presses_stop_flags_within_about_an_hour_as_suspect():
    rows = _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE)
    # the first judge able to see it: 1 h 15 min after the walks began
    i, p = P.ped_health(con_of(rows), start=T0, end=T0 + pd.Timedelta(seconds=FIRE + 4500), periods=True)
    q = _walks(p[p.Detector == 2])
    assert len(q) == 1 and i[i.Input == 2].status.iloc[0] == "suspect"
    assert "walks jumped with no presses" in i[i.Input == 2].reason.iloc[0]
    assert (T0 + pd.Timedelta(seconds=FIRE) - q.start.iloc[0]) <= pd.Timedelta(minutes=30)
    # and 30 minutes in it does not yet
    _, p2 = P.ped_health(con_of(rows), start=T0, end=T0 + pd.Timedelta(seconds=FIRE + 1800), periods=True)
    assert _walks(p2).empty


def _judged_at(rows, seconds, **kw):
    i, p = P.ped_health(con_of(rows), start=T0, end=T0 + pd.Timedelta(seconds=seconds), periods=True, **kw)
    return i[i.Input == 2].iloc[0], _walks(p[p.Detector == 2])


@pytest.mark.parametrize("persons", [120, 400])
def test_it_stays_suspect_however_strong_until_the_walks_have_run_long_enough(persons):
    fire = 2 * DAY + 13 * 3600     # 19:00: those hours usually bring next to no walks, so the jump is as strong as it gets
    rows = _log(persons_day=persons, stop_at=fire, walk_every_from=fire)
    i, q = _judged_at(rows, fire + S.NP_BAD_H * 3600 - 900)
    assert len(q) == 1 and 0.5 <= q.severity.iloc[0] < 0.8 and i.status == "suspect"
    i, q = _judged_at(rows, fire + S.NP_BAD_H * 3600)
    assert len(q) == 1 and q.severity.iloc[0] >= 0.8 and i.status == "bad" and "walks at nearly every green for 2.5 logged hours" in i.reason


def test_switched_off_a_long_run_stays_suspect():
    i, q = _judged_at(_log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE), 3 * DAY, options={"walks_bad": False})
    assert len(q) == 1 and q.severity.iloc[0] < 0.8     # the sudden stop of the same silence is judged by itself


def test_only_the_hours_with_walks_rate_it():
    # walks for 2 hours, then the run of no presses goes on for hours with no extra walks: not long enough
    rows = _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE, walk_every_until=FIRE + 7200)
    i, q = _judged_at(rows, 3 * DAY)
    assert len(q) == 1 and q.severity.iloc[0] < 0.8     # the sudden stop of the same silence is judged by itself


def test_walks_with_no_green_logged_do_not_rate_it():
    # the phase's greens are not in the log for the run: walks at every green cannot be said
    rows = [r for r in _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE) if not (r[0] >= FIRE and r[1] == 1 and r[2] == 2)]
    i, q = _judged_at(rows, 3 * DAY)
    assert len(q) == 1 and q.severity.iloc[0] < 0.8     # the sudden stop of the same silence is judged by itself


def test_the_confidence_of_every_rule_rated_by_a_floor_is_at_least_a_half():
    assert P.SEV_FLOOR == {"sudden stop": 0.5, "long hold": 0.5, "walks without presses": 0.5}


def test_a_flagged_stop_with_a_weak_score_reaches_the_report_limit():
    # zero for 36 logged hours from about 34 presses a day, flagged by the simple test only: its score alone is far under 0.5
    i, p = _periods(_log(persons_day=34, stop_at=1.5 * DAY))
    q = p[p.condition == "sudden stop"]
    assert len(q) == 1 and i.status == "suspect" and q.severity.iloc[0] == 0.5


def test_too_little_look_back_does_not_judge():
    # the run starts under 24 logged hours into the window: no usual level for those hours of day
    fire = 20 * 3600
    assert _walks(_periods(_log(persons_day=120, stop_at=fire, walk_every_from=fire))[1]).empty


def test_the_look_back_limit_is_in_logged_hours():
    assert S.NP_MIN_LOOK_H == 24 and f">= {S.NP_MIN_LOOK_H} * 4" in dict(P._steps("events"))["f_nop"]


def test_walks_at_few_of_the_greens_do_not_flag():
    # every 4th green walks: 9 walks an hour, 3 x usual at the night hours, but a share of 0.25: not a held button
    rows = _log(persons_day=20, stop_at=1.5 * DAY, walk_every_from=1.5 * DAY, walk_every_n=4)
    assert _walks(_periods(rows)[1]).empty


def test_a_button_of_a_phase_on_recall_all_along_does_not_flag():
    assert _walks(_periods(_log(persons_day=30, recall=True, stop_at=FIRE))[1]).empty


def test_a_quiet_button_with_no_extra_walks_does_not_flag():
    assert _walks(_periods(_log(persons_day=40, stop_at=FIRE))[1]).empty


def test_walks_brought_by_ped_calls_of_another_input_are_not_this_rule():
    rows = _log(persons_day=60, stop_at=FIRE, second_input=(FIRE, FIRE + 7200))
    assert _walks(_periods(rows)[1]).empty


def test_all_buttons_of_a_signal_jumping_together_is_a_plan_not_a_button():
    a = _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE)
    b = _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE, ch=6, ph=6, seed=2)
    assert _walks(P.ped_health(con_of(a + b), start=T0, end=T0 + pd.Timedelta(days=3), periods=True)[1]).empty
    assert not _walks(P.ped_health(con_of(a), start=T0, end=T0 + pd.Timedelta(days=3), periods=True)[1]).empty


def test_a_stuck_on_with_its_on_in_the_log_stays_stuck_on_alone():
    rows = _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE) + [(float(FIRE) - 5, 90, 2), (float(FIRE) - 4.95, 45, 2)]
    _, p = _periods(rows)
    assert (p.condition == "stuck on").any() and _walks(p).empty


@pytest.mark.parametrize("seed", [1, 2])
def test_any_cutting_into_updates_gives_the_batch_answer(seed):
    rows = _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE, seed=seed) + _log(persons_day=60, ch=6, ph=6, seed=seed + 5, stop_at=1.2 * DAY)
    con = con_of(rows)
    end = T0 + pd.Timedelta(days=3)
    rng = np.random.default_rng(seed)
    cuts = sorted(T0 + pd.to_timedelta(rng.uniform(0, 3 * DAY, 30), unit="s")) + [end]
    fed(con, cuts, horizon_days=3)
    got, want = ped_of(con, end, 3), batch(con, end, 3)
    assert not _walks(want[1]).empty
    same_ped(got, want, f"seed {seed}")


def _few(every=None, fire=FIRE, presses=((0.4, 2), (1.4, 2))):
    """a button pressed a couple of times in the look-back, a walk with each press; from `fire` a walk at every `every`-th cycle of 100 s"""
    rows, walks = [], set()
    for f, _ in presses:
        t = f * DAY
        rows += _press(t)
        walks.add(int(t // 100) + 1)
    for g in range(0, int(3 * DAY) // 100):
        s = g * 100.0
        rows += [(s, 1, 2), (s + 40, 8, 2), (s + 50, 1, 4)]
        if g in walks or (every and s >= fire and (g % every) == 0):
            rows += [(s + 0.5, 21, 2), (s + 8, 22, 2), (s + 20, 23, 2)]
    return rows


def test_walks_jump_on_a_button_with_very_few_presses_flags():
    i, p = _periods(_few(every=1))
    assert not _walks(p).empty and i.status == "bad"


def test_a_button_on_recall_all_along_with_almost_no_presses_does_not_flag():
    rows = list(_few()) + [r for g in range(0, int(3 * DAY) // 100) for r in ((g * 100.0 + 0.5, 21, 2), (g * 100.0 + 8, 22, 2), (g * 100.0 + 20, 23, 2))]
    assert _walks(_periods(sorted(set(rows)))[1]).empty


def test_the_findings_table_reads_the_check():
    con = con_of(_log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE))
    f = health(con, "events", start=T0, end=T0 + pd.Timedelta(days=3), vehicle=False, ped=True)
    q = f[(f.Detector == 2) & (f.Label == "walks without presses")]
    assert len(q) == 1 and q.Detail.iloc[0].startswith("walks jumped with no presses") and q.Confidence.iloc[0] >= 0.8
    # judged after an hour and a quarter it is suspect
    f = health(con, "events", start=T0, end=T0 + pd.Timedelta(seconds=FIRE + 4500), vehicle=False, ped=True)
    q = f[(f.Detector == 2) & (f.Label == "walks without presses")]
    assert len(q) == 1 and 0.5 <= q.Confidence.iloc[0] < 0.8


def test_the_rows_do_not_read_a_number():
    # the same log with the input and phase numbers changed gives the same answer
    a = _log(persons_day=120, stop_at=FIRE, walk_every_from=FIRE)
    b = [(t, e, {2: 11, 4: 3}.get(n, n)) for t, e, n in a]
    xp, yp = _periods(a)[1], _periods(b, ch=11)[1]
    assert len(_walks(xp)) == 1 == len(_walks(yp)) and _walks(xp).start.iloc[0] == _walks(yp).start.iloc[0]


# --- a very long ON of a presence input over a log hole ---------------------------------------------------------------------------
def _presence(hold_s=6 * 3600, holes=(3 * 3600,), hole_s=40.0):
    """a signal that logs every 10 s; an input whose presses last 8 s (presence), held ON for hold_s from 02:00 on day 2, the signal silent hole_s at each hole"""
    rows = []
    on = DAY + 2 * 3600
    for t in np.arange(0, 2 * DAY, 3600.0):
        if t < on - 3600 or t > on + hold_s:
            for k in range(3):
                rows += [(t + k * 60, 90, 2), (t + k * 60 + 8, 89, 2), (t + k * 60 + 0.05, 45, 2)]
    rows += [(float(on), 90, 2), (float(on) + 0.05, 45, 2), (float(on + hold_s), 89, 2)]
    t = 0.0
    while t < 2 * DAY:
        if not any(on + h <= t < on + h + hole_s for h in holes):
            rows.append((t, 1, 2))
        t += 10.0
    return rows


def _stuck(rows):
    i, p = P.ped_health(con_of(rows), start=T0, end=T0 + pd.Timedelta(days=2), periods=True)
    return i[i.Input == 2].iloc[0], p[(p.Detector == 2) & (p.condition == "stuck on")]


def test_a_long_presence_on_over_a_log_hole_is_a_stuck_period():
    i, s = _stuck(_presence())
    assert i.Type == "presence" and len(s) == 1 and i.status == "bad"
    assert (s.end.iloc[0] - s.start.iloc[0]) == pd.Timedelta(hours=6) and "held 6 h" in i.reason


def test_a_presence_on_with_no_long_stretch_between_holes_stays_unjudged():
    # 20 minutes held, a hole every 5 minutes: a lost OFF and ON cannot be told from a hold
    i, s = _stuck(_presence(hold_s=1200, holes=(300, 600, 900)))
    assert i.Type == "presence" and s.empty


def test_a_short_presence_on_with_no_hole_is_still_judged_on_its_length():
    assert len(_stuck(_presence(hold_s=900, holes=()))[1]) == 1
    assert _stuck(_presence(hold_s=500, holes=()))[1].empty


# --- wording -----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ratio,has,hasnt", [(2.322e12, "far above its phase mates", " x "), (100.0, "far above its phase mates", " x "), (99.0, "99.0 x", "far above"), (5.25, "5.2 x", "far above")])
def test_a_rise_against_almost_silent_mates_reads_in_plain_words(ratio, has, hasnt):
    import duckdb
    import re
    s = dict((n, q) for n, q, _ in H.build("events", "detectors").ctes)["evid"]
    expr = re.search(r"WHEN 'rise' THEN (CASE WHEN x\.mr_r >= 100 .*? END)", s, re.S).group(1)
    got = duckdb.connect().execute(f"SELECT {expr} FROM (SELECT {ratio}::DOUBLE AS mr_r) x").fetchone()[0]
    assert has in got and hasnt not in got and "2322" not in got
