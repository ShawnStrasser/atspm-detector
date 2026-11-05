# FROZEN ORACLE (stage 0 of the incremental-health design): src/atspm_detector/ped_health_duckdb.py as of 2026-10-29 (ped round 5 + ped movement
# model merged).  Test-only; never edited.  Changes against the original: imports and weight paths only.
"""Pedestrian push-button health in DuckDB SQL (research notes 123, 129, 155): one row per ped input (push button)
of each signal, from the controller's own ped events.  Separate from the detector classifier and its health: the
classifier never sees ped events; this module reads nothing else.

    ped_health(con, events_table="events", start=None, end=None) -> DataFrame
    ped_health(con, ..., periods=True) -> (inputs, periods)

events_table: DeviceId, Timestamp, EventId, Parameter (the controller log, local time).  Codes used: 89 / 90 ped input
off / on (Parameter = input), 45 ped call (phase), 21 / 22 / 23 walk / ped clearance / don't walk (phase), 67 ped
overlap walk (overlap), 1 begin green (phase), 102 / 104 preempt call input on / off, 105 / 111 preempt entry / begin
exit and 110 max presence exceeded (preempt number), 8 begin yellow (phase); any event of the signal counts as "the
signal was logging".  A log gap: 2 whole clock minutes or more in which the signal logged nothing at all; a log hole:
30 s or more without any event of the signal (a lost block of the log: our Dec 2024 feed loses 1-3 min blocks at about
:13 and :43 past each hour; option log_gaps).

Per input (one row each with presses in the window):
  Input       the ped input channel (Parameter of 89 / 90)
  Phase       the phase it calls: the phase of the ped call (45) the controller logs within 0.25 s of its presses
              (>= 1 % of presses; most presses, then the phase numbered like the input; a button that calls two
              phases at once gets a config note), else the phase whose walk follows >= 90 % of presses within 180 s
              (ped recall; ties: the phase whose walk holds the fewest presses for its share of time), else none
  Type        pulse (< 0.5 % of presses 0.6-10 s long), hold (10 % of presses <= 0.2 s), hold_min (a ~0.3 s minimum
              ON), presence (median ON >= 5 s: ON while a person waits), unknown (< 30 presses)
  presses, status (ok / config_issue / suspect / bad / not_enough_data), reason, config (wiring and timing notes,
              not faults).  config_issue: the button works but the controller's set-up keeps its presses from being
              served (informational, like a lane closure for vehicle detectors: not a fault of the button, but listed)
Checks (per calendar day; the input's status is its worst day: bad > suspect > config_issue > ok):
  stuck on          bad      one ON >= 300 s (presence input 600 s) or through >= 3 walks of its phase, or (push
                             inputs, user review 2026-10-29) ON >= 180 s through >= 2 walks of its phase; an ON with
                             no OFF before the log ends counts as ON to the log's end, if >= 3 walks came in it (a
                             lost OFF brings no walks).  Not counted: the log-limit ON
                             (the 31st / 61st press of an unbroken run of presses < 3 s apart, >= 5 s long), and the
                             3-walk rule (not the 180-s / 2-walk one) on a day the input is ON >= 60 s together with
                             another input.  An ON with a log hole inside it is judged on its walks only (>= 3 walks,
                             push inputs): its length is not known
  machine-regular   bad      >= 50 % of its >= 20 press gaps of >= 5 s the same length (+-0.15 s), whether or not
                             the input places ped calls (user review 2026-10-29)
  never calls       config_issue  >= 10 presses that day, no phase, no ped overlap and not machine-regular: no ped
                             movement is assigned to the button.  A log without any ped overlap walk 67 cannot tell
                             an overlap button apart: then a config note "never calls a walk not checked", no finding
                             (option log_gaps)
  call dropped      config_issue  a ped call (45, logged outside the walk) stood through >= 2 green starts of its
                             phase without a walk (the green that serves the walk is not counted; usually the
                             controller cleared it: the next 45 comes before any walk) and no preemption during the
                             wait; charged to the input whose press placed it, else to every input of the phase.
                             A wait with any preemption active at any time during it (user review 2026-10-29) is
                             excused as a whole: a note ("omitted during preemption"), not a finding.  A preemption
                             (any preempt number) runs from its call input on (102) or entry (105) to its logged end:
                             the first input off (104), exit (111) or max presence (110) of that preempt number after
                             it; with none logged it is taken as still on, for at most 6 h (user review 2026-10-29);
                             plus 120 s recovery as the atspm package's phase-wait check.  A log without any preempt
                             event cannot tell the two apart: every wait counts as call dropped and the reason says
                             the preemption was not checked.  A wait ends at a log hole (only the greens before it
                             count).  Option late_call (off): the first green does not count when the call came after
                             the yellow that led into it began (the last yellow 8 of the signal <= 15 s before the
                             green): the controller picks its next phases about when that yellow starts
  dead              suspect  >= 6 h of 07-19 h without a waiting ped while the signal was logging (no log gap in
                             the hour) and the input's own daytime level expects >= 20 waits in them
Config notes (status stays ok, user review 2026-10): served by a ped overlap (no ped call; >= 80 % of presses followed
by that overlap's walk within 180 s), phase on ped recall, one button wired to two inputs (>= 95 % identical ONs),
calls two ped phases (>= 20 % of its ONs >= 1 s identical to another input's, or both ON >= 60 s within 5 s), ped
calls omitted during preemption, never calls a walk not checked (a log without any ped overlap walk).
not_enough_data: no finding and fewer than 10 presses in the window.
assign=True (the ped movement model, ped_assign): + Phases, Overlaps, assign_confidence per input; an input with >= 10
presses whose presses call no phase or ped overlap is a config_issue (user ruling 2026-10-29) with a config note (a log
without any ped overlap walk: the note "ped movement not checked" only, option log_gaps).
With periods=True also one row per finding: DeviceId, Detector (= Input), start, end, condition, severity (0-1: 0.5
at the check's limit, 0.8 at its "clearly broken" mark; 0 for an "omitted during preemption" note), limit_s (stuck
on, dead), in_closure (always false), whole_window (always false: every finding has its own time).
options: switch the 2026-10-29 rules (on by default but late_call; all off = the first package pass of note 155
exactly): regular_bad (machine-regular always bad), config_issue (never calls / call dropped -> config_issue, not
suspect), preempt_wait (a preemption during a call wait excuses the whole wait), stuck_2walks (ON >= 180 s through >= 2
walks is stuck), orphan_clearance (a ped clearance 22 or don't walk 23 whose walk 21 is missing from the log counts as
a walk, from the phase's green just before it unless the phase's yellow came in between: then that green's start was
lost too and the walk starts at the 22 / 23 itself: lost data never reads as a call that stood), log_gaps (a log gap or
hole, or a log without any ped overlap walk, never makes a finding: see stuck on, never calls, call dropped, dead);
proposed, off by default: late_call (see call dropped).
"""
from __future__ import annotations

import math

import pandas as pd

PREFIX = "pq_"
PRE_CODES = (102, 104, 105, 110, 111)
CODES = (1, 8, 21, 22, 23, 45, 67, 89, 90) + PRE_CODES
# user review 2026-10-29 (note 155b): approved, on; all off = the first package pass (note 155)
OPT = {"regular_bad": True, "config_issue": True, "preempt_wait": True, "stuck_2walks": True,
       # data loss never makes a finding: a lost walk start counts as a walk; a log gap ends a wait / an ON's length
       "orphan_clearance": True, "log_gaps": True,
       # proposed, OFF until approved: a call placed after the yellow before a green began is not owed that green
       "late_call": False}
# a preemption (user review 2026-10-29): from 102 / 105 to the first 104 / 110 / 111 of its number after it (Sep 2026:
# 111 missing after 1,676 of 2,425 entries, the 104 is there); none logged: still on, at most 6 h; recovery 120 s as
# the atspm package's phase-wait check (preempt_recovery_seconds)
PRE = {"cap_s": 21600, "recovery_s": 120}
NOTE = "omitted during preemption"
LIM = {"call_s": 0.25, "registers": 0.01, "follow_s": 180, "walk_share": 0.9, "overlap_share": 0.8,
       "second_phase": 0.3, "typed": 30, "stuck_s": 300, "stuck_presence_s": 600, "stuck_walks": 3,
       "stuck2_s": 180, "stuck2_walks": 2,
       "logcap_gap_s": 3, "logcap_s": 5,
       "joint_s": 60, "joint_start_s": 5, "mirror_s": 0.15, "mirror_on_s": 1, "mirror_ons": 10, "mirror_share": 0.2,
       "mirror_full": 0.95, "regular_gap_s": 5, "regular_tol_s": 0.15, "regular_gaps": 20, "regular_share": 0.5,
       "no_call": 10, "drop_greens": 2, "serve_s": 10, "dead_h": 6, "dead_waits": 20, "wait_gap_s": 120,
       "recall_share": 0.9, "recall_walks": 50, "few": 10, "gap_min": 2, "hole_s": 30, "orphan_s": 120,
       "late_s": 15}
# severity: 0.5 at the limit (lo), 0.8 at the "clearly broken" mark (hi), on log strength
SEV = {"stuck on": (0.5, 1.0), "machine-regular presses": (0.5, 0.75), "never calls a walk": (10, 100),
       "call dropped": (2, 20), "dead": (20, 40)}
COLS = ["DeviceId", "Input", "Phase", "Type", "presses", "status", "reason", "config"]
PERIOD_COLS = ["DeviceId", "Detector", "start", "end", "condition", "severity", "limit_s", "in_closure", "whole_window"]


def _steps(E: str, opt: dict | None = None) -> list:
    """the temp tables, in order: (name, SELECT).  pq_wpart (dev, DeviceId, s, e, lo, hi) holds the signals of one
    group and their windows; times are epoch seconds (DOUBLE), days / hours count from the epoch (local time)."""
    L = LIM
    opt = {**OPT, **(opt or {})}
    on = {k: str(bool(v)).lower() for k, v in opt.items()}         # option -> SQL true / false
    pw = f"FROM {E} e JOIN {PREFIX}wpart w ON e.DeviceId = w.DeviceId WHERE e.Timestamp >= w.lo AND e.Timestamp < w.hi"
    late = f"""ga AS (SELECT g1.*, g.ts AS gf FROM g1 ASOF LEFT JOIN pq_green g ON g.dev = g1.dev AND g.ph = g1.ph AND g1.tc < g.ts),
gy AS (SELECT ga.*, coalesce(ga.tc >= y.ts AND ga.gf - y.ts <= {L['late_s']}, false)::INT AS late
       FROM ga ASOF LEFT JOIN pq_yel y ON y.dev = ga.dev AND ga.gf > y.ts),""" if opt["late_call"] else \
        "gy AS (SELECT g1.*, 0 AS late FROM g1),"
    return [
        # events, de-duplicated; minutes and hours in which the signal logged anything
        ("ev", f"""SELECT DISTINCT w.dev, epoch_ns(e.Timestamp) / 1e9 AS ts, e.EventId::INT AS ev, e.Parameter::INT AS p
                   {pw} AND e.EventId IN {CODES}"""),
        ("b10", f"""SELECT w.dev, epoch_ns(e.Timestamp) // 10000000000 AS k, min(epoch_ns(e.Timestamp)) / 1e9 AS t0,
                           max(epoch_ns(e.Timestamp)) / 1e9 AS t1 {pw} GROUP BY ALL"""),
        ("mins", "SELECT dev, k // 6 AS m, max(t1) AS tmax FROM pq_b10 GROUP BY ALL"),
        ("hours", "SELECT dev, m // 60 AS h, max(tmax) AS tmax FROM pq_mins GROUP BY ALL"),
        # log gaps (option log_gaps): >= 2 whole clock minutes without any event of the signal, [g0, g1) epoch seconds
        ("gap", f"""
SELECT dev, (m + 1) * 60.0 AS g0, nm * 60.0 AS g1
FROM (SELECT dev, m, lead(m) OVER (PARTITION BY dev ORDER BY m) AS nm FROM pq_mins)
WHERE nm - m > {L['gap_min']} AND {on['log_gaps']}"""),
        # log holes (option log_gaps): >= 30 s without any event of the signal (exact: from 10-s buckets), [g0, g1)
        ("hole", f"""
SELECT dev, t1 AS g0, nt0 AS g1
FROM (SELECT dev, t1, lead(t0) OVER (PARTITION BY dev ORDER BY k) AS nt0 FROM pq_b10)
WHERE nt0 - t1 >= {L['hole_s']} AND {on['log_gaps']}"""),
        # hours the signal logged without a gap (the dead check's "logging" hours)
        ("alive", """SELECT h.dev, h.h FROM pq_hours h
                      ANTI JOIN pq_gap g ON g.dev = h.dev AND g.g0 < (h.h + 1) * 3600 AND g.g1 > h.h * 3600"""),
        # ONs: 90 -> the next 89 of the input (none before the next 90: no OFF); the last ON with nothing after it is
        # ON until the log ends (open_s)
        ("press", """
SELECT x.dev, x.ch, x.t_on, x.t_off, x.t_off - x.t_on AS dur, CASE WHEN x.nev IS NULL THEN l.last - x.t_on END AS open_s,
       floor(x.t_on / 86400)::BIGINT AS day
FROM (SELECT dev, p AS ch, ts AS t_on, ev, lead(ev) OVER w AS nev, CASE WHEN lead(ev) OVER w = 89 THEN lead(ts) OVER w END AS t_off
      FROM pq_ev WHERE ev IN (89, 90) WINDOW w AS (PARTITION BY dev, p ORDER BY ts, ev)) x
JOIN (SELECT dev, max(tmax) AS last FROM pq_hours GROUP BY dev) l USING (dev) WHERE x.ev = 90"""),
        # green starts, yellow starts, walk cycles (21, then 22, then 23), ped calls, ped overlap walks
        ("green", "SELECT dev, p AS ph, ts, row_number() OVER (PARTITION BY dev, p ORDER BY ts) AS k FROM pq_ev WHERE ev = 1"),
        ("yel", "SELECT dev, p AS ph, ts FROM pq_ev WHERE ev = 8"),
        # (option orphan_clearance: a clearance 22 or don't walk 23 with no walk 21 since the last 23 is a walk whose 21
        # (and 22) was lost from the log; it starts at the phase's green start <= 120 s before it, else at itself; not
        # at a green the phase already ended (a yellow of the phase > 1 s before the 22 / 23 came after it): that
        # green's start was lost too)
        ("walk", f"""
WITH x AS (SELECT dev, p, ts, ev, lag(ev) OVER w AS e0, lead(ev) OVER w AS e1, lead(ts) OVER w AS t1,
                  lead(ev, 2) OVER w AS e2, lead(ts, 2) OVER w AS t2
           FROM pq_ev WHERE ev IN (21, 22, 23) WINDOW w AS (PARTITION BY dev, p ORDER BY ts, ev)),
y AS (SELECT * FROM x WHERE ev = 21 OR (ev IN (22, 23) AND coalesce(e0, 23) = 23 AND {on['orphan_clearance']}))
SELECT y.dev, y.p AS ph, CASE WHEN y.ev = 21 OR g.ts IS NULL OR y.ts - g.ts > {L['orphan_s']}
                                   OR coalesce(v.ts > g.ts, false) THEN y.ts ELSE g.ts END AS t21,
       CASE WHEN y.ev = 22 THEN y.ts WHEN y.ev = 21 AND y.e1 = 22 THEN y.t1 END AS t22,
       CASE WHEN y.ev = 23 THEN y.ts WHEN y.ev = 22 AND y.e1 = 23 THEN y.t1
            WHEN y.ev = 21 AND y.e1 = 22 AND y.e2 = 23 THEN y.t2 WHEN y.ev = 21 AND y.e1 = 23 THEN y.t1 END AS t23
FROM y ASOF LEFT JOIN pq_green g ON g.dev = y.dev AND g.ph = y.p AND y.ts >= g.ts
ASOF LEFT JOIN pq_yel v ON v.dev = y.dev AND v.ph = y.p AND y.ts - 1 > v.ts"""),
        ("call", "SELECT dev, p AS ph, ts FROM pq_ev WHERE ev = 45"),
        ("ovw", "SELECT dev, p AS ov, ts FROM pq_ev WHERE ev = 67"),
        # preemptions (any preempt number): from the call input on (102) or an entry (105) to the first input off
        # (104), exit (111) or max presence (110) of that number after it; none logged: still on, at most 6 h (user
        # review 2026-10-29); + 120 s recovery (as the atspm package)
        ("pre", f"""
SELECT dev, ts AS ps, least(coalesce(en, ts + {PRE['cap_s']}), ts + {PRE['cap_s']}) + {PRE['recovery_s']} AS pe
FROM (SELECT dev, ts, ev, min(CASE WHEN ev IN (104, 110, 111) THEN ts END) OVER f AS en
      FROM pq_ev WHERE ev IN {PRE_CODES}
      WINDOW f AS (PARTITION BY dev, p ORDER BY ts, ev ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING))
WHERE ev IN (102, 105) AND {on['preempt_wait']}"""),
        # signals that logged any preempt event (none in the whole log: the preemption check could not run), any ped
        # overlap walk (none in the whole log: an overlap button cannot be told apart)
        ("prelog", f"SELECT DISTINCT dev FROM pq_ev WHERE ev IN {PRE_CODES}"),
        ("ovlog", "SELECT DISTINCT dev FROM pq_ev WHERE ev = 67"),
        # type and the ped-call phase of each input
        ("inp0", f"""
WITH n AS (SELECT dev, ch, count(*) AS presses,
                  count(*) FILTER (round(dur, 1) BETWEEN 0.6 AND 10) / count(*) AS f06,
                  quantile_cont(round(dur, 1), 0.1) AS q10, quantile_cont(round(dur, 1), 0.5) AS q50
           FROM pq_press GROUP BY ALL),
m AS (SELECT p.dev, p.ch, p.t_on, c.ts FROM pq_press p ASOF JOIN pq_call c ON c.dev = p.dev AND p.t_on <= c.ts
      WHERE c.ts - p.t_on <= {L['call_s']}),
k AS (SELECT *, row_number() OVER (PARTITION BY dev, ch ORDER BY k DESC, ph = ch DESC, ph) AS r FROM (
        SELECT m.dev, m.ch, c.ph, count(*) AS k, any_value(t.n) AS n45
        FROM m JOIN pq_call c ON c.dev = m.dev AND c.ts = m.ts    -- every phase called at that instant
        JOIN (SELECT dev, ch, count(*) AS n FROM m GROUP BY ALL) t ON t.dev = m.dev AND t.ch = m.ch GROUP BY ALL)),
a AS (SELECT dev, ch, any_value(n45) AS n45, any_value(ph) FILTER (r = 1) AS ph45,
             any_value(ph) FILTER (r = 2 AND k >= {L['second_phase']} * n45) AS ph2 FROM k GROUP BY ALL)
SELECT n.dev, n.ch, n.presses, coalesce(a.n45, 0) >= {L['registers']} * n.presses AS registers, a.ph45, a.ph2,
       CASE WHEN n.presses < {L['typed']} THEN 'unknown' WHEN n.q50 >= 5 THEN 'presence' WHEN n.f06 < 0.005 THEN 'pulse'
            WHEN n.q10 <= 0.2 THEN 'hold' ELSE 'hold_min' END AS type
FROM n LEFT JOIN a USING (dev, ch)"""),
        # no ped call: the phase whose walk follows >= 90 % of its presses within 180 s (fewest presses in that walk
        # for its share of time first)
        ("fb", f"""
WITH pp AS (SELECT p.dev, p.ch, p.t_on, w.ph FROM pq_press p JOIN pq_inp0 i USING (dev, ch)
            JOIN (SELECT DISTINCT dev, ph FROM pq_walk) w USING (dev) WHERE NOT i.registers),
f1 AS (SELECT pp.*, w.t21 AS nxt FROM pp ASOF LEFT JOIN pq_walk w ON w.dev = pp.dev AND w.ph = pp.ph AND pp.t_on <= w.t21),
f2 AS (SELECT f1.*, w.t22 AS pt22 FROM f1 ASOF LEFT JOIN pq_walk w ON w.dev = f1.dev AND w.ph = f1.ph AND f1.t_on >= w.t21),
s AS (SELECT dev, ch, ph, avg(CASE WHEN nxt - t_on <= {L['follow_s']} THEN 1 ELSE 0 END) AS share_walk,
             avg(CASE WHEN t_on < pt22 THEN 1 ELSE 0 END) AS share_inwalk FROM f2 GROUP BY ALL),
ws AS (SELECT dev, ph, sum(t22 - t21) AS wsec FROM pq_walk GROUP BY ALL),
sp AS (SELECT dev, max(t_on) - min(t_on) AS span FROM pq_press GROUP BY dev)
SELECT dev, ch, ph FROM (
  SELECT s.*, row_number() OVER (PARTITION BY s.dev, s.ch ORDER BY s.share_walk >= {L['walk_share']} DESC,
         s.share_inwalk / greatest(ws.wsec / nullif(sp.span, 0), 1e-6) ASC NULLS LAST, s.share_walk DESC, s.ph) AS r
  FROM s JOIN ws USING (dev, ph) JOIN sp USING (dev)) WHERE r = 1 AND share_walk >= {L['walk_share']}"""),
        # still none: the ped overlap whose walk (67) follows >= 80 % of its presses within 180 s
        ("ovl", f"""
WITH pp AS (SELECT p.dev, p.ch, p.t_on, o.ov FROM pq_press p JOIN pq_inp0 i USING (dev, ch)
            ANTI JOIN pq_fb f USING (dev, ch) JOIN (SELECT DISTINCT dev, ov FROM pq_ovw) o USING (dev)
            WHERE NOT i.registers),
f AS (SELECT pp.*, o.ts FROM pp ASOF LEFT JOIN pq_ovw o ON o.dev = pp.dev AND o.ov = pp.ov AND pp.t_on <= o.ts)
SELECT dev, ch, arg_max(ov, share * 1000 - ov) AS ov FROM (
  SELECT dev, ch, ov, avg(CASE WHEN ts - t_on <= {L['follow_s']} THEN 1 ELSE 0 END) AS share FROM f GROUP BY ALL)
WHERE share >= {L['overlap_share']} GROUP BY ALL"""),
        ("inp", """
SELECT i.dev, i.ch, i.presses, i.registers, i.type, CASE WHEN i.registers THEN i.ph45 ELSE f.ph END AS ph,
       CASE WHEN i.registers THEN i.ph2 END AS ph2, o.ov
FROM pq_inp0 i LEFT JOIN pq_fb f USING (dev, ch) LEFT JOIN pq_ovl o USING (dev, ch)"""),
        # every ON with its phase's walk state, the 45 it placed, the gap since the input's previous ON, the log-limit
        # mark and the waiting-group key (don't walk: its cycle; before any walk: a new group after a 120-s gap)
        ("pa", f"""
WITH x AS (SELECT p.*, i.ph, i.type FROM pq_press p JOIN pq_inp i USING (dev, ch)),
y AS (SELECT x.*, w.t21, w.t22, w.t23 FROM x ASOF LEFT JOIN pq_walk w ON w.dev = x.dev AND w.ph = x.ph AND x.t_on >= w.t21),
z AS (SELECT y.*, c.ts AS c45, y.t_on - lag(y.t_on) OVER (PARTITION BY y.dev, y.ch ORDER BY y.t_on) AS gap
      FROM y ASOF LEFT JOIN pq_call c ON c.dev = y.dev AND c.ph = y.ph AND y.t_on <= c.ts),
r AS (SELECT z.*, sum(CASE WHEN gap IS NULL OR gap > {L['logcap_gap_s']} THEN 1 ELSE 0 END) OVER o AS run3,
             sum(CASE WHEN gap IS NULL OR gap > {L['wait_gap_s']} THEN 1 ELSE 0 END) OVER o AS run120
      FROM z WINDOW o AS (PARTITION BY dev, ch ORDER BY t_on ROWS UNBOUNDED PRECEDING))
SELECT dev, ch, ph, type, t_on, t_off, dur, open_s, day, t23, gap, run120,
       CASE WHEN t21 IS NULL THEN 'nowalk' WHEN t_on < t22 THEN 'walk' WHEN t_on < t23 THEN 'fdw' ELSE 'dw' END AS state,
       coalesce(c45 - t_on <= {L['call_s']}, false) AS call45,
       dur >= {L['logcap_s']} AND row_number() OVER (PARTITION BY dev, ch, run3 ORDER BY t_on) IN (31, 61) AS logcap
FROM r"""),
        # ON >= 60 s together with another input (starts within 5 s)
        ("joint", f"""
WITH q AS (SELECT dev, ch, day, t_on, dur FROM pq_pa WHERE dur >= {L['joint_s']})
SELECT a.dev, a.ch, a.day, b.ch AS other, least(a.dur, b.dur) AS both_s FROM q a JOIN q b
  ON a.dev = b.dev AND a.ch <> b.ch AND abs(a.t_on - b.t_on) <= {L['joint_start_s']}"""),
        # identical ONs: share of an input's ONs >= 1 s (>= 10 of them) whose nearest ON of another input starts and
        # ends within 0.15 s
        ("mirror", f"""
WITH a AS (SELECT dev, ch, t_on, t_off FROM pq_pa WHERE dur >= {L['mirror_on_s']}),
k AS (SELECT dev, ch, count(*) AS n FROM a GROUP BY ALL HAVING count(*) >= {L['mirror_ons']}),
b AS (SELECT dev, ch AS o, t_on, t_off FROM pq_pa WHERE dur IS NOT NULL),
x AS (SELECT a.*, o.o FROM a JOIN k USING (dev, ch) JOIN (SELECT DISTINCT dev, o FROM b) o ON o.dev = a.dev AND o.o <> a.ch),
p AS (SELECT x.*, b.t_on AS pb, b.t_off AS pe FROM x ASOF LEFT JOIN b ON b.dev = x.dev AND b.o = x.o AND x.t_on >= b.t_on),
n AS (SELECT p.*, b.t_on AS nb, b.t_off AS ne FROM p ASOF LEFT JOIN b ON b.dev = p.dev AND b.o = p.o AND p.t_on <= b.t_on),
h AS (SELECT dev, ch, o, CASE WHEN coalesce(t_on - pb, 1e9) <= coalesce(nb - t_on, 1e9) THEN pb ELSE nb END AS mb,
             CASE WHEN coalesce(t_on - pb, 1e9) <= coalesce(nb - t_on, 1e9) THEN pe ELSE ne END AS me, t_on, t_off FROM n),
s AS (SELECT h.dev, h.ch, h.o, count(*) FILTER (abs(mb - t_on) <= {L['mirror_s']} AND abs(me - t_off) <= {L['mirror_s']})
             / any_value(k.n) AS share FROM h JOIN k USING (dev, ch) GROUP BY ALL)
SELECT s.dev, s.ch, s.o, s.share, greatest(s.share, coalesce(t.share, 0)) >= {L['mirror_full']} AS full_pair
FROM s LEFT JOIN s t ON t.dev = s.dev AND t.ch = s.o AND t.o = s.ch"""),
        # ---- findings: kind, dev, ch, day, a / z (period), x (strength), n (count), other (number for the text)
        ("f_stuck", f"""
WITH l AS (SELECT *, coalesce(dur, open_s) AS held FROM pq_pa WHERE NOT logcap AND coalesce(dur, open_s) >= 10),
n AS (SELECT l.dev, l.ch, l.t_on, count(w.t21) AS walks FROM l LEFT JOIN pq_walk w
      ON w.dev = l.dev AND w.ph = l.ph AND w.t21 > l.t_on AND w.t21 < l.t_on + l.held GROUP BY ALL),
-- ONs with a log hole (>= 30 s silent) inside: their length is not known (option log_gaps)
g AS (SELECT DISTINCT l.dev, l.ch, l.t_on, true AS gap FROM l JOIN pq_hole p
      ON p.dev = l.dev AND p.g0 < l.t_on + l.held AND p.g1 > l.t_on)
SELECT 'stuck on' AS kind, l.dev, l.ch, l.day, l.t_on AS a, l.t_on + l.held AS z,
       greatest(l.held / CASE WHEN l.type = 'presence' THEN {L['stuck_presence_s']} ELSE {L['stuck_s']} END,
                CASE WHEN l.type = 'presence' THEN 0 ELSE n.walks / {L['stuck_walks']} END) AS x,
       n.walks AS n, (l.dur IS NULL)::INT AS other, l.held AS held,
       CASE WHEN l.type = 'presence' THEN {L['stuck_presence_s']} ELSE {L['stuck_s']} END AS lim
FROM l JOIN n USING (dev, ch, t_on) LEFT JOIN g USING (dev, ch, t_on)
WHERE CASE WHEN l.dur IS NULL THEN n.walks >= {L['stuck_walks']}
           WHEN coalesce(g.gap, false) THEN l.type <> 'presence' AND n.walks >= {L['stuck_walks']}
                AND NOT EXISTS (SELECT 1 FROM pq_joint j WHERE j.dev = l.dev AND j.ch = l.ch AND j.day = l.day)
           WHEN l.type = 'presence' THEN l.held >= {L['stuck_presence_s']}
           ELSE l.held >= {L['stuck_s']} OR (n.walks >= {L['stuck_walks']}
                AND NOT EXISTS (SELECT 1 FROM pq_joint j WHERE j.dev = l.dev AND j.ch = l.ch AND j.day = l.day))
                OR (l.held >= {L['stuck2_s']} AND n.walks >= {L['stuck2_walks']} AND {on['stuck_2walks']})
           END"""),
        ("f_regular", f"""
WITH g AS (SELECT dev, ch, day, gap, count(*) OVER (PARTITION BY dev, ch, day ORDER BY gap
                  RANGE BETWEEN {L['regular_tol_s']} PRECEDING AND {L['regular_tol_s']} FOLLOWING) AS k,
                  count(*) OVER (PARTITION BY dev, ch, day) AS n
           FROM pq_pa WHERE gap >= {L['regular_gap_s']} AND type <> 'presence'),
r AS (SELECT dev, ch, day, max(k) / any_value(n) AS x, arg_max(gap, k) AS every_s, any_value(n) AS n FROM g GROUP BY ALL
      HAVING any_value(n) >= {L['regular_gaps']} AND max(k) >= {L['regular_share']} * any_value(n))
SELECT 'machine-regular presses' AS kind, r.dev, r.ch, r.day, min(p.t_on) AS a, max(p.t_on) AS z, any_value(r.x) AS x,
       any_value(r.n) AS n, any_value(r.every_s) AS every_s
FROM r JOIN pq_pa p USING (dev, ch, day) GROUP BY ALL"""),
        ("f_never", f"""
SELECT 'never calls a walk' AS kind, p.dev, p.ch, p.day, min(p.t_on) AS a, max(p.t_on) AS z, count(*) AS x, count(*) AS n
FROM pq_pa p JOIN pq_inp i USING (dev, ch)
WHERE i.ph IS NULL AND i.ov IS NULL AND i.type <> 'presence'
  AND NOT EXISTS (SELECT 1 FROM pq_f_regular r WHERE r.dev = p.dev AND r.ch = p.ch AND r.day = p.day)
GROUP BY ALL HAVING count(*) >= {L['no_call']}"""),
        # standing calls (45 outside the walk) of phases that walk: ended by the next walk or the next 45 (dropped)
        ("f_drop", f"""
WITH c AS (SELECT dev, ph, ts AS tc, lead(ts) OVER (PARTITION BY dev, ph ORDER BY ts) AS tn FROM pq_call
           SEMI JOIN (SELECT DISTINCT dev, ph FROM pq_walk) USING (dev, ph)),
p AS (SELECT c.*, w.t21 AS pt21, w.t22 AS pt22 FROM c ASOF LEFT JOIN pq_walk w ON w.dev = c.dev AND w.ph = c.ph AND c.tc >= w.t21),
n AS (SELECT p.*, w.t21 AS tw FROM p ASOF LEFT JOIN pq_walk w ON w.dev = p.dev AND w.ph = p.ph AND p.tc < w.t21),
e1 AS (SELECT n.dev, n.ph, n.tc, coalesce(least(n.tw, n.tn), s.e) AS t_end FROM n
       JOIN {PREFIX}wpart s USING (dev) WHERE NOT coalesce(n.tc < coalesce(n.pt22, n.pt21 + 7), false)),
-- a wait ends at the first log hole (>= 30 s silent) after the call (option log_gaps: a walk lost in it must not
-- read as a wait)
e AS (SELECT e1.dev, e1.ph, e1.tc, least(e1.t_end, coalesce(p.g0, e1.t_end)) AS t_end FROM e1
      ASOF LEFT JOIN pq_hole p ON p.dev = e1.dev AND e1.tc <= p.g0),
g0 AS (SELECT e.*, g.k AS k0, e.t_end - {L['serve_s']} AS te FROM e ASOF LEFT JOIN pq_green g
       ON g.dev = e.dev AND g.ph = e.ph AND e.tc >= g.ts),
g1 AS (SELECT g0.*, g.k AS k1 FROM g0 ASOF LEFT JOIN pq_green g ON g.dev = g0.dev AND g.ph = g0.ph AND g0.te > g.ts),
-- option late_call: the first green after the call is not owed when the call came after the yellow before it began
-- (the signal's last yellow start <= 15 s before that green)
{late}
d0 AS (SELECT dev, ph, tc, t_end, greatest(coalesce(k1, 0) - coalesce(k0, 0) - late, 0) AS greens FROM gy
       WHERE coalesce(k1, 0) - coalesce(k0, 0) - late >= {L['drop_greens']}),
-- any preemption active at any time during the wait (in progress when the call came, or starting before the wait
-- ended; + recovery) excuses the whole wait (user review 2026-10-29): "omitted during preemption"
pf AS (SELECT DISTINCT d0.dev, d0.ph, d0.tc FROM d0 JOIN {PREFIX}pre p
       ON p.dev = d0.dev AND p.ps < d0.t_end AND p.pe >= d0.tc),
d AS (SELECT d0.*, pf.tc IS NOT NULL AS pre FROM d0 LEFT JOIN pf USING (dev, ph, tc)),
b AS (SELECT dev, ph, ch, t_on FROM pq_pa WHERE call45),
w AS (SELECT d.*, b.ch AS by_ch, b.t_on FROM d ASOF LEFT JOIN b ON b.dev = d.dev AND b.ph = d.ph AND d.tc >= b.t_on)
SELECT CASE WHEN w.pre THEN '{NOTE}' ELSE 'call dropped' END AS kind, w.dev, i.ch,
       floor(w.tc / 86400)::BIGINT AS day, w.tc AS a, w.t_end AS z, w.greens AS x, w.greens AS n, w.ph AS other,
       w.dev IN (SELECT dev FROM {PREFIX}prelog) AS pre_log
FROM w JOIN pq_inp i ON i.dev = w.dev AND i.ph = w.ph
     AND (i.ch = w.by_ch OR NOT coalesce(w.tc - w.t_on <= 0.5, false))
WHERE i.type <> 'presence'"""),
        # dead: longest run of 07-19 h logging hours with no waiting group against the input's own daytime level
        ("f_dead", f"""
WITH wg AS (SELECT dev, ch, min(t_on) AS t FROM pq_pa WHERE state IN ('dw', 'nowalk')
            GROUP BY dev, ch, state = 'dw', CASE WHEN state = 'dw' THEN t23 ELSE -run120 END),
cnt AS (SELECT dev, ch, floor(t / 3600)::BIGINT AS h, count(*) AS n FROM wg GROUP BY ALL),
hrs AS (SELECT w.dev, x.h FROM {PREFIX}wpart w, LATERAL (SELECT unnest(range(
          (floor(w.s / 86400) * 24)::BIGINT, (ceil(w.e / 86400) * 24)::BIGINT)) AS h) x
        WHERE x.h % 24 BETWEEN 7 AND 18),
grid AS (SELECT i.dev, i.ch, h.h, h.h // 24 AS day, coalesce(c.n, 0) AS n, a.h IS NOT NULL AS alive
         FROM (SELECT DISTINCT dev, ch FROM wg) i JOIN pq_inp p USING (dev, ch) JOIN hrs h USING (dev)
         LEFT JOIN cnt c ON c.dev = i.dev AND c.ch = i.ch AND c.h = h.h LEFT JOIN pq_alive a ON a.dev = i.dev AND a.h = h.h
         WHERE p.type <> 'presence'),
lvl AS (SELECT dev, ch, sum(n) FILTER (alive) / greatest(count(*) FILTER (alive), 1) AS per_h FROM grid GROUP BY ALL),
runs AS (SELECT dev, ch, day, min(h) AS h0, count(*) AS hours FROM (
           SELECT *, h - row_number() OVER (PARTITION BY dev, ch, day ORDER BY h) AS g FROM grid WHERE alive AND n = 0)
         GROUP BY dev, ch, day, g),
best AS (SELECT * FROM runs QUALIFY row_number() OVER (PARTITION BY dev, ch, day ORDER BY hours DESC, h0) = 1)
SELECT 'dead' AS kind, b.dev, b.ch, b.day, b.h0 * 3600.0 AS a, (b.h0 + b.hours) * 3600.0 AS z,
       b.hours * l.per_h AS x, b.hours AS n, {L['dead_h'] * 3600.0} AS lim
FROM best b JOIN lvl l USING (dev, ch) WHERE b.hours >= {L['dead_h']} AND b.hours * l.per_h >= {L['dead_waits']}"""),
        # ped recall: >= 90 % of the phase's (>= 50) walks come with no ped call since the previous don't walk
        ("recall", f"""
WITH w AS (SELECT dev, ph, t21 + 1 AS tq, coalesce(lag(t23) OVER (PARTITION BY dev, ph ORDER BY t21), -1e18) AS p23 FROM pq_walk),
x AS (SELECT w.*, c.ts FROM w ASOF LEFT JOIN pq_call c ON c.dev = w.dev AND c.ph = w.ph AND w.tq >= c.ts)
SELECT dev, ph FROM x GROUP BY ALL
HAVING count(*) >= {L['recall_walks']} AND avg(CASE WHEN ts > p23 THEN 0 ELSE 1 END) >= {L['recall_share']}"""),
    ]


FINDINGS = ("f_stuck", "f_regular", "f_never", "f_drop", "f_dead")


def _run_group(con, steps, keep: dict) -> dict:
    """run the steps for the signals in pq_wpart; return the tables named in keep (name -> SELECT columns)."""
    names = [n for n, _ in steps]
    try:
        for n, sql in steps:
            con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}{n} AS {sql}")
        return {k: con.execute(f"SELECT {cols} FROM {PREFIX}{k}").df() for k, cols in keep.items()}
    finally:
        for n in names:
            con.execute(f"DROP TABLE IF EXISTS {PREFIX}{n}")


def _sev(x, lo, hi):
    if not x or x <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-math.log(4.0) * math.log(x / lo) / math.log(max(hi / lo, 1.05))))


def _dur(s):
    s = float(s)
    if s < 300:
        return f"{s:.0f} s"
    m = int(round(s / 60))
    return f"{m} min" if m < 60 else f"{m // 60} h" + (f" {m % 60} min" if m % 60 else "")


def _ts(x):
    return pd.to_datetime((pd.Series(x, dtype="float64") * 1000).round().astype("int64"), unit="ms")


def _reason(kind, g, registers, presence, opt=OPT, pre_any=True):
    """plain words for one check's findings of one input (g: its rows).  pre_any: the log holds a preempt event of
    some signal (none: that part of the check could not run)."""
    if kind == "stuck on":
        r = g.loc[g.held.idxmax()]
        end = " (still ON when the log ends)" if r.other else ""
        w = f"{int(r.n)} walk" + "s" * (int(r.n) != 1)
        lim = "limit 10 min for a presence input" if presence else \
            "limit 5 min, 3 walks, or 3 min through 2 walks" if opt["stuck_2walks"] else "limit 5 min or 3 walks"
        return f"stuck ON: held {_dur(r.held)}{end} through {w} ({lim})"
    if kind == "machine-regular presses":
        r = g.loc[g.x.idxmax()]
        calls = "and it places ped calls" if registers else "places no ped call"
        return (f"presses every {r.every_s:.1f} s like a machine ({100 * r.x:.0f} % of gaps the same; true peds stay "
                f"under 22 %), {calls}")
    if kind == "never calls a walk":
        cfg = ": no ped movement seems assigned to it (configuration)" if opt["config_issue"] else ""
        return (f"pressed {int(g.n.sum()):,} times on {len(g)} day{'s' * (len(g) > 1)} but no ped call or walk "
                f"ever followed{cfg}")
    if kind == "call dropped":
        r = g.loc[g.x.idxmax()]
        pre = ""
        if opt["preempt_wait"]:
            pre = (" with no preemption during the wait" if g.pre_log.eq(True).any() else
                   " with no preempt event logged at this signal" if pre_any else
                   " (preemption not checked: the log holds no preempt events)")
        cfg = " (configuration: ped timing or settings)" if opt["config_issue"] else ""
        late = (" (a green that began just after a call placed in the clearance before it not counted)"
                if opt["late_call"] else "")
        return (f"{len(g)} ped call{'s' * (len(g) > 1)} for P{int(r.other)} stood through {int(r.x)} greens without a "
                f"walk{' (the longest)' * (len(g) > 1)}{pre}; limit 2{late}, usually the controller dropped the "
                f"call{cfg}")
    r = g.loc[g.x.idxmax()]
    return (f"no waiting ped for {int(r.n)} h from {_ts([r.a])[0]:%a %H:%M} when ~{r.x:.0f} were expected "
            f"(limit 6 h / 20)")


def _config(inp, mirror, joint, recall, stuck, notes=None, nov=None):
    """config notes per input (dev, ch) -> text; no recall note for a stuck input (its held call brings the walks).
    notes: the "omitted during preemption" rows (not a fault, a note); nov: the "never calls a walk" rows of a log
    without any ped overlap walk (not judged)."""
    out = {}

    def add(k, t):
        out[k] = out.get(k, []) + [t]
    if notes is not None:
        for (dev, ch), g in notes.groupby(["dev", "ch"]):
            add((dev, ch), f"omitted during preemption: {len(g)} ped call{'s' * (len(g) > 1)} for "
                           f"P{int(g.other.iloc[0])} waited up to {int(g.n.max())} greens while a preemption ran "
                           f"(not a fault)")
    if nov is not None:
        for (dev, ch), g in nov.groupby(["dev", "ch"]):
            add((dev, ch), f"never calls a walk not checked: pressed {int(g.n.sum()):,} times on {len(g)} "
                           f"day{'s' * (len(g) > 1)} without a ped call, but the log holds no ped overlap walk "
                           f"events, so a button served by a ped overlap cannot be told apart")
    rec = set(map(tuple, recall[["dev", "ph"]].itertuples(index=False)))
    for r in inp.itertuples(index=False):
        if pd.notna(r.ov):
            add((r.dev, r.ch), f"serves ped overlap {int(r.ov)}: its presses bring that overlap's walk "
                               "(no ped call is logged for an overlap)")
        if pd.notna(r.ph2):
            add((r.dev, r.ch), f"calls P{int(r.ph)} and P{int(r.ph2)} at once")
        if pd.notna(r.ph) and (r.dev, r.ph) in rec and (r.dev, r.ch) not in stuck:
            add((r.dev, r.ch), f"P{int(r.ph)} on ped recall: it walks without presses")
    for r in mirror.sort_values("share", ascending=False).drop_duplicates(["dev", "ch"]).itertuples(index=False):
        if r.full_pair and r.share >= LIM["mirror_share"]:
            add((r.dev, r.ch), f"same button as ped {int(r.o)} (one button wired to two inputs)")
    for r in mirror[(mirror.share >= LIM["mirror_share"]) & ~mirror.full_pair].sort_values(
            "share", ascending=False).drop_duplicates(["dev", "ch"]).itertuples(index=False):
        add((r.dev, r.ch), f"calls two ped phases: {100 * r.share:.0f} % of its presses of 1 s or more also switch "
                           f"ped {int(r.o)}")
    for r in joint.sort_values(["both_s", "other"], ascending=[False, True]).drop_duplicates(
            ["dev", "ch"]).itertuples(index=False):
        add((r.dev, r.ch), f"switches ON together with ped {int(r.other)} (both ON {_dur(r.both_s)})")
    return {k: "; ".join(v) for k, v in out.items()}


def ped_health(con, events_table: str = "events", start=None, end=None, periods: bool = False,
               max_events: int | None = 3_000_000, options: dict | None = None, assign: bool = False):
    """Push-button health of every ped input in `events_table` (DeviceId, Timestamp, EventId, Parameter; local time)
    on an existing DuckDB connection; nothing is copied out.  start / end: the window (end exclusive); default each
    signal's first event to one second past its last.  Returns a DataFrame (COLS: DeviceId, Input, Phase, Type,
    presses, status, reason, config), or with periods=True (inputs, periods) where periods has one row per finding
    (PERIOD_COLS).  Signals are processed in groups of at most `max_events` ped events; the steps run as temp tables
    (prefix pq_), dropped after each group.  options: switches over OPT (regular_bad, config_issue, preempt_wait,
    stuck_2walks, orphan_clearance, log_gaps, late_call).  assign=True also runs the ped movement model
    (ped_assign.add_assignments): + Phases, Overlaps, assign_confidence per input; an input with >= 10 presses whose
    presses call nothing gets a config note and, when it was ok, the status config_issue."""
    rng = [f"Timestamp >= TIMESTAMP '{pd.Timestamp(start)}'"] if start is not None else []
    rng += [f"Timestamp < TIMESTAMP '{pd.Timestamp(end)}'"] if end is not None else []
    where = " AND ".join(["DeviceId IS NOT NULL", "Timestamp IS NOT NULL"] + rng)
    s0 = f"{pd.Timestamp(start).value / 1e9!r}" if start is not None else "w0"
    s1 = f"{pd.Timestamp(end).value / 1e9!r}" if end is not None else "w1"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE {PREFIX}wall AS
        SELECT (dense_rank() OVER (ORDER BY CAST(DeviceId AS VARCHAR)))::INT AS dev, DeviceId, {s0} AS s, {s1} AS e, n
        FROM (SELECT DeviceId, epoch_ns(min(Timestamp)) / 1e9 AS w0, epoch_ns(max(Timestamp)) / 1e9 + 1 AS w1,
                     count(*) FILTER (EventId IN {CODES}) AS n
              FROM {events_table} WHERE {where} GROUP BY DeviceId HAVING count(*) FILTER (EventId IN (89, 90)) > 0)""")
    sig = con.execute(f"SELECT dev, n FROM {PREFIX}wall ORDER BY dev").fetchall()
    groups, cur, size = [], [], 0
    for d, n in sig:
        if cur and max_events is not None and size + n > max_events:
            groups.append(cur)
            cur, size = [], 0
        cur.append(d)
        size += n
    groups += [cur] if cur else []
    opt = {**OPT, **(options or {})}
    steps = _steps(events_table, opt)
    keep = {"inp": "dev, ch, presses, registers, type, ph, ph2, ov", "mirror": "*", "joint": "*", "recall": "*",
            "prelog": "*", "ovlog": "*", **{f: "*" for f in FINDINGS}}
    parts = []
    try:
        for g in groups:
            con.execute(f"""CREATE OR REPLACE TEMP TABLE {PREFIX}wpart AS SELECT dev, DeviceId, s, e,
                make_timestamp((s * 1e6)::BIGINT) AS lo, make_timestamp((e * 1e6)::BIGINT) AS hi
                FROM {PREFIX}wall WHERE dev IN (SELECT unnest(?))""", [list(g)])
            parts.append(_run_group(con, steps, keep))
        devs = con.execute(f"SELECT dev, DeviceId FROM {PREFIX}wall").df()
    finally:
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wpart")
        con.execute(f"DROP TABLE IF EXISTS {PREFIX}wall")
    T = {k: pd.concat([p[k] for p in parts], ignore_index=True) if parts else pd.DataFrame() for k in keep}
    res = _assemble(T, devs, periods, opt)
    if assign:
        from .ped_assign_v2 import add_assignments
        inp = add_assignments(con, res[0] if periods else res, events_table, start, end, options)
        res = (inp, res[1]) if periods else inp
    return res


def _assemble(T: dict, devs: pd.DataFrame, periods: bool, opt: dict = OPT):
    inp = T["inp"]
    if inp.empty:
        out = pd.DataFrame(columns=COLS)
        return (out, pd.DataFrame(columns=PERIOD_COLS)) if periods else out
    # every finding column, also when no finding of a kind that has it was made (each kind has its own columns)
    cols = ["kind", "dev", "ch", "day", "a", "z", "x", "n", "other", "held", "every_s", "lim", "pre_log"]
    parts = [T[f] for f in FINDINGS if not T[f].empty]
    fnd = pd.concat(parts, ignore_index=True).reindex(columns=cols) if parts else pd.DataFrame(columns=cols)
    pre_any, ov_any = not T["prelog"].empty, not T["ovlog"].empty
    inp = inp.sort_values(["dev", "ch"]).reset_index(drop=True)
    reg = dict(zip(zip(inp.dev, inp.ch), inp.registers))
    calls = pd.Series([reg.get(k, False) for k in zip(fnd.dev, fnd.ch)], index=fnd.index, dtype=bool)
    regular = fnd.kind == "machine-regular presses"
    fnd["bad"] = (fnd.kind == "stuck on") | (regular & (calls | opt["regular_bad"]))
    fnd["cfg"] = fnd.kind.isin(["never calls a walk", "call dropped"]) & opt["config_issue"]
    notes, fnd = fnd[fnd.kind == NOTE], fnd[fnd.kind != NOTE]
    # a log without any ped overlap walk (67) cannot tell a button served by an overlap apart (option log_gaps: data
    # loss never makes a finding): never calls a walk is a config note, not a finding
    nov = fnd[fnd.kind == "never calls a walk"] if opt["log_gaps"] and not ov_any else None
    fnd = fnd if nov is None else fnd[fnd.kind != "never calls a walk"]
    stuck = set(zip(fnd.dev[fnd.kind == "stuck on"], fnd.ch[fnd.kind == "stuck on"]))
    cfg = _config(inp, T["mirror"], T["joint"], T["recall"], stuck, notes, nov)
    by = {k: g for k, g in fnd.groupby(["dev", "ch"])}
    rows = []
    for r in inp.itertuples(index=False):
        g = by.get((r.dev, r.ch))
        if g is not None:
            status = "bad" if g.bad.any() else "config_issue" if g.cfg.all() else "suspect"
            reason = "; ".join(_reason(k, g[g.kind == k], r.registers, r.type == "presence", opt, pre_any)
                               for k in
                               ("stuck on", "machine-regular presses", "never calls a walk", "call dropped", "dead")
                               if (g.kind == k).any())
        elif r.presses < LIM["few"]:
            status, reason = "not_enough_data", f"only {r.presses} presses: too few to judge"
        else:
            status, reason = "ok", "no problem found"
        rows.append((r.dev, r.ch, r.ph, r.type, r.presses, status, reason, cfg.get((r.dev, r.ch), "")))
    out = pd.DataFrame(rows, columns=["dev"] + COLS[1:]).merge(devs, on="dev")[COLS]
    out["Phase"] = out.Phase.astype("Int64")
    if not periods:
        return out
    p = pd.concat([fnd, notes], ignore_index=True).merge(devs, on="dev")
    sev = [0.0 if k == NOTE else max(_sev(x, *SEV[k]), 0.8 if b else 0.0)
           for k, x, b in zip(p.kind, p.x.astype(float), p.bad)]
    per = pd.DataFrame({"DeviceId": p.DeviceId, "Detector": p.ch, "start": _ts(p.a), "end": _ts(p.z),
                        "condition": p.kind, "severity": [round(s, 3) for s in sev], "limit_s": p.lim.astype("float64"),
                        "in_closure": False, "whole_window": False})
    per = per.sort_values(["DeviceId", "Detector", "start"], key=lambda s: s.astype(str) if s.name == "DeviceId" else s)
    return out, per.reset_index(drop=True)[PERIOD_COLS]
