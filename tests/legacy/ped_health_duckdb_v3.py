"""FROZEN oracle (stage 5 of the incremental-health design): ped_health as it stood with the final rules of the 2026-11-02 review, before
the ped judge moved to the store's log; templates ped_judge_v3.sql and ped_windows_v3.sql beside this file.  Do not edit.

Pedestrian push-button health in DuckDB SQL (research notes 123, 129, 155): one row per ped input (push button)
of each signal, from the controller's own ped events.  Separate from the detector classifier and its health: the
classifier never sees ped events; this module reads nothing else.

    ped_health(con, events_table="events", start=None, end=None) -> DataFrame
    ped_health(con, ..., periods=True) -> (inputs, periods)

events_table: DeviceId, Timestamp, EventId, Parameter (the controller log, local time).  Codes used: 89 / 90 ped input
off / on (Parameter = input), 45 ped call (phase), 21 / 22 / 23 walk / ped clearance / don't walk (phase), 67 ped
overlap walk (overlap), 1 begin green (phase), 102 / 104 preempt call input on / off, 105 / 111 preempt entry / begin
exit and 110 max presence exceeded (preempt number), 8 begin yellow (phase); any event of the signal counts as "the
signal was logging".  A log gap: 2 whole clock minutes or more in which the signal logged nothing at all; a log hole:
30 s or more without any event of the signal (a lost block of the log; option log_gaps).

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
  stuck on          bad      (an ON with no OFF seen, or an OFF whose ON came before the window: only when >= 3 walks came and
                             the phase walks at >= 80 % of its greens from that ON, and not already in the 2 h before it, nor still
                             in the 2 h after its OFF.  Without that it is lost data: no finding.  user 2026-11-02)
                             one ON >= 300 s (presence input 600 s) or through >= 3 walks of its phase, or (push
                             inputs, user review 2026-10-29) ON >= 180 s through >= 2 walks of its phase; an ON with
                             no OFF before the log ends counts as ON to the log's end, if >= 3 walks came in it (a
                             lost OFF brings no walks).  Not counted: the log-limit ON
                             (the 31st / 61st press of an unbroken run of presses < 3 s apart, >= 5 s long), and the
                             3-walk rule (not the 180-s / 2-walk one) on a day the input is ON >= 60 s together with
                             another input.  An ON with a log hole inside it is judged on its walks only (>= 3 walks,
                             push inputs): its length is not known.  When every stuck period of the input has 0 walks it stays
                             bad and also carries a configuration note (user review 2026-11-02)
  machine-regular   bad      >= 50 % of its >= 20 press gaps of >= 5 s the same length (+-0.15 s), whether or not
                             the input places ped calls (user review 2026-10-29)
  never calls       config_issue  >= 10 presses that day, no phase, no ped overlap and not machine-regular: no ped
                             movement is assigned to the button.  A log without any ped overlap walk 67 cannot tell
                             an overlap button apart: then a config note "never calls a walk not checked", no finding
                             (option log_gaps)
  call dropped      note (never a status: user review 2026-11-02; one wait reads "possible configuration issue (low
                             confidence)") a ped call (45, logged outside the walk: a call placed during its phase's
                             running walk, to the tick the walk ends, is served) stood through >= 2 green starts of its
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
                             count).  Option late_call: the first green does not count when the call came after
                             the yellow that led into it began (the last yellow 8 of the signal <= 15 s before the
                             green): the controller picks its next phases about when that yellow starts
  long hold         suspect  a push input ON 150-179 s through >= 2 walks of its phase, no log hole inside
  sudden stop       suspect / bad  its longest run of zero presses while the signal logs, against the presses its own level
                             at those times of day expects, allowing for clumps and for day-to-day swings (ped_stop.py):
                             flagged at a chance <= 1 in 1,000 OR zero for >= 24 logged hours at >= 30 presses a day
                             (suspect); bad at <= 1 in 1,000 with bursts capped; run >= 6 logged hours, >= 20 presses
                             in >= 6 separate hours outside it.  Inferred only from the window; no list of buttons
Config notes (status stays ok, user review 2026-10): served by a ped overlap (no ped call; >= 80 % of presses followed
by that overlap's walk within 180 s), phase on ped recall, one button wired to two inputs (>= 95 % identical ONs),
calls two ped phases (>= 20 % of its ONs >= 1 s identical to another input's, or both ON >= 60 s within 5 s), ped
calls omitted during preemption, never calls a walk not checked (a log without any ped overlap walk).
not_enough_data: no finding and fewer than 10 presses in the window.
assign=True (the ped movement model, ped_assign): + Phases, Overlaps, assign_confidence per input; an input with >= 10
presses whose presses call no phase or ped overlap is a config_issue (user ruling 2026-10-29) with a config note (a log
without any ped overlap walk: the note "ped movement not checked" only, option log_gaps); an input the log cannot tie
to a movement because a phase of the signal is in ped recall gets the config note "ped in recall", never a config_issue.
With periods=True also one row per finding: DeviceId, Detector (= Input), start, end, condition, severity (0-1: 0.5
at the check's limit, 0.8 at its "clearly broken" mark; 0 for an "omitted during preemption" note), limit_s (stuck
on), in_closure (always false), whole_window (always false: every finding has its own time).
options: switch the rules (on by default): regular_bad (machine-regular always bad), config_issue (never calls / call dropped -> config_issue, not
suspect), preempt_wait (a preemption during a call wait excuses the whole wait), stuck_2walks (ON >= 180 s through >= 2
walks is stuck), orphan_clearance (a ped clearance 22 or don't walk 23 whose walk 21 is missing from the log counts as
a walk, from the phase's green just before it unless the phase's yellow came in between: then that green's start was
lost too and the walk starts at the 22 / 23 itself: lost data never reads as a call that stood), log_gaps (a log gap or
hole, or a log without any ped overlap walk, never makes a finding: see stuck on, never calls, call dropped), late_call
(see call dropped), stop_test (sudden stop).
The SQL lives in Jinja templates: hq/sql/ped/judge.sql (the steps pq_*), hq/sql/ped/windows.sql (the windows).
"""
from __future__ import annotations

import math

import pandas as pd

from pathlib import Path

from atspm_detector import ped_stop as _stop
from atspm_detector.hq import plan as _plan
from atspm_detector.hq import render as _render
from atspm_detector.hq import run as _run

HERE = Path(__file__).resolve().parent


def _template(name, **ctx):
    return _render.ENV.from_string((HERE / name).read_text(encoding="utf-8")).render(**ctx)


PREFIX = "pq_"
PRE_CODES = (102, 104, 105, 110, 111)
CODES = (1, 8, 21, 22, 23, 45, 67, 89, 90) + PRE_CODES
# user review 2026-10-29 (note 155b): approved, on; all off = the first package pass (note 155)
OPT = {"regular_bad": True, "config_issue": True, "preempt_wait": True, "stuck_2walks": True,
       # data loss never makes a finding: a lost walk start counts as a walk; a log gap ends a wait / an ON's length
       "orphan_clearance": True, "log_gaps": True,
       # a call placed after the yellow before a green began is not owed that green; the sudden-stop test
       "late_call": True, "stop_test": True}
# a preemption (user review 2026-10-29): from 102 / 105 to the first 104 / 110 / 111 of its number after it (Sep 2026:
# 111 missing after 1,676 of 2,425 entries, the 104 is there); none logged: still on, at most 6 h; recovery 120 s as
# the atspm package's phase-wait check (preempt_recovery_seconds)
PRE = {"cap_s": 21600, "recovery_s": 120}
NOTE = "omitted during preemption"
LIM = {"call_s": 0.25, "registers": 0.01, "follow_s": 180, "walk_share": 0.9, "overlap_share": 0.8,
       "second_phase": 0.3, "typed": 30, "stuck_s": 300, "stuck_presence_s": 600, "stuck_walks": 3,
       "stuck2_s": 180, "stuck2_walks": 2, "stuck3_s": 150, "walk_tick_s": 0.15, "recall_share": 0.8, "recall_min_greens": 5,
       "recall_look_s": 7200,
       "logcap_gap_s": 3, "logcap_s": 5,
       "joint_s": 60, "joint_start_s": 5, "mirror_s": 0.15, "mirror_on_s": 1, "mirror_ons": 10, "mirror_share": 0.2,
       "mirror_full": 0.95, "regular_gap_s": 5, "regular_tol_s": 0.15, "regular_gaps": 20, "regular_share": 0.5,
       "no_call": 10, "drop_greens": 2, "serve_s": 10, "wait_gap_s": 120,
       "recall_note_share": 0.9, "recall_walks": 50, "few": 10, "gap_min": 2, "hole_s": 30, "orphan_s": 120,
       "late_s": 15}
# severity: 0.5 at the limit (lo), 0.8 at the "clearly broken" mark (hi), on log strength
SEV = {"stuck on": (0.5, 1.0), "machine-regular presses": (0.5, 0.75), "never calls a walk": (10, 100),
       "call dropped": (2, 20), "long hold": (1.0, 1.5), "sudden stop": (18.5, 36.0)}
COLS = ["DeviceId", "Input", "Phase", "Type", "presses", "status", "reason", "config"]
PERIOD_COLS = ["DeviceId", "Detector", "start", "end", "condition", "severity", "limit_s", "in_closure", "whole_window"]


def _steps(E: str, opt: dict | None = None, carry: bool = False) -> list:
    """the temp tables, in order: (name, SELECT) -- the template hq/sql/ped/judge.sql.  pq_wpart (dev, DeviceId, s, e,
    lo, hi) holds the signals of one group and their windows; times are epoch seconds (DOUBLE), days / hours count
    from the epoch (local time)."""
    ctx = dict(E=E, opt={**OPT, **(opt or {})}, L=LIM, PRE=PRE, NOTE=NOTE, CODES=CODES, PRE_CODES=PRE_CODES, carry=carry,
               STOP=_stop.stop_sql("pq_stopc", "pq_stopl"),
               STOPLIM={"p_suspect": _stop.P_SUSPECT, "min_run_h": _stop.MIN_RUN_H, "min_presses": _stop.MIN_PRESSES,
                        "min_active_h": _stop.MIN_ACTIVE_H})
    return [(n, s) for n, s, _ in _plan.steps(_template("ped_judge_v3.sql", **ctx))]


FINDINGS = ("f_stuck", "f_hold", "f_regular", "f_never", "f_drop", "f_stop")


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
        end = (" (already ON when the log starts)" if r.before is True else "") + (" (still ON when the log ends)" if r.other else "")
        w = f"{int(r.n)} walk" + "s" * (int(r.n) != 1)
        lim = "limit 10 min for a presence input" if presence else \
            "limit 5 min, 3 walks, or 3 min through 2 walks" if opt["stuck_2walks"] else "limit 5 min or 3 walks"
        every = ", the phase walking every cycle from then" if r.other or r.before is True else ""
        return f"stuck ON: held {'at least ' if r.before is True else ''}{_dur(r.held)}{end} through {w}{every} ({lim})"
    if kind == "machine-regular presses":
        r = g.loc[g.x.idxmax()]
        calls = "and it places ped calls" if registers else "places no ped call"
        return (f"presses every {r.every_s:.1f} s like a machine ({100 * r.x:.0f} % of gaps the same; true peds stay "
                f"under 22 %), {calls}")
    if kind == "never calls a walk":
        cfg = ": no ped movement seems assigned to it (configuration)" if opt["config_issue"] else ""
        return (f"pressed {int(g.n.sum()):,} times on {len(g)} day{'s' * (len(g) > 1)} but no ped call or walk "
                f"ever followed{cfg}")
    if kind == "long hold":
        r = g.loc[g.held.idxmax()]
        return f"held ON {r.held:.0f} s through {int(r.n)} walks (suspect at 150-179 s; 180 s is bad)"
    r = g.loc[g.x.idxmax()]
    return stop_reason(r.n, _ts([r.a])[0], r.held, r.every_s, int(r.other), _ts([r.z])[0] if r.ended else None, r.last_ev,
                       bool(r.before))


def stop_cause(last_ev, walks, stuck=False):
    """what the log says about a silent button.  Zero presses with walks is never a fault by itself: stuck ON needs an ON with no OFF
    and the phase walking every cycle from it (the stuck on check); that ON without it is lost data; walks without it, plain recall."""
    w = f"{walks:,} walks came" if walks else "no walks"
    if stuck:
        return "last button event is an ON with no OFF and the phase walks every cycle from it: stuck ON"
    if last_ev == "on":
        return f"last button event is an ON with no OFF, but no walking every cycle from it: lost data, not stuck ({w})"
    if last_ev == "off":
        return f"last button event is an OFF, so not held on; {w}" + (": recall or something else (inferred)" if walks else ": quiet, or a dead button (inferred)")
    return f"{w}; no button event stored from then, so stuck ON and recall cannot be told apart"


def stop_reason(run_h, start, expected, chance, walks, end=None, last_ev=None, before=False, stuck=False):
    """plain words for one sudden stop.  end: the next press (a period that ended has an end time, no more)."""
    when = f"{start:%a %m-%d %H:%M}{' or earlier' if before else ''} to " + (f"{end:%a %m-%d %H:%M}" if end is not None else "the end of the log")
    return (f"no presses for {run_h:.0f} logged hours, {when}; its own level expects ~{expected:.0f} "
            f"(chance about 1 in {1 / max(chance, 1e-12):,.0f} allowing for clumps); {stop_cause(last_ev, walks, stuck)}")


def _wait_note(g, pre_any, opt):
    """the note for one input's call-dropped rows: never a status."""
    r = g.loc[g.x.idxmax()]
    pre = ""
    if opt["preempt_wait"]:
        pre = (" with no preemption during the wait" if g.pre_log.eq(True).any() else
               " with no preempt event logged at this signal" if pre_any else
               " (preemption not checked: the log holds no preempt events)")
    if len(g) == 1:
        return (f"possible configuration issue (low confidence): 1 ped call for P{int(r.other)} waited through "
                f"{int(r.x)} greens without a walk{pre}")
    return f"{len(g)} ped calls for P{int(r.other)} waited, up to {int(r.x)} greens without a walk{pre} (not a fault)"


def _config(inp, mirror, joint, recall, stuck, notes=None, nov=None, waits=None, pre_any=True, opt=OPT, no_walks=()):
    """config notes per input (dev, ch) -> text; no recall note for a stuck input (its held call brings the walks).
    notes: the "omitted during preemption" rows (not a fault, a note); nov: the "never calls a walk" rows of a log
    without any ped overlap walk (not judged); no_walks: inputs whose every stuck period has 0 walks."""
    out = {}

    def add(k, t):
        out[k] = out.get(k, []) + [t]
    if notes is not None:
        for (dev, ch), g in notes.groupby(["dev", "ch"]):
            add((dev, ch), f"omitted during preemption: {len(g)} ped call{'s' * (len(g) > 1)} for "
                           f"P{int(g.other.iloc[0])} waited up to {int(g.n.max())} greens while a preemption ran "
                           f"(not a fault)")
    if waits is not None:
        for (dev, ch), g in waits.groupby(["dev", "ch"]):
            add((dev, ch), _wait_note(g, pre_any, opt))
    if nov is not None:
        for (dev, ch), g in nov.groupby(["dev", "ch"]):
            add((dev, ch), f"never calls a walk not checked: pressed {int(g.n.sum()):,} times on {len(g)} "
                           f"day{'s' * (len(g) > 1)} without a ped call, but the log holds no ped overlap walk "
                           f"events, so a button served by a ped overlap cannot be told apart")
    for k in no_walks:
        add(k, "possible configuration issue: stuck ON through 0 walks, no walk comes from this input")
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
               max_events: int | None = 3_000_000, options: dict | None = None, assign: bool = False, carry=None):
    """Push-button health of every ped input in `events_table` (DeviceId, Timestamp, EventId, Parameter; local time)
    on an existing DuckDB connection; nothing is copied out.  start / end: the window (end exclusive); default each
    signal's first event to one second past its last.  Returns a DataFrame (COLS: DeviceId, Input, Phase, Type,
    presses, status, reason, config), or with periods=True (inputs, periods) where periods has one row per finding
    (PERIOD_COLS).  Signals are processed in groups of at most `max_events` ped events; the steps run as temp tables
    (prefix pq_), dropped after each group.  options: switches over OPT (regular_bad, config_issue, preempt_wait,
    stuck_2walks, orphan_clearance, log_gaps, late_call).  assign=True also runs the ped movement model
    (ped_assign.add_assignments): + Phases, Overlaps, assign_confidence per input; an input with >= 10 presses whose
    presses call nothing gets a config note and, when it was ok, the status config_issue; an input in ped recall gets the
    config note "ped in recall" only."""
    if carry is not None:  # ONs still open from an earlier chunk (ped_stop.open_ons): DeviceId, Input, t_on, Phase
        con.register("pq_carry", carry)
    wall = _plan.steps(_template("ped_windows_v3.sql", E=events_table, start=start, end=end, CODES=CODES, carry=carry is not None))[0][1]
    con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}wall AS {wall}")
    sig = con.execute(f"SELECT dev, n FROM {PREFIX}wall ORDER BY dev").fetchall()
    groups = _run.groups(sig, max_events)
    opt = {**OPT, **(options or {})}
    steps = _steps(events_table, opt, carry is not None)
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
        from atspm_detector.ped_assign import add_assignments
        inp = add_assignments(con, res[0] if periods else res, events_table, start, end, options)
        res = (inp, res[1]) if periods else inp
    return res


def _assemble(T: dict, devs: pd.DataFrame, periods: bool, opt: dict = OPT):
    inp = T["inp"]
    if inp.empty:
        out = pd.DataFrame(columns=COLS)
        return (out, pd.DataFrame(columns=PERIOD_COLS)) if periods else out
    # every finding column, also when no finding of a kind that has it was made (each kind has its own columns)
    cols = ["kind", "dev", "ch", "day", "a", "z", "x", "n", "other", "held", "every_s", "lim", "pre_log", "chance_cap", "ended", "before", "last_ev"]
    parts = [T[f] for f in FINDINGS if not T[f].empty]
    fnd = pd.concat(parts, ignore_index=True).reindex(columns=cols) if parts else pd.DataFrame(columns=cols)
    pre_any, ov_any = not T["prelog"].empty, not T["ovlog"].empty
    inp = inp.sort_values(["dev", "ch"]).reset_index(drop=True)
    reg = dict(zip(zip(inp.dev, inp.ch), inp.registers))
    calls = pd.Series([reg.get(k, False) for k in zip(fnd.dev, fnd.ch)], index=fnd.index, dtype=bool)
    regular = fnd.kind == "machine-regular presses"
    fnd["bad"] = ((fnd.kind == "stuck on") | (regular & (calls | opt["regular_bad"]))
                  | ((fnd.kind == "sudden stop") & (fnd.chance_cap.astype(float) <= _stop.P_BAD)))
    fnd["cfg"] = (fnd.kind == "never calls a walk") & opt["config_issue"]
    # a call that waited is a note, never a status (user review 2026-11-02)
    notes, waits, fnd = fnd[fnd.kind == NOTE], fnd[fnd.kind == "call dropped"], fnd[~fnd.kind.isin([NOTE, "call dropped"])]
    # a log without any ped overlap walk (67) cannot tell a button served by an overlap apart (option log_gaps: data
    # loss never makes a finding): never calls a walk is a config note, not a finding
    nov = fnd[fnd.kind == "never calls a walk"] if opt["log_gaps"] and not ov_any else None
    fnd = fnd if nov is None else fnd[fnd.kind != "never calls a walk"]
    stuck = set(zip(fnd.dev[fnd.kind == "stuck on"], fnd.ch[fnd.kind == "stuck on"]))
    held = fnd[fnd.kind == "stuck on"]
    no_walks = [k for k, n in held.groupby(["dev", "ch"]).n.max().items() if n == 0]
    cfg = _config(inp, T["mirror"], T["joint"], T["recall"], stuck, notes, nov, waits, pre_any, opt, no_walks)
    by = {k: g for k, g in fnd.groupby(["dev", "ch"])}
    rows = []
    for r in inp.itertuples(index=False):
        g = by.get((r.dev, r.ch))
        if g is not None:
            status = "bad" if g.bad.any() else "config_issue" if g.cfg.all() else "suspect"
            reason = "; ".join(_reason(k, g[g.kind == k], r.registers, r.type == "presence", opt, pre_any)
                               for k in
                               ("stuck on", "long hold", "machine-regular presses", "never calls a walk", "sudden stop")
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
    p = pd.concat([fnd, notes, waits], ignore_index=True).merge(devs, on="dev")
    sev = [0.0 if k in (NOTE, "call dropped") else max(_sev(x, *SEV[k]), 0.8 if b else 0.0)
           for k, x, b in zip(p.kind, p.x.astype(float), p.bad)]
    per = pd.DataFrame({"DeviceId": p.DeviceId, "Detector": p.ch, "start": _ts(p.a), "end": _ts(p.z),
                        "condition": p.kind, "severity": [round(s, 3) for s in sev], "limit_s": p.lim.astype("float64"),
                        "in_closure": False, "whole_window": False})
    per = per.sort_values(["DeviceId", "Detector", "start"], key=lambda s: s.astype(str) if s.name == "DeviceId" else s)
    return out, per.reset_index(drop=True)[PERIOD_COLS]
