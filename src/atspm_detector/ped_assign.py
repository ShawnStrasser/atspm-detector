"""Which phases and ped overlaps each pedestrian input calls, from the controller's own ped events (research note 156).
The ped counterpart of the vehicle phase ranker: one small LightGBM model (weights/ped_assign, onnxruntime) scores every
(ped input, movement) candidate; the features are DuckDB SQL on your connection (nothing is copied out; template
hq/sql/ped/assign.sql).

    ped_assignments(con, events_table="events", start=None, end=None) -> DataFrame, one row per ped input with presses:
      DeviceId, Input     the ped input (Parameter of 90 / 89)
      Phases, Overlaps    the phases / ped overlaps it calls (sorted lists): every candidate with p >= threshold
                          (manifest, 0.5)
      no_movement         True when no candidate reaches the threshold and none is in ped recall: the presses call
                          nothing the log can see (a configuration issue when the input is wired, or a spare input)
      in_recall           True when no candidate reaches the threshold and the signal has a movement in ped recall that
                          this button shows no calls on: the movement walks whether anyone pushes or not, so the log
                          cannot show which one the button belongs to.  No answer, no guess, not a configuration issue.
      confidence          probability that the whole set is right (product over the scored candidates of max(p, 1 - p));
                          empty for an in_recall input
      support             presses in the window;  low_support  too little to judge: support < 10, or no answer and a
                          phase here walked too few times to tell whether it is in ped recall (_thin)
      top_movement, top_p the best scored candidate ('ph4' / 'ovl10') and its probability (also for no_movement inputs);
                          when two candidates tie for best, top_movement is empty: no number may break a tie
    return_candidates=True also returns the candidate rows (DeviceId, Input, kind, movement, p, features, walk_nocall,
      gated, thin); a gated row (ped recall, no calls from this button) has no p.

No number reaches the model: no input, phase or overlap number is a feature, no tie is broken by one, and the base rates
are summed in whole milliseconds so that a rank cannot move with the row order.  tests/test_ped_invariance.py renumbers
the raw log and requires every value to follow.  Phase 1-16 and overlap 1-32 are a valid-range filter only.

    ped_detectors(con, events_table="events", start=None, end=None, health=True) -> the same answers in predict()'s
      columns and dtypes, one row per (ped input, movement it calls): function_pred "Ped" / "Ped_Overlap",
      phase_pred the phase / ped overlap; health_* from ped_health (see ped_detectors).

Candidates: every ped input with presses x every phase 1-16 with a ped call or walk (45 / 21) and every ped overlap 1-32
with an overlap walk / clearance (67 / 68) at that signal in the window.  A configured movement that never shows ped
activity in the window (e.g. "phase 2 + overlap 2" where overlap 2 never times a walk) cannot be seen in the log.
Codes (Indiana hi-res): 90 / 89 ped input on / off (Parameter = input); 45 ped call registered, 21 / 22 / 23 walk /
ped clearance / solid don't walk (Parameter = phase); 67 / 68 ped overlap walk / clearance (Parameter = overlap).
Overlaps have no ped-call code, so the call features are NULL for overlap candidates.

Features (times in seconds; "press" = a 90 on input d; windows from the press):
  n_press        presses of d (support)                    n_calls / n_walks  45 / walk events of the movement
  call1, call3   share of presses with a 45 on p in [-0.2, 1] / [-0.2, 3] s
  call_base      chance of a 45 on p in a random 1.2 s window (time covered / logged time)
  call_lift      ln((call1 + .01) / (call_base + .01))
  lag_call_med   median press -> next 45 on p, capped at 120 (no 45 within 120 s counts 120)
  free_n         presses while the movement is idle: phase last state 23 (don't walk, no pending call), overlap last 68
  free_nocall    share of free presses with no 45 on p within 3 s (evidence against; phase only)
  free_nowalk    share of free presses with no walk on the movement within 120 s (evidence against)
  served         share of presses followed by a walk (21 / 67) on the movement within 120 s, or landing while its walk
                 is already on
  walk_base      chance of a walk within 120 s of a random time;   served_lift  ln((served + .01) / (walk_base + .01))
  cov1           share of p's 45 events with a press on d in [-1, +0.2] s (does d explain p's calls)
  walk_cov       share of the movement's walks with a press on d since the previous walk (and <= 120 s before)
  rank_call      rank of p among d's phase candidates by call_lift (1 = best);  margin_call  call1 - best other call1
  rank_served    rank among all d's candidates by served_lift;                  margin_served  served_lift - best other
  rank_cov       rank of d among the signal's inputs for this movement by cov1 (phase) / walk_cov (overlap)
  n_cand         candidates for d;  is_overlap
Ped recall (read by the rule _gated, not by the model):
  walk_nocall    share of the phase's walks with no ped call (45) on it since its previous walk; does not depend on
                 press traffic.  Phases only: an overlap has no call code, its value is 0 and it is never gated
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import trees_onnx
from .hq import plan as _plan
from .hq import render as _render

PREFIX = "pa_"
CODES = (21, 22, 23, 45, 67, 68, 90)
FEATURES = ['n_press', 'n_calls', 'n_walks', 'call1', 'call3', 'call_base', 'call_lift', 'lag_call_med', 'free_n',
            'free_nocall', 'free_nowalk', 'served', 'walk_base', 'served_lift', 'cov1', 'walk_cov', 'rank_call',
            'margin_call', 'rank_served', 'margin_served', 'rank_cov', 'n_cand', 'is_overlap']
EXTRA = ['walk_nocall']    # read by the recall rule, never by the model
COLS = ['DeviceId', 'Input', 'Phases', 'Overlaps', 'no_movement', 'in_recall', 'confidence', 'support', 'low_support',
        'top_movement', 'top_p']
LOW_SUPPORT = 10
# ped recall (user ruling; measured on Sep + Oct 2026, note 156): a phase in recall walks with no ped call since its
# previous walk at least half of the time (>= 5 walks; fewer: too few to tell, _thin); a button's own evidence on it is a
# ped call on >= 5 % of its presses and >= half of the phase's calls.  37 of 37 recall rows with such evidence are
# configured links.
RECALL_SHARE = 0.5
RECALL_MIN_WALKS = 5
EV_CALL1 = 0.05
EV_COV1 = 0.5
WEIGHTS = Path(__file__).resolve().parent / "weights" / "ped_assign"


def _steps(E: str, start=None, end=None) -> list:
    """the feature steps, in order: (name, SELECT) -- the template hq/sql/ped/assign.sql (temp tables pa_*)."""
    return [(n, q) for n, q, _ in _plan.steps(_render.render("ped/assign.sql", E=E, start=start, end=end, CODES=CODES))]


def ped_features(con, events_table: str = "events", start=None, end=None) -> pd.DataFrame:
    """The candidate rows (DeviceId, Input, kind, movement + FEATURES) for every ped input with presses in
    `events_table` (DeviceId, Timestamp, EventId, Parameter; local time), optionally limited to [start, end).
    An input at a signal with no ped call / walk / ped overlap walk in the window has no candidate rows."""
    return _features(con, events_table, start, end)[0]


def _features(con, events_table, start, end) -> tuple:
    """(ped_features, inputs): inputs = every ped input with presses (DeviceId, Input, n_press), candidates or not"""
    steps = _steps(events_table, start, end)
    try:
        for name, sql in steps:
            con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}{name} AS {sql}")
        df = con.execute(f"""SELECT dev AS DeviceId, d AS Input, kind, m AS movement, {', '.join(FEATURES + EXTRA)}
                             FROM {PREFIX}feat ORDER BY ALL""").df()
        inp = con.execute(f"SELECT dev AS DeviceId, d AS Input, n_press FROM {PREFIX}inp ORDER BY ALL").df()
    finally:
        for name, _ in steps:
            con.execute(f"DROP TABLE IF EXISTS {PREFIX}{name}")
    for c in ('Input', 'movement'):
        df[c] = df[c].astype('int32')
    inp["Input"] = inp.Input.astype('int32')
    return df, inp


def _gated(C: pd.DataFrame) -> pd.Series:
    """True: the movement is in ped recall and this button shows no calls of its own on it.  Behaviour only; a gated row is
    never predicted, whatever the model would say."""
    recall = (C.walk_nocall >= RECALL_SHARE) & (C.n_walks >= RECALL_MIN_WALKS)
    return recall & ~_own_calls(C)


def _own_calls(C: pd.DataFrame) -> pd.Series:
    return (C.call1.fillna(0) >= EV_CALL1) & (C.cov1.fillna(0) >= EV_COV1)


def _thin(C: pd.DataFrame) -> pd.Series:
    """True: a phase that walked too few times in the window to tell whether it is in ped recall, its walks not shown to
    be called, and no calls from this button.  A button left without an answer next to one is not enough data, never a
    configuration issue."""
    few = (C.is_overlap == 0) & (C.n_walks > 0) & (C.n_walks < RECALL_MIN_WALKS)
    return few & ((C.walk_nocall >= RECALL_SHARE) | (C.n_calls == 0)) & ~_own_calls(C)


def _untied(g: pd.DataFrame):
    """the first row of g (sorted by p, best first), None when the next row ties it: no number may break a tie"""
    if g.empty or (len(g) > 1 and g.p.iloc[0] == g.p.iloc[1]):
        return None
    return g.iloc[0]


def _manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / "manifest.json").read_text())


def ped_assignments(con, events_table: str = "events", start=None, end=None, threshold: float | None = None,
                    return_candidates: bool = False, model_dir=None):
    """Phases and ped overlaps called by every ped input with presses in `events_table` (see the module docstring for
    the columns).  An input at a signal whose window holds no ped call / walk / ped overlap walk has no candidate:
    no_movement True, confidence 1, top_movement None.  threshold: decision probability (default: the manifest's,
    0.5); the ped recall rule (_gated) comes first."""
    model_dir = Path(model_dir) if model_dir is not None else WEIGHTS
    man = _manifest(model_dir)
    thr = float(man["threshold"] if threshold is None else threshold)
    C, inputs = _features(con, events_table, start, end)
    if inputs.empty:
        out = pd.DataFrame(columns=COLS)
        return (out, C.assign(p=pd.Series(dtype=float))) if return_candidates else out
    C["gated"] = _gated(C) if len(C) else pd.Series(dtype=bool)
    C["thin"] = _thin(C) if len(C) else pd.Series(dtype=bool)
    if C.empty:
        C["p"] = pd.Series(dtype=float)
    else:
        model = trees_onnx.load(model_dir / man["files"][0])
        if list(model.feature_names) != FEATURES:
            raise ValueError(f"ped_assign model features {model.feature_names} != {FEATURES}")
        X = C[FEATURES].astype(float).to_numpy(dtype=np.float64, na_value=np.nan)
        C["p"] = model.predict_by(X, C.DeviceId.astype(str).to_numpy())
        C.loc[C.gated, "p"] = np.nan
    C["pred"] = C.p >= thr
    C["label"] = np.where(C.kind == "phase", "ph", "ovl") + C.movement.astype(str)
    C["sure"] = np.maximum(C.p, 1 - C.p)
    groups = dict(iter(C.groupby(["DeviceId", "Input"], sort=False))) if len(C) else {}
    rows = []
    for dev, inp, n in inputs.itertuples(index=False):
        n, g = int(n), groups.get((dev, inp))
        if g is None:       # the window logs no ped movement at this signal: the presses call nothing it can see
            rows.append((dev, int(inp), [], [], True, False, 1.0, n, n < LOW_SUPPORT, None, np.nan))
            continue
        sel = g[g.pred]
        scored = g[~g.gated].sort_values("p", ascending=False)
        ph = sorted(int(m) for m in sel.movement[sel.kind == "phase"])
        ov = sorted(int(m) for m in sel.movement[sel.kind == "overlap"])
        rec = not (ph or ov) and bool(g.gated.any())
        top = None if rec else _untied(scored)
        low = n < LOW_SUPPORT or (not (ph or ov) and not rec and bool(g.thin.any()))
        rows.append((dev, int(inp), ph, ov, not (ph or ov) and not rec, rec,
                     np.nan if rec else float(np.prod(np.sort(scored.sure.to_numpy()))), n, low,
                     None if top is None else top.label,
                     np.nan if rec or scored.empty else float(scored.p.iloc[0])))
    out = pd.DataFrame(rows, columns=COLS)
    if return_candidates:
        return out, C.drop(columns=["pred", "label", "sure"])
    return out


NO_MOVEMENT_NOTE = ("its presses call no phase or ped overlap in the log (check this ped detector's call phase / "
                    "call overlap setting)")
NO_MOVEMENT_REASON = "no ped movement: its presses call no phase or ped overlap in the log (configuration)"
RECALL_NOTE = "ped in recall"
THIN_NOTE = "a phase here walked too few times in this window to tell whether it is in ped recall"
NOT_CHECKED_NOTE = ("ped movement not checked: its presses call no phase, but the log holds no ped overlap walk "
                    "events, so a button served by a ped overlap cannot be told apart")


def _assigned(con, events_table, start, end, threshold=None, model_dir=None) -> tuple:
    """(ped_assignments, candidates, overlap_seen) for one window.  overlap_seen: some signal with presses logs a ped
    overlap walk (67) in the window (an overlap is a candidate only then; ped_health's `ovlog` step reads the same)."""
    a, C = ped_assignments(con, events_table, start, end, threshold, return_candidates=True, model_dir=model_dir)
    return a, C, bool(len(C) and (C.kind == "overlap").any())


def _checked(overlap_seen: bool, options: dict | None) -> bool:
    """a 'calls nothing' answer can be trusted: the log shows overlap walks, or option log_gaps is off"""
    from .ped_health_duckdb import OPT
    return overlap_seen or not {**OPT, **(options or {})}["log_gaps"]


def add_assignments(con, inputs: pd.DataFrame, events_table: str = "events", start=None, end=None,
                    options: dict | None = None, assigned: tuple | None = None) -> pd.DataFrame:
    """ped_health's input table + the model's answer per input: Phases, Overlaps (lists), assign_confidence.  An input
    with >= 10 presses that calls nothing is a configuration issue (user ruling 2026-10-29): the config note
    NO_MOVEMENT_NOTE and, when its status was ok, status config_issue with reason NO_MOVEMENT_REASON (option
    config_issue off: the note only).  A log without any ped overlap walk cannot tell an overlap button apart (option
    log_gaps: lost data never makes a finding): then only the note NOT_CHECKED_NOTE.  An input in ped recall (the log
    cannot show which phase it belongs to) gets the config note RECALL_NOTE: never a config_issue, never a guess (user
    ruling 2026-11-02).  Same window as ped_health.
    options: ped_health_duckdb.OPT overrides; assigned: _assigned() of this window, when already computed."""
    from .ped_health_duckdb import OPT
    opt = {**OPT, **(options or {})}
    a, _, overlap_seen = assigned if assigned is not None else _assigned(con, events_table, start, end)
    checked = _checked(overlap_seen, options)
    a = a[["DeviceId", "Input", "Phases", "Overlaps", "confidence", "no_movement", "in_recall", "low_support"]].rename(
        columns={"confidence": "assign_confidence"})
    out = inputs.merge(a, on=["DeviceId", "Input"], how="left")
    none = out.no_movement.eq(True) & out.low_support.eq(False)
    if "config" in out.columns:
        note = NO_MOVEMENT_NOTE if checked else NOT_CHECKED_NOTE
        cur = out["config"].fillna("").astype(str)
        out.loc[none, "config"] = [f"{c}; {note}" if c else note for c in cur[none]]
        rec = out.in_recall.eq(True)
        cur = out["config"].fillna("").astype(str)
        out.loc[rec, "config"] = [f"{c}; {RECALL_NOTE}" if c else RECALL_NOTE for c in cur[rec]]
    if checked and opt["config_issue"] and "status" in out.columns:
        ok = none & out.status.eq("ok")
        out.loc[ok, "status"] = "config_issue"
        if "reason" in out.columns:
            out.loc[ok, "reason"] = NO_MOVEMENT_REASON
    if "status" in out.columns:
        rc = out.in_recall.eq(True) & out.status.eq("config_issue")
        out.loc[rc, "status"] = "ok"
        if "reason" in out.columns:
            out.loc[rc, "reason"] = "no problem found"
    for c in ("Phases", "Overlaps"):
        out[c] = [v if isinstance(v, list) else [] for v in out[c]]
    return out.drop(columns=["no_movement", "in_recall", "low_support"])


def ped_detectors(con, events_table: str = "events", start=None, end=None, health: bool = True,
                  options: dict | None = None, threshold: float | None = None, model_dir=None) -> pd.DataFrame:
    """The ped inputs in the vehicle classification schema: exactly predict()'s columns and dtypes (OUT_COLS +
    EXTRA_COLS), so `pd.concat([predict(...), ped_detectors(con, "events")])` is one detector table.  One row per
    (ped input, movement it calls), as atspm's detector_config has one row per (detector, phase):
      Detector        the ped input (Parameter of 90 / 89; its number may equal a vehicle detector's channel)
      function_pred   "Ped" (phase_pred = the phase it calls) or "Ped_Overlap" (phase_pred = the ped overlap it calls)
      phase_prob      the model's probability for that movement; phase_2nd / phase_2nd_prob: the best movement of
                      the same kind (phase / ped overlap) it does not call; phase_margin = phase_prob - phase_2nd_prob
      function_prob   1 (a ped input is known from its codes 89 / 90)
      n_actuations    presses; minutes_of_data: the signal's ped log span; n_candidate_phases: movements scored
    Every ped input with presses has a row.  An input in ped recall has one row without an answer and without a guess:
    status "ped in recall: ...", review_flag False.  An input that calls nothing (>= 10 presses; also when the window logs no
    ped movement at its signal at all), or with fewer than 10 presses, has one row without an answer (the best movement,
    if any was scored, in phase_guess / function_guess): status "config issue: ..." or "not enough data: ...",
    review_flag True.  Vehicle-only columns (p_*, lanes, setback, night speed) are empty.
    health=True also runs the ped health on the same window: health_status is its status, with the movement model's
    config_issue (add_assignments); ped_health(..., assign=True) gives the findings.
    options: ped_health_duckdb.OPT overrides; threshold / model_dir: as ped_assignments."""
    from .ped_health_duckdb import ped_health
    from .pipeline import EXTRA_COLS, OUT_COLS
    cols = OUT_COLS + EXTRA_COLS
    assigned = _assigned(con, events_table, start, end, threshold, model_dir)
    a, C, overlap_seen = assigned
    if a.empty:
        return _ped_frame(pd.DataFrame(columns=cols), cols)
    in_recall = set(zip(a.DeviceId[a.in_recall], a.Input[a.in_recall]))
    thin = set(zip(a.DeviceId[a.low_support], a.Input[a.low_support]))
    checked = _checked(overlap_seen, options)
    thr = float(_manifest(Path(model_dir) if model_dir is not None else WEIGHTS)["threshold"]
                if threshold is None else threshold)
    span = con.execute(_plan.steps(_render.render("ped/span.sql", E=events_table, start=start, end=end,
                                                  CODES=CODES))[0][1]).fetchall()
    span = dict(span)
    C = C.assign(label=np.where(C.kind == "phase", "Ped", "Ped_Overlap"), pick=C.p >= thr)
    groups = dict(iter(C.groupby(["DeviceId", "Input"], sort=False))) if len(C) else {}
    none = C.iloc[:0]
    rows = []
    for dev, inp, n in a[["DeviceId", "Input", "support"]].itertuples(index=False):
        n, g = int(n), groups.get((dev, inp), none)
        ncand = len(g)
        g = g[~g.gated].sort_values("p", ascending=False)
        base = {"DeviceId": dev, "Detector": int(inp), "n_actuations": n, "minutes_of_data": round(span[dev], 2),
                "n_candidate_phases": ncand, "function_guess_prob": 1.0}
        if (dev, inp) in in_recall:     # the log cannot show which movement: no answer and no guess
            rows.append({**base, "status": f"{RECALL_NOTE}: a phase here walks whether anyone pushes or not, so the log "
                                           "cannot show which phase this button calls",
                         "review_flag": False, "review_reason": RECALL_NOTE})
            continue
        top = _untied(g)    # a tie (or no candidate: no ped movement logged at the signal): no guess
        if top is not None:
            base.update(phase_guess=int(top.movement), phase_guess_prob=float(top.p), function_guess=top.label)
        sel, rest = g[g.pick], g[~g.pick]
        if n < LOW_SUPPORT or sel.empty:
            if n < LOW_SUPPORT:
                st, rs = f"not enough data: {n} press{'es' * (n != 1)} in sample (need >= {LOW_SUPPORT})", \
                    "not enough data"
            elif (dev, inp) in thin:
                st, rs = f"not enough data: {THIN_NOTE}", "not enough data"
            elif checked:
                st, rs = f"config issue: {NO_MOVEMENT_NOTE}", "no ped movement"
            else:
                st, rs = f"cannot classify: {NOT_CHECKED_NOTE}", "no ped overlap events"
            rows.append({**base, "status": st, "review_flag": True, "review_reason": rs})
            continue
        for r in sel.itertuples(index=False):
            alt = rest[rest.kind == r.kind]
            nxt = _untied(alt)
            low = r.p < 0.5
            rows.append({**base, "phase_pred": int(r.movement), "phase_prob": float(r.p),
                         "phase_2nd": None if nxt is None else int(nxt.movement),
                         "phase_2nd_prob": float(alt.p.iloc[0]) if len(alt) else np.nan,
                         "function_pred": r.label, "function_prob": 1.0, "phase_guess": int(r.movement),
                         "phase_guess_prob": float(r.p), "function_guess": r.label,
                         "status": "ok; low confidence: movement probability below 0.5" if low else "ok",
                         "review_flag": bool(low), "review_reason": "low confidence phase" if low else ""})
    out = pd.DataFrame(rows).reindex(columns=cols)
    out["phase_margin"] = out.phase_prob - out.phase_2nd_prob.fillna(0)
    if health:
        h = ped_health(con, events_table, start, end, options=options)
        h = add_assignments(con, h, events_table, start, end, options, assigned)
        h = h.rename(columns={"Input": "Detector", "status": "health_status"})
        out = out.drop(columns=["health_status"]).merge(h[["DeviceId", "Detector", "health_status"]],
                                                        on=["DeviceId", "Detector"], how="left")
    return _ped_frame(out, cols)


def _ped_frame(out: pd.DataFrame, cols: list) -> pd.DataFrame:
    """predict()'s column order and dtypes on a ped table (columns it lacks stay empty)."""
    out = out.reindex(columns=cols)
    for c in ("phase_pred", "phase_2nd", "phase_guess", "n_actuations", "phase_n_lanes", "night_speed_vehicles"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    obj = ("function_pred", "status", "function_guess", "lanes", "setback_confidence", "health_status",
           "health_reason", "health_watch", "health_categories", "health_config", "health_signal_note",
           "review_reason")
    text = ("function_pred", "status", "function_guess", "lanes", "setback_confidence", "review_reason")
    str_type = pd.Series(["a"]).dtype       # object on pandas 2, str on pandas 3, as in predict()
    for c in cols:
        if c == "DeviceId":
            continue
        if c in obj:
            out[c] = out[c].astype(object).where(out[c].notna(), None)
            if c in text:
                out[c] = out[c].astype(str_type)
        elif out[c].dtype != "Int64":
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["Detector"] = out.Detector.astype("int32")
    out["n_candidate_phases"] = out.n_candidate_phases.astype("int64")
    out["review_flag"] = out.review_flag.astype(bool)
    out = out.sort_values(["DeviceId", "Detector", "function_pred", "phase_pred"],
                          key=lambda s: s.astype(str) if s.name == "DeviceId" else s)
    return out.reset_index(drop=True)
