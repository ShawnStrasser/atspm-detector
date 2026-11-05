# FROZEN ORACLE (stage 0 of the incremental-health design): src/atspm_detector/ped_assign.py as of 2026-10-29 (ped round 5 + ped movement
# model merged).  Test-only; never edited.  Changes against the original: imports and weight paths only.
"""Which phases and ped overlaps each pedestrian input calls, from the controller's own ped events (research note 156).
The ped counterpart of the vehicle phase ranker: one small LightGBM model (weights/ped_assign, onnxruntime) scores every
(ped input, movement) candidate; the features are DuckDB SQL on your connection (nothing is copied out).

    ped_assignments(con, events_table="events", start=None, end=None) -> DataFrame, one row per ped input with presses:
      DeviceId, Input     the ped input (Parameter of 90 / 89)
      Phases, Overlaps    the phases / ped overlaps it calls (sorted lists): every candidate with p >= threshold
                          (manifest, 0.5)
      no_movement         True when no candidate reaches the threshold: the presses call nothing the log can see
                          (a configuration issue when the input is wired, or a spare input with stray presses)
      confidence          probability that the whole set is right (product over the candidates of max(p, 1 - p))
      support             presses in the window;  low_support  support < 10 (too few presses to judge)
      top_movement, top_p the best candidate ('ph4' / 'ovl10') and its probability (also for no_movement inputs)
    return_candidates=True also returns the scored candidate rows (DeviceId, Input, kind, movement, p, features).

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
  walk120        share of presses followed by a walk (21 / 67) on the movement within 120 s
  walk_base      chance of a walk within 120 s of a random time;   walk_lift  ln((walk120 + .01) / (walk_base + .01))
  cov1           share of p's 45 events with a press on d in [-1, +0.2] s (does d explain p's calls)
  walk_cov       share of the movement's walks with a press on d since the previous walk (and <= 120 s before)
  rank_call      rank of p among d's phase candidates by call_lift (1 = best);  margin_call  call1 - best other call1
  rank_walk      rank among all d's candidates by walk_lift;                    margin_walk  walk_lift - best other
  rank_cov       rank of d among the signal's inputs for this movement by cov1 (phase) / walk_cov (overlap)
  n_cand         candidates for d;  same_number  d == movement (a prior: most inputs call the phase of their number);
  is_overlap
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atspm_detector import trees_onnx

PREFIX = "pa_"
CODES = (21, 22, 23, 45, 67, 68, 90)
FEATURES = ['n_press', 'n_calls', 'n_walks', 'call1', 'call3', 'call_base', 'call_lift', 'lag_call_med', 'free_n',
            'free_nocall', 'free_nowalk', 'walk120', 'walk_base', 'walk_lift', 'cov1', 'walk_cov', 'rank_call',
            'margin_call', 'rank_walk', 'margin_walk', 'rank_cov', 'n_cand', 'same_number', 'is_overlap']
COLS = ['DeviceId', 'Input', 'Phases', 'Overlaps', 'no_movement', 'confidence', 'support', 'low_support',
        'top_movement', 'top_p']
LOW_SUPPORT = 10
WEIGHTS = Path(__file__).resolve().parent / "ped_assign_weights_v1"     # the model this oracle was frozen with


def _steps(E: str, where: str) -> list:
    P = PREFIX
    return [
        ("ev", f"""SELECT DeviceId AS dev, epoch_ms(Timestamp) / 1000.0 AS ts, EventId::INT AS ev, Parameter::INT AS p
                   FROM {E} WHERE EventId IN {CODES} AND {where}"""),
        ("dur", f"SELECT dev, greatest(max(ts) - min(ts), 3600) AS dur FROM {P}ev GROUP BY dev"),
        ("press", f"SELECT dev, p AS d, ts FROM {P}ev WHERE ev = 90"),
        ("inp", f"SELECT dev, d, count(*) AS n_press FROM {P}press GROUP BY ALL"),
        ("mev", f"""SELECT dev, CASE WHEN ev IN (67, 68) THEN 'overlap' ELSE 'phase' END AS kind, p AS m, ts,
                          CASE ev WHEN 45 THEN 'call' WHEN 21 THEN 'walk' WHEN 67 THEN 'walk'
                                  WHEN 22 THEN 'clr' WHEN 68 THEN 'clr' ELSE 'dw' END AS typ
                   FROM {P}ev WHERE (ev IN (21, 22, 23, 45) AND p BETWEEN 1 AND 16) OR (ev IN (67, 68) AND p BETWEEN 1 AND 32)"""),
        # state for the "idle" test: one row per instant; same-instant events (a 45 logged with the 21 / 22 / 23)
        # resolved walk > clearance > call > don't walk, so the ASOF match is deterministic
        ("st", f"""SELECT dev, kind, m, ts, arg_max(typ, CASE typ WHEN 'walk' THEN 4 WHEN 'clr' THEN 3 WHEN 'call' THEN 2
                   ELSE 1 END) AS typ FROM {P}mev GROUP BY ALL"""),
        ("mov", f"""SELECT dev, kind, m, count(*) FILTER (typ = 'call') AS n_calls, count(*) FILTER (typ = 'walk') AS n_walks
                   FROM {P}mev GROUP BY ALL HAVING count(*) FILTER (typ IN ('call', 'walk')) > 0"""),
        ("calls", f"SELECT dev, kind, m, ts FROM {P}mev WHERE typ = 'call'"),
        ("walks", f"""SELECT dev, kind, m, ts, ts - lag(ts) OVER (PARTITION BY dev, kind, m ORDER BY ts) AS gap
                      FROM {P}mev WHERE typ = 'walk'"""),
        ("cg", f"""SELECT dev, kind, m, sum(g) AS s FROM (SELECT dev, kind, m,
                    least(1.2, coalesce(ts - lag(ts) OVER (PARTITION BY dev, kind, m ORDER BY ts), 1.2)) AS g FROM {P}calls)
                  GROUP BY ALL"""),
        ("wg", f"SELECT dev, kind, m, sum(least(120, coalesce(gap, 120))) AS s FROM {P}walks GROUP BY ALL"),
        ("base", f"""SELECT o.dev, o.kind, o.m, least(1, coalesce(cg.s, 0) / d.dur) AS call_base,
                      least(1, coalesce(wg.s, 0) / d.dur) AS walk_base
                    FROM {P}mov o JOIN {P}dur d USING (dev) LEFT JOIN {P}cg cg USING (dev, kind, m)
                    LEFT JOIN {P}wg wg USING (dev, kind, m)"""),
        ("pc", f"SELECT p.dev, p.d, o.kind, o.m, p.ts, p.ts - 0.2 AS ts_lo FROM {P}press p JOIN {P}mov o USING (dev)"),
        ("pcx", f"""SELECT pc.dev, pc.d, pc.kind, pc.m, pc.ts,
                     c.ts - pc.ts AS lag_call, w.ts - pc.ts AS lag_walk, s.typ AS last_typ
                   FROM {P}pc pc
                   ASOF LEFT JOIN {P}calls c ON pc.dev = c.dev AND pc.kind = c.kind AND pc.m = c.m AND c.ts >= pc.ts_lo
                   ASOF LEFT JOIN {P}walks w ON pc.dev = w.dev AND pc.kind = w.kind AND pc.m = w.m AND w.ts >= pc.ts
                   ASOF LEFT JOIN {P}st s ON pc.dev = s.dev AND pc.kind = s.kind AND pc.m = s.m AND pc.ts_lo > s.ts"""),
        ("agg", f"""SELECT dev, d, kind, m,
                     avg((coalesce(lag_call, 1e9) <= 1.0)::INT) AS call1,
                     avg((coalesce(lag_call, 1e9) <= 3.0)::INT) AS call3,
                     median(least(coalesce(lag_call, 120), 120)) AS lag_call_med,
                     count(*) FILTER (free) AS free_n,
                     avg((coalesce(lag_call, 1e9) > 3.0)::INT) FILTER (free) AS free_nocall,
                     avg((coalesce(lag_walk, 1e9) > 120)::INT) FILTER (free) AS free_nowalk,
                     avg((coalesce(lag_walk, 1e9) <= 120)::INT) AS walk120
                   FROM (SELECT *, CASE WHEN kind = 'phase' THEN last_typ = 'dw' ELSE last_typ = 'clr' END AS free FROM {P}pcx)
                   GROUP BY ALL"""),
        ("cm", f"SELECT c.dev, i.d, c.kind, c.m, c.ts, c.ts + 0.2 AS ts_hi FROM {P}calls c JOIN {P}inp i USING (dev)"),
        ("cov", f"""SELECT cm.dev, cm.d, cm.kind, cm.m, avg((cm.ts - coalesce(p.ts, -1e12) <= 1.0)::INT) AS cov1
                   FROM {P}cm cm ASOF LEFT JOIN {P}press p ON cm.dev = p.dev AND cm.d = p.d AND cm.ts_hi >= p.ts
                   GROUP BY ALL"""),
        ("wm", f"SELECT w.dev, i.d, w.kind, w.m, w.ts, coalesce(w.gap, 1e9) AS gap FROM {P}walks w JOIN {P}inp i USING (dev)"),
        ("wcov", f"""SELECT wm.dev, wm.d, wm.kind, wm.m,
                      avg((wm.ts - coalesce(p.ts, -1e12) <= least(120, wm.gap))::INT) AS walk_cov
                    FROM {P}wm wm ASOF LEFT JOIN {P}press p ON wm.dev = p.dev AND wm.d = p.d AND wm.ts >= p.ts
                    GROUP BY ALL"""),
        ("f0", f"""SELECT a.dev, a.d, a.kind, a.m, i.n_press, o.n_calls, o.n_walks,
                    CASE WHEN a.kind = 'phase' THEN a.call1 END AS call1,
                    CASE WHEN a.kind = 'phase' THEN a.call3 END AS call3,
                    CASE WHEN a.kind = 'phase' THEN b.call_base END AS call_base,
                    CASE WHEN a.kind = 'phase' THEN ln((a.call1 + .01) / (b.call_base + .01)) END AS call_lift,
                    CASE WHEN a.kind = 'phase' THEN a.lag_call_med END AS lag_call_med,
                    a.free_n, CASE WHEN a.kind = 'phase' THEN a.free_nocall END AS free_nocall, a.free_nowalk,
                    a.walk120, b.walk_base, ln((a.walk120 + .01) / (b.walk_base + .01)) AS walk_lift,
                    CASE WHEN a.kind = 'phase' THEN coalesce(c.cov1, 0) END AS cov1, coalesce(w.walk_cov, 0) AS walk_cov,
                    (a.d = a.m)::INT AS same_number, (a.kind = 'overlap')::INT AS is_overlap
                  FROM {P}agg a JOIN {P}inp i USING (dev, d) JOIN {P}mov o USING (dev, kind, m)
                  JOIN {P}base b USING (dev, kind, m)
                  LEFT JOIN {P}cov c USING (dev, d, kind, m) LEFT JOIN {P}wcov w USING (dev, d, kind, m)"""),
        ("feat", f"""SELECT *,
                    CASE WHEN kind = 'phase' THEN rank() OVER (PARTITION BY dev, d, kind ORDER BY call_lift DESC) END AS rank_call,
                    CASE WHEN kind = 'phase' THEN call1 - coalesce(max(call1) OVER (PARTITION BY dev, d, kind ORDER BY call1 DESC
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW), 0) END AS margin_call,
                    rank() OVER (PARTITION BY dev, d ORDER BY walk_lift DESC) AS rank_walk,
                    walk_lift - coalesce(max(walk_lift) OVER (PARTITION BY dev, d ORDER BY walk_lift DESC
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW), 0) AS margin_walk,
                    rank() OVER (PARTITION BY dev, kind, m ORDER BY CASE WHEN kind = 'phase' THEN cov1 ELSE walk_cov END DESC) AS rank_cov,
                    count(*) OVER (PARTITION BY dev, d) AS n_cand
                  FROM {P}f0"""),
    ]


def ped_features(con, events_table: str = "events", start=None, end=None) -> pd.DataFrame:
    """The candidate rows (DeviceId, Input, kind, movement + FEATURES) for every ped input with presses in
    `events_table` (DeviceId, Timestamp, EventId, Parameter; local time), optionally limited to [start, end).
    An input at a signal with no ped call / walk / ped overlap walk in the window has no candidate rows."""
    return _features(con, events_table, start, end)[0]


def _features(con, events_table, start, end) -> tuple:
    """(ped_features, inputs): inputs = every ped input with presses (DeviceId, Input, n_press), candidates or not"""
    rng = [f"Timestamp >= TIMESTAMP '{pd.Timestamp(start)}'"] if start is not None else []
    rng += [f"Timestamp < TIMESTAMP '{pd.Timestamp(end)}'"] if end is not None else []
    where = " AND ".join(["DeviceId IS NOT NULL", "Timestamp IS NOT NULL"] + rng)
    steps = _steps(events_table, where)
    try:
        for name, sql in steps:
            con.execute(f"CREATE OR REPLACE TEMP TABLE {PREFIX}{name} AS {sql}")
        df = con.execute(f"""SELECT dev AS DeviceId, d AS Input, kind, m AS movement, {', '.join(FEATURES)}
                             FROM {PREFIX}feat ORDER BY ALL""").df()
        inp = con.execute(f"SELECT dev AS DeviceId, d AS Input, n_press FROM {PREFIX}inp ORDER BY ALL").df()
    finally:
        for name, _ in steps:
            con.execute(f"DROP TABLE IF EXISTS {PREFIX}{name}")
    for c in ('Input', 'movement'):
        df[c] = df[c].astype('int32')
    inp["Input"] = inp.Input.astype('int32')
    return df, inp


def _manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / "manifest.json").read_text())


def ped_assignments(con, events_table: str = "events", start=None, end=None, threshold: float | None = None,
                    return_candidates: bool = False, model_dir=None):
    """Phases and ped overlaps called by every ped input with presses in `events_table` (see the module docstring for
    the columns).  An input at a signal whose window holds no ped call / walk / ped overlap walk has no candidate:
    no_movement True, confidence 1, top_movement None.  threshold: decision probability (default: the manifest's,
    0.5)."""
    model_dir = Path(model_dir) if model_dir is not None else WEIGHTS
    man = _manifest(model_dir)
    thr = float(man["threshold"] if threshold is None else threshold)
    C, inputs = _features(con, events_table, start, end)
    if inputs.empty:
        out = pd.DataFrame(columns=COLS)
        return (out, C.assign(p=pd.Series(dtype=float))) if return_candidates else out
    if C.empty:
        C["p"] = pd.Series(dtype=float)
    else:
        model = trees_onnx.load(model_dir / man["files"][0])
        if list(model.feature_names) != FEATURES:
            raise ValueError(f"ped_assign model features {model.feature_names} != {FEATURES}")
        X = C[FEATURES].astype(float).to_numpy(dtype=np.float64, na_value=np.nan)
        C["p"] = model.predict_by(X, C.DeviceId.astype(str).to_numpy())
    C["pred"] = C.p >= thr
    C["label"] = np.where(C.kind == "phase", "ph", "ovl") + C.movement.astype(str)
    C["sure"] = np.maximum(C.p, 1 - C.p)
    groups = dict(iter(C.groupby(["DeviceId", "Input"], sort=False))) if len(C) else {}
    rows = []
    for dev, inp, n in inputs.itertuples(index=False):
        n, g = int(n), groups.get((dev, inp))
        if g is None:       # the window logs no ped movement at this signal: the presses call nothing it can see
            rows.append((dev, int(inp), [], [], True, 1.0, n, n < LOW_SUPPORT, None, np.nan))
            continue
        sel = g[g.pred]
        top = g.loc[g.p.idxmax()]
        ph = sorted(int(m) for m in sel.movement[sel.kind == "phase"])
        ov = sorted(int(m) for m in sel.movement[sel.kind == "overlap"])
        rows.append((dev, int(inp), ph, ov, not (ph or ov), float(np.prod(g.sure)), n, n < LOW_SUPPORT,
                     top.label, float(top.p)))
    out = pd.DataFrame(rows, columns=COLS)
    if return_candidates:
        return out, C.drop(columns=["pred", "label", "sure"])
    return out


NO_MOVEMENT_NOTE = ("its presses call no phase or ped overlap in the log (check this ped detector's call phase / "
                    "call overlap setting)")
NO_MOVEMENT_REASON = "no ped movement: its presses call no phase or ped overlap in the log (configuration)"
NOT_CHECKED_NOTE = ("ped movement not checked: its presses call no phase, but the log holds no ped overlap walk "
                    "events, so a button served by a ped overlap cannot be told apart")


def _assigned(con, events_table, start, end, threshold=None, model_dir=None) -> tuple:
    """(ped_assignments, candidates, overlap_seen) for one window.  overlap_seen: some signal with presses logs a ped
    overlap walk (67) in the window (an overlap is a candidate only then; ped_health's `ovlog` step reads the same)."""
    a, C = ped_assignments(con, events_table, start, end, threshold, return_candidates=True, model_dir=model_dir)
    return a, C, bool(len(C) and (C.kind == "overlap").any())


def _checked(overlap_seen: bool, options: dict | None) -> bool:
    """a 'calls nothing' answer can be trusted: the log shows overlap walks, or option log_gaps is off"""
    from .ped_health_duckdb_v2 import OPT
    return overlap_seen or not {**OPT, **(options or {})}["log_gaps"]


def add_assignments(con, inputs: pd.DataFrame, events_table: str = "events", start=None, end=None,
                    options: dict | None = None, assigned: tuple | None = None) -> pd.DataFrame:
    """ped_health's input table + the model's answer per input: Phases, Overlaps (lists), assign_confidence.  An input
    with >= 10 presses that calls nothing is a configuration issue (user ruling 2026-10-29): the config note
    NO_MOVEMENT_NOTE and, when its status was ok, status config_issue with reason NO_MOVEMENT_REASON (option
    config_issue off: the note only).  A log without any ped overlap walk cannot tell an overlap button apart (option
    log_gaps: lost data never makes a finding): then only the note NOT_CHECKED_NOTE.  Same window as ped_health.
    options: ped_health_duckdb.OPT overrides; assigned: _assigned() of this window, when already computed."""
    from .ped_health_duckdb_v2 import OPT
    opt = {**OPT, **(options or {})}
    a, _, overlap_seen = assigned if assigned is not None else _assigned(con, events_table, start, end)
    checked = _checked(overlap_seen, options)
    a = a[["DeviceId", "Input", "Phases", "Overlaps", "confidence", "no_movement", "low_support"]].rename(
        columns={"confidence": "assign_confidence"})
    out = inputs.merge(a, on=["DeviceId", "Input"], how="left")
    none = out.no_movement.eq(True) & out.low_support.eq(False)
    if "config" in out.columns:
        note = NO_MOVEMENT_NOTE if checked else NOT_CHECKED_NOTE
        cur = out["config"].fillna("").astype(str)
        out.loc[none, "config"] = [f"{c}; {note}" if c else note for c in cur[none]]
    if checked and opt["config_issue"] and "status" in out.columns:
        ok = none & out.status.eq("ok")
        out.loc[ok, "status"] = "config_issue"
        if "reason" in out.columns:
            out.loc[ok, "reason"] = NO_MOVEMENT_REASON
    for c in ("Phases", "Overlaps"):
        out[c] = [v if isinstance(v, list) else [] for v in out[c]]
    return out.drop(columns=["no_movement", "low_support"])


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
    Every ped input with presses has a row.  An input that calls nothing (>= 10 presses; also when the window logs no
    ped movement at its signal at all), or with fewer than 10 presses, has one row without an answer (the best movement,
    if any was scored, in phase_guess / function_guess): status "config issue: ..." or "not enough data: ...",
    review_flag True.  Vehicle-only columns (p_*, lanes, setback, night speed, health_score) are empty.
    health=True also runs ped_health on the same window: health_status / health_reason / health_config are its
    status, reason and config, with the movement model's config_issue (add_assignments).
    options: ped_health_duckdb.OPT overrides; threshold / model_dir: as ped_assignments."""
    from .ped_health_duckdb_v2 import ped_health
    from atspm_detector.pipeline import EXTRA_COLS, OUT_COLS
    # frozen: predict()'s health columns as they stood (the package kept only health_status in stage 6)
    cols = OUT_COLS + EXTRA_COLS
    i = cols.index("health_status")
    cols = cols[:i] + ["health_status", "health_score", "health_reason", "health_watch", "health_categories", "health_config",
                       "health_signal_note"] + cols[i + 1:]
    assigned = _assigned(con, events_table, start, end, threshold, model_dir)
    a, C, overlap_seen = assigned
    if a.empty:
        return _ped_frame(pd.DataFrame(columns=cols), cols)
    checked = _checked(overlap_seen, options)
    thr = float(_manifest(Path(model_dir) if model_dir is not None else WEIGHTS)["threshold"]
                if threshold is None else threshold)
    rng = [f"Timestamp >= TIMESTAMP '{pd.Timestamp(start)}'"] if start is not None else []
    rng += [f"Timestamp < TIMESTAMP '{pd.Timestamp(end)}'"] if end is not None else []
    where = " AND ".join(["DeviceId IS NOT NULL", "Timestamp IS NOT NULL", f"EventId IN {CODES}"] + rng)
    span = con.execute(f"""SELECT DeviceId, (epoch_ns(max(Timestamp)) - epoch_ns(min(Timestamp))) / 6e10 AS m
                           FROM {events_table} WHERE {where} GROUP BY DeviceId""").fetchall()
    span = dict(span)
    C = C.assign(label=np.where(C.kind == "phase", "Ped", "Ped_Overlap"), pick=C.p >= thr)
    groups = dict(iter(C.groupby(["DeviceId", "Input"], sort=False))) if len(C) else {}
    none = C.iloc[:0]
    rows = []
    for dev, inp, n in a[["DeviceId", "Input", "support"]].itertuples(index=False):
        n, g = int(n), groups.get((dev, inp), none)
        g = g.sort_values(["p", "kind", "movement"], ascending=[False, False, True])
        base = {"DeviceId": dev, "Detector": int(inp), "n_actuations": n, "minutes_of_data": round(span[dev], 2),
                "n_candidate_phases": len(g), "function_guess_prob": 1.0}
        if len(g):          # no candidate (no ped movement logged at the signal): no guess either
            top = g.iloc[0]
            base.update(phase_guess=int(top.movement), phase_guess_prob=float(top.p), function_guess=top.label)
        sel, rest = g[g.pick], g[~g.pick]
        if n < LOW_SUPPORT or sel.empty:
            if n < LOW_SUPPORT:
                st, rs = f"not enough data: {n} press{'es' * (n != 1)} in sample (need >= {LOW_SUPPORT})", \
                    "not enough data"
            elif checked:
                st, rs = f"config issue: {NO_MOVEMENT_NOTE}", "no ped movement"
            else:
                st, rs = f"cannot classify: {NOT_CHECKED_NOTE}", "no ped overlap events"
            rows.append({**base, "status": st, "review_flag": True, "review_reason": rs})
            continue
        for r in sel.itertuples(index=False):
            alt = rest[rest.kind == r.kind]
            nxt = alt.iloc[0] if len(alt) else None
            low = r.p < 0.5
            rows.append({**base, "phase_pred": int(r.movement), "phase_prob": float(r.p),
                         "phase_2nd": None if nxt is None else int(nxt.movement),
                         "phase_2nd_prob": np.nan if nxt is None else float(nxt.p),
                         "function_pred": r.label, "function_prob": 1.0, "phase_guess": int(r.movement),
                         "phase_guess_prob": float(r.p), "function_guess": r.label,
                         "status": "ok; low confidence: movement probability below 0.5" if low else "ok",
                         "review_flag": bool(low), "review_reason": "low confidence phase" if low else ""})
    out = pd.DataFrame(rows).reindex(columns=cols)
    out["phase_margin"] = out.phase_prob - out.phase_2nd_prob.fillna(0)
    if health:
        h = ped_health(con, events_table, start, end, options=options)
        h = add_assignments(con, h, events_table, start, end, options, assigned)
        h = h.rename(columns={"Input": "Detector", "status": "health_status", "reason": "health_reason",
                              "config": "health_config"})
        out = out.drop(columns=["health_status", "health_reason", "health_config"]).merge(
            h[["DeviceId", "Detector", "health_status", "health_reason", "health_config"]],
            on=["DeviceId", "Detector"], how="left")
        out[["health_watch", "health_categories", "health_signal_note"]] = ""
    return _ped_frame(out, cols)


def _ped_frame(out: pd.DataFrame, cols: list) -> pd.DataFrame:
    """predict()'s column order and dtypes on a ped table (columns it lacks stay empty)."""
    out = out.reindex(columns=cols)
    for c in ("phase_pred", "phase_2nd", "phase_guess", "n_actuations", "phase_n_lanes", "night_speed_vehicles"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    obj = ("function_pred", "status", "function_guess", "lanes", "setback_confidence", "health_status",
           "health_reason", "health_watch", "health_categories", "health_config", "health_signal_note",
           "review_reason")
    for c in cols:
        if c == "DeviceId":
            continue
        if c in obj:
            out[c] = out[c].astype(object).where(out[c].notna(), None)
        elif out[c].dtype != "Int64":
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["Detector"] = out.Detector.astype("int32")
    out["n_candidate_phases"] = out.n_candidate_phases.astype("int64")
    out["review_flag"] = out.review_flag.astype(bool)
    out = out.sort_values(["DeviceId", "Detector", "function_pred", "phase_pred"],
                          key=lambda s: s.astype(str) if s.name == "DeviceId" else s)
    return out.reset_index(drop=True)
