"""Classifiability gate: which channels cannot be classified at all, and what to tell the user.

This is NOT the detector health output -- that is `health_core.py` (status / score / reason / bad
periods, notes 38-47), reported next to phase and function.  The gate below only decides when the
phase / function answer is withheld.  Stage 02 found that the only detector pathology that costs
phase accuracy is the absence of usable actuations, so the gate is a small, high-precision set:
no actuations, stuck ON for most of the window, or electrical chatter.  (near_zero_volume is kept
as a reason but does not withhold the answer -- the actuation-count rule in pipeline.py does that.)

Detector fault events (83-88) are not used anywhere (user decision 2026-09-28); the former
`controller_fault_events` flag and the advisory "suspect" rules are gone -- the health output
replaces them.
"""
from __future__ import annotations

import pandas as pd

# Ordered; first match wins, so the reason column is the *primary* reason.
FAILED_RULES = [
    # No actuation in the whole window (no completed ON and no ON left open).  Fold-0 accuracy on these: 6% (mean top
    # prob 0.11).
    ("no_events", lambda d: (d["n_on"].fillna(0) == 0) & ~(_open_on(d) > 0) & (d["frac_time_on"].fillna(0) < 0.90)),
    # Permanently occupied (an ON that never ends counts until the window end): the ON/OFF pattern carries no phase
    # information.  Checked before the volume rule, so a channel stuck ON is reported as stuck, not as quiet (115d).
    ("stuck_on", lambda d: d["frac_time_on"].fillna(0) >= 0.90),
    # Turned ON and never OFF (no completed actuation, an 82 with no 81 after it): stuck ON from that moment (115d).
    ("never_off", lambda d: (d["n_on"].fillna(0) == 0) & (_open_on(d) > 0)),
    # < 20 actuations/day is not a working traffic detector.  Fold-0 accuracy: 21%.
    ("near_zero_volume", lambda d: d["on_per_day"].fillna(0) < 20),
    # >10 Hz toggling for at least one minute: electrical chatter.
    ("chatter_storm", lambda d: d["max_on_per_min"].fillna(0) >= 600),
]

def _open_on(d: pd.DataFrame) -> pd.Series:
    """seconds of the channel's final ON that never ended inside the window (0 if none / not measured)."""
    return d["open_on_s"].fillna(0) if "open_on_s" in d.columns else pd.Series(0.0, index=d.index)


USER_STATUS = {
    "no_events": "cannot classify: detector produced no actuations in the analysis window "
                 "(channel not wired, card removed, or detector failed)",
    "near_zero_volume": "cannot classify: detector produced almost no actuations "
                        "(<20/day) - likely failed",
    "stuck_on": "cannot classify: detector stuck ON for most of the analysis window "
                "- likely failed (shorted loop / card fault)",
    "never_off": "cannot classify: detector stuck ON (turned on and never off in the analysis window) "
                 "- likely failed (shorted loop / card fault)",
    "chatter_storm": "cannot classify: detector chattering (>10 actuations/second) "
                     "- likely failed (open loop / noise)",
}


def flag_detectors(df: pd.DataFrame) -> pd.DataFrame:
    """Add `gate_flag` ('ok' / 'failed') and `gate_reason` from the metric columns pipeline.py builds."""
    df = df.copy()
    flag = pd.Series("ok", index=df.index, dtype=object)
    reason = pd.Series("ok", index=df.index, dtype=object)
    for name, pred in reversed(FAILED_RULES):
        hit = pred(df).fillna(False).to_numpy()
        flag[hit] = "failed"
        reason[hit] = name
    df["gate_flag"] = flag
    df["gate_reason"] = reason
    return df


def status_for_user(reason: str = "") -> str:
    """One-line, end-user facing status string for a withheld detector."""
    return USER_STATUS.get(reason, "cannot classify: detector failed")
