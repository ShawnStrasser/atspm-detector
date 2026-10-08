"""Lanes redo, task 1 -- does the LABEL table respect the per-lane rule?

Engineer's rule: a phase has 1-3 lanes (usually 1-2); per LANE at most one Advance, one
Presence, one Count; Yellow_Red may span lanes; the rest is Other.  So the labels imply
a lane lower bound  L_min = max(#Advance, #Presence, #Count)  per (signal, phase), and
only L_min > 3 is a real violation.  A3 counted ">= 2 of a class on a phase" (or on one
radar unit) as a violation, which is the wrong reading.

Unlocked signals only, `?` rows dropped.  Phase = the official timing target.

    python lr1_labels.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import lr_common as C


def per_group(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = d.groupby(keys).func5
    t = pd.DataFrame({c: g.apply(lambda s, c=c: int((s == c).sum()))
                      for c in ["Advance", "Presence", "Count", "Yellow_Red", "Other"]})
    t["n_det"] = g.size()
    t["L_min"] = t[C.APC].max(axis=1)
    return t


def dist(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in s.clip(upper=5).value_counts().sort_index().items()}


def main():
    d = C.labelled_detectors()
    d = d[d.target.notna()]
    res = {}
    for tag, sub in (("all_labelled", d), ("actuating", d[d.real])):
        t = per_group(sub, ["DeviceId", "target"])
        t = t[t[C.APC].sum(axis=1) > 0]
        over = t[t.L_min > 3]
        res[tag] = {
            "n_signals": int(sub.DeviceId.nunique()), "n_phases": int(len(t)),
            "L_min_dist": dist(t.L_min),
            "share_L_min_gt3": round(float((t.L_min > 3).mean()), 4),
            "share_L_min_ge2_old_A3_reading": round(float((t.L_min >= 2).mean()), 4),
            "gt3_driven_by": {c: int((over[c] > 3).sum()) for c in C.APC},
            "gt3_phase_or_overlap": over.index.get_level_values("target").str[0]
                                    .value_counts().to_dict(),
            "per_class_max_dist": {c: dist(t[c]) for c in C.APC},
        }
        C.log(f"{tag}: {json.dumps(res[tag])}")
    t = per_group(d[d.real], ["DeviceId", "target"])
    t[t.L_min > 3].reset_index().to_csv(C.WORK / "lr1_gt3_phases.csv", index=False)

    # ---- the 44-ish signals whose channel text names the lane
    L = d[d.lane.notna()]
    sig = L.DeviceId.unique()
    Ls = d[d.DeviceId.isin(sig)]
    r = {"n_signals": int(len(sig)), "n_lane_tagged_detectors": int(len(L))}
    # (i) true per-lane test: one (signal, phase, lane-token) group
    gl = per_group(L, ["DeviceId", "target", "lane"])
    gl = gl[gl[C.APC].sum(axis=1) > 0]
    r["per_true_lane"] = {
        "n_groups": int(len(gl)),
        "share_any_class_gt1": round(float((gl.L_min > 1).mean()), 4),
        "by_class_gt1": {c: int((gl[c] > 1).sum()) for c in C.APC}}
    # (ii) the per-phase reading on those signals: L_min vs the number of distinct lane
    # tokens on the phase
    gp = per_group(Ls, ["DeviceId", "target"])
    gp["n_tokens"] = Ls.groupby(["DeviceId", "target"]).lane.nunique()
    gp = gp[gp[C.APC].sum(axis=1) > 0]
    r["per_phase"] = {"n_phases": int(len(gp)), "L_min_dist": dist(gp.L_min),
                      "share_L_min_gt3": round(float((gp.L_min > 3).mean()), 4),
                      "share_L_min_ge2": round(float((gp.L_min >= 2).mean()), 4)}
    # (iii) A3's figure: the radar/camera UNIT treated as the lane.  A unit watches an
    # approach, so the right test is per unit <= (#lanes it covers) of each class,
    # #lanes = distinct lane tokens under that unit (or 3 when none is written).
    U = Ls[Ls.unit.notna()]
    gu = per_group(U, ["DeviceId", "target", "unit"])
    gu["n_tokens"] = U.groupby(["DeviceId", "target", "unit"]).lane.nunique()
    gu = gu[(gu[C.APC].sum(axis=1) > 0) & (gu.n_det > 1)]
    cap = gu.n_tokens.where(gu.n_tokens > 0, 3)
    r["per_unit"] = {
        "n_groups": int(len(gu)),
        "old_reading_any_class_gt1": round(float((gu.L_min > 1).mean()), 4),
        "correct_reading_gt_lanes_named": round(float((gu.L_min > cap).mean()), 4),
        "correct_reading_gt3": round(float((gu.L_min > 3).mean()), 4)}
    # the same on ALL unlocked signals with unit text (A3 used 82 signals)
    U2 = d[d.unit.notna()]
    gu2 = per_group(U2, ["DeviceId", "target", "unit"])
    gu2["n_tokens"] = U2.groupby(["DeviceId", "target", "unit"]).lane.nunique()
    gu2 = gu2[(gu2[C.APC].sum(axis=1) > 0) & (gu2.n_det > 1)]
    cap2 = gu2.n_tokens.where(gu2.n_tokens > 0, 3)
    r["per_unit_all_signals"] = {
        "n_signals": int(U2.DeviceId.nunique()), "n_groups": int(len(gu2)),
        "old_reading_any_class_gt1": round(float((gu2.L_min > 1).mean()), 4),
        "correct_reading_gt_lanes_named_or_3": round(float((gu2.L_min > cap2).mean()), 4)}
    res["lane_text_signals"] = r
    C.log(json.dumps(r, indent=1))
    # lane-tagged table for later validation (same shape a cabinet-print table would take)
    ln = (L.groupby(["DeviceId", "target"]).lane.nunique().rename("n_lane_tokens")
          .reset_index())
    ln.to_parquet(C.WORK / "lr1_lane_tokens.parquet", index=False)
    json.dump(res, open(C.WORK / "lr1_labels.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
