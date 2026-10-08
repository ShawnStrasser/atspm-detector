"""Lanes from cabinet prints (note 30), step 1 -- lane-label tables.

Source: research/labels/function_labels_v3.parquet (read only).  Rows used: print_confidence
high, tier complete_high / complete_mixed, not unusual_layout, timing target a phase, the
print's diagram phase equal to the timing phase, a vehicle lane (lane_type not bike /
departure / other).  Locked signals are filtered out (and asserted absent).

  nlanes_labels.parquet  (DeviceId, target "P2", n_lanes, n_det_high, all_high)
      n_lanes = the print's n_lanes_phase (total lanes over all approaches).  A phase is
      kept only when every print row on it agrees on n_lanes_phase and no row (any
      confidence) sits in a lane beyond it.  This is the file lr4_decode.py looks for.
  lane_pairs_labels.parquet  (DeviceId, target, da, db, same_lane, span_a/b, f_a/f_b)
      every pair of high rows on the same timing phase of a kept phase; same_lane = the
      lane intervals [lane_index, lane_index + lanes_spanned - 1] overlap (a spanning
      detector is same-lane with every lane it spans).

Validation / training truth for the lane model only -- never a function or phase input.

    python lp1_labels.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import lr_common as C

V3 = C.REPO / "research" / "labels" / "function_labels_v3.parquet"
NONVEH = {"bike", "departure", "other"}


def load() -> pd.DataFrame:
    d = pd.read_parquet(V3)
    d["DeviceId"] = d.DeviceId.str.lower()
    assert not d.DeviceId.isin(C.locked_ids()).any(), "locked signal in v3"
    d = d[d.tier.isin(["complete_high", "complete_mixed"]) & ~d.unusual_layout
          & (d.phase_target_type == "phase") & d.phase_target.notna()].copy()
    d["span"] = d.lanes_spanned.fillna(1).astype(int)
    d["end"] = d.lane_index.astype("float") + d.span - 1
    d["ph_num"] = pd.to_numeric(d.phase_target.str[1:], errors="coerce")
    return d


def main():
    d = load()
    res = {"rows_complete_tiers": int(len(d)), "signals": int(d.DeviceId.nunique())}
    g = d.groupby(["DeviceId", "phase_target"])
    ph = g.agg(n_lanes=("n_lanes_phase", "max"), n_lanes_min=("n_lanes_phase", "min"),
               max_end=("end", "max"),
               n_high=("print_confidence", lambda s: int((s == "high").sum())),
               n_rows=("detector", "size"),
               all_high=("print_confidence", lambda s: bool((s == "high").all())),
               diag_ok=("phase_diagram", "min"))
    ph["diag_mismatch"] = g.apply(
        lambda s: bool(((s.phase_diagram.notna()) & (s.phase_diagram.astype("float")
                                                       != s.ph_num)).any()),
        include_groups=False)
    ph = ph.reset_index()
    keep = (ph.n_lanes.notna() & (ph.n_lanes == ph.n_lanes_min) & (ph.n_lanes >= 1)
            & ~(ph.max_end > ph.n_lanes) & ~ph.diag_mismatch & (ph.n_high >= 1))
    res["phases_with_lane_info"] = int(ph.n_lanes.notna().sum())
    res["phases_dropped"] = {
        "n_lanes_disagree": int((ph.n_lanes != ph.n_lanes_min).sum()),
        "lane_beyond_n_lanes": int((ph.max_end > ph.n_lanes).sum()),
        "diagram_phase_mismatch": int(ph.diag_mismatch.sum())}
    nl = ph[keep][["DeviceId", "phase_target", "n_lanes", "n_high", "n_rows",
                   "all_high"]].rename(columns={"phase_target": "target",
                                                "n_high": "n_det_high"})
    nl["n_lanes"] = nl.n_lanes.astype(int)
    nl.to_parquet(C.WORK / "nlanes_labels.parquet", index=False)
    res["nlanes_table"] = {"phases": int(len(nl)), "signals": int(nl.DeviceId.nunique()),
                           "dist": nl.n_lanes.value_counts().sort_index().to_dict(),
                           "all_rows_high": int(nl.all_high.sum())}

    # ---- pairs: both high, vehicle lane, same timing phase, phase kept
    h = d[(d.print_confidence == "high") & d.lane_index.notna()
          & ~d.lane_type.isin(NONVEH)]
    h = h.merge(nl[["DeviceId", "target"]], left_on=["DeviceId", "phase_target"],
                right_on=["DeviceId", "target"])
    cols = ["DeviceId", "target", "detector", "lane_index", "span", "end", "print_function"]
    a = h[cols]
    P = a.merge(a, on=["DeviceId", "target"], suffixes=("_a", "_b"))
    P = P[P.detector_a < P.detector_b]
    P["same_lane"] = ((P.lane_index_a <= P.end_b) & (P.lane_index_b <= P.end_a)).astype(int)
    P = P.rename(columns={"detector_a": "da", "detector_b": "db", "span_a": "span_a",
                          "print_function_a": "f_a", "print_function_b": "f_b"})
    P = P[["DeviceId", "target", "da", "db", "same_lane", "span_a", "span_b",
           "lane_index_a", "lane_index_b", "f_a", "f_b"]]
    P.to_parquet(C.WORK / "lane_pairs_labels.parquet", index=False)
    res["pairs"] = {"n": int(len(P)), "signals": int(P.DeviceId.nunique()),
                    "phases": int(P.groupby(["DeviceId", "target"]).ngroups),
                    "share_same": round(float(P.same_lane.mean()), 4),
                    "with_spanning": int(((P.span_a > 1) | (P.span_b > 1)).sum())}
    C.log(json.dumps(res, indent=1, default=str))
    json.dump(res, open(C.WORK / "lp1_labels.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
