"""Note 51 step 3: dump ~10 example detectors per top confusion pair (realistic set) with behaviour stats on the
longest window, the same-phase context (labelled phase siblings: truth, modal prediction, stats) and label source.

    python err51_examples.py > %DC_WORK%/trackA/err51/examples.txt
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

D = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), "trackA/err51/")
PAIRS = [("Other", "Advance"), ("Other", "Presence"), ("Yellow_Red", "Count"), ("Advance", "Other"),
         ("Advance", "Presence"), ("Presence", "Advance"), ("Count", "Presence")]
ORDER = ["full", "h24", "h6", "h3", "h1", "m30", "m10", "m5"]
ST = ["det_on_per_hour", "det_occ_frac", "det_dur_med", "px_pulse_frac", "px_occ_red_all", "px_span_to_green",
      "px_g_first_med", "on_lift_green", "yr_hit_yr", "lagsib_best_lag", "lagsib_best_coinc"]
pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_colwidth", 38)


def main():
    fr = pd.read_parquet(D + "rows.parquet")
    df = fr[fr.setR].copy()
    df["wo"] = df.wgroup.map({g: i for i, g in enumerate(ORDER)})
    det = df.groupby(["DeviceId", "Detector"])
    modal = det.pred.agg(lambda s: s.value_counts().index[0]).rename("pred_modal")
    accd = det.ok.mean().rename("det_acc")
    best = df.sort_values(["wo", "win"]).drop_duplicates(["DeviceId", "Detector"]).set_index(["DeviceId", "Detector"])
    best = best.join(modal).join(accd)
    rng = np.random.default_rng(51)
    for a, b in PAIRS:
        cand = best[(best.truth == a) & (best.pred_modal == b)]
        pick = cand.iloc[rng.choice(len(cand), min(10, len(cand)), replace=False)] if len(cand) else cand
        print(f"\n\n################ {a} -> {b}: {len(cand)} detectors with this modal error; showing {len(pick)}")
        for (sid, d), r in pick.iterrows():
            print(f"\n## {r.DeviceName} d{d} truth {r.truth} ({r.truth_src}; config '{r.config_function}', print "
                  f"'{r.print_function}'/{r.print_subtype}/{r.print_conf}) tech {r.technology} lane {r.lane_type} "
                  f"phase {r.phase_target}; det acc {r.det_acc:.2f}; win {r.wgroup} pred {r.pred} p {r.p_max:.2f}; "
                  f"desc '{r.description}'; val {r.validated} {r.failed_checks if isinstance(r.failed_checks, str) else ''}")
            print("   own: " + ", ".join(f"{c}={r[c]:.3g}" for c in ST if pd.notna(r[c])))
            sib = best.loc[sid]
            sib = sib[(sib.phase_target == r.phase_target)]
            print(sib[["truth", "pred_modal", "det_acc", "technology", "lane_type", "print_subtype"] + ST[:7]]
                  .round(3).to_string())


if __name__ == "__main__":
    main()
