"""Note 77: compile the fit77 refits into `%DC_WORK%/final_v3_candidate_v3/weights` (assemble76 with FIT = v3fit77:
lanes D trained on both pair orientations, stackers on the channel-order-free context, setback P50 refit on the
symmetric travel-time block).  GRU and funcnet ONNX untouched.

    python assemble77.py      -> weights rewritten + v3fit77/trees_parity_frames.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
sys.path.insert(0, str(rpath.CODE / "final76"))
import assemble75 as A  # noqa: E402

A.FIT = A.DC_WORK / "final_v3_work" / "v3fit77"
_plan = A.plan


def plan():
    return [(A.FIT / "phase" / Path(src).name, rel) if rel.startswith("phase_lgbm_v5") else (src, rel)
            for src, rel in _plan()]


A.plan = plan

if __name__ == "__main__":
    A.main()
    pm = A.WTS / "phase_lgbm_v5.json"
    m = json.load(open(pm))
    f76 = json.load(open(A.FIT / "phase" / "phase_fit76.json"))
    assert m["features"] == f76["features"], "phase feature list changed"
    m["note76"] = {"n_estimators": f76["n_estimators"], "rule": f76["rule"],
                   "features": "phase- and channel-order-free (note 76: tie-averaged excl_partner_diff / partner, "
                               "float32-rounded cross-candidate ranks)"}
    json.dump(m, open(pm, "w"), indent=1)
    lm = json.load(open(A.WTS / "lanes" / "lane_model.json"))
    assert "both" in lm.get("orientation", ""), "lane model is not the note-77 symmetric fit"
    print("assembled")
