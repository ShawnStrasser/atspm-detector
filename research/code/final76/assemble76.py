"""Note 76: compile the fit76 refits into `%DC_WORK%/final_v3_candidate_v3/weights` (assemble75 with FIT = v3fit76 and the
phase ranker taken from v3fit76/phase instead of candidate v2).  GRU and funcnet (siba) ONNX files are untouched; the
q10 / q90 setback band and setback_pair_hgb still come from candidate v2, the P50 setback models are fit75's (copied).

    python assemble76.py      -> weights rewritten + v3fit76/trees_parity_frames.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
import assemble75 as A  # noqa: E402

A.FIT = A.DC_WORK / "final_v3_work" / "v3fit76"
_plan = A.plan


def plan():
    out = [(A.FIT / "phase" / Path(src).name, rel) if rel.startswith("phase_lgbm_v5") else (src, rel)
           for src, rel in _plan()]
    return out


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
    print("assembled")
