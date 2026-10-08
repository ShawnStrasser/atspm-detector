"""Note 76: function refit on the phase-order-free frame and scoring inside the current champion.

    python f76_function.py fit       # 229-feature arm (note 57 recipe, t57_function.stage_fit) on frame v6e with the new
                                     # excl_partner_diff, 3 seeds x 6 folds -> f76/function/v6e/<cfg>/
    python f76_function.py stack     # note-67/69 context stacker (s74.cmd_stack, main net = siba x69_siba, 3 seeds) on the
                                     # new tree probabilities -> f76/s74/f76/
    python f76_function.py compare   # s74.cmd_compare siba (champion .9169) vs f76: gate .9, >= 30 / 5 / 10 min, E and R

Held fixed (as note 57): the frame's predicted phase (v6e), lanes D and the pick / health inputs (they read the old
229-arm probabilities and the old excl_partner_diff); siba OOF unchanged.  Locked_v2 asserted absent (V / cand64).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import v3_retrain as V  # noqa: E402
import t57_function as T57F  # noqa: E402

F76 = DC_WORK / "final_v3_work" / "f76"
ARM = os.environ.get("F76_ARM", "")       # "" = phase-order fix only; "c" = + channel-order fix (frame_v6e_c, lag76.py)
CFG = "drop:pp_xcand+pp_pdiff+pp_v2+yr+ratio+phctx"
_orig = V.set_frame


def _set_frame(tag):
    _orig(tag)
    assert tag == "v6e"
    V.OUT = F76 / ("frame_v6e_c" if ARM == "c" else "frame_v6e")
    V.FEATS = V.OUT / "feat_frame.parquet"


V.set_frame = _set_frame
T57F.OUT = F76 / ("function_c" if ARM == "c" else "function")


def new_dir() -> Path:
    return T57F.cfg_dir("v6e", CFG)


def stage_fit(a):
    T57F.stage_fit(argparse.Namespace(frame="v6e", cfg=CFG, seeds="0,1,2", threads=6))
    old = json.load(open(DC_WORK / "trees57" / "function" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx" /
                         "cols.json"))["cols"]
    new = json.load(open(new_dir() / "cols.json"))["cols"]
    assert old == new, "feature list differs from the 229-feature arm"
    print(f"fit done: {len(new)} features, same list as the note-57 arm -> {new_dir()}")


def _s74():
    sys.path.insert(0, str(Path(rpath.CODE) / "evaluation"))
    import s59_step6 as S59
    import s74
    S59.FUNC_DIR = new_dir()
    V.set_frame = _orig          # scoring frame / keys / baseline run = the original v6e paths (same rows)
    s74.OUT = F76 / "s74"
    s74.OUT.mkdir(parents=True, exist_ok=True)
    return s74


def stage_stack(a):
    s74 = _s74()
    s74.cmd_stack(argparse.Namespace(name="f76" + ARM, main="x69_siba", extra=""))


def stage_compare(a):
    s74 = _s74()
    s74.cmd_compare(argparse.Namespace(a="siba", b="f76" + ARM))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "stack", "compare"])
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
