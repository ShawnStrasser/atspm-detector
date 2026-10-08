"""Note 88: quick speed / RAM check of candidate v4d (bench84 recipe: fresh process per signal, 4 threads, siba filter ON;
cold = first call, warm = median of 3 more) on the four bench signals at 3 h only.

    python bench88.py   -> %DC_WORK%/final_v3_work/f88/bench/r_<sig>_h3.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final84"))
import bench84 as B4  # noqa: E402

B4.V4B = B4.B.W / "final_v3_candidate_v4d"
B4.out_dir = lambda cfg: B4.B.W / "final_v3_work" / "f88" / "bench"
B4.B.LENS = [x for x in B4.B.LENS if x in ("h3", "3h", "180")] or B4.B.LENS[1:2]
print("lengths:", B4.B.LENS, flush=True)
B4.run("mean3")
