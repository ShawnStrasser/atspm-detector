"""Note 86: quick speed / RAM check of candidate v4e vs v4d, paired on the same (busy: GPU job running) machine (bench84
recipe: fresh process per signal, 4 threads, siba filter ON; cold = first call, warm = median of 3 more), 3 h only.

    python bench86.py v4d|v4e   -> %DC_WORK%/final_v3_work/f86/bench_<pkg>/r_<sig>_h3.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final84"))
import bench84 as B4  # noqa: E402

PKG = sys.argv[1]
assert PKG in ("v4d", "v4e")
B4.V4B = B4.B.W / f"final_v3_candidate_{PKG}"
B4.out_dir = lambda cfg: B4.B.W / "final_v3_work" / "f86" / f"bench_{PKG}"
B4.B.LENS = [x for x in B4.B.LENS if x in ("h3", "3h", "180")] or B4.B.LENS[1:2]
print("package:", B4.V4B, "lengths:", B4.B.LENS, flush=True)
B4.run("mean3")
