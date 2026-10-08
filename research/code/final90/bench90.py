"""Note 90: speed / RAM of the final package v4f vs v4e (GRU phase net), paired on the same machine load -- bench84 recipe
(one signal per predict() call, fresh process, 4 threads, siba filter ON = default; cold = first call, warm = median of
3 more, peak working set).  Signals: typical r8 / r11, busiest by events / channels; 30 min / 3 h / 24 h.

    python bench90.py [v4e,v4f]   -> %DC_WORK%/final_v3_work/f90/bench_<tag>/r_<sig>_<len>.json + summary
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final84"))
import bench84 as B4  # noqa: E402

tags = (sys.argv[1] if len(sys.argv) > 1 else "v4f,v4e").split(",")
for tag in tags:
    B4.V4B = B4.B.W / f"final_v3_candidate_{tag}"
    B4.out_dir = lambda cfg, t=tag: B4.B.W / "final_v3_work" / "f90" / f"bench_{t}"
    print("==", tag, B4.B.SIGS, B4.B.LENS, flush=True)
    B4.run("mean3")
    B4.B.OUT = B4.out_dir("mean3")
    B4.B.cmd_summary(None)
