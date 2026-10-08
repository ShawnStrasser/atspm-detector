"""Stage 33: CPU cost of the final_v3 candidate with the network on K pieces.

Runs the candidate package (`%DC_WORK%/final_v3_candidate`) end to end on note 32's
20-signal bench extract with `gru_blend.phase_probs(max_chunks=K)`; K = 0 means all pieces.

    python research/code/neural/pieces_bench.py 4 h6     # -> one json line
"""
import json
import os
import sys
import time
from pathlib import Path

import psutil

W = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
PKG = W / "final_v3_candidate"
sys.path.insert(0, str(PKG))
import gru_blend as gb  # noqa: E402
import predict as P  # noqa: E402

mc = int(sys.argv[1])
L = sys.argv[2]
LENS = {"h3": ("2026-09-21 06:00:00", "2026-09-21 09:00:00"),
        "h6": ("2026-09-20 06:00:00", "2026-09-20 12:00:00")}
P.predict(PKG / "sample_events.parquet", min_actuations=1)          # warm-up
g0 = gb.phase_probs
t = [0.0]


def gw(*a, **k):
    if mc:
        k["max_chunks"] = mc
    s = time.perf_counter()
    r = g0(*a, **k)
    t[0] += time.perf_counter() - s
    return r


gb.phase_probs = gw
s = time.perf_counter()
res = P.predict(W / "final_v3_work" / "extracts" / f"bench_{L}.parquet", *LENS[L])
w = time.perf_counter() - s
n = res.DeviceId.nunique()
print(json.dumps({"length": L, "max_chunks": mc, "signals": int(n),
                  "s_per_signal": round(w / n, 3), "gru_s_per_signal": round(t[0] / n, 3),
                  "peak_wset_mb": round(psutil.Process().memory_info().peak_wset / 2**20)}))
