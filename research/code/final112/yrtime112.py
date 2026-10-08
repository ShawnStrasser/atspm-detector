"""Note 112: wall time of the dead yellow / red-clearance pair features (features_yellowred.SQL_YR + merge,
predict.build_features; no v6 model reads a yr_* column) on the 4 bench signals at 3 h.

    python yrtime112.py <package dir> [L=h3]     -> %DC_WORK%/s112/yrtime112.json
One fresh process, 4 threads; per signal 1 cold + 3 warm predict() calls; build_features is re-compiled from its own
source with timers around the yr block, so the answers are unchanged.  Bench extracts are the note-71 non-locked ones.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
SIGS = ["typical_r8", "typical_r11", "busiest_ev", "busiest_ch"]


def main(pkg: Path, L: str):
    sys.path.insert(0, str(pkg))
    import predict as P
    src = inspect.getsource(P.build_features)
    a = "    try:\n        yr = f3.build(con, WIN, dm, f3.SQL_YR)"
    b = "        log(f\"yr features unavailable"
    assert a in src and b in src
    src = src.replace(a, "    _t_yr = time.perf_counter()\n" + a)
    i = src.index(b)
    j = src.index("\n", i) + 1
    src = src[:j] + "    _YRT.append(time.perf_counter() - _t_yr)\n" + src[j:]
    P.__dict__.setdefault("time", time)
    P._YRT = []
    exec(compile(src, "build_features_timed", "exec"), P.__dict__)
    out = {}
    for s in SIGS:
        f = str(W / "bench71" / f"ev_{s}_{L}.parquet")
        P.predict(f, threads=4)
        yr, tot = [], []
        for _ in range(3):
            P._YRT.clear()
            t0 = time.perf_counter()
            P.predict(f, threads=4)
            tot.append(time.perf_counter() - t0)
            yr.append(sum(P._YRT))
        out[s] = {"yr_s": float(np.median(yr)), "warm_s": float(np.median(tot)), "yr_all": yr, "warm_all": tot,
                  "share": float(np.median(yr) / np.median(tot))}
        print(s, out[s], flush=True)
    o = W / "s112" / f"yrtime112_{L}.json"
    json.dump({"pkg": str(pkg), "L": L, "profile": os.environ.get("DC_FAST_PROFILE"), "res": out}, open(o, "w"), indent=1)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "h3")
