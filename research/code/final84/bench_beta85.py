"""Monday table: speed / peak RAM of the GitHub beta (final_v2, %DC_WORK%/final_v3_work/final_v2_ref) with bench84's protocol:
one signal per predict() call, fresh process, 4 threads, cold = first call, warm = median of 3 more, peak working set;
same bench71 extracts (typical r8 / r11, busiest by channels / events; 30 min / 3 h / 24 h).

    python bench_beta85.py run   -> %DC_WORK%/final_v3_work/f84/bench_beta/
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PKG = W / "final_v3_work" / "final_v2_ref"
B71 = W / "bench71"
OUT = W / "final_v3_work" / "f84" / "bench_beta"
SIGS = ["typical_r8", "typical_r11", "busiest_ch", "busiest_ev"]
LENS = ["m30", "h3", "h24"]


def one(sig, L, out):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import numpy as np
    import psutil
    proc = psutil.Process()
    sys.path.insert(0, str(PKG))
    import predict as P
    f = str(B71 / f"ev_{sig}_{L}.parquet")
    s = time.perf_counter()
    res = P.predict(f, threads=4)
    cold = time.perf_counter() - s
    peak_cold = proc.memory_info().peak_wset / 2**20
    warm = []
    for _ in range(3):
        s = time.perf_counter()
        P.predict(f, threads=4)
        warm.append(time.perf_counter() - s)
    json.dump(dict(sig=sig, L=L, channels=int(res.Detector.nunique()), cold_s=cold, warm_s=float(np.median(warm)),
                   warm_all=warm, peak_cold_mb=peak_cold, peak_all_mb=proc.memory_info().peak_wset / 2**20),
              open(out, "w"), indent=1)


def run():
    import duckdb
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for sig in SIGS:
        for L in LENS:
            ids = duckdb.sql(f"select distinct lower(DeviceId) d from '{B71 / f'ev_{sig}_{L}.parquet'}'").df().d
            assert not set(ids) & lk
            o = OUT / f"r_{sig}_{L}.json"
            subprocess.run([sys.executable, "-W", "ignore", __file__, "--one", sig, L, str(o)], check=True, env=env)
            r = json.load(open(o))
            print(f"beta {sig} {L}: cold {r['cold_s']:.2f}s warm {r['warm_s']:.2f}s peak {r['peak_all_mb']:.0f} MB "
                  f"({r['channels']} ch)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(*sys.argv[2:5])
    else:
        run()
