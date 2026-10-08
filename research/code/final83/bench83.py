"""Note 83: bench75 speed / RAM on `%DC_WORK%/final_v3_candidate_v4` (one signal per predict() call, fresh process,
4 threads; cold = first call, warm = median of 3 more calls; peak working set).  Signals: typical r8 / r11, busiest by
channels / events; 30 min / 3 h / 24 h.  `--filter` runs with the siba candidate filter ON (DC_SIBA_FILTER=1).

    python bench83.py run [--filter] [--lens m30,h3,h24]   -> %DC_WORK%/final_v3_work/f83/bench[_filter]/
    python bench83.py summary [--filter]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final75"))
import bench75 as B  # noqa: E402

B.PKG = B.W / "final_v3_candidate_v4"


def out_dir(filt: bool) -> Path:
    return B.W / "final_v3_work" / "f83" / ("bench_filter" if filt else "bench")


def run(filt: bool, lens):
    import duckdb
    import pandas as pd
    o_d = out_dir(filt)
    o_d.mkdir(parents=True, exist_ok=True)
    lk = set(pd.read_csv(B.W / "official" / "locked_v2.csv").DeviceId.str.lower())
    env = dict(os.environ, DC_SIBA_FILTER="1" if filt else "0", CUDA_VISIBLE_DEVICES="")
    for sig in B.SIGS:
        for L in lens:
            ids = duckdb.sql(f"select distinct lower(DeviceId) d from '{B.ev_file(sig, L)}'").df().d
            assert not set(ids) & lk
            o = o_d / f"r_{sig}_{L}.json"
            subprocess.run([sys.executable, "-W", "ignore", __file__, "--one", sig, L, str(o)], check=True, env=env)
            r = json.load(open(o))
            print(f"{sig} {L}: cold {r['cold_s']:.2f}s warm {r['warm_s']:.2f}s peak {r['peak_all_mb']:.0f} MB "
                  f"({r['channels']} ch)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        B.one(*sys.argv[2:5])
    else:
        filt = "--filter" in sys.argv
        lens = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--lens=")), "m30,h3,h24").split(",")
        if sys.argv[1] == "run":
            run(filt, lens)
        B.OUT = out_dir(filt)
        B.cmd_summary(None)
