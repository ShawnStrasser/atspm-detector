"""Note 84: bench75 speed / RAM of candidate v4b, siba filter ON (the v4b default), one signal per predict() call, fresh
process, 4 threads (cold = first call, warm = median of 3 more, peak working set).  Two configurations:
  mean3    the package as shipped (3 siba members, stacker 'mean3')
  single   a copy with members 1 / 2 removed (FuncNet skips absent members -> stacker 'single' = the fallback path)
Signals: typical r8 / r11, busiest by channels / events; 30 min / 3 h / 24 h.

    python bench84.py run mean3|single    -> %DC_WORK%/final_v3_work/f84/bench_<cfg>/
    python bench84.py summary mean3|single
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final75"))
import bench75 as B  # noqa: E402

V4B = B.W / "final_v3_candidate_v4b"
SINGLE = B.W / "final_v3_work" / "f84" / "pkg_single"


def pkg(cfg: str) -> Path:
    if cfg == "mean3":
        return V4B
    if not SINGLE.exists():
        shutil.copytree(V4B, SINGLE, ignore=shutil.ignore_patterns("__pycache__"))
        for f in (SINGLE / "weights" / "funcnet").glob("x74_sibafull4l_s[12]_*.onnx"):
            f.unlink()
    return SINGLE


def out_dir(cfg: str) -> Path:
    return B.W / "final_v3_work" / "f84" / f"bench_{cfg}"


def run(cfg: str):
    import duckdb
    import pandas as pd
    o_d = out_dir(cfg)
    o_d.mkdir(parents=True, exist_ok=True)
    lk = set(pd.read_csv(B.W / "official" / "locked_v2.csv").DeviceId.str.lower())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("DC_SIBA_FILTER", None)                      # package default (ON)
    for sig in B.SIGS:
        for L in B.LENS:
            ids = duckdb.sql(f"select distinct lower(DeviceId) d from '{B.ev_file(sig, L)}'").df().d
            assert not set(ids) & lk
            o = o_d / f"r_{sig}_{L}.json"
            subprocess.run([sys.executable, "-W", "ignore", __file__, "--one", str(pkg(cfg)), sig, L, str(o)],
                           check=True, env=env)
            r = json.load(open(o))
            print(f"{cfg} {sig} {L}: cold {r['cold_s']:.2f}s warm {r['warm_s']:.2f}s peak {r['peak_all_mb']:.0f} MB "
                  f"({r['channels']} ch)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        B.PKG = Path(sys.argv[2])
        sys.path.insert(0, str(B.PKG))
        import predict as P
        import function_stage as fs
        assert P.SIBA_FILTER, "v4b default must be filter ON"
        B.one(*sys.argv[3:6])
        m = fs.models(B.PKG / "weights")
        r = json.load(open(sys.argv[5]))
        r["members"], r["stacker_variant"] = len(m["net"].members), m["stacker"].variant
        json.dump(r, open(sys.argv[5], "w"), indent=1)
    else:
        cfg = sys.argv[2]
        if sys.argv[1] == "run":
            run(cfg)
        B.OUT = out_dir(cfg)
        B.cmd_summary(None)
