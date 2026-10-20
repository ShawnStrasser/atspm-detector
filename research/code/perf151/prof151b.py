"""Note 151: tottime by bucket (DuckDB / onnxruntime / pandas / numpy / package python) for one warm predict on
k signals of h24_b (seed 151 order).   python prof151b.py <k> [n_reps]"""
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("P151_SRC") or str(REPO / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from atspm_detector import pipeline as PP  # noqa: E402


def bucket(fn):
    f, _, name = fn
    if "duckdb" in name or "duckdb" in f:
        return "duckdb"
    if "onnxruntime" in f or "onnxruntime" in name or name.startswith("<method 'run' of 'onnxruntime"):
        return "onnx"
    if "pandas" in f:
        return "pandas"
    if "numpy" in f:
        return "numpy"
    if "atspm_detector" in f:
        return "pkg:" + Path(f).stem
    return "other:" + name[:40]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    rng = np.random.default_rng(151)
    pick = [ids[i] for i in rng.permutation(len(ids))][:k]
    ev = pd.read_parquet(DCW / "s151" / "h24b_events.parquet", filters=[("DeviceId", "in", pick)])
    kw = dict(start="2026-09-28 00:00", end="2026-09-29 00:00")
    PP.predict(ev, **kw)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    PP.predict(ev, **kw)
    pr.disable()
    wall = time.perf_counter() - t0
    st = pstats.Stats(pr)
    agg = {}
    for fn, (cc, nc, tt, ct, callers) in st.stats.items():
        b = bucket(fn)
        agg[b] = agg.get(b, 0.0) + tt
    print(f"{k} signals, {len(ev):,} events, wall {wall:.2f}s (profiled)")
    for b, t in sorted(agg.items(), key=lambda x: -x[1])[:30]:
        print(f"  {b:45s} {t:7.3f}")
    st.sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
