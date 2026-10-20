"""Note 151: where does predict() time go for one 24-h signal (h24_b)?  cProfile of a warm call + stage wrappers.
    python prof151.py [k] [--src <package parent>]  (k = index into the seed-151 sample of h24_b ids)
"""
import cProfile
import functools
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

T = {}


def timed(mod, name):
    f = getattr(mod, name)

    @functools.wraps(f)
    def g(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[name] = T.get(name, 0.0) + time.perf_counter() - t0
    setattr(mod, name, g)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    rng = np.random.default_rng(151)
    pick = rng.choice(ids, 10, replace=False)
    dev = pick[k]
    ev = pd.read_parquet(DCW / "s151" / "h24b_events.parquet", filters=[("DeviceId", "==", dev)])
    kw = dict(start="2026-09-28 00:00", end="2026-09-29 00:00")
    for n in ("load_events", "build_chunk_tables", "detector_universe", "signal_facts", "gate_frame",
              "build_features", "score", "score_function", "post_outputs", "_assemble", "on_table",
              "_health_events"):
        timed(PP, n)
    from atspm_detector import blend as gb, health_v4 as hv4, setback as sb, night_speed as ns
    timed(gb, "streams_for")
    timed(hv4, "assess")
    timed(sb, "setback")
    timed(ns, "night_speed")
    PP.predict(ev, **kw)
    T.clear()
    t0 = time.perf_counter()
    PP.predict(ev, **kw)
    print(f"events {len(ev):,}  wall {time.perf_counter()-t0:.2f}s")
    for a, b in sorted(T.items(), key=lambda x: -x[1]):
        print(f"  {a:22s} {b:6.3f}")
    pr = cProfile.Profile()
    pr.enable()
    PP.predict(ev, **kw)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(60)


if __name__ == "__main__":
    main()
