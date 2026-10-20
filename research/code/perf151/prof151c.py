"""Note 151: stage times of one warm predict on the first k h24_b signals (seed-151 order) in ONE batch.
    python prof151c.py <k> [--cprofile]"""
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
from atspm_detector import pipeline as PP, features as F1, features_partner as F2, similarity as SM  # noqa: E402
from atspm_detector import features_expert as FX, function_stage as FS, setback as SB, blend as GB  # noqa: E402
from atspm_detector import health_v4 as HV, streams as ST, decode as DE, function as FN  # noqa: E402

T = {}


def timed(mod, name, key=None):
    f = getattr(mod, name)
    key = key or f"{mod.__name__.split('.')[-1]}.{name}"

    @functools.wraps(f)
    def g(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[key] = T.get(key, 0.0) + time.perf_counter() - t0
    setattr(mod, name, g)


def main():
    if os.environ.get("P151_NOGC"):
        import gc
        gc.disable()
    k = int(sys.argv[1])
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    rng = np.random.default_rng(151)
    pick = [ids[i] for i in rng.permutation(len(ids))][:k]
    day = str(DCW / "s151" / "h24b_events.parquet")
    kw = dict(start="2026-09-28 00:00", end="2026-09-29 00:00", device_ids=pick, batch_signals=k)
    for mod, names in ((PP, ["load_events", "build_chunk_tables", "detector_universe", "signal_facts", "gate_frame",
                             "build_features", "score", "score_function", "post_outputs", "_assemble", "on_table",
                             "_health_events", "_setback", "_windows"]),
                       (F1, ["apply_window", "shared_parts", "_mask_features", "finalise", "call_reverse"]),
                       (F2, ["_partner_features", "add_partner_diffs"]), (SM, ["build_window", "build_lead_window"]),
                       (FX, ["build"]), (FS, ["run"]), (HV, ["assess"]), (ST, ["build_streams_batch"]),
                       (DE, ["assemble"]), (PP, ["siba_phase_probs", "_function_frame"])):
        for n in names:
            timed(mod, n)
    PP.predict(day, **kw)
    T.clear()
    t0 = time.perf_counter()
    if "--cprofile" in sys.argv:
        pr = cProfile.Profile()
        pr.enable()
    PP.predict(day, **kw)
    wall = time.perf_counter() - t0
    print(f"{k} signals: wall {wall:.2f}s = {wall / k:.3f} s/signal")
    for a, b in sorted(T.items(), key=lambda x: -x[1]):
        print(f"  {a:34s} {b:7.2f}  {1000 * b / k:7.1f} ms/signal")
    if "--cprofile" in sys.argv:
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(30)


if __name__ == "__main__":
    main()
