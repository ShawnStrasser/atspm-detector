"""Note 151: where health time goes (sub-steps of health_v4.assess) over k h24_b signals in one batch."""
import functools
import os
import sys
import time
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("P151_SRC") or str(REPO / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from atspm_detector import pipeline as PP, health_v4 as HV, health_v4_stats as HS, health_core as HC  # noqa: E402
from atspm_detector import health_counts as HK  # noqa: E402

T = {}


def timed(mod, name):
    f = getattr(mod, name)
    key = f"{mod.__name__.split('.')[-1]}.{name}"

    @functools.wraps(f)
    def g(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[key] = T.get(key, 0.0) + time.perf_counter() - t0
    setattr(mod, name, g)


def main():
    k = int(sys.argv[1])
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    rng = np.random.default_rng(151)
    pick = [ids[i] for i in rng.permutation(len(ids))][:k]
    day = str(DCW / "s151" / "h24b_events.parquet")
    kw = dict(start="2026-09-28 00:00", end="2026-09-29 00:00", device_ids=pick)
    for n in ("assess", "_stage1", "limits", "_time_of_day", "scores", "status", "_outputs", "config_notes",
              "episode_flags", "_common_note"):
        timed(HV, n)
    for n in ("occ_bins", "bins_ctx", "q5_stats", "too_fast", "act_stats", "act118", "slopes", "like_corr",
              "fast_even", "occ_hi", "episodes_ctx", "episode_overlap", "dropout_clean"):
        timed(HS, n)
    for n in ("health", "events_to_bins", "on_episodes"):
        timed(HC, n)
    for n in ("spiky", "hourly_geh", "twin_matrix", "common_holds"):
        timed(HK, n)
    timed(PP, "_health_events")
    PP.predict(day, **kw)
    T.clear()
    PP.predict(day, **kw)
    for a, b in sorted(T.items(), key=lambda x: -x[1]):
        print(f"  {a:34s} {b:7.2f}  {1000 * b / k:7.1f} ms/signal")


if __name__ == "__main__":
    main()
