"""Note 151: same 50 h24_b signals, one predict per batch size (fresh process each): wall and peak memory.
    python bsize151.py <batch_signals> [n]"""
import os
import sys
import time
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("P151_SRC") or str(REPO / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from atspm_detector import pipeline as PP  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from run151 import peak_mb  # noqa: E402


def main():
    bs = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    rng = np.random.default_rng(151)
    pick = [ids[i] for i in rng.permutation(len(ids))][:n]
    day = str(DCW / "s151" / "h24b_events.parquet")
    PP.predict(str(Path(PP.__file__).parent / "data" / "sample_events.parquet"))
    t0 = time.perf_counter()
    out = PP.predict(day, start="2026-09-28 00:00", end="2026-09-29 00:00", device_ids=pick, batch_signals=bs)
    w = time.perf_counter() - t0
    print(f"batch_signals={bs}: {n} signals {w:.1f}s = {w / n:.3f} s/signal, peak {peak_mb():.0f} MB, {len(out)} rows")


if __name__ == "__main__":
    main()
