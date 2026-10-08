"""Note 38: rule statistics on the REAL data, fixed windows (Sept 2026) + Dec 2024 full window.

    python hb_real.py            -> %DC_WORK%/health/real_stats.parquet
"""
from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")


def one(args):
    per, dev = args
    B = H.load_B(per, dev)
    if B is None:
        return None
    out = []
    wins = H.EVAL_WIN.get(per, {"full": (str(B["start"]), 72)})
    for w, (s, h) in wins.items():
        a, b = H.win_bins(B, s, h)
        if b - a < 12:
            continue
        st = H.stats_of(H.slice_B(B, a, b))
        st.insert(0, "window", w)
        st.insert(0, "period", per)
        st.insert(0, "DeviceId", dev)
        out.append(st)
    return pd.concat(out) if out else None


def main():
    jobs = [(per, f.stem) for per in ("stg", "dec") for f in sorted((H.BINS / per).glob("*.npz"))]
    t = time.time()
    with Pool(6) as p:
        res = [r for r in p.imap_unordered(one, jobs, chunksize=4) if r is not None]
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(H.HB / "real_stats.parquet")
    print(len(df), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
