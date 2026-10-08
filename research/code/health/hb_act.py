"""Note 40: actuation-level (ON / OFF event) health statistics on the real Sept 2026 windows.

Per detector and evaluation window (hb_data.EVAL_WIN['stg']): ON durations, OFF gaps, ON->ON
intervals, bursts, repeated identical intervals (rhythm), ONs locked to color changes,
5-min flow-occupancy consistency (all ONs and green-starting ONs of the detector's
best-matching phase), and the histogram distance to the nearest sibling detector.
Hi-res log only (health_core.act_stats); training signals only (folds_v4 minus locked_v2).

    python hb_act.py [--period stg|dec] [--workers 8]  -> %DC_WORK%/health/act_stats[_dec].parquet
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
EV = {"stg": H.DCW / "official" / "stg" / "cache" / "events", "dec": H.DCW / "cache" / "events"}
WIN = {"stg": H.EVAL_WIN["stg"], "dec": {"full": ("2024-12-02 00:00", 72)}}


def one(args):
    per, dev = args
    p = EV[per] / f"DeviceId={dev}"
    if not p.is_dir():
        return None
    ev = pd.read_parquet(p)
    out = []
    for w, (s, h) in WIN[per].items():
        e = pd.Timestamp(s) + pd.Timedelta(hours=h)
        st = hc.act_stats(ev, s, e)
        if st is None or st.empty:
            continue
        st.insert(0, "window", w)
        st.insert(0, "DeviceId", dev)
        out.append(st)
    return pd.concat(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--period", default="stg")
    a = ap.parse_args()
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = sorted({d.lower() for d in folds.DeviceId} - locked)
    if a.limit:
        devs = devs[: a.limit]
    t = time.time()
    with Pool(a.workers) as p:
        res = [r for r in p.imap_unordered(one, [(a.period, d) for d in devs], chunksize=2) if r is not None]
    df = pd.concat(res, ignore_index=True)
    df.insert(1, "period", a.period)
    sfx = "" if a.period == "stg" else "_dec"
    df.to_parquet(H.HB / (f"act_stats{sfx}.parquet" if not a.limit else "act_stats_test.parquet"))
    print(len(df), df.DeviceId.nunique(), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
