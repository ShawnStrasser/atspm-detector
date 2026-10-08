"""Step 1 of note 38: 5-min detector bins for every non-locked signal, both periods.

For each DeviceId in folds_v4.csv (never the locked list) and each event cache (Sept 2026
staging, Dec 2024) run health_core.events_to_bins over the whole window and save the arrays
to %DC_WORK%/health/bins/{period}/{DeviceId}.npz.  The expected channel list passed in is
the labelled / print / card channels of the signal (evaluation only: it just lets a silent
channel appear as a row, as a controller's detector table would in production).

    python hb_build.py [--period stg|dec|both]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import health_core as hc  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
OUT = DCW / "health" / "bins"
PER = {"stg": (DCW / "official" / "stg" / "cache" / "events", "2026-09-18 16:15", "2026-09-21 10:25"),
       "dec": (DCW / "cache" / "events", "2024-12-02 00:00", "2024-12-05 00:00")}


def expected_channels() -> dict:
    v3 = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v3.parquet",
                         columns=["DeviceId", "detector"])
    card = pd.read_parquet(DCW / "cabinet" / "card_channels.parquet",
                           columns=["DeviceId", "detector", "in_use"])
    card = card[card.in_use.fillna(False).astype(bool)]
    a = pd.concat([v3[["DeviceId", "detector"]], card[["DeviceId", "detector"]]])
    a["DeviceId"] = a.DeviceId.str.lower()
    a = a[pd.to_numeric(a.detector, errors="coerce").between(1, 64)]
    return a.groupby("DeviceId").detector.apply(lambda s: sorted(set(int(x) for x in s))).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="both")
    a = ap.parse_args()
    folds = pd.read_csv(DCW / "folds_v4.csv")
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = [d.lower() for d in folds.DeviceId if d.lower() not in locked]
    exp = expected_channels()
    for per in (["stg", "dec"] if a.period == "both" else [a.period]):
        root, t0, t1 = PER[per]
        (OUT / per).mkdir(parents=True, exist_ok=True)
        n, t = 0, time.time()
        for d in devs:
            p = root / f"DeviceId={d}"
            if not p.is_dir():
                continue
            f = OUT / per / f"{d}.npz"
            if f.exists():
                continue
            ev = pd.read_parquet(p)
            B = hc.events_to_bins(ev, t0, t1, exp.get(d))
            np.savez_compressed(f, **{k: v for k, v in B.items()
                                      if k in ("n_on", "occ", "n_chat", "n_flt", "dmax", "dets",
                                               "cov", "hour", "dur_max", "listed")},
                                start=str(B["start"]), bin_s=B["bin_s"])
            n += 1
            if n % 50 == 0:
                print(per, n, f"{time.time() - t:.0f}s", flush=True)
        print(per, "done", n, f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
