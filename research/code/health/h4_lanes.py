"""Note 46 input: lanes spanned per detector (note 42's lane_output.lanes) on the full window of both
periods, from the OOF predicted phase / function (health3/oof_pf.parquet, wgroup full).  Used by the
health v4 `rapid` rule: a detector spanning 2+ lanes sees two vehicles side by side as two quick ONs.
The lane pair model was fitted on all training signals (note 42) - in-sample for these signals; it
only sets which rapid-actuation limit applies, never a health label.  Training signals only.

    python h4_lanes.py -> %DC_WORK%/health4/lanes_full.parquet
"""
from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lanes"))
import hb_data as H  # noqa: E402
import lane_output as LO  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health4"
EVR = {"stg": H.DCW / "official" / "stg" / "cache" / "events", "dec": H.DCW / "cache" / "events"}
FULL = {"stg": ("2026-09-18 16:15", "2026-09-21 10:25"), "dec": ("2024-12-02 00:00", "2024-12-05 00:00")}
PF = PM = None


def init():
    global PF, PM
    pf = pd.read_parquet(H.DCW / "health3" / "oof_pf.parquet")
    PF = {k: g for k, g in pf[pf.wgroup == "full"].groupby(["period", "DeviceId"])}
    PM = LO.PairModel(LO.default_dir())


def one(dev):
    out = []
    for per in ("stg", "dec"):
        p = EVR[per] / f"DeviceId={dev}"
        g = PF.get((per, dev))
        if g is None or not p.is_dir():
            continue
        ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin([81, 82])).to_pandas()
        t0, t1 = pd.Timestamp(FULL[per][0]), pd.Timestamp(FULL[per][1])
        ev = ev[(ev.Timestamp >= t0) & (ev.Timestamp < t1)]
        ev["DeviceId"] = dev
        pr = pd.DataFrame({"DeviceId": dev, "Detector": g.detector.astype(int), "phase_pred": g.pred_phase,
                           "function_pred": g.pred_function})
        _, det = LO.lanes(ev, pr, PM, start=t0, end=t1)
        if len(det):
            out.append(det.assign(period=per))
    return pd.concat(out, ignore_index=True) if out else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = sorted({d.lower() for d in folds.DeviceId} - locked)
    t = time.time()
    with Pool(6, initializer=init) as p:
        res = [r for r in p.imap_unordered(one, devs, chunksize=4) if r is not None]
    df = pd.concat(res, ignore_index=True)
    assert not df.DeviceId.isin(locked).any()
    df.to_parquet(OUT / "lanes_full.parquet", index=False)
    print(df.shape, df.groupby("period").n_lanes_spanned.value_counts().to_dict(), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
