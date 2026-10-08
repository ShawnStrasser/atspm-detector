"""Note 46 (f): rolling function consistency - the function head run on consecutive 2-h windows through the
66-h Sept-2026 sample (33 windows) of every training signal, out of fold (h4_pkg.FunctionScorer: the fold
model that never saw the signal; phase fixed at the out-of-fold site answer).  Locked never read.

    python h4_rolling.py [--period stg] [--hours 2] -> %DC_WORK%/health4/rolling_{period}.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health4"
EVR = {"stg": H.DCW / "official" / "stg" / "cache" / "events", "dec": H.DCW / "cache" / "events"}
FULL = {"stg": ("2026-09-18 16:15", "2026-09-21 10:25"), "dec": ("2024-12-02 00:00", "2024-12-05 00:00")}
IN = FO = SC = None
ARGS = None


def init(args):
    global IN, FO, SC, ARGS
    import h4_pkg  # noqa: F401  (loads the package once per process)
    ARGS = args
    x = pd.read_parquet(OUT / "inputs.parquet")
    x = x[(x.wgroup == "full") & (x.period == args.period)]
    IN = {k: g for k, g in x.groupby("DeviceId")}
    f4 = pd.read_csv(H.DCW / "folds_v4.csv")
    FO = dict(zip(f4.DeviceId.str.lower(), f4.fold.astype(int)))
    SC = {}


def one(dev):
    import h4_pkg
    g = IN.get(dev)
    p = EVR[ARGS.period] / f"DeviceId={dev}"
    if g is None or not p.is_dir():
        return None
    k = FO[dev]
    if k not in SC:
        SC[k] = h4_pkg.FunctionScorer(k)
    ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin([1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173])
                                ).to_pandas()
    ev["DeviceId"] = dev
    ph = dict(zip(g.detector.astype(int), g.pred_phase.astype(int)))
    pp = dict(zip(g.detector.astype(int), g.top_prob.astype(float)))
    t0, t1 = pd.Timestamp(FULL[ARGS.period][0]), pd.Timestamp(FULL[ARGS.period][1])
    out, j = [], 0
    while t0 + pd.Timedelta(hours=ARGS.hours * (j + 1)) <= t1 + pd.Timedelta(minutes=30):
        a = t0 + pd.Timedelta(hours=ARGS.hours * j)
        b = min(a + pd.Timedelta(hours=ARGS.hours), t1)
        try:
            r = SC[k].score(ev, a, b, ph, pp)
        except Exception as exc:  # noqa: BLE001 - a thin window must not stop the run
            r = None
            print(dev, j, type(exc).__name__, exc, flush=True)
        if r is not None:
            out.append(r.assign(k=j, t0=a, t1=b))
        j += 1
    return pd.concat(out, ignore_index=True).assign(DeviceId=dev, fold=k) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="stg")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = sorted({d.lower() for d in folds.DeviceId} - locked)
    if a.limit:
        devs = devs[:a.limit]
    t = time.time()
    res = []
    with Pool(a.procs, initializer=init, initargs=(a,)) as p:
        for n, r in enumerate(p.imap_unordered(one, devs, chunksize=1)):
            if r is not None:
                res.append(r)
            if n % 50 == 0:
                print(n, f"{time.time() - t:.0f}s", flush=True)
    df = pd.concat(res, ignore_index=True)
    assert not df.DeviceId.isin(locked).any()
    df.to_parquet(OUT / f"rolling_{a.period}.parquet", index=False)
    print(df.shape, df.DeviceId.nunique(), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
