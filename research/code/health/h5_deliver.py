"""Note 47: deliverable health v5 (health_core defaults) for every training signal (locked_v2 never read), both
Sept 2026 periods, whole span each, same out-of-fold site inputs as note 46:
    stg  Fri 18 Sep 16:15 - Mon 21 Sep 10:25 (official stg event cache)
    w40  Sat 26 Sep 16:15 - Tue 29 Sep 00:00 (h4_w40 event cache of the late-Sept pull)

    python h5_deliver.py -> %DC_WORK%/health/health_v5.parquet
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
import h4_final as F  # noqa: E402
import h4_w40 as W  # noqa: E402
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
PER = {"stg": (F.EVR["stg"], F.FULL["stg"]), "w40": (W.EVC, W.SPAN)}
COLS = ["period", "DeviceId", "detector", "status", "health_score", "reason", "not_checked", "bad_periods", "n_on",
        "cov_h", "pred_function", "f_conf", "ph_conf", "lanes_spanned", "n_families", "night_ratio", "short2"]


def one(dev):
    out = []
    ph, fn, ln, pc = F.inputs("stg", dev)
    for per, (root, (t0, t1)) in PER.items():
        p = root / f"DeviceId={dev}"
        if not p.is_dir():
            continue
        ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
        h = hc.health(ev, pd.Timestamp(t0), pd.Timestamp(t1), F.EXP.get(dev), ph or None, fn or None, ln or None,
                      pc or None)
        h["bad_periods"] = F.periods(h)
        h.insert(0, "DeviceId", dev)
        h.insert(0, "period", per)
        out.append(h[[c for c in COLS if c in h] + [c for c in h if c.startswith("s_")]])
    return pd.concat(out, ignore_index=True) if out else None


def main():
    t = time.time()
    devs = W.devs()
    with Pool(6, initializer=F.init, initargs=(["final"],)) as p:
        res = [r for r in p.imap_unordered(one, devs, chunksize=2) if r is not None]
    df = pd.concat(res, ignore_index=True)
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not df.DeviceId.isin(locked).any()
    df.to_parquet(H.HB / "health_v5.parquet", index=False)
    print(df.shape, df.groupby("period").status.value_counts().to_dict(), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
