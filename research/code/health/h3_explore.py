"""Note 43 step 1: shape / choppiness statistics and stuck-on episodes for every training signal,
Sept-2026 evaluation windows of note 38 (hb_data.EVAL_WIN), with the OOF predicted phase as the
reference group (h3_oof.py).  Research only.

    python h3_explore.py -> %DC_WORK%/health3/shape_stats.parquet, episodes.parquet
"""
from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health3"
EV = H.DCW / "official" / "stg" / "cache" / "events"
WG = {"2h": "h3", "6h": "h6", "24h": "h24", "full": "full"}
PF = None


def init():
    global PF
    pf = pd.read_parquet(OUT / "oof_pf.parquet", columns=["period", "DeviceId", "detector", "wgroup", "pred_phase"])
    pf = pf[pf.period == "stg"]
    PF = {k: dict(zip(g.detector, g.pred_phase)) for k, g in pf.groupby(["DeviceId", "wgroup"])}


def one(dev):
    B = H.load_B("stg", dev)
    p = EV / f"DeviceId={dev}"
    if B is None or not p.is_dir():
        return None
    ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    rows, eps = [], []
    for w, (s, h) in H.EVAL_WIN["stg"].items():
        a, b = H.win_bins(B, s, h)
        if b - a < 12:
            continue
        S = H.slice_B(B, a, b)
        t0, t1 = S["start"], S["start"] + pd.Timedelta(seconds=(b - a) * S["bin_s"])
        ph = PF.get((dev, WG[w.split("_")[0]]), {})
        cyc = hc.cycle_counts(ev, t0, t1, ph) if ph else None
        st = hc.shape_stats(S, ph or None, cyc)
        st0 = hc.shape_stats(S, None, None)[["detector", "chop15"]].rename(columns={"chop15": "chop15_blind"})
        st = st.merge(st0, on="detector", how="left")
        ep = hc.on_episodes(ev, t0, t1)
        if len(ep):
            ep.insert(0, "window", w)
            eps.append(ep)
            g = ep.sort_values("dur_s").groupby("detector").last()
            st["ep_dur"] = st.detector.map(g.dur_s)
            st["ep_no_off"] = st.detector.map(g.no_off)
            st["ep_co"] = st.detector.map(g.co_stuck)
        st.insert(0, "window", w)
        rows.append(st)
    if not rows:
        return None
    st = pd.concat(rows, ignore_index=True)
    st.insert(0, "DeviceId", dev)
    ep = pd.concat(eps, ignore_index=True).assign(DeviceId=dev) if eps else None
    return st, ep


def main():
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = [f.stem for f in sorted((H.BINS / "stg").glob("*.npz")) if f.stem not in locked]
    t = time.time()
    with Pool(6, initializer=init) as p:
        res = [r for r in p.imap_unordered(one, devs, chunksize=4) if r is not None]
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(OUT / "shape_stats.parquet")
    pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True).to_parquet(OUT / "episodes.parquet")
    print(len(res), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
