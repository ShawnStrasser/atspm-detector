"""Note 43 step 2: rule statistics per (signal, window, detector) in three reference variants.

    blind   = note 38/40: every detector compared with the rest of the signal
    phase   = compared with the other live detectors on its OOF-predicted phase (66-h prediction:
              "the classifier's answer for the site"); twins (same zone on two inputs) excluded
    phase_w = the same with the prediction made on a window of the same length (h3 for 2 h, h6,
              h24, full) - what a short-sample user would have

Plus shape statistics (choppiness, phase correlation, elasticity), cycle-level choppiness and
stuck-on episodes (ON -> ON with no OFF counted).  Training signals only; locked never read.

    python h3_eval_build.py -> %DC_WORK%/health3/eval_stats.parquet, eval_episodes.parquet
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


def stats_variant(S, ev, t0, t1, ph):
    n, cov = S["n_on"].astype(float), S["cov"]
    live = n[:, cov].sum(1) > 0
    if ph is None:
        st = hc.det_stats(S)
        sh = hc.shape_stats(S, None, None)
    else:
        tw = hc.twins(n, cov)
        ref, kind, _ = hc._refs(S["dets"], live, ph, tw)
        st = hc.det_stats(S, ref)
        sh = hc.shape_stats(S, ph, hc.cycle_counts(ev, t0, t1, ph))
    return st.merge(sh, on="detector", how="left")


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
        ep = hc.on_episodes(ev, t0, t1)
        if len(ep):
            eps.append(ep.assign(window=w))
        for var, ph in (("blind", None), ("phase", PF.get((dev, "full"))),
                        ("phase_w", PF.get((dev, WG[w.split("_")[0]])))):
            if var != "blind" and not ph:
                continue
            st = stats_variant(S, ev, t0, t1, ph)
            if len(ep):
                g = ep.sort_values("dur_s").groupby("detector").last()
                st["ep_dur"] = st.detector.map(g.dur_s)
                st["ep_no_off"] = st.detector.map(g.no_off).astype(float)
                st["ep_co"] = st.detector.map(g.co_stuck).astype(float)
            st.insert(0, "variant", var)
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
    st = pd.concat([r[0] for r in res], ignore_index=True)
    st.to_parquet(OUT / "eval_stats.parquet")
    pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True).to_parquet(OUT / "eval_episodes.parquet")
    print(len(res), st.shape, f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
