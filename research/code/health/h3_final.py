"""Note 43 step 4: run the packaged health_core.health() on every training signal and window.

Modes per (signal, window):
    v3_blind  log only (no phase / function input)
    v3_phase  + OOF predicted phase (66-h prediction)
    v3_pf     + OOF predicted phase and function (66-h prediction)            <- candidate
    v3_pf_w   + OOF predictions made on a window of the same length (h3 / h6 / h24 / full)
    v3_shuf   CONTROL: phase and function predictions permuted among the signal's detectors
Sept 2026 windows of note 38 (hb_data.EVAL_WIN); the channel list = hb_build.expected_channels().
Also writes the deliverable: full-window v3_pf health for both periods (Sept 2026 66 h, Dec 2024
72 h) -> %DC_WORK%/health/health_v3.parquet.  Training signals only; locked never read.

    python h3_final.py -> %DC_WORK%/health3/final_eval.parquet, %DC_WORK%/health/health_v3.parquet
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_build as HB  # noqa: E402
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health3"
EVR = {"stg": H.DCW / "official" / "stg" / "cache" / "events", "dec": H.DCW / "cache" / "events"}
FULL = {"stg": ("2026-09-18 16:15", "2026-09-21 10:25"), "dec": ("2024-12-02 00:00", "2024-12-05 00:00")}
WG = {"2h": "h3", "6h": "h6", "24h": "h24", "full": "full"}
KEEP = ["detector", "health_score", "status", "n_on", "cov_h", "chop15", "chop_lim", "ref_kind", "drop_ref",
        "drop_lam", "drop_rec", "drop_in_stuck", "ep_dur", "ep_no_off", "ep_co", "ep_rec", "pred_function"]
PF = EXP = None


def init():
    global PF, EXP
    pf = pd.read_parquet(OUT / "oof_pf.parquet", columns=["period", "DeviceId", "detector", "wgroup",
                                                          "pred_phase", "pred_function"])
    PF = {k: (dict(zip(g.detector, g.pred_phase)), dict(zip(g.detector, g.pred_function)))
          for k, g in pf.groupby(["period", "DeviceId", "wgroup"])}
    EXP = HB.expected_channels()


def _shuffle(ph, fn, seed):
    rng = np.random.default_rng(seed)
    k = list(ph)
    p = rng.permutation(len(k))
    return ({k[i]: ph[k[j]] for i, j in enumerate(p)}, {k[i]: fn.get(k[j]) for i, j in enumerate(p)})


def slim(h):
    x = h[[c for c in KEEP + [c for c in h if c.startswith("s_")] if c in h]].copy()
    return x


def one(dev):
    out = []
    p = EVR["stg"] / f"DeviceId={dev}"
    if p.is_dir():
        ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
        for w, (s, hrs) in H.EVAL_WIN["stg"].items():
            t0 = pd.Timestamp(s)
            t1 = t0 + pd.Timedelta(hours=hrs)
            if w == "full":
                t1 = pd.Timestamp(FULL["stg"][1])
            wl = w.split("_")[0]
            ph, fn = PF.get(("stg", dev, "full"), ({}, {}))
            phw, fnw = PF.get(("stg", dev, WG[wl]), ({}, {}))
            modes = {"v3_blind": (None, None), "v3_phase": (ph, None), "v3_pf": (ph, fn), "v3_pf_w": (phw, fnw),
                     "v3_shuf": _shuffle(ph, fn, hash((dev, w)) % 2**32)}
            for m, (a, b) in modes.items():
                h = hc.health(ev, t0, t1, EXP.get(dev), a or None, b or None)
                x = slim(h)
                x.insert(0, "mode", m)
                x.insert(0, "window", w)
                x.insert(0, "DeviceId", dev)
                if w == "full" and m == "v3_pf":
                    x["bad_periods"] = h.bad_periods.map(lambda L: json.dumps(
                        [{**d, "start": str(d["start"]), "end": str(d["end"])} for d in L]))
                    x["reason"] = h.reason
                out.append(x)
    # the deliverable for Dec 2024 (the other training period)
    p = EVR["dec"] / f"DeviceId={dev}"
    if p.is_dir():
        ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
        ph, fn = PF.get(("dec", dev, "full"), ({}, {}))
        h = hc.health(ev, *FULL["dec"], EXP.get(dev), ph or None, fn or None)
        x = slim(h)
        x["bad_periods"] = h.bad_periods.map(lambda L: json.dumps(
            [{**d, "start": str(d["start"]), "end": str(d["end"])} for d in L]))
        x["reason"] = h.reason
        x.insert(0, "mode", "v3_pf")
        x.insert(0, "window", "dec_full")
        x.insert(0, "DeviceId", dev)
        out.append(x)
    return pd.concat(out, ignore_index=True) if out else None


def main():
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = sorted({d.lower() for d in folds.DeviceId} - locked)
    t = time.time()
    with Pool(6, initializer=init) as p:
        res = [r for r in p.imap_unordered(one, devs, chunksize=2) if r is not None]
    df = pd.concat(res, ignore_index=True)
    assert not df.DeviceId.isin(locked).any()
    df.to_parquet(OUT / "final_eval.parquet")
    d = df[(df.window.isin(["full", "dec_full"])) & (df["mode"] == "v3_pf")].copy()
    d["period"] = np.where(d.window == "dec_full", "dec", "stg")
    d = d[["period", "DeviceId", "detector", "status", "health_score", "reason", "bad_periods", "n_on", "cov_h",
           "pred_function"] + [c for c in d if c.startswith("s_")]]
    d.to_parquet(H.HB / "health_v3.parquet", index=False)
    print(df.shape, d.shape, d.groupby("period").status.value_counts().to_dict(), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
