"""Note 46: run health_core.health() v4 on every training signal and window (note 38's harness), one mode per
idea so each can be judged on its own.

Modes per (signal, window), all with the SAME out-of-fold inputs (h4_inputs.py, the 66-h "site answer"):
    v3        every v4 option off (= the note-43 scorer on these inputs)
    v4        every option on
    v4-X      v4 without option X (lanes, partner, recur, rapid_hour, conf)
    v4_shuf   CONTROL: lanes spanned and phase confidence permuted among the signal's detectors
Sept 2026 windows of note 38 (hb_data.EVAL_WIN); channel list = hb_build.expected_channels().  Also writes the
deliverable: full-window v4 for both periods -> %DC_WORK%/health/health_v4.parquet.  Locked never read.

    python h4_final.py [--modes v3,v4,...] -> %DC_WORK%/health4/final_eval.parquet
"""
from __future__ import annotations

import argparse
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
OUT = H.DCW / "health4"
EVR = {"stg": H.DCW / "official" / "stg" / "cache" / "events", "dec": H.DCW / "cache" / "events"}
FULL = {"stg": ("2026-09-18 16:15", "2026-09-21 10:25"), "dec": ("2024-12-02 00:00", "2024-12-05 00:00")}
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
OPTS = list(hc.OPTS)
DEFAULTS = dict(hc.OPTS)
ALL_MODES = ["v3", "v4"] + [f"v4-{k}" for k in OPTS if k != "group"] + ["v4_shuf"]
KEEP = ["detector", "health_score", "status", "n_on", "cov_h", "chop15", "chop_lim", "ref_kind", "drop_lam", "drop_rec",
        "ep_dur", "ep_co", "ep_rec", "ep_flow", "ep_days", "ep_partner", "rapid", "rapid_h", "n_days", "rec_share",
        "rec_ndates", "spk_plaus", "spk_fast", "spk_dur_ratio", "f_conf", "ph_conf", "p_bike", "lanes_spanned",
        "pred_function"]
IN = EXP = MODES = None


def init(modes):
    global IN, EXP, MODES
    x = pd.read_parquet(OUT / "inputs.parquet")
    x = x[x.wgroup == "full"]
    IN = {k: g for k, g in x.groupby(["period", "DeviceId"])}
    EXP = HB.expected_channels()
    MODES = modes


def inputs(per, dev):
    g = IN.get((per, dev))
    if g is None:
        return {}, {}, {}, {}
    ph = dict(zip(g.detector, g.pred_phase.astype(float)))
    fn = {int(d): {c: float(v) for c, v in zip(C7, row)} for d, row in zip(g.detector, g[[f"p_{c}" for c in C7]].to_numpy())}
    return ph, fn, dict(zip(g.detector, g.n_lanes_spanned)), dict(zip(g.detector, g.top_prob))


def run(ev, t0, t1, dev, mode, ph, fn, ln, pc, seed):
    on = {k: mode != "v3" for k in OPTS}
    if mode == "final":                    # the kept configuration = health_core's defaults
        on = dict(DEFAULTS)
    if mode.startswith("v4-"):
        on[mode[3:]] = False
    if mode == "v4_shuf":
        rng = np.random.default_rng(seed)
        k = list(ln)
        ln = dict(zip(k, np.array([ln[x] for x in k])[rng.permutation(len(k))]))
        k = list(pc)
        pc = dict(zip(k, np.array([pc[x] for x in k])[rng.permutation(len(k))]))
    old = dict(hc.OPTS)
    hc.OPTS.update(on)
    try:
        return hc.health(ev, t0, t1, EXP.get(dev), ph or None, fn or None, ln or None, pc or None)
    finally:
        hc.OPTS.clear()
        hc.OPTS.update(old)


def slim(h):
    return h[[c for c in KEEP + [c for c in h if c.startswith("s_")] if c in h]].copy()


def periods(h):
    return h.bad_periods.map(lambda L: json.dumps([{**d, "start": str(d["start"]), "end": str(d["end"])} for d in L]))


def one(dev):
    out = []
    p = EVR["stg"] / f"DeviceId={dev}"
    if p.is_dir():
        ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
        ph, fn, ln, pc = inputs("stg", dev)
        for w, (s, hrs) in H.EVAL_WIN["stg"].items():
            t0 = pd.Timestamp(s)
            t1 = pd.Timestamp(FULL["stg"][1]) if w == "full" else t0 + pd.Timedelta(hours=hrs)
            for m in MODES:
                h = run(ev, t0, t1, dev, m, ph, fn, ln, pc, hash((dev, w)) % 2**32)
                x = slim(h)
                if m in ("v3", "v4", "final"):
                    x["bad_periods"] = periods(h)
                    x["reason"] = h.reason
                    x["not_checked"] = h.not_checked
                x.insert(0, "mode", m)
                x.insert(0, "window", w)
                x.insert(0, "DeviceId", dev)
                out.append(x)
    if "final" in MODES:                   # the deliverable for Dec 2024 (the other training period)
        p = EVR["dec"] / f"DeviceId={dev}"
        if p.is_dir():
            ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
            ph, fn, ln, pc = inputs("dec", dev)
            h = run(ev, *[pd.Timestamp(x) for x in FULL["dec"]], dev, "final", ph, fn, ln, pc, 0)
            x = slim(h)
            x["bad_periods"] = periods(h)
            x["reason"] = h.reason
            x["not_checked"] = h.not_checked
            x.insert(0, "mode", "final")
            x.insert(0, "window", "dec_full")
            x.insert(0, "DeviceId", dev)
            out.append(x)
    return pd.concat(out, ignore_index=True) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="final_eval.parquet")
    a = ap.parse_args()
    modes = a.modes.split(",")
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = sorted({d.lower() for d in folds.DeviceId} - locked)
    if a.limit:
        devs = devs[:: max(1, len(devs) // a.limit)][: a.limit]
    t = time.time()
    with Pool(a.procs, initializer=init, initargs=(modes,)) as p:
        res = []
        for k, r in enumerate(p.imap_unordered(one, devs, chunksize=2)):
            if r is not None:
                res.append(r)
            if k % 50 == 0:
                print(k, f"{time.time() - t:.0f}s", flush=True)
    df = pd.concat(res, ignore_index=True)
    assert not df.DeviceId.isin(locked).any()
    df.to_parquet(OUT / a.out)
    if "final" in modes and not a.limit:
        d = df[df.window.isin(["full", "dec_full"]) & (df["mode"] == "final")].copy()
        d["period"] = np.where(d.window == "dec_full", "dec", "stg")
        d = d[["period", "DeviceId", "detector", "status", "health_score", "reason", "not_checked", "bad_periods",
               "n_on", "cov_h", "pred_function", "f_conf", "ph_conf", "lanes_spanned"] + [c for c in d if c.startswith("s_")]]
        d.to_parquet(H.HB / "health_v4.parquet", index=False)
        print(d.groupby("period").status.value_counts().to_dict())
    print(df.shape, f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
