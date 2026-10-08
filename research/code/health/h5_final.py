"""Note 47: run health_core.health() v5 on every training signal and window (note 38's harness, h4_final's
inputs), one mode per change so each can be judged on its own.  Locked never read.

Modes (same out-of-fold inputs as note 46):
    v5          continuous ON (a) + short ONs (b) + night drop (b'); grading unchanged
    v5_gate[N]  v5 + a continuous ON counts as stuck only if >= N (default 30) ONs were expected meanwhile
    v5_grade    v5 + two independent findings = bad (c)
    v5-min_on   v5 without the short-ON check
    v5-night    v5 without the night-drop check
    v5_a        only (a): both new checks off
    v5_shuf     CONTROL: the short-ON and night-drop statistics permuted among the signal's detectors
v4 for comparison = %DC_WORK%/health4/final_eval_final.parquet (mode "final").

    python h5_final.py [--modes ...] -> %DC_WORK%/health5/final_eval.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h4_final as F  # noqa: E402
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health5"
ALL_MODES = ["v5", "v5_gate", "v5_grade", "v5-min_on", "v5-night", "v5_a", "v5_shuf"]
KEEP = F.KEEP + ["n_families", "ep_lam", "n_dur", "short1", "short2", "short_blk_min", "short_blk_max", "short_blk",
                 "night_ratio", "night_n", "night_exp", "night_med_dur", "med_dur", "dur_max"]
SHUF = ["n_dur", "short1", "short2", "short_blk_min", "short_blk_max", "short_blk", "night_ratio", "night_n",
        "night_exp", "night_ref", "night_med_dur"]
MODES = None
_rules = hc.rules


def init(modes):
    global MODES
    F.init(["final"])
    MODES = modes


def run(ev, t0, t1, dev, mode, ph, fn, ln, pc, seed):
    opts = dict(hc.OPTS)
    lam = hc.STUCK_LAM
    try:
        hc.OPTS.update(min_on=True, night_drop=True, grade2=False)
        if mode.startswith("v5_gate"):
            hc.STUCK_LAM = float(mode[7:] or 30)
        if mode == "v5_grade":
            hc.OPTS["grade2"] = True
        if mode in ("v5-min_on", "v5_a"):
            hc.OPTS["min_on"] = False
        if mode in ("v5-night", "v5_a"):
            hc.OPTS["night_drop"] = False
        if mode == "v5_shuf":
            rng = np.random.default_rng(seed)

            def shuf(st, B):
                st = st.copy()
                p = rng.permutation(len(st))
                for c in SHUF:
                    if c in st:
                        st[c] = st[c].to_numpy()[p]
                return _rules(st, B)
            hc.rules = shuf
        return hc.health(ev, t0, t1, F.EXP.get(dev), ph or None, fn or None, ln or None, pc or None)
    finally:
        hc.OPTS.clear()
        hc.OPTS.update(opts)
        hc.STUCK_LAM = lam
        hc.rules = _rules


def one(dev):
    out = []
    p = F.EVR["stg"] / f"DeviceId={dev}"
    if not p.is_dir():
        return None
    ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    ph, fn, ln, pc = F.inputs("stg", dev)
    for w, (s, hrs) in H.EVAL_WIN["stg"].items():
        t0 = pd.Timestamp(s)
        t1 = pd.Timestamp(F.FULL["stg"][1]) if w == "full" else t0 + pd.Timedelta(hours=hrs)
        for m in MODES:
            h = run(ev, t0, t1, dev, m, ph, fn, ln, pc, hash((dev, w)) % 2**32)
            x = h[[c for c in KEEP + [c for c in h if c.startswith("s_")] if c in h]].copy()
            if m in ("v5", "v5_grade", "v5_gate"):
                x["bad_periods"] = F.periods(h)
                x["reason"] = h.reason
            x.insert(0, "mode", m)
            x.insert(0, "window", w)
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
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with Pool(a.procs, initializer=init, initargs=(modes,)) as p:
        res = []
        for k, r in enumerate(p.imap_unordered(one, devs, chunksize=2)):
            if r is not None:
                res.append(r)
            if k % 100 == 0:
                print(k, f"{time.time() - t:.0f}s", flush=True)
    df = pd.concat(res, ignore_index=True)
    assert not df.DeviceId.isin(locked).any()
    df.to_parquet(OUT / a.out)
    print(df.shape, f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
