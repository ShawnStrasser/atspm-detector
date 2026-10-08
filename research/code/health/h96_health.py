"""Note 96: re-run the CURRENT package health (final_v3_candidate_v4f health_core, note-83 tick fix included) on the
note-79 w40 windows with the note-79 inputs (stg OOF phase / function probabilities / lanes / phase confidence of the
matching length, health4/inputs.parquet), keeping every statistic behind the rules (stuck episode, silent stretch,
level change point ...) for the note-96 context resolver.  Training signals only (folds_v4 minus locked_v2, asserted).
Hi-res log only; no fault events, no prints / config.  CPU, --procs <= 4.

    python h96_health.py [--procs 4] [--limit N] [--twopass]   -> %DC_WORK%/health96/health.parquet (health2.parquet)
--twopass (R9): after pass 1, the detectors called bad lose their predicted phase, so they no longer serve as the
yardstick of their phase mates, and every other detector is judged again.
"""
from __future__ import annotations

import argparse
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import h96_occ as O

warnings.filterwarnings("ignore")
hc = O.hc
IN = None
TWOPASS = False


def init(two=False):
    global IN, TWOPASS
    TWOPASS = two
    x = pd.read_parquet(O.DCW / "health4" / "inputs.parquet")
    x = x[(x.period == "stg") & x.wgroup.isin(["m30", "h3", "h24"])]
    x["DeviceId"] = x.DeviceId.str.lower()
    IN = {k: g for k, g in x.groupby(["wgroup", "DeviceId"])}


def inputs(wg, dev):
    g = IN.get((wg, dev))
    if g is None:
        return {}, {}, {}, {}
    ph = dict(zip(g.detector.astype(int), g.pred_phase.astype(float)))
    fn = {int(d): {c: float(v) for c, v in zip(O.C7, row)} for d, row in zip(g.detector, g[[f"p_{c}" for c in O.C7]].to_numpy())}
    return ph, fn, dict(zip(g.detector.astype(int), g.n_lanes_spanned)), dict(zip(g.detector.astype(int), g.top_prob))


def one(dev_dir):
    dev = dev_dir.name.split("=", 1)[1].lower()
    e = ds.dataset(dev_dir).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    e = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
    out = []
    for w, (s, h) in O.WIN.items():
        t0 = pd.Timestamp(s)
        t1 = t0 + pd.Timedelta(hours=h)
        ph, fn, ln, pc = inputs(w.split("_")[0], dev)
        try:
            x = hc.health(e, t0, t1, None, ph or None, fn or None, ln or None, pc or None)
        except Exception as exc:  # noqa: BLE001
            print("ERR", dev, w, type(exc).__name__, exc, flush=True)
            continue
        if not len(x):
            continue
        x = x.copy()
        if TWOPASS:
            # R9: re-judge every other detector with the detectors pass 1 called bad removed from the phase yardstick
            bad = set(x.loc[x.status == "bad", "detector"].astype(int))
            if bad and ph:
                ph2 = {k: (np.nan if k in bad else v) for k, v in ph.items()}
                y = hc.health(e, t0, t1, None, ph2, fn or None, ln or None, pc or None)
                y = y[~y.detector.isin(bad)]
                x = pd.concat([x[x.detector.isin(bad)], y], ignore_index=True)
            x["pass2"] = bool(bad and ph)
        for c in x.columns:
            if x[c].dtype == object and c not in ("status", "reason", "not_checked", "ref_kind", "drop_ref",
                                                  "night_ref", "fn"):
                x[c] = x[c].astype(str)
        x.insert(0, "window", w)
        x.insert(0, "DeviceId", dev)
        out.append(x)
    return pd.concat(out, ignore_index=True) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--twopass", action="store_true")
    a = ap.parse_args()
    global TWOPASS
    TWOPASS = a.twopass
    locked = set(pd.read_csv(O.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    folds = set(pd.read_csv(O.DCW / "folds_v4.csv").DeviceId.str.lower())
    dirs = [p for p in sorted(O.EVD.iterdir()) if p.name.split("=", 1)[1].lower() in folds - locked]
    assert not any(p.name.split("=", 1)[1].lower() in locked for p in dirs)
    if a.limit:
        dirs = dirs[:a.limit]
    t = time.time()
    with Pool(a.procs, initializer=init, initargs=(a.twopass,)) as pool:
        res = pool.map(one, dirs, chunksize=2)
    H = pd.concat([r for r in res if r is not None], ignore_index=True)
    for c in H.columns:
        if H[c].dtype == object:
            H[c] = H[c].astype(str)
    H.to_parquet(O.OUT / (("health2.parquet" if a.twopass else "health.parquet") if not a.limit else "health_smoke.parquet"))
    print(len(dirs), "signals", len(H), "rows", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
