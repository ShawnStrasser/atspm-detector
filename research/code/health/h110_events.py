"""Note 110: event pass for the v110 health-check fixes (one pass over the w40 events of the 763 training signals).

Per signal x note-79 w40 window (h96_occ.WIN), from the hi-res log only (package health_core.events_to_bins: 81 / 82,
de-duplicated, channels > 64 dropped) and the classifier's own outputs (health4/inputs.parquet: predicted phase,
function probabilities, lanes spanned), with the SAME two-pass phase dict as the package run (pass-1 bad detectors lose
their phase for everyone else; read from health108/base.parquet `status`):

  shape.parquet  per detector-window: chop15 recomputed with the package reference (check vs the saved value),
                 chop15 with the v110 reference (fix 1: phase mates >= 2 non-twin, else a non-twin phase mate of the
                 same function, else >= 2 same-function detectors elsewhere on the signal, else none = 'no yardstick'), chop15 against the whole signal (what the old fallback
                 used), reference kind, number of twin phase mates.
  b15.parquet    15-min bins (3-h and 24-h windows): own count x, rest of the signal S, v110 reference R, covered,
                 clock hour - input of the time-of-day aware count-drop check (fix 2).
  eps.parquet    every continuous ON >= 60 s (ON to the next OFF, comms gaps excluded as the package): t0, dur, n_on,
                 open_end - input of the total-stuck-time check (fix 3) and the per-episode queue test (fix 4).

CPU, --procs <= 4.  Locked_v2 asserted absent.

    python h110_events.py [--procs 4] [--limit N]   -> %DC_WORK%/health110/{shape,b15,eps}.parquet
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

import h96_occ as O  # noqa: E402
import h96_health as HH  # noqa: E402

warnings.filterwarnings("ignore")
hc = O.hc
OUT = O.DCW / "health110"
BAD1 = None
C7 = O.C7


def init(bad1):
    global BAD1
    HH.init()
    BAD1 = bad1


def refs110(dets, live, ph, fnl, tw):
    """v110 reference: >= 2 non-twin live phase mates (package rule), else >= 1 non-twin live phase mate of the SAME
    function (like with like), else >= 2 non-twin live detectors of the same function elsewhere on the signal, else
    none ('no yardstick').  Never the whole signal mixed across functions."""
    idx = np.arange(len(dets))
    ref, kind = [], []
    for i in range(len(dets)):
        ok = live & (idx != i) & ~tw[i]
        if not np.isfinite(ph[i]):
            ref.append(np.array([], int))
            kind.append("none")
            continue
        same = np.where(ok & (ph == ph[i]))[0]
        if len(same) >= 2:
            ref.append(same)
            kind.append("phase")
            continue
        sf = [j for j in same if fnl[j] is not None and fnl[j] == fnl[i]]
        if len(sf) >= 1:
            ref.append(np.array(sf, int))
            kind.append("same_fn")
            continue
        sg = [j for j in np.where(ok)[0] if fnl[j] is not None and fnl[j] == fnl[i]]
        if len(sg) >= 2:
            ref.append(np.array(sg, int))
            kind.append("same_fn_sig")
        else:
            ref.append(np.array([], int))
            kind.append("none")
    return ref, kind


def chop(a, ok, i, refi, h=8):
    if refi is None:
        return hc.rel_disp(a[i], None, ok, h)[0]
    if not len(refi):
        return np.nan
    return hc.rel_disp(a[i], a[refi].sum(0), ok, h)[0]


def one(dev_dir):
    dev = dev_dir.name.split("=", 1)[1].lower()
    e = ds.dataset(dev_dir).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    e = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
    S_rows, B_rows, E_rows = [], [], []
    for w, (s, hrs_) in O.WIN.items():
        t0 = pd.Timestamp(s)
        t1 = t0 + pd.Timedelta(hours=hrs_)
        ew = e[(e.Timestamp >= t0) & (e.Timestamp < t1)]
        if not ew.EventId.isin((81, 82)).any():
            continue
        ph, fn, ln, pc = HH.inputs(w.split("_")[0], dev)
        B = hc.events_to_bins(ew, t0, t1)
        dets = B["dets"]
        if not len(dets):
            continue
        n, cov = B["n_on"].astype(float), B["cov"]
        live = n[:, cov].sum(1) > 0
        tw = hc.twins(n, cov)
        bad = BAD1.get((dev, w), set())
        phd1 = {int(k): v for k, v in (ph or {}).items() if v is not None and np.isfinite(v)}
        phd2 = {k: v for k, v in phd1.items() if k not in bad}
        fl = hc._labels(fn or None)
        fnl = [fl.get(int(d)) for d in dets]
        a15 = hc._agg(n, 3)
        ok15 = hc._agg(cov[None].astype(float), 3)[0] == 3
        idx = np.arange(len(dets))
        refs = {}
        for tag, phd in (("p1", phd1), ("p2", phd2)):
            r_old, k_old, phv = hc._refs(dets, live, phd, tw)
            r_new, k_new = refs110(dets, live, phv, fnl, tw)
            refs[tag] = (r_old, k_old, r_new, k_new, phv)
        hb = B["hour"][: len(ok15) * 3: 3]
        for i, d in enumerate(dets):
            d = int(d)
            tag = "p1" if (d in bad or not bad or not phd1) else "p2"
            r_old, k_old, r_new, k_new, phv = refs[tag]
            enough = n[i, cov].sum() >= 30
            sig = np.where(live & (idx != i) & ~tw[i])[0]
            twm = int((tw[i] & np.isfinite(phv) & (phv == phv[i])).sum()) if np.isfinite(phv[i]) else 0
            S_rows.append(dict(DeviceId=dev, window=w, detector=d, pass_used=tag, n_on5=float(n[i, cov].sum()),
                               ref_old=k_old[i], n_ref_old=len(r_old[i]), ref110=k_new[i], n_ref110=len(r_new[i]),
                               ref110_dets="+".join(str(int(dets[j])) for j in r_new[i]), n_twin_ph=twm,
                               chop_old=chop(a15, ok15, i, r_old[i]) if enough and len(r_old[i]) else np.nan,
                               chop110=chop(a15, ok15, i, r_new[i]) if enough else np.nan,
                               chop_sig=chop(a15, ok15, i, sig) if enough and len(sig) else np.nan))
            if not w.startswith("m30"):
                Ri = a15[r_new[i]].sum(0) if len(r_new[i]) else np.full(a15.shape[1], np.nan)
                B_rows.append(pd.DataFrame({"DeviceId": dev, "window": w, "detector": d, "b": np.arange(a15.shape[1]),
                                            "x": a15[i], "S": a15.sum(0) - a15[i], "R": Ri, "ok": ok15, "hour": hb}))
        ep = hc.on_episodes(ew, t0, t1, min_s=60.0)
        if len(ep):
            ep = ep.assign(DeviceId=dev, window=w)
            E_rows.append(ep)
    S = pd.DataFrame(S_rows)
    Bf = pd.concat(B_rows, ignore_index=True) if B_rows else None
    E = pd.concat(E_rows, ignore_index=True) if E_rows else None
    return S, Bf, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(O.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    folds = set(pd.read_csv(O.DCW / "folds_v4.csv").DeviceId.str.lower())
    dirs = [p for p in sorted(O.EVD.iterdir()) if p.name.split("=", 1)[1].lower() in folds - locked]
    assert not any(p.name.split("=", 1)[1].lower() in locked for p in dirs)
    if a.limit:
        dirs = dirs[:a.limit]
    X = pd.read_parquet(O.DCW / "health108" / "base.parquet", columns=["DeviceId", "window", "detector", "status"])
    bad1 = {k: set(g.detector.astype(int)) for k, g in X[X.status.eq("bad")].groupby(["DeviceId", "window"])}
    t = time.time()
    with Pool(min(a.procs, 4), initializer=init, initargs=(bad1,)) as pool:
        res = pool.map(one, dirs, chunksize=2)
    sfx = "" if not a.limit else "_smoke"
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(OUT / f"shape{sfx}.parquet")
    pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True).to_parquet(OUT / f"b15{sfx}.parquet")
    pd.concat([r[2] for r in res if r[2] is not None], ignore_index=True).to_parquet(OUT / f"eps{sfx}.parquet")
    print(len(dirs), "signals", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
