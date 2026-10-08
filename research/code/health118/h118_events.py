"""Note 118a: one event pass for the v118 health-check fixes (stuck / extension time / erratic counts / count drops).

Per training signal (folds_v4 minus locked_v2, asserted) x note-79 w40 window, from the hi-res log only (81 / 82,
exact duplicates dropped, channels > 64 dropped; comms gap = > 120 s with no event of any allowed code; no fault
events).  Pure numpy after the parquet read (no pandas frame of events).  Per detector:

  n_on, n_rep        ONs; ONs logged again with no OFF since the previous ON (extension-time pattern, not vehicles)
  n_start            ONs that START a continuous ON (= n_on - n_rep): the vehicle actuations
  ioi_lt05_c / ioi_lt1_c / burst_frac_c   ON->ON < 0.5 s / < 1 s and burst share (>= 5 starts each < 1 s apart),
                     on STARTS only, so re-logged ONs never count as 'too fast'
  max5_c             most starts in one fixed 5-min bin
  n_cont, n_ge2, n_ge5, n_ge60, long_max_s   continuous ONs (start -> next OFF, comms gaps excluded): number, number
                     >= 2 s / >= 5 s / >= 60 s, the longest - input of the 'not set to pulse' configuration note

    python h118_events.py [--procs 4] [--limit N]   -> %DC_WORK%/s118/act118.parquet
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
EVD = DCW / "health4" / "w40_events"
OUT = DCW / "s118"
ALLOWED = (1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173)
WIN = {"m30_a": ("2026-09-27 08:00", 0.5), "m30_b": ("2026-09-27 12:00", 0.5), "m30_c": ("2026-09-27 17:00", 0.5),
       "m30_d": ("2026-09-28 07:30", 0.5), "h3_a": ("2026-09-27 12:00", 3), "h3_b": ("2026-09-28 06:00", 3),
       "h24_a": ("2026-09-27 00:00", 24), "h24_b": ("2026-09-28 00:00", 24)}
GAP_S = 120.0


def _runs(mask, brk):
    """per element: length of the run of True it belongs to; runs broken where brk is True."""
    m = mask & ~brk
    st = m & ~np.r_[False, m[:-1]]
    rid = np.cumsum(st) * m
    cnt = np.bincount(rid)
    cnt[0] = 0
    return cnt[rid]


def stats(t, eid, par, T):
    """t seconds from window start (float), sorted arbitrary; returns list of dicts per detector."""
    ut = np.unique(t)
    g0 = ut[:-1][np.diff(ut) > GAP_S]

    def spans_gap(a, b):
        if not len(g0):
            return np.zeros(len(a), bool)
        k = np.searchsorted(g0, a)
        return (k < len(g0)) & (g0[np.minimum(k, len(g0) - 1)] < b)

    dm = np.isin(eid, (81, 82)) & (par <= 64)
    td, ed, pd_ = t[dm], eid[dm], par[dm]
    o = np.lexsort((ed != 82, td, pd_))
    td, ed, pd_ = td[o], ed[o], pd_[o]
    if not len(td):
        return []
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    on = ed == 82
    rep = on & same_prev & np.r_[False, ed[:-1] == 82]
    start = on & ~rep
    # continuous ON: start -> next OFF of the same channel (T if none)
    offs = np.where(ed == 81)[0]
    n = len(td)
    k = np.searchsorted(offs, np.arange(n))
    p = offs[np.minimum(k, max(len(offs) - 1, 0))] if len(offs) else np.zeros(n, int)
    okk = (k < len(offs)) & (pd_[p] == pd_) if len(offs) else np.zeros(n, bool)
    end = np.where(okk, td[p] if len(offs) else T, T)
    si = np.where(start)[0]
    cdur = end[si] - td[si]
    cdur = np.where(spans_gap(td[si], end[si]), np.nan, cdur)
    sch, st_t = pd_[si], td[si]
    # ON->ON on starts, same channel, not over a comms gap
    same = np.r_[False, sch[1:] == sch[:-1]]
    ioi = np.r_[np.nan, np.diff(st_t)]
    ioi[~same] = np.nan
    ioi[np.r_[False, spans_gap(st_t[:-1], st_t[1:])]] = np.nan
    jt = np.rint(ioi * 10.0)
    fast = np.nan_to_num(jt, nan=990.0) < 10
    rl = _runs(fast, ~same)                         # run of < 1 s intervals; >= 4 intervals = burst of >= 5 ONs
    inb = (rl >= 4)
    # ONs in a burst = intervals in bursts + one per burst
    nb5 = (st_t // 300).astype(np.int64)
    rows = []
    chans, first = np.unique(sch, return_index=True)
    last = np.r_[first[1:], len(sch)]
    n_on_c = np.bincount(pd_[on], minlength=65)
    n_rep_c = np.bincount(pd_[rep], minlength=65)
    for c, a, b in zip(chans, first, last):
        j = jt[a:b]
        f = np.isfinite(j)
        ib = inb[a:b]
        nbur = int(ib.sum()) + int((ib & ~np.r_[False, ib[:-1]]).sum())
        ns = b - a
        cd = cdur[a:b]
        cf = cd[np.isfinite(cd)]
        r = dict(detector=int(c), n_on=int(n_on_c[c]), n_rep=int(n_rep_c[c]), n_start=int(ns),
                 max5_c=int(np.bincount(nb5[a:b] - nb5[a:b].min()).max()),
                 n_cont=int(len(cf)), n_ge2=int((cf >= 2).sum()), n_ge5=int((cf >= 5).sum()),
                 n_ge60=int((cf >= 60).sum()), long_max_s=float(cf.max()) if len(cf) else np.nan)
        if f.sum() >= 5:
            r.update(ioi_lt05_c=float((j[f] < 5).mean()), ioi_lt1_c=float((j[f] < 10).mean()),
                     burst_frac_c=nbur / ns)
        rows.append(r)
    return rows


def one(dev_dir):
    dev = dev_dir.name.split("=", 1)[1].lower()
    tb = ds.dataset(dev_dir).to_table(filter=ds.field("EventId").isin(list(ALLOWED)),
                                      columns=["Timestamp", "EventId", "Parameter"])
    ts = pc.cast(tb["Timestamp"], "int64").to_numpy()
    eid = tb["EventId"].to_numpy().astype(np.int64)
    par = tb["Parameter"].to_numpy().astype(np.int64)
    # exact duplicates out
    o = np.lexsort((par, eid, ts))
    ts, eid, par = ts[o], eid[o], par[o]
    keep = np.r_[True, (np.diff(ts) != 0) | (np.diff(eid) != 0) | (np.diff(par) != 0)]
    ts, eid, par = ts[keep], eid[keep], par[keep]
    out = []
    for w, (s, h) in WIN.items():
        t0 = pd.Timestamp(s).value // 1000
        t1 = t0 + int(h * 3600e6)
        m = (ts >= t0) & (ts < t1)
        if not np.isin(eid[m], (81, 82)).any():
            continue
        for r in stats((ts[m] - t0) / 1e6, eid[m], par[m], h * 3600.0):
            r.update(DeviceId=dev, window=w)
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    folds = set(pd.read_csv(DCW / "folds_v4.csv").DeviceId.str.lower())
    dirs = [p for p in sorted(EVD.iterdir()) if p.name.split("=", 1)[1].lower() in folds - locked]
    assert not any(p.name.split("=", 1)[1].lower() in locked for p in dirs)
    if a.limit:
        dirs = dirs[:a.limit]
    t = time.time()
    with Pool(min(a.procs, 4)) as pool:
        res = pool.map(one, dirs, chunksize=4)
    A = pd.DataFrame([r for rr in res for r in rr])
    A.to_parquet(OUT / ("act118.parquet" if not a.limit else "act118_smoke.parquet"))
    print(len(dirs), "signals", len(A), "rows", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
