"""Note 108: per-detector actuation statistics for the baseline study of NORMAL behaviour (one event pass).

For every training signal (folds_v4 minus locked_v2, asserted) and every note-79 w40 window (h96_occ.WIN: 4 x 30 min,
2 x 3 h, 2 x 24 h; Sat 26 - Mon 28 Sep 2026) this computes, from the hi-res log only (81 / 82; de-duplicated; channels
> 64 dropped), per detector:

  n_on        ON events
  rep_frac    share of ON events logged again while already ON (no OFF in between) - a logging pattern, not a vehicle
  clean ONs   ONs whose previous and next event on the channel is an OFF: n_clean, dur_p10 / p50 / p90 (s),
              short_frac (<= 0.2 s, i.e. <= 2 ticks)
  gaps        OFF -> next ON: gap_p10 / gap_p50 (s), chat_frac (< 3 ticks = < 0.3 s)
  ON -> ON    ioi_lt05 / ioi_lt1 (< 5 / < 10 ticks), burst_frac (ONs in runs of >= 5 with every interval < 1 s)
  max_on_s    longest continuous ON (first ON -> next OFF), the package's definition
  rep_time_s  time held ON by continuous ONs that contain an ON logged again without an OFF

Intervals that span a comms gap (> 120 s with no event of any code) are dropped.  No fault events, no prints / config.
CPU, --procs <= 6.

    python h108_events.py [--procs 6] [--limit N]   -> %DC_WORK%/health108/act.parquet
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
OUT = O.DCW / "health108"
GAP_S = 120.0


def _ticks(x):
    return np.rint(np.asarray(x, float) * 10.0)


def _runs_of(mask, min_len):
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]
    L = b - a
    k = L >= min_len
    return int(L[k].sum()), int(k.sum())


def stats(e, t0, t1):
    T = (t1 - t0).total_seconds()
    t = (e.Timestamp - t0).dt.total_seconds().to_numpy()
    eid, par = e.EventId.to_numpy().astype(int), e.Parameter.to_numpy().astype(int)
    ut = np.unique(t)
    gi = np.diff(ut) > GAP_S
    g0 = ut[:-1][gi]

    def spans_gap(a, b):
        if not len(g0):
            return np.zeros(len(a), bool)
        k = np.searchsorted(g0, a)
        kk = np.minimum(k, len(g0) - 1)
        return (k < len(g0)) & (g0[kk] < b)

    dm = np.isin(eid, (81, 82)) & (par <= 64)
    td, ed, pd_ = t[dm], eid[dm], par[dm]
    o = np.lexsort((ed != 82, td, pd_))
    td, ed, pd_ = td[o], ed[o], pd_[o]
    rows = []
    for c in np.unique(pd_):
        m = pd_ == c
        x, ev = td[m], ed[m]
        on = ev == 82
        n_on = int(on.sum())
        if n_on == 0:
            continue
        prev_e = np.r_[0, ev[:-1]]
        next_e = np.r_[ev[1:], 0]
        next_t = np.r_[x[1:], np.nan]
        prev_t = np.r_[np.nan, x[:-1]]
        rep = on & (prev_e == 82)
        clean = on & (prev_e != 82) & (next_e == 81)
        dur = (next_t - x)[clean]
        dur = dur[~spans_gap(x[clean], x[clean] + dur)]
        gm = on & (prev_e == 81)
        gap = (x - prev_t)[gm]
        gap = gap[~spans_gap(x[gm] - gap, x[gm])]
        xo = x[on]
        ioi = np.diff(xo)
        ioi[spans_gap(xo[:-1], xo[1:])] = np.nan
        jt = _ticks(ioi)
        k_, nr = _runs_of(np.nan_to_num(jt, nan=990.0) < 10, 4)
        # continuous ONs: first ON (prev not ON) -> next OFF
        st = np.where(on & (prev_e != 82))[0]
        offs = np.where(ev == 81)[0]
        kk = np.searchsorted(offs, st)
        okk = kk < len(offs)
        cl = x[offs[kk[okk]]] - x[st[okk]] if okk.any() else np.array([])
        if okk.any():                                   # an ON over a comms gap is not counted (as the package)
            cl = np.where(spans_gap(x[st[okk]], x[offs[kk[okk]]]), 0.0, cl)
        # time held ON by continuous ONs that contain an ON logged again (no OFF between): a logging pattern
        if okk.any():
            cr = np.cumsum(rep)
            nrep = cr[offs[kk[okk]] - 1] - cr[st[okk]]
            rep_time = float(cl[nrep > 0].sum())
        else:
            rep_time = 0.0
        r = dict(detector=int(c), n_on=n_on, rep_frac=float(rep.sum() / n_on), n_clean=int(len(dur)),
                 max_on_s=float(cl.max()) if len(cl) else np.nan, rep_time_s=rep_time)
        if len(dur) >= 5:
            q = np.quantile(dur, [.1, .5, .9])
            r.update(dur_p10=q[0], dur_p50=q[1], dur_p90=q[2], short_frac=float((_ticks(dur) <= 2).mean()))
        if len(gap) >= 5:
            r.update(gap_p10=float(np.quantile(gap, .1)), gap_p50=float(np.median(gap)),
                     chat_frac=float((_ticks(gap) < 3).mean()))
        f = np.isfinite(jt)
        if f.sum() >= 5:
            r.update(ioi_lt05=float((jt[f] < 5).mean()), ioi_lt1=float((jt[f] < 10).mean()),
                     burst_frac=(k_ + nr) / n_on)
        rows.append(r)
    return rows


def one(dev_dir):
    dev = dev_dir.name.split("=", 1)[1].lower()
    e = ds.dataset(dev_dir).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    e = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
    out = []
    for w, (s, h) in O.WIN.items():
        t0 = pd.Timestamp(s)
        t1 = t0 + pd.Timedelta(hours=h)
        ew = e[(e.Timestamp >= t0) & (e.Timestamp < t1)]
        if not ew.EventId.isin((81, 82)).any():
            continue
        for r in stats(ew, t0, t1):
            r.update(DeviceId=dev, window=w)
            out.append(r)
    return pd.DataFrame(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(O.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    folds = set(pd.read_csv(O.DCW / "folds_v4.csv").DeviceId.str.lower())
    dirs = [p for p in sorted(O.EVD.iterdir()) if p.name.split("=", 1)[1].lower() in folds - locked]
    assert not any(p.name.split("=", 1)[1].lower() in locked for p in dirs)
    if a.limit:
        dirs = dirs[:a.limit]
    t = time.time()
    with Pool(min(a.procs, 6)) as pool:
        res = pool.map(one, dirs, chunksize=4)
    A = pd.concat([r for r in res if r is not None], ignore_index=True)
    A.to_parquet(OUT / ("act.parquet" if not a.limit else "act_smoke.parquet"))
    print(len(dirs), "signals", len(A), "rows", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
