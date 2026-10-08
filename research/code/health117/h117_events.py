"""Note 117: RED vs GREEN event pass - actuations, fast ONs and occupancy by the colour of the detector's own phase.

One pass over the w40 events of the 763 training signals (locked_v2 asserted absent), every note-79 window (4 x 30 min,
2 x 3 h, 2 x 24 h).  Hi-res log only (81 / 82 de-duplicated, channels > 64 dropped; colour 1 / 8 / 9 / 10 of the phase)
plus the classifier's own predicted phase (health4/inputs.parquet, matching window length) - the phase number is only a
join key to that phase's colour events, never a feature.

Vehicle ONs = ONs that START a continuous ON (previous event on the channel is an OFF).  An ON logged again without an
OFF (extension) is counted separately (n_inner) and never as a fast actuation (user, Oct 7 point 4).

Colour state of the phase at each ON: G (begin green 1 -> begin yellow 8), Y (8 -> end yellow 9 / red clearance 10),
Rg (first 2 s of red: late-yellow tails), R (rest of red, up to the next 1), U (unknown: no colour, before the first
change, or over a comms gap > 120 s).  Occupancy is split exactly over the states with cumulative state-time curves.

  bins117.parquet   per detector-window x 5-min bin: ONs per state, fast ONs (ON->ON < 10 ticks) G+Y / R, chatter
                    (OFF->ON < 3 ticks) G+Y / R, occupancy s G+Y / R, phase seconds in G+Y / R.
  det117.parquet    per detector-window: totals per state, fast < 0.5 / < 1 s, burst ONs (>= 5 ONs all < 1 s) per
                    state, inner ONs, per-cycle flow (ONs in green+yellow / (green+yellow s), cycles >= 10 s), ONs per
                    red, longest burst.

    python h117_events.py [--procs 4] [--limit N]    -> %DC_WORK%/s117/{bins117,det117}.parquet
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
OUT = Path(os.environ["H117_OUT"]) if os.environ.get("H117_OUT") else DCW / "s117"   # note 118c: rerun elsewhere
WIN = {"m30_a": ("2026-09-27 08:00", .5), "m30_b": ("2026-09-27 12:00", .5), "m30_c": ("2026-09-27 17:00", .5),
       "m30_d": ("2026-09-28 07:30", .5), "h3_a": ("2026-09-27 12:00", 3), "h3_b": ("2026-09-28 06:00", 3),
       "h24_a": ("2026-09-27 00:00", 24), "h24_b": ("2026-09-28 00:00", 24)}
ALLOWED = [1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173]   # no fault events 83-88
COLOUR = {1: 0, 8: 1, 9: 2, 10: 2}                                # G, Y, R
G, Y, RG, R, U = 0, 1, 2, 3, 4
RED_GUARD = 2.0
GAP_S = 120.0
BIN_S = 300
T0 = pd.Timestamp("2026-09-26 00:00")
PH = None


def init():
    global PH
    x = pd.read_parquet(DCW / "health4" / "inputs.parquet", columns=["period", "DeviceId", "detector", "wgroup", "pred_phase"])
    x = x[(x.period == "stg") & x.wgroup.isin(["m30", "h3", "h24"])]
    x["DeviceId"] = x.DeviceId.str.lower()
    PH = {k: dict(zip(g.detector.astype(int), g.pred_phase.astype(float))) for k, g in x.groupby(["wgroup", "DeviceId"])}


def load(dev_dir):
    t = ds.dataset(dev_dir).to_table(filter=ds.field("EventId").isin(ALLOWED))
    ts = pc.cast(t["Timestamp"], "int64").to_numpy() / 1e6 - T0.value / 1e9      # s since T0
    eid = t["EventId"].to_numpy().astype(np.int16)
    par = t["Parameter"].to_numpy().astype(np.int32)
    keep = ~(((eid == 81) | (eid == 82)) & (par > 64))
    ts, eid, par = ts[keep], eid[keep], par[keep]
    o = np.lexsort((par, eid, ts))
    ts, eid, par = ts[o], eid[o], par[o]
    dup = np.r_[False, (ts[1:] == ts[:-1]) & (eid[1:] == eid[:-1]) & (par[1:] == par[:-1])]
    return ts[~dup], eid[~dup], par[~dup]


def phase_timeline(tc, ec, Tlen, g0, g1):
    """segments of one phase: boundaries b (m+1), state s (m), valid v (m).  Red split into Rg (first 2 s) + R."""
    st = np.array([COLOUR[int(e)] for e in ec], np.int8)
    keep = np.r_[True, st[1:] != st[:-1]]                         # 9 then 10 -> one red change
    tc, st = tc[keep], st[keep]
    b, s = [], []
    for i in range(len(tc)):
        a = tc[i]
        z = tc[i + 1] if i + 1 < len(tc) else Tlen
        if st[i] == 2:
            m = min(a + RED_GUARD, z)
            b += [a, m]
            s += [RG, R]
        else:
            b.append(a)
            s.append(G if st[i] == 0 else Y)
    b = np.array(b + [Tlen], float)
    s = np.array(s, np.int8)
    v = np.ones(len(s), bool)
    if len(g0):                                                   # a segment containing a comms gap is unknown
        k = np.searchsorted(g0, b[:-1])
        kk = np.minimum(k, len(g0) - 1)
        v &= ~((k < len(g0)) & (g0[kk] < b[1:]))
    return b, s, v


def cum_curves(b, s, v):
    d = np.diff(b)
    return {k: np.r_[0.0, np.cumsum(d * ((s == k) & v))] for k in (G, Y, RG, R)}


def one(dev_dir):
    dev = dev_dir.name.split("=", 1)[1].lower()
    ts_all, eid_all, par_all = load(dev_dir)
    BINS, DETS = [], []
    for w, (s0, hrs) in WIN.items():
        t0 = (pd.Timestamp(s0) - T0).total_seconds()
        Tlen = hrs * 3600.0
        i0, i1 = np.searchsorted(ts_all, [t0, t0 + Tlen])
        t, eid, par = ts_all[i0:i1] - t0, eid_all[i0:i1], par_all[i0:i1]
        dm = (eid == 81) | (eid == 82)
        if not dm.any():
            continue
        nb = int(np.ceil(Tlen / BIN_S))
        edges = np.minimum(np.arange(nb + 1) * BIN_S, Tlen)
        ut = np.unique(t)
        gi = np.diff(ut) > GAP_S
        g0, g1 = ut[:-1][gi], ut[1:][gi]
        phd = PH.get((w.split("_")[0], dev), {})
        # ---- colour timelines per phase
        cm = np.isin(eid, (1, 8, 9, 10))
        tl = {}
        for p in np.unique(par[cm & (eid == 1)]):
            mm = cm & (par == p)
            if (eid[mm] == 1).sum() < 3:
                continue
            b, s, v = phase_timeline(t[mm], eid[mm], Tlen, g0, g1)
            cc = cum_curves(b, s, v)
            secs = {k: np.diff(np.interp(edges, b, cc[k])) for k in cc}
            tl[int(p)] = (b, s, v, cc, secs)
        # ---- detector events, (channel, time, ON before OFF)
        td, ed, pd_ = t[dm], eid[dm], par[dm]
        o = np.lexsort((ed != 82, td, pd_))
        td, ed, pd_ = td[o], ed[o], pd_[o]
        n = len(td)
        same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
        prev_e = np.r_[0, ed[:-1]]
        prev_t = np.r_[np.nan, td[:-1]]
        on = ed == 82
        start = on & ~(same_prev & (prev_e == 82))
        inner = on & ~start
        offs = np.where(ed == 81)[0]
        k = np.searchsorted(offs, np.arange(n), "left")
        if len(offs):
            pp = offs[np.minimum(k, len(offs) - 1)]
            okk = (k < len(offs)) & (pd_[pp] == pd_)
            c_end = np.where(okk, td[pp], Tlen)
        else:
            c_end = np.full(n, Tlen)
        gap_off = np.where(same_prev & (prev_e == 81), td - prev_t, np.nan)
        si = np.where(start)[0]
        ts_, ch_, dur_ = td[si], pd_[si], c_end[si] - td[si]
        goff = gap_off[si]
        if len(g0) and len(si):                                   # ON over a comms gap: duration unknown
            kq = np.searchsorted(g0, ts_)
            kk = np.minimum(kq, len(g0) - 1)
            hit = (kq < len(g0)) & (g0[kk] < ts_ + dur_)
            dur_[hit] = np.nan
        n_inner = np.bincount(np.searchsorted(np.unique(pd_), pd_[inner]), minlength=len(np.unique(pd_)))
        chans = np.unique(pd_)
        for ci, c in enumerate(chans):
            m = ch_ == c
            x, du, go = ts_[m], dur_[m], goff[m]
            nn = len(x)
            ioi = np.r_[np.nan, np.diff(x)]
            with np.errstate(invalid="ignore"):
                tk = np.rint(ioi / 0.1)
                f1 = tk < 10
                f05 = tk < 5
                chat = np.rint(go / 0.1) < 3
            # bursts: runs of >= 4 consecutive fast intervals = >= 5 ONs
            rs = np.diff(np.r_[0, f1.astype(np.int8), 0])
            a_, z_ = np.where(rs == 1)[0], np.where(rs == -1)[0]
            ln = z_ - a_
            inb = np.zeros(nn, bool)
            for aa, zz in zip(a_[ln >= 4], z_[ln >= 4]):
                inb[aa - 1:zz] = True
            p = phd.get(int(c), np.nan)
            T = tl.get(int(p)) if np.isfinite(p) else None
            if T is not None:
                b, s, v, cc, secs = T
                j = np.searchsorted(b, x, "right") - 1
                okj = (j >= 0) & (j < len(s))
                jj = np.clip(j, 0, len(s) - 1)
                state = np.where(okj & v[jj], s[jj], U).astype(np.int8)
                fin = np.isfinite(du)
                end = np.minimum(x + np.nan_to_num(du), Tlen)
                occ = {kk_: np.where(fin, np.interp(end, b, cc[kk_]) - np.interp(x, b, cc[kk_]), 0.0) for kk_ in cc}
                # cycles: G segment index running count
                cyc_of_seg = np.cumsum(s == G) - 1
                cyc = np.where(okj, cyc_of_seg[jj], -1)
                nc = int(cyc_of_seg[-1]) + 1 if len(s) else 0
                segd = np.diff(b)
                gy_seg = (s == G) | (s == Y)
                r_seg = (s == RG) | (s == R)
                cvalid = np.ones(max(nc, 1), bool)
                np.logical_and.at(cvalid, np.clip(cyc_of_seg, 0, None), v | (cyc_of_seg < 0))
                gy_dur = np.bincount(np.clip(cyc_of_seg, 0, None), weights=segd * gy_seg * (cyc_of_seg >= 0), minlength=max(nc, 1))
                r_dur = np.bincount(np.clip(cyc_of_seg, 0, None), weights=segd * r_seg * (cyc_of_seg >= 0), minlength=max(nc, 1))
                gyon = (state == G) | (state == Y)
                ron = (state == RG) | (state == R)
                n_gy_c = np.bincount(np.clip(cyc[gyon], 0, None), minlength=max(nc, 1))
                n_r_c = np.bincount(np.clip(cyc[ron & (cyc >= 0)], 0, None), minlength=max(nc, 1))
                fullc = cvalid & (gy_dur >= 10) & (r_dur > 0)
                # the last cycle ends at the window end: keep only cycles whose red is complete (next G exists)
                if nc:
                    fullc[nc - 1] = False
                flow_c = n_gy_c[fullc] / gy_dur[fullc] * 3600
                ronc = n_r_c[fullc]
            else:
                state = np.full(nn, U, np.int8)
                occ = {kk_: np.zeros(nn) for kk_ in (G, Y, RG, R)}
                secs = None
                flow_c = ronc = np.array([])
            bi = np.minimum((x // BIN_S).astype(int), nb - 1)
            gy = (state == G) | (state == Y)
            rr = state == R
            cnt = lambda msk: np.bincount(bi[msk], minlength=nb)  # noqa: E731
            wsum = lambda val, msk: np.bincount(bi[msk], weights=val[msk], minlength=nb)  # noqa: E731
            bins = dict(b=np.arange(nb, dtype=np.int16),
                        n=cnt(np.ones(nn, bool)), nG=cnt(state == G), nY=cnt(state == Y), nRg=cnt(state == RG),
                        nR=cnt(rr), nU=cnt(state == U), fGY=cnt(gy & f1), fR=cnt(rr & f1), fU=cnt((state == U) & f1),
                        cGY=cnt(gy & chat), cR=cnt(rr & chat), bGY=cnt(gy & inb), bR=cnt(rr & inb),
                        # note 118c: fast ONs that are NOT chatter re-triggers (OFF -> ON < 3 ticks), chatter in U
                        xGY=cnt(gy & f1 & ~chat), xR=cnt(rr & f1 & ~chat), xU=cnt((state == U) & f1 & ~chat),
                        cU=cnt((state == U) & chat),
                        oGY=wsum(occ[G] + occ[Y], np.ones(nn, bool)), oR=wsum(occ[R], np.ones(nn, bool)),
                        oAll=wsum(np.nan_to_num(np.minimum(du, Tlen - x)), np.ones(nn, bool)))
            if secs is not None:
                bins.update(sGY=secs[G] + secs[Y], sR=secs[R], sRg=secs[RG])
            else:
                bins.update(sGY=np.zeros(nb), sR=np.zeros(nb), sRg=np.zeros(nb))
            Bdf = pd.DataFrame({k_: v_.astype(np.float32) if k_ != "b" else v_ for k_, v_ in bins.items()})
            Bdf.insert(0, "detector", np.int16(c))
            Bdf.insert(0, "window", w)
            Bdf.insert(0, "DeviceId", dev)
            BINS.append(Bdf)
            r = dict(DeviceId=dev, window=w, detector=int(c), phase=p, has_colour=T is not None, n_on=nn,
                     n_inner=int(n_inner[ci]), n_dur=int(np.isfinite(du).sum()), med_dur=float(np.nanmedian(du)) if np.isfinite(du).any() else np.nan,
                     pulse=float(np.nanmean(du[np.isfinite(du)] <= 0.15)) if np.isfinite(du).any() else np.nan,
                     n_cyc=len(flow_c),
                     cyc_flow_p50=float(np.median(flow_c)) if len(flow_c) else np.nan,
                     cyc_flow_p95=float(np.quantile(flow_c, .95)) if len(flow_c) else np.nan,
                     cyc_flow_max=float(flow_c.max()) if len(flow_c) else np.nan,
                     red_on_mean=float(ronc.mean()) if len(ronc) else np.nan,
                     red_on_p95=float(np.quantile(ronc, .95)) if len(ronc) else np.nan,
                     red_on_max=float(ronc.max()) if len(ronc) else np.nan,
                     burst_max=int(ln.max() + 1) if len(ln) else 0)
            for nm, kk_ in (("G", [G]), ("Y", [Y]), ("Rg", [RG]), ("R", [R]), ("U", [U]), ("GY", [G, Y])):
                msk = np.isin(state, kk_)
                r[f"n_{nm}"] = int(msk.sum())
                r[f"f1_{nm}"] = int((msk & f1).sum())
                r[f"f05_{nm}"] = int((msk & f05).sum())
                r[f"b_{nm}"] = int((msk & inb).sum())
                r[f"c_{nm}"] = int((msk & chat).sum())
            if secs is not None:
                for nm, kk_ in (("G", G), ("Y", Y), ("Rg", RG), ("R", R)):
                    r[f"s_{nm}"] = float(secs[kk_].sum())
                    r[f"o_{nm}"] = float(occ[kk_].sum())
            DETS.append(r)
    B = pd.concat(BINS, ignore_index=True) if BINS else None
    D = pd.DataFrame(DETS) if DETS else None
    return B, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    folds = set(pd.read_csv(DCW / "folds_v4.csv").DeviceId.str.lower())
    dirs = [p for p in sorted(EVD.iterdir()) if p.name.split("=", 1)[1].lower() in folds - locked]
    assert not any(p.name.split("=", 1)[1].lower() in locked for p in dirs)
    if a.only:
        dirs = [p for p in dirs if p.name.split("=", 1)[1].lower() in a.only.split(",")]
    if a.limit:
        dirs = dirs[:a.limit]
    t = time.time()
    with Pool(min(a.procs, 4), initializer=init) as pool:
        res = pool.map(one, dirs, chunksize=2)
    sfx = "" if not (a.limit or a.only) else "_smoke"
    pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True).to_parquet(OUT / f"bins117{sfx}.parquet")
    pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True).to_parquet(OUT / f"det117{sfx}.parquet")
    print(len(dirs), "signals", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
