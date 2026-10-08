"""Note 122: red vs green as a HEALTH signal - per-cycle event pass with three colour sources.

Same w40 windows / signals as note 117 (763 training signals, locked_v2 asserted absent).  Hi-res log only (81 / 82
de-duplicated, channels > 64 dropped; colour 1 / 8 / 9 / 10).  For every detector-window the detector's vehicle ONs
(ONs that start a continuous ON, h117 rule) are read against the colours of three phases:
  pred   the classifier's predicted phase (production view; the phase number is only a join key)
  true   the official timing phase (labels v4q phase_target; evaluation only)
  plac   a placebo: a random OTHER phase with greens at the signal (seeded per detector-window)
Only complete, gap-free cycles count (begin green -> begin yellow -> red -> next begin green).  Red = after the first
2 s of red (late-yellow tails), as note 117.

Per source (prefix p_ / t_ / x_):
  ncyc, sGY, sR          cycles, green+yellow seconds, red seconds
  nGY, nR                vehicle ONs in green+yellow / in red
  oGY, oR                ON seconds in green+yellow / red
  bg, by                 share of cycles the zone is ON at begin green / at begin yellow
  cg, cr                 share of cycles with >= 1 ON in green+yellow / in red
  e5, nd                 share of 'demand' cycles (>= 2 ONs in green, or ON at begin green) whose first ON is within
                         5 s of begin green (ON at begin green counts as 0 s);  nd = number of demand cycles
  lagf                   median (first-ON lag in green / green length) over cycles with an ON in green
Histograms (pred, true) of ON-start lag since begin green, 1-s bins 0..149 + 150+ -> hist122.npz (charts only).

    python h122_events.py [--procs 4] [--limit N]   -> %DC_WORK%/s122/{cyc122.parquet, hist122.npz}
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "health117"))
import h117_events as E  # noqa: E402

DCW = E.DCW
OUT = DCW / "s122"
REPO = Path(__file__).resolve().parents[3]
G, Y, RG, R = E.G, E.Y, E.RG, E.R
NH = 151
PH = TRU = None


def init():
    global PH, TRU
    E.init()
    PH = E.PH
    L = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v4q.parquet",
                        columns=["DeviceId", "detector", "phase_target", "phase_target_type"])
    L = L[L.phase_target_type == "phase"]
    L["p"] = pd.to_numeric(L.phase_target.str.replace("P", "", regex=False), errors="coerce")
    L["DeviceId"] = L.DeviceId.str.lower()
    TRU = {k: dict(zip(g.detector.astype(int), g.p)) for k, g in L.dropna(subset=["p"]).groupby("DeviceId")}


def cycles(b, s, v):
    """complete gap-free cycles: arrays begin green, begin yellow, begin red, next begin green."""
    gi = np.where(s == G)[0]
    out = []
    for a, z in zip(gi[:-1], gi[1:]):
        if not v[a:z].all():
            continue
        ss = s[a:z]
        yi = np.where(ss == Y)[0]
        ri = np.where(ss == RG)[0]
        if not len(ri):
            continue
        cy = b[a + yi[0]] if len(yi) else b[a + ri[0]]
        cr = b[a + ri[0]]
        if cy - b[a] < 2 or b[z] - cr < 2.5:
            continue
        out.append((b[a], cy, cr, b[z]))
    return np.array(out, float).reshape(-1, 4)


def cum_on(x, du, t):
    """total ON seconds up to times t (intervals [x, x+du), sorted, non-overlapping)."""
    if not len(x):
        return np.zeros(len(t))
    cd = np.r_[0.0, np.cumsum(du)]
    i = np.searchsorted(x, t, "right") - 1
    ii = np.clip(i, 0, len(x) - 1)
    part = np.where(i >= 0, np.clip(t - x[ii], 0, du[ii]), 0.0)
    return np.where(i >= 0, cd[ii], 0.0) + part


def on_at(x, e, t):
    i = np.searchsorted(x, t, "right") - 1
    ii = np.clip(i, 0, max(len(x) - 1, 0))
    return (i >= 0) & (e[ii] > t) if len(x) else np.zeros(len(t), bool)


def stats(x, du, C):
    nan = np.nan
    if not len(C):
        return dict(ncyc=0), None
    cg, cy, cr, cn = C.T
    r0 = np.minimum(cr + E.RED_GUARD, cn)
    e = x + du
    k0 = np.searchsorted(x, cg, "left")
    k1 = np.searchsorted(x, cr, "left")
    kr0 = np.searchsorted(x, r0, "left")
    kr1 = np.searchsorted(x, cn, "left")
    ngy, nr = k1 - k0, kr1 - kr0
    F = lambda t: cum_on(x, du, t)  # noqa: E731
    ogy = F(cr) - F(cg)
    o_r = F(cn) - F(r0)
    bg = on_at(x, e, cg)
    by = on_at(x, e, cy)
    ng_only = np.searchsorted(x, cy, "left") - k0                     # ONs in green proper
    first = np.where(ng_only > 0, x[np.clip(k0, 0, max(len(x) - 1, 0))] - cg if len(x) else nan, nan)
    lag0 = np.where(bg, 0.0, first)
    dem = bg | (ng_only >= 2)
    gl = cy - cg
    with np.errstate(invalid="ignore"):
        lagf = first / gl
    d = dict(ncyc=len(C), sGY=float((cr - cg).sum()), sR=float((cn - r0).sum()), nGY=int(ngy.sum()), nR=int(nr.sum()),
             oGY=float(ogy.sum()), oR=float(o_r.sum()), bg=float(bg.mean()), by=float(by.mean()),
             cg=float((ngy > 0).mean()), cr=float((nr > 0).mean()), nd=int(dem.sum()),
             e5=float((lag0[dem] <= 5).mean()) if dem.any() else nan,
             lagf=float(np.nanmedian(lagf)) if np.isfinite(lagf).any() else nan,
             gmed=float(np.median(gl)), cmed=float(np.median(cn - cg)))
    # lag histogram since begin green (every ON inside a complete cycle)
    j = np.searchsorted(cg, x, "right") - 1
    ok = (j >= 0) & (x < cn[np.clip(j, 0, len(cn) - 1)])
    lag = x[ok] - cg[j[ok]]
    h = np.bincount(np.minimum(lag.astype(int), NH - 1), minlength=NH).astype(np.uint16)
    return d, h


def one(dev_dir):
    dev = dev_dir.name.split("=", 1)[1].lower()
    ts_all, eid_all, par_all = E.load(dev_dir)
    rows, H, K = [], [], []
    tru = TRU.get(dev, {})
    for w, (s0, hrs) in E.WIN.items():
        t0 = (pd.Timestamp(s0) - E.T0).total_seconds()
        Tlen = hrs * 3600.0
        i0, i1 = np.searchsorted(ts_all, [t0, t0 + Tlen])
        t, eid, par = ts_all[i0:i1] - t0, eid_all[i0:i1], par_all[i0:i1]
        dm = (eid == 81) | (eid == 82)
        if not dm.any():
            continue
        ut = np.unique(t)
        gi = np.diff(ut) > E.GAP_S
        g0, g1 = ut[:-1][gi], ut[1:][gi]
        phd = PH.get((w.split("_")[0], dev), {})
        cm = np.isin(eid, (1, 8, 9, 10))
        CY = {}
        for p in np.unique(par[cm & (eid == 1)]):
            mm = cm & (par == p)
            if (eid[mm] == 1).sum() < 3:
                continue
            b, s, v = E.phase_timeline(t[mm], eid[mm], Tlen, g0, g1)
            C = cycles(b, s, v)
            if len(C) >= 3:
                CY[int(p)] = C
        if not CY:
            continue
        phases = sorted(CY)
        # vehicle ONs per channel (h117 rule: ON that starts a continuous ON; end = next OFF)
        td, ed, pd_ = t[dm], eid[dm], par[dm]
        o = np.lexsort((ed != 82, td, pd_))
        td, ed, pd_ = td[o], ed[o], pd_[o]
        n = len(td)
        same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
        prev_e = np.r_[0, ed[:-1]]
        start = (ed == 82) & ~(same_prev & (prev_e == 82))
        offs = np.where(ed == 81)[0]
        k = np.searchsorted(offs, np.arange(n), "left")
        if len(offs):
            pp = offs[np.minimum(k, len(offs) - 1)]
            c_end = np.where((k < len(offs)) & (pd_[pp] == pd_), td[pp], Tlen)
        else:
            c_end = np.full(n, Tlen)
        si = np.where(start)[0]
        xs, ch, du = td[si], pd_[si], c_end[si] - td[si]
        if len(g0) and len(si):
            kq = np.searchsorted(g0, xs)
            kk = np.minimum(kq, len(g0) - 1)
            du[(kq < len(g0)) & (g0[kk] < xs + du)] = 0.0              # duration unknown over a comms gap
        for c in np.unique(pd_):
            m = ch == c
            x, dd = xs[m], du[m]
            pp_ = phd.get(int(c), np.nan)
            pt = tru.get(int(c), np.nan)
            pred = int(pp_) if np.isfinite(pp_) and int(pp_) in CY else None
            true = int(pt) if np.isfinite(pt) and int(pt) in CY else None
            others = [q for q in phases if q != pred and q != true]
            rng = np.random.default_rng(zlib.crc32(f"{dev}|{w}|{int(c)}".encode()))
            plac = int(rng.choice(others)) if others else None
            r = dict(DeviceId=dev, window=w, detector=int(c), n_on=len(x), pred=pred, true=true, plac=plac,
                     true_raw=pt)
            for pre, ph in (("p_", pred), ("t_", true), ("x_", plac)):
                if ph is None:
                    continue
                d, h = stats(x, dd, CY[ph])
                r.update({pre + kk_: vv for kk_, vv in d.items()})
                if pre != "x_" and h is not None:
                    H.append(h)
                    K.append((dev, w, int(c), pre[0]))
            rows.append(r)
    return pd.DataFrame(rows), H, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    folds = set(pd.read_csv(DCW / "folds_v4.csv").DeviceId.str.lower())
    dirs = [p for p in sorted(E.EVD.iterdir()) if p.name.split("=", 1)[1].lower() in folds - locked]
    assert not any(p.name.split("=", 1)[1].lower() in locked for p in dirs)
    if a.limit:
        dirs = dirs[:a.limit]
    t = time.time()
    with Pool(min(a.procs, 4), initializer=init) as pool:
        res = pool.map(one, dirs, chunksize=2)
    sfx = "_smoke" if a.limit else ""
    pd.concat([r[0] for r in res if len(r[0])], ignore_index=True).to_parquet(OUT / f"cyc122{sfx}.parquet")
    H = np.stack([h for r in res for h in r[1]]) if any(r[1] for r in res) else np.zeros((0, NH), np.uint16)
    K = pd.DataFrame([k for r in res for k in r[2]], columns=["DeviceId", "window", "detector", "src"])
    np.savez_compressed(OUT / f"hist122{sfx}.npz", H=H)
    K.to_parquet(OUT / f"histkey122{sfx}.parquet")
    print(len(dirs), "signals", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
