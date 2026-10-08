"""Note 106, idea 1: vehicle-path features per detector-window, phase-anonymous, from the hi-res ON times only.

For every pair of detectors on the same PREDICTED phase (frame pred_phase, OOF) in a Sept-2026 window >= 30 min, the
ON-time correlogram (b - a, 0.5 s bins, |lag| < 15 s; lane_output.xcorr_hist) gives
  zero-lag excess (|lag| < 1 s)                  -> co-located / stacked / spanning zones
  forward peak 1-12 s (b follows a)              -> a is upstream of b: travel time = peak lag, strength = excess share
  backward peak 1-12 s (b precedes a)            -> a is downstream of b
(excess = 3-bin run at the peak minus chance from the |lag| >= 10 s bins, divided by a's ONs = share of a's actuations
that propagate).  Per detector: best downstream / upstream partner (share, lag, sharpness, volume ratio, partner-side
share), number of strong down / up edges, chain position, chain depth (longest path of strong edges), farthest travel
time, any-mate propagation shares, co-location, and the same restricted to its lane-mates (lanes D of the fixed OOF lane
chain, lanes5g).  No label, no print fact, no function prediction.  NaN for windows < 30 min, < 10 ONs, or no mates.

    python path106.py [--workers 6]          -> %DC_WORK%/s106/feat_vp.parquet (frame order of s106/keys.parquet)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
for _d in ("", "lanes", "trackA"):
    sys.path.insert(0, str(CODE / _d) if _d else str(CODE))
import rpath  # noqa: F401,E402
import lane_output as LO  # noqa: E402
from common import DC_WORK  # noqa: E402

OUT = DC_WORK / "s106"
STG = DC_WORK / "official" / "stg" / "cache" / "det_intervals.parquet"
KEY = ["DeviceId", "Detector", "period", "win"]
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
MIN_ON = 10
MID = LO.MID
BASE = np.abs(MID) >= 10
ZERO = np.abs(MID) < 1.0
FWD = (MID >= 1.0) & (MID <= 12.0)
BWD = (MID <= -1.0) & (MID >= -12.0)
STRONG_SHARE, STRONG_RATIO = 0.15, np.log(2.0)
COLS = ["vp_dn_share", "vp_dn_lag", "vp_dn_ratio", "vp_dn_lv", "vp_dn_rshare",
        "vp_up_share", "vp_up_lag", "vp_up_ratio", "vp_up_lv", "vp_up_rshare",
        "vp_n_dn", "vp_n_up", "vp_pos", "vp_dn_depth", "vp_up_depth", "vp_dn_lagmax", "vp_up_lagmax",
        "vp_z_max", "vp_z_lv", "vp_n_z", "vp_any_dn", "vp_any_up",
        "vp_lane_ndn", "vp_lane_nup", "vp_lane_first", "vp_lane_last"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run3(h, band):
    """max 3-bin run inside a band (band bins are contiguous); returns (value, centre lag)."""
    x = h[band]
    mids = MID[band]
    if len(x) < 3:
        return 0.0, np.nan
    r = x[:-2] + x[1:-1] + x[2:]
    j = int(np.argmax(r))
    return float(r[j]), float(mids[j + 1])


def pair(ta, tb):
    """statistics of one unordered pair from a's side and b's side."""
    h = LO.xcorr_hist(ta, np.zeros(len(ta), bool), tb)[0]
    base = max(h[BASE].mean(), 0.2)
    na, nb = len(ta), len(tb)
    zexc = h[ZERO].sum() - ZERO.sum() * base
    fv, fl = run3(h, FWD)          # b after a
    bv, bl = run3(h, BWD)          # b before a
    return dict(base=base, zexc=zexc, za=zexc / na, zb=zexc / nb,
                f_exc=fv - 3 * base, f_lag=fl, f_ratio=np.log(max(fv, 0.5) / (3 * base)),
                b_exc=bv - 3 * base, b_lag=-bl if np.isfinite(bl) else np.nan, b_ratio=np.log(max(bv, 0.5) / (3 * base)),
                na=na, nb=nb)


def any_share(t, others, lo, hi):
    """excess share of t's ONs with some other ON in [t+lo, t+hi] (lo<hi, may be negative) over chance (same span
    shifted 20 s further out)."""
    if len(others) == 0 or len(t) == 0:
        return np.nan

    def frac(a, b):
        i = np.searchsorted(others, t + a, "left")
        j = np.searchsorted(others, t + b, "right")
        return float(np.mean(j > i))
    sh = 20.0 if hi > 0 else -20.0
    return frac(lo, hi) - frac(lo + sh, hi + sh)


def longest(adj, n):
    """longest path length (edges) from every node along adj (DAG assumed; cycles cut by depth limit)."""
    memo = {}

    def f(i, depth):
        if depth > n:
            return 0
        if i in memo:
            return memo[i]
        v = 0
        for j in adj[i]:
            v = max(v, 1 + f(j, depth + 1))
        memo[i] = v
        return v
    return [f(i, 0) for i in range(n)]


def group_feats(dets, T, lanes, prs=None):
    """dets list, T {det: ON times}, lanes {det: set of lane ids or empty} -> {det: feature dict}."""
    n = len(dets)
    S = {}
    for i in range(n):
        for j in range(i + 1, n):
            S[(i, j)] = pair(T[dets[i]], T[dets[j]])
            if prs is not None:
                s_ = S[(i, j)]
                mn = max(min(s_["na"], s_["nb"]), 1)
                prs.append(dict(da=dets[i], db=dets[j], dir=(s_["f_exc"] - s_["b_exc"]) / mn,
                                f_sh=s_["f_exc"] / s_["na"], b_sh=s_["b_exc"] / s_["na"], f_lag=s_["f_lag"],
                                b_lag=s_["b_lag"], z=s_["zexc"] / mn if "zexc" in s_ else np.nan))
    # directional view from each detector: list of (partner, share_dn, lag_dn, ratio_dn, rshare_dn, share_up, ...)
    view = {i: [] for i in range(n)}
    for (i, j), s in S.items():
        na, nb = s["na"], s["nb"]
        # from i: downstream partner j uses forward (j after i); upstream partner j uses backward
        view[i].append(dict(p=j, dn=s["f_exc"] / na, dn_lag=s["f_lag"], dn_ratio=s["f_ratio"], dn_r=s["f_exc"] / nb,
                            up=s["b_exc"] / na, up_lag=s["b_lag"], up_ratio=s["b_ratio"], up_r=s["b_exc"] / nb,
                            z=s["za"], zr=s["zb"], lv=np.log(nb / na)))
        view[j].append(dict(p=i, dn=s["b_exc"] / nb, dn_lag=s["b_lag"], dn_ratio=s["b_ratio"], dn_r=s["b_exc"] / na,
                            up=s["f_exc"] / nb, up_lag=s["f_lag"], up_ratio=s["f_ratio"], up_r=s["f_exc"] / na,
                            z=s["zb"], zr=s["za"], lv=np.log(na / nb)))
    strong_dn = {i: [v["p"] for v in view[i] if v["dn"] >= STRONG_SHARE and v["dn_ratio"] >= STRONG_RATIO
                     and v["dn"] > v["z"] and v["dn"] > v["up"]] for i in range(n)}
    strong_up = {i: [v["p"] for v in view[i] if v["up"] >= STRONG_SHARE and v["up_ratio"] >= STRONG_RATIO
                     and v["up"] > v["z"] and v["up"] > v["dn"]] for i in range(n)}
    ddep, udep = longest(strong_dn, n), longest(strong_up, n)
    allt = {i: np.sort(np.concatenate([T[dets[j]] for j in range(n) if j != i])) for i in range(n)}
    out = {}
    for i in range(n):
        V = view[i]
        bd = max(V, key=lambda v: v["dn"])
        bu = max(V, key=lambda v: v["up"])
        bz = max(V, key=lambda v: min(v["z"], v["zr"]))
        nd, nu = len(strong_dn[i]), len(strong_up[i])
        lag_d = [v["dn_lag"] for v in V if v["p"] in strong_dn[i]]
        lag_u = [v["up_lag"] for v in V if v["p"] in strong_up[i]]
        li = lanes.get(dets[i], set())
        mates = [j for j in range(n) if j != i and li and (lanes.get(dets[j], set()) & li)]
        r = dict(vp_dn_share=bd["dn"], vp_dn_lag=bd["dn_lag"], vp_dn_ratio=bd["dn_ratio"], vp_dn_lv=bd["lv"],
                 vp_dn_rshare=bd["dn_r"],
                 vp_up_share=bu["up"], vp_up_lag=bu["up_lag"], vp_up_ratio=bu["up_ratio"], vp_up_lv=bu["lv"],
                 vp_up_rshare=bu["up_r"],
                 vp_n_dn=nd, vp_n_up=nu, vp_pos=nu / (nu + nd) if nu + nd else np.nan,
                 vp_dn_depth=ddep[i], vp_up_depth=udep[i],
                 vp_dn_lagmax=max(lag_d) if lag_d else np.nan, vp_up_lagmax=max(lag_u) if lag_u else np.nan,
                 vp_z_max=min(bz["z"], bz["zr"]), vp_z_lv=bz["lv"],
                 vp_n_z=sum(1 for v in V if min(v["z"], v["zr"]) >= 0.3),
                 vp_any_dn=any_share(T[dets[i]], allt[i], 1.0, 12.0),
                 vp_any_up=any_share(T[dets[i]], allt[i], -12.0, -1.0))
        if mates:
            r["vp_lane_ndn"] = sum(1 for j in mates if j in strong_dn[i])
            r["vp_lane_nup"] = sum(1 for j in mates if j in strong_up[i])
            r["vp_lane_first"] = float(r["vp_lane_nup"] == 0 and r["vp_lane_ndn"] > 0)
            r["vp_lane_last"] = float(r["vp_lane_ndn"] == 0 and r["vp_lane_nup"] > 0)
        out[dets[i]] = r
    return out


def work(args):
    dev, g = args
    import pyarrow.dataset as ds
    import a2_features as A2F
    tab = ds.dataset(str(STG)).to_table(filter=ds.field("DeviceId") == dev,
                                        columns=["Detector", "t_on", "t_off"]).to_pandas().drop_duplicates()
    tab = tab[tab.t_off.notna()]
    tab["t"] = tab.t_on.astype("datetime64[us]").astype("int64") / 1e6
    by = {int(d): np.sort(x.t.to_numpy()) for d, x in tab.groupby("Detector")}
    wins = {n: (pd.Timestamp(t0).value / 1e9, secs) for n, t0, secs in A2F.WINDOWS["stg"]}
    rows, PR, prs = [], [], []
    for (w, ph), gg in g.groupby(["win", "pred_phase"]):
        t0, secs = wins[w]
        T, lanes = {}, {}
        for d, ls in zip(gg.Detector.astype(int), gg.lanes5g):
            x = by.get(d, np.zeros(0))
            x = x[(x >= t0) & (x < t0 + secs)]
            if len(x) >= MIN_ON:
                T[d] = x
                lanes[d] = {int(v) for v in ls.split(",")} if isinstance(ls, str) and ls else set()
        dets = sorted(T)
        if len(dets) < 2:
            continue
        for d, r in group_feats(dets, T, lanes, prs).items():
            r.update(DeviceId=dev, Detector=d, win=w)
            rows.append(r)
        for p in prs:
            p.update(DeviceId=dev, win=w)
        PR.extend(prs)
        prs = []
    return pd.DataFrame(rows), pd.DataFrame(PR)


def main(a):
    k = pd.read_parquet(OUT / "keys.parquet")
    m = (k.period == "stg") & k.wgroup.isin(GE30) & k.pred_phase.notna()
    sub = k[m]
    jobs = [(dev, g[["Detector", "win", "pred_phase", "lanes5g"]]) for dev, g in sub.groupby("DeviceId")]
    if a.limit:
        jobs = jobs[:a.limit]
    log(f"{len(jobs)} signals, {len(sub):,} detector-windows")
    from multiprocessing import Pool
    out, outp, t0 = [], [], time.time()
    with Pool(a.workers) as pool:
        for i, (x, xp) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            out.append(x)
            outp.append(xp)
            if i % 100 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    PP = pd.concat(outp, ignore_index=True)
    PP["period"] = "stg"
    PP.astype({c: np.float32 for c in ["dir", "f_sh", "b_sh", "f_lag", "b_lag", "z"]}).to_parquet(
        OUT / ("pairs_vp_test.parquet" if a.limit else "pairs_vp.parquet"), index=False)
    F = pd.concat(out, ignore_index=True)
    F["period"] = "stg"
    F["Detector"] = F.Detector.astype(k.Detector.dtype)
    x = k[KEY].merge(F, on=KEY, how="left")
    assert len(x) == len(k)
    for c in COLS:
        x[c] = x[c].astype(np.float32)
    x = x[KEY + COLS]
    x.to_parquet(OUT / ("feat_vp_test.parquet" if a.limit else "feat_vp.parquet"), index=False)
    cov = x[COLS].notna().mean().round(3).to_dict()
    log(f"wrote {len(x):,} rows ({time.time()-t0:.0f}s); coverage {cov}")
    log(x.loc[m.to_numpy(), COLS].describe().T.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
