"""Note 56: hi-res-only inputs for the pick rule, second version (ln6_pick.py = note 55's first version).

  stack health  Replaces note 55's per-window health_core status (inert: it judged each detector against fixed limits,
                not against its stack partner). Within a hi-res stack (co-located or spanning-cover members of one
                predicted phase, the same union as ln6) every member is measured against ONE shared independent
                reference R = the counts of the phase's other detectors outside the stack (predicted Count first, else
                any; else the rest of the signal), binned (1 min up to 3 h, 5 min above):
                  chi    Poisson dispersion of the member's binned counts around a_d * R_b (a_d = its own level):
                         erratic spikes / bursts give a large chi, a healthy partner on the same lane does not
                  surge  share of its ONs inside bins with n_b > 3 e_b + 10 (count bursts nobody else sees)
                  drop   share of bins where the reference says >= 5 ONs were expected and it logged none
                  chat   share of OFF -> next ON gaps < 0.3 s (chatter)
                  lvl    its count / the predicted-Count total of the lanes it sits on (NaN if unknown)
                Because all members share R, comparing them is fair; the pick rule uses the RELATIVE values (member
                vs its best partner): rel_chi = chi / min partner chi, etc. (atspm_pick56.py sets the flag).
  span v2       For every detector covering >= 2 non-co-located peers of its phase (ln6's candidate, before the gain
                test): the user's side-by-side cue. Lane detectors in different lanes see side-by-side vehicles as
                near-simultaneous ONs; one zone spanning those lanes sees them as ONE ON:
                  coinc_x  excess (over chance) near-simultaneous (|dt| <= 1 s) ON pairs between the peers, per peer ON
                  merge    share of those coincident pairs that the candidate covers with a single ON (its own lags)
                  r_lo / r_hi  candidate ONs / sum of peer ONs in the quietest / busiest third of 1-min bins
                           (spanning: r_hi < r_lo - side-by-side vehicles merge at peak)
                plus ln6's gain / flag (span1) for comparison. A detector covering exactly ONE peer gets
                  one_ratio (its ONs / the peer's) and one_back (share of its ONs the peer matches).

    python ln7_stackhealth.py [--run ...] [--lanes v3s] [--tag v3s] [--workers 4]
-> %DC_WORK%/lanes/ln7_stackhealth_<tag>.parquet. No technology, print or config fact is read. locked_v2 asserted
absent. CPU only.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ln1_cues as L1  # noqa: E402
import ln6_pick as L6  # noqa: E402

warnings.filterwarnings("ignore")
OUT = L1.OUT
RUN = "run_51ba131222_exclude_min5_clean_valnc_h3_drfp"
C7 = L6.C7
MIN_ON, MIN_SECS = L6.MIN_ON, L6.MIN_SECS
COVER, COLOC, TOL = L6.COVER, L6.COLOC, L6.TOL
COINC = 1.0          # side-by-side: two peer ONs within 1 s


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pair_mats(on, dets, secs):
    n = len(dets)
    M, L = np.zeros((n, n)), np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                M[a, b], L[a, b] = L6.match_frac(on[dets[a]], on[dets[b]], secs)
    return M, L


def clusters(n, M, cand):
    """union-find over co-location (either way >= COVER) and spanning-cover (candidate <-> its kept peers)."""
    par = list(range(n))

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for a in range(n):
        for b in range(a + 1, n):
            if max(M[a, b], M[b, a]) >= COVER:
                par[root(a)] = root(b)
    for i, keep in cand.items():
        for j in keep:
            par[root(i)] = root(j)
    out = {}
    for i in range(n):
        out.setdefault(root(i), []).append(i)
    return [m for m in out.values() if len(m) >= 2]


def coinc_merge(tS, peers, lags, secs):
    """side-by-side cue for a candidate spanning zone tS over peer lanes `peers` (ON times), lags = t_S - t_peer."""
    xs, merged, tot = 0.0, 0, 0
    npeer = sum(len(p) for p in peers)
    for a in range(len(peers)):
        for b in range(a + 1, len(peers)):
            ta, tb = peers[a], peers[b]
            if len(ta) == 0 or len(tb) == 0:
                continue
            lo = np.searchsorted(tb, ta - COINC, "left")
            hi = np.searchsorted(tb, ta + COINC, "right")
            c = hi - lo
            k = int(c.sum())
            chance = len(ta) * len(tb) * 2 * COINC / secs
            xs += k - chance
            ia = np.flatnonzero(c > 0)
            if not len(ia):
                continue
            # pair = a's ON with its nearest b ON (one of the two neighbours of the insertion point)
            jn = np.clip(np.searchsorted(tb, ta[ia]), 0, len(tb) - 1)
            jp = np.clip(jn - 1, 0, len(tb) - 1)
            j = np.where(np.abs(tb[jn] - ta[ia]) <= np.abs(tb[jp] - ta[ia]), jn, jp)
            xa, xb = ta[ia] + lags[a], tb[j] + lags[b]
            s0, s1 = np.minimum(xa, xb) - TOL, np.maximum(xa, xb) + TOL
            m = np.searchsorted(tS, s1, "right") - np.searchsorted(tS, s0, "left")
            tot += int((m >= 1).sum())
            merged += int((m == 1).sum())
    return xs / max(npeer, 1), (merged / tot if tot else np.nan), tot


def ratio_lohi(tS, peers, t0, secs):
    edges = t0 + np.arange(0, secs + 60, 60.0)
    s = np.histogram(tS, edges)[0].astype(float)
    k = sum(np.histogram(p, edges)[0] for p in peers).astype(float)
    m = k > 0
    if m.sum() < 9:
        return np.nan, np.nan
    q1, q2 = np.quantile(k[m], [1 / 3, 2 / 3])
    lo, hi = m & (k <= q1), m & (k >= q2)
    if k[lo].sum() == 0 or k[hi].sum() == 0 or q2 <= q1:
        return np.nan, np.nan
    return float(s[lo].sum() / k[lo].sum()), float(s[hi].sum() / k[hi].sum())


def health_rel(on, off_gap, members, ref_t, t0, secs, lanes, cls, phase_dets):
    binw = 60.0 if secs <= 3 * 3600 else 300.0
    edges = t0 + np.arange(0, secs + binw, binw)
    R = np.histogram(ref_t, edges)[0].astype(float)
    NR = R.sum()
    out = {}
    counts = [d for d in phase_dets if cls.get(d) == "Count"]
    for d in members:
        x = np.histogram(on[d], edges)[0].astype(float)
        r = {}
        if NR >= MIN_ON:
            a = x.sum() / NR
            e = a * R
            m = (e > 0) | (x > 0)
            r["chi"] = float(np.mean((x[m] - e[m]) ** 2 / (e[m] + 1))) if m.any() else np.nan
            sb = x > 3 * e + 10
            r["surge"] = float(x[sb].sum() / max(x.sum(), 1))
            ex = e >= 5
            r["drop"] = float(((x == 0) & ex).sum() / max(ex.sum(), 1)) if ex.sum() >= 3 else np.nan
        g = off_gap.get(d, np.zeros(0))
        r["chat"] = float((g < 0.3).mean()) if len(g) >= MIN_ON else np.nan
        ld = lanes.get(d, frozenset())
        same = [c for c in counts if c != d and c not in members and ld and lanes.get(c) and (lanes[c] & ld)]
        r["lvl"] = float(len(on[d]) / max(sum(len(on[c]) for c in same), 1)) if same else np.nan
        out[d] = r
    return out


def work(args):
    dev, period, g = args
    import a2_features as A2F
    import pyarrow.dataset as ds
    tab = ds.dataset(str(L6.CACHE[period])).to_table(filter=ds.field("DeviceId") == dev,
                                                     columns=["Detector", "t_on", "dur"]).to_pandas().drop_duplicates()
    tab["t"] = tab.t_on.astype("datetime64[us]").astype("int64").to_numpy() / 1e6
    tab = tab.sort_values(["Detector", "t"])
    on_all, gap_all = {}, {}
    for d, x in tab.groupby("Detector"):
        t, du = x.t.to_numpy(), x.dur.to_numpy(float)
        on_all[int(d)] = t
        gap_all[int(d)] = (t, np.r_[np.nan, t[1:] - (t[:-1] + du[:-1])])
    rows = []
    for name, t0w, secs in A2F.WINDOWS[period]:
        if secs < MIN_SECS:
            continue
        w = g[g.win == name]
        if w.empty:
            continue
        t0 = pd.Timestamp(t0w).value / 1e9
        t1 = t0 + secs
        on, gp = {}, {}
        for d in w.Detector.astype(int):
            t = on_all.get(d, np.zeros(0))
            k = (t >= t0) & (t < t1)
            on[d] = t[k]
            if d in gap_all:
                gg = gap_all[d][1][k]
                gp[d] = gg[np.isfinite(gg)]
        lanes = {int(d): frozenset(int(v) for v in s.split(",")) if isinstance(s, str) and s else frozenset()
                 for d, s in zip(w.Detector, w.lanes)}
        cls = dict(zip(w.Detector.astype(int), w.func))
        alld = [d for d in on if len(on[d]) >= MIN_ON]
        for ph_, gg_ in w.groupby("pred_phase"):
            dets = L6.order_dets(on, [int(d) for d in gg_.Detector if len(on[int(d)]) >= MIN_ON], t0)
            if len(dets) < 2:
                continue
            n = len(dets)
            M, L = pair_mats(on, dets, secs)
            cand, one = {}, {}
            for i in range(n):
                cov = [j for j in range(n) if j != i and M[j, i] >= COVER and len(on[dets[i]]) > len(on[dets[j]])]
                cov = sorted(cov, key=lambda j: -M[j, i])
                keep = []
                for j in cov:
                    if all(M[j, k] < COLOC and M[k, j] < COLOC for k in keep):
                        keep.append(j)
                if len(keep) >= 2:
                    cand[i] = keep
                elif len(keep) == 1:
                    one[i] = keep[0]
            cl = clusters(n, M, cand)
            # span v2 features for every candidate
            sp = {}
            for i, keep in cand.items():
                tS = on[dets[i]]
                peers = [on[dets[j]] for j in keep]
                lags = [L[j, i] for j in keep]                 # t_S - t_peer at the pair's best lag
                u = L6.union_frac(tS, peers, lags, secs)
                gain = u - max(M[i, j] for j in keep)
                xs, mg, npair = coinc_merge(tS, peers, lags, secs)
                rlo, rhi = ratio_lohi(tS, peers, t0, secs)
                sp[dets[i]] = dict(span1=bool(gain >= L6.SPAN_GAIN), gain=float(gain), peers=",".join(str(dets[j]) for j in keep),
                                   coinc_x=float(xs), merge=float(mg), n_pair=int(npair), r_lo=rlo, r_hi=rhi,
                                   sum_ratio=float(len(tS) / max(sum(len(p) for p in peers), 1)))
            # stack health
            hr = {}
            for m in cl:
                mem = [dets[i] for i in m]
                outside = [d for d in dets if d not in mem]
                ref = [d for d in outside if cls.get(d) == "Count"] or outside
                kind = "phase_count" if ref and cls.get(ref[0]) == "Count" else "phase" if ref else "signal"
                if not ref:
                    ref = [d for d in alld if d not in mem]
                ref_t = np.concatenate([on[d] for d in ref]) if ref else np.zeros(0)
                h = health_rel(on, gp, mem, ref_t, t0, secs, lanes, cls, dets)
                cid = "-".join(str(d) for d in sorted(mem))
                for d in mem:
                    hr[d] = dict(clus=cid, clus_n=len(mem), ref_kind=kind, **h[d])
            for i, j in one.items():                          # single covered peer (e.g. radar zone over ONE loop)
                sp[dets[i]] = dict(one_peer=str(dets[j]), one_ratio=float(len(on[dets[i]]) / max(len(on[dets[j]]), 1)),
                                   one_back=float(M[i, j]))         # share of ITS ONs matched by the peer
            for d in set(sp) | set(hr):
                rows.append(dict(DeviceId=dev, period=period, win=name, Detector=d, n_on=len(on[d]),
                                 **hr.get(d, {}), **{f"sp_{k}": v for k, v in sp.get(d, {}).items()}))
    return pd.DataFrame(rows)


def main(run, lanes_tag, tag, workers, limit=None):
    import a2_features as A2F
    k = L6.frame(run, lanes_tag)
    big = {p: {n for n, _, s in A2F.WINDOWS[p] if s >= MIN_SECS} for p in A2F.WINDOWS}
    k = k[[w in big[p] for p, w in zip(k.period, k.win)]]
    jobs = [(dev, per, g[["Detector", "win", "pred_phase", "func", "lanes"]].copy())
            for (dev, per), g in k.groupby(["DeviceId", "period"])]
    if limit:
        jobs = jobs[:limit]
    log(f"{len(jobs)} signal-periods, {len(k):,} detector-windows")
    out, t0 = [], time.time()
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            out.append(r)
            if i % 100 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    D = pd.concat(out, ignore_index=True)
    assert not D.DeviceId.isin(L1.locked()).any()
    D.to_parquet(OUT / f"ln7_stackhealth_{tag}.parquet", index=False)
    log(f"wrote {len(D):,} rows; in stacks {int(D.clus.notna().sum()):,}; span candidates {int(D.sp_gain.notna().sum()):,} "
        f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--lanes", default="v3s")
    ap.add_argument("--tag", default="v3s")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.run, a.lanes, a.tag, a.workers, a.limit)
