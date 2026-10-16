"""Detector health -- statistics from the hi-res log for ONE signal window (notes 104-144).

Every function here is a production port of one research event pass (research/code/health*/ ...), computed on the
window's prepared event arrays (`health_core.Prep`) and the classifier's own answers (predicted phase, function,
lane span).  numpy + pandas only; no loop over events (loops are over detectors, phases or episodes).  The phase
number is only a key to that phase's color events and to "who are its phase mates" -- never a feature.

Research source of each block (same arithmetic, same order):
  occ_bins / bins_ctx            h96_occ.one, h104_resolve.load_bins          (15-min bins: counts, % ON, traffic, congestion;
                                                                              detector x bin matrices since note 128)
  slopes / like_corr / occ_hi    h108_base.bin_stats, h108_resolve.context, score_v4c.occ_hi_c (Count zone held ON)
  act_stats / act118             h108_events.stats, h118_events.stats         (ON durations, extension, starts)
  shape110                       h110_events.one                              (yardstick, 15-min bins)
  level110 / drop_evidence       h110_resolve.level110, h118_resolve.drop_evidence
  erratic                        h118_erratic.off_bins
  episodes_ctx                   h110_resolve.episodes + traffic_evidence     (queue test per stuck ON)
  q5_stats / fast_even           h117_events.one + h118c_fast, h135_fast      (too many for the traffic; rule F)
  too_fast                       h141_events item 1                           (ON-to-ON gaps too short for vehicles)
  day_profile / night_day        h116_data + tod116                           (busy at night, 24-h samples)
  dropout_clean                  h118c_dropout                                (goes silent, clean yardstick)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import health_core as hc

FNS = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")
TRAF = ("Advance", "Count")
GAP_S = 120.0


def _nan(n):
    return np.full(n, np.nan)


def _corr(a, b):
    """h104_resolve._corr"""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _spans_gap(g0, a, b):
    if not len(g0):
        return np.zeros(len(a), bool)
    k = np.searchsorted(g0, a)
    kk = np.minimum(k, len(g0) - 1)
    return (k < len(g0)) & (g0[kk] < b)


# ============================================================================ 15-min bins (h96_occ, h104 load_bins)
# note 128: the research computed these per detector x bin in long pandas tables; the package keeps every statistic
# as a detector x bin matrix.  Where the research summed with pandas (compensated sums), `_ksum` adds in the same
# order with the same compensation, so every value is bit-identical; medians use the same two middle values.
def _ksum(M):
    """pandas groupby sum / count of the rows of M taken top to bottom (NaN skipped): Kahan-compensated exactly as
    pandas' group_sum.  Returns (sum, count) per column."""
    nb = M.shape[1] if M.ndim == 2 else 0
    s, c, k = np.zeros(nb), np.zeros(nb), np.zeros(nb, np.int64)
    with np.errstate(invalid="ignore"):
        for row in M:
            ok = ~np.isnan(row)
            y = row - c
            t = s + y
            cn = (t - s) - y
            cn = np.where(cn != cn, 0.0, cn)
            s = np.where(ok, t, s)
            c = np.where(ok, cn, c)
            k += ok
    return s, k


def _ksum_rows(M):
    """_ksum of every ROW of M separately (its columns taken left to right, NaN skipped) -- the same compensated
    additions, one column at a time for all rows together.  Returns (sum, count) per row."""
    nr = M.shape[0]
    s, c, k = np.zeros(nr), np.zeros(nr), np.zeros(nr, np.int64)
    with np.errstate(invalid="ignore"):
        for j in range(M.shape[1]):
            col = M[:, j]
            ok = ~np.isnan(col)
            y = col - c
            t = s + y
            cn = (t - s) - y
            cn = np.where(cn != cn, 0.0, cn)
            s = np.where(ok, t, s)
            c = np.where(ok, cn, c)
            k += ok
    return s, k


def _median_excl(V):
    """per row i and column j: the median of column j's finite values without row i (= np.nanmedian(V[others], 0):
    the two middle values added and halved); NaN where nothing is left."""
    nd, nb = V.shape
    out = np.full((nd, nb), np.nan)
    fin = np.isfinite(V)
    for j in range(nb):
        f = fin[:, j]
        v = np.sort(V[f, j])
        m = len(v)
        if not m:
            continue
        p = np.where(f, np.searchsorted(v, np.where(f, V[:, j], 0.0), "left"), m)
        mm = np.where(f, m - 1, m)
        ok = mm > 0
        h = mm // 2
        lo = np.where(mm % 2 == 1, h, h - 1)
        vv = np.r_[v, np.nan]
        a = vv[np.clip(np.where(lo < p, lo, lo + 1), 0, m)]
        b = vv[np.clip(np.where(h < p, h, h + 1), 0, m)]
        out[ok, j] = ((a + b) / 2.0)[ok]
    return out


def occ_bins(P, start, end, bs, fn, phase):
    """counts n, fraction ON occ (NaN in uncovered bins), traffic reference ref, congestion index cong per detector x
    bin (bs seconds), plus per-detector mean ON fraction, total and share of the reference.  fn / phase: dicts."""
    B = hc.events_to_bins(P, start, end, bin_s=bs)
    dets = B["dets"]
    n = B["n_on"].astype(float)
    occ = B["occ"].astype(float) / bs
    cov = B["cov"]
    n[:, ~cov] = np.nan
    occ[:, ~cov] = np.nan
    nd, nb = n.shape
    F = np.array([fn.get(int(d)) for d in dets], dtype=object)
    Ph = np.array([phase.get(int(d), np.nan) for d in dets], dtype=float)
    with np.errstate(all="ignore"), _quiet():
        tot = np.nansum(n, 1)
        mocc = np.nanmean(occ, 1) if nb else np.full(nd, np.nan)
        rel = occ / np.where(mocc[:, None] > 0.01, mocc[:, None], np.nan)
    istraf = np.isin(F.astype(str), TRAF) & np.array([f is not None for f in F])
    ref = np.full((nd, nb), np.nan)
    cong = np.full((nd, nb), np.nan)
    share = np.full(nd, np.nan)
    idx = np.arange(nd)
    ref_mem = []
    if nd - 1 >= 3:
        cong = _median_excl(rel)
    for i in range(nd):
        oth = idx != i
        same = oth & (Ph == Ph[i]) if np.isfinite(Ph[i]) else np.zeros(nd, bool)
        if (same & istraf).any():
            rm = same & istraf
        elif (oth & istraf).any():
            rm = oth & istraf
        else:
            rm = oth
        r = np.nansum(n[rm], 0) if rm.any() else np.full(nb, np.nan)
        r[~cov] = np.nan
        ref[i] = r
        ref_mem.append(np.flatnonzero(rm))
        share[i] = tot[i] / max(np.nansum(r), 1)
    return dict(dets=dets, n=n, occ=occ, cov=cov, ref=ref, cong=cong, tot=tot, mocc=mocc, share_ref=share,
                hour=B["hour"], fn=F, phase=Ph, bs=bs, nb=nb, B=B, ref_mem=ref_mem)


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")
        return self

    def __exit__(self, *a):
        self._w.__exit__(*a)


def bins_ctx(O, status1):
    """the research 'bins_ctx' table (h104 load_bins) as detector x bin matrices: ph (per detector, phase or -1),
    st (pass-1 status per detector, None if missing), hl (healthy and covered), peer_occ (mean % ON of the healthy
    phase mates in the bin)."""
    dets, nb = O["dets"], O["nb"]
    nd = len(dets)
    st = np.array([status1.get(int(d), None) for d in dets], dtype=object)
    ph = np.where(np.isfinite(O["phase"]), O["phase"], -1.0)
    occ = O["occ"]
    hl = (st != "bad")[:, None] & ~np.isnan(occ) if nd else np.zeros((0, nb), bool)
    ho = np.where(hl, occ, np.nan)
    peer = np.full((nd, nb), np.nan)
    for p in np.unique(ph):
        mem = np.flatnonzero(ph == p)
        hs_, hc_ = _ksum(ho[mem])
        if p < 0:
            continue
        for i in mem:
            den = hc_ - hl[i].astype(int)
            with np.errstate(invalid="ignore", divide="ignore"):
                peer[i] = (hs_ - np.nan_to_num(ho[i], nan=0.0)) / np.where(den > 0, den, np.nan)
    return dict(O, dets_i=dets.astype(int), ph=ph, st=st, hl=hl, ho=ho, peer_occ=peer)


# ============================================================================ h108 bin statistics
def slopes(M, m30):
    """count / time-ON elasticity vs phase traffic, all bins and the busier half (h108_base.bin_stats)."""
    if m30:
        return pd.DataFrame(columns=["detector", "elhi_cnt", "elhi_occ"])
    n, occ, ref = M["n"], M["occ"], M["ref"]
    nd, nb = n.shape
    ok = ~np.isnan(n) & ~np.isnan(ref)
    rmed = np.array([np.median(ref[i][ok[i]]) if ok[i].any() else np.nan for i in range(nd)])
    with np.errstate(invalid="ignore"):
        use = ok & (ref >= rmed[:, None])
    has = use.any(1)
    L = int(use.sum(1).max()) if nd else 0
    Z = np.full((7, nd, max(L, 1)), np.nan)
    for i in np.flatnonzero(has):
        r_, n_, o_ = ref[i][use[i]], n[i][use[i]], occ[i][use[i]]
        lr, ln, lo = np.log(r_ + 1.0), np.log(n_ + 1.0), np.log(o_ + 0.005)
        k = len(lr)
        Z[0, i, :k], Z[1, i, :k], Z[2, i, :k], Z[3, i, :k] = 1.0, lr, ln, lo
        Z[4, i, :k], Z[5, i, :k], Z[6, i, :k] = lr ** 2, lr * ln, lr * lo
    one, slr, sln, slo, x2, xn, xo = (_ksum_rows(Z[j])[0] for j in range(7))
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = x2 - slr ** 2 / one
        good = (one >= 6) & (vx > 0.5)
        cnt_ = np.where(good, (xn - slr * sln / one) / vx, np.nan)
        occ_ = np.where(good, (xo - slr * slo / one) / vx, np.nan)
    out = pd.DataFrame({"detector": M["dets_i"][has], "elhi_cnt": cnt_[has], "elhi_occ": occ_[has]})
    return out.astype({"detector": "int64", "elhi_cnt": "float64", "elhi_occ": "float64"})


def like_corr(M, m30):
    """c_occ_like: own 15-min time ON vs healthy same-function phase mates (else all healthy mates) (h108 context)."""
    if m30:
        return pd.DataFrame(columns=["detector", "c_occ_like"])
    occ, ho, hl, ph, peer = M["occ"], M["ho"], M["hl"], M["ph"], M["peer_occ"]
    fn = M["fn"]
    nd, nb = occ.shape
    same = np.full((nd, nb), np.nan)
    key = [(p, f) if isinstance(f, str) else None for p, f in zip(ph, fn)]
    for k in set(x for x in key if x is not None):
        mem = [i for i in range(nd) if key[i] == k]
        fs, fc = _ksum(ho[mem])
        if k[0] < 0:
            continue
        for i in mem:
            den = fc - hl[i].astype(int)
            with np.errstate(invalid="ignore", divide="ignore"):
                same[i] = (fs - np.nan_to_num(ho[i], nan=0.0)) / np.where(den > 0, den, np.nan)

    def corr(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < 6:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0, 1])
    out = []
    with _quiet():
        for i, d in enumerate(M["dets_i"]):
            out.append((int(d), corr(occ[i], same[i]), corr(occ[i], peer[i])))
    C = pd.DataFrame(out, columns=["detector", "c_occ_same", "c_occ_all"]).astype(
        {"detector": "int64", "c_occ_same": "float64", "c_occ_all": "float64"})
    C["c_occ_like"] = C.c_occ_same.fillna(C.c_occ_all)
    return C[["detector", "c_occ_like"]]


GREEN_GRACE_S, GREEN_SHARE, GREEN_MIN_S = 5.0, 0.20, 60.0     # note 135


def _per_bin(s, f, bs, nb):
    """seconds of the sorted, disjoint intervals [s, f) in each bin of bs seconds (the last bin takes the rest)."""
    out = np.zeros(nb)
    if not len(s) or not nb:
        return out
    edges = np.arange(nb + 1, dtype=np.float64) * bs
    edges[-1] = np.inf
    cum = np.r_[0.0, np.cumsum(f - s)]

    def cover(x):                      # covered seconds before x
        j = np.searchsorted(s, x, "right")
        c = cum[j]
        jm = j - 1
        ok = jm >= 0
        over = np.zeros(len(x))
        over[ok] = np.maximum(f[jm[ok]] - x[ok], 0.0)
        return c - over
    cv = cover(edges[:-1])
    return np.diff(np.r_[cv, cum[-1]])


def _isect(a0, a1, b0, b1):
    """intersection of two sorted lists of disjoint intervals."""
    if not len(a0) or not len(b0):
        return np.zeros(0), np.zeros(0)
    # every a-interval against the b-intervals it can touch
    lo = np.searchsorted(b1, a0, "right")
    hi = np.searchsorted(b0, a1, "left")
    n = np.maximum(hi - lo, 0)
    if not n.sum():
        return np.zeros(0), np.zeros(0)
    ia = np.repeat(np.arange(len(a0)), n)
    ib = np.concatenate([np.arange(x, y) for x, y in zip(lo[n > 0], hi[n > 0])])
    s, f = np.maximum(a0[ia], b0[ib]), np.minimum(a1[ia], b1[ib])
    k = f > s
    return s[k], f[k]


def _green_after_grace(P, T, p):
    """green intervals of phase p after their first GREEN_GRACE_S seconds (the colour_bins timeline: events 1 / 8 /
    9 / 10, state changes only; a green with a comms gap in it is not used)."""
    m = np.isin(P.eid, (1, 8, 9, 10)) & (P.par == int(p))
    tc, ec = P.t[m], P.eid[m]
    o = np.lexsort((ec, tc))
    tc, ec = tc[o], ec[o]
    if not len(tc):
        return np.zeros(0), np.zeros(0)
    st = _COL[ec]
    keep = np.r_[True, st[1:] != st[:-1]]
    tc, st = tc[keep], st[keep]
    a, z = tc, np.r_[tc[1:], T]
    g = st == 0
    if len(P.g0):
        k = np.searchsorted(P.g0, a)
        kk = np.minimum(k, len(P.g0) - 1)
        g &= ~((k < len(P.g0)) & (P.g0[kk] < z))
    a, z = a[g] + GREEN_GRACE_S, z[g]
    k = z > a
    return a[k], z[k]


def _on_intervals(P, T, ch):
    """continuous ONs of one channel as health_core.events_to_bins counts them for the occupancy."""
    V = hc._memo(P, ("onview_all", float(T)), lambda: _on_view_all(P, T))
    i = V["idx"].get(int(ch))
    if i is None:
        return np.zeros(0), np.zeros(0)
    m = V["ci"] == i
    s, f = V["s"][m], V["f"][m]
    lead = V["lead"].get(int(ch))
    if lead is not None:
        s, f = np.r_[0.0, s], np.r_[lead, f]
    o = np.argsort(s, kind="stable")
    return s[o], f[o]


def _on_view_all(P, T):
    dets = np.unique(P.pd_).astype(int)
    V = hc._on_view(P, T, dets)
    fin = np.isfinite(V["dur"])
    return dict(idx={int(d): i for i, d in enumerate(dets)}, ci=V["ci"][fin], s=V["ton"][fin],
                f=np.minimum(V["ton"][fin] + V["dur"][fin], T),
                lead={int(c): float(t) for t, c in zip(V["lead_t"], V["lead_c"])})


def green_on(P, T, ch, p, bs, nb):
    """note 135: per bin, the phase's green time after its first 5 s (gav) and the zone's ON time inside it (gon)."""
    gs, gf = _green_after_grace(P, T, p)
    os_, of_ = _on_intervals(P, T, ch)
    is_, if_ = _isect(os_, of_, gs, gf)
    return _per_bin(gs, gf, bs, nb), _per_bin(is_, if_, bs, nb)


def occ_hi(M, cmode, refs, P=None, T=None, bins: dict | None = None):
    """the Count 'held ON' minutes (score_v4c.occ_hi_c; the Count-zone variant of 'stuck on').  cmode: detector ->
    ''/pulse/normal/unknown (only detectors of the scored table; every one of them is a phase mate for the 'quiet'
    test, only Count zones are scored).  Note 135: with the window's events P a Count zone's held-ON period also needs
    the zone ON for >= 20 % of its phase's green time after the first 5 s of each green (>= 60 s of such green in the
    period): a car waiting through the red leaves soon after green starts.  `bins` (a dict, optional) receives
    {detector: bool array of the held-ON periods} for the bad-period list."""
    cols = ["detector", "hi_min_c", "hi_occ_max"]
    dets = M["dets_i"]
    cm = [cmode.get(int(d)) for d in dets]
    keep = np.array([isinstance(c, str) and c != "unknown" for c in cm], bool)
    if not keep.any():
        return pd.DataFrame(columns=cols)
    idx = np.flatnonzero(keep)
    n, occ, ref, cong, ph = M["n"][idx], M["occ"][idx], M["ref"][idx], M["cong"][idx], M["ph"][idx]
    fn = M["fn"][idx]
    bs = int(M["bs"])
    with _quiet(), np.errstate(invalid="ignore", divide="ignore"):
        rmax = np.nanmax(ref, 1) if ref.shape[1] else np.full(len(idx), np.nan)
        tl = np.clip(np.floor(5 * ref / np.where(rmax > 0, rmax, np.nan)[:, None]), 0, 4)
    # the pulse-zone limit per traffic level (refs.occ_lim_pulse / t_pass)
    lp = np.full(tl.shape, np.nan)
    for t in range(5):
        lp[tl == t] = refs._occ_p.get((bs, t), np.nan)
    tp = refs._tp.get(bs, np.nan)
    ps = np.full(occ.shape, np.nan)
    pc = np.zeros(occ.shape, np.int64)
    for p in np.unique(ph):
        mem = np.flatnonzero(ph == p)
        s_, k_ = _ksum(occ[mem])
        ps[mem], pc[mem] = s_, k_
    with np.errstate(invalid="ignore", divide="ignore"):
        pocc = (ps - np.nan_to_num(occ, nan=0.0)) / np.where(pc > 1, pc - 1, np.nan)
        pocc = np.where(ph[:, None] >= 0, pocc, np.nan)
        quiet = (cong < 1.5) & ~(pocc >= 0.40)
        cnt = (fn == "Count")[:, None]
        lim_c = np.where(cnt, np.fmax(np.fmax(lp, n * tp / bs), 0.20), np.nan)
        hi_c = (occ > lim_c) & quiet & (tl <= 1)
    if P is not None:
        for r in np.flatnonzero(cnt[:, 0] & hi_c.any(1)):
            p = ph[r]
            if not p >= 0:
                hi_c[r] = False
                continue
            gav, gon = green_on(P, T, int(dets[idx[r]]), int(p), bs, hi_c.shape[1])
            hi_c[r] &= (gav >= GREEN_MIN_S) & (gon >= GREEN_SHARE * gav)
    hic = hi_c.sum(1)
    mx = np.array([occ[r][hi_c[r]].max() if hi_c[r].any() else np.nan for r in range(len(idx))], float)
    if bins is not None:
        bins.update({int(dets[idx[r]]): hi_c[r] for r in np.flatnonzero(hi_c.any(1))})
    return pd.DataFrame({"detector": dets[idx].astype("int64"), "hi_min_c": hic * bs / 60, "hi_occ_max": mx})


# ============================================================================ per-actuation statistics
def _median_sorted(v, a, m):
    """np.median of the m values v[a:a + m] (v sorted within each group), one per group: the middle value, or for an
    even count numpy's linear interpolation b - (b - a) * 0.5."""
    lo = a + (m - 1) // 2
    hi = np.minimum(lo + 1, a + m - 1)
    va, vb = v[lo], v[hi]
    return np.where(m % 2 == 1, va, vb - (vb - va) * 0.5)


def act_stats(P):
    """h108_events.stats for every channel at once: clean ON durations (an ON after a non-ON, ended by an OFF, not
    over a comms gap) -> dur_p50 (>= 5 of them; Count mode), longest continuous ON (max_on_s), time held ON by
    continuous ONs containing an ON logged again without an OFF (rep_time_s), share of such re-logged ONs (rep_frac).
    Channels with no ON have no row."""
    cols = ["detector", "rep_frac", "max_on_s", "rep_time_s", "dur_p50"]
    td, ed, pd_ = P.td, P.ed, P.pd_
    g0 = P.g0
    if not len(td):
        return pd.DataFrame(columns=cols)
    chans, first = np.unique(pd_, return_index=True)
    nch = len(chans)
    ci = np.repeat(np.arange(nch), np.diff(np.r_[first, len(pd_)]))
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    same_next = np.r_[pd_[1:] == pd_[:-1], False]
    prev_e = np.where(same_prev, np.r_[0, ed[:-1]], 0)
    next_e = np.where(same_next, np.r_[ed[1:], 0], 0)
    next_t = np.r_[td[1:], np.nan]
    on = ed == 82
    rep = on & (prev_e == 82)
    clean = on & (prev_e != 82) & (next_e == 81)
    ic = np.flatnonzero(clean)
    dur = next_t[ic] - td[ic]
    keep = ~_spans_gap(g0, td[ic], td[ic] + dur)
    ic, dur = ic[keep], dur[keep]
    n_on = np.bincount(ci[on], minlength=nch)
    # continuous ONs: start = an ON after a non-ON; it ends at the channel's next OFF
    st = np.flatnonzero(on & (prev_e != 82))
    offs = np.flatnonzero(ed == 81)
    kk = np.searchsorted(offs, st)
    okk = kk < len(offs)
    st, kk = st[okk], kk[okk]
    oi = offs[kk]
    same = pd_[oi] == pd_[st]
    st, oi = st[same], oi[same]
    cl = td[oi] - td[st]
    cl = np.where(_spans_gap(g0, td[st], td[oi]), 0.0, cl)
    cr = np.cumsum(rep)
    nrep = cr[oi - 1] - cr[st]
    cst = ci[st]
    rep_time = np.zeros(nch)
    sel = nrep > 0
    if sel.any():                                  # np.sum per channel (its own pairwise order), rows in channel order
        ks = cst[sel]
        u, f0 = np.unique(ks, return_index=True)
        rep_time[u] = [float(x.sum()) for x in np.split(cl[sel], f0[1:])]
    max_on = np.full(nch, np.nan)
    if len(cl):
        u, f0 = np.unique(cst, return_index=True)
        max_on[u] = np.maximum.reduceat(cl, f0)
    p50 = np.full(nch, np.nan)
    if len(dur):                                   # the clean ONs are in channel order: sort each channel's own
        u, f0, m = np.unique(ci[ic], return_index=True, return_counts=True)
        for k, a0, mm in zip(u, f0, m):
            if mm >= 5:
                v = np.sort(dur[a0:a0 + mm])
                p50[k] = _median_sorted(v, np.array([0]), np.array([mm]))[0]
    has = n_on > 0
    out = pd.DataFrame({"detector": chans[has].astype(np.int64),
                        "rep_frac": np.bincount(ci[rep], minlength=nch)[has] / n_on[has],
                        "max_on_s": max_on[has], "rep_time_s": rep_time[has], "dur_p50": p50[has]})
    return out[cols]


def act118(P, T):
    """h118_events.stats for every channel at once: ONs logged again without an OFF (n_rep), vehicle starts, and the
    continuous-ON lengths (not over a comms gap): >= 5 s, >= 60 s, longest (pulse zones that hold ON)."""
    td, ed, pd_ = P.td, P.ed, P.pd_
    cols = ["detector", "n_on_a", "n_rep", "n_start", "n_ge5", "n_ge60", "long_max_s"]
    if not len(td):
        return pd.DataFrame(columns=cols)
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    on = ed == 82
    rep = on & same_prev & np.r_[False, ed[:-1] == 82]
    end, start = hc.cont_on(P, T)
    si = np.where(start)[0]
    cdur = end[si] - td[si]
    cdur = np.where(_spans_gap(P.g0, td[si], end[si]), np.nan, cdur)
    sch = pd_[si]
    chans, first, cnt = np.unique(sch, return_index=True, return_counts=True)
    if not len(chans):
        return pd.DataFrame(columns=cols)
    k = np.repeat(np.arange(len(chans)), cnt)
    fin = np.isfinite(cdur)
    n_on_c = np.bincount(pd_[on], minlength=65)
    n_rep_c = np.bincount(pd_[rep], minlength=65)
    mx = np.full(len(chans), -np.inf)
    np.maximum.at(mx, k[fin], cdur[fin])
    return pd.DataFrame({"detector": chans.astype(np.int64), "n_on_a": n_on_c[chans].astype(np.int64),
                         "n_rep": n_rep_c[chans].astype(np.int64), "n_start": cnt.astype(np.int64),
                         "n_ge5": np.bincount(k[fin & (cdur >= 5)], minlength=len(chans)).astype(np.int64),
                         "n_ge60": np.bincount(k[fin & (cdur >= 60)], minlength=len(chans)).astype(np.int64),
                         "long_max_s": np.where(np.isfinite(mx), mx, np.nan)}, columns=cols)


# ============================================================================ stuck episodes + queue context (h110)
F4_COVER = 0.75
R1_FULL, R1_LIGHT = 0.90, 0.30


def episode_overlap(E):
    """co5: other detectors held ON (>= 300 s episodes) through at least half of the episode."""
    if not len(E):
        return np.zeros(0, int)
    s = E.t0.astype("int64").to_numpy() / 1e9          # epoch seconds, as the research (same float rounding)
    f = E.t1.astype("int64").to_numpy() / 1e9
    c = E.detector.to_numpy()
    ov = np.minimum(f[:, None], f[None]) - np.maximum(s[:, None], s[None])
    m = (ov >= 0.5 * (f - s)[:, None]) & (c[:, None] != c[None])
    return np.array([len(set(c[row].tolist())) for row in m], int)


BUSY_MIN, BUSY_OCC = 60.0, 0.5      # note 135: a held ON >= 1 h is a queue only if the same traffic holds it elsewhere
REF_HELD = 0.95                      # ... unless a detector of that traffic yardstick was itself held ON in the hold


def busy_traffic(o, ref, ia, iz, bs):
    """note 135 (research h135_lift.busy): the bins outside the ON (and one bin either side) whose phase traffic `ref`
    was at least the ON's mean traffic -> their minutes and the detector's median % ON in them."""
    nb = len(o)
    bo = np.arange(nb)
    inep = (bo >= ia) & (bo <= iz - 1)
    rh = ref[inep & np.isfinite(ref)]
    if not len(rh):
        return dict(busy_min_e=0.0, busy_occ_e=np.nan)
    r_h = float(np.mean(rh))
    with np.errstate(invalid="ignore"):
        m = ((bo < ia - 1) | (bo > iz)) & np.isfinite(ref) & np.isfinite(o) & (ref >= r_h)
    k = int(m.sum())
    return dict(busy_min_e=k * bs / 60.0, busy_occ_e=float(np.median(o[m])) if k else np.nan)


def ref_held(occ, mem, ia, iz):
    """note 135 fix (research h135_lift.ref_held): a detector of the traffic yardstick `mem` (rows of occ) >= 95 % ON in
    a bin of the hold [ia, iz - 1] -- its counts were buried under the queue, the hold's traffic is under-measured."""
    a, z = max(ia, 0), max(min(iz, occ.shape[1]), 0)
    if not len(mem) or z <= a:
        return False
    with np.errstate(invalid="ignore"):
        return bool((occ[mem, a:z] >= REF_HELD).any())


def episodes_ctx(E, M, phase_x, qtype, typ):
    """queue context per stuck ON (h110_resolve.episodes + traffic_evidence).  E: episodes (ts / tf seconds from the
    window start, dur_s), M: the bins_ctx matrices, phase_x: detector -> phase (scored table only), qtype: type -> p95
    % ON of healthy detectors, typ: detector -> type.  Per phase the mates' bins are a small matrix; the mates' mean
    % ON per bin is the research's pandas mean over the same rows (same compensated sum; cached per set of mates)."""
    cols = ["n_hpeer_e", "phx_h_e", "corr_h_e", "refx_e", "light_e", "cover_e", "trafx_e", "busy_min_e", "busy_occ_e",
            "ref_held_e"]
    if not len(E):
        return pd.DataFrame(columns=cols)
    dets = M["dets_i"]
    bs = int(M["bs"]) if len(dets) else 900
    nb = int(M["nb"]) if len(dets) else 0
    row = {int(d): i for i, d in enumerate(dets)}
    qt_all = np.array([qtype.get(typ.get(int(d)), np.nan) if typ.get(int(d)) is not None else np.nan for d in dets],
                      float)
    piv = {}
    for p in np.unique(M["ph"]):
        mem = np.flatnonzero(M["ph"] == p)
        piv[float(p)] = (dets[mem], M["occ"][mem], M["n"][mem], M["st"][mem], M["fn"][mem], qt_all[mem], mem)
    bo = np.arange(nb)
    pm_cache, out = {}, []
    for r in E.itertuples():
        p = phase_x.get(int(r.detector), np.nan)
        res = {}
        i0 = row.get(int(r.detector))
        if i0 is None or not nb:
            out.append(res)
            continue
        ia, iz = int(r.ts // bs), int(np.ceil(r.tf / bs))
        inep = (bo >= ia) & (bo <= iz - 1)
        o, ref = M["occ"][i0], M["ref"][i0]
        keep = []
        phx_h, corr_h, cover = np.nan, np.nan, np.nan
        if np.isfinite(p) and p in piv:
            dl, occ, n, st, fn, qt, mem = piv[p]
            a0, a1 = max(ia, 0), max(min(iz, nb), 0)
            cand = (dl != r.detector) & np.array([x != "bad" for x in st])
            for j in np.flatnonzero(cand):
                seg = occ[j, a0:a1]
                if a1 > a0 and (seg >= .99).sum() / (a1 - a0) >= .8:
                    continue
                keep.append(int(dl[j]))
            if keep:
                kk = tuple(keep)
                if (p, kk) not in pm_cache:
                    s_, k_ = _ksum(occ[np.isin(dl, keep)])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        pm_cache[(p, kk)] = np.where(k_ > 0, s_ / np.where(k_ > 0, k_, 1), np.nan)
                pm = pm_cache[(p, kk)]
                with _quiet():
                    phx_h = np.nanmean(pm[inep]) / max(np.nanmean(pm), 1e-9) if inep.any() else np.nan
                corr_h = _corr(o[~inep], pm[~inep])
                ki = np.isin(dl, keep)
                with np.errstate(invalid="ignore"):
                    hit = (occ[ki, a0:a1] >= qt[ki, None]).any(0)
                nb_ = int(inep.sum())
                cover = int(hit.sum()) / max(nb_, 1) if nb_ else np.nan
        with _quiet():
            refx = np.nanmean(ref[inep]) / max(np.nanmean(ref), 1e-9) if inep.any() else np.nan
            rm = np.nanmean(ref)
        near = (bo >= ia - 1) & (bo <= iz)
        po = M["peer_occ"][i0]
        with _quiet():
            quiet = ~(po > np.nanmean(po)) if np.isfinite(po).any() else np.ones(len(o), bool)
        with np.errstate(invalid="ignore"):
            light = ~near & (o >= R1_FULL) & (ref <= R1_LIGHT * rm) & quiet
        res.update(n_hpeer_e=len(keep), phx_h_e=phx_h, corr_h_e=corr_h, refx_e=refx, light_e=int(light.sum()),
                   cover_e=cover)
        res.update(busy_traffic(o, ref, ia, iz, bs))
        if "ref_mem" in M:
            res["ref_held_e"] = ref_held(M["occ"], M["ref_mem"][i0], ia, iz)
        # traffic evidence: the phase's Advance / Count detectors (else its other mates) during the ON
        if np.isfinite(p) and p in piv:
            dl, occ, n, st, fn, qt, mem = piv[p]
            oth = dl != r.detector
            tr = oth & np.isin(fn.astype(str), TRAF) & np.array([f is not None for f in fn])
            if not tr.any():
                tr = oth
            if tr.any():
                tot = np.nansum(n[tr], 0)                     # integer counts: exact in any order
                ja, jz = ia, iz
                fa, fz = int(np.ceil(r.ts / bs)), int(r.tf // bs)
                if fz > fa:
                    ja, jz = fa, fz
                ins = tot[max(ja, 0):max(min(jz, nb), 0)]
                res["trafx_e"] = ins.mean() / max(tot.mean(), 1e-9) if len(ins) else np.nan
        out.append(res)
    C = pd.DataFrame(out, index=E.index)
    for c in cols:
        if c not in C:
            C[c] = np.nan
    return C


# ============================================================================ per-start view, too fast, too many
G_, Y_, RG_, R_, U_ = 0, 1, 2, 3, 4
RED_GUARD = 2.0
BIN5 = 300


_COL = np.full(256, -1, np.int8)
_COL[[1, 8, 9, 10]] = [0, 1, 2, 2]                    # begin green -> G, begin yellow -> Y, end yellow / red clear -> red


def _phase_timeline(tc, ec, Tlen, g0):
    """segments of one phase (h117 phase_timeline, vectorised): boundaries b (m+1), state s (m), valid v (m); red
    split into Rg (its first 2 s) + R."""
    st = _COL[np.asarray(ec, int)]
    keep = np.r_[True, st[1:] != st[:-1]]                      # 9 then 10 -> one red change
    tc, st = np.asarray(tc, float)[keep], st[keep]
    red = st == 2
    z = np.r_[tc[1:], Tlen]
    idx = np.repeat(np.arange(len(tc)), np.where(red, 2, 1))
    first = np.r_[True, idx[1:] != idx[:-1]] if len(idx) else np.zeros(0, bool)
    b = np.where(first, tc[idx], np.minimum(tc[idx] + RED_GUARD, z[idx]))
    s = np.where(red[idx], np.where(first, RG_, R_), np.where(st[idx] == 0, G_, Y_)).astype(np.int8)
    b = np.r_[b, Tlen].astype(float)
    v = np.ones(len(s), bool)
    if len(g0):
        k = np.searchsorted(g0, b[:-1])
        kk = np.minimum(k, len(g0) - 1)
        v &= ~((k < len(g0)) & (g0[kk] < b[1:]))
    return b, s, v


def start_view(P, T):
    """the vehicle starts of the window (an ON whose previous event on its channel is not an ON), in channel / time
    order, computed once per window: times x, channel row ci into chans (every channel with 81 / 82 events), 5-min bin
    cell (ci * nb + bin), fast (start -> start < 1 s, whole ticks) and re-trigger (OFF -> ON < 0.3 s) flags."""
    return hc._memo(P, ("starts", float(T)), lambda: _start_view(P, T))


def _start_view(P, T):
    td, ed, pd_ = P.td, P.ed, P.pd_
    nb = int(np.ceil(T / BIN5))
    chans = np.unique(pd_)
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    prev_e = np.r_[0, ed[:-1]]
    prev_t = np.r_[np.nan, td[:-1]]
    start = (ed == 82) & ~(same_prev & (prev_e == 82))
    gap_off = np.where(same_prev & (prev_e == 81), td - prev_t, np.nan)
    si = np.where(start)[0]
    x, ch_, go = td[si], pd_[si], gap_off[si]
    ci = np.searchsorted(chans, ch_)
    ioi = np.r_[np.nan, np.diff(x)]
    ioi[np.r_[True, ch_[1:] != ch_[:-1]]] = np.nan              # first start of each channel
    with np.errstate(invalid="ignore"):
        f1 = np.rint(ioi / 0.1) < 10
        chat = np.rint(go / 0.1) < 3
    cell = ci * nb + np.minimum((x // BIN5).astype(int), nb - 1)
    return dict(chans=chans, nb=nb, x=x, ci=ci, cell=cell, f1=f1, chat=chat)


def q5_stats(P, T, phase, lanes, bins: dict | None = None):
    """too many for the traffic: per detector the busiest 5 min in vehicles per hour per lane -- vehicle starts in
    green + yellow of its predicted phase per green + yellow second (q5_gy; >= 60 s of it in the bin) and all starts
    (q5_all), with the float32 arithmetic of the research SQL (h117 / h118c).  Returns (table, colour state of every
    start); `bins` receives the per-5-min series for the bad-period list."""
    cols = ["detector", "q5_gy", "q5_all"]
    V = start_view(P, T)
    chans, nb = V["chans"], V["nb"]
    nd = len(chans)
    if not nd:
        return pd.DataFrame(columns=cols), np.zeros(0, np.int8)
    t, eid, par = P.t, P.eid, P.par
    edges = np.minimum(np.arange(nb + 1) * BIN5, T)
    g0 = P.g0
    tl = {}
    # color events in time order (ties: as the research file order after its (time, code, param) sort)
    cm = np.isin(eid, (1, 8, 9, 10))
    tc, ec, pc = t[cm], eid[cm], par[cm]
    o = np.lexsort((pc, ec, tc))
    ts_, es_, ps_ = tc[o], ec[o], pc[o]
    for p in np.unique(ps_[es_ == 1]):
        mm = ps_ == p
        if (es_[mm] == 1).sum() < 3:
            continue
        b, s_, v = _phase_timeline(ts_[mm], es_[mm], T, g0)
        d = np.diff(b)
        cg = np.r_[0.0, np.cumsum(d * ((s_ == G_) & v))]
        cy = np.r_[0.0, np.cumsum(d * ((s_ == Y_) & v))]
        tl[int(p)] = (b, s_, v, np.diff(np.interp(edges, b, cg)) + np.diff(np.interp(edges, b, cy)))
    x, ci = V["x"], V["ci"]
    state = np.full(len(x), U_, np.int8)
    sGY = np.zeros((nd, nb))
    for k, c in enumerate(chans):
        p = phase.get(int(c), np.nan)
        Tl = tl.get(int(p)) if p is not None and np.isfinite(p) else None
        if Tl is None:
            continue
        b, s_, v, secs = Tl
        m = ci == k
        j = np.searchsorted(b, x[m], "right") - 1
        okj = (j >= 0) & (j < len(s_))
        jj = np.clip(j, 0, len(s_) - 1)
        state[m] = np.where(okj & v[jj], s_[jj], U_).astype(np.int8)
        sGY[k] = secs
    gyst = (state == G_) | (state == Y_)
    cell = V["cell"]
    n = np.bincount(cell, minlength=nd * nb).astype(np.float32).reshape(nd, nb)
    ngy = np.bincount(cell[gyst], minlength=nd * nb).astype(np.float32).reshape(nd, nb)
    s32 = sGY.astype(np.float32)
    ln = np.array([lanes.get(int(c), np.nan) for c in chans], float)
    ln = np.fmax(np.where(np.isnan(ln), 1.0, ln), 1.0)[:, None]
    f32 = np.float32
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"), _quiet():
        q5_gy = np.where(s32 >= 60, (ngy / s32 * f32(3600)).astype(np.float64) / ln, np.nan)
        q5_all = (n / f32(300) * f32(3600)).astype(np.float64) / ln
        q5g = np.nanmax(q5_gy, 1) if nb else np.full(nd, np.nan)
        q5a = np.nanmax(q5_all, 1) if nb else np.full(nd, np.nan)
    if bins is not None:
        bins.update(det=chans.astype(np.int64), q5_gy=q5_gy, q5_all=q5_all)
    return pd.DataFrame({"detector": chans.astype(np.int64), "q5_gy": q5g, "q5_all": q5a}), state


PHI_MAX, PHI_MIN_N = 2.0, 8          # note 135


def fast_even(P, T, state):
    """note 135 (research h135_fast): per detector, are its fast starts (< 1 s after the previous start, not a
    re-trigger < 0.3 s after an OFF) the same share of its starts in every 15-min period?  fe_n = periods with starts,
    fe_p = share, fe_phi = dispersion of the per-period counts around that share (1 = as even as chance allows).
    state: the colour state of every start (q5_stats); as the research, starts in the first 2 s of red are left out
    of the fast and re-trigger counts."""
    cols = ["detector", "fe_n", "fe_p", "fe_phi"]
    V = start_view(P, T)
    chans, nb = V["chans"], V["nb"]
    nd = len(chans)
    if not nd or not len(V["x"]):
        return pd.DataFrame(columns=cols)
    cell, f1, chat = V["cell"], V["f1"], V["chat"]
    nrg = state != RG_
    x5 = np.bincount(cell[f1 & ~chat & nrg], minlength=nd * nb).reshape(nd, nb).astype(np.float64)
    n5 = (np.bincount(cell, minlength=nd * nb) - np.bincount(cell[chat & nrg], minlength=nd * nb)).reshape(nd, nb)
    n5 = n5.astype(np.float64)
    nq = -(-nb // 3)
    pad = np.zeros((nd, nq * 3 - nb))
    x = np.concatenate([x5, pad], 1).reshape(nd, nq, 3).sum(2)          # integer counts: exact in any order
    n = np.concatenate([n5, pad], 1).reshape(nd, nq, 3).sum(2)
    ok = n > 0
    k = ok.sum(1)
    sx, sn = np.where(ok, x, 0).sum(1), np.where(ok, n, 0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = sx / sn
        c = np.where(ok, (x - p[:, None] * n) ** 2 / (p[:, None] * (1 - p[:, None]) * n), 0.0).sum(1)
        good = (k >= 2) & (p > 0) & (p < 1)
        phi = np.where(good, c / np.maximum(k - 1, 1), np.nan)
    return pd.DataFrame({"detector": chans.astype(np.int64), "fe_n": k.astype(np.int64), "fe_p": p,
                         "fe_phi": phi})[cols]


STOPBAR = ("Count", "Presence", "Yellow_Red")
TF_MIN_ON, TF_MIN_GAPS = 50, 3        # too-fast: scored from 50 ONs, a finding needs >= 3 too-fast gaps


def too_fast(P, fn, lanes, gap_sb=0.8, gap_other=0.6, bins: dict | None = None):
    """note 141 item 1: every ON-to-ON gap (every ON, also an ON logged again without an OFF) shorter than 0.8 s /
    lanes (Count, Presence, Yellow_Red) or 0.6 s / lanes (the others), compared in whole 0.1-s ticks; per detector
    N ONs, n_fast gaps, fast_rate per 1,000 ONs, pair_share (share of the short gaps whose next gap is not short:
    pairs of ONs, note 147e).  fn / lanes: detector -> predicted function / lane span.  `bins` receives
    {detector: (hour of each fast gap, hour of each ON)} for the bad-period list."""
    cols = ["detector", "N", "n_fast", "fast_rate", "gap_lim", "pair_share"]
    td, ed, pd_ = P.td, P.ed, P.pd_
    on = ed == 82
    x, ch = td[on], pd_[on]
    if not len(x):
        return pd.DataFrame(columns=cols)
    chans, N = np.unique(ch, return_counts=True)
    ci = np.repeat(np.arange(len(chans)), N)
    lim = np.array([(gap_sb if fn.get(int(c)) in STOPBAR else gap_other) / max(lanes.get(int(c), 1.0), 1.0)
                    for c in chans], float)
    same = np.r_[False, ch[1:] == ch[:-1]]
    tk = np.rint(np.r_[np.nan, np.diff(x)] / 0.1)
    with np.errstate(invalid="ignore"):
        f = same & (tk < (lim * 10)[ci])
    nf = np.bincount(ci[f], minlength=len(chans))
    nxt_same = np.r_[same[1:], False]
    nxt_f = np.r_[f[1:], False]
    g = f & nxt_same
    ng = np.bincount(ci[g], minlength=len(chans))
    npair = np.bincount(ci[g & ~nxt_f], minlength=len(chans))
    if bins is not None:
        hb = (x // 3600).astype(np.int64)
        for k in np.flatnonzero(nf):
            m = ci == k
            bins[int(chans[k])] = (hb[m & f], hb[m])
    return pd.DataFrame({"detector": chans.astype(np.int64), "N": N.astype(np.int64), "n_fast": nf.astype(np.int64),
                         "fast_rate": 1000.0 * nf / np.maximum(N, 1), "gap_lim": lim,
                         "pair_share": np.where(ng > 0, npair / np.maximum(ng, 1), np.nan)})


# ============================================================================ time of day (h116, 24-h samples)
NIGHT_TOD = (1, 5)
CNT_FLOOR, MIN_N, MIN_NIGHT, MIN_MATE = 0.5, 50, 20, 200
Z_LO, Z_HI, D_PH = 3.5, 6.0, 0.30
SIG_SHARE, SIG_MIN = 0.20, 5


def day_profile(n15, occ15):
    """hourly counts N and mean fraction ON O from 96 15-min bins (NaN = not covered), as h116_data: a 30-min value
    is the sum / mean of its covered 15-min bins (NaN when neither is); N needs both 30-min halves, O averages them."""
    a = n15.reshape(len(n15), 48, 2)
    q = occ15.reshape(len(occ15), 48, 2)
    nb = np.isfinite(a).sum(2)
    with _quiet():
        H = np.where(nb > 0, np.nansum(a, 2), np.nan)
        Q = np.where(np.isfinite(q).sum(2) > 0, np.nanmean(q, 2), np.nan)
        N = H.reshape(len(H), 24, 2).sum(2)
        O = np.nanmean(Q.reshape(len(Q), 24, 2), 2)
    return N, O


def _busiest4(A):
    k = 4
    Ac = np.concatenate([A, A[:, :k - 1]], 1)
    cs = np.cumsum(np.pad(Ac, ((0, 0), (1, 0))), 1)
    return ((cs[:, k:] - cs[:, :-k]) / k).max(1)


def tod_features(N, O):
    N = np.nan_to_num(np.asarray(N, float))
    O = np.clip(np.nan_to_num(np.asarray(O, float)), 0, 1)
    a, b = NIGHT_TOD
    nn = N[:, a:b].mean(1)
    r = np.sqrt((nn + CNT_FLOOR) / (_busiest4(N) + CNT_FLOOR))
    on = np.arcsin(np.sqrt(O[:, a:b].mean(1)))
    od = np.arcsin(np.sqrt(np.clip(_busiest4(O), 0, 1)))
    lv = np.log10(1 + N.sum(1) * 24.0 / 24)
    return np.column_stack([r, on, od, lv]), N[:, a:b].sum(1)


def band_of(rate):
    rate = np.asarray(rate, float)
    return np.where(rate < 20.0, "low", np.where(rate < 100.0, "medium", "high"))


def tod_z(X, fn, span, band, ok, ref):
    """note-116 night z per detector-day from a reference {group: model}; NaN when not scored."""
    groups = ref["groups"]
    z = np.full(len(X), np.nan)
    gk = []
    for f, s, b, o in zip(fn, span, band, ok):
        k = None
        if o:
            for kk in (f"{f}|{s}|{b}", f"{f}|*|{b}", f"{f}|{s}|*", f"{f}|*|*"):
                if kk in groups:
                    k = kk
                    break
        gk.append(k)
    for i, k in enumerate(gk):
        if k is None:
            continue
        g = groups[k]
        A = np.array([1.0, X[i, 2], X[i, 3]])
        zr = (X[i, 0] - A @ np.asarray(g["bc"], float)) / float(g["sc"])
        if g["occ"]:
            zo = (X[i, 1] - A[[0, 2]] @ np.asarray(g["bo"], float)) / float(g["so"])
            zr = max(zr, zo)
        z[i] = (zr - float(g["cal_med"])) / float(g["cal_mad"])
    return z, gk


def phase_delta(r, phase, tot, ok):
    """own night level minus the median of its phase mates' (same predicted phase, ok, >= 200 a day)."""
    m = len(r)
    mate_r = np.full(m, np.nan)
    good = ok & (tot >= MIN_MATE)
    for p in np.unique(phase[np.isfinite(phase)]):
        idx = np.flatnonzero(phase == p)
        if len(idx) < 2:
            continue
        for j, i in enumerate(idx):
            mm = good[idx].copy()
            mm[j] = False
            if mm.any():
                mate_r[i] = np.median(r[idx][mm])
    return r - mate_r, mate_r



def night_day(N, O, fn, span, phase, ok_day, ref_own, ref_alt, alts):
    """note-116 night check for ONE signal-day with the model-unsure rule (h118c_band.build): z = own type's z, the
    smallest over the plausible functions when the model is unsure; phase-mate layer dph; level."""
    X, n_night = tod_features(N, O)
    tot = np.nansum(np.nan_to_num(N), 1)
    ok = ok_day
    band = band_of(tot / 24)
    z, gk = tod_z(X, fn, span, band, ok, ref_own)
    dph, mate_r = phase_delta(X[:, 0], phase, tot, ok & np.isfinite(z))
    z_eff = z.copy()
    for f in FNS:
        m = np.array([f in a for a in alts]) & (np.asarray(fn) != f)
        if not m.any():
            continue
        za, _ = tod_z(X, [f] * len(X), span, band, ok, ref_alt)
        z_eff = np.where(m & np.isfinite(za) & np.isfinite(z_eff), np.fmin(z_eff, za), z_eff)
    return pd.DataFrame(dict(tod_z=z_eff, tod_dph=dph, tod_n_night=n_night, tod_r=X[:, 0], tod_mate_r=mate_r,
                             tod_odd=ok & (np.nan_to_num(z_eff, nan=-9) >= Z_LO), tod_group=gk))


def busy_level(z, dph, n_night, zh, P=None):
    """busy at night (note 147e rule at the limit zh per detector): every finding needs the night level >= .30 above
    its phase mates' (P['mates_delta']); suspect from z >= zh x 3.5 / 6, bad from z >= zh; >= 20 night ONs; NaN z = not
    scored.  Returns the score: .8 bad, .5 suspect, 0, NaN."""
    P = P or {"mates_delta": D_PH, "z_x": Z_LO / Z_HI, "min_night": MIN_NIGHT}
    z, zh = np.asarray(z, float), np.asarray(zh, float)
    d = np.nan_to_num(np.asarray(dph, float), nan=-1)
    with np.errstate(invalid="ignore"):
        flag = (np.asarray(n_night, float) >= P["min_night"]) & (d >= P["mates_delta"]) & (z >= zh * P["z_x"])
        bad = flag & (z >= zh)
    return np.where(np.isfinite(z), np.where(bad, 0.8, np.where(flag, 0.5, 0.0)), np.nan)


# ============================================================================ goes silent with a clean yardstick
def _best_run(n, occ, cov, i, S):
    x = n[i]
    n_on = x[cov].sum()
    quiet = (x == 0) & (occ[i] < 1) & cov
    s0, s1 = hc._runs(quiet)
    best = (0.0, -1, -1)
    if len(s0) and n_on > 0:
        cS = np.r_[0, np.cumsum(np.where(cov, S, 0))]
        for a, b in zip(s0, s1):
            s_in = cS[b] - cS[a]
            s_out = cS[-1] - s_in
            lam = n_on / max(s_out, 1) * s_in
            if lam > best[0]:
                best = (lam, a, b)
    return best


def dropout_clean(B5, cands, phase_x, flagged):
    """h118c_dropout for the candidate detectors (drop_lam >= 30): the silent-run expectation with phase mates that
    carry a (non-silence) finding left out.  Returns {detector: dict}."""
    n, occ, cov = B5["n_on"].astype(float), B5["occ"].astype(float), B5["cov"]
    dets = np.array([int(d) for d in B5["dets"]])
    live = n[:, cov].sum(1) > 0
    phase = {int(d): phase_x.get(int(d), np.nan) for d in dets}
    ref, kind, _ = hc._refs(dets, live, phase, hc.twins_of(B5))
    fl = np.array([bool(flagged.get(int(d), False)) for d in dets])
    tot = n.sum(0)
    out = {}
    for d in cands:
        if int(d) not in set(dets.tolist()):
            continue
        i = int(np.flatnonzero(dets == int(d))[0])
        if kind[i] == "phase":
            S0 = n[ref[i]].sum(0)
            cand = np.array(ref[i])
        else:
            S0 = tot - n[i]
            cand = np.flatnonzero(np.arange(len(dets)) != i)
        lam0 = _best_run(n, occ, cov, i, S0)
        dropped = [int(dets[j]) for j in cand if fl[j]]
        if kind[i] == "phase":
            keep = [j for j in ref[i] if not fl[j]]
            if len(keep) >= 2:
                S1 = n[keep].sum(0)
            else:
                oth = [j for j in range(len(dets)) if j != i and not fl[j]]
                S1 = n[oth].sum(0)
                dropped = [int(dets[j]) for j in range(len(dets)) if j != i and fl[j]]
        else:
            oth = [j for j in range(len(dets)) if j != i and not fl[j]]
            S1 = n[oth].sum(0)
        lam1 = _best_run(n, occ, cov, i, S1)
        out[int(d)] = dict(drop_lam_chk=lam0[0], drop_lam_c=lam1[0], drop_b0_c=lam1[1], drop_b1_c=lam1[2],
                           n_dropped=len(dropped), dropped="+".join(map(str, dropped)))
    return out
