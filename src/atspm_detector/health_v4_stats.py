"""Detector health v4 -- statistics from the hi-res log for ONE signal window (notes 104-121, adopted in note 124).

Every function here is a production port of one research event pass (research/code/health*/ ...), computed on the
window's prepared event arrays (`health_core.Prep`) and the classifier's own answers (predicted phase, function,
lane span).  numpy + pandas only; no loop over events (loops are over detectors, phases or episodes).  The phase
number is only a key to that phase's colour events and to "who are its phase mates" -- never a feature.

Research source of each block (same arithmetic, same order):
  occ_bins / bins_ctx            h96_occ.one, h104_resolve.load_bins          (15-min bins: counts, % ON, traffic, congestion;
                                                                              detector x bin matrices since note 128)
  n3_stats                       h104_resolve.n3_stats                        (erratic time ON)
  slopes / like_corr / occ_hi    h108_base.bin_stats, h108_resolve.context / occ_hi, score_v4c.occ_hi_c
  act_stats / act118             h108_events.stats, h118_events.stats         (ON durations, extension, starts)
  shape110                       h110_events.one                              (yardstick, 15-min bins, episodes)
  level110 / drop_evidence       h110_resolve.level110, h118_resolve.drop_evidence
  erratic                        h118_erratic.off_bins
  episodes_ctx                   h110_resolve.episodes + traffic_evidence     (queue test per stuck ON)
  colour_stats                   h117_events.one + h117_study + h118c_fast    (red / green, too fast, too many;
                                                                              the research SQL in numpy since note 128)
  day_profile / night / band     h116_data + tod116 + h118c_band              (time of day, 24-h samples)
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
        share[i] = tot[i] / max(np.nansum(r), 1)
    return dict(dets=dets, n=n, occ=occ, cov=cov, ref=ref, cong=cong, tot=tot, mocc=mocc, share_ref=share,
                hour=B["hour"], fn=F, phase=Ph, bs=bs, nb=nb, B=B)


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


# ============================================================================ N3 erratic time ON (h104 n3_stats)
N3_X, N3_PT, N3_FULL = 3.0, 0.05, 0.90


def _roll_median(A, w=9, minp=3):
    """per row: centred rolling median over w bins with at least minp finite values (= pandas
    rolling(w, center=True, min_periods=minp).median())."""
    from numpy.lib.stride_tricks import sliding_window_view
    nd, nb = A.shape
    h = w // 2
    Ap = np.concatenate([np.full((nd, h), np.nan), A, np.full((nd, h), np.nan)], 1)
    W = sliding_window_view(Ap, w, axis=1)
    cnt = np.isfinite(W).sum(2)
    Ws = np.sort(W, axis=2)                                    # NaN last
    lo = np.take_along_axis(Ws, np.clip((cnt - 1) // 2, 0, w - 1)[..., None], 2)[..., 0]
    hi = np.take_along_axis(Ws, np.clip(cnt // 2, 0, w - 1)[..., None], 2)[..., 0]
    with np.errstate(invalid="ignore"):
        m = (lo + hi) / 2.0
    return np.where((cnt >= minp) & (cnt > 0), m, np.nan)


def n3_stats(M):
    """h104 n3_stats: minutes of time ON per detector its counts do not explain (n3_exc) in how many bins (n3_n)."""
    n, occ, cong, peer, bs = M["n"], M["occ"], M["cong"], M["peer_occ"], M["bs"]
    pl, cl = _roll_median(peer), _roll_median(cong)
    with np.errstate(invalid="ignore"):
        busy2 = np.where(~np.isnan(peer), peer >= 1.5 * np.clip(pl, .01, None), cong >= 1.5 * np.clip(cl, .05, None))
        q = (n >= 3) & ~busy2 & (occ < N3_FULL)
        val = occ * bs / n
    sel, dbar = [], []
    for i in range(len(M["dets_i"])):
        v = val[i][q[i]]
        if len(v) >= 4:
            sel.append(i)
            dbar.append(np.median(v))
    sel = np.array(sel, int)
    with np.errstate(invalid="ignore"):
        e = n[sel] * np.array(dbar, float)[:, None] / bs
        o_ = occ[sel]
        spk = (n[sel] >= 1) & (o_ < N3_FULL) & (o_ >= N3_X * e) & (o_ - e >= N3_PT) & ~busy2[sel]
        exc = np.where(spk, (o_ - e) * bs / 60, 0.0)
    out = pd.DataFrame({"detector": M["dets_i"][sel], "n3_n": spk.sum(1), "n3_exc": _ksum_rows(exc)[0]})
    return out.astype({"detector": "int64", "n3_n": "int64", "n3_exc": "float64"})


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


def occ_hi(M, cmode, refs):
    """occ_hi8 (h108 occ_hi, every class) and the Count 'held ON' minutes (score_v4c.occ_hi_c).  cmode: detector ->
    ''/pulse/normal/unknown (only detectors of the scored table)."""
    cols = ["detector", "occ_hi8", "hi_min_c", "hi_occ_max"]
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
    # per-type limits (refs.occ_lim / occ_lim_pulse / t_pass, looked up once per detector and level)
    lim = np.full(tl.shape, np.nan)
    lp = np.full(tl.shape, np.nan)
    for r, (f, c) in enumerate(zip(fn, [cm[i] for i in idx])):
        for t in range(5):
            m = tl[r] == t
            if m.any():
                if isinstance(f, str):
                    lim[r, m] = refs._occ.get((f, c, bs, t), np.nan)
                lp[r, m] = refs._occ_p.get((bs, t), np.nan)
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
        hi = (occ > lim) & quiet
        cnt = (fn == "Count")[:, None]
        lim_c = np.where(cnt, np.fmax(np.fmax(lp, n * tp / bs), 0.20), lim)
        hi_c = (occ > lim_c) & quiet & np.where(cnt, tl <= 1, True)
    hi_n8, hic = hi.sum(1), hi_c.sum(1)
    mx = np.array([occ[r][hi_c[r]].max() if hi_c[r].any() else np.nan for r in range(len(idx))], float)
    return pd.DataFrame({"detector": dets[idx].astype("int64"), "occ_hi8": hi_n8 * bs >= 1800,
                         "hi_min_c": hic * bs / 60, "hi_occ_max": mx})


# ============================================================================ per-actuation statistics
def _runs_of(mask, min_len):
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]
    ln = b - a
    k = ln >= min_len
    return int(ln[k].sum()), int(k.sum())


def act_stats(P):
    """h108_events.stats: clean ON durations (dur_p50 -> Count mode), longest continuous ON, time held ON by
    continuous ONs containing an ON logged again without an OFF (rep_time_s)."""
    td, ed, pd_ = P.td, P.ed, P.pd_
    g0 = P.g0
    rows = []
    if not len(td):
        return pd.DataFrame(columns=["detector", "dur_p50", "max_on_s", "rep_time_s", "rep_frac"])
    chans, first = np.unique(pd_, return_index=True)
    last = np.r_[first[1:], len(pd_)]
    for c, a0, a1 in zip(chans, first, last):
        x, ev = td[a0:a1], ed[a0:a1]
        on = ev == 82
        n_on = int(on.sum())
        if n_on == 0:
            continue
        prev_e = np.r_[0, ev[:-1]]
        next_e = np.r_[ev[1:], 0]
        next_t = np.r_[x[1:], np.nan]
        rep = on & (prev_e == 82)
        clean = on & (prev_e != 82) & (next_e == 81)
        dur = (next_t - x)[clean]
        dur = dur[~_spans_gap(g0, x[clean], x[clean] + dur)]
        st = np.where(on & (prev_e != 82))[0]
        offs = np.where(ev == 81)[0]
        kk = np.searchsorted(offs, st)
        okk = kk < len(offs)
        cl = x[offs[kk[okk]]] - x[st[okk]] if okk.any() else np.array([])
        if okk.any():
            cl = np.where(_spans_gap(g0, x[st[okk]], x[offs[kk[okk]]]), 0.0, cl)
            cr = np.cumsum(rep)
            nrep = cr[offs[kk[okk]] - 1] - cr[st[okk]]
            rep_time = float(cl[nrep > 0].sum())
        else:
            rep_time = 0.0
        r = dict(detector=int(c), rep_frac=float(rep.sum() / n_on), max_on_s=float(cl.max()) if len(cl) else np.nan,
                 rep_time_s=rep_time, dur_p50=np.nan)
        if len(dur) >= 5:
            r["dur_p50"] = float(np.quantile(dur, [.1, .5, .9])[1])
        rows.append(r)
    return pd.DataFrame(rows)


def _runlen(mask, brk):
    m = mask & ~brk
    st = m & ~np.r_[False, m[:-1]]
    rid = np.cumsum(st) * m
    cnt = np.bincount(rid)
    cnt[0] = 0
    return cnt[rid]


def act118(P, T):
    """h118_events.stats: ONs logged again without an OFF (n_rep), vehicle starts, continuous-ON lengths (pulse
    zones that hold ON)."""
    td, ed, pd_ = P.td, P.ed, P.pd_
    cols = ["detector", "n_on_a", "n_rep", "n_start", "n_ge5", "n_ge60", "long_max_s"]
    if not len(td):
        return pd.DataFrame(columns=cols)
    g0 = P.g0
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    on = ed == 82
    rep = on & same_prev & np.r_[False, ed[:-1] == 82]
    # the continuous ONs (= start & end above in the research: health_core.continuous_on, shared per window)
    end, start = hc.cont_on(P, T)
    si = np.where(start)[0]
    cdur = end[si] - td[si]
    cdur = np.where(_spans_gap(g0, td[si], end[si]), np.nan, cdur)
    sch = pd_[si]
    rows = []
    chans, first = np.unique(sch, return_index=True)
    last = np.r_[first[1:], len(sch)]
    n_on_c = np.bincount(pd_[on], minlength=65)
    n_rep_c = np.bincount(pd_[rep], minlength=65)
    for c, a, b in zip(chans, first, last):
        cd = cdur[a:b]
        cf = cd[np.isfinite(cd)]
        rows.append(dict(detector=int(c), n_on_a=int(n_on_c[c]), n_rep=int(n_rep_c[c]), n_start=int(b - a),
                         n_ge5=int((cf >= 5).sum()), n_ge60=int((cf >= 60).sum()),
                         long_max_s=float(cf.max()) if len(cf) else np.nan))
    return pd.DataFrame(rows, columns=cols)


# ============================================================================ h110 yardstick, 15-min bins, episodes
def refs110(dets, live, ph, fnl, tw):
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


def shape110(B5, phase, bad, fprob):
    """yardstick kind (ref110) per detector and the 15-min bins x (own), S (rest of signal), R (yardstick), ok.
    B5 = 5-min events_to_bins; phase: raw predicted phases; bad: pass-1 bad detectors; fprob: function probabilities."""
    dets = B5["dets"]
    n, cov = B5["n_on"].astype(float), B5["cov"]
    live = n[:, cov].sum(1) > 0
    tw = hc.twins_of(B5)
    phd1 = {int(k): v for k, v in (phase or {}).items() if v is not None and np.isfinite(v)}
    phd2 = {k: v for k, v in phd1.items() if k not in bad}
    fl = hc._labels(fprob or None)
    fnl = [fl.get(int(d)) for d in dets]
    a15 = hc._agg(n, 3)
    ok15 = hc._agg(cov[None].astype(float), 3)[0] == 3
    refs = {}
    for tag, phd in (("p1", phd1), ("p2", phd2)):
        _, _, phv = hc._refs(dets, live, phd, tw)
        r_new, k_new = refs110(dets, live, phv, fnl, tw)
        refs[tag] = (r_new, k_new)
    S_rows, x, S, R = [], [], [], []
    tot = a15.sum(0)
    for i, d in enumerate(dets):
        d = int(d)
        tag = "p1" if (d in bad or not bad or not phd1) else "p2"
        r_new, k_new = refs[tag]
        S_rows.append(dict(detector=d, ref110=k_new[i], ref110_dets="+".join(str(int(dets[j])) for j in r_new[i])))
        x.append(a15[i])
        S.append(tot - a15[i])
        R.append(a15[r_new[i]].sum(0) if len(r_new[i]) else np.full(a15.shape[1], np.nan))
    hb = np.asarray(B5["hour"])[: len(ok15) * 3: 3]
    return pd.DataFrame(S_rows), dict(dets=dets.astype(int), x=np.array(x).reshape(len(dets), -1),
                                      S=np.array(S).reshape(len(dets), -1), R=np.array(R).reshape(len(dets), -1),
                                      ok=ok15, hour=hb)


# ============================================================================ count drops (h110 F2, h118 G4)
F2_MIN_E, F2_OWN = 0.02, 0.5


def level110(x, S, ok, w, min_h_bins=8, min_e=F2_MIN_E, own_fall=True):
    m = ok & np.isfinite(S) & np.isfinite(w)
    xs, gs = x[m], (S * w)[m]
    if len(xs) < 8 or xs.sum() < 20 or gs.sum() <= 0:
        return np.nan, 0.0, -1
    cx, cs = np.cumsum(xs), np.cumsum(gs)
    X_, SS = cx[-1], cs[-1]
    k = np.arange(2, len(xs) - 1)
    x1, s1, x2, s2 = cx[k - 1], cs[k - 1], X_ - cx[k - 1], SS - cs[k - 1]
    p0 = X_ / max(SS, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        def ll(xx, ss):
            p = np.where(ss > 0, xx / np.maximum(ss, 1e-9), 0)
            return np.where(xx > 0, xx * np.log(np.maximum(p, 1e-12)), 0) - p * ss
        llr = ll(x1, s1) + ll(x2, s2) - (X_ * np.log(max(p0, 1e-12)) - p0 * SS)
    mh = max(2, min(min_h_bins, len(xs) // 4))
    sup = (k >= mh) & (len(xs) - k >= mh) & (s1 >= min_e * SS) & (s2 >= min_e * SS)
    if own_fall:
        with np.errstate(divide="ignore", invalid="ignore"):
            rat = ((x2 + .5) / np.maximum(s2, 1e-9)) / ((x1 + .5) / np.maximum(s1, 1e-9))
        sup &= (rat < 1) & (x2 / (len(xs) - k) <= F2_OWN * x1 / k)
    llr = np.where(sup, llr, -np.inf)
    if not np.isfinite(llr).any():
        return np.nan, 0.0, -1
    j = int(np.nanargmax(llr))
    ratio = float(((x2[j] + .5) / max(s2[j], 1e-9)) / ((x1[j] + .5) / max(s1[j], 1e-9)))
    return ratio, float(llr[j]), int(np.where(m)[0][k[j]])


def drop_evidence(x, S, ok, w, cut):
    """h118 drop_evidence for one detector: own counts per 15 min before / after the cut, expected after, guard."""
    m = ok & np.isfinite(S)
    gg = S * w
    bb = np.arange(len(x))
    bef, aft = m & (bb < cut), m & (bb >= cut)
    x1, s1, x2, s2 = x[bef].sum(), gg[bef].sum(), x[aft].sum(), gg[aft].sum()
    sh = x1 / max(s1, 1e-9)
    exp_after = sh * s2 / max(aft.sum(), 1)
    own_max = x[m].max() if m.any() else np.nan
    return dict(own_before=x1 / max(bef.sum(), 1), own_after=x2 / max(aft.sum(), 1), exp_after=exp_after,
                g4_unscored0=bool(exp_after > 3.0 * max(own_max, 1.0)), exp_series=np.where(m, sh * gg, np.nan))


# ============================================================================ erratic counts (h118_erratic)
ER_H, ER_Z, ER_MIN_ABS, ER_C, ER_MIN_REF = 2, 3.0, 5.0, 0.15, 10.0


def _movsum(a, h):
    c = np.r_[0.0, np.cumsum(a)]
    n = len(a)
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return c[hi] - c[lo]


def off_bins(x, r, ok, c=ER_C, h=ER_H, z=ER_Z, min_abs=ER_MIN_ABS):
    x = np.where(ok, x, 0.0).astype(float)
    r = np.where(ok & np.isfinite(r), r, 0.0).astype(float)
    okf = ok.astype(float)
    X = _movsum(x, h) - x
    Rr = _movsum(r, h) - r
    N = _movsum(okf, h) - okf
    s = X / np.maximum(Rr, 1e-9)
    e = s * r
    sd = np.sqrt(s * (r + 1) * (1 + s) + (c * e) ** 2)
    scored = ok & (N >= 2) & (Rr >= ER_MIN_REF) & ((X + x) > 0)
    hw = np.maximum(z * sd, min_abs)
    off = scored & (np.abs(x - e) > hw)
    return e, hw, off, scored


def erratic(x, rr, ok, pre):
    """n_off / n_sc / exc for one detector against yardstick rr (h118_erratic.run)."""
    if not np.isfinite(rr).any():
        return {}
    e, hw, off, sc = off_bins(x, rr, ok)
    xx = np.where(ok, x, 0.0)
    return {f"n_off_{pre}": int(off.sum()), f"n_sc_{pre}": int(sc.sum()),
            f"exc_{pre}": float(np.where(off, np.abs(xx - e) - hw, 0.0).sum() / max(xx[sc].sum(), 1.0))}


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


def episodes_ctx(E, M, phase_x, qtype, typ):
    """queue context per stuck ON (h110_resolve.episodes + traffic_evidence).  E: episodes (ts / tf seconds from the
    window start, dur_s), M: the bins_ctx matrices, phase_x: detector -> phase (scored table only), qtype: type -> p95
    % ON of healthy detectors, typ: detector -> type.  Per phase the mates' bins are a small matrix; the mates' mean
    % ON per bin is the research's pandas mean over the same rows (same compensated sum; cached per set of mates)."""
    cols = ["n_hpeer_e", "phx_h_e", "corr_h_e", "refx_e", "light_e", "cover_e", "trafx_e"]
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


# ============================================================================ red / green statistics (h117, h118c)
G_, Y_, RG_, R_, U_ = 0, 1, 2, 3, 4
COLOUR = {1: 0, 8: 1, 9: 2, 10: 2}
RED_GUARD = 2.0


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


def colour_bins(P, T, phase):
    """per detector x 5-min bin: vehicle starts by colour state of its predicted phase, fast starts (ON -> ON < 1 s),
    chatter re-triggers, phase seconds per state (h117_events.one).  Returns (long DataFrame, {detector: n starts}).
    (note 128: every detector at once -- one count per bin and kind, the same numbers as the per-detector loop.)"""
    t, eid, par = P.t, P.eid, P.par
    BIN = 300
    nb = int(np.ceil(T / BIN))
    edges = np.minimum(np.arange(nb + 1) * BIN, T)
    g0 = P.g0
    tl = {}
    # colour events in time order (ties: as the research file order after its (time, code, param) sort; a stable
    # sort of the colour events alone gives them in the same order as sorting every event)
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
        cc = {k: np.r_[0.0, np.cumsum(d * ((s_ == k) & v))] for k in (G_, Y_, RG_, R_)}
        secs = {k: np.diff(np.interp(edges, b, cc[k])) for k in cc}
        tl[int(p)] = (b, s_, v, secs)
    td, ed, pd_ = P.td, P.ed, P.pd_
    n = len(td)
    nstart = {}
    if not n:
        return pd.DataFrame(), nstart
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    prev_e = np.r_[0, ed[:-1]]
    prev_t = np.r_[np.nan, td[:-1]]
    on = ed == 82
    start = on & ~(same_prev & (prev_e == 82))
    gap_off = np.where(same_prev & (prev_e == 81), td - prev_t, np.nan)
    si = np.where(start)[0]
    x, ch_, go = td[si], pd_[si], gap_off[si]
    chans = np.unique(pd_)
    nd = len(chans)
    ci = np.searchsorted(chans, ch_)
    ioi = np.r_[np.nan, np.diff(x)]
    ioi[np.r_[True, ch_[1:] != ch_[:-1]]] = np.nan              # first start of each channel
    with np.errstate(invalid="ignore"):
        f1 = np.rint(ioi / 0.1) < 10
        chat = np.rint(go / 0.1) < 3
    state = np.full(len(x), U_, np.int8)
    S3 = np.zeros((3, nd, nb))
    for k, c in enumerate(chans):
        nstart[int(c)] = int((ci == k).sum())
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
        S3[0, k], S3[1, k], S3[2, k] = secs[G_] + secs[Y_], secs[R_], secs[RG_]
    cell = ci * nb + np.minimum((x // BIN).astype(int), nb - 1)
    gy = (state == G_) | (state == Y_)
    rr = state == R_
    uu = state == U_

    def cnt(msk):
        return np.bincount(cell[msk], minlength=nd * nb).astype(np.float32)
    d = dict(detector=np.repeat(chans.astype(np.int64), nb), b=np.tile(np.arange(nb), nd),
             n=cnt(np.ones(len(x), bool)), nG=cnt(state == G_), nY=cnt(state == Y_), nR=cnt(rr), nU=cnt(uu),
             fGY=cnt(gy & f1), fR=cnt(rr & f1), fU=cnt(uu & f1), cGY=cnt(gy & chat), cR=cnt(rr & chat),
             cU=cnt(uu & chat), xGY=cnt(gy & f1 & ~chat), xR=cnt(rr & f1 & ~chat), xU=cnt(uu & f1 & ~chat),
             sGY=S3[0].ravel().astype(np.float32), sR=S3[1].ravel().astype(np.float32),
             sRg=S3[2].ravel().astype(np.float32))
    return pd.DataFrame(d), nstart


def _win_median(R, h=6):
    """per row of the float32 matrix R (NaN = NULL): DuckDB median(x) OVER (ORDER BY column ROWS BETWEEN h
    PRECEDING AND h FOLLOWING) -- the middle value, for an even count lo + (hi - lo) * 0.5 with the difference in
    float32 and the result rounded to float32 (quantile_cont on FLOAT); NaN where the frame holds no value."""
    from numpy.lib.stride_tricks import sliding_window_view
    nd, nb = R.shape
    w = 2 * h + 1
    pad = np.full((nd, h), np.nan, np.float32)
    W = np.sort(sliding_window_view(np.concatenate([pad, R, pad], 1), w, axis=1), axis=2)
    k = np.isfinite(W).sum(2)
    half = k // 2
    hi = np.take_along_axis(W, np.clip(half, 0, w - 1)[..., None], 2)[..., 0]
    lo = np.take_along_axis(W, np.clip(half - 1, 0, w - 1)[..., None], 2)[..., 0]
    with np.errstate(invalid="ignore"):
        ev = (lo.astype(np.float64) + (hi - lo).astype(np.float64) * 0.5).astype(np.float32)
    out = np.where(k % 2 == 1, hi, ev).astype(np.float32)
    out[k == 0] = np.nan
    return out


def _colour_medians(cb):
    """the six rolling medians of the research SQL (m7 / m8: rates by colour state, raw and without chatter), from
    the float32 bins in the same float32 arithmetic as the SQL (rows: detector-major, every bin present)."""
    nd = len(np.unique(cb.detector.to_numpy()))
    nb = len(cb) // max(nd, 1)
    f = {c: cb[c].to_numpy(np.float32).reshape(nd, nb) for c in ("nG", "nY", "nR", "nU", "cGY", "cR", "cU", "sGY",
                                                                  "sR", "sRg")}
    z = np.float32(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        nGYa = f["nG"] + f["nY"]
        sU = np.maximum(np.float32(300) - f["sGY"] - f["sR"] - f["sRg"], z)
        sGY, sR = f["sGY"], f["sR"]

        def rates(g, r, u):
            return (np.where(sGY >= 30, g / sGY, np.nan).astype(np.float32),
                    np.where(sR >= 30, r / sR, np.nan).astype(np.float32),
                    np.where(sU >= 30, u / sU, np.nan).astype(np.float32))
        r7 = rates(nGYa, f["nR"], f["nU"])
        r8 = rates(np.maximum(nGYa - f["cGY"], z), np.maximum(f["nR"] - f["cR"], z), np.maximum(f["nU"] - f["cU"], z))
    out = {}
    for tag, rr in (("7", r7), ("8", r8)):
        for nm, a in zip(("mg", "mr", "mu"), rr):
            out[nm + tag] = _win_median(a).ravel()
    return out


def _fast_stats(cb):
    """the research SQL of h117_study.stats + h118c_fast.fast_stats (run in DuckDB until note 128) in numpy, with the
    same arithmetic: FLOAT (float32) columns and operations where DuckDB used FLOAT, DOUBLE where it used DOUBLE, sums
    added row after row in bin order (as DuckDB on one thread), exp / sqrt the same libm calls.  cb: detector-major
    rows (every bin present) with the six rolling medians already in it.  -> one row per detector."""
    f32 = np.float32
    nd = len(np.unique(cb.detector.to_numpy()))
    nb = len(cb) // max(nd, 1)
    g = {c: cb[c].to_numpy(f32).reshape(nd, nb) for c in ("n", "nG", "nY", "nR", "nU", "fGY", "fR", "fU", "cGY",
                                                           "cR", "cU", "xGY", "xR", "xU", "sGY", "sR", "sRg", "mg7",
                                                           "mr7", "mu7", "mg8", "mr8", "mu8")}
    ln = cb.ln.to_numpy(np.float64).reshape(nd, nb)
    z = f32(0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        nGYa = g["nG"] + g["nY"]
        sGY, sR = g["sGY"], g["sR"]
        sU = np.maximum(f32(300) - sGY - g["sR"] - g["sRg"], z)
        q5_gy = np.where(sGY >= 60, (nGYa / sGY * f32(3600)).astype(np.float64) / ln, np.nan)
        q5_all = (g["n"] / f32(300) * f32(3600)).astype(np.float64) / ln

        def expect(nGY, nR, nU, mg, mr, mu):
            def term(x, s, m, cap=None):
                r = x / s
                lo = np.fmin(r, np.where(np.isnan(m), r, m))
                if cap is not None:
                    lo = np.fmin(lo, f32(cap))
                return x.astype(np.float64) * (1.0 - np.exp(-(lo.astype(np.float64))))
            eg = np.where(sGY > 0, term(nGY, sGY, mg), 0.0)
            er = np.where(sR > 0, term(nR, sR, mr), 0.0)
            eu = np.where(sU > 0, term(nU, sU, mu, 50), nU.astype(np.float64))
            return eg, er, eu

        def agg(fa, fb, fc, eg, er, eu):
            fo = (fa + fb + fc).astype(np.float64)
            e3 = eg + er + eu
            spk = ((fo - eg - er - eu) / np.sqrt(e3 + 1) >= 4) & (fo >= 5)
            return (np.cumsum(fo, 1)[:, -1] if nb else np.zeros(nd), np.cumsum(e3, 1)[:, -1] if nb else np.zeros(nd),
                    spk.sum(1).astype(np.float64))
        e7 = expect(nGYa, g["nR"], g["nU"], g["mg7"], g["mr7"], g["mu7"])
        fo_all, fem_all, n_spk = agg(g["fGY"], g["fR"], g["fU"], *e7)
        e8 = expect(np.maximum(nGYa - g["cGY"], z), np.maximum(g["nR"] - g["cR"], z), np.maximum(g["nU"] - g["cU"], z),
                    g["mg8"], g["mr8"], g["mu8"])
        fo_c, fem_c, n_spk_c = agg(g["xGY"], g["xR"], g["xU"], *e8)
        with _quiet():
            q5g = np.nanmax(q5_gy, 1) if nb else np.full(nd, np.nan)
            q5a = np.nanmax(q5_all, 1) if nb else np.full(nd, np.nan)
    det = cb.detector.to_numpy()[::nb] if nb else np.unique(cb.detector.to_numpy())
    return pd.DataFrame({"detector": det.astype(np.int64), "q5_gy": q5g, "q5_all": q5a, "fo_all": fo_all,
                         "fem_all": fem_all, "n_spk": n_spk, "fo_c": fo_c, "fem_c": fem_c, "n_spk_c": n_spk_c})


def colour_stats(CB, lanes):
    """too-fast / too-many statistics per detector from the colour bins (h117_study.stats + h118c_fast.fast_stats;
    note 128: the research DuckDB query ported to numpy with the same float32 / float64 arithmetic, so health opens no
    database connection).  lanes: detector -> model lane span.  Rows sorted by detector."""
    cols = ["detector", "q5_gy", "q5_all", "fo_all", "fem_all", "n_spk", "fo_c", "fem_c", "n_spk_c"]
    if CB is None or not len(CB):
        return pd.DataFrame(columns=cols)
    cb = CB.assign(ln=np.fmax(pd.to_numeric(CB.detector.map(lanes), errors="coerce").fillna(1).to_numpy(float), 1),
                   **_colour_medians(CB))
    A = _fast_stats(cb)
    A["zf"] = (A.fo_all - A.fem_all) / np.sqrt(A.fem_all + 1)
    A["zf_c"] = (A.fo_c - A.fem_c) / np.sqrt(A.fem_c + 1)
    return A


# ============================================================================ time of day (h116, h118c band)
NIGHT_TOD = (1, 5)
CNT_FLOOR, MIN_N, MIN_NIGHT, MIN_MATE = 0.5, 50, 20, 200
OCC_FNS = ("Presence", "Other", "Mid")
Z_LO, Z_HI, D_PH = 3.5, 6.0, 0.30
ZH, K_MIN, K_BAD, MATE_EX = 4.0, 3, 7, 2.0
CONG_X, CONG_MIN, CONG_ABS, MIN_GAP = 1.5, 0.20, 0.50, 30
SIG_SHARE, SIG_MIN = 0.20, 5
FLOOR_C = 0.02


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


def tod_decide(z, dph, n_night):
    d = np.nan_to_num(dph, nan=-1)
    hi = z >= Z_HI
    flag = (n_night >= MIN_NIGHT) & (hi | ((z >= Z_LO) & (d >= D_PH)))
    bad = flag & hi & (d >= D_PH)
    return flag, np.where(bad, "bad", np.where(flag, "suspect", "ok"))


def band_feats(N, O):
    tot = np.nansum(N, 1, keepdims=True)
    c = np.sqrt(np.nan_to_num(N) / np.clip(tot, 1, None))
    a = np.arcsin(np.sqrt(np.clip(np.nan_to_num(O), 0, 1)))
    return c, a


def band_group(fn, span, band, day, avail):
    for keys, k in (((fn, span, band, day), 4), ((fn, band, day), 3), ((fn, span, day), 3), ((fn, day), 2)):
        g = "|".join(map(str, keys)) + f"#{k}"
        if g in avail:
            return g
    return None


def band_z(C, A, tot, fn, span, band, day, ok, ref):
    """per-hour z of the counts shape against the type band (h118c_band.oof_z, one reference); band lo / med / hi."""
    n = len(C)
    Zc = np.full((n, 24), np.nan)
    BNDc = np.full((n, 3, 24), np.nan)
    BNDa = np.full((n, 3, 24), np.nan)
    avail = set(ref["avail"])
    G = np.array([band_group(f, s, b, d, avail) if o else None for f, s, b, d, o in zip(fn, span, band, day, ok)],
                 object)
    for i in range(n):
        g = G[i]
        if g is None or g not in ref["groups"]:
            continue
        rg = ref["groups"][g]
        med, sc = np.asarray(rg["mc"], float), np.asarray(rg["sc"], float)
        pois = 1.0 / (4.0 * max(tot[i], 1))
        s = np.sqrt(sc ** 2 + 1.0 * pois)
        Zc[i] = (C[i] - med) / s
        BNDc[i] = np.stack([med - 4 * s, med, med + 4 * s])
        ma, sa = np.asarray(rg["ma"], float), np.asarray(rg["sa"], float)
        BNDa[i] = np.stack([ma - 4 * sa, ma, ma + 4 * sa])
    return Zc, BNDc, BNDa, G


def runs(Zm, zh):
    n, h = Zm.shape
    best = np.zeros(n, int)
    sign = np.zeros(n, int)
    at = np.full((n, 2), -1)
    for s in (1, -1):
        ab = np.nan_to_num(s * Zm, nan=-9) >= zh
        cur = np.zeros(n, int)
        for j in range(h):
            cur = np.where(ab[:, j], cur + 1, 0)
            better = cur > best
            best = np.where(better, cur, best)
            sign = np.where(better, s, sign)
            at[better, 0] = j - cur[better] + 1
            at[better, 1] = j
    return best, sign, at


def _mates(phase, ok, tot):
    good = ok & (tot >= MIN_MATE) & np.isfinite(phase)
    out = {}
    for i in range(len(phase)):
        if not np.isfinite(phase[i]):
            out[i] = np.array([], int)
            continue
        mm = np.flatnonzero((phase == phase[i]) & good)
        out[i] = mm[mm != i]
    return out


def band_day(N, O, fn, span, lanes, phase, ok, day, ref_own, ref_alt, alts):
    """the adopted 'unusual daily pattern' rule for the detectors of ONE signal-day (h118c_band.final with
    alt_run and mate_z).  ref_own / ref_alt: band references (production: the same; research: out-of-fold)."""
    n = len(N)
    tot = np.nansum(N, 1)
    band = band_of(tot / 24)
    C, A = band_feats(N, O)
    Zc, BND, BNDa, G = band_z(C, A, tot, fn, span, band, [day] * n, ok, ref_own)
    # model unsure: the shortest run over the plausible functions' bands
    b_alt = np.full(n, 99)
    for f in FNS:
        m = np.array([f in a for a in alts]) & (np.asarray(fn) != f)
        if not m.any():
            continue
        fn2 = np.where(m, f, np.asarray(fn, object))
        Z2, _, _, _ = band_z(C, A, tot, fn2, span, band, [day] * n, ok & m, ref_alt)
        b, _, _ = runs(Z2, ZH)
        b_alt = np.where(m & np.isfinite(Z2).all(1), np.minimum(b_alt, b), b_alt)
    b, sgn, at = runs(Zc, ZH)
    b_own = b.copy()
    b = np.minimum(b, b_alt)
    mates = _mates(phase, ok, tot)
    ex = np.full(n, np.nan)
    for i in np.flatnonzero(at[:, 0] >= 0):
        mm = mates[i]
        if not len(mm) or not np.isfinite(phase[i]):
            continue
        a, z = at[i]
        with _quiet():
            v = np.nanmedian(sgn[i] * Zc[mm, a:z + 1], 1)
            ex[i] = np.nanmedian(v)
    med = np.clip(BND[:, 1], 0, None) ** 2 * tot[:, None]
    obs, exp_, cong = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    with _quiet():
        day_on = np.nanmean(O[:, 7:19], 1)
    for i in np.flatnonzero(at[:, 0] >= 0):
        a, z = at[i]
        obs[i] = np.nansum(N[i, a:z + 1])
        exp_[i] = np.nansum(med[i, a:z + 1])
        if sgn[i] < 0:
            with _quiet():
                ro = np.nanmean(O[i, a:z + 1])
            cong[i] = (ro >= CONG_ABS) or (ro >= max(CONG_X * day_on[i], CONG_MIN))
    run_ok = (b >= K_MIN) & ok & (np.abs(obs - exp_) >= MIN_GAP)
    mate = np.nan_to_num(ex, nan=-9) >= MATE_EX
    flag = run_ok & ~mate & ~cong
    # phase layer: the same run rule against its phase mates' own day
    Zm = np.full((n, 24), np.nan)
    for i in range(n):
        mm = mates[i]
        if len(mm) < 2 or not ok[i] or not np.isfinite(phase[i]):
            continue
        md = np.median(C[mm], 0)
        sp = 1.4826 * np.median(np.abs(C[mm] - md), 0)
        sc = np.sqrt(sp ** 2 + 1 / (4 * max(tot[i], 1)) + 1 / (4 * np.median(tot[mm])) + FLOOR_C ** 2)
        Zm[i] = (C[i] - md) / sc
    bm, sm, am = runs(Zm, ZH)
    obs_m, exp_m, cong_m = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    for i in np.flatnonzero(bm >= K_MIN):
        a, z = am[i]
        mm = mates[i]
        sh = np.median(N[mm] / np.clip(np.nansum(N[mm], 1, keepdims=True), 1, None), 0) * tot[i]
        obs_m[i] = np.nansum(N[i, a:z + 1])
        exp_m[i] = np.nansum(sh[a:z + 1])
        with _quiet():
            if sm[i] < 0:
                ro = np.nanmean(O[i, a:z + 1])
                cong_m[i] = (ro >= CONG_ABS) or (ro >= max(CONG_X * day_on[i], CONG_MIN))
            else:
                cong_m[i] = np.nanmedian(np.nanmean(O[mm, a:z + 1], 1)) >= CONG_ABS
    run_ok_m = (bm >= K_MIN) & ok & (np.abs(obs_m - exp_m) >= MIN_GAP)
    flag_m = run_ok_m & ~cong_m
    use_m = flag_m & (~flag | (bm > b))
    F = pd.DataFrame(dict(bd_flag=flag | flag_m, bd_ev_from=np.where(use_m, "mates", np.where(flag, "type", "")),
                          bd_ev_h=np.where(use_m, bm, b), bd_ev_dir=np.where(use_m, sm, sgn),
                          bd_ev_h0=np.where(use_m, am[:, 0], at[:, 0]), bd_ev_h1=np.where(use_m, am[:, 1], at[:, 1]),
                          bd_ev_obs=np.where(use_m, obs_m, obs), bd_ev_exp=np.where(use_m, exp_m, exp_),
                          bd_run_ok=run_ok, bd_run_h_own=b_own))
    lo = np.clip(BND[:, 0], 0, None) ** 2 * tot[:, None]
    hi = np.clip(BND[:, 2], 0, None) ** 2 * tot[:, None]
    ba = np.sin(np.clip(BNDa, 0, np.pi / 2)) ** 2 * 100
    return F, dict(lo=lo, med=med, hi=hi, olo=ba[:, 0], omed=ba[:, 1], ohi=ba[:, 2])


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
    _, lvl = tod_decide(np.where(np.isfinite(z_eff), z_eff, -9.0), dph, n_night)
    return pd.DataFrame(dict(tod_z=z_eff, tod_dph=dph, tod_n_night=n_night, tod_r=X[:, 0], tod_mate_r=mate_r,
                             tod_level=np.where(np.isfinite(z_eff), lvl, "not scored"),
                             tod_odd=ok & (np.nan_to_num(z_eff, nan=-9) >= Z_LO), tod_group=gk))


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
