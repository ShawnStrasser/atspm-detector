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


def _kmean_rows(M):
    """pandas groupby mean of each ROW of M (the columns taken left to right, NaN skipped)."""
    s, k = _ksum(M.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, s / np.where(k > 0, k, 1), np.nan)


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
    with _quiet():
        m = np.nanmedian(W, 2) if nb else np.zeros((nd, 0))
    return np.where(cnt >= minp, m, np.nan)


def n3_stats(M):
    """h104 n3_stats: minutes of time ON per detector its counts do not explain (n3_exc) in how many bins (n3_n)."""
    n, occ, cong, peer, bs = M["n"], M["occ"], M["cong"], M["peer_occ"], M["bs"]
    pl, cl = _roll_median(peer), _roll_median(cong)
    with np.errstate(invalid="ignore"):
        busy2 = np.where(~np.isnan(peer), peer >= 1.5 * np.clip(pl, .01, None), cong >= 1.5 * np.clip(cl, .05, None))
        q = (n >= 3) & ~busy2 & (occ < N3_FULL)
        val = occ * bs / n
    rows = []
    for i, d in enumerate(M["dets_i"]):
        v = val[i][q[i]]
        if len(v) < 4:
            continue
        dbar = np.median(v)
        with np.errstate(invalid="ignore"):
            e = n[i] * dbar / bs
            spk = (n[i] >= 1) & (occ[i] < N3_FULL) & (occ[i] >= N3_X * e) & (occ[i] - e >= N3_PT) & ~busy2[i]
            exc = np.where(spk, (occ[i] - e) * bs / 60, 0.0)
        rows.append((int(d), int(spk.sum()), float(_ksum(exc[:, None])[0][0])))
    out = pd.DataFrame(rows, columns=["detector", "n3_n", "n3_exc"])
    return out.astype({"detector": "int64", "n3_n": "int64", "n3_exc": "float64"})


# ============================================================================ h108 bin statistics
def slopes(M, m30):
    """count / time-ON elasticity vs phase traffic, all bins and the busier half (h108_base.bin_stats)."""
    if m30:
        return pd.DataFrame(columns=["detector", "elhi_cnt", "elhi_occ"])
    n, occ, ref = M["n"], M["occ"], M["ref"]
    rows = []
    for i, d in enumerate(M["dets_i"]):
        ok = ~np.isnan(n[i]) & ~np.isnan(ref[i])
        if not ok.any():
            continue
        r_, n_, o_ = ref[i][ok], n[i][ok], occ[i][ok]
        lr, ln, lo = np.log(r_ + 1.0), np.log(n_ + 1.0), np.log(o_ + 0.005)
        hi = r_ >= np.median(r_)
        if not hi.any():
            continue
        lr, ln, lo = lr[hi], ln[hi], lo[hi]
        S = np.column_stack([np.ones(len(lr)), lr, ln, lo, lr ** 2, lr * ln, lr * lo])
        one, slr, sln, slo, x2, xn, xo = (a[None] for a in _ksum(S)[0])       # 1-element arrays: array arithmetic
        with np.errstate(invalid="ignore", divide="ignore"):
            vx = x2 - slr ** 2 / one
            good = bool(((one >= 6) & (vx > 0.5))[0])
            rows.append((int(d), float(((xn - slr * sln / one) / vx)[0]) if good else np.nan,
                         float(((xo - slr * slo / one) / vx)[0]) if good else np.nan))
    out = pd.DataFrame(rows, columns=["detector", "elhi_cnt", "elhi_occ"])
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


