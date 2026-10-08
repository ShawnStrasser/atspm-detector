import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_core.py")
s = open(p, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:90], s.count(a))
    s = s.replace(a, b)


rep('''    if isinstance(ev, Prep):
        return _memo(ev, ("ep", start, end, float(min_s)), lambda: _on_episodes(ev, start, end, min_s)).copy()
    return _on_episodes(ev, start, end, min_s)''', '''    if isinstance(ev, Prep):
        # the >= 15-min list is the >= 5-min list cut at 15 min (both compare ONs held >= CO_MIN_S for co_stuck)
        base = float(min(CO_MIN_S, min_s))
        E0 = _memo(ev, ("ep", start, end, base), lambda: _on_episodes(ev, start, end, base))
        return E0.copy() if float(min_s) == base else E0[E0.dur_s >= min_s].reset_index(drop=True)
    return _on_episodes(ev, start, end, min_s)''')
open(p, "w", encoding="utf-8").write(s)

p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4_stats.py")
s = open(p, encoding="utf-8").read()
rep('''def _kmean_rows(M):''', '''def _ksum_rows(M):
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


def _kmean_rows(M):''')
# rolling median by sorting (the two middle values added and halved, as pandas / numpy)
rep('''    cnt = np.isfinite(W).sum(2)
    with _quiet():
        m = np.nanmedian(W, 2) if nb else np.zeros((nd, 0))
    return np.where(cnt >= minp, m, np.nan)''', '''    cnt = np.isfinite(W).sum(2)
    Ws = np.sort(W, axis=2)                                    # NaN last
    lo = np.take_along_axis(Ws, np.clip((cnt - 1) // 2, 0, w - 1)[..., None], 2)[..., 0]
    hi = np.take_along_axis(Ws, np.clip(cnt // 2, 0, w - 1)[..., None], 2)[..., 0]
    with np.errstate(invalid="ignore"):
        m = (lo + hi) / 2.0
    return np.where((cnt >= minp) & (cnt > 0), m, np.nan)''')
# n3: the minutes summed for all detectors at once
rep('''    rows = []
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
    out = pd.DataFrame(rows, columns=["detector", "n3_n", "n3_exc"])''', '''    sel, dbar = [], []
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
    out = pd.DataFrame({"detector": M["dets_i"][sel], "n3_n": spk.sum(1), "n3_exc": _ksum_rows(exc)[0]})''')
# slopes: the seven sums for every detector at once (each detector's busier-half bins packed to the left)
rep('''    n, occ, ref = M["n"], M["occ"], M["ref"]
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
    out = pd.DataFrame(rows, columns=["detector", "elhi_cnt", "elhi_occ"])''', '''    n, occ, ref = M["n"], M["occ"], M["ref"]
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
    out = pd.DataFrame({"detector": M["dets_i"][has], "elhi_cnt": cnt_[has], "elhi_occ": occ_[has]})''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
