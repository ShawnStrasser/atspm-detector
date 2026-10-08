import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_core.py")
s = open(p, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:90], s.count(a))
    s = s.replace(a, b)


rep('''def _act_stats(ev, start, end, min_on: int = 20, light: bool = False):
    T = (end - start).total_seconds()
    P_ = _prep(ev, start, end)
    if P_.empty:
        return None''', '''def _act_light(P_, start, end, min_on: int = 20):
    """act_stats(light=True) for every detector at once (note 128): the columns the packaged rules read (ACOLS:
    pulse share, median ON, ON -> ON shares, bursts, short ONs overall and by 3-h block), the same numbers as the
    per-detector loop (shares = counts / counts, medians = the middle values halved as np.median)."""
    T = (end - start).total_seconds()
    g0, g1 = P_.g0, P_.g1
    cov_s = T - float((g1 - g0).sum())

    def spans_gap(a, b):
        if not len(g0):
            return np.zeros(len(a), bool)
        k = np.searchsorted(g0, a)
        kk = np.minimum(k, len(g0) - 1)
        return (k < len(g0)) & (g0[kk] < b)
    td, ed, pd_ = P_.td, P_.ed, P_.pd_
    same_next = np.r_[pd_[1:] == pd_[:-1], False]
    nxt_t, nxt_e = np.r_[td[1:], np.nan], np.r_[ed[1:], 0]
    on = ed == 82
    dur = np.where(same_next & (nxt_e == 81), nxt_t - td, np.nan)[on]
    ton, ch = td[on], pd_[on]
    if not len(ton):
        return None
    s0 = start.hour * 3600 + start.minute * 60 + start.second
    blk = (((s0 + ton) // 10800) % 8).astype(int)                 # 3-h clock block of each ON
    bad = spans_gap(ton, ton + np.nan_to_num(dur, nan=0.0))
    dur[bad] = np.nan
    dets, first = np.unique(ch, return_index=True)
    nd = len(dets)
    n_c = np.r_[first[1:], len(ch)] - first
    k = np.repeat(np.arange(nd), n_c)
    same = np.r_[False, ch[1:] == ch[:-1]]                         # the ON has an earlier ON on its channel
    ioi = np.r_[np.nan, np.diff(ton)]
    ioi[~same] = np.nan
    ioi[np.r_[False, spans_gap(ton[:-1], ton[1:])] & same] = np.nan
    fd = np.isfinite(dur)
    fj = np.isfinite(ioi)

    def cnt(msk, size=nd, key=k):
        return np.bincount(key[msk], minlength=size)
    with np.errstate(invalid="ignore"):
        jt = _ticks(ioi)
        nf, n_pulse = cnt(fd), cnt(fd & (dur <= 0.15))
        n_sh1, n_sh2 = cnt(fd & (dur < 0.15)), cnt(fd & (dur < SHORT_S))
        nj, n_j05, n_j1 = cnt(fj), cnt(fj & (jt < IOI_TICKS_05)), cnt(fj & (jt < IOI_TICKS_1))
        bm = (np.nan_to_num(jt, nan=990.0) < IOI_TICKS_1) & same
    # bursts: >= 4 successive ON->ON intervals under 1 s (5 ONs inside ~4 s), per channel
    r0, r1 = _runs(bm)
    L = r1 - r0
    big = L >= 4
    burst_len = np.bincount(k[r0[big]], weights=L[big], minlength=nd).astype(np.int64)
    burst_n = np.bincount(k[r0[big]], minlength=nd)
    # median ON per channel (finite durations, sorted within channel)
    df, kf = dur[fd], k[fd]
    v = df[np.lexsort((df, kf))]
    st0 = np.cumsum(nf) - nf
    # short ONs by 3-h block
    cb = kf * 8 + blk[fd]
    cntb = np.bincount(cb, minlength=nd * 8).reshape(nd, 8)
    shb = np.bincount(cb[df < SHORT_S], minlength=nd * 8).reshape(nd, 8)
    rows = []
    for c in range(nd):
        n = int(n_c[c])
        r = {"detector": int(dets[c]), "a_n_on": n}
        if n < min_on or nf[c] < min_on:
            rows.append(r)
            continue
        m = int(nf[c])
        a = int(st0[c])
        med = v[a + m // 2] if m % 2 else (v[a + m // 2 - 1] + v[a + m // 2]) / 2.0
        r["pulse_frac"] = float(n_pulse[c] / nf[c])
        r["med_dur"] = float(med)
        r["ioi_lt05"] = float(n_j05[c] / nj[c]) if nj[c] else np.nan
        r["ioi_lt1"] = float(n_j1[c] / nj[c]) if nj[c] else np.nan
        k_, nr = int(burst_len[c]), int(burst_n[c])
        r["burst_frac"], r["n_burst"] = (k_ + nr) / n, nr
        r["n_dur"] = m
        r["short1"] = float(n_sh1[c] / nf[c])
        r["short2"] = float(n_sh2[c] / nf[c])
        sb = [(shb[c, b] / cntb[c, b], b) for b in range(8) if cntb[c, b] >= SHORT_BLK_MIN]
        if sb:
            r["short_blk_min"] = float(min(sb)[0])
            r["short_blk_max"], r["short_blk"] = float(max(sb)[0]), int(max(sb)[1])
        rows.append(r)
    st = pd.DataFrame(rows)
    st["a_cov_h"] = cov_s / 3600
    return st


def _act_stats(ev, start, end, min_on: int = 20, light: bool = False):
    T = (end - start).total_seconds()
    P_ = _prep(ev, start, end)
    if P_.empty:
        return None
    if light:
        return _act_light(P_, start, end, min_on)''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
