import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4_stats.py")
s = open(p, encoding="utf-8").read()
a = s.index("def colour_bins(P, T, phase):")
b = s.index("# note 128: the six rolling medians")
new = '''def colour_bins(P, T, phase):
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


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("patched")
