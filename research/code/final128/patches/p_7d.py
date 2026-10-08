import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_core.py")
s = open(p, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:90], s.count(a))
    s = s.replace(a, b)


# continuous ONs computed once per window
rep('''def continuous_on(td, ed, pd_, T):''', '''def cont_on(P_, T):
    """continuous_on of a Prep's detector events, computed once per window (read only)."""
    return _memo(P_, ("cont", float(T)), lambda: continuous_on(P_.td, P_.ed, P_.pd_, T))


def continuous_on(td, ed, pd_, T):''')
rep('''    c_end, c_start = continuous_on(td, ed, pd_, T)
    dur = np.where(c_start, c_end - td, np.nan)[on]''', '''    c_end, c_start = cont_on(P_, T)
    dur = np.where(c_start, c_end - td, np.nan)[on]''')
rep('''    end_t, start_ = continuous_on(td, ed, pd_, T)
    oe_all = end_t >= T''', '''    end_t, start_ = cont_on(P_, T) if isinstance(ev, Prep) else continuous_on(td, ed, pd_, T)
    oe_all = end_t >= T''')
# continuous-ON starts per channel: one split instead of a mask per channel
rep('''def _cont_on_starts0(ev, start, end) -> dict:''', '''def _cont_on_starts0(ev, start, end) -> dict:
    P_ = _prep(ev, start, end)
    t, par = P_.td, P_.pd_
    if not len(t):
        return {}
    st_ = cont_on(P_, (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds())[1]
    idx = np.flatnonzero(st_)
    if not len(idx):
        return {}
    ch = par[idx]
    u, first = np.unique(ch, return_index=True)               # the events are ordered by channel
    return {int(c): t[a] for c, a in zip(u, np.split(idx, first[1:]))}


def _cont_on_starts_ref(ev, start, end) -> dict:''')
# clock hour of every ON in integer nanoseconds (pandas' float-seconds conversion reproduced)
rep('''def _on_hours(on: dict, start) -> dict:
    """{detector: clock hour of each ON} (one Timestamp conversion for all detectors)."""
    ks = list(on)
    hh = (start + pd.to_timedelta(np.concatenate([on[k][0] for k in ks]), unit="s")).hour.to_numpy()
    cut = np.cumsum([len(on[k][0]) for k in ks])[:-1]
    return dict(zip(ks, np.split(hh, cut)))''', '''def _sec_ns(v) -> np.ndarray:
    """float seconds -> integer nanoseconds exactly as pd.to_timedelta(v, unit="s") (whole seconds, then the fraction
    rounded to 9 decimals and truncated)."""
    v = np.asarray(v, dtype=np.float64)
    base = v.astype(np.int64)
    return base * 1_000_000_000 + (np.round(v - base, 9) * 1_000_000_000).astype(np.int64)


def _on_hours(on: dict, start) -> dict:
    """{detector: clock hour of each ON} (= (start + pd.to_timedelta(t, unit="s")).hour, in integer nanoseconds)."""
    ks = list(on)
    ns = int(pd.Timestamp(start).value) + _sec_ns(np.concatenate([on[k][0] for k in ks]))
    hh = ((ns // 3_600_000_000_000) % 24).astype(np.int32)
    cut = np.cumsum([len(on[k][0]) for k in ks])[:-1]
    return dict(zip(ks, np.split(hh, cut)))''')
open(p, "w", encoding="utf-8").write(s)

p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4_stats.py")
s = open(p, encoding="utf-8").read()
rep('''    g0 = P.g0
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    on = ed == 82
    rep = on & same_prev & np.r_[False, ed[:-1] == 82]
    start = on & ~rep
    offs = np.where(ed == 81)[0]
    n = len(td)
    k = np.searchsorted(offs, np.arange(n))
    p = offs[np.minimum(k, max(len(offs) - 1, 0))] if len(offs) else np.zeros(n, int)
    okk = (k < len(offs)) & (pd_[p] == pd_) if len(offs) else np.zeros(n, bool)
    end = np.where(okk, td[p] if len(offs) else T, T)''', '''    g0 = P.g0
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    on = ed == 82
    rep = on & same_prev & np.r_[False, ed[:-1] == 82]
    # the continuous ONs (= start & end above in the research: health_core.continuous_on, shared per window)
    end, start = hc.cont_on(P, T)''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
