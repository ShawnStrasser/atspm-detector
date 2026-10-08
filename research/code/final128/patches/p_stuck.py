import os
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4.py")
s = open(p, encoding="utf-8").read()
a = s.index("def _stuck_stats(X, E, lim):")
b = s.index("def _night_frac_v(t0, b0, b1):")
new = '''def _kahan(vals):
    """pandas groupby sum of one group (compensated, in order)."""
    sm = cp = 0.0
    for v in vals:
        if v != v:
            continue
        y = v - cp
        t = sm + y
        cp = (t - sm) - y
        if cp != cp:
            cp = 0.0
        sm = t
    return sm


def _stuck_stats(X, E, lim):
    """per scored detector: its continuous ONs over the limit `lim` and how many of them are queues (Q1 / Q1b),
    shared (held ON with >= 3 others) or counted, with the counted time (research h110 F3 / F4; note 128: numpy, the
    groupby of the research reproduced -- counts, a compensated time sum, NaN rows for detectors without one)."""
    cols = ["n_over", "n_ep", "tot_s", "n_q1", "n_q1b", "n_shared", "n_shared_unq"]
    n = len(X)
    if not len(E):
        return pd.DataFrame(np.nan, index=np.arange(n), columns=cols)
    pos = {int(d): i for i, d in enumerate(X.detector.to_numpy())}
    ri = np.array([pos.get(int(d), -1) for d in E.detector.to_numpy()], int)
    L = np.where(ri >= 0, np.asarray(lim, float)[np.maximum(ri, 0)], np.nan)
    num = lambda c: E[c].to_numpy(float)  # noqa: E731
    dur = num("dur_s")
    with np.errstate(invalid="ignore"):
        keep = (ri >= 0) & (dur >= L)
        q_ok = ((np.nan_to_num(num("n_hpeer_e"), nan=0.0) > 0) & (num("phx_h_e") >= 1.5) & (num("corr_h_e") >= Q1_CORR)
                & (num("refx_e") >= 0.5) & (np.nan_to_num(num("light_e"), nan=1.0) == 0) & (num("trafx_e") >= TRAF_X))
        q1 = q_ok & (dur < 3600)
        q1b = q_ok & (dur >= 3600) & (num("cover_e") >= F4_COVER)
        shared = num("co5") >= 3
    count = ~q1 & ~q1b & ~shared
    cdur = np.where(count, dur, 0.0)
    sh_unq = shared & ~q1 & ~q1b
    if not keep.any():
        return pd.DataFrame(np.nan, index=np.arange(n), columns=cols)
    r = ri[keep]
    u = np.unique(r)
    idx = [np.flatnonzero(r == k) for k in u]
    cnt = lambda a: np.array([int(a[keep][ii].sum()) for ii in idx], np.int64)  # noqa: E731
    S = pd.DataFrame({"n_over": np.array([len(ii) for ii in idx], np.int64), "n_ep": cnt(count),
                      "tot_s": np.array([_kahan(cdur[keep][ii].tolist()) for ii in idx], float), "n_q1": cnt(q1),
                      "n_q1b": cnt(q1b), "n_shared": cnt(shared), "n_shared_unq": cnt(sh_unq)}, index=u)
    return S.reindex(np.arange(n))


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("patched")
