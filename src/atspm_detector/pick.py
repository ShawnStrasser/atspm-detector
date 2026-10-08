"""Hi-res-only inputs of the per-lane function decode and of the context stacker (notes 55 / 56 / 59 / 62).

All from ON times on the sample window [t0, t0 + secs) and the model's own predicted phase / function / lanes; no
technology, print or config fact.  Phase numbers are grouping keys only.  Research originals (identical arithmetic):
`research/code/lanes/ln6_pick.py` (span / co-location / track), `ln7_stackhealth.py` (stack clusters),
`trackA/s62_short.py` (twin pairs for short samples).

  span_feats(on, dets, secs)            -> {det: (span flag, gain, covered peers)}, {det: co-located peers}
  track_feats(on, dets, cls, lanes, t0, secs)   -> {det: corr of binned counts with the phase's predicted Count total}
  stack_sizes(on, dets, M)              -> {det: stack size} (structure only; no health statistic, note 114)
  twin_pairs(on, groups, secs, t0)      -> [(a, b, m_ab, m_ba, lag)] for the short-sample twin decode
Channel-order free (note 77): every per-phase loop runs over the detectors in a canonical behavioural order
(`order_dets`: actuations, then the ON offsets -- lanes.content_key), so ties between equal coverages, the greedy choice
of covered peers and the pair orientation never depend on channel numbers.  Callers pass the window start t0.
"""
from __future__ import annotations

import numpy as np

MIN_ON = 10
MIN_SECS = 1800
LAGMAX, BIN, TOL = 2.0, 0.25, 0.5
COVER = 0.5
COLOC = 0.5
SPAN_GAIN = 0.15
TWIN_MIN_N = 3


def content_key(t, t0: float) -> tuple:
    """= lanes.content_key: (actuations, sum and a hash of the ON offsets from t0 in 0.1 s); never the channel."""
    r = np.rint((np.asarray(t, float) - t0) * 10.0).astype(np.int64)
    return int(len(r)), int(r.sum()), int(((r % 65521) * (r % 65519) % 1000003).sum())


def order_dets(on: dict, dets, t0: float) -> list:
    """detectors in canonical behavioural order (busiest first); equal keys = identical ON times."""
    return sorted((int(d) for d in dets), key=lambda d: content_key(on.get(d, ()), t0), reverse=True)


def match_frac(ta: np.ndarray, tb: np.ndarray, secs: float) -> tuple[float, float]:
    """(chance-corrected share of a's ONs with a b ON within TOL of the pair's best lag, best lag)."""
    if len(ta) == 0 or len(tb) == 0:
        return 0.0, 0.0
    lo = np.searchsorted(tb, ta - LAGMAX, "left")
    hi = np.searchsorted(tb, ta + LAGMAX, "right")
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        return 0.0, 0.0
    ia = np.repeat(np.arange(len(ta)), cnt)
    jb = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt) + np.arange(tot)
    d = tb[jb] - ta[ia]
    h, e = np.histogram(d, bins=np.arange(-LAGMAX, LAGMAX + BIN, BIN))
    hs = np.convolve(h, np.ones(4), "same")
    lag = float(e[np.argmax(hs)] + BIN / 2)
    ok = np.abs(d - lag) <= TOL
    hit = np.zeros(len(ta), bool)
    hit[ia[ok]] = True
    raw = float(hit.mean())
    chance = 1.0 - np.exp(-len(tb) / secs * 2 * TOL)
    return max(0.0, (raw - chance) / max(1e-9, 1.0 - chance)), lag


def union_frac(ta, peers, lags, secs) -> float:
    hit = np.zeros(len(ta), bool)
    rate = 0.0
    for tb, lag in zip(peers, lags):
        lo = np.searchsorted(tb, ta + lag - TOL, "left")
        hi = np.searchsorted(tb, ta + lag + TOL, "right")
        hit |= hi > lo
        rate += len(tb) / secs
    chance = 1.0 - np.exp(-rate * 2 * TOL)
    return max(0.0, (float(hit.mean()) - chance) / max(1e-9, 1.0 - chance))


def pair_mats(on, dets, secs):
    n = len(dets)
    M, L = np.zeros((n, n)), np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                M[a, b], L[a, b] = match_frac(on[dets[a]], on[dets[b]], secs)
    return M, L


def span_feats(on: dict, dets: list, secs: float, M=None, L=None):
    n = len(dets)
    if M is None:
        M, L = pair_mats(on, dets, secs)
    out, coloc = {}, {}
    for i in range(n):
        coloc[dets[i]] = ",".join(str(dets[j]) for j in range(n) if j != i and max(M[i, j], M[j, i]) >= COVER)
    for i in range(n):
        cov = [j for j in range(n) if j != i and M[j, i] >= COVER and len(on[dets[i]]) > len(on[dets[j]])]
        if len(cov) < 2:
            out[dets[i]] = (False, 0.0, "")
            continue
        cov = sorted(cov, key=lambda j: -M[j, i])
        keep = []
        for j in cov:
            if all(M[j, k] < COLOC and M[k, j] < COLOC for k in keep):
                keep.append(j)
        if len(keep) < 2:
            out[dets[i]] = (False, 0.0, "")
            continue
        u = union_frac(on[dets[i]], [on[dets[j]] for j in keep], [L[i, j] for j in keep], secs)
        best = max(M[i, j] for j in keep)
        g = u - best
        out[dets[i]] = (bool(g >= SPAN_GAIN), float(g), ",".join(str(dets[j]) for j in keep))
    return out, coloc


def track_feats(on: dict, dets: list, cls: dict, lanes: dict, t0: float, secs: float) -> dict:
    nb = int(min(288, max(6, round(secs / 300.0))))
    edges = t0 + np.linspace(0, secs, nb + 1)
    cnt = {d: np.histogram(on[d], bins=edges)[0].astype(float) for d in dets}
    counts = [d for d in dets if cls.get(d) == "Count"]
    out = {}
    for d in dets:
        ref = [c for c in counts if c != d]
        if not ref:
            out[d] = np.nan
            continue
        ld = lanes.get(d, frozenset())
        same = [c for c in ref if ld and lanes.get(c) and (lanes[c] & ld)]
        tot = sum(cnt[c] for c in (same or ref))
        x = cnt[d]
        out[d] = float(np.corrcoef(x, tot)[0, 1]) if x.std() > 0 and tot.std() > 0 else np.nan
    return out


# ------------------------------------------------------------------------------------------- stacks (ln7 structure)
def _clusters(n, M, cand):
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


def stack_sizes(on: dict, dets: list, M) -> dict:
    """{det: size of its hi-res stack} for the members of a stack (co-located or spanning-cover cluster of >= 2) of ONE
    predicted phase; `dets` in canonical order with >= MIN_ON ONs and M = pair_mats(on, dets, secs) of exactly that list
    (the matrix span_feats uses).  Structure only (note 114): the stack-relative health statistics are gone."""
    n = len(dets)
    if n < 2:
        return {}
    cand = {}
    for i in range(n):
        cov = [j for j in range(n) if j != i and M[j, i] >= COVER and len(on[dets[i]]) > len(on[dets[j]])]
        cov = sorted(cov, key=lambda j: -M[j, i])
        keep = []
        for j in cov:
            if all(M[j, k] < COLOC and M[k, j] < COLOC for k in keep):
                keep.append(j)
        if len(keep) >= 2:
            cand[i] = keep
    return {dets[i]: len(m) for m in _clusters(n, M, cand) for i in m}


# ------------------------------------------------------------------------------------------- twins (short samples)
def twin_pairs(on: dict, groups: dict, secs: float, t0: float) -> list:
    """[(a, b, m_ab, m_ba, lag)] for every pair on the same predicted phase, both with >= 3 ONs (s62 pair_work); pair
    orientation (the sign of lag) from the canonical order, not the channel order."""
    out = []
    for _, dets in groups.items():
        big = order_dets(on, [int(d) for d in dets if len(on.get(int(d), ())) >= TWIN_MIN_N], t0)
        for i, a in enumerate(big):
            for b in big[i + 1:]:
                mab, lag = match_frac(on[a], on[b], secs)
                mba, _ = match_frac(on[b], on[a], secs)
                out.append((a, b, mab, mba, lag))
    return out
