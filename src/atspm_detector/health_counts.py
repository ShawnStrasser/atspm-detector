"""Count-based health measures of 1.1.0 (notes 147d-147m, 150), vectorised over the detectors of a signal (2-D numpy on
the 5-min bins of health_core.events_to_bins; no per-detector loops).  numpy + pandas only.

  hourly_geh   hourly GEH against a yardstick: count drops (drop point, GEH after it, occupancy before / after, own
               and yardstick) and misses at night (GEH of the average night hour 21-05).
               GEH = sqrt(2 (M - C)^2 / (M + C)), M = observed hourly count, C = share x the yardstick's hourly count;
               signed + when M < C (under-counting).  Yardstick = the same-phase detector it tracks (15-min counts,
               corr >= .7), else the other live detectors of its phase (>= 2, twins left out), else the rest of the
               signal.  Share = median M / R over the covered day hours 07-19 with R >= 30 (>= 4 of them), else
               total M / total R over those hours.  Hours = clock hours fully covered by the log.
  spiky        10-min bursts: own level = median of the covered bins within +-1 h leaving out the bin and its two
               neighbours; expected = max(level, 1) x the phase mates' own rise (their count / their level when they
               count >= 10, never below 1); a spike bin has count >= 3 x expected AND GEH(count, expected) >= 5;
               consecutive spike bins = one burst.
  common_holds every continuous ON >= 5 min held at the same time as other detectors' holds: common = all of them
               start AND end within 5 s of it (one shared failure, never a queue).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import health_core as hc

HOUR_NS = 3_600_000_000_000
DAY_H, NIGHT_H = (7, 19), (21, 5)
PARTNER_CORR = 0.7


def nanmedian_last(A):
    """np.nanmedian over the last axis (NaN where a row has no finite value), by one sort: faster than np.nanmedian on
    many short rows, same values."""
    A = np.sort(np.asarray(A, float), axis=-1)                # NaN sorts last
    cnt = np.isfinite(A).sum(-1)
    lo = np.take_along_axis(A, np.maximum((cnt - 1) // 2, 0)[..., None], -1)[..., 0]
    hi = np.take_along_axis(A, np.maximum(cnt // 2, 0)[..., None], -1)[..., 0]
    return np.where(cnt > 0, (lo + hi) / 2, np.nan)


def geh(m, c):
    """unsigned GEH of observed m vs expected c (elementwise)."""
    m, c = np.asarray(m, float), np.asarray(c, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(2 * (m - c) ** 2 / np.maximum(m + c, 1e-9))


def sgeh(m, c):
    """signed GEH: + when m < c (under-counting)."""
    g = geh(m, c)
    return np.where(np.asarray(m, float) < np.asarray(c, float), g, -g)


# ============================================================================ yardsticks
def _phase_vec(dets, phase):
    return np.array([phase.get(int(d), np.nan) for d in dets], float)


def yardsticks(B, phase):
    """per detector of B: the yardstick weight matrix W (R = W @ counts), the occupancy mates matrix (partner or
    phase reference; empty row = none) and the kind ('partner' / 'phase' / 'signal').  Cached in B."""
    key = ("_yard", tuple(sorted((int(k), float(v)) for k, v in phase.items())))
    if key in B:
        return B[key]
    dets = B["dets"]
    nd = len(dets)
    n, cov = B["n_on"].astype(float), B["cov"]
    live = n[:, cov].sum(1) > 0
    tw = hc.twins_of(B)
    ph = _phase_vec(dets, phase)
    eye = np.eye(nd, dtype=bool)
    same = np.isfinite(ph)[:, None] & (ph[:, None] == ph[None, :]) & ~eye
    # phase reference (health_core._refs): other live same-phase detectors, no twin, >= 2 of them
    ref = same & live[None, :] & ~tw
    is_ph = ref.sum(1) >= 2
    # partner (health_core._partner): best 15-min correlation among the live same-phase detectors, >= .7
    cand = same & live[None, :]
    n15, ok = hc._counts15(B)
    part = np.full(nd, -1)
    if ok.sum() >= 16 and cand.any():
        X = n15[:, ok].astype(float)
        sd = X.std(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.corrcoef(X) if nd > 1 else np.ones((1, 1))
        good = cand & (sd > 0)[None, :] & (sd > 0)[:, None] & np.isfinite(C)
        k1 = np.where(good, np.round(C, 12), -np.inf)
        m = good & (k1 == k1.max(1, initial=-np.inf)[:, None])
        ys, yw = X.sum(1), X @ np.arange(1, X.shape[1] + 1, dtype=float)
        k2 = np.where(m, ys[None, :], -np.inf)
        m &= k2 == k2.max(1, initial=-np.inf)[:, None]
        k3 = np.where(m, yw[None, :], -np.inf)
        m &= k3 == k3.max(1, initial=-np.inf)[:, None]
        has = m.any(1)
        best = np.argmax(m, 1)
        cbest = C[np.arange(nd), best]
        part = np.where(has & (cbest >= PARTNER_CORR), best, -1)
    W = np.where(is_ph[:, None], ref, ~eye).astype(float)
    Wm = np.where(is_ph[:, None], ref, False)
    hp = part >= 0
    W[hp] = 0.0
    Wm[hp] = False
    W[np.flatnonzero(hp), part[hp]] = 1.0
    Wm[np.flatnonzero(hp), part[hp]] = True
    kind = np.where(hp, "partner", np.where(is_ph, "phase", "signal"))
    out = dict(W=W, Wm=Wm.astype(float), has_m=Wm.any(1), kind=kind, live=live, tw=tw, ph=ph)
    B[key] = out
    return out


# ============================================================================ hourly GEH (count drops, night)
def _hours(B):
    """5-min bins -> clock hours: hour index of each bin, the fully covered hours, their hour of day and start."""
    nb = len(B["cov"])
    s0 = int(pd.Timestamp(B["start"]).value)
    t = s0 + np.arange(nb, dtype=np.int64) * int(B["bin_s"]) * 1_000_000_000
    hid = t // HOUR_NS
    u, inv, cnt = np.unique(hid, return_inverse=True, return_counts=True)
    cov_h = np.bincount(inv, weights=B["cov"].astype(float), minlength=len(u))
    per_h = HOUR_NS // (int(B["bin_s"]) * 1_000_000_000)
    ch = (cnt == per_h) & (cov_h == per_h)
    return inv, ch, (u % 24).astype(int), pd.to_datetime(u * HOUR_NS)


def _share(M, R, mask):
    """share of the detector in its yardstick over the hours `mask` (2-D): median M / R over the hours with R >= 30
    when >= 4 of them, else total M / total R."""
    ok = mask & (R >= 30)
    with np.errstate(divide="ignore", invalid="ignore"):
        med = nanmedian_last(np.where(ok, M / R, np.nan))
    tot = (M * mask).sum(1) / np.maximum((R * mask).sum(1), 1)
    return np.where(ok.sum(1) >= 4, med, tot)


def hourly_geh(B, phase) -> pd.DataFrame:
    """per detector of B: geh_ok (scored), geh_night, night_n, night_exp, geh_drop, drop_t (start of the first hour
    after the drop), own_before / own_after / exp_after (per hour), occ_own / occ_mates (mean hourly occupancy after
    / before the drop point)."""
    dets = B["dets"]
    nd = len(dets)
    cols = ["detector", "geh_ok", "geh_night", "geh_night_n", "geh_night_exp", "geh_drop", "drop_t", "own_before", "own_after",
            "exp_after", "occ_own", "occ_mates"]
    if not nd:
        return pd.DataFrame(columns=cols)
    inv, ch, hod, h0 = _hours(B)
    H = len(ch)
    n = B["n_on"].astype(float)
    occ = B["occ"].astype(float)
    Mh = np.zeros((nd, H))
    Oh = np.zeros((nd, H))
    np.add.at(Mh.T, inv, n.T)
    np.add.at(Oh.T, inv, occ.T)
    Y = yardsticks(B, phase)
    R = Y["W"] @ Mh
    Om = Y["Wm"] @ Oh
    day = ch & (hod >= DAY_H[0]) & (hod < DAY_H[1])
    nt = ch & ((hod >= NIGHT_H[0]) | (hod < NIGHT_H[1]))
    D2 = np.broadcast_to(day, (nd, H))
    elig = ((Mh * day).sum(1) >= 20) & ((R * day).sum(1) > 0)
    s = _share(Mh, R, D2)
    C = s[:, None] * R
    # misses at night: the average night hour
    nn = int(nt.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        mn, cn = (Mh * nt).sum(1) / max(nn, 1), (C * nt).sum(1) / max(nn, 1)
    g_night = np.where(elig & (nn >= 4), sgeh(mn, cn), np.nan)
    # count drops: the hour with the largest step up in under-counting (two-sample, scaled), >= 3 covered hours each side
    sg = np.where(ch[None, :], sgeh(Mh, C), 0.0)
    cs, cc = np.cumsum(sg, 1), np.cumsum(ch)
    na = cc[:-1].astype(float)                       # covered hours before k = 1 .. H-1
    nbk = float(cc[-1]) - na
    sa, sb = cs[:, :-1], cs[:, -1:] - cs[:, :-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = (sb / nbk - sa / na) * np.sqrt(na * nbk / (na + nbk))
    valid = (na >= 3) & (nbk >= 3)
    stat = np.where(valid[None, :], stat, -np.inf)
    has_k = elig & valid.any()
    k = np.argmax(stat, 1) + 1                      # first hour after the drop point
    idx = np.arange(H)
    pre = ch[None, :] & (idx[None, :] < k[:, None])
    post = ch[None, :] & (idx[None, :] >= k[:, None])
    s2 = _share(Mh, R, pre)
    re = (pre & (R >= 30)).sum(1) >= 3
    sg2 = np.where(re[:, None], np.where(ch[None, :], sgeh(Mh, s2[:, None] * R), 0.0), sg)
    med = nanmedian_last(np.where(post, sg2, np.nan))
    g_drop = np.where(has_k & (med > 0), med, np.nan)
    sh = np.where(re, s2, s)
    npre, npost = np.maximum(pre.sum(1), 1), np.maximum(post.sum(1), 1)
    occ_own = (Oh * post).sum(1) / npost / np.maximum((Oh * pre).sum(1) / npre, 1e-6)
    occ_m = (Om * post).sum(1) / npost / np.maximum((Om * pre).sum(1) / npre, 1e-6)
    return pd.DataFrame({
        "detector": dets.astype(int), "geh_ok": elig, "geh_night": g_night,
        "geh_night_n": np.where(np.isfinite(g_night), (Mh * nt).sum(1), np.nan),
        "geh_night_exp": np.where(np.isfinite(g_night), (C * nt).sum(1), np.nan),
        "geh_drop": g_drop, "drop_t": np.where(has_k, h0.to_numpy()[np.minimum(k, H - 1)], np.datetime64("NaT", "ns")),
        "own_before": (Mh * pre).sum(1) / npre, "own_after": (Mh * post).sum(1) / npost,
        "exp_after": sh * (R * post).sum(1) / npost,
        "occ_own": np.where(has_k, occ_own, np.nan), "occ_mates": np.where(has_k & Y["has_m"], occ_m, np.nan)})


# ============================================================================ spiky counts (10-min bursts)
def _level(x, c, w):
    """median of the covered bins within +-w bins of t leaving out t-1, t, t+1 (>= 3 of them), else NaN."""
    nd, T = x.shape
    S = np.full((nd, T, 2 * (w - 1)), np.nan)
    j = 0
    for d in range(-w, w + 1):
        if abs(d) <= 1:
            continue
        if d < 0:
            S[:, -d:, j] = np.where(c[:, :d], x[:, :d], np.nan)
        else:
            S[:, :-d, j] = np.where(c[:, d:], x[:, d:], np.nan)
        j += 1
    b = nanmedian_last(S)
    return np.where(np.isfinite(S).sum(2) >= 3, b, np.nan)


def spiky(B, phase, hours, P) -> tuple[pd.DataFrame, dict]:
    """per detector of B: spk_bursts (bursts per day), spk_gmax (largest GEH), spk_daily (vehicles per day), spk_lvl
    (0 / 1 suspect / 2 bad by P = limits['spiky']); and {detector: [(first bin, end bin), ...]} of the bursts (bins of
    P['bin_min'] minutes from the window start)."""
    dets = B["dets"]
    nd = len(dets)
    k = int(P["bin_min"] * 60 // int(B["bin_s"]))
    n, cov = B["n_on"].astype(float), B["cov"]
    T = n.shape[1] // k
    cols = ["detector", "spk_bursts", "spk_gmax", "spk_daily", "spk_lvl"]
    if not nd or T < 2 * P["window_bins"]:
        return pd.DataFrame(columns=cols), {}
    own = n[:, :T * k].reshape(nd, T, k).sum(2)
    c1 = cov[:T * k].reshape(T, k).all(1)
    c = np.broadcast_to(c1, own.shape)
    Y = yardsticks(B, phase)
    same = np.isfinite(Y["ph"])[:, None] & (Y["ph"][:, None] == Y["ph"][None, :]) & ~np.eye(nd, dtype=bool)
    Wt = same & Y["live"][None, :] & ~Y["tw"]
    m = np.where(Wt.any(1)[:, None], Wt.astype(float) @ own, 0.0)
    w = int(P["window_bins"])
    b, bm = _level(own, c, w), _level(m, c, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        rise = np.where((m >= P["mates_min"]) & np.isfinite(bm), m / np.maximum(bm, 1.0), 1.0)
    ex = np.maximum(b, 1.0) * np.maximum(rise, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = own / ex
    g = geh(own, ex)
    sp = np.isfinite(b) & c & (ratio >= P["ratio"]) & (g >= P["geh"]) & (own > ex)
    st = sp & ~np.c_[np.zeros((nd, 1), bool), sp[:, :-1]]
    en = sp & ~np.c_[sp[:, 1:], np.zeros((nd, 1), bool)]
    days = max(float(hours) / 24.0, 1.0)
    nb = st.sum(1) / days
    gmax = np.where(sp, g, 0.0).max(1)
    daily = own.sum(1) * 24.0 / max(float(hours), 1e-9)
    lvl = np.where(nb >= P["bad"], 2, np.where(nb >= P["suspect"], 1, 0))
    lvl = np.where((daily < P["daily_low"]) & (lvl == 2) & (gmax < P["geh_big"]), 1, lvl)
    lvl = np.where(daily < P["daily_none"], 0, lvl)
    r0, t0 = np.nonzero(st)
    r1, t1 = np.nonzero(en)
    runs = {}
    for r, a, z in zip(r0.tolist(), t0.tolist(), t1.tolist()):
        runs.setdefault(int(dets[r]), []).append((a, z + 1))
    return pd.DataFrame({"detector": dets.astype(int), "spk_bursts": nb, "spk_gmax": gmax, "spk_daily": daily,
                         "spk_lvl": lvl}), runs


# ============================================================================ holds shared with other detectors
def twin_matrix(B):
    """detector-channel x detector-channel bool matrix of twins (one zone wired to two inputs: health_core.twins on the
    whole sample's 5-min counts), indexed by channel number."""
    dets = np.asarray(B["dets"], int)
    m = int(dets.max()) + 1 if len(dets) else 1
    T = np.zeros((m, m), bool)
    if len(dets):
        T[np.ix_(dets, dets)] = hc.twins_of(B)
    return T


def _not_twin(T, a, b):
    if T is None:
        return True
    n = len(T)
    ok = (a < n) & (b < n)
    return ~np.where(ok, T[np.minimum(a, n - 1), np.minimum(b, n - 1)], False)


def common_holds(s, f, c, tol=5.0, block=2048, twin=None):
    """per hold (start s, end f in seconds, detector c): overlaps another detector's hold, and is a COMMON hold (every
    overlapping hold starts AND ends within `tol` s of it).  A twin (twin_matrix: the same zone on another input) never
    counts as another detector here.  Blocked broadcast (memory <= block x n)."""
    s, f, c = (np.asarray(v) for v in (s, f, c))
    n = len(s)
    anyov, common = np.zeros(n, bool), np.zeros(n, bool)
    for a in range(0, n, block):
        z = min(a + block, n)
        ov = ((np.minimum(f[a:z, None], f[None, :]) > np.maximum(s[a:z, None], s[None, :])) & (c[a:z, None] != c[None, :])
              & _not_twin(twin, c[a:z, None], c[None, :]))
        near = (np.abs(s[None, :] - s[a:z, None]) <= tol) & (np.abs(f[None, :] - f[a:z, None]) <= tol)
        anyov[a:z] = ov.any(1)
        common[a:z] = anyov[a:z] & ~(ov & ~near).any(1)
    return anyov, common


def common_mates(s, f, c, i, twin=None):
    """the other (non-twin) detectors whose holds overlap hold i (sorted)."""
    ov = (np.minimum(f[i], f) > np.maximum(s[i], s)) & (c != c[i]) & _not_twin(twin, np.full(len(c), c[i]), c)
    return sorted(set(int(x) for x in c[ov]))
