"""Note 116 exploration methods (hourly / block features, Mahalanobis, kNN, k-means, conditional night level). The production
subset lives in tod116.py.

Idea (user, Oct 7): every detector type has a normal 24-h shape. Group all detectors by TYPE (the classifier's function x
lane span x volume band; thin groups merged), learn the group's normal day from BOTH actuations and time ON, and score
how far a detector-day sits from its group. Unsupervised: no health labels are used anywhere.

Features of one detector-day (hourly counts N[24], hourly mean fraction ON O[24]):
    s_h  = sqrt(N_h / sum N)            shape of the day's actuations (Hellinger: equalises Poisson noise)
    a_h  = arcsin(sqrt(O_h))            time ON per hour, LEVEL kept (variance-stabilised fraction)
    ds_h, da_h = first differences      (hour-to-hour change; only shifts weight in the distance)
    logv = log10(1 + sum N)             volume level inside the band
Distance = robust Mahalanobis in the group's standardised space: centre = group median, scale = robust SD per feature,
covariance from the trimmed group (worst 10 % dropped twice) shrunk 30 % towards identity. Score = sqrt(d2 / p),
then calibrated per group: z = (log score - median) / MAD-scale of the group's out-of-fold scores, flag at z >= T.

Partial samples (3-12 h): the same on the available hours only; the group statistics are rebuilt on the fly from the
stored reference profiles restricted to those hours (shares renormalised over the window), calibration by split-half.

Reference file (npz): groups, per group mu/scale/centre/precision/cal + up to REF_CAP reference profiles (s, o, logv).
"""
from __future__ import annotations

import json

import numpy as np

FNS = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")
BANDS = (20.0, 100.0)            # actuations per hour: low < 20 <= medium < 100 <= high
MIN_N = 50                       # actuations needed in the sample (24 h) to score
SHRINK = 0.30
TRIM = 0.10
ZCLIP = 8.0
REF_CAP = 1500
CC_FLOOR = 0.5                   # actuations per hour added to both ends of the counts contrast
CO_FLOOR = 0.01                  # fraction ON added to both ends of the time-ON contrast (pulse zones carry ~0)


def band_of(rate):
    rate = np.asarray(rate, float)
    return np.where(rate < BANDS[0], "low", np.where(rate < BANDS[1], "medium", "high"))


def span_of(lanes):
    lanes = np.asarray(lanes, float)
    return np.where(lanes >= 2, "2+", "1")


# ---------------------------------------------------------------------------------------------------------- features
def features(N, O, hours=None, diffs=True):
    """N, O: (m, 24) arrays (NaN allowed outside `hours`). hours: bool mask (24,) or None = all. Returns (m, p)."""
    N = np.asarray(N, float)
    O = np.asarray(O, float)
    if hours is None:
        hours = np.ones(N.shape[1], bool)
    n = np.nan_to_num(N[:, hours])
    o = np.clip(np.nan_to_num(O[:, hours]), 0, 1)
    tot = n.sum(1, keepdims=True)
    s = np.sqrt(n / np.clip(tot, 1, None))
    a = np.arcsin(np.sqrt(o))
    parts = [s, a]
    if diffs and s.shape[1] > 1:
        contiguous = np.diff(np.flatnonzero(hours)) == 1
        parts += [np.diff(s, axis=1)[:, contiguous], np.diff(a, axis=1)[:, contiguous]]
    hrs = hours.sum()
    parts.append(np.log10(1 + tot * len(hours) / hrs))    # daily-equivalent volume
    return np.hstack(parts)


def feature_names(hours=None, diffs=True):
    h = np.flatnonzero(np.ones(24, bool) if hours is None else hours)
    names = [f"s{i}" for i in h] + [f"a{i}" for i in h]
    if diffs and len(h) > 1:
        c = np.diff(h) == 1
        names += [f"ds{i}" for i in h[1:][c]] + [f"da{i}" for i in h[1:][c]]
    return names + ["logv"]


# ---------------------------------------------------------------------------------------------------------- block features
BLOCKS = ((0, 5), (5, 7), (7, 10), (10, 15), (15, 19), (19, 22), (22, 24))     # clock hours [a, b)
BLOCK_NAMES = ("night 0-5", "early 5-7", "AM 7-10", "midday 10-15", "PM 15-19", "evening 19-22", "late 22-24")


def features_blk(N, O, H=None, Q=None, hours=None, rough=True):
    """compact day shape: per clock block the log hourly rate relative to the sample mean (counts) and the log
    time-ON ratio vs the sample mean; overall time-ON level; 30-min roughness of counts / time ON; volume level.
    N, O (m, 24) hourly; H, Q (m, 48) 30-min (for roughness; optional). hours: bool (24,) available clock hours."""
    N = np.nan_to_num(np.asarray(N, float))
    O = np.clip(np.nan_to_num(np.asarray(O, float)), 0, 1)
    if hours is None:
        hours = np.ones(24, bool)
    hrs = hours.sum()
    tot = N[:, hours].sum(1)
    rate = (tot + 1.0) / hrs
    oavg = O[:, hours].mean(1)
    parts, names = [], []
    for (a, b), nm in zip(BLOCKS, BLOCK_NAMES):
        hb = np.zeros(24, bool)
        hb[a:b] = True
        hb &= hours
        if hb.sum() == 0:
            continue
        r = (N[:, hb].sum(1) + 0.5 * hb.sum() / 24 * 24 / hrs) / hb.sum()
        parts.append(np.log(r / rate))
        names.append("c " + nm)
        parts.append(np.log((O[:, hb].mean(1) + 0.002) / (oavg + 0.002)))
        names.append("o " + nm)
    parts.append(np.arcsin(np.sqrt(oavg)))
    names.append("o level")
    if rough and H is not None:
        hh = np.repeat(hours, 2)
        h = np.sqrt(np.clip(np.nan_to_num(np.asarray(H, float))[:, hh], 0, None))
        q = np.sqrt(np.clip(np.nan_to_num(np.asarray(Q, float))[:, hh], 0, 1))
        parts.append(np.abs(np.diff(h, axis=1)).mean(1) / np.clip(h.mean(1), 0.05, None))
        names.append("c rough")
        parts.append(np.abs(np.diff(q, axis=1)).mean(1) / np.clip(q.mean(1), 0.02, None))
        names.append("o rough")
    parts.append(np.log10(1 + tot * 24.0 / hrs))
    names.append("logv")
    return np.column_stack(parts), names


def _roll4(A, k=4):
    """circular 4-h rolling mean of (m, 24) hourly values."""
    A = np.nan_to_num(np.asarray(A, float))
    Ac = np.concatenate([A, A[:, :k - 1]], 1)
    cs = np.cumsum(np.pad(Ac, ((0, 0), (1, 0))), 1)
    return (cs[:, k:] - cs[:, :-k]) / k


NIGHT = (1, 5)                   # clock hours [1, 5): the quietest traffic of the day everywhere


def features_night(N, O):
    """night level vs the busiest 4 h, counts and time ON: log((mean 01-05 h + floor) / (busiest 4-h mean + floor));
    + conditioners o_day (arcsin sqrt of the busiest 4-h time ON) and logv. Columns: [log nc, log no, o_day, logv, night time ON (arcsin sqrt)]."""
    N = np.nan_to_num(np.asarray(N, float))
    O = np.clip(np.nan_to_num(np.asarray(O, float)), 0, 1)
    rn, ro = _roll4(N), _roll4(O)
    a, b = NIGHT
    nc = (N[:, a:b].mean(1) + CC_FLOOR) / (rn.max(1) + CC_FLOOR)
    no = (O[:, a:b].mean(1) + CO_FLOOR) / (ro.max(1) + CO_FLOOR)
    return np.column_stack([np.log(nc), np.log(no), np.arcsin(np.sqrt(ro.max(1))), np.log10(1 + N.sum(1)),
                            np.arcsin(np.sqrt(O[:, a:b].mean(1)))])


def features_flat(N, O):
    """day/night contrast (amplitude of the daily cycle) for counts and time ON + the conditioners.
    log cc = log(quietest 4 h / busiest 4 h) of counts, log co the same for time ON; o_day = arcsin sqrt of the busiest
    4-h time ON (a saturated long zone keeps its counts flat); logv. Columns: [log cc, log co, o_day, logv]."""
    rn, ro = _roll4(N), _roll4(np.clip(np.nan_to_num(O), 0, 1))
    tot = np.nansum(N, 1)
    cc = (rn.min(1) + CC_FLOOR) / (rn.max(1) + CC_FLOOR)
    co = (ro.min(1) + CO_FLOOR) / (ro.max(1) + CO_FLOOR)
    return np.column_stack([np.log(cc), np.log(co), np.arcsin(np.sqrt(ro.max(1))), np.log10(1 + tot)])


# ---------------------------------------------------------------------------------------------------------- fitting
def _robust_scale(X):
    med = np.median(X, 0)
    mad = 1.4826 * np.median(np.abs(X - med), 0)
    sd = X.std(0)
    return med, np.maximum(np.maximum(mad, 0.25 * sd), 1e-3)


def fit_group(X, shrink=SHRINK, trim=TRIM, iters=2):
    """Robust centre/scale + shrunk precision on the trimmed group. Returns dict and the kept-row mask."""
    keep = np.ones(len(X), bool)
    for _ in range(iters + 1):
        med, sc = _robust_scale(X[keep])
        Z = np.clip((X - med) / sc, -ZCLIP, ZCLIP)
        mu = Z[keep].mean(0)
        C = np.cov(Z[keep], rowvar=False)
        S = (1 - shrink) * C + shrink * np.eye(C.shape[0]) * np.mean(np.diag(C))
        P = np.linalg.inv(S)
        D = Z - mu
        d2 = np.einsum("ij,jk,ik->i", D, P, D)
        if _ < iters:
            keep = d2 <= np.quantile(d2, 1 - trim)
    return dict(med=med, scale=sc, mu=mu, prec=P), keep


def fit_cond(X, cond, trim=TRIM, iters=2, min_scale=1e-3):
    """per shape feature j (not in `cond`): x_j = b . [1, X[:, cond]] fitted by trimmed least squares; robust residual
    scale (MAD). Returns dict(cond, shape, B (len(cond)+1, n_shape), scale (n_shape,))."""
    cond = np.asarray(cond, int)
    shape = np.setdiff1d(np.arange(X.shape[1]), cond)
    A = np.column_stack([np.ones(len(X)), X[:, cond]])
    B = np.zeros((A.shape[1], len(shape)))
    sc = np.zeros(len(shape))
    for j, c in enumerate(shape):
        y = X[:, c]
        keep = np.ones(len(X), bool)
        for it in range(iters + 1):
            b = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
            r = y - A @ b
            if it < iters:
                keep = np.abs(r) <= np.quantile(np.abs(r), 1 - trim)
        rk = r[keep]
        s = 1.4826 * np.median(np.abs(rk - np.median(rk)))
        B[:, j], sc[j] = b, max(s, 0.25 * rk.std(), min_scale)
    return dict(cond=cond, shape=shape, B=B, rscale=sc)


def cond_z(X, m):
    """signed residual z per shape feature (n, n_shape)."""
    A = np.column_stack([np.ones(len(X)), X[:, m["cond"]]])
    return np.clip((X[:, m["shape"]] - A @ m["B"]) / m["rscale"], -ZCLIP * 2, ZCLIP * 2)


def zmax_score(Zr, top=1):
    a = np.sort(np.abs(Zr), 1)[:, ::-1]
    return a[:, :top].mean(1)


def maha(X, g):
    Z = np.clip((X - g["med"]) / g["scale"], -ZCLIP, ZCLIP)
    D = Z - g["mu"]
    return np.sqrt(np.einsum("ij,jk,ik->i", D, g["prec"], D) / X.shape[1])


def zrms(X, g):
    Z = np.clip((X - g["med"]) / g["scale"], -ZCLIP, ZCLIP)
    return np.sqrt(np.mean(Z ** 2, 1))


def knn(X, R, g, k=10, exclude_self=False):
    """mean Euclidean distance (standardised, clipped) to the k nearest reference rows R (already standardised)."""
    Z = np.clip((X - g["med"]) / g["scale"], -ZCLIP, ZCLIP)
    out = np.empty(len(Z))
    for i0 in range(0, len(Z), 512):                          # chunked, vectorised
        z = Z[i0:i0 + 512]
        d2 = (z ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None, :] - 2 * z @ R.T
        d2 = np.maximum(d2, 0)
        kk = k + 1 if exclude_self else k
        part = np.partition(d2, kk - 1, axis=1)[:, :kk]
        part.sort(1)
        if exclude_self:
            part = part[:, 1:]
        out[i0:i0 + 512] = np.sqrt(part).mean(1) / np.sqrt(X.shape[1])
    return out


def kmeans(Z, k=4, iters=30, seed=0):
    """plain k-means++ (numpy). Returns centroids, labels."""
    rng = np.random.default_rng(seed)
    C = [Z[rng.integers(len(Z))]]
    for _ in range(1, k):
        d2 = np.min(((Z[:, None, :] - np.array(C)[None]) ** 2).sum(2), 1)
        C.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(2), 1)
        newC = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, lab


# ---------------------------------------------------------------------------------------------------------- groups
def group_key(fn, span, band):
    return f"{fn}|{span}|{band}"


def resolve_group(fn, span, band, available):
    """finest available group: fn|span|band -> fn|*|band -> fn|span|* -> fn|*|*."""
    for k in (group_key(fn, span, band), group_key(fn, "*", band), group_key(fn, span, "*"), group_key(fn, "*", "*")):
        if k in available:
            return k
    return None


# ---------------------------------------------------------------------------------------------------------- reference I/O
def save_reference(path, groups, meta):
    """groups: {key: dict(med, scale, mu, prec, cal_med, cal_mad, ref_n, ref_o, ref_tot)}"""
    arr = {}
    keys = sorted(groups)
    for i, k in enumerate(keys):
        for f, v in groups[k].items():
            arr[f"g{i}_{f}"] = np.asarray(v, np.float32 if f.startswith("ref_") else np.float64)
    meta = dict(meta, groups=keys)
    np.savez_compressed(path, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arr)


def load_reference(path):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    groups = {}
    for i, k in enumerate(meta["groups"]):
        pre = f"g{i}_"
        groups[k] = {f[len(pre):]: z[f] for f in z.files if f.startswith(pre)}
    return groups, meta


# ---------------------------------------------------------------------------------------------------------- scoring
def score_day(N, O, fn, lanes, ref, threshold=None):
    """Score full-day samples. N, O (m, 24); fn (m,), lanes (m,). Returns dict of arrays: group, score, z, flag, and
    per-hour standardised deviations zs (counts share) / za (time ON) for explanation and charts (m, 24)."""
    groups, meta = ref
    T = meta["threshold"] if threshold is None else threshold
    N = np.asarray(N, float)
    O = np.asarray(O, float)
    m = len(N)
    tot = np.nansum(N, 1)
    band = band_of(tot / 24.0)
    span = span_of(lanes)
    X = features(N, O)
    out = dict(group=np.array([None] * m, object), score=np.full(m, np.nan), z=np.full(m, np.nan),
               flag=np.zeros(m, bool), zs=np.full((m, 24), np.nan), za=np.full((m, 24), np.nan))
    ok = (tot >= MIN_N) & np.isfinite(N).all(1) & np.isfinite(O).all(1)
    gk = np.array([resolve_group(f, s, b, groups) if o else None for f, s, b, o in zip(fn, span, band, ok)], object)
    for k in set(gk[gk != None]):                                        # noqa: E711
        i = np.flatnonzero(gk == k)
        g = groups[k]
        sc = maha(X[i], g)
        z = (np.log(sc) - g["cal_med"]) / g["cal_mad"]
        Z = (X[i] - g["med"]) / g["scale"]
        out["group"][i] = k
        out["score"][i], out["z"][i], out["flag"][i] = sc, z, z >= T
        out["zs"][i], out["za"][i] = Z[:, :24], Z[:, 24:48]
    return out


def score_partial(N, O, hours, fn, lanes, ref, threshold=None, min_n=MIN_N):
    """Score samples covering only `hours` (bool (24,), same for all rows). Group statistics are rebuilt from the
    stored reference profiles restricted to those hours; calibration by split-half of the references."""
    groups, meta = ref
    T = meta.get("threshold_partial", meta["threshold"]) if threshold is None else threshold
    hours = np.asarray(hours, bool)
    N = np.asarray(N, float)
    O = np.asarray(O, float)
    m = len(N)
    tot = np.nansum(N[:, hours], 1)
    hrs = hours.sum()
    # volume band from the window rate scaled by the typical share of these clock hours (all types)
    share = np.asarray(meta["all_share"])[hours].sum()
    band = band_of(tot / max(share, 1e-6) / 24.0)
    span = span_of(lanes)
    X = features(N, O, hours)
    out = dict(group=np.array([None] * m, object), score=np.full(m, np.nan), z=np.full(m, np.nan),
               flag=np.zeros(m, bool))
    ok = (tot >= min_n) & np.isfinite(N[:, hours]).all(1) & np.isfinite(O[:, hours]).all(1)
    gk = np.array([resolve_group(f, s, b, groups) if o else None for f, s, b, o in zip(fn, span, band, ok)], object)
    for k in set(gk[gk != None]):                                        # noqa: E711
        i = np.flatnonzero(gk == k)
        g = groups[k]
        rn = g["ref_n"].astype(float)
        ro = g["ref_o"].astype(float)
        R = features(rn, ro, hours)
        rok = rn[:, hours].sum(1) >= min_n
        R = R[rok]
        if len(R) < 60:
            continue
        rng = np.random.default_rng(0)
        half = rng.random(len(R)) < 0.5
        ga, _ = fit_group(R[half])                       # fit on one half, calibrate on the other, score with the same fit
        cal = np.log(maha(R[~half], ga))
        med = np.median(cal)
        mad = max(1.4826 * np.median(np.abs(cal - med)), 1e-3)
        sc = maha(X[i], ga)
        z = (np.log(sc) - med) / mad
        out["group"][i] = k
        out["score"][i], out["z"][i], out["flag"][i] = sc, z, z >= T
    return out
