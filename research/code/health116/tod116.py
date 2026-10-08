"""Note 116: TIME-OF-DAY health check - "the night does not go down" - numpy only (production-ready: no sklearn /
scipy / torch / pandas at scoring time).

What it asks: every vehicle detector sees its quietest traffic between 01:00 and 05:00. How high is its night level
compared with its own busiest 4 hours, against detectors of the same TYPE (classifier function x lane span x volume
band, thin groups merged)? Two layers:
  1. TYPE: counts night level  r = sqrt((mean count/h 01-05 + 0.5) / (busiest 4-h mean count/h + 0.5)), predicted
     from the zone's daytime saturation (busiest 4-h time ON) and its volume (a saturated long zone keeps its counts
     flat in the day, so its night ratio is naturally higher); for Presence / Other / Mid also the absolute night
     time ON. One-sided robust z (only 'too busy at night' counts), calibrated per group -> z.
  2. PHASE (context): dph = its r minus the median r of its phase mates (same signal, same predicted phase,
     >= 200 actuations a day). Phase mates measure the same movement, so their night traffic should match.
Finding: night actuations >= 20 AND (z >= Z_HI OR (z >= Z_LO AND dph >= D_PH)). The phase number is only a grouping
key (which detectors are mates); no channel or phase number is a feature.

Full day (24 h) is the main use. Partial samples (>= 10 h that include 01-05 and a daytime peak) rebuild the
group model on the fly from the stored reference profiles restricted to the same clock hours.
"""
from __future__ import annotations

import json

import numpy as np

FNS = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")
OCC_FNS = ("Presence", "Other", "Mid")        # types whose job is time ON: their night % ON is a second channel
BANDS = (20.0, 100.0)                         # actuations per hour: low < 20 <= medium < 100 <= high
NIGHT = (1, 5)                                # clock hours [1, 5)
CNT_FLOOR = 0.5                               # actuations / h added to both ends of the night ratio
MIN_N = 50                                    # actuations in the sample needed to score
MIN_NIGHT = 20                                # night (01-05) actuations needed for a finding
MIN_MATE = 200                                # daily actuations for a detector to serve as a phase mate
MIN_GROUP = 150
TRIM = 0.10
OCC_MIN_SCALE = 0.05                          # arcsin-sqrt units (~0.25 % ON near 0)
Z_LO, Z_HI, D_PH = 3.5, 6.0, 0.30


def band_of(rate):
    rate = np.asarray(rate, float)
    return np.where(rate < BANDS[0], "low", np.where(rate < BANDS[1], "medium", "high"))


def span_of(lanes):
    return np.where(np.asarray(lanes, float) >= 2, "2+", "1")


def group_key(fn, span, band):
    return f"{fn}|{span}|{band}"


def resolve_group(fn, span, band, available):
    for k in (group_key(fn, span, band), group_key(fn, "*", band), group_key(fn, span, "*"), group_key(fn, "*", "*")):
        if k in available:
            return k
    return None


# ---------------------------------------------------------------------------------------------------------- features
def _busiest4(A, hours):
    """busiest 4-h rolling mean inside the available hours (circular only for a full day)."""
    m, k = A.shape[0], 4
    if hours.all():
        Ac = np.concatenate([A, A[:, :k - 1]], 1)
        cs = np.cumsum(np.pad(Ac, ((0, 0), (1, 0))), 1)
        return ((cs[:, k:] - cs[:, :-k]) / k).max(1)
    best = np.full(m, -np.inf)
    for s in range(0, 24 - k + 1):
        if hours[s:s + k].all():
            best = np.maximum(best, A[:, s:s + k].mean(1))
    return best


def features(N, O, hours=None):
    """N, O: (m, 24) hourly counts and mean fraction ON (NaN allowed outside `hours`).
    Returns X (m, 4) = [r (sqrt night ratio), night time ON (arcsin sqrt), o_day (arcsin sqrt busiest 4-h ON), logv],
    and night actuations (m,)."""
    hours = np.ones(24, bool) if hours is None else np.asarray(hours, bool)
    N = np.where(hours, np.nan_to_num(np.asarray(N, float)), 0.0)
    O = np.where(hours, np.clip(np.nan_to_num(np.asarray(O, float)), 0, 1), 0.0)
    a, b = NIGHT
    nn = N[:, a:b].mean(1)
    r = np.sqrt((nn + CNT_FLOOR) / (_busiest4(N, hours) + CNT_FLOOR))
    on = np.arcsin(np.sqrt(O[:, a:b].mean(1)))
    od = np.arcsin(np.sqrt(np.clip(_busiest4(O, hours), 0, 1)))
    lv = np.log10(1 + N.sum(1) * 24.0 / hours.sum())
    return np.column_stack([r, on, od, lv]), N[:, a:b].sum(1)


# ---------------------------------------------------------------------------------------------------------- model
def _fit_cond(y, A, trim=TRIM, iters=2, min_scale=1e-3):
    """trimmed least squares y ~ A; robust residual scale."""
    keep = np.ones(len(y), bool)
    for it in range(iters + 1):
        b = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        r = y - A @ b
        if it < iters:
            keep = np.abs(r) <= np.quantile(np.abs(r), 1 - trim)
    rk = r[keep]
    s = 1.4826 * np.median(np.abs(rk - np.median(rk)))
    return b, max(s, 0.25 * rk.std(), min_scale)


def fit_group(X, fn):
    """X from features(); returns the group's model (counts channel always, time-ON channel for OCC_FNS)."""
    A = np.column_stack([np.ones(len(X)), X[:, 2], X[:, 3]])
    bc, sc = _fit_cond(X[:, 0], A)
    g = dict(bc=bc, sc=np.array(sc), occ=np.array(fn in OCC_FNS))
    if fn in OCC_FNS:
        bo, so = _fit_cond(X[:, 1], A[:, [0, 2]], min_scale=OCC_MIN_SCALE)
        g.update(bo=bo, so=np.array(so))
    return g


def raw_z(X, g):
    """one-sided residual z (max over channels), the counts channel's expected r, and each channel's z."""
    A = np.column_stack([np.ones(len(X)), X[:, 2], X[:, 3]])
    exp_r = A @ g["bc"]
    z = (X[:, 0] - exp_r) / g["sc"]
    zo = np.full(len(X), -np.inf)
    if bool(g["occ"]):
        exp_o = A[:, [0, 2]] @ g["bo"]
        zo = (X[:, 1] - exp_o) / g["so"]
    return np.fmax(z, zo), exp_r, z, zo


def phase_delta(r, sig, phase, tot, ok):
    """own r minus the median r of phase mates (same signal + predicted phase, ok, >= MIN_MATE a day), self out.
    sig / phase: 1-D keys (phase used ONLY to say who are mates). Returns dph (NaN without a mate), mates' r."""
    m = len(r)
    mate_r = np.full(m, np.nan)
    good = ok & (tot >= MIN_MATE)
    keys = np.array([f"{s}#{p}" if p is not None and p == p else f"{s}#none{i}"
                     for i, (s, p) in enumerate(zip(sig, phase))])         # no predicted phase -> no mates
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    cuts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = order[a:b]
        if len(idx) < 2:
            continue
        vals, gd = r[idx], good[idx]
        for j, i in enumerate(idx):
            mm = gd.copy()
            mm[j] = False
            if mm.any():
                mate_r[i] = np.median(vals[mm])
    return r - mate_r, mate_r


def decide(z, dph, n_night):
    """finding + level: 'bad' when far from the type AND the phase mates disagree, else 'suspect'."""
    d = np.nan_to_num(dph, nan=-1)
    hi = z >= Z_HI
    flag = (n_night >= MIN_NIGHT) & (hi | ((z >= Z_LO) & (d >= D_PH)))
    bad = flag & hi & (d >= D_PH)
    return flag, np.where(bad, "bad", np.where(flag, "suspect", "ok"))


# ---------------------------------------------------------------------------------------------------------- I/O
def save_reference(path, groups, meta):
    arr = {}
    keys = sorted(groups)
    for i, k in enumerate(keys):
        for f, v in groups[k].items():
            arr[f"g{i}_{f}"] = np.asarray(v, np.float32) if f.startswith("ref_") else np.asarray(v)
    meta = dict(meta, groups=keys)
    np.savez_compressed(path, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arr)


def load_reference(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta"]).decode())
    groups = {}
    for i, k in enumerate(meta["groups"]):
        pre = f"g{i}_"
        groups[k] = {f[len(pre):]: z[f] for f in z.files if f.startswith(pre)}
    return groups, meta


# ---------------------------------------------------------------------------------------------------------- scoring
def window_ok(hours):
    """a partial sample can be scored when it covers 01-05, >= 10 h and a whole daytime peak (07-10 or 15-19)."""
    a, b = NIGHT
    h = np.asarray(hours, bool)
    return bool(h[a:b].all() and h.sum() >= 10 and (h[7:10].all() or h[15:19].all()))


def score(N, O, fn, lanes, sig, phase, ref, hours=None):
    """Score detector samples of ONE clock window (`hours`, default full day). Returns dict of arrays:
    group, z, z_cnt, z_occ, r, exp_r, mate_r, dph, n_night, flag, level."""
    groups, meta = ref
    hours = np.ones(24, bool) if hours is None else np.asarray(hours, bool)
    full = bool(hours.all())
    N = np.asarray(N, float)
    O = np.asarray(O, float)
    m = len(N)
    out = dict(group=np.array([None] * m, object), z=np.full(m, np.nan), z_cnt=np.full(m, np.nan),
               z_occ=np.full(m, np.nan), exp_r=np.full(m, np.nan))
    X, n_night = features(N, O, hours)
    tot = np.nansum(np.where(hours, N, 0), 1)
    tot_day = tot * 24.0 / hours.sum()
    ok = (tot >= MIN_N) & np.isfinite(N[:, hours]).all(1) & np.isfinite(O[:, hours]).all(1)
    if not full and not window_ok(hours):
        ok[:] = False
    band = band_of(tot_day / 24.0)
    span = span_of(lanes)
    gk = np.array([resolve_group(f, s, b, groups) if o else None for f, s, b, o in zip(fn, span, band, ok)], object)
    for k in sorted(set(gk[gk != None])):                                            # noqa: E711
        i = np.flatnonzero(gk == k)
        g = groups[k]
        if full:
            gm, cal_med, cal_mad = g, float(g["cal_med"]), float(g["cal_mad"])
        else:
            gm, cal_med, cal_mad = _partial_model(g, k.split("|")[0], hours)
            if gm is None:
                ok[i] = False
                continue
        zr, exp_r, zc, zo = raw_z(X[i], gm)
        out["group"][i] = k
        out["z"][i] = (zr - cal_med) / cal_mad
        out["z_cnt"][i], out["z_occ"][i], out["exp_r"][i] = zc, zo, exp_r
    out["r"], out["n_night"] = X[:, 0], n_night
    dph, mate_r = phase_delta(X[:, 0], sig, phase, tot_day, ok)
    out["dph"], out["mate_r"] = dph, mate_r
    scored = np.isfinite(out["z"])
    flag, level = decide(np.where(scored, out["z"], -9.0), dph, n_night)
    out["flag"] = flag & scored
    out["level"] = np.where(scored, level, "not scored")
    return out


def _partial_model(g, fn, hours, min_ref=60):
    """group model on the reference profiles restricted to `hours`; split-half calibration."""
    rn, ro = g["ref_n"].astype(float), g["ref_o"].astype(float)
    R, _ = features(rn, ro, hours)
    R = R[rn[:, hours].sum(1) >= MIN_N]
    if len(R) < min_ref:
        return None, None, None
    half = np.random.default_rng(0).random(len(R)) < 0.5
    ga = fit_group(R[half], fn)
    cal = raw_z(R[~half], ga)[0]
    med = float(np.median(cal))
    mad = max(1.4826 * float(np.median(np.abs(cal - med))), 1e-3)
    return ga, med, mad
