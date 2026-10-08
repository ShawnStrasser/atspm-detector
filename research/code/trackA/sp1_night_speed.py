"""Note 60 -- deterministic night-time free-flow speed per phase (approach), hi-res only, numpy only.

Not trained: speed = (advance setback - stop-bar zone edge) / travel time, over isolated night-time vehicles that
cross a predicted Advance zone and then its lane-mate stop-bar zone (Count > Yellow_Red > Presence) while the phase
is green at both.  Robust median, with a count and the inter-quartile spread.  NaN when the sample has no night
period or too few matches.  Inputs are the classifier's own outputs (function, phase, lanes, setback); nothing
from the print or the config export.  Self-contained so it can move into model/ unchanged.

Times are seconds on a LOCAL clock (epoch of the local wall time, so hour of day = t mod 86400 / 3600).

Per-vehicle rules (defaults in PARAMS):
  night        advance ON in [night_start, night_end) local hours (00:00-05:00)
  isolated     no other ON of the advance zone within iso_s before or after (one vehicle, no platoon)
  no queue     stop-bar zone not occupied at the advance ON and no stop-bar ON in the sb_quiet_s before it
  green        phase green >= green_lead_s before the advance ON and still green >= green_tail_s after the
               stop-bar ON (no vehicle that saw red, or yellow, while travelling)
  matched      exactly one stop-bar ON in (t_adv, t_adv + d / v_min]; it is >= d / v_max after t_adv (15-80 mph)
  speed        d_eff / (t_sb - t_adv), d_eff = setback - stop-bar zone upstream edge (Presence 20 ft, else 0)
Chance matches: the same matching against the stop-bar zone shifted by a few tens of seconds gives n_bg (expected
chance matches); an answer needs n >= min_n and n >= min_snr x n_bg.  Estimate = median of the vehicles within
+-25 % of the histogram mode of log speed (chance matches are spread, the same-vehicle travel time is a peak).
Answer per detector and per phase (all the phase's advance vehicles pooled): median mph, n, n_bg, p25, p75;
NaN with a reason ('no_night', 'no_partner', 'no_setback', 'too_few', 'weak_peak') otherwise.
"""
from __future__ import annotations

import numpy as np

FTS_PER_MPH = 5280.0 / 3600.0
EDGE_FT = {"Count": 0.0, "Yellow_Red": 0.0, "Presence": 20.0}
ROLE_ORDER = ("Count", "Yellow_Red", "Presence")
PARAMS = dict(night_start=0.0, night_end=5.0, iso_s=10.0, sb_quiet_s=3.0, green_lead_s=3.0, green_tail_s=1.0,
              v_min=15.0, v_max=80.0, min_d_ft=50.0, min_n=5, min_snr=3.0, mode_win=0.25, bg_shifts=(-61.0, 47.0, 89.0))


def _hour(t):
    return np.mod(t, 86400.0) / 3600.0


def night_seconds(t0: float, t1: float, night_start=0.0, night_end=5.0) -> float:
    """Seconds of [t0, t1) that fall inside the nightly window (local clock)."""
    if t1 <= t0:
        return 0.0
    d0 = np.floor(t0 / 86400.0) * 86400.0
    tot, d = 0.0, d0
    while d < t1:
        a, b = d + night_start * 3600.0, d + night_end * 3600.0
        tot += max(0.0, min(b, t1) - max(a, t0))
        d += 86400.0
    return tot


def _in_green(t, gs, ge, lead, tail_t):
    """bool per t: some green interval [gs, ge) has gs <= t - lead and ge >= tail_t (tail_t per element)."""
    if len(gs) == 0:
        return np.zeros(len(t), bool)
    i = np.searchsorted(gs, t - lead, side="right") - 1        # last green that started early enough
    ok = i >= 0
    ii = np.clip(i, 0, None)
    return ok & (ge[ii] >= tail_t) & (gs[ii] <= t - lead)


def vehicle_speeds(adv_on, sb_on, sb_off, green_start, green_end, setback_ft, partner_role, p=None, shift=0.0):
    """Per-vehicle speeds (mph) for ONE advance detector and its lane-mate stop-bar zone(s).

    adv_on: advance ON times; sb_on / sb_off: stop-bar ON / OFF times (several zones of the same lane may be
    merged); green_start / green_end: the phase's green intervals (green_end = yellow onset).  All 1-D float
    arrays on the local clock.  shift moves the stop-bar times (background estimate: a shifted stop-bar zone has
    the same rhythm but no same-vehicle link).  Returns (speeds_mph, tt_s, t_adv) of the accepted vehicles."""
    p = {**PARAMS, **(p or {})}
    e = np.empty(0)
    if setback_ft is None or not np.isfinite(setback_ft):
        return e, e, e
    d = float(setback_ft) - EDGE_FT.get(partner_role, 0.0)
    if d < p["min_d_ft"] or len(adv_on) == 0 or len(sb_on) == 0:
        return e, e, e
    a = np.sort(np.asarray(adv_on, float))
    o = np.argsort(sb_on)
    s_on, s_off = np.asarray(sb_on, float)[o] + shift, np.asarray(sb_off, float)[o] + shift
    g0 = np.asarray(green_start, float)
    og = np.argsort(g0)
    g0, g1 = g0[og], np.asarray(green_end, float)[og]
    h = _hour(a)
    night = (h >= p["night_start"]) & (h < p["night_end"])
    gap_prev = np.diff(a, prepend=-np.inf)
    gap_next = np.diff(a, append=np.inf)
    keep = night & (gap_prev >= p["iso_s"]) & (gap_next >= p["iso_s"])
    a = a[keep]
    if len(a) == 0:
        return e, e, e
    tmin, tmax = d / (p["v_max"] * FTS_PER_MPH), d / (p["v_min"] * FTS_PER_MPH)
    # no queue: last stop-bar ON before t_adv is >= sb_quiet_s earlier and already OFF
    j = np.searchsorted(s_on, a, side="left") - 1
    jj = np.clip(j, 0, None)
    quiet = (j < 0) | ((a - s_on[jj] >= p["sb_quiet_s"]) & (s_off[jj] <= a))
    # exactly one stop-bar ON in (t_adv, t_adv + tmax]
    k0 = np.searchsorted(s_on, a, side="right")
    k1 = np.searchsorted(s_on, a + tmax, side="right")
    one = (k1 - k0) == 1
    t_sb = s_on[np.clip(k0, 0, len(s_on) - 1)]
    tt = t_sb - a
    ok = quiet & one & (tt >= tmin)
    ok &= _in_green(a, g0, g1, p["green_lead_s"], t_sb + p["green_tail_s"])
    tt, a = tt[ok], a[ok]
    return d / tt / FTS_PER_MPH, tt, a


def background(adv_on, sb_on, sb_off, green_start, green_end, setback_ft, partner_role, p=None) -> float:
    """Expected number of chance matches: mean count over the stop-bar zone shifted by p['bg_shifts'] seconds."""
    p = {**PARAMS, **(p or {})}
    return float(np.mean([len(vehicle_speeds(adv_on, sb_on, sb_off, green_start, green_end, setback_ft,
                                             partner_role, p, shift=h)[0]) for h in p["bg_shifts"]]))


def summarise(v, n_bg=0.0, p=None) -> dict:
    """Mode-anchored robust median: histogram mode of log speed (5 % bins, 3-bin smoothing), then median / p25 / p75
    of the vehicles within +-mode_win of the mode.  Answer only when n >= min_n and n >= min_snr x background."""
    p = {**PARAMS, **(p or {})}
    v = np.asarray(v, float)
    n = int(len(v))
    out = dict(speed_mph=np.nan, n=n, n_bg=round(float(n_bg), 2), n_core=0, p25=np.nan, p75=np.nan)
    if n < p["min_n"]:
        return {**out, "reason": "too_few"}
    if n < p["min_snr"] * n_bg:
        return {**out, "reason": "weak_peak"}
    lv = np.log(v)
    edges = np.arange(np.log(p["v_min"]) - 0.05, np.log(p["v_max"]) + 0.1, 0.05)
    h, _ = np.histogram(lv, edges)
    hs = np.convolve(h, np.ones(3), "same")
    m = 0.5 * (edges[:-1] + edges[1:])[int(np.argmax(hs))]
    core = v[np.abs(lv - m) <= np.log1p(p["mode_win"])]
    if len(core) < p["min_n"]:
        return {**out, "n_core": int(len(core)), "reason": "too_few"}
    q = np.percentile(core, [25, 50, 75])
    return {**out, "speed_mph": float(q[1]), "n_core": int(len(core)), "p25": float(q[0]), "p75": float(q[2]),
            "reason": ""}


def night_speed(sample_t0, sample_t1, detectors, on_times, off_times, greens, p=None):
    """Per-detector and per-phase night free-flow speed for one signal sample.

    detectors: list of dicts {det, phase, function, lanes (tuple of int), setback_ft} -- the classifier's
      outputs for every active detector (function in Advance / Presence / Count / Yellow_Red / ...).
    on_times / off_times: {det: array}; greens: {phase: (green_start array, green_end array)}.
    Returns (det_rows, phase_rows): lists of dicts."""
    p = {**PARAMS, **(p or {})}
    has_night = night_seconds(sample_t0, sample_t1, p["night_start"], p["night_end"]) > 0
    det_rows, pooled = [], {}
    sb = [x for x in detectors if x["function"] in ROLE_ORDER]
    for x in detectors:
        if x["function"] != "Advance":
            continue
        ph = x["phase"]
        pooled.setdefault(ph, [])
        row = dict(det=x["det"], phase=ph, partner=None, role=None)
        if not has_night:
            det_rows.append({**row, **summarise([], 0, p), "reason": "no_night"})
            continue
        lanes = set(x.get("lanes") or ())
        mates = [y for y in sb if y["phase"] == ph and lanes & set(y.get("lanes") or ())]
        if not mates:
            det_rows.append({**row, **summarise([], 0, p), "reason": "no_partner"})
            continue
        role = min((y["function"] for y in mates), key=ROLE_ORDER.index)
        ids = [y["det"] for y in mates if y["function"] == role]
        s_on = np.concatenate([np.asarray(on_times.get(i, []), float) for i in ids])
        s_off = np.concatenate([np.asarray(off_times.get(i, []), float) for i in ids])
        sb_ft = x.get("setback_ft")
        if sb_ft is None or not np.isfinite(sb_ft):
            det_rows.append({**row, "partner": ids, "role": role, **summarise([], 0, p), "reason": "no_setback"})
            continue
        gs, ge = greens.get(ph, (np.empty(0), np.empty(0)))
        args = (on_times.get(x["det"], np.empty(0)), s_on, s_off, gs, ge, sb_ft, role, p)
        v, tt, _ = vehicle_speeds(*args)
        nb = background(*args)
        pooled[ph].append((v, nb))
        det_rows.append({**row, "partner": ids, "role": role, **summarise(v, nb, p)})
    ph_rows = []
    for ph, vs in pooled.items():
        if not has_night:
            ph_rows.append(dict(phase=ph, **summarise([], 0, p), reason_ph="no_night"))
            continue
        v = np.concatenate([a for a, _ in vs]) if vs else np.empty(0)
        r = summarise(v, sum(b for _, b in vs), p)
        ph_rows.append(dict(phase=ph, **r, reason_ph=r["reason"] if vs else "no_partner"))
    return det_rows, ph_rows
