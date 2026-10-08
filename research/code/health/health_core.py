"""Detector health from the hi-res log (notes 38, 40, 43, 46, 47).  numpy + pandas only.  Expert RULES:
every limit is a fixed number set once on the training population (p99.5 of presumed-healthy
detectors); nothing is fitted per signal and no learned model is involved.

    health(events_df, start, end, detectors=None, phase=None, function=None) -> one row per detector

`events_df`: columns Timestamp, EventId, Parameter (the controller log, local time).
`detectors`: IGNORED since note 83 (user: health is judged from the log only).  The former
"dead channel" check (a listed channel silent for the whole window) is removed; a channel with no
actuation in the window has no row.
Note 83: every OFF -> ON gap and ON -> ON interval is compared in whole 0.1-s log ticks (chatter
< 3 ticks, rapid < 5 / < 10 ticks), never in float seconds (a gap of exactly 0.3 s was counted only
sometimes); the chatter / rapid limits were recalibrated to keep the healthy-detector fire rates.
"Undercounts vs partner" (note 79 c1) is an INFORMATION note only (never a status change).
`phase` / `function` (optional, note 43): {detector: predicted phase} and {detector: predicted
function label, or a dict of class probabilities} from the phase / function model run first.
Predictions only - never a wiring table or a print.  With `phase` a detector is compared with the
other detectors on its predicted phase (silence expected from them; spikes they share are traffic);
with `function` a predicted Bike detector is allowed to be silent.

Only actuations (81 / 82) plus comms coverage from any allowed event are used.  Detector fault
events (83-88) are NOT used (user, 2026-09-28).
Pipeline: events -> 5-min bins per channel + per-actuation intervals + ON episodes -> per-detector
statistics against a reference (the other detectors on its predicted phase, else the rest of the
signal) -> rule scores (0 = fine, 1 = certainly bad, NaN = could not be checked) -> health_score =
prod(1 - score), status ok / suspect / bad / not_enough_data, one plain-English reason, and
`bad_periods` (stuck-on and silent episodes with start / end, so only those can be dropped).

Continuous ON (note 47, the user 2026-09-29): an ON followed by more ONs before the next OFF is NORMAL
(detector extension: the next vehicle arrives before the extension ends) - it is one continuous ON from
the first ON to the next OFF.  A missing OFF is never evidence of a fault; stuck-on is judged only from
the length of a continuous ON.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

BIN_S = 300                     # 5-min bins
AGG = 3                         # 3 x 5 min = 15-min bins for shape / correlation checks
ALLOWED = (1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173)   # no fault events 83-88
MAXCH = 64
GAP_S = 120.0                   # no event of any code for this long = comms gap
CHAT_GAP = 0.3                  # OFF -> next ON within 0.3 s = re-trigger (chatter)
TICK_S = 0.1                    # the hi-res log clock
CHAT_TICKS = 3                  # note 83: chatter = OFF -> next ON gap < 3 ticks (compared in whole ticks)
IOI_TICKS_05, IOI_TICKS_1 = 5, 10   # note 83: rapid = ON -> ON interval < 5 / < 10 ticks


def _ticks(x):
    """seconds -> whole 0.1-s log ticks (NaN stays NaN); gaps are compared in ticks, never in float seconds."""
    return np.rint(np.asarray(x, dtype=float) / TICK_S)
NIGHT = (0, 5)                  # clock hours [0, 5)
DAY = (7, 19)                   # clock hours [7, 19)

# ---- rule limits (set once on the training population, note 38; not tuned per signal)
LIM = dict(
    dead_lam=(15.0, 60.0),        # expected ONs in the window (from siblings) for suspect / bad
    drop_lam=(30.0, 100.0),       # expected ONs inside the silent run
    stuck_s=(900.0, 3600.0),     # longest single ON
    chat=(0.263, 0.60),          # share of ONs that re-trigger < 3 ticks (< 0.3 s) after the OFF (n >= 50); note 83:
    #                              suspect 0.30 -> 0.263 keeps the presumed-healthy fire rate (.054 %) in whole ticks
    max5=(150, 250),             # ONs in one 5-min bin (no lane count in the log)
    nightday=(3.0, 8.0),         # (night/day rate) / max(signal's night/day, .15), when > 1
    level=(0.15, 0.05),          # after/before share ratio at the best change point
    disp=(12.0, 30.0),           # 15-min residual variance / Poisson variance (note 38; not scored since 43)
    corr_gap=(0.60, 0.80),       # expected minus observed 15-min correlation with the signal
)
SUSPECT_H, BAD_H = 0.70, 0.25    # health_score cut-offs
# ---- note 43
# choppiness: suspect limit of chop15 (15-min dispersion around the local share of the reference),
# p99.5 of presumed-healthy by window length; bad at 2x.  Phase reference / rest-of-signal reference.
CHOP_LIM = {"phase": ((2, 10.5), (6, 10.5), (24, 6.0), (66, 5.0)),
            "signal": ((2, 12.0), (6, 11.2), (24, 8.0), (66, 7.5))}
BIKE_K = 0.02          # a predicted Bike detector's expected share vs a typical detector's (median .06 % vs 3.6 %)
CO_STUCK_N = 3         # held ON together with >= 3 others = shared event: capped at suspect, period listed
STUCK_LAM = None       # note 47: a continuous ON counts as stuck only if >= this many ONs were expected
#                        meanwhile (None = always).  Replaces note 43's "no OFF logged" gate - a missing OFF
#                        is normal (extension) and is no longer treated differently from any other ON.
REC_RATIO = (0.5, 2.0)  # share after / before an episode (6 h each side) for a clean recovery
REC_MIN_BINS = 36      # >= 3 h of data after the episode (and before it) to judge the recovery
REC_BINS = 72          # 6 h each side
CO_MIN_S = 300.0       # another detector's ON counts as "held ON at the same time" from 5 min
NAMES = {"dropout": "sudden silence", "stuck": "stuck-on",
         "chatter": "chatter (needs 50 actuations)",
         "rapid": "rapid re-actuation (needs 50 actuations)", "volume": "volume",
         "night_day": "night vs day (needs 2 h of each)",
         "level": "drop in level (needs 2 h and 20 actuations)",
         "choppy": "choppy counts (needs 2 h and 30 actuations)",
         "corr": "moves with the signal (needs 12 h)",
         "rapid_hour": "rapid re-actuation in the worst hour (needs an hour with 50 actuations)",
         "short_on": "implausibly short ONs (needs 50 actuations; not for pulse-mode detectors)",
         "night_drop": "night count vs its partner / phase (needs 2 h of night and day and 30 expected at night)"}


# ============================================================================ binning
def continuous_on(td, ed, pd_, T):
    """Detector events sorted by (channel, time, ON before OFF).  Returns, per event, the time of the
    next OFF of the same channel (T = still ON at the window end) and whether the event is an ON that
    STARTS a continuous ON (its previous event on the channel is not an ON).  ONs inside a continuous
    ON (ON -> ON, no OFF between) are normal with extension and belong to the ON that started it."""
    n = len(td)
    offs = np.where(ed == 81)[0]
    k = np.searchsorted(offs, np.arange(n), "left")
    p = offs[np.minimum(k, max(len(offs) - 1, 0))] if len(offs) else np.zeros(n, int)
    ok = (k < len(offs)) & (pd_[p] == pd_) if len(offs) else np.zeros(n, bool)
    end = np.where(ok, td[p] if len(offs) else T, T)
    prev_on = np.r_[False, (pd_[1:] == pd_[:-1]) & (ed[:-1] == 82)]
    start = (ed == 82) & ~prev_on
    return end, start


def events_to_bins(ev: pd.DataFrame, start, end, detectors=None, bin_s: int = BIN_S) -> dict:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    T = (end - start).total_seconds()
    nb = int(np.ceil(T / bin_s))
    e = ev[["Timestamp", "EventId", "Parameter"]]
    e = e[(e.Timestamp >= start) & (e.Timestamp < end) & e.EventId.isin(ALLOWED)]
    e = e[~(e.EventId.isin((81, 82)) & (e.Parameter > MAXCH))].drop_duplicates()
    t = (e.Timestamp - start).dt.total_seconds().to_numpy()
    eid = e.EventId.to_numpy().astype(int)
    par = e.Parameter.to_numpy().astype(int)
    cov = np.bincount((t // bin_s).astype(int), minlength=nb)[:nb] > 0
    ut = np.unique(t)
    gi = np.diff(ut) > GAP_S
    gaps0 = ut[:-1][gi]                             # comms-gap start times
    dm = np.isin(eid, (81, 82))
    chans = set(np.unique(par[dm]).tolist())
    if detectors is not None:
        chans |= {int(x) for x in detectors}
    dets = np.array(sorted(chans), dtype=int)
    nd = len(dets)
    idx = {c: i for i, c in enumerate(dets)}
    out = {k: np.zeros((nd, nb), np.float32) for k in ("n_on", "occ", "n_chat")}
    out["dmax"] = np.zeros((nd, nb), np.float32)
    dur_max = np.zeros(nd)
    # --- ON intervals (order: time, ON before OFF at the same instant, as dq_core)
    td, ed, pd_ = t[dm], eid[dm], par[dm]
    o = np.lexsort((ed != 82, td, pd_))
    td, ed, pd_ = td[o], ed[o], pd_[o]
    same_next = np.r_[pd_[1:] == pd_[:-1], False]
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    nxt_t, nxt_e = np.r_[td[1:], np.nan], np.r_[ed[1:], 0]
    prv_t, prv_e = np.r_[np.nan, td[:-1]], np.r_[0, ed[:-1]]
    on = ed == 82
    # continuous ON: from the ON that starts it to the next OFF (ONs inside it are extension re-calls);
    # still ON at the window end -> to the end.  Inner ONs get NaN (their time is already counted).
    c_end, c_start = continuous_on(td, ed, pd_, T)
    dur = np.where(c_start, c_end - td, np.nan)[on]
    gap = np.where(same_prev & (prv_e == 81), td - prv_t, np.nan)[on]
    ton, ci = td[on], np.array([idx[c] for c in pd_[on]], dtype=int)
    if len(ton):
        long_ = np.where(dur > 60)[0]                     # an ON over a comms gap is not "stuck"
        if len(long_) and len(gaps0):
            k = np.searchsorted(gaps0, ton[long_])
            hit = (k < len(gaps0)) & (gaps0[np.minimum(k, len(gaps0) - 1)] < ton[long_] + dur[long_])
            dur[long_[hit]] = np.nan
        b0 = np.minimum((ton // bin_s).astype(int), nb - 1)
        np.add.at(out["n_on"], (ci, b0), 1)
        with np.errstate(invalid="ignore"):
            np.add.at(out["n_chat"], (ci, b0), (_ticks(gap) < CHAT_TICKS).astype(float))
        dd = np.nan_to_num(dur, nan=0.0)
        np.maximum.at(out["dmax"], (ci, b0), dd)
        np.maximum.at(dur_max, ci, dd)
        # occupancy, split across bins
        fin = np.isfinite(dur)
        s, f, c = ton[fin], np.minimum(ton[fin] + dur[fin], T), ci[fin]
        bs, bf = (s // bin_s).astype(int), np.minimum((f // bin_s).astype(int), nb - 1)
        nspan = bf - bs + 1
        rep = np.repeat(np.arange(len(s)), nspan)
        bb = bs[rep] + (np.arange(len(rep)) - np.repeat(np.cumsum(nspan) - nspan, nspan))
        ov = np.minimum(f[rep], (bb + 1) * bin_s) - np.maximum(s[rep], bb * bin_s)
        np.add.at(out["occ"], (c[rep], bb), np.clip(ov, 0, None))
    # a channel whose first event is an OFF was ON when the window opened: occupied until then
    first = np.r_[True, pd_[1:] != pd_[:-1]]
    lead = first & (ed == 81)
    for tt, cc in zip(td[lead], pd_[lead]):
        bf = min(int(tt // bin_s), nb - 1)
        i = idx[int(cc)]
        out["occ"][i, :bf] += bin_s
        out["occ"][i, bf] += tt - bf * bin_s
    hrs = ((start + pd.to_timedelta(np.arange(nb) * bin_s, unit="s")).hour).to_numpy()
    out.update(dets=dets, cov=cov, hour=hrs, dur_max=dur_max, start=start, bin_s=bin_s,
               listed=None if detectors is None else np.isin(dets, [int(x) for x in detectors]))
    return out


# ======================================================================== statistics
def _agg(a, k=AGG):
    n = a.shape[-1] // k * k
    return a[..., :n].reshape(*a.shape[:-1], -1, k).sum(-1)


def _runs(mask):
    """start, end (exclusive) of True runs in a 1-D bool array."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def det_stats(B: dict, ref: list | None = None) -> pd.DataFrame:
    """`ref` (note 43): per detector, the row indices of its reference detectors (the other live
    detectors on its predicted phase); None = the rest of the signal (note 38)."""
    n, occ, cov = B["n_on"].astype(float), B["occ"].astype(float), B["cov"]
    nd, nb = n.shape
    hrs = B["hour"]
    cov_h = cov.sum() * B["bin_s"] / 3600
    tot = n.sum(0)
    rows = []
    n15, cov15 = _agg(n), _agg(cov[None].astype(float))[0] == AGG
    hr15 = hrs[: len(cov15) * AGG: AGG]
    active = (n.sum(1) > 0) | (occ.sum(1) > 0)
    for i in range(nd):
        r = {"detector": int(B["dets"][i])}
        x, S = n[i], (tot - n[i] if ref is None else n[ref[i]].sum(0))   # this detector / its reference
        # the reference = other detectors that are alive (a dead sibling adds nothing)
        r["n_on"] = x[cov].sum()
        r["cov_h"] = cov_h
        r["sib_on"] = S[cov].sum()
        r["n_sib"] = int(active.sum() - active[i])
        r["share"] = r["n_on"] / max(r["sib_on"], 1)
        r["rate_h"] = r["n_on"] / max(cov_h, 1e-9)
        r["max5"] = x.max() if nb else 0
        r["occ_frac"] = occ[i][cov].sum() / max(cov.sum() * B["bin_s"], 1)
        r["dur_max"] = B["dur_max"][i]
        r["chat_frac"] = B["n_chat"][i].sum() / max(r["n_on"], 1)
        # ---- silent runs: bins with no ON and (almost) no occupancy, on covered bins;
        # expected ONs in the run = the detector's share outside the run x the siblings inside it
        quiet = (x == 0) & (occ[i] < 1) & cov
        s0, s1 = _runs(quiet)
        best = (0.0, -1, -1)
        if len(s0) and r["n_on"] > 0:
            cS = np.r_[0, np.cumsum(np.where(cov, S, 0))]
            for a, b in zip(s0, s1):
                s_in = cS[b] - cS[a]
                s_out = cS[-1] - s_in
                lam = r["n_on"] / max(s_out, 1) * s_in
                if lam > best[0]:
                    best = (lam, a, b)
        r["drop_lam"], r["drop_b0"], r["drop_b1"] = best
        r["drop_to_end"] = best[2] == nb or (best[2] > 0 and not cov[best[2]:].any())
        # other live detectors that were silent through the same run (shared unit / comms?)
        r["co_silent"] = 0
        if best[2] - best[1] >= 3:
            q = ((n[:, best[1]:best[2]] == 0) & (occ[:, best[1]:best[2]] < 1)).mean(1) >= 0.8
            r["co_silent"] = int((q & active).sum() - 1)
        # dead for the whole window: expected from the typical share of a live detector
        r["dead_lam"] = np.nan
        if r["n_on"] == 0 and occ[i].sum() < 1:
            live = active.copy(); live[i] = False
            med_share = np.median(n[live][:, cov].sum(1) / max(tot[cov].sum(), 1)) if live.any() else np.nan
            r["dead_lam"] = med_share * tot[cov].sum() * 0.25   # a quarter of the median live share
        # ---- level shift of the share (15-min bins, best single change point, Poisson LLR)
        xs, Ss = n15[i][cov15], (n15.sum(0) - n15[i] if ref is None else n15[ref[i]].sum(0))[cov15]
        r["level_ratio"], r["level_llr"] = np.nan, 0.0
        if len(xs) >= 8 and xs.sum() >= 20:
            cx, cs = np.cumsum(xs), np.cumsum(Ss)
            X, SS = cx[-1], cs[-1]
            k = np.arange(2, len(xs) - 1)
            x1, s1_, x2, s2 = cx[k - 1], cs[k - 1], X - cx[k - 1], SS - cs[k - 1]
            p0 = X / max(SS, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                def ll(xx, ss):
                    p = np.where(ss > 0, xx / np.maximum(ss, 1e-9), 0)
                    return np.where(xx > 0, xx * np.log(np.maximum(p, 1e-12)), 0) - p * ss
                llr = ll(x1, s1_) + ll(x2, s2) - (np.where(True, X * np.log(max(p0, 1e-12)), 0) - p0 * SS)
            j = int(np.nanargmax(llr))
            r["level_llr"] = float(llr[j])
            r["level_ratio"] = float(((x2[j] + .5) / max(s2[j], 1)) / ((x1[j] + .5) / max(s1_[j], 1)))
            r["level_b"] = int(k[j]) * AGG
        # ---- 15-min shape: correlation with the signal and residual dispersion
        r["corr"], r["corr_exp"], r["disp"] = np.nan, np.nan, np.nan
        if len(xs) >= 8 and xs.sum() >= 30 and Ss.std() > 0:
            E = Ss * xs.sum() / max(Ss.sum(), 1)
            if xs.std() > 0:
                r["corr"] = float(np.corrcoef(xs, Ss)[0, 1])
            r["corr_exp"] = float(np.sqrt(E.var() / (E.var() + E.mean()))) if E.mean() > 0 else np.nan
            r["disp"] = float(((xs - E) ** 2).mean() / max(E.mean(), 1e-9))
        r["corr_gap"] = (r["corr_exp"] - (r["corr"] if np.isfinite(r["corr"]) else 0.0)
                         if np.isfinite(r["corr_exp"]) else np.nan)
        # extra shape statistics (model features only; the rules do not use them)
        r["zero15_exp"], r["max_dev_1h"] = np.nan, np.nan
        if len(xs) >= 4 and xs.sum() >= 5 and Ss.sum() > 0:
            E = Ss * xs.sum() / Ss.sum()
            big = E >= 3
            r["zero15_exp"] = float(((xs == 0) & big).sum() / max(big.sum(), 1))
            m = len(xs) // 4 * 4
            if m >= 8:
                xh, Eh = xs[:m].reshape(-1, 4).sum(1), E[:m].reshape(-1, 4).sum(1)
                r["max_dev_1h"] = float(np.max(np.abs(np.log((xh + 1) / (Eh + 1)))))
        r["mean_on_s"] = occ[i][cov].sum() / max(r["n_on"], 1)
        r["occ_max_bin"] = float((occ[i][cov] / B["bin_s"]).max()) if cov.any() else np.nan
        # ---- night vs day (needs >= 2 h of each in the window)
        nm = cov15 & (hr15 >= NIGHT[0]) & (hr15 < NIGHT[1])
        dmk = cov15 & (hr15 >= DAY[0]) & (hr15 < DAY[1])
        r["night_day"], r["sig_night_day"] = np.nan, np.nan
        if nm.sum() >= 8 and dmk.sum() >= 8 and n15[i][dmk].sum() + n15[i][nm].sum() >= 30:
            r["night_day"] = n15[i][nm].mean() / max(n15[i][dmk].mean(), 0.25)
            sib = n15.sum(0) - n15[i]
            r["sig_night_day"] = sib[nm].mean() / max(sib[dmk].mean(), 0.25)
        rows.append(r)
    st = pd.DataFrame(rows)
    # sibling-relative dispersion (robust z within the signal)
    ld = np.log(st["disp"])
    med, mad = np.nanmedian(ld), np.nanmedian(np.abs(ld - np.nanmedian(ld)))
    st["disp_z"] = (ld - med) / max(1.4826 * mad, 0.25) if np.isfinite(med) else np.nan
    st["listed"] = B["listed"] if B["listed"] is not None else True
    return st


# ============================================================================= rules
def _score(x, lo, hi):
    """0 below the suspect limit lo; .35 at lo rising to 1 at the bad limit hi; NaN stays NaN."""
    if x is None or not np.isfinite(x):
        return np.nan
    if x < lo:
        return 0.0
    return float(0.35 + 0.65 * np.clip((x - lo) / (hi - lo), 0, 1))


def _hm(b, B):
    t = B["start"] + pd.Timedelta(seconds=int(b) * B["bin_s"])
    return t.strftime("%a %d %b %H:%M")


def _fn(r):
    """predicted function label of a stats row (None when not supplied)."""
    f = r.get("pred_function")
    return f if isinstance(f, str) else None


def _f(r, k):
    v = r.get(k, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _recur_text(r, s: dict) -> str:
    """(a) note 46: spikes that recur at the same time of day and look like vehicles lean towards
    traffic (capped at suspect on 2-3 days, cleared from 4 days); spikes whose actuations do not look
    like vehicles are said so.  Edits s["choppy"] in place; returns the text to append."""
    nd, rs, pl = _f(r, "n_days"), _f(r, "rec_share"), _f(r, "spk_plaus")
    fast, dr = _f(r, "spk_fast"), _f(r, "spk_dur_ratio")
    tx = ""
    if np.isfinite(nd) and nd >= RECUR_DAYS[0] and np.isfinite(rs) and rs >= RECUR_SHARE and _f(r, "rec_ndates") >= 2:
        firm = nd >= RECUR_DAYS[1] and _f(r, "rec_ndates") >= 3
        wk = _f(r, "spk_weekend")
        tx = (f"; the spikes recur at the same time of day ({r.get('spk_tod')}) on "
              f"{int(_f(r, 'rec_ndates'))} different dates in {nd:.1f} days of data"
              + (" including the weekend" if np.isfinite(wk) and wk > 0.2 else ""))
        if pl == 1:
            s["choppy"] = min(s["choppy"], 0.0 if firm else 0.35)
            tx += (" and the actuations inside them look like vehicles (gaps and ON times as usual) - probably "
                   "real traffic that only this detector's lane sees (school, shift, event), or a daily sensor "
                   "effect such as low sun on a camera"
                   + ("" if firm else "; judged on fewer than 4 days, so kept as suspect"))
        elif pl == 0:
            tx += " but the actuations inside them do not look like vehicles"
    elif np.isfinite(nd) and nd >= RECUR_DAYS[0] and np.isfinite(rs):
        tx = "; the spikes do not recur at the same time on other days"
    if pl == 0:
        bits = []
        if np.isfinite(fast) and fast >= 1:
            bits.append(f"{_f(r, 'spk_ioi_lt1'):.0%} of them come within 1 s of the previous one")
        if np.isfinite(dr) and not (1 / 3 <= dr <= 3):
            bits.append(f"their ON time is {dr:.1f}x its usual")
        if bits:
            tx += " (inside the spikes " + " and ".join(bits) + ")"
    return tx


def _bike_k(r, fn) -> float:
    """expected-volume factor for silence: a Bike loop is often silent.  With class probabilities
    (note 46, OPTS conf) the factor is weighted by P(Bike) instead of the argmax label."""
    pb = _f(r, "p_bike")
    if OPTS["conf"] and np.isfinite(pb):
        return 1.0 - pb * (1.0 - BIKE_K)
    return BIKE_K if fn == "Bike" else 1.0


FAMILY = {"chatter": "fast", "rapid": "fast", "rapid_hour": "fast", "short_on": "short", "dead": "silent",
          "dropout": "silent", "stuck": "stuck", "level": "level", "night_drop": "level", "choppy": "shape",
          "corr": "shape", "volume": "shape", "night_day": "shape"}


def rules(st: pd.DataFrame, B: dict) -> pd.DataFrame:
    L = LIM
    out = []
    for _, r in st.iterrows():
        s, why, note = {}, {}, []
        capped = set()          # findings capped at suspect because only a period is affected (note 47 c)
        fn = _fn(r)
        onph = r.get("drop_ref") == "phase"
        # note 83: the "dead channel" check is removed (log-only rule, user); a whole-window-silent channel has no
        # row (health() ignores the channel list), so `dead` is never true here
        dead = False
        # sudden drop to zero while its reference keeps counting
        if not dead and r.n_on > 0:
            s["dropout"] = 0.0 if r.get("drop_in_stuck") is True else _score(r.drop_lam, *L["drop_lam"])
            if s["dropout"] > 0:
                end = "and stayed silent to the end" if r.drop_to_end else f"until {_hm(r.drop_b1, B)}"
                who = "the other detectors on its phase" if onph else "the other detectors"
                why["dropout"] = (f"went silent at {_hm(r.drop_b0, B)} {end} "
                                  f"({(r.drop_b1 - r.drop_b0) * B['bin_s'] / 3600:.1f} h; about "
                                  f"{r.drop_lam:.0f} actuations expected from {who})")
                if r.get("co_silent", 0) >= max(3, 0.5 * r.n_sib):
                    why["dropout"] += (f"; {int(r.co_silent)} of the {int(r.n_sib)} other detectors went "
                                       f"silent with it - likely a cabinet or communications problem, "
                                       f"not this detector alone")
                elif r.get("co_silent", 0) > 0:
                    why["dropout"] += (f"; {int(r.co_silent)} other detector(s) went silent at the "
                                       f"same time (shared card or unit?)")
                rr = _f(r, "drop_rec")
                if not r.drop_to_end and np.isfinite(rr):
                    pt = _f(r, "drop_partner")
                    vs = f" relative to d{int(pt)}, which it tracks" if np.isfinite(pt) else ""
                    if REC_RATIO[0] <= rr <= REC_RATIO[1]:
                        s["dropout"] = min(s["dropout"], 0.35)
                        capped.add("dropout")
                        why["dropout"] += (f"; afterwards it counted its usual share again ({rr:.0%} of "
                                           f"before{vs}) - the data outside this period looks usable")
                    else:
                        why["dropout"] += (f"; after it came back it counted {rr:.0%} of its earlier "
                                           f"share - not a clean recovery")
        # stuck-on: the longest continuous ON (ON to the next OFF; further ONs inside it are normal
        # extension re-calls, note 47).  Inside health() only the episodes that count (ep_dur) are used.
        sd = _f(r, "ep_dur") if "ep_dur" in r else r.dur_max
        sd = 0.0 if not np.isfinite(sd) else sd
        s["stuck"] = _score(sd, *L["stuck_s"])
        if sd >= L["stuck_s"][0]:
            t0, t1 = r.get("ep_t0"), r.get("ep_t1")
            span = (f" ({pd.Timestamp(t0).strftime('%a %d %b %H:%M')} - {pd.Timestamp(t1).strftime('%H:%M')})"
                    if isinstance(t0, pd.Timestamp) and isinstance(t1, pd.Timestamp) else "")
            why["stuck"] = f"stayed ON continuously for {sd / 60:.0f} min (from an ON to the next OFF){span}"
            lm = _f(r, "ep_lam")
            if np.isfinite(lm):
                why["stuck"] += f", while its usual share of the traffic meanwhile was about {lm:.0f} actuations"
            co, rr = _f(r, "ep_co"), _f(r, "ep_rec")
            fw, nd = _f(r, "ep_flow"), _f(r, "ep_days")
            daily = (f"; it happened at about this time on {int(nd)} days" if np.isfinite(nd) and nd >= 2 else "")
            if np.isfinite(co) and co >= CO_STUCK_N:
                # a shared event: the period is unusable, but it is not this detector's fault alone
                s["stuck"] = min(s["stuck"], 0.35)
                capped.add("stuck")
                if OPTS["group"] and np.isfinite(fw) and fw >= FLOW_OK:
                    why["stuck"] += (f"; {int(co)} other detectors were held ON with it while the rest of its phase "
                                     f"kept counting ({fw:.0%} of its usual share) - sensors failing together "
                                     f"(e.g. one radar / camera unit or card), not a real queue{daily}: "
                                     f"drop the period, keep the detector")
                else:
                    why["stuck"] += (f"; {int(co)} other detectors were held ON at the same time - a shared event "
                                     f"(real queue, closure or works, or cabinet-wide), not this detector alone: "
                                     f"drop the period, keep the detector{daily}")
            else:
                if np.isfinite(rr):
                    pt = _f(r, "ep_partner")
                    vs = f" relative to d{int(pt)}, which it tracks" if np.isfinite(pt) else ""
                    if REC_RATIO[0] <= rr <= REC_RATIO[1]:
                        s["stuck"] = min(s["stuck"], 0.35)
                        capped.add("stuck")
                        why["stuck"] += (f"; afterwards it counted its usual share again ({rr:.0%} of "
                                         f"before{vs}) - the data outside this period looks usable")
                    else:
                        why["stuck"] += f"; after it recovered it counted {rr:.0%} of its earlier share{vs}"
                if np.isfinite(co) and co > 0:
                    why["stuck"] += f"; {int(co)} other detector(s) held ON at the same time (shared card?)"
                why["stuck"] += daily
        if r.n_on >= 50:
            s["chatter"] = _score(r.chat_frac, *L["chat"])
            if r.chat_frac >= L["chat"][0]:
                why["chatter"] = f"{r.chat_frac:.0%} of actuations re-trigger less than 0.3 s after the previous OFF"
        rp = r.get("rapid", np.nan)
        if rp is not None and np.isfinite(rp):
            s["rapid"] = _score(rp, 1.0, 2.0)
            if rp >= 1.0:
                why["rapid"] = (f"{r.ioi_lt1:.0%} of actuations start within 1 s of the previous one "
                                f"({int(r.n_burst)} bursts of 5 or more) - faster than separate vehicles "
                                f"can pass; chatter, crosstalk or a failing card/amplifier")
        rh = _f(r, "rapid_h")
        if OPTS["rapid_hour"] and np.isfinite(rh):
            s["rapid_hour"] = _score(rh, *RAPID_H_LIM)
            if s["rapid_hour"] > 0:
                t0 = B["start"] + pd.Timedelta(seconds=_f(r, "rapid_h_t"))
                why["rapid_hour"] = (f"in its worst hour ({t0.strftime('%a %d %b %H:%M')}) actuations came within 1 s "
                                     f"of each other {rh:.1f}x as often as healthy detectors like it allow")
        # note 47 (b): ONs too short for a vehicle on a detector that is not in pulse mode
        nd_ = _f(r, "n_dur")
        if np.isfinite(nd_) and nd_ >= SHORT_MIN_N:
            bmin = _f(r, "short_blk_min")
            mode_sh = bmin if np.isfinite(bmin) else _f(r, "short2")
            if mode_sh < PULSE_MODE:
                sa, sb_ = _f(r, "short2"), _f(r, "short_blk_max")
                v = np.nanmax([sa / SHORT_LIM[0], sb_ / SHORT_LIM[1] if np.isfinite(sb_) else np.nan])
                s["short_on"] = _score(v, 1.0, 2.0)
                if v >= 1.0:
                    ft = LOOP_FT.get(fn, 6)
                    mph = (ft + CAR_FT) / 0.2 * 0.6818
                    why["short_on"] = (f"{sa:.0%} of its ONs last 0.2 s or less although it is not set to pulse "
                                       f"(median ON {_f(r, 'med_dur'):.1f} s); a {CAR_FT}-ft car crossing a {ft}-ft "
                                       f"zone in 0.2 s would be doing {mph:.0f} mph")
                    kb = _f(r, "short_blk")
                    if np.isfinite(sb_) and np.isfinite(kb) and sb_ >= 1.5 * max(sa, 1e-9):
                        why["short_on"] += (f"; worst {int(kb) * 3:02d}:00-{int(kb) * 3 + 3:02d}:00 "
                                            f"({sb_:.0%}) - it changes with the time of day")
                if not OPTS["min_on"]:          # note 47: informative only - it added no catch at +0.5 pt FA
                    del s["short_on"]
                    if "short_on" in why:
                        note.append(why.pop("short_on"))
            elif OPTS["min_on"]:
                s["short_on"] = 0.0
        # note 47 (b'): stops counting at night while the detector it tracks (or its phase) keeps counting
        nr = _f(r, "night_ratio")
        if OPTS["night_drop"] and np.isfinite(nr):
            s["night_drop"] = _score(-np.log(max(nr, 1e-6)), -np.log(NIGHT_DROP_LIM[0]), -np.log(NIGHT_DROP_LIM[1]))
            if s["night_drop"] > 0:
                why["night_drop"] = (f"at night ({NIGHT_H[0]:02d}:00-{NIGHT_H[1]:02d}:00) it counted "
                                     f"{_f(r, 'night_n'):.0f} actuations where about {_f(r, 'night_exp'):.0f} were "
                                     f"expected from {r.get('night_ref')} ({nr:.0%}) - it misses vehicles at night")
                mn, md_ = _f(r, "night_med_dur"), _f(r, "med_dur")
                if np.isfinite(mn) and np.isfinite(md_) and mn < 0.5 * md_:
                    why["night_drop"] += (f"; the few it sees at night are short (median ON {mn:.1f} s vs "
                                          f"{md_:.1f} s overall)")
        s["volume"] = _score(r.max5, *L["max5"])
        if r.max5 >= L["max5"][0]:
            why["volume"] = f"{r.max5:.0f} actuations in one 5-min bin (implausible)"
        if np.isfinite(r.night_day):
            ratio = r.night_day / max(r.sig_night_day, 0.15)
            s["night_day"] = _score(ratio, *L["nightday"]) if r.night_day > 1.0 else 0.0
            if s["night_day"] > 0:
                why["night_day"] = (f"counts more at night than by day ({r.night_day:.1f}x the day rate; "
                                    f"the other detectors {r.sig_night_day:.2f}x)")
        if np.isfinite(r.level_ratio):
            s["level"] = 0.0 if r.level_llr <= 50 else _score(
                -np.log(r.level_ratio), -np.log(L["level"][0]), -np.log(L["level"][1]))
            if s["level"] > 0:
                why["level"] = (f"dropped to {r.level_ratio:.0%} of its earlier share of the signal's "
                                f"traffic from {_hm(r.level_b, B)}")
        # choppy (note 43; replaces note 38's `erratic`): 15-min counts around the local share of
        # the reference - spikes that the other detectors on the phase share are traffic, not a fault
        ch, cl = _f(r, "chop15"), _f(r, "chop_lim")
        if np.isfinite(ch) and np.isfinite(cl):
            s["choppy"] = _score(ch, cl, 2 * cl)
            if s["choppy"] > 0:
                who = ("the other detectors on its phase" if r.get("ref_kind") == "phase"
                       else "the rest of the signal")
                why["choppy"] = (f"its 15-min counts jump around {ch:.0f}x more than {who} explain "
                                 f"(spikes and dips they do not share; healthy detectors stay under {cl:.0f}x)")
                if OPTS["recur"]:
                    c0 = s["choppy"]
                    why["choppy"] += _recur_text(r, s)
                    if s["choppy"] < c0:
                        capped.add("choppy")
        if np.isfinite(r.corr_gap) and r.cov_h >= 12:
            s["corr"] = _score(r.corr_gap, *L["corr_gap"])
            if s["corr"] > 0:
                why["corr"] = (f"does not rise and fall with the rest of the signal "
                               f"(correlation {r['corr'] if np.isfinite(r['corr']) else 0:.2f}, "
                               f"expected about {r.corr_exp:.2f})")
        vals = {k: v for k, v in s.items() if np.isfinite(v)}
        hs = float(np.prod([1 - v for v in vals.values()])) if vals else np.nan
        # enough data? the silent-run / dead checks need a meaningful expected volume
        if dead:
            lam = r.dead_lam * _bike_k(r, fn)
            can_see = bool(np.isfinite(lam) and lam >= L["dead_lam"][0])
        else:
            can_see = r.n_on >= 20
        if np.isfinite(hs) and hs < BAD_H:
            status = "bad"
        elif np.isfinite(hs) and hs < SUSPECT_H:
            status = "suspect"
        elif not can_see:
            status = "not_enough_data"
        else:
            status = "ok"
        # note 47 (c): independent findings add up - two different kinds of evidence, each at least at its
        # suspect limit and not limited to a period, make the detector bad
        fams = sorted({FAMILY[k] for k, v in vals.items() if v >= 0.35 and k not in capped and k in FAMILY})
        if OPTS["grade2"] and status == "suspect" and len(fams) >= 2:
            status = "bad"
            note.append("bad because " + str(len(fams)) + " independent findings agree")
        top = sorted(((v, k) for k, v in vals.items() if k in why), reverse=True)
        reason = "; ".join(why[k] for v, k in top[:2]) if top and status in ("bad", "suspect") else ""
        if status == "not_enough_data":
            reason = (f"only {r.n_on:.0f} actuations in {r.cov_h:.1f} h - too few to judge" if not dead else
                      ("silent, but a bike detector is often silent for hours" if fn == "Bike"
                       else "silent, but the signal was too quiet to expect actuations"))
        if status == "ok":
            reason = f"{r.n_on:,.0f} actuations in {r.cov_h:.1f} h, nothing unusual"
        fc = _f(r, "f_conf")
        if np.isfinite(fc) and fc < CONF_FUNC_LOW and status in ("bad", "suspect"):
            note.append(f"the classifier is unsure what this detector is ({fn} at {fc:.0%})")
        if note:
            reason += "; note: " + "; ".join(note)
        skipped = [NAMES[k] for k in NAMES if k not in vals and not (k == "dead" and not dead)
                   and not (k == "dropout" and dead) and not (k == "short_on" and not OPTS["min_on"])]
        if OPTS["recur"] and s.get("choppy", 0) > 0:
            nd = _f(r, "n_days")
            if not (np.isfinite(nd) and nd >= RECUR_DAYS[0]):
                skipped.append("recurring spikes by time of day (needs >= 2 days; firm from 4)")
            elif nd < RECUR_DAYS[1]:
                skipped.append("firm recurrence call (needs >= 4 days; judged on %.1f)" % nd)
        out.append(dict(detector=r.detector, health_score=hs, status=status, reason=reason,
                        not_checked=", ".join(skipped), n_families=len(fams),
                        **{f"s_{k}": v for k, v in s.items()}))
    return pd.DataFrame(out)


# ================================================================ actuation level (note 40)
LOGB = np.r_[0.0, np.logspace(np.log10(0.15), np.log10(600.0), 19), np.inf]   # histogram edges, s


def _js(p, q):
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log(p / m), 0).sum()
        b = np.where(q > 0, q * np.log(q / m), 0).sum()
    return float(0.5 * (a + b) / np.log(2))


def _runs_of(mask, min_len):
    """number of elements in True runs of length >= min_len."""
    s0, s1 = _runs(mask)
    L = s1 - s0
    return int(L[L >= min_len].sum()), int((L >= min_len).sum())


def act_stats(ev: pd.DataFrame, start, end, min_on: int = 20, light: bool = False) -> pd.DataFrame | None:
    """Per-detector statistics of the individual actuations in [start, end) (note 40).

    ON duration, OFF gap (OFF -> next ON), ON -> ON interval; any interval that spans a comms
    gap (> GAP_S with no event of any code) is dropped.  Colour state (1 = begin green,
    8 = begin yellow) is used only phase-anonymously: each detector's best-matching phase is
    the one whose green holds the most ON starts relative to its green time.
    `light=True` computes only what the packaged `rapid` rule needs (durations, intervals, bursts);
    the other statistics were tested in note 40 and did not beat loosening the existing rules."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    T = (end - start).total_seconds()
    e = ev[["Timestamp", "EventId", "Parameter"]]
    e = e[(e.Timestamp >= start) & (e.Timestamp < end) & e.EventId.isin(ALLOWED)]
    e = e[~(e.EventId.isin((81, 82)) & (e.Parameter > MAXCH))].drop_duplicates()
    if e.empty:
        return None
    t = (e.Timestamp - start).dt.total_seconds().to_numpy()
    eid = e.EventId.to_numpy().astype(int)
    par = e.Parameter.to_numpy().astype(int)
    ut = np.unique(t)
    gi = np.diff(ut) > GAP_S
    g0, g1 = ut[:-1][gi], ut[1:][gi]
    cov_s = T - float((g1 - g0).sum())

    def spans_gap(a, b):
        if not len(g0):
            return np.zeros(len(a), bool)
        k = np.searchsorted(g0, a)
        kk = np.minimum(k, len(g0) - 1)
        return (k < len(g0)) & (g0[kk] < b)

    # colour: green intervals per phase, and all colour-change instants
    cm = np.isin(eid, (1, 7, 8, 9, 10, 11))
    tc = np.sort(t[cm])
    greens = {}
    for p in ([] if light else np.unique(par[eid == 1])):
        s_ = np.sort(t[(eid == 1) & (par == p)])
        y_ = np.sort(t[(eid == 8) & (par == p)])
        k = np.searchsorted(y_, s_, "right")
        e_ = np.where(k < len(y_), y_[np.minimum(k, len(y_) - 1)], T)
        ok = ~spans_gap(s_, e_)
        if ok.sum() >= 3:
            greens[int(p)] = (s_[ok], e_[ok], float((e_[ok] - s_[ok]).sum()) / max(cov_s, 1))
    # detector events: time order, ON before OFF at the same instant
    dm = np.isin(eid, (81, 82))
    td, ed, pd_ = t[dm], eid[dm], par[dm]
    o = np.lexsort((ed != 82, td, pd_))
    td, ed, pd_ = td[o], ed[o], pd_[o]
    same_next = np.r_[pd_[1:] == pd_[:-1], False]
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    nxt_t, nxt_e = np.r_[td[1:], np.nan], np.r_[ed[1:], 0]
    prv_t, prv_e = np.r_[np.nan, td[:-1]], np.r_[0, ed[:-1]]
    on = ed == 82
    dur = np.where(same_next & (nxt_e == 81), nxt_t - td, np.nan)[on]
    gap = np.where(same_prev & (prv_e == 81), td - prv_t, np.nan)[on]
    ton, ch = td[on], pd_[on]
    s0 = start.hour * 3600 + start.minute * 60 + start.second
    blk = (((s0 + ton) // 10800) % 8).astype(int)                 # 3-h clock block of each ON
    bad = spans_gap(ton, ton + np.nan_to_num(dur, nan=0.0))
    dur[bad] = np.nan
    gap[spans_gap(ton - np.nan_to_num(gap, nan=0.0), ton)] = np.nan
    nb = int(np.ceil(T / BIN_S))
    covb = np.bincount((t // BIN_S).astype(int), minlength=nb)[:nb] > 0
    dets = np.unique(ch)
    H, rows = {}, []
    for c in dets:
        m = ch == c
        x, d, g = ton[m], dur[m], gap[m]
        n = len(x)
        r = {"detector": int(c), "a_n_on": n}
        if n < min_on:
            rows.append(r)
            continue
        ioi = np.diff(x)
        ioi[spans_gap(x[:-1], x[1:])] = np.nan
        df, gf, jf = d[np.isfinite(d)], g[np.isfinite(g)], ioi[np.isfinite(ioi)]
        if len(df) < min_on:
            rows.append(r)
            continue
        r["pulse_frac"] = float((df <= 0.15).mean())
        r["med_dur"] = float(np.median(df))
        r["p90_dur"] = float(np.quantile(df, .9))
        gt, jt = _ticks(gf), _ticks(jf)                  # note 83: whole 0.1-s ticks
        r["gap_lt03"] = float((gt < CHAT_TICKS).mean()) if len(gf) else np.nan
        r["gap_lt1"] = float((gt < IOI_TICKS_1).mean()) if len(gf) else np.nan
        r["ioi_lt05"] = float((jt < IOI_TICKS_05).mean()) if len(jf) else np.nan
        r["ioi_lt1"] = float((jt < IOI_TICKS_1).mean()) if len(jf) else np.nan
        # bursts: >= 4 successive ON->ON intervals under 1 s (5 ONs inside ~4 s)
        k_, nr = _runs_of(np.nan_to_num(_ticks(ioi), nan=990.0) < IOI_TICKS_1, 4)
        r["burst_frac"], r["n_burst"] = (k_ + nr) / n, nr
        # note 47: implausibly short ONs (a vehicle crossing a 6-ft loop in <= 0.2 s = 70+ mph), overall
        # and in the worst 3-h clock block; the mode (pulse or not) is read from the MOST normal block, so
        # a detector whose ONs collapse only part of the day is still judged as non-pulse
        fin = np.isfinite(d)
        r["n_dur"] = int(fin.sum())
        r["short1"] = float((df < 0.15).mean())
        r["short2"] = float((df < SHORT_S).mean())
        bk, dk = blk[m][fin], d[fin]
        sb = [((dk[bk == b] < SHORT_S).mean(), b) for b in range(8) if (bk == b).sum() >= SHORT_BLK_MIN]
        if sb:
            r["short_blk_min"] = float(min(sb)[0])
            r["short_blk_max"], r["short_blk"] = float(max(sb)[0]), int(max(sb)[1])
        if light:
            rows.append(r)
            continue
        # rhythm: >= 4 successive identical intervals (0.1-s clock) between 0.3 and 30 s
        eq = (np.abs(np.diff(ioi)) < 0.05) & (ioi[1:] >= 0.3) & (ioi[1:] <= 30)
        k_, nr = _runs_of(np.nan_to_num(eq, nan=0).astype(bool), 4)
        r["rhythm_frac"], r["n_rhythm"] = (k_ + nr + nr) / n, nr
        # toggling: identical ON and identical OFF durations repeated (non-pulse ONs)
        dd = np.abs(np.diff(d)) < 0.05
        gg = np.abs(np.diff(g)) < 0.05
        tog = dd & gg & (d[1:] > 0.15) & (g[1:] < 30)
        k_, nr = _runs_of(tog, 3)
        r["toggle_frac"] = (k_ + nr) / n
        r["long_frac"] = float((df > 60).mean())
        r["long_time"] = float(df[(df > 120) & (df <= 900)].sum()) / max(cov_s, 1)
        # ONs 0-0.25 s after a colour change (crosstalk / controller-locked), vs chance
        if len(tc) > 10:
            k = np.searchsorted(tc, x, "left") - 1
            dt = np.where(k >= 0, x - tc[np.maximum(k, 0)], np.inf)
            lk = float(((dt > 0) & (dt <= 0.25)).mean())
            chance = len(tc) / max(cov_s, 1) * 0.25
            r["lock_frac"], r["lock_ratio"] = lk, lk / max(chance, 1e-4)
        # flow-occupancy, 5-min bins (ON's duration counted in its start bin, capped at 300 s)
        b = np.minimum((x // BIN_S).astype(int), nb - 1)
        nbin = np.bincount(b, minlength=nb).astype(float)
        obin = np.bincount(b, weights=np.clip(np.nan_to_num(d, nan=0.0), 0, 300), minlength=nb)
        el = covb & (nbin + obin > 0)
        if el.sum() >= 12 and nbin[el].std() > 0 and obin[el].std() > 0:
            r["fo_corr"] = float(pd.Series(nbin[el]).corr(pd.Series(obin[el]), method="spearman"))
        e3 = covb & (nbin >= 3)
        if e3.sum() >= 6:
            mb = obin[e3] / nbin[e3]
            md = np.median(mb)
            if md > 0:
                r["fo_hi"] = float((mb > 6 * md).mean())
                r["fo_lo"] = float((mb < md / 6).mean())
        r["sticky_bins"] = int((covb & (obin > 0.5 * BIN_S) & (nbin <= 2)).sum())
        # green-starting ONs of the best-matching phase, >= 5 s into green ("free flow")
        best, lift = None, 0.0
        for p, (s_, e_, frac) in greens.items():
            if frac < 0.02:
                continue
            k = np.searchsorted(s_, x, "right") - 1
            ing = (k >= 0) & (x < e_[np.maximum(k, 0)])
            lf = ing.mean() / frac
            if lf > lift:
                best, lift, ingb, kb = p, lf, ing, k
        r["green_lift"] = lift
        if best is not None:
            s_ = greens[best][0]
            free = ingb & (x - s_[np.maximum(kb, 0)] >= 5) & np.isfinite(d)
            r["n_free"] = int(free.sum())
            if free.sum() >= 30:
                fd = d[free]
                r["free_med_dur"] = float(np.median(fd))
                r["free_pulse_frac"] = float((fd <= 0.15).mean())
                fb = np.minimum((x[free] // (BIN_S * AGG)).astype(int), nb // AGG)
                nf = np.bincount(fb).astype(float)
                of = np.bincount(fb, weights=np.clip(fd, 0, 300))
                e3 = nf >= 3
                if e3.sum() >= 4:
                    mb = of[e3] / nf[e3]
                    md = np.median(mb)
                    if md > 0:
                        r["fog_dev"] = float(((mb > 4 * md) | (mb < md / 4)).mean())
        H[int(c)] = (np.histogram(df, LOGB)[0] / len(df),
                     np.histogram(jf, LOGB)[0] / max(len(jf), 1) if len(jf) else None)
        rows.append(r)
    if not rows:
        return None
    st = pd.DataFrame(rows)
    st["a_cov_h"] = cov_s / 3600
    if light:
        return st
    # distance to the most similar sibling (>= 50 ONs) in duration and interval histograms
    big = [k for k in H if st.loc[st.detector == k, "a_n_on"].iat[0] >= 50]
    nn = {}
    for k in H:
        best = (np.nan, np.nan)
        for j in big:
            if j == k or H[k][1] is None or H[j][1] is None:
                continue
            v = (_js(H[k][0], H[j][0]), _js(H[k][1], H[j][1]))
            if not np.isfinite(best[0]) or sum(v) < sum(best):
                best = v
        nn[k] = best
    st["nn_js_dur"] = st.detector.map(lambda k: nn.get(k, (np.nan, np.nan))[0])
    st["nn_js_ioi"] = st.detector.map(lambda k: nn.get(k, (np.nan, np.nan))[1])
    return st


# behaviour group from the log (pulse-mode or median ON duration) -> suspect limits of the
# `rapid` rule: share of ON->ON intervals < 0.5 s, < 1 s, share of ONs in bursts (>= 5 ONs with
# every interval < 1 s).  = p99.8 of 8,666 presumed-healthy detectors (>= 50 ONs, all windows).
RAPID_LIM = {"pulse": (0.180, 0.401, 0.070), "short": (0.129, 0.276, 0.037),
             "mid": (0.062, 0.145, 0.065), "long": (0.064, 0.161, 0.058),
             "vlong": (0.051, 0.110, 0.032)}
# note 46: a detector spanning 2+ lanes (lane output, note 42) sees side-by-side vehicles as quick
# successive ONs: its limits = max(the above, p99.8 of presumed-healthy spanning detectors)
RAPID_LIM_SPAN = {"pulse": (0.180, 0.401, 0.070), "short": (0.154, 0.347, 0.037),
                  "mid": (0.062, 0.165, 0.065), "long": (0.109, 0.237, 0.058),
                  "vlong": (0.051, 0.127, 0.032)}
RAPID_MIN_ON = 50
# note 83: the rapid statistics are now counted in whole ticks; on 135k presumed-healthy detector-windows only 133 of
# 197k ioi_lt1 values moved (max .02) and these limits keep every fire rate exactly -> unchanged
ACOLS = ["a_n_on", "pulse_frac", "med_dur", "ioi_lt05", "ioi_lt1", "burst_frac", "n_burst", "rapid",
         "n_dur", "short1", "short2", "short_blk_min", "short_blk_max", "short_blk"]


def behaviour_group(pulse_frac, med_dur):
    pf, md = np.asarray(pulse_frac, float), np.asarray(med_dur, float)
    g = np.select([pf > .5, md <= .5, md <= 1.5, md <= 4, md > 4],
                  ["pulse", "short", "mid", "long", "vlong"], default="")
    return g


def rapid_ratio(a: pd.DataFrame, span2=None) -> pd.Series:
    """max over (ioi_lt05, ioi_lt1, burst_frac) of value / the behaviour group's suspect limit;
    NaN below RAPID_MIN_ON actuations.  >= 1 = suspect, >= 2 = bad.  `span2`: optional bool per
    row, True = the detector spans 2+ lanes (note 46: RAPID_LIM_SPAN)."""
    g = behaviour_group(a["pulse_frac"], a["med_dur"])
    sp = np.zeros(len(a), bool) if span2 is None else np.asarray(span2, bool)
    lim = np.array([(RAPID_LIM_SPAN if s else RAPID_LIM).get(k, (np.nan,) * 3) for k, s in zip(g, sp)],
                   float).reshape(-1, 3)
    v = a[["ioi_lt05", "ioi_lt1", "burst_frac"]].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.nanmax(np.where(np.isfinite(v), v / lim, np.nan), axis=1) if len(a) else np.array([])
    r = np.where(a["a_n_on"].to_numpy(float) >= RAPID_MIN_ON, r, np.nan)
    return pd.Series(r, index=a.index)


# ===================================================== v3 (note 43): shape against the same-phase detectors
# Optional inputs: `phase` = {detector: predicted phase} and `function` = {detector: predicted
# function} from the phase / function model (never a wiring table).  Without them the reference of
# a detector is the rest of the signal.
H_REF = 2.0                       # hours either side for the local share in rel_disp
SHAPE_COLS = ("chop5", "nb5", "chop15", "nb15", "chop15_self", "chop15_sig", "corr_ph", "corr_ph_exp",
              "beta15", "chop30", "nb30")


def _movsum(a, h):
    """centred moving sum over +-h along the last axis (edges truncated)."""
    n = a.shape[-1]
    c = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, -1)], -1)
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return c[..., hi] - c[..., lo]


def rel_disp(x, r, ok, h):
    """Pearson dispersion of counts x around a locally scaled reference: e_t = s_t r_t where s_t is
    x's share of r in the +-h neighbourhood WITHOUT bin t.  Poisson noise on both gives ~1; spikes
    that the reference does not share push it up.  r=None: around x's own local mean.
    Returns (D, bins used)."""
    x = np.where(ok, x, 0.0).astype(float)
    okf = ok.astype(float)
    X = _movsum(x, h) - x
    if r is None:
        N = _movsum(okf, h) - okf
        e = X / np.maximum(N, 1)
        v = e * (1 + 1 / np.maximum(N, 1))
        m = ok & (N >= 2)
    else:
        r = np.where(ok, r, 0.0).astype(float)
        R = _movsum(r, h) - r
        s = X / np.maximum(R, 1e-9)
        e, v = s * r, s * r * (1 + s)
        m = ok & (R > 0) & (X + x > 0)
    m &= (e + x) > 0
    if m.sum() < 6:
        return np.nan, int(m.sum())
    return float(((x - e) ** 2)[m].sum() / max(v[m].sum(), 1e-9)), int(m.sum())


def cycle_counts(ev: pd.DataFrame, start, end, phase: dict) -> dict:
    """{phase: (cycle start seconds, {detector: ONs per cycle})}; a cycle = begin green to the next
    begin green of the phase; cycles that contain a comms gap (> GAP_S) are dropped."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    e = ev[["Timestamp", "EventId", "Parameter"]]
    e = e[(e.Timestamp >= start) & (e.Timestamp < end) & e.EventId.isin(ALLOWED)].drop_duplicates()
    t = (e.Timestamp - start).dt.total_seconds().to_numpy()
    eid, par = e.EventId.to_numpy().astype(int), e.Parameter.to_numpy().astype(int)
    ut = np.unique(t)
    gi = np.diff(ut) > GAP_S
    g0 = ut[:-1][gi]
    out, byp = {}, {}
    for d, p in phase.items():
        if p is not None and np.isfinite(p):
            byp.setdefault(int(p), []).append(int(d))
    for p, ds_ in byp.items():
        gs = np.sort(t[(eid == 1) & (par == p)])
        if len(gs) < 8:
            continue
        a, b = gs[:-1], gs[1:]
        ok = np.ones(len(a), bool)
        if len(g0):
            k = np.searchsorted(g0, a)
            ok = ~((k < len(g0)) & (g0[np.minimum(k, len(g0) - 1)] < b))
        cnt = {}
        for d in ds_:
            x = np.sort(t[(eid == 82) & (par == d)])
            c = np.searchsorted(x, b) - np.searchsorted(x, a)
            cnt[d] = c[ok].astype(float)
        out[p] = (a[ok], cnt)
    return out


def twins(n, cov, tol=0.1):
    """pairs of channels carrying (almost) the same actuations = one zone wired to two inputs:
    5-min counts differ by < tol of their sum.  Returns a bool matrix."""
    x = n[:, cov]
    tot = x.sum(1)
    tw = np.zeros((len(n), len(n)), bool)
    for i in np.where(tot > 0)[0]:
        dd = np.abs(x - x[i]).sum(1) / np.maximum(tot + tot[i], 1)
        tw[i] = (dd < tol) & (tot > 0)
    np.fill_diagonal(tw, False)
    return tw


def _refs(dets, live, phase, tw):
    """reference of each detector: the other live detectors on its (predicted) phase when there are
    >= 2 of them, else the rest of the signal's live detectors; a twin (same zone on another input)
    is never a reference."""
    ph = np.array([phase.get(int(d), np.nan) if phase else np.nan for d in dets], float)
    idx = np.arange(len(dets))
    ref, kind = [], []
    for i in range(len(dets)):
        ok = live & (idx != i) & ~tw[i]
        same = np.where(ok & (ph == ph[i]))[0] if np.isfinite(ph[i]) else []
        if len(same) >= 2:              # one sibling is too fragile: if it fails, this one looks bad
            ref.append(same)
            kind.append("phase")
        else:
            ref.append(np.where(ok)[0])
            kind.append("signal")
    return ref, kind, ph


def shape_stats(B: dict, phase: dict | None = None, cyc: dict | None = None) -> pd.DataFrame:
    """Per-detector choppiness / traffic-following statistics against the same-phase reference."""
    n, cov = B["n_on"].astype(float), B["cov"]
    dets = B["dets"]
    live = n[:, cov].sum(1) > 0
    tw = twins(n, cov)
    ref, kind, ph = _refs(dets, live, phase, tw)
    idx = np.arange(len(dets))
    bph = 3600 / B["bin_s"]
    rows = []
    agg = {k: (_agg(n, k), _agg(cov[None].astype(float), k)[0] == k) for k in (1, 3, 6)}
    for i, d in enumerate(dets):
        r = {"detector": int(d), "ref_kind": kind[i], "n_ref": len(ref[i]), "pred_phase": ph[i],
             "n_twin": int(tw[i].sum())}
        if n[i, cov].sum() < 30 or not len(ref[i]):
            rows.append(r)
            continue
        for k, nm in ((1, "5"), (3, "15"), (6, "30")):
            a, ok = agg[k]
            h = max(int(round(H_REF * bph / k)), 1)
            r[f"chop{nm}"], r[f"nb{nm}"] = rel_disp(a[i], a[ref[i]].sum(0), ok, h)
            if nm == "15":
                r["chop15_self"], _ = rel_disp(a[i], None, ok, h)
                sig = a[np.where(live & (idx != i) & ~tw[i])[0]].sum(0)
                r["chop15_sig"], _ = rel_disp(a[i], sig, ok, h)
                x, y = a[i][ok], a[ref[i]].sum(0)[ok]
                if len(x) >= 8 and y.std() > 0 and y.sum() > 0:
                    E = y * x.sum() / y.sum()
                    r["corr_ph"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
                    r["corr_ph_exp"] = float(np.sqrt(E.var() / (E.var() + E.mean()))) if E.mean() > 0 else np.nan
                    lx, ly = np.log(x + 0.5), np.log(y + 0.5)
                    if len(x) >= 16 and ly.std() > 0.3:
                        r["beta15"] = float(np.polyfit(ly, lx, 1)[0])
        rows.append(r)
    st = pd.DataFrame(rows)
    for c in SHAPE_COLS:
        if c not in st:
            st[c] = np.nan
    # cycle level: ONs per green-to-green cycle of the detector's predicted phase
    if cyc:
        cc = {}
        for p, (t0, cnt) in cyc.items():
            ks = [k for k in cnt if cnt[k].sum() > 0]
            nc = len(t0)
            h = max(int(round(H_REF * 3600 / max(np.median(np.diff(t0)) if nc > 1 else 120, 30))), 2)
            for k in cnt:
                others = [j for j in ks if j != k and not np.array_equal(cnt[j], cnt[k])
                          and np.abs(cnt[j] - cnt[k]).sum() >= 0.1 * (cnt[j] + cnt[k]).sum()]
                if cnt[k].sum() >= 30:
                    cc[k] = rel_disp(cnt[k], np.sum([cnt[j] for j in others], 0) if others else None,
                                     np.ones(nc, bool), h)[0]
        st["chop_cyc"] = st.detector.map(cc)
    # the same statistic on the reference detectors: does the whole phase spike, or only this one?
    for c in ("chop15", "chop5"):
        if c in st:
            v = st[c].to_numpy(float)
            st[c + "_ref"] = [np.nanmedian(v[ref[i]]) if kind[i] == "phase" and np.isfinite(v[ref[i]]).any()
                              else np.nan for i in range(len(dets))]
    return st


def on_episodes(ev: pd.DataFrame, start, end, min_s: float = 900.0) -> pd.DataFrame:
    """Continuous ONs held >= min_s (stuck-on), with start / end.  A continuous ON runs from the ON that
    starts it to the next OFF; further ONs inside it (no OFF between) are normal extension re-calls and
    do not end it (note 47).  `n_on` = ONs logged inside it.  An ON over a comms gap is not counted."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    T = (end - start).total_seconds()
    e = ev[["Timestamp", "EventId", "Parameter"]]
    e = e[(e.Timestamp >= start) & (e.Timestamp < end) & e.EventId.isin(ALLOWED)]
    e = e[~(e.EventId.isin((81, 82)) & (e.Parameter > MAXCH))].drop_duplicates()
    t = (e.Timestamp - start).dt.total_seconds().to_numpy()
    eid, par = e.EventId.to_numpy().astype(int), e.Parameter.to_numpy().astype(int)
    ut = np.unique(t)
    gi = np.diff(ut) > GAP_S
    g0 = ut[:-1][gi]
    dm = np.isin(eid, (81, 82))
    td, ed, pd_ = t[dm], eid[dm], par[dm]
    o = np.lexsort((ed != 82, td, pd_))
    td, ed, pd_ = td[o], ed[o], pd_[o]
    end_t, start_ = continuous_on(td, ed, pd_, T)
    oe_all = end_t >= T                                   # still ON when the window closed
    # ONs inside each continuous ON: ON count between its start and its OFF on the same channel
    ci = np.cumsum(ed == 82)
    offp = np.searchsorted(td + pd_ * 1e7, end_t + pd_ * 1e7, "left")      # position of the ending OFF
    nin = ci[np.minimum(offp, len(td)) - 1] - ci + 1 if len(td) else np.zeros(0, int)
    m = start_ & (end_t - td >= min(CO_MIN_S, min_s))
    s, f, c, nf, oe = td[m], end_t[m], pd_[m], nin[m], oe_all[m]
    if len(g0) and len(s):
        k = np.searchsorted(g0, s)
        hit = (k < len(g0)) & (g0[np.minimum(k, len(g0) - 1)] < f)
        s, f, c, nf, oe = s[~hit], f[~hit], c[~hit], nf[~hit], oe[~hit]
    # other detectors held ON (>= 5 min) through at least half of the same time: a shared event
    ov = np.minimum(f[:, None], f[None]) - np.maximum(s[:, None], s[None])
    co = ((ov >= 0.5 * (f - s)[:, None]) & (c[:, None] != c[None]))
    co = np.array([len(set(c[row].tolist())) for row in co], dtype=int)
    k = (f - s) >= min_s
    return pd.DataFrame({"detector": c[k].astype(int), "t0": start + pd.to_timedelta(s[k], unit="s"),
                         "t1": start + pd.to_timedelta(f[k], unit="s"), "dur_s": (f - s)[k], "n_on": nf[k],
                         "open_end": oe[k], "co_stuck": co[k]})


def _chop_lim(kind: str, hours: float) -> float:
    """suspect limit of chop15 for a window of `hours` (log-interpolated between the tabled lengths)."""
    tab = CHOP_LIM[kind]
    h = np.log(np.clip(hours, tab[0][0], tab[-1][0]))
    return float(np.interp(h, np.log([x for x, _ in tab]), [y for _, y in tab]))


def _recovery(B: dict, i: int, a: int, b: int, ref) -> float:
    """share of the reference in the 6 h after bins [a, b) / in the 6 h before; NaN when either
    side has < 3 h of data."""
    n, cov = B["n_on"], B["cov"]
    nb = n.shape[1]

    def sh(lo, hi):
        c = cov[lo:hi]
        x = n[i, lo:hi][c].sum()
        S = (n[:, lo:hi].sum(0) - n[i, lo:hi] if ref is None else n[ref, lo:hi].sum(0))[c].sum()
        return (x + 0.5) / max(S, 1.0), int(c.sum())

    if a < REC_MIN_BINS or nb - b < REC_MIN_BINS:
        return np.nan
    s0, c0 = sh(max(a - REC_BINS, 0), a)
    s1, c1 = sh(b, min(b + REC_BINS, nb))
    return s1 / s0 if c0 >= REC_MIN_BINS and c1 >= REC_MIN_BINS else np.nan


# ============================================================== v4 (note 46): the user's review ideas
# Every new check is optional (OPTS) and gated by how much data the sample holds; what could not run
# is listed in `not_checked` with the amount it needs.
OPTS = dict(lanes=True,        # (e) lane-aware rapid limits (needs `lanes`)
            partner=True,      # (b) recovery judged against the sibling that tracked the detector before
            recur=True,        # (a) choppy spikes: recurrence by time of day + are the extra ONs vehicles?
            group=True,        # (c) shared stuck-on: did the rest of the phase keep counting?
            rapid_hour=False,  # (d) rapid re-actuation inside the worst hour - tested, OFF: +0-2 catches vs loosening
            conf=False)        # (g) model confidence weighs the phase / Bike exemptions - tested, OFF: no gain
#                                  (the "classifier is unsure" note in the reason is always given)
SPIKE_Z, SPIKE_MIN = 3.0, 5.0     # a 15-min spike: z >= 3 above the local-share expectation and >= 5 extra ONs
RECUR_TOL = 2                     # same time of day = within +-2 15-min slots (+-30 min)
RECUR_DAYS = (2, 4)               # recurrence judged from 2 days (weak evidence), firm from 4 days
RECUR_SHARE = 0.6                 # >= 60 % of the spike excess recurs at the same time on another day
SPIKE_MIN_ON = 30                 # actuations inside the spikes needed to judge them
PARTNER_CORR = 0.7                # recovery partner: best 15-min correlation with a same-phase sibling
FLOW_OK = 0.5                     # a same-phase sibling counted >= 50 % of its expected share in the episode
RAPID_H_MIN_ON = 50               # ONs in an hour to judge its rapid share
RAPID_H_LIM = (2.0, 4.0)          # worst hour: share of ON->ON < 1 s / the group limit; suspect / bad
CONF_PHASE_MIN = 0.5              # phase reference only when the phase model is at least this sure
CONF_FUNC_LOW = 0.5               # below this the reason says the function is uncertain
GROUP_LT1 = 1                     # index of ioi_lt1 in RAPID_LIM tuples
# ---- note 47 (the user's 2026-09-29 review)
OPTS.update(min_on=False,         # (b) implausibly short ONs on a non-pulse detector - tested, OFF as a score (+0.5 pt
            #                     FA at 66 h, no new catch); above its limit it is still reported as a note
            night_drop=True,      # (b') stops counting at night while its partner / phase does not
            grade2=False)         # (c) two independent suspect findings = bad
SHORT_S = 0.25                    # "short" ON: <= 0.2 s (<= 2 ticks of the 0.1-s log clock)
SHORT_BLK_MIN = 30                # ONs with a duration in a 3-h clock block to judge it
SHORT_MIN_N = 50                  # ONs with a duration to judge the detector at all
PULSE_MODE = 0.7                  # pulse mode: >= 70 % short ONs even in its most normal 3-h block
SHORT_LIM = (0.44, 0.53)          # suspect limits: short share overall / in the worst 3-h block (p99.5 of presumed-
#                                   healthy non-pulse detectors, 24-h and 66-h windows pooled); bad at 2x
NIGHT_H = (21, 5)                 # night for the night-drop check: 21:00-05:00
NIGHT_MIN_EXP = 30.0              # night ONs expected from the reference before the check runs
NIGHT_DROP_LIM = (0.19, 0.063)    # night count / expected: suspect = p0.5 of presumed-healthy (24 h + 66 h), bad = 1/3 of it
LOOP_FT = {"Advance": 6, "Count": 6, "Yellow_Red": 6, "Presence": 20}   # typical zone length by function
CAR_FT = 15


def on_arrays(ev: pd.DataFrame, start, end) -> dict:
    """{detector: (ON times s from start, ON durations s (NaN unknown), ON->ON intervals (NaN over a
    comms gap))} - the per-actuation view used by the v4 checks."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    e = ev[["Timestamp", "EventId", "Parameter"]]
    e = e[(e.Timestamp >= start) & (e.Timestamp < end) & e.EventId.isin(ALLOWED)]
    e = e[~(e.EventId.isin((81, 82)) & (e.Parameter > MAXCH))].drop_duplicates()
    t = (e.Timestamp - start).dt.total_seconds().to_numpy()
    eid, par = e.EventId.to_numpy().astype(int), e.Parameter.to_numpy().astype(int)
    ut = np.unique(t)
    gi = np.diff(ut) > GAP_S
    g0 = ut[:-1][gi]
    dm = np.isin(eid, (81, 82))
    td, ed, pd_ = t[dm], eid[dm], par[dm]
    o = np.lexsort((ed != 82, td, pd_))
    td, ed, pd_ = td[o], ed[o], pd_[o]
    same_next = np.r_[pd_[1:] == pd_[:-1], False]
    nxt_t, nxt_e = np.r_[td[1:], np.nan], np.r_[ed[1:], 0]
    on = ed == 82
    dur = np.where(same_next & (nxt_e == 81), nxt_t - td, np.nan)[on]
    ton, ch = td[on], pd_[on]
    out = {}
    for c in np.unique(ch):
        m = ch == c
        x, d = ton[m], dur[m]
        ioi = np.r_[np.nan, np.diff(x)]
        if len(g0) and len(x) > 1:
            k = np.searchsorted(g0, x[:-1])
            hit = (k < len(g0)) & (g0[np.minimum(k, len(g0) - 1)] < x[1:])
            ioi[1:][hit] = np.nan
        out[int(c)] = (x, d, ioi)
    return out


def _zres(x, r, ok, h):
    """expected counts e and standardised residuals z of x around the local share of reference r
    (the rel_disp model, per bin)."""
    x, r = np.where(ok, x, 0.0).astype(float), np.where(ok, r, 0.0).astype(float)
    X, R = _movsum(x, h) - x, _movsum(r, h) - r
    s = X / np.maximum(R, 1e-9)
    e = s * r
    z = (x - e) / np.sqrt(np.maximum(e * (1 + s), 1.0))
    return e, np.where(ok & (R > 0), z, np.nan)


def _grp_lim(pulse_frac, med_dur, span2: bool):
    g = str(behaviour_group(pulse_frac, med_dur)) if np.isfinite(med_dur) else ""
    lim = (RAPID_LIM_SPAN if span2 else RAPID_LIM).get(g)
    return lim[GROUP_LT1] if lim else np.nan


def spike_stats(B: dict, i: int, ref_rows, on: tuple | None, lt1_lim: float) -> dict:
    """(a)/(d): where does the detector's 15-min excess over its reference sit, does it recur at the
    same time of day on other days, and do the actuations inside the spikes look like vehicles?"""
    out = {}
    n15 = _agg(B["n_on"].astype(float))
    ok = _agg(B["cov"][None].astype(float))[0] == AGG
    x, r = n15[i], n15[ref_rows].sum(0)
    h = max(int(round(H_REF * 3600 / (B["bin_s"] * AGG))), 1)
    e, z = _zres(x, r, ok, h)
    spk = ok & np.isfinite(z) & (z >= SPIKE_Z) & (x - e >= SPIKE_MIN)
    t15 = B["start"] + pd.to_timedelta(np.arange(len(x)) * B["bin_s"] * AGG, unit="s")
    date = t15.normalize().to_numpy()
    slot = (t15.hour * 4 + t15.minute // 15).to_numpy()
    out["n_days"] = round(float(ok.sum()) * B["bin_s"] * AGG / 86400.0, 2)     # days of data in the sample
    out["spk_n"] = int(spk.sum())
    exc = np.where(spk, x - e, 0.0)
    out["spk_excess"] = float(exc.sum())
    out["spk_share"] = float(exc.sum() / max(np.clip(np.where(ok, x - e, 0), 0, None).sum(), 1e-9))
    out["spk_periods"] = [(int(a) * AGG, int(b) * AGG) for a, b in zip(*_runs(spk))]
    if spk.sum():
        si = np.where(spk)[0]
        rec = np.zeros(len(si), bool)
        ndates = []
        for j, k in enumerate(si):
            dd = np.abs(slot[si] - slot[k])
            dd = np.minimum(dd, 96 - dd)
            other = (dd <= RECUR_TOL) & (date[si] != date[k])
            rec[j] = other.any()
            ndates.append(len(set(date[si][dd <= RECUR_TOL].tolist())))
        out["rec_share"] = float(exc[si][rec].sum() / max(exc[si].sum(), 1e-9))
        out["rec_ndates"] = int(max(ndates))
        wk = pd.DatetimeIndex(t15[si]).dayofweek >= 5
        out["spk_weekend"] = float(exc[si][wk].sum() / max(exc[si].sum(), 1e-9))
        u = np.unique(slot[si][rec] if rec.any() else slot[si])
        runs, a_ = [], u[0]
        for p_, q_ in zip(u[:-1], u[1:]):
            if q_ - p_ > 1:
                runs.append((a_, p_))
                a_ = q_
        runs.append((a_, u[-1]))
        hm = lambda k: f"{k // 4:02d}:{(k % 4) * 15:02d}"  # noqa: E731
        out["spk_tod"] = ", ".join(f"{hm(a_)}-{hm(b_ + 1) if b_ < 95 else '24:00'}" for a_, b_ in runs[:4])
    # actuations inside the spikes vs outside
    if on is not None and spk.sum():
        tt, dur, ioi = on
        b15 = np.minimum((tt // (B["bin_s"] * AGG)).astype(int), len(x) - 1)
        ins = spk[b15]
        if ins.sum() >= SPIKE_MIN_ON:
            iv = ioi[ins & np.isfinite(ioi)]
            out["spk_ioi_lt1"] = float((_ticks(iv) < IOI_TICKS_1).mean()) if len(iv) else np.nan
            di, do = dur[ins & np.isfinite(dur)], dur[~ins & np.isfinite(dur)]
            out["spk_dur_ratio"] = (float(np.median(di) / max(np.median(do), 0.05))
                                    if len(di) >= 10 and len(do) >= 10 else np.nan)
            fast = out["spk_ioi_lt1"] / lt1_lim if np.isfinite(lt1_lim) and lt1_lim > 0 else np.nan
            out["spk_fast"] = fast
            dr = out["spk_dur_ratio"]
            out["spk_plaus"] = float(bool((not np.isfinite(fast) or fast < 1.0)
                                          and (not np.isfinite(dr) or 1 / 3 <= dr <= 3)))
    return out


def rapid_hour(on: tuple, lt1_lim: float) -> tuple[float, float]:
    """(d): worst clock hour's share of ON->ON < 1 s over the behaviour group's limit (hours with
    >= RAPID_H_MIN_ON actuations), and that hour's start (s from the window start)."""
    tt, _, ioi = on
    if not np.isfinite(lt1_lim) or len(tt) < RAPID_H_MIN_ON:
        return np.nan, np.nan
    hb = (tt // 3600).astype(int)
    m = np.isfinite(ioi)
    n = np.bincount(hb[m])
    f = np.bincount(hb[m], weights=(_ticks(ioi[m]) < IOI_TICKS_1).astype(float), minlength=len(n))
    okh = n >= RAPID_H_MIN_ON
    if not okh.any():
        return np.nan, np.nan
    sh = np.where(okh, f / np.maximum(n, 1), -1)
    k = int(sh.argmax())
    return float(sh[k] / lt1_lim), float(k * 3600)


def _partner(B: dict, i: int, cand, a: int, b: int):
    """(b): the same-phase sibling whose 15-min counts tracked detector i best outside bins [a, b)
    (corr >= PARTNER_CORR), as a row index list, else None."""
    if cand is None or not len(cand):
        return None, np.nan
    n15 = _agg(B["n_on"].astype(float))
    ok = _agg(B["cov"][None].astype(float))[0] == AGG
    ok[a // AGG: -(-b // AGG)] = False
    if ok.sum() < 16:
        return None, np.nan
    best, bc, bk = None, -1.0, None
    w = np.arange(1, int(ok.sum()) + 1, dtype=float)
    for j in cand:
        y, x = n15[j][ok], n15[i][ok]
        if y.std() > 0 and x.std() > 0:
            c = float(np.corrcoef(x, y)[0, 1])
            # note 77: equal correlations are broken by the sibling's own counts, never by its channel order
            k = (round(c, 12), float(y.sum()), float((y * w).sum()))
            if bk is None or k > bk:
                best, bc, bk = j, c, k
    return ([best], bc) if best is not None and bc >= PARTNER_CORR else (None, bc)


def _flow(B: dict, rows, a: int, b: int) -> float:
    """(c): best ratio, over `rows` (same-phase siblings not held ON), of their ONs in bins [a, b)
    to the number expected from their share of the rest of the signal outside it; NaN when none
    was expected to count >= 10."""
    n, cov = B["n_on"].astype(float), B["cov"]
    tot = n.sum(0)
    out = np.nan
    ins = np.zeros(n.shape[1], bool)
    ins[a:b] = True
    for j in rows:
        S = tot - n[j]
        lam = n[j][~ins & cov].sum() / max(S[~ins & cov].sum(), 1) * S[ins & cov].sum()
        if lam >= 10:
            v = n[j][ins & cov].sum() / lam
            out = v if not np.isfinite(out) else max(out, v)
    return out


def _probs(function: dict | None) -> dict:
    """{detector: {class: prob}} where probabilities were given."""
    return {int(d): v for d, v in (function or {}).items() if isinstance(v, dict) and v}


def _labels(function: dict | None) -> dict:
    """{detector: label} from labels or from class-probability dicts (argmax)."""
    out = {}
    for d, v in (function or {}).items():
        if isinstance(v, dict) and v:
            out[int(d)] = max(v, key=v.get)
        elif isinstance(v, str):
            out[int(d)] = v
    return out


def health(events_df: pd.DataFrame, start, end, detectors=None, phase: dict | None = None,
           function: dict | None = None, lanes: dict | None = None,
           phase_conf: dict | None = None) -> pd.DataFrame:
    """Per-detector health table for one signal over [start, end).  Columns: detector,
    health_score, status, reason, not_checked, bad_periods (list of {start, end, what,
    recovered}), the rule scores s_* and the statistics behind them.
    v4 (note 46) optional inputs: `lanes` = {detector: lanes spanned} (lane output, note 42);
    `phase_conf` = {detector: probability of its predicted phase}; `function` may carry class
    probabilities (dicts) - used as the model's confidence."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    B = events_to_bins(events_df, start, end, None)       # note 83: the channel list is ignored (no dead check)
    if len(B["dets"]) == 0:
        return pd.DataFrame(columns=["detector", "health_score", "status", "reason", "bad_periods"])
    dets = B["dets"]
    n, cov = B["n_on"].astype(float), B["cov"]
    live = n[:, cov].sum(1) > 0
    phase = {int(k): v for k, v in (phase or {}).items() if v is not None and np.isfinite(v)}
    if OPTS["conf"] and phase_conf:                # (g) an unsure phase is no yardstick
        phase = {k: v for k, v in phase.items() if not (phase_conf.get(k, 1.0) < CONF_PHASE_MIN)}
    ref, kind, _ = _refs(dets, live, phase, twins(n, cov))
    # silence is expected from the other detectors on the predicted phase when there are >= 2
    ph2 = [i for i in range(len(dets)) if kind[i] == "phase" and len(ref[i]) >= 2]
    st = det_stats(B)
    st["sig_on"] = st["sib_on"]
    st["drop_ref"] = "signal"
    if ph2:
        sp = det_stats(B, [ref[i] if i in ph2 else np.where(np.arange(len(dets)) != i)[0]
                           for i in range(len(dets))])
        cols = ["drop_lam", "drop_b0", "drop_b1", "drop_to_end", "co_silent"]
        st.loc[ph2, cols] = sp.loc[ph2, cols].to_numpy()
        st.loc[ph2, "drop_ref"] = "phase"
    sh = shape_stats(B, phase or None, None)
    hours = cov.sum() * B["bin_s"] / 3600
    sh["chop_lim"] = [_chop_lim(k if k == "phase" else "signal", hours) for k in sh.ref_kind]
    st = st.merge(sh.drop(columns=[c for c in ("pred_phase",) if c in sh]), on="detector", how="left")
    fl = _labels(function)
    pr = _probs(function)
    st["pred_function"] = st.detector.map(fl)
    st["pred_phase"] = st.detector.map(phase) if phase else np.nan
    st["p_bike"] = st.detector.map({d: float(v.get("Bike", 0.0)) for d, v in pr.items()})
    st["f_conf"] = st.detector.map({d: float(max(v.values())) for d, v in pr.items()})
    st["ph_conf"] = st.detector.map({int(k): float(v) for k, v in (phase_conf or {}).items()})
    span = {int(k): float(v) for k, v in (lanes or {}).items() if v is not None and np.isfinite(v)}
    st["lanes_spanned"] = st.detector.map(span)
    span2 = (st.lanes_spanned.fillna(0).ge(2) & OPTS["lanes"]).to_numpy(bool)
    a = act_stats(events_df, start, end, light=True)
    if a is not None and len(a):
        # note 56: a short / quiet window where no detector reaches min_on has no pulse_frac / med_dur columns
        a = a.reindex(columns=list(a.columns) + [c for c in ACOLS if c not in a and c != "rapid"])
        a["rapid"] = rapid_ratio(a, a.detector.map(dict(zip(st.detector, span2))).fillna(False).to_numpy(bool))
        a = a.reindex(columns=list(a.columns) + [c for c in ACOLS if c not in a])
        st = st.merge(a[["detector"] + ACOLS], on="detector", how="left")
    else:
        for c in ACOLS:
            st[c] = np.nan
    idx = {int(d): i for i, d in enumerate(dets)}
    # ---- v4: per-actuation view, worst-hour rapid share, spikes (recurrence + plausibility)
    on = on_arrays(events_df, start, end)
    lt1 = {int(d): _grp_lim(pf, md, bool(s2))
           for d, pf, md, s2 in zip(st.detector, st.pulse_frac.astype(float), st.med_dur.astype(float), span2)}
    spk_per, rh_per = {}, {}
    new = {}
    for j, r in st.iterrows():
        d = int(r.detector)
        i = idx[d]
        if OPTS["rapid_hour"] and d in on and np.isfinite(r.a_n_on) and r.a_n_on >= RAPID_MIN_ON:
            v, t0 = rapid_hour(on[d], lt1[d])
            new.setdefault("rapid_h", {})[j], new.setdefault("rapid_h_t", {})[j] = v, t0
            if np.isfinite(t0):
                rh_per[d] = (int(t0 // B["bin_s"]), int((t0 + 3600) // B["bin_s"]))
        ch, cl = r.get("chop15", np.nan), r.get("chop_lim", np.nan)
        if OPTS["recur"] and np.isfinite(ch) and np.isfinite(cl) and ch >= 0.5 * cl and len(ref[i]):
            sp_ = spike_stats(B, i, ref[i], on.get(d), lt1[d])
            spk_per[d] = sp_.pop("spk_periods", [])
            for k, v in sp_.items():
                new.setdefault(k, {})[j] = v
    for c in ("rapid_h", "rapid_h_t", "n_days", "spk_n", "spk_excess", "spk_share", "rec_share", "rec_ndates",
              "spk_weekend", "spk_tod", "spk_ioi_lt1", "spk_dur_ratio", "spk_fast", "spk_plaus"):
        st[c] = pd.Series(new.get(c, {}), dtype=object if c == "spk_tod" else float).reindex(st.index)
    # ---- episodes: stuck-on (incl. ON with no OFF) and silent runs -> bad periods + recovery
    rref = [ref[i] if kind[i] == "phase" and len(ref[i]) >= 2 else None for i in range(len(dets))]
    # recovery partner candidates: every other live detector on the predicted phase, twins included
    # (a same-lane loop or the same zone on another input is the best yardstick for "back to normal")
    phv = np.array([phase.get(int(d), np.nan) for d in dets], float)
    pcand = [list(np.where(live & (np.arange(len(dets)) != i) & (phv == phv[i]))[0]) if np.isfinite(phv[i]) else []
             for i in range(len(dets))]
    # ---- note 47 (b'): night count against the detector it tracks (best 15-min correlation >= .7 with a
    # same-phase sibling, twins included), else the other detectors on its phase (>= 2), else the signal
    if OPTS["night_drop"]:
        hr = B["hour"]
        nt = cov & ((hr >= NIGHT_H[0]) | (hr < NIGHT_H[1]))
        dy = cov & (hr >= DAY[0]) & (hr < DAY[1])
        nd = {c: {} for c in ("night_ratio", "night_n", "night_exp", "night_ref", "night_med_dur")}
        if nt.sum() >= 24 and dy.sum() >= 24:
            for j, r in st.iterrows():
                d = int(r.detector)
                i = idx[d]
                if n[i, dy].sum() < 20:
                    continue
                pref, _ = _partner(B, i, pcand[i], 0, 0)
                if pref is not None:
                    R, who = n[pref].sum(0), f"d{int(dets[pref[0]])}, which it tracks"
                elif rref[i] is not None:
                    R, who = n[rref[i]].sum(0), "the other detectors on its phase"
                else:
                    R, who = n.sum(0) - n[i], "the rest of the signal"
                exp_n = n[i, dy].sum() / max(R[dy].sum(), 1) * R[nt].sum()
                if exp_n < NIGHT_MIN_EXP:
                    continue
                xn = n[i, nt].sum()
                nd["night_ratio"][j], nd["night_n"][j], nd["night_exp"][j] = (xn + .5) / (exp_n + .5), xn, exp_n
                nd["night_ref"][j] = who
                if d in on:
                    tt, du, _ = on[d]
                    h_ = (start + pd.to_timedelta(tt, unit="s")).hour
                    mk = ((h_ >= NIGHT_H[0]) | (h_ < NIGHT_H[1])) & np.isfinite(du)
                    if mk.sum() >= 5:
                        nd["night_med_dur"][j] = float(np.median(du[mk]))
        for c, v in nd.items():
            st[c] = pd.Series(v, dtype=object if c == "night_ref" else float).reindex(st.index)
    per = {int(d): [] for d in dets}
    ep = on_episodes(events_df, start, end)
    S_all = n.sum(0)
    nb = n.shape[1]
    best = {}
    held = {}
    for r in ep.itertuples():
        held.setdefault(int(r.detector), []).append((r.t0, r.t1))

    def tod(t):
        return t.hour * 60 + t.minute

    for r in ep.itertuples():
        i = idx.get(int(r.detector))
        if i is None:
            continue
        a0 = int((r.t0 - start).total_seconds() // B["bin_s"])
        b0 = min(int(np.ceil((r.t1 - start).total_seconds() / B["bin_s"])), nb)
        x, S = n[i], S_all - n[i]
        outm = np.ones(nb, bool)
        outm[a0:b0] = False
        lam = x[outm & cov].sum() / max(S[outm & cov].sum(), 1) * S[a0:b0][cov[a0:b0]].sum()
        if STUCK_LAM is not None and lam < STUCK_LAM:
            continue                                  # held ON while the signal was quiet: not implausible
        pref, pc = (_partner(B, i, [k for k in pcand[i] if not any(t0 < r.t1 and t1 > r.t0 for t0, t1
                                                                  in held.get(int(dets[k]), []))], a0, b0)
                    if OPTS["partner"] else (None, np.nan))
        rec = np.nan if r.open_end else _recovery(B, i, a0, b0, pref if pref is not None else rref[i])
        flow = np.nan
        if OPTS["group"] and r.co_stuck >= CO_STUCK_N and kind[i] == "phase":
            free = [k for k in ref[i] if not any(t0 < r.t1 and t1 > r.t0 for t0, t1 in held.get(int(dets[k]), []))]
            flow = _flow(B, free, a0, b0)
        # the same detector held ON at about the same clock time on other days (a daily pattern)
        rec_days = len({t0.normalize() for t0, _ in held.get(int(r.detector), [])
                        if min(abs(tod(t0) - tod(r.t0)), 1440 - abs(tod(t0) - tod(r.t0))) <= 60})
        per[int(r.detector)].append(dict(start=r.t0, end=r.t1, what="stuck on",
                                         recovered=None if not np.isfinite(rec) else bool(REC_RATIO[0] <= rec <= REC_RATIO[1])))
        if r.dur_s > best.get(int(r.detector), (0,))[0]:
            best[int(r.detector)] = (r.dur_s, r.t0, r.t1, float(lam), float(r.co_stuck), rec, flow, rec_days,
                                     pc, np.nan if pref is None else int(dets[pref[0]]))
    for c, k in (("ep_dur", 0), ("ep_t0", 1), ("ep_t1", 2), ("ep_lam", 3), ("ep_co", 4), ("ep_rec", 5),
                 ("ep_flow", 6), ("ep_days", 7), ("ep_pcorr", 8), ("ep_partner", 9)):
        st[c] = st.detector.map({d: v[k] for d, v in best.items()})
    st["drop_rec"] = np.nan
    st["drop_in_stuck"] = False
    st["drop_partner"] = np.nan
    for j, r in st.iterrows():
        if r.n_on > 0 and r.drop_lam >= LIM["drop_lam"][0] and r.drop_b1 > r.drop_b0:
            i = idx[int(r.detector)]
            r0 = start + pd.Timedelta(seconds=int(r.drop_b0) * B["bin_s"])
            r1 = start + pd.Timedelta(seconds=int(r.drop_b1) * B["bin_s"])
            ov = sum(max((min(p["end"], r1) - max(p["start"], r0)).total_seconds(), 0)
                     for p in per[int(r.detector)])
            if ov >= 0.5 * (r1 - r0).total_seconds():
                st.loc[j, "drop_in_stuck"] = True
                continue
            pref, _ = (_partner(B, i, pcand[i], int(r.drop_b0), int(r.drop_b1))
                       if OPTS["partner"] else (None, np.nan))
            rec = np.nan if r.drop_to_end else _recovery(B, i, int(r.drop_b0), int(r.drop_b1),
                                                          pref if pref is not None else rref[i])
            st.loc[j, "drop_rec"] = rec
            if pref is not None:
                st.loc[j, "drop_partner"] = int(dets[pref[0]])
            per[int(r.detector)].append(dict(
                start=r0, end=r1, what="silent",
                recovered=None if not np.isfinite(rec) else bool(REC_RATIO[0] <= rec <= REC_RATIO[1])))
    h = rules(st, B)
    # v4: the periods behind a choppy / worst-hour rapid call are listed too, so only they can be dropped
    sc = dict(zip(h.detector, h["s_choppy"])) if "s_choppy" in h else {}
    sr = dict(zip(h.detector, h["s_rapid_hour"])) if "s_rapid_hour" in h else {}

    def tb(b):
        return start + pd.Timedelta(seconds=int(b) * B["bin_s"])

    for d, L in spk_per.items():
        if sc.get(d, 0) >= 0.35:
            per[d] += [dict(start=tb(a_), end=tb(min(b_, nb)), what="spike", recovered=None) for a_, b_ in L]
    for d, (a_, b_) in rh_per.items():
        if sr.get(d, 0) >= 0.35:
            per[d].append(dict(start=tb(a_), end=tb(min(b_, nb)), what="rapid re-actuation", recovered=None))
    h["bad_periods"] = h.detector.map(lambda d: sorted(per.get(int(d), []), key=lambda x: x["start"]))
    # note 83: "undercounts vs partner" -- an information note only (status and score unchanged)
    pn = partner_notes(events_df, start, end, phase, fl)
    if pn:
        h["reason"] = [r_ + ("; " if "; note: " in r_ else "; note: ") + pn[int(d)] if int(d) in pn else r_
                       for d, r_ in zip(h.detector, h.reason.fillna(""))]
    return h.merge(st, on="detector")


# ===================================================== note 83: undercounts vs partner (information note only)
# Note 79 candidate (c1): the detector's count over the count of its best same-phase partner (best 15-min count
# correlation; in a window under 1 h the busiest phase-mate), judged as -log(ratio) against a limit per (own,
# partner) predicted function and sample length = p99.5 of presumed-healthy detectors (w40 windows, >= 30 rows per
# pair else the pooled limit; note 82 sheet).  Needs >= 20 actuations on the detector and on its partner.  It is
# REPORTED as a note ("counts x % of d_k ...") and never changes the status: it was real on the independent tier
# (+7-8 pt recall, CI > 0) but roughly doubles flags at 3 h, so the user judges it from the review sheet first.
PARTNER_MIN_ON = 20
PARTNER_LIM = {                 # note-82 calibration (h82_review.partner_limits), -log(ratio) limits
    0.5: (5.1545, {"Advance>?": 4.4218, "Advance>Advance": 3.8466, "Advance>Count": 3.3523, "Advance>Mid": 4.5916, "Advance>Other": 3.0019, "Advance>Presence": 2.3750, "Advance>Yellow_Red": 2.4305, "Bike>?": 6.0336, "Bike>Advance": 6.9078, "Bike>Count": 5.9745, "Bike>Mid": 6.2855, "Bike>Other": 6.7051, "Bike>Presence": 4.7705, "Count>?": 5.5941, "Count>Advance": 4.1667, "Count>Count": 4.2822, "Count>Mid": 4.2947, "Count>Other": 3.8873, "Count>Presence": 3.7727, "Count>Yellow_Red": 3.8737, "Mid>?": 1.2176, "Mid>Advance": 0.5745, "Mid>Mid": 1.0591, "Mid>Presence": 0.8110, "Other>?": 5.2643, "Other>Advance": 4.9157, "Other>Count": 4.5086, "Other>Mid": 5.5187, "Other>Other": 2.9040, "Other>Presence": 4.1475, "Other>Yellow_Red": 4.8531, "Presence>?": 3.4138, "Presence>Advance": 3.2881, "Presence>Count": 3.5264, "Presence>Mid": 4.2690, "Presence>Other": 4.3390, "Presence>Presence": 3.7304, "Presence>Yellow_Red": 4.1194, "Yellow_Red>?": 4.8588, "Yellow_Red>Advance": 3.4658, "Yellow_Red>Count": 4.0634, "Yellow_Red>Other": 3.2462, "Yellow_Red>Presence": 2.5919}),
    3.0: (5.9638, {"Advance>?": 3.7228, "Advance>Advance": 3.9833, "Advance>Count": 1.7203, "Advance>Mid": 1.7080, "Advance>Other": 1.3448, "Advance>Presence": 1.3863, "Advance>Yellow_Red": 1.3894, "Bike>?": 6.9078, "Bike>Advance": 6.9078, "Bike>Count": 6.8988, "Bike>Other": 6.8540, "Bike>Presence": 6.9078, "Count>?": 6.5542, "Count>Advance": 4.5480, "Count>Count": 3.8984, "Count>Other": 4.0784, "Count>Presence": 2.6414, "Count>Yellow_Red": 1.5107, "Mid>?": 0.3941, "Mid>Advance": 0.3183, "Mid>Presence": 0.8566, "Other>?": 5.9443, "Other>Advance": 6.2317, "Other>Count": 4.9953, "Other>Other": 5.1804, "Other>Presence": 4.8928, "Other>Yellow_Red": 1.1757, "Presence>?": 2.6213, "Presence>Advance": 2.3877, "Presence>Count": 2.0137, "Presence>Other": 3.1926, "Presence>Presence": 3.3256, "Presence>Yellow_Red": 1.5008, "Yellow_Red>Count": 2.5091, "Yellow_Red>Other": 2.9614, "Yellow_Red>Presence": 3.0364, "Yellow_Red>Yellow_Red": 0.6473}),
    24.0: (6.5121, {"Advance>?": 2.0676, "Advance>Advance": 3.1403, "Advance>Count": 2.3424, "Advance>Mid": 1.2601, "Advance>Other": 1.6837, "Advance>Presence": 0.9946, "Advance>Yellow_Red": 1.0494, "Bike>?": 6.9078, "Bike>Advance": 6.9078, "Bike>Count": 6.8309, "Bike>Mid": 6.9078, "Bike>Other": 6.9078, "Bike>Presence": 6.9078, "Count>?": 4.9946, "Count>Advance": 3.1644, "Count>Count": 3.4639, "Count>Other": 3.6144, "Count>Presence": 1.4712, "Count>Yellow_Red": 0.9375, "Mid>?": 0.2691, "Mid>Advance": 0.4409, "Mid>Presence": 0.3440, "Other>?": 4.9022, "Other>Advance": 5.0206, "Other>Count": 4.2433, "Other>Other": 4.3813, "Other>Presence": 4.0402, "Other>Yellow_Red": 1.0981, "Presence>?": 1.7474, "Presence>Advance": 1.9663, "Presence>Count": 1.7517, "Presence>Other": 1.9432, "Presence>Presence": 2.0663, "Presence>Yellow_Red": 1.8103, "Yellow_Red>?": 1.2385, "Yellow_Red>Count": 2.1246, "Yellow_Red>Presence": 0.6413, "Yellow_Red>Yellow_Red": 0.5527}),
}
PARTNER_WLEN = (0.5, 3.0, 24.0)   # calibration lengths (h); a sample takes the nearest on a log scale


def _cont_on_starts(ev: pd.DataFrame, start, end) -> dict:
    """{detector: ON start times (s from start) of continuous ONs} (ON after an OFF; note-79 counting)."""
    e = ev[(ev.Timestamp >= start) & (ev.Timestamp < end) & ev.EventId.isin((81, 82)) & (ev.Parameter <= MAXCH)]
    e = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
    if e.empty:
        return {}
    t = (e.Timestamp - start).dt.total_seconds().to_numpy()
    eid, par = e.EventId.to_numpy().astype(int), e.Parameter.to_numpy().astype(int)
    o = np.lexsort((eid != 82, t, par))                    # channel, time, ON before OFF
    t, eid, par = t[o], eid[o], par[o]
    prev_on = np.r_[False, (par[1:] == par[:-1]) & (eid[:-1] == 82)]
    st_ = (eid == 82) & ~prev_on
    return {int(c): t[st_ & (par == c)] for c in np.unique(par[st_])}


def partner_notes(ev: pd.DataFrame, start, end, phase: dict, flabel: dict) -> dict:
    """{detector: note text} for detectors whose count sits below the healthy limit relative to their partner."""
    if not PARTNER_LIM or not phase:
        return {}
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    T = (end - start).total_seconds()
    hrs = T / 3600.0
    wl = min(PARTNER_WLEN, key=lambda w: abs(np.log(max(hrs, 1e-3) / w)))
    pooled, by_grp = PARTNER_LIM.get(wl, (np.nan, {}))
    on = _cont_on_starts(ev, start, end)
    nb = max(int(T // 900), 1)
    cnt = {d: np.bincount(np.minimum((x // 900).astype(int), nb - 1), minlength=nb) for d, x in on.items()}
    n = {d: len(x) for d, x in on.items()}
    ckey = {d: int(_ticks(x).sum()) for d, x in on.items()}            # content tie-break, never the channel number
    out = {}
    for d in sorted(on):
        p = phase.get(d)
        if p is None or not np.isfinite(p) or n[d] < PARTNER_MIN_ON:
            continue
        mates = [k for k in on if k != d and phase.get(k) == p and n[k] >= PARTNER_MIN_ON]
        if not mates:
            continue
        best, bk = None, None
        if nb >= 4:
            for k in mates:
                a, b = cnt[d], cnt[k]
                c = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
                if np.isfinite(c):
                    key = (round(float(c), 12), n[k], ckey[k])
                    if bk is None or key > bk:
                        best, bk = k, key
        if best is None:
            best = max(mates, key=lambda k: (n[k], ckey[k]))
        ratio = n[d] / max(n[best], 1)
        own, pfn = flabel.get(d), flabel.get(best)
        lim = by_grp.get(f"{own if own else '?'}>{pfn if pfn else '?'}", pooled)
        if not np.isfinite(lim) or -np.log(max(ratio, 1e-3)) < lim:
            continue
        out[d] = (f"undercounts vs partner: counted {n[d]:,} vs {n[best]:,} on d{best}, the same-phase detector it "
                  f"follows best ({ratio:.0%}; healthy {own or '?'} / {pfn or '?'} pairs go down to {np.exp(-lim):.0%}) "
                  f"- information only, not a fault call")
    return out
