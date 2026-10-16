"""Detector health, stage 1: the per-detector rule checks from the hi-res log that the health output builds on
(notes 38-47, slimmed in 1.1.0 / note 144).  numpy + pandas only.

    health(events_df, start, end, detectors=None, phase=None, function=None) -> one row per detector

`events_df`: columns Timestamp, EventId, Parameter (the controller log, local time), or a `Prep` of the window
(`prep_arrays`: predict passes its de-duplicated event arrays directly; every check reads the same prepared arrays).
`detectors` is ignored (health is judged from the log only); a channel with no actuation in the window has no row.
`phase` / `function` (optional): {detector: predicted phase} and {detector: predicted function label, or a dict of
class probabilities} from the phase / function model run first -- predictions only, never a wiring table or a print.

Stage 1 gives the measures the six health checks read (goes silent, stuck-on holds with their recovery, the level
drop, night counts against a partner) and a pass-1 status from those four rules only (1.1.0: chatter, rapid
re-actuation, 5-min volume, night vs day, choppy counts, correlation with the signal, short ONs and the
'undercounts vs partner' note are gone; the health output has its own too-fast, too-many and erratic checks).
The pass-1 status decides which detectors are a yardstick for the others (a 'bad' one is nobody's yardstick).

Only actuations (81 / 82) plus comms coverage from any allowed event are used.  Detector fault events (83-88) are NOT
used (user, 2026-09-28).  Continuous ON (note 47): an ON followed by more ONs before the next OFF is normal
(extension); it is one continuous ON from the first ON to the next OFF, and stuck-on is judged only from its length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import timedelta as _td

BIN_S = 300                     # 5-min bins
AGG = 3                         # 3 x 5 min = 15-min bins
ALLOWED = (1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173)   # no fault events 83-88
MAXCH = 64
GAP_S = 120.0                   # no event of any code for this long = comms gap
TICK_S = 0.1                    # the hi-res log clock
CHAT_TICKS = 3                  # chatter = OFF -> next ON gap < 3 ticks (compared in whole ticks)


def _ticks(x):
    """seconds -> whole 0.1-s log ticks (NaN stays NaN); gaps are compared in ticks, never in float seconds."""
    return np.rint(np.asarray(x, dtype=float) / TICK_S)


DAY = (7, 19)                   # clock hours [7, 19)
LIM = dict(
    drop_lam=(30.0, 100.0),       # expected ONs inside the silent run
    stuck_s=(900.0, 3600.0),     # longest single ON
    level=(0.15, 0.05),          # after/before share ratio at the best change point
)
SUSPECT_H, BAD_H = 0.70, 0.25    # health_score cut-offs
CO_STUCK_N = 3         # held ON together with >= 3 others = shared event: capped at suspect, period listed
REC_RATIO = (0.5, 2.0)  # share after / before an episode (6 h each side) for a clean recovery
REC_MIN_BINS = 36      # >= 3 h of data after the episode (and before it) to judge the recovery
REC_BINS = 72          # 6 h each side
CO_MIN_S = 300.0       # another detector's ON counts as "held ON at the same time" from 5 min
PARTNER_CORR = 0.7     # recovery partner: best 15-min correlation with a same-phase sibling
NIGHT_H = (21, 5)                 # night for the night-drop check: 21:00-05:00
NIGHT_MIN_EXP = 30.0              # night ONs expected from the reference before the check runs
NIGHT_DROP_LIM = (0.19, 0.063)    # night count / expected: suspect / bad
DAY_NS = 86_400_000_000_000


class Prep:
    """The cleaned events of one signal window, prepared ONCE per health() call and shared by every check: times in
    seconds from the window start (t), codes (eid), parameters (par), the comms gaps (g0 / g1: no event of any code
    for > GAP_S) and the detector events sorted by (channel, time, ON before OFF) (td / ed / pd_)."""
    __slots__ = ("t", "eid", "par", "ut", "g0", "g1", "td", "ed", "pd_", "empty", "cache")


def rows_of(df: pd.DataFrame) -> list:
    """note 128: light row views over a (wide) frame in place of iterrows / itertuples: `r.col`, `r["col"]`,
    `r.get(col, default)` and `col in r` give the values iterrows gives on a mixed-type frame (each column turned into
    a Python list once, when first read), without building a Series or a tuple of every column for every row."""
    cols, cache = set(df.columns), {}

    def col(k):
        v = cache.get(k)
        if v is None:
            v = cache[k] = df[k].tolist()
        return v

    class Row:
        __slots__ = ("_i",)

        def __init__(self, i):
            self._i = i

        def __getattr__(self, k):
            if k not in cols:
                raise AttributeError(k)
            return col(k)[self._i]

        def __getitem__(self, k):
            if k not in cols:
                raise KeyError(k)
            return col(k)[self._i]

        def __contains__(self, k):
            return k in cols

        def get(self, k, default=None):
            return col(k)[self._i] if k in cols else default

    return [Row(i) for i in range(len(df))]


def _memo(P_, key, fn):
    """note 128: results that depend only on the window's events (bins, per-actuation tables, episodes) are computed
    once per Prep and shared by both stage-1 passes and the v4 layer (callers never edit them in place)."""
    c = getattr(P_, "cache", None)
    if c is None:
        return fn()
    if key not in c:
        c[key] = fn()
    return c[key]


def prep_arrays(t, eid, par) -> Prep:
    """`Prep` from already-cleaned arrays (allowed codes only, no detector channel > 64, no duplicate rows, times in
    seconds from the window start, all inside the window) -- predict passes them straight from its event table."""
    P_ = Prep()
    P_.t = np.asarray(t, dtype=np.float64)
    P_.eid = np.asarray(eid).astype(int)
    P_.par = np.asarray(par).astype(int)
    P_.empty = len(P_.t) == 0
    P_.cache = {}
    P_.ut = np.unique(P_.t)
    gi = np.diff(P_.ut) > GAP_S
    P_.g0, P_.g1 = P_.ut[:-1][gi], P_.ut[1:][gi]
    dm = np.isin(P_.eid, (81, 82))
    td, ed, pd_ = P_.t[dm], P_.eid[dm], P_.par[dm]
    o = np.lexsort((ed != 82, td, pd_))
    P_.td, P_.ed, P_.pd_ = td[o], ed[o], pd_[o]
    return P_


def _prep(ev, start, end) -> Prep:
    """A `Prep` from a raw event frame (Timestamp, EventId, Parameter), or `ev` itself when it already is one."""
    if isinstance(ev, Prep):
        return ev
    e = ev[["Timestamp", "EventId", "Parameter"]]
    e = e[(e.Timestamp >= start) & (e.Timestamp < end) & e.EventId.isin(ALLOWED)]
    e = e[~(e.EventId.isin((81, 82)) & (e.Parameter > MAXCH))].drop_duplicates()
    return prep_arrays((e.Timestamp - start).dt.total_seconds().to_numpy(), e.EventId.to_numpy(),
                       e.Parameter.to_numpy())


# ============================================================================ binning
def cont_on(P_, T):
    """continuous_on of a Prep's detector events, computed once per window (read only)."""
    return _memo(P_, ("cont", float(T)), lambda: continuous_on(P_.td, P_.ed, P_.pd_, T))


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
    if detectors is None and isinstance(ev, Prep):
        return _memo(ev, ("bins", start, end, int(bin_s)), lambda: _events_to_bins(ev, start, end, None, bin_s))
    return _events_to_bins(ev, start, end, detectors, bin_s)


def twins_of(B: dict):
    """twins() of a bins dict, computed once per bins dict."""
    if "_tw" not in B:
        B["_tw"] = twins(B["n_on"].astype(float), B["cov"])
    return B["_tw"]


def _on_view(P_, T, dets):
    """the per-ON view behind events_to_bins (independent of the bin size): ON times, channel rows, continuous-ON
    durations (NaN inside a continuous ON or over a comms gap), chatter flags, the longest ON per channel and the
    channels that were ON when the window opened."""
    gaps0 = P_.g0                             # comms-gap start times
    td, ed, pd_ = P_.td, P_.ed, P_.pd_
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    prv_t, prv_e = np.r_[np.nan, td[:-1]], np.r_[0, ed[:-1]]
    on = ed == 82
    # continuous ON: from the ON that starts it to the next OFF (ONs inside it are extension re-calls);
    # still ON at the window end -> to the end.  Inner ONs get NaN (their time is already counted).
    c_end, c_start = cont_on(P_, T)
    dur = np.where(c_start, c_end - td, np.nan)[on]
    gap = np.where(same_prev & (prv_e == 81), td - prv_t, np.nan)[on]
    ton, ci = td[on], np.searchsorted(dets, pd_[on]).astype(int)
    nd = len(dets)
    dur_max = np.zeros(nd)
    chat = np.zeros(0)
    if len(ton):
        long_ = np.where(dur > 60)[0]                     # an ON over a comms gap is not "stuck"
        if len(long_) and len(gaps0):
            k = np.searchsorted(gaps0, ton[long_])
            hit = (k < len(gaps0)) & (gaps0[np.minimum(k, len(gaps0) - 1)] < ton[long_] + dur[long_])
            dur[long_[hit]] = np.nan
        with np.errstate(invalid="ignore"):
            chat = (_ticks(gap) < CHAT_TICKS).astype(float)
        # longest ON per channel (ci is sorted: the events are ordered by channel)
        dd = np.nan_to_num(dur, nan=0.0)
        u, first = np.unique(ci, return_index=True)
        dur_max[u] = np.maximum(np.maximum.reduceat(dd, first), 0.0)
    first = np.r_[True, pd_[1:] != pd_[:-1]]
    lead = first & (ed == 81)
    return dict(ton=ton, ci=ci, dur=dur, chat=chat, dur_max=dur_max, lead_t=td[lead], lead_c=pd_[lead])


def _events_to_bins(ev, start, end, detectors=None, bin_s: int = BIN_S) -> dict:
    T = (end - start).total_seconds()
    nb = int(np.ceil(T / bin_s))
    P_ = _prep(ev, start, end)
    t, eid, par = P_.t, P_.eid, P_.par
    cov = np.bincount((t // bin_s).astype(int), minlength=nb)[:nb] > 0
    chans = set(_memo(P_, "chans", lambda: np.unique(P_.pd_).tolist()))   # pd_ = the 81 / 82 parameters
    if detectors is not None:
        chans |= {int(x) for x in detectors}
    dets = np.array(sorted(chans), dtype=int)
    nd = len(dets)
    idx = {c: i for i, c in enumerate(dets)}
    out = {k: np.zeros((nd, nb), np.float32) for k in ("n_on", "occ", "n_chat")}
    V = (_memo(P_, ("onview", start, end), lambda: _on_view(P_, T, dets)) if detectors is None
         else _on_view(P_, T, dets))
    ton, ci, dur = V["ton"], V["ci"], V["dur"]
    if len(ton):
        b0 = np.minimum((ton // bin_s).astype(int), nb - 1)
        cell = ci * nb + b0
        out["n_on"] = np.bincount(cell, minlength=nd * nb).reshape(nd, nb).astype(np.float32)
        out["n_chat"] = np.bincount(cell, weights=V["chat"], minlength=nd * nb).reshape(nd, nb).astype(np.float32)
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
    for tt, cc in zip(V["lead_t"], V["lead_c"]):
        bf = min(int(tt // bin_s), nb - 1)
        i = idx[int(cc)]
        out["occ"][i, :bf] += bin_s
        out["occ"][i, bf] += tt - bf * bin_s
    # clock hour of each bin start (integer nanoseconds: = (start + k * bin_s seconds).hour)
    hrs = ((int(start.value) + np.arange(nb, dtype=np.int64) * (int(bin_s) * 1_000_000_000))
           // 3_600_000_000_000) % 24
    out.update(dets=dets, cov=cov, hour=hrs.astype(np.int32), dur_max=V["dur_max"].copy(), start=start, bin_s=bin_s,
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


def _silent(n, occ, cov, active, i, S, n_on) -> dict:
    """silent runs: bins with no ON and (almost) no occupancy, on covered bins; expected ONs in the run = the
    detector's share outside the run x the reference S inside it; and the other live detectors silent with it."""
    nb = n.shape[1]
    r = {}
    x = n[i]
    quiet = (x == 0) & (occ[i] < 1) & cov
    s0, s1 = _runs(quiet)
    best = (0.0, -1, -1)
    if len(s0) and n_on > 0:
        cS = np.r_[0, np.cumsum(np.where(cov, S, 0))]
        for a, b in zip(s0, s1):
            s_in = cS[b] - cS[a]
            s_out = cS[-1] - s_in
            lam = n_on / max(s_out, 1) * s_in
            if lam > best[0]:
                best = (lam, a, b)
    r["drop_lam"], r["drop_b0"], r["drop_b1"] = best
    r["drop_to_end"] = best[2] == nb or (best[2] > 0 and not cov[best[2]:].any())
    # other live detectors that were silent through the same run (shared unit / comms?)
    r["co_silent"] = 0
    if best[2] - best[1] >= 3:
        q = ((n[:, best[1]:best[2]] == 0) & (occ[:, best[1]:best[2]] < 1)).mean(1) >= 0.8
        r["co_silent"] = int((q & active).sum() - 1)
    return r


def drop_stats(B: dict, refs: dict) -> pd.DataFrame:
    """the silent-run columns of det_stats (drop_lam, drop_b0, drop_b1, drop_to_end, co_silent) for the detectors in
    `refs` ({row: reference rows}) only, indexed by row (note 128: the phase-reference pass needs nothing else)."""
    n, occ, cov = B["n_on"].astype(float), B["occ"].astype(float), B["cov"]
    active = (n.sum(1) > 0) | (occ.sum(1) > 0)
    rows = []
    for i, rf in refs.items():
        x, S = n[i], n[rf].sum(0)
        rows.append(_silent(n, occ, cov, active, i, S, x[cov].sum()))
    return pd.DataFrame(rows, index=list(refs))


def det_stats(B: dict) -> pd.DataFrame:
    """per detector against the rest of the signal (computed once per bins dict): ONs, the silent run, the level
    shift of its share (15-min bins, best single change point, Poisson LLR)."""
    if "_ds" not in B:
        B["_ds"] = _det_stats(B)
    return B["_ds"].copy()


def _det_stats(B: dict) -> pd.DataFrame:
    n, occ, cov = B["n_on"].astype(float), B["occ"].astype(float), B["cov"]
    nd, nb = n.shape
    cov_h = cov.sum() * B["bin_s"] / 3600
    tot = n.sum(0)
    rows = []
    n15, cov15 = _agg(n), _agg(cov[None].astype(float))[0] == AGG
    active = (n.sum(1) > 0) | (occ.sum(1) > 0)
    for i in range(nd):
        r = {"detector": int(B["dets"][i])}
        x, S = n[i], tot - n[i]
        r["n_on"] = x[cov].sum()
        r["cov_h"] = cov_h
        r["n_sib"] = int(active.sum() - active[i])
        r["dur_max"] = B["dur_max"][i]
        r.update(_silent(n, occ, cov, active, i, S, r["n_on"]))
        r["level_ratio"], r["level_llr"] = np.nan, 0.0
        xs, Ss = n15[i][cov15], (n15.sum(0) - n15[i])[cov15]
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
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================= rules
def _score(x, lo, hi):
    """0 below the suspect limit lo; .35 at lo rising to 1 at the bad limit hi; NaN stays NaN."""
    if x is None or not np.isfinite(x):
        return np.nan
    if x < lo:
        return 0.0
    return float(0.35 + 0.65 * np.clip((x - lo) / (hi - lo), 0, 1))


def _hm(b, B):
    t = B["start"] + _td(seconds=int(b) * B["bin_s"])
    return t.strftime("%a %d %b %H:%M")


def _f(r, k):
    v = r.get(k, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def rules(st: pd.DataFrame, B: dict) -> pd.DataFrame:
    """the four stage-1 rules (sudden silence, stuck-on, night count, level drop) -> score, pass-1 status, reason."""
    L = LIM
    out = []
    for r in rows_of(st):
        s, why = {}, {}
        onph = r.get("drop_ref") == "phase"
        if r.n_on > 0:
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
                    if REC_RATIO[0] <= rr <= REC_RATIO[1]:
                        s["dropout"] = min(s["dropout"], 0.35)
                        why["dropout"] += (f"; afterwards it counted its usual share again ({rr:.0%} of "
                                           f"before) - the data outside this period looks usable")
                    else:
                        why["dropout"] += (f"; after it came back it counted {rr:.0%} of its earlier "
                                           f"share - not a clean recovery")
        # stuck-on: the longest continuous ON that counts (ep_dur; ONs inside it are extension re-calls)
        sd = _f(r, "ep_dur") if "ep_dur" in r else r.dur_max
        sd = 0.0 if not np.isfinite(sd) else sd
        s["stuck"] = _score(sd, *L["stuck_s"])
        if sd >= L["stuck_s"][0]:
            t0, t1 = r.get("ep_t0"), r.get("ep_t1")
            span = (f" ({pd.Timestamp(t0).strftime('%a %d %b %H:%M')} - {pd.Timestamp(t1).strftime('%H:%M')})"
                    if isinstance(t0, pd.Timestamp) and isinstance(t1, pd.Timestamp) else "")
            why["stuck"] = f"stayed ON continuously for {sd / 60:.0f} min (from an ON to the next OFF){span}"
            co, rr = _f(r, "ep_co"), _f(r, "ep_rec")
            if np.isfinite(co) and co >= CO_STUCK_N:
                # a shared event: the period is unusable, but it is not this detector's fault alone
                s["stuck"] = min(s["stuck"], 0.35)
                why["stuck"] += (f"; {int(co)} other detectors were held ON at the same time - a shared event "
                                 f"(real queue, closure or works, or cabinet-wide), not this detector alone: "
                                 f"drop the period, keep the detector")
            else:
                if np.isfinite(rr):
                    if REC_RATIO[0] <= rr <= REC_RATIO[1]:
                        s["stuck"] = min(s["stuck"], 0.35)
                        why["stuck"] += (f"; afterwards it counted its usual share again ({rr:.0%} of "
                                         f"before) - the data outside this period looks usable")
                    else:
                        why["stuck"] += f"; after it recovered it counted {rr:.0%} of its earlier share"
                if np.isfinite(co) and co > 0:
                    why["stuck"] += f"; {int(co)} other detector(s) held ON at the same time (shared card?)"
        # stops counting at night while the detector it tracks (or its phase) keeps counting (note 47 b')
        nr = _f(r, "night_ratio")
        if np.isfinite(nr):
            s["night_drop"] = _score(-np.log(max(nr, 1e-6)), -np.log(NIGHT_DROP_LIM[0]), -np.log(NIGHT_DROP_LIM[1]))
            if s["night_drop"] > 0:
                why["night_drop"] = (f"at night ({NIGHT_H[0]:02d}:00-{NIGHT_H[1]:02d}:00) it counted "
                                     f"{_f(r, 'night_n'):.0f} actuations where about {_f(r, 'night_exp'):.0f} were "
                                     f"expected from {r.get('night_ref')} ({nr:.0%}) - it misses vehicles at night")
        if np.isfinite(r.level_ratio):
            s["level"] = 0.0 if r.level_llr <= 50 else _score(
                -np.log(r.level_ratio), -np.log(L["level"][0]), -np.log(L["level"][1]))
            if s["level"] > 0:
                why["level"] = (f"dropped to {r.level_ratio:.0%} of its earlier share of the signal's "
                                f"traffic from {_hm(r.level_b, B)}")
        vals = {k: v for k, v in s.items() if np.isfinite(v)}
        hs = float(np.prod([1 - v for v in vals.values()])) if vals else np.nan
        if np.isfinite(hs) and hs < BAD_H:
            status = "bad"
        elif np.isfinite(hs) and hs < SUSPECT_H:
            status = "suspect"
        elif r.n_on < 20:
            status = "not_enough_data"
        else:
            status = "ok"
        top = sorted(((v, k) for k, v in vals.items() if k in why), reverse=True)
        reason = "; ".join(why[k] for v, k in top[:2]) if top and status in ("bad", "suspect") else ""
        if status == "not_enough_data":
            reason = f"only {r.n_on:.0f} actuations in {r.cov_h:.1f} h - too few to judge"
        if status == "ok":
            reason = f"{r.n_on:,.0f} actuations in {r.cov_h:.1f} h, nothing unusual"
        out.append(dict(detector=r.detector, health_score=hs, status=status, reason=reason,
                        **{f"s_{k}": v for k, v in s.items()}))
    return pd.DataFrame(out)


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


def on_episodes(ev: pd.DataFrame, start, end, min_s: float = 900.0) -> pd.DataFrame:
    """Continuous ONs held >= min_s (stuck-on), with start / end.  A continuous ON runs from the ON that
    starts it to the next OFF; further ONs inside it (no OFF between) are normal extension re-calls and
    do not end it (note 47).  `n_on` = ONs logged inside it.  An ON over a comms gap is not counted."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if isinstance(ev, Prep):
        # the >= 15-min list is the >= 5-min list cut at 15 min (both compare ONs held >= CO_MIN_S for co_stuck)
        base = float(min(CO_MIN_S, min_s))
        E0 = _memo(ev, ("ep", start, end, base), lambda: _on_episodes(ev, start, end, base))
        return E0.copy() if float(min_s) == base else E0[E0.dur_s >= min_s].reset_index(drop=True)
    return _on_episodes(ev, start, end, min_s)


def _on_episodes(ev, start, end, min_s: float = 900.0) -> pd.DataFrame:
    T = (end - start).total_seconds()
    P_ = _prep(ev, start, end)
    g0 = P_.g0
    td, ed, pd_ = P_.td, P_.ed, P_.pd_
    end_t, start_ = cont_on(P_, T) if isinstance(ev, Prep) else continuous_on(td, ed, pd_, T)
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


def _counts15(B: dict):
    """15-min counts of every detector and the fully covered 15-min bins, computed once per window (cached in B; the
    coverage mask is returned as a copy because callers edit it)."""
    if "_n15" not in B:
        B["_n15"] = _agg(B["n_on"].astype(float))
        B["_ok15"] = _agg(B["cov"][None].astype(float))[0] == AGG
    return B["_n15"], B["_ok15"].copy()


def _partner(B: dict, i: int, cand, a: int, b: int):
    """(b): the same-phase sibling whose 15-min counts tracked detector i best outside bins [a, b)
    (corr >= PARTNER_CORR), as a row index list, else None."""
    if cand is None or not len(cand):
        return None, np.nan
    n15, ok = _counts15(B)
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


def _merge_same(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """left.merge(right, on="detector", how=how) -- as one concat when both hold the same detectors in the same order
    and share no other column (then the merge adds nothing and changes no type)."""
    if (len(left) == len(right) and np.array_equal(left.detector.to_numpy(), right.detector.to_numpy())
            and not (set(left.columns) & set(right.columns)) - {"detector"}):
        r = right.drop(columns="detector")
        r.index = left.index
        return pd.concat([left, r], axis=1)
    return left.merge(right, on="detector", how=how)


def _add_cols(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """df with the new columns `cols` appended in order (one concat instead of a setitem each)."""
    if not cols:
        return df
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def health(events_df: pd.DataFrame, start, end, detectors=None, phase: dict | None = None,
           function: dict | None = None, lanes: dict | None = None,
           phase_conf: dict | None = None) -> pd.DataFrame:
    """Per-detector stage-1 health table for one signal over [start, end).  Columns: detector, health_score, status,
    reason, bad_periods (list of {start, end, what, recovered}), the rule scores s_* and the statistics behind them.
    `lanes` / `phase_conf` are accepted for compatibility and not used."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    events_df = _prep(events_df, start, end)
    B = events_to_bins(events_df, start, end, None)
    if len(B["dets"]) == 0:
        return pd.DataFrame(columns=["detector", "health_score", "status", "reason", "bad_periods"])
    dets = B["dets"]
    n, cov = B["n_on"].astype(float), B["cov"]
    live = n[:, cov].sum(1) > 0
    phase = {int(k): v for k, v in (phase or {}).items() if v is not None and np.isfinite(v)}
    ref, kind, _ = _refs(dets, live, phase, twins_of(B))
    # silence is expected from the other detectors on the predicted phase when there are >= 2
    ph2 = [i for i in range(len(dets)) if kind[i] == "phase" and len(ref[i]) >= 2]
    st = det_stats(B)
    st["drop_ref"] = "signal"
    if ph2:
        sp = drop_stats(B, {i: ref[i] for i in ph2})
        for c in ["drop_lam", "drop_b0", "drop_b1", "drop_to_end", "co_silent"]:
            v = st[c].to_numpy().copy()
            v[ph2] = sp[c].to_numpy()
            st[c] = v
        v = st["drop_ref"].to_numpy().copy()
        v[ph2] = "phase"
        st["drop_ref"] = v
    fl = _labels(function)
    st["pred_function"] = st.detector.map(fl)
    st["pred_phase"] = st.detector.map(phase) if phase else np.nan
    idx = {int(d): i for i, d in enumerate(dets)}
    rref = [ref[i] if kind[i] == "phase" and len(ref[i]) >= 2 else None for i in range(len(dets))]
    # recovery partner candidates: every other live detector on the predicted phase, twins included
    phv = np.array([phase.get(int(d), np.nan) for d in dets], float)
    pcand = [list(np.where(live & (np.arange(len(dets)) != i) & (phv == phv[i]))[0]) if np.isfinite(phv[i]) else []
             for i in range(len(dets))]
    # night count against the detector it tracks (best 15-min correlation >= .7 with a same-phase sibling, twins
    # included), else the other detectors on its phase (>= 2), else the signal
    hr = B["hour"]
    nt = cov & ((hr >= NIGHT_H[0]) | (hr < NIGHT_H[1]))
    dy = cov & (hr >= DAY[0]) & (hr < DAY[1])
    nd = {c: {} for c in ("night_ratio", "night_n", "night_exp", "night_ref")}
    if nt.sum() >= 24 and dy.sum() >= 24:
        for j, r in zip(st.index, rows_of(st)):
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
    st = _add_cols(st, {c: pd.Series(v, dtype=object if c == "night_ref" else float).reindex(st.index)
                        for c, v in nd.items()})
    # ---- episodes: stuck-on (incl. ON with no OFF) and silent runs -> bad periods + recovery
    per = {int(d): [] for d in dets}
    ep = on_episodes(events_df, start, end)
    nb = n.shape[1]
    best = {}
    held = {}
    for r in ep.itertuples():
        held.setdefault(int(r.detector), []).append((r.t0, r.t1))
    for r in ep.itertuples():
        i = idx.get(int(r.detector))
        if i is None:
            continue
        a0 = int((r.t0 - start).total_seconds() // B["bin_s"])
        b0 = min(int(np.ceil((r.t1 - start).total_seconds() / B["bin_s"])), nb)
        pref, pc = _partner(B, i, [k for k in pcand[i] if not any(t0 < r.t1 and t1 > r.t0 for t0, t1
                                                                  in held.get(int(dets[k]), []))], a0, b0)
        rec = np.nan if r.open_end else _recovery(B, i, a0, b0, pref if pref is not None else rref[i])
        per[int(r.detector)].append(dict(start=r.t0, end=r.t1, what="stuck on",
                                         recovered=None if not np.isfinite(rec) else bool(REC_RATIO[0] <= rec <= REC_RATIO[1])))
        if r.dur_s > best.get(int(r.detector), (0,))[0]:
            best[int(r.detector)] = (r.dur_s, r.t0, r.t1, float(r.co_stuck), rec)
    epc = {c: st.detector.map({d: v[k] for d, v in best.items()})
           for c, k in (("ep_dur", 0), ("ep_t0", 1), ("ep_t1", 2), ("ep_co", 3), ("ep_rec", 4))}
    nst = len(st)
    d_rec, d_in = np.full(nst, np.nan), np.zeros(nst, bool)
    for j, r in enumerate(rows_of(st)):
        if r.n_on > 0 and r.drop_lam >= LIM["drop_lam"][0] and r.drop_b1 > r.drop_b0:
            i = idx[int(r.detector)]
            r0 = start + _td(seconds=int(r.drop_b0) * B["bin_s"])
            r1 = start + _td(seconds=int(r.drop_b1) * B["bin_s"])
            ov = sum(max((min(p["end"], r1) - max(p["start"], r0)).total_seconds(), 0)
                     for p in per[int(r.detector)])
            if ov >= 0.5 * (r1 - r0).total_seconds():
                d_in[j] = True
                continue
            pref, _ = _partner(B, i, pcand[i], int(r.drop_b0), int(r.drop_b1))
            rec = np.nan if r.drop_to_end else _recovery(B, i, int(r.drop_b0), int(r.drop_b1),
                                                          pref if pref is not None else rref[i])
            d_rec[j] = rec
            per[int(r.detector)].append(dict(
                start=r0, end=r1, what="silent",
                recovered=None if not np.isfinite(rec) else bool(REC_RATIO[0] <= rec <= REC_RATIO[1])))
    epc.update(drop_rec=d_rec, drop_in_stuck=d_in)
    st = _add_cols(st, epc)
    h = rules(st, B)
    h["bad_periods"] = h.detector.map(lambda d: sorted(per.get(int(d), []), key=lambda x: x["start"]))
    return _merge_same(h, st, how="inner")
