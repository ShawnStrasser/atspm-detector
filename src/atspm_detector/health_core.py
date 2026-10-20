"""Detector health, stage 1: the per-detector rule checks from the hi-res log that the health output builds on
(notes 38-47, slimmed in 1.1.0 / note 144; computed for many signals at once since note 152).  numpy + pandas only.

    health(events_df, start, end, detectors=None, phase=None, function=None) -> one row per detector

`events_df`: columns Timestamp, EventId, Parameter (the controller log, local time), or a `Prep` of the window
(`prep_arrays`: predict passes its de-duplicated event arrays directly; every check reads the same prepared arrays).
A `Prep` may hold several signals sharing the window [start, end) (`prep_group`, note 152): signal s's channel c is
then the detector key s * BASE + c and its phase p the phase key s * PB + p, so every per-detector and per-phase
computation runs for all signals of the batch at once; every signal-wide quantity (references 'the rest of the
signal', comms coverage and gaps, twins, holds at the same time) stays inside its own signal.  One signal = keys equal
to the channel / phase numbers.
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
COLOUR = (1, 8, 9, 10)
MAXCH = 64
BASE = 128                      # detector key of a batch: signal * BASE + channel
PB = 1 << 16                    # phase key of a batch: signal * PB + phase (event parameters are < 65536)
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
    """The cleaned events of one signal window (or of several signals sharing the window, `prep_group`), prepared ONCE
    and shared by every check: times in seconds from the window start (t, with the signal of each event tsig), codes
    (eid), parameters (par; colour events carry the phase key), the comms gaps (g0 / g1: no event of any code for
    > GAP_S, signal gsig, sorted by signal and time) and the detector events sorted by (detector key, time, ON before
    OFF) (td / ed / pd_ = the detector key)."""
    __slots__ = ("t", "eid", "par", "ut", "usig", "g0", "g1", "td", "ed", "pd_", "empty", "cache", "S", "tsig",
                 "gsig", "names", "covs", "csorted")


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


def _gaps(ut):
    gi = np.diff(ut) > GAP_S
    return ut[:-1][gi], ut[1:][gi]


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
    P_.g0, P_.g1 = _gaps(P_.ut)
    dm = np.isin(P_.eid, (81, 82))
    td, ed, pd_ = P_.t[dm], P_.eid[dm], P_.par[dm]
    o = np.lexsort((ed != 82, td, pd_))
    P_.td, P_.ed, P_.pd_ = td[o], ed[o], pd_[o]
    P_.S = 1
    P_.tsig = np.zeros(len(P_.t), np.int64)
    P_.usig = np.zeros(len(P_.ut), np.int64)
    P_.gsig = np.zeros(len(P_.g0), np.int64)
    P_.names = None
    P_.covs = None
    P_.csorted = False
    return P_


def prep_group(preps: list, names=None) -> Prep:
    """one `Prep` for several signals sharing the window: signal s's detector events get the key s * BASE + channel and
    its colour events the phase key s * PB + phase; arrays stacked in signal order (each signal's own order kept)."""
    if len(preps) == 1 and names is None:
        return preps[0]
    G = Prep()
    S = len(preps)
    G.S = S
    G.names = names
    ns = [len(p.t) for p in preps]
    G.t = np.concatenate([p.t for p in preps]) if S else np.zeros(0)
    G.tsig = np.repeat(np.arange(S, dtype=np.int64), ns)
    G.eid = np.concatenate([p.eid for p in preps]) if S else np.zeros(0, int)
    par = np.concatenate([p.par for p in preps]) if S else np.zeros(0, int)
    G.par = np.where(np.isin(G.eid, COLOUR), G.tsig * PB + par,
                     np.where(np.isin(G.eid, (81, 82)), G.tsig * BASE + par, par))
    G.empty = len(G.t) == 0
    G.cache = {}
    G.ut = np.concatenate([p.ut for p in preps]) if S else np.zeros(0)
    G.usig = np.repeat(np.arange(S, dtype=np.int64), [len(p.ut) for p in preps])
    G.covs = None
    G.csorted = False
    G.g0 = np.concatenate([p.g0 for p in preps]) if S else np.zeros(0)
    G.g1 = np.concatenate([p.g1 for p in preps]) if S else np.zeros(0)
    G.gsig = np.repeat(np.arange(S, dtype=np.int64), [len(p.g0) for p in preps])
    nd = [len(p.td) for p in preps]
    G.td = np.concatenate([p.td for p in preps]) if S else np.zeros(0)
    G.ed = np.concatenate([p.ed for p in preps]) if S else np.zeros(0, int)
    G.pd_ = (np.concatenate([p.pd_ for p in preps]) + np.repeat(np.arange(S, dtype=np.int64) * BASE, nd)) if S \
        else np.zeros(0, int)
    return G


def _prep(ev, start, end) -> Prep:
    """A `Prep` from a raw event frame (Timestamp, EventId, Parameter), or `ev` itself when it already is one."""
    if isinstance(ev, Prep):
        return ev
    e = ev[["Timestamp", "EventId", "Parameter"]]
    e = e[(e.Timestamp >= start) & (e.Timestamp < end) & e.EventId.isin(ALLOWED)]
    e = e[~(e.EventId.isin((81, 82)) & (e.Parameter > MAXCH))].drop_duplicates()
    return prep_arrays((e.Timestamp - start).dt.total_seconds().to_numpy(), e.EventId.to_numpy(),
                       e.Parameter.to_numpy())


def spans_gap(P_, sig, a, b):
    """per interval [a, b) of signal sig (sig ascending): a comms gap of that signal starts in it (k = the first gap
    start >= a, g0[k] < b), as the one-signal searchsorted test."""
    out = np.zeros(len(a), bool)
    g0, gs = P_.g0, P_.gsig
    if not len(g0) or not len(a):
        return out
    sig = np.asarray(sig)
    for s in np.unique(gs):
        q0, q1 = np.searchsorted(sig, s, "left"), np.searchsorted(sig, s, "right")
        if q1 <= q0:
            continue
        g = g0[np.searchsorted(gs, s, "left"):np.searchsorted(gs, s, "right")]
        k = np.searchsorted(g, a[q0:q1])
        kk = np.minimum(k, len(g) - 1)
        out[q0:q1] = (k < len(g)) & (g[kk] < b[q0:q1])
    return out


def sorted_unique(a, return_index=False, return_counts=False):
    """np.unique of an already sorted array (same values, first indices and counts) without sorting it again."""
    a = np.asarray(a)
    first = np.flatnonzero(np.r_[True, a[1:] != a[:-1]]) if len(a) else np.zeros(0, np.int64)
    out = [a[first]]
    if return_index:
        out.append(first)
    if return_counts:
        out.append(np.diff(np.r_[first, len(a)]))
    return out[0] if len(out) == 1 else tuple(out)


# ============================================================================ binning
def _next_off(ed):
    """per detector event: index of the first OFF at or after it (any channel; len = none) -- the searchsorted of the
    event index in the OFF positions, by one backward running minimum."""
    n = len(ed)
    idx = np.where(np.asarray(ed) == 81, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1] if n else idx


def _cont_all(P_, T):
    """(end, start, index of the ending OFF or -1) of every detector event, computed once per window."""
    def f():
        return _continuous(P_.td, P_.ed, P_.pd_, T, _next_off(P_.ed))
    return _memo(P_, ("cont", float(T)), f)


def cont_on(P_, T):
    """continuous_on of a Prep's detector events, computed once per window (read only)."""
    end, start, _ = _cont_all(P_, T)
    return end, start


def _cont_off(P_, T):
    """per detector event: index of the OFF that ends its continuous ON (-1: still ON at the window end)."""
    return _cont_all(P_, T)[2]


def _continuous(td, ed, pd_, T, nxt):
    n = len(td)
    pc = np.minimum(nxt, max(n - 1, 0))
    ok = (nxt < n) & (pd_[pc] == pd_) if n else np.zeros(0, bool)
    end = np.where(ok, td[pc] if n else T, T)
    prev_on = np.r_[False, (pd_[1:] == pd_[:-1]) & (ed[:-1] == 82)]
    start = (ed == 82) & ~prev_on
    return end, start, np.where(ok, pc, -1)


def continuous_on(td, ed, pd_, T):
    """Detector events sorted by (channel, time, ON before OFF).  Returns, per event, the time of the
    next OFF of the same channel (T = still ON at the window end) and whether the event is an ON that
    STARTS a continuous ON (its previous event on the channel is not an ON).  ONs inside a continuous
    ON (ON -> ON, no OFF between) are normal with extension and belong to the ON that started it."""
    end, start, _ = _continuous(td, ed, pd_, T, _next_off(ed))
    return end, start


def events_to_bins(ev: pd.DataFrame, start, end, detectors=None, bin_s: int = BIN_S) -> dict:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if detectors is None and isinstance(ev, Prep):
        return _memo(ev, ("bins", start, end, int(bin_s)), lambda: _events_to_bins(ev, start, end, None, bin_s))
    return _events_to_bins(ev, start, end, detectors, bin_s)


def twins_of(B: dict):
    """twins() of a bins dict (row x row, only inside a signal), computed once per bins dict."""
    if "_tw" not in B:
        B["_tw"] = twins_rows(B)
    return B["_tw"]


def on_view(P_, T):
    """_on_view over every detector key of the Prep, computed once per window."""
    return _memo(P_, ("onview", float(T)), lambda: _on_view(P_, T, _memo(P_, "chans", lambda: sorted_unique(P_.pd_)),
                                                             True))


def _on_view(P_, T, dets, all_keys=False):
    """the per-ON view behind events_to_bins (independent of the bin size): ON times, channel rows, continuous-ON
    durations (NaN inside a continuous ON or over a comms gap), chatter flags, the longest ON per channel and the
    channels that were ON when the window opened."""
    td, ed, pd_ = P_.td, P_.ed, P_.pd_
    same_prev = np.r_[False, pd_[1:] == pd_[:-1]]
    prv_t, prv_e = np.r_[np.nan, td[:-1]], np.r_[0, ed[:-1]]
    on = ed == 82
    # continuous ON: from the ON that starts it to the next OFF (ONs inside it are extension re-calls);
    # still ON at the window end -> to the end.  Inner ONs get NaN (their time is already counted).
    c_end, c_start = cont_on(P_, T)
    dur = np.where(c_start, c_end - td, np.nan)[on]
    gap = np.where(same_prev & (prv_e == 81), td - prv_t, np.nan)[on]
    if all_keys:                                          # dets = every key, sorted: the row is the key's rank
        rank = np.cumsum(np.r_[True, pd_[1:] != pd_[:-1]]) - 1 if len(pd_) else np.zeros(0, np.int64)
        ton, ci = td[on], rank[on].astype(int)
    else:
        ton, ci = td[on], np.searchsorted(dets, pd_[on]).astype(int)
    nd = len(dets)
    dur_max = np.zeros(nd)
    chat = np.zeros(0)
    if len(ton):
        long_ = np.where(dur > 60)[0]                     # an ON over a comms gap is not "stuck"
        if len(long_) and len(P_.g0):
            hit = spans_gap(P_, pd_[on][long_] // BASE, ton[long_], ton[long_] + dur[long_])
            dur[long_[hit]] = np.nan
        with np.errstate(invalid="ignore"):
            chat = (_ticks(gap) < CHAT_TICKS).astype(float)
        # longest ON per channel (ci is sorted: the events are ordered by channel)
        dd = np.nan_to_num(dur, nan=0.0)
        u, first = sorted_unique(ci, return_index=True)
        dur_max[u] = np.maximum(np.maximum.reduceat(dd, first), 0.0)
    first = np.r_[True, pd_[1:] != pd_[:-1]]
    lead = first & (ed == 81)
    return dict(ton=ton, ci=ci, dur=dur, chat=chat, dur_max=dur_max, lead_t=td[lead], lead_c=pd_[lead])


def _on_bins(V, T):
    """5-min bin index of every ON and the start / end of every continuous ON (the occupancy intervals), computed once
    per window for every bin size that is a multiple of 5 min."""
    ton, ci, dur = V["ton"], V["ci"], V["dur"]
    fin = np.isfinite(dur)
    s, f = ton[fin], np.minimum(ton[fin] + dur[fin], T)
    on = (ton // BIN_S).astype(np.int64)
    return dict(on=on, s=s, f=f, c=ci[fin], s5=on[fin], f5=(f // BIN_S).astype(np.int64))


def coverage(P_, nb, bin_s):
    """bins of bin_s seconds with any event of the signal (signals x bins): a bin [k bin_s, (k + 1) bin_s) is covered
    when one of the signal's event times falls in it (the bin index t // bin_s, compared exactly)."""
    def f():
        if getattr(P_, "covs", None) is not None:
            return P_.covs(nb, bin_s)
        if getattr(P_, "ut", None) is not None:
            edges = np.arange(nb + 1, dtype=np.float64) * bin_s
            us = P_.usig
            out = np.zeros((P_.S, nb), bool)
            for s_ in range(P_.S):
                u = P_.ut[np.searchsorted(us, s_, "left"):np.searchsorted(us, s_, "right")]
                out[s_] = np.diff(np.searchsorted(u, edges, "left")) > 0
            return out
        idx = (P_.t // bin_s).astype(int)
        ok = idx < nb
        return (np.bincount(P_.tsig[ok] * nb + idx[ok], minlength=P_.S * nb)[:P_.S * nb] > 0).reshape(P_.S, nb)
    return _memo(P_, ("cov", int(nb), int(bin_s)), f)


def _events_to_bins(ev, start, end, detectors=None, bin_s: int = BIN_S) -> dict:
    T = (end - start).total_seconds()
    nb = int(np.ceil(T / bin_s))
    P_ = _prep(ev, start, end)
    cov = coverage(P_, nb, bin_s)
    chans = _memo(P_, "chans", lambda: sorted_unique(P_.pd_))   # pd_ = the 81 / 82 detector keys, sorted
    if detectors is not None:
        chans = sorted(set(chans.tolist()) | {int(x) for x in detectors})
    dets = np.asarray(chans, dtype=int)
    nd = len(dets)
    idx = {int(c): i for i, c in enumerate(dets)}
    out = {k: np.zeros((nd, nb), np.float32) for k in ("n_on", "occ", "n_chat")}
    V = on_view(P_, T) if detectors is None else _on_view(P_, T, dets)
    ton, ci, dur = V["ton"], V["ci"], V["dur"]
    if len(ton):
        # bin indices: t // bin_s; for a multiple of 5 min from the 5-min index (floor(floor(t / 300) / k) is exact)
        k5 = int(bin_s) // BIN_S if int(bin_s) % BIN_S == 0 and detectors is None else 0
        Q = _memo(P_, ("onbins", float(T)), lambda: _on_bins(V, T)) if k5 else None
        b0 = np.minimum(Q["on"] // k5 if k5 else (ton // bin_s).astype(int), nb - 1)
        cell = ci * nb + b0
        out["n_on"] = np.bincount(cell, minlength=nd * nb).reshape(nd, nb).astype(np.float32)
        out["n_chat"] = np.bincount(cell, weights=V["chat"], minlength=nd * nb).reshape(nd, nb).astype(np.float32)
        # occupancy, split across bins
        if k5:
            s, f, c = Q["s"], Q["f"], Q["c"]
            bs, bf = Q["s5"] // k5, np.minimum(Q["f5"] // k5, nb - 1)
        else:
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
    out.update(dets=dets, cov=cov, rsig=dets // BASE, S=P_.S, hour=hrs.astype(np.int32), dur_max=V["dur_max"].copy(),
               start=start, bin_s=bin_s, listed=None if detectors is None else np.isin(dets, [int(x) for x in detectors]))
    return out


def colour_events(P_):
    """the colour events (1 / 8 / 9 / 10) sorted by (phase key, time, code), with each phase key's slice; computed
    once per window."""
    def f():
        if getattr(P_, "csorted", False):             # colour events only, already in this order (the health query)
            ts, es, ps = P_.t, P_.eid, P_.par
        else:
            cm = np.isin(P_.eid, COLOUR)
            tc, ec, pc = P_.t[cm], P_.eid[cm], P_.par[cm]
            o = np.lexsort((ec, tc, pc))
            ts, es, ps = tc[o], ec[o], pc[o]
        up, f0, cnt = sorted_unique(ps, return_index=True, return_counts=True)
        return dict(t=ts, e=es, p=ps, up=up, f0=f0, cnt=cnt,
                    pos={int(p): (int(a), int(a + m)) for p, a, m in zip(up, f0, cnt)})
    return _memo(P_, "colour", f)


# ============================================================================ batch layout helpers
def norm_bins(B: dict) -> dict:
    """a bins dict made by hand for one signal (1-D cov, no rsig) -> the batch form (cov signals x bins, rsig)."""
    if "rsig" not in B:
        B["rsig"] = np.zeros(len(B["dets"]), np.int64)
        B["cov"] = np.atleast_2d(np.asarray(B["cov"], bool))
        B["S"] = 1
    return B


def layout(B: dict) -> dict:
    """rows of a bins dict per signal: rows of signal s = ptr[s]:ptr[s + 1] (rows are ordered by signal), local row
    index li, the largest signal's row count m; pad(x) puts row values into a signals x m (x ...) array."""
    if "_lay" in B:
        return B["_lay"]
    norm_bins(B)
    rs = np.asarray(B["rsig"], np.int64)
    S = int(B["cov"].shape[0])
    cnt = np.bincount(rs, minlength=S)
    ptr = np.r_[0, np.cumsum(cnt)]
    li = np.arange(len(rs)) - ptr[rs]
    m = int(cnt.max()) if len(cnt) and len(rs) else 0
    valid = np.zeros((S, max(m, 1)), bool)
    valid[rs, li] = True
    B["_lay"] = dict(rs=rs, li=li, ptr=ptr, cnt=cnt, S=S, m=max(m, 1), valid=valid)
    return B["_lay"]


def pad(L, x, fill=0.0):
    x = np.asarray(x)
    out = np.full((L["S"], L["m"]) + x.shape[1:], fill, dtype=np.result_type(x.dtype, np.asarray(fill).dtype))
    out[L["rs"], L["li"]] = x
    return out


def _rows_of_signal(L, s):
    return np.arange(L["ptr"][s], L["ptr"][s + 1])


def twins_rows(B, tol=0.1):
    """pairs of channels of the same signal carrying (almost) the same actuations = one zone wired to two inputs:
    5-min counts (covered bins) differ by < tol of their sum.  Returns signals x m x m bool (local rows)."""
    L = layout(B)
    n = B["n_on"].astype(float)
    x = np.where(B["cov"][L["rs"]], n, 0.0)
    tot = x.sum(1)
    xp, tp = pad(L, x), pad(L, tot)
    S, m = L["S"], L["m"]
    tw = np.zeros((S, m, m), bool)
    for i in range(m):                                  # all signals' i-th rows at once
        dd = np.abs(xp - xp[:, i:i + 1, :]).sum(2) / np.maximum(tp + tp[:, i:i + 1], 1)
        tw[:, i, :] = (dd < tol) & (tp > 0) & (tp[:, i:i + 1] > 0) & L["valid"]
    tw &= L["valid"][:, :, None]
    idx = np.arange(m)
    tw[:, idx, idx] = False
    return tw


def twins(n, cov, tol=0.1):
    """pairs of channels carrying (almost) the same actuations = one zone wired to two inputs (one signal):
    5-min counts differ by < tol of their sum.  Returns a bool matrix."""
    B = dict(dets=np.arange(len(n)), n_on=np.asarray(n), cov=np.asarray(cov, bool))
    return twins_rows(B, tol)[0][:len(n), :len(n)]


def refs_rows(B, live, phase_r):
    """reference of each detector (padded signals x m x m, local rows): the other live detectors on its (predicted)
    phase when there are >= 2 of them, else the rest of the signal's live detectors; a twin (same zone on another
    input) is never a reference.  phase_r: phase key per row (NaN = none).  Also the kind per row."""
    L = layout(B)
    m = L["m"]
    lp = pad(L, live, False) & L["valid"]
    pp = pad(L, np.asarray(phase_r, float), np.nan)
    eye = np.eye(m, dtype=bool)[None]
    ok = lp[:, None, :] & ~eye & ~twins_of(B)
    same = ok & (pp[:, :, None] == pp[:, None, :]) & np.isfinite(pp)[:, :, None]
    isph = same.sum(2) >= 2                     # one sibling is too fragile: if it fails, this one looks bad
    ref = np.where(isph[:, :, None], same, ok)
    kind = np.where(isph[L["rs"], L["li"]], "phase", "signal")
    return ref, kind


def ref_list(L, ref, i):
    """global row indices of row i's reference rows (padded ref)."""
    s = L["rs"][i]
    return L["ptr"][s] + np.flatnonzero(ref[s, L["li"][i]])


def ref_sum(L, ref, X):
    """sum over each row's reference rows of X (rows x k; integer-valued: exact in any order) -> rows x k."""
    Xp = pad(L, X)
    return np.matmul(ref.astype(float), Xp)[L["rs"], L["li"]]


# ======================================================================== statistics


def _agg(a, k=AGG):
    n = a.shape[-1] // k * k
    return a[..., :n].reshape(*a.shape[:-1], -1, k).sum(-1)


def _runs(mask):
    """start, end (exclusive) of True runs in a 1-D bool array."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def best_runs(quiet, covr, S, n_on):
    """per row: the silent run (True run of `quiet`) with the most expected ONs lam = n_on / max(ONs of the reference
    outside the run, 1) x reference ONs inside it (reference S counted on covered bins only); the first run with the
    largest lam > 0 -> (lam, start, end), else (0, -1, -1).  Rows with n_on = 0 have none."""
    R, nb = quiet.shape
    lam = np.zeros(R)
    b0 = np.full(R, -1, np.int64)
    b1 = np.full(R, -1, np.int64)
    if not R:
        return lam, b0, b1
    d = np.diff(np.c_[np.zeros((R, 1), np.int8), quiet.astype(np.int8), np.zeros((R, 1), np.int8)], axis=1)
    r0, a = np.nonzero(d == 1)
    r1, b = np.nonzero(d == -1)
    keep = n_on[r0] > 0
    r0, a, b = r0[keep], a[keep], b[keep]
    if not len(r0):
        return lam, b0, b1
    cS = np.c_[np.zeros(R), np.cumsum(np.where(covr, S, 0), 1)]
    s_in = cS[r0, b] - cS[r0, a]
    s_out = cS[r0, -1] - s_in
    lr = n_on[r0] / np.maximum(s_out, 1) * s_in
    pos = lr > 0
    r0, a, b, lr = r0[pos], a[pos], b[pos], lr[pos]
    if not len(r0):
        return lam, b0, b1
    mx = np.full(R, -np.inf)
    np.maximum.at(mx, r0, lr)
    hit = lr == mx[r0]
    rr, first = np.unique(r0[hit], return_index=True)
    j = np.flatnonzero(hit)[first]
    lam[rr], b0[rr], b1[rr] = lr[j], a[j], b[j]
    return lam, b0, b1


def _silent_rows(B, L, rows, S, n_on):
    """_silent for the given rows at once: drop_lam, drop_b0, drop_b1, drop_to_end, co_silent."""
    n, occ, cov = B["n_on"].astype(float), B["occ"].astype(float), B["cov"]
    nb = n.shape[1]
    rs = L["rs"][rows]
    covr = cov[rs]
    quiet = (n[rows] == 0) & (occ[rows] < 1) & covr
    lam, b0, b1 = best_runs(quiet, covr, S, n_on)
    # drop_to_end: the run reaches the last bin, or no covered bin after it
    last = np.array([np.flatnonzero(c)[-1] if c.any() else -1 for c in cov])
    to_end = (b1 == nb) | ((b1 > 0) & (last[rs] < b1))
    co = np.zeros(len(rows), np.int64)
    sel = np.flatnonzero(b1 - b0 >= 3)
    if len(sel):
        active = (n.sum(1) > 0) | (occ.sum(1) > 0)
        Q = (n == 0) & (occ < 1)
        Qc = pad(L, np.c_[np.zeros(len(n), np.int64), np.cumsum(Q, 1)])
        ap = pad(L, active, False)
        s_ = rs[sel]
        cnt = Qc[s_, :, b1[sel]] - Qc[s_, :, b0[sel]]
        q = cnt / (b1[sel] - b0[sel])[:, None] >= 0.8
        co[sel] = (q & ap[s_]).sum(1) - 1
    return dict(drop_lam=lam, drop_b0=b0, drop_b1=b1, drop_to_end=to_end, co_silent=co)


def drop_stats(B: dict, refs) -> pd.DataFrame:
    """the silent-run columns of det_stats (drop_lam, drop_b0, drop_b1, drop_to_end, co_silent) for the rows `rows`
    against their reference rows `ref` (padded), indexed by row (note 128: the phase-reference pass needs nothing
    else).  refs = (rows, ref)."""
    rows, ref = refs
    L = layout(B)
    n = B["n_on"].astype(float)
    S = ref_sum(L, ref, n)[rows]
    n_on = (n[rows] * B["cov"][L["rs"][rows]]).sum(1)
    return pd.DataFrame(_silent_rows(B, L, rows, S, n_on), index=rows)


def det_stats(B: dict) -> pd.DataFrame:
    """per detector against the rest of the signal (computed once per bins dict): ONs, the silent run, the level
    shift of its share (15-min bins, best single change point, Poisson LLR)."""
    if "_ds" not in B:
        B["_ds"] = _det_stats(B)
    return B["_ds"].copy()


def _sig_sum(L, X):
    """per row: the sum of X over all rows of its signal (integer-valued: exact)."""
    return np.add.reduceat(X, L["ptr"][:-1][L["cnt"] > 0], axis=0)[np.cumsum(L["cnt"] > 0)[L["rs"]] - 1] \
        if len(X) else X


def _det_stats(B: dict) -> pd.DataFrame:
    L = layout(B)
    n, occ, cov = B["n_on"].astype(float), B["occ"].astype(float), B["cov"]
    nd, nb = n.shape
    rs = L["rs"]
    covr = cov[rs]
    cov_h = (cov.sum(1) * B["bin_s"] / 3600)[rs]
    active = (n.sum(1) > 0) | (occ.sum(1) > 0)
    n_on = (n * covr).sum(1)
    S = _sig_sum(L, n) - n
    out = {"detector": B["dets"].astype(int), "n_on": n_on, "cov_h": cov_h,
           "n_sib": (_sig_sum(L, active.astype(np.int64)) - active).astype(int), "dur_max": B["dur_max"]}
    out.update(_silent_rows(B, L, np.arange(nd), S, n_on))
    out["level_ratio"], out["level_llr"] = np.full(nd, np.nan), np.zeros(nd)
    lb = np.full(nd, np.nan)
    n15 = _agg(n)
    cov15 = _agg(cov.astype(float)) == AGG                       # signals x 15-min bins
    S15 = _sig_sum(L, n15) - n15
    c15 = cov15[rs]
    ln = c15.sum(1)
    pos = np.cumsum(c15, 1) - 1                                  # compacted position of each covered bin
    W = int(ln.max()) if nd else 0
    xs = np.zeros((nd, max(W, 1)))
    ss = np.zeros((nd, max(W, 1)))
    r_, b_ = np.nonzero(c15)
    xs[r_, pos[r_, b_]] = n15[r_, b_]
    ss[r_, pos[r_, b_]] = S15[r_, b_]
    xsum = np.where(np.arange(xs.shape[1])[None] < ln[:, None], xs, 0).sum(1)
    ok = (ln >= 8) & (xsum >= 20)
    if ok.any():
        rr = np.flatnonzero(ok)
        cx, cs = np.cumsum(xs[rr], 1), np.cumsum(ss[rr], 1)
        X, SS = cx[np.arange(len(rr)), ln[rr] - 1], cs[np.arange(len(rr)), ln[rr] - 1]
        k = np.arange(2, xs.shape[1] - 1) if xs.shape[1] > 3 else np.zeros(0, int)
        valid = k[None, :] < (ln[rr] - 1)[:, None]
        x1, s1_ = cx[:, k - 1], cs[:, k - 1]
        x2, s2 = X[:, None] - x1, SS[:, None] - s1_
        p0 = X / np.maximum(SS, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            def ll(xx, ss_):
                p = np.where(ss_ > 0, xx / np.maximum(ss_, 1e-9), 0)
                return np.where(xx > 0, xx * np.log(np.maximum(p, 1e-12)), 0) - p * ss_
            llr = ll(x1, s1_) + ll(x2, s2) - (X * np.log(np.maximum(p0, 1e-12)) - p0 * SS)[:, None]
        llr = np.where(valid, llr, -np.inf)
        j = np.argmax(llr, 1)
        ar = np.arange(len(rr))
        out["level_llr"][rr] = llr[ar, j]
        out["level_ratio"][rr] = ((x2[ar, j] + .5) / np.maximum(s2[ar, j], 1)) / ((x1[ar, j] + .5) /
                                                                                 np.maximum(s1_[ar, j], 1))
        lb[rr] = k[j] * AGG
    df = pd.DataFrame(out)
    if np.isfinite(lb).any():
        df["level_b"] = lb.astype(np.int64) if np.isfinite(lb).all() else lb
    return df


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


def pairs_in_signal(sig):
    """all ordered pairs (i, j), i != j, of items of the same signal (sig ascending), i-major."""
    sig = np.asarray(sig)
    n = len(sig)
    if not n:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    u, first, cnt = sorted_unique(sig, return_index=True, return_counts=True)
    k = np.repeat(np.arange(len(u)), cnt)
    m = cnt[k]
    ii = np.repeat(np.arange(n), m)
    jj = np.repeat(first[k], m) + (np.arange(len(ii)) - np.repeat(np.cumsum(m) - m, m))
    keep = ii != jj
    return ii[keep], jj[keep]


def distinct_count(ii, vals, mask, n):
    """per item i: the number of distinct vals[j] over its pairs (i, j) with mask."""
    if not len(ii) or not mask.any():
        return np.zeros(n, int)
    a, v = ii[mask], vals[mask]
    u = np.unique(a * (int(v.max()) + 1) + v)
    return np.bincount(u // (int(v.max()) + 1), minlength=n)[:n].astype(int)


def _on_episodes(ev, start, end, min_s: float = 900.0) -> pd.DataFrame:
    T = (end - start).total_seconds()
    P_ = _prep(ev, start, end)
    td, ed, pd_ = P_.td, P_.ed, P_.pd_
    if isinstance(ev, Prep):
        end_t, start_ = cont_on(P_, T)
        po = _cont_off(P_, T)
    else:
        end_t, start_ = continuous_on(td, ed, pd_, T)
        P_.cache = {}
        po = _cont_off(P_, T)
    # continuous ONs held >= the base length; ONs inside each: ON count between its start and its OFF on the same
    # channel (the position of the ending OFF = the first event of its channel at that time; still ON = the end of
    # the channel's events)
    n = len(td)
    m = np.flatnonzero(start_ & (end_t - td >= min(CO_MIN_S, min_s)))
    if len(m):
        ci = np.cumsum(ed == 82)
        p = po[m]
        gp = np.maximum(p, 0)
        while True:                                       # step back over events of the channel at the same time
            back = (p >= 0) & (gp > 0)
            back[back] &= (pd_[gp[back] - 1] == pd_[gp[back]]) & (td[gp[back] - 1] == td[gp[back]])
            if not back.any():
                break
            gp = np.where(back, gp - 1, gp)
        offp = np.where(p >= 0, gp, np.searchsorted(pd_, pd_[m], "right"))
        nin = ci[np.minimum(offp, n) - 1] - ci[m] + 1
    else:
        nin = np.zeros(0, int)
    s, f, c, nf, oe = td[m], end_t[m], pd_[m], nin, end_t[m] >= T
    if len(P_.g0) and len(s):
        hit = spans_gap(P_, c // BASE, s, f)
        s, f, c, nf, oe = s[~hit], f[~hit], c[~hit], nf[~hit], oe[~hit]
    # other detectors (same signal) held ON (>= 5 min) through at least half of the same time: a shared event
    ii, jj = pairs_in_signal(c // BASE)
    ov = np.minimum(f[ii], f[jj]) - np.maximum(s[ii], s[jj])
    co = distinct_count(ii, c[jj], (ov >= 0.5 * (f - s)[ii]) & (c[ii] != c[jj]), len(s))
    k = (f - s) >= min_s
    return pd.DataFrame({"detector": c[k].astype(int), "t0": start + pd.to_timedelta(s[k], unit="s"),
                         "t1": start + pd.to_timedelta(f[k], unit="s"), "dur_s": (f - s)[k], "n_on": nf[k],
                         "open_end": oe[k], "co_stuck": co[k]})


def _recovery(B: dict, i: int, a: int, b: int, ref) -> float:
    """share of the reference in the 6 h after bins [a, b) / in the 6 h before; NaN when either
    side has < 3 h of data.  ref None = the rest of the signal."""
    L = layout(B)
    n, cov = B["n_on"], B["cov"][L["rs"][i]]
    rows = _rows_of_signal(L, L["rs"][i])
    nb = n.shape[1]

    def sh(lo, hi):
        c = cov[lo:hi]
        x = n[i, lo:hi][c].sum()
        S = (n[rows, lo:hi].sum(0) - n[i, lo:hi] if ref is None else n[ref, lo:hi].sum(0))[c].sum()
        return (x + 0.5) / max(S, 1.0), int(c.sum())

    if a < REC_MIN_BINS or nb - b < REC_MIN_BINS:
        return np.nan
    s0, c0 = sh(max(a - REC_BINS, 0), a)
    s1, c1 = sh(b, min(b + REC_BINS, nb))
    return s1 / s0 if c0 >= REC_MIN_BINS and c1 >= REC_MIN_BINS else np.nan


def _counts15(B: dict):
    """15-min counts of every detector and the fully covered 15-min bins of every signal (signals x bins), computed
    once per window (cached in B; the coverage mask is returned as a copy because callers edit it)."""
    if "_n15" not in B:
        norm_bins(B)
        B["_n15"] = _agg(B["n_on"].astype(float))
        B["_ok15"] = _agg(B["cov"].astype(float)) == AGG
    return B["_n15"], B["_ok15"].copy()


def _corr2(B, i, j, okkey, x, y):
    """np.corrcoef(x, y)[0, 1], cached per (row, row, mask) in B (both passes ask the same pairs)."""
    c = B.setdefault("_corr", {})
    k = (i, j, okkey)
    if k not in c:
        c[k] = float(np.corrcoef(x, y)[0, 1])
    return c[k]


def _partner(B: dict, i: int, cand, a: int, b: int):
    """(b): the same-phase sibling whose 15-min counts tracked detector i best outside bins [a, b)
    (corr >= PARTNER_CORR), as a row index list, else None."""
    if cand is None or not len(cand):
        return None, np.nan
    n15, ok = _counts15(B)
    ok = ok[layout(B)["rs"][i]]
    ok[a // AGG: -(-b // AGG)] = False
    if ok.sum() < 16:
        return None, np.nan
    best, bc, bk = None, -1.0, None
    w = np.arange(1, int(ok.sum()) + 1, dtype=float)
    x = n15[i][ok]
    xsd = x.max() != x.min()
    for j in cand:
        y = n15[j][ok]
        if xsd and y.max() != y.min():
            c = _corr2(B, int(i), int(j), (a // AGG, -(-b // AGG)), x, y)
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
           phase_conf: dict | None = None, signals=None) -> pd.DataFrame:
    """Per-detector stage-1 health table over [start, end) for every signal of the Prep (keys: see the module text).
    Columns: detector, health_score, status, reason, bad_periods (list of {start, end, what, recovered}), the rule
    scores s_* and the statistics behind them.  `signals` (optional): only these signals' rows are returned (the
    others are not computed beyond the shared bins).  `lanes` / `phase_conf` are accepted for compatibility and not
    used."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    events_df = _prep(events_df, start, end)
    B = events_to_bins(events_df, start, end, None)
    if len(B["dets"]) == 0:
        return pd.DataFrame(columns=["detector", "health_score", "status", "reason", "bad_periods"])
    L = layout(B)
    dets = B["dets"]
    rs, rch = L["rs"], dets % BASE
    nr = len(dets)
    n, cov = B["n_on"].astype(float), B["cov"]
    covr = cov[rs]
    live = (n * covr).sum(1) > 0
    phase = {int(k): v for k, v in (phase or {}).items() if v is not None and np.isfinite(v)}
    phv = np.array([phase.get(int(d), np.nan) for d in dets], float)
    keep = np.ones(nr, bool) if signals is None else np.isin(rs, list(signals))
    ref, kind = refs_rows(B, live, phv)
    # silence is expected from the other detectors on the predicted phase when there are >= 2
    ph2 = np.flatnonzero(kind == "phase")
    st = det_stats(B)
    st["drop_ref"] = "signal"
    if len(ph2):
        sp = drop_stats(B, (ph2, ref))
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
    isref = kind == "phase"
    rref = [ref_list(L, ref, i) if isref[i] else None for i in range(nr)]
    # recovery partner candidates: every other live detector on the predicted phase, twins included
    lp = pad(L, live, False)
    pp = pad(L, phv, np.nan)
    eye = np.eye(L["m"], dtype=bool)[None]
    pc_m = lp[:, None, :] & ~eye & (pp[:, :, None] == pp[:, None, :]) & np.isfinite(pp)[:, :, None]
    pcand = [list(ref_list(L, pc_m, i)) if np.isfinite(phv[i]) else [] for i in range(nr)]
    # night count against the detector it tracks (best 15-min correlation >= .7 with a same-phase sibling, twins
    # included), else the other detectors on its phase (>= 2), else the signal
    hr = B["hour"]
    ntm = cov & ((hr >= NIGHT_H[0]) | (hr < NIGHT_H[1]))[None]
    dym = cov & ((hr >= DAY[0]) & (hr < DAY[1]))[None]
    dy_r, nt_r = (n * dym[rs]).sum(1), (n * ntm[rs]).sum(1)
    sig_ok = (ntm.sum(1) >= 24) & (dym.sum(1) >= 24)
    rdy, rnt = ref_sum(L, ref, np.c_[dy_r, nt_r]).T
    sdy, snt = _sig_sum(L, np.c_[dy_r, nt_r]).T
    nd = {c: {} for c in ("night_ratio", "night_n", "night_exp", "night_ref")}
    for i in np.flatnonzero(keep & sig_ok[rs] & (dy_r >= 20)):
        pref, _ = _partner(B, i, pcand[i], 0, 0)
        if pref is not None:
            Rd, Rn, who = dy_r[pref[0]], nt_r[pref[0]], f"d{int(rch[pref[0]])}, which it tracks"
        elif rref[i] is not None:
            Rd, Rn, who = rdy[i], rnt[i], "the other detectors on its phase"
        else:
            Rd, Rn, who = sdy[i] - dy_r[i], snt[i] - nt_r[i], "the rest of the signal"
        exp_n = dy_r[i] / max(Rd, 1) * Rn
        if exp_n < NIGHT_MIN_EXP:
            continue
        xn = nt_r[i]
        nd["night_ratio"][i], nd["night_n"][i], nd["night_exp"][i] = (xn + .5) / (exp_n + .5), xn, exp_n
        nd["night_ref"][i] = who
    st = _add_cols(st, {c: pd.Series(v, dtype=object if c == "night_ref" else float).reindex(st.index)
                        for c, v in nd.items()})
    # ---- episodes: stuck-on (incl. ON with no OFF) and silent runs -> bad periods + recovery
    per = {int(d): [] for d in dets}
    ep = on_episodes(events_df, start, end)
    idx = {int(d): i for i, d in enumerate(dets)}
    nb = n.shape[1]
    best = {}
    held = {}
    for r in ep.itertuples():
        held.setdefault(int(r.detector), []).append((r.t0, r.t1))
    for r in ep.itertuples():
        i = idx.get(int(r.detector))
        if i is None or not keep[i]:
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
    d_rec, d_in = np.full(nr, np.nan), np.zeros(nr, bool)
    dl, db0, db1, dn = (st[c].to_numpy() for c in ("drop_lam", "drop_b0", "drop_b1", "n_on"))
    dte = st.drop_to_end.to_numpy()
    for j in np.flatnonzero(keep & (dn > 0) & (dl >= LIM["drop_lam"][0]) & (db1 > db0)):
        d = int(dets[j])
        r0 = start + _td(seconds=int(db0[j]) * B["bin_s"])
        r1 = start + _td(seconds=int(db1[j]) * B["bin_s"])
        ov = sum(max((min(p["end"], r1) - max(p["start"], r0)).total_seconds(), 0) for p in per[d])
        if ov >= 0.5 * (r1 - r0).total_seconds():
            d_in[j] = True
            continue
        pref, _ = _partner(B, j, pcand[j], int(db0[j]), int(db1[j]))
        rec = np.nan if dte[j] else _recovery(B, j, int(db0[j]), int(db1[j]), pref if pref is not None else rref[j])
        d_rec[j] = rec
        per[d].append(dict(start=r0, end=r1, what="silent",
                           recovered=None if not np.isfinite(rec) else bool(REC_RATIO[0] <= rec <= REC_RATIO[1])))
    epc.update(drop_rec=d_rec, drop_in_stuck=d_in)
    st = _add_cols(st, epc)
    if not keep.all():
        st = st[keep].reset_index(drop=True)
    h = rules(st, B)
    h["bad_periods"] = h.detector.map(lambda d: sorted(per.get(int(d), []), key=lambda x: x["start"]))
    return _merge_same(h, st, how="inner")
