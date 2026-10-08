"""Detector health v4 (research v4d, notes 104-121; adopted in note 124): per-detector status, score, plain reason,
bad periods, health categories with severity, configuration notes and a signal-level note, from the hi-res log and
the classifier's own answers (predicted phase, function, lane span).  numpy + pandas + DuckDB only.

    assess(prep, start, end, phase, fn_label, fn_prob, lanes, phase_conf) -> (per-detector DataFrame, signal note)

Layers (each a port of the research scorer, same rules, same limits):
  1  the rule checks of health_core (two passes: detectors pass 1 calls bad are nobody's yardstick in pass 2);
  2  per-type statistics from the log (health_v4_stats) against limits fitted ONCE on the training population
     (healthy detectors of the same type = function x lane span [x Count mode] x sample length; p99.5 / p99.8),
     shipped in weights/health/health_v4_refs.json;
  3  the resolver: queue / traffic / no-yardstick / borderline rules, the model-unsure rule (function probability
     < .7: least strict limit over the plausible types), the time-of-day checks (24-h samples), and the severity rule
     (two independent findings = bad; a problem lasting 4 h or more of the day = bad: busy at night, unusual daily
     pattern for >= 4 h, silent >= 4 h, unexplained time ON in >= 16 15-min periods; a Count zone held ON in light
     traffic = suspect even with extension).
Checks that need a whole day (busy at night, unusual daily pattern) run only on samples that contain a full day;
shorter samples get every other check with the limits of the nearest research sample length (30 min / 3 h / 24 h).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import health_core as hc
from .common import timedelta as _td
from . import health_v4_stats as hs

FNS = hs.FNS
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
REF_FILE = Path(__file__).resolve().parent / "weights" / "health" / "health_v4_refs.json"

# ---- constants of the research scorer (h108 / h110 / h118 / score_v4c / score_v4d)
BORDER, LOW_N = 0.42, 100
STUCK_CLIP, STUCK_BAD = (300.0, 900.0), 3600.0
Q2_CORR, Q1_CORR = 0.80, 0.70
D1_SHARE, D1_MIN_S = 0.70, 600.0
F5_TOP, F5_PLAUS = 0.70, 0.15
F4_COVER = 0.75
TRAF_X = 0.5
NO_SHARED = ("Advance", "Count", "Yellow_Red")
QUEUE_OK = {"Presence", "Other", "Mid"}
C1_N, C1_SHARE = 20, 0.05
C2_N60, C2_N5 = 1, 3
G3_MIN_OFF, G3_MIN_SC = 2, 6
SAT = 1800.0
ZF_FLOOR, SPK_FLOOR, EXC_FLOOR = 3.0, 2.0, 0.02
PROF_LEVEL = {"suspect": 0.50, "bad": 0.80}
NIGHT_SIL = (21, 5)
STAT = ("dropout", "rapid", "level", "choppy", "night_drop", "prof")
FAMILY = {"chatter": "fast", "rapid": "fast", "dropout": "silent", "stuck": "stuck", "level": "level",
          "night_drop": "level", "choppy": "shape", "volume": "shape", "prof": "shape", "occspk": "occ",
          "shape24": "shape", "count_on": "stuck"}
QUIET = {"dropout", "level", "night_drop"}
SILENT_FAM = {"dropout", "level", "night_drop"}
CATEGORIES = {
    "stuck": "Stuck on", "dropout": "Goes silent", "level": "Count drops", "night_drop": "Misses vehicles at night",
    "choppy": "Erratic counts", "rapid": "Too-fast actuations", "volume": "Too many for the traffic",
    "chatter": "Chattering", "occspk": "Erratic time ON", "prof": "Busy at night", "shape24": "Unusual daily pattern",
    "count_on": "Count zone held ON", "occ_hi": "ON longer than its kind",
    "C1": "Extension time on a count zone", "C2": "Set to pulse but holds ON"}
LANE_CATS = {"stuck", "choppy", "rapid", "volume", "chatter", "occspk", "prof", "shape24"}   # note 121
ORDER = ["stuck", "count_on", "dropout", "night_drop", "level", "chatter", "rapid", "volume", "choppy", "occspk",
         "prof", "shape24"]
OUT_COLS = ["detector", "health_status", "health_score", "health_reason", "health_bad_periods", "health_categories",
            "health_config", "health_watch"]


# ============================================================================ references
class Refs:
    """the frozen population tables (weights/health/health_v4_refs.json); `tod` / `band` hold the 24-h references
    used for the detector's own type (`*_own`) and for the plausible alternatives when the model is unsure (`*_alt`)."""

    def __init__(self, d: dict, tod_own=None, tod_alt=None, band_own=None, band_alt=None):
        self.d = d
        self.cell = {v: {c: self._levels(t) for c, t in d["cell"][v].items()} for v in d["cell"]}
        self.g3 = {v: self._levels(t) for v, t in d["g3"].items()}
        self.f117 = {c: self._levels(t) for c, t in d["fast117"].items()}
        self.f118 = {c: self._levels(t) for c, t in d["fast118c"].items()}
        self._occ = {(r[0], r[1], int(r[2]), int(r[3])): r[4] for r in d["occ_lim"]}
        self._occ_p = {(int(r[0]), int(r[1])): r[2] for r in d["occ_lim_pulse"]}
        self._tp = {int(r[0]): r[1] for r in d["count_t_pass"]}
        self.share_med = {(r[0], r[1]): r[2] for r in d["share_med"]}
        self.qtype = {r[0]: r[1] for r in d["q_type"]}
        self.tod_w = {(r[0], r[1], r[2], int(r[3])): r[4] for r in d["tod_w"]}
        self.tod_wb = {(r[0], r[1], int(r[2])): r[3] for r in d["tod_wb"]}
        self.tod_own = tod_own or d["tod"]
        self.tod_alt = tod_alt or d["tod"]
        self.band_own = band_own or d["band"]
        self.band_alt = band_alt or d["band"]

    @staticmethod
    def _levels(t):
        return [(tuple(lv["keys"]), {tuple(str(x) for x in r[:-1]): r[-1] for r in lv["rows"]}) for lv in t]

    @staticmethod
    def lookup(levels, D: pd.DataFrame) -> np.ndarray:
        """first finite value over the key levels (research `lookup`; keys compared as text, as astype(str))."""
        n = len(D)
        lim = np.full(n, np.nan)
        if not n:
            return lim
        sc = {}
        for keys, tab in levels:
            for k in keys:
                if k not in sc:
                    sc[k] = [str(v) for v in D[k].tolist()]
            m = np.array([tab.get(v, np.nan) for v in zip(*[sc[k] for k in keys])], float)
            fill = np.isnan(lim) & np.isfinite(m)
            lim[fill] = m[fill]
        return lim

    def occ_lim(self, fn, cmode, bs, tl):
        return np.array([self._occ.get((f, c, int(b), int(t)), np.nan) if (isinstance(f, str) and t == t) else np.nan
                         for f, c, b, t in zip(fn, cmode, bs, tl)], float)

    def occ_lim_pulse(self, bs, tl):
        return np.array([self._occ_p.get((int(b), int(t)), np.nan) if t == t else np.nan for b, t in zip(bs, tl)], float)

    def t_pass(self, bs):
        return np.array([self._tp.get(int(b), np.nan) for b in bs], float)


@lru_cache(maxsize=2)
def _load_refs(path: str = str(REF_FILE)) -> Refs:
    with open(path, encoding="utf-8") as f:
        return Refs(json.load(f))


def score(x, lo, hi):
    """0 below the limit, .35 at it, 1 at `hi` (research h108 score)."""
    x, lo, hi = (np.asarray(v, float) for v in (x, lo, hi))
    with np.errstate(invalid="ignore"):
        s = 0.35 + 0.65 * np.clip((x - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
        return np.where(~np.isfinite(x) | ~np.isfinite(lo), np.nan, np.where(x < lo, 0.0, s))


def window_group(hours: float) -> str:
    """research sample length whose limits apply: 30 min, 3 h or 24 h (nearest on a log scale)."""
    return "m30" if hours < np.sqrt(0.5 * 3) else ("h3" if hours < np.sqrt(3 * 24) else "h24")


def _cmode(fn, dur_p50):
    return np.where(np.asarray(fn) != "Count", "",
                    np.where(pd.isna(dur_p50), "unknown", np.where(np.asarray(dur_p50, float) <= 0.25, "pulse", "normal")))


def _alts(P):
    """plausible functions when the model is unsure (top probability < .7): classes with >= .15."""
    top = np.nanmax(np.where(np.isfinite(P), P, -1), 1) if len(P) else np.zeros(0)
    return [set(c for c, v in zip(C7, row) if np.isfinite(v) and v >= F5_PLAUS) if 0 <= t < F5_TOP else set()
            for t, row in zip(top, P)]


# ============================================================================ the resolver (research h110 / h118 / v4 / v4c)
def _alt_has(X) -> dict:
    """{class: bool per row} -- the class is among the row's plausible alternatives and is not its own label."""
    al = list(X.alt_set)
    fn = X.fn.to_numpy(object)
    return {fa: np.array([fa in a for a in al], bool) & (fn != fa) for fa in C7}


def _limits(X, refs, v, ah=None):
    """the per-type limits of the chatter / stuck / erratic-time-ON checks, loosened to the least strict plausible type
    when the model is unsure.  Returns {column: array} (no column is written into X)."""
    base = X[["fn", "span", "cmode", "wg"]]
    lim = {col: refs.lookup(refs.cell[v][col], base) for col in ("chat_frac", "stuck_x", "n3_exc")}
    low = (X.f_top < F5_TOP).to_numpy()
    ah = ah if ah is not None else _alt_has(X)
    for fa in C7:
        m = low & ah[fa]
        if not m.any():
            continue
        D = base[m].copy()
        D["fn"] = fa
        D["cmode"] = np.where(fa != "Count", "", np.where(X.loc[m, "dur_p50"].isna(), "unknown",
                                                          np.where(X.loc[m, "dur_p50"] <= 0.25, "pulse", "normal")))
        for col in ("chat_frac", "stuck_x", "n3_exc"):
            la = refs.lookup(refs.cell[v][col], D)
            cur = lim[col][m]
            with np.errstate(invalid="ignore"):
                looser = la > cur
            lim[col][np.flatnonzero(m)[looser]] = la[looser]
    return {f"lim_{col}": a for col, a in lim.items()}


def _kahan(vals):
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


def _night_frac_v(t0, b0, b1):
    """share of the 5-min bins [b0, b1) (from t0) whose clock hour is night (21:00-05:00), one value per row
    (clock hours in integer nanoseconds of the day)."""
    out = np.zeros(len(b0))
    if not len(b0):
        return out
    sod = int(pd.Timestamp(t0).value) % 86_400_000_000_000
    for i, (a, b) in enumerate(zip(b0, b1)):
        if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
            continue
        hrs = ((sod + np.arange(int(a), int(b), dtype=np.int64) * 300_000_000_000) // 3_600_000_000_000) % 24
        out[i] = float(np.mean((hrs >= NIGHT_SIL[0]) | (hrs < NIGHT_SIL[1])))
    return out


def _f0(a):
    """Series.fillna(0) of a float column, as an array."""
    return np.nan_to_num(np.asarray(a, float), nan=0.0)


def _resolve(X0, E0, refs, v):
    """one research resolver pass, v = 'v4' (score_v4.run_v4) or 'v4c' (score_v4c.run_c).  Returns X with the
    scores s8_*, status st8, rules8, watch8, cleared8, left8, score8, dq8, cfg.  (note 128: the new columns are built
    as arrays and joined once; the arithmetic is the research's, step for step.)"""
    X = X0.reset_index(drop=True)
    n = len(X)
    E = E0.copy()
    if len(E):
        E.loc[E.fn.isin(NO_SHARED), "co5"] = 0
    ah = _alt_has(X)
    C = _limits(X, refs, v, ah)
    base = X[["fn", "span", "cmode", "wg"]]
    # stuck: episodes over the per-type limit; their number has its own limit
    sl = np.clip(C["lim_stuck_x"], *STUCK_CLIP)
    S0 = _stuck_stats(X, E, sl)
    C["n_ep"] = S0.n_ep.fillna(0).to_numpy()
    lim_n_ep = refs.lookup(refs.cell[v]["n_ep"], base)
    low = X.alt_fns.ne("").to_numpy()
    for fa in C7:
        m = low & ah[fa]
        if m.any():
            la = refs.lookup(refs.cell[v]["n_ep"], base[m].assign(fn=fa))
            lim_n_ep[m] = np.fmax(lim_n_ep[m], la)
    C["lim_n_ep"] = lim_n_ep
    # ---- scores (h108 scores -> h110 scores110 -> h118 scores118 -> v4 / v4c)
    S = {}
    for k in ("dropout", "level", "night_drop"):
        S[k] = pd.to_numeric(X[f"s_{k}"], errors="coerce").to_numpy()
    S["choppy"] = np.full(n, np.nan)
    n50 = X.n_on.to_numpy(float) >= 50
    lcf = C["lim_chat_frac"]
    S["chatter"] = np.where(n50, score(X.chat_frac, lcf, 2 * lcf), np.nan)
    S["rapid"] = np.full(n, np.nan)
    S["volume"] = np.full(n, np.nan)
    C["stuck_lim8"] = sl
    S["stuck"] = score(X.stuck_x.fillna(0), sl, np.fmax(STUCK_BAD, 2 * sl))
    n3ok = (X.n3_n >= 2).to_numpy()
    n3_exc = X.n3_exc.to_numpy(float)
    l3e = C["lim_n3_exc"]
    S["occspk"] = np.where(n3ok, score(n3_exc, l3e, 4 * l3e), np.where(np.isfinite(n3_exc), 0.0, np.nan))
    rep_s = X.rep_time_s.to_numpy(float)
    with np.errstate(invalid="ignore"):
        n3_dq = (S["occspk"] >= .35) & (rep_s / 60 >= 0.5 * n3_exc)
    C["n3_dq"] = n3_dq
    S["occspk"] = np.where(n3_dq, 0.0, S["occspk"])
    S["prof"] = np.full(n, np.nan)
    cap = (pd.to_numeric(X.s_stuck, errors="coerce") == 0.35).to_numpy()
    with np.errstate(invalid="ignore"):
        S["stuck"] = np.where(cap & (S["stuck"] > .35), 0.35, S["stuck"])
    # F2 level
    ratio, llr = X.lv_ratio110.to_numpy(float), X.lv_llr110.to_numpy(float)
    vv = -np.log(np.maximum(np.nan_to_num(ratio, nan=1.0), 1e-9))
    s_lv = score(vv, -np.log(0.15), -np.log(0.05))
    S["level"] = np.where(np.isfinite(ratio), np.where(llr <= 50, 0.0, s_lv), np.nan)
    # F3 / F4 stuck (the same episodes and limits as above)
    St = {c: S0[c].to_numpy() for c in S0.columns}
    for c in S0.columns:
        C[f"st_{c}"] = St[c]
    tot = _f0(St["tot_s"])
    s_tot = score(tot, sl, np.fmax(STUCK_BAD, 2 * sl))
    s_n = score(_f0(St["n_ep"]), lim_n_ep + 1, 2 * (lim_n_ep + 1))
    s_ = np.fmax(s_tot, np.nan_to_num(s_n))
    single = _f0(St["n_ep"]) <= 1
    s_ = np.where(cap & single & (s_ > .35), .35, s_)
    q1_all = ((_f0(St["n_over"]) > 0) & (_f0(St["n_ep"]) == 0) & (_f0(St["n_shared_unq"]) == 0) &
              (_f0(St["n_q1"]) + _f0(St["n_q1b"]) > 0))
    C["q1_all"] = q1_all
    shared_only = (_f0(St["n_ep"]) == 0) & (_f0(St["n_shared_unq"]) > 0)
    stk = S["stuck"]
    with np.errstate(invalid="ignore"):
        s_ = np.where(shared_only, np.where(stk >= .35, .35, stk), s_)
    s_ = np.where(q1_all, 0.0, s_)
    S["stuck"] = s_
    # G3 erratic counts
    D3 = X[["fn", "span", "band", "wg"]]
    L3 = refs.lookup(refs.g3[v], D3)
    for fa in C7:
        m = low & ah[fa]
        if not m.any():
            continue
        la = refs.lookup(refs.g3[v], D3[m].assign(fn=fa))
        with np.errstate(invalid="ignore"):
            lo_ = la > L3[m]
        ii = np.flatnonzero(m)[lo_]
        L3[ii] = la[lo_]
    if v == "v4c":
        L3 = np.fmax(L3, EXC_FLOOR)
    C["lim_exc"] = L3
    ok3 = (X.exc_15.notna() & X.n_sc_15.ge(G3_MIN_SC) & X.ref110.ne("none")).to_numpy()
    s3 = score(X.exc_15, L3, 2 * L3)
    s3 = np.where(X.n_off_15.fillna(0) >= G3_MIN_OFF, s3, 0.0)
    S["choppy"] = np.where(ok3, s3, np.nan)
    with np.errstate(invalid="ignore"):
        C["noyard_chop"] = (X.ref110.eq("none") & (X.exc_sig >= L3) & (X.n_off_sig.fillna(0) >= G3_MIN_OFF) &
                            X.n_sc_sig.ge(G3_MIN_SC)).to_numpy()
    S["level"] = np.where(X.g4_unscored.fillna(False).to_numpy(bool), np.nan, S["level"])
    # ---- too fast / too many / busy at night
    adv = X.fn.eq("Advance").to_numpy()
    enough = X.n_on117.fillna(0).to_numpy() >= 50
    with np.errstate(invalid="ignore", divide="ignore"):
        if v == "v4":
            lz = refs.lookup(refs.f117["zf"], X[["fn", "span", "wg"]])
            lk = refs.lookup(refs.f117["n_spk"], X[["fn", "span", "wg"]])
            lg = np.fmax(refs.lookup(refs.f117["q5_gy"], X[["fn", "span", "wg"]]), SAT)
            la_ = np.fmax(refs.lookup(refs.f117["q5_all"], X[["fn", "span", "wg"]]), SAT)
            sc_ = lambda a, b: np.where(np.isfinite(a) & np.isfinite(b) & (b > 0), a / b, np.nan)  # noqa: E731
            x_zf = sc_(X.zf.to_numpy(float), lz)
            nspk = X.n_spk.to_numpy(float)
            x_spk = np.where(nspk > lk, nspk / np.fmax(lk, 1), 0)
            fx = np.fmax(np.nan_to_num(x_zf), x_spk)
            vol_x = np.where(adv, sc_(X.q5_all.to_numpy(float), la_), sc_(X.q5_gy.to_numpy(float), lg))
            C["lim_zf_use"], C["lim_spk_use"], C["lim_vol_use"] = lz, lk, np.where(adv, la_, lg)
            C["fo_use"], C["fem_use"], C["zf_use"], C["n_spk_use"] = X.fo_all, X.fem_all, X.zf, X.n_spk
        else:
            lz = _least_strict(refs.f118["zf_c"], X, ZF_FLOOR, ah)
            lk = _least_strict(refs.f118["n_spk_c"], X, SPK_FLOOR, ah)
            zfc = X.zf_c.to_numpy(float)
            x_zf = np.where(np.isfinite(zfc), zfc / lz, np.nan)
            nspk = X.n_spk_c.to_numpy(float)
            x_spk = np.where(nspk > lk, nspk / lk, 0.0)
            fx = np.where(X.n_on117.to_numpy(float) >= 50, np.fmax(np.nan_to_num(x_zf), x_spk), np.nan)
            lg = _least_strict(refs.f118["q5_gy"], X, None, ah)
            la_ = _least_strict(refs.f118["q5_all"], X, None, ah)
            lv = np.where(adv, np.fmax(la_, SAT), np.fmax(lg, SAT))
            q = np.where(adv, X.q5_all, X.q5_gy).astype(float)
            vol_x = np.where(np.isfinite(q) & (lv > 0), q / lv, np.nan)
            C["lim_zf_use"], C["lim_spk_use"], C["lim_vol_use"] = lz, lk, lv
            C["fo_use"], C["fem_use"], C["zf_use"], C["n_spk_use"] = X.fo_c, X.fem_c, X.zf_c, X.n_spk_c
    C["x_zf"], C["x_spk"], C["fast_x"], C["vol_x"] = x_zf, x_spk, fx, vol_x
    S["rapid"] = np.where(enough & np.isfinite(fx), score(fx, 1.0, 2.0), np.nan)
    S["volume"] = np.where(np.isfinite(vol_x), score(vol_x, 1.0, 2.0), np.nan)
    lvc = X.tod_level.fillna("not scored")
    pv = np.where(lvc.eq("not scored"), np.nan, lvc.map(PROF_LEVEL).fillna(0.0))
    dro = _f0(S["dropout"])
    S["prof"] = np.where((dro >= .35) & np.isfinite(pv), 0.0, pv)
    occ_hi8 = X.occ_hi8.to_numpy()
    changed_hi8 = False
    if v == "v4c":
        nf = _night_frac_v(X.t_start.iat[0] if n else None, X.drop_b0.astype(float).to_numpy(),
                           X.drop_b1.astype(float).to_numpy())
        silent = (dro >= .35) & (nf >= .8)
        C["silent_night"] = silent
        nd = np.asarray(S["night_drop"], float)
        S["night_drop"] = np.where(silent, np.fmax(np.nan_to_num(nd), dro), nd)
        S["dropout"] = np.where(silent, 0.0, S["dropout"])
        run = X.bd_ev_h.to_numpy(float)
        sh = np.where(X.bd_flag.fillna(False).astype(bool), score(run, 3.0, 7.0), 0.0)
        down = X.bd_ev_dir.to_numpy(float) < 0
        with np.errstate(invalid="ignore"):
            other = (_f0(S["level"]) >= .35) | (_f0(S["night_drop"]) >= .35)
            sh = np.where(down & other, 0.0, sh)
            sh = np.where(~down & (np.nan_to_num(S["prof"]) >= .35), 0.0, sh)
            dr = _f0(S["dropout"])
            both = down & (dr >= .35) & (sh >= .35)
            S["dropout"] = np.where(both & (sh >= dr), 0.0, np.where(both, np.fmax(dr, BORDER), S["dropout"]))
            sh = np.where(both & (sh < dr), 0.0, sh)
        S["shape24"] = np.where(X.bd_flag.notna().to_numpy(), sh, np.nan)
        hm = X.hi_min_c.fillna(0).to_numpy(float)
        S["count_on"] = np.where(X.count_on_ok, score(hm, 30.0, 240.0), np.nan)
        m_hi = X.count_on_ok.to_numpy() & (hm >= 30)
        if m_hi.any():
            occ_hi8 = occ_hi8.copy()
            occ_hi8[m_hi] = False
            changed_hi8 = True
    for k, a in S.items():
        C[f"s8_{k}"] = np.asarray(a)
    C["queue_pat8"] = ((X.elhi_occ > 0) & (X.elhi_occ > X.elhi_cnt)).to_numpy()
    # ---- the rules (h110 resolve)
    rows = []
    cols = list(S)
    Sv = np.column_stack([np.asarray(S[k], float) for k in cols]) if n else np.zeros((0, len(cols)))
    L = {c: X[c].tolist() for c in ("status2", "status", "c_occ_like", "n_on", "share_x", "hours", "n_hmates",
                                    "reason", "fn", "occ", "rep_time_s")}
    qp, q1a, nyc, n3d = C["queue_pat8"].tolist(), q1_all.tolist(), C["noyard_chop"].tolist(), n3_dq.tolist()
    sq1b, oh8 = pd.Series(St["n_q1b"]).tolist(), occ_hi8.tolist()
    for i in range(n):
        s = {k: Sv[i, j] for j, k in enumerate(cols) if np.isfinite(Sv[i, j])}
        find = {k: val for k, val in s.items() if val >= .35}
        notes, watch, fired = [], [], []
        status, c_occ_like = L["status"][i], L["c_occ_like"][i]
        if L["status2"][i] != status:
            fired.append("R9")
        for k in list(find):
            rule = None
            if k in ("choppy", "level", "occspk") and bool(qp[i]) and np.isfinite(c_occ_like) and \
                    c_occ_like >= Q2_CORR:
                rule = ("Q2", "clear")
            elif k == "rapid" and L["n_on"][i] < LOW_N and np.isfinite(L["share_x"][i]) and L["share_x"][i] >= 0.1:
                rule = ("R5", "watch")
            elif k == "dropout" and L["hours"][i] < 1:
                rule = ("R6", "watch")
            if rule:
                fired.append(rule[0])
                (watch if rule[1] == "watch" else notes).append(k)
                del find[k]
        if bool(q1a[i]):
            fired.append("Q1b" if (sq1b[i] or 0) > 0 else "Q1")
            notes.append("stuck")
        nh = L["n_hmates"][i]
        if "R9" in fired and np.isfinite(nh) and nh == 0 and status in ("suspect", "bad"):
            fired.append("Y")
            watch.append("no_yardstick")
        if bool(nyc[i]):
            fired.append("Y1")
            watch.append("no_yardstick_chop")
        if len(find) == 1:
            k, val = next(iter(find.items()))
            if k in STAT and val < BORDER:
                fired.append("R7")
                watch.append(k)
                del find[k]
        if "last 0.2 s or less" in str(L["reason"][i]) and L["fn"][i] == "Presence":
            fired.append("R8")
            watch.append("short_on")
        if bool(oh8[i]) and "occspk" not in find and "stuck" not in find and \
                not (np.isfinite(c_occ_like) and c_occ_like >= .5):
            fired.append("N1")
            watch.append("occ_hi")
        dq = []
        occ, rts = L["occ"][i], L["rep_time_s"][i]
        on_s = occ * L["hours"][i] * 3600 if np.isfinite(occ) else np.nan
        if np.isfinite(rts) and rts >= D1_MIN_S and np.isfinite(on_s) and rts >= D1_SHARE * on_s:
            dq.append("D1")
        if bool(n3d[i]):
            dq.append("D1n3")
        hsc = float(np.prod([1 - val for k, val in s.items() if k in find or val < .35])) if s else np.nan
        if np.isfinite(hsc) and hsc < .25:
            st = "bad"
        elif np.isfinite(hsc) and hsc < .70:
            st = "suspect"
        elif watch:
            st = "watch"
        elif L["n_on"][i] < 20:
            st = "not_enough_data"
        else:
            st = "ok"
        fams = {FAMILY[k] for k, val in find.items() if not (k in ("stuck", "dropout") and val == .35)}
        if st == "suspect" and len(fams) >= 2:
            st = "bad"
        rows.append(dict(st8=st, rules8=",".join(fired), watch8=",".join(watch), cleared8=",".join(notes),
                         left8=",".join(sorted(find)), score8=hsc, dq8=",".join(dq)))
    Xo = X.assign(occ_hi8=occ_hi8) if changed_hi8 else X
    R = pd.concat([Xo, pd.DataFrame(C, index=X.index), pd.DataFrame(rows, index=X.index)], axis=1)
    R["cfg"] = ""
    c1 = R.fn.eq("Count") & (R.dq8.str.contains("D1") | ((R.n_rep >= C1_N) & (R.n_rep >= C1_SHARE * R.n_on)))
    R.loc[c1, "cfg"] = "C1"
    c2 = R.fn.eq("Count") & R.cmode.eq("pulse") & ((R.n_ge60 >= C2_N60) | (R.n_ge5 >= C2_N5)) & ~R.cfg.str.contains("C1")
    R.loc[c2, "cfg"] = (R.loc[c2, "cfg"] + ",C2").str.strip(",")
    return R


def _least_strict(levels, X, floor, ah=None):
    D = X[["fn", "span", "wg"]]
    lim = Refs.lookup(levels, D)
    ah = ah if ah is not None else _alt_has(X)
    for f in C7:
        m = ah[f]
        if not m.any():
            continue
        la = Refs.lookup(levels, D[m].assign(fn=f))
        lim[m] = np.fmax(lim[m], la)
    if floor is not None:
        lim = np.fmax(lim, floor)
    return lim


_rows = hc.rows_of        # light row views (note 128)


PERSIST_H = 4.0          # note 121 rule P, adopted 2026-10-08 (user decision): a finding lasting >= 4 h of the day


def _has(R, k):
    return np.array([k in str(x).split(",") for x in R.left8.fillna("")], bool)


def persists(R):
    """score_v4d.persists: busy at night (01-05 h by construction), unusual daily pattern run >= 4 h, silent >= 4 h,
    unexplained time ON in >= 16 15-min periods."""
    silent_h = (pd.to_numeric(R.drop_b1, errors="coerce") - pd.to_numeric(R.drop_b0, errors="coerce")) * 300 / 3600
    bd_h = pd.to_numeric(R.bd_ev_h, errors="coerce") if "bd_ev_h" in R else pd.Series(np.nan, index=R.index)
    n3 = pd.to_numeric(R.n3_n, errors="coerce") if "n3_n" in R else pd.Series(np.nan, index=R.index)
    return (_has(R, "prof") | (_has(R, "shape24") & (bd_h >= PERSIST_H).to_numpy()) |
            (_has(R, "dropout") & (silent_h >= PERSIST_H).to_numpy()) |
            (_has(R, "occspk") & (n3 >= 4 * PERSIST_H).to_numpy()))


def persisting_categories(r):
    """the categories of one detector-row that lasted >= 4 h (shown as 'bad' when rule P made the detector bad)."""
    left = str(r.left8).split(",")
    out = set()
    if "prof" in left:
        out.add("prof")
    if "shape24" in left and np.isfinite(_f(r, "bd_ev_h")) and _f(r, "bd_ev_h") >= PERSIST_H:
        out.add("shape24")
    if "dropout" in left and np.isfinite(_f(r, "drop_b1") - _f(r, "drop_b0")) and             (_f(r, "drop_b1") - _f(r, "drop_b0")) * 300 / 3600 >= PERSIST_H:
        out.add("dropout")
    if "occspk" in left and np.isfinite(_f(r, "n3_n")) and _f(r, "n3_n") >= 4 * PERSIST_H:
        out.add("occspk")
    return out


def _f(r, c):
    v = pd.to_numeric(getattr(r, c, np.nan), errors="coerce")
    return float(v) if v is not None and np.isfinite(v) else np.nan


def severity_v4d(R):
    """score_v4d rule A (two independent findings -> bad), P (a finding lasting >= 4 h of the day -> bad; note 121,
    adopted 2026-10-08) and C (Count zone held ON by extension -> suspect)."""
    st = R.st8.copy()
    n_ev = []
    for r in _rows(R):
        ev = set()
        for k in str(r.left8).split(","):
            if not k:
                continue
            val = getattr(r, f"s8_{k}", np.nan)
            if not (np.isfinite(val) and val >= .35):
                continue
            if k in ("stuck", "dropout") and val == .35:
                continue
            quiet = k in QUIET or (k == "shape24" and r.bd_ev_dir < 0)
            ev.add("quiet" if quiet else k)
        n_ev.append(len(ev))
    R["n_ev"] = n_ev
    rule = np.full(len(R), "", object)
    a = st.eq("suspect").to_numpy() & (R.n_ev.to_numpy() >= 2)
    st[a], rule[a] = "bad", "A"
    p = st.eq("suspect").to_numpy() & ~a & persists(R)
    st[p], rule[p] = "bad", "P"
    c = (R.fn.eq("Count") & R.c1_pre.astype(bool) & (R.hi_min_c >= 30)).to_numpy() & st.isin(["ok", "watch"]).to_numpy()
    st[c], rule[c] = "suspect", "C"
    R["st_v4d"], R["sev_rule"] = st.to_numpy(), rule
    return R


# ============================================================================ main entry
def _stage1(P, start, end, phase, fprob, lanes, pconf, stage1=None):
    """package health_core rules, two passes (h96_health --twopass)."""
    f = stage1 or hc.health
    h1 = f(P, start, end, None, phase or None, fprob or None, lanes or None, pconf or None)
    if not len(h1):
        return h1, {}
    bad = set(h1.loc[h1.status == "bad", "detector"].astype(int))
    st1 = dict(zip(h1.detector.astype(int), h1.status))
    sc1 = dict(zip(h1.detector.astype(int), h1.health_score))
    x = h1
    if bad and phase:
        ph2 = {k: (np.nan if k in bad else v) for k, v in phase.items()}
        y = f(P, start, end, None, ph2, fprob or None, lanes or None, pconf or None)
        y = y[~y.detector.isin(bad)]
        x = pd.concat([h1[h1.detector.isin(bad)], y], ignore_index=True)
    x = x.copy()
    x["status1"] = x.detector.astype(int).map(st1)
    x["score1"] = x.detector.astype(int).map(sc1)
    return x, st1


def _left(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """X.merge(df, on="detector", how="left") for a df with one row per detector: its columns looked up by detector
    (reindex fills a missing detector exactly as the merge does) and joined in one concat."""
    if df.detector.duplicated().any() or (set(X.columns) & set(df.columns)) - {"detector"}:
        return X.merge(df, on="detector", how="left")
    R = df.set_index("detector").reindex(X.detector.to_numpy())
    R.index = X.index
    return pd.concat([X, R], axis=1)


def _add(X: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """X with new columns appended in order, one concat (scalars broadcast as a setitem would)."""
    return pd.concat([X, pd.DataFrame(cols, index=X.index)], axis=1)


def _day_type(ts) -> str:
    """the research day whose tables apply: Sunday (weekend) or Monday (weekday)."""
    return "h24_a" if pd.Timestamp(ts).dayofweek >= 5 else "h24_b"


def assess(P, start, end, phase: dict, fn_label: dict, fn_prob: dict, lanes: dict, phase_conf: dict | None = None,
           refs: Refs | None = None, stage1=None, keep_all: bool = False):
    """Health v4 for one signal window.  P = health_core.Prep of the window (allowed events only, de-duplicated);
    phase / fn_label / fn_prob / lanes / phase_conf: the classifier's answers per detector (raw predicted phase,
    final function label, class probabilities, lane span, phase probability).  Returns (table, signal note).
    `keep_all` returns every internal column (research comparison)."""
    refs = refs or _load_refs()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    T = (end - start).total_seconds()
    hours = T / 3600.0
    wg = window_group(hours)
    m30 = wg == "m30"
    bs = 300 if m30 else 900
    phase = {int(k): float(v) for k, v in (phase or {}).items() if v is not None and np.isfinite(v)}
    fn_label = {int(k): v for k, v in (fn_label or {}).items() if isinstance(v, str)}
    fn_prob = {int(k): v for k, v in (fn_prob or {}).items() if isinstance(v, dict) and v}
    lanes = {int(k): float(v) for k, v in (lanes or {}).items() if v is not None and np.isfinite(v)}
    H, st1 = _stage1(P, start, end, phase, fn_prob, lanes, phase_conf, stage1)
    if not len(H):
        return pd.DataFrame(columns=OUT_COLS), ""
    H = H.rename(columns={"status": "status2", "status1": "status"})
    H["detector"] = H.detector.astype(int)
    for c in ("s_dropout", "s_stuck", "s_level", "s_night_drop", "ep_dur", "drop_lam", "drop_b0", "drop_b1",
              "chat_frac", "night_n", "night_exp", "reason"):
        if c not in H:                                   # a check that did not run on this sample has no column
            H[c] = np.nan
    if "drop_in_stuck" not in H:
        H["drop_in_stuck"] = False
    # ---- the scored table: detectors with a predicted function the checks know
    X = H[H.detector.map(fn_label).isin(FNS)].copy().reset_index(drop=True)
    rest = H[~H.detector.isin(X.detector)].copy()
    if not len(X):
        return _fallback(rest), ""
    Pm = np.array([[fn_prob.get(int(d), {}).get(c, np.nan) for c in C7] for d in X.detector], float)
    top = np.nanmax(np.where(np.isfinite(Pm), Pm, -1), 1)
    alts = _alts(Pm)
    O = hs.occ_bins(P, start, end, bs, fn_label, phase)
    dmap = {int(d): i for i, d in enumerate(O["dets"])}
    cols = {"fn": X.detector.map(fn_label), "phase": X.detector.map(phase).astype(float),
            "lanes": X.detector.map(lanes).astype(float)}
    cols.update({f"p_{c}": Pm[:, j] for j, c in enumerate(C7)})
    cols.update(f_top=np.where(top >= 0, top, np.nan), alt_set=alts,
                alt_fns=[",".join(c for c in C7 if c in a) for a in alts], hours=hours, wg=wg, t_start=start,
                occ=[O["mocc"][dmap[d]] if d in dmap else np.nan for d in X.detector],
                share_ref=[O["share_ref"][dmap[d]] if d in dmap else np.nan for d in X.detector])
    X = _add(X, cols)
    # ---- bins, context
    M = hs.bins_ctx(O, st1)
    N3 = hs.n3_stats(M)
    A = hs.act_stats(P)
    X = _left(_left(X, N3), A)
    span = pd.Series(np.where(X.lanes >= 2, "2+", "1"), index=X.index)
    rate = X.n_on / X.hours
    share_med = pd.Series([refs.share_med.get((f, wg), np.nan) for f in X.fn], index=X.index)
    X = _add(X, {"span": span, "rate": rate, "band": hs.band_of(rate), "cmode": _cmode(X.fn, X.dur_p50),
                 "type": X.fn + " " + span, "share_med": share_med, "share_x": X.share_ref / share_med,
                 "stuck_x": np.fmax(pd.to_numeric(X.ep_dur, errors="coerce"), X.max_on_s)})
    X = _left(_left(X, hs.slopes(M, m30)), hs.like_corr(M, m30))
    for c in ("elhi_cnt", "elhi_occ", "c_occ_like"):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    hm = X.status.ne("bad").astype(int)
    ph = X.phase.fillna(-1)
    X["n_hmates"] = (hm.groupby(ph).transform("sum") - hm).where(X.phase.notna())
    OH = hs.occ_hi(M, dict(zip(X.detector, X.cmode)), refs)
    X = _left(X, OH)
    X["occ_hi8"] = X.occ_hi8.fillna(False).astype(bool)
    # ---- 5-min bins: yardstick, 15-min series, episodes
    B5 = hc.events_to_bins(P, start, end, None)
    bad1 = set(X.detector[X.status.eq("bad")])
    SH, b15 = hs.shape110(B5, phase, bad1, fn_prob)
    X = _left(X, SH)
    bi = {d: i for i, d in enumerate(b15["dets"])}
    nb15 = b15["x"].shape[1] if len(b15["dets"]) else 0
    dow = ((int(start.value) + np.arange(nb15, dtype=np.int64) * 900_000_000_000) // hc.DAY_NS + 3) % 7
    tday = np.where(dow >= 5, "h24_a", "h24_b").astype(object)          # = _day_type of each 15-min bin start
    wcache = {}
    lv, ev, dr = [], [], []
    for r in _rows(X):
        i = bi.get(int(r.detector))
        if i is None:
            lv.append((np.nan, 0.0, -1))
            ev.append({})
            dr.append({})
            continue
        x, S_, R_, ok = b15["x"][i], b15["S"][i], b15["R"][i], b15["ok"]
        wk = (r.type, r.band)
        if wk not in wcache:
            wcache[wk] = np.array([refs.tod_w.get((r.type, r.band, dd, int(hh)), refs.tod_wb.get((r.band, dd, int(hh)), 1.0))
                                   for dd, hh in zip(tday, b15["hour"])], float)
        w = wcache[wk]
        res = hs.level110(x, S_, ok, w)
        lv.append(res)
        e = dict(hs.erratic(x, R_, ok, "15"))
        e.update({k.replace("_15", "_sig"): v for k, v in hs.erratic(x, S_, ok, "15").items()})
        ev.append(e)
        if np.isfinite(res[0]) and res[2] >= 0:
            d = hs.drop_evidence(x, S_, ok, w, res[2])
            d["drop_at"] = start + _td(minutes=15 * res[2])
            dr.append(d)
        else:
            dr.append({})
    cols = {"lv_ratio110": [a[0] for a in lv], "lv_llr110": [a[1] for a in lv], "lv_b110": [a[2] for a in lv]}
    EV = pd.DataFrame(ev, index=X.index)
    for c in ("n_off_15", "n_sc_15", "exc_15", "n_off_sig", "n_sc_sig", "exc_sig"):
        cols[c] = EV[c] if c in EV else np.nan
    DR = pd.DataFrame(dr, index=X.index)
    for c in ("own_before", "own_after", "exp_after", "drop_at"):
        cols[c] = DR[c] if c in DR else np.nan
    g4 = DR.g4_unscored0.fillna(False).astype(bool) if "g4_unscored0" in DR else False
    cols["g4_unscored0"] = g4
    cols["g4_unscored"] = g4
    X = _left(_add(X, cols), hs.act118(P, T))
    # ---- episodes (continuous ONs >= 5 min) with the queue / traffic test
    # continuous ONs >= 300 s (the research listed >= 60 s and kept >= 300 s: the same rows, and on_episodes' own
    # pairwise 'held ON together' matrix then spans only these ONs, not every ON of a minute or more)
    ep = hc.on_episodes(P, start, end, min_s=300.0)
    E = ep[ep.dur_s >= 300].copy().reset_index(drop=True)
    E["ts"] = (E.t0 - start).dt.total_seconds()
    E["tf"] = (E.t1 - start).dt.total_seconds()
    E["co5"] = hs.episode_overlap(E)
    E = E.join(hs.episodes_ctx(E, M, dict(zip(X.detector, X.phase)), refs.qtype, dict(zip(X.detector, X.type))))
    E["fn"] = E.detector.map(dict(zip(X.detector, X.fn)))
    E["co5_orig"] = E.co5
    # ---- red / green: too fast, too many
    CB, nstart = hs.colour_bins(P, T, phase)
    CS = hs.colour_stats(CB, lanes)
    X = _left(X, CS)
    X = _add(X, {"n_on117": X.detector.map(nstart).astype(float)})
    # ---- time of day (a whole day in the sample)
    X, sig_note, day_info = _time_of_day(X, O, start, end, refs)
    # ---- v4 pass (only to find phase mates with a finding for the 'goes silent' yardstick), then v4c
    cand = X[(pd.to_numeric(X.drop_lam, errors="coerce") >= 30) & (X.n_on > 0)].detector.tolist()
    X["tod_level"] = X.tod_level_v4
    if cand:
        R4 = _resolve(X, E, refs, "v4")
        flagged = {int(d): bool(set(x for x in str(l).split(",") if x) - SILENT_FAM)
                   for d, l in zip(R4.detector, R4.left8)}
        DC = hs.dropout_clean(B5, cand, dict(zip(X.detector, X.phase)), flagged)
    else:
        DC = {}
    X["drop_lam"] = pd.to_numeric(X.drop_lam, errors="coerce")
    X["drop_b0"] = pd.to_numeric(X.drop_b0, errors="coerce")
    X["drop_b1"] = pd.to_numeric(X.drop_b1, errors="coerce")
    cols = {c: X.detector.map({d: v[c] for d, v in DC.items()}).astype(float)
            for c in ("drop_lam_chk", "drop_lam_c", "drop_b0_c", "drop_b1_c", "n_dropped")}
    cols["dropped"] = X.detector.map({d: v["dropped"] for d, v in DC.items()})
    X = _add(X, cols)
    m = (X.n_dropped.fillna(0) > 0) & (X.drop_lam_chk > 0)
    X.loc[m, "drop_lam"] = X.loc[m, "drop_lam"] * X.loc[m, "drop_lam_c"] / X.loc[m, "drop_lam_chk"]
    X.loc[m, "drop_b0"] = X.loc[m, "drop_b0_c"]
    X.loc[m, "drop_b1"] = X.loc[m, "drop_b1_c"]
    sd = pd.to_numeric(X.s_dropout, errors="coerce")
    new = np.where(X.drop_in_stuck.astype(str).eq("True"), 0.0,
                   [np.nan if not np.isfinite(v) else (0.0 if v < 30 else .35 + .65 * min((v - 30) / 70, 1))
                    for v in X.drop_lam.astype(float)])
    X["s_dropout"] = np.where(m, new, sd)
    X["tod_level"] = X.tod_level_c
    c1 = X.fn.eq("Count") & ((X.n_rep >= C1_N) & (X.n_rep >= C1_SHARE * X.n_on) |
                             ((X.rep_time_s >= D1_MIN_S) & (X.rep_time_s >= D1_SHARE * X.occ * X.hours * 3600)))
    sure_count = np.array([(not a) or a <= {"Count"} for a in alts])
    X["count_on_ok"] = X.fn.eq("Count").to_numpy() & sure_count & ~c1.fillna(False).to_numpy()
    X["c1_pre"] = c1.fillna(False).to_numpy()
    noq = X.fn.isin(NO_SHARED).to_numpy() & np.array([not (a & QUEUE_OK) for a in alts])
    E4 = E.copy()
    if len(E4):
        E4["noq"] = E4.detector.map(dict(zip(X.detector, noq))).fillna(False).astype(bool)
        E4.loc[E4.noq, "n_hpeer_e"] = 0
        qc = E4.cover_e.fillna(0) >= F4_COVER
        E4.loc[qc, "phx_h_e"] = np.fmax(E4.loc[qc, "phx_h_e"].fillna(0), 1.5)
    R = _resolve(X, E4, refs, "v4c")
    R = severity_v4d(R)
    out = _outputs(R, E4, start, end, day_info)
    if len(rest):
        out = pd.concat([out, _fallback(rest)], ignore_index=True)
    out = out.sort_values("detector").reset_index(drop=True)
    if keep_all:
        return out, sig_note, R, E4
    return out, sig_note


# ============================================================================ time of day (24-h samples)
def _full_days(start, end):
    """the whole days (00:00-24:00) inside [start, end); a 24-h sample that does not start at midnight is scored by
    clock hour as one day."""
    d0 = start.normalize() + (_td(days=1) if start != start.normalize() else pd.Timedelta(0))
    out = []
    d = d0
    while d + _td(days=1) <= end:
        out.append(d)
        d += _td(days=1)
    if not out and (end - start) >= _td(hours=24):
        out.append(start)
    return out


def _time_of_day(X, O, start, end, refs):
    cols_bd = ["bd_flag", "bd_ev_from", "bd_ev_h", "bd_ev_dir", "bd_ev_h0", "bd_ev_h1", "bd_ev_obs", "bd_ev_exp"]
    cols = {"tod_level_v4": "not scored", "tod_level_c": "not scored"}
    cols.update({c: np.nan for c in cols_bd})
    cols["bd_flag"] = pd.Series([np.nan] * len(X), dtype=object, index=X.index)
    cols["tod_day"] = pd.NaT
    X = _add(X, cols)
    note, info = "", {}
    if X.wg.iloc[0] != "h24" or O["bs"] != 900:
        return X, note, info
    days = _full_days(start, end)
    if not days:
        return X, note, info
    dmap = {int(d): i for i, d in enumerate(O["dets"])}
    idx = np.array([dmap.get(int(d), -1) for d in X.detector])
    best = {}
    sig = []
    for d0 in days:
        j0 = int(round((d0 - start).total_seconds() / 900))
        if j0 < 0 or j0 + 96 > O["nb"]:
            continue
        n15 = np.full((len(X), 96), np.nan)
        o15 = np.full((len(X), 96), np.nan)
        hv = idx >= 0
        n15[hv] = O["n"][idx[hv], j0:j0 + 96]
        o15[hv] = O["occ"][idx[hv], j0:j0 + 96]
        if d0 != d0.normalize():                      # not midnight-aligned: order the 96 bins by clock time
            sh = int(((d0 - d0.normalize()).total_seconds() // 900) % 96)
            n15, o15 = np.roll(n15, sh, 1), np.roll(o15, sh, 1)
        N, Od = hs.day_profile(n15, o15)
        tot = np.nansum(N, 1)
        ok = np.isfinite(N).all(1) & np.isfinite(Od).all(1) & (tot >= hs.MIN_N)
        dty = _day_type(d0 + _td(hours=12))
        fn = X.fn.to_numpy(object)
        span = X.span.to_numpy(object)
        ph = X.phase.to_numpy(float)
        alts = list(X.alt_set)
        nd = hs.night_day(N, Od, fn, span, ph, ok, refs.tod_own, refs.tod_alt, alts)
        nd_v4 = hs.night_day(N, Od, fn, span, ph, ok, refs.tod_own, refs.tod_alt, [set()] * len(X))
        F, bnd = hs.band_day(N, Od, fn, span, X.lanes.to_numpy(float), ph, ok, dty, refs.band_own, refs.band_alt, alts)
        odd = ok & (F.bd_run_ok.to_numpy() | nd.tod_odd.to_numpy())
        if odd.sum() >= hs.SIG_MIN and odd.sum() >= hs.SIG_SHARE * ok.sum():
            sig.append((d0, int(odd.sum()), int(ok.sum())))
        rank = {"bad": 3, "suspect": 2, "ok": 1, "not scored": 0}
        for i in range(len(X)):
            key = (rank[nd.tod_level.iat[i]], int(bool(F.bd_flag.iat[i])) * F.bd_ev_h.iat[i])
            if i not in best or key > best[i][0]:
                best[i] = (key, d0, nd.iloc[i], nd_v4.tod_level.iat[i], F.iloc[i],
                           {k: v[i] for k, v in bnd.items()}, N[i], Od[i], ok[i])
    n = len(X)
    num = ("tod_z", "tod_dph", "tod_n_night", "tod_r", "tod_mate_r", "bd_ev_h", "bd_ev_dir", "bd_ev_h0", "bd_ev_h1",
           "bd_ev_obs", "bd_ev_exp")
    V = {c: [np.nan] * n for c in num}
    V.update({"tod_level_c": ["not scored"] * n, "tod_level_v4": ["not scored"] * n, "bd_flag": [np.nan] * n,
              "bd_ev_from": [np.nan] * n, "tod_day": [pd.NaT] * n})
    for i, (key, d0, ndr, lv4, Fr, bnd, N_i, O_i, ok_i) in best.items():
        V["tod_level_c"][i], V["tod_level_v4"][i], V["tod_day"][i] = ndr.tod_level, lv4, d0
        for c in ("tod_z", "tod_dph", "tod_n_night", "tod_r", "tod_mate_r"):
            V[c][i] = float(ndr[c])
        for c in cols_bd:
            V[c][i] = Fr[c] if c in ("bd_flag", "bd_ev_from") else float(Fr[c])
        info[int(X.detector.iat[i])] = dict(day=d0, N=N_i, O=O_i, ok=ok_i, **bnd)
    for c, v in V.items():                         # whole columns (pandas 3 refuses a string into a float cell)
        X[c] = pd.Series(v, index=X.index, dtype=float if c in num else ("datetime64[ns]" if c == "tod_day" else object))
    if sig:
        d0, k, n = max(sig, key=lambda s: s[1] / max(s[2], 1))
        note = (f"{k} of {n} detectors at this signal had an unusual day on {d0:%a %d %b}: "
                "a signal-wide event or an input change, not one bad detector")
    return X, note, info


# ============================================================================ plain outputs
def _lanes_txt(r):
    n = r.lanes
    if n is None or not np.isfinite(n):
        return ""
    n = max(int(round(n)), 1)
    return f"covers {n} lane{'s' if n > 1 else ''}"


def _dur(s):
    s = float(s)
    if s < 300:
        return f"{s:.0f} s"
    m = int(round(s / 60))
    return f"{m} min" if m < 60 else (f"{m // 60} h" + (f" {m % 60} min" if m % 60 else ""))


def _hm(t):
    return pd.Timestamp(t).strftime("%H:%M")


def _label(r):
    p = f"P{int(r.phase)} " if np.isfinite(r.phase) else ""
    return f"det {int(r.detector)}: {p}{str(r.fn).replace('Yellow_Red', 'Yellow-red')}"


def _evidence(k, r, eps, start):
    """one short line per category with the numbers the check used."""
    if k == "stuck":
        e = eps.get(int(r.detector), [])
        cnt = [x for x in e if x["label"] != "queue"]
        tot = sum(x["dur_s"] for x in cnt)
        if len(cnt) > 1:
            return f"held ON {len(cnt)} times, {_dur(tot)} in total"
        if cnt:
            return f"held ON {_dur(tot)} from {cnt[0]['t0']:%a %H:%M}"
        return f"held ON up to {_dur(r.stuck_x)}"
    if k == "dropout":
        a = r.t_start + _td(seconds=300 * float(r.drop_b0))
        b = r.t_start + _td(seconds=300 * float(r.drop_b1))
        return f"no actuations {_hm(a)}-{_hm(b)}; ~{float(r.drop_lam):,.0f} expected"
    if k == "night_drop":
        if bool(getattr(r, "silent_night", False)):
            a = r.t_start + _td(seconds=300 * float(r.drop_b0))
            b = r.t_start + _td(seconds=300 * float(r.drop_b1))
            return f"no actuations {_hm(a)}-{_hm(b)}; ~{float(r.drop_lam):,.0f} expected"
        nn, ne = pd.to_numeric(getattr(r, "night_n", np.nan), errors="coerce"), \
            pd.to_numeric(getattr(r, "night_exp", np.nan), errors="coerce")
        return f"night {nn:,.0f} actuations vs ~{ne:,.0f} expected" if np.isfinite(ne) else "fewer actuations at night"
    if k == "level":
        t = f" at {pd.Timestamp(r.drop_at):%a %H:%M}" if pd.notna(r.drop_at) else ""
        return f"counts dropped{t}: {r.own_before:.0f} to {r.own_after:.0f} per 15 min; ~{r.exp_after:.0f} expected"
    if k == "choppy":
        return (f"{int(r.n_off_15)} of {int(r.n_sc_15)} 15-min periods outside the expected range; "
                f"{100 * r.exc_15:.0f} % of counts (limit {100 * r.lim_exc:.0f} %)")
    if k == "rapid":
        if r.x_spk >= np.nan_to_num(r.x_zf):
            return f"{int(r.n_spk_use)} bursts of back-to-back ONs; normal at most {int(r.lim_spk_use)}"
        normal = r.fem_use + r.lim_zf_use * (r.fem_use + 1) ** 0.5
        return f"{r.fo_use:,.0f} back-to-back vehicle ONs; normal up to {normal:,.0f}"
    if k == "volume":
        q = r.q5_all if r.fn == "Advance" else r.q5_gy
        return f"busiest 5 min {q:,.0f} veh/h/lane vs limit {r.lim_vol_use:,.0f}"
    if k == "chatter":
        return f"{100 * r.chat_frac:.1f} % of ONs turn on again within 0.3 s (limit {100 * r.lim_chat_frac:.1f} %)"
    if k == "occspk":
        return f"{r.n3_exc:.0f} min of time ON its counts do not explain (limit {r.lim_n3_exc:.0f})"
    if k == "prof":
        return (f"night level {100 * r.tod_r ** 2:.0f} % of its busiest hours; phase mates "
                + (f"{100 * r.tod_mate_r ** 2:.0f} %" if np.isfinite(r.tod_mate_r) else "none"))
    if k == "shape24":
        a, b = int(r.bd_ev_h0), int(r.bd_ev_h1) + 1
        who = "its phase mates" if r.bd_ev_from == "mates" else "its type"
        return f"{r.bd_ev_obs:,.0f} actuations {a:02d}-{b:02d} h; {who} ~{r.bd_ev_exp:,.0f}"
    if k == "count_on":
        mx = r.hi_occ_max if np.isfinite(r.hi_occ_max) else np.nan
        return f"ON up to {100 * mx:.0f} % in light traffic for {r.hi_min_c:.0f} min"
    if k == "occ_hi":
        return "ON longer than similar zones for 30 min or more"
    if k == "C1":
        return f"{int(r.n_rep):,} of {int(r.n_on):,} ONs logged again without an OFF"
    if k == "C2":
        return f"{int(r.n_ge5)} ONs over 5 s, longest {_dur(r.long_max_s)}" if np.isfinite(r.long_max_s) else \
            "holds ON although set to pulse"
    return ""


def _stuck_eps(R, E, start):
    """every continuous ON over the type limit, labelled counted / queue / shared (h118 stuck_list)."""
    out = {}
    if not len(E):
        return out
    Ex = E.merge(R[["detector", "stuck_lim8"]], on="detector", how="inner")
    Ex = Ex[Ex.dur_s >= Ex.stuck_lim8]
    if not len(Ex):
        return out
    q_ok = (Ex.n_hpeer_e.fillna(0) > 0) & (Ex.phx_h_e >= 1.5) & (Ex.corr_h_e >= Q1_CORR) & (Ex.refx_e >= 0.5) & \
        (Ex.light_e.fillna(1) == 0) & (Ex.trafx_e >= TRAF_X)
    q1 = q_ok & (Ex.dur_s < 3600)
    q1b = q_ok & (Ex.dur_s >= 3600) & (Ex.cover_e >= F4_COVER)
    co = np.where(Ex.fn.isin(NO_SHARED), 0, Ex.co5)
    sh = (co >= 3) & ~q1 & ~q1b
    lab = np.where(q1 | q1b, "queue", np.where(sh, "shared", "counted"))
    for r, l in zip(Ex.sort_values("t0").itertuples(), lab[np.argsort(Ex.t0.to_numpy(), kind="stable")]):
        out.setdefault(int(r.detector), []).append(dict(t0=r.t0, t1=r.t1, dur_s=float(r.dur_s), label=l,
                                                        open_end=bool(r.open_end)))
    return out


def _outputs(R, E, start, end, day_info):
    eps = _stuck_eps(R, E, start)
    rows = []
    for r in _rows(R):
        st = r.st_v4d
        cats = [k for k in str(r.left8).split(",") if k and k in CATEGORIES]
        cats = sorted(cats, key=lambda k: (-(getattr(r, f"s8_{k}", 0) or 0), ORDER.index(k) if k in ORDER else 99))
        watch = [w for w in str(r.watch8).split(",") if w]
        cfg = [c for c in str(r.cfg).split(",") if c in CATEGORIES]
        if r.sev_rule == "C":
            cats = cats + ["count_on"] if "count_on" not in cats else cats
        pers = persisting_categories(r) if r.sev_rule == "P" else set()
        lane = _lanes_txt(r)
        lab = _label(r) + (f" ({lane})" if lane and set(cats) & LANE_CATS else "")
        parts, catl = [], []
        for k in cats:
            v = getattr(r, f"s8_{k}", np.nan)
            sev = "bad" if (np.isfinite(v) and v >= 0.75) else "suspect"
            if r.sev_rule == "C" and k == "count_on":
                sev = "suspect"
            elif r.sev_rule == "P" and k in pers:
                sev = "bad"
            catl.append(f"{CATEGORIES[k]} ({sev})")
            parts.append(f"{CATEGORIES[k]}: {_evidence(k, r, eps, start)}")
        if "occ_hi" in watch:
            catl.append(f"{CATEGORIES['occ_hi']} (watch)")
        wtxt = []
        for w in watch:
            if w == "occ_hi":
                wtxt.append("ON longer than similar zones")
            elif w in ("no_yardstick", "no_yardstick_chop"):
                wtxt.append("no healthy phase mate to compare with")
            elif w == "short_on":
                wtxt.append("many ONs last 0.2 s or less for a presence zone")
            elif w in CATEGORIES:
                wtxt.append(f"borderline {CATEGORIES[w].lower()}")
        if st in ("suspect", "bad"):
            reason = f"{lab} - " + "; ".join(parts) if parts else f"{lab} - {st}"
        elif st == "watch":
            reason = f"{lab} - watch: " + "; ".join(dict.fromkeys(wtxt)) if wtxt else f"{lab} - watch"
        elif st == "not_enough_data":
            reason = f"{lab} - too few actuations to judge"
        else:
            reason = f"{lab} - nothing unusual"
        conf = "; ".join(f"{CATEGORIES[c]}: {_evidence(c, r, eps, start)}" for c in cfg)
        per = []
        if "stuck" in cats or st in ("suspect", "bad"):
            for x in eps.get(int(r.detector), []):
                if x["label"] != "queue" and "stuck" in cats:
                    per.append(dict(start=str(x["t0"]), end=str(x["t1"]), what="stuck on", recovered=None))
        for k, what in (("dropout", "silent"), ("night_drop", "silent at night")):
            if k in cats and np.isfinite(r.drop_b0) and np.isfinite(r.drop_b1) and \
                    (k == "dropout" or bool(getattr(r, "silent_night", False))):
                per.append(dict(start=str(r.t_start + _td(seconds=300 * float(r.drop_b0))),
                                end=str(r.t_start + _td(seconds=300 * float(r.drop_b1))), what=what,
                                recovered=None))
        if "level" in cats and pd.notna(r.drop_at):
            per.append(dict(start=str(r.drop_at), end=str(end), what="counts dropped", recovered=None))
        if "shape24" in cats and pd.notna(r.tod_day):
            a = pd.Timestamp(r.tod_day) + _td(hours=int(r.bd_ev_h0))
            per.append(dict(start=str(a), end=str(a + _td(hours=int(r.bd_ev_h1) - int(r.bd_ev_h0) + 1)),
                            what="unusual daily pattern", recovered=None))
        per = sorted(per, key=lambda x: x["start"])
        rows.append(dict(detector=int(r.detector), health_status=st,
                         health_score=float(r.score8) if np.isfinite(r.score8) else np.nan,
                         health_reason=reason, health_bad_periods=per, health_categories="; ".join(catl),
                         health_config=conf, health_watch="; ".join(dict.fromkeys(wtxt))))
    return pd.DataFrame(rows, columns=OUT_COLS)


def _fallback(H):
    """detectors without a known predicted function: the rule checks of health_core as they are."""
    out = pd.DataFrame({"detector": H.detector.astype(int), "health_status": H.status2 if "status2" in H else H.status,
                        "health_score": H.health_score,
                        "health_reason": [f"det {int(d)}: " + str(x).split("; note: ")[0] for d, x in
                                          zip(H.detector, H.reason.fillna(""))],
                        "health_bad_periods": [[dict(start=str(p["start"]), end=str(p["end"]), what=p["what"],
                                                     recovered=p.get("recovered")) for p in (L or [])]
                                               if isinstance(L, list) else [] for L in H.bad_periods],
                        "health_categories": "", "health_config": "", "health_watch": ""})
    return out
