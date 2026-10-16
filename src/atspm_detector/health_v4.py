"""Detector health (1.1.0, notes 144, 147m, 150): per-detector status, score, plain reason, bad periods, health
categories with severity, configuration notes and a signal-level note, from the hi-res log and the classifier's own
answers (predicted phase, function, lane span).  numpy + pandas only.

    assess(prep, start, end, phase, fn_label, fn_prob, lanes, phase_conf) -> (per-detector DataFrame, signal note)

Six checks set the status (the user-reviewed set, notes 140-147m):
  1 Stuck on                 a continuous ON over the type's limit that is not a queue (Q1 / Q1b, note 135 lift) and
                             not held ON with >= 3 others (suspect at most); held ON together with other detectors, all
                             starting AND ending within 5 s = one common failure (bad, signal note); on a Count zone
                             also 'held ON' through light traffic;
  2 Counts fall              goes silent (expected actuations in the longest silent run) or the count drops (hourly GEH
                             after the drop point, only when its own occupancy fell too) -- one finding, never two;
  3 Misses vehicles at night GEH of the average night hour (21-05) against its yardstick; a night silence with < 100
                             vehicles expected is suspect at most;
  4 Busy at night            01-05 h level z against its type, and busier than its phase mates (whole day in the sample);
  5 Too-fast actuations      ON-to-ON gaps < 0.8 s / lanes (Count, Presence, Yellow_Red) or 0.6 s / lanes (others) per
                             1,000 ONs (2+ lanes: the 1-lane limit); Too many for the traffic = busiest 5 min veh/h/lane;
  6 Spiky counts             bursts in 10-min counts (>= 3 x its own level of the surrounding 2 h AND GEH >= 5, beyond
                             the phase mates' own rise); 2 a day suspect, 3 bad; < 200 vehicles a day never, < 500
                             suspect at most unless a burst reaches GEH 10.
Bike zones: no counts-fall or night checks.  Count drops, night misses and spiky counts need >= 3 h of data.
Limits: weights/health/health_limits.json holds [suspect, bad] per check x type (function | lane span [| Count mode]):
a single finding is suspect from the first number and BAD from the second.  24-h samples (and longer) use the
user-reviewed limits; shorter samples the 1.0.1 limits (weights/health/health_v4_refs.json).  When the model is unsure
of the function (top probability < .7) the least strict limit over the plausible functions applies.
Score per finding: 0 below the suspect limit, .35 at it, .75 at the bad limit (linear), 1 at most.
Severity: product of (1 - score) < .25 bad, < .70 suspect; findings of two families or two independent kinds of
evidence = bad; busy at night or silent >= 4 h = bad (rule P); a Count zone with extension held ON = suspect (rule C);
too-fast alone, spread evenly with traffic on a Presence / Other / Mid zone = suspect (rule F).
Watch (information): borderline single finding, no healthy phase mate to compare with.
Config notes: extension time on a count zone (C1), set to pulse but holds ON (C2).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import health_core as hc
from . import health_counts as hk
from .common import timedelta as _td
from . import health_v4_stats as hs

FNS = hs.FNS
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
WDIR = Path(__file__).resolve().parent / "weights" / "health"
REF_FILE = WDIR / "health_v4_refs.json"
LIM_FILE = WDIR / "health_limits.json"

# ---- constants of the research scorer (h108 / h110 / h118 / score_v4c / score_v4d / score_v4e)
BORDER, STUCK_CLIP = 0.42, (300.0, 900.0)
Q2_CORR, Q1_CORR = 0.80, 0.70
D1_SHARE, D1_MIN_S = 0.70, 600.0
F5_TOP, F5_PLAUS = 0.70, 0.15
F4_COVER = 0.75
TRAF_X = 0.5
LIFT_CAP = 0.35                    # note 135: a hold the traffic does not explain is a suspect finding at most
F_FNS = ("Presence", "Other", "Mid")   # note 135 rule F: only zones where extra fast ONs do not inflate counts
NO_SHARED = ("Advance", "Count", "Yellow_Red")
QUEUE_OK = {"Presence", "Other", "Mid"}
C1_N, C1_SHARE = 20, 0.05
C2_N60, C2_N5 = 1, 3
NIGHT_SIL = (21, 5)
PERSIST_H = 4.0                    # note 121 rule P (user decision 2026-10-08)
FAST_PER_H = 3                     # bad periods of too-fast actuations: hours with >= 3 too-fast gaps over the limit
PAIRS_SHARE = 0.9                  # note 147e: >= 90 % of the short gaps are single pairs -> 'counts each vehicle twice'
NO_COUNTS = ("Bike",)              # note 147e: bike zones have no counts-fall or night-miss check
KEYS = ("stuck", "count_on", "dropout", "level", "night_drop", "prof", "rapid", "volume", "spiky")
STAT = ("dropout", "rapid", "level", "night_drop", "prof", "spiky")
FAMILY = {"rapid": "fast", "dropout": "silent", "stuck": "stuck", "level": "level", "night_drop": "level",
          "volume": "shape", "prof": "shape", "count_on": "stuck", "spiky": "shape"}
EVID = {"stuck": "stuck", "count_on": "stuck", "dropout": "quiet", "level": "quiet", "night_drop": "quiet",
        "rapid": "fast", "volume": "fast", "prof": "prof", "spiky": "spiky"}
CATEGORY = {"stuck": "Stuck on", "count_on": "Stuck on", "dropout": "Counts fall", "level": "Counts fall",
            "night_drop": "Misses vehicles at night", "prof": "Busy at night", "rapid": "Too-fast actuations",
            "volume": "Too many for the traffic", "spiky": "Spiky counts"}
CONFIG = {"C1": "Extension time on a count zone", "C2": "Set to pulse but holds ON"}
LANE_CATS = {"stuck", "rapid", "volume", "prof", "spiky"}
GEH_COLS = ("geh_night", "geh_night_n", "geh_night_exp", "geh_drop", "drop_t", "own_before", "own_after", "exp_after",
            "occ_own", "occ_mates", "spk_bursts", "spk_gmax", "spk_daily", "spk_lvl")
OUT_COLS = ["detector", "health_status", "health_score", "health_reason", "health_bad_periods", "health_categories",
            "health_config", "health_watch"]


# ============================================================================ references and limits
class Refs:
    """the frozen 1.0.1 population tables (weights/health/health_v4_refs.json); `tod` holds the 24-h night models used
    for the detector's own type (`tod_own`) and for the plausible alternatives when the model is unsure (`tod_alt`)."""

    def __init__(self, d: dict, tod_own=None, tod_alt=None, **_):
        self.d = d
        self.cell = {c: self._levels(t) for c, t in d["cell"]["v4c"].items() if c in ("stuck_x", "n_ep")}
        self.g3 = self._levels(d["g3"]["v4c"])
        self.f118 = {c: self._levels(d["fast118c"][c]) for c in ("q5_gy", "q5_all")}
        self._occ_p = {(int(r[0]), int(r[1])): r[2] for r in d["occ_lim_pulse"]}
        self._tp = {int(r[0]): r[1] for r in d["count_t_pass"]}
        self.qtype = {r[0]: r[1] for r in d["q_type"]}
        self.tod_own = tod_own or d["tod"]
        self.tod_alt = tod_alt or d["tod"]

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


@lru_cache(maxsize=2)
def _load_refs(path: str = str(REF_FILE)) -> Refs:
    with open(path, encoding="utf-8") as f:
        return Refs(json.load(f))


@lru_cache(maxsize=2)
def _load_limits(path: str = str(LIM_FILE)) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score(x, lo, bad):
    """the finding's score: 0 below the suspect limit `lo`, .35 at it, .75 at the bad limit `bad` (linear), 1 at most.
    One finding is bad from the bad limit on (severity: 1 - score < .25)."""
    x, lo, bad = (np.asarray(v, float) for v in (x, lo, bad))
    with np.errstate(invalid="ignore"):
        s = np.minimum(0.35 + 0.40 * np.maximum((x - lo) / np.maximum(bad - lo, 1e-9), 0), 1.0)
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


def _cm(fa, dp):
    """the Count mode a plausible function takes for the refs tables: '' unless Count; from the median ON."""
    return "" if fa != "Count" else ("unknown" if dp != dp else ("pulse" if dp <= 0.25 else "normal"))


def _lk(levels, row):
    """Refs.lookup for one row (a dict of key column -> text): the first finite value over the key levels."""
    for keys, tab in levels:
        v = tab.get(tuple(row[k] for k in keys))
        if v is not None:
            v = float(v)
            if v == v:
                return v
    return np.nan


def limits(X, refs, LJ):
    """per-detector limits of every check for the sample's length class -> (suspect, bad) dicts of arrays.  24-h samples
    (and longer): the [suspect, bad] tables of health_limits.json; shorter: the 1.0.1 tables (refs) by function x lane
    span x Count mode x length class with the bad starts of LJ['short'].  Loosened to the least strict plausible
    function when the model is unsure.  Plain dict lookups (a signal has a few dozen detectors: faster than arrays)."""
    n = len(X)
    cur, sh, h, tf = LJ["current"], LJ["short"], LJ["h24"], LJ["too_fast"]
    h24 = bool(n) and X.wg.iat[0] == "h24"
    rows = [dict(zip(("fn", "span", "cmode", "wg", "band"), r)) for r in
            zip(*(X[c].astype(str).tolist() for c in ("fn", "span", "cmode", "wg", "band")))]
    dp = X.dur_p50.to_numpy(float)
    cm_alt = ["pulse" if d <= 0.25 else "normal" for d in dp]
    alts = [(i, fa) for i, a in enumerate(X.alt_set) for fa in C7 if fa in a and fa != rows[i]["fn"]]

    def key(f, i, cm=None):
        return f"{f}|{rows[i]['span']}" + (f"|{cm}" if (cm and f == "Count") else "")

    def pick(v, j):
        v = v if j is None or v is None else v[j]
        return np.nan if v is None else float(v)

    def tab(t, j=None, split=False):
        v = np.array([pick(t.get(key(r["fn"], i, ("normal" if r["cmode"] == "normal" else "pulse") if split else None)), j)
                      for i, r in enumerate(rows)], float)
        for i, fa in alts:
            a = pick(t.get(key(fa, i, cm_alt[i] if split else None)), j)
            if a == a and not a <= v[i]:
                v[i] = a
        return v

    def ref(levels, cm_fn=False):
        v = np.array([_lk(levels, r) for r in rows], float)
        for i, fa in alts:
            a = _lk(levels, dict(rows[i], fn=fa, **({"cmode": _cm(fa, dp[i])} if cm_fn else {})))
            if a == a and (a > v[i] if cm_fn else not a <= v[i]):     # 1.0.1 stuck_x: a NaN own value stays
                v[i] = a
        return v
    L, B = {}, {}
    lne = ref(refs.cell["n_ep"])
    L["n_ep"], B["n_ep"] = lne + 1, (lne + 1) * sh["n_holds_bad_x"]
    L["rapid"] = tab(tf["per_1000"])
    B["rapid"] = L["rapid"] * tf["bad_x"]
    L["level"], B["level"] = np.full(n, LJ["count_drops_geh"][0]), np.full(n, LJ["count_drops_geh"][1])
    L["night_drop"], B["night_drop"] = np.full(n, LJ["night_geh"][0]), np.full(n, LJ["night_geh"][1])
    L["count_on"], B["count_on"] = np.full(n, LJ["count_held_on_min"][0]), np.full(n, LJ["count_held_on_min"][1])
    if h24:
        L["stuck"] = np.fmax(tab(h["stuck_s"], 0, True), LJ["hold_floor_s"])
        B["stuck"] = np.fmax(tab(h["stuck_s"], 1, True), L["stuck"])
        L["dropout"], B["dropout"] = tab(h["silent_exp"], 0), tab(h["silent_exp"], 1)
        L["prof"] = np.array([pick(h["busy_z"].get(key(r["fn"], i)), None) for i, r in enumerate(rows)], float)
        L["volume"] = np.full(n, LJ["too_many_h24_veh_h_lane"][0])
        B["volume"] = np.full(n, LJ["too_many_h24_veh_h_lane"][1])
    else:
        L["stuck"] = np.clip(ref(refs.cell["stuck_x"], cm_fn=True), STUCK_CLIP[0], STUCK_CLIP[1])
        a, b, c = sh["stuck_bad"]
        B["stuck"] = np.fmax(a + b * L["stuck"], c * L["stuck"])
        L["dropout"] = np.full(n, cur["silent_exp"])
        B["dropout"] = L["dropout"] * sh["silent_bad_x"]
        L["prof"] = np.full(n, cur["busy_z"])
        lg, la = ref(refs.f118["q5_gy"]), ref(refs.f118["q5_all"])
        L["volume"] = np.fmax(np.where(X.fn.eq("Advance").to_numpy(), la, lg), cur["too_many_veh_h_lane"])
        B["volume"] = L["volume"] * sh["too_many_bad_x"]
    return L, B


# ============================================================================ the checks
def _f0(a):
    """Series.fillna(0) of a float column, as an array."""
    return np.nan_to_num(np.asarray(a, float), nan=0.0)


def _lifted(E, q_ok, dur):
    """note 135: a hold >= 1 h the queue test excuses (Q1b) whose traffic did not hold the detector elsewhere in the
    sample (>= 60 min of bins at least as busy, median % ON there < 50 %) is not excused -- unless a detector of
    the traffic yardstick was itself held ON during the hold (ref_held_e: the hold's traffic is under-measured)."""
    if "busy_min_e" not in E:
        return np.zeros(len(E), bool)
    bm, bo = E.busy_min_e.to_numpy(float), E.busy_occ_e.to_numpy(float)
    rh = (E.ref_held_e.fillna(False).astype(bool).to_numpy() if "ref_held_e" in E else np.zeros(len(E), bool))
    with np.errstate(invalid="ignore"):
        return (q_ok & (dur >= 3600) & (E.cover_e.to_numpy(float) >= F4_COVER) & (bm >= hs.BUSY_MIN)
                & (bo < hs.BUSY_OCC) & ~rh)


def episode_flags(E):
    """per continuous ON >= 300 s: queue < 1 h (q1), queue >= 1 h covered by queued mates (q1b), queue excuse lifted
    by the traffic test (lifted), held ON with >= 3 others (shared; never on Advance / Count / Yellow_Red), held ON
    together with other detectors that all start and end within 5 s of it (common: one shared failure, never excused),
    counted."""
    num = lambda c: E[c].to_numpy(float)  # noqa: E731
    dur = num("dur_s")
    common = E.common.to_numpy(bool) if "common" in E else np.zeros(len(E), bool)
    with np.errstate(invalid="ignore"):
        q_ok = ((np.nan_to_num(num("n_hpeer_e"), nan=0.0) > 0) & (num("phx_h_e") >= 1.5) & (num("corr_h_e") >= Q1_CORR)
                & (num("refx_e") >= 0.5) & (np.nan_to_num(num("light_e"), nan=1.0) == 0) & (num("trafx_e") >= TRAF_X))
        q1 = q_ok & (dur < 3600) & ~common
        lifted = _lifted(E, q_ok, dur)
        q1b = q_ok & (dur >= 3600) & (num("cover_e") >= F4_COVER) & ~lifted & ~common
        co = np.where(E.fn.isin(NO_SHARED).to_numpy(), 0, num("co5"))
        shared = (co >= 3) & ~common
    return dict(dur=dur, q1=q1, q1b=q1b, lifted=q_ok & lifted, shared=shared, counted=~q1 & ~q1b & ~shared,
                common=common)


def _stuck(X, E, F, L, Lb, ln1, lnb):
    """stuck on (h110 F3 / F4 + note 135, 147e): counted holds >= the limit L; score of their total time (bad from Lb),
    number of them against ln1 = lim_n_ep + 1 (bad from lnb); a single recovered hold, shared-only holds and
    lifted-only holds are suspect at most; queue-only holds are cleared; a common hold (held ON together with other
    detectors, same start and end) at the limit is bad.  Returns (score, per-detector numbers)."""
    n = len(X)
    pos = {int(d): i for i, d in enumerate(X.detector.to_numpy())}
    ri = np.array([pos.get(int(d), -1) for d in E.detector.to_numpy()], int) if len(E) else np.zeros(0, int)
    ok = ri >= 0
    Le = np.where(ok, np.asarray(L, float)[np.maximum(ri, 0)], np.inf)
    keep = ok & (F["dur"] >= Le) if len(E) else np.zeros(0, bool)

    def cnt(mask, w=None):
        m = keep & mask
        return np.bincount(ri[m], weights=None if w is None else w[m], minlength=n)[:n] if len(E) else np.zeros(n)
    one = np.ones(len(E), bool)
    n_over, n_ep = cnt(one), cnt(F["counted"])
    tot = cnt(F["counted"], F["dur"]).astype(float)
    n_q = cnt(F["q1"] | F["q1b"])
    n_q1b = cnt(F["q1b"])
    n_sh_unq = cnt(F["shared"] & ~F["q1"] & ~F["q1b"])
    n_lift = cnt(F["counted"] & F["lifted"])
    n_common = cnt(F["common"])
    s_tot = score(tot, L, Lb)
    s_n = score(n_ep, ln1, lnb)
    s = np.fmax(s_tot, np.nan_to_num(s_n))
    cap = (pd.to_numeric(X.s_stuck, errors="coerce") == 0.35).to_numpy()
    s = np.where(cap & (n_ep <= 1) & (s > .35), .35, s)
    q1_all = (n_over > 0) & (n_ep == 0) & (n_sh_unq == 0) & (n_q > 0)
    shared_only = (n_ep == 0) & (n_sh_unq > 0)
    s = np.where(shared_only, .35, s)
    s = np.where(q1_all, 0.0, s)
    lift_only = (n_ep > 0) & (n_ep == n_lift)
    s = np.where(lift_only & (s > LIFT_CAP), LIFT_CAP, s)
    s = np.where(n_common > 0, 1.0, s)
    return s, dict(st_n_ep=n_ep, st_tot_s=tot, st_q1_all=q1_all, st_q1b=n_q1b, st_lift_only=lift_only,
                   st_common=n_common)


def _night_frac_v(t0, b0, b1):
    """share of the 5-min bins [b0, b1) (from t0) whose clock hour is night (21:00-05:00), one value per row."""
    out = np.zeros(len(b0))
    b0, b1 = np.asarray(b0, float), np.asarray(b1, float)
    ok = np.isfinite(b0) & np.isfinite(b1) & (b1 > b0)
    if not ok.any():
        return out
    a, b = b0[ok].astype(np.int64), b1[ok].astype(np.int64)
    sod = int(pd.Timestamp(t0).value) % 86_400_000_000_000
    hrs = ((sod + np.arange(int(b.max()), dtype=np.int64) * 300_000_000_000) // 3_600_000_000_000) % 24
    cn = np.r_[0, np.cumsum((hrs >= NIGHT_SIL[0]) | (hrs < NIGHT_SIL[1]))]
    out[ok] = (cn[b] - cn[a]) / (b - a)
    return out


def scores(X, E, F, L, B, LJ):
    """the check scores (0 below the suspect limit, .35 at it, .75 at the bad limit, 1 at most; NaN = not checked) ->
    {key: array} and the extra per-detector columns the text needs.  L / B: suspect / bad limits (limits())."""
    S, C = {}, {}
    S["stuck"], C0 = _stuck(X, E, F, L["stuck"], B["stuck"], L["n_ep"], B["n_ep"])
    C.update(C0)
    hm = _f0(X.hi_min_c)
    S["count_on"] = np.where(X.count_on_ok.to_numpy(bool), score(hm, L["count_on"], B["count_on"]), np.nan)
    dl = X.drop_lam.to_numpy(float)
    sd = score(dl, L["dropout"], B["dropout"])
    cap = ((pd.to_numeric(X.s_dropout, errors="coerce") == 0.35) & (X.drop_lam > 30.001)).to_numpy()
    sd = np.where(cap & (sd > .35), .35, sd)
    sd = np.where(X.drop_in_stuck.astype(str).eq("True").to_numpy(), 0.0, sd)
    sd = np.where(X.n_on.to_numpy(float) > 0, sd, np.nan)
    # count drops: hourly GEH after the drop point, only when the detector's own occupancy fell against its mates'
    gok = X.geh_ok.fillna(False).astype(bool).to_numpy()
    gd = X.geh_drop.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = X.occ_own.to_numpy(float) / np.maximum(X.occ_mates.to_numpy(float), 1e-9)
    conf = np.isfinite(rel) & (rel < LJ["occ_confirm"])
    C["occ_rel"] = rel
    S["level"] = np.where(gok, np.where(np.isfinite(gd) & conf, score(gd, L["level"], B["level"]), 0.0), np.nan)
    # misses at night: GEH of the average night hour; a silence lying >= 80 % at night is judged here
    nd = score(X.geh_night.to_numpy(float), L["night_drop"], B["night_drop"])
    nf = _night_frac_v(X.t_start.iat[0] if len(X) else None, X.drop_b0.to_numpy(float), X.drop_b1.to_numpy(float))
    silent = (np.nan_to_num(sd) >= .35) & (nf >= .8)
    moved = silent & (np.nan_to_num(sd) > np.nan_to_num(nd) + 1e-9)
    nd2 = np.where(moved, sd, nd)
    low = moved & (np.nan_to_num(dl, nan=0.0) < LJ["night_silence_low_exp"])   # few vehicles expected: suspect at most
    nd2 = np.where(low, np.minimum(nd2, 0.5), nd2)
    C["silent_night"] = silent
    S["dropout"] = np.where(silent, 0.0, sd)
    S["night_drop"] = nd2
    bike = X.fn.isin(NO_COUNTS).to_numpy()
    for k in ("dropout", "level", "night_drop"):
        S[k] = np.where(bike, np.nan, S[k])
    pv = hs.busy_level(X.tod_z.to_numpy(float), X.tod_dph.to_numpy(float), X.tod_n_night.to_numpy(float), L["prof"],
                       LJ["busy"])
    S["prof"] = np.where(np.nan_to_num(S["dropout"]) >= .35, np.where(np.isfinite(pv), 0.0, pv), pv)
    fr, nfast, N = X.fast_rate.to_numpy(float), X.n_fast.to_numpy(float), X.N.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(nfast >= hs.TF_MIN_GAPS, fr / L["rapid"], 0.0)
        S["rapid"] = np.where(N >= hs.TF_MIN_ON, score(x + 1e-9 * (x > 0), 1.0, B["rapid"] / L["rapid"]), np.nan)
    q5 = np.where(X.fn.eq("Advance").to_numpy(), X.q5_all.to_numpy(float), X.q5_gy.to_numpy(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        S["volume"] = np.where(np.isfinite(q5), score(q5 / L["volume"], 1.0, B["volume"] / L["volume"]), np.nan)
    lv = X.spk_lvl.to_numpy(float)
    S["spiky"] = np.where(np.isfinite(lv), np.where(lv >= 2, 1.0, np.where(lv >= 1, 0.5, 0.0)), np.nan)
    C["q5_use"] = q5
    return S, C


def status(X, S, C):
    """the resolver (h110 / v4c rules on the six checks) and severity (score_v4d A / P / C, score_v4e F), vectorised.
    Returns a dict of per-detector arrays: st, left (comma list of findings), watch, rule, score, n_ev."""
    n = len(X)
    K = list(KEYS)
    V = np.column_stack([np.asarray(S[k], float) for k in K]) if n else np.zeros((0, len(K)))
    j = {k: i for i, k in enumerate(K)}
    fin = np.isfinite(V)
    find = fin & (V >= .35)
    watch = {w: np.zeros(n, bool) for w in ("no_yardstick", "dropout_short")}
    c_like = X.c_occ_like.to_numpy(float)
    with np.errstate(invalid="ignore"):
        qp = X.queue_pat8.to_numpy(bool) & (c_like >= Q2_CORR)
    find[:, j["level"]] &= ~qp                                       # Q2: a queue pattern explains a count drop
    r6 = find[:, j["dropout"]] & (X.hours.to_numpy(float) < 1)       # R6: silence in a sample under 1 h -> watch
    find[:, j["dropout"]] &= ~r6
    watch["dropout_short"] = r6
    both = find[:, j["dropout"]] & find[:, j["level"]]               # counts fall: one finding, the larger score
    keep_d = V[:, j["dropout"]] >= V[:, j["level"]]
    find[:, j["level"]] &= ~(both & keep_d)
    find[:, j["dropout"]] &= ~(both & ~keep_d)
    nh = X.n_hmates.to_numpy(float)
    r9 = (X.status2.astype(str) != X.status.astype(str)).to_numpy()
    watch["no_yardstick"] = r9 & (nh == 0) & X.status.isin(["suspect", "bad"]).to_numpy()
    one = find.sum(1) == 1                                         # R7: one borderline statistical finding -> watch
    stat_m = np.zeros(len(K), bool)
    stat_m[[j[k] for k in STAT]] = True
    r7 = one & (find & stat_m[None, :] & (V < BORDER)).any(1)
    border = np.where(r7[:, None], find, False)
    find &= ~r7[:, None]
    Vf = np.where(find, V, 0.0)
    hsc = np.prod(1 - Vf, 1)
    any_fin = fin.any(1)
    hsc = np.where(any_fin, hsc, np.nan)
    has_w = r7 | r6 | watch["no_yardstick"]
    n_on = X.n_on.to_numpy(float)
    with np.errstate(invalid="ignore"):
        st = np.where(hsc < .25, "bad", np.where(hsc < .70, "suspect",
                      np.where(has_w, "watch", np.where(n_on < 20, "not_enough_data", "ok")))).astype(object)
    cap35 = np.zeros_like(find)
    for k in ("stuck", "dropout"):
        cap35[:, j[k]] = V[:, j[k]] == .35
    real = find & ~cap35
    fam_keys = sorted(set(FAMILY.values()))
    fam = np.column_stack([(real & np.array([FAMILY[k] == f for k in K])[None, :]).any(1) for f in fam_keys]) \
        if n else np.zeros((0, len(fam_keys)), bool)
    st = np.where((st == "suspect") & (fam.sum(1) >= 2), "bad", st).astype(object)
    ev_keys = sorted(set(EVID.values()))
    ev = np.column_stack([(real & np.array([EVID[k] == e for k in K])[None, :]).any(1) for e in ev_keys]) \
        if n else np.zeros((0, len(ev_keys)), bool)
    n_ev = ev.sum(1)
    rule = np.full(n, "", object)
    a = (st == "suspect") & (n_ev >= 2)
    st[a], rule[a] = "bad", "A"
    silent_h = (X.drop_b1.to_numpy(float) - X.drop_b0.to_numpy(float)) * 300 / 3600
    with np.errstate(invalid="ignore"):
        pers = find[:, j["prof"]] | (find[:, j["dropout"]] & (silent_h >= PERSIST_H))
    p = (st == "suspect") & ~a & pers
    st[p], rule[p] = "bad", "P"
    c = (X.fn.eq("Count") & X.c1_pre.astype(bool) & (X.hi_min_c >= 30)).to_numpy() & np.isin(st, ["ok", "watch"])
    st[c], rule[c] = "suspect", "C"
    with np.errstate(invalid="ignore"):
        even = ((X.fe_n.to_numpy(float) >= hs.PHI_MIN_N) & (X.fe_phi.to_numpy(float) <= hs.PHI_MAX)
                & X.fn.isin(F_FNS).to_numpy())
    only_rapid = find[:, j["rapid"]] & (find.sum(1) == 1)
    f = (st == "bad") & only_rapid & even
    st[f], rule[f] = "suspect", "F"
    left = [",".join(k for k in K if find[i, j[k]]) for i in range(n)]
    bl = [",".join(k for k in K if border[i, j[k]]) for i in range(n)]
    return dict(st=st, left=np.array(left, object), border=np.array(bl, object), rule=rule, score=hsc, n_ev=n_ev,
                silent_h=silent_h, **{f"w_{k}": v for k, v in watch.items()})


def config_notes(X):
    """C1 extension time on a count zone (ONs logged again without an OFF, or most of its time ON in such holds);
    C2 a Count zone in pulse mode that holds ON."""
    occ, rts = X.occ.to_numpy(float), X.rep_time_s.to_numpy(float)
    on_s = occ * X.hours.to_numpy(float) * 3600
    with np.errstate(invalid="ignore"):
        d1 = (rts >= D1_MIN_S) & (rts >= D1_SHARE * on_s)
        nrep, non = X.n_rep.to_numpy(float), X.n_on.to_numpy(float)
        c1 = X.fn.eq("Count").to_numpy() & (d1 | ((nrep >= C1_N) & (nrep >= C1_SHARE * non)))
        c2 = (X.fn.eq("Count") & X.cmode.eq("pulse")).to_numpy() & (
            (X.n_ge60.to_numpy(float) >= C2_N60) | (X.n_ge5.to_numpy(float) >= C2_N5)) & ~c1
    return np.where(c1, "C1", np.where(c2, "C2", ""))


# ============================================================================ main entry
def _stage1(P, start, end, phase, fprob, lanes, pconf, stage1=None):
    """stage-1 rules (health_core), two passes: detectors pass 1 calls bad are nobody's yardstick in pass 2."""
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


def _left(X: pd.DataFrame, *dfs: pd.DataFrame) -> pd.DataFrame:
    """X.merge(df, on="detector", how="left") for each df with one row per detector: its columns looked up by
    detector (reindex fills a missing detector exactly as the merge does), all joined in one concat."""
    parts, seen = [X], set(X.columns)
    for df in dfs:
        if df.detector.duplicated().any() or (seen & set(df.columns)) - {"detector"}:
            return _left_seq(X, dfs)
        R = df.set_index("detector").reindex(X.detector.to_numpy())
        R.index = X.index
        parts.append(R)
        seen |= set(df.columns)
    return pd.concat(parts, axis=1)


def _left_seq(X, dfs):
    for df in dfs:
        X = X.merge(df, on="detector", how="left")
    return X


def _add(X: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """X with new columns appended in order, one concat (scalars broadcast as a setitem would)."""
    return pd.concat([X, pd.DataFrame(cols, index=X.index)], axis=1)


def _day_type(ts) -> str:
    """the research day whose tables apply: Sunday (weekend) or Monday (weekday)."""
    return "h24_a" if pd.Timestamp(ts).dayofweek >= 5 else "h24_b"


def assess(P, start, end, phase: dict, fn_label: dict, fn_prob: dict, lanes: dict, phase_conf: dict | None = None,
           refs: Refs | None = None, stage1=None, keep_all: bool = False, limits_json: dict | None = None):
    """Health for one signal window.  P = health_core.Prep of the window (allowed events only, de-duplicated);
    phase / fn_label / fn_prob / lanes / phase_conf: the classifier's answers per detector (raw predicted phase,
    final function label, class probabilities, lane span, phase probability).  Returns (table, signal note).
    `keep_all` also returns every internal column and the episode table (research comparison)."""
    refs = refs or _load_refs()
    LJ = limits_json or _load_limits()
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
        return (pd.DataFrame(columns=OUT_COLS), "") if not keep_all else (pd.DataFrame(columns=OUT_COLS), "", None, None)
    H = H.rename(columns={"status": "status2", "status1": "status"})
    H["detector"] = H.detector.astype(int)
    for c in ("s_dropout", "s_stuck", "s_night_drop", "ep_dur", "drop_lam", "drop_b0", "drop_b1", "night_ratio",
              "night_n", "night_exp", "reason"):
        if c not in H:                                   # a check that did not run on this sample has no column
            H[c] = np.nan
    if "drop_in_stuck" not in H:
        H["drop_in_stuck"] = False
    # ---- the scored table: detectors with a predicted function the checks know
    X = H[H.detector.map(fn_label).isin(FNS)].copy().reset_index(drop=True)
    rest = H[~H.detector.isin(X.detector)].copy()
    if not len(X):
        out = _fallback(rest)
        return (out, "") if not keep_all else (out, "", None, None)
    Pm = np.array([[fn_prob.get(int(d), {}).get(c, np.nan) for c in C7] for d in X.detector], float)
    top = np.nanmax(np.where(np.isfinite(Pm), Pm, -1), 1)
    alts = _alts(Pm)
    O = hs.occ_bins(P, start, end, bs, fn_label, phase)
    dmap = {int(d): i for i, d in enumerate(O["dets"])}
    cols = {"fn": X.detector.map(fn_label), "phase": X.detector.map(phase).astype(float),
            "lanes": X.detector.map(lanes).astype(float)}
    cols.update(f_top=np.where(top >= 0, top, np.nan), alt_set=alts,
                alt_fns=[",".join(c for c in C7 if c in a) for a in alts], hours=hours, wg=wg, t_start=start,
                occ=[O["mocc"][dmap[d]] if d in dmap else np.nan for d in X.detector])
    X = _add(X, cols)
    # ---- bins, context
    M = hs.bins_ctx(O, st1)
    per_bins = {"bs": bs, "hi": {}, "q5": {}, "fast": {}, "spiky": {}}
    Q5, cstate = hs.q5_stats(P, T, phase, lanes, per_bins["q5"])
    tf = LJ["too_fast"]
    TF = hs.too_fast(P, fn_label, lanes, tf["gap_s_stopbar"], tf["gap_s_other"], per_bins["fast"])
    X = _left(X, hs.act_stats(P), hs.act118(P, T), hs.slopes(M, m30), hs.like_corr(M, m30), Q5,
              hs.fast_even(P, T, cstate), TF)
    span = pd.Series(np.where(X.lanes >= 2, "2+", "1"), index=X.index)
    rate = X.n_on / X.hours
    X = _add(X, {"span": span, "rate": rate, "band": hs.band_of(rate), "cmode": _cmode(X.fn, X.dur_p50),
                 "type": X.fn + " " + span,
                 "stuck_x": np.fmax(pd.to_numeric(X.ep_dur, errors="coerce"), X.max_on_s)})
    for c in ("elhi_cnt", "elhi_occ", "c_occ_like"):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X["queue_pat8"] = ((X.elhi_occ > 0) & (X.elhi_occ > X.elhi_cnt)).to_numpy()
    hm = X.status.ne("bad").astype(int)
    ph = X.phase.fillna(-1)
    X["n_hmates"] = (hm.groupby(ph).transform("sum") - hm).where(X.phase.notna())
    # ---- 5-min bins: hourly GEH (count drops, night misses) and 10-min bursts (spiky counts), from 3 h of data
    B5 = hc.events_to_bins(P, start, end, None)
    X = _left(X, hs.occ_hi(M, dict(zip(X.detector, X.cmode)), refs, P, T, per_bins["hi"]))
    if hours >= LJ["min_hours_counts"]:
        SP, per_bins["spiky"] = hk.spiky(B5, phase, hours, LJ["spiky"])
        X = _left(X, hk.hourly_geh(B5, phase), SP)
    else:
        X = _add(X, {c: np.nan for c in GEH_COLS})
        X["geh_ok"] = False
    # ---- episodes (continuous ONs >= 5 min) with the queue / traffic test
    ep = hc.on_episodes(P, start, end, min_s=300.0)
    E = ep[ep.dur_s >= 300].copy().reset_index(drop=True)
    E["ts"] = (E.t0 - start).dt.total_seconds()
    E["tf"] = (E.t1 - start).dt.total_seconds()
    E["co5"] = hs.episode_overlap(E)
    TW = hk.twin_matrix(B5)
    E["n_ov_any"], E["common"] = hk.common_holds(E.ts.to_numpy(float), E.tf.to_numpy(float), E.detector.to_numpy(int),
                                                 LJ["common_hold_s"], twin=TW)
    E = E.join(hs.episodes_ctx(E, M, dict(zip(X.detector, X.phase)), refs.qtype, dict(zip(X.detector, X.type))))
    E["fn"] = E.detector.map(dict(zip(X.detector, X.fn)))
    # ---- limits, time of day (a whole day in the sample)
    L, Bl = limits(X, refs, LJ)
    X, sig_note, day_info = _time_of_day(X, O, start, end, refs, L["prof"], LJ["busy"])
    # ---- Count zone held ON / extension; the queue excuse is never for a counting zone
    cfg = config_notes(X)
    c1 = cfg == "C1"
    sure_count = np.array([(not a) or a <= {"Count"} for a in alts])
    X["count_on_ok"] = X.fn.eq("Count").to_numpy() & sure_count & ~c1
    X["c1_pre"] = c1
    noq = X.fn.isin(NO_SHARED).to_numpy() & np.array([not (a & QUEUE_OK) for a in alts])
    if len(E):
        E["noq"] = E.detector.map(dict(zip(X.detector, noq))).fillna(False).astype(bool)
        E.loc[E.noq, "n_hpeer_e"] = 0
        qc = E.cover_e.fillna(0) >= F4_COVER
        E.loc[qc, "phx_h_e"] = np.fmax(E.loc[qc, "phx_h_e"].fillna(0), 1.5)
    F = episode_flags(E)
    F["twin"] = TW
    # ---- goes silent: a pre-pass finds phase mates with a finding of their own; they leave the yardstick
    for c in ("drop_lam", "drop_b0", "drop_b1"):
        X[c] = pd.to_numeric(X[c], errors="coerce")
    cand = X[(X.drop_lam >= 30) & (X.n_on > 0)].detector.tolist()
    if cand:
        S0, C0 = scores(X, E, F, L, Bl, LJ)
        R0 = status(X, S0, C0)
        quietk = {"dropout", "level", "night_drop"}
        flagged = {int(d): bool(set(x for x in l.split(",") if x) - quietk) for d, l in zip(X.detector, R0["left"])}
        DC = hs.dropout_clean(B5, cand, dict(zip(X.detector, X.phase)), flagged)
    else:
        DC = {}
    cols = {c: X.detector.map({d: v[c] for d, v in DC.items()}).astype(float)
            for c in ("drop_lam_chk", "drop_lam_c", "drop_b0_c", "drop_b1_c", "n_dropped")}
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
    # ---- the checks, status, severity, outputs
    S, C = scores(X, E, F, L, Bl, LJ)
    R = status(X, S, C)
    X = _add(X, {**{f"s8_{k}": np.asarray(v, float) for k, v in S.items()},
                 **{f"lim_{k}": np.asarray(v, float) for k, v in L.items()},
                 **{f"bad_{k}": np.asarray(v, float) for k, v in Bl.items()}, **C,
                 "health_st": R["st"], "left": R["left"], "border": R["border"], "sev_rule": R["rule"],
                 "score": R["score"], "n_ev": R["n_ev"], "silent_h": R["silent_h"], "w_no_yardstick": R["w_no_yardstick"],
                 "w_dropout_short": R["w_dropout_short"],
                 "cfg": cfg})
    out = _outputs(X, E, F, start, end, per_bins)
    held = _common_note(X, E, F)
    sig_note = "; ".join(x for x in (sig_note, held) if x)
    if len(rest):
        out = pd.concat([out, _fallback(rest)], ignore_index=True)
    out = out.sort_values("detector").reset_index(drop=True)
    if keep_all:
        return out, sig_note, X, E
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


def _time_of_day(X, O, start, end, refs, zlim, busy=None):
    """busy at night per whole day of the sample (the day with the strongest finding is kept); signal note when many
    detectors had an unusual night (a signal-wide event)."""
    num = ("tod_z", "tod_dph", "tod_n_night", "tod_r", "tod_mate_r")
    n = len(X)
    V = {c: np.full(n, np.nan) for c in num}
    tod_day = [pd.NaT] * n
    note, info = "", {}
    days = _full_days(start, end) if (n and X.wg.iloc[0] == "h24" and O["bs"] == 900) else []
    dmap = {int(d): i for i, d in enumerate(O["dets"])}
    idx = np.array([dmap.get(int(d), -1) for d in X.detector])
    best = {}
    sig = []
    for d0 in days:
        j0 = int(round((d0 - start).total_seconds() / 900))
        if j0 < 0 or j0 + 96 > O["nb"]:
            continue
        n15 = np.full((n, 96), np.nan)
        o15 = np.full((n, 96), np.nan)
        hv = idx >= 0
        n15[hv] = O["n"][idx[hv], j0:j0 + 96]
        o15[hv] = O["occ"][idx[hv], j0:j0 + 96]
        if d0 != d0.normalize():                      # not midnight-aligned: order the 96 bins by clock time
            sh = int(((d0 - d0.normalize()).total_seconds() // 900) % 96)
            n15, o15 = np.roll(n15, sh, 1), np.roll(o15, sh, 1)
        N, Od = hs.day_profile(n15, o15)
        tot = np.nansum(N, 1)
        ok = np.isfinite(N).all(1) & np.isfinite(Od).all(1) & (tot >= hs.MIN_N)
        nd = hs.night_day(N, Od, X.fn.to_numpy(object), X.span.to_numpy(object), X.phase.to_numpy(float), ok,
                          refs.tod_own, refs.tod_alt, list(X.alt_set))
        odd = ok & nd.tod_odd.to_numpy()
        if odd.sum() >= hs.SIG_MIN and odd.sum() >= hs.SIG_SHARE * ok.sum():
            sig.append((d0, int(odd.sum()), int(ok.sum())))
        lvl = hs.busy_level(nd.tod_z, nd.tod_dph, nd.tod_n_night, zlim, busy)
        key = np.where(np.isfinite(lvl), lvl, -1.0)
        for i in range(n):
            if i not in best or key[i] > best[i][0]:
                best[i] = (key[i], d0, nd.iloc[i])
                info[int(X.detector.iat[i])] = dict(day=d0)
    for i, (_, d0, ndr) in best.items():
        for c in num:
            V[c][i] = float(ndr[c])
        tod_day[i] = d0
    X = _add(X, {**V, "tod_day": pd.Series(tod_day, index=X.index, dtype="datetime64[ns]")})
    if sig:
        d0, k, m = max(sig, key=lambda s: s[1] / max(s[2], 1))
        note = (f"{k} of {m} detectors at this signal had an unusual night on {d0:%a %d %b}: "
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


def _evidence(k, r, eps):
    """one short line per finding with the numbers the check used."""
    if k == "stuck":
        e = eps.get(int(r.detector), [])
        cnt = [x for x in e if x["label"] != "queue"]
        tot = sum(x["dur_s"] for x in cnt)
        nq = [x for x in cnt if x["label"] == "not_queue"]
        why = ""
        if nq and len(nq) == len(cnt):            # note 135: excused as a queue before, the traffic does not explain it
            why = (f"; not a queue: at the same traffic it was ON {100 * min(x['busy_occ'] for x in nq):.0f} % "
                   "of the time")
        tog = [x for x in cnt if x["label"] == "together"]
        if tog:                                   # note 147e: one common failure with other detectors
            mates = ", ".join(str(m) for m in sorted({m for x in tog for m in x["mates"]}))
            when = (f"{_dur(tog[0]['dur_s'])} from {tog[0]['t0']:%a %H:%M}" if len(tog) == 1 else
                    f"{len(tog)} times from {tog[0]['t0']:%a %H:%M}, {_dur(sum(x['dur_s'] for x in tog))} in total")
            return f"held ON {when} together with det {mates} (same start and end: a common failure)"
        if len(cnt) > 1:
            return f"held ON {len(cnt)} times, {_dur(tot)} in total" + why
        if cnt:
            return f"held ON {_dur(tot)} from {cnt[0]['t0']:%a %H:%M}" + why
        return f"held ON up to {_dur(r.stuck_x)}"
    if k == "count_on":
        mx = r.hi_occ_max if np.isfinite(r.hi_occ_max) else np.nan
        return f"count zone ON up to {100 * mx:.0f} % in light traffic for {r.hi_min_c:.0f} min"
    if k == "dropout" or (k == "night_drop" and bool(r.silent_night)):
        a = r.t_start + _td(seconds=300 * float(r.drop_b0))
        b = r.t_start + _td(seconds=300 * float(r.drop_b1))
        return f"no actuations {_hm(a)}-{_hm(b)}; ~{float(r.drop_lam):,.0f} expected"
    if k == "night_drop":
        nn, ne = float(r.geh_night_n), float(r.geh_night_exp)
        return (f"night {nn:,.0f} actuations vs ~{ne:,.0f} expected (GEH {r.geh_night:.1f})" if np.isfinite(ne)
                else "fewer actuations at night")
    if k == "level":
        t = f" from {pd.Timestamp(r.drop_t):%a %H:%M}" if pd.notna(r.drop_t) else ""
        return (f"counts dropped{t}: {r.own_after:.0f} per hour where its phase mates' traffic gives ~{r.exp_after:.0f} "
                f"(GEH {r.geh_drop:.1f}); its occupancy fell too")
    if k == "rapid":
        txt = (f"{int(r.n_fast):,} of {int(r.N):,} ONs follow the previous ON within {r.gap_lim:.2g} s "
               f"({r.fast_rate:.0f} per 1,000; limit {r.lim_rapid:.0f})")
        if np.isfinite(r.pair_share) and r.pair_share >= PAIRS_SHARE:   # note 147e: short gaps come in pairs
            txt += "; counts each vehicle twice"
        if r.sev_rule == "F":                     # note 135 rule F: an even excess
            txt += ", spread evenly with traffic - flickers as vehicles pass, check zone setup"
        return txt
    if k == "spiky":
        return (f"{r.spk_bursts:.0f} bursts{' a day' if r.hours > 24.5 else ''} in 10-min counts at 3 x its usual "
                f"level or more (largest GEH {r.spk_gmax:.0f}); ~{r.spk_daily:,.0f} vehicles a day")
    if k == "volume":
        return f"busiest 5 min {r.q5_use:,.0f} veh/h/lane vs limit {r.lim_volume:,.0f}"
    if k == "prof":
        return (f"night level {100 * r.tod_r ** 2:.0f} % of its busiest hours; phase mates "
                + (f"{100 * r.tod_mate_r ** 2:.0f} %" if np.isfinite(r.tod_mate_r) else "none"))
    if k == "C1":
        return f"{int(r.n_rep):,} of {int(r.n_on):,} ONs logged again without an OFF"
    if k == "C2":
        return f"{int(r.n_ge5)} ONs over 5 s, longest {_dur(r.long_max_s)}" if np.isfinite(r.long_max_s) else \
            "holds ON although set to pulse"
    return ""


def _stuck_eps(X, E, F):
    """every continuous ON over the detector's stuck-on limit, labelled counted / queue / shared / not_queue."""
    out = {}
    if not len(E):
        return out
    lim = dict(zip(X.detector.astype(int), X.lim_stuck))
    Le = E.detector.astype(int).map(lim).to_numpy(float)
    with np.errstate(invalid="ignore"):
        m = F["dur"] >= Le
    if not m.any():
        return out
    q = F["q1"] | F["q1b"]
    sh = F["shared"] & ~q
    lab = np.where(F["common"], "together",
                   np.where(q, "queue", np.where(sh, "shared", np.where(F["lifted"], "not_queue", "counted"))))
    s_, f_, c_ = E.ts.to_numpy(float), E.tf.to_numpy(float), E.detector.to_numpy(int)
    bo = E.busy_occ_e.to_numpy(float) if "busy_occ_e" in E else np.full(len(E), np.nan)
    ii = np.flatnonzero(m)
    ii = ii[np.argsort(E.t0.to_numpy()[ii], kind="stable")]
    t0, t1, det = E.t0.tolist(), E.t1.tolist(), E.detector.astype(int).tolist()
    for i in ii:
        out.setdefault(det[i], []).append(dict(t0=t0[i], t1=t1[i], dur_s=float(F["dur"][i]), label=lab[i],
                                               busy_occ=float(bo[i]),
                                               mates=hk.common_mates(s_, f_, c_, i, F.get("twin")) if F["common"][i] else []))
    return out


def _common_note(X, E, F):
    """signal note for holds shared by several detectors with the same start and end (a common failure)."""
    if not len(E) or not F["common"].any():
        return ""
    lim = dict(zip(X.detector.astype(int), X.lim_stuck))
    Le = E.detector.astype(int).map(lim).to_numpy(float)
    with np.errstate(invalid="ignore"):
        ii = np.flatnonzero(F["common"] & (F["dur"] >= Le))
    if not len(ii):
        return ""
    s_, f_, c_ = E.ts.to_numpy(float), E.tf.to_numpy(float), E.detector.to_numpy(int)
    groups = {}
    for i in ii[np.argsort(s_[ii], kind="stable")]:
        g = tuple(sorted(set(hk.common_mates(s_, f_, c_, i, F.get("twin"))) | {int(c_[i])}))
        t = groups.setdefault(g, [i, set()])
        t[1].add(round(s_[i] / 60))                            # one event per start minute
    parts = []
    for g, (i, ev) in sorted(groups.items(), key=lambda x: s_[x[1][0]]):
        k = len(ev)
        parts.append(f"{len(g)} detectors (det {', '.join(map(str, g))}) held ON together "
                     + (f"from {E.t0.iat[i]:%a %H:%M} for {_dur(F['dur'][i])}" if k == 1 else
                        f"{k} times from {E.t0.iat[i]:%a %H:%M}")
                     + " (same start and end): one common failure, check the cabinet first")
    return "; ".join(parts)


def _runs_to_periods(mask, t0, bin_s, end, what):
    """consecutive flagged bins (bin k = [t0 + k * bin_s, t0 + (k + 1) * bin_s), clipped to the window end) -> one
    period each."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for i, j in zip(a, b):
        s0 = t0 + _td(seconds=float(i) * bin_s)
        s1 = min(t0 + _td(seconds=float(j) * bin_s), end)
        out.append(dict(start=str(s0), end=str(s1), what=what, recovered=None))
    return out


def _periods(r, keys, eps, per_bins, start, end):
    """the bad periods of a flagged detector: where each check saw the problem."""
    d = int(r.detector)
    out = []
    if "stuck" in keys:
        out += [dict(start=str(x["t0"]), end=str(x["t1"]), what="stuck on", recovered=None)
                for x in eps.get(d, []) if x["label"] != "queue"]
    if "count_on" in keys and d in per_bins["hi"]:
        out += _runs_to_periods(per_bins["hi"][d], start, int(per_bins["bs"]), end, "count zone held ON")
    for k, what in (("dropout", "silent"), ("night_drop", "silent at night")):
        if k in keys and np.isfinite(r.drop_b0) and np.isfinite(r.drop_b1) and (k == "dropout" or bool(r.silent_night)):
            out.append(dict(start=str(r.t_start + _td(seconds=300 * float(r.drop_b0))),
                            end=str(r.t_start + _td(seconds=300 * float(r.drop_b1))), what=what, recovered=None))
    if "level" in keys and pd.notna(r.drop_t):
        out.append(dict(start=str(pd.Timestamp(r.drop_t)), end=str(end), what="counts dropped", recovered=None))
    if "rapid" in keys and d in per_bins["fast"]:
        hf, ha = per_bins["fast"][d]
        nh = int(max(ha.max() if len(ha) else 0, hf.max() if len(hf) else 0)) + 1
        cf, ca = np.bincount(hf, minlength=nh), np.bincount(ha, minlength=nh)
        with np.errstate(invalid="ignore", divide="ignore"):
            hit = (cf >= FAST_PER_H) & (1000.0 * cf / np.maximum(ca, 1) >= r.lim_rapid)
        out += _runs_to_periods(hit, start, 3600, end, "too-fast actuations")
    if "volume" in keys and per_bins["q5"].get("det") is not None:
        i = np.flatnonzero(per_bins["q5"]["det"] == d)
        if len(i) and np.isfinite(r.lim_volume):
            q = per_bins["q5"]["q5_all" if r.fn == "Advance" else "q5_gy"][i[0]]
            with np.errstate(invalid="ignore"):
                out += _runs_to_periods(q > r.lim_volume, start, 300, end, "too many for the traffic")
    if "prof" in keys and pd.notna(r.tod_day):
        a = pd.Timestamp(r.tod_day) + _td(hours=hs.NIGHT_TOD[0])
        out.append(dict(start=str(a), end=str(a + _td(hours=hs.NIGHT_TOD[1] - hs.NIGHT_TOD[0])), what="busy at night",
                        recovered=None))
    if "spiky" in keys and d in per_bins["spiky"]:
        bm = _td(minutes=10)
        out += [dict(start=str(start + a * bm), end=str(min(start + b * bm, end)), what="spiky counts", recovered=None)
                for a, b in per_bins["spiky"][d]]
    return sorted(out, key=lambda x: x["start"])


def _outputs(X, E, F, start, end, per_bins):
    eps = _stuck_eps(X, E, F)
    rows = []
    for r in hc.rows_of(X):
        st = r.health_st
        keys = [k for k in str(r.left).split(",") if k]
        if r.sev_rule == "C" and "count_on" not in keys:
            keys.append("count_on")
        sc = {k: (getattr(r, f"s8_{k}") if np.isfinite(getattr(r, f"s8_{k}")) else 0.0) for k in keys}
        # one category per check: its severity is its worst finding's
        cats = {}
        for k in keys:
            c = CATEGORY[k]
            sev = "bad" if sc[k] >= 0.75 else "suspect"
            if (r.sev_rule == "C" and k == "count_on") or (r.sev_rule == "F" and k == "rapid"):
                sev = "suspect"
            elif r.sev_rule == "P" and (k == "prof" or (k == "dropout" and r.silent_h >= PERSIST_H)):
                sev = "bad"
            v = cats.setdefault(c, {"sev": sev, "s": sc[k], "keys": []})
            v["keys"].append(k)
            if sev == "bad":
                v["sev"] = "bad"
            v["s"] = max(v["s"], sc[k])
        order = sorted(cats, key=lambda c: (-cats[c]["s"], KEYS.index(cats[c]["keys"][0])))
        lane = _lanes_txt(r)
        lab = _label(r) + (f" ({lane})" if lane and set(keys) & LANE_CATS else "")
        catl = [f"{c} ({cats[c]['sev']})" for c in order]
        parts = [f"{c}: " + "; ".join(_evidence(k, r, eps) for k in cats[c]["keys"]) for c in order]
        wtxt = []
        if r.w_no_yardstick:
            wtxt.append("no healthy phase mate to compare with")
        for k in [x for x in str(r.border).split(",") if x] + (["dropout"] if r.w_dropout_short else []):
            wtxt.append(f"borderline {CATEGORY[k].lower()}")
        wtxt = list(dict.fromkeys(wtxt))
        if st in ("suspect", "bad"):
            reason = f"{lab} - " + "; ".join(parts) if parts else f"{lab} - {st}"
        elif st == "watch":
            reason = f"{lab} - watch: " + "; ".join(wtxt) if wtxt else f"{lab} - watch"
        elif st == "not_enough_data":
            reason = f"{lab} - too few actuations to judge"
        else:
            reason = f"{lab} - nothing unusual"
        conf = f"{CONFIG[r.cfg]}: {_evidence(r.cfg, r, eps)}" if r.cfg else ""
        per = _periods(r, keys, eps, per_bins, start, end) if st in ("suspect", "bad") else []
        rows.append(dict(detector=int(r.detector), health_status=st,
                         health_score=float(r.score) if np.isfinite(r.score) else np.nan,
                         health_reason=reason, health_bad_periods=per, health_categories="; ".join(catl),
                         health_config=conf, health_watch="; ".join(wtxt)))
    return pd.DataFrame(rows, columns=OUT_COLS)


def _fallback(H):
    """detectors without a known predicted function: the stage-1 rule checks as they are."""
    return pd.DataFrame({"detector": H.detector.astype(int), "health_status": H.status2 if "status2" in H else H.status,
                         "health_score": H.health_score,
                         "health_reason": [f"det {int(d)}: " + str(x) for d, x in zip(H.detector, H.reason.fillna(""))],
                         "health_bad_periods": [[dict(start=str(p["start"]), end=str(p["end"]), what=p["what"],
                                                      recovered=p.get("recovered")) for p in (L or [])]
                                                if isinstance(L, list) else [] for L in H.bad_periods],
                         "health_categories": "", "health_config": "", "health_watch": ""})
