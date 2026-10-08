"""Lane output (note 42): per predicted phase the number of lanes, per detector the lane(s) it covers.

    from lane_output import lanes
    phase_tab, det_tab = lanes(events, predictions)            # numpy / pandas only

Runs AFTER the phase / function prediction and reads only the hi-res log (81 / 82) plus the
predicted phase and function of each detector.  Phase-anonymous: the phase number is used only
as a grouping key (which detectors share a phase), never as a feature.

Lane numbering (a convention, not geometry -- the log cannot tell left from right, nor one
approach from another; lanes are counted per phase in TOTAL, over all its approaches):
    lane 1 = the busiest lane of the phase, lane 2 the next, ...  A lane's volume is the largest
    actuation rate among the single-lane Advance / Presence / Count / Yellow_Red detectors in it.
A detector that spans lanes lists all of them ("1,2").

Pipeline
 1. ON times of completed actuations (82 followed by 81; de-duplicated, channels <= 64).
 2. Pair cues for every pair of detectors on the same predicted phase (both >= MIN_ON
    actuations): ON-time cross-correlogram (zero lag, 2-8 s lead; all hours and 22-06),
    1-min high-pass count correlation and 15-min count correlation (off-peak / peak thirds of
    the signal's own 15-min volume), volumes, window length, predicted-function pair type.
    Same definitions as research/code/trackA/lr2_pairs.py (SQL there, numpy here).
 3. Pair model (LightGBM binary, exported text, run by model/lgbm_numpy.NumpyBooster):
    P(same lane).
 4. Constrained decode per phase: for L = 1..4 lanes, the lane-set assignment maximising
        sum over pairs of log P(same) [lane sets overlap] or log P(diff) [disjoint]
        + log P(span size | function) - lam * L,
    subject to <= 1 single-lane Advance / Presence / Count / Yellow_Red per lane and every lane
    holding at least one single-lane "anchor" (a detector predicted A / P / C / YR).  Mid / Other
    detectors join lanes (often spanning) but never create one.  Bike detectors get no lane.
    Exhaustive over the anchors' partitions (<= 9 anchors), then coordinate ascent over every
    detector's lane set (any non-empty subset: spanning allowed).  L = argmax.
 5. Confidence: n_lanes_conf = softmax of the best score per L; a detector's lane_conf =
    n_lanes_conf * sigmoid(best score - best score with that detector moved elsewhere).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ON = 10                 # detectors with fewer actuations in the window get no lane
MAX_LANES = 4
MAX_CHANNEL = 64
ANCHOR = ("Advance", "Presence", "Count", "Yellow_Red")
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
NOLANE = ("Bike",)
HALF = 0.5                  # correlogram bin, s
MAXLAG = 15.0
LAGS = np.arange(-30, 30)   # bin lb covers [lb/2, lb/2 + 0.5)
MID = LAGS * HALF + 0.25

CUE_COLS = ["z0_ratio_all", "z0_sharp_all", "z0_exc_all", "z0_ratio_q", "z0_sharp_q", "z0_exc_q",
            "lead_ratio_all", "lead_sharp_all", "lead_exc_all", "lead_asym_abs",
            "lead_ratio_q", "lead_sharp_q", "lead_exc_q",
            "c_all", "c_off", "c_peak", "c_div", "r_shift", "hp_all", "hp_off", "hp_peak", "bal"]
CTX_COLS = ["log_na", "log_nb", "log_ratio", "log_rate_a", "log_rate_b", "n_det_phase",
            "win_hours"]
PT_COLS = [f"pt_{c}" for c in C7]
FEATURES = CUE_COLS + CTX_COLS + PT_COLS


# ------------------------------------------------------------------------------ events
def on_times(events: pd.DataFrame) -> dict:
    """{(DeviceId, det): (t seconds since epoch sorted, hour-of-day)}: the ON (82) of every
    completed actuation -- an 82 whose next 81/82 event on the channel is an 81 (the interval rule
    of model/predict.build_chunk_tables; unpaired ONs dropped).  Channels > 64 dropped,
    exact duplicate rows dropped."""
    e = events
    e = e[e.EventId.isin([81, 82]) & (e.Parameter <= MAX_CHANNEL)][
        ["DeviceId", "Timestamp", "EventId", "Parameter"]].drop_duplicates()
    ts = pd.to_datetime(e.Timestamp)
    e = e.assign(t=(ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(),
                 h=ts.dt.hour.to_numpy(), k=(e.EventId == 81).astype(int).to_numpy())
    e = e.sort_values(["DeviceId", "Parameter", "t", "k"], kind="stable")
    nxt = e.groupby(["DeviceId", "Parameter"], sort=False).k.shift(-1)
    e = e[(e.k == 0) & (nxt == 1)]
    out = {}
    for (dev, det), g in e.groupby(["DeviceId", "Parameter"], sort=False):
        out[(str(dev), int(det))] = (g.t.to_numpy().astype(float), g.h.to_numpy())
    return out


def content_key(t, t0: float) -> tuple:
    """note 77 (= package lanes.content_key): (actuations, sum and a hash of the ON offsets from t0 in 0.1 s) -- the
    canonical behavioural order of a phase's detectors in the decode; never the channel number."""
    r = np.rint((np.asarray(t, float) - t0) * 10.0).astype(np.int64)
    return int(len(r)), int(r.sum()), int(((r % 65521) * (r % 65519) % 1000003).sum())


def _unit(ts) -> float:
    return pd.Timestamp(ts).value / 1e9


# ------------------------------------------------------------------------------ cues
def xcorr_hist(ta: np.ndarray, qa: np.ndarray, tb: np.ndarray):
    """Counts of b - a in 0.5 s bins over |lag| < 15 s, for every a and for quiet-hour a."""
    lo = np.searchsorted(tb, ta - MAXLAG, "right")
    hi = np.searchsorted(tb, ta + MAXLAG, "left")
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        return np.zeros(60), np.zeros(60)
    ia = np.repeat(np.arange(len(ta)), cnt)
    start = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt)
    jb = start + np.arange(tot)
    lb = np.floor((tb[jb] - ta[ia]) / HALF).astype(np.int64) + 30
    lb = np.clip(lb, 0, 59)
    return (np.bincount(lb, minlength=60).astype(float),
            np.bincount(lb, weights=qa[ia].astype(float), minlength=60))


@np.errstate(divide="ignore", invalid="ignore")
def _cor_cues(M: np.ndarray, tag: str) -> dict:
    """lr2_pairs.cues for a stack of correlograms (rows = pairs)."""
    base = M[:, np.abs(MID) >= 10].mean(1).clip(min=0.5)
    z = np.abs(MID) < 0.5
    fwd = (MID >= 1.5) & (MID <= 8.5)
    bwd = (MID <= -1.5) & (MID >= -8.5)
    near = (np.abs(MID) >= 1.0) & (np.abs(MID) < 3.0)
    d = {}
    d[f"z0_ratio_{tag}"] = np.log(M[:, z].mean(1).clip(min=0.1) / base)
    d[f"z0_sharp_{tag}"] = np.log(M[:, z].mean(1).clip(min=0.1) / M[:, near].mean(1).clip(min=0.5))
    d[f"z0_exc_{tag}"] = (M[:, z] - base[:, None]).sum(1)
    pf, pb = M[:, fwd].max(1), M[:, bwd].max(1)
    d[f"lead_ratio_{tag}"] = np.log(np.maximum(pf, pb) / base)
    band = np.where(pf >= pb, 0, 1)
    MM = np.where(band[:, None] == 0, M[:, fwd], M[:, bwd][:, ::-1])
    run = MM[:, :-1] + MM[:, 1:]
    d[f"lead_sharp_{tag}"] = np.log(run.max(1).clip(min=0.5) / (2 * np.median(MM, 1)).clip(min=0.5))
    d[f"lead_exc_{tag}"] = run.max(1) - 2 * base
    d[f"lead_asym_{tag}"] = np.log((M[:, fwd].sum(1) + 1) / (M[:, bwd].sum(1) + 1))
    return d


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else np.nan


def signal_pair_cues(on: dict, dets: list, pairs: list, t0: float, t1: float) -> pd.DataFrame:
    """Cues for `pairs` [(da, db) with da < db] of one signal.  `on` {det: (t, hour)} for the
    detectors of the signal; `dets` = every detector of the signal used for the volume levels
    (>= MIN_ON actuations).  Times in seconds; window [t0, t1)."""
    nm = int((t1 - t0) // 60)
    if nm < 1 or not pairs:
        return pd.DataFrame()
    rel, quiet = {}, {}
    for d in dets:
        t, h = on[d]
        m = (t >= t0) & (t < t1)
        rel[d] = t[m] - t0
        quiet[d] = (h[m] >= 22) | (h[m] < 6)
    # minute counts, 15-min block residuals and volume levels (lr2 SQL_MIN)
    idx = {d: i for i, d in enumerate(dets)}
    C = np.zeros((len(dets), nm))
    for d, i in idx.items():
        mm = np.floor(rel[d] / 60).astype(np.int64)
        mm = mm[(mm >= 0) & (mm < nm)]
        C[i] = np.bincount(mm, minlength=nm)[:nm]
    blk = np.arange(nm) // 15
    nb = int(blk[-1]) + 1
    bsize = np.bincount(blk, minlength=nb).astype(float)
    B = np.zeros((len(dets), nb))
    for b in range(nb):
        B[:, b] = C[:, blk == b].sum(1)
    R = C - (B / bsize)[:, blk]
    tn = B.sum(0)
    if nb > 1:
        srt = np.sort(tn)
        pr = np.searchsorted(srt, tn, "left") / (nb - 1)
    else:
        pr = np.zeros(1)
    lv = np.where(pr >= 0.67, 2, np.where(pr <= 0.33, 0, 1))
    mlv = lv[blk]
    rows, Ha, Hq = [], [], []
    for da, db in pairs:
        ta, tb = rel[da], rel[db]
        h, hq = xcorr_hist(ta, quiet[da], tb)
        Ha.append(h)
        Hq.append(hq)
        ia, ib = idx[da], idx[db]
        ra, rb = R[ia], R[ib]
        ba, bb = B[ia], B[ib]
        r = {"da": da, "db": db, "n_a": len(ta), "n_b": len(tb),
             "nq_a": int(quiet[da].sum()), "nq_b": int(quiet[db].sum()),
             "hp_all": _corr(ra, rb), "hp_off": _corr(ra[mlv == 0], rb[mlv == 0]),
             "hp_peak": _corr(ra[mlv == 2], rb[mlv == 2]),
             "c_all": _corr(ba, bb), "c_off": _corr(ba[lv == 0], bb[lv == 0]),
             "c_peak": _corr(ba[lv == 2], bb[lv == 2])}
        sa2, sa0 = ba[lv == 2].sum(), ba[lv == 0].sum()
        r["r_peak"] = bb[lv == 2].sum() / sa2 if sa2 > 0 else np.nan
        r["r_off"] = bb[lv == 0].sum() / sa0 if sa0 > 0 else np.nan
        rows.append(r)
    P = pd.DataFrame(rows)
    Ha, Hq = np.array(Ha), np.array(Hq)
    for k, v in _cor_cues(Ha, "all").items():
        P[k] = v
    qc = _cor_cues(Hq, "q")
    noq = Hq.sum(1) == 0
    for k, v in qc.items():
        P[k] = np.where(noq, np.nan, v)
    for t, na, nb_ in (("all", "n_a", "n_b"), ("q", "nq_a", "nq_b")):
        mn = np.minimum(P[na], P[nb_]).clip(lower=1)
        P[f"z0_exc_{t}"] = P[f"z0_exc_{t}"] / mn
        P[f"lead_exc_{t}"] = P[f"lead_exc_{t}"] / mn
    P["bal"] = np.minimum(P.n_a, P.n_b) / np.maximum(P.n_a, P.n_b).clip(lower=1)
    P["c_div"] = P.c_off - P.c_peak
    P["r_shift"] = -np.abs(np.log(P.r_peak.clip(lower=1e-3) / P.r_off.clip(lower=1e-3)))
    P["lead_asym_abs"] = P.lead_asym_all.abs()
    return P.replace([np.inf, -np.inf], np.nan)


def add_context(P: pd.DataFrame, func: dict, n_det_phase: dict, hours: float) -> pd.DataFrame:
    """Volumes, phase size, window length and the predicted-function pair type.
    `func` {det: 7-class function}, `n_det_phase` {det: #detectors (>= MIN_ON) on its phase}."""
    P = P.copy()
    P["log_na"] = np.log1p(P.n_a)
    P["log_nb"] = np.log1p(P.n_b)
    P["log_ratio"] = np.abs(P.log_na - P.log_nb)
    P["log_rate_a"] = np.log1p(P.n_a / hours)
    P["log_rate_b"] = np.log1p(P.n_b / hours)
    P["n_det_phase"] = P.da.map(n_det_phase).astype(float)
    P["win_hours"] = hours
    fa, fb = P.da.map(func), P.db.map(func)
    for c in C7:
        P[f"pt_{c}"] = np.where(fa.notna() & fb.notna(),
                                (fa == c).astype(float) + (fb == c).astype(float), np.nan)
    return P


# ------------------------------------------------------------------------------ decode
_RGS: dict = {}


def partitions(n: int, L: int) -> np.ndarray:
    """All set partitions of n items into exactly L blocks (restricted growth strings)."""
    key = (n, L)
    if key in _RGS:
        return _RGS[key]
    out = []

    def rec(i, cur, mx):
        if n - i < L - (mx + 1):
            return
        if i == n:
            if mx + 1 == L:
                out.append(cur.copy())
            return
        for b in range(min(mx + 2, L)):
            cur.append(b)
            rec(i + 1, cur, max(mx, b))
            cur.pop()
    rec(0, [], -1)
    _RGS[key] = np.array(out, dtype=np.int8).reshape(-1, n)
    return _RGS[key]


class Decoder:
    """Lane-set decode of one phase.  `span_prior[f][k]` = log P(k lanes spanned | function f,
    phase with >= 2 lanes); lam = per-lane penalty; beta = logit shift of P(same)."""

    def __init__(self, span_prior: dict, lam: float = 0.0, beta: float = 0.0,
                 role_pen: float = 4.0, max_exhaustive: int = 9):
        self.span_prior, self.lam, self.beta = span_prior, lam, beta
        self.role_pen, self.max_ex = role_pen, max_exhaustive

    def _sp(self, f: str, k: int) -> float:
        d = self.span_prior.get(f) or self.span_prior.get("Other") or {}
        return d.get(k, min(d.values())) if d else 0.0

    def decode(self, P: np.ndarray, func: list, vol: np.ndarray) -> dict:
        """P: n x n P(same lane) (diag ignored, NaN = unknown -> 0.5); func: predicted function
        per detector; vol: actuation rate.  Returns lanes (list of sorted lane tuples), n_lanes,
        n_lanes_conf, lane_conf, scores per L."""
        n = len(func)
        p = np.clip(np.where(np.isnan(P), 0.5, P), 0.02, 0.98)
        lg = np.log(p / (1 - p)) + self.beta
        ls = -np.log1p(np.exp(-lg))          # log sigmoid = log P(same)
        ld = -np.log1p(np.exp(lg))           # log P(diff)
        np.fill_diagonal(ls, 0)
        np.fill_diagonal(ld, 0)
        W = ls - ld                          # gain of "same" over "diff"
        base = ld[np.triu_indices(n, 1)].sum()
        anc = np.array([f in ANCHOR for f in func])
        if not anc.any():
            anc[:] = True
        role = np.array([func[i] if func[i] in ANCHOR else "" for i in range(n)], object)
        n_anc = int(anc.sum())
        best = {}
        for L in range(1, min(MAX_LANES, n_anc) + 1):
            r = self._best_L(L, W, base, anc, role, func)
            if r is not None:
                best[L] = r
        Ls = sorted(best)
        sc = np.array([best[L][0] - self.lam * L for L in Ls])
        pl = np.exp(sc - sc.max())
        pl /= pl.sum()
        Lb = Ls[int(np.argmax(sc))]
        score, sets = best[Lb]
        conf = self._det_conf(sets, Lb, W, anc, role, func, score) * float(pl.max())
        # lane numbering: by volume, busiest first
        lv = np.zeros(Lb)
        for l in range(Lb):
            m = [i for i in range(n) if sets[i] == (1 << l) and anc[i]]
            lv[l] = max(vol[m]) if m else 0.0
        order = np.argsort(-lv, kind="stable")
        remap = {int(old): new for new, old in enumerate(order)}
        lanes = [tuple(sorted(remap[l] + 1 for l in range(Lb) if s >> l & 1)) for s in sets]
        return {"lanes": lanes, "n_lanes": Lb, "n_lanes_conf": float(pl.max()),
                "lane_conf": conf, "lane_volume": lv[order], "scores": dict(zip(Ls, sc))}

    # -- scoring of a full assignment (bitmasks)
    def _score(self, sets, L, W, base, anc, role, func) -> float:
        n = len(sets)
        s = np.array(sets)
        same = (s[:, None] & s[None, :]) != 0
        tot = base + (W * same)[np.triu_indices(n, 1)].sum() if n > 1 else base
        for i in range(n):
            if L > 1:
                tot += self._sp(func[i], bin(sets[i]).count("1"))
        return tot - self._viol(sets, L, anc, role) * self.role_pen

    @staticmethod
    def _viol(sets, L, anc, role) -> int:
        v = 0
        for l in range(L):
            b = 1 << l
            single = [i for i in range(len(sets)) if sets[i] == b and anc[i]]
            if not single:
                return 10 ** 6                # a lane with no single-lane anchor
            rs = [role[i] for i in single]
            v += len(rs) - len(set(rs))
        return v

    def _best_L(self, L, W, base, anc, role, func):
        n = len(func)
        ai = np.flatnonzero(anc)
        na = len(ai)
        full = (1 << L) - 1
        starts = []
        if L == 1:
            starts.append([1] * n)
        else:
            if na <= self.max_ex:
                R = partitions(na, L)                      # m x na
                Wa = W[np.ix_(ai, ai)]
                iu = np.triu_indices(na, 1)
                same = (R[:, iu[0]] == R[:, iu[1]])
                sc = same @ Wa[iu]
                rl = role[ai]
                pen = np.zeros(len(R))
                for rr in set(rl):
                    m = rl == rr
                    if m.sum() < 2:
                        continue
                    cnt = np.stack([(R[:, m] == b).sum(1) for b in range(L)], 1)
                    pen += np.clip(cnt - 1, 0, None).sum(1)
                sc = sc - pen * self.role_pen
                top = np.argsort(-sc, kind="stable")[:4]
                cand = [R[k] for k in top]
            else:
                cand = [self._linkage(W[np.ix_(ai, ai)], L)]
            for lab in cand:
                sets = [0] * n
                for j, i in enumerate(ai):
                    sets[i] = 1 << int(lab[j])
                for i in range(n):                        # non-anchors: best single lane
                    if not anc[i]:
                        gains = [sum(W[i, k] for k in range(n) if k != i and sets[k] >> l & 1)
                                 for l in range(L)]
                        sets[i] = 1 << int(np.argmax(gains)) if n > 1 else 1
                starts.append(sets)
        best = None
        for sets in starts:
            sets = list(sets)
            cur = self._score(sets, L, W, base, anc, role, func)
            for _ in range(8):                             # coordinate ascent
                moved = False
                for i in range(n):
                    old = sets[i]
                    bv, bs = cur, old
                    for s in range(1, full + 1):
                        if s == old:
                            continue
                        sets[i] = s
                        v = self._score(sets, L, W, base, anc, role, func)
                        if v > bv + 1e-9:
                            bv, bs = v, s
                    sets[i] = bs
                    if bs != old:
                        cur, moved = bv, True
                if not moved:
                    break
            if cur > -1e5 and (best is None or cur > best[0]):
                best = (cur, list(sets))
        return best

    @staticmethod
    def _linkage(W, L):
        n = len(W)
        groups = {i: [i] for i in range(n)}
        while len(groups) > L:
            keys = list(groups)
            bv, bi, bj = -np.inf, None, None
            for a, b in itertools.combinations(keys, 2):
                v = float(np.mean(W[np.ix_(groups[a], groups[b])]))
                if v > bv:
                    bv, bi, bj = v, a, b
            groups[bi] += groups.pop(bj)
        lab = np.empty(n, int)
        for k, idx in enumerate(groups.values()):
            lab[idx] = k
        return lab

    def _det_conf(self, sets, L, W, anc, role, func, score) -> np.ndarray:
        n = len(sets)
        out = np.ones(n)
        if L == 1:
            return out
        base = 0.0  # constant cancels
        full = (1 << L) - 1
        for i in range(n):
            old = sets[i]
            alt = -np.inf
            for s in range(1, full + 1):
                if s == old:
                    continue
                sets[i] = s
                alt = max(alt, self._score(sets, L, W, base, anc, role, func))
            sets[i] = old
            ref = self._score(sets, L, W, base, anc, role, func)
            out[i] = 1.0 / (1.0 + np.exp(-(ref - alt))) if np.isfinite(alt) else 1.0
        return out


# ------------------------------------------------------------------------------ model
class PairModel:
    """P(same lane) from the exported LightGBM text models (seed bag), numpy only."""

    def __init__(self, model_dir):
        import sys
        md = Path(model_dir)
        meta = json.loads((md / "lane_model.json").read_text())
        here = Path(__file__).resolve()
        mod = here.parents[3] / "model"
        if str(mod) not in sys.path:
            sys.path.insert(0, str(mod))
        from lgbm_numpy import NumpyBoosterBag
        self.bag = NumpyBoosterBag([md / f for f in meta["pair_models"]])
        self.features = meta["features"]
        self.decoder = Decoder({f: {int(k): v for k, v in d.items()}
                                for f, d in meta["span_prior"].items()},
                               lam=meta["lam"], beta=meta["beta"], role_pen=meta["role_pen"])
        self.min_on = meta.get("min_on", MIN_ON)

    def predict(self, P: pd.DataFrame) -> np.ndarray:
        return self.bag.predict(P[self.features])


def _func_col(pr):
    for c in ("function_pred", "function_guess"):
        if c in pr.columns:
            return c
    raise ValueError("predictions need function_pred or function_guess")


def lanes(events: pd.DataFrame, predictions: pd.DataFrame, model_dir=None, start=None, end=None):
    """-> (phase table, detector table).  `predictions` = model.predict output (DeviceId,
    Detector, phase_pred / phase_guess, function_pred / function_guess).  `start` / `end`
    (optional) = the sample window; default = the span of the events."""
    pm = model_dir if isinstance(model_dir, PairModel) else PairModel(model_dir or default_dir())
    on = on_times(events)
    pr = predictions.copy()
    pr["DeviceId"] = pr.DeviceId.astype(str)
    phc = "phase_pred" if "phase_pred" in pr.columns else "phase_guess"
    pr["phase_use"] = pr[phc]
    if "phase_guess" in pr.columns:
        pr["phase_use"] = pr.phase_use.fillna(pr["phase_guess"])
    fn = pr[_func_col(pr)]
    if "function_guess" in pr.columns:
        fn = fn.fillna(pr["function_guess"])
    pr["func_use"] = fn.astype(object)
    ts = pd.to_datetime(events.Timestamp)
    t0 = _unit(start) if start is not None else ts.min().value / 1e9
    t1 = _unit(end) if end is not None else ts.max().value / 1e9 + 1.0
    hours = (t1 - t0) / 3600.0
    ph_rows, det_rows = [], []
    for dev, g in pr.groupby("DeviceId", sort=True):
        n_on = {}
        for d in g.Detector.astype(int):
            t = on.get((dev, d), (np.zeros(0), np.zeros(0)))[0]
            n_on[d] = int(((t >= t0) & (t < t1)).sum())
        grp, func = eligible(g, n_on, pm.min_on)
        probs = {}
        pairs = [(a, b) for a, b in itertools.combinations(sorted(grp), 2) if grp[a] == grp[b]]
        if pairs:
            loc = {d: on[(dev, d)] for d, k in n_on.items() if k >= pm.min_on and (dev, d) in on}
            P = signal_pair_cues(loc, sorted(loc), pairs, t0, t1)
            P = add_context(P, func, group_sizes(grp), hours)
            for a, b, p in zip(P.da, P.db, pm.predict(P)):
                probs[(int(a), int(b))] = float(p)
        ph_t, det_t = decode_groups(dev, g, n_on, hours, probs, pm.decoder, pm.min_on)
        ph_rows.append(ph_t)
        det_rows.append(det_t)
    ph = pd.concat(ph_rows, ignore_index=True) if ph_rows else pd.DataFrame()
    de = pd.concat(det_rows, ignore_index=True) if det_rows else pd.DataFrame()
    return ph, de


def eligible(g: pd.DataFrame, n_on: dict, min_on: int):
    """{det: phase} of the detectors that get a lane (phase known, >= min_on actuations, not
    Bike) and {det: function} of all."""
    det = g.Detector.astype(int).to_numpy()
    func = dict(zip(det, g.func_use))
    grp = {}
    for d, p, f in zip(det, g.phase_use, g.func_use):
        if pd.notna(p) and n_on.get(d, 0) >= min_on and f not in NOLANE:
            grp[int(d)] = p
    return grp, func


def group_sizes(grp: dict) -> dict:
    s = pd.Series(grp, dtype=object)
    return s.groupby(s).transform("size").to_dict() if len(s) else {}


def decode_groups(dev, g, n_on: dict, hours: float, probs: dict, decoder: "Decoder",
                  min_on: int = MIN_ON, keys: dict | None = None):
    """Decode every predicted phase of one signal.  g: its predictions (Detector, phase_use,
    func_use); probs {(da, db): P(same lane)} for pairs on the same phase (either orientation).  keys {det: content_key}
    (note 77): the phase's detectors are decoded in this behavioural order (busiest first), never in channel order;
    keys=None keeps the old channel order (reproduces notes 42-76)."""
    grp, func = eligible(g, n_on, min_on)
    ph_rows, det_rows = [], []
    s = pd.Series(grp, dtype=object)
    for phase, members in s.groupby(s):
        ds = sorted(members.index, key=lambda d: keys[d], reverse=True) if keys else sorted(members.index)
        n = len(ds)
        M = np.full((n, n), np.nan)
        for i, j in itertools.combinations(range(n), 2):
            M[i, j] = M[j, i] = probs.get((ds[i], ds[j]), probs.get((ds[j], ds[i]), np.nan))
        fl = [func[d] for d in ds]
        vol = np.array([n_on[d] / hours for d in ds])
        r = decoder.decode(M, fl, vol)
        ph_rows.append({"DeviceId": dev, "phase": int(phase), "n_detectors": n,
                        "n_lanes": r["n_lanes"], "n_lanes_conf": round(r["n_lanes_conf"], 3),
                        "lane_volumes_per_hour": ",".join(f"{v:.0f}" for v in r["lane_volume"])})
        for d, ln, c in zip(ds, r["lanes"], r["lane_conf"]):
            det_rows.append({"DeviceId": dev, "Detector": d, "phase": int(phase),
                             "function": func[d], "lanes": ",".join(map(str, ln)),
                             "n_lanes_spanned": len(ln), "lane_conf": round(float(c), 3),
                             "lane_note": ""})
    done = {r["Detector"] for r in det_rows}
    for d, p, f in zip(g.Detector.astype(int), g.phase_use, g.func_use):
        if d in done:
            continue
        k = n_on.get(d, 0)
        note = ("no phase" if pd.isna(p) else "bike detector" if f in NOLANE
                else f"too few actuations ({k} < {min_on})")
        det_rows.append({"DeviceId": dev, "Detector": int(d),
                         "phase": None if pd.isna(p) else int(p), "function": f, "lanes": "",
                         "n_lanes_spanned": 0, "lane_conf": np.nan, "lane_note": note})
    return (pd.DataFrame(ph_rows),
            pd.DataFrame(det_rows).sort_values("Detector").reset_index(drop=True))


def default_dir() -> Path:
    import os
    return Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work")) / "lanes" / "model"
