"""Number-free cross-detector structure, the input to the joint per-signal phase decoder.

1. Similarity (phi) graph: how similarly two detector channels actuate in time.  Detectors on the same approach see the
   same vehicles / platoon within the same second; the opposing approach of a concurrent pair (the 2 <-> 6 problem) is
   independent even though both are green together.  phi over `BIN_SECS`-second bins of "detector had an actuation":

       phi(a,b) = (n_ab*N - n_a*n_b) / sqrt(n_a (N-n_a) n_b (N-n_b))

   Output: DeviceId, win, Detector, other, phi, n_common (top-K neighbours per detector, ties kept).

2. Lead-neighbour graph (note 113): detector b is a LEAD neighbour of a when one of them is followed by the other within
   1..LEAD_SECS s more often than the signal's green state explains (advance -> stop bar, upstream -> downstream: the
   same approach).  Expectation per stratum = green mask x time since the last green-state change (0-2, 3-7, 8-19,
   20+ s); the mask is only an equality key (renumbering the phases permutes strata and changes nothing):

       r(a,b) = max(O_ab - E_ab, O_ba - E_ba) / sqrt(n_a n_b),   O_ab = #(onset-second of a, onset-second of b in
       +1..+LEAD_SECS),  E_ab = sum_s m_a,s rate_b,s.   Kept: r > 0, top LEAD_TOPK per detector with ties kept, w = r^2.

   Computed in numpy from the per-detector onset seconds (sorted-array searches, integer counts; memory grows with the
   number of onsets, not with the sample length).  The research definition is SQL (pairs113.py); the counts are
   identical and the expectations agree to float rounding.

No phase number and no channel number or channel order is read anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_SECS = 2.0
TOPK = 8

SQL_SIM = f"""
WITH b AS (
  SELECT DISTINCT dev, det, (t_on / {BIN_SECS})::BIGINT AS bin FROM onev
), na AS (
  SELECT dev, det, count(*) AS n FROM b GROUP BY 1,2
), pa AS (
  SELECT x.dev, x.det AS a, y.det AS c, count(*) AS n_ab
  FROM b x JOIN b y ON x.dev = y.dev AND x.bin = y.bin AND x.det < y.det
  GROUP BY 1,2,3
)
SELECT p.dev, p.a, p.c, p.n_ab, ka.n AS na, kc.n AS nc
FROM pa p JOIN na ka ON ka.dev=p.dev AND ka.det=p.a
          JOIN na kc ON kc.dev=p.dev AND kc.det=p.c
WHERE p.n_ab >= 3
"""


def build_window(con, win: str, secs: float, devmap: pd.DataFrame) -> pd.DataFrame:
    d = con.sql(SQL_SIM).df()
    if not len(d):
        return pd.DataFrame()
    N = max(secs / BIN_SECS, 2.0)
    na, nc, nab = d.na.to_numpy(float), d.nc.to_numpy(float), d.n_ab.to_numpy(float)
    denom = np.sqrt(np.clip(na * (N - na) * nc * (N - nc), 1e-9, None))
    d["phi"] = (nab * N - na * nc) / denom
    d = d.merge(devmap, on="dev", how="left")
    # symmetrise
    x = d[["DeviceId", "a", "c", "phi", "n_ab"]].rename(columns={"a": "Detector", "c": "other"})
    y = d[["DeviceId", "c", "a", "phi", "n_ab"]].rename(columns={"c": "Detector", "a": "other"})
    out = pd.concat([x, y], ignore_index=True).rename(columns={"n_ab": "n_common"})
    out = out.sort_values(["DeviceId", "Detector", "phi"], ascending=[True, True, False])
    # top-K by phi; a tie at the cut keeps every tied neighbour (never decided by channel order, note 76)
    r = out.phi.astype(np.float32).groupby([out.DeviceId, out.Detector], sort=False).rank(
        method="min", ascending=False)
    out = out[(r <= TOPK).to_numpy()]
    out["win"] = win
    return out


# ------------------------------------------------------------------ lead neighbours (note 113)
LEAD_SECS = 8
LEAD_TOPK = 8

EMPTY_LEAD = pd.DataFrame({"DeviceId": pd.Series(dtype=object), "win": pd.Series(dtype=object),
                           "Detector": pd.Series(dtype="int64"), "other": pd.Series(dtype="int64"),
                           "w": pd.Series(dtype="float64")})


def lead_from_counts(d: pd.DataFrame) -> pd.DataFrame:
    """(DeviceId, win, Detector, other, x_ab, x_ba, na, nb) -> the kept lead edges (DeviceId, win, Detector, other, w)."""
    r = np.maximum(d.x_ab.to_numpy(np.float64), d.x_ba.fillna(0).to_numpy(np.float64)) / np.sqrt(
        np.maximum(d.na.to_numpy(np.float64) * d.nb.to_numpy(np.float64), 1.0))
    d = d.assign(r=r)
    d = d[d.r > 0]
    if not len(d):
        return EMPTY_LEAD.copy()
    keep = d.groupby(["DeviceId", "win", "Detector"], sort=False).r.rank(method="min", ascending=False) <= LEAD_TOPK
    d = d[keep]
    return pd.DataFrame({"DeviceId": d.DeviceId.to_numpy(), "win": d.win.to_numpy(),
                         "Detector": d.Detector.astype("int64").to_numpy(), "other": d.other.astype("int64").to_numpy(),
                         "w": (d.r ** 2).to_numpy(np.float64)})


def _lag_pairs(B: np.ndarray, lag: int):
    """Index pairs (i, j) with B[j] == B[i] + lag, B sorted ascending."""
    lo = np.searchsorted(B, B + lag, "left")
    cnt = np.searchsorted(B, B + lag, "right") - lo
    tot = int(cnt.sum())
    if not tot:
        e = np.zeros(0, np.int64)
        return e, e
    src = np.repeat(np.arange(len(B)), cnt)
    dst = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt) + np.arange(tot)
    return src, dst


def _lead_counts(det: np.ndarray, b: np.ndarray, seg: pd.DataFrame):
    """One signal: onset seconds (det, b) distinct; seg = green-state segments (t0, t1, mask) with t1 > t0.
    -> (dets, O [D,D] lead counts, E [D,D] expected or None, has_e [D,D] pairs sharing a stratum, n_a [D])."""
    dets, di = np.unique(det, return_inverse=True)
    D = len(dets)
    o = np.argsort(b, kind="stable")
    B, Dd = b[o].astype(np.int64), di[o].astype(np.int64)
    na = np.bincount(Dd, minlength=D).astype(float)
    O = np.zeros(D * D, np.int64)
    for lag in range(1, LEAD_SECS + 1):
        s, t = _lag_pairs(B, lag)
        if len(s):
            O += np.bincount(Dd[s] * D + Dd[t], minlength=D * D)
    O = O.reshape(D, D).astype(float)
    np.fill_diagonal(O, 0.0)
    # strata: every whole second of a segment (b0 = ceil(t0) .. ceil(t1) - 1) -> mask x time since the change
    t0 = seg.t0.to_numpy(np.float64)
    b0 = np.ceil(t0).astype(np.int64)
    b1 = np.ceil(seg.t1.to_numpy(np.float64)).astype(np.int64) - 1
    ok = b1 >= b0
    if not ok.any():
        return dets, O, None, None, na
    t0, b0, b1, mk = t0[ok], b0[ok], b1[ok], seg["mask"].to_numpy(np.int64)[ok]
    L = b1 - b0 + 1
    rep = np.repeat(np.arange(len(L)), L)
    sec = b0[rep] + (np.arange(int(L.sum())) - np.repeat(np.cumsum(L) - L, L))
    el = sec - t0[rep]
    code = mk[rep] * 4 + np.where(el < 3, 0, np.where(el < 8, 1, np.where(el < 20, 2, 3)))
    su, sinv = np.unique(code, return_inverse=True)
    S = len(su)
    Ns = np.bincount(sinv, minlength=S).astype(float)
    srt = np.argsort(sec, kind="stable")
    sec_s, str_s = sec[srt], sinv[srt]

    def stratum(x):
        k = np.searchsorted(sec_s, x)
        kk = np.minimum(k, len(sec_s) - 1)
        hit = (k < len(sec_s)) & (sec_s[kk] == x)
        return np.where(hit, str_s[kk], -1)

    sa = stratum(B)
    m = sa >= 0
    rate = np.bincount(Dd[m] * S + sa[m], minlength=D * S).reshape(D, S) / Ns
    M = np.zeros(D * S)
    for lag in range(1, LEAD_SECS + 1):
        sl = stratum(B + lag)
        k = sl >= 0
        M += np.bincount(Dd[k] * S + sl[k], minlength=D * S)
    M = M.reshape(D, S)
    E = M @ rate.T
    has_e = (M > 0).astype(float) @ (rate > 0).astype(float).T > 0
    np.fill_diagonal(E, 0.0)
    np.fill_diagonal(has_e, False)
    return dets, O, E, has_e, na


def build_lead_window(con, win: str, devmap: pd.DataFrame) -> pd.DataFrame:
    """Lead-neighbour graph of the current window (needs the `onev` and `gs` tables of features.apply_window)."""
    ab = con.sql("SELECT DISTINCT dev, det, floor(t_on)::BIGINT AS b FROM onev").df()
    if not len(ab):
        return EMPTY_LEAD.copy()
    gs = con.sql("SELECT dev, t0, t1, mask FROM gs WHERE t1 > t0").df()
    gseg = {k: v for k, v in gs.groupby("dev", sort=False)}
    empty = gs.iloc[:0]
    parts = []
    for dev, a in ab.groupby("dev", sort=False):
        dets, O, E, has_e, na = _lead_counts(a.det.to_numpy(), a.b.to_numpy(), gseg.get(dev, empty))
        if E is None:
            E = np.zeros_like(O)
            has_e = np.zeros(O.shape, bool)
        pres = (O > 0) | has_e                       # the pairs the research SQL produces (an O or an E row)
        X = O - E
        ii, jj = np.nonzero(pres)
        if not len(ii):
            continue
        parts.append(pd.DataFrame({"dev": dev, "Detector": dets[ii], "other": dets[jj], "x_ab": X[ii, jj],
                                   "x_ba": np.where(pres[jj, ii], X[jj, ii], np.nan), "na": na[ii], "nb": na[jj]}))
    if not parts:
        return EMPTY_LEAD.copy()
    d = pd.concat(parts, ignore_index=True).merge(devmap, on="dev").drop(columns="dev")
    d["win"] = win
    return lead_from_counts(d)
