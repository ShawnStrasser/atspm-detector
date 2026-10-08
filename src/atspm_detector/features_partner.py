"""Extra phase-anonymous pair features: partner geometry and fine ON timing.

Produced as separate columns keyed by (DeviceId, Detector, cand_phase, win) and merged
onto the base table from `features.py`.  The fine-timing and conditioned-call SQL read the shared detector-ON x
candidate tables `j` / `lo2` of `features.shared_parts` (built once per call).

Families
--------
1. **Partner geometry** -- for every candidate p, the candidate q whose green overlaps
   p's most (found generically, no phase numbers): co-green share, exclusive-green
   seconds each way, green-start / green-end lead of p over q.  Plus `__pdiff` columns
   = feature(p) - feature(q) for the features that separate concurrent pairs, which is
   the only way to tell 2 from 6.
2. **Exclusive-green detail with evidence** -- ON rate and occupancy lift restricted to
   (p green & q not green), shrunk toward 1 by the number of ONs observed there.
3. **Fine ON timing** -- sub-second bins of time-since-begin-green, per-cycle first-ON
   latency (median / share <= 1 s / <= 3 s), share of the phase's cycles with any
   actuation, queue-release lag quantiles.
4. **Conditioned call linkage** -- event 43 within fine bins of an ON, restricted to ONs
   where the candidate is RED (a green phase cannot register a vehicle call) and to ONs
   where some other phase is green; chance corrected.
5. **Shrinkage** -- Bayesian-smoothed versions of the key lifts for low-volume detectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import tied_best

# features whose partner-difference is informative for concurrent pairs
PDIFF_FEATS = [
    "on_lift_green", "occ_lift_green", "f_on_green", "f_occ_green",
    "dtg_h0", "dtg_h1", "dtg_h2", "first_on_med", "first_on_le1", "first_on_le3",
    "cyc_hit_frac", "release_lag_le2", "release_lag_med", "queue_end_frac",
    "call43_red_lift", "call43_b0", "burst_g2",
]


# ------------------------------------------------------------------ SQL parts (read features.shared_parts' j / lo2)
# fine ON timing + release
SQL_FINE = """
WITH percyc AS (
  SELECT dev, det, p, cyc, min(dtg) AS first_dtg
  FROM j WHERE st = 0 AND cyc IS NOT NULL GROUP BY 1,2,3,4
), ncyc AS (
  SELECT dev, p, count(*) AS n_cyc FROM cyc_win GROUP BY 1,2
), fc AS (
  SELECT pc.dev, pc.det, pc.p, count(*) AS n_hit,
         least(median(pc.first_dtg), 600.0) AS first_on_med,
         avg(CASE WHEN pc.first_dtg <= 1 THEN 1 ELSE 0 END) AS first_on_le1,
         avg(CASE WHEN pc.first_dtg <= 3 THEN 1 ELSE 0 END) AS first_on_le3
  FROM percyc pc GROUP BY 1,2,3
), rel AS (
  -- lo2 = long ONs with the candidate's last begin-green at or before the END of the occupancy: if it started
  -- DURING the occupancy, that green released this queue (here: ONs over 3 s)
  SELECT dev, det, p,
         median(CASE WHEN gs > t_on THEN t_off - gs END)                        AS release_lag_med,
         avg(CASE WHEN gs > t_on AND t_off - gs <= 2 THEN 1 ELSE 0 END)         AS release_lag_le2,
         avg(CASE WHEN gs > t_on THEN 1 ELSE 0 END)                             AS queue_end_frac,
         count(*)                                                              AS n_long
  FROM lo2 WHERE dur > 3 GROUP BY 1,2,3
), fine AS (
  SELECT dev, det, p,
    avg(CASE WHEN dtg < 0.5 THEN 1 ELSE 0 END) FILTER (st=0)                   AS dtg_h0,
    avg(CASE WHEN dtg >= 0.5 AND dtg < 1.5 THEN 1 ELSE 0 END) FILTER (st=0)    AS dtg_h1,
    avg(CASE WHEN dtg >= 1.5 AND dtg < 3.0 THEN 1 ELSE 0 END) FILTER (st=0)    AS dtg_h2,
    avg(CASE WHEN dtg >= 3.0 AND dtg < 6.0 THEN 1 ELSE 0 END) FILTER (st=0)    AS dtg_h3,
    count(*) FILTER (st=0 AND dtg < 2)                                         AS n_on_g2,
    avg(CASE WHEN tog <= 3 THEN 1 ELSE 0 END) FILTER (st=2)                    AS tog_h0,
    avg(dur) FILTER (st=0 AND dtg < 2)                                         AS dur_g2
  FROM j GROUP BY 1,2,3
)
SELECT f.dev, f.det, f.p, f.dtg_h0, f.dtg_h1, f.dtg_h2, f.dtg_h3, f.tog_h0, f.dur_g2,
       f.n_on_g2 / (2.0 * greatest(n.n_cyc,1)) AS burst_g2,
       fc.first_on_med, fc.first_on_le1, fc.first_on_le3,
       fc.n_hit / nullif(n.n_cyc,0)::DOUBLE AS cyc_hit_frac,
       least(r.release_lag_med, 120.0) AS release_lag_med, r.release_lag_le2, r.queue_end_frac,
       r.n_long
FROM fine f
LEFT JOIN fc  ON fc.dev=f.dev AND fc.det=f.det AND fc.p=f.p
LEFT JOIN rel r ON r.dev=f.dev AND r.det=f.det AND r.p=f.p
LEFT JOIN ncyc n ON n.dev=f.dev AND n.p=f.p
"""

# conditioned / fine call linkage: the per-(detector, candidate) 43 bins come from features.SQL_CALLAGG (one pass over
# j); this adds the candidate's red seconds and 43 count for the chance correction (per dev, p)
SQL_CALLS2_CTX = """
WITH red_secs AS (SELECT dev, p, sum(ng - rs) AS sec_red FROM cyc_win WHERE ng IS NOT NULL GROUP BY 1,2),
n43 AS (SELECT dev, p, count(*) AS n43_tot FROM calls WHERE ev = 43 GROUP BY 1,2)
SELECT coalesce(r.dev, k.dev) AS dev, coalesce(r.p, k.p) AS p, r.sec_red, k.n43_tot
FROM red_secs r FULL OUTER JOIN n43 k ON r.dev = k.dev AND r.p = k.p
"""


# --------------------------------------------------- partner / exclusive green
def _partner_features(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (dev, p): best partner q by co-green seconds + exclusive-green behaviour."""
    om = con.sql("SELECT dev, det, mask, n_on, occ FROM onmask").df()
    mt = con.sql("SELECT dev, mask, secs FROM masktime").df()
    cand = con.sql("SELECT * FROM cand").df()
    cyc = con.sql("SELECT dev, p, cyc, gs, ge FROM cyc_win").df()
    rows, prows = [], []
    for dev, cd in cand.groupby("dev"):
        phases = sorted(int(x) for x in cd.p.unique())
        mtd = mt[mt.dev == dev]
        omd = om[om.dev == dev]
        if not len(mtd):
            continue
        masks = mtd["mask"].to_numpy(dtype=np.int64)
        secs = mtd["secs"].to_numpy(dtype=float)
        T = secs.sum()
        if T <= 0:
            continue
        bits = {p: ((masks >> (p - 1)) & 1).astype(bool) for p in phases}
        tp = {p: secs[bits[p]].sum() for p in phases}
        # partner = candidate with the largest co-green time, then the longest-green one.  A tie left after both
        # keys (e.g. two phases with identical greens) is never broken by phase order: every partner quantity below
        # is the mean over the tied partners (note 76).
        partners, cog, exc_p, exc_q, exc_pq = {}, {}, {}, {}, {}
        for p in phases:
            qs = [q for q in phases if q != p]
            if not qs:
                partners[p], cog[p], exc_p[p], exc_q[p], exc_pq[p] = [], -1.0, 0.0, 0.0, []
                continue
            t = np.array([secs[bits[p] & bits[q]].sum() for q in qs])
            m1 = tied_best(t)
            g = np.array([tp[q] if m else -np.inf for q, m in zip(qs, m1)])
            ts = [q for q, m in zip(qs, tied_best(g)) if m]
            partners[p] = ts
            cog[p] = float(t.max())
            exc_pq[p] = [(secs[bits[p] & ~bits[q]].sum(), secs[~bits[p] & bits[q]].sum()) for q in ts]
            exc_p[p] = float(np.mean([a for a, _ in exc_pq[p]]))
            exc_q[p] = float(np.mean([b for _, b in exc_pq[p]]))
        # green-start / end lead of p over its partner (cycle matched, nearest start)
        cd_cyc = cyc[cyc.dev == dev]
        starts = {p: np.sort(cd_cyc.loc[cd_cyc.p == p, "gs"].to_numpy()) for p in phases}
        ends = {p: np.sort(cd_cyc.loc[cd_cyc.p == p, "ge"].to_numpy()) for p in phases}

        def _lead(sp, sq):
            i = np.clip(np.searchsorted(sq, sp), 0, len(sq) - 1)
            d1 = sp - sq[i]
            i2 = np.clip(i - 1, 0, len(sq) - 1)
            d2 = sp - sq[i2]
            d = np.where(np.abs(d1) < np.abs(d2), d1, d2)
            return float(np.median(d[np.abs(d) < 120])) if np.any(np.abs(d) < 120) else np.nan

        lead_s, lead_e = {}, {}
        for p in phases:
            ls, le = [], []
            for q in partners[p]:
                if len(starts[p]) < 3 or len(starts.get(q, [])) < 3:
                    ls.append(np.nan)
                    le.append(np.nan)
                    continue
                ls.append(_lead(starts[p], starts[q]))
                le.append(_lead(ends[p], ends[q]) if len(ends[p]) >= 3 and len(ends[q]) >= 3 else np.nan)
            lead_s[p], lead_e[p] = _nanmean(ls), _nanmean(le)
        for p in phases:
            ts = partners[p]
            prows.append(dict(dev=dev, p=p, partner=(ts[0] if ts else None),
                              partner_set=",".join(str(q) for q in ts),
                              cogreen_secs=cog[p], cogreen_frac=cog[p] / tp[p] if tp[p] > 0 else np.nan,
                              excl_secs_p=exc_p[p], excl_secs_q=exc_q[p],
                              excl_secs_min=min(exc_p[p], exc_q[p]),
                              partner_lead_start=lead_s[p], partner_lead_end=lead_e[p]))
        # detector-level exclusive-green behaviour vs the partner (mean over tied partners)
        PEX = ("pex_lift_p", "pex_lift_q", "pex_diff", "pex_occ_diff", "pex_diff_shrunk", "pex_n_on", "pex_share_p")
        for det, g in omd.groupby("det"):
            gm = g["mask"].to_numpy(dtype=np.int64)
            gn = g["n_on"].to_numpy(dtype=float)
            go = g["occ"].to_numpy(dtype=float)
            N = gn.sum()
            if N == 0:
                continue
            rate, orate = N / T, go.sum() / T
            gbits = {p: ((gm >> (p - 1)) & 1).astype(bool) for p in phases}
            for p in phases:
                r = dict(dev=dev, det=det, p=p)
                vals = []
                for q, (ep, eq) in zip(partners[p], exc_pq[p]):
                    if not (ep > 60 and eq > 60):
                        continue
                    mp = gbits[p] & ~gbits[q]
                    mq = ~gbits[p] & gbits[q]
                    np_on, nq_on = gn[mp].sum(), gn[mq].sum()
                    lp = (np_on / ep) / rate
                    lq = (nq_on / eq) / rate
                    op = (go[mp].sum() / ep) / orate if orate > 0 else np.nan
                    oq = (go[mq].sum() / eq) / orate if orate > 0 else np.nan
                    k = 8.0  # shrink toward "no difference" when few ONs were seen
                    n_ev = np_on + nq_on
                    d = np.log1p(lp) - np.log1p(lq)
                    vals.append((lp, lq, d, np.log1p(op) - np.log1p(oq), d * n_ev / (n_ev + k), n_ev,
                                 np_on / n_ev if n_ev > 0 else np.nan))
                if vals:
                    r.update({c: _nanmean([v[i] for v in vals]) for i, c in enumerate(PEX)})
                else:
                    r.update(pex_lift_p=np.nan, pex_lift_q=np.nan, pex_diff=np.nan,
                             pex_occ_diff=np.nan, pex_diff_shrunk=np.nan,
                             pex_n_on=0.0, pex_share_p=np.nan)
                rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(prows)


def _nanmean(v) -> float:
    a = np.asarray(v, dtype=float)
    a = a[~np.isnan(a)] if a.size else a
    return float(a.mean()) if a.size else np.nan


# --------------------------------------------------------------------- build
def build_window(con, win: str, secs: float, parts: dict, devmap: pd.DataFrame) -> pd.DataFrame:
    pex, pmeta = _partner_features(con)
    if not len(pex):
        return pd.DataFrame()
    out = pex.merge(parts["fine"], on=["dev", "det", "p"], how="outer")
    out = out.merge(parts["calls2"], on=["dev", "det", "p"], how="left")
    out = out.merge(pmeta, on=["dev", "p"], how="left")
    # chance-corrected red-restricted call lift
    red_rate = (out["n43_tot"] / out["sec_red"].replace(0, np.nan))
    out["call43_red_lift"] = (out["call43_red_035"] /
                              (0.35 * red_rate.replace(0, np.nan))).clip(upper=200)
    out = out.rename(columns={"partner": "partner_phase"})   # a KEY, never a model input
    out.drop(columns=["n43_tot", "sec_red", "n_on_red_c"], inplace=True, errors="ignore")
    out = out.merge(devmap, on="dev", how="left").drop(columns=["dev"])
    out = out.rename(columns={"det": "Detector", "p": "cand_phase"})
    out["win"] = win
    return out


def add_partner_diffs(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Add `<f>__pdiff` = f(p) - f(partner(p)) for every feature in `feats`.

    `partner_phase` is used only as a JOIN KEY here and is never a model input, so the
    model stays phase-anonymous (the partner is found from green overlap, not numbering).
    """
    feats = list(dict.fromkeys(f for f in feats if f in df.columns))
    if not feats or "partner_phase" not in df.columns:
        return df
    key = ["DeviceId", "win", "Detector"]
    right = df[key + ["cand_phase"] + feats].rename(
        columns={"cand_phase": "partner_phase", **{f: f + "__pv" for f in feats}})
    if "partner_set" in df.columns:
        # tied partners (note 76): f(p) minus the mean of f over every tied partner (NaN values skipped)
        ps = df["partner_set"].fillna("").astype(str)
        multi = ps.str.contains(",").to_numpy()
        if multi.any():
            lft = df.loc[multi, key].copy()
            lft["_row"] = np.flatnonzero(multi)
            lft["partner_phase"] = ps[multi].str.split(",").to_numpy()
            lft = lft.explode("partner_phase")
            lft["partner_phase"] = lft.partner_phase.astype(int).astype(df["cand_phase"].dtype)
            mm = lft.merge(right, on=key + ["partner_phase"], how="left").groupby("_row")[
                [f + "__pv" for f in feats]].mean()
    m = df[key + ["partner_phase"]].merge(right, on=key + ["partner_phase"], how="left")
    pv = {f: m[f + "__pv"].to_numpy().copy() for f in feats}
    if "partner_set" in df.columns and multi.any():
        for f in feats:
            pv[f][mm.index.to_numpy()] = mm[f + "__pv"].to_numpy().astype(pv[f].dtype)
    new = {f + "__pdiff": df[f].to_numpy() - pv[f] for f in feats}
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
