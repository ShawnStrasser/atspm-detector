"""Phase-anonymous detector/phase pair features -- the phase ranker's main input.

One row per (DeviceId, Detector, cand_phase, win) for every active detector channel and every candidate phase of that
signal: how the detector behaves relative to that phase's green, yellow and red, whether a phase call follows its
actuations, how it behaves when that phase is green and its usual partner is not.

No phase number, detector channel number or label-derived quantity is ever a feature; `DeviceId`, `Detector`,
`cand_phase`, `win` are keys only.  All features are rates / shares / lifts (duration invariant); the amount of evidence
is exposed through `win_secs`, `det_n_on`, `det_on_per_hour`, `n_cycles`, so a 30-minute and a 3-day sample look the
same to the model.

pipeline.py calls `apply_window` -> `shared_parts` (every detector-ON x candidate-cycle join, built ONCE and read by this
module and `features_partner`) -> `build_window` -> `finalise`.  Research builder of the training tables:
`research/code/features/build_features.py` (identical definitions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["DeviceId", "Detector", "cand_phase", "win"]
K3 = ["dev", "det", "p"]

# features that get a within-(signal, detector, window) rank / z-score companion
RANK_FEATS = [
    "on_lift_green", "occ_lift_green", "f_on_green", "f_occ_green",
    "release_frac", "release_frac_long", "straddle_frac",
    "call43_fwd_lift", "call43_rev_frac", "call44_fwd_lift", "call43_rev_onnow",
    "excl_lift_min", "excl_lift_mean", "excl_diff_min", "excl_diff_mean",
    "excl_partner_diff", "queue_occ_pre_green", "burst_rate_g4", "ext_corr_late",
    "f_on_red_last10", "dtg_b0", "dtr_b4", "tog_b0", "late_green_rate",
    "on_lift_green_coord", "on_lift_green_free", "solo_lift", "ext_green_gain",
]


def apply_window(con, w0: float, w1: float) -> None:
    """Restrict the chunk tables to [w0, w1) and rebuild the mask aggregates."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE onev AS SELECT * FROM onev_all "
                f"WHERE t_on >= {w0} AND t_on < {w1}")
    # cycles: keep a lookback so an ON early in the window finds its green
    con.execute(f"CREATE OR REPLACE TEMP TABLE cyc AS SELECT * FROM cyc_all "
                f"WHERE gs >= {w0 - 900} AND gs < {w1}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE cyc_win AS SELECT * FROM cyc_all "
                f"WHERE gs >= {w0} AND gs < {w1}")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE gs AS
        SELECT dev, greatest(t0,{w0}) AS t0, least(t1,{w1}) AS t1, mask
        FROM gs_all WHERE t1 > {w0} AND t0 < {w1}""")
    con.execute(f"CREATE OR REPLACE TEMP TABLE calls AS SELECT * FROM calls_all "
                f"WHERE t >= {w0} AND t < {w1} + 10")
    con.execute("""CREATE OR REPLACE TEMP TABLE onmask AS
        WITH a AS (
          SELECT o.dev, o.det, o.t_on, o.dur, coalesce(g.mask, 0) AS mask
          FROM onev o ASOF LEFT JOIN gs g ON o.dev = g.dev AND o.t_on >= g.t0
        )
        SELECT a.dev, a.det, a.mask, coalesce(c.is_coord, false) AS is_coord,
               count(*) AS n_on, sum(a.dur) AS occ
        FROM a ASOF LEFT JOIN coordiv c ON a.dev = c.dev AND a.t_on >= c.t0
        GROUP BY 1,2,3,4""")
    con.execute("""CREATE OR REPLACE TEMP TABLE masktime AS
        SELECT g.dev, g.mask, coalesce(c.is_coord, false) AS is_coord, sum(g.t1 - g.t0) AS secs
        FROM gs g ASOF LEFT JOIN coordiv c ON g.dev = c.dev AND g.t0 >= c.t0
        GROUP BY 1,2,3""")


# --------------------------------------------------------------------- SQL
# every detector ON x every candidate phase, with its cycle (the candidate's last begin-green at or before the ON) --
# built once, read by the state, cycle, call and fine-timing aggregates of this module and features_partner; only the
# columns they read are kept (peak memory)
SQL_J = """
CREATE OR REPLACE TEMP TABLE j AS
SELECT oc.dev, oc.det, oc.p, oc.t_on, oc.dur, y.cyc, y.green_secs,
       (oc.t_on - y.gs)::FLOAT AS dtg, (oc.t_on - y.rs)::FLOAT AS dtr,
       (y.ng - oc.t_on)::FLOAT AS tog, (y.ge - oc.t_on)::FLOAT AS to_end_green,
       (CASE WHEN oc.t_on < y.ge THEN 0 WHEN oc.t_on < y.rs THEN 1 ELSE 2 END)::UTINYINT AS st
FROM (SELECT o.dev, o.det, o.t_on, o.dur, c.p FROM onev o JOIN cand c USING (dev)) oc
ASOF LEFT JOIN cyc y ON oc.dev = y.dev AND oc.p = y.p AND oc.t_on >= y.gs
"""

# every call aggregate of the pair features in ONE pass over j: the candidate's next 43 / 44 call at or after each ON
# (forward linkage, fine 43 bins, 43 while the candidate is red); the ASOF joins read only (dev, det, p, t_on, st)
SQL_CALLAGG = """
WITH c43 AS (SELECT dev, p, t FROM calls WHERE ev = 43), c44 AS (SELECT dev, p, t FROM calls WHERE ev = 44),
a AS (SELECT j.dev, j.det, j.p, j.t_on, j.st, k.t AS t43 FROM j
      ASOF LEFT JOIN c43 k ON j.dev=k.dev AND j.p=k.p AND j.t_on <= k.t),
b AS (SELECT a.*, k.t AS t44 FROM a ASOF LEFT JOIN c44 k ON a.dev=k.dev AND a.p=k.p AND a.t_on <= k.t)
SELECT dev, det, p,
  avg(CASE WHEN t43 - t_on <= 0.35 THEN 1 ELSE 0 END) AS call43_fwd_035,
  avg(CASE WHEN t43 - t_on <= 1.0  THEN 1 ELSE 0 END) AS call43_fwd_1,
  avg(CASE WHEN t44 - t_on <= 0.35 THEN 1 ELSE 0 END) AS call44_fwd_035,
  avg(CASE WHEN t44 - t_on <= 1.0  THEN 1 ELSE 0 END) AS call44_fwd_1,
  avg(CASE WHEN t43 - t_on <= 0.15 THEN 1 ELSE 0 END)                     AS call43_b0,
  avg(CASE WHEN t43 - t_on > 0.15 AND t43 - t_on <= 0.5 THEN 1 ELSE 0 END) AS call43_b1,
  avg(CASE WHEN t43 - t_on > 0.5 AND t43 - t_on <= 2.0 THEN 1 ELSE 0 END)  AS call43_b2,
  avg(CASE WHEN t43 - t_on <= 0.35 THEN 1 ELSE 0 END) FILTER (st=2)        AS call43_red_035,
  count(*) FILTER (st=2)                                                   AS n_on_red_c
FROM b GROUP BY 1,2,3
"""

# long ONs (> 1.5 s) x candidate, with the candidate's last begin-green at or before the END of the ON (if it started
# during the occupancy, that green released this queue); features_partner reads the > 3 s subset
SQL_LO2 = """
CREATE OR REPLACE TEMP TABLE lo2 AS
SELECT lo.*, y.gs FROM (SELECT o.dev, o.det, o.t_on, o.t_off, o.dur, c.p FROM onev o JOIN cand c USING (dev)
                        WHERE o.dur > 1.5) lo
ASOF LEFT JOIN cyc y ON lo.dev=y.dev AND lo.p=y.p AND lo.t_off >= y.gs"""

SQL_STATE_AGG = """
SELECT dev, det, p,
  count(*)                                         AS n_on_pair,
  avg(CASE WHEN st=0 THEN 1 ELSE 0 END)            AS f_on_green,
  avg(CASE WHEN st=1 THEN 1 ELSE 0 END)            AS f_on_yellow,
  avg(CASE WHEN st=2 THEN 1 ELSE 0 END)            AS f_on_red,
  sum(CASE WHEN st=0 THEN dur ELSE 0 END)/nullif(sum(dur),0) AS f_occ_green,
  sum(CASE WHEN st=1 THEN dur ELSE 0 END)/nullif(sum(dur),0) AS f_occ_yellow,
  sum(CASE WHEN st=2 THEN dur ELSE 0 END)/nullif(sum(dur),0) AS f_occ_red,
  avg(dur) FILTER (st=0)                           AS dur_mean_green,
  avg(dur) FILTER (st=2)                           AS dur_mean_red,
  avg(CASE WHEN dtg <  2 THEN 1 ELSE 0 END) FILTER (st=0) AS dtg_b0,
  avg(CASE WHEN dtg >= 2 AND dtg <  5 THEN 1 ELSE 0 END) FILTER (st=0) AS dtg_b1,
  avg(CASE WHEN dtg >= 5 AND dtg < 10 THEN 1 ELSE 0 END) FILTER (st=0) AS dtg_b2,
  avg(CASE WHEN dtg >=10 AND dtg < 20 THEN 1 ELSE 0 END) FILTER (st=0) AS dtg_b3,
  avg(CASE WHEN dtg >=20 THEN 1 ELSE 0 END)              FILTER (st=0) AS dtg_b4,
  avg(CASE WHEN dtr <  5 THEN 1 ELSE 0 END) FILTER (st=2) AS dtr_b0,
  avg(CASE WHEN dtr >= 5 AND dtr < 15 THEN 1 ELSE 0 END) FILTER (st=2) AS dtr_b1,
  avg(CASE WHEN dtr >=15 AND dtr < 30 THEN 1 ELSE 0 END) FILTER (st=2) AS dtr_b2,
  avg(CASE WHEN dtr >=30 AND dtr < 60 THEN 1 ELSE 0 END) FILTER (st=2) AS dtr_b3,
  avg(CASE WHEN dtr >=60 THEN 1 ELSE 0 END)              FILTER (st=2) AS dtr_b4,
  avg(CASE WHEN tog <  3 THEN 1 ELSE 0 END) FILTER (st=2) AS tog_b0,
  avg(CASE WHEN tog >= 3 AND tog < 10 THEN 1 ELSE 0 END) FILTER (st=2) AS tog_b1,
  avg(CASE WHEN tog >=10 AND tog < 30 THEN 1 ELSE 0 END) FILTER (st=2) AS tog_b2,
  avg(CASE WHEN tog >=30 THEN 1 ELSE 0 END)              FILTER (st=2) AS tog_b3,
  avg(dur) FILTER (st=2 AND tog <= 8)              AS queue_occ_pre_green,
  avg(CASE WHEN st=2 AND tog <= 10 THEN 1 ELSE 0 END) AS f_on_red_last10,
  count(*) FILTER (st=0 AND dtg < 4)               AS n_on_g4,
  count(*) FILTER (st=0 AND dtg < 8)               AS n_on_g8,
  count(*) FILTER (st=0 AND to_end_green <= 3)     AS n_on_late_green,
  avg(dur) FILTER (st=0 AND dtg < 4)               AS dur_mean_g4
FROM j GROUP BY 1,2,3
"""

SQL_CYC = """
WITH ca AS (
  SELECT dev, det, p, cyc, any_value(green_secs) AS g,
         count(*) FILTER (dtg >= 3) AS n_late,
         max(CASE WHEN to_end_green <= 2 THEN 1 ELSE 0 END) AS late2
  FROM j WHERE st = 0 AND green_secs IS NOT NULL GROUP BY 1,2,3,4
), tot AS (
  SELECT dev, p, count(*) AS n_cyc, sum(green_secs) AS sg, sum(green_secs*green_secs) AS sgg
  FROM cyc_win WHERE green_secs IS NOT NULL GROUP BY 1,2
), agg AS (
  SELECT dev, det, p, count(*) AS n_cyc_act,
         sum(n_late) AS sn, sum(n_late*n_late) AS snn, sum(n_late*g) AS sng,
         sum(CASE WHEN late2=1 THEN g ELSE 0 END) AS sg_late2, sum(late2) AS n_late2
  FROM ca GROUP BY 1,2,3
)
SELECT a.dev, a.det, a.p,
       a.n_cyc_act / nullif(t.n_cyc,0)::DOUBLE AS cyc_active_frac,
       (t.n_cyc*a.sng - a.sn*t.sg) /
         nullif(sqrt(greatest(t.n_cyc*a.snn - a.sn*a.sn,0)) *
                sqrt(greatest(t.n_cyc*t.sgg - t.sg*t.sg,0)), 0) AS ext_corr_late,
       (a.sg_late2/nullif(a.n_late2,0)) - (t.sg/nullif(t.n_cyc,0)) AS ext_green_gain,
       a.n_late2 / nullif(t.n_cyc,0)::DOUBLE AS late2_frac
FROM agg a JOIN tot t ON t.dev=a.dev AND t.p=a.p
"""

SQL_RELEASE = """
SELECT dev, det, p,
  avg(CASE WHEN gs IS NOT NULL AND gs > t_on AND t_off - gs <= 6 THEN 1 ELSE 0 END) AS straddle_frac,
  avg(CASE WHEN gs IS NOT NULL AND gs > t_on AND t_off - gs <= 6 THEN 1 ELSE 0 END)
      FILTER (dur > 5)                                                              AS release_frac_long,
  avg(CASE WHEN gs IS NOT NULL AND gs > t_on THEN exp(-greatest(t_off-gs,0)/3.0) ELSE 0 END) AS release_frac,
  count(*) FILTER (dur > 5)                                                         AS n_long_on,
  count(*)                                                                          AS n_on15
FROM lo2 GROUP BY 1,2,3
"""

SQL_CTX = """
WITH c AS (
  SELECT dev, p, count(*) AS n_cycles, avg(green_secs) AS green_mean,
         stddev_samp(green_secs) AS green_sd, median(green_secs) AS green_med,
         sum(green_secs) AS green_total,
         avg(ng - gs) FILTER (ng IS NOT NULL AND ng - gs < 600) AS cycle_mean
  FROM cyc_win WHERE green_secs IS NOT NULL GROUP BY 1,2
), k AS (
  SELECT dev, p, count(*) FILTER (ev=43) AS n43, count(*) FILTER (ev=44) AS n44
  FROM calls GROUP BY 1,2
), tot AS (SELECT dev, sum(green_total) AS tg, max(n_cycles) AS mx FROM c GROUP BY 1)
SELECT c.dev, c.p, c.n_cycles, c.green_mean, c.green_sd, c.green_med, c.cycle_mean,
       c.green_total / nullif(t.tg,0) AS cand_green_share,
       c.n_cycles / nullif(t.mx,0)::DOUBLE AS cand_cycle_ratio,
       coalesce(k.n43,0) / nullif(c.n_cycles,0)::DOUBLE AS call43_per_cycle,
       coalesce(k.n44,0) / nullif(c.n_cycles,0)::DOUBLE AS call44_per_cycle,
       (coalesce(k.n43,0) / nullif(c.n_cycles,0)::DOUBLE < 0.25)::INT AS looks_recall,
       coalesce(k.n43,0) AS n43_win
FROM c JOIN tot t USING (dev) LEFT JOIN k USING (dev, p)
"""

SQL_DETSTAT = """SELECT dev, det, count(*) n, sum(dur) o, quantile_cont(dur,0.5) dq50, quantile_cont(dur,0.9) dq90,
       avg(CASE WHEN dur<0.5 THEN 1 ELSE 0 END) fshort, avg(CASE WHEN dur>5 THEN 1 ELSE 0 END) flong,
       avg(dur) dmean, max(dur) dmax FROM onev GROUP BY 1,2"""


def call_reverse(con) -> pd.DataFrame:
    """Reverse call linkage per (dev, det, p): for every 43 / 44 call of the candidate, the time since the detector's
    last ON (search per detector; the SQL ASOF 'call >= ON' join of the research definition)."""
    on = con.sql("SELECT dev, det, t_on, t_off FROM onev ORDER BY dev, det, t_on").df()
    cl = con.sql("SELECT dev, p, ev, t FROM calls").df()
    out = []
    for dev, o in on.groupby("dev", sort=False):
        c = cl[cl.dev == dev]
        if not len(c):
            continue
        t = c.t.to_numpy()
        e43 = c.ev.to_numpy() == 43
        pu, pinv = np.unique(c.p.to_numpy(np.int64), return_inverse=True)
        P_ = len(pu)
        n43 = np.bincount(pinv, e43, P_)
        n44 = np.bincount(pinv, ~e43, P_)
        d43 = np.where(n43 > 0, n43, np.nan)
        d44 = np.where(n44 > 0, n44, np.nan)
        dets = o.det.to_numpy()
        brk = np.flatnonzero(np.diff(dets)) + 1
        ton_all, toff_all = o.t_on.to_numpy(), o.t_off.to_numpy()
        for s0, s1 in zip(np.r_[0, brk], np.r_[brk, len(dets)]):
            ton, toff = ton_all[s0:s1], toff_all[s0:s1]
            i = np.searchsorted(ton, t, side="right") - 1      # last ON with t_on <= t
            has = i >= 0
            ii = np.maximum(i, 0)
            dt = np.where(has, t - ton[ii], np.nan)
            onnow = has & (toff[ii] >= t)
            k35, k1 = dt <= 0.35, dt <= 1.0
            out.append(pd.DataFrame({"dev": dev, "det": dets[s0], "p": pu,
                                     "call43_rev_035": np.bincount(pinv, k35 & e43, P_) / d43,
                                     "call43_rev_frac": np.bincount(pinv, k1 & e43, P_) / d43,
                                     "call43_rev_onnow": np.bincount(pinv, onnow & e43, P_) / d43,
                                     "call44_rev_frac": np.bincount(pinv, k1 & ~e43, P_) / d44,
                                     "n_call43": n43.astype(np.int64)}))
    if not out:
        return pd.DataFrame({"dev": pd.Series(dtype="int16"), "det": pd.Series(dtype="int16"),
                             "p": pd.Series(dtype="int16"), "call43_rev_035": pd.Series(dtype=float),
                             "call43_rev_frac": pd.Series(dtype=float), "call43_rev_onnow": pd.Series(dtype=float),
                             "call44_rev_frac": pd.Series(dtype=float), "n_call43": pd.Series(dtype="int64")})
    return pd.concat(out, ignore_index=True)


# Peak memory: `j` has one row per detector ON x candidate phase (7 days, 40 channels: ~8 M rows).  Every aggregate
# read from `j` / `lo2` is grouped by (dev, det, p[, cyc]), so the join is built and aggregated per BLOCK of detectors
# of at most J_BLOCK_ROWS rows and the per-block results are concatenated: the same groups, the same rows per group,
# bounded memory.  A sample whose join fits in one block (a day or less on almost every signal) runs exactly as before.
J_BLOCK_ROWS = 1_500_000


def _det_blocks(con) -> list:
    """[(dev, det) list per block] so that each block's ON x candidate rows <= J_BLOCK_ROWS (a detector is never
    split); one block (None) when everything fits."""
    w = con.sql("SELECT o.dev, o.det, count(*) * any_value(c.nc) AS n FROM onev o "
                "JOIN (SELECT dev, count(*) AS nc FROM cand GROUP BY 1) c USING (dev) "
                "GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    if sum(int(r[2]) for r in w) <= J_BLOCK_ROWS:
        return [None]
    blocks, cur, n = [], [], 0
    for dev, det, k in w:
        if cur and n + int(k) > J_BLOCK_ROWS:
            blocks.append(cur)
            cur, n = [], 0
        cur.append((int(dev), int(det)))
        n += int(k)
    if cur:
        blocks.append(cur)
    return blocks


def _j_aggregates(con, f2, src: str = "onev") -> tuple:
    """The aggregates read from `j` / `lo2` (state, cycle, release, fine timing, call pass) for the ON intervals in
    table `src` (`onev`, or one block of it)."""
    con.execute(SQL_J if src == "onev" else SQL_J.replace("FROM onev o JOIN cand", f"FROM {src} o JOIN cand"))
    con.execute(SQL_LO2 if src == "onev" else SQL_LO2.replace("FROM onev o JOIN cand", f"FROM {src} o JOIN cand"))
    try:
        out = (con.sql(SQL_STATE_AGG).df(), con.sql(SQL_CYC).df(), con.sql(SQL_RELEASE).df(),
               con.sql(f2.SQL_FINE).df(), con.sql(SQL_CALLAGG).df())
    finally:
        con.execute("DROP TABLE IF EXISTS j")
        con.execute("DROP TABLE IF EXISTS lo2")
    return out


def shared_parts(con) -> dict:
    """Every SQL aggregate of the pair features (this module + features_partner) from ONE detector-ON x candidate join
    `j` and ONE long-ON release join `lo2` per block of detectors (one block unless the sample is long); both tables
    are dropped afterwards."""
    from . import features_partner as f2
    res = []
    try:
        for blk in _det_blocks(con):
            if blk is None:
                res.append(_j_aggregates(con, f2))
                continue
            con.execute("CREATE OR REPLACE TEMP TABLE onev_keys (dev SMALLINT, det SMALLINT)")
            con.executemany("INSERT INTO onev_keys VALUES (?, ?)", blk)
            con.execute("CREATE OR REPLACE TEMP TABLE onev_j AS SELECT o.* FROM onev o JOIN onev_keys k "
                        "ON o.dev = k.dev AND o.det = k.det")
            res.append(_j_aggregates(con, f2, "onev_j"))
    finally:
        con.execute("DROP TABLE IF EXISTS onev_j")
        con.execute("DROP TABLE IF EXISTS onev_keys")
    if len(res) == 1:
        st, cy, rl, fi, ca = res[0]
    else:
        st, cy, rl, fi, ca = (pd.concat([r[i] for r in res], ignore_index=True) for i in range(5))
    parts = {"state": st, "cyc": cy, "release": rl, "fine": fi}
    parts["call_fwd"] = ca[K3 + ["call43_fwd_035", "call43_fwd_1", "call44_fwd_035", "call44_fwd_1"]]
    parts["calls2"] = ca[K3 + ["call43_b0", "call43_b1", "call43_b2", "call43_red_035", "n_on_red_c"]].merge(
        con.sql(f2.SQL_CALLS2_CTX).df(), on=["dev", "p"], how="left")
    parts["call_rev"] = call_reverse(con)
    parts["ctx"] = con.sql(SQL_CTX).df()
    parts["detstat"] = con.sql(SQL_DETSTAT).df()
    return parts


# ------------------------------------------------------- mask-based features
TIE_REL = 1e-9      # co-green seconds within this relative distance of the best are a tie (sums differ by ulps only)


def tied_best(v) -> np.ndarray:
    """Boolean mask of the entries of `v` tied with its maximum.  Phase-order free: a tie is never broken by which
    phase comes first, the caller averages over every tied entry instead (note 76)."""
    v = np.asarray(v, dtype=float)
    mx = v.max()
    return v >= mx - TIE_REL * max(abs(mx), 1.0)


def partner_mean(diffs, cot) -> float:
    """`excl_partner_diff`: the exclusive-green difference against the phase with the most co-green time; when several
    phases tie on co-green time (typically 0 s: no concurrent phase), the mean over all of them (note 76)."""
    return float(np.mean(np.asarray(diffs, dtype=float)[tied_best(cot)]))


def _mask_features(con) -> pd.DataFrame:
    om = con.sql("SELECT * FROM onmask").df()
    mt = con.sql("SELECT * FROM masktime").df()
    cand = con.sql("SELECT * FROM cand").df()
    rows = []
    for dev, cd in cand.groupby("dev"):
        phases = sorted(int(x) for x in cd.p.unique())
        mtd = mt[mt.dev == dev]
        omd = om[om.dev == dev]
        if not len(mtd) or not len(omd):
            continue
        masks = mtd["mask"].to_numpy(dtype=np.int64)
        secs = mtd["secs"].to_numpy(dtype=float)
        is_c = mtd["is_coord"].to_numpy(dtype=bool)
        T = secs.sum()
        if T <= 0:
            continue
        Tc, Tf = secs[is_c].sum(), secs[~is_c].sum()
        bits = {p: ((masks >> (p - 1)) & 1).astype(bool) for p in phases}
        t_anyg = secs[masks != 0].sum()
        for det, g in omd.groupby("det"):
            gm = g["mask"].to_numpy(dtype=np.int64)
            gc = g["is_coord"].to_numpy(dtype=bool)
            gn = g["n_on"].to_numpy(dtype=float)
            go = g["occ"].to_numpy(dtype=float)
            N, O = gn.sum(), go.sum()
            if N == 0:
                continue
            gbits = {p: ((gm >> (p - 1)) & 1).astype(bool) for p in phases}
            rate, orate = N / T, O / T
            nc, nf = gn[gc].sum(), gn[~gc].sum()
            anyg_lift = (gn[gm != 0].sum() / t_anyg) / rate if t_anyg > 5 else np.nan
            base = {}
            for p in phases:
                bp, gp = bits[p], gbits[p]
                tp = secs[bp].sum()
                solo_m = bp & (masks == (1 << (p - 1)))
                gsolo_m = gp & (gm == (1 << (p - 1)))
                solo_t = secs[solo_m].sum()
                tpc = secs[bp & is_c].sum()
                tpf = secs[bp & ~is_c].sum()
                base[p] = dict(
                    green_share=tp / T,
                    on_lift_green=(gn[gp].sum() / tp) / rate if tp > 5 else np.nan,
                    occ_lift_green=(go[gp].sum() / tp) / orate if tp > 5 and orate > 0 else np.nan,
                    on_lift_green_coord=(gn[gp & gc].sum() / tpc) / (nc / Tc)
                    if tpc > 60 and Tc > 60 and nc > 0 else np.nan,
                    on_lift_green_free=(gn[gp & ~gc].sum() / tpf) / (nf / Tf)
                    if tpf > 60 and Tf > 60 and nf > 0 else np.nan,
                    solo_lift=(gn[gsolo_m].sum() / solo_t) / rate if solo_t > 30 else np.nan,
                    solo_share=solo_t / tp if tp > 0 else np.nan,
                    anygreen_lift=anyg_lift,
                )
            for p in phases:
                lp, lq, diffs, cot = [], [], [], []
                for q in phases:
                    if q == p:
                        continue
                    m_pq = bits[p] & ~bits[q]
                    m_qp = ~bits[p] & bits[q]
                    t_pq, t_qp = secs[m_pq].sum(), secs[m_qp].sum()
                    if t_pq < 120 or t_qp < 120:
                        continue
                    l_pq = (gn[gbits[p] & ~gbits[q]].sum() / t_pq) / rate
                    l_qp = (gn[~gbits[p] & gbits[q]].sum() / t_qp) / rate
                    lp.append(l_pq)
                    lq.append(l_qp)
                    diffs.append(np.log1p(l_pq) - np.log1p(l_qp))
                    cot.append(secs[bits[p] & bits[q]].sum())
                r = dict(dev=dev, det=det, p=p, **base[p])
                if diffs:
                    r.update(excl_lift_min=float(np.min(lp)), excl_lift_mean=float(np.mean(lp)),
                             excl_lift_max=float(np.max(lp)),
                             excl_lift_other_max=float(np.max(lq)),
                             excl_lift_other_mean=float(np.mean(lq)),
                             excl_diff_min=float(np.min(diffs)),
                             excl_diff_mean=float(np.mean(diffs)),
                             excl_partner_diff=partner_mean(diffs, cot),
                             n_excl_pairs=len(diffs))
                else:
                    r.update(excl_lift_min=np.nan, excl_lift_mean=np.nan, excl_lift_max=np.nan,
                             excl_lift_other_max=np.nan, excl_lift_other_mean=np.nan,
                             excl_diff_min=np.nan, excl_diff_mean=np.nan,
                             excl_partner_diff=np.nan, n_excl_pairs=0)
                rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- assembly
def build_window(con, win: str, secs: float, parts: dict, devmap: pd.DataFrame) -> pd.DataFrame:
    m = _mask_features(con)
    if not len(m):
        return pd.DataFrame()
    a = parts["state"]
    e = parts["call_rev"].astype({"det": a.det.dtype, "p": a.p.dtype})
    out = m.merge(a, on=K3, how="outer")
    for t in (parts["cyc"], parts["release"], parts["call_fwd"], e):
        out = out.merge(t, on=K3, how="left")
    out = out.merge(parts["ctx"], on=["dev", "p"], how="left")
    out = out.merge(parts["detstat"].rename(columns={"n": "det_n_on", "o": "det_occ", "dq50": "det_dur_med",
                                                     "dq90": "det_dur_q90", "fshort": "det_frac_short",
                                                     "flong": "det_frac_long", "dmean": "det_dur_mean",
                                                     "dmax": "det_dur_max"}),
                    on=["dev", "det"], how="left")
    out = out.merge(devmap, on="dev", how="left").drop(columns=["dev"])
    out = out.rename(columns={"det": "Detector", "p": "cand_phase"})
    out["win"] = win
    out["win_secs"] = secs
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    hrs = df["win_secs"] / 3600.0
    gs = df["green_share"].replace(0, np.nan)
    df["det_on_per_hour"] = df["det_n_on"] / hrs
    df["det_occ_frac"] = df["det_occ"] / df["win_secs"]
    df["log_det_n_on"] = np.log1p(df["det_n_on"])
    df["log_win_hours"] = np.log(hrs)
    df["n_on_per_cycle"] = df["det_n_on"] / df["n_cycles"].clip(lower=1)
    df["burst_rate_g4"] = df["n_on_g4"] / (4.0 * df["n_cycles"].clip(lower=1))
    df["burst_rate_g8"] = df["n_on_g8"] / (8.0 * df["n_cycles"].clip(lower=1))
    df["late_green_rate"] = df["n_on_late_green"] / (3.0 * df["n_cycles"].clip(lower=1))
    df["burst_lift_g4"] = df["burst_rate_g4"] / (df["det_n_on"] / df["win_secs"]).replace(0, np.nan)
    df["f_on_green_over_share"] = df["f_on_green"] / gs
    df["f_occ_green_over_share"] = df["f_occ_green"] / gs
    df["dur_ratio_red_green"] = df["dur_mean_red"] / df["dur_mean_green"].replace(0, np.nan)
    df["long_on_share"] = df["n_long_on"] / df["det_n_on"].clip(lower=1)
    df["call43_rev_per_on"] = df["n_call43"] / df["det_n_on"].clip(lower=1)
    # chance-corrected forward-call lift: P(call <=0.35 s after ON) / (call rate * 0.35)
    call_rate = (df["n43_win"] / df["win_secs"]).replace(0, np.nan)
    df["call43_fwd_lift"] = df["call43_fwd_035"] / (0.35 * call_rate)
    df["call44_fwd_lift"] = df["call44_fwd_035"] / (0.35 * call_rate)
    df["n43_per_hour"] = df["n43_win"] / hrs
    df["n_cycles_per_hour"] = df["n_cycles"] / hrs
    df.drop(columns=["n_on_g4", "n_on_g8", "n_on_late_green", "n_long_on",
                     "n_call43", "n43_win", "det_occ", "green_total", "n_on15"],
            inplace=True, errors="ignore")
    return df


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-candidate companions (rank, z-score, gap to the best, is-best) of RANK_FEATS within each detector.
    Computed on the float32-rounded value (the precision the feature is stored and scored at): two candidates whose
    values are equal in exact arithmetic can differ by a few ulps depending on DuckDB's summation order, which changes
    with the phase numbering; rounding first makes them a true tie (note 76).  One grouping, one pass per statistic."""
    codes = df.groupby([df[c] for c in ("DeviceId", "win", "Detector")], sort=False).ngroup().to_numpy()
    df["n_cand"] = np.bincount(codes)[codes].astype("float32")
    fs = [f for f in RANK_FEATS if f in df.columns]
    V = df[fs].astype(np.float32).astype(np.float64)
    g = V.groupby(codes)
    rk = g.rank(pct=True, method="average")
    mu, sd, mx = g.transform("mean"), g.transform("std"), g.transform("max")
    new = {}
    for f in fs:
        v = V[f]
        new[f"{f}__rank"] = rk[f]
        new[f"{f}__z"] = (v - mu[f]) / sd[f].replace(0, np.nan)
        new[f"{f}__mgap"] = v - mx[f]
        new[f"{f}__argmax"] = (v >= mx[f]).astype("float32")
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = add_derived(df)
    df = add_rank_features(df)
    num = [c for c in df.columns if df[c].dtype.kind == "f"]
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    cast = {c: np.float32 for c in df.columns if df[c].dtype == np.float64}
    cast.update({c: np.int8 for c in df.columns if df[c].dtype == bool})
    cast.update({"Detector": np.int16, "cand_phase": np.int16})
    return df.astype(cast)
