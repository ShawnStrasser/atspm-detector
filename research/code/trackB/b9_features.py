"""Track B step B9: neighbour-trace summary features for the LightGBM pair ranker.

For every (signal, window, detector d, candidate c) summarise how the detectors that
actuate *with* d (its similarity neighbours, phi of 2-s "active" bins, `model/similarity.py`,
top-8 per detector, weight w = phi^2 as in the decoder) behave against the SAME candidate c,
using their own raw first-stage features -- not their ranker probabilities, which is all the
joint decoder sees.  Phase-anonymous: `cand_phase`, `partner_phase` and detector numbers are
join keys only.

Families (all prefixed `nb_`):
  nb_<f>          w-weighted mean over the neighbours of feature f(neighbour, c)
  nb_<f>__dself   own f(d, c) minus the neighbourhood mean
  nb_<f>__pc      neighbourhood mean for c minus the neighbourhood mean for c's partner
                  (the candidate whose green overlaps c's most) -- the concurrent-pair axis
  nb_vote_<f>     w-weighted share of neighbours whose argmax of raw f is c
  nb_pool_*       exclusive-green evidence POOLED over d and its neighbours: actuations
                  seen while c is green and its partner is not, against the partner-only
                  time, as a log-odds against the time share (the per-detector version is
                  `pex_*`; pooling is what a thin 30-minute sample lacks)
  nb_wsum, nb_n, nb_phi_max   neighbourhood size / strength (per detector)

Only the 701 training signals are read (DEV Dec-2024 + NEWTRAIN Sept-2026); the locked
signals are never loaded.

    python research/code/trackB/b9_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
from neural.data2 import training_signals  # noqa: E402

import duckdb  # noqa: E402

FEAT = DC_WORK / "features"
SFEAT = DC_WORK / "official" / "stg" / "features"
OUT = DC_WORK / "trackB" / "b9"

BASE_F = ["on_lift_green", "occ_lift_green", "f_on_green", "solo_lift",
          "excl_lift_mean", "excl_diff_mean", "excl_partner_diff",
          "call43_fwd_lift", "call43_rev_frac", "release_frac", "queue_occ_pre_green",
          "dtg_b0"]
V2_F = ["pex_diff_shrunk", "pex_lift_p", "call43_red_lift", "first_on_le3", "cyc_hit_frac"]
VOTE_F = ["on_lift_green", "excl_diff_mean", "call43_fwd_lift", "excl_partner_diff"]
DSELF_F = ["on_lift_green", "f_on_green", "excl_diff_mean", "pex_diff_shrunk",
           "call43_fwd_lift", "call43_red_lift"]
PC_F = ["on_lift_green", "f_on_green", "excl_lift_mean", "pex_lift_p", "call43_red_lift",
        "first_on_le3", "cyc_hit_frac", "call43_fwd_lift"]
SIDES = {"dec": dict(base=[FEAT / "pair_features_windows.parquet",
                           FEAT / "pair_features_windows_B.parquet"],
                     v2=[FEAT / "pair_features_v2_extra.parquet",
                         FEAT / "pair_features_v2_extra_B.parquet"],
                     sim=[FEAT / "det_similarity.parquet", FEAT / "det_similarity_B.parquet"],
                     suffix=""),
         "stg": dict(base=[SFEAT / "pair_features_stg.parquet"],
                     v2=[SFEAT / "pair_features_v2_stg.parquet"],
                     sim=[SFEAT / "det_similarity_stg.parquet"], suffix="@stg")}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _lst(fs) -> str:
    return "[" + ",".join(f"'{Path(f).as_posix()}'" for f in fs if Path(f).exists()) + "]"


def build_side(con, side: str, ids: list[str]) -> pd.DataFrame:
    S = SIDES[side]
    con.execute("CREATE OR REPLACE TEMP TABLE ids AS SELECT unnest(?) AS DeviceId", [ids])
    bcols = ", ".join(f"b.{c}" for c in BASE_F + [f"{v}__argmax" for v in VOTE_F])
    vcols = ", ".join(f"v.{c}" for c in V2_F)
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE pf AS
      SELECT lower(b.DeviceId) AS dev, b.win, b.Detector::INT AS det, b.cand_phase::INT AS c,
             {bcols}, {vcols}, v.partner_phase::INT AS q,
             v.pex_n_on, v.pex_share_p, v.excl_secs_p, v.excl_secs_q
      FROM read_parquet({_lst(S['base'])}, union_by_name=true) b
      LEFT JOIN read_parquet({_lst(S['v2'])}, union_by_name=true) v
        ON v.DeviceId = b.DeviceId AND v.win = b.win AND v.Detector = b.Detector
           AND v.cand_phase = b.cand_phase
      WHERE lower(b.DeviceId) IN (SELECT DeviceId FROM ids)""")
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE nb AS
      SELECT lower(DeviceId) AS dev, win, Detector::INT AS det, other::INT AS oth,
             phi, phi*phi AS w
      FROM read_parquet({_lst(S['sim'])}, union_by_name=true)
      WHERE phi > 0 AND lower(DeviceId) IN (SELECT DeviceId FROM ids)""")
    n_pf = con.sql("SELECT count(*) FROM pf").fetchone()[0]
    log(f"  {side}: pair rows {n_pf:,}, neighbour edges "
        f"{con.sql('SELECT count(*) FROM nb').fetchone()[0]:,}")
    agg_f = BASE_F + V2_F
    wm = ",\n".join(
        f"sum(n.w * o.{f}) / nullif(sum(CASE WHEN o.{f} IS NOT NULL THEN n.w END), 0) AS nb_{f}"
        for f in agg_f)
    vt = ",\n".join(f"sum(n.w * (o.{f}__argmax > 0.5)::INT) / sum(n.w) AS nb_vote_{f}"
                    for f in VOTE_F)
    # pooled exclusive-green evidence: ONs while c green & partner not (np) vs the reverse (nq)
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE agg AS
      SELECT n.dev, n.win, n.det, o.c,
             {wm},
             {vt},
             sum(n.w * o.pex_n_on * o.pex_share_p) AS nbp_np,
             sum(n.w * o.pex_n_on * (1 - o.pex_share_p)) AS nbp_nq
      FROM nb n JOIN pf o ON o.dev = n.dev AND o.win = n.win AND o.det = n.oth
      GROUP BY 1, 2, 3, 4""")
    per_det = con.sql("""
      SELECT dev, win, det, sum(w) AS nb_wsum, count(*) AS nb_n, max(phi) AS nb_phi_max
      FROM nb GROUP BY 1, 2, 3""").df()
    own_cols = ", ".join(f"p.{f}" for f in set(DSELF_F))
    d = con.sql(f"""
      SELECT p.dev, p.win, p.det, p.c, p.q, {own_cols},
             p.pex_n_on, p.pex_share_p, p.excl_secs_p, p.excl_secs_q, a.* EXCLUDE (dev, win, det, c)
      FROM pf p LEFT JOIN agg a ON a.dev = p.dev AND a.win = p.win AND a.det = p.det
                               AND a.c = p.c""").df()
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    d = d.merge(per_det, on=["dev", "win", "det"], how="left")
    # --- own minus neighbourhood
    for f in DSELF_F:
        d[f"nb_{f}__dself"] = d[f] - d[f"nb_{f}"]
    # --- neighbourhood mean for c minus for its partner q (q is a join key only)
    key = ["dev", "win", "det"]
    right = d[key + ["c"] + [f"nb_{f}" for f in PC_F]].rename(
        columns={"c": "q", **{f"nb_{f}": f"_q_{f}" for f in PC_F}})
    m = d[key + ["q"]].merge(right, on=key + ["q"], how="left")
    for f in PC_F:
        d[f"nb_{f}__pc"] = d[f"nb_{f}"].to_numpy() - m[f"_q_{f}"].to_numpy()
    # --- pooled exclusive-green log-odds, with and without the detector itself
    own_np = (d.pex_n_on * d.pex_share_p).fillna(0).to_numpy()
    own_nq = (d.pex_n_on * (1 - d.pex_share_p)).fillna(0).to_numpy()
    tlog = np.log((d.excl_secs_p + 1) / (d.excl_secs_q + 1)).to_numpy()
    nbp, nbq = d.nbp_np.fillna(0).to_numpy(), d.nbp_nq.fillna(0).to_numpy()
    a = 0.5
    d["nb_pool_logodds"] = np.log((own_np + nbp + a) / (own_nq + nbq + a)) - tlog
    d["nb_pool_logodds_o"] = np.log((nbp + a) / (nbq + a)) - tlog
    d["nb_pool_n"] = own_np + own_nq + nbp + nbq
    d["nb_pool_n_o"] = nbp + nbq
    d.loc[d.q.isna(), ["nb_pool_logodds", "nb_pool_logodds_o"]] = np.nan
    keep = [c for c in d.columns if c.startswith("nb_")]
    out = d[["dev", "win", "det", "c"] + keep].rename(
        columns={"dev": "DeviceId", "det": "Detector", "c": "cand_phase"})
    out["DeviceId"] = out.DeviceId + S["suffix"]
    for c in keep:
        out[c] = out[c].astype(np.float32)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = training_signals()
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=6;")
    con.execute(f"SET temp_directory='{(DC_WORK / 'tmp' / 'duckdb').as_posix()}'")
    parts = []
    for side in ("dec", "stg"):
        ids = sorted(s[s.period == side].DeviceId.str.lower().unique())
        t0 = time.time()
        parts.append(build_side(con, side, ids))
        log(f"{side}: {len(parts[-1]):,} rows, {parts[-1].DeviceId.nunique()} signals, "
            f"{time.time() - t0:.0f}s")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT / "nb_features.parquet", index=False)
    cols = [c for c in out.columns if c.startswith("nb_")]
    log(f"wrote {OUT / 'nb_features.parquet'}: {out.shape}, {len(cols)} features")
    log("non-null share: " + ", ".join(f"{c} {out[c].notna().mean():.2f}" for c in cols[:8]))


if __name__ == "__main__":
    main()
