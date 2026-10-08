"""Setback distance, first look (note 31) -- estimate an advance detector's distance from the
stop bar from behaviour only.

Truth: cabinet-print `distance_ft` (`%DC_WORK%/cabinet/print_labels.parquet`), training/dev
signals only (locked TEST / NEWTEST filtered out), `unusual_layout` excluded, single numbers
only (zone ranges "160-320" and lists are counted, not scored).  Phase = official timing
(`function_labels_v3.phase_target`, phase type only); in production it would be the predicted
phase.  Events: the staging cache (Sept 2026, 66 h), which matches the prints' vintage.

Per advance/mid detector (phase-anonymous: only its own phase's color state is used):
  (a) travel time: ON cross-correlogram advance -> stop-bar partner (a Presence / Count /
      Yellow_Red zone on the same phase, same lane when both lane indices are known), advance
      ONs in green >= 8 s after green start with dur <= 2 s.  tau = smoothed peak lag.
      d_a1 = tau * 58.7 ft/s (40 mph);  d_a2 = tau * L_eff / median ON (L_eff = 22 ft).
  (b) queue spill-back: per cycle, the first advance ON starting in red that is a stopped car
      (dur >= 5 s or still on at next green).  n_q = ONs on the loop between red start and it
      (the cars queued ahead in its lane); w_q = its OFF time after next green (start wave).
      d_b1 = spacing * (median n_q + 1);  d_b2 = w * median(w_q).  Constants fitted per fold.
  (c) LightGBM on all the features, target log distance, signal-grouped folds_v3.

    python sb1_setback.py [--hours 66] [--start 0]   (window = start .. start+hours from 16:15 Fri)
"""
from __future__ import annotations
import argparse
import os
import re
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(os.environ.get("DC_REPO") or Path(__file__).resolve().parents[3])
OUT = DCW / "trackA" / "setback"
STG = DCW / "official" / "stg" / "cache"
T0 = pd.Timestamp("2026-09-18 16:15:00")
V_FREE = 58.7          # ft/s, 40 mph
L_EFF = 22.0           # ft, vehicle + loop


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set[str]:
    a = pd.read_csv(DCW / "data" / "splits" / "test_config.csv").DeviceId
    b = pd.read_csv(DCW / "official" / "newtest_signals.csv").DeviceId
    return set(a.str.lower()) | set(b.str.lower())


def parse_dist(s):
    if s is None or pd.isna(s):
        return np.nan, "none"
    s = str(s).strip()
    try:
        return float(s), "single"
    except ValueError:
        pass
    if re.fullmatch(r"-?\d+(\.\d+)?\s*-\s*\d+(\.\d+)?", s):
        return np.nan, "range"
    return np.nan, "other"


def truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(DCW / "cabinet" / "print_labels.parquet")
    p["dev"] = p.DeviceId.str.lower()
    p = p[~p.dev.isin(locked())]
    v = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v3.parquet",
                        columns=["DeviceId", "detector", "phase_target", "phase_target_type"])
    v["dev"] = v.DeviceId.str.lower()
    p = p.merge(v[["dev", "detector", "phase_target", "phase_target_type"]],
                on=["dev", "detector"], how="left")
    assert not p.dev.isin(locked()).any()
    p = p[(p.unusual_layout != True) & (p.phase_target_type == "phase")].copy()
    p["phase"] = p.phase_target.str[1:].astype(int)
    pd_ = p.distance_ft.map(parse_dist)
    p["dist"] = [x[0] for x in pd_]
    p["dist_kind"] = [x[1] for x in pd_]
    p["lane_index"] = p.lane_index.astype("float")
    tgt = p[p.function.isin(["Advance", "Mid"]) & (p.dist_kind != "none")].copy()
    # partner candidates: EVERY v3 detector on the same timing phase (its v3 function --
    # print, else config -- is carried so the eval can restrict to stop-bar partners)
    w = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v3.parquet",
                        columns=["DeviceId", "detector", "phase_target", "phase_target_type",
                                 "function", "lane_index", "lanes_spanned", "unusual_layout"])
    w["dev"] = w.DeviceId.str.lower()
    w = w[w.dev.isin(set(tgt.dev)) & (w.phase_target_type == "phase")].copy()
    assert not w.dev.isin(locked()).any()
    w["phase"] = w.phase_target.str[1:].astype(int)
    w["lane_index"] = w.lane_index.astype("float")
    w["lanes_spanned"] = w.lanes_spanned.astype("float")
    sb = w[["dev", "detector", "phase", "function", "lane_index", "lanes_spanned"]].copy()
    return tgt, sb


def connect():
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    (DCW / "tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


SQL_LAG = """
WITH a AS (
  SELECT o.dev, o.det AS da, o.t, floor(o.t / 20)::INT AS bk
  FROM onc o JOIN tg ON o.dev = tg.dev AND o.det = tg.detector
  WHERE o.state = 'G' AND o.since >= 8 AND o.dur <= 2
), b AS (
  SELECT dev, det AS db, t,
         unnest([floor(t / 20)::INT, floor(t / 20)::INT + 1]) AS bk
  FROM onc
)
SELECT a.dev, a.da, b.db, floor((b.t - a.t) / 0.2)::INT AS lb, count(*) AS n
FROM a JOIN prs p ON a.dev = p.dev AND a.da = p.da
JOIN b ON a.dev = b.dev AND p.db = b.db AND a.bk = b.bk
WHERE b.t - a.t > 0 AND b.t - a.t < 20
GROUP BY ALL
"""

SQL_Q = """
WITH r AS (
  SELECT o.*, row_number() OVER (PARTITION BY dev, det, cyc ORDER BY t) AS k
  FROM onc o SEMI JOIN tg ON o.dev = tg.dev AND o.det = tg.detector
  WHERE o.state = 'R'
), s AS (
  SELECT dev, det, cyc, min(k) AS kstop
  FROM r WHERE dur >= 5 OR t + dur >= t_next
  GROUP BY ALL
), st AS (
  SELECT s.dev, s.det, s.cyc, s.kstop - 1 AS n_q, r.t - r.t_red AS t_stop_in_red,
         r.t + r.dur - r.t_next AS w_q, r.t_next - r.t_red AS red_len, r.t AS t_stop,
         r.t_red, r.t_next
  FROM s JOIN r ON s.dev = r.dev AND s.det = r.det AND s.cyc = r.cyc AND s.kstop = r.k
  WHERE r.t_next - r.t_red < 300 AND r.t + r.dur - r.t_next < 60
)
-- n_y: ONs on the loop from 4 s before yellow (cars that could not clear) up to the stop
SELECT st.*, (SELECT count(*) FROM onc o WHERE o.dev = st.dev AND o.det = st.det
              AND o.t >= st.t_red - 8 AND o.t < st.t_stop) AS n_y
FROM st
"""


def features(hours: float, start: float) -> pd.DataFrame:
    tgt, sb = truth()
    log(f"targets: {len(tgt)} ({tgt.function.value_counts().to_dict()}), "
        f"{tgt.dev.nunique()} signals; kinds {tgt.dist_kind.value_counts().to_dict()}")
    # partners: same phase; same lane when both indices known and the stop-bar zone is 1 lane
    pr = tgt[["dev", "detector", "phase", "lane_index"]].merge(
        sb, on=["dev", "phase"], suffixes=("", "_b"))
    pr["same_lane"] = (pr.lane_index == pr.lane_index_b) & (pr.lanes_spanned == 1)
    pr["lane_known"] = pr.lane_index.notna() & pr.lane_index_b.notna()
    pr = pr[pr.detector != pr.detector_b]
    prs = pr.rename(columns={"detector": "da", "detector_b": "db"})[
        ["dev", "da", "db", "function", "same_lane", "lane_known"]]
    t0 = T0 + pd.Timedelta(hours=float(start))
    t1 = t0 + pd.Timedelta(hours=float(hours))
    e0 = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
    con = connect()
    con.register("tg_df", tgt[["dev", "detector", "phase"]])
    con.execute("CREATE TEMP TABLE tg AS SELECT * FROM tg_df")
    con.register("prs_df", prs)
    con.execute("CREATE TEMP TABLE prs AS SELECT * FROM prs_df")
    alld = pd.concat([tgt[["dev", "detector", "phase"]],
                      pr[["dev", "detector_b", "phase"]].rename(
                          columns={"detector_b": "detector"})]).drop_duplicates(
        ["dev", "detector"])
    con.register("ad_df", alld)
    con.execute("CREATE TEMP TABLE ad AS SELECT * FROM ad_df")
    di, pc = (STG / "det_intervals.parquet").as_posix(), (STG / "phase_cycles.parquet").as_posix()
    con.execute(f"""CREATE TEMP TABLE cyc AS
        SELECT lower(DeviceId) AS dev, Phase::INT AS phase, cyc,
               epoch_ms(green_start) / 1000.0 - {e0} AS t_g,
               epoch_ms(yellow_start) / 1000.0 - {e0} AS t_y,
               epoch_ms(red_start) / 1000.0 - {e0} AS t_red,
               epoch_ms(next_green) / 1000.0 - {e0} AS t_next
        FROM read_parquet('{pc}') WHERE next_green IS NOT NULL
          AND lower(DeviceId) IN (SELECT DISTINCT dev FROM ad)""")
    con.execute(f"""CREATE TEMP TABLE onw AS
        SELECT lower(i.DeviceId) AS dev, i.Detector::INT AS det, ad.phase,
               epoch_ms(i.t_on) / 1000.0 - {e0} AS t, i.dur,
               (hour(i.t_on) >= 22 OR hour(i.t_on) < 6) AS quiet
        FROM read_parquet('{di}') i
        JOIN ad ON lower(i.DeviceId) = ad.dev AND i.Detector = ad.detector
        WHERE i.t_on >= TIMESTAMP '{t0}' AND i.t_on < TIMESTAMP '{t1}'""")
    con.execute("""CREATE TEMP TABLE onc AS
        SELECT o.*, c.cyc, c.t_red, c.t_next, t - c.t_g AS since,
               CASE WHEN t < c.t_y THEN 'G' WHEN t >= c.t_red THEN 'R' ELSE 'Y' END AS state
        FROM onw o ASOF JOIN cyc c ON o.dev = c.dev AND o.phase = c.phase AND o.t >= c.t_g
        WHERE o.t < c.t_next""")
    log(f"ONs in window {con.sql('SELECT count(*) FROM onc').fetchone()[0]:,}")
    base = con.sql("""SELECT o.dev, o.det AS detector, count(*) AS n_on_w,
            median(dur) AS dur_med, quantile_cont(dur, 0.25) AS dur_q25,
            median(dur) FILTER (state = 'G' AND since >= 8 AND dur <= 2) AS dur_ff,
            avg((dur <= 0.15)::INT) AS pulse_share,
            count(*) FILTER (state = 'R') AS n_red,
            count(DISTINCT cyc) AS n_cyc
        FROM onc o JOIN tg ON o.dev = tg.dev AND o.det = tg.detector GROUP BY ALL""").df()
    ncyc = con.sql("SELECT dev, phase, count(*) AS cyc_phase, "
                   "avg(t_next - t_red) AS red_mean, avg(t_y - t_g) AS green_mean "
                   "FROM cyc GROUP BY ALL").df()
    lag = con.sql(SQL_LAG).df()
    log(f"lag histogram {lag.shape}")
    q = con.sql(SQL_Q).df()
    log(f"queue events {q.shape}")
    # --- (a) correlogram peak per pair, best partner per advance detector
    bins = np.arange(0, 100)
    H = lag.groupby(["dev", "da", "db", "lb"]).n.sum().unstack("lb").reindex(
        columns=bins, fill_value=0).fillna(0)
    M = H.to_numpy(float)
    Ms = np.apply_along_axis(lambda r: np.convolve(r, np.ones(5) / 5, "same"), 1, M)
    lo = 2                                      # ignore lags < 0.4 s
    k = Ms[:, lo:90].argmax(1) + lo
    pk = Ms[np.arange(len(Ms)), k]
    basel = np.median(Ms[:, lo:95], 1).clip(min=0.2)
    P = pd.DataFrame({"tau": (k + 0.5) * 0.2, "pk_ratio": np.log(pk.clip(min=0.2) / basel),
                      "pk_n": M.sum(1)}, index=H.index).reset_index()
    P = P.merge(prs, on=["dev", "da", "db"])
    P["score"] = P.pk_ratio + 0.5 * P.same_lane
    P["sbp"] = P.function.isin(["Presence", "Count", "Yellow_Red"])
    P.to_parquet(OUT / f"sb1_pairs_h{hours:g}_s{start:g}.parquet", index=False)
    cols = ["tau", "pk_ratio", "pk_n", "same_lane", "lane_known", "partner", "partner_fn"]
    def pick(sub, tag):
        b = (sub.sort_values("score", ascending=False).drop_duplicates(["dev", "da"])
             .rename(columns={"da": "detector", "db": "partner", "function": "partner_fn"}))
        return b[["dev", "detector"] + cols].rename(
            columns={c: f"{c}{tag}" for c in cols})
    best = pick(P[P.sbp], "").merge(pick(P, "_any"), on=["dev", "detector"], how="outer")
    # --- (b) queue spill-back aggregates
    qa = q.groupby(["dev", "det"]).agg(
        q_cyc=("cyc", "size"), nq_med=("n_q", "median"), nq_mean=("n_q", "mean"),
        nq_q75=("n_q", lambda s: s.quantile(.75)), wq_med=("w_q", "median"),
        ny_med=("n_y", "median"), ny_q75=("n_y", lambda s: s.quantile(.75)),
        ny_max=("n_y", lambda s: s.quantile(.9)), wq_q75=("w_q", lambda s: s.quantile(.75)),
        tstop_med=("t_stop_in_red", "median")).reset_index().rename(columns={"det": "detector"})
    F = (tgt.merge(base, on=["dev", "detector"], how="left")
         .merge(ncyc, on=["dev", "phase"], how="left")
         .merge(best, on=["dev", "detector"], how="left")
         .merge(qa, on=["dev", "detector"], how="left"))
    F["q_share"] = F.q_cyc.fillna(0) / F.cyc_phase
    F["on_per_cyc"] = F.n_on_w / F.cyc_phase
    for c in ("same_lane", "lane_known", "same_lane_any", "lane_known_any"):
        F[c] = F[c].astype(float)
    F["hours"], F["start"] = hours, start
    return F


def main(hours: float, start: float):
    OUT.mkdir(parents=True, exist_ok=True)
    F = features(hours, start)
    fn = OUT / f"sb1_feat_h{hours:g}_s{start:g}.parquet"
    F.drop(columns=[c for c in F.columns if F[c].dtype == object and c not in
                    ("dev", "DeviceId", "function", "distance_ft", "dist_kind", "technology",
                     "partner_fn", "partner_fn_any", "subtype", "phase_target")]).to_parquet(fn, index=False)
    log(f"wrote {fn} ({len(F)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=66)
    ap.add_argument("--start", type=float, default=0)
    a = ap.parse_args()
    main(a.hours, a.start)
