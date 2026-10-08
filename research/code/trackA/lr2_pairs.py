"""Lanes redo, task 2a -- detector-pair lane cues on the lane-text signals.

For every pair of detectors on the same timing phase (unlocked signals whose channel
text names RL / CL / LL somewhere), over the full window of each period:

  correlogram   ON-time cross-correlogram of b relative to a, 0.5 s bins, |lag| < 15 s,
                all hours and quiet hours (22-06) separately.  Gives the zero-lag
                co-location cue and the 2-8 s advance->stop-bar lead cue.
  minute corr   correlation of 1-minute counts after removing each detector's 15-min
                mean (high-pass: the common-demand part is gone, what is left is "the
                same vehicles"), overall and in the quiet / busy thirds of 15-min bins.
  15-min corr   binned-count correlation and count ratio off-peak vs peak (A2's
                lane-divergence cue, recomputed here for the same pairs).

Output: lr2_corr.parquet (the raw correlogram), lr2_pairs.parquet (one row per pair).
`--mode all`: every unlocked signal, pairs on the same PREDICTED phase of the full
window -> lr2_pairs_all.parquet (input to the constrained decode, lr4_decode.py).

    python lr2_pairs.py
    python lr2_pairs.py --mode all
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import lr_common as C

WIN = {"dec": (C.DCW / "cache", pd.Timestamp("2024-12-02 00:00:00"), 72 * 3600),
       "stg": (C.DCW / "official" / "stg" / "cache", pd.Timestamp("2026-09-18 16:15:00"),
               66 * 3600)}

SQL_CORR = """
WITH a AS (
  SELECT o.dev, o.det AS da, p.db, o.t, o.quiet, floor(o.t / 20)::INT AS bk
  FROM onw o JOIN prs p ON o.dev = p.dev AND o.det = p.da
), b AS (
  SELECT dev, det AS db, t,
         unnest([floor(t / 20)::INT - 1, floor(t / 20)::INT, floor(t / 20)::INT + 1]) AS bk
  FROM onw
)
SELECT a.dev, a.da, a.db, floor((b.t - a.t) / 0.5)::INT AS lb, a.quiet, count(*) AS n
FROM a JOIN b ON a.dev = b.dev AND a.db = b.db AND a.bk = b.bk
WHERE abs(b.t - a.t) < 15
GROUP BY ALL
"""

SQL_MIN = """
WITH cnt AS (SELECT dev, det, floor(t / 60)::INT AS m, count(*) AS n FROM onw GROUP BY ALL),
grid AS (SELECT d.dev, d.det, m.m FROM (SELECT DISTINCT dev, det FROM onw) d
         CROSS JOIN (SELECT unnest(range(0, {nm})) AS m) m),
g AS (SELECT g.dev, g.det, g.m, g.m // 15 AS blk, coalesce(c.n, 0)::DOUBLE AS n
      FROM grid g LEFT JOIN cnt c ON g.dev = c.dev AND g.det = c.det AND g.m = c.m),
r AS (SELECT *, n - avg(n) OVER (PARTITION BY dev, det, blk) AS res FROM g),
tot AS (SELECT dev, blk, sum(n) AS tn FROM g GROUP BY ALL),
lvl AS (SELECT dev, blk, CASE WHEN rk >= 0.67 THEN 2 WHEN rk <= 0.33 THEN 0 ELSE 1 END AS lv
        FROM (SELECT dev, blk, percent_rank() OVER (PARTITION BY dev ORDER BY tn) AS rk
              FROM tot)),
r2 AS (SELECT r.*, lvl.lv FROM r JOIN lvl ON r.dev = lvl.dev AND r.blk = lvl.blk),
blk AS (SELECT dev, det, blk, any_value(lv) AS lv, sum(n) AS n FROM r2 GROUP BY ALL),
m AS (
  SELECT a.dev, a.det AS da, b.det AS db,
         corr(a.res, b.res) AS hp_all,
         corr(a.res, b.res) FILTER (a.lv = 0) AS hp_off,
         corr(a.res, b.res) FILTER (a.lv = 2) AS hp_peak
  FROM r2 a JOIN r2 b ON a.dev = b.dev AND a.m = b.m
  JOIN prs p ON a.dev = p.dev AND a.det = p.da AND b.det = p.db
  GROUP BY ALL
), q AS (
  SELECT a.dev, a.det AS da, b.det AS db,
         corr(a.n, b.n) AS c_all,
         corr(a.n, b.n) FILTER (a.lv = 0) AS c_off,
         corr(a.n, b.n) FILTER (a.lv = 2) AS c_peak,
         sum(b.n) FILTER (a.lv = 2) / nullif(sum(a.n) FILTER (a.lv = 2), 0) AS r_peak,
         sum(b.n) FILTER (a.lv = 0) / nullif(sum(a.n) FILTER (a.lv = 0), 0) AS r_off
  FROM blk a JOIN blk b ON a.dev = b.dev AND a.blk = b.blk
  JOIN prs p ON a.dev = p.dev AND a.det = p.da AND b.det = p.db
  GROUP BY ALL
)
SELECT * FROM m FULL JOIN q USING (dev, da, db)
"""


def cues(corr: pd.DataFrame, nq: pd.DataFrame) -> pd.DataFrame:
    """Correlogram -> per-pair cues.  Baseline = mean bin count at 10 <= |lag| < 15 s
    (same cycle-scale coupling, no same-vehicle pairing)."""
    lags = np.arange(-30, 30)                      # bin lb covers [lb/2, lb/2 + 0.5)
    mid = lags * 0.5 + 0.25
    out = []
    for quiet_tag, sub in (("all", corr), ("q", corr[corr.quiet])):
        H = (sub.groupby(["dev", "da", "db", "lb"]).n.sum().unstack("lb")
             .reindex(columns=lags, fill_value=0).fillna(0))
        M = H.to_numpy(float)
        base = M[:, np.abs(mid) >= 10].mean(1).clip(min=0.5)
        z = (np.abs(mid) < 0.5)                    # 2 bins: |lag| < 0.5 s
        fwd = (mid >= 1.5) & (mid <= 8.5)
        bwd = (mid <= -1.5) & (mid >= -8.5)
        near = (np.abs(mid) >= 1.0) & (np.abs(mid) < 3.0)
        d = pd.DataFrame(index=H.index)
        d[f"z0_ratio_{quiet_tag}"] = np.log(M[:, z].mean(1).clip(min=0.1) / base)
        d[f"z0_sharp_{quiet_tag}"] = np.log(M[:, z].mean(1).clip(min=0.1)
                                            / M[:, near].mean(1).clip(min=0.5))
        d[f"z0_exc_{quiet_tag}"] = (M[:, z] - base[:, None]).sum(1)
        pf, pb = M[:, fwd].max(1), M[:, bwd].max(1)
        d[f"lead_ratio_{quiet_tag}"] = np.log(np.maximum(pf, pb) / base)
        # sharpness: best 1 s (2-bin) run in the 1.5-8.5 s band vs the band's median
        band = np.where(pf >= pb, 0, 1)
        Mf, Mb = M[:, fwd], M[:, bwd][:, ::-1]
        MM = np.where(band[:, None] == 0, Mf, Mb)
        run = MM[:, :-1] + MM[:, 1:]
        d[f"lead_sharp_{quiet_tag}"] = np.log(run.max(1).clip(min=0.5)
                                              / (2 * np.median(MM, 1)).clip(min=0.5))
        d[f"lead_exc_{quiet_tag}"] = (run.max(1) - 2 * base)
        d[f"lead_lag_{quiet_tag}"] = np.where(band == 0, 1, -1) * (
            1.5 + 0.5 * run.argmax(1) + 0.5)
        d[f"lead_asym_{quiet_tag}"] = np.log((M[:, fwd].sum(1) + 1) / (M[:, bwd].sum(1) + 1))
        out.append(d)
    d = pd.concat(out, axis=1).reset_index()
    d = d.merge(nq.rename(columns={"det": "da", "n": "n_a", "nq": "nq_a"}), on=["dev", "da"])
    d = d.merge(nq.rename(columns={"det": "db", "n": "n_b", "nq": "nq_b"}), on=["dev", "db"])
    for t, na, nb in (("all", "n_a", "n_b"), ("q", "nq_a", "nq_b")):
        mn = np.minimum(d[na], d[nb]).clip(lower=1)
        d[f"z0_exc_{t}"] = d[f"z0_exc_{t}"] / mn          # extra coincidences per ON
        d[f"lead_exc_{t}"] = d[f"lead_exc_{t}"] / mn
    d["bal"] = np.minimum(d.n_a, d.n_b) / np.maximum(d.n_a, d.n_b).clip(lower=1)
    return d


def dets_lanetext() -> dict:
    """Unlocked lane-text signals, pairs on the same TIMING phase (truth), both periods."""
    lab = C.labelled_detectors()
    sig = lab[lab.lane.notna()].DeviceId.unique()
    d = lab[lab.DeviceId.isin(sig) & lab.target.notna()][
        ["DeviceId", "Detector", "target"]].rename(
        columns={"DeviceId": "dev", "Detector": "det", "target": "grp"})
    d["det"] = d.det.astype("int16")
    C.log(f"lanetext: {len(sig)} signals, {len(d)} labelled detectors")
    return {"dec": d, "stg": d}


def dets_all() -> dict:
    """Every unlocked signal in the function OOF table, pairs on the same PREDICTED
    phase of the full window (what the decoder sees)."""
    k = pd.read_parquet(C.WORK / "a2_oof_keys.parquet",
                        columns=["DeviceId", "Detector", "period", "wgroup", "pred_phase"])
    k = k[k.wgroup == "full"]
    k["DeviceId"] = k.DeviceId.str.lower()
    k = k[~k.DeviceId.isin(C.locked_ids())]
    out = {}
    for per in ("dec", "stg"):
        d = k[k.period == per][["DeviceId", "Detector", "pred_phase"]].rename(
            columns={"DeviceId": "dev", "Detector": "det", "pred_phase": "grp"})
        d["det"] = d.det.astype("int16")
        d["grp"] = d.grp.astype(str)
        out[per] = d.drop_duplicates(["dev", "det"])
        C.log(f"all/{per}: {d.dev.nunique()} signals, {len(d)} detectors")
    return out


def main(mode: str):
    detsets = dets_lanetext() if mode == "lanetext" else dets_all()
    con = C.connect()
    allc, allp = [], []
    for per, (cache, t0, secs) in WIN.items():
        tp = time.time()
        dets = detsets[per]
        con.register("dets_df", dets)
        con.execute("CREATE OR REPLACE TEMP TABLE dets AS SELECT * FROM dets_df")
        con.execute("""CREATE OR REPLACE TEMP TABLE prs AS
                       SELECT a.dev, a.det AS da, b.det AS db
                       FROM dets a JOIN dets b ON a.dev = b.dev AND a.grp = b.grp
                       AND a.det < b.det""")
        t1 = t0 + pd.Timedelta(seconds=secs)
        e0 = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
        di = (cache / "det_intervals.parquet").as_posix()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE onw AS
            SELECT lower(i.DeviceId) AS dev, i.Detector::SMALLINT AS det,
                   epoch_ms(i.t_on) / 1000.0 - {e0} AS t,
                   (hour(i.t_on) >= 22 OR hour(i.t_on) < 6) AS quiet
            FROM read_parquet('{di}') i
            SEMI JOIN dets d ON lower(i.DeviceId) = d.dev AND i.Detector = d.det
            WHERE i.t_on >= TIMESTAMP '{t0}' AND i.t_on < TIMESTAMP '{t1}'""")
        nq = con.sql("""SELECT dev, det, count(*) AS n, count(*) FILTER (quiet) AS nq
                        FROM onw GROUP BY ALL HAVING count(*) >= 20""").df()
        con.execute("DELETE FROM onw WHERE NOT EXISTS "
                    "(SELECT 1 FROM nq WHERE nq.dev = onw.dev AND nq.det = onw.det)")
        C.log(f"[{per}] {len(nq)} actuating detectors on {nq.dev.nunique()} signals")
        corr = con.sql(SQL_CORR).df()
        corr["period"] = per
        if mode == "lanetext":
            allc.append(corr)
        C.log(f"[{per}] correlogram {corr.shape} ({time.time()-tp:.0f}s)")
        mm = con.sql(SQL_MIN.format(nm=int(secs // 60))).df()
        C.log(f"[{per}] minute/15-min corr {mm.shape} ({time.time()-tp:.0f}s)")
        cu = cues(corr, nq).merge(mm, on=["dev", "da", "db"], how="left")
        cu["period"] = per
        allp.append(cu)
    tag = "" if mode == "lanetext" else "_all"
    if allc:
        pd.concat(allc, ignore_index=True).to_parquet(C.WORK / "lr2_corr.parquet",
                                                      index=False)
    P = pd.concat(allp, ignore_index=True).rename(columns={"dev": "DeviceId"})
    P = P.replace([np.inf, -np.inf], np.nan)
    P.to_parquet(C.WORK / f"lr2_pairs{tag}.parquet", index=False)
    C.log(f"wrote {len(P)} pairs")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lanetext", choices=["lanetext", "all"])
    main(ap.parse_args().mode)
