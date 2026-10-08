"""Permissive-phase recognition from the hi-res log alone (note 50 experiment, 2026-09-29).

Protected-permissive (FYA) left turns and right-turn lanes stay ~4 pt harder for the function head (notes 44, 45).
FYA events 32 / 33 are NOT allowed as model inputs, so the permissive clue is rebuilt from allowed codes only
(81 / 82 detector, 1 / 8 / 10 color, 43 / 44 phase call), per window and per candidate phase; the detectors of a
phase are those whose PREDICTED phase (frame pred_phase) it is -- never the print, the timing or a phase number.

Per detector (on its predicted phase p):
  n_wait, leave     red waits (ON starting in p's red, >= 3 s) and the share that leave before green (OFF > 2 s
                    before green, no new ON within 3 s) -- note 44's statistic; RTOR in a right-turn lane leaves too
  on_green          share of p's green starts at which the detector is ON (stop-bar-ness, from behaviour)
Per phase (pm_ph_*):
  leave_maxwait     leave on the detector with the most red waits (note 44 "any label" version, AUC .82 there)
  leave_sb          leave on the detector maximising n_wait * on_green (behaviour-chosen stop-bar zone)
  leave_wmean       n_wait-weighted mean leave;  leave_max  largest leave (n_wait >= 10)
  wait_per_cyc      red waits per cycle over the phase's detectors
  drop43            of phase calls (43) placed in p's red, share withdrawn (44) >= 1 s before p's green
  n43red_per_cyc    those calls per cycle;  green_share, cyc_ratio (green starts / the window's busiest phase)
Per detector relation (pm_det_*): own leave, own leave - leave_sb, is the stop-bar pick, own share of the phase's
red waits. A permissive-phase SCORE (pm_score) is fitted per phase in pm_score.py (OOF) and joined afterwards.

    python pm_features.py --period stg | dec   -> %DC_WORK%/trackA/pm/pm_{det,ph}_{period}.parquet
    python pm_features.py --labels             -> %DC_WORK%/trackA/pm/pm_labels.parquet (EVALUATION ONLY: FYA-event
                                                  counts per signal / period / phase from the raw pulls; never a
                                                  model input, used as the phase-score TARGET and for AUCs)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import a2_features as A2F

DCW = A2F.DCW
OUT = DCW / "trackA" / "pm"
FRAME = DCW / "function_v4" / "funcframe_v6.parquet"
LOCKED_V2 = DCW / "official" / "locked_v2.csv"
EVENTS = {"dec": DCW / "cache" / "events", "stg": DCW / "official" / "stg" / "cache" / "events"}
RAW32 = {"stg": (DCW / "data" / "staging_other" / "*/*.parquet").as_posix(),
         "dec": (DCW / "data" / "raw" / "Train_Dec_*.parquet").as_posix()}
MIN_WAIT = 10
log = A2F.log


def connect():
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'; SET threads=6; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


SQL_DET = """
WITH o AS (SELECT onw.*, tgt.p FROM onw JOIN tgt USING (dev, det)),
nx AS (SELECT *, lead(t_on) OVER (PARTITION BY dev, det ORDER BY t_on) AS nxt FROM o),
j AS (SELECT nx.*, c.gs, c.ge, c.ng FROM nx ASOF JOIN cycw c ON nx.dev = c.dev AND nx.p = c.p AND nx.t_on >= c.gs),
k AS (SELECT *, t_on >= ge AS in_r FROM j WHERE t_on < ng)
SELECT dev, det, count(*) FILTER (in_r AND dur >= 3) AS n_wait,
       avg(CASE WHEN in_r AND dur >= 3 THEN (t_off < ng - 2 AND coalesce(nxt, 1e12) > least(t_off + 3, ng))::INT END)
           AS leave
FROM k GROUP BY 1, 2
"""
SQL_ONG = """
WITH dg AS (SELECT tgt.dev, tgt.det, c.gs FROM tgt JOIN cycw c ON tgt.dev = c.dev AND tgt.p = c.p WHERE c.gs >= {t0}),
a AS (SELECT dg.*, onw.t_off FROM dg ASOF LEFT JOIN onw ON dg.dev = onw.dev AND dg.det = onw.det AND dg.gs >= onw.t_on)
SELECT dev, det, avg((coalesce(t_off, 0) > gs)::INT) AS on_green, count(*) AS n_gs FROM a GROUP BY 1, 2
"""
SQL_PH = """SELECT dev, p, count(*) AS n_cyc, sum(least(ge, {t1}) - greatest(gs, {t0})) / {secs} AS green_share
            FROM cycw WHERE gs >= {t0} GROUP BY 1, 2"""
SQL_43 = """
WITH c43 AS (SELECT * FROM calls WHERE e = 43), c44 AS (SELECT * FROM calls WHERE e = 44),
j AS (SELECT c43.dev, c43.p, c43.t, c.ge, c.ng FROM c43 ASOF JOIN cycw c ON c43.dev = c.dev AND c43.p = c.p
      AND c43.t >= c.gs),
r AS (SELECT * FROM j WHERE t >= ge AND t < ng),
k AS (SELECT r.*, c44.t AS t44 FROM r ASOF LEFT JOIN c44 ON r.dev = c44.dev AND r.p = c44.p AND c44.t >= r.t)
SELECT dev, p, count(*) AS n43red, avg((coalesce(t44, 1e12) < ng - 1)::INT) AS drop43 FROM k GROUP BY 1, 2
"""


def phase_agg(t: pd.DataFrame, ph: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """t: per detector (dev, det, p, n, n_wait, leave, on_green); ph: per (dev, p) cycles / calls."""
    t = t.copy()
    t["lv"] = t.leave.where(t.n_wait >= MIN_WAIT)
    t["sbw"] = t.n_wait.fillna(0) * t.on_green.fillna(0)
    g = t.groupby(["dev", "p"], sort=False)
    rows = []
    for (dev, p), d in g:
        ok = d[d.lv.notna()]
        if len(ok):
            mw = ok.loc[ok.n_wait.idxmax()]
            sb = ok.loc[ok.sbw.idxmax()]
            rows.append((dev, p, mw.lv, sb.lv, int(sb.det), np.average(ok.lv, weights=ok.n_wait), ok.lv.max(),
                         d.n_wait.fillna(0).sum()))
        else:
            rows.append((dev, p, np.nan, np.nan, -1, np.nan, np.nan, d.n_wait.fillna(0).sum()))
    P = pd.DataFrame(rows, columns=["dev", "p", "pm_ph_leave_maxwait", "pm_ph_leave_sb", "sb_det",
                                    "pm_ph_leave_wmean", "pm_ph_leave_max", "tot_wait"])
    P = ph.merge(P, on=["dev", "p"], how="left")
    P["pm_ph_wait_per_cyc"] = P.tot_wait.fillna(0) / P.n_cyc
    P["pm_ph_drop43"] = P.drop43.where(P.n43red >= 5)
    P["pm_ph_n43red_per_cyc"] = P.n43red.fillna(0) / P.n_cyc
    P["pm_ph_green_share"] = P.green_share
    P["pm_ph_cyc_ratio"] = P.n_cyc / P.groupby("dev").n_cyc.transform("max")
    t = t.merge(P[["dev", "p", "pm_ph_leave_sb", "sb_det", "tot_wait"]], on=["dev", "p"], how="left")
    t["pm_det_leave"] = t.lv
    t["pm_det_leave_minus_sb"] = t.lv - t.pm_ph_leave_sb
    t["pm_det_is_sb"] = (t.det == t.sb_det).astype(float).where(t.sb_det.notna() & (t.sb_det >= 0))
    t["pm_det_wait_share"] = (t.n_wait / t.tot_wait).where(t.tot_wait > 0)
    return t, P


PH_FEATS = ["pm_ph_leave_maxwait", "pm_ph_leave_sb", "pm_ph_leave_wmean", "pm_ph_leave_max", "pm_ph_wait_per_cyc",
            "pm_ph_drop43", "pm_ph_n43red_per_cyc", "pm_ph_green_share", "pm_ph_cyc_ratio"]
DET_FEATS = ["pm_det_leave", "pm_det_leave_minus_sb", "pm_det_is_sb", "pm_det_wait_share"]


def build(period: str, only: str | None = None) -> None:
    t_all = time.time()
    con = connect()
    cache = A2F.CACHE[period]
    di, pc = (cache / "det_intervals.parquet").as_posix(), (cache / "phase_cycles.parquet").as_posix()
    ev = (EVENTS[period] / "*" / "*.parquet").as_posix()
    fr = pd.read_parquet(FRAME, columns=["DeviceId", "Detector", "win", "period", "pred_phase"])
    fr = fr[fr.period == period]
    fr["DeviceId"] = fr.DeviceId.str.lower()
    lock = set(pd.read_csv(LOCKED_V2).DeviceId.astype(str).str.lower())
    assert not fr.DeviceId.isin(lock).any(), "locked signal in the frame"
    log(f"[{period}] targets {fr.shape}, {fr.DeviceId.nunique()} signals")
    dparts, pparts = [], []
    for name, t0, secs in A2F.WINDOWS[period]:
        if only and name != only:
            continue
        tw = time.time()
        t1 = t0 + pd.Timedelta(seconds=secs)
        e0 = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
        e1 = e0 + secs
        tg = fr[fr.win == name][["DeviceId", "Detector", "pred_phase"]].rename(
            columns={"DeviceId": "dev", "Detector": "det", "pred_phase": "p"}).astype({"det": "int16", "p": "int16"})
        con.register("tgt_df", tg)
        con.execute("CREATE OR REPLACE TEMP TABLE tgt AS SELECT * FROM tgt_df")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE onw AS
            SELECT lower(DeviceId) AS dev, Detector::SMALLINT AS det, epoch_ms(t_on) / 1000.0 AS t_on,
                   epoch_ms(t_off) / 1000.0 AS t_off, dur::DOUBLE AS dur
            FROM read_parquet('{di}')
            WHERE t_on >= TIMESTAMP '{t0}' AND t_on < TIMESTAMP '{t1}' AND Detector BETWEEN 1 AND 64
              AND dur IS NOT NULL AND lower(DeviceId) IN (SELECT DISTINCT dev FROM tgt)""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cycw AS
            SELECT lower(DeviceId) AS dev, Phase::SMALLINT AS p, epoch_ms(green_start) / 1000.0 AS gs,
                   epoch_ms(coalesce(yellow_start, red_start, next_green)) / 1000.0 AS ge,
                   epoch_ms(next_green) / 1000.0 AS ng
            FROM read_parquet('{pc}')
            WHERE Phase BETWEEN 1 AND 16 AND next_green IS NOT NULL
              AND epoch_ms(next_green - green_start) / 1000.0 <= 900
              AND green_start >= TIMESTAMP '{t0 - pd.Timedelta(minutes=15)}' AND green_start < TIMESTAMP '{t1}'
              AND lower(DeviceId) IN (SELECT DISTINCT dev FROM tgt)""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE calls AS
            SELECT DISTINCT lower(DeviceId) AS dev, Parameter::SMALLINT AS p, EventId::SMALLINT AS e,
                   epoch_ms(Timestamp) / 1000.0 AS t
            FROM read_parquet('{ev}', hive_partitioning = true)
            WHERE EventId IN (43, 44) AND Timestamp >= TIMESTAMP '{t0}' AND Timestamp < TIMESTAMP '{t1}'
              AND lower(DeviceId) IN (SELECT DISTINCT dev FROM tgt)""")
        det = con.sql(SQL_DET).df()
        ong = con.sql(SQL_ONG.format(t0=e0)).df()
        ph = con.sql(SQL_PH.format(t0=e0, t1=e1, secs=secs)).df()
        c43 = con.sql(SQL_43).df()
        ph = ph.merge(c43, on=["dev", "p"], how="left")
        t = tg.merge(det, on=["dev", "det"], how="left").merge(ong[["dev", "det", "on_green"]], on=["dev", "det"],
                                                               how="left")
        t, P = phase_agg(t, ph)
        d = t[["dev", "det", "p"] + DET_FEATS].rename(columns={"dev": "DeviceId", "det": "Detector"})
        d["win"], d["period"] = name, period
        P = P[["dev", "p"] + PH_FEATS].rename(columns={"dev": "DeviceId"})
        P["win"], P["period"] = name, period
        dparts.append(d)
        pparts.append(P)
        log(f"[{period}] {name}: det {len(d)} (leave {int(d.pm_det_leave.notna().sum())}), phases {len(P)} "
            f"(leave_sb {int(P.pm_ph_leave_sb.notna().sum())}, drop43 {int(P.pm_ph_drop43.notna().sum())}) "
            f"({time.time() - tw:.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.concat(dparts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    P = pd.concat(pparts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    sfx = f"_{only}" if only else ""
    D.to_parquet(OUT / f"pm_det_{period}{sfx}.parquet", index=False)
    P.to_parquet(OUT / f"pm_ph_{period}{sfx}.parquet", index=False)
    log(f"wrote pm_det_{period} {D.shape}, pm_ph_{period} {P.shape} ({time.time() - t_all:.0f}s)")


def labels() -> None:
    """EVALUATION / TARGET ONLY: FYA begin-permissive events (32) per signal, period and phase in the raw pulls."""
    con = connect()
    out = []
    for per, g in RAW32.items():
        out.append(con.sql(f"""SELECT lower(DeviceId) AS DeviceId, '{per}' AS period, Parameter::INT AS p,
                                      count(*) AS n32
                               FROM read_parquet('{g}') WHERE EventId = 32 GROUP BY ALL""").df())
    L = pd.concat(out, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    L.to_parquet(OUT / "pm_labels.parquet", index=False)
    log(f"labels: {len(L)} (signal, period, phase) rows with FYA events; signals logging: "
        f"{L.groupby('period').DeviceId.nunique().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dec", "stg"])
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.labels:
        labels()
    else:
        build(a.period, a.only)
