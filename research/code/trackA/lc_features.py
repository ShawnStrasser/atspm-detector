"""Label-check statistics as MODEL features (note 45 experiment, 2026-09-29).

The label check (note 29) judges a labelled detector with statistics the engineer uses. Those that need only the
hi-res log and no phase number are rebuilt here per detector-window, with the detector's PREDICTED phase (frame
column pred_phase) as the only grouping key -- never the print, the timing, a lane or a technology:

  lc_hold            share of ONs starting in the (predicted) phase's red that stay ON until its green
  lc_on_green        share of the phase's green starts at which the detector is ON
  lc_red_arr         (share of ONs starting in red) / (red share of the time): arrivals on red vs chance
  lc_leave           of red waits (ON in red lasting >= 3 s) the share that leave before green (permissive / RTOR)
  lc_phase_leave     the largest lc_leave on the detector's predicted phase (a permissive-phase clue)
  lc_occ_rel         occupancy / median occupancy of the other detectors on the predicted phase
  lc_cnt_rel         actuations / median actuations of the other detectors on the predicted phase
  lc_moreocc_fewer   share of those siblings that this detector out-occupies (>= 1.2x) with FEWER actuations
                     (the presence side of a presence / count pair)
  lc_lessocc_more    the reverse (the count side)
  lc_lead_out        max over siblings of (share of the sibling's ONs preceded 1.5-9 s by this detector - chance)
  lc_lead_in         max over siblings of (share of this detector's ONs preceded 1.5-9 s by the sibling - chance)
  lc_order_pos       (#siblings it leads - #siblings leading it) / #siblings with a lead value (upstream > 0)

NaN when the window lacks the condition (fewer than 10 red ONs, no sibling, a partner so busy that chance explains
the order, ...). Windows and caches are a2_features' (22 per period).

    python lc_features.py --period stg | dec        -> %DC_WORK%/trackA/lc/feat_lc_{period}.parquet
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import a2_features as A2F

DCW = A2F.DCW
OUT = DCW / "trackA" / "lc"
FRAME = DCW / "function_v4" / "funcframe_v6.parquet"
LOCKED_V2 = DCW / "official" / "locked_v2.csv"
SAMPLE_Y = 400          # ONs per detector sampled for the order statistic
LEAD = (1.5, 9.0)
CHANCE = (37.5, 45.0)
BUSY = 0.60
MIN_RED, MIN_ONS, MIN_WAIT, MIN_Y = 10, 20, 10, 20
FEATS = ["lc_hold", "lc_on_green", "lc_red_arr", "lc_leave", "lc_phase_leave", "lc_occ_rel", "lc_cnt_rel",
         "lc_moreocc_fewer", "lc_lessocc_more", "lc_lead_out", "lc_lead_in", "lc_order_pos"]
log = A2F.log


def connect():
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=8; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


SQL_COL = """
WITH o AS (SELECT onw.*, tgt.p FROM onw JOIN tgt USING (dev, det)),
nx AS (SELECT *, lead(t_on) OVER (PARTITION BY dev, det ORDER BY t_on) AS nxt FROM o),
j AS (SELECT nx.*, c.gs, c.ge, c.ng FROM nx ASOF JOIN cycw c ON nx.dev = c.dev AND nx.p = c.p AND nx.t_on >= c.gs),
k AS (SELECT *, t_on >= ge AS in_r FROM j WHERE t_on < ng)
SELECT dev, det, count(*) FILTER (in_r) AS n_r, count(*) FILTER (NOT in_r) AS n_g,
       avg(CASE WHEN in_r THEN (t_off >= ng - 0.05)::INT END) AS hold,
       count(*) FILTER (in_r AND dur >= 3) AS n_wait,
       avg(CASE WHEN in_r AND dur >= 3 THEN (t_off < ng - 2 AND coalesce(nxt, 1e12) > least(t_off + 3, ng))::INT END)
           AS leave
FROM k GROUP BY 1, 2
"""
SQL_RED = """SELECT dev, p, sum(ng - ge) / nullif(sum(ng - gs), 0) AS red_share, count(*) AS n_cyc
             FROM cycw WHERE gs >= {t0} GROUP BY 1, 2"""
SQL_ONG = """
WITH dg AS (SELECT tgt.dev, tgt.det, c.gs FROM tgt JOIN cycw c ON tgt.dev = c.dev AND tgt.p = c.p WHERE c.gs >= {t0}),
a AS (SELECT dg.*, onw.t_off FROM dg ASOF LEFT JOIN onw ON dg.dev = onw.dev AND dg.det = onw.det AND dg.gs >= onw.t_on)
SELECT dev, det, avg((coalesce(t_off, 0) > gs)::INT) AS on_green, count(*) AS n_gs FROM a GROUP BY 1, 2
"""
SQL_OCC = """SELECT dev, det, count(*) AS n, sum(least(t_off, {t1}) - greatest(t_on, {t0})) / {secs} AS occ
             FROM onw GROUP BY 1, 2"""
SQL_ORDER = f"""
WITH q AS (SELECT pr.dev, pr.x, pr.y, ys.t FROM pr JOIN ys ON pr.dev = ys.dev AND pr.y = ys.det),
a AS (SELECT q.*, onw.t_on AS tx1 FROM q ASOF LEFT JOIN onw ON q.dev = onw.dev AND q.x = onw.det AND q.t - {LEAD[0]} >= onw.t_on),
b AS (SELECT a.*, onw.t_on AS tx2 FROM a ASOF LEFT JOIN onw ON a.dev = onw.dev AND a.x = onw.det AND a.t - {CHANCE[0]} >= onw.t_on)
SELECT dev, x, y, count(*) AS n_y, avg(coalesce(t - tx1 <= {LEAD[1]}, false)::INT) AS sh_before,
       avg(coalesce(t - tx2 <= {CHANCE[1]}, false)::INT) AS sh_chance
FROM b GROUP BY ALL
"""


def sibling_feats(t: pd.DataFrame, O: pd.DataFrame) -> pd.DataFrame:
    """Per detector: occupancy / count relative to its predicted-phase siblings, the phase's largest leave share,
    and the order features from the pair table O (x -> y lead)."""
    t = t.copy()
    rows = []
    for (dev, p), g in t.groupby(["dev", "p"], sort=False):
        occ, n = g.occ.to_numpy(float), g.n.to_numpy(float)
        lv = g.leave.where(g.n_wait >= MIN_WAIT).to_numpy(float)
        pl = np.nanmax(lv) if np.isfinite(lv).any() else np.nan
        for i in range(len(g)):
            o = np.r_[occ[:i], occ[i + 1:]]
            c = np.r_[n[:i], n[i + 1:]]
            ok = c >= 5
            if n[i] < 5 or not ok.any():
                rows.append((np.nan, np.nan, np.nan, np.nan, pl))
                continue
            o, c = o[ok], c[ok]
            mo, mc = np.median(o), np.median(c)
            rows.append((occ[i] / mo if mo > 0 else np.nan, n[i] / mc if mc > 0 else np.nan,
                         float(np.mean((occ[i] >= 1.2 * o) & (n[i] < c))),
                         float(np.mean((o >= 1.2 * occ[i]) & (c < n[i]))), pl))
    idx = [ix for _, g in t.groupby(["dev", "p"], sort=False) for ix in g.index]
    R = pd.DataFrame(rows, index=idx, columns=["lc_occ_rel", "lc_cnt_rel", "lc_moreocc_fewer", "lc_lessocc_more",
                                               "lc_phase_leave"])
    t = t.join(R)
    if len(O):
        O = O.copy()
        O["lead"] = (O.sh_before - O.sh_chance).where((O.n_y >= MIN_Y) & (O.sh_chance <= BUSY))
        O = O[O.lead.notna()]
        out_ = O.groupby(["dev", "x"]).lead.max().rename("lc_lead_out")
        in_ = O.groupby(["dev", "y"]).lead.max().rename("lc_lead_in")
        lead_n = O.assign(f=(O.lead >= 0.1).astype(int))
        up = lead_n.groupby(["dev", "x"]).f.sum()
        dn = lead_n.groupby(["dev", "y"]).f.sum()
        nx = O.groupby(["dev", "x"]).size()
        ny = O.groupby(["dev", "y"]).size()
        k = list(zip(t.dev, t.det))
        u = up.reindex(k).fillna(0).to_numpy()
        dd = dn.reindex(k).fillna(0).to_numpy()
        m = np.maximum(nx.reindex(k).fillna(0).to_numpy(), ny.reindex(k).fillna(0).to_numpy())
        t["lc_lead_out"] = out_.reindex(k).to_numpy()
        t["lc_lead_in"] = in_.reindex(k).to_numpy()
        t["lc_order_pos"] = np.where(m > 0, (u - dd) / np.maximum(m, 1), np.nan)
    else:
        t["lc_lead_out"] = t["lc_lead_in"] = t["lc_order_pos"] = np.nan
    return t


def build(period: str, only: str | None = None) -> None:
    t_all = time.time()
    con = connect()
    cache = A2F.CACHE[period]
    di, pc = (cache / "det_intervals.parquet").as_posix(), (cache / "phase_cycles.parquet").as_posix()
    fr = pd.read_parquet(FRAME, columns=["DeviceId", "Detector", "win", "period", "pred_phase"])
    fr = fr[fr.period == period]
    lock = set(pd.read_csv(LOCKED_V2).DeviceId.astype(str).str.lower()) if LOCKED_V2.exists() else set()
    assert not fr.DeviceId.str.lower().isin(lock).any(), "locked signal in the frame"
    log(f"[{period}] targets {fr.shape}, {fr.DeviceId.nunique()} signals")
    parts = []
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
            SELECT DeviceId AS dev, Detector::SMALLINT AS det, epoch_ms(t_on) / 1000.0 AS t_on,
                   epoch_ms(t_off) / 1000.0 AS t_off, dur::DOUBLE AS dur
            FROM read_parquet('{di}')
            WHERE t_on >= TIMESTAMP '{t0}' AND t_on < TIMESTAMP '{t1}' AND Detector BETWEEN 1 AND 64
              AND dur IS NOT NULL AND DeviceId IN (SELECT DISTINCT dev FROM tgt)""")
        # cycles whose green starts up to 15 min before the window (an ON early in the window joins its cycle)
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cycw AS
            SELECT DeviceId AS dev, Phase::SMALLINT AS p, epoch_ms(green_start) / 1000.0 AS gs,
                   epoch_ms(coalesce(yellow_start, red_start, next_green)) / 1000.0 AS ge,
                   epoch_ms(next_green) / 1000.0 AS ng
            FROM read_parquet('{pc}')
            WHERE Phase BETWEEN 1 AND 16 AND next_green IS NOT NULL
              AND epoch_ms(next_green - green_start) / 1000.0 <= 900
              AND green_start >= TIMESTAMP '{t0 - pd.Timedelta(minutes=15)}' AND green_start < TIMESTAMP '{t1}'
              AND DeviceId IN (SELECT DISTINCT dev FROM tgt)""")
        col = con.sql(SQL_COL).df()
        red = con.sql(SQL_RED.format(t0=e0)).df()
        ong = con.sql(SQL_ONG.format(t0=e0)).df()
        occ = con.sql(SQL_OCC.format(t0=e0, t1=e1, secs=secs)).df()
        t = tg.merge(occ, on=["dev", "det"], how="left").merge(col, on=["dev", "det"], how="left")
        t = t.merge(red, on=["dev", "p"], how="left").merge(ong, on=["dev", "det"], how="left")
        t["n"] = t.n.fillna(0)
        t["occ"] = t.occ.fillna(0)
        # sibling pairs on the same predicted phase, both with >= 5 actuations
        s = t[t.n >= 5][["dev", "det", "p"]]
        pr = s.merge(s, on=["dev", "p"], suffixes=("_x", "_y"))
        pr = pr[pr.det_x != pr.det_y].rename(columns={"det_x": "x", "det_y": "y"})[["dev", "x", "y"]]
        O = pd.DataFrame()
        if len(pr):
            con.register("pr_df", pr.astype({"x": "int16", "y": "int16"}))
            con.execute("CREATE OR REPLACE TEMP TABLE pr AS SELECT * FROM pr_df")
            con.execute(f"""CREATE OR REPLACE TEMP TABLE ys AS
                SELECT dev, det, t_on AS t FROM onw WHERE (dev, det) IN (SELECT DISTINCT dev, y FROM pr)
                QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY hash(t_on)) <= {SAMPLE_Y}""")
            O = con.sql(SQL_ORDER).df()
        t = sibling_feats(t, O)
        t["lc_hold"] = t.hold.where(t.n_r >= MIN_RED)
        t["lc_red_arr"] = ((t.n_r / (t.n_r + t.n_g)) / t.red_share).where((t.n_r + t.n_g) >= MIN_ONS)
        t["lc_leave"] = t.leave.where(t.n_wait >= MIN_WAIT)
        t["lc_on_green"] = t.on_green.where(t.n_gs >= 5)
        f = t[["dev", "det"] + FEATS].rename(columns={"dev": "DeviceId", "det": "Detector"})
        f["win"], f["period"] = name, period
        parts.append(f)
        log(f"[{period}] {name}: {len(f)} rows, pairs {len(O)}, non-null "
            f"{ {c: int(f[c].notna().sum()) for c in ('lc_hold', 'lc_leave', 'lc_occ_rel', 'lc_lead_out')} } "
            f"({time.time() - tw:.0f}s)")
    df = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    for c in FEATS:
        df[c] = df[c].astype(np.float32)
    df["Detector"] = df.Detector.astype("int16")
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / (f"feat_lc_{period}" + (f"_{only}" if only else "") + ".parquet")
    df.to_parquet(dest, index=False)
    log(f"wrote {dest} {df.shape} ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=["dec", "stg"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    build(a.period, a.only)
