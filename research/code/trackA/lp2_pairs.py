"""Lanes from cabinet prints (note 30), step 2 -- pair cues for the print signals.

Same cues as note 17 (lr2_pairs.py: correlogram zero-lag / 2-8 s lead, 15-min count
correlation off-peak / peak, 1-min high-pass count correlation), computed for every pair
of detectors on the same TIMING phase at the complete-tier print signals (not only the
labelled pairs, so the n_lanes estimate can see every detector on a phase).  Staging window
only: the prints document the current wiring, the Dec-2024 window may predate a rewire.

Output: lp2_pairs.parquet (one row per pair).

    python lp2_pairs.py
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import lr_common as C
import lr2_pairs as L2
import lp1_labels as LP1


def connect():
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(C.DCW / 'tmp').as_posix()}'")
    return con


def main():
    d = LP1.load()
    dets = d[["DeviceId", "detector", "phase_target"]].rename(
        columns={"DeviceId": "dev", "detector": "det", "phase_target": "grp"})
    dets["det"] = dets.det.astype("int16")
    dets = dets.drop_duplicates(["dev", "det"])
    C.log(f"{dets.dev.nunique()} signals, {len(dets)} detectors")
    cache, t0, secs = L2.WIN["stg"]
    con = connect()
    tp = time.time()
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
    con.register("nq_df", nq)
    con.execute("CREATE OR REPLACE TEMP TABLE nq AS SELECT * FROM nq_df")
    con.execute("DELETE FROM onw WHERE NOT EXISTS "
                "(SELECT 1 FROM nq WHERE nq.dev = onw.dev AND nq.det = onw.det)")
    C.log(f"{len(nq)} actuating detectors on {nq.dev.nunique()} signals")
    corr = con.sql(L2.SQL_CORR).df()
    C.log(f"correlogram {corr.shape} ({time.time()-tp:.0f}s)")
    mm = con.sql(L2.SQL_MIN.format(nm=int(secs // 60))).df()
    C.log(f"minute/15-min corr {mm.shape} ({time.time()-tp:.0f}s)")
    cu = L2.cues(corr, nq).merge(mm, on=["dev", "da", "db"], how="left")
    cu["period"] = "stg"
    P = cu.rename(columns={"dev": "DeviceId"}).replace([np.inf, -np.inf], np.nan)
    tg = dets.rename(columns={"dev": "DeviceId", "det": "da", "grp": "target"})
    P = P.merge(tg, on=["DeviceId", "da"], how="left")
    P.to_parquet(C.WORK / "lp2_pairs.parquet", index=False)
    C.log(f"wrote {len(P)} pairs")


if __name__ == "__main__":
    main()
