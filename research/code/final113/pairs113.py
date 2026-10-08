"""Note 113: behaviour-only detector-pair statistics for the phase decoder (no channel number in any form).

Per (signal, window, detector a, detector b), from actuation ONSETS in 1-s bins, stratified by the signal's green state
(stratum = green mask x time since the last green-state change: 0-2, 3-7, 8-19, 20+ s; the mask is used only as an
equality key, so renumbering the phases permutes strata and changes nothing):
  O0 / E0   same-second co-onsets, observed vs expected if a and b were independent within each stratum
            (E0 = sum_s n_a,s n_b,s / N_s): removes the "both go at the same begin-green" coincidence that makes
            phi pair the two concurrent through approaches;
  OL / EL   a -> b within +1..+8 s (advance -> stop-bar travel), observed pairs vs the stratum expectation
            (EL = sum_s m_a,s n_b,s / N_s, m_a,s = number of (onset, lag) landing in stratum s);
  na, nb, N onset-bins of a and b, window bins (for a plain 1-s phi).
Runs the shipped staging cache (DC_WORK re-pointed at dc_work/official/stg, as build_stg_features.py), all 22
windows, the 761 pool signals of note 95 (locked_v2 asserted absent).  DuckDB 10 GB / 4 threads.

    python pairs113.py [--chunk 12] [--limit 0]   -> %HOME%/dc_work/s113/pairs113.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HOME_WORK = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
STG = HOME_WORK / "official" / "stg"
os.environ["DC_WORK"] = str(STG)
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import build_features as bf  # noqa: E402
import windows_stg  # noqa: E402,F401
from common import connect  # noqa: E402

OUT = HOME_WORK / "s113"
LAG = 8

SQL = f"""
CREATE OR REPLACE TEMP TABLE ab AS
  SELECT DISTINCT dev, det, floor(t_on)::BIGINT AS b FROM onev WHERE det BETWEEN 1 AND 64;
CREATE OR REPLACE TEMP TABLE allb AS
  WITH seg AS (SELECT dev, t0, t1, mask, ceil(t0)::BIGINT AS b0, (ceil(t1)::BIGINT - 1) AS b1 FROM gs WHERE t1 > t0)
  SELECT dev, b, (mask::BIGINT * 4 + CASE WHEN b - t0 < 3 THEN 0 WHEN b - t0 < 8 THEN 1
                                         WHEN b - t0 < 20 THEN 2 ELSE 3 END) AS s
  FROM (SELECT dev, t0, mask, unnest(range(b0, b1 + 1)) AS b FROM seg WHERE b1 >= b0);
CREATE OR REPLACE TEMP TABLE abs_ AS
  SELECT a.dev, a.det, a.b, coalesce(x.s, -1) AS s FROM ab a LEFT JOIN allb x ON x.dev = a.dev AND x.b = a.b;
CREATE OR REPLACE TEMP TABLE ns AS SELECT dev, s, count(*)::DOUBLE AS N FROM allb GROUP BY 1, 2;
CREATE OR REPLACE TEMP TABLE r AS
  SELECT a.dev, a.det, a.s, count(*)::DOUBLE / any_value(n.N) AS rate, count(*)::DOUBLE AS n
  FROM abs_ a JOIN ns n ON n.dev = a.dev AND n.s = a.s GROUP BY 1, 2, 3;
CREATE OR REPLACE TEMP TABLE m AS
  SELECT a.dev, a.det, x.s, count(*)::DOUBLE AS m
  FROM (SELECT dev, det, b + unnest(range(1, {LAG + 1})) AS bl FROM ab) a JOIN allb x ON x.dev = a.dev AND x.b = a.bl
  GROUP BY 1, 2, 3;
CREATE OR REPLACE TEMP TABLE e0 AS
  SELECT x.dev, x.det AS a, y.det AS c, sum(x.n * y.rate) AS E0
  FROM r x JOIN r y ON x.dev = y.dev AND x.s = y.s AND x.det < y.det GROUP BY 1, 2, 3;
CREATE OR REPLACE TEMP TABLE el AS
  SELECT x.dev, x.det AS a, y.det AS c, sum(x.m * y.rate) AS EL
  FROM m x JOIN r y ON x.dev = y.dev AND x.s = y.s AND x.det <> y.det GROUP BY 1, 2, 3;
CREATE OR REPLACE TEMP TABLE o0 AS
  SELECT x.dev, x.det AS a, y.det AS c, count(*)::DOUBLE AS O0
  FROM ab x JOIN ab y ON x.dev = y.dev AND x.b = y.b AND x.det < y.det GROUP BY 1, 2, 3;
CREATE OR REPLACE TEMP TABLE ol AS
  SELECT x.dev, x.det AS a, y.det AS c, count(*)::DOUBLE AS OL
  FROM (SELECT dev, det, b + unnest(range(1, {LAG + 1})) AS bl FROM ab) x JOIN ab y ON x.dev = y.dev AND y.b = x.bl
  WHERE x.det <> y.det GROUP BY 1, 2, 3;
CREATE OR REPLACE TEMP TABLE na AS SELECT dev, det, count(*)::DOUBLE AS n FROM ab GROUP BY 1, 2;
"""

SQL_OUT = """
WITH sym AS (
  SELECT dev, a, c, O0, 0.0 AS E0 FROM o0 UNION ALL SELECT dev, c, a, O0, 0.0 FROM o0
  UNION ALL SELECT dev, a, c, 0.0, E0 FROM e0 UNION ALL SELECT dev, c, a, 0.0, E0 FROM e0),
s0 AS (SELECT dev, a, c, sum(O0) AS O0, sum(E0) AS E0 FROM sym GROUP BY 1, 2, 3),
lag AS (SELECT dev, a, c, sum(OL) AS OL, sum(EL) AS EL FROM
        (SELECT dev, a, c, OL, 0.0 AS EL FROM ol UNION ALL SELECT dev, a, c, 0.0, EL FROM el) GROUP BY 1, 2, 3)
SELECT coalesce(s0.dev, lag.dev) AS dev, coalesce(s0.a, lag.a) AS a, coalesce(s0.c, lag.c) AS c,
       coalesce(s0.O0, 0) AS O0, coalesce(s0.E0, 0) AS E0, coalesce(lag.OL, 0) AS OL, coalesce(lag.EL, 0) AS EL
FROM s0 FULL OUTER JOIN lag ON s0.dev = lag.dev AND s0.a = lag.a AND s0.c = lag.c
"""


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_lean(con, devs):
    """Only the two tables this note needs (bf.load_chunk also loads calls / cycles / coordination)."""
    dev_list = ",".join("'" + d + "'" for d in devs)
    C = STG / "cache"
    con.execute("CREATE OR REPLACE TEMP TABLE devmap AS SELECT DeviceId, row_number() OVER (ORDER BY DeviceId)::SMALLINT AS dev "
                f"FROM (SELECT unnest([{dev_list}]) AS DeviceId)")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE onev_all AS
        SELECT m.dev, i.Detector::SMALLINT AS det, epoch_ms(i.t_on)/1000.0 AS t_on
        FROM read_parquet('{(C / 'det_intervals.parquet').as_posix()}') i
        JOIN devmap m USING (DeviceId) WHERE i.DeviceId IN ({dev_list})""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE gs_all AS
        SELECT m.dev, epoch_ms(g.t_start)/1000.0 AS t0, epoch_ms(g.t_end)/1000.0 AS t1, g.mask
        FROM read_parquet('{(C / 'green_state.parquet').as_posix()}') g
        JOIN devmap m USING (DeviceId) WHERE g.t_end IS NOT NULL AND g.DeviceId IN ({dev_list})""")


def signals() -> list[str]:
    p = pd.read_parquet(HOME_WORK / "s95" / "phase" / "pool.parquet", columns=["DeviceId"]).DeviceId.unique()
    raw = sorted({d.replace("@stg", "") for d in p})
    lk = set(pd.read_csv(HOME_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not any(d.lower() in lk for d in raw)
    meta = pd.read_parquet(STG / "cache" / "signal_meta.parquet", columns=["DeviceId"]).DeviceId
    have = {d.lower(): d for d in meta}
    out = [have[d.lower()] for d in raw if d.lower() in have]
    log(f"{len(raw)} pool signals, {len(out)} in the staging cache")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="pairs113.parquet")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    devs = signals()
    if a.limit:
        devs = devs[:a.limit]
    con = connect(threads=4)
    wins = bf.WINDOW_SETS["stgall"]
    parts, t0 = [], time.time()
    for i in range(0, len(devs), a.chunk):
        ch, t1 = devs[i:i + a.chunk], time.time()
        tl = time.time()
        load_lean(con, ch)
        dm = con.sql("SELECT * FROM devmap").df()
        tl = time.time() - tl
        for w in wins:
            w0 = bf._epoch(w)
            w1 = w0 + w["secs"]
            con.execute(f"CREATE OR REPLACE TEMP TABLE onev AS SELECT * FROM onev_all WHERE t_on >= {w0} AND t_on < {w1}")
            con.execute(f"CREATE OR REPLACE TEMP TABLE gs AS SELECT dev, greatest(t0,{w0}) AS t0, least(t1,{w1}) AS t1, mask "
                        f"FROM gs_all WHERE t1 > {w0} AND t0 < {w1}")
            con.execute(SQL)
            d = con.sql(SQL_OUT).df()
            n = con.sql("SELECT * FROM na").df()
            if not len(d):
                continue
            d = d.merge(n.rename(columns={"det": "a", "n": "na"}), on=["dev", "a"], how="left")
            d = d.merge(n.rename(columns={"det": "c", "n": "nb"}), on=["dev", "c"], how="left")
            d = d.merge(dm, on="dev").drop(columns="dev")
            d["win"], d["N"] = w["win"], float(w["secs"])
            for c in ("O0", "E0", "OL", "EL", "na", "nb", "N"):
                d[c] = d[c].astype(np.float32)
            d["a"], d["c"] = d.a.astype(np.int16), d.c.astype(np.int16)
            parts.append(d)
        log(f"chunk {i // a.chunk + 1}/{(len(devs) + a.chunk - 1) // a.chunk} {time.time() - t1:.0f}s "
            f"(load {tl:.0f}s) elapsed {time.time() - t0:.0f}s")
    df = pd.concat(parts, ignore_index=True)
    df["DeviceId"] = df.DeviceId + "@stg"
    df.to_parquet(OUT / a.out, index=False)
    log(f"wrote {OUT / a.out}: {len(df):,} rows")


if __name__ == "__main__":
    main()
