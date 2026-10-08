"""Note 82 (health review sheet): the classifiability-gate metrics of the package (`health.py` / predict.gate_frame) on
the note-79 w40 windows (26-28 Sep 2026), every non-locked signal.  Same definitions as predict.py: an ON counts when the
next event of the channel is an OFF; frac_time_on = sum(ON->OFF) / window; on_per_day; max ONs in one clock minute.
DuckDB, threads 4, memory 10GB, temp in %DC_WORK%/tmp.

    python h82_gate.py -> %DC_WORK%/health82/gate_w40.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h79_run as R  # noqa: E402
import hb_data as H  # noqa: E402

OUT = H.DCW / "health82"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    con = duckdb.connect()
    con.execute(f"SET threads=4; SET memory_limit='10GB'; SET temp_directory='{(H.DCW / 'tmp').as_posix()}'")
    src = (H.DCW / "health4" / "w40_events").as_posix() + "/*/*.parquet"
    wins = pd.DataFrame([(w, pd.Timestamp(s), pd.Timestamp(s) + pd.Timedelta(hours=h), h)
                         for w, (s, h) in R.WIN["w40"].items()], columns=["window", "w0", "w1", "hours"])
    con.register("wins", wins)
    df = con.execute(f"""
        WITH ev AS (SELECT DISTINCT lower(DeviceId) AS DeviceId, Timestamp AS ts, EventId::INT AS EventId,
                           Parameter::INT AS det
                    FROM read_parquet('{src}', hive_partitioning = true)
                    WHERE EventId IN (81, 82) AND Parameter <= 64),
             we AS (SELECT w."window" AS wn, w.hours, w.w0, w.w1, ev.* FROM ev JOIN wins w ON ev.ts >= w.w0 AND ev.ts < w.w1),
             d AS (SELECT *, LEAD(ts) OVER w_ AS nts, LEAD(EventId) OVER w_ AS nev FROM we
                   WINDOW w_ AS (PARTITION BY wn, DeviceId, det ORDER BY ts, CASE WHEN EventId = 82 THEN 0 ELSE 1 END)),
             o AS (SELECT wn, hours, DeviceId, det, ts, epoch_ms(nts - ts) / 1000.0 AS dur,
                          date_trunc('minute', ts) AS mn
                   FROM d WHERE EventId = 82 AND nev = 81),
             pm AS (SELECT wn, DeviceId, det, mn, count(*) AS c FROM o GROUP BY ALL)
        SELECT a.wn, a.hours, a.DeviceId, a.det AS detector, a.n_on, a.n_on / (a.hours / 24.0) AS on_per_day,
               a.occ / (a.hours * 3600.0) AS frac_time_on, a.longest_on_s, b.max_on_per_min
        FROM (SELECT wn, hours, DeviceId, det, count(*) AS n_on, sum(dur) AS occ, max(dur) AS longest_on_s
              FROM o GROUP BY ALL) a
        JOIN (SELECT wn, DeviceId, det, max(c) AS max_on_per_min FROM pm GROUP BY ALL) b
          USING (wn, DeviceId, det)
    """).df()
    df = df[~df.DeviceId.isin(locked)].rename(columns={"wn": "window"})
    df["gate"] = "ok"
    df.loc[df.max_on_per_min >= 600, "gate"] = "chatter_storm"
    df.loc[df.frac_time_on >= 0.90, "gate"] = "stuck_on"
    df.loc[df.on_per_day < 20, "gate"] = "near_zero_volume"
    df.to_parquet(OUT / "gate_w40.parquet")
    print(df.groupby(["window", "gate"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
