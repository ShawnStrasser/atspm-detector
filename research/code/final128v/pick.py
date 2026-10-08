"""pick 10 non-locked pool signals never used in a package parity / bench / robustness set; extract 24 h (2026-09-20 06:00 -> 09-24 06:00)"""
import os
import duckdb, json, numpy as np, pandas as pd, glob
from pathlib import Path
W = Path(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work"))); O = W / "s128v" / "ev"
c = duckdb.connect(); c.execute("set threads=4; set memory_limit='10GB'")
used = set(pd.read_csv(W / "s115/parity/signals115.csv").DeviceId.str.lower())
used |= {r["DeviceId"].lower() for r in json.loads((W / "s118c/plot/rows.json").read_text())}
for pat in ["bench71/*.parquet", "s115v/ev/*.parquet", "s115dv/ev/*.parquet", "s124v/ev/*.parquet", "s124v/rob/week_*.parquet", "s128/clip/*.parquet"]:
    for f in glob.glob(str(W / pat)):
        try:
            used |= {d.lower() for d in c.sql(f"select distinct DeviceId from '{Path(f).as_posix()}'").df().DeviceId.dropna()}
        except Exception as e:
            print("skip", f, e)
print("used", len(used))
pool = {d.replace("@stg", "").lower() for d in c.sql(f"select distinct DeviceId from '{(W/'s95/phase/pool.parquet').as_posix()}'").df().DeviceId}
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
meta = c.sql(f"select DeviceId, n_det_channels, t0, t1 from '{(W/'official/stg/cache/signal_meta.parquet').as_posix()}'").df()
low = meta.DeviceId.str.lower()
meta = meta[low.isin(pool) & ~low.isin(lk) & ~low.isin(used)]
meta = meta[(meta.t0 <= "2026-09-20 06:00") & (meta.t1 >= "2026-09-21 06:00")].copy()
print("eligible", len(meta))
meta["q"] = pd.qcut(meta.n_det_channels.rank(method="first"), 5, labels=False)
rng = np.random.default_rng(1281)
pick = pd.concat([g.iloc[rng.choice(len(g), 2, replace=False)] for _, g in meta.groupby("q")]).sort_values("n_det_channels")
assert not pick.DeviceId.str.lower().isin(lk | used).any()
rows = []
for i, r in enumerate(pick.itertuples()):
    src = (W / "official/stg/cache/events" / f"DeviceId={r.DeviceId}").as_posix()
    f = O / f"x{i:02d}.parquet"
    c.execute(f"""COPY (SELECT '{r.DeviceId}' AS DeviceId, Timestamp, EventId, Parameter FROM read_parquet('{src}/*.parquet')
        WHERE Timestamp >= TIMESTAMP '2026-09-20 06:00:00' AND Timestamp < TIMESTAMP '2026-09-21 06:00:00') TO '{f.as_posix()}' (FORMAT parquet)""")
    n, t0, t1 = c.sql(f"select count(*), min(Timestamp), max(Timestamp) from '{f.as_posix()}'").fetchone()
    rows.append(dict(case=f"x{i:02d}", DeviceId=r.DeviceId, n_det_channels=r.n_det_channels, n_events=n, t0=t0, t1=t1))
p = pd.DataFrame(rows); p.to_csv(O / "signals.csv", index=False); print(p.to_string())
