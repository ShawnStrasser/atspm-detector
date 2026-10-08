"""Note 115 parity set: 40 non-locked pool signals (seed 115, stratified by detector-channel count), one 24-h event
extract each (2026-09-19 06:00 -> 2026-09-20 06:00, allowed + some disallowed codes kept: the package filters them).
30-min / 3-h cases are cut from it by predict(start=, end=).  Local cache only, no new pull."""
import duckdb, numpy as np, pandas as pd
import os
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
OUT = W / "s115" / "parity"
c = duckdb.connect(); c.execute("set threads=4; set memory_limit='10GB'")
pool = {d.replace("@stg", "").lower() for d in c.sql(f"select distinct DeviceId from '{(W/'s95/phase/pool.parquet').as_posix()}'").df().DeviceId}
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
meta = c.sql(f"select DeviceId, n_det_channels, t0, t1 from '{(W/'official/stg/cache/signal_meta.parquet').as_posix()}'").df()
meta = meta[meta.DeviceId.str.lower().isin(pool) & ~meta.DeviceId.str.lower().isin(lk)]
meta = meta[(meta.t0 <= "2026-09-19 06:00") & (meta.t1 >= "2026-09-20 06:00")].copy()
meta["q"] = pd.qcut(meta.n_det_channels.rank(method="first"), 4, labels=False)
rng = np.random.default_rng(115)
pick = pd.concat([g.iloc[rng.choice(len(g), 10, replace=False)] for _, g in meta.groupby("q")]).sort_values("n_det_channels")
assert not pick.DeviceId.str.lower().isin(lk).any()
rows = []
for i, r in enumerate(pick.itertuples()):
    src = (W / "official/stg/cache/events" / f"DeviceId={r.DeviceId}").as_posix()
    f = OUT / "ev" / f"s{i:02d}.parquet"
    c.execute(f"""COPY (SELECT '{r.DeviceId}' AS DeviceId, Timestamp, EventId, Parameter
        FROM read_parquet('{src}/*.parquet')
        WHERE Timestamp >= TIMESTAMP '2026-09-19 06:00:00' AND Timestamp < TIMESTAMP '2026-09-20 06:00:00')
        TO '{f.as_posix()}' (FORMAT parquet)""")
    n = c.sql(f"select count(*) from '{f.as_posix()}'").fetchone()[0]
    rows.append(dict(case=f"s{i:02d}", DeviceId=r.DeviceId, n_det_channels=r.n_det_channels, n_events=n))
pd.DataFrame(rows).to_csv(OUT / "signals115.csv", index=False)
print(pd.DataFrame(rows).describe())
