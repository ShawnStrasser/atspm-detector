"""Pick 10 NEW non-locked signals (not in 115 / 115v / 115d / bench71), 2 per channel-count quintile; extract 24 h."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import duckdb, numpy as np, pandas as pd
from pathlib import Path
W = Path(DCW); O = W / "s115dv"
c = duckdb.connect(); c.execute("set threads=4; set memory_limit='10GB'")
pool = {d.replace("@stg", "").lower() for d in c.sql(f"select distinct DeviceId from '{(W/'s95/phase/pool.parquet').as_posix()}'").df().DeviceId}
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
used = set(pd.read_csv(W/"s115/parity/signals115.csv").DeviceId.str.lower()) | set(pd.read_csv(W/"s115v/signals_v.csv").DeviceId.str.lower())
b71 = {x.lower() for x in c.sql(f"select distinct DeviceId from read_parquet('{(W/'bench71').as_posix()}/ev_*.parquet')").df().DeviceId}
excl = lk | used | b71
meta = c.sql(f"select * from '{(W/'official/stg/cache/signal_meta.parquet').as_posix()}'").df()
L = meta.DeviceId.str.lower()
m = meta[L.isin(pool) & ~L.isin(excl)]
m = m[(m.t0 <= "2026-09-20 06:00") & (m.t1 >= "2026-09-21 06:00")].copy()
print("eligible", len(m))
m["q"] = pd.qcut(m.n_det_channels.rank(method="first"), 5, labels=False)
rng = np.random.default_rng(20261008)
pick = pd.concat([g.iloc[rng.choice(len(g), 2, replace=False)] for _, g in m.groupby("q")]).sort_values("n_det_channels")
assert not pick.DeviceId.str.lower().isin(excl).any()
rows = []
for i, r in enumerate(pick.itertuples()):
    src = (W / "official/stg/cache/events" / f"DeviceId={r.DeviceId}").as_posix()
    f = O / "ev" / f"n{i:02d}.parquet"
    c.execute(f"""COPY (SELECT '{r.DeviceId}' AS DeviceId, Timestamp, EventId, Parameter
        FROM read_parquet('{src}/*.parquet')
        WHERE Timestamp >= TIMESTAMP '2026-09-20 06:00:00' AND Timestamp < TIMESTAMP '2026-09-21 06:00:00')
        TO '{f.as_posix()}' (FORMAT parquet)""")
    n = c.sql(f"select count(*) from '{f.as_posix()}'").fetchone()[0]
    rows.append(dict(case=f"n{i:02d}", DeviceId=r.DeviceId, n_det_channels=r.n_det_channels, n_events=n))
d = pd.DataFrame(rows); d.to_csv(O / "signals_n.csv", index=False); print(d)
