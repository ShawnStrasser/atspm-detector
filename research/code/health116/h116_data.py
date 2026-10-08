"""Note 116 data: per detector-day (Sun 27 = h24_a, Mon 28 = h24_b) hourly and 30-min counts + mean fraction ON from the
note-96 15-min bins (hi-res log, dummy > 64 dropped, de-duplicated), joined to the classifier's OOF outputs (function,
lane span, phase) and the v110 health status (comparison only). Training signals only (locked_v2 asserted absent).

    python h116_data.py   -> %DC_WORK%/s116/day.parquet  (one row per detector-day; arrays as columns n0..n23, o0..o23,
                                                         h0..h47 (30-min counts), q0..q47 (30-min fraction ON))
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s116"
FNS = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]


def main():
    OUT.mkdir(exist_ok=True)
    c = duckdb.connect()
    c.execute("set memory_limit='10GB'; set threads=4")
    src = str(DCW / "health96" / "bins_h24.parquet").replace("\\", "/")
    # hourly and 30-min aggregates in one pass (vectorised in DuckDB, pivot in numpy)
    B = c.sql(f"""select DeviceId, "window", detector, b // 2 as hh, sum(n) as n, avg(occ) as occ, count(n) as nb
                  from '{src}' group by all""").df()
    key = ["DeviceId", "window", "detector"]
    K = B[key].drop_duplicates().reset_index(drop=True)
    K["row"] = np.arange(len(K))
    B = B.merge(K, on=key)
    H = np.full((len(K), 48), np.nan)
    Q = H.copy()
    NB = np.zeros((len(K), 48))
    H[B.row, B.hh] = B.n
    Q[B.row, B.hh] = B.occ
    NB[B.row, B.hh] = B.nb
    H[NB == 0] = np.nan
    Q[NB == 0] = np.nan
    N = H.reshape(len(K), 24, 2).sum(2)                          # NaN only if both halves missing
    N[np.isnan(H.reshape(len(K), 24, 2)).all(2)] = np.nan
    O = np.nanmean(Q.reshape(len(K), 24, 2), 2) if True else None
    D = pd.concat([K[key],
                   pd.DataFrame(N, columns=[f"n{h}" for h in range(24)]),
                   pd.DataFrame(O, columns=[f"o{h}" for h in range(24)]),
                   pd.DataFrame(H, columns=[f"h{h}" for h in range(48)]),
                   pd.DataFrame(Q, columns=[f"q{h}" for h in range(48)])], axis=1)
    D["nbins15"] = NB.sum(1)
    X = pd.read_parquet(DCW / "health108" / "base.parquet",
                        columns=key + ["wg", "fn", "span", "type", "phase", "lanes", "f_conf", "n_on", "technology"])
    X = X[X.wg.eq("h24")].drop(columns="wg")
    R = pd.read_parquet(DCW / "health110" / "resolved110.parquet", columns=key + ["new_status", "rules"])
    R = R.rename(columns={"new_status": "status110", "rules": "rules110"})
    D = D.merge(X, on=key, how="inner").merge(R, on=key, how="left")
    D = D[D.fn.isin(FNS)].reset_index(drop=True)
    locked = pd.read_csv(DCW / "official" / "locked_v2.csv")
    lcol = [c for c in locked.columns if "device" in c.lower() or c.lower() in ("signal", "signalid", "deviceid")]
    print("locked columns:", locked.columns.tolist()[:6], "->", lcol)
    for col in lcol:
        vals = set(locked[col].astype(str).str.lower())
        assert not D.DeviceId.str.lower().isin(vals).any(), "locked signal in data"
    sig = pd.read_csv(DCW / "health110" / "review" / "review_rows110.csv")[["signal", "dev"]].drop_duplicates()
    D = D.merge(sig.rename(columns={"dev": "DeviceId"}), on="DeviceId", how="left")
    D.to_parquet(OUT / "day.parquet")
    print(D.shape, D.window.value_counts().to_dict(), "complete days:", (D.nbins15 == 96).mean().round(3))


if __name__ == "__main__":
    main()
