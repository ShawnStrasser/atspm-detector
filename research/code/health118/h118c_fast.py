"""Note 118c item 4: too-fast actuations without re-triggers, limit floors, model-unsure limits for the 117 checks.

* A fast ON (ON -> ON < 1 s) that is a re-trigger (OFF -> ON < 3 ticks = chatter, its own check) or an ON logged again
  without an OFF (already excluded in 117) never counts as 'too fast'; the random-arrival expectation uses the same
  ONs (vehicle starts minus re-triggers).  2B058 d46 (3 h Mon): 863 fast starts, 388 of them 0.2 s after an OFF.
* Limits (p99.8 of the 117 healthy set, fn x span x length, fallbacks; cells >= 100) get floors so that a limit
  near 0 cannot turn noise into a finding: excess >= 3 Poisson SD, bursts >= 2 (note 115v F6 analogue).
* Model unsure (top function < 70 %): each limit = the least strict over the plausible functions (>= 15 %).
* Volume (busiest 5-min flow per lane) unchanged except the model-unsure rule.

    python h118c_fast.py   -> %DC_WORK%/s118c/fast118c.parquet   (needs s118c/bins117.parquet from h117_events with
                                                                  H117_OUT=%DC_WORK%/s118c)
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s118c"
KEY = ["DeviceId", "window", "detector"]
Q, MIN_CELL, SAT = 0.998, 100, 1800.0
KEYS = (["fn", "span", "wg"], ["span", "wg"], ["wg"])
ZF_FLOOR, SPK_FLOOR = 3.0, 2.0
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]


def fast_stats():
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=4")
    src = (OUT / "bins117.parquet").as_posix()
    return con.execute(f"""
        WITH b AS (SELECT DeviceId, "window", detector, b, xGY, xR, xU,
                          greatest(nG + nY - cGY, 0) AS nGY, greatest(nR - cR, 0) AS nR, greatest(nU - cU, 0) AS nU,
                          sGY, sR, greatest(300 - sGY - sR - sRg, 0) AS sU FROM read_parquet('{src}')),
        r AS (SELECT *, CASE WHEN sGY >= 30 THEN nGY / sGY END AS rg, CASE WHEN sR >= 30 THEN nR / sR END AS rr,
                        CASE WHEN sU >= 30 THEN nU / sU END AS ru FROM b),
        m AS (SELECT *, median(rg) OVER w AS mg, median(rr) OVER w AS mr, median(ru) OVER w AS mu FROM r
              WINDOW w AS (PARTITION BY DeviceId, "window", detector ORDER BY b ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING)),
        e AS (SELECT *, CASE WHEN sGY > 0 THEN nGY * (1 - exp(-least(nGY / sGY, coalesce(mg, nGY / sGY)))) ELSE 0 END AS eg,
                        CASE WHEN sR > 0 THEN nR * (1 - exp(-least(nR / sR, coalesce(mr, nR / sR)))) ELSE 0 END AS er,
                        CASE WHEN sU > 0 THEN nU * (1 - exp(-least(nU / sU, coalesce(mu, nU / sU), 50))) ELSE nU END AS eu
              FROM m)
        SELECT DeviceId, "window", detector, sum(xGY + xR + xU) AS fo_c, sum(eg + er + eu) AS fem_c,
               sum(CASE WHEN (xGY + xR + xU - eg - er - eu) / sqrt(eg + er + eu + 1) >= 4
                         AND xGY + xR + xU >= 5 THEN 1 ELSE 0 END) AS n_spk_c
        FROM e GROUP BY ALL""").df()


def tables(X, col, hl):
    T = []
    for keys in KEYS:
        g = X[hl & X[col].notna()].groupby(keys)[col].agg(["count", lambda s: s.quantile(Q)])
        g.columns = ["cnt", "lim"]
        T.append((keys, g[g.cnt >= MIN_CELL].lim))
    return T


def lookup(T, D):
    lim = pd.Series(np.nan, index=D.index)
    for keys, tab in T:
        m = D[keys].merge(tab.rename("lim"), left_on=keys, right_index=True, how="left").lim.to_numpy()
        fill = lim.isna().to_numpy() & np.isfinite(m)
        lim[fill] = m[fill]
    return lim


def alt_fns(X):
    """plausible functions when the model is unsure (top < .7): classes with probability >= .15 (as v110 F5)."""
    P = X[[f"p_{c}" for c in C7]].to_numpy(float)
    top = np.nanmax(np.where(np.isfinite(P), P, -1), 1)
    out = []
    for t, row in zip(top, P):
        out.append([c for c, v in zip(C7, row) if np.isfinite(v) and v >= .15] if 0 <= t < .7 else [])
    return out


def least_strict(X, T, alts, floor=None):
    lim = lookup(T, X[["fn", "span", "wg"]])
    for f in C7:
        m = np.array([f in al for al in alts]) & X.fn.ne(f).to_numpy()
        if not m.any():
            continue
        la = lookup(T, X.loc[m, ["fn", "span", "wg"]].assign(fn=f))
        lim[m] = np.fmax(lim[m], la)
    if floor is not None:
        lim = np.fmax(lim, floor)
    return lim


def main():
    S7 = pd.read_parquet(DCW / "s117" / "stats117.parquet",
                         columns=KEY + ["fn", "span", "wg", "n_on", "healthy", "zf", "n_spk", "fo_all", "fem_all",
                                        "q5_gy", "q5_all", "lim_q5_gy", "lim_q5_all", "lim_zf", "lim_n_spk",
                                        "new_fast_x", "new_vol_x"])
    A = fast_stats()
    X = S7.merge(A, on=KEY, how="left")
    P = pd.read_parquet(DCW / "s118b" / "resolved_v4.parquet", columns=KEY + [f"p_{c}" for c in C7])
    X = X.merge(P, on=KEY, how="left")
    X["zf_c"] = (X.fo_c - X.fem_c) / np.sqrt(X.fem_c + 1)
    H = X.healthy.fillna(False).astype(bool)
    alts = alt_fns(X)
    X["alt117"] = [",".join(a) for a in alts]
    X["lim_zf_c"] = least_strict(X, tables(X, "zf_c", H), alts, ZF_FLOOR)
    X["lim_spk_c"] = least_strict(X, tables(X, "n_spk_c", H), alts, SPK_FLOOR)
    X["lim_zf_c_own"] = np.fmax(lookup(tables(X, "zf_c", H), X[["fn", "span", "wg"]]), ZF_FLOOR)
    enough = X.n_on >= 50
    x_zf = np.where(np.isfinite(X.zf_c), X.zf_c / X.lim_zf_c, np.nan)
    x_spk = np.where(X.n_spk_c > X.lim_spk_c, X.n_spk_c / X.lim_spk_c, 0.0)
    X["x_zf_c"], X["x_spk_c"] = x_zf, x_spk
    X["fast_x_c"] = np.where(enough, np.fmax(np.nan_to_num(x_zf), x_spk), np.nan)
    # volume: model-unsure -> least strict over the plausible functions (Advance uses all-time flow)
    lg = least_strict(X, tables(X, "q5_gy", H), alts)
    la = least_strict(X, tables(X, "q5_all", H), alts)
    adv = X.fn.eq("Advance")
    X["lim_vol_c"] = np.where(adv, np.fmax(la, SAT), np.fmax(lg, SAT))
    q = np.where(adv, X.q5_all, X.q5_gy)
    X["vol_x_c"] = np.where(np.isfinite(q) & (X.lim_vol_c > 0), q / X.lim_vol_c, np.nan)
    X.drop(columns=[f"p_{c}" for c in C7]).to_parquet(OUT / "fast118c.parquet")
    for wg, g in X.groupby("wg"):
        e = g.n_on >= 50
        print(wg, "fast fires /100 v4 -> 118c:", round(100 * (g.new_fast_x[e] >= 1).mean(), 3), "->",
              round(100 * (g.fast_x_c[e] >= 1).mean(), 3), "| volume:", round(100 * (g.new_vol_x >= 1).mean(), 3), "->",
              round(100 * (g.vol_x_c >= 1).mean(), 3))
    lt = X.groupby(["fn", "span", "wg"])[["lim_zf_c_own"]].first().unstack().round(1)
    print(lt.to_string())


if __name__ == "__main__":
    main()
