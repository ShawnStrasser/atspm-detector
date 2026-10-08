"""Stacked-detector review, step 1: pair statistics for every pair of detectors on the same timing phase.

    python stacked_pairs.py        -> %DC_WORK%/cabinet/stacked/pairs.parquet

Sept-2026 66-h window (official/stg/cache/det_intervals.parquet), function_labels_v3 signals minus locked_v2
(asserted). Per ordered pair (a, b): counts, onset match (share of a's ONs with a b ON starting within +-1.5 s, and
within +-0.3 s), median lag, chance level, 15-min count correlation, ON-time overlap (0.1-s grid), durations.
Cleansing / review only: nothing here is a model input.
"""
from __future__ import annotations

import itertools
import time

import duckdb
import numpy as np
import pandas as pd

from cab_common import DC_WORK, REPO

OUT = DC_WORK / "cabinet" / "stacked"
IVF = (DC_WORK / "official/stg/cache/det_intervals.parquet").as_posix()
V3 = REPO / "research/labels/function_labels_v3.parquet"
TOL, TIGHT, MIN_N = 1.5, 0.3, 50


def load_labels() -> pd.DataFrame:
    d = pd.read_parquet(V3)
    locked = set(pd.read_csv(DC_WORK / "official/locked_v2.csv").DeviceId.str.lower())
    d["dev"] = d.DeviceId.str.lower()
    d = d[~d.dev.isin(locked)].copy()
    assert not set(d.dev) & locked
    return d


def onset_match(a: np.ndarray, b: np.ndarray):
    """a, b sorted onset times in seconds. Returns share of a matched within TOL / TIGHT, median lag (b - a)."""
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    i = np.searchsorted(b, a)
    lo = b[np.clip(i - 1, 0, len(b) - 1)] - a
    hi = b[np.clip(i, 0, len(b) - 1)] - a
    lag = np.where(np.abs(lo) <= np.abs(hi), lo, hi)
    m = np.abs(lag) <= TOL
    return m.mean(), (np.abs(lag) <= TIGHT).mean(), (np.median(lag[m]) if m.any() else np.nan)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = load_labels()
    lab = lab[lab.phase_target.notna()]
    cn = duckdb.connect()
    cn.execute("SET memory_limit='10GB'")
    cn.execute("SET threads=8")
    cn.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    rows, t0 = [], time.time()
    devs = sorted(lab.dev.unique())
    cn.execute(f"""CREATE TEMP TABLE iv AS SELECT lower(DeviceId) dev, Detector det, t_on, t_off, dur FROM '{IVF}'
                   WHERE lower(DeviceId) IN ({",".join(f"'{x}'" for x in devs)})""")
    for k, dev in enumerate(devs):
        L = lab[lab.dev == dev]
        iv = cn.execute("SELECT det, t_on, t_off, dur FROM iv WHERE dev = ? ORDER BY det, t_on", [dev]).df()
        if iv.empty:
            continue
        base = iv.t_on.min().floor("15min")
        iv["s_on"] = (iv.t_on - base).dt.total_seconds().to_numpy()
        iv["s_off"] = (iv.t_off - base).dt.total_seconds().to_numpy()
        tmax = float(np.nanmax(iv[["s_on", "s_off"]].to_numpy())) + 1
        span_h = tmax / 3600
        nb = int(tmax // 900) + 1
        per = {}
        for det, g in iv.groupby("det"):
            if len(g) < MIN_N:
                continue
            on = g.s_on.to_numpy()
            off = np.where(np.isfinite(g.s_off.to_numpy()), g.s_off.to_numpy(), on)
            cnt = np.bincount((on // 900).astype(int), minlength=nb)
            grid = np.zeros(int(tmax * 10) + 2, np.int32)
            np.add.at(grid, (on * 10).astype(int), 1)
            np.add.at(grid, np.maximum((off * 10).astype(int), (on * 10).astype(int) + 1), -1)
            occ = np.cumsum(grid) > 0
            dur = g.dur.to_numpy()
            per[int(det)] = dict(on=on, cnt=cnt, occ=occ, n=len(on), dmed=float(np.nanmedian(dur)),
                                 pulse=float(np.nanmean(dur <= 0.15)), occ_share=float(occ.mean()))
        for ph, P in L.groupby("phase_target"):
            dets = [int(x) for x in P.detector if int(x) in per]
            for a, b in itertools.combinations(sorted(dets), 2):
                A, B = per[a], per[b]
                ab, ab_t, lag_ab = onset_match(A["on"], B["on"])
                ba, ba_t, _ = onset_match(B["on"], A["on"])
                ca, cb = A["cnt"], B["cnt"]
                act = (ca + cb) > 0
                corr = float(np.corrcoef(ca[act], cb[act])[0, 1]) if act.sum() > 8 and ca[act].std() > 0 and cb[act].std() > 0 else np.nan
                inter = float((A["occ"] & B["occ"]).sum())
                rows.append(dict(dev=dev, DeviceName=L.DeviceName.iloc[0], phase=ph, a=a, b=b, n_a=A["n"], n_b=B["n"],
                                 match_ab=ab, match_ba=ba, tight_ab=ab_t, tight_ba=ba_t, lag_ab=lag_ab,
                                 chance_ab=min(1.0, B["n"] / (span_h * 3600) * 2 * TOL),
                                 chance_ba=min(1.0, A["n"] / (span_h * 3600) * 2 * TOL), corr15=corr,
                                 occ_a=A["occ_share"], occ_b=B["occ_share"],
                                 ovl_a=inter / max(A["occ"].sum(), 1), ovl_b=inter / max(B["occ"].sum(), 1),
                                 dmed_a=A["dmed"], dmed_b=B["dmed"], pulse_a=A["pulse"], pulse_b=B["pulse"], hours=span_h))
        if k % 50 == 0:
            print(k, len(devs), dev, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "pairs.parquet", index=False)
    print("pairs", len(out), "signals", out.dev.nunique())


if __name__ == "__main__":
    main()
