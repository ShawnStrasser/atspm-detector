"""Note 108b: can the time-of-day profile check run on samples shorter than 24 h?

For every 24-h detector-day of note 108 (prof.parquet: hourly counts n0-n23 and hourly time-ON shares o0-o23; Sun 27 /
Mon 28 separately) a partial sample of H hours (3 h: 8 windows, 6 h: 4, 12 h: 2, non-overlapping, clock-aligned) is
cut out.  The detector's hourly shares are renormalised over the available hours; the reference is the PRESAVED
type + volume-band profile of healthy detectors restricted to the same hours and renormalised (median per hour), the
distance is the same total-variation distance, the limit p99.5 of healthy distances per (type, band, day, hours)
cell (fallback function + band, band; >= 100 healthy).  Volume band from the partial sample's own rate.

Honest estimate: 2-fold split by signal (seed 108) - references and limits fitted on one half, applied to the other.
Recall = share of the 24-h profile flags (prof.parquet f_prof, p99.5) that the partial check flags on the same
detector-day (any / mean over windows); false-flag rate = share of presumed-healthy (hl_prof) detector-windows flagged.
Hi-res log only (counts / time ON), classifier outputs (type); training signals only (locked_v2 absent upstream).

    python h108b_profile_partial.py   -> %DC_WORK%/health108/v3c/partial_profile.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
H8 = DCW / "health108"
Q, MIN_CELL, MIN_ON = 0.995, 100, 50


def band_of(rate):
    return np.where(rate < 20, "low", np.where(rate < 100, "medium", "high"))


def run():
    P = pd.read_parquet(H8 / "prof.parquet")
    nc = [f"n{h}" for h in range(24)]
    oc = [f"o{h}" for h in range(24)]
    rng = np.random.default_rng(108)
    devs = P.DeviceId.unique()
    fold = dict(zip(devs, rng.integers(0, 2, len(devs))))
    P["fold"] = P.DeviceId.map(fold)
    N = P[nc].to_numpy(float)
    Oc = P[oc].to_numpy(float)                       # shares of the day's time ON; renormalised below
    out = []
    for H in (3, 6, 12, 24):
        for h0 in range(0, 24, H):
            hs = list(range(h0, h0 + H))
            n = N[:, hs]
            o = Oc[:, hs]
            tot = n.sum(1)
            S = n / np.clip(tot[:, None], 1, None)
            So = o / np.clip(o.sum(1, keepdims=True), 1e-9, None)
            X = pd.DataFrame({"DeviceId": P.DeviceId, "window": P.window, "detector": P.detector, "fn": P.fn,
                              "type": P.type, "fold": P.fold, "hl": P.hl_prof & (tot >= MIN_ON),
                              "ok": tot >= MIN_ON, "f24": P.f_prof.fillna(False).astype(bool),
                              "band": band_of(tot / H)})
            X["d_cnt"], X["d_occ"] = np.nan, np.nan
            for tf in (0, 1):                         # fit on fold 1 - tf, apply to fold tf
                fit, app = (X.fold != tf).to_numpy(), (X.fold == tf).to_numpy()
                for nm, A in (("cnt", S), ("occ", So)):
                    med = np.full(A.shape, np.nan)
                    for keys in (["band", "window"], ["fn", "band", "window"], ["type", "band", "window"]):
                        g = X[fit & X.hl.to_numpy()].groupby(keys).groups
                        for k, idx in g.items():
                            if len(idx) < MIN_CELL:
                                continue
                            m = app & (X[keys] == pd.Series(k if isinstance(k, tuple) else (k,), index=keys)).all(
                                1).to_numpy()
                            med[m] = np.median(A[idx], 0)
                    d = 0.5 * np.abs(A - med).sum(1)
                    X.loc[app, f"d_{nm}"] = d[app]
                    # limit from the fit half's healthy distances (same reference built on the fit half itself)
                    medf = np.full(A.shape, np.nan)
                    for keys in (["band", "window"], ["fn", "band", "window"], ["type", "band", "window"]):
                        g = X[fit & X.hl.to_numpy()].groupby(keys).groups
                        for k, idx in g.items():
                            if len(idx) < MIN_CELL:
                                continue
                            m = fit & (X[keys] == pd.Series(k if isinstance(k, tuple) else (k,), index=keys)).all(
                                1).to_numpy()
                            medf[m] = np.median(A[idx], 0)
                    df_ = 0.5 * np.abs(A - medf).sum(1)
                    F = X[fit].assign(dd=df_[fit])
                    lim = pd.Series(np.nan, index=X.index)
                    for keys in (["type", "band", "window"], ["fn", "band", "window"], ["band", "window"]):
                        t = F[F.hl].groupby(keys).dd.agg(["count", lambda s: s.quantile(Q)])
                        t.columns = ["c", "l"]
                        t = t[t.c >= MIN_CELL]
                        mm = X[keys].merge(t, left_on=keys, right_index=True, how="left").l.to_numpy()
                        fill = lim.isna().to_numpy() & np.isfinite(mm) & app
                        lim[fill] = mm[fill]
                    X.loc[app, f"lim_{nm}"] = lim[app]
            X["flag"] = X.ok & ((X.d_cnt >= X.lim_cnt) | (X.d_occ >= X.lim_occ))
            X["H"], X["h0"] = H, h0
            out.append(X[["DeviceId", "window", "detector", "H", "h0", "ok", "hl", "f24", "flag"]])
    R = pd.concat(out, ignore_index=True)
    R.to_parquet(H8 / "v3c" / "partial_profile.parquet")
    rows = []
    for H, g in R.groupby("H"):
        per = g.groupby(["DeviceId", "window", "detector"]).agg(f24=("f24", "first"), anyf=("flag", "any"),
                                                               mf=("flag", "mean"), ok=("ok", "mean"))
        f = g[g.ok]
        rows.append(dict(H=H, n_windows=int(g.h0.nunique()), coverage=g.ok.mean(),
                         recall_mean_window=f[f.f24].flag.mean(), recall_any_window=per[per.f24].anyf.mean(),
                         flag_rate=f.flag.mean(), false_flag_healthy=f[f.hl].flag.mean(),
                         share_of_flags_also_24h=f[f.flag].f24.mean()))
        for h0, gg in f.groupby("h0"):
            rows.append(dict(H=H, h0=h0, coverage=g[g.h0 == h0].ok.mean(), recall_mean_window=gg[gg.f24].flag.mean(),
                             flag_rate=gg.flag.mean(), false_flag_healthy=gg[gg.hl].flag.mean(),
                             share_of_flags_also_24h=gg[gg.flag].f24.mean()))
    T = pd.DataFrame(rows)
    T.to_csv(H8 / "v3c" / "partial_profile.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
