"""Note 108: TIME-OF-DAY PROFILE check (user idea, 2026-10-06).

Per detector and 24-h sample (Sun 27 / Mon 28 Sep, separately - weekend and weekday profiles differ): the hourly share
of the day's actuations (and of the day's time ON).  Normal = the presumed-healthy detectors of the same TYPE (model
function x model lane span) and VOLUME BAND (log: < 20 / 20-100 / > 100 actuations per hour) on the same day.
Distance = total-variation distance to the type+band median profile (0.5 x sum |share - median share|: the share of
its day's actuations that sit in the 'wrong' hours), and the number of hours outside the healthy p2.5-p97.5 band.
Limit = p99.5 of the healthy distances of the cell (fallback: function x band, then band; >= 100 healthy per cell).
Healthy for this check = no finding outside the shape family (choppy / corr / volume / night_day), >= 50 actuations,
a full day of bins.  3-h variant: the same on the 12 15-min bins of each 3-h sample (shares of the 3 h).

    python h108_profile.py      -> health108/prof.parquet (24 h), prof3.parquet (3 h), limits, overlap printout
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h108_base as B  # noqa: E402

OUT = B.OUT
Q = 0.995
MIN_CELL = 100


def _limits(P, key_cols, dcol, hl):
    """p99.5 of healthy distance per cell with fallbacks; returns a Series aligned to P."""
    lim = pd.Series(np.nan, index=P.index)
    src = pd.Series("", index=P.index)
    for keys in key_cols:
        g = P[hl].groupby(keys)[dcol].agg(["count", lambda s: s.quantile(Q)])
        g.columns = ["cnt", "lim"]
        g = g[g.cnt >= MIN_CELL]
        m = P[keys].merge(g, left_on=keys, right_index=True, how="left")
        fill = lim.isna() & m.lim.notna().to_numpy()
        lim[fill] = m.lim.to_numpy()[fill]
        src[fill] = "+".join(keys)
    return lim, src


def tv(S, med):
    return 0.5 * np.abs(S - med).sum(1)


def build24(X):
    Pr = pd.read_parquet(OUT / "prof_raw.parquet")
    nc = [f"n{h}" for h in range(24)]
    oc = [f"occ{h}" for h in range(24)]
    Pr = Pr.merge(X[X.wg.eq("h24")][["DeviceId", "window", "detector", "fn", "span", "type", "n_on", "hl_shape",
                                       "status", "new_status", "technology"]], on=["DeviceId", "window", "detector"])
    Pr["band"] = B.band_of(Pr.n_on / 24)
    N = Pr[nc].to_numpy(float)
    Oc = Pr[oc].to_numpy(float)
    S = N / np.clip(N.sum(1, keepdims=True), 1, None)
    So = Oc / np.clip(Oc.sum(1, keepdims=True), 1e-9, None)
    for h in range(24):
        Pr[f"s{h}"], Pr[f"o{h}"] = S[:, h], So[:, h]
    Pr["hl_prof"] = Pr.hl_shape & (Pr.n_on >= 50) & (Pr.nbins >= 96)
    Pr["d_cnt"], Pr["d_occ"], Pr["n_out"] = np.nan, np.nan, np.nan
    # reference = median / p2.5 / p97.5 per (type, band, window), fallback (fn, band, window), (band, window)
    sc, oc2 = [f"s{h}" for h in range(24)], [f"o{h}" for h in range(24)]
    ref = {}
    for keys in (["type", "band", "window"], ["fn", "band", "window"], ["band", "window"]):
        for k, g in Pr[Pr.hl_prof].groupby(keys):
            if len(g) >= MIN_CELL:
                ref[(tuple(keys), k)] = (g[sc].median().to_numpy(), g[sc].quantile(.025).to_numpy(),
                                         g[sc].quantile(.975).to_numpy(), g[oc2].median().to_numpy())
    src = []
    MED = np.full((len(Pr), 24), np.nan)
    LO, HI, MO = MED.copy(), MED.copy(), MED.copy()
    for i, r in enumerate(Pr[["type", "fn", "band", "window"]].itertuples(index=False)):
        for keys, k in ((("type", "band", "window"), (r.type, r.band, r.window)),
                        (("fn", "band", "window"), (r.fn, r.band, r.window)), (("band", "window"), (r.band, r.window))):
            if (keys, k) in ref:
                MED[i], LO[i], HI[i], MO[i] = ref[(keys, k)]
                src.append("+".join(keys))
                break
        else:
            src.append("")
    Pr["ref_src"] = src
    Pr["d_cnt"] = tv(S, MED)
    Pr["d_occ"] = tv(So, MO)
    Pr["n_out"] = ((S < LO) | (S > HI)).sum(1)
    Pr.loc[Pr.n_on < 50, ["d_cnt", "d_occ", "n_out"]] = np.nan
    for c in ("d_cnt", "d_occ"):
        hl = Pr.hl_prof & Pr[c].notna()
        Pr[c + "_lim"], Pr[c + "_src"] = _limits(Pr, (["type", "band", "window"], ["fn", "band", "window"],
                                                      ["band", "window"]), c, hl)
    Pr["prof_x"] = np.fmax(Pr.d_cnt / Pr.d_cnt_lim, Pr.d_occ / Pr.d_occ_lim)
    Pr["f_prof"] = Pr.prof_x >= 1
    keep = ["DeviceId", "window", "detector", "fn", "span", "type", "band", "n_on", "technology", "hl_prof", "ref_src",
            "d_cnt", "d_occ", "n_out", "d_cnt_lim", "d_occ_lim", "prof_x", "f_prof"] + sc + oc2 + nc
    Pr[keep].to_parquet(OUT / "prof.parquet")
    # healthy band table for charts
    rows = []
    for (t, b, w), g in Pr[Pr.hl_prof].groupby(["type", "band", "window"]):
        if len(g) < 30:
            continue
        for h in range(24):
            rows.append(dict(type=t, band=b, window=w, h=h, n=len(g), med=g[f"s{h}"].median(),
                             lo=g[f"s{h}"].quantile(.025), hi=g[f"s{h}"].quantile(.975),
                             omed=g[f"o{h}"].median(), cnt_med=g[f"n{h}"].median(),
                             cnt_lo=g[f"n{h}"].quantile(.025), cnt_hi=g[f"n{h}"].quantile(.975)))
    pd.DataFrame(rows).to_parquet(OUT / "prof_band.parquet")
    return Pr


def build3(X):
    Bn = pd.read_parquet(B.DCW / "health96" / "bins.parquet", columns=["DeviceId", "window", "detector", "b", "n", "occ"])
    Bn = Bn[Bn.window.str.startswith("h3")]
    P = Bn.pivot_table(index=["DeviceId", "window", "detector"], columns="b", values=["n", "occ"])
    P.columns = [f"{a}{b}" for a, b in P.columns]
    P = P.reset_index().merge(X[X.wg.eq("h3")][["DeviceId", "window", "detector", "fn", "span", "type", "n_on",
                                                "hl_shape"]], on=["DeviceId", "window", "detector"])
    P["band"] = B.band_of(P.n_on / 3)
    nc, oc = [f"n{b}" for b in range(12)], [f"occ{b}" for b in range(12)]
    N, Oc = P[nc].to_numpy(float), P[oc].to_numpy(float)
    full = np.isfinite(N).all(1)
    S = N / np.clip(np.nansum(N, 1, keepdims=True), 1, None)
    So = Oc / np.clip(np.nansum(Oc, 1, keepdims=True), 1e-9, None)
    P["hl_prof"] = P.hl_shape & (P.n_on >= 50) & full
    MED = np.full_like(S, np.nan)
    MO = MED.copy()
    for keys in (["band", "window"], ["fn", "band", "window"], ["type", "band", "window"]):   # finest last wins
        for k, g in P[P.hl_prof].groupby(keys):
            if len(g) < MIN_CELL:
                continue
            m = (P[keys] == pd.Series(k if isinstance(k, tuple) else (k,), index=keys)).all(1).to_numpy()
            MED[m] = np.nanmedian(S[g.index], 0)
            MO[m] = np.nanmedian(So[g.index], 0)
    P["d_cnt"], P["d_occ"] = tv(S, MED), tv(So, MO)
    P.loc[(P.n_on < 50) | ~full, ["d_cnt", "d_occ"]] = np.nan
    for c in ("d_cnt", "d_occ"):
        P[c + "_lim"], _ = _limits(P, (["type", "band", "window"], ["fn", "band", "window"], ["band", "window"]), c,
                                   P.hl_prof & P[c].notna())
    P["prof_x"] = np.fmax(P.d_cnt / P.d_cnt_lim, P.d_occ / P.d_occ_lim)
    P["f_prof"] = P.prof_x >= 1
    P[["DeviceId", "window", "detector", "type", "band", "n_on", "d_cnt", "d_occ", "d_cnt_lim", "d_occ_lim", "prof_x",
       "f_prof"]].to_parquet(OUT / "prof3.parquet")
    return P


def report(X, Pr, P3):
    pd.set_option("display.width", 250)
    lim = Pr.groupby(["type", "band", "window"]).agg(n=("hl_prof", "sum"), d_cnt_lim=("d_cnt_lim", "first"),
                                                    d_occ_lim=("d_occ_lim", "first"))
    print(lim.round(3).to_string())
    J = X[X.wg.eq("h24")].merge(Pr[["DeviceId", "window", "detector", "f_prof", "prof_x"]], how="left",
                                on=["DeviceId", "window", "detector"])
    J["f_prof"] = J.f_prof.fillna(False).astype(bool)
    has = lambda s, k: s.fillna("").str.split(",").apply(lambda l: k in l)  # noqa: E731
    J["n5"] = has(J.rules, "N5")
    n = len(J)
    print(f"\n24 h: {n} detector-windows; profile fires {100 * J.f_prof.mean():.2f} per 100 "
          f"(counts {100 * (Pr.d_cnt >= Pr.d_cnt_lim).mean():.2f}, time ON {100 * (Pr.d_occ >= Pr.d_occ_lim).mean():.2f})")
    for k, m in (("night_day", J.f_night_day), ("corr", J.f_corr), ("N5 watch", J.n5), ("choppy", J.f_choppy),
                 ("level", J.f_level), ("night_drop", J.f_night_drop), ("any old shape (night_day|corr|N5)",
                                                                       J.f_night_day | J.f_corr | J.n5)):
        print(f"  {k:36s} fires {m.sum():5d} ({100 * m.mean():.2f}/100); also profile {100 * J.f_prof[m].mean():5.1f} %;"
              f" profile fires also flagged by it {100 * m[J.f_prof].mean():5.1f} %")
    old_any = J[[f"f_{k}" for k in B.CHK]].any(axis=1) | J.n5
    print(f"  profile fires with NO old finding: {(J.f_prof & ~old_any).sum()} ({100 * (J.f_prof & ~old_any).mean():.2f}"
          f"/100); old status of profile fires: {J[J.f_prof].status.value_counts().to_dict()}")
    print("  profile fire rate per 100 by type:", (100 * J.groupby("type").f_prof.mean()).round(2).to_dict())
    print("  by band:", (100 * J.groupby(B.band_of(J.rate)).f_prof.mean()).round(2).to_dict())
    print("  by technology (analysis only):", (100 * J.groupby("technology").f_prof.mean()).round(2).to_dict())
    J3 = X[X.wg.eq("h3")].merge(P3[["DeviceId", "window", "detector", "f_prof"]], how="left",
                                on=["DeviceId", "window", "detector"])
    J3["f_prof"] = J3.f_prof.fillna(False).astype(bool)
    print(f"\n3 h: profile fires {100 * J3.f_prof.mean():.2f} per 100; overlap with choppy "
          f"{100 * J3.f_prof[J3.f_choppy].mean():.1f} % of choppy fires; with corr / night_day (never run at 3 h): "
          f"{J3.f_corr.sum()} / {J3.f_night_day.sum()} fires")
    # same detector: 24-h profile flag vs its 3-h profile flag (same day)
    a = J[["DeviceId", "detector", "window", "f_prof"]].assign(day=lambda d: d.window.str[-1])
    b = J3[["DeviceId", "detector", "window", "f_prof"]].assign(day=lambda d: d.window.str[-1])
    m = a.merge(b, on=["DeviceId", "detector", "day"], suffixes=("24", "3"))
    print(f"  of 24-h profile fires, the same day's 3-h sample also fires: {100 * m.f_prof3[m.f_prof24].mean():.1f} %")
    J[["DeviceId", "window", "detector", "f_prof", "prof_x"]].to_parquet(OUT / "prof_join24.parquet")


def main():
    X = pd.read_parquet(OUT / "base.parquet")
    Pr = build24(X)
    P3 = build3(X)
    report(X, Pr, P3)


if __name__ == "__main__":
    main()
