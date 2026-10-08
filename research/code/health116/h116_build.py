"""Note 116 build: out-of-fold evaluation of the chosen time-of-day check (tod116, numpy only), sensitivity grid,
final reference file, per-detector scores for the review signals, partial-window test, plot-ready data, clusters.

    python h116_build.py   -> %DC_WORK%/s116/{tod116_ref.npz, oof_final.parquet, grid.csv, scores_review.csv,
                                           partial.csv, plot_ready.parquet, clusters.parquet, final_summary.json}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h116_study as S  # noqa: E402
import tod116 as T  # noqa: E402

OUT = S.OUT
REF_CAP = 1500


def groups_for(D, rows):
    """group key per row from the groups that have >= MIN_GROUP rows among `rows` (pooled days)."""
    avail = set()
    tr = D[rows]
    for keys, f in ((["fn", "span", "band"], lambda k: T.group_key(*k)), (["fn", "band"], lambda k: T.group_key(k[0], "*", k[1])),
                    (["fn", "span"], lambda k: T.group_key(k[0], k[1], "*")), (["fn"], lambda k: T.group_key(k[0], "*", "*"))):
        for k, n in tr.groupby(keys).size().items():
            if n >= T.MIN_GROUP:
                avail.add(f(k if isinstance(k, tuple) else (k,)))
    return avail


def members(D, k, rows):
    fn, span, band = k.split("|")
    m = rows & (D.fn.to_numpy() == fn)
    if span != "*":
        m &= D.span.to_numpy() == span
    if band != "*":
        m &= D.band.to_numpy() == band
    return m


def fit_all(D, X, rows, rng):
    avail = groups_for(D, rows)
    G = {}
    for k in avail:
        m = members(D, k, rows)
        g = T.fit_group(X[m], k.split("|")[0])
        idx = np.flatnonzero(m)
        pick = rng.permutation(idx)[:REF_CAP]
        g["ref_n"] = D.loc[pick, [f"n{h}" for h in range(24)]].to_numpy(float)
        g["ref_o"] = D.loc[pick, [f"o{h}" for h in range(24)]].to_numpy(float)
        G[k] = g
    return G


def oof(D, X):
    """2-fold by signal, pooled days: raw z per row, then per-group calibration on the out-of-fold raw z."""
    rng = np.random.default_rng(116)
    raw = np.full(len(D), np.nan)
    grp = np.array([None] * len(D), object)
    exp_r = raw.copy()
    zc, zo = raw.copy(), raw.copy()
    folds = {}
    for f in (0, 1):
        tr = ((D.fold != f) & D.ok).to_numpy()
        te = ((D.fold == f) & D.ok).to_numpy()
        G = fit_all(D, X, tr, rng)
        folds[f] = G
        gk = np.array([T.resolve_group(a, b, c, G) for a, b, c in zip(D.fn, D.span, D.band)], object)
        for k in G:
            i = np.flatnonzero(te & (gk == k))
            if len(i):
                raw[i], exp_r[i], zc[i], zo[i] = T.raw_z(X[i], G[k])
                grp[i] = k
    cal = pd.DataFrame(dict(grp=grp, raw=raw)).dropna().groupby("grp").raw.agg(
        med="median", mad=lambda s: max(1.4826 * np.median(np.abs(s - np.median(s))), 1e-3))
    med = pd.Series(grp).map(cal["med"]).to_numpy(float)
    mad = pd.Series(grp).map(cal["mad"]).to_numpy(float)
    return grp, (raw - med) / mad, exp_r, zc, zo, cal, folds


def evaluate(D, z, dph, nnight, tag, zlo=T.Z_LO, zhi=T.Z_HI, dd=T.D_PH):
    ok = np.isfinite(z)
    d = np.nan_to_num(dph, nan=-1)
    f = ok & (nnight >= T.MIN_NIGHT) & ((z >= zhi) | ((z >= zlo) & (d >= dd)))
    r = S.summarise(D, np.where(ok, f.astype(float), np.nan), 0.5, tag)
    return r, f


def main():
    D, N, O, H, Q = S.load()
    X, nnight = T.features(N, O)
    grp, z, exp_r, zc, zo, cal, folds = oof(D, X)
    # phase layer per window (detector-days of the same day only)
    dph = np.full(len(D), np.nan)
    mate = dph.copy()
    for w in ("h24_a", "h24_b"):
        i = np.flatnonzero((D.window == w).to_numpy())
        d_, m_ = T.phase_delta(X[i, 0], D.DeviceId.to_numpy()[i], D.phase.to_numpy()[i], D.tot.to_numpy()[i],
                               D.ok.to_numpy()[i] & np.isfinite(z[i]))
        dph[i], mate[i] = d_, m_
    rows = []
    for zlo in (3.0, 3.5, 4.0):
        for zhi in (5.0, 6.0, 8.0):
            for dd in (0.2, 0.3, 0.4):
                r, _ = evaluate(D, z, dph, nnight, f"{zlo}/{zhi}/{dd}", zlo, zhi, dd)
                rows.append(r)
    pd.DataFrame(rows).drop(columns=[c for c in rows[0] if c.endswith("_z")]).to_csv(OUT / "grid.csv", index=False)
    r_fin, flag = evaluate(D, z, dph, nnight, "final")
    _, flag_type_only = evaluate(D, z, np.full(len(D), 9.0), nnight, "type only (no phase layer)", T.Z_LO, T.Z_LO, 0)
    r_type = S.summarise(D, np.where(np.isfinite(z), flag_type_only.astype(float), np.nan), .5, "type layer only z>=3.5")
    r_hi = S.summarise(D, np.where(np.isfinite(z), ((z >= T.Z_HI) & (nnight >= T.MIN_NIGHT)).astype(float), np.nan), .5,
                       "type layer only z>=6")
    lvl = np.where(flag & (z >= T.Z_HI) & (np.nan_to_num(dph, nan=-1) >= T.D_PH), "bad", np.where(flag, "suspect", "ok"))
    lvl = np.where(np.isfinite(z), lvl, "not scored")
    F = D[["DeviceId", "signal", "window", "detector", "fn", "span", "band", "phase", "tot", "status110", "rules110",
           "technology"]].copy()
    F["group"], F["z"], F["z_cnt"], F["z_occ"], F["r"], F["exp_r"] = grp, z, zc, zo, X[:, 0], exp_r
    F["night_ratio"] = X[:, 0] ** 2
    F["mate_r"], F["dph"], F["n_night"], F["flag"], F["level"] = mate, dph, nnight, flag, lvl
    F.to_parquet(OUT / "oof_final.parquet")
    # summary numbers
    okm = np.isfinite(z)
    summ = dict(final=r_fin, type_only_35=r_type, type_only_6=r_hi,
                rate_by_fn={k: round(100 * v, 2) for k, v in F[okm].groupby("fn").flag.mean().items()},
                rate_by_tech_analysis_only={str(k): round(100 * v, 2) for k, v in F[okm].groupby("technology").flag.mean().items()},
                level_counts=F[okm].level.value_counts().to_dict(),
                v110_status_of_flags=F[flag].status110.value_counts().to_dict(),
                scored_share=round(okm.mean(), 3), n_rows=int(len(F)),
                signals_with_ge3_flags_per_day=int((F[flag].groupby(["DeviceId", "window"]).size() >= 3).sum()),
                no_mate_share=round(float(np.isnan(dph[okm]).mean()), 3))
    # final reference: fit on all ok rows (pooled days); calibration from the out-of-fold z
    rng = np.random.default_rng(7)
    G = fit_all(D, X, D.ok.to_numpy(), rng)
    for k, g in G.items():
        if k in cal.index:
            g["cal_med"], g["cal_mad"] = np.array(cal.loc[k, "med"]), np.array(cal.loc[k, "mad"])
        else:   # group only formed on all data: calibrate in-sample
            m = members(D, k, D.ok.to_numpy())
            rz = T.raw_z(X[m], g)[0]
            g["cal_med"] = np.array(np.median(rz))
            g["cal_mad"] = np.array(max(1.4826 * np.median(np.abs(rz - np.median(rz))), 1e-3))
        m = members(D, k, D.ok.to_numpy())
        sh = N[m] / N[m].sum(1, keepdims=True)
        g["chart_share_q"] = np.quantile(sh, [.1, .5, .9], axis=0)          # typical day, for charts only
        g["chart_on_q"] = np.quantile(O[m], [.1, .5, .9], axis=0)
        g["n_fit"] = np.array(m.sum())
    allsh = N[D.ok.to_numpy()] / N[D.ok.to_numpy()].sum(1, keepdims=True)
    meta = dict(version="tod116", data="w40 Sun 27 + Mon 28 Sep 2026, 760 training signals, locked_v2 absent",
                Z_LO=T.Z_LO, Z_HI=T.Z_HI, D_PH=T.D_PH, MIN_NIGHT=T.MIN_NIGHT, MIN_N=T.MIN_N, NIGHT=list(T.NIGHT),
                all_share=np.median(allsh, 0).round(5).tolist())
    T.save_reference(OUT / "tod116_ref.npz", G, meta)
    ref = T.load_reference(OUT / "tod116_ref.npz")
    summ["ref_bytes"] = (OUT / "tod116_ref.npz").stat().st_size
    summ["groups"] = {k: int(G[k]["n_fit"]) for k in sorted(G)}
    # production scorer on all rows (in-sample; checks the code path end to end and the share it flags)
    P = []
    for w in ("h24_a", "h24_b"):
        i = np.flatnonzero((D.window == w).to_numpy())
        o = T.score(N[i], O[i], D.fn.to_numpy()[i], D.lanes.to_numpy()[i], D.DeviceId.to_numpy()[i],
                    D.phase.to_numpy()[i], ref)
        P.append(pd.DataFrame(dict(row=i, z_prod=o["z"], flag_prod=o["flag"], level_prod=o["level"])))
    P = pd.concat(P).sort_values("row")
    F["z_prod"], F["flag_prod"], F["level_prod"] = P.z_prod.to_numpy(), P.flag_prod.to_numpy(), P.level_prod.to_numpy()
    summ["prod_in_sample_rate"] = round(100 * F.flag_prod[np.isfinite(F.z_prod)].mean(), 2)
    summ["prod_vs_oof_flag_agreement"] = round(float((F.flag_prod == F.flag)[okm].mean()), 4)
    F.to_parquet(OUT / "oof_final.parquet")
    rev = F[F.signal.notna()].copy()
    rev[["signal", "window", "detector", "fn", "span", "band", "group", "tot", "n_night", "night_ratio", "exp_r", "r",
         "mate_r", "dph", "z", "z_cnt", "z_occ", "flag", "level", "z_prod", "level_prod", "status110"]].round(3).to_csv(
        OUT / "scores_review.csv", index=False)
    summ["partial"] = partial(D, N, O, F, folds, cal)
    plot_ready(D, N, O, F, G)
    clusters(D, N)
    (OUT / "final_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ[k] for k in summ if k not in ("groups",)}, indent=1, default=str)[:6000])


def partial(D, N, O, F, folds, cal):
    """partial windows scored with the production scorer on the held-out fold (refs = the other fold)."""
    rows = []
    wins = {"00-12": (0, 12), "00-10": (0, 10), "00-08": (0, 8), "00-06": (0, 6), "06-18": (6, 18), "12-24": (12, 24)}
    for nm, (a, b) in wins.items():
        hours = np.zeros(24, bool)
        hours[a:b] = True
        flags, sc = np.zeros(len(D), bool), np.zeros(len(D), bool)
        for f in (0, 1):
            G = {k: dict(g) for k, g in folds[1 - f].items()}           # fitted on the other fold's signals
            meta = dict(Z_LO=T.Z_LO)
            for w in ("h24_a", "h24_b"):
                i = np.flatnonzero(((D.fold == f) & (D.window == w)).to_numpy())
                o = T.score(N[i], O[i], D.fn.to_numpy()[i], D.lanes.to_numpy()[i], D.DeviceId.to_numpy()[i],
                            D.phase.to_numpy()[i], (G, meta), hours=hours)
                flags[i], sc[i] = o["flag"], np.isfinite(o["z"])
        f24 = F.flag.to_numpy()
        r = dict(window=nm, hours=int(b - a), scored=round(sc.mean(), 3),
                 rate=round(100 * flags[sc].mean(), 2) if sc.any() else None,
                 recall_of_24h=round(float(flags[f24 & sc].mean()), 3) if (f24 & sc).any() else None,
                 also_24h=round(float(f24[flags].mean()), 3) if flags.any() else None)
        for nmc, cases in (("must_bad", S.MUST_BAD), ("must_ok", S.MUST_OK)):
            idx = S.case_mask(D, cases)
            r[nmc] = f"{int(flags[idx[idx >= 0]].sum())}/{int(sc[idx[idx >= 0]].sum())}"
        rows.append(r)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "partial.csv", index=False)
    return P.to_dict("records")


def plot_ready(D, N, O, F, G):
    """long table for one chart per flagged / case detector-day: own hourly counts and % ON, its type's typical day
    (median share x its own daily total, 10-90 % band), phase mates' median day scaled to its busiest 4 h."""
    want = F.flag.to_numpy() | F.signal.notna().to_numpy()
    rows = []
    key = D.DeviceId.astype(str) + "#" + D.window + "#" + D.phase.astype(str)
    for i in np.flatnonzero(want & np.isfinite(F.z.to_numpy())):
        k = F.group.iat[i]
        g = G.get(k)
        if g is None:
            continue
        q = g["chart_share_q"] * D.tot.iat[i]
        mates = np.flatnonzero((key == key.iat[i]).to_numpy() & D.ok.to_numpy() & (D.tot.to_numpy() >= T.MIN_MATE))
        mates = mates[mates != i]
        own4 = T._busiest4(N[[i]], np.ones(24, bool))[0]
        if len(mates):
            mn = N[mates] / np.clip(T._busiest4(N[mates], np.ones(24, bool))[:, None], 1, None)
            mate_day = np.median(mn, 0) * own4
            mate_on = np.median(O[mates], 0)
        else:
            mate_day = mate_on = np.full(24, np.nan)
        for h in range(24):
            rows.append((D.DeviceId.iat[i], D.signal.iat[i], D.window.iat[i], int(D.detector.iat[i]), h, N[i, h],
                         100 * O[i, h], q[0, h], q[1, h], q[2, h], 100 * g["chart_on_q"][1, h], mate_day[h],
                         100 * mate_on[h], bool(F.flag.iat[i])))
    P = pd.DataFrame(rows, columns=["DeviceId", "signal", "window", "detector", "hour", "count", "pct_on", "type_lo",
                                    "type_med", "type_hi", "type_pct_on_med", "mates_count", "mates_pct_on", "flag"])
    P.to_parquet(OUT / "plot_ready.parquet")


def clusters(D, N, k=4):
    """the natural day shapes per type (function x lane span): k-means on sqrt hourly shares, numpy."""
    import h116_methods as M
    rows = []
    ok = D.ok.to_numpy()
    for (fn, span), g in D[ok].groupby(["fn", "span"]):
        if len(g) < 200:
            continue
        idx = g.index.to_numpy()
        Z = np.sqrt(N[idx] / N[idx].sum(1, keepdims=True))
        C, lab = M.kmeans(Z, k=k, seed=0)
        for j in range(k):
            for h in range(24):
                rows.append((fn, span, j, int((lab == j).sum()), round((lab == j).mean(), 3), h, float(C[j, h] ** 2)))
    pd.DataFrame(rows, columns=["fn", "span", "cluster", "n", "share", "hour", "hourly_share"]).to_parquet(
        OUT / "clusters.parquet")


if __name__ == "__main__":
    main()
