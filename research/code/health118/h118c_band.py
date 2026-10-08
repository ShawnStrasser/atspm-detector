"""Note 118c item 1: one more serious attempt at the user's full-profile idea (type groups + the type's normal 24 h of
counts AND % ON + first differences + an unsupervised outlier rule), aimed at what the night check cannot see
(peaks at odd hours, flat all day).

Per detector-day (s116/day.parquet: hourly counts N, hourly mean fraction ON O; w40 Sun 27 / Mon 28, 760 training
signals, locked_v2 absent - asserted in h116_data):
    counts shape  c_h = sqrt(N_h / day total)          (Hellinger: equal Poisson noise)
    time ON level a_h = arcsin(sqrt(O_h))
    first differences dc_h, da_h
Type group = function x lane span x volume band x DAY (Sun / Mon shapes differ), thin groups merged (>= 150 rows).
Band = the group's per-hour median +- robust SD (1.4826 MAD, floored), fitted OUT OF FOLD (2 folds by signal).
Robust per-hour z; an hour is abnormal at |z| >= ZH; finding = a RUN of >= K consecutive abnormal hours in the same
direction (counts or % ON separately), or >= K abnormal hour-to-hour changes. Phase context: excused when the phase
mates (same predicted phase, >= 200 / day) sit outside the band in the same hours and direction (median z >= ZH / 2)
- then it is the movement's traffic, not the detector.

    python h118c_band.py study   -> %DC_WORK%/s118c/band_grid.csv, band_oof.parquet (best config), band_new.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "health116"))  # h116_study, tod116
import h116_study as S  # noqa: E402

DCW = S.DCW
OUT = DCW / "s118c"
MIN_GROUP = 150
FLOOR_C, FLOOR_A = 0.02, 0.02          # robust-SD floors (sqrt share ~ .02 = 1 % share at 1/24; arcsin .02 = .04 % ON)
MIN_MATE = 200


def feats(N, O):
    tot = np.nansum(N, 1, keepdims=True)
    c = np.sqrt(np.nan_to_num(N) / np.clip(tot, 1, None))
    a = np.arcsin(np.sqrt(np.clip(np.nan_to_num(O), 0, 1)))
    return c, a


def group_keys(D, rows, target=None):
    """fn|span|band|day with fallbacks; groups need >= MIN_GROUP rows among `rows`; keys for `target` (default D)."""
    avail = set()
    tr = D[rows]
    T_ = D if target is None else target
    lv = (("fn", "span", "band", "window"), ("fn", "band", "window"), ("fn", "span", "window"), ("fn", "window"))
    for keys in lv:
        for k, n in tr.groupby(list(keys)).size().items():
            if n >= MIN_GROUP:
                avail.add((keys, k))
    out = np.array([None] * len(T_), object)
    for keys in lv:
        kk = list(zip(*[T_[c] for c in keys]))
        for i, k in enumerate(kk):
            if out[i] is None and (keys, k) in avail:
                out[i] = "|".join(map(str, k)) + f"#{len(keys)}"
    return out


def band_fit(M, floor):
    med = np.median(M, 0)
    sc = 1.4826 * np.median(np.abs(M - med), 0)
    return med, np.maximum(sc, floor)


def oof_z(D, C, A):
    """out-of-fold per-hour z for counts shape, time ON level and their first differences; plus the band (for charts)."""
    n = len(D)
    Z = {k: np.full((n, 24 if k in ("c", "a") else 23), np.nan) for k in ("c", "a", "dc", "da")}
    BND = {k: np.full((n, 3, 24), np.nan) for k in ("c", "a")}
    G = np.array([None] * n, object)
    dc, da = np.diff(C, axis=1), np.diff(A, axis=1)
    for f in (0, 1):
        tr = ((D.fold != f) & D.ok).to_numpy()
        te = ((D.fold == f) & D.ok).to_numpy()
        gk = group_keys(D, tr)
        for g in sorted(set(gk[te & (gk != None)])):                      # noqa: E711
            m_tr = tr & (gk == g)
            m_te = te & (gk == g)
            pois = 1.0 / (4.0 * np.clip(D.tot.to_numpy(float)[m_te], 1, None))      # Poisson var of sqrt(share)
            for nm, M, fl in (("c", C, FLOOR_C), ("a", A, FLOOR_A), ("dc", dc, FLOOR_C), ("da", da, FLOOR_A)):
                med, sc = band_fit(M[m_tr], fl)
                if nm in ("c", "dc"):
                    k = 1.0 if nm == "c" else 2.0                                   # a difference adds two hours
                    sc = np.sqrt(sc[None] ** 2 + k * pois[:, None])
                Z[nm][m_te] = (M[m_te] - med) / sc
                if nm in BND:
                    s3 = sc if sc.ndim == 2 else np.broadcast_to(sc, (m_te.sum(), len(sc)))
                    BND[nm][m_te] = np.stack([med[None] - 4 * s3, np.broadcast_to(med, s3.shape), med[None] + 4 * s3], 1)
            G[m_te] = g
    return Z, BND, G


def runs(Zm, zh):
    """longest run of consecutive hours beyond +zh (or -zh), signed; and the run's hours (first, last)."""
    n, h = Zm.shape
    best = np.zeros(n, int)
    sign = np.zeros(n, int)
    at = np.full((n, 2), -1)
    for s in (1, -1):
        ab = np.nan_to_num(s * Zm, nan=-9) >= zh
        cur = np.zeros(n, int)
        for j in range(h):
            cur = np.where(ab[:, j], cur + 1, 0)
            better = cur > best
            best = np.where(better, cur, best)
            sign = np.where(better, s, sign)
            at[better, 0] = j - cur[better] + 1
            at[better, 1] = j
    return best, sign, at


def mate_excuse(D, Zm, at, sign, zh):
    """median z of the phase mates over the same hours, same direction (mates = same day, signal, predicted phase,
    ok, >= MIN_MATE a day, self excluded)."""
    n = len(D)
    out = np.full(n, np.nan)
    key = (D.DeviceId.astype(str) + "#" + D.window + "#" + D.phase.astype(str)).to_numpy()
    good = (D.ok & (D.tot >= MIN_MATE)).to_numpy() & D.phase.notna().to_numpy()
    idx = np.flatnonzero(at[:, 0] >= 0)
    groups = pd.Series(np.arange(n)).groupby(key).indices
    for i in idx:
        mm = groups[key[i]]
        mm = mm[(mm != i) & good[mm]]
        if not len(mm) or not D.phase.notna().iat[i]:
            continue
        a, b = at[i]
        v = np.nanmedian(sign[i] * Zm[mm, a:b + 1], 1)
        out[i] = np.nanmedian(v)
    return out


def evaluate(D, flag, tag, night_flag):
    ok = D.ok.to_numpy()
    z = np.where(ok, flag.astype(float), np.nan)
    r = S.summarise(D, z, 0.5, tag)
    r = {k: v for k, v in r.items() if not k.endswith("_z")}
    r["rate"] = round(100 * flag[ok].mean(), 2)
    r["new_vs_night"] = round(100 * (flag & ~night_flag)[ok].mean(), 2)
    r["also_night"] = round(float(night_flag[flag].mean()), 3) if flag.any() else None
    return r


def study():
    OUT.mkdir(parents=True, exist_ok=True)
    D, N, O, H, Q = S.load()
    C, A = feats(N, O)
    Z, BND, G = oof_z(D, C, A)
    T = pd.read_parquet(DCW / "s116" / "oof_final.parquet", columns=["DeviceId", "window", "detector", "flag", "level"])
    night = D[["DeviceId", "window", "detector"]].merge(T, how="left").flag.fillna(False).to_numpy(bool)
    rows, keep = [], {}
    for zh in (3.0, 4.0, 5.0):
        R = {nm: runs(Z[nm], zh) for nm in Z}
        EX = {nm: mate_excuse(D, Z[nm], R[nm][2], R[nm][1], zh) for nm in ("c", "a")}
        for K in (2, 3, 4, 6):
            for use_d in (False, True):
                for excuse in (False, True):
                    f = np.zeros(len(D), bool)
                    for nm in ("c", "a"):
                        fi = R[nm][0] >= K
                        if excuse:
                            fi &= ~(np.nan_to_num(EX[nm], nan=-9) >= zh / 2)
                        f |= fi
                    if use_d:
                        for nm in ("dc", "da"):
                            f |= (np.nan_to_num(np.abs(Z[nm]), nan=0) >= zh).sum(1) >= K
                    f &= D.ok.to_numpy()
                    tag = f"zh{zh:g}_K{K}_{'diff' if use_d else 'nodiff'}_{'excuse' if excuse else 'noexc'}"
                    r = evaluate(D, f, tag, night)
                    rows.append(r)
                    keep[tag] = f
                    print({k: r[k] for k in ("cfg", "must_bad", "must_ok", "oth_bad", "oth_ok", "2B069", "rate",
                                             "new_vs_night", "jaccard")}, flush=True)
    G_ = pd.DataFrame(rows)
    G_.to_csv(OUT / "band_grid.csv", index=False)
    # save per-row run statistics for the most promising configs (chosen after reading the grid)
    np.savez_compressed(OUT / "band_z.npz", c=Z["c"], a=Z["a"], dc=Z["dc"], da=Z["da"], bc=BND["c"], ba=BND["a"])
    P = D[["DeviceId", "signal", "window", "detector", "fn", "span", "band", "phase", "tot", "status110"]].copy()
    P["group"] = G
    P["night_flag"] = night
    for zh in (3.0, 4.0, 5.0):
        for nm in ("c", "a"):
            b, s, at = runs(Z[nm], zh)
            P[f"run_{nm}_{zh:g}"], P[f"sgn_{nm}_{zh:g}"] = b, s
            P[f"h0_{nm}_{zh:g}"], P[f"h1_{nm}_{zh:g}"] = at[:, 0], at[:, 1]
            P[f"ex_{nm}_{zh:g}"] = mate_excuse(D, Z[nm], at, s, zh)
        for nm in ("dc", "da"):
            P[f"nd_{nm}_{zh:g}"] = (np.nan_to_num(np.abs(Z[nm]), nan=0) >= zh).sum(1)
    P["ok"] = D.ok.to_numpy()
    P.to_parquet(OUT / "band_oof.parquet")


def alt_run(D, C, alts):
    """model unsure (top function < 70 %): the counts-shape run length against each plausible function's band
    (same out-of-fold fit); returns the SHORTEST run over the plausible functions (= least strict)."""
    n = len(D)
    best = np.full(n, 99)
    tot = D.tot.to_numpy(float)
    for f in FNS:
        m = np.array([f in a for a in alts]) & D.fn.ne(f).to_numpy()
        if not m.any():
            continue
        D2 = D.copy()
        D2.loc[m, "fn"] = f
        Zc = np.full((n, 24), np.nan)
        for fo in (0, 1):
            tr = ((D.fold != fo) & D.ok).to_numpy()
            te = ((D.fold == fo) & D.ok).to_numpy() & m
            gk_tr = group_keys(D, tr)
            gk2 = group_keys(D, tr, target=D2)
            for g in sorted(set(gk2[te & (gk2 != None)])):                 # noqa: E711
                m_tr = tr & (gk_tr == g)
                m_te = te & (gk2 == g)
                med, sc = band_fit(C[m_tr], FLOOR_C)
                sc = np.sqrt(sc[None] ** 2 + 1.0 / (4.0 * np.clip(tot[m_te], 1, None))[:, None])
                Zc[m_te] = (C[m_te] - med) / sc
        b, _, _ = runs(Zc, ZH)
        best = np.where(m & np.isfinite(Zc).all(1), np.minimum(best, b), best)
    return best


FNS = ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")


def mate_z(D, C, floor=FLOOR_C):
    """per-hour z of its counts shape against its phase mates' median shape (>= 2 mates: same day, signal and
    predicted phase, >= MIN_MATE a day); scale = the mates' own spread + Poisson noise of both + floor."""
    n = len(D)
    Zm = np.full((n, 24), np.nan)
    key = (D.DeviceId.astype(str) + "#" + D.window + "#" + D.phase.astype(str)).to_numpy()
    good = (D.ok & (D.tot >= MIN_MATE)).to_numpy() & D.phase.notna().to_numpy()
    tot = D.tot.to_numpy(float)
    for ii in pd.Series(np.arange(n)).groupby(key).indices.values():
        for i in ii:
            mm = ii[(ii != i) & good[ii]]
            if len(mm) < 2 or not D.ok.iat[i] or not D.phase.notna().iat[i]:
                continue
            med = np.median(C[mm], 0)
            sp = 1.4826 * np.median(np.abs(C[mm] - med), 0)
            sc = np.sqrt(sp ** 2 + 1 / (4 * max(tot[i], 1)) + 1 / (4 * np.median(tot[mm])) + floor ** 2)
            Zm[i] = (C[i] - med) / sc
    return Zm


# ---------------------------------------------------------------------------------------------------------- final rule
ZH, K_MIN, K_BAD = 4.0, 3, 7          # |z| per hour, consecutive hours for a finding (score .35 at 3 h, 1 at 7 h)
MATE_EX = ZH / 2                      # phase mates outside the band the same way (median z >= 2): traffic, not the detector
CONG_X, CONG_MIN, CONG_ABS = 1.5, 0.20, 0.50   # quiet run while its own % ON is >= 50 %, or >= 1.5x its 07-19 mean and >= 20 %: queue
MIN_GAP = 30                          # actuations between it and its type's normal over the run
SIG_SHARE, SIG_MIN = 0.20, 5          # signal-wide note: >= 20 % (and >= 5) of its scored detectors with a run


def final(D, N, O, Z, BND, b_alt=None, Zm=None):
    """the adopted (or tested) full-profile rule on counts; per detector-day. b_alt: shortest run over the plausible
    functions when the model is unsure (F5); the finding needs that run too."""
    b, sgn, at = runs(Z["c"], ZH)
    b_own = b.copy()
    if b_alt is not None:
        b = np.minimum(b, b_alt)
    ex = mate_excuse(D, Z["c"], at, sgn, ZH)
    tot = D.tot.to_numpy(float)
    med = np.clip(BND["c"][:, 1], 0, None) ** 2 * tot[:, None]           # type normal, in actuations per hour
    lo = np.clip(BND["c"][:, 0], 0, None) ** 2 * tot[:, None]
    hi = np.clip(BND["c"][:, 2], 0, None) ** 2 * tot[:, None]
    n = len(D)
    obs, exp_, cong = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    day_on = np.nanmean(O[:, 7:19], 1)
    for i in np.flatnonzero(at[:, 0] >= 0):
        a, z = at[i]
        obs[i] = np.nansum(N[i, a:z + 1])
        exp_[i] = np.nansum(med[i, a:z + 1])
        if sgn[i] < 0:
            ro = np.nanmean(O[i, a:z + 1])
            cong[i] = (ro >= CONG_ABS) or (ro >= max(CONG_X * day_on[i], CONG_MIN))
    run_ok = (b >= K_MIN) & D.ok.to_numpy() & (np.abs(obs - exp_) >= MIN_GAP)
    mate = np.nan_to_num(ex, nan=-9) >= MATE_EX
    flag = run_ok & ~mate & ~cong
    # phase layer: the same run rule against its phase mates' own day (no type involved, so no model-unsure rule)
    bm, sm, am = runs(Zm, ZH) if Zm is not None else (np.zeros(n, int), np.zeros(n, int), np.full((n, 2), -1))
    obs_m, exp_m, cong_m = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    key = (D.DeviceId.astype(str) + "#" + D.window + "#" + D.phase.astype(str)).to_numpy()
    good = (D.ok & (D.tot >= MIN_MATE)).to_numpy() & D.phase.notna().to_numpy()
    grp = pd.Series(np.arange(n)).groupby(key).indices
    for i in np.flatnonzero(bm >= K_MIN):
        a, z = am[i]
        mm = grp[key[i]]
        mm = mm[(mm != i) & good[mm]]
        sh = np.median(N[mm] / np.clip(np.nansum(N[mm], 1, keepdims=True), 1, None), 0) * tot[i]
        obs_m[i] = np.nansum(N[i, a:z + 1])
        exp_m[i] = np.nansum(sh[a:z + 1])
        if sm[i] < 0:
            ro = np.nanmean(O[i, a:z + 1])
            cong_m[i] = (ro >= CONG_ABS) or (ro >= max(CONG_X * day_on[i], CONG_MIN))
        else:                     # busier than its mates while THEY were queued (their counts flatten): no evidence
            cong_m[i] = np.nanmedian(np.nanmean(O[mm, a:z + 1], 1)) >= CONG_ABS
    run_ok_m = (bm >= K_MIN) & D.ok.to_numpy() & (np.abs(obs_m - exp_m) >= MIN_GAP)
    flag_m = run_ok_m & ~cong_m
    F = D[["DeviceId", "signal", "window", "detector", "fn", "span", "band", "phase", "tot"]].copy()
    F["run_h"], F["run_h_own"], F["run_dir"], F["run_h0"], F["run_h1"] = b, b_own, sgn, at[:, 0], at[:, 1]
    F["run_obs"], F["run_exp"], F["mate_z"], F["cong"] = obs, exp_, ex, cong
    F["run_ok"], F["mate_excused"], F["flag_type"] = run_ok, run_ok & mate, flag
    F["mrun_h"], F["mrun_dir"], F["mrun_h0"], F["mrun_h1"] = bm, sm, am[:, 0], am[:, 1]
    F["mrun_obs"], F["mrun_exp"], F["mrun_cong"], F["flag_mates"] = obs_m, exp_m, cong_m, flag_m
    F["flag"] = flag | flag_m
    # the evidence shown: the longer of the two runs (type band / phase mates)
    use_m = flag_m & (~flag | (bm > b))
    F["ev_from"] = np.where(use_m, "mates", np.where(flag, "type", ""))
    for c_, a_, m_ in (("ev_h", b, bm), ("ev_dir", sgn, sm), ("ev_h0", at[:, 0], am[:, 0]), ("ev_h1", at[:, 1], am[:, 1]),
                       ("ev_obs", obs, obs_m), ("ev_exp", exp_, exp_m)):
        F[c_] = np.where(use_m, m_, a_)
    # signal-wide: many of a signal's detectors outside their type's band on the same day (before excuses)
    g = F[D.ok.to_numpy()].groupby(["DeviceId", "window"])
    sw = g.run_ok.agg(["sum", "count"])
    sw = sw[(sw["sum"] >= SIG_MIN) & (sw["sum"] >= SIG_SHARE * sw["count"])]
    F["sig_wide"] = pd.MultiIndex.from_frame(F[["DeviceId", "window"]]).isin(sw.index)
    ba = np.sin(np.clip(BND["a"], 0, np.pi / 2)) ** 2 * 100                 # % ON band (arcsin sqrt back to %)
    cols = {f"t{nm}{h}": (lo, med, hi)[j][:, h] for j, nm in enumerate(("lo", "med", "hi")) for h in range(24)}
    cols.update({f"{nm}{h}": ba[:, j, h] for j, nm in enumerate(("olo", "omed", "ohi")) for h in range(24)})
    return pd.concat([F, pd.DataFrame(cols, index=F.index)], axis=1)


def build():
    D, N, O, H, Q = S.load()
    C, A = feats(N, O)
    Z, BND, G = oof_z(D, C, A)
    P = pd.read_parquet(DCW / "s118b" / "resolved_v4.parquet", columns=["DeviceId", "window", "detector"] +
                        [f"p_{c}" for c in FNS])
    P = D[["DeviceId", "window", "detector"]].merge(P, how="left")
    Pm = P[[f"p_{c}" for c in FNS]].to_numpy(float)
    top = np.nanmax(np.where(np.isfinite(Pm), Pm, -1), 1)
    alts = [[c for c, v in zip(FNS, row) if np.isfinite(v) and v >= .15] if 0 <= t < .7 else []
            for t, row in zip(top, Pm)]
    F = final(D, N, O, Z, BND, alt_run(D, C, alts), mate_z(D, C))
    F["alt_fns"] = [",".join(a) for a in alts]
    F["group"] = G
    # note-116 night check with the model-unsure rule: z = the smallest over the plausible functions (production
    # reference for the alternative types; own type keeps its out-of-fold z)
    import tod116 as T6
    T = pd.read_parquet(DCW / "s116" / "oof_final.parquet",
                        columns=["DeviceId", "window", "detector", "z", "dph", "n_night", "level"])
    T = D[["DeviceId", "window", "detector"]].merge(T, how="left")
    ref = T6.load_reference(DCW / "s116" / "tod116_ref.npz")
    z_eff = T.z.to_numpy(float).copy()
    for f in FNS:
        m = np.array([f in a for a in alts]) & D.fn.ne(f).to_numpy()
        for w in ("h24_a", "h24_b"):
            i = np.flatnonzero(m & (D.window == w).to_numpy())
            if not len(i):
                continue
            o = T6.score(N[i], O[i], np.array([f] * len(i)), D.lanes.to_numpy()[i], D.DeviceId.to_numpy()[i],
                         D.phase.to_numpy()[i], ref)
            za = o["z"]
            z_eff[i] = np.where(np.isfinite(za) & np.isfinite(z_eff[i]), np.fmin(z_eff[i], za), z_eff[i])
    flag6, lvl6 = T6.decide(np.where(np.isfinite(z_eff), z_eff, -9.0), T.dph.to_numpy(float), T.n_night.to_numpy(float))
    F["tod_z_c"] = z_eff
    F["tod_level_c"] = np.where(np.isfinite(z_eff), lvl6, "not scored")
    F["tod_level_v4"] = T.level.to_numpy()
    # signal-wide note: >= 20 % (and >= 5) of a signal's scored detectors outside their type's normal day the same
    # day - a >= 3-h run outside the band OR a night level >= 3.5 robust SD over the type - before any phase excuse
    F["ok"] = D.ok.to_numpy()
    F["odd_day"] = F.ok & (F.run_ok | (np.nan_to_num(z_eff, nan=-9) >= T6.Z_LO))
    sw = F[F.ok].groupby(["DeviceId", "window"]).odd_day.agg(["sum", "count"])
    sw = sw[(sw["sum"] >= SIG_MIN) & (sw["sum"] >= SIG_SHARE * sw["count"])]
    F["sig_wide"] = pd.MultiIndex.from_frame(F[["DeviceId", "window"]]).isin(sw.index)
    F["sig_odd_n"] = F.groupby(["DeviceId", "window"]).odd_day.transform("sum")
    F["sig_n"] = F.groupby(["DeviceId", "window"]).ok.transform("sum")
    print("signal-wide notes:", sw.to_string())
    print("night check, model-unsure rule: flags", int((F.tod_level_v4.isin(["suspect", "bad"])).sum()), "->",
          int((F.tod_level_c.isin(["suspect", "bad"])).sum()))
    T = pd.read_parquet(DCW / "s116" / "oof_final.parquet", columns=["DeviceId", "window", "detector", "flag"])
    night = D[["DeviceId", "window", "detector"]].merge(T, how="left").flag.fillna(False).to_numpy(bool)
    F["night_flag"] = night
    F = pd.concat([F, pd.DataFrame({**{f"n{h}": N[:, h] for h in range(24)}, **{f"o{h}": 100 * O[:, h] for h in range(24)}},
                                   index=F.index)], axis=1)
    F["ok"] = D.ok.to_numpy()
    F.to_parquet(OUT / "band_final.parquet")
    r = evaluate(D, F.flag.to_numpy(), "final", night)
    print({k: v for k, v in r.items()})
    ok = F.ok
    print("type-band flags", round(100 * F.flag_type[ok].mean(), 2), "phase-mate flags", round(100 * F.flag_mates[ok].mean(), 2),
          "either", round(100 * F.flag[ok].mean(), 2))
    print("run_ok", round(100 * F.run_ok[ok].mean(), 2), "mate-excused", round(100 * F.mate_excused[ok].mean(), 2),
          "queue-excused", round(100 * (F.run_ok & F.cong & ~F.mate_excused)[ok].mean(), 2),
          "flag", round(100 * F.flag[ok].mean(), 2), "up", round(100 * (F.flag & (F.run_dir > 0))[ok].mean(), 2),
          "sig-wide signal-days", int(F[F.sig_wide].groupby(["DeviceId", "window"]).ngroups),
          "2B069 sig-wide", F[(F.signal == "2B069")].groupby("window").sig_wide.any().to_dict())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        study()
