"""Note 110: health resolver v110 = v108 (p99.8 per-type limits, profile p99.5; h108_resolve) with six fixes for the
check artefacts the note-108b charts exposed.  Same data (w40 windows, 763 training signals, locked_v2 absent -
asserted upstream); hi-res log + classifier outputs only.  Each fix can be switched on alone (before / after per fix).

  F1 erratic counts: reference = >= 2 non-twin phase mates (as before), else a non-twin phase mate of the SAME function,
     else >= 2 same-function detectors elsewhere on the signal; else NO yardstick: not scored, and 'watch (no yardstick
     for erratic counts)' when the old whole-signal comparison would have fired.  Limits re-fitted on the new values.
  F2 count drops: expected 15-min count = its reference's count x the HOURLY relative share that healthy detectors of
     its type + volume band show (measured, same day), not a flat share; each side of the cut >= 2 h (1/4 of a short
     sample), and the detector's OWN counts per 15 min must at least halve after the cut - a detector counting on
     while the signal wakes up at dawn is no drop.  Same LLR > 50 / ratio limits (.15 / .05) as the package.
  F3 stuck on: every continuous ON over the per-type limit counts: total stuck time (suspect at the limit, bad at 60
     min) and number of episodes (suspect above the p99.8 of healthy detectors of the type); a shared episode (>= 3
     others held ON) is dropped as before; the package's 'recovered -> suspect at most' only for a single episode.
  F4 queue (Q1) per episode, no 60-min cap when healthy same-phase mates were QUEUED through it: >= 75 % of its
     15-min periods have a healthy mate above the p95 % ON of healthy detectors of that mate's type (measured).
     Every queue clear (Q1 and Q1b) also needs TRAFFIC: the phase's Advance / Count detectors (else its other
     mates) still count >= half their usual per 15 min in the 15-min periods fully inside the ON (orchestrator, 2026-10-06).
  F5 the model's function probability < .7: every limit is the least strict over the plausible types (classes with
     probability >= .15, same lane span); the row says so.
  F6 profile: a finding from 12 h of sample, 'watch' at 3-12 h (partial-sample version of note 108b, hours restricted
     and renormalised, limit p99.5 per type x band x day x hours), not run below 3 h.

    python h110_resolve.py            -> %DC_WORK%/health110/resolved110.parquet, rates110.csv, fix_rates110.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h108_resolve as R8  # noqa: E402
import h104_resolve as R4  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 300)
DCW = R8.DCW
H8 = DCW / "health108"
OUT = DCW / "health110"
R8.Q, R8.PQ, R8.TAG = 0.998, 0.995, "998p995"
ALL = frozenset({1, 2, 3, 4, 5, 6})
F5_TOP, F5_PLAUS = 0.70, 0.15
F4_COVER, F4_QPCT = 0.75, 0.95
F2_MIN_H, F2_MIN_E, F2_OWN = 2.0, 0.02, 0.5
TRAF_FN, TRAF_X = ("Advance", "Count"), 0.5
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
WIN_H = {"m30": 0.5, "h3": 3.0, "h24": 24.0}
S_WIN = R4.V.S_WIN


# ------------------------------------------------------------------ inputs
def base():
    """v108 inputs exactly as h108_resolve.build (q998, profile q995) + v108 result columns for comparison."""
    X = pd.read_parquet(H8 / "base.parquet")
    X["stuck_x"] = np.fmax(pd.to_numeric(X.ep_dur, errors="coerce"), X.max_on_s)
    X = X.merge(pd.read_parquet(H8 / "ctx108.parquet"), on=["DeviceId", "window", "detector"], how="left")
    P24 = pd.read_parquet(H8 / "prof.parquet", columns=["DeviceId", "window", "detector", "type", "fn", "band",
                                                       "hl_prof", "d_cnt", "d_occ"])
    for c in ("d_cnt", "d_occ"):
        P24[c + "_lim"], _ = R8.cell_limit(P24, c, P24.hl_prof, (["type", "band", "window"], ["fn", "band", "window"],
                                                                 ["band", "window"]), q=R8.PQ)
    P24["prof_x"] = np.fmax(P24.d_cnt / P24.d_cnt_lim, P24.d_occ / P24.d_occ_lim)
    X = X.merge(P24[["DeviceId", "window", "detector", "prof_x", "d_cnt", "d_occ", "d_cnt_lim", "d_occ_lim"]],
                on=["DeviceId", "window", "detector"], how="left")
    ph = X.phase.fillna(-1)
    X["hm"] = X.status.ne("bad").astype(int)
    X["n_hmates"] = X.groupby(["DeviceId", "window", ph]).hm.transform("sum") - X.hm
    X.loc[X.phase.isna(), "n_hmates"] = np.nan
    X = R8.occ_hi(X)
    V8 = pd.read_parquet(H8 / "resolved108_q998p995.parquet", columns=["DeviceId", "window", "detector", "st8", "rules8",
                                                                      "watch8", "left8", "cleared8"])
    X = X.merge(V8.rename(columns={c: c + "_v108" for c in ("st8", "rules8", "watch8", "left8", "cleared8")}),
                on=["DeviceId", "window", "detector"], how="left")
    # F1 statistics
    S = pd.read_parquet(OUT / "shape.parquet")
    X = X.merge(S[["DeviceId", "window", "detector", "chop_old", "chop110", "chop_sig", "ref110", "n_ref110",
                   "ref110_dets", "n_twin_ph", "ref_old"]], on=["DeviceId", "window", "detector"], how="left")
    # F5 function probabilities (the classifier's own output, same inputs as the package run)
    I = pd.read_parquet(DCW / "health4" / "inputs.parquet")
    I = I[(I.period == "stg") & I.wgroup.isin(["m30", "h3", "h24"])]
    I["DeviceId"] = I.DeviceId.str.lower()
    I = I[["DeviceId", "wgroup", "detector"] + [f"p_{c}" for c in C7]].drop_duplicates(["DeviceId", "wgroup",
                                                                                       "detector"])
    X = X.merge(I.rename(columns={"wgroup": "wg"}), on=["DeviceId", "wg", "detector"], how="left")
    return X


# ------------------------------------------------------------------ F2: time-of-day aware count drop
def tod_weights():
    """hourly relative share per type x band x day: median over healthy (level family) 24-h detectors of
    (own count / rest of signal in hour h) / (own / rest over the day).  From the 15-min bins of the event pass."""
    f = OUT / "tod_w.parquet"
    if f.exists():
        return pd.read_parquet(f)
    X = pd.read_parquet(H8 / "base.parquet", columns=["DeviceId", "window", "detector", "type", "band", "hl_level",
                                                     "wg"])
    X = X[X.wg.eq("h24") & X.hl_level]
    b = pd.read_parquet(OUT / "b15.parquet", columns=["DeviceId", "window", "detector", "x", "S", "ok", "hour"])
    b = b[b.window.str.startswith("h24") & b.ok].merge(X, on=["DeviceId", "window", "detector"])
    g = b.groupby(["DeviceId", "window", "detector", "type", "band", "hour"])[["x", "S"]].sum().reset_index()
    tot = g.groupby(["DeviceId", "window", "detector"])[["x", "S"]].transform("sum")
    g["rel"] = ((g.x + 0.5) / (g.S + 1.0)) / ((tot.x + 0.5) / (tot.S + 1.0))
    g = g[tot.x >= 50]
    W = g.groupby(["type", "band", "window", "hour"]).rel.agg(["median", "count"]).reset_index()
    W = W.rename(columns={"median": "w", "count": "n_w"})
    W.to_parquet(f)
    return W


def level110(x, S, ok, w, min_h_bins=8, min_e=F2_MIN_E, own_fall=True):
    """package change point (Poisson LLR, best single cut, covered 15-min bins) of x against g = S x w (w = expected
    hourly relative share of its type).  F2 cut candidates: each side >= min(min_h_bins, 1/4 of the bins) covered bins
    and >= min_e of the expected count, the share falls (after < before) AND the detector's OWN counts per 15 min
    fall too - a reference that ramps up at dawn while the detector counts on is no drop of the detector."""
    m = ok & np.isfinite(S) & np.isfinite(w)
    xs, gs = x[m], (S * w)[m]
    if len(xs) < 8 or xs.sum() < 20 or gs.sum() <= 0:
        return np.nan, 0.0, -1
    cx, cs = np.cumsum(xs), np.cumsum(gs)
    X_, SS = cx[-1], cs[-1]
    k = np.arange(2, len(xs) - 1)
    x1, s1, x2, s2 = cx[k - 1], cs[k - 1], X_ - cx[k - 1], SS - cs[k - 1]
    p0 = X_ / max(SS, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        def ll(xx, ss):
            p = np.where(ss > 0, xx / np.maximum(ss, 1e-9), 0)
            return np.where(xx > 0, xx * np.log(np.maximum(p, 1e-12)), 0) - p * ss
        llr = ll(x1, s1) + ll(x2, s2) - (X_ * np.log(max(p0, 1e-12)) - p0 * SS)
    mh = max(2, min(min_h_bins, len(xs) // 4))
    sup = (k >= mh) & (len(xs) - k >= mh) & (s1 >= min_e * SS) & (s2 >= min_e * SS)
    if own_fall:
        rat = ((x2 + .5) / np.maximum(s2, 1e-9)) / ((x1 + .5) / np.maximum(s1, 1e-9))
        sup &= (rat < 1) & (x2 / (len(xs) - k) <= F2_OWN * x1 / k)
    llr = np.where(sup, llr, -np.inf)
    if not np.isfinite(llr).any():
        return np.nan, 0.0, -1
    j = int(np.nanargmax(llr))
    ratio = float(((x2[j] + .5) / max(s2[j], 1e-9)) / ((x1[j] + .5) / max(s1[j], 1e-9)))
    return ratio, float(llr[j]), int(np.where(m)[0][k[j]])


def level_all(X):
    f = OUT / "level110.parquet"
    if f.exists():
        return pd.read_parquet(f)
    W = tod_weights()
    b = pd.read_parquet(OUT / "b15.parquet")
    T = X[["DeviceId", "window", "detector", "type", "band"]]
    b = b.merge(T, on=["DeviceId", "window", "detector"], how="inner")
    b["day"] = np.where(b.window.isin(["h24_a", "h3_a"]), "h24_a", "h24_b")
    W2 = W.rename(columns={"window": "day"})
    b = b.merge(W2[W2.n_w >= 30][["type", "band", "day", "hour", "w"]], on=["type", "band", "day", "hour"], how="left")
    # fallback: band x day over all types
    Wb = W.groupby(["band", "window", "hour"]).apply(lambda d: np.average(d.w, weights=d.n_w)).rename("wb").reset_index()
    b = b.merge(Wb.rename(columns={"window": "day"}), on=["band", "day", "hour"], how="left")
    b["w"] = b.w.fillna(b.wb).fillna(1.0)
    out = []
    for k, g in b.groupby(["DeviceId", "window", "detector"], sort=False):
        g = g.sort_values("b")
        x, S, ok, w = (g[c].to_numpy(float) for c in ("x", "S", "ok", "w"))
        ok = ok.astype(bool)
        rt, lr, cb = level110(x, S, ok, w)
        ra, la, ca = level110(x, S, ok, np.ones_like(w))               # F2 without the hourly weights (ablation)
        r0, l0, c0 = level110(x, S, ok, np.ones_like(w), min_h_bins=2, min_e=0.0, own_fall=False)     # = the package statistic
        out.append((*k, rt, lr, cb, r0, l0, c0, ra, la, ca))
    L = pd.DataFrame(out, columns=["DeviceId", "window", "detector", "lv_ratio110", "lv_llr110", "lv_b110",
                                   "lv_ratio_chk", "lv_llr_chk", "lv_b_chk", "lv_ratio_nw", "lv_llr_nw", "lv_b_nw"])
    L.to_parquet(f)
    return L


def level_score(ratio, llr):
    v = -np.log(np.maximum(ratio, 1e-9))
    s = R8.score(v, -np.log(0.15), -np.log(0.05))
    return np.where(np.isfinite(ratio), np.where(llr <= 50, 0.0, s), np.nan)


# ------------------------------------------------------------------ F3 / F4: every stuck episode, per-episode queue test
def episodes(X):
    """every continuous ON >= 300 s with its queue context (r1_context per episode, + F4 queued-mate coverage)."""
    f = OUT / "eps_ctx.parquet"
    if f.exists():
        return pd.read_parquet(f)
    E = pd.read_parquet(OUT / "eps.parquet")
    E = E[E.dur_s >= 300].copy()
    # shared episode = >= 3 other detectors held ON >= 5 min through at least half of it (package CO_STUCK_N)
    co = []
    for k, g in E.groupby(["DeviceId", "window"], sort=False):
        s, f_, c = (g.t0.astype("int64").to_numpy() / 1e9, g.t1.astype("int64").to_numpy() / 1e9,
                    g.detector.to_numpy())
        ov = np.minimum(f_[:, None], f_[None]) - np.maximum(s[:, None], s[None])
        m = (ov >= 0.5 * (f_ - s)[:, None]) & (c[:, None] != c[None])
        co.append(pd.Series([len(set(c[row].tolist())) for row in m], index=g.index))
    E["co5"] = pd.concat(co)
    Bn = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet")
    Bn = Bn.merge(X[["DeviceId", "window", "detector", "type"]], on=["DeviceId", "window", "detector"], how="left")
    # F4 measured pattern: p95 of 15-min % ON of healthy detectors per type (24-h and 3-h windows, 15-min bins)
    q = Bn[Bn.hl & Bn.bs.eq(900) & Bn.occ.notna()].groupby("type").occ.quantile(F4_QPCT).rename("q_type")
    q.to_frame().to_csv(OUT / "queue_q95_by_type.csv")
    Bn = Bn.merge(q, left_on="type", right_index=True, how="left")
    G = {k: g for k, g in Bn.groupby(["DeviceId", "window"])}
    ph = X.set_index(["DeviceId", "window", "detector"]).phase
    out = []
    for r in E.itertuples():
        b = G.get((r.DeviceId, r.window))
        p = ph.get((r.DeviceId, r.window, int(r.detector)), np.nan)
        res = dict(i=r.Index)
        if b is None:
            out.append(res)
            continue
        bs = int(b.bs.iloc[0])
        t0 = pd.Timestamp(S_WIN[r.window][0])
        ia, iz = int((r.t0 - t0).total_seconds() // bs), int(np.ceil((r.t1 - t0).total_seconds() / bs))
        own = b[b.detector == r.detector].set_index("b").sort_index()
        if not len(own):
            out.append(res)
            continue
        inep = own.index.to_series().between(ia, iz - 1).to_numpy()
        o, ref = own.occ.to_numpy(float), own.ref.to_numpy(float)
        pp = b[(b.ph == p) & (b.detector != r.detector) & b.status.ne("bad")] if np.isfinite(p) else b.iloc[:0]
        keep = []
        for k, g in pp.groupby("detector"):
            ge = g[g.b.between(ia, iz - 1)]
            if len(ge) and (ge.occ >= .99).mean() >= .8:
                continue
            keep.append(k)
        pp = pp[pp.detector.isin(keep)]
        phx_h, corr_h, cover = np.nan, np.nan, np.nan
        if len(pp):
            pm = pp.groupby("b").occ.mean().reindex(own.index).to_numpy(float)
            phx_h = np.nanmean(pm[inep]) / max(np.nanmean(pm), 1e-9) if inep.any() else np.nan
            corr_h = R4._corr(o[~inep], pm[~inep])
            qd = pp[pp.b.between(ia, iz - 1)]
            qb = qd[qd.occ >= qd.q_type].b.unique()
            nb_ = int(inep.sum())
            cover = len(set(qb.tolist()) & set(own.index[inep].tolist())) / max(nb_, 1) if nb_ else np.nan
        refx = np.nanmean(ref[inep]) / max(np.nanmean(ref), 1e-9) if inep.any() else np.nan
        rm = np.nanmean(ref)
        near = own.index.to_series().between(ia - 1, iz).to_numpy()
        po = own.peer_occ.to_numpy(float)
        quiet = ~(po > np.nanmean(po)) if np.isfinite(po).any() else np.ones(len(o), bool)
        light = ~near & (o >= R4.R1_FULL) & (ref <= R4.R1_LIGHT * rm) & quiet
        res.update(n_hpeer_e=len(keep), phx_h_e=phx_h, corr_h_e=corr_h, refx_e=refx, light_e=int(light.sum()),
                   cover_e=cover, mates_e="+".join(str(int(k)) for k in keep))
        out.append(res)
    C = pd.DataFrame(out).set_index("i")
    E = E.join(C)
    E.to_parquet(f)
    return E


def traffic_evidence(X, E):
    """Orchestrator decision 2026-10-06: a queue needs traffic.  Per episode: the phase's Advance / Count detectors
    (self excluded; else every other phase mate) - their 15-min counts during the ON / their mean over the sample
    (trafx_e; a queue needs >= 0.5).  A whole phase held ON with zero counts is no congestion."""
    f = OUT / "eps_traf.parquet"
    if f.exists():
        return E.join(pd.read_parquet(f))
    Bn = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet", columns=["DeviceId", "window", "detector", "b", "n", "bs",
                                                                         "ph", "fn"])
    G = {k: g for k, g in Bn.groupby(["DeviceId", "window"])}
    ph = X.set_index(["DeviceId", "window", "detector"]).phase
    out = []
    for r in E.itertuples():
        b = G.get((r.DeviceId, r.window))
        p = ph.get((r.DeviceId, r.window, int(r.detector)), np.nan)
        if b is None or not np.isfinite(p):
            out.append((r.Index, np.nan, ""))
            continue
        bs = int(b.bs.iloc[0])
        t0 = pd.Timestamp(S_WIN[r.window][0])
        ia, iz = int((r.t0 - t0).total_seconds() // bs), int(np.ceil((r.t1 - t0).total_seconds() / bs))
        m = b[(b.ph == p) & (b.detector != r.detector)]
        tr = m[m.fn.isin(TRAF_FN)]
        if not len(tr):
            tr = m
        if not len(tr):
            out.append((r.Index, np.nan, ""))
            continue
        tot = tr.groupby("b").n.sum()
        fa = int(np.ceil((r.t0 - t0).total_seconds() / bs))      # 15-min periods fully inside the ON
        fz = int((r.t1 - t0).total_seconds() // bs)
        if fz > fa:
            ia, iz = fa, fz
        ins = tot.reindex(range(ia, iz)).dropna()
        x = ins.mean() / max(tot.mean(), 1e-9) if len(ins) else np.nan
        out.append((r.Index, x, "+".join(str(int(k)) for k in sorted(tr.detector.unique()))))
    T = pd.DataFrame(out, columns=["i", "trafx_e", "traf_dets"]).set_index("i")
    T.to_parquet(f)
    return E.join(T)


def stuck_stats(X, E, fixes, lim):
    """per detector-window: episodes over the limit, total stuck time and count (uncleared, not shared), Q1 per
    episode.  lim = per-row stuck limit (s).  Returns DataFrame indexed like X."""
    E = E.merge(pd.DataFrame({"DeviceId": X.DeviceId, "window": X.window, "detector": X.detector, "L": lim,
                              "ri": np.arange(len(X))}), on=["DeviceId", "window", "detector"], how="inner")
    E = E[E.dur_s >= E.L].copy()
    q_ok = (E.n_hpeer_e.fillna(0) > 0) & (E.phx_h_e >= 1.5) & (E.corr_h_e >= R8.Q1_CORR) & (E.refx_e >= 0.5) & \
        (E.light_e.fillna(1) == 0) & (E.trafx_e >= TRAF_X)
    E["q1"] = q_ok & (E.dur_s < 3600)
    E["q1b"] = q_ok & (E.dur_s >= 3600) & (E.cover_e >= F4_COVER) if 4 in fixes else False
    E["shared"] = E.co5 >= 3
    E["count"] = ~E.q1 & ~E.q1b & ~E.shared
    g = E.groupby("ri")
    S = pd.DataFrame({"n_over": g.size(), "n_ep": g["count"].sum(),
                      "tot_s": g.apply(lambda d: d.dur_s[d["count"]].sum()),
                      "n_q1": g.q1.sum(), "n_q1b": g.q1b.sum(), "n_shared": g.shared.sum(),
                      "n_shared_unq": g.apply(lambda d: (d.shared & ~d.q1 & ~d.q1b).sum())})
    return S.reindex(np.arange(len(X)))


# ------------------------------------------------------------------ F6: profile on partial samples (3-h windows)
def prof_partial():
    f = OUT / "prof3.parquet"
    if f.exists():
        return pd.read_parquet(f)
    P = pd.read_parquet(H8 / "prof.parquet")
    out = []
    for w3, (w24, h0) in {"h3_a": ("h24_a", 12), "h3_b": ("h24_b", 6)}.items():
        p = P[P.window == w24]
        hs = list(range(h0, h0 + 3))
        n = p[[f"n{h}" for h in hs]].to_numpy(float)
        o = p[[f"o{h}" for h in hs]].to_numpy(float)
        tot = n.sum(1)
        A = {"cnt": n / np.clip(tot[:, None], 1, None), "occ": o / np.clip(o.sum(1, keepdims=True), 1e-9, None)}
        Y = pd.DataFrame({"DeviceId": p.DeviceId.to_numpy(), "window": w3, "detector": p.detector.to_numpy(),
                          "fn": p.fn.to_numpy(), "type": p.type.to_numpy(),
                          "band": np.where(tot / 3 < 20, "low", np.where(tot / 3 < 100, "medium", "high")),
                          "ok": tot >= 50})
        Y["hl"] = p.hl_prof.to_numpy() & Y.ok
        for nm, M in A.items():
            med = np.full(M.shape, np.nan)
            for keys in (["band"], ["fn", "band"], ["type", "band"]):
                for kk, idx in Y[Y.hl].groupby(keys).groups.items():
                    if len(idx) < R8.MIN_CELL:
                        continue
                    kk = kk if isinstance(kk, tuple) else (kk,)
                    m = (Y[keys] == pd.Series(kk, index=keys)).all(1).to_numpy()
                    med[m] = np.median(M[Y.index.get_indexer(idx)], 0)
            Y[f"d3_{nm}"] = 0.5 * np.abs(M - med).sum(1)
            Y[f"d3_{nm}_lim"], _ = R8.cell_limit(Y.assign(x=Y[f"d3_{nm}"]), "x", Y.hl, (["type", "band"], ["fn", "band"],
                                                                                         ["band"]), q=R8.PQ)
        Y["prof3_x"] = np.where(Y.ok, np.fmax(Y.d3_cnt / Y.d3_cnt_lim, Y.d3_occ / Y.d3_occ_lim), np.nan)
        out.append(Y)
    Y = pd.concat(out, ignore_index=True)
    Y.to_parquet(f)
    return Y


# ------------------------------------------------------------------ F5: limits over plausible types
LIMCOLS = ("chop15", "chat_frac", "ioi_lt05", "ioi_lt1", "burst_frac", "max5", "stuck_x", "n3_exc", "rep_frac")


def cell_tables(X, cols):
    """per KEYS level, the p-quantile limit table of each column (presumed healthy, leave-own-check-out)."""
    n50 = X.n_on >= 50
    fam = {"chop15": ("shape", None), "chat_frac": ("fast", n50), "ioi_lt05": ("fast", n50), "ioi_lt1": ("fast", n50),
           "burst_frac": ("fast", n50), "max5": ("shape", None), "stuck_x": ("stuck", None),
           "n3_exc": ("occ", X.n3_n >= 2), "rep_frac": ("fast", n50), "n_ep": ("stuck", None)}
    T = {}
    for col in cols:
        f, extra = fam[col]
        hl = X[f"hl_{f}"] & (extra if extra is not None else True) & X[col].notna()
        T[col] = []
        for keys in R8.KEYS:
            g = X[hl].groupby(keys)[col].agg(["count", lambda s: s.quantile(R8.Q)])
            g.columns = ["cnt", "lim"]
            T[col].append((keys, g[g.cnt >= R8.MIN_CELL].lim))
    return T


def lookup(T, col, D):
    """limit for rows of D (columns fn, span, cmode, wg) with the KEYS fallback."""
    lim = pd.Series(np.nan, index=D.index)
    for keys, tab in T[col]:
        m = D[keys].merge(tab.rename("lim"), left_on=keys, right_index=True, how="left").lim.to_numpy()
        fill = lim.isna().to_numpy() & np.isfinite(m)
        lim[fill] = m[fill]
    return lim


def limits110(X, fixes):
    X = X.copy()
    cols = LIMCOLS
    T = cell_tables(X, cols)
    base_ = X[["fn", "span", "cmode", "wg"]]
    for col in cols:
        X[f"lim_{col}"] = lookup(T, col, base_)
    pc = X[[f"p_{c}" for c in C7]].to_numpy(float)
    top = np.nanmax(np.where(np.isfinite(pc), pc, -1), 1)
    X["f_top"] = np.where(top >= 0, top, np.nan)
    X["alt_fns"] = ""
    if 5 in fixes:
        low = X.f_top < F5_TOP
        X.loc[low, "alt_fns"] = [",".join(c for c, v in zip(C7, row) if np.isfinite(v) and v >= F5_PLAUS)
                                 for row in pc[low.to_numpy()]]
        for fn_alt in C7:
            m = low & X.alt_fns.str.split(",").apply(lambda l, f=fn_alt: f in l) & X.fn.ne(fn_alt)
            if not m.any():
                continue
            D = base_[m].copy()
            D["fn"] = fn_alt
            D["cmode"] = np.where(fn_alt != "Count", "", np.where(X.loc[m, "dur_p50"].isna(), "unknown",
                                                                   np.where(X.loc[m, "dur_p50"] <= 0.25, "pulse", "normal")))
            for col in cols:
                la = lookup(T, col, D)
                cur = X.loc[m, f"lim_{col}"]
                looser = la > cur
                X.loc[m & looser.reindex(X.index, fill_value=False), f"lim_{col}"] = la[looser]
                X.loc[m & looser.reindex(X.index, fill_value=False), f"lsrc_{col}"] = fn_alt
    return X


# ------------------------------------------------------------------ main
def prepare():
    X = base()
    X["chop15_v108"] = X.chop15
    L = level_all(X)
    X = X.merge(L, on=["DeviceId", "window", "detector"], how="left")
    E = traffic_evidence(X, episodes(X))
    P3 = prof_partial()
    X = X.merge(P3[["DeviceId", "window", "detector", "prof3_x", "d3_cnt", "d3_occ", "d3_cnt_lim", "d3_occ_lim"]],
                on=["DeviceId", "window", "detector"], how="left")
    return X, E


def prof_alt_limits(X):
    """F5 for the profile: least strict d_cnt / d_occ limit over the plausible types (same band, same day)."""
    P = pd.read_parquet(H8 / "prof.parquet", columns=["DeviceId", "window", "detector", "type", "fn", "band", "hl_prof",
                                                     "d_cnt", "d_occ"])
    tabs = {}
    for c in ("d_cnt", "d_occ"):
        g = P[P.hl_prof].groupby(["type", "band", "window"])[c].agg(["count", lambda s: s.quantile(R8.PQ)])
        g.columns = ["cnt", "lim"]
        tabs[c] = g[g.cnt >= R8.MIN_CELL].lim
    return tabs


def run(X0, E, fixes):
    X = X0.copy()
    if 1 in fixes:
        X["chop15"] = X.chop110
    else:
        X["chop15"] = X.chop15_v108
    X = limits110(X, fixes)
    if 3 in fixes:
        # number of episodes over the per-type limit: needs a first pass for its own limit
        sl = X.lim_stuck_x.clip(*R8.STUCK_CLIP)
        S0 = stuck_stats(X, E, fixes, sl.to_numpy())
        X["n_ep"] = S0.n_ep.fillna(0).to_numpy()
        T = cell_tables(X, ("n_ep",))
        X["lim_n_ep"] = lookup(T, "n_ep", X[["fn", "span", "cmode", "wg"]])
        if 5 in fixes:
            low = X.alt_fns.ne("")
            for fn_alt in C7:
                m = low & X.alt_fns.str.split(",").apply(lambda l, f=fn_alt: f in l) & X.fn.ne(fn_alt)
                if m.any():
                    D = X.loc[m, ["fn", "span", "cmode", "wg"]].assign(fn=fn_alt)
                    la = lookup(T, "n_ep", D)
                    X.loc[m, "lim_n_ep"] = np.fmax(X.loc[m, "lim_n_ep"], la)
    if 5 in fixes:
        tabs = prof_alt_limits(X)
        low = X.alt_fns.ne("") & X.wg.eq("h24")
        for fn_alt in C7:
            m = low & X.alt_fns.str.split(",").apply(lambda l, f=fn_alt: f in l) & X.fn.ne(fn_alt)
            if not m.any():
                continue
            key = pd.DataFrame({"type": fn_alt + " " + X.loc[m, "span"], "band": X.loc[m, "band"],
                                "window": X.loc[m, "window"]})
            for c in ("d_cnt", "d_occ"):
                la = key.merge(tabs[c].rename("l"), left_on=["type", "band", "window"], right_index=True,
                               how="left").l.to_numpy()
                cur = X.loc[m, f"{c}_lim"].to_numpy()
                X.loc[m, f"{c}_lim"] = np.fmax(cur, la)
        X["prof_x"] = np.fmax(X.d_cnt / X.d_cnt_lim, X.d_occ / X.d_occ_lim)
    return resolve(X, E, fixes)


def scores110(X, E, fixes):
    S = R8.scores(X)                                   # v108 scores with the (possibly re-fitted) limits
    sl = X.lim_stuck_x.clip(*R8.STUCK_CLIP)
    X["stuck_lim8"] = sl
    if 1 in fixes:
        X["noyard_chop"] = X.ref110.eq("none") & (X.chop_sig >= X.lim_chop15)
    else:
        X["noyard_chop"] = False
    if 2 in fixes:
        S["level"] = level_score(X.lv_ratio110.to_numpy(float), X.lv_llr110.to_numpy(float))
    # stuck: v108 = longest ON; F3 = all episodes; F4 = per-episode Q1 incl. > 60 min
    X["q1_all"] = False
    if 3 in fixes or 4 in fixes:
        St = stuck_stats(X, E, fixes, sl.to_numpy())
        for c in St.columns:
            X[f"st_{c}"] = St[c].to_numpy()
        if 3 in fixes:
            tot = X.st_tot_s.fillna(0).to_numpy(float)
            s_tot = R8.score(tot, sl, np.fmax(R8.STUCK_BAD, 2 * sl))
            s_n = R8.score(X.st_n_ep.fillna(0).to_numpy(float), X.lim_n_ep + 1, 2 * (X.lim_n_ep + 1))
            s_ = np.fmax(s_tot, np.nan_to_num(s_n))
            single = X.st_n_ep.fillna(0) <= 1
            cap = (pd.to_numeric(X.s_stuck, errors="coerce") == 0.35) & single
            s_ = np.where(cap & (s_ > .35), .35, s_)
            # a shared ON (>= 3 others held ON) that is not itself a queue keeps the detector at suspect at most
            X["q1_all"] = (X.st_n_over.fillna(0) > 0) & (X.st_n_ep.fillna(0) == 0) & \
                (X.st_n_shared_unq.fillna(0) == 0) & (X.st_n_q1.fillna(0) + X.st_n_q1b.fillna(0) > 0)
            shared_only = (X.st_n_ep.fillna(0) == 0) & (X.st_n_shared_unq.fillna(0) > 0)
            s_ = np.where(shared_only, np.where(S.stuck >= .35, .35, S.stuck), s_)
            s_ = np.where(X.q1_all, 0.0, s_)       # every episode over the limit explained by a queue
            S["stuck"] = s_
        else:
            # F4 alone: the v108 longest-ON score, cleared when its (longest) episode passes Q1b
            X["q1b_long"] = (X.st_n_q1b.fillna(0) > 0) & (X.stuck_x >= 3600)
    if 6 in fixes:
        hrs = X.wg.map(WIN_H)
        S["prof"] = np.where(hrs >= 12, R8.score(X.prof_x, 1.0, 2.0), np.nan)
        X["prof3_watch"] = (hrs >= 3) & (hrs < 12) & (X.prof3_x >= 1.0)
    else:
        X["prof3_watch"] = False
    return S


def resolve(X, E, fixes):
    S = scores110(X, E, fixes)
    for k in S.columns:
        X[f"s8_{k}"] = S[k]
    X["queue_pat8"] = (X.elhi_occ > 0) & (X.elhi_occ > X.elhi_cnt)
    rows = []
    q1b_long = X.get("q1b_long", pd.Series(False, index=X.index)).to_numpy()
    for i, r in enumerate(X.itertuples(index=False)):
        s = {k: S.iat[i, j] for j, k in enumerate(S.columns) if np.isfinite(S.iat[i, j])}
        find = {k: v for k, v in s.items() if v >= .35}
        notes, watch, fired = [], [], []
        if r.status2 != r.status:
            fired.append("R9")
        for k in list(find):
            rule = None
            if k == "stuck":
                if 3 in fixes:
                    rule = None                      # Q1 / Q1b already applied per episode
                else:
                    lf = r.light_full == 0 if np.isfinite(r.light_full) else False
                    if r.n_hpeer and r.n_hpeer > 0:
                        queue = np.isfinite(r.ep_phx_h) and r.ep_phx_h >= 1.5
                        corr_ok = np.isfinite(r.c_occ_hpeer) and r.c_occ_hpeer >= R8.Q1_CORR
                    else:
                        queue, corr_ok = False, False
                    ed = pd.to_numeric(r.ep_dur, errors="coerce")
                    if np.isfinite(ed) and ed < 3600 and np.isfinite(r.ep_refx) and r.ep_refx >= 0.5 and queue and \
                            corr_ok and lf:
                        rule = ("Q1", "clear")
                    elif q1b_long[i]:
                        rule = ("Q1b", "clear")
            elif k in ("choppy", "level", "occspk") and bool(r.queue_pat8) and np.isfinite(r.c_occ_like) and \
                    r.c_occ_like >= R8.Q2_CORR:
                rule = ("Q2", "clear")
            elif k == "rapid" and r.n_on < R8.LOW_N and np.isfinite(r.share_x) and r.share_x >= 0.1:
                rule = ("R5", "watch")
            elif k == "dropout" and r.hours < 1:
                rule = ("R6", "watch")
            if rule:
                fired.append(rule[0])
                (watch if rule[1] == "watch" else notes).append(k)
                del find[k]
        if 3 in fixes and bool(r.q1_all):
            fired.append("Q1b" if (r.st_n_q1b or 0) > 0 else "Q1")
            notes.append("stuck")
        if "R9" in fired and np.isfinite(r.n_hmates) and r.n_hmates == 0 and r.status in ("suspect", "bad"):
            fired.append("Y")
            watch.append("no_yardstick")
        if bool(r.noyard_chop):
            fired.append("Y1")
            watch.append("no_yardstick_chop")
        if bool(r.prof3_watch):
            fired.append("P3")
            watch.append("prof_partial")
        if len(find) == 1:
            k, v = next(iter(find.items()))
            if k in R8.STAT and v < R8.BORDER:
                fired.append("R7")
                watch.append(k)
                del find[k]
        if "last 0.2 s or less" in str(r.reason) and r.fn == "Presence":
            fired.append("R8")
            watch.append("short_on")
        if bool(r.occ_hi8) and "occspk" not in find and "stuck" not in find and \
                not (np.isfinite(r.c_occ_like) and r.c_occ_like >= .5):
            fired.append("N1")
            watch.append("occ_hi")
        dq = []
        on_s = r.occ * r.hours * 3600 if np.isfinite(r.occ) else np.nan
        if np.isfinite(r.rep_time_s) and r.rep_time_s >= R8.D1_MIN_S and np.isfinite(on_s) and \
                r.rep_time_s >= R8.D1_SHARE * on_s:
            dq.append("D1")
        if bool(r.n3_dq):
            dq.append("D1n3")
        hs = float(np.prod([1 - v for k, v in s.items() if k in find or v < .35])) if s else np.nan
        if np.isfinite(hs) and hs < .25:
            st = "bad"
        elif np.isfinite(hs) and hs < .70:
            st = "suspect"
        elif watch:
            st = "watch"
        elif r.n_on < 20:
            st = "not_enough_data"
        else:
            st = "ok"
        fams = {R8.FAMILY[k] for k, v in find.items() if not (k in ("stuck", "dropout") and v == .35)}
        if st == "suspect" and len(fams) >= 2:
            st = "bad"
        rows.append(dict(st8=st, rules8=",".join(fired), watch8=",".join(watch), cleared8=",".join(notes),
                         left8=",".join(sorted(find)), score8=hs, dq8=",".join(dq)))
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


FL = ("suspect", "bad")


def rate_row(R, tag):
    out = []
    for wg in ("m30", "h3", "h24"):
        x = R[R.wg == wg]
        out.append(dict(run=tag, window=wg, n=len(x), flag=100 * x.st8.isin(FL).mean(), bad=100 * x.st8.eq("bad").mean(),
                        watch=100 * x.st8.eq("watch").mean(),
                        **{f"f_{k}": 100 * (x[f"s8_{k}"] >= .35).mean() for k in ("choppy", "level", "stuck", "prof")}))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X0, E = prepare()
    chk = X0[X0.chop_old.notna() | X0.chop15_v108.notna()]
    print("F1 check: package chop15 recomputed = saved:",
          f"{np.mean(np.isclose(chk.chop_old, chk.chop15_v108, rtol=1e-6, equal_nan=True)):.4f}")
    lc = X0[X0.level_ratio.notna() & X0.wg.ne("m30")]
    print("F2 check: package level ratio recomputed = saved:",
          f"{np.mean(np.isclose(lc.lv_ratio_chk, lc.level_ratio, rtol=1e-4, equal_nan=True)):.4f}")
    runs = {"v108": frozenset(), "F1": {1}, "F2": {2}, "F3": {3}, "F4": {4}, "F3+F4": {3, 4}, "F5": {5}, "F6": {6},
            "v110": ALL}
    rates, keep = [], {}
    for tag, fx in runs.items():
        R = run(X0, E, set(fx))
        rates += rate_row(R, tag)
        keep[tag] = R[["DeviceId", "window", "detector", "st8", "rules8", "watch8", "left8", "cleared8"]]
        if tag == "v108":
            same = (R.st8.to_numpy() == R.st8_v108.to_numpy()).mean()
            print("v108 reproduced by the v110 code with every fix off:", f"{same:.5f}")
        if tag == "v110":
            R.to_parquet(OUT / "resolved110.parquet")
        print(tag, "done", flush=True)
    T = pd.DataFrame(rates).round(3)
    T.to_csv(OUT / "fix_rates110.csv", index=False)
    print(T.to_string(index=False))
    base_ = keep["v108"].rename(columns={"st8": "st_v108"})
    for tag in runs:
        if tag == "v108":
            continue
        m = base_.merge(keep[tag], on=["DeviceId", "window", "detector"])
        m["wg"] = m.window.str.split("_").str[0]
        for wg, g in m.groupby("wg"):
            up = (g.st8.isin(FL) & ~g.st_v108.isin(FL)).mean() * 100
            dn = (~g.st8.isin(FL) & g.st_v108.isin(FL)).mean() * 100
            print(f"{tag:6s} {wg:4s} newly flagged {up:.3f} / unflagged {dn:.3f} per 100")
    pd.concat([v.assign(run=k) for k, v in keep.items()]).to_parquet(OUT / "runs110.parquet")


if __name__ == "__main__":
    main()
