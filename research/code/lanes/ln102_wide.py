"""Note 102: the 'wide right lane' lane-count error (user, lane review v1) -- an extra stop-bar loop on the right of a
through / right lane (the lane widens at the corner radius, vehicles pull right) is still ONE lane.

Fix tried with hi-res data only (no print fact is a model input):
  W   interval pair features of two detectors (ON / OFF in 0.1 s ticks): occupancy of each, share of the lower-occupancy
      detector's ON time / ON starts / whole intervals covered by the other, lift over chance (all hours and 22-06),
      same-vehicle share (ON and OFF both within 1 s), duration ratio.  Symmetric by construction (lo / hi by ON time),
      so both orientations get the same values.
  X   decoder 'stack exemption': the one-role-per-lane penalty is waived for a pair whose P(same) >= tau (the two are
      then a stacked pair in one lane; the ATSPM pick later calls the weaker one Other).  tau in the per-fold grid.
Setup = the 2026 lane recipe of package v5 (pkg95.pkglanes): lanes D pair model (cues + context + pair type + function
block of the 2026 function-tree OOF function_c_v4q26), both orientations, labels = print lanes on v4q (+ the user's
lane answers of review v1), Sept-2026 data only, six folds (folds_v4) x 3 seeds, lam (+ tau) picked per held-out fold.
locked_v2 asserted absent.  CPU <= 6 workers.

    python ln102_wide.py setup               working dir f102/ln8: v4q truth (+ user lane answers), cue / key copies
    python ln102_wide.py wcues [--workers 6] interval pair features for every Sept pair       -> f102/ln8/wcues.parquet
    python ln102_wide.py fit --var base|W|Wshuf                                             -> f102/ln8/p_<var>.parquet
    python ln102_wide.py decode --var V      Sept-scope grid lam x tau, per-fold pick        -> f102/ln8/pick_<var>.json
    python ln102_wide.py full --var V        every Sept window >= 30 min, picked settings    -> f102/ln8/lanes_<cfg>.parquet
    python ln102_wide.py eval                n_lanes exact (14 windows, note-99 scope) overall / pattern / CIs
    python ln102_wide.py atspm               ATSPM function score through the lane step (P_v5 fixed, lanes swapped)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
for _d in ("", "lanes", "final77", "evaluation", "trackA"):
    sys.path.insert(0, str(CODE / _d) if _d else str(CODE))
import rpath  # noqa: F401,E402
import lane_output as LO  # noqa: E402
import ln8_validate as L8  # noqa: E402

DCW = L8.DCW
REPO = L8.REPO
F95 = DCW / "final_v3_work" / "f95" / "ln8"
F77 = DCW / "final_v3_work" / "f77" / "ln8"
OUT = DCW / "final_v3_work" / "f102" / "ln8"
TREES95 = DCW / "final_v3_work" / "f76" / "function_c_v4q26" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx"
STG = DCW / "official" / "stg" / "cache" / "det_intervals.parquet"
CORR = REPO / "research" / "labels" / "lane_truth_corrections_user_v1.csv"
LAMS = [2.0, 3.0, 4.0]
TAUS = [None, 0.90, 0.95]
W_COLS = ["w_occ_lo", "w_occ_hi", "w_cov_lo", "w_cov_hi", "w_lift", "w_st_lo", "w_st_hi", "w_stl_lo", "w_stl_hi",
          "w_nest_lo", "w_nest_hi", "w_sv_lo", "w_sv_hi", "w_dur_lr", "w_cov_lo_q", "w_lift_q"]
REVIEW_WINS = ["m30_a", "m30_b", "m30_c", "m30_d", "h1_a", "h1_b", "h1_c", "h3_a", "h3_b", "h6_a", "h6_b",
               "h24_a", "h24_b", "full66"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def patch():
    """point the ln8 / of77 machinery at the f102 working dir and the 2026 function-tree OOF."""
    import of77
    L8.OUT = OUT
    L8.FUNC_DIR = TREES95
    of77.O8 = OUT
    return of77


# ================================================================================================ setup
def stage_setup(a):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("cues.parquet", "dets.parquet", "truth_text_pairs.parquet", "truth_det.parquet", "truth_phase.parquet"):
        shutil.copy(F95 / f, OUT / f)
    for f in ("cues_rev.parquet", "keys.parquet"):
        shutil.copy(F77 / f, OUT / f)
    det = pd.read_parquet(OUT / "truth_det.parquet")
    ph = pd.read_parquet(OUT / "truth_phase.parquet")
    C = pd.read_csv(CORR, dtype={"signal": str})
    L = pd.read_parquet(rpath.LABELS_CURRENT, columns=["DeviceId", "DeviceName"]).drop_duplicates()
    nm = dict(zip(L.DeviceName, L.DeviceId.str.lower()))
    lk = L8.locked()
    n_ph = n_det = 0
    for r in C.itertuples():
        dev = nm[r.signal]
        assert dev not in lk
        if r.kind == "phase":
            m = (ph.DeviceId == dev) & (ph.target == r.target)
            assert m.sum() == 1, r
            ph.loc[m, "n_lanes"] = int(r.value)
            n_ph += 1
        else:
            m = (det.DeviceId == dev) & (det.target == r.target) & (det.det == int(r.det))
            assert m.sum() == 1, r
            li, sp = (int(x) for x in str(r.value).split(":"))
            det.loc[m, "lane_index"] = li
            det.loc[m, "span"] = sp
            det.loc[m, "end"] = li + sp - 1
            n_det += 1
    det.to_parquet(OUT / "truth_det.parquet", index=False)
    ph.to_parquet(OUT / "truth_phase.parquet", index=False)
    assert not det.DeviceId.isin(lk).any() and not ph.DeviceId.isin(lk).any()
    log(f"truth v4q + user lane answers: {n_ph} phases, {n_det} detector rows changed; {len(ph)} phases / "
        f"{ph.DeviceId.nunique()} signals")


# ================================================================================================ interval features
def _occ(iv_on, iv_off, n):
    """bool occupancy array (ticks) from interval tick arrays (clipped to [0, n))."""
    d = np.zeros(n + 1, np.int32)
    s, e = np.clip(iv_on, 0, n), np.clip(iv_off, 0, n)
    k = e > s
    np.add.at(d, s[k], 1)
    np.add.at(d, e[k], -1)
    return np.cumsum(d[:n]) > 0


def pair_w(A, B, hq):
    """A / B = (on ticks, off ticks, occupancy bool, cumsum of occupancy); hq = bool quiet-hour mask per tick."""
    oa, ob = A[2], B[2]
    n = len(oa)
    ta, tb = int(oa.sum()), int(ob.sum())
    lo, hi = (A, B) if (ta, len(A[0])) <= (tb, len(B[0])) else (B, A)
    tl, th = int(lo[2].sum()), int(hi[2].sum())
    both = int(np.count_nonzero(lo[2] & hi[2]))
    eps = 1e-5
    r = {"w_occ_lo": np.log10(max(tl / n, eps)), "w_occ_hi": np.log10(max(th / n, eps)),
         "w_cov_lo": both / tl if tl else np.nan, "w_cov_hi": both / th if th else np.nan}
    r["w_lift"] = np.log(max(both, 0.5) * n / max(tl * th, 1))
    out = {}
    for tag, x, y in (("lo", lo, hi), ("hi", hi, lo)):
        s, e = x[0], x[1]
        k = (s >= 0) & (s < n) & (e > s)
        s, e = s[k], np.minimum(e[k], n)
        if len(s) == 0:
            out[tag] = (np.nan, np.nan, np.nan, np.nan)
            continue
        occ_y = y[2].mean()
        st = y[2][s].mean()                                   # other detector already ON at this ON start
        cs = y[3]
        cov = (cs[e] - cs[s]) / (e - s)                       # share of this interval covered by the other
        nest = float(np.mean(cov >= 0.999))
        # same vehicle: an ON of the other within 1 s AND its OFF within 1 s
        ys, ye = y[0], y[1]
        j = np.searchsorted(ys, s)
        sv = np.zeros(len(s), bool)
        for dj in (-1, 0, 1):
            jj = np.clip(j + dj, 0, max(len(ys) - 1, 0))
            if len(ys):
                sv |= (np.abs(ys[jj] - s) <= 10) & (np.abs(ye[jj] - e) <= 10)
        out[tag] = (st, np.log(max(st, 1e-3) / max(occ_y, 1e-4)), nest, float(sv.mean()))
    r["w_st_lo"], r["w_stl_lo"], r["w_nest_lo"], r["w_sv_lo"] = out["lo"]
    r["w_st_hi"], r["w_stl_hi"], r["w_nest_hi"], r["w_sv_hi"] = out["hi"]
    dl, dh = np.median(lo[1] - lo[0]) if len(lo[0]) else np.nan, np.median(hi[1] - hi[0]) if len(hi[0]) else np.nan
    r["w_dur_lr"] = np.log(max(dl, 1) / max(dh, 1)) if np.isfinite(dl) and np.isfinite(dh) else np.nan
    if hq.any():
        lq, hq_ = lo[2] & hq, hi[2] & hq
        tlq, thq = int(lq.sum()), int(hq_.sum())
        bq = int(np.count_nonzero(lq & hi[2]))
        nq = int(hq.sum())
        r["w_cov_lo_q"] = bq / tlq if tlq >= 20 else np.nan
        r["w_lift_q"] = np.log(max(bq, 0.5) * nq / max(tlq * thq, 1)) if tlq >= 20 and thq >= 20 else np.nan
    else:
        r["w_cov_lo_q"] = r["w_lift_q"] = np.nan
    return r


def wcue_work(args):
    dev, pairs = args
    import a2_features as A2F
    import pyarrow.dataset as ds
    tab = ds.dataset(str(STG)).to_table(filter=ds.field("DeviceId") == dev,
                                        columns=["Detector", "t_on", "t_off"]).to_pandas().drop_duplicates()
    tab = tab[tab.t_off.notna()]
    tab["s"] = (tab.t_on.astype("datetime64[us]").astype("int64") // 100000).to_numpy()   # 0.1 s ticks
    tab["e"] = (tab.t_off.astype("datetime64[us]").astype("int64") // 100000).to_numpy()
    tab["e"] = np.maximum(tab.e, tab.s + 1)
    by = {int(d): g.sort_values("s") for d, g in tab.groupby("Detector")}
    wins = {n: (int(pd.Timestamp(t0).value // 100_000_000), int(secs * 10)) for n, t0, secs in A2F.WINDOWS["stg"]}
    rows = []
    for w, g in pairs.groupby("win"):
        t0, n = wins[w]
        tick_h = ((t0 + np.arange(n)) // 36000) % 24
        hq = (tick_h >= 22) | (tick_h < 6)
        dets = sorted(set(g.da.astype(int)) | set(g.db.astype(int)))
        T = {}
        for d in dets:
            x = by.get(d)
            if x is None:
                s = e = np.zeros(0, np.int64)
            else:
                m = (x.s.to_numpy() >= t0) & (x.s.to_numpy() < t0 + n)
                s, e = x.s.to_numpy()[m] - t0, x.e.to_numpy()[m] - t0
            o = _occ(s, e, n)
            T[d] = (s, np.minimum(e, n), o, np.concatenate([[0], np.cumsum(o)]))
        for da, db in zip(g.da.astype(int), g.db.astype(int)):
            r = pair_w(T[da], T[db], hq)
            r.update(DeviceId=dev, win=w, da=da, db=db)
            rows.append(r)
    return pd.DataFrame(rows)


def stage_wcues(a):
    P = pd.read_parquet(OUT / "cues.parquet", columns=["DeviceId", "period", "win", "da", "db"])
    P = P[P.period == "stg"]
    assert not P.DeviceId.isin(L8.locked()).any()
    jobs = [(dev, g[["win", "da", "db"]]) for dev, g in P.groupby("DeviceId")]
    if a.limit:
        jobs = jobs[:a.limit]
    log(f"{len(jobs)} signals, {len(P):,} Sept pairs")
    from multiprocessing import Pool
    out, t0 = [], time.time()
    with Pool(a.workers) as pool:
        for i, x in enumerate(pool.imap_unordered(wcue_work, jobs, chunksize=1)):
            out.append(x)
            if i % 50 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    W = pd.concat(out, ignore_index=True)
    W["period"] = "stg"
    for c in W_COLS:
        W[c] = W[c].astype(np.float32)
    W.to_parquet(OUT / ("wcues_test.parquet" if a.limit else "wcues.parquet"), index=False)
    log(f"wrote {len(W):,} rows ({time.time()-t0:.0f}s); NaN share {W[W_COLS].isna().mean().round(3).to_dict()}")


# ================================================================================================ fit
def label_pairs(Pf, med: bool):
    """same_lane labels on the Sept training windows: high-confidence print rows (as ln8), or + medium (variant m)."""
    import ln2_pairmodel as L2
    det = pd.read_parquet(OUT / "truth_det.parquet")
    ph = pd.read_parquet(OUT / "truth_phase.parquet")
    if med:
        det = det.assign(high_veh=det.print_confidence.isin(["high", "medium"]) & det.lane_index.notna()
                         & ~det.lane_type.isin(L8.L1.NONVEH))
    T = L2.truth_pairs(det, ph)[["DeviceId", "da", "db", "same_lane", "n_lanes"]]
    tw = ((Pf.period == "stg") & Pf.win.isin(L8.TRAIN_WINS)).to_numpy()
    x = Pf.loc[tw, ["DeviceId", "da", "db"]].merge(T, on=["DeviceId", "da", "db"], how="left")
    assert len(x) == int(tw.sum())
    y = np.full(len(Pf), np.nan)
    y[tw] = x.same_lane.to_numpy()
    return y


def frames(var: str):
    """(D, Pf, Pr, names, Xf, Xr) on Sept rows; W block appended when 'W' in var (permuted across rows when 'shuf')."""
    of77 = patch()
    D, Pf, Pr = of77.sym_frames()
    keep = (Pf.period == "stg").to_numpy()
    Pf, Pr = Pf[keep].reset_index(drop=True), Pr[keep].reset_index(drop=True)
    D = D[D.period == "stg"].reset_index(drop=True)
    assert not Pf.DeviceId.isin(L8.locked()).any()
    c2 = L8.c2_matrix(Pf, D, np.arange(len(Pf)))
    names = LO.FEATURES + c2[1]
    Xf = np.hstack([Pf[LO.FEATURES].to_numpy(np.float32), c2[0]])
    Xr = np.hstack([Pr[LO.FEATURES].to_numpy(np.float32), c2[0]])
    if "W" in var:
        kk = ["DeviceId", "period", "win", "da", "db"]
        W = Pf[kk].merge(pd.read_parquet(OUT / "wcues.parquet"), on=kk, how="left")
        assert len(W) == len(Pf)
        Z = W[W_COLS].to_numpy(np.float32)
        log(f"W block: {Z.shape}, rows without W {np.isnan(Z[:, 0]).mean():.4f}")
        if "shuf" in var:
            Z = Z[np.random.default_rng(102).permutation(len(Z))]
        Xf, Xr = np.hstack([Xf, Z]), np.hstack([Xr, Z])
        names = names + W_COLS
    if "T" in var:
        kk = ["DeviceId", "period", "win", "da", "db"]
        T = Pf[kk].merge(pd.read_parquet(OUT / "tri.parquet"), on=kk, how="left")
        assert len(T) == len(Pf)
        Z = T[T_COLS].to_numpy(np.float32)
        if "Tshuf" in var:
            Z = Z[np.random.default_rng(103).permutation(len(Z))]
        Xf, Xr = np.hstack([Xf, Z]), np.hstack([Xr, Z])
        names = names + T_COLS
    return D, Pf, Pr, names, Xf, Xr


T_COLS = ["t_adv", "t_anchor", "t_any", "t_n_adv_both", "t_conflict"]


def stage_tri(a):
    """second-stage 'shared neighbour' pair features from the OOF base P(same): for a pair (a, b) on one predicted
    phase, over every third detector k of that phase: max_k min(P(a,k), P(b,k)) for k predicted Advance / any anchor /
    anyone; # Advance k with both P >= .8; max_k |P(a,k) - P(b,k)| (a third detector that tells them apart)."""
    of77 = patch()
    q = pd.read_parquet(OUT / "p_base.parquet")
    q = q[q.same_pred]
    D = L8.load_dets()
    D = D[D.period == "stg"]
    fn = {(d, w, int(x)): f for d, w, x, f in zip(D.DeviceId, D.win, D.Detector, D.func)}
    rows = []
    for (dev, w), g in q.groupby(["DeviceId", "win"]):
        dets = sorted(set(g.da.astype(int)) | set(g.db.astype(int)))
        ix = {d: i for i, d in enumerate(dets)}
        Pm = np.full((len(dets), len(dets)), np.nan)
        for x, y, p in zip(g.da.astype(int), g.db.astype(int), g.p_same):
            Pm[ix[x], ix[y]] = Pm[ix[y], ix[x]] = p
        f = np.array([fn.get((dev, w, d), "") for d in dets], object)
        adv, anc = f == "Advance", np.isin(f, list(LO.ANCHOR))
        for x, y in zip(g.da.astype(int), g.db.astype(int)):
            i, j = ix[x], ix[y]
            m = np.ones(len(dets), bool)
            m[[i, j]] = False
            m &= ~np.isnan(Pm[i]) & ~np.isnan(Pm[j])
            mn = np.fmin(Pm[i], Pm[j])
            r = {"DeviceId": dev, "win": w, "da": x, "db": y,
                 "t_adv": np.nanmax(np.where(m & adv, mn, np.nan)) if (m & adv).any() else np.nan,
                 "t_anchor": np.nanmax(np.where(m & anc, mn, np.nan)) if (m & anc).any() else np.nan,
                 "t_any": np.nanmax(np.where(m, mn, np.nan)) if m.any() else np.nan,
                 "t_n_adv_both": float((m & adv & (mn >= .8)).sum()),
                 "t_conflict": np.nanmax(np.where(m, np.abs(Pm[i] - Pm[j]), np.nan)) if m.any() else np.nan}
            rows.append(r)
    T = pd.DataFrame(rows)
    T["period"] = "stg"
    T.to_parquet(OUT / "tri.parquet", index=False)
    log(f"tri: {len(T):,} pairs")


def is_med(var: str) -> bool:
    return var.split("_")[0].endswith("m")


def stage_fit(a):
    import lightgbm as lgb
    import ln2_pairmodel as L2
    var = a.var
    D, Pf, Pr, names, Xf, Xr = frames(var)
    y = label_pairs(Pf, med=is_med(var))
    lab = ~np.isnan(y)
    fo = Pf.fold.to_numpy()
    log(f"{var}: {len(Pf):,} Sept pairs, labelled {lab.sum():,} (same {np.nanmean(y):.3f}), {len(names)} features")
    out = np.zeros(len(Pf))
    gain = np.zeros(len(names))
    for f in range(6):
        te, tr = fo == f, lab & (fo != f)
        X = np.vstack([Xf[tr], Xr[tr]])
        yy = np.concatenate([y[tr], y[tr]]).astype(int)
        for s in L8.SEEDS:
            m = lgb.LGBMClassifier(random_state=s, **dict(L8.PARAMS, n_jobs=a.workers)).fit(X, yy)
            out[te] += 0.5 * (m.predict_proba(Xf[te])[:, 1] + m.predict_proba(Xr[te])[:, 1]) / len(L8.SEEDS)
            gain += m.booster_.feature_importance("gain")
        log(f"  fold {f} done")
    Q = Pf[["DeviceId", "period", "win", "da", "db", "same_pred", "fold"]].copy()
    Q["same_lane"] = label_pairs(Pf, med=False)
    Q["n_lanes"] = Pf.n_lanes.to_numpy()
    Q["p_same"] = out.astype(np.float32)
    Q.to_parquet(OUT / f"p_{var}.parquet", index=False)
    r = {"labelled_train": int(lab.sum())}
    qq = Q[Q.same_lane.notna()]
    for wg in ["m30", "h6", "h24", "full"]:
        mm = (qq.win.map(L8.wgroup) == wg).to_numpy() & (qq.n_lanes >= 2).to_numpy()
        r[wg] = L2.auc(qq.same_lane.to_numpy()[mm], qq.p_same.to_numpy()[mm])
    gi = pd.Series(gain / gain.sum(), index=names).sort_values(ascending=False)
    r["top_gain"] = gi.head(25).round(4).to_dict()
    r["W_gain_share"] = float(gi[[c for c in W_COLS if c in gi.index]].sum()) if "W" in var else 0.0
    json.dump(r, open(OUT / f"fit_{var}.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in r.items() if k != "top_gain"}))


# ================================================================================================ decode
class DemoteDecoder(LO.Decoder):
    """tau: a same-role anchor pair with P(same) >= tau is a stack -> the lower-volume one is decoded as Other
    (joins a lane, never founds one; the role rule then does not split the pair)."""

    def __init__(self, span_prior, lam=0.0, tau=None, adv=None):
        super().__init__(span_prior, lam=lam)
        self.tau = tau
        self.adv = adv                     # log prior of an Advance spanning >= 2 lanes (None = data prior)

    def _sp(self, f, k):
        if self.adv is not None and f == "Advance" and k >= 2:
            return self.adv
        return super()._sp(f, k)

    def decode(self, P, func, vol):
        func = list(func)
        if self.tau is not None and len(func) > 1:
            n = len(func)
            cand = [(P[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                    if func[i] in LO.ANCHOR and func[i] == func[j] and np.isfinite(P[i, j]) and P[i, j] >= self.tau]
            for p, i, j in sorted(cand, key=lambda t: (-t[0], -max(vol[t[1]], vol[t[2]]))):
                if func[i] in LO.ANCHOR and func[i] == func[j]:
                    k = j if (vol[j], -j) < (vol[i], -i) else i
                    func[k] = "Other"
        return super().decode(P, func, vol)


def cfg_name(var, lam, tau, adv=None):
    return f"{var}@{lam:g}|" + ("none" if tau is None else f"{tau:g}") + ("" if adv is None else f"|a{adv:g}")


def parse_cfg(g):
    f = g.split("@")[1].split("|")
    lam = float(f[0])
    tau = None if f[1] == "none" else float(f[1])
    adv = float(f[2][1:]) if len(f) > 2 else None
    return lam, tau, adv


def dec_job(args):
    dev, Dd, probs, cfgs, keys = args
    dets, phs = [], []
    for name, var, mode, lam, prior in cfgs:
        _, tau, adv = parse_cfg(name)
        dec = DemoteDecoder(prior, lam=lam, tau=tau, adv=adv)
        for (period, w), g in Dd.groupby(["period", "win"]):
            gg = g.rename(columns={"pred_phase": "phase_use", "func": "func_use"})
            n_on = dict(zip(gg.Detector.astype(int), gg.n_on))
            pt, dt = LO.decode_groups(dev, gg, n_on, g.hours.iloc[0], probs.get(var, {}).get((period, w), {}), dec,
                                      keys=keys[(period, w)])
            dt["function"] = gg.set_index(gg.Detector.astype(int)).func_use.reindex(dt.Detector).to_numpy()
            dt["period"], dt["win"], dt["cfg"] = period, w, name
            dets.append(dt)
            if len(pt):
                pt["period"], pt["win"], pt["cfg"] = period, w, name
                phs.append(pt)
    return (pd.concat(dets, ignore_index=True), pd.concat(phs, ignore_index=True) if phs else pd.DataFrame())


def run_dec(D, q, cfgs, workers, tag):
    of77 = patch()
    of77.dec_job = dec_job
    return of77.run_decodes(D, q, cfgs, workers, tag)


def stage_decode(a):
    of77 = patch()
    det, ph, T, D, D9, pri = of77._priors()
    var = a.var
    q = pd.read_parquet(OUT / f"p_{var}.parquet")
    q = q[q.same_pred & q.win.isin(L8.TRAIN_WINS) & q.DeviceId.isin(set(ph.DeviceId))]
    taus = [None] if a.notau else TAUS
    advs = [None if x == "none" else float(x) for x in a.advs.split(",")]
    grid = [cfg_name("Dsym", lam, tau, adv) for lam in LAMS for tau in taus for adv in advs]
    cfgs = {f: [(g, "Dsym", "func", parse_cfg(g)[0], pri[f]) for g in grid] for f in range(6)}
    var_o = var + a.suffix
    DT, PT = run_dec(D9, q, cfgs, a.workers, f"sept {var_o}")
    DT.to_parquet(OUT / f"sept_det_{var_o}.parquet", index=False)
    PT.to_parquet(OUT / f"sept_ph_{var_o}.parquet", index=False)
    res = {}
    for g in grid:
        NL, PR, DX = L8.eval_cfg(DT[DT.cfg == g], PT[PT.cfg == g], D9, det, ph, T)
        ob = L8.obj_by_fold(NL, PR, DX)
        res[g] = {f: (ob[f]["nl_exact"] or 0) + (ob[f]["pair_acc_multi"] or 0) for f in range(6)}
        log(f"  {g}: total {sum(res[g].values()):.4f}")
    pick = {f: max(grid, key=lambda g: sum(res[g][k] for k in range(6) if k != f)) for f in range(6)}
    log(f"{var_o} picked {[pick[f] for f in range(6)]}")
    json.dump({"grid_obj": res, "pick": pick}, open(OUT / f"pick_{var_o}.json", "w"), indent=1)


def stage_full(a):
    """decode every Sept window >= 30 min with the per-fold pick (tag = var, or var + '.notau' with tau forced off)."""
    of77 = patch()
    det, ph, T, D, D9, pri = of77._priors()
    var = a.var
    pk = json.load(open(OUT / f"pick_{var}{a.suffix}.json"))["pick"]
    q = pd.read_parquet(OUT / f"p_{var}.parquet", columns=["DeviceId", "period", "win", "da", "db", "p_same", "same_pred"])
    q = q[q.same_pred]
    Ds = D[(D.period == "stg") & D.win.isin(REVIEW_WINS)]
    tag = var + a.suffix + (".notau" if a.notau else "")

    def cfg(f):
        lam, tau, adv = parse_cfg(pk[str(f)])
        if a.notau:
            tau = None
        return [(cfg_name("Dsym", lam, tau, adv), "Dsym", "func", lam, pri[f])]
    DT, PT = run_dec(Ds, q, {f: cfg(f) for f in range(6)}, a.workers, f"full {tag}")
    DT = DT.merge(Ds[["DeviceId", "period", "win", "Detector", "n_on"]], on=["DeviceId", "period", "win", "Detector"],
                  how="left")
    DT.to_parquet(OUT / f"lanes_{tag}.parquet", index=False)
    PT.to_parquet(OUT / f"lanes_{tag}_ph.parquet", index=False)
    log(f"{tag}: {len(DT):,} det-windows, {len(PT):,} phase-windows")


# ================================================================================================ eval
SB = {"stopbar_presence", "stopbar_count", "stopbar_secondary", "adaptive_stopbar"}
REF77 = F77 / "lanes_D.func"


def pattern_phases() -> pd.DataFrame:
    """print phases with the wide-right-lane layout: one print lane holding >= 2 single-lane stop-bar zones of the same
    print function (Presence / Count), or a stop-bar zone plus an 'Other' zone drawn on the right (lane type R / TR);
    any print confidence.  Plus the user-confirmed phases of review v1.  Evaluation only."""
    det = pd.read_parquet(OUT / "truth_det.parquet")
    L = pd.read_parquet(rpath.LABELS_CURRENT, columns=["DeviceId", "detector", "print_subtype"])
    L["DeviceId"] = L.DeviceId.str.lower()
    L = L.rename(columns={"detector": "det"}).drop_duplicates(["DeviceId", "det"])
    L["det"] = L.det.astype(int)
    d = det.merge(L, on=["DeviceId", "det"], how="left")
    d = d[d.lane_index.notna() & (d.span == 1)]
    sb = d.print_subtype.isin(SB) & d.print_function.isin(["Presence", "Count"])
    oth = (d.print_function == "Other") & d.lane_type.isin(["R", "TR"]) & ~d.print_subtype.isin(
        ["advance", "advance_presence", "long_zone", "mid", "eta", "departure", "bike"])
    k = ["DeviceId", "target", "lane_index"]
    twin = d[sb].groupby(k + ["print_function"]).size().gt(1).groupby(level=[0, 1, 2]).any()
    n_sb = d[sb].groupby(k).size()
    n_ot = d[oth].groupby(k).size()
    lanes = pd.DataFrame({"twin": twin, "n_sb": n_sb, "n_ot": n_ot}).fillna(0)
    lanes["wide"] = (lanes.twin > 0) | ((lanes.n_sb >= 1) & (lanes.n_ot >= 1))
    P = lanes[lanes.wide].reset_index()[["DeviceId", "target"]].drop_duplicates()
    P["user_confirmed"] = False
    names = pd.read_parquet(rpath.LABELS_CURRENT, columns=["DeviceId", "DeviceName"]).drop_duplicates()
    nm = dict(zip(names.DeviceName, names.DeviceId.str.lower()))
    uc = pd.DataFrame({"DeviceId": [nm[s] for s in ("01030", "01072", "03044", "04059")],
                       "target": ["P8", "P4", "P4", "P8"], "user_confirmed": True})
    P = pd.concat([P[~P.set_index(["DeviceId", "target"]).index.isin(uc.set_index(["DeviceId", "target"]).index)], uc])
    return P.reset_index(drop=True)


def sample_rows(tag) -> pd.DataFrame:
    """phase-window rows (14 Sept windows >= 30 min) truth vs model n_lanes; truth phase must exist as a predicted phase."""
    ph = pd.read_parquet(OUT / "truth_phase.parquet")
    ph["ph_num"] = ph.target.str[1:].astype(int)
    f = (str(REF77) + "_ph.parquet") if tag == "ref77" else OUT / f"lanes_{tag}_ph.parquet"
    P = pd.read_parquet(f)
    P = P[(P.period == "stg") & P.win.isin(REVIEW_WINS)]
    x = ph.merge(P[["DeviceId", "phase", "win", "n_lanes", "n_lanes_conf"]], left_on=["DeviceId", "ph_num"],
                 right_on=["DeviceId", "phase"], suffixes=("", "_p"))
    x["ok"] = (x.n_lanes == x.n_lanes_p).astype(float)
    return x


def stage_eval(a):
    tags = a.tags.split(",")
    pat = pattern_phases()
    pk = set(zip(pat.DeviceId, pat.target))
    uc = set(zip(pat[pat.user_confirmed].DeviceId, pat[pat.user_confirmed].target))
    R = {t: sample_rows(t) for t in tags}
    key = ["DeviceId", "target", "win"]
    common = None
    for t in tags:
        s = R[t][key]
        common = s if common is None else common.merge(s, on=key)
    res = {"pattern_phases": len(pat), "pattern_signals": int(pat.DeviceId.nunique())}
    for t in tags:
        x = R[t].merge(common, on=key)
        x["pat"] = [(d, g) in pk for d, g in zip(x.DeviceId, x.target)]
        x["uc"] = [(d, g) in uc for d, g in zip(x.DeviceId, x.target)]
        R[t] = x
        r = {"n": int(len(x)), "exact": round(float(x.ok.mean()), 4),
             "over": round(float((x.n_lanes_p > x.n_lanes).mean()), 4),
             "under": round(float((x.n_lanes_p < x.n_lanes).mean()), 4),
             "pattern": {"n": int(x.pat.sum()), "exact": round(float(x[x.pat].ok.mean()), 4),
                         "over": round(float((x[x.pat].n_lanes_p > x[x.pat].n_lanes).mean()), 4)},
             "not_pattern": round(float(x[~x.pat].ok.mean()), 4),
             "user_confirmed_4": {f"{d[:4]}|{g}": round(float(x[(x.DeviceId == d) & (x.target == g)].ok.mean()), 3)
                                  for d, g in sorted(uc)},
             "by_truth": {int(k): round(float(v), 4) for k, v in x.groupby("n_lanes").ok.mean().items()},
             "by_wg": {w: round(float(x[x.win.map(L8.wgroup) == w].ok.mean()), 4)
                       for w in ["m30", "h1", "h3", "h6", "h24", "full"]}}
        res[t] = r
        log(f"{t:12s} exact {r['exact']} (over {r['over']}, under {r['under']}) | pattern {r['pattern']} | "
            f"other {r['not_pattern']} | by truth {r['by_truth']}")
    ref = tags[0]
    for t in tags[1:]:
        c = {}
        for nm_, m in (("all", lambda x: x.ok == x.ok), ("pattern", lambda x: x.pat), ("not_pattern", lambda x: ~x.pat),
                       ("truth1", lambda x: x.n_lanes == 1), ("truth2", lambda x: x.n_lanes == 2),
                       ("truth3+", lambda x: x.n_lanes >= 3)):
            d0, d1 = R[ref][m(R[ref])], R[t][m(R[t])]
            c[nm_] = [round(100 * (d1.ok.mean() - d0.ok.mean()), 2)] + L8.boot(d0, d1)
        # phase-level changes (majority over samples)
        j = R[ref].merge(R[t], on=key, suffixes=("_0", "_1"))
        c["sample_flips"] = {"fixed": int(((j.ok_0 == 0) & (j.ok_1 == 1)).sum()),
                             "broken": int(((j.ok_0 == 1) & (j.ok_1 == 0)).sum())}
        res[f"{t} vs {ref}"] = c
        log(f"{t} vs {ref}: {c}")
    json.dump(res, open(OUT / f"eval_{a.out}.json", "w"), indent=1, default=str)


# ================================================================================================ atspm
def lanes_file(tag) -> Path:
    return Path(str(REF77) + ".parquet") if tag == "ref77" else OUT / f"lanes_{tag}.parquet"


def stage_atspm(a):
    """v5 function OOF probabilities (f95/oof/P_v5.npy) held fixed; only the lane step's inputs change (lanes5g of
    the Sept rows + lane_conf for the .9 gate).  Pick inputs (ln6 / ln7) and the stacker's lane context held fixed.
    Gate .9 decode, v4q truth, Sept-2026 rows, E / R by length pool; paired signal CI vs the first tag."""
    os.environ.setdefault("F76_ARM", "c")
    sys.path.insert(0, str(CODE / "final95"))
    import func95 as FN
    import cand64 as C
    import of77
    F, F4 = FN.fsetup()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    P = np.load(FN.OOF95 / "P_v5.npy").astype(float)
    assert len(P) == len(fr)
    st = (fr.period == "stg").to_numpy()
    sig = fr.DeviceId.to_numpy()
    lanes0 = fr.lanes5g.copy()
    k = ["DeviceId", "Detector", "period", "win"]
    ok, pred = {}, {}
    for t in a.tags.split(","):
        L = pd.read_parquet(lanes_file(t), columns=k + ["phase", "lanes"])
        L = L[(L.period == "stg") & (L.lanes != "")].astype({"Detector": fr.Detector.dtype})
        x = fr[k].merge(L, on=k, how="left")
        assert len(x) == len(fr)
        new = x.lanes.where(x.phase.eq(fr.pred_phase.to_numpy()), None).where(~fr.wgroup.isin(["m5", "m10"]), None)
        fr["lanes5g"] = np.where(st, new.to_numpy(object), lanes0.to_numpy(object))
        lc = of77.lane_ctx(fr, lanes_file(t))[:, 2]
        lc = np.where(st, lc, np.nan)
        E["fr"] = fr
        lg = fr.lanes5g.copy()                                   # = s74.gate_ok, keeping the decoded classes
        fr["lanes5g"] = lg.where(~(lc < s74.GATE), None)
        pr2, cr = S.run_decode(E, P)
        fr["lanes5g"] = lg
        ok[t] = S.ok_cols(E, cr)
        pred[t] = np.array(E["C7"], object)[pr2]
        log(f"  {t}: decoded")
    fr["lanes5g"] = lanes0
    out = fr.loc[st, ["DeviceId", "Detector", "win", "wgroup", "pred_phase", "truth_v3s"]].copy()
    for t in pred:
        out[f"func_{t}"] = pred[t][st]
        out[f"okE_{t}"] = ok[t]["E"][st]
    out.to_parquet(OUT / f"func_{a.out}.parquet", index=False)
    tags = list(ok)
    res = {}
    tr = fr.truth_v3s.to_numpy(object)
    for stn in ("E", "R"):
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            pm = st & fr.wgroup.isin(fams).to_numpy()
            sc = pm.copy()
            for t in tags:
                sc &= ~np.isnan(ok[t][stn])
            r = {"n": int(sc.sum())}
            for t in tags:
                r[t] = C.acc_ci(ok[t][stn][sc], sig[sc])
            for t in tags[1:]:
                r[f"{t} - {tags[0]}"] = C.delta_ci(ok[tags[0]][stn][sc], ok[t][stn][sc], sig[sc])
            res[f"{stn}_{pool}"] = r
            log(f"{stn}_{pool}: {r}")
    json.dump(res, open(OUT / f"atspm_{a.out}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--var", default="base")
    ap.add_argument("--notau", action="store_true")
    ap.add_argument("--tags", default="base,W")
    ap.add_argument("--advs", default="none")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="main")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
