"""Note 77: channel-order-free lanes D / pick inputs / decodes -- six-fold OOF rebuild and scoring inside the champion.

Everything that note 76 left channel-order dependent, rebuilt with the order-free code (research lane_output.py,
ln6_pick.py / ln7_stackhealth.py ORDER_FREE, atspm_decode._pair_order, s62 canonical twin pairs):
  cues     pair cues of every ln8 pair in the REVERSED orientation (b, a) + content keys of every detector-window
           -> f77/ln8/cues_rev.parquet, keys.parquet
  fit      lanes D pair model trained on BOTH orientations (labelled rows twice), OOF = mean over both orientations,
           six folds x 3 seeds (note-58 recipe otherwise)                         -> f77/ln8/p_Dsym.parquet
  decode   Sept-scope lam grid {2,3,4} per held-out fold (note 58), decode in canonical behavioural order
                                                                                     -> f77/ln8/decode_pick.json, sept_*
  laneeval lane outputs vs print lanes, new vs note-58 D.func (n_lanes exact, lane set, pair acc; signal CI)
  full     every frame window >= 30 min, per-fold lam                               -> f77/ln8/lanes_D.func(_ph).parquet
  pick     ln6 (span / co-location / track) + ln7 (stack health) on the new lanes, canonical order
                                                                       -> %DC_WORK%/lanes/ln6_pick_D229o.parquet, ln7_*
  hff      function-free stack health for the stacker context (s59 health (c), stack part) -> f77/stackhealth_ff.parquet
  twins    short-window twin pairs (5 / 10 min) in canonical orientation          -> f77/pairs_short.parquet
  stack    context stacker (s74 recipe, main net siba x69_siba, trees = note-76 arm c) on the new context
                                                                                     -> f77/s74/f77/stack_ctx_s*.npy
  compare  champion after note 76 (f76c stack, old lanes / pick / twins) vs this, gate .9, paired signal bootstrap
Inputs held as note 58 / 76: frame v6e predicted phase, note-57 229-arm probabilities for the lane / pick inputs, siba OOF.
locked_v2 asserted absent (frame loaders).  CPU only, <= 6 workers.

    python of77.py cues|fit|decode|laneeval|full|pick|hff|twins|stack|compare [--workers 6]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import lane_output as LO  # noqa: E402
import ln8_validate as L8  # noqa: E402

DCW = L8.DCW
F77 = DCW / "final_v3_work" / "f77"
O8 = F77 / "ln8"
TAG = "D229o"                      # ln6 / ln7 outputs in %DC_WORK%/lanes (attach_pick_inputs reads that folder)
LAMS = [2.0, 3.0, 4.0]
log = L8.log


# ================================================================================================ cues
def cue_work(args):
    """L8.cue_work with the pairs reversed + content keys of every eligible detector-window."""
    dev, period, fk, tt = args
    import a2_features as A2F
    import pyarrow.dataset as ds
    tab = ds.dataset(str(L8.CACHE[period])).to_table(filter=ds.field("DeviceId") == dev,
                                                     columns=["Detector", "t_on"]).to_pandas().drop_duplicates()
    ton = tab.t_on.astype("datetime64[us]")
    tab["t"] = ton.astype("int64").to_numpy() / 1e6
    tab["h"] = ton.dt.hour.to_numpy()
    on = {}
    for d, g in tab.groupby("Detector"):
        o = np.argsort(g.t.to_numpy(), kind="stable")
        on[int(d)] = (g.t.to_numpy()[o], g.h.to_numpy()[o])
    krows, prows = [], []
    for name, t0w, secs in A2F.WINDOWS[period]:
        if name.startswith(("m5", "m10")):
            continue
        g = fk[fk.win == name]
        if g.empty:
            continue
        t0 = pd.Timestamp(t0w).value / 1e9
        t1 = t0 + secs
        n_on = {}
        for d in g.Detector.astype(int):
            t = on.get(d, (np.zeros(0),))[0]
            m = (t >= t0) & (t < t1)
            n_on[d] = int(m.sum())
            k = LO.content_key(t[m], t0)
            krows.append((dev, period, name, d, k[0], k[1], k[2]))
        elig = {int(d): p for d, p in zip(g.Detector.astype(int), g.pred_phase)
                if n_on[int(d)] >= LO.MIN_ON and pd.notna(p) and int(d) in on}
        trainwin = period == "stg" and name in L8.TRAIN_WINS
        pairs = [(a, b) for a, b in itertools.combinations(sorted(elig), 2)
                 if elig[a] == elig[b] or (trainwin and a in tt and b in tt and tt[a] == tt[b])]
        if not pairs:
            continue
        loc = {d: on[d] for d, k in n_on.items() if k >= LO.MIN_ON and d in on}
        P = LO.signal_pair_cues(loc, sorted(loc), [(b, a) for a, b in pairs], t0, t1)
        P = P.rename(columns={"da": "db", "db": "da"})            # keyed by the canonical (a < b) pair, cues of (b, a)
        P["same_pred"] = [elig[a] == elig[b] for a, b in pairs]
        P["DeviceId"], P["period"], P["win"], P["hours"] = dev, period, name, secs / 3600.0
        prows.append(P)
    K = pd.DataFrame(krows, columns=["DeviceId", "period", "win", "Detector", "k_n", "k_s1", "k_s2"])
    return K, (pd.concat(prows, ignore_index=True) if prows else None)


def stage_cues(a):
    O8.mkdir(parents=True, exist_ok=True)
    k = L8.frame_keys()
    k = k[~k.wgroup.isin(["m5", "m10"])]
    det = pd.read_parquet(L8.OUT / "truth_det.parquet")
    jobs = []
    for (dev, period), g in k.groupby(["DeviceId", "period"]):
        tt = {}
        if period == "stg":
            t = det[det.DeviceId == dev]
            tt = dict(zip(t.det.astype(int), t.target))
        jobs.append((dev, period, g[["Detector", "win", "pred_phase"]].copy(), tt))
    log(f"{len(jobs)} signal-periods")
    K, P, t0 = [], [], time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for i, (kk, p) in enumerate(pool.imap_unordered(cue_work, jobs, chunksize=2)):
            K.append(kk)
            if p is not None:
                P.append(p)
            if i % 200 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    K = pd.concat(K, ignore_index=True)
    P = pd.concat(P, ignore_index=True)
    for c in P.columns:
        if P[c].dtype == np.float64:
            P[c] = P[c].astype(np.float32)
    old = pd.read_parquet(L8.OUT / "cues.parquet", columns=["DeviceId", "period", "win", "da", "db"])
    kk = ["DeviceId", "period", "win", "da", "db"]
    chk = old.merge(P[kk], on=kk, how="outer", indicator=True)._merge.value_counts().to_dict()
    log(f"pair sets old vs reversed: {chk}")
    K.to_parquet(O8 / "keys.parquet", index=False)
    P.to_parquet(O8 / "cues_rev.parquet", index=False)
    log(f"wrote {len(K):,} keys, {len(P):,} reversed pairs ({time.time()-t0:.0f}s)")


# ================================================================================================ fit
def sym_frames():
    """(P forward, P reversed) pair frames (L8.pair_frame), rows aligned on the canonical pair key."""
    D = L8.load_dets()
    kk = ["DeviceId", "period", "win", "da", "db"]
    Pf = L8.pair_frame(pd.read_parquet(L8.OUT / "cues.parquet"), D)
    Pr = L8.pair_frame(pd.read_parquet(O8 / "cues_rev.parquet"), D)
    Pr = Pf[kk].merge(Pr, on=kk, how="left")
    assert len(Pr) == len(Pf) and (Pr.same_lane.isna() == Pf.same_lane.isna()).all()
    for P in (Pf, Pr):
        P["n_det_all"] = P.n_det_all.astype(float)
    return D, Pf, Pr


def stage_fit(a):
    import lightgbm as lgb
    import ln2_pairmodel as L2
    D, Pf, Pr = sym_frames()
    c2 = L8.c2_matrix(Pf, D, np.arange(len(Pf)))           # min / max of both detectors: symmetric
    Xf = np.hstack([Pf[LO.FEATURES].to_numpy(np.float32), c2[0]])
    Xr = np.hstack([Pr[LO.FEATURES].to_numpy(np.float32), c2[0]])
    d = (Pf[LO.CUE_COLS + ["log_na", "log_rate_a"]].to_numpy(float) -
         Pr[LO.CUE_COLS + ["log_nb", "log_rate_b"]].to_numpy(float))
    log(f"reversed vs forward cues: {int((np.abs(np.nan_to_num(d)) > 1e-6).any(1).sum()):,} of {len(d):,} pairs differ "
        f"in some cue (asymmetric definitions)")
    lab = Pf.labelled.to_numpy()
    y = Pf.same_lane.to_numpy()
    fo = Pf.fold.to_numpy()
    out = np.zeros(len(Pf))
    per_seed = np.zeros((3, len(Pf)))
    orient = np.zeros((2, len(Pf)))
    for f in range(6):
        te = fo == f
        tr = lab & (fo != f)
        X = np.vstack([Xf[tr], Xr[tr]])
        yy = np.concatenate([y[tr], y[tr]]).astype(int)
        for si, s in enumerate(L8.SEEDS):
            m = lgb.LGBMClassifier(random_state=s, **dict(L8.PARAMS, n_jobs=a.workers)).fit(X, yy)
            pf, pr = m.predict_proba(Xf[te])[:, 1], m.predict_proba(Xr[te])[:, 1]
            per_seed[si, te] = 0.5 * (pf + pr)
            orient[0, te] += pf / 3
            orient[1, te] += pr / 3
            out[te] += 0.5 * (pf + pr) / 3
        log(f"  fold {f} done")
    keep = ["DeviceId", "period", "win", "da", "db", "same_pred", "same_lane", "n_lanes", "fold"]
    Q = Pf[keep].copy()
    Q["p_same"] = out.astype(np.float32)
    Q["p_fwd"] = orient[0].astype(np.float32)
    Q["p_rev"] = orient[1].astype(np.float32)
    Q.to_parquet(O8 / "p_Dsym.parquet", index=False)
    old = pd.read_parquet(L8.OUT / "p_D.parquet")
    assert (old[["DeviceId", "win", "da", "db"]].to_numpy() == Q[["DeviceId", "win", "da", "db"]].to_numpy()).all()
    r = {"orientation_gap_abs_mean": float(np.mean(np.abs(orient[0] - orient[1]))),
         "orientation_gap_p99": float(np.quantile(np.abs(orient[0] - orient[1]), .99)),
         "vs_old_abs_mean": float(np.mean(np.abs(old.p_same.to_numpy() - out)))}
    qq = Q[Q.same_lane.notna()]
    oq = old[old.same_lane.notna()]
    for wg in ["m30", "h6", "h24", "full"]:
        mm = (qq.win.map(L8.wgroup) == wg).to_numpy() & (qq.n_lanes >= 2).to_numpy()
        r[wg] = {"auc_multi_new": L2.auc(qq.same_lane.to_numpy()[mm], qq.p_same.to_numpy()[mm]),
                 "auc_multi_old": L2.auc(oq.same_lane.to_numpy()[mm], oq.p_same.to_numpy()[mm]),
                 "per_seed_new": [L2.auc(Q.same_lane.to_numpy()[Q.same_lane.notna().to_numpy()][mm], per_seed[s][
                     Q.same_lane.notna().to_numpy()][mm]) for s in range(3)]}
    json.dump(r, open(O8 / "fit.json", "w"), indent=1)
    log(json.dumps(r))


# ================================================================================================ decode
def keys_by_win() -> dict:
    K = pd.read_parquet(O8 / "keys.parquet")
    out = {}
    for (dev, per, w), g in K.groupby(["DeviceId", "period", "win"]):
        out[(dev, per, w)] = {int(d): (int(a), int(b), int(c)) for d, a, b, c in
                              zip(g.Detector, g.k_n, g.k_s1, g.k_s2)}
    return out


def dec_job(args):
    """L8.dec_job (func mode only) with the canonical behavioural decode order."""
    dev, Dd, probs, cfgs, keys = args
    dets, phs = [], []
    for name, var, mode, lam, prior in cfgs:
        dec = LO.Decoder(prior, lam=lam)
        for (period, w), g in Dd.groupby(["period", "win"]):
            hours = g.hours.iloc[0]
            gg = g.rename(columns={"pred_phase": "phase_use", "func": "func_use"})
            n_on = dict(zip(gg.Detector.astype(int), gg.n_on))
            pr = probs.get(var, {}).get((period, w), {})
            pt, dt = LO.decode_groups(dev, gg, n_on, hours, pr, dec, keys=keys[(period, w)])
            dt["function"] = gg.set_index(gg.Detector.astype(int)).func_use.reindex(dt.Detector).to_numpy()
            dt["period"], dt["win"], dt["cfg"] = period, w, name
            dets.append(dt)
            if len(pt):
                pt["period"], pt["win"], pt["cfg"] = period, w, name
                phs.append(pt)
    return (pd.concat(dets, ignore_index=True),
            pd.concat(phs, ignore_index=True) if phs else pd.DataFrame())


def run_decodes(D, ptab: pd.DataFrame, cfgs_by_fold, workers, tag):
    import a2_features as A2F
    secs = {(p, n): s for p in A2F.WINDOWS for n, _, s in A2F.WINDOWS[p]}
    D = D.assign(hours=[secs[(p, w)] / 3600.0 for p, w in zip(D.period, D.win)])
    kw = keys_by_win()
    byd = dict(tuple(ptab.groupby("DeviceId")))
    jobs = []
    for dev, Dd in D.groupby("DeviceId"):
        t = byd.get(dev)
        pr = {}
        if t is not None:
            pr["Dsym"] = {(p, w): {(int(x), int(y)): float(z) for x, y, z in zip(h.da, h.db, h.p_same)}
                          for (p, w), h in t.groupby(["period", "win"])}
        ks = {(p, w): kw[(dev, p, w)] for p, w in Dd[["period", "win"]].drop_duplicates().itertuples(index=False)}
        jobs.append((dev, Dd, pr, cfgs_by_fold[int(Dd.fold.iloc[0])], ks))
    DT, PT, t0 = [], [], time.time()
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for i, (x, y) in enumerate(pool.imap_unordered(dec_job, jobs, chunksize=2)):
            DT.append(x)
            if len(y):
                PT.append(y)
            if i % 100 == 0:
                log(f"  [{tag}] {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    return pd.concat(DT, ignore_index=True), pd.concat(PT, ignore_index=True)


def _priors():
    det, ph, T = L8.truth_tabs()
    D = L8.load_dets()
    sigs = set(ph.DeviceId)
    D9 = D[(D.period == "stg") & D.win.isin(L8.TRAIN_WINS) & D.DeviceId.isin(sigs)].copy()
    fsig = D9.groupby("DeviceId").fold.first()
    pri = {f: L8.span_priors(det, ph, D9.rename(columns={"Detector": "det"}), set(fsig.index[fsig != f]))[0]
           for f in range(6)}
    return det, ph, T, D, D9, pri


def stage_decode(a):
    det, ph, T, D, D9, pri = _priors()
    q = pd.read_parquet(O8 / "p_Dsym.parquet")
    q = q[q.same_pred & (q.period == "stg") & q.win.isin(L8.TRAIN_WINS) & q.DeviceId.isin(set(ph.DeviceId))]
    grid = [f"D.func@{lam:g}" for lam in LAMS]
    cfgs = {f: [(g, "Dsym", "func", float(g.split("@")[1]), pri[f]) for g in grid] for f in range(6)}
    DT, PT = run_decodes(D9, q, cfgs, a.workers, "sept")
    DT.to_parquet(O8 / "sept_dec_det.parquet", index=False)
    PT.to_parquet(O8 / "sept_dec_ph.parquet", index=False)
    res = {}
    for g in grid:
        NL, PR, DX = L8.eval_cfg(DT[DT.cfg == g], PT[PT.cfg == g], D9, det, ph, T)
        ob = L8.obj_by_fold(NL, PR, DX)
        res[g] = {f: (ob[f]["nl_exact"] or 0) + (ob[f]["pair_acc_multi"] or 0) for f in range(6)}
        log(f"  {g}: {json.dumps(res[g])}")
    pick = {"D.func": {f: max(grid, key=lambda g: sum(res[g][k] for k in range(6) if k != f)) for f in range(6)}}
    old = json.load(open(L8.OUT / "decode_pick.json"))["pick"]["D.func"]
    log(f"picked {[pick['D.func'][f] for f in range(6)]} (note 58: {[old[str(f)] for f in range(6)]})")
    json.dump({"grid_obj": res, "pick": pick}, open(O8 / "decode_pick.json", "w"), indent=1)


def _sept_rows(DT, PT, pk, D9, det, ph, T, fold):
    import ln3_decode as L3
    m = DT.cfg == DT.DeviceId.map(fold).astype(int).map({int(f): g for f, g in pk.items()})
    mp = PT.cfg == PT.DeviceId.map(fold).astype(int).map({int(f): g for f, g in pk.items()})
    NL, PR, DX = L8.eval_cfg(DT[m], PT[mp], D9, det, ph, T)
    h = NL.n_lanes_p.notna()
    rows = {"nl": NL[h].assign(ok=(NL.n_lanes_p == NL.n_lanes)[h]),
            "lane": DX[DX.covered].assign(ok=DX[DX.covered].exact.astype(float)),
            "pair": PR[PR.covered & (PR.n_lanes >= 2)].assign(
                ok=(PR.pred_same == (PR.same_lane == 1))[PR.covered & (PR.n_lanes >= 2)].astype(float))}
    return rows, L3.score(NL, PR, DX)


def stage_laneeval(a):
    det, ph, T, D, D9, pri = _priors()
    fold = D9.groupby("DeviceId").fold.first()
    new = _sept_rows(pd.read_parquet(O8 / "sept_dec_det.parquet"), pd.read_parquet(O8 / "sept_dec_ph.parquet"),
                     json.load(open(O8 / "decode_pick.json"))["pick"]["D.func"], D9, det, ph, T, fold)
    pk = json.load(open(L8.OUT / "decode_pick.json"))["pick"]["D.func"]
    old = _sept_rows(pd.read_parquet(L8.OUT / "sept_dec_det.parquet"), pd.read_parquet(L8.OUT / "sept_dec_ph.parquet"),
                     pk, D9, det, ph, T, fold)
    res = {"old": old[1], "new": new[1], "ci_pt": {}}
    for k in ("nl", "lane", "pair"):
        d0, d1 = old[0][k], new[0][k]
        res["ci_pt"][k] = [round(100 * (d1.ok.mean() - d0.ok.mean()), 2)] + L8.boot(d0, d1)
    json.dump(res, open(O8 / "laneeval.json", "w"), indent=1, default=str)
    log(f"old {old[1]}\nnew {new[1]}\nnew - old (pt [CI]): {res['ci_pt']}")


def stage_full(a):
    det, ph, T, D, D9, pri = _priors()
    pk = json.load(open(O8 / "decode_pick.json"))["pick"]["D.func"]
    q = pd.read_parquet(O8 / "p_Dsym.parquet", columns=["DeviceId", "period", "win", "da", "db", "p_same", "same_pred"])
    q = q[q.same_pred]
    cfgs = {f: [("D.func", "Dsym", "func", float(pk[str(f)].split("@")[1]), pri[f])] for f in range(6)}
    DT, PT = run_decodes(D, q, cfgs, a.workers, "full")
    DT = DT.merge(D[["DeviceId", "period", "win", "Detector", "n_on"]], on=["DeviceId", "period", "win", "Detector"],
                  how="left")
    DT.to_parquet(O8 / "lanes_D.func.parquet", index=False)
    PT.to_parquet(O8 / "lanes_D.func_ph.parquet", index=False)
    # how much moved vs note 58's D.func
    o = pd.read_parquet(L8.OUT / "lanes_D.func.parquet")
    k = ["DeviceId", "period", "win", "Detector"]
    m = o.merge(DT, on=k, suffixes=("_o", "_n"))
    lc = (m.lane_conf_o.fillna(-1) - m.lane_conf_n.fillna(-1)).abs()
    log(f"{len(DT):,} det-windows (old {len(o):,}); lanes string changed {(m.lanes_o != m.lanes_n).mean():.4f}; "
        f"lane_conf |diff| > .001 {(lc > .001).mean():.4f}, max {lc.max():.3f}; gate side (.9) flips "
        f"{((m.lane_conf_o.fillna(0) >= .9) != (m.lane_conf_n.fillna(0) >= .9)).mean():.4f}")


# ================================================================================================ pick inputs
def frame_D77(run=None, lanes_tag=None):
    import s59_step6 as S59
    S59.LANES_D = O8 / "lanes_D.func.parquet"
    return S59.frame_D(run, lanes_tag)


def ln6_work(args):
    """ln6_pick.work without the per-window health_core run (its 'health' column is not read downstream)."""
    import ln6_pick as L6
    L6.EVR = {"stg": Path("__none__"), "dec": Path("__none__")}
    return L6.work(args)


def stage_pick(a):
    import a2_features as A2F
    import ln6_pick as L6
    import ln7_stackhealth as L7
    assert L6.ORDER_FREE
    L6.frame = frame_D77
    k = frame_D77()
    big = {p: {n for n, _, s in A2F.WINDOWS[p] if s >= L6.MIN_SECS} for p in A2F.WINDOWS}
    k = k[[w in big[p] for p, w in zip(k.period, k.win)]]
    from multiprocessing import Pool
    f6 = L6.OUT / f"ln6_pick_{TAG}.parquet"
    if not f6.exists():
        jobs = [(dev, per, g[["Detector", "win", "pred_phase", "func", "lanes"] + [f"P_{c}" for c in L6.C7]].copy())
                for (dev, per), g in k.groupby(["DeviceId", "period"])]
        log(f"ln6: {len(jobs)} signal-periods")
        t0 = time.time()
        with Pool(a.workers) as pool:
            out = list(pool.imap_unordered(ln6_work, jobs, chunksize=1))
        D = pd.concat(out, ignore_index=True)
        D.to_parquet(f6, index=False)
        log(f"ln6 wrote {len(D):,}; span {int(D.span.sum())} ({time.time()-t0:.0f}s)")
    if not (L6.OUT / f"ln7_stackhealth_{TAG}.parquet").exists():
        L7.main(None, None, TAG, a.workers)
    k6 = ["DeviceId", "period", "win", "Detector"]
    for f in ("ln6_pick", "ln7_stackhealth"):
        o = pd.read_parquet(L6.OUT / f"{f}_D229.parquet")
        n = pd.read_parquet(L6.OUT / f"{f}_{TAG}.parquet")
        m = o.merge(n, on=k6, how="outer", suffixes=("_o", "_n"), indicator=True)
        r = {"rows_old": len(o), "rows_new": len(n), "unmatched": int((m._merge != "both").sum())}
        m = m[m._merge == "both"]
        for c in ("span", "track", "chi", "clus_n", "sp_span1"):
            if c + "_o" in m:
                x, y = m[c + "_o"], m[c + "_n"]
                if x.dtype == object or x.dtype == bool:
                    r[c] = int((x.astype(str) != y.astype(str)).sum())
                else:
                    r[c] = int(((x - y).abs() > 1e-9).sum() + (x.isna() != y.isna()).sum())
        for c in ("span_peers", "coloc_peers"):
            if c + "_o" in m:
                st = lambda s: frozenset(v for v in str(s).split(",") if v) if isinstance(s, str) else frozenset()  # noqa
                r[c] = int((m[c + "_o"].map(st) != m[c + "_n"].map(st)).sum())
        log(f"{f}: {r}")


def hff_work(args):
    import ln7_stackhealth as L7
    dev, period, g = args
    return L7.work((dev, period, g.assign(func="", lanes="")))


def stage_hff(a):
    import a2_features as A2F
    import s59_step6 as S59
    k = S59.frame_keys()
    big = {(p, n) for p in ("dec", "stg") for n, _, s in A2F.WINDOWS[p] if s >= 1800}
    k = k[[(p, w) in big for p, w in zip(k.period, k.win)]]
    jobs = [(dev, per, g[["Detector", "win", "pred_phase"]].copy()) for (dev, per), g in k.groupby(["DeviceId", "period"])]
    log(f"hff: {len(jobs)} signal-periods")
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers) as pool:
        S = [s for s in pool.imap_unordered(hff_work, jobs, chunksize=1) if len(s)]
    S = pd.concat(S, ignore_index=True)
    S.to_parquet(F77 / "stackhealth_ff.parquet", index=False)
    o = pd.read_parquet(S59.OUT / "stackhealth_ff.parquet")
    k6 = ["DeviceId", "period", "win", "Detector"]
    m = o.merge(S, on=k6, how="outer", suffixes=("_o", "_n"), indicator=True)
    log(f"hff wrote {len(S):,} (old {len(o):,}); unmatched {int((m._merge != 'both').sum())}; clus_n changed "
        f"{int((m.clus_n_o.fillna(-1) != m.clus_n_n.fillna(-1)).sum())}; chi changed "
        f"{int(((m.chi_o - m.chi_n).abs() > 1e-9).sum())} ({time.time()-t0:.0f}s)")


def stage_twins(a):
    import s62_short as S62
    import s59_step6 as B
    import ln6_pick as L6
    assert L6.ORDER_FREE
    k = B.frame_keys()
    k = k[k.pred_phase.notna() & k.wgroup.isin(["m5", "m10"])]
    jobs = [(dev, per, g[["Detector", "win", "pred_phase"]].copy()) for (dev, per), g in k.groupby(["DeviceId", "period"])]
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers) as pool:
        F = pd.concat(list(pool.imap_unordered(S62.pair_work, jobs, chunksize=2)), ignore_index=True)
    F.to_parquet(F77 / "pairs_short.parquet", index=False)
    o = pd.read_parquet(S62.PAIRS_F)
    key = lambda d: (d.DeviceId + "|" + d.period + "|" + d.win + "|" +  # noqa: E731
                     np.minimum(d.da, d.db).astype(str) + "|" + np.maximum(d.da, d.db).astype(str))
    tw = lambda d: set(key(d[(np.minimum(d.m_ab, d.m_ba) >= 0.4) & (d.lag.abs() <= 1.0)]))  # noqa: E731
    to, tn = tw(o), tw(F)
    mm = o.assign(k=key(o)).merge(F.assign(k=key(F)), on="k", suffixes=("_o", "_n"))
    log(f"twins: {len(F):,} pairs (old {len(o):,}); twin pairs old {len(to):,} new {len(tn):,}, only old "
        f"{len(to - tn)}, only new {len(tn - to)}; orientation flipped {int((mm.da_o != mm.da_n).sum()):,}, "
        f"|lag| changed {int((mm.lag_o.abs() - mm.lag_n.abs()).abs().gt(1e-9).sum())} ({time.time()-t0:.0f}s)")


# ================================================================================================ stack / compare
_SF = {}


def scoring_frame77(pick_tag=None):
    """S59.scoring_frame with the note-77 lanes and pick inputs."""
    if "x" not in _SF:
        import atspm_score as S
        import atspm_pick55 as AP
        import t57_function as T57
        fr = S.load(T57.BASE_RUN)
        fr = S.attach_lanes(fr, "../final_v3_work/f77/ln8/lanes_D.func")
        fr = S.attach_pick_inputs(fr, f"ln6_pick_{TAG}", f"ln7_stackhealth_{TAG}")
        _SF["x"] = (fr, AP.score_rows(fr))
    return _SF["x"]


def lane_ctx(fr, lanes_f: Path) -> np.ndarray:
    """phase_n_lanes, phase_n_lanes_conf, lane_conf per frame row (cand64.attach_side recipe)."""
    k = ["DeviceId", "Detector", "period", "win"]
    L = pd.read_parquet(lanes_f, columns=k + ["phase", "lane_conf"]).rename(columns={"phase": "lane_phase"})
    L = L.astype({"Detector": fr.Detector.dtype})
    x = fr[k].merge(L, on=k, how="left")
    Lp = pd.read_parquet(str(lanes_f).replace(".parquet", "_ph.parquet"),
                         columns=["DeviceId", "period", "win", "phase", "n_lanes", "n_lanes_conf"])
    Lp = Lp.rename(columns={"phase": "lane_phase", "n_lanes": "phase_n_lanes", "n_lanes_conf": "phase_n_lanes_conf"})
    x = x.merge(Lp, on=["DeviceId", "period", "win", "lane_phase"], how="left")
    assert len(x) == len(fr)
    return x[["phase_n_lanes", "phase_n_lanes_conf", "lane_conf"]].to_numpy(float)


def health_feats77(k):
    """S59.health_feats (function-free health context) with the note-77 stack-health table."""
    import s59_step6 as S59
    real = pd.read_parquet

    def rp(p, *a, **kw):
        if Path(p).name == "stackhealth_ff.parquet":
            p = F77 / "stackhealth_ff.parquet"
        return real(p, *a, **kw)
    pd.read_parquet = rp
    try:
        return S59.health_feats(k)
    finally:
        pd.read_parquet = real


def setup_new():
    """s74.setup('x69_siba') on the note-77 frame, trees = note-76 arm c (f76/function_c), context cache prefilled."""
    sys.path.insert(0, str(rpath.CODE / "evaluation"))
    sys.path.insert(0, str(rpath.CODE / "final76"))
    assert os.environ.get("F76_ARM") == "c", "run with F76_ARM=c (trees = note-76 arm c)"
    import f76_function as F
    s74 = F._s74()
    import s59_step6 as S59
    import s62_short as S62
    import s67_decider as S
    S59.scoring_frame = scoring_frame77
    S62.PAIRS_F = F77 / "pairs_short.parquet"
    s74.OUT = F77 / "s74"
    E = s74.setup("x69_siba")
    fr = E["fr"]
    S._CTX.clear()
    S._CTX["H"] = health_feats77(fr[S59.KEY].reset_index(drop=True)).to_numpy(np.float32)
    S._CTX["ln"] = lane_ctx(fr, O8 / "lanes_D.func.parquet")
    return s74, S, E


def stage_stack(a):
    s74, S, E = setup_new()
    s74.cmd_stack(argparse.Namespace(name="f77", main="x69_siba", extra=""))


def stage_compare(a):
    import cand64 as C
    import s59_step6 as S59
    import s62_short as S62
    # baseline: the note-76 champion (f76c stack, note-58 lanes, D229 pick inputs, old twins)
    sys.path.insert(0, str(rpath.CODE / "evaluation"))
    sys.path.insert(0, str(rpath.CODE / "final76"))
    assert os.environ.get("F76_ARM") == "c", "run with F76_ARM=c (trees = note-76 arm c)"
    import f76_function as F
    s74 = F._s74()
    import s67_decider as S
    E0 = s74.setup("x69_siba")
    lc0 = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["lane_conf"]).lane_conf.to_numpy(float)
    ok0 = s74.gate_ok(E0, s74.load_stack("f76c"), lc0)
    sig0 = E0["fr"].DeviceId.to_numpy()
    rows0 = {k: v.copy() for k, v in E0["rows"].items()}
    fr0 = E0["fr"][["DeviceId", "Detector", "period", "win", "wgroup", "truth_v3s"]].copy()
    # new
    S59.scoring_frame = scoring_frame77
    S62.PAIRS_F = F77 / "pairs_short.parquet"
    E1 = s74.setup("x69_siba")
    fr = E1["fr"]
    assert (fr[["DeviceId", "Detector", "period", "win"]].to_numpy() == fr0[["DeviceId", "Detector", "period", "win"]]
            .to_numpy()).all()
    for k in rows0:
        assert (rows0[k] == E1["rows"][k]).all(), k
    lc1 = lane_ctx(fr, O8 / "lanes_D.func.parquet")[:, 2]
    s74.OUT = F77 / "s74"
    ok1 = s74.gate_ok(E1, s74.load_stack("f77"), lc1)
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    res = {}
    for s in ("E", "R"):
        A, B = ok0[s], ok1[s]
        sc = ~np.isnan(A) & ~np.isnan(B)
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            m = sc & fr.wgroup.isin(fams).to_numpy()
            res[f"{s}_{pool}"] = {"n": int(m.sum()), "f76c": C.acc_ci(A[m], sig0[m]), "f77": C.acc_ci(B[m], sig0[m]),
                                  "b_minus_a": C.delta_ci(A[m], B[m], sig0[m])}
        m = sc & fr.wgroup.isin(C.GE30).to_numpy()
        res[f"{s}_ge30_by_class"] = {c: {"n": int((m & (cl == c)).sum()),
                                         "d": C.delta_ci(A[m & (cl == c)], B[m & (cl == c)], sig0[m & (cl == c)])}
                                     for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        res[f"{s}_ge30_by_fold"] = {int(k): round(100 * float(np.mean(B[m & (fr.fold == k).to_numpy()])
                                                               - np.mean(A[m & (fr.fold == k).to_numpy()])), 3)
                                    for k in range(6)}
        res[f"{s}_rows_changed_ge30"] = int((A[m] != B[m]).sum())
    # per-seed new stacks (seed noise reference)
    seeds = {}
    for sd in (0, 1, 2):
        P = np.load(F77 / "s74" / "f77" / f"stack_ctx_s{sd}.npy").astype(float)
        o = s74.gate_ok(E1, P, lc1)["E"]
        m = ~np.isnan(o) & fr.wgroup.isin(C.GE30).to_numpy()
        seeds[sd] = round(float(np.mean(o[m])), 4)
    res["E_ge30_new_per_seed"] = seeds
    json.dump(res, open(F77 / "compare.json", "w"), indent=1, default=str)
    for k, v in res.items():
        log(f"{k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cues", "fit", "decode", "laneeval", "full", "pick", "hff", "twins", "stack",
                                      "compare"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    F77.mkdir(parents=True, exist_ok=True)
    O8.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
