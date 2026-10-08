"""Note 59 (validation plan step 6): pending function additions, each vs the CURRENT set-up.

Current set-up (baseline): function = note-57 arm (229 features, 3 seeds, blend phase input = frame v6e); lanes = joint
pair model D (note 58, `lanes/ln8/lanes_D.func.parquet`); decode = stack-scoped pick, stack_loser "nonatspm_ap" (default),
pick inputs REBUILT here on D lanes + the 229-feature arm (`lanes/ln6_pick_D229`, `lanes/ln7_stackhealth_D229`).
Scoring = ATSPM-only stack-aware score (note 54/55), everything / realistic, paired signal bootstrap; six folds OOF.

Additions tested (each = one new feature family / class / weighting on top of the baseline):
  (a) lanes   lane features of the detector (phase n_lanes, lanes spanned, lane rank, lane-mates' volumes and function
              probabilities). Lanes come from D, which itself uses function features + OOF function probabilities, so the
              TRAINING rows of outer fold k get NESTED lanes: function refit on folds not in {k, j} -> probs for fold j,
              D pair model refit on folds not in {k, j} on those probs -> pairs of fold j -> lane decode of fold j. The
              test rows (fold k) get the ordinary OOF D lanes. No fold-k label reaches fold k's features.
  (b) eta     radar ETA zones (print subtype `eta`) as an 8th, non-ATSPM class; `etaadv` = eta + radar advance-presence
              zones. Decoded with Other := max(Other, ETA) (= 8-class argmax); judged by the ATSPM score only.
  (c) health  health_core v5 on the window + note-56 stack-relative statistics, computed FUNCTION-FREE (predicted phase
              only, no function probabilities, no lanes) so no circularity / nesting is needed. Windows >= 30 min.
  (d) offpeak per-detector behaviour computed only on the off-peak part of the sample (outside weekday 06:30-09:00 and
              15:30-18:30), NaN when the window has < 15 min of it; `opw` = training rows weighted 1 + off-peak share.
Every feature family gets a shuffled control (`<cfg>_shuf`: the new columns permuted across rows).

    python s59_step6.py pick                 # rebuild ln6 / ln7 pick inputs on D lanes
    python s59_step6.py inner                # nested function OOF (outer k, inner j), 3 seeds
    python s59_step6.py npair                # nested D pair models + lane decodes
    python s59_step6.py lanefeat | health | offpeak
    python s59_step6.py fit --cfg lanes --seeds 0
    python s59_step6.py score --cfgs lanes@s0,lanes_shuf@s0 --base base@s0 --tag screen
Out: %DC_WORK%/s59/. locked_v2 asserted absent everywhere. CPU only (6 threads).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import v3_retrain as V  # noqa: E402

DCW = V.DCW
OUT = DCW / "s59"
FUNC_DIR = DCW / "trees57" / "function" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx"
LANES_D = DCW / "lanes" / "ln8" / "lanes_D.func.parquet"
LANES_D_PH = DCW / "lanes" / "ln8" / "lanes_D.func_ph.parquet"
PICK_TAG = "D229"
VAR = "first.all.wi"
SEEDS = (0, 1, 2)
C7 = list(V.C7)
KEY = V.KEY


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return V.locked_signals()


def setup():
    import t57_function as T57
    T57.setup("v6e")


def frame_keys(extra=()) -> pd.DataFrame:
    setup()
    k = pd.read_parquet(V.FEATS, columns=KEY + ["wgroup", "fold", "pred_phase", "det_n_on"] + list(extra))
    k["DeviceId"] = k.DeviceId.str.lower()
    assert not k.DeviceId.isin(locked()).any()
    return k


def std_probs(k: pd.DataFrame, seeds=SEEDS, d: Path = FUNC_DIR) -> np.ndarray:
    """OOF probabilities of a config dir (frame row order), mean over `seeds`."""
    P = None
    fo = k.fold.to_numpy()
    for s in seeds:
        for f in range(6):
            x = np.load(d / f"P_{VAR}_s{s}_f{f}.npy")
            if P is None:
                P = np.zeros((len(k), x.shape[1]), np.float32)
            P[fo == f] += x
    return P / len(seeds)


# ------------------------------------------------------------------------------------------------ pick inputs on D
def frame_D(run=None, lanes_tag=None) -> pd.DataFrame:
    """ln6_pick.frame() with the note-57 229-feature arm probabilities and D lanes."""
    k = frame_keys()
    k = k.drop(columns=["wgroup"])
    P = std_probs(k)
    for i, c in enumerate(C7):
        k[f"P_{c}"] = P[:, i]
    k["func"] = np.array(C7, object)[P.argmax(1)]
    Ln = pd.read_parquet(LANES_D, columns=["DeviceId", "Detector", "period", "win", "phase", "lanes"])
    Ln = Ln.astype({"Detector": k.Detector.dtype})
    k = k.merge(Ln, on=["DeviceId", "Detector", "period", "win"], how="left")
    k["lanes"] = k.lanes.where(k.phase.eq(k.pred_phase), "").fillna("")
    return k


def stage_pick(a):
    import ln6_pick as L6
    import ln7_stackhealth as L7
    L6.frame = frame_D
    t0 = time.time()
    if not (DCW / "lanes" / f"ln6_pick_{PICK_TAG}.parquet").exists():
        L6.main(None, None, PICK_TAG, a.workers)
    log(f"ln6 done ({time.time()-t0:.0f}s)")
    if not (DCW / "lanes" / f"ln7_stackhealth_{PICK_TAG}.parquet").exists():
        L7.main(None, None, PICK_TAG, a.workers)
    log(f"ln7 done ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ training frame
_TR = {}


def load_train():
    """(frame with the 229 columns, cols, label table, y, ok) - the note-57 arm's training set, frame row order."""
    if "x" in _TR:
        return _TR["x"]
    setup()
    cols = json.load(open(FUNC_DIR / "cols.json"))["cols"]
    meta = KEY + ["wgroup", "fold", "pred_phase", "det_n_on", "health_flag"]
    fr = pd.read_parquet(V.FEATS, columns=meta + [c for c in cols if c not in meta])
    assert not fr.DeviceId.str.lower().isin(locked()).any()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, VAR)
    _TR["x"] = (fr, cols, lab, y, ok)
    return _TR["x"]


def lgb_fit(X, yi, w, trm, vam, tem, seed, threads, ncls):
    import lightgbm as lgb
    import a2_model as A2
    prm = dict(A2.FUNC_PARAMS, n_jobs=threads, num_class=ncls, seed=seed, bagging_seed=seed + 1,
               feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    n = prm.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n, **prm)
    m.fit(X[trm], yi[trm], sample_weight=w[trm], eval_set=[(X[vam], yi[vam])], eval_sample_weight=[w[vam]],
          eval_metric="multi_logloss", callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    return m.predict_proba(X[tem]).astype(np.float32), int(m.best_iteration_ or 0)


# ------------------------------------------------------------------------------------------------ (a) nested lanes
def stage_inner(a):
    """function refit on folds not in {k, j} (early stopping on a third fold v) -> probabilities for fold j."""
    fr, cols, lab, y, ok = load_train()
    yi = pd.Series(y).map({c: i for i, c in enumerate(C7)}).fillna(-1).astype(int).to_numpy()
    X = fr[cols].to_numpy(np.float32)
    fo = fr.fold.to_numpy()
    w = np.ones(len(fr))
    d = OUT / "inner"
    d.mkdir(parents=True, exist_ok=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        for k in range(6):
            for j in range(6):
                if j == k:
                    continue
                f = d / f"P_k{k}_j{j}_s{s}.npy"
                if f.exists():
                    continue
                v = next(x % 6 for x in range(j + 1, j + 7) if x % 6 not in (k, j))
                t0 = time.time()
                trm = ok & ~np.isin(fo, [k, j, v])
                P, nt = lgb_fit(X, yi, w, trm, ok & (fo == v), fo == j, s, a.threads, 7)
                np.save(f, P)
                log(f"inner s{s} k{k} j{j} (val {v}): {nt} trees, {time.time()-t0:.0f}s")


def inner_probs(k: int, fo: np.ndarray, seeds=SEEDS) -> np.ndarray:
    """frame-aligned probabilities for outer fold k: rows of folds j != k from the nested fits; fold-k rows = NaN."""
    P = np.full((len(fo), 7), np.nan, np.float32)
    for j in range(6):
        if j == k:
            continue
        P[fo == j] = np.mean([np.load(OUT / "inner" / f"P_k{k}_j{j}_s{s}.npy") for s in seeds], 0)
    return P


def dets_with(P: np.ndarray, k: pd.DataFrame) -> pd.DataFrame:
    """ln8 dets.parquet + predicted function / probabilities from P (frame-aligned) + phase-group sizes."""
    import lane_output as LO
    D = pd.read_parquet(DCW / "lanes" / "ln8" / "dets.parquet")
    kk = k[["DeviceId", "period", "win", "Detector"]].copy()
    kk["Detector"] = kk.Detector.astype(int)
    ok = ~np.isnan(P[:, 0])
    kk = kk[ok]
    Pk = P[ok]
    kk["func"] = np.array(LO.C7, object)[Pk.argmax(1)]
    for i, c in enumerate(LO.C7):
        kk[f"P_{c}"] = Pk[:, i]
    D = D.merge(kk, on=["DeviceId", "period", "win", "Detector"], how="inner")
    el = (D.n_on >= LO.MIN_ON) & D.pred_phase.notna()
    gk = ["DeviceId", "period", "win", "pred_phase"]
    D["n_det_all"] = np.where(el, D.assign(e=el).groupby(gk).e.transform("sum"), np.nan)
    nb = el & (D.func != "Bike")
    D["n_det_nb"] = np.where(el, D.assign(e=nb).groupby(gk).e.transform("sum"), np.nan)
    return D


def stage_npair(a):
    """per outer k: D pair model refit on folds not in {k, j} with the nested probabilities, pairs of fold j, lane decode
    of fold j (lam = note 58's pick for fold j, span prior from signals not in {k, j})."""
    import lightgbm as lgb
    import lane_output as LO
    import ln8_validate as L8
    k0 = frame_keys()
    fo = k0.fold.to_numpy()
    cues = pd.read_parquet(DCW / "lanes" / "ln8" / "cues.parquet")
    cues["_fold"] = cues.DeviceId.map(k0.groupby("DeviceId").fold.first()).fillna(-1).astype(int)
    assert (cues._fold >= 0).mean() > 0.99, "cues signals missing from the frame"
    pk = json.load(open(DCW / "lanes" / "ln8" / "decode_pick.json"))["pick"]["D.func"]
    det, ph, T = L8.truth_tabs()
    sigs = set(ph.DeviceId)
    d = OUT / "nlanes"
    d.mkdir(parents=True, exist_ok=True)
    for k in range(6):
        if (d / f"lanes_k{k}.parquet").exists():
            continue
        t0 = time.time()
        D = dets_with(inner_probs(k, fo), k0)
        P = L8.pair_frame(cues[cues._fold != k].drop(columns="_fold").reset_index(drop=True), D)
        P["n_det_all"] = P.n_det_all.astype(float)
        assert P.fa.notna().all(), "pair without nested function"
        lab = P.labelled.to_numpy()
        pf = P.fold.to_numpy()
        names = None
        Xl, il = None, np.flatnonzero(lab)
        c2 = L8.c2_matrix(P, D, il)
        Xl = np.hstack([P.loc[il, LO.FEATURES].to_numpy(np.float32), c2[0]])
        yl = P.same_lane.to_numpy()[il].astype(int)
        names = LO.FEATURES + c2[1]
        out = []
        for j in range(6):
            if j == k:
                continue
            tr = pf[il] != j
            ij = np.flatnonzero(pf == j)
            c2j = L8.c2_matrix(P, D, ij)
            Xj = np.hstack([P.loc[ij, LO.FEATURES].to_numpy(np.float32), c2j[0]])
            p = np.zeros(len(ij))
            for s in SEEDS:
                m = lgb.LGBMClassifier(random_state=s, **dict(L8.PARAMS, n_jobs=a.threads)).fit(Xl[tr], yl[tr])
                p += m.predict_proba(Xj)[:, 1] / len(SEEDS)
            q = P.loc[ij, ["DeviceId", "period", "win", "da", "db", "same_pred"]].copy()
            q["p_same"] = p.astype(np.float32)
            out.append(q)
            log(f"  k{k} j{j}: pair model on {int(tr.sum()):,} labelled, {len(ij):,} pairs ({time.time()-t0:.0f}s)")
        Q = pd.concat(out, ignore_index=True)
        Q.to_parquet(d / f"pairs_k{k}.parquet", index=False)
        # lane decode of every fold j != k
        D9 = D[(D.period == "stg") & D.win.isin(L8.TRAIN_WINS) & D.DeviceId.isin(sigs)]
        fsig = D9.groupby("DeviceId").fold.first()
        Dsp = D9.rename(columns={"Detector": "det"})
        cfgs = {}
        for j in range(6):
            if j == k:
                continue
            pri = L8.span_priors(det, ph, Dsp, set(fsig.index[~fsig.isin([k, j])]))
            cfgs[j] = [("D.func", "D", "func", float(pk[str(j)].split("@")[1]), pri[0])]
        DT, PT = L8.run_decodes(D, {"D": Q[Q.same_pred]}, cfgs, a.workers, f"k{k}")
        DT = DT.merge(D[["DeviceId", "period", "win", "Detector", "n_on"]], on=["DeviceId", "period", "win", "Detector"],
                      how="left")
        assert not DT.DeviceId.isin(locked()).any()
        DT.to_parquet(d / f"lanes_k{k}.parquet", index=False)
        PT.to_parquet(d / f"lanes_k{k}_ph.parquet", index=False)
        log(f"k{k}: {len(DT):,} det-windows, lane known {(DT.lanes != '').mean():.3f} ({time.time()-t0:.0f}s)")


LANE_COLS = ["ln_nl", "ln_nl_conf", "ln_span", "ln_conf", "ln_rank", "ln_ndet", "ln_nmates", "ln_volrank",
             "ln_lr_max", "ln_mA", "ln_mP", "ln_mC", "ln_mY"]


def lane_feats(k: pd.DataFrame, DT: pd.DataFrame, PT: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    """frame-aligned lane features (NaN where the detector has no lane on its predicted phase)."""
    kk = k[KEY + ["pred_phase", "det_n_on"]].copy()
    kk["_i"] = np.arange(len(kk))
    for i, c in enumerate(C7[:4]):
        kk[f"P{i}"] = P[:, i]
    L = DT[["DeviceId", "Detector", "period", "win", "phase", "lanes", "n_lanes_spanned", "lane_conf"]]
    L = L[L.lanes != ""].astype({"Detector": kk.Detector.dtype})
    x = kk.merge(L, on=["DeviceId", "Detector", "period", "win"], how="inner")
    x = x[x.phase == x.pred_phase]
    Pt = PT[["DeviceId", "period", "win", "phase", "n_lanes", "n_lanes_conf", "n_detectors"]]
    x = x.merge(Pt, on=["DeviceId", "period", "win", "phase"], how="left")
    x["lmin"] = x.lanes.map(lambda s: min(int(v) for v in s.split(",")))
    e = x[["_i", "DeviceId", "period", "win", "phase", "lanes", "det_n_on", "P0", "P1", "P2", "P3"]].copy()
    e["lane"] = e.lanes.str.split(",")
    e = e.explode("lane")
    m = e.merge(e[["DeviceId", "period", "win", "phase", "lane", "_i", "det_n_on", "P0", "P1", "P2", "P3"]],
                on=["DeviceId", "period", "win", "phase", "lane"], suffixes=("", "_m"))
    m = m[m._i != m._i_m].drop_duplicates(["_i", "_i_m"])
    m["big"] = (m.det_n_on_m > m.det_n_on).astype(int)
    g = m.groupby("_i")
    agg = pd.DataFrame({"ln_nmates": g.size(), "mx_on": g.det_n_on_m.max(), "n_bigger": g.big.sum(),
                        "ln_mA": g.P0_m.max(), "ln_mP": g.P1_m.max(), "ln_mC": g.P2_m.max(), "ln_mY": g.P3_m.max()})
    x = x.join(agg, on="_i")
    x["ln_nmates"] = x.ln_nmates.fillna(0)
    x["ln_volrank"] = x.n_bigger.fillna(0) + 1
    x["ln_lr_max"] = np.log1p(x.det_n_on) - np.log1p(x.mx_on)
    x["ln_nl"], x["ln_nl_conf"], x["ln_span"], x["ln_conf"] = x.n_lanes, x.n_lanes_conf, x.n_lanes_spanned, x.lane_conf
    x["ln_rank"] = x.lmin / x.n_lanes
    x["ln_ndet"] = x.n_detectors
    out = np.full((len(kk), len(LANE_COLS)), np.nan, np.float32)
    out[x._i.to_numpy()] = x[LANE_COLS].to_numpy(np.float32)
    return pd.DataFrame(out, columns=LANE_COLS)


def stage_lanefeat(a):
    k = frame_keys()
    fo = k.fold.to_numpy()
    d = OUT / "lanefeat"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    F = lane_feats(k, pd.read_parquet(LANES_D), pd.read_parquet(LANES_D_PH), std_probs(k))
    F.to_parquet(d / "std.parquet", index=False)
    log(f"std: rows with lanes {F.ln_nl.notna().mean():.3f} ({time.time()-t0:.0f}s)")
    for kk in range(6):
        P = inner_probs(kk, fo)
        P = np.nan_to_num(P)
        F = lane_feats(k, pd.read_parquet(OUT / "nlanes" / f"lanes_k{kk}.parquet"),
                       pd.read_parquet(OUT / "nlanes" / f"lanes_k{kk}_ph.parquet"), P)
        F.loc[fo == kk] = np.nan
        F.to_parquet(d / f"k{kk}.parquet", index=False)
        log(f"k{kk}: rows with lanes {F.ln_nl.notna().mean():.3f} ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ (c) health, function-free
HF_COLS = ["hf_score", "hf_status", "hf_nfam", "hf_s_dropout", "hf_s_stuck", "hf_s_chatter", "hf_s_rapid", "hf_s_volume",
           "hf_s_level", "hf_s_choppy", "hf_chi", "hf_surge", "hf_drop", "hf_chat", "hf_rel_chi", "hf_clus_n"]


def health_work(args):
    dev, period, g = args
    import a2_features as A2F
    import health_core as hc
    import ln6_pick as L6
    import ln7_stackhealth as L7
    import pyarrow.dataset as ds
    p = L6.EVR[period] / f"DeviceId={dev}"
    ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas() if p.is_dir() else None
    rows = []
    for name, t0w, secs in A2F.WINDOWS[period]:
        if secs < L6.MIN_SECS or ev is None:
            continue
        w = g[g.win == name]
        if w.empty:
            continue
        ph = {int(d): float(v) for d, v in zip(w.Detector, w.pred_phase)}
        try:
            h = hc.health(ev, pd.Timestamp(t0w), pd.Timestamp(t0w) + pd.Timedelta(seconds=secs), None, ph, None, None, None)
        except Exception as e:                       # noqa: BLE001
            print(f"health failed {dev} {period} {name}: {e}", flush=True)
            continue
        h = h.assign(DeviceId=dev, period=period, win=name)
        rows.append(h.reindex(columns=["DeviceId", "period", "win", "detector", "health_score", "status", "n_families", "s_dropout",
                       "s_stuck", "s_chatter", "s_rapid", "s_volume", "s_level", "s_choppy"]))
    H = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    S = L7.work((dev, period, g.assign(func="", lanes="")))
    return H, S


def stage_health(a):
    k = frame_keys()
    big = {("dec", n) for n, _, s in __import__("a2_features").WINDOWS["dec"] if s >= 1800} | \
          {("stg", n) for n, _, s in __import__("a2_features").WINDOWS["stg"] if s >= 1800}
    k = k[[(p, w) in big for p, w in zip(k.period, k.win)]]
    jobs = [(dev, per, g[["Detector", "win", "pred_phase"]].copy()) for (dev, per), g in k.groupby(["DeviceId", "period"])]
    log(f"{len(jobs)} signal-periods")
    Hs, Ss, t0 = [], [], time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for i, (h, s) in enumerate(pool.imap_unordered(health_work, jobs, chunksize=1)):
            Hs.append(h)
            Ss.append(s)
            if i % 100 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    H = pd.concat(Hs, ignore_index=True)
    S = pd.concat([s for s in Ss if len(s)], ignore_index=True)
    assert not H.DeviceId.isin(locked()).any()
    H.to_parquet(OUT / "health_core_ff.parquet", index=False)
    S.to_parquet(OUT / "stackhealth_ff.parquet", index=False)
    log(f"health rows {len(H):,} {H.status.value_counts().to_dict()}; stack rows {len(S):,} ({time.time()-t0:.0f}s)")


def health_feats(k: pd.DataFrame) -> pd.DataFrame:
    H = pd.read_parquet(OUT / "health_core_ff.parquet").rename(columns={"detector": "Detector"})
    S = pd.read_parquet(OUT / "stackhealth_ff.parquet")
    k6 = ["DeviceId", "Detector", "period", "win"]
    H = H.astype({"Detector": k.Detector.dtype})
    S = S.astype({"Detector": k.Detector.dtype})
    st = {"ok": 0.0, "suspect": 1.0, "bad": 2.0}
    H["hf_status"] = H.status.map(st)
    H = H.rename(columns={"health_score": "hf_score", "n_families": "hf_nfam",
                          **{c: f"hf_{c}" for c in ("s_dropout", "s_stuck", "s_chatter", "s_rapid", "s_volume", "s_level",
                                                    "s_choppy")}})
    S = S[S.clus.notna()].copy()
    g = S.groupby(["DeviceId", "period", "win", "clus"]).chi
    lo1 = g.transform("min")
    second = g.transform(lambda v: np.sort(v.dropna().to_numpy())[1] if v.notna().sum() > 1 else np.nan)
    S["chi_p"] = np.where(S.chi.eq(lo1) & S.chi.notna(), second, lo1)
    S["hf_rel_chi"] = S.chi / (S.chi_p + 0.5)
    S = S.rename(columns={"chi": "hf_chi", "surge": "hf_surge", "drop": "hf_drop", "chat": "hf_chat", "clus_n": "hf_clus_n"})
    x = k[k6].merge(H[k6 + [c for c in HF_COLS if c in H]], on=k6, how="left")
    x = x.merge(S[k6 + ["hf_chi", "hf_surge", "hf_drop", "hf_chat", "hf_rel_chi", "hf_clus_n"]].drop_duplicates(k6),
                on=k6, how="left")
    assert len(x) == len(k)
    return x[HF_COLS].astype(np.float32).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ (d) off-peak
OP_COLS = ["op_rate", "op_rate_ratio", "op_pulse", "op_dur_med", "op_dur_p90", "op_on_red_share", "op_occ_red",
           "op_occ_green", "op_ons_per_green", "op_first_lag", "op_on_at_gs", "op_last_off_frac"]
PEAKS = ((390, 540), (930, 1110))            # weekday 06:30-09:00, 15:30-18:30 (minutes after midnight)
OP_MIN_S = 900


def peak_sql(col: str) -> str:
    m = f"(hour({col}) * 60 + minute({col}))"
    return (f"(isodow({col}) <= 5 AND (({m} >= {PEAKS[0][0]} AND {m} < {PEAKS[0][1]}) OR "
            f"({m} >= {PEAKS[1][0]} AND {m} < {PEAKS[1][1]})))")


def offpeak_secs(t0, secs) -> float:
    t = pd.date_range(pd.Timestamp(t0), periods=int(secs // 60), freq="60s")
    mm = t.hour * 60 + t.minute
    pk = (t.dayofweek <= 4) & (((mm >= PEAKS[0][0]) & (mm < PEAKS[0][1])) | ((mm >= PEAKS[1][0]) & (mm < PEAKS[1][1])))
    return float((~pk).sum() * 60)


def stage_offpeak(a):
    import duckdb
    import a2_features as A2F
    k = frame_keys()
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute(f"SET threads={a.threads}")
    (DCW / "tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    cache = {"dec": DCW / "cache", "stg": DCW / "official" / "stg" / "cache"}
    parts, t00 = [], time.time()
    for period in ("dec", "stg"):
        di = (cache[period] / "det_intervals.parquet").as_posix()
        pc = (cache[period] / "phase_cycles.parquet").as_posix()
        for name, t0, secs in A2F.WINDOWS[period]:
            ops = offpeak_secs(t0, secs)
            tg = k[(k.period == period) & (k.win == name)][["DeviceId", "Detector", "pred_phase", "det_n_on"]].dropna()
            if ops < OP_MIN_S or tg.empty:
                continue
            t1 = pd.Timestamp(t0) + pd.Timedelta(seconds=secs)
            tg = tg.rename(columns={"DeviceId": "dev", "Detector": "det", "pred_phase": "p"}).astype({"det": int, "p": int})
            con.register("tg", tg)
            con.execute(f"""CREATE OR REPLACE TEMP TABLE o AS
                SELECT DISTINCT o.DeviceId AS dev, o.Detector::INT AS det, o.t_on, o.t_off, o.dur, tg.p
                FROM read_parquet('{di}') o JOIN tg ON lower(o.DeviceId) = tg.dev AND o.Detector::INT = tg.det
                WHERE o.t_on >= TIMESTAMP '{t0}' AND o.t_on < TIMESTAMP '{t1}' AND NOT {peak_sql('o.t_on')}
                  AND o.dur IS NOT NULL""")
            con.execute(f"""CREATE OR REPLACE TEMP TABLE c AS
                SELECT lower(DeviceId) AS dev, Phase::INT AS p, green_start AS gs,
                       coalesce(yellow_start, red_start, next_green) AS ge, next_green AS ng
                FROM read_parquet('{pc}')
                WHERE next_green IS NOT NULL AND green_start >= TIMESTAMP '{t0}' AND green_start < TIMESTAMP '{t1}'
                  AND NOT {peak_sql('green_start')}
                  AND lower(DeviceId) IN (SELECT DISTINCT dev FROM tg)""")
            q = con.sql(f"""
              WITH oc AS (
                SELECT o.*, c.gs, c.ge, c.ng FROM o ASOF LEFT JOIN c ON o.dev = c.dev AND o.p = c.p AND o.t_on >= c.gs),
              oc2 AS (
                SELECT *, (gs IS NOT NULL AND t_on < ng) AS incyc, (t_on < ge) AS ing FROM oc),
              cyc AS (
                SELECT tg.dev, tg.det, c.gs, c.ge, c.ng FROM tg JOIN c ON tg.dev = c.dev AND tg.p = c.p),
              per_cyc AS (
                SELECT dev, det, gs, min(t_on) FILTER (ing) AS first_on, max(least(t_off, ge)) FILTER (ing) AS last_off,
                       count(*) FILTER (ing) AS n_g, bool_or(NOT ing AND t_off >= ng) AS on_next
                FROM oc2 WHERE incyc GROUP BY 1, 2, 3),
              cy AS (
                SELECT cyc.dev, cyc.det, count(*) AS n_cyc,
                       sum(epoch(cyc.ge - cyc.gs)) AS g_s, sum(epoch(cyc.ng - cyc.ge)) AS r_s,
                       avg(coalesce(pc.n_g, 0)) AS ons_per_green,
                       median(epoch(pc.first_on - cyc.gs)) AS first_lag,
                       avg(coalesce(pc.on_next, false)::INT) AS on_at_gs,
                       median(epoch(pc.last_off - cyc.gs) / nullif(epoch(cyc.ge - cyc.gs), 0)) AS last_off_frac
                FROM cyc LEFT JOIN per_cyc pc ON cyc.dev = pc.dev AND cyc.det = pc.det AND cyc.gs = pc.gs
                GROUP BY 1, 2),
              od AS (
                SELECT dev, det, count(*) AS n, avg((dur <= 0.15)::INT) AS pulse, median(dur) AS dur_med,
                       quantile_cont(dur, 0.9) AS dur_p90,
                       avg((incyc AND NOT ing)::INT) FILTER (incyc) AS on_red_share,
                       sum(CASE WHEN incyc AND NOT ing THEN epoch(least(t_off, ng) - t_on) ELSE 0 END) AS red_occ,
                       sum(CASE WHEN incyc AND ing THEN epoch(least(t_off, ge) - t_on)
                                ELSE 0 END) AS green_occ
                FROM oc2 GROUP BY 1, 2)
              SELECT tg.dev, tg.det, tg.det_n_on, od.n, od.pulse, od.dur_med, od.dur_p90, od.on_red_share,
                     od.red_occ / nullif(cy.r_s, 0) AS occ_red, od.green_occ / nullif(cy.g_s, 0) AS occ_green,
                     cy.ons_per_green, cy.first_lag, cy.on_at_gs, cy.last_off_frac, cy.n_cyc
              FROM tg LEFT JOIN od ON tg.dev = od.dev AND tg.det = od.det
                      LEFT JOIN cy ON tg.dev = cy.dev AND tg.det = cy.det""").df()
            h = ops / 3600.0
            q["n"] = q.n.fillna(0)
            f = pd.DataFrame({"DeviceId": q.dev, "Detector": q.det, "period": period, "win": name})
            f["op_rate"] = np.log1p(q.n / h)
            f["op_rate_ratio"] = (q.n / h) / np.maximum(q.det_n_on / (secs / 3600.0), 1e-3)
            for c_new, c_old in (("op_pulse", "pulse"), ("op_dur_med", "dur_med"), ("op_dur_p90", "dur_p90"),
                                 ("op_on_red_share", "on_red_share"), ("op_occ_red", "occ_red"),
                                 ("op_occ_green", "occ_green"), ("op_ons_per_green", "ons_per_green"),
                                 ("op_first_lag", "first_lag"), ("op_on_at_gs", "on_at_gs"),
                                 ("op_last_off_frac", "last_off_frac")):
                f[c_new] = q[c_old].astype(float)
            few = q.n < 5                                   # behaviour features need some actuations
            f.loc[few, [c for c in OP_COLS if c not in ("op_rate", "op_rate_ratio")]] = np.nan
            nocyc = q.n_cyc.fillna(0) < 3
            f.loc[nocyc, ["op_on_red_share", "op_occ_red", "op_occ_green", "op_ons_per_green", "op_first_lag",
                          "op_on_at_gs", "op_last_off_frac"]] = np.nan
            f["op_share"] = ops / secs
            parts.append(f)
            log(f"[{period}] {name}: off-peak {ops/secs:.2f} of the window, {len(f):,} det ({time.time()-t00:.0f}s)")
    F = pd.concat(parts, ignore_index=True)
    assert not F.DeviceId.isin(locked()).any()
    F.to_parquet(OUT / "offpeak.parquet", index=False)
    log(f"wrote {len(F):,} rows")


def op_share(k: pd.DataFrame) -> np.ndarray:
    import a2_features as A2F
    sh = {(p, n): offpeak_secs(t0, s) / s for p in A2F.WINDOWS for n, t0, s in A2F.WINDOWS[p]}
    return np.array([sh[(p, w)] for p, w in zip(k.period, k.win)])


def offpeak_feats(k: pd.DataFrame) -> pd.DataFrame:
    F = pd.read_parquet(OUT / "offpeak.parquet").astype({"Detector": k.Detector.dtype})
    x = k[KEY].merge(F[KEY + OP_COLS], on=KEY, how="left")
    assert len(x) == len(k)
    return x[OP_COLS].astype(np.float32).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ fit
ETA_SUB = {"eta": {"eta"}, "etaadv": {"eta", "advance_presence"}}


def stage_fit(a):
    fr, cols, lab, y, ok = load_train()
    cfg = a.cfg
    shuf = cfg.endswith("_shuf")
    fam = cfg[:-5] if shuf else cfg
    classes = list(C7)
    w = np.ones(len(fr))
    y = y.copy()
    fo = fr.fold.to_numpy()
    X0 = fr[cols].to_numpy(np.float32)
    extra = {}                        # outer fold -> extra block (frame rows); key None = same for every fold
    names = []
    if fam == "lanes":
        names = LANE_COLS
        std = pd.read_parquet(OUT / "lanefeat" / "std.parquet").to_numpy(np.float32)
        for k in range(6):
            nk = pd.read_parquet(OUT / "lanefeat" / f"k{k}.parquet").to_numpy(np.float32)
            nk[fo == k] = std[fo == k]                     # test rows: ordinary OOF lanes
            extra[k] = nk
    elif fam == "health":
        names = HF_COLS
        extra[None] = health_feats(fr).to_numpy(np.float32)
    elif fam == "offpeak":
        names = OP_COLS
        extra[None] = offpeak_feats(fr).to_numpy(np.float32)
    elif fam == "opw":
        w = 1.0 + op_share(fr)
    elif fam in ETA_SUB:
        v3 = pd.read_parquet(V.LABELS_V3, columns=["DeviceId", "detector", "print_subtype", "technology"])
        v3 = v3.assign(DeviceId=v3.DeviceId.str.lower(), Detector=v3.detector.astype(fr.Detector.dtype))
        x = fr[["DeviceId", "Detector"]].merge(v3[["DeviceId", "Detector", "print_subtype", "technology"]],
                                              on=["DeviceId", "Detector"], how="left")
        m = (x.print_subtype.isin(ETA_SUB[fam]) & x.technology.eq("radar")).to_numpy() & (y == "Other")
        y[m] = "ETA"
        classes = classes + ["ETA"]
        log(f"{fam}: {int((m & ok).sum()):,} training rows ({x[m & ok].drop_duplicates(['DeviceId', 'Detector']).shape[0]} "
            f"detectors) Other -> ETA")
    else:
        assert fam == "base", cfg
    if shuf:
        rng = np.random.default_rng(59)
        perm = rng.permutation(len(fr))
        extra = {kk: v[perm] for kk, v in extra.items()}
    yi = pd.Series(y).map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int).to_numpy()
    d = OUT / "fit" / cfg
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"cfg": cfg, "classes": classes, "n_cols": len(cols) + len(names), "extra": names},
              open(d / "cols.json", "w"), indent=1)
    for s in [int(x) for x in a.seeds.split(",")]:
        for k in range(6):
            f = d / f"P_{VAR}_s{s}_f{k}.npy"
            if f.exists():
                continue
            t0 = time.time()
            e = extra.get(k, extra.get(None))
            X = X0 if e is None else np.hstack([X0, e])
            inner = (k + 1) % 6
            P, nt = lgb_fit(X, yi, w, ok & (fo != k) & (fo != inner), ok & (fo == inner), fo == k, s, a.threads,
                            len(classes))
            np.save(f, P)
            log(f"{cfg} s{s} f{k}: {nt} trees, {time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------------ score
_SC = {}


def scoring_frame(pick_tag: str):
    if pick_tag in _SC:
        return _SC[pick_tag]
    import t57_function as T57
    import atspm_score as S
    import atspm_pick55 as AP
    fr = S.load(T57.BASE_RUN)
    fr = S.attach_lanes(fr, "ln8/lanes_D.func")
    fr = S.attach_pick_inputs(fr, f"ln6_pick_{pick_tag}", f"ln7_stackhealth_{pick_tag}")
    rows = AP.score_rows(fr)
    _SC[pick_tag] = (fr, rows)
    return _SC[pick_tag]


def spec_probs(spec: str, fr: pd.DataFrame) -> np.ndarray:
    """spec = cfg[@sN][@pk=TAG]; cfg 'base' = the note-57 arm. Rows aligned to fr (S.load order)."""
    parts = spec.split("@")
    cfg = parts[0]
    seeds = [int(o[1:]) for o in parts[1:] if o.startswith("s") and o[1:].isdigit()]
    d = FUNC_DIR if cfg == "base" else OUT / "fit" / cfg
    if not seeds:
        seeds = sorted({int(p.name.split("_s")[1].split("_")[0]) for p in d.glob(f"P_{VAR}_s*_f5.npy")})
    k = frame_keys()
    P = std_probs(k, seeds, d)
    if P.shape[1] == 8:                                  # ETA class: the best non-ATSPM member keeps its probability
        P = np.concatenate([P[:, :6], np.maximum(P[:, 6:7], P[:, 7:8])], 1)
    k["_i"] = np.arange(len(k))
    k["Detector"] = k.Detector.astype(fr.Detector.dtype)
    idx = fr[KEY].merge(k[KEY + ["_i"]], on=KEY, how="left")._i.to_numpy()
    assert not np.isnan(idx).any()
    return P[idx.astype(int)], seeds


def evaluate(spec: str):
    import atspm_score as S
    pk = next((o[3:] for o in spec.split("@") if o.startswith("pk=")), PICK_TAG)
    fr, rows = scoring_frame(pk)
    P, seeds = spec_probs(spec, fr)
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = P[:, i]
    pred = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
    return {s: S.credit(fr, pred, "truth_v3s", r, True) for s, r in rows.items()}, seeds


WG = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]


def stage_score(a):
    import atspm_score as S
    base, bseeds = evaluate(a.base)
    res = {"base": a.base, "results": {}}
    specs = [a.base] + [s for s in a.cfgs.split(",") if s and s != a.base]
    for spec in specs:
        D, seeds = (base, bseeds) if spec == a.base else evaluate(spec)
        r = {"seeds": seeds}
        for sname in ("everything", "realistic"):
            s = S.summ(D[sname])
            r[sname] = {"n": s["n"], "signals": s["signals"], "atspm": s["atspm"], "acc7": s["acc7"],
                        "secondary": s["secondary_acc"], "stack_extra": s["stack_extra"],
                        "by_window": {g: S.summ(D[sname][D[sname].wgroup == g])["atspm"] for g in WG}}
            if spec != a.base:
                r[sname]["d_atspm_pt"] = round(100 * (D[sname].ok_a.mean() - base[sname].ok_a.mean()), 3)
                r[sname]["ci_atspm_pt"] = S.boot(base[sname], D[sname])
                bw = {}
                for g in WG:
                    b0, b1 = base[sname][base[sname].wgroup == g], D[sname][D[sname].wgroup == g]
                    bw[g] = [round(100 * (b1.ok_a.mean() - b0.ok_a.mean()), 3)] + S.boot(b0, b1)
                r[sname]["d_by_window_pt_ci"] = bw
        res["results"][spec] = r
        e, rr = r["everything"], r["realistic"]
        log(f"{spec:28s} ATSPM E {e['atspm']:.4f} R {rr['atspm']:.4f} n {e['n']:,} "
            + (f"| dE {e['d_atspm_pt']:+.3f} {e['ci_atspm_pt']} dR {rr['d_atspm_pt']:+.3f} {rr['ci_atspm_pt']}"
               if spec != a.base else "(base)"))
        if spec != a.base:
            log("    by window E: " + " ".join(f"{g} {v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]"
                                           for g, v in e["d_by_window_pt_ci"].items()))
        else:
            log("    by window E: " + " ".join(f"{g} {v:.4f}" for g, v in e["by_window"].items()))
    json.dump(res, open(OUT / f"score_{a.tag}.json", "w"), indent=1, default=str)
    log(f"-> {OUT / f'score_{a.tag}.json'}")


def stage_agg(a):
    """pooled deltas vs base on short (5 / 10 min) and lane-gated (>= 30 min) windows."""
    import atspm_score as S
    base, _ = evaluate(a.base)
    out = {}
    for spec in [s for s in a.cfgs.split(",") if s]:
        D, _ = evaluate(spec)
        out[spec] = {}
        for sname in ("everything", "realistic"):
            for nm, gs in (("short", ["m5", "m10"]), ("ge30", WG[2:])):
                b0, b1 = base[sname][base[sname].wgroup.isin(gs)], D[sname][D[sname].wgroup.isin(gs)]
                out[spec][f"{sname}_{nm}"] = [round(100 * (b1.ok_a.mean() - b0.ok_a.mean()), 3)] + S.boot(b0, b1)
        log(f"{spec}: {out[spec]}")
    json.dump(out, open(OUT / f"agg_{a.tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cfg", default="")
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--base", default="base")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)
