"""Note 58 (validation plan step 5) -- LANES: separate model vs outputs of the main model, re-validated on v3s.

Question: does the lane output need its own pair model fed by the function head (note 42 "as is"), or does a
function-free (pure behaviour) grouping, or lanes derived from the function model alone, do as well?

Truth: print lanes of function_labels_v3s (ln1_cues.truth rules: complete tiers, not unusual (v3s flag, so the 33
radar-over-loop signals are back), timing target a phase, diagram agrees); validation-only second source = RL / CL /
LL lane text in the channel description (lr_common.lane_token; same lane = same token; phases whose tagged detectors
come from > 1 sensor unit dropped). Inputs: frame v6e predicted phase + the note-57 function arm (229 features,
3-seed mean, OOF). locked_v2 asserted absent. CPU only.

Stages
  truth   v3s print truth (det / phase) + text pairs                               -> ln8/truth_*.parquet
  cues    pair cues (lane_output.signal_pair_cues) for every frame signal, both periods, every window >= 30 min,
          pairs on the same PREDICTED phase (+ same truth phase on the nine Sept training windows)
                                                                                    -> ln8/cues.parquet, dets.parquet
  fit     pair models, six folds x 3 seeds, labelled = print-lane pairs on the nine Sept windows (note-42 recipe):
            A   cues + context + predicted-function pair type (= note 42)
            B   cues + context, no function (pure behaviour)
            C2  function model only: its 229 features + 7 probabilities of both detectors (min / max), no lane cues
            D   A + C2 (joint)
            A_old  note 54's six fold models as they are (trained on v3 truth / old function), applied to these cues
                                                                                    -> ln8/p_<var>.parquet
  decode  lane decode; modes  func = note-42 decoder (function anchors, <= 1 A/P/C/YR per lane, span prior by
          function);  free = every detector an anchor, no role rule, pooled span prior (pure behaviour);
          rule (C1) = no pair model: n_lanes = max(#A, #P, #C) predicted, k-th busiest of each role = lane k.
          lam picked per held-out fold on the other five (Sept scope).        -> ln8/dec_<cfg>_{det,ph}.parquet
  eval    n_lanes exact vs print, pairwise / lane-set accuracy, text pairs, signal bootstrap CI -> ln8/eval.json
  full    chosen configs decoded on every frame signal / window >= 30 min (ln5 format) -> ln8/lanes_<cfg>.parquet
  atspm   ATSPM score through the per-lane decode (stack pick, stack_loser nonatspm_ap), note-57 function arm,
          pick inputs fixed at ln6_pick_v3s                                          -> ln8/atspm.json

    python ln8_validate.py truth|cues|fit|decode|eval|full|atspm [--workers 6]
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
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lane_output as LO  # noqa: E402
import ln1_cues as L1  # noqa: E402

DCW = L1.DCW
REPO = L1.REPO
OUT = DCW / "lanes" / "ln8"
V3S = rpath.LABELS_CURRENT   # note 81: v4l (was function_labels_v3s)
FUNC_CFG = "drop:pp_xcand+pp_pdiff+pp_v2+yr+ratio+phctx"          # note-57 decision: 229 features, 3 seeds
FUNC_DIR = DCW / "trees57" / "function" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx"
CACHE = {"dec": DCW / "cache" / "det_intervals.parquet",
         "stg": DCW / "official" / "stg" / "cache" / "det_intervals.parquet"}
TRAIN_WINS = list(L1.WINS)                    # the nine Sept-2026 windows of note 42
WG = {"m30": "m30", "h1": "h1", "h3": "h3", "h6": "h6", "h24": "h24", "full": "full"}
PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=6, verbose=-1)
SEEDS = (0, 1, 2)
CTX_B = ["log_na", "log_nb", "log_ratio", "log_rate_a", "log_rate_b", "n_det_all", "win_hours"]
LAM_FUNC = [2.0, 3.0, 4.0]
LAM_FREE = [1.0, 2.0, 3.0, 4.0, 5.0]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wgroup(win: str) -> str:
    return "full" if win.startswith("full") else win.split("_")[0]


def locked() -> set:
    return L1.locked()


# ------------------------------------------------------------------------------------------------ truth
def stage_truth(a):
    OUT.mkdir(parents=True, exist_ok=True)
    L1.V3 = V3S
    ph, det = L1.truth()
    ph.to_parquet(OUT / "truth_phase.parquet", index=False)
    det.to_parquet(OUT / "truth_det.parquet", index=False)
    old = pd.read_parquet(DCW / "lanes" / "ln1_truth_phase.parquet")
    log(f"v3s truth: {len(ph)} phases / {ph.DeviceId.nunique()} signals (note 42 v3: {len(old)} / "
        f"{old.DeviceId.nunique()}), n_lanes {ph.n_lanes.value_counts().sort_index().to_dict()}")
    # lane text (validation only)
    sys.path.insert(0, str(REPO / "research" / "code" / "trackA"))
    import lr_common as C
    pl = pd.read_parquet(C.PLANS)[["DeviceId", "Detector", "description"]]
    pl["DeviceId"] = pl.DeviceId.str.lower()
    off = pd.read_parquet(C.OFFICIAL)[["DeviceId", "Detector", "target"]]
    off["DeviceId"] = off.DeviceId.str.lower()
    x = off.drop_duplicates(["DeviceId", "Detector"]).merge(pl.drop_duplicates(["DeviceId", "Detector"]),
                                                            on=["DeviceId", "Detector"])
    x = x[~x.DeviceId.isin(locked()) & x.target.astype(str).str.startswith("P")]
    x["tok"] = C.lane_token(x.description)
    x["unit"] = C.unit_token(x.description).fillna("-")
    x = x[x.tok.notna()]
    nu = x.groupby(["DeviceId", "target"]).unit.nunique()
    keep = nu[nu == 1].index
    x = x.set_index(["DeviceId", "target"]).loc[keep].reset_index()
    T = x.merge(x, on=["DeviceId", "target"], suffixes=("_a", "_b"))
    T = T[T.Detector_a < T.Detector_b]
    T = T.assign(da=T.Detector_a.astype(int), db=T.Detector_b.astype(int),
                 same_lane=(T.tok_a == T.tok_b).astype(int))
    T["n_tok"] = T.groupby(["DeviceId", "target"]).tok_a.transform("nunique")
    T = T[["DeviceId", "target", "da", "db", "same_lane", "n_tok"]]
    assert not T.DeviceId.isin(locked()).any()
    T.to_parquet(OUT / "truth_text_pairs.parquet", index=False)
    log(f"text pairs: {len(T)} / {T.DeviceId.nunique()} signals, same {T.same_lane.mean():.3f}")


# ------------------------------------------------------------------------------------------------ frame / function
def frame_keys() -> pd.DataFrame:
    import v3_retrain as V
    V.set_frame("v6e")
    k = pd.read_parquet(V.FEATS, columns=V.KEY + ["wgroup", "fold", "pred_phase"])
    k["DeviceId"] = k.DeviceId.str.lower()
    assert not k.DeviceId.isin(locked()).any()
    return k


def func_probs(k: pd.DataFrame) -> np.ndarray:
    """note-57 function arm (229 features), 3-seed mean OOF, rows aligned to frame_keys()."""
    P = np.zeros((len(k), 7), np.float32)
    fo = k.fold.to_numpy()
    for s in SEEDS:
        for f in range(6):
            P[fo == f] += np.load(FUNC_DIR / f"P_first.all.wi_s{s}_f{f}.npy")
    return P / len(SEEDS)


# ------------------------------------------------------------------------------------------------ cues
def cue_work(args):
    dev, period, fk, tt = args
    import a2_features as A2F
    import pyarrow.dataset as ds
    tab = ds.dataset(str(CACHE[period])).to_table(filter=ds.field("DeviceId") == dev,
                                                  columns=["Detector", "t_on"]).to_pandas().drop_duplicates()
    ton = tab.t_on.astype("datetime64[us]")
    tab["t"] = ton.astype("int64").to_numpy() / 1e6
    tab["h"] = ton.dt.hour.to_numpy()
    on = {}
    for d, g in tab.groupby("Detector"):
        o = np.argsort(g.t.to_numpy(), kind="stable")
        on[int(d)] = (g.t.to_numpy()[o], g.h.to_numpy()[o])
    drows, prows = [], []
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
            n_on[d] = int(((t >= t0) & (t < t1)).sum())
        drows.append(pd.DataFrame({"DeviceId": dev, "period": period, "win": name, "Detector": g.Detector.astype(int),
                                   "n_on": g.Detector.astype(int).map(n_on).to_numpy(),
                                   "pred_phase": g.pred_phase.to_numpy()}))
        elig = {int(d): p for d, p in zip(g.Detector.astype(int), g.pred_phase)
                if n_on[int(d)] >= LO.MIN_ON and pd.notna(p) and int(d) in on}
        trainwin = period == "stg" and name in TRAIN_WINS
        pairs = [(a, b) for a, b in itertools.combinations(sorted(elig), 2)
                 if elig[a] == elig[b] or (trainwin and a in tt and b in tt and tt[a] == tt[b])]
        if not pairs:
            continue
        loc = {d: on[d] for d, k in n_on.items() if k >= LO.MIN_ON and d in on}
        P = LO.signal_pair_cues(loc, sorted(loc), pairs, t0, t1)
        P["same_pred"] = [elig[a] == elig[b] for a, b in pairs]
        P["DeviceId"], P["period"], P["win"], P["hours"] = dev, period, name, secs / 3600.0
        prows.append(P)
    return (pd.concat(drows, ignore_index=True) if drows else None,
            pd.concat(prows, ignore_index=True) if prows else None)


def stage_cues(a):
    k = frame_keys()
    k = k[~k.wgroup.isin(["m5", "m10"])]
    det = pd.read_parquet(OUT / "truth_det.parquet")
    jobs = []
    for (dev, period), g in k.groupby(["DeviceId", "period"]):
        tt = {}
        if period == "stg":
            t = det[det.DeviceId == dev]
            tt = dict(zip(t.det.astype(int), t.target))
        jobs.append((dev, period, g[["Detector", "win", "pred_phase"]].copy(), tt))
    if a.limit:
        jobs = jobs[:a.limit]
    log(f"{len(jobs)} signal-periods")
    D, P, t0 = [], [], time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for i, (d, p) in enumerate(pool.imap_unordered(cue_work, jobs, chunksize=2)):
            if d is not None:
                D.append(d)
            if p is not None:
                P.append(p)
            if i % 100 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    D = pd.concat(D, ignore_index=True)
    P = pd.concat(P, ignore_index=True)
    fold = k.groupby("DeviceId").fold.first()
    D["fold"] = D.DeviceId.map(fold).astype(int)
    for c in P.columns:
        if P[c].dtype == np.float64:
            P[c] = P[c].astype(np.float32)
    D.to_parquet(OUT / "dets.parquet", index=False)
    P.to_parquet(OUT / "cues.parquet", index=False)
    log(f"wrote {len(D):,} detector-windows, {len(P):,} pairs ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ fit
def load_dets() -> pd.DataFrame:
    """dets.parquet + predicted function (note-57 arm) + phase-group sizes."""
    D = pd.read_parquet(OUT / "dets.parquet")
    k = frame_keys()
    P = func_probs(k)
    k = k.assign(func=np.array(LO.C7, object)[P.argmax(1)])
    for i, c in enumerate(LO.C7):
        k[f"P_{c}"] = P[:, i]
    k["Detector"] = k.Detector.astype(int)
    D = D.merge(k[["DeviceId", "period", "win", "Detector", "func"] + [f"P_{c}" for c in LO.C7]],
                on=["DeviceId", "period", "win", "Detector"], how="left")
    assert D.func.notna().all()
    el = (D.n_on >= LO.MIN_ON) & D.pred_phase.notna()
    gk = ["DeviceId", "period", "win", "pred_phase"]
    D["n_det_all"] = np.where(el, D.assign(e=el).groupby(gk).e.transform("sum"), np.nan)
    nb = el & (D.func != "Bike")
    D["n_det_nb"] = np.where(el, D.assign(e=nb).groupby(gk).e.transform("sum"), np.nan)
    return D


def pair_frame(P: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """context (lane_output.add_context, vectorised) + labels + fold."""
    k = ["DeviceId", "period", "win"]
    Da = D[k + ["Detector", "func", "n_det_all", "n_det_nb", "fold"]]
    P = P.merge(Da.rename(columns={"Detector": "da", "func": "fa"}), on=k + ["da"], how="left")
    P = P.merge(D[k + ["Detector", "func"]].rename(columns={"Detector": "db", "func": "fb"}), on=k + ["db"], how="left")
    P["log_na"] = np.log1p(P.n_a)
    P["log_nb"] = np.log1p(P.n_b)
    P["log_ratio"] = np.abs(P.log_na - P.log_nb)
    P["log_rate_a"] = np.log1p(P.n_a / P.hours)
    P["log_rate_b"] = np.log1p(P.n_b / P.hours)
    P["n_det_phase"] = P.n_det_nb.astype(float)
    P["win_hours"] = P.hours.astype(float)
    for c in LO.C7:
        P[f"pt_{c}"] = (P.fa == c).astype(float) + (P.fb == c).astype(float)
    import ln2_pairmodel as L2
    det = pd.read_parquet(OUT / "truth_det.parquet")
    ph = pd.read_parquet(OUT / "truth_phase.parquet")
    T = L2.truth_pairs(det, ph)[["DeviceId", "da", "db", "same_lane", "n_lanes"]]
    tw = (P.period == "stg") & P.win.isin(TRAIN_WINS)
    x = P.loc[tw, ["DeviceId", "da", "db"]].merge(T, on=["DeviceId", "da", "db"], how="left")
    assert len(x) == int(tw.sum()), "duplicate truth pairs"
    P["same_lane"] = np.nan
    P["n_lanes"] = np.nan
    P.loc[tw, "same_lane"] = x.same_lane.to_numpy()
    P.loc[tw, "n_lanes"] = x.n_lanes.to_numpy()
    P["labelled"] = P.same_lane.notna()
    return P


def c2_matrix(P: pd.DataFrame, D: pd.DataFrame, rows: np.ndarray) -> tuple[np.ndarray, list]:
    """function-model features of both detectors (229 frame columns + 7 probabilities), min / max."""
    import v3_retrain as V
    cols = json.load(open(FUNC_DIR / "cols.json"))["cols"]
    V.set_frame("v6e")
    k = ["DeviceId", "period", "win", "Detector"]
    need = P.loc[rows, ["DeviceId", "period", "win"]].drop_duplicates()
    F = pd.read_parquet(V.FEATS, columns=V.KEY + cols)
    F["DeviceId"] = F.DeviceId.str.lower()
    F["Detector"] = F.Detector.astype(int)
    F = F.merge(need, on=["DeviceId", "period", "win"])
    pc = [f"P_{c}" for c in LO.C7]
    F = F.merge(D[k + pc], on=k, how="left")
    allc = cols + pc
    F["_i"] = np.arange(len(F))
    sub = P.loc[rows, ["DeviceId", "period", "win", "da", "db"]]
    ia = sub.merge(F[k + ["_i"]].rename(columns={"Detector": "da"}), on=["DeviceId", "period", "win", "da"],
                   how="left")._i.to_numpy()
    ib = sub.merge(F[k + ["_i"]].rename(columns={"Detector": "db"}), on=["DeviceId", "period", "win", "db"],
                   how="left")._i.to_numpy()
    M = F[allc].to_numpy(np.float32)
    A, B = M[ia.astype(int)], M[ib.astype(int)]
    X = np.hstack([np.fmin(A, B), np.fmax(A, B)])
    names = [f"mn_{c}" for c in allc] + [f"mx_{c}" for c in allc]
    return X, names


def variant_X(P, D, var, rows, c2=None):
    if var in ("A", "A_old"):
        return P.loc[rows, LO.FEATURES].to_numpy(np.float32), LO.FEATURES
    if var == "B":
        X = P.loc[rows, LO.CUE_COLS + CTX_B].to_numpy(np.float32)
        return X, LO.CUE_COLS + CTX_B
    if var == "C2":
        return c2
    if var in ("D", "Dshuf"):
        X = P.loc[rows, LO.FEATURES].to_numpy(np.float32)
        Z = c2[0]
        if var == "Dshuf":                      # control: C2 block permuted across pair rows (links broken)
            Z = Z[np.random.default_rng(58).permutation(len(Z))]
        return np.hstack([X, Z]), LO.FEATURES + c2[1]
    raise ValueError(var)


def stage_fit(a):
    import lightgbm as lgb
    import ln2_pairmodel as L2
    D = load_dets()
    P = pair_frame(pd.read_parquet(OUT / "cues.parquet"), D)
    P["n_det_all"] = P.n_det_all.astype(float)
    lab = P.labelled.to_numpy()
    sigs = set(pd.read_parquet(OUT / "truth_phase.parquet").DeviceId)
    sept = ((P.period == "stg") & P.win.isin(TRAIN_WINS) & P.DeviceId.isin(sigs)).to_numpy()
    log(f"pairs {len(P):,}; labelled pair-windows {lab.sum():,} ({P[lab].DeviceId.nunique()} signals); "
        f"Sept scope {sept.sum():,}")
    fo = P.fold.to_numpy()
    res = {"labelled": int(lab.sum()), "pairs": int(len(P)), "same_share": float(P.same_lane[lab].mean())}
    keep = ["DeviceId", "period", "win", "da", "db", "same_pred", "same_lane", "n_lanes", "fold"]
    c2 = None
    for var in a.vars.split(","):
        t0 = time.time()
        rows = np.arange(len(P))
        if var in ("C2", "D", "Dshuf") and c2 is None:
            c2 = c2_matrix(P, D, np.arange(len(P)))
            log(f"C2 matrix {c2[0].shape}")
        X, names = variant_X(P, D, var, rows, c2)
        y = P.same_lane.to_numpy()[rows]
        lr, fr_ = lab[rows], fo[rows]
        out = np.zeros(len(rows))
        per_seed = np.zeros((len(SEEDS), len(rows)))
        for f in range(6):
            te = fr_ == f
            if var == "A_old":
                pm = LO.PairModel(DCW / "lanes" / "fold_models" / f"f{f}")
                out[te] = pm.bag.predict(pd.DataFrame(X[te], columns=names)[pm.features])
                continue
            tr = lr & (fr_ != f)
            for si, s in enumerate(SEEDS):
                m = lgb.LGBMClassifier(random_state=s, **PARAMS).fit(X[tr], y[tr].astype(int))
                p = m.predict_proba(X[te])[:, 1]
                per_seed[si, te] = p
                out[te] += p / len(SEEDS)
        Q = P.iloc[rows][keep].copy()
        Q["p_same"] = out.astype(np.float32)
        Q.to_parquet(OUT / f"p_{var}.parquet", index=False)
        # AUC on labelled Sept pair-windows by window group / >= 2-lane phases
        r = {}
        qq = Q[Q.same_lane.notna()]
        wgq = qq.win.map(wgroup)
        for wg in ["m30", "h6", "h24", "full"]:
            m = (wgq == wg).to_numpy()
            mm = m & (qq.n_lanes >= 2).to_numpy()
            r[wg] = {"all": L2.auc(qq.same_lane.to_numpy()[m], qq.p_same.to_numpy()[m]),
                     "multi": L2.auc(qq.same_lane.to_numpy()[mm], qq.p_same.to_numpy()[mm])}
            if var != "A_old":
                lm = (Q.same_lane.notna() & (Q.win.map(wgroup) == wg) & (Q.n_lanes >= 2)).to_numpy()
                r[wg]["multi_per_seed"] = [L2.auc(Q.same_lane.to_numpy()[lm], per_seed[si][lm])
                                           for si in range(len(SEEDS))]
        res[var] = r
        log(f"{var}: {json.dumps(r)} ({time.time()-t0:.0f}s)")
    old = {}
    if (OUT / "fit.json").exists():
        old = json.load(open(OUT / "fit.json"))
    old.update(res)
    json.dump(old, open(OUT / "fit.json", "w"), indent=1, default=str)


# ------------------------------------------------------------------------------------------------ decode
class FreeDecoder(LO.Decoder):
    """Function-free decode: every detector may found a lane, no role rule, one pooled span prior."""

    def decode(self, P, func, vol):
        n = len(func)
        p = np.clip(np.where(np.isnan(P), 0.5, P), 0.02, 0.98)
        lg = np.log(p / (1 - p)) + self.beta
        ls = -np.log1p(np.exp(-lg))
        ld = -np.log1p(np.exp(lg))
        np.fill_diagonal(ls, 0)
        np.fill_diagonal(ld, 0)
        W = ls - ld
        base = ld[np.triu_indices(n, 1)].sum()
        anc = np.ones(n, bool)
        role = np.array([f"r{i}" for i in range(n)], object)      # unique: no role rule
        fn = ["Any"] * n
        best = {}
        for L in range(1, min(LO.MAX_LANES, n) + 1):
            r = self._best_L(L, W, base, anc, role, fn)
            if r is not None:
                best[L] = r
        Ls = sorted(best)
        sc = np.array([best[L][0] - self.lam * L for L in Ls])
        pl = np.exp(sc - sc.max())
        pl /= pl.sum()
        Lb = Ls[int(np.argmax(sc))]
        score, sets = best[Lb]
        conf = self._det_conf(sets, Lb, W, anc, role, fn, score) * float(pl.max())
        lv = np.zeros(Lb)
        for l in range(Lb):
            m = [i for i in range(n) if sets[i] == (1 << l)]
            lv[l] = max(vol[m]) if m else 0.0
        order = np.argsort(-lv, kind="stable")
        remap = {int(old): new for new, old in enumerate(order)}
        lanes = [tuple(sorted(remap[l] + 1 for l in range(Lb) if s >> l & 1)) for s in sets]
        return {"lanes": lanes, "n_lanes": Lb, "n_lanes_conf": float(pl.max()), "lane_conf": conf,
                "lane_volume": lv[order], "scores": dict(zip(Ls, sc))}


def span_priors(det, ph, D9, sigs) -> tuple[dict, dict]:
    """by predicted function (ln3_decode.span_prior) and pooled ("Any"), from training-fold signals."""
    import ln3_decode as L3
    pf = L3.span_prior(det, ph, D9, sigs)
    x = det[det.high_veh & det.phase_kept & det.DeviceId.isin(sigs)]
    x = x.merge(ph[ph.n_lanes >= 2][["DeviceId", "target"]], on=["DeviceId", "target"])
    c = x.span.clip(upper=LO.MAX_LANES).value_counts()
    v = np.array([c.get(k, 0) + 1.0 for k in range(1, LO.MAX_LANES + 1)], float)
    v = np.log(v / v.sum())
    return pf, {"Any": {k: float(v[k - 1]) for k in range(1, LO.MAX_LANES + 1)}}


def rule_decode(dev, g, n_on, hours):
    """C1: no pair model. n_lanes = max(#A, #P, #C) predicted (1..4); within each A/P/C/YR role the k-th busiest
    detector is lane k (YR beyond n_lanes wraps); Mid / Other join the lane whose busiest anchor volume is nearest."""
    grp, func = LO.eligible(g, n_on, LO.MIN_ON)
    ph_rows, det_rows = [], []
    s = pd.Series(grp, dtype=object)
    for phase, members in s.groupby(s):
        ds = sorted(members.index)
        cnt = {r: sum(func[d] == r for d in ds) for r in ("Advance", "Presence", "Count")}
        L = int(min(LO.MAX_LANES, max(1, max(cnt.values()))))
        lane = {}
        for r in LO.ANCHOR:
            rs = sorted([d for d in ds if func[d] == r], key=lambda d: -n_on[d])
            for k, d in enumerate(rs):
                lane[d] = (k % L) + 1
        lv = {l: max([n_on[d] for d in lane if lane[d] == l] or [0]) for l in range(1, L + 1)}
        for d in ds:
            if d not in lane:
                lane[d] = min(lv, key=lambda l: abs(np.log1p(lv[l]) - np.log1p(n_on[d])))
        ph_rows.append({"DeviceId": dev, "phase": int(phase), "n_detectors": len(ds), "n_lanes": L,
                        "n_lanes_conf": np.nan, "lane_volumes_per_hour": ""})
        for d in ds:
            det_rows.append({"DeviceId": dev, "Detector": d, "phase": int(phase), "function": func[d],
                             "lanes": str(lane[d]), "n_lanes_spanned": 1, "lane_conf": np.nan, "lane_note": ""})
    done = {r["Detector"] for r in det_rows}
    for d, p, f in zip(g.Detector.astype(int), g.phase_use, g.func_use):
        if d not in done:
            det_rows.append({"DeviceId": dev, "Detector": int(d), "phase": None if pd.isna(p) else int(p),
                             "function": f, "lanes": "", "n_lanes_spanned": 0, "lane_conf": np.nan, "lane_note": "n/a"})
    return pd.DataFrame(ph_rows), pd.DataFrame(det_rows)


def dec_job(args):
    """args: dev, D (its det-windows), probs {var: {(period, win): {(a, b): p}}}, cfgs [(name, var, mode, lam, prior)]"""
    dev, Dd, probs, cfgs = args
    dets, phs = [], []
    for name, var, mode, lam, prior in cfgs:
        if mode == "rule":
            dec = None
        elif mode == "free":
            dec = FreeDecoder(prior, lam=lam)
        else:
            dec = LO.Decoder(prior, lam=lam)
        for (period, w), g in Dd.groupby(["period", "win"]):
            hours = g.hours.iloc[0]
            gg = g.rename(columns={"pred_phase": "phase_use", "func": "func_use"})
            if mode == "free":
                gg = gg.assign(func_use="Any")
            n_on = dict(zip(gg.Detector.astype(int), gg.n_on))
            if mode == "rule":
                pt, dt = rule_decode(dev, gg, n_on, hours)
            else:
                pr = probs.get(var, {}).get((period, w), {})
                pt, dt = LO.decode_groups(dev, gg, n_on, hours, pr, dec)
            dt["function"] = gg.set_index(gg.Detector.astype(int)).func_use.reindex(dt.Detector).to_numpy() \
                if mode != "free" else g.set_index(g.Detector.astype(int)).func.reindex(dt.Detector).to_numpy()
            dt["period"], dt["win"], dt["cfg"] = period, w, name
            dets.append(dt)
            if len(pt):
                pt["period"], pt["win"], pt["cfg"] = period, w, name
                phs.append(pt)
    return (pd.concat(dets, ignore_index=True),
            pd.concat(phs, ignore_index=True) if phs else pd.DataFrame())


def run_decodes(D, probs_tab: dict, cfgs_by_fold, workers, tag):
    """probs_tab {var: DataFrame(DeviceId, period, win, da, db, p_same)}."""
    import a2_features as A2F
    secs = {(p, n): s for p in A2F.WINDOWS for n, _, s in A2F.WINDOWS[p]}
    D = D.assign(hours=[secs[(p, w)] / 3600.0 for p, w in zip(D.period, D.win)])
    byv = {v: dict(tuple(t.groupby("DeviceId"))) for v, t in probs_tab.items()}
    jobs = []
    for dev, Dd in D.groupby("DeviceId"):
        f = int(Dd.fold.iloc[0])
        pr = {}
        for v, d in byv.items():
            t = d.get(dev)
            if t is None:
                continue
            pr[v] = {(p, w): {(int(x), int(y)): float(z) for x, y, z in zip(h.da, h.db, h.p_same)}
                     for (p, w), h in t.groupby(["period", "win"])}
        jobs.append((dev, Dd, pr, cfgs_by_fold[f]))
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


def eval_cfg(DT, PT, D9, det, ph, T):
    import ln3_decode as L3
    import ln2_pairmodel as L2
    L2.WG.update({w: wgroup(w) for w in TRAIN_WINS})
    DT = DT[DT.period == "stg"].copy()
    PT = PT[PT.period == "stg"].copy()
    DT["wg"] = DT.win.map(wgroup)
    return L3.eval_all(DT, PT, D9.rename(columns={"Detector": "det"}), det, ph, T)


def obj_by_fold(NL, PR, DX):
    import ln3_decode as L3
    return {f: (L3.score(NL, PR, DX, lambda x: x.fold == f, lambda x: x.fold == f, lambda x: x.fold == f))
            for f in range(6)}


DEC_CFGS = [("A_old.func", "A_old", "func"), ("A.func", "A", "func"), ("B.free", "B", "free"),
            ("B.func", "B", "func"), ("A.free", "A", "free"), ("C2.func", "C2", "func"), ("D.func", "D", "func"),
            ("C1.rule", None, "rule")]


def truth_tabs():
    import ln2_pairmodel as L2
    det = pd.read_parquet(OUT / "truth_det.parquet")
    ph = pd.read_parquet(OUT / "truth_phase.parquet")
    T = L2.truth_pairs(det, ph)[["DeviceId", "target", "da", "db", "same_lane", "n_lanes"]]
    return det, ph, T


def stage_decode(a):
    """Sept scope (print-lane signals, nine windows): lam grid per config, picked per held-out fold on the other five."""
    det, ph, T = truth_tabs()
    D = load_dets()
    sigs = set(ph.DeviceId)
    D9 = D[(D.period == "stg") & D.win.isin(TRAIN_WINS) & D.DeviceId.isin(sigs)].copy()
    fsig = D9.groupby("DeviceId").fold.first()
    Dsp = D9.rename(columns={"Detector": "det"})
    pri = {f: span_priors(det, ph, Dsp, set(fsig.index[fsig != f])) for f in range(6)}
    old_prior = {f: {fn: {int(k): v for k, v in d.items()} for fn, d in
                     json.load(open(DCW / "lanes" / "fold_models" / f"f{f}" / "lane_model.json"))["span_prior"].items()}
                 for f in range(6)}
    use = [c for c in DEC_CFGS if c[0] in a.cfgs.split(",")] if a.cfgs else DEC_CFGS
    probs = {}
    for _, var, _ in use:
        if var and var not in probs and (OUT / f"p_{var}.parquet").exists():
            q = pd.read_parquet(OUT / f"p_{var}.parquet", columns=["DeviceId", "period", "win", "da", "db", "p_same",
                                                                    "same_pred"])
            probs[var] = q[q.same_pred & (q.period == "stg") & q.win.isin(TRAIN_WINS) & q.DeviceId.isin(sigs)]
    grid = []
    for name, var, mode in use:
        if var and var not in probs:
            log(f"skip {name}: no p_{var}")
            continue
        lf = [float(x) for x in a.lam_free.split(",")] if a.lam_free else LAM_FREE
        lams = [3.0] if name == "A_old.func" else [0.0] if mode == "rule" else lf if mode == "free" else LAM_FUNC
        for lam in lams:
            grid.append((f"{name}@{lam:g}", var, mode, lam))

    def cfgs_for(f):
        out = []
        for g, var, mode, lam in grid:
            prior = (old_prior[f] if g.startswith("A_old") else pri[f][1] if mode == "free" else pri[f][0])
            out.append((g, var, mode, lam, prior))
        return out
    if a.reuse and (OUT / "sept_dec_det.parquet").exists():
        DT = pd.read_parquet(OUT / "sept_dec_det.parquet")
        PT = pd.read_parquet(OUT / "sept_dec_ph.parquet")
    else:
        DT, PT = run_decodes(D9, probs, {f: cfgs_for(f) for f in range(6)}, a.workers, "sept")
        if a.merge:
            gs = {g for g, *_ in grid}
            D0 = pd.read_parquet(OUT / "sept_dec_det.parquet")
            P0 = pd.read_parquet(OUT / "sept_dec_ph.parquet")
            DT = pd.concat([D0[~D0.cfg.isin(gs)], DT], ignore_index=True)
            PT = pd.concat([P0[~P0.cfg.isin(gs)], PT], ignore_index=True)
            names = {g.split("@")[0] for g in gs}
            for g in DT.cfg.unique():
                if g.split("@")[0] in names and g not in gs:
                    grid.append((g, g.split(".")[0], g.split("@")[0].split(".")[1], float(g.split("@")[1])))
        DT.to_parquet(OUT / "sept_dec_det.parquet", index=False)
        PT.to_parquet(OUT / "sept_dec_ph.parquet", index=False)
    log("decoded; scoring grid")
    res = {}
    for g, var, mode, lam in grid:
        NL, PR, DX = eval_cfg(DT[DT.cfg == g], PT[PT.cfg == g], D9, det, ph, T)
        ob = obj_by_fold(NL, PR, DX)
        res[g] = {f: (ob[f]["nl_exact"] or 0) + (ob[f]["pair_acc_multi"] or 0) for f in range(6)}
        log(f"  {g}: {json.dumps(res[g])}")
    pick = {}
    for name, var, mode in use:
        gs = [g for g, *_ in grid if g.split("@")[0] == name]
        if not gs:
            continue
        pick[name] = {f: max(gs, key=lambda g: sum(res[g][k] for k in range(6) if k != f)) for f in range(6)}
        log(f"{name}: picked {[pick[name][f] for f in range(6)]}")
    if a.merge:
        o = json.load(open(OUT / "decode_pick.json"))
        o["grid_obj"].update(res)
        o["pick"].update(pick)
        res, pick = o["grid_obj"], o["pick"]
    json.dump({"grid_obj": res, "pick": pick}, open(OUT / "decode_pick.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ eval
def boot(d0: pd.DataFrame, d1: pd.DataFrame, col="ok", reps=1000, seed=58) -> list:
    a = d0.groupby("DeviceId")[col].agg(["sum", "size"])
    b = d1.groupby("DeviceId")[col].agg(["sum", "size"])
    s = a.index.union(b.index)
    a, b = a.reindex(s, fill_value=0).to_numpy(float), b.reindex(s, fill_value=0).to_numpy(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=reps).astype(float)
    diff = (w @ b[:, 0]) / (w @ b[:, 1]) - (w @ a[:, 0]) / (w @ a[:, 1])
    return [round(float(np.percentile(diff, 2.5)) * 100, 2), round(float(np.percentile(diff, 97.5)) * 100, 2)]


def oof_cfg(DT, PT, pick, name):
    """rows of the per-fold picked lam for config `name`."""
    D = DT[DT.cfg.str.split("@").str[0] == name]
    P_ = PT[PT.cfg.str.split("@").str[0] == name]
    return D, P_, pick


def stage_eval(a):
    det, ph, T = truth_tabs()
    TX = pd.read_parquet(OUT / "truth_text_pairs.parquet")
    D = load_dets()
    sigs = set(ph.DeviceId)
    D9 = D[(D.period == "stg") & D.win.isin(TRAIN_WINS) & D.DeviceId.isin(sigs)].copy()
    fold = D9.groupby("DeviceId").fold.first()
    DT = pd.read_parquet(OUT / "sept_dec_det.parquet")
    PT = pd.read_parquet(OUT / "sept_dec_ph.parquet")
    pk = json.load(open(OUT / "decode_pick.json"))["pick"]
    import ln3_decode as L3
    rows, res = {}, {}
    for name in pk:
        sel = DT.cfg.isin(set(pk[name].values()))
        fsel = DT.DeviceId.map(fold).astype(int)
        m = sel & (DT.cfg == fsel.map({int(f): g for f, g in pk[name].items()}))
        mp = PT.cfg == PT.DeviceId.map(fold).astype(int).map({int(f): g for f, g in pk[name].items()})
        NL, PR, DX = eval_cfg(DT[m], PT[mp], D9, det, ph, T)
        r = {"all": L3.score(NL, PR, DX)}
        for wg in ["m30", "h6", "h24", "full"]:
            r[wg] = L3.score(NL, PR, DX, lambda x: x.wg == wg, lambda x: x.wg == wg, lambda x: x.wg == wg)
        # text pairs
        dd = DT[m & (DT.period == "stg")][["DeviceId", "Detector", "win", "phase", "lanes"]]
        x = TX.merge(dd.rename(columns={"Detector": "da"}), on=["DeviceId", "da"]).merge(
            dd.rename(columns={"Detector": "db"}), on=["DeviceId", "db", "win"], suffixes=("_a", "_b"))
        x = x[(x.lanes_a != "") & (x.lanes_b != "")]
        sa, sb = x.lanes_a.map(L3.lanes_set), x.lanes_b.map(L3.lanes_set)
        x["pred_same"] = (x.phase_a == x.phase_b) & np.array([len(p & q) > 0 for p, q in zip(sa, sb)])
        x["ok"] = x.pred_same == (x.same_lane == 1)
        x["wg"] = x.win.map(wgroup)
        r["text"] = {"pairs": int(len(x)), "signals": int(x.DeviceId.nunique()), "acc": round(float(x.ok.mean()), 4),
                     "acc_multi": round(float(x[x.n_tok >= 2].ok.mean()), 4),
                     "by_wg": {wg: round(float(x[x.wg == wg].ok.mean()), 4) for wg in ["m30", "h6", "h24", "full"]}}
        res[name] = r
        h = NL.n_lanes_p.notna()
        rows[name] = {"nl": NL[h].assign(ok=(NL.n_lanes_p == NL.n_lanes)[h]),
                      "lane": DX[DX.covered].assign(ok=DX[DX.covered].exact.astype(float)),
                      "pair": PR[PR.covered & (PR.n_lanes >= 2)].assign(
                          ok=(PR.pred_same == (PR.same_lane == 1))[PR.covered & (PR.n_lanes >= 2)].astype(float)),
                      "text": x.assign(ok=x.ok.astype(float))}
        a_ = r["all"]
        log(f"{name:12s} nl {a_['nl_exact']} (multi {a_['nl_exact_multi']}) lane {a_['lane_exact']} "
            f"pair_multi {a_['pair_acc_multi']} cov {a_['det_cov']} text {r['text']['acc']} | full nl "
            f"{r['full']['nl_exact']} m30 nl {r['m30']['nl_exact']}")
    ci = {}
    for ref in ("A.func", "A_old.func"):
        if ref not in rows:
            continue
        for name in rows:
            if name == ref:
                continue
            c = {}
            for k in ("nl", "lane", "pair", "text"):
                d0, d1 = rows[ref][k], rows[name][k]
                c[k] = [round(100 * (d1.ok.mean() - d0.ok.mean()), 2)] + boot(d0, d1)
                if k in ("nl", "lane"):
                    for wg in ("m30", "full"):
                        e0, e1 = d0[d0.wg == wg], d1[d1.wg == wg]
                        c[f"{k}_{wg}"] = [round(100 * (e1.ok.mean() - e0.ok.mean()), 2)] + boot(e0, e1)
            ci[f"{name} vs {ref}"] = c
            log(f"{name} vs {ref}: {c}")
    json.dump({"metrics": res, "ci_pt": ci}, open(OUT / "eval.json", "w"), indent=1, default=str)


# ------------------------------------------------------------------------------------------------ full + atspm
def stage_full(a):
    """decode every frame signal / window >= 30 min with the per-fold picked lam -> ln5-format lanes files."""
    det, ph, T = truth_tabs()
    D = load_dets()
    sigs = set(ph.DeviceId)
    D9 = D[(D.period == "stg") & D.win.isin(TRAIN_WINS) & D.DeviceId.isin(sigs)]
    fsig = D9.groupby("DeviceId").fold.first()
    Dsp = D9.rename(columns={"Detector": "det"})
    pri = {f: span_priors(det, ph, Dsp, set(fsig.index[fsig != f])) for f in range(6)}
    pk = json.load(open(OUT / "decode_pick.json"))["pick"]
    for name in a.cfgs.split(","):
        var, mode = name.split(".")
        probs = {}
        if mode != "rule":
            q = pd.read_parquet(OUT / f"p_{var}.parquet", columns=["DeviceId", "period", "win", "da", "db", "p_same",
                                                                    "same_pred"])
            probs[var] = q[q.same_pred]

        def cfg(f):
            lam = float(pk[name][str(f)].split("@")[1])
            prior = pri[f][1] if mode == "free" else pri[f][0]
            return [(name, var, mode, lam, prior)]
        DT, PT = run_decodes(D, probs, {f: cfg(f) for f in range(6)}, a.workers, name)
        DT = DT.merge(D[["DeviceId", "period", "win", "Detector", "n_on"]], on=["DeviceId", "period", "win", "Detector"],
                      how="left")
        DT.to_parquet(OUT / f"lanes_{name}.parquet", index=False)
        PT.to_parquet(OUT / f"lanes_{name}_ph.parquet", index=False)
        log(f"{name}: {len(DT):,} det-windows, lane known {(DT.lanes != '').mean():.3f}")


def stage_atspm(a):
    sys.path.insert(0, str(REPO / "research" / "code" / "trackA"))
    import t57_function as T57
    import atspm_score as S
    import atspm_pick55 as AP
    base = S.load(T57.BASE_RUN)
    P = T57.load_P("v6e", FUNC_CFG, list(SEEDS), base)
    rows = AP.score_rows(base)
    WGS = ["m30", "h1", "h3", "h6", "h24", "full"]
    out, Ds = {}, {}
    lanes = {"ln5_v3s (as was)": "ln5_lanes_v3s"}
    lanes.update({n: f"ln8/lanes_{n}" for n in a.cfgs.split(",")})
    for name, lf in list(lanes.items()) + [("argmax (no lanes)", None)]:
        if lf is None:
            pred = P.argmax(1)
            fr = base
        else:
            fr = S.attach_lanes(base.copy(), lf)
            fr = AP.attach_pick(fr, "ln6_pick_v3s")
            G = AP.groups(fr)
            pred = AP.decode_all(fr, P, G, "pick")
        Ds[name] = {s: S.credit(fr, pred, "truth_v3s", r, True) for s, r in rows.items()}
        out[name] = {}
        for s in ("everything", "realistic"):
            d = Ds[name][s]
            sm = S.summ(d)
            out[name][s] = {"atspm": sm["atspm"], "n": sm["n"], "stack_extra": sm["stack_extra"],
                            "by_wg": {g: S.summ(d[d.wgroup == g])["atspm"] for g in WGS}}
        log(f"{name:24s} ATSPM E {out[name]['everything']['atspm']} R {out[name]['realistic']['atspm']} "
            f"full E {out[name]['everything']['by_wg']['full']}")
    for rn in ("ln5_v3s (as was)", a.cfgs.split(",")[0]):
        for name in Ds:
            if name == rn:
                continue
            c = {}
            for s in ("everything", "realistic"):
                d0, d1 = Ds[rn][s], Ds[name][s]
                c[s] = [round(100 * (d1.ok_a.mean() - d0.ok_a.mean()), 3)] + S.boot(d0, d1)
                for g in ("m30", "h6", "full"):
                    e0, e1 = d0[d0.wgroup == g], d1[d1.wgroup == g]
                    c[f"{s}_{g}"] = [round(100 * (e1.ok_a.mean() - e0.ok_a.mean()), 3)] + S.boot(e0, e1)
            out[name][f"vs {rn}"] = c
            log(f"{name} vs {rn}: {c['everything']} R {c['realistic']} full {c['everything_full']}")
    json.dump(out, open(OUT / "atspm.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["truth", "cues", "fit", "decode", "eval", "full", "atspm"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vars", default="A_old,A,B,C2,D")
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--lam-free", default="", dest="lam_free")
    ap.add_argument("--merge", action="store_true", help="decode: add these configs to the saved Sept decode + picks")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)
