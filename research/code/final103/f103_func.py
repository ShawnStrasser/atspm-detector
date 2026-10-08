"""Note 103: can ONE LightGBM per task replace scorer -> TCN -> context stacker?  FUNCTION side (note-95 2026-only
setup: Sept-2026 training rows, v4q truth, six folds folds_v4, gate .9 lane rule).

    set F76_ARM=c & python f103_func.py ctx            context columns aligned to the stacker frame -> s103/func/ctx_*.parquet
    set F76_ARM=c & python f103_func.py fit --arm A1   one 7-class LightGBM = the function-trees recipe (note 95: 229
                                                       features, A2.FUNC_PARAMS, inner-fold early stopping, 3 seeds) on
                                                       229 features + the arm's context columns
    set F76_ARM=c & python f103_func.py score          argmax + lane rule (gate .9 decode), per length, CIs vs full v5b

Arms:
  A1   229 + the stacker's PREDICTION-FREE context (16: log minutes, stack cluster size, pick / span / track, span and
       co-location peers, D-lane columns, phase-mate count, lane-mate count, log actuations).  ONE LightGBM (3 seeds).
       (lanes / pick inputs come from the fixed upstream lane chain, as for every arm and the current stacker.)
  A1o  A1 + the stacker's prediction-based context from the six-fold OOF trees (own tree probs, entropy / margin, phase-
       mates' best ATSPM prob, own rank in the phase, lane-mates' max ATSPM prob).  Needs the trees' prediction -> TWO
       LightGBM stages in production (= the stacker with the raw features folded in).
  A2   A1 + the siba network OUTPUT (v5b w32 3-seed mean, six-fold OOF): its 7 log-probs, entropy, margin, and the
       prediction-based context computed on the network's probabilities (no tree model needed).  siba x3 + ONE LightGBM.
References (saved OOF, same rows): full v5b (P_w32m3 = w32 mean3 -> v5 stacker recipe), no networks (P_v5_nonet),
fast le2h (w32m3 <= 1 h, nonet above), trees alone (2026 trees mean3 -> lane rule).
locked_v2 asserted absent by the loaders.  CPU <= 6 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
import func95 as G  # noqa: E402

OUT = G.DC_WORK / "s103" / "func"
NET = "x100_w32"
THREADS = int(os.environ.get("F103_THREADS", "6"))
KEY = ["DeviceId", "Detector", "period", "win"]
FREE = ["log_minutes", "hf_clus_n", "pk_span", "pk_unhealthy", "pk_track", "n_span_peers", "n_coloc_peers", "nl_self",
        "lane_min", "n_lanes_phase", "phase_n_lanes", "phase_n_lanes_conf", "lane_conf", "n_ph", "lm_n", "log1p_det_n_on"]
PCTX = ["phm_A", "phm_P", "phm_C", "phm_Y", "rk_A", "rk_P", "rk_C", "rk_Y", "lm_A", "lm_P", "lm_C", "lm_Y"]
C7S = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
TREE_OWN = [f"lt_{c}" for c in C7S] + ["ent_t", "margin_t"]
NET_OWN = [f"ln_{c}" for c in C7S] + ["ent_n", "margin_n"]
POOLS = {"m5": ["m5"], "m10": ["m10"], "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h24": ["h24"],
         "ge30": ["m30", "h1", "h3", "h6", "h24", "full"]}
LE = ["m5", "m10", "m30", "h1"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def x47(F, S, E, Pt, Pn, Pctx):
    """the 47 stacker columns with stack_X on (Pt, Pn) and ctx_X on Pctx."""
    X = np.hstack([S.stack_X(E["fr"], Pt, Pn), S.ctx_X(E, Pctx)])
    assert X.shape[1] == 62
    return np.delete(X, F.HEALTH15, axis=1)


def stage_ctx(a):
    F, F4 = G.fsetup()
    OUT.mkdir(parents=True, exist_ok=True)
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr, Pt = E["fr"], E["Pt"]
    Pn = s74.net_probs(fr, NET)
    st = (fr.period == "stg").to_numpy()
    log(f"{NET}: coverage stg {np.mean(~np.isnan(Pn[st, 0])):.4f}")
    has = ~np.isnan(Pn[:, 0])
    Pn_f = np.where(has[:, None], Pn, 1.0 / 7)          # missing net rows (dec period mostly): uniform, flagged
    k = fr[KEY].copy()
    k["DeviceId"] = k.DeviceId.str.lower()
    ix = {n: i for i, n in enumerate(names)}
    Xt = x47(F, S, E, Pt, Pt, Pt)                       # context on the OOF trees
    Xn = x47(F, S, E, Pn_f, Pn_f, Pn_f)                 # context on the network alone
    free = pd.DataFrame(Xt[:, [ix[c] for c in FREE]], columns=FREE)
    assert np.allclose(np.nan_to_num(free.to_numpy()), np.nan_to_num(Xn[:, [ix[c] for c in FREE]]))
    trees = pd.DataFrame(Xt[:, [ix[c] for c in TREE_OWN + PCTX]], columns=["t_" + c for c in TREE_OWN + PCTX])
    net = pd.DataFrame(Xn[:, [ix[c] for c in NET_OWN + PCTX]], columns=["n_" + c for c in NET_OWN + PCTX])
    net.loc[~has, :] = np.nan
    net["n_missing"] = (~has).astype(float)
    for nm, D in (("free", free), ("trees", trees), ("net", net)):
        pd.concat([k.reset_index(drop=True), D.astype(np.float32)], axis=1).to_parquet(OUT / f"ctx_{nm}.parquet", index=False)
        log(f"ctx_{nm}: {D.shape}")


ARMS = {"A1": ["free"], "A1o": ["free", "trees"], "A2": ["free", "net"]}


def stage_fit(a):
    import lightgbm as lgb
    import t57_function as T57F
    import f76_function  # noqa: F401
    import v3_retrain as V
    import a2_model as A2
    assert Path(V.LABEL_SETS["v3s"][0]).name == "function_labels_v4q.parquet"
    G.patch_2026(V)
    T57F.OUT = G.TREES95
    T57F.setup("v6e")
    fr, _ = V.load_feats()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, T57F.VAR)
    yi = pd.Series(y).map({c: i for i, c in enumerate(V.C7)}).fillna(-1).astype(int).to_numpy()
    assert list(V.C7) == C7S
    cols = json.load(open(G.TREES95 / G.CFG_SUB / "cols.json"))["cols"]
    assert len(cols) == 229
    k = fr[KEY].copy()
    k["DeviceId"] = k.DeviceId.str.lower()
    new = []
    for part in ARMS[a.arm]:
        D = pd.read_parquet(OUT / f"ctx_{part}.parquet")
        D["Detector"] = D.Detector.astype(k.Detector.dtype)
        m = k.merge(D, on=KEY, how="left")
        assert len(m) == len(fr)
        c2 = [c for c in D.columns if c not in KEY]
        log(f"ctx_{part}: {len(c2)} cols, frame rows without a context row {int(m[KEY[0]].isna().sum())}")
        for c in c2:
            fr[c] = m[c].to_numpy()
        new += c2
    cols = cols + new
    d = OUT / a.arm
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"arm": a.arm, "n_cols": len(cols), "cols": cols, "train_rows": int(ok.sum())}, open(d / "cols.json", "w"),
              indent=1)
    log(f"{a.arm}: {len(cols)} features, {int(ok.sum()):,} training rows (Sept-2026)")
    folds = fr.fold.to_numpy()
    fk = fr[KEY].copy()
    fk["DeviceId"] = fk.DeviceId.str.lower()
    fk.to_parquet(d / "keys.parquet", index=False)
    for s in [int(x) for x in a.seeds.split(",")]:
        for kf in range(6):
            f = d / f"P_s{s}_f{kf}.npy"
            if f.exists():
                continue
            t0 = time.time()
            inner = (kf + 1) % 6
            trm = ok & (folds != kf) & (folds != inner)
            vam = ok & (folds == inner)
            prm = dict(A2.FUNC_PARAMS, n_jobs=THREADS, num_class=7, seed=s, bagging_seed=s + 1,
                       feature_fraction_seed=s + 2, data_random_seed=s + 3)
            n = prm.pop("n_estimators")
            mdl = lgb.LGBMClassifier(n_estimators=n, **prm)
            mdl.fit(fr.loc[trm, cols], yi[trm], eval_set=[(fr.loc[vam, cols], yi[vam])], eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            np.save(f, mdl.predict_proba(fr.loc[folds == kf, cols]).astype(np.float32))
            with open(d / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"seed": s, "fold": kf, "trees": int(mdl.best_iteration_ or 0),
                                     "secs": round(time.time() - t0, 1)}) + "\n")
            log(f"  {a.arm} s{s} f{kf}: {mdl.best_iteration_} trees, {time.time() - t0:.0f}s")
            if s == 0 and kf == 0:
                imp = pd.Series(mdl.booster_.feature_importance("gain"), index=cols)
                (imp / imp.sum()).sort_values(ascending=False).head(40).to_json(d / "gain_f0_s0.json", indent=1)


def arm_P(arm, fr, seeds=None):
    d = OUT / arm
    fk = pd.read_parquet(d / "keys.parquet")
    # fold of each trees-frame row: from the frame itself (same order as keys.parquet)
    import v3_retrain as V
    import t57_function as T57F
    T57F.setup("v6e")
    folds = pd.read_parquet(V.FEATS, columns=["fold"]).fold.to_numpy()
    assert len(folds) == len(fk)
    have = sorted({int(p.name.split("_s")[1].split("_")[0]) for p in d.glob("P_s*_f5.npy")})
    seeds = seeds or have
    Ps = []
    for s in seeds:
        P = np.zeros((len(fk), 7), np.float32)
        for k in range(6):
            P[folds == k] = np.load(d / f"P_s{s}_f{k}.npy")
        Ps.append(P)
    P = np.mean(Ps, 0)
    fk["_i"] = np.arange(len(fk))
    fk["Detector"] = fk.Detector.astype(fr.Detector.dtype)
    idx = fr[KEY].assign(DeviceId=fr.DeviceId.str.lower()).merge(fk, on=KEY, how="left")._i.to_numpy()
    assert not np.isnan(idx).any()
    return P[idx.astype(int)].astype(float), seeds


def stage_score(a):
    import cand64 as C
    import of77
    F, F4 = G.fsetup()
    s74, S, E, Xs, y, trm, _ = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    st = (fr.period == "stg").to_numpy()
    le = fr.wgroup.isin(LE).to_numpy()[:, None]
    full = np.load(G.DC_WORK / "s100" / "b" / "P_w32m3.npy").astype(float)
    nonet = np.load(G.OOF95 / "P_v5_nonet.npy").astype(float)
    P = {"full_v5b": full, "no_nets": nonet, "fast_le2h": np.where(le, full, nonet), "trees_alone": E["Pt"].astype(float)}
    seeds = {}
    for arm in a.arms.split(","):
        if (OUT / arm / "P_s0_f5.npy").exists():
            P[arm], seeds[arm] = arm_P(arm, fr)
            if len(seeds[arm]) > 1 and a.per_seed:
                for s in seeds[arm]:
                    P[f"{arm}@s{s}"], _ = arm_P(arm, fr, [s])
    ok = {}
    for k_, v in P.items():
        t0 = time.time()
        ok[k_] = s74.gate_ok(E, v, lc)
        a_ = v.argmax(1)
        log(f"decoded {k_} ({time.time() - t0:.0f}s)")
    res = {"seeds": seeds, "n": {}, "acc": {}, "delta_vs_full": {}, "delta_vs_no_nets": {}}
    for stn in ("E", "R"):
        for pool, fams in POOLS.items():
            sc = st & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k_][stn]) for k_ in ok])
            res["n"].setdefault(stn, {})[pool] = [int(sc.sum()), int(len(set(sig[sc])))]
            for k_ in ok:
                res["acc"].setdefault(k_, {}).setdefault(stn, {})[pool] = C.acc_ci(ok[k_][stn][sc], sig[sc])
                if k_ != "full_v5b":
                    res["delta_vs_full"].setdefault(k_, {}).setdefault(stn, {})[pool] = C.delta_ci(
                        ok["full_v5b"][stn][sc], ok[k_][stn][sc], sig[sc])
                if k_ != "no_nets":
                    res["delta_vs_no_nets"].setdefault(k_, {}).setdefault(stn, {})[pool] = C.delta_ci(
                        ok["no_nets"][stn][sc], ok[k_][stn][sc], sig[sc])
    # by class >= 30 min, E
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    m = st & fr.wgroup.isin(POOLS["ge30"]).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k_]["E"]) for k_ in ok])
    res["E_ge30_by_class_vs_full"] = {k_: {c: C.delta_ci(ok["full_v5b"]["E"][m & (cl == c)], ok[k_]["E"][m & (cl == c)],
                                                         sig[m & (cl == c)]) for c in ["Advance", "Presence", "Count",
                                                                                       "Yellow_Red", "nonATSPM"]}
                                      for k_ in ok if k_ != "full_v5b"}
    f = OUT / "score103.json"
    json.dump(res, open(f, "w"), indent=1)
    for k_ in ok:
        log(f"{k_:14s} E " + " ".join(f"{p} {res['acc'][k_]['E'][p][0]}" for p in POOLS))
        if k_ != "full_v5b":
            log(f"{'':14s} dE vs full " + " ".join(f"{p} {res['delta_vs_full'][k_]['E'][p]}" for p in POOLS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ctx", "fit", "score"])
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="A1,A1o,A2")
    ap.add_argument("--per_seed", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
