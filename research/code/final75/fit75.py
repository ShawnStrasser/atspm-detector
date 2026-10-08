"""Note 75: full-data refits (CPU) of every model of the integrated champion that can be refitted without a GPU.

No out-of-fold leakage inside the package: each model below is fitted on ALL non-locked training rows of its own recipe,
with the number of boosting rounds fixed from that recipe's six-fold OOF runs (mean best iteration), no early stopping.

  func      229-feature function trees (note 57 arm: v6e frame, v3s labels, note-55 row filter), 3 seeds,
            n_estimators = mean best iteration of the 18 note-57 fold fits (233)
  decoder   joint phase decoder (note-57 recipe) on the champion's FIRST-STAGE input = trees bag + p3 GRU 0.5 / 0.5 with the
            note-73b candidate filter (GRU only where tree p >= .01), all labelled rows, seed 0, n_estimators = mean best
            iteration of the six note-73b thr-.01 fold fits (203)
  lanes     D pair model (note 58: lane cues + context + predicted-function pair type + the function model's 229
            features and 7 probabilities of both detectors, min / max), all labelled print-lane pairs, 3 seeds, 300 rounds
            (fixed recipe); lam = 3 (picked in 4 of 6 folds), span prior from every print-lane signal
  stacker   context stacker (note 67 / 69 / 74: 20 probability columns + 42 hi-res context columns, siba as the net),
            all OOF rows that its OOF version trained on, 3 seeds, 150 rounds (fixed recipe)
  setback   sb7 'all' P50 (note 58), one model per window group (m30 / h6 / h24 / full), 3 seeds, 300 rounds (fixed)

Phase trees are NOT refitted: the candidate-v2 package's `phase_lgbm_v5_s{0,1,2}` ARE the full-data fit of the same
recipe on the same 772-signal pool (note 37 = note-57 recipe; note 57 re-ran it on folds_v4 for OOF only).

    python fit75.py func|decoder|lanes|stacker|setback|all     -> %DC_WORK%/final_v3_work/v3fit/
locked_v2 asserted absent everywhere. CPU only, 6 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

OUT = DC_WORK / "final_v3_work" / "v3fit"
THREADS = 6
SEEDS = (0, 1, 2)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())


def save_json(obj, f):
    json.dump(obj, open(f, "w"), indent=1, default=str)


# ================================================================================================ function trees
def stage_func(a):
    import lightgbm as lgb
    import s59_step6 as S59
    import a2_model as A2
    d = OUT / "function"
    d.mkdir(parents=True, exist_ok=True)
    fr, cols, lab, y, ok = S59.load_train()
    assert not fr.DeviceId.str.lower().isin(locked()).any()
    rec = [json.loads(x) for x in open(S59.FUNC_DIR / "timing.jsonl")]
    n_est = int(round(np.mean([r["trees"] for r in rec])))
    assert len(rec) == 18
    C7 = list(S59.C7)
    yi = pd.Series(y).map({c: i for i, c in enumerate(C7)}).fillna(-1).astype(int).to_numpy()
    m = ok & (yi >= 0)
    X = fr.loc[m, cols]
    log(f"function: {len(cols)} features, {int(m.sum()):,} rows, {fr.loc[m, 'DeviceId'].nunique()} signals, "
        f"n_estimators {n_est}")
    files = []
    for s in SEEDS:
        f = d / f"function229_s{s}.txt"
        files.append(f.name)
        if f.exists():
            continue
        t0 = time.time()
        prm = dict(A2.FUNC_PARAMS, n_jobs=THREADS, num_class=len(C7), seed=s, bagging_seed=s + 1,
                   feature_fraction_seed=s + 2, data_random_seed=s + 3)
        prm.pop("n_estimators")
        mdl = lgb.LGBMClassifier(n_estimators=n_est, **prm).fit(X, yi[m])
        mdl.booster_.save_model(str(f))
        log(f"  seed {s}: {time.time()-t0:.0f}s")
    by_cls = pd.Series(np.array(C7, object)[yi[m]]).value_counts().to_dict()
    save_json({"classes": C7, "features": cols, "n_models": 3, "files": files, "n_estimators": n_est,
               "n_estimators_rule": "mean best iteration of the 18 note-57 six-fold fits (3 seeds) of the 229-feature arm; "
                                    "no early stopping on the final fit",
               "params": {k: v for k, v in A2.FUNC_PARAMS.items() if k not in ("n_jobs", "n_estimators")},
               "averaging": "mean of the seeds' class probabilities",
               "recipe": "note 57 arm drop:pp_xcand+pp_pdiff+pp_v2+yr+ratio+phctx on frame v6e, v3s labels, note-55 rows "
                         "(min_on 5, clean, require_validated, not-checkable-high, health3, Dec-role fp rule), first.all.wi",
               "trained_on": {"n_rows": int(m.sum()), "n_signals": int(fr.loc[m, "DeviceId"].nunique()),
                              "rows_by_class": by_cls}}, d / "function229.json")


# ================================================================================================ decoder
def stage_decoder(a):
    import lightgbm as lgb
    import cand64 as C
    import decode_train as dec
    import train_official as T
    import prof73b as PB
    d = OUT / "decoder"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "decode_v3.txt"
    need = C.KEY4 + ["fold", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(C.T57P / "pool.parquet", columns=need)
    simc = pd.read_parquet(C.T57P / "pool_sim.parquet")
    assert not C.plain(comb.DeviceId).isin(locked()).any()
    q = pd.read_parquet(C.OUT / "phase_oof.parquet")
    for c in C.KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all(), c
    keep = q.p0_bag.to_numpy() >= 0.01
    p0 = PB.blend(q, keep)
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[C.KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[C.KEY4 + ["Phase", "y"]], on=C.KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    lab = X.Phase.notna().to_numpy()
    iters = [104, 259, 309, 207, 109, 230]            # note 73b thr .01 fold fits (oof73b.log)
    n_est = int(round(np.mean(iters)))
    log(f"decoder: {int(lab.sum()):,} labelled pair rows of {len(X):,}; n_estimators {n_est}")
    P = dict(T.BIN_PARAMS, n_jobs=THREADS, bagging_seed=0, feature_fraction_seed=100, seed=0)
    P.pop("n_estimators")
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=n_est, **P).fit(X.loc[lab, cols2], X.y[lab])
    m.booster_.save_model(str(f))
    log(f"  fitted ({time.time()-t0:.0f}s)")
    save_json({"features": cols2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
               "n_estimators_rule": "mean best iteration of the six note-73b thr-.01 fold fits " + str(iters),
               "params": {k: v for k, v in P.items() if k != "n_jobs"},
               "first_stage_input": "trees bag (3 seeds) 0.5 / p3 GRU 0.5, GRU only on candidates with tree p >= .01 "
                                    "(renormalised over the kept candidates), six-fold OOF of both (cand64 phase_oof)",
               "trained_on": {"labelled_pair_rows": int(lab.sum()), "signals": int(C.plain(X.DeviceId[lab]).nunique())}},
              d / "decode_v3.json")


# ================================================================================================ lanes D
def stage_lanes(a):
    import lightgbm as lgb
    import lane_output as LO
    import ln8_validate as L8
    d = OUT / "lanes"
    d.mkdir(parents=True, exist_ok=True)
    D = L8.load_dets()
    P = L8.pair_frame(pd.read_parquet(L8.OUT / "cues.parquet"), D)
    P["n_det_all"] = P.n_det_all.astype(float)
    assert not P.DeviceId.isin(locked()).any()
    il = np.flatnonzero(P.labelled.to_numpy())
    c2 = L8.c2_matrix(P, D, il)
    X = np.hstack([P.loc[il, LO.FEATURES].to_numpy(np.float32), c2[0]])
    names = LO.FEATURES + c2[1]
    y = P.same_lane.to_numpy()[il].astype(int)
    log(f"lanes D: {len(il):,} labelled pair-windows, {P.loc[il, 'DeviceId'].nunique()} signals, {X.shape[1]} features")
    files = []
    for s in SEEDS:
        f = d / f"lane_pair_D_s{s}.txt"
        files.append(f.name)
        if f.exists():
            continue
        m = lgb.LGBMClassifier(random_state=s, **dict(L8.PARAMS, n_jobs=THREADS)).fit(pd.DataFrame(X, columns=names), y)
        m.booster_.save_model(str(f))
        log(f"  seed {s} done")
    det, ph, T = L8.truth_tabs()
    D9 = D[(D.period == "stg") & D.win.isin(L8.TRAIN_WINS) & D.DeviceId.isin(set(ph.DeviceId))]
    pri = L8.span_priors(det, ph, D9.rename(columns={"Detector": "det"}), set(D9.DeviceId))[0]
    pk = json.load(open(L8.OUT / "decode_pick.json"))["pick"]["D.func"]
    lams = [float(v.split("@")[1]) for v in pk.values()]
    lam = float(pd.Series(lams).mode().iloc[0])
    save_json({"pair_models": files, "features": names, "n_cue_context_features": len(LO.FEATURES),
               "c2_columns": json.load(open(L8.FUNC_DIR / "cols.json"))["cols"] + [f"P_{c}" for c in LO.C7],
               "span_prior": {fn: {str(k): v for k, v in dd.items()} for fn, dd in pri.items()},
               "lam": lam, "lam_fold_picks": lams, "beta": 0.0, "role_pen": 4.0, "min_on": LO.MIN_ON,
               "params": {k: v for k, v in L8.PARAMS.items() if k != "n_jobs"},
               "note": "note 58 joint pair model D: cues + context + predicted-function pair type + function model block "
                       "(229 features + 7 probabilities of both detectors, min / max); decode = note-42 func decoder",
               "trained_on": {"labelled_pairs": int(len(il)), "signals": int(P.loc[il, "DeviceId"].nunique()),
                              "same_share": float(y.mean())}}, d / "lane_model.json")


# ================================================================================================ stacker
def stage_stacker(a):
    import lightgbm as lgb
    import cand64 as C
    import s67_decider as S
    import s74 as S74
    d = OUT / "stacker"
    d.mkdir(parents=True, exist_ok=True)
    E = S74.setup("x69_siba")
    fr, C7 = E["fr"], E["C7"]
    Pt, Pn = E["Pt"], E["Pn"]
    X = np.hstack([S.stack_X(fr, Pt, Pn), S.ctx_X(E, C.W_TREE * Pt + (1 - C.W_TREE) * Pn)])
    y = np.array([C7.index(t) if isinstance(t, str) and t in C7 else -1 for t in fr.truth_v3s.to_numpy(object)])
    trm = (y >= 0) & ~fr.exclude_train_score.fillna(False).to_numpy(bool) & (fr.det_n_on >= 5).to_numpy()
    log(f"stacker: X {X.shape}, training rows {int(trm.sum()):,} ({fr.DeviceId[trm].nunique()} signals)")
    np.save(d / "X_check.npy", X[:2000].astype(np.float32))
    fr[["DeviceId", "Detector", "period", "win"]].iloc[:2000].to_parquet(d / "X_check_keys.parquet", index=False)
    files = []
    prm0 = dict(objective="multiclass", num_class=7, learning_rate=0.05, num_leaves=7, max_depth=3,
                min_data_in_leaf=300, lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                num_threads=THREADS, verbose=-1)
    for s in SEEDS:
        f = d / f"stacker_s{s}.txt"
        files.append(f.name)
        if f.exists():
            continue
        m = lgb.train(dict(prm0, seed=s), lgb.Dataset(X[trm], y[trm]), num_boost_round=150)
        m.save_model(str(f))
        log(f"  seed {s} done")
    names = ([f"lt_{c}" for c in C7] + [f"ln_{c}" for c in C7] + ["log_minutes", "ent_t", "ent_n", "margin_t", "margin_n",
                                                                  "agree"] +
             [f"hf_{i}" for i in range(16)] + ["pk_span", "pk_unhealthy", "pk_track", "n_span_peers", "n_coloc_peers",
                                               "nl_self", "lane_min", "n_lanes_phase", "phase_n_lanes",
                                               "phase_n_lanes_conf", "lane_conf", "n_ph", "phm_A", "phm_P", "phm_C",
                                               "phm_Y", "rk_A", "rk_P", "rk_C", "rk_Y", "lm_n", "lm_A", "lm_P", "lm_C",
                                               "lm_Y", "log1p_det_n_on"])
    assert len(names) == X.shape[1], (len(names), X.shape)
    save_json({"classes": C7, "n_features": int(X.shape[1]), "feature_names": names, "files": files,
               "params": {k: v for k, v in prm0.items() if k != "num_threads"}, "num_boost_round": 150,
               "net": "siba (x69_siba) 3-seed mean per fold, OOF; tree weight of the context blend 0.6",
               "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique())}},
              d / "stacker.json")


def stage_stacker_single(a):
    """note 75: the stacker for a SINGLE function-net member in production.  The note-69 stacker was trained on the
    3-seed mean of the siba fold models; fed one model it loses ~0.2 pt (member75.py).  Trained instead on the three
    single-seed versions of every OOF row (3 x rows; label repeated), it loses ~0.12 pt.  Same recipe otherwise."""
    import lightgbm as lgb
    import cand64 as C
    import s67_decider as S
    import s74 as S74
    d = OUT / "stacker"
    d.mkdir(parents=True, exist_ok=True)
    E = S74.setup("x69_siba")
    fr, C7 = E["fr"], E["C7"]
    Pt = E["Pt"]
    y = np.array([C7.index(t) if isinstance(t, str) and t in C7 else -1 for t in fr.truth_v3s.to_numpy(object)])
    trm = (y >= 0) & ~fr.exclude_train_score.fillna(False).to_numpy(bool) & (fr.det_n_on >= 5).to_numpy()
    Xs = []
    for s in (0, 1, 2):
        P1 = S74.net_probs(fr, f"x69_siba:{s}")
        P1 = np.where(np.isnan(P1[:, :1]), Pt, P1)
        Xs.append(np.hstack([S.stack_X(fr, Pt, P1), S.ctx_X(E, C.W_TREE * Pt + (1 - C.W_TREE) * P1)])[trm])
    X = np.vstack(Xs)
    yy = np.concatenate([y[trm]] * 3)
    log(f"stacker (single member): X {X.shape}")
    prm0 = dict(objective="multiclass", num_class=7, learning_rate=0.05, num_leaves=7, max_depth=3,
                min_data_in_leaf=300, lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                num_threads=THREADS, verbose=-1)
    for s in SEEDS:
        f = d / f"stacker_single_s{s}.txt"
        if not f.exists():
            lgb.train(dict(prm0, seed=s), lgb.Dataset(X, yy), num_boost_round=150).save_model(str(f))
            log(f"  seed {s} done")


# ================================================================================================ setback sb7
def stage_setback(a):
    import lightgbm as lgb
    import sb5_setback as SB
    import sb7_validate as S7
    d = OUT / "setback"
    d.mkdir(parents=True, exist_ok=True)
    F = pd.read_parquet(S7.OUT / "feat.parquet")
    assert not F.dev.isin(locked()).any()
    F["wg"] = F.win.map(S7.wgroup)
    L = F.dist_kind.eq("single") & F.print_fn.isin(SB.TARGET) & (F.n_on > 0)
    meta = {"groups": {}, "features": SB.FEATS, "params": SB.LGB, "quantile": 0.5,
            "group_rule_hours": {"m30": [0, 1.7320508], "h6": [1.7320508, 12.0], "h24": [12.0, 39.799497],
                                 "full": [39.799497, 1e9]},
            "group_rule": "nearest training window length on a log scale (geometric means of 0.5 / 6 / 24 / 66 h)"}
    for wg in S7.WGS:
        tr = F[L & (F.wg == wg)]
        Lf = SB.fit_L(tr)
        trp = SB.add_physics(tr, Lf)
        yy = SB._lg(trp.dist.to_numpy())
        files = []
        for s in SEEDS:
            f = d / f"setback_{wg}_q50_s{s}.txt"
            files.append(f.name)
            if not f.exists():
                m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=s, **dict(SB.LGB, n_jobs=THREADS))
                m.fit(trp[SB.FEATS].astype(float), yy)
                m.booster_.save_model(str(f))
        meta["groups"][wg] = {"files": files, "L_fit": Lf, "n_train": int(len(tr)), "signals": int(tr.dev.nunique())}
        log(f"setback {wg}: {len(tr)} rows, L_fit {Lf:.1f}")
    save_json(meta, d / "setback_p50.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["func", "decoder", "lanes", "stacker", "stacker_single", "setback", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = ["func", "decoder", "lanes", "stacker", "stacker_single", "setback"] if a.stage == "all" else [a.stage]
    for s in st:
        t0 = time.time()
        globals()[f"stage_{s}"](a)
        log(f"== {s} done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
