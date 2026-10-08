"""Note 37: the PHASE trees re-fitted with the 71 released NEWTEST signals in the pool.

Exactly the final_v1 recipe (`fit_final_v1.py`: 3-seed LightGBM pair ranker bag -> joint
decoder, official timing labels, the 22-window mix, same parameters), on the stage-12/13
pool (709 signals, 701 with rows) PLUS the 71 released signals (`official/newtest_released.csv`).

Folds: the 709 keep the fold the stage-12/13 phase OOF used (folds.csv + newtrain_folds.csv),
so the old rows are held out exactly as before; the released 71 take their folds_v4 fold
(1-5).  NB folds_v4 itself differs from the phase map on 22 old NEWTRAIN signals (it
inherited the function head's folds for 29 of them) -- the phase map is kept on purpose so
the same-rows comparison with stage 13 changes the training data and nothing else.

The locked hold-out (`official/locked_v2.csv`, 115 signals) is asserted absent everywhere.
Changes from fit_final_v1 that are NOT recipe changes: n_jobs 10 -> 8 (shared CPU), feature
columns held as float32 (RAM), per-fold checkpoints (resumable).

    python phase_v3_trees.py --stage oof      # six-fold OOF -> work/trees_oof_bywindow.parquet
    python phase_v3_trees.py --stage oof --control   # same code, 709 only (recipe-noise control)
    python phase_v3_trees.py --stage models   # final ranker bag + decoder -> phase_v3/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK, N_FOLDS  # noqa: E402
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
import fit_final_v1 as F  # noqa: E402

OUTROOT = DC_WORK / "final_v3_work" / "phase_v3"
WORK = OUTROOT / "work"
CKPT = WORK / "trees_ckpt"
RELEASED = DC_WORK / "official" / "newtest_released.csv"
LOCKED = DC_WORK / "official" / "locked_v2.csv"
FOLDS_V4 = DC_WORK / "folds_v4.csv"
SEEDS = F.SEEDS
NJ = 8
T.RANK_PARAMS["n_jobs"] = NJ
T.BIN_PARAMS["n_jobs"] = NJ
KEY4 = ["DeviceId", "Detector", "win", "cand_phase"]
log = F.log


def plain(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("@stg", "", regex=False).str.lower()


def locked_ids() -> set:
    return set(pd.read_csv(LOCKED).DeviceId.astype(str).str.lower())


def released_ids() -> list[str]:
    r = sorted(pd.read_csv(RELEASED).DeviceId.astype(str).str.lower())
    assert len(r) == 71 and not set(r) & locked_ids()
    return r


def build_pool_v3(control: bool = False):
    comb, simc, fc = F.build_pool()                  # the stage-12 pool, unchanged
    if control:                                      # same code, no released signals
        comb[fc] = comb[fc].astype(np.float32)
        log(f"CONTROL pool {comb.shape}; {comb.DeviceId.nunique()} signals")
        return comb, simc, fc
    nt = pd.read_csv(DC_WORK / "official" / "newtrain_folds.csv")
    fm = dict(zip(nt.DeviceId.str.lower(), nt.fold))
    stg = comb.src == "STG"
    assert (comb.loc[stg, "DeviceId"].str.replace("@stg", "").str.lower().map(fm)
            == comb.loc[stg, "fold"]).all(), "pool fold != newtrain_folds.csv"
    rel = released_ids()
    sp = pd.read_csv(DC_WORK / "official" / "new_signal_split.csv")
    ids = sorted(d for d in sp.DeviceId if d.lower() in set(rel))
    assert len(ids) == 71
    df, sim = T.load_stg(keep=set(ids))
    off = T._official_phase().copy()
    off["DeviceId"] = off.DeviceId + "@stg"
    df = T.attach_labels(df, off)
    df = df[df.Phase.notna()].reset_index(drop=True)     # = run_train.build_extra
    f4 = pd.read_csv(FOLDS_V4)
    f4m = dict(zip(f4.DeviceId.str.lower(), f4.fold))
    df["fold"] = plain(df.DeviceId).map(f4m)
    assert df.fold.notna().all() and set(df.fold.unique()) <= {1, 2, 3, 4, 5}
    df["Phase_hand"] = np.nan
    df["y_hand"] = 0
    df["scorable_hand"] = False
    df = T.add_scorable(df)
    df["src"] = "REL"
    log(f"released: {df.shape}, {df.DeviceId.nunique()} signals")
    for c in fc:
        assert c in df.columns, c
    comb = pd.concat([comb, df[[c for c in comb.columns if c in df.columns]]],
                     ignore_index=True)
    simc = pd.concat([simc, sim], ignore_index=True)
    comb = comb.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    comb["scorable"] = comb.scorable.fillna(False).astype(bool)
    comb[fc] = comb[fc].astype(np.float32)
    lk = locked_ids()
    assert not plain(comb.DeviceId).isin(lk).any(), "locked signal in the pool"
    assert not plain(simc.DeviceId).isin(lk).any()
    log(f"pool v3 {comb.shape}; {comb.DeviceId.nunique()} signals; "
        f"{int(comb.Phase.notna().sum()):,} labelled rows; by src "
        f"{comb.groupby('src').DeviceId.nunique().to_dict()}")
    return comb, simc, fc


def rp(sd: int) -> dict:
    return dict(T.RANK_PARAMS, bagging_seed=sd, feature_fraction_seed=sd + 100,
                data_random_seed=sd + 200, seed=sd)


def stage_oof(a) -> None:
    global CKPT
    if a.control:
        CKPT = WORK / "trees_ckpt_control"
    CKPT.mkdir(parents=True, exist_ok=True)
    comb, simc, fc = build_pool_v3(a.control)
    keyf = CKPT / "rows.parquet"
    if not keyf.exists():
        comb[KEY4 + ["src", "fold"]].to_parquet(keyf, index=False)
    else:
        k0 = pd.read_parquet(keyf)
        assert len(k0) == len(comb) and (k0.DeviceId.values == comb.DeviceId.values).all()
    lab = comb.Phase.notna().to_numpy()
    t0 = time.time()
    for k in range(N_FOLDS):
        te = (comb.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        base = (~te) & lab
        for j, sd in enumerate(SEEDS):
            f = CKPT / f"S_f{k}_s{sd}.npy"
            if f.exists():
                continue
            tr = comb[base & (comb.fold != inner).to_numpy()]
            va = comb[base & (comb.fold == inner).to_numpy()]
            m = T._fit_rank(tr, va, fc, rp(sd))
            del tr, va
            np.save(f, m.predict(comb.loc[te, fc]))
            log(f"  fold {k} seed {sd}: {m.best_iteration_} trees ({time.time()-t0:.0f}s)")
    S = np.zeros((len(SEEDS), len(comb)))
    for k in range(N_FOLDS):
        te = (comb.fold == k).to_numpy()
        for j, sd in enumerate(SEEDS):
            S[j, te] = np.load(CKPT / f"S_f{k}_s{sd}.npy")
    p0 = F.bag_prob(comb, list(S))
    per_seed = np.stack([T.to_prob(comb, S[j]) for j in range(len(SEEDS))])

    # joint decoder on the bagged OOF first-stage probabilities (fit_final_v1.oof_bagged)
    pr = comb[KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[KEY4 + ["Phase", "fold", "y"]], on=KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    s2 = np.zeros(len(X))
    labX = X.Phase.notna().to_numpy()
    for k in range(N_FOLDS):
        te = (X.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        base = (~te) & labX
        tr = X[base & (X.fold != inner).to_numpy()]
        va = X[base & (X.fold == inner).to_numpy()]
        P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
        log(f"  decoder fold {k}: {m.best_iteration_} trees")
    X["p2"] = T.norm_prob(X, s2)
    back = comb[KEY4].merge(X[KEY4 + ["p2"]], on=KEY4, how="left")
    ob = comb[KEY4 + ["src", "fold"]].copy()
    ob["prob"] = back.p2.to_numpy()
    ob["p0"] = p0
    for j, sd in enumerate(SEEDS):
        ob[f"p0_s{sd}"] = per_seed[j]
    dest = WORK / ("trees_oof_control_bywindow.parquet" if a.control
                   else "trees_oof_bywindow.parquet")
    ob.to_parquet(dest, index=False)
    log(f"wrote {dest} ({len(ob):,} rows) "
        f"in {time.time()-t0:.0f}s")


def stage_models(a) -> None:
    """fit_final_v1.stage_models on the v3 pool: ranker seeds trained on folds 1-5 with
    fold 0 as the early-stopping set (as final_v1 was), decoder on the OOF p0."""
    OUTROOT.mkdir(parents=True, exist_ok=True)
    comb, simc, fc = build_pool_v3()
    lab = comb.Phase.notna().to_numpy()
    iters, scores = [], []
    for j, sd in enumerate(SEEDS):
        f = OUTROOT / f"phase_lgbm_v5_s{j}.txt"
        if not f.exists():
            tr = comb[lab & (comb.fold != 0).to_numpy()]
            va = comb[lab & (comb.fold == 0).to_numpy()]
            m = T._fit_rank(tr, va, fc, rp(sd))
            del tr, va
            m.booster_.save_model(str(f))
            log(f"ranker seed {sd}: {m.best_iteration_} trees")
        b = lgb.Booster(model_file=str(f))
        iters.append(int(b.current_iteration()))
        scores.append(b.predict(comb[fc], num_threads=NJ))
    json.dump({"features": fc, "params": dict(T.RANK_PARAMS), "n_models": len(SEEDS),
               "seeds": SEEDS, "best_iterations": iters, "objective": "lambdarank",
               "averaging": "mean of per-detector-normalised (softmax) probabilities",
               "pool": "709 stage-12/13 training signals + 71 released NEWTEST (note 37)"},
              open(OUTROOT / "phase_lgbm_v5.json", "w"), indent=1)

    q = pd.read_parquet(WORK / "trees_oof_bywindow.parquet", columns=KEY4 + ["p0"])
    pr = comb[KEY4].merge(q, on=KEY4, how="left")
    log(f"decoder trains on OOF first-stage probabilities "
        f"({int(pr.p0.notna().sum()):,}/{len(pr):,} rows matched)")
    pr["p0"] = pr.p0.fillna(pd.Series(F.bag_prob(comb, scores), index=pr.index))
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[KEY4 + ["Phase", "fold", "y"]], on=KEY4, how="left")
    X = X[X.Phase.notna()].sort_values(["win", "DeviceId", "Detector", "cand_phase"]) \
                          .reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0)
    n = P.pop("n_estimators")
    m2 = lgb.LGBMClassifier(n_estimators=n, **P)
    dtr, dva = X[X.fold != 0], X[X.fold == 0]
    m2.fit(dtr[cols], dtr.y, eval_set=[(dva[cols], dva.y)], eval_metric="binary_logloss",
           callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    m2.booster_.save_model(str(OUTROOT / "decode_lgbm_v5.txt"))
    json.dump({"features": cols, "champion": "decode_sim_adj_binary", "mode": "binary",
               "temperature": 1.0, "best_iteration": int(m2.best_iteration_ or n),
               "params": dict(P, n_estimators=n)},
              open(OUTROOT / "decode_lgbm_v5.json", "w"), indent=1)
    log(f"decoder: {m2.best_iteration_} trees")

    from lgbm_numpy import NumpyBooster, NumpyBoosterBag
    smp = comb.sample(min(20000, len(comb)), random_state=0)
    files = [str(OUTROOT / f"phase_lgbm_v5_s{j}.txt") for j in range(len(SEEDS))]
    a1 = np.mean([lgb.Booster(model_file=f).predict(smp[fc]) for f in files], axis=0)
    b1 = NumpyBoosterBag(files).predict(smp[fc].astype(np.float64))
    smpX = X.sample(min(20000, len(X)), random_state=0)
    a2 = np.asarray(m2.predict_proba(smpX[cols])[:, 1])
    b2 = NumpyBooster(OUTROOT / "decode_lgbm_v5.txt").predict(smpX[cols])
    ver = {"ranker_bag_max_abs_diff": float(np.max(np.abs(a1 - b1))),
           "decoder_max_abs_diff": float(np.max(np.abs(a2 - b2))),
           "ranker_best_iterations": iters,
           "decoder_best_iteration": int(m2.best_iteration_ or n),
           "pool_signals": int(comb.DeviceId.nunique()),
           "pool_labelled_rows": int(lab.sum())}
    json.dump(ver, open(OUTROOT / "trees_fit_info.json", "w"), indent=1)
    log("numpy backend / fit info: " + json.dumps(ver))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["oof", "models"])
    ap.add_argument("--control", action="store_true",
                    help="oof only: the same code on the 709 old signals (no released)")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
