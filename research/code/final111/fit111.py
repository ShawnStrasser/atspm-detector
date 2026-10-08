"""Note 111: full-data refits for package v6 = note-109 design C (one network kind: the three w32 siba members give BOTH
the phase answer -- their pair phase head -- and the function answer; every tree stage at ONE seed).

    python fit111.py dec          joint decoder on trees-s0 0.5 / siba-head (x100_w32 x3) 0.5, net on tree p >= .01
                                  (= note-109 OOF arm dec_w32m3_rs0); n = mean of its six fold iterations -> v3fit111/decoder
    python fit111.py dectrees     fast-profile decoder on the 1-seed ranker ALONE: six-fold OOF (iterations + OOF column
                                  -> s111/phase/q111.parquet), then one full-data fit -> v3fit111/decode_trees
    set F76_ARM=c & python fit111.py stack       stackers mean3 / single / nonet at ONE seed (seed 0) on the 1-seed
                                  (seed-0) function-trees OOF, Sept-2026 rows, v4q (pkg95 recipes) -> v3fit111/stacker
    set F76_ARM=c & python fit111.py oofnonet    six-fold OOF of the 1-seed nonet stacker (trees s0, seed 0) -> s111/func
Ranker / function trees / setback P50: the existing seed-0 full-data fits (v3fit95) ARE the 1-seed refits of the same
recipe (same data, params, n_estimators rule, seed 0), so they are reused, not refitted.  CPU <= 6 threads; locked_v2
asserted absent by every loader.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
for _d in ("final95", "final109", "final103", "final100"):
    sys.path.insert(0, str(CODE / _d))
sys.path.insert(0, str(CODE))
import rpath  # noqa: E402,F401

NJ = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _phase():
    import phase95 as P
    import p109_phase as P109
    P.NJ = P.C.NJ = P.T57.NJ = P109.NJ = NJ
    return P, P109


def DCW():
    import cand64 as C
    return C.DC_WORK


def p0_s0(P, comb):
    t = pd.read_parquet(P.P95 / "full" / "oof.parquet", columns=P.KEY4 + ["p0_s0"])
    for c in P.KEY4:
        assert (t[c].to_numpy() == comb[c].to_numpy()).all(), c
    return t.p0_s0.to_numpy()


def _fit_decoder(P, comb, simc, p0, n_est, out: Path, meta: dict):
    import lightgbm as lgb
    import decode_train as dec
    import train_official as T
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[P.KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[P.KEY4 + ["Phase", "y"]], on=P.KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    labm = X.Phase.notna().to_numpy()
    prm = dict(T.BIN_PARAMS, n_jobs=NJ, bagging_seed=0, feature_fraction_seed=100, seed=0)
    prm.pop("n_estimators")
    out.mkdir(parents=True, exist_ok=True)
    m = lgb.LGBMClassifier(n_estimators=n_est, **prm).fit(X.loc[labm, cols2], X.y[labm])
    m.booster_.save_model(str(out / "decode_v3.txt"))
    np.save(out / "X_check.npy", X.loc[labm, cols2].to_numpy(np.float64)[:2000])
    json.dump({"features": cols2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
               "params": {k: v for k, v in prm.items() if k != "n_jobs"},
               "trained_on": {"labelled_pair_rows": int(labm.sum()), "period": "Sept-2026 only"}, **meta},
              open(out / "decode_v3.json", "w"), indent=1)
    log(f"decoder -> {out}: n_estimators {n_est}, {int(labm.sum()):,} labelled rows")


def stage_dec():
    P, P109 = _phase()
    comb, simc = P.pool()
    its = json.load(open(P109.OUT / "dec_iters.json"))["w32m3_rs0"]
    nn = P109.net_col(comb, "w32m3")
    p0 = P.blend(comb, p0_s0(P, comb), nn)
    # the same first-stage input as the note-109 OOF arm (q109 p2_w32m3_rs0 was decoded from exactly this blend)
    n_est = int(round(np.mean(its)))
    _fit_decoder(P, comb, simc, p0, n_est, DCW() / "final_v3_work" / "v3fit111" / "decoder", {
        "n_estimators_rule": f"mean best iteration of the six note-109 fold fits (arm w32m3_rs0) {its}",
        "first_stage_input": "note 109 design C: 2026-only phase ranker ONE seed (s0) 0.5 / siba w32 phase head "
                             "(3 members x100_w32, six-fold OOF, mean of member probabilities) 0.5, network only on "
                             "candidates with ranker p >= .01",
        "arm": "w32m3_rs0"})


def stage_dectrees():
    import p90_phase as P90
    P, _ = _phase()
    P90.C.NJ = NJ
    comb, simc = P.pool()
    p0 = p0_s0(P, comb)
    out = DCW() / "s111" / "phase"
    out.mkdir(parents=True, exist_ok=True)
    fq = out / "q111.parquet"
    q = comb[P.KEY4].copy()
    q["p2_trees_s0"], its = P90.decode_iters(comb, simc, p0)
    q.to_parquet(fq, index=False)
    json.dump({"trees_s0": its}, open(out / "dec_iters111.json", "w"), indent=1)
    log(f"trees-only (ranker s0) decoder folds: {its}")
    n_est = int(round(np.mean(its)))
    _fit_decoder(P, comb, simc, p0, n_est, DCW() / "final_v3_work" / "v3fit111" / "decode_trees", {
        "n_estimators_rule": f"mean best iteration of the six note-111 trees-only (ranker s0) fold fits {its}",
        "first_stage_input": "note 111: 2026-only phase ranker ONE seed (s0) alone, no phase network (fast profiles)",
        "arm": "trees_s0_dec"})


def _stack_setup():
    """pkg95 package-stacker inputs (same loaders as v5 / v5b / v5c's stackers) with the trees OOF swapped for seed 0."""
    import pkg95 as PK
    import f109_func as F109
    PK._mask()
    PK.O88._pkg_out()
    PK.O88._v4m_everywhere()
    s74, S, E, Xs, y, trm, names = PK.F.stacker_inputs()
    fr = E["fr"]
    assert not fr.DeviceId.str.lower().isin(PK.F.F75.locked()).any()
    Pt3 = E["Pt"]
    chk = F109.trees_seeds(fr, [0, 1, 2])
    assert np.allclose(chk, Pt3, atol=1e-5), "trees OOF loader disagrees with the stacker's own trees columns"
    E = dict(E)
    E["Pt"] = F109.trees_seeds(fr, [0])
    trm = trm & (fr.period == "stg").to_numpy()
    return PK, s74, S, E, y, trm, names, fr


def stage_stack():
    import lightgbm as lgb
    PK, s74, S, E, y, trm, names, fr = _stack_setup()
    F, F4 = PK.F, PK.F4
    d = DCW() / "final_v3_work" / "v3fit111" / "stacker"
    d.mkdir(parents=True, exist_ok=True)
    prm = dict(F.PRM0, num_threads=NJ, seed=0)
    base = {"classes": E["C7"], "n_features": len(names), "feature_names": names, "params": F.PRM0,
            "num_boost_round": 150, "labels": "v4q (note 95); trees = 2026-only OOF of seed 0 ONLY (note 111)",
            "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique()), "period": "Sept-2026"}}
    # mean3: trees s0 + the 3-member-mean siba OOF (x100_w32), ONE stacker seed
    X = F4.net_X(s74, S, E, s74.net_probs(fr, "x100_w32"))
    np.save(d / "X_check_mean3.npy", X[:2000].astype(np.float32))
    lgb.train(prm, lgb.Dataset(X[trm], y[trm]), num_boost_round=150).save_model(str(d / "stacker_mean3_s0.txt"))
    json.dump({**base, "files": ["stacker_mean3_s0.txt"], "variant": "mean3 (note 111 design C): 1-seed trees OOF + "
               "x100_w32 3-member-mean siba OOF (filtered), ONE stacker seed"}, open(d / "stacker_mean3.json", "w"), indent=1)
    log("mean3 done")
    # single: three single-member versions of every row (3 x rows), ONE stacker seed
    Xs = [F4.net_X(s74, S, E, s74.net_probs(fr, f"x100_w32:{s}")) for s in (0, 1, 2)]
    np.save(d / "X_check_single.npy", Xs[0][:2000].astype(np.float32))
    lgb.train(prm, lgb.Dataset(np.vstack([x[trm] for x in Xs]), np.concatenate([y[trm]] * 3)),
              num_boost_round=150).save_model(str(d / "stacker_single_s0.txt"))
    json.dump({**base, "files": ["stacker_single_s0.txt"], "variant": "single (note 111): 1-seed trees OOF + each of "
               "the three single-member x100_w32 OOFs (3 x rows), ONE stacker seed"}, open(d / "stacker_single.json", "w"),
              indent=1)
    log("single done")
    # nonet: no function network (net columns NaN), ONE stacker seed
    X = F4.net_X(s74, S, E, np.full_like(E["Pt"], np.nan))
    np.save(d / "X_check_nonet.npy", X[:2000].astype(np.float32))
    lgb.train(prm, lgb.Dataset(X[trm], y[trm]), num_boost_round=150).save_model(str(d / "stacker_nonet_s0.txt"))
    json.dump({**base, "files": ["stacker_nonet_s0.txt"], "variant": "nonet (note 98 recipe, note 111 inputs): no "
               "function network, 1-seed trees OOF, ONE stacker seed"}, open(d / "stacker_nonet.json", "w"), indent=1)
    log("nonet done")


def stage_oofnonet():
    """six-fold OOF of the nonet stacker on the 1-seed trees (for the fast-profile headline)."""
    import lightgbm as lgb
    import func95 as G
    import f109_func as F109
    F, F4 = G.fsetup()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    E = dict(E)
    E["Pt"] = F109.trees_seeds(fr, [0])
    X = F4.net_X(s74, S, E, np.full_like(E["Pt"], np.nan))
    trm = trm & (fr.period == "stg").to_numpy()
    fo = fr.fold.to_numpy()
    R = np.zeros((len(y), 7))
    for k in range(6):
        tr, te = trm & (fo != k), fo == k
        m = lgb.train(dict(F.PRM0, num_threads=NJ, seed=0), lgb.Dataset(X[tr], y[tr]), num_boost_round=150)
        R[te] = m.predict(X[te])
    out = DCW() / "s111" / "func"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "P_nonet_t0_s0.npy", R.astype(np.float32))
    log("nonet 1-seed OOF written")


if __name__ == "__main__":
    t0 = time.time()
    globals()[f"stage_{sys.argv[1]}"]()
    log(f"== {sys.argv[1]} done ({time.time() - t0:.0f}s)")
