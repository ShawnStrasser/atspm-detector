"""Note 90: the phase network of the final package = TCN ad_all (note 87b tie, user rule tie -> faster).

OOF (note-87b pipeline: f76 trees 3 seeds, 0.5 / 0.5 with the net on candidates with tree p >= .01, decoder re-fitted
six-fold OOF on folds_v4 with the note-57 recipe) for
  tcn_ad     ad_all_e100 fold models, note-53 inference (first-candidate partner tie)
  tcn_ad76   the same fold models re-inferred with the note-76 tcn53.py (content partner tie-break; note-86 GPU queue)
scored against the GRU p3 arm (= the current package) with p87_phase's scorer (cand64 / t57 rows, E and R).
The decoder fold fits' best iterations are recorded; the package decoder = one full-data fit on the blend with
n_estimators = their mean (fit76 recipe).

    python p90_phase.py decode --arms tcn_ad[,tcn_ad76]   -> %DC_WORK%/s90/p87_q.parquet + s90/dec_iters.json
    python p90_phase.py score                             -> %DC_WORK%/s90/p87_phase.json
    python p90_phase.py fitdec --arm tcn_ad76             -> %DC_WORK%/final_v3_work/v3fit90/decoder/decode_v3.{txt,json}
CPU, LightGBM 2 threads (the function chain runs next to it at 4).  locked_v2 asserted absent.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "final87"))
import rpath  # noqa: F401,E402
import p87_phase as P87  # noqa: E402
import cand64 as C  # noqa: E402

NJ = int(os.environ.get("P90_THREADS", "2"))
C.NJ = NJ
OUT = C.DC_WORK / "s90"
P87.OUT = OUT
FIT = C.DC_WORK / "final_v3_work" / "v3fit90" / "decoder"
F76P = P87.F76P


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def decode_iters(comb, simc, p0):
    """cand64.decode (note-57 recipe, OOF over folds_v4) that also returns the six fold fits' best iterations."""
    import lightgbm as lgb
    import train_official as T
    import decode_train as dec
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[C.KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[C.KEY4 + ["Phase", "fold", "y"]], on=C.KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    s2 = np.zeros(len(X))
    labX = X.Phase.notna().to_numpy()
    fx = X.fold.to_numpy()
    iters = []
    for k in range(C.N_FOLDS):
        te = fx == k
        inner = (k + 1) % C.N_FOLDS
        base = (~te) & labX
        tr, va = X[base & (fx != inner)], X[base & (fx == inner)]
        P = dict(T.BIN_PARAMS, n_jobs=NJ, bagging_seed=0, feature_fraction_seed=100, seed=0)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(tr[cols2], tr.y, eval_set=[(va[cols2], va.y)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2[te] = m.predict_proba(X.loc[te, cols2])[:, 1]
        iters.append(int(m.best_iteration_))
        log(f"    decoder fold {k}: {m.best_iteration_} trees")
    X["p2"] = T.norm_prob(X, s2)
    return comb[C.KEY4].merge(X[C.KEY4 + ["p2"]], on=C.KEY4, how="left").p2.to_numpy(), iters


def _pool():
    need = C.KEY4 + ["fold", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(F76P / "pool.parquet", columns=need)
    simc = pd.read_parquet(F76P / "pool_sim.parquet")
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    return comb, simc


def stage_decode(a):
    comb, simc = _pool()
    q = P87.base()
    for c in C.KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all(), c
    OUT.mkdir(parents=True, exist_ok=True)
    fq, fi = OUT / "p87_q.parquet", OUT / "dec_iters.json"
    done = pd.read_parquet(fq) if fq.exists() else q[C.KEY4].copy()
    its = json.load(open(fi)) if fi.exists() else {}
    if "p2_gru" not in done:
        f76 = pd.read_parquet(F76P / "f76_q.parquet")
        for c in C.KEY4:
            assert (f76[c].to_numpy() == q[c].to_numpy()).all(), c
        done["p2_gru"] = f76.p2_new.to_numpy()
        done["pn_gru"] = q.p_nn.to_numpy()
        its["gru"] = [116, 258, 453, 227, 107, 310]          # fit76 (the current package decoder's fold fits)
    for arm in a.arms.split(","):
        if f"p2_{arm}" in done and not a.force:
            log(f"{arm} cached"); continue
        t0 = time.time()
        nn = P87.tcn_probs(q, arm)
        done[f"pn_{arm}"] = nn
        done[f"p2_{arm}"], its[arm] = decode_iters(comb, simc, P87.blend(q, q.p0t.to_numpy(), nn))
        done.to_parquet(fq, index=False)
        json.dump(its, open(fi, "w"), indent=1)
        log(f"{arm} decoded in {time.time()-t0:.0f}s, iterations {its[arm]}")
    if "tcn_ad" in done and "p2_tcn_ad" in done:
        ref = pd.read_parquet(C.DC_WORK / "s87" / "p87_q.parquet", columns=["p2_tcn_ad"])
        log(f"tcn_ad vs note 87b decode: max |dp2| {np.abs(ref.p2_tcn_ad.to_numpy() - done.p2_tcn_ad.to_numpy()).max():.2e}")


def stage_score(a):
    P87.stage_score(a)


def stage_fitdec(a):
    """full-data decoder on the chosen arm's blend (fit76.stage_decoder recipe, n = mean of the six fold iterations)."""
    import lightgbm as lgb
    import decode_train as dec
    import train_official as T
    comb, simc = _pool()
    q = P87.base()
    its = json.load(open(OUT / "dec_iters.json"))[a.arm]
    nn = P87.tcn_probs(q, a.arm)
    p0 = P87.blend(q, q.p0t.to_numpy(), nn)
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[C.KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[C.KEY4 + ["Phase", "y"]], on=C.KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    labm = X.Phase.notna().to_numpy()
    n_est = int(round(np.mean(its)))
    P = dict(T.BIN_PARAMS, n_jobs=NJ, bagging_seed=0, feature_fraction_seed=100, seed=0)
    P.pop("n_estimators")
    FIT.mkdir(parents=True, exist_ok=True)
    m = lgb.LGBMClassifier(n_estimators=n_est, **P).fit(X.loc[labm, cols2], X.y[labm])
    m.booster_.save_model(str(FIT / "decode_v3.txt"))
    np.save(FIT / "X_check.npy", X.loc[labm, cols2].to_numpy(np.float64)[:2000])
    src = {"tcn_ad": "note-53 inference", "tcn_ad76": "note-86 re-inference with the note-76 tcn53.py"}[a.arm]
    json.dump({"features": cols2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
               "n_estimators_rule": f"mean best iteration of the six note-90 fold fits {its}",
               "params": {k: v for k, v in P.items() if k != "n_jobs"},
               "first_stage_input": "note-76 trees bag (3 seeds, phase-order-free features) 0.5 / TCN ad_all (1 s, 15 "
                                    f"channels; six-fold OOF of the ad_all_e100 fold models, {src}) 0.5, network only on "
                                    "candidates with tree p >= .01",
               "arm": a.arm, "trained_on": {"labelled_pair_rows": int(labm.sum())}},
              open(FIT / "decode_v3.json", "w"), indent=1)
    log(f"decoder ({a.arm}): n_estimators {n_est}, {int(labm.sum()):,} labelled rows -> {FIT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["decode", "score", "fitdec"])
    ap.add_argument("--arms", default="tcn_ad")
    ap.add_argument("--arm", default="tcn_ad")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
