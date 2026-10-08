"""Note 76: full-data refits of every tree model that reads the changed features (fit75 recipes, fixed rounds = mean
best iteration of the note-76 six-fold fits), for the candidate package.  -> %DC_WORK%/final_v3_work/v3fit76/

    python fit76.py phase      # pair ranker, patched pool (f76/phase), 3 seeds, n = mean of the 18 f76 fold fits
    python fit76.py decoder    # decoder on the f76 first-stage blend (new trees bag + p3 GRU, filter .01), n = mean of 6
    python fit76.py func       # 229 function trees on the chosen frame (F76_ARM: '' or 'c'), fit75.stage_func
    python fit76.py lanes      # lanes D on the same frame + the new OOF function probabilities, fit75.stage_lanes
    python fit76.py stacker    # context stacker (3-seed + single-member) on the new OOF tree probabilities
    python fit76.py setback    # unchanged models copied from v3fit (setback reads none of the changed columns)
One stage per process (stages patch module paths differently).  locked_v2 asserted absent.  CPU, 6 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
from common import DC_WORK  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
import fit75 as F75  # noqa: E402

F76 = DC_WORK / "final_v3_work" / "f76"
OUT = DC_WORK / "final_v3_work" / "v3fit76"
ARM = os.environ.get("F76_ARM", "")
F75.OUT = OUT
log = F75.log


def _func_paths(patch_frame: bool):
    import v3_retrain as V
    import s59_step6 as S59
    import t57_function as T57F
    T57F.OUT = F76 / ("function_c" if ARM == "c" else "function")
    d = T57F.cfg_dir("v6e", "drop:pp_xcand+pp_pdiff+pp_v2+yr+ratio+phctx")
    S59.FUNC_DIR = d
    if patch_frame:
        orig = V.set_frame

        def sf(tag):
            orig(tag)
            V.OUT = F76 / ("frame_v6e_c" if ARM == "c" else "frame_v6e")
            V.FEATS = V.OUT / "feat_frame.parquet"
        V.set_frame = sf
        V.set_frame("v6e")
    return d


def stage_phase(a):
    import lightgbm as lgb
    import t57_phase as T57
    import train_official as T
    T57.OUT = F76 / "phase"
    comb, simc, fc = T57.load_pool()
    assert not T57.plain(comb.DeviceId).isin(F75.locked()).any()
    rec = [json.loads(x) for x in open(F76 / "phase" / "full" / "timing.jsonl")]
    assert len(rec) == 18
    n_est = int(round(np.mean([r["trees"] for r in rec])))
    lab = comb[comb.Phase.notna()].sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    d = OUT / "phase"
    d.mkdir(parents=True, exist_ok=True)
    log(f"phase ranker: {len(fc)} features, {len(lab):,} labelled pair rows, n_estimators {n_est}")
    for s in (0, 1, 2):
        f = d / f"phase_lgbm_v5_s{s}.txt"
        if f.exists():
            continue
        P = T57.rp(s)
        P.pop("n_estimators")
        t0 = time.time()
        m = lgb.LGBMRanker(n_estimators=n_est, **P).fit(lab[fc], lab["y"], group=T._groups(lab))
        m.booster_.save_model(str(f))
        log(f"  seed {s}: {time.time() - t0:.0f}s")
    F75.save_json({"features": fc, "n_estimators": n_est, "rule": "mean best iteration of the 18 note-76 fold fits",
                   "trained_on": {"labelled_pair_rows": int(len(lab))}}, d / "phase_fit76.json")


def stage_decoder(a):
    import lightgbm as lgb
    import cand64 as C
    import decode_train as dec
    import train_official as T
    import f76_phase as FP
    d = OUT / "decoder"
    d.mkdir(parents=True, exist_ok=True)
    need = C.KEY4 + ["fold", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(F76 / "phase" / "pool.parquet", columns=need)
    simc = pd.read_parquet(F76 / "phase" / "pool_sim.parquet")
    assert not C.plain(comb.DeviceId).isin(F75.locked()).any()
    q = pd.read_parquet(C.OUT / "phase_oof.parquet")
    nq = pd.read_parquet(F76 / "phase" / "f76_q.parquet")
    for c in C.KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all() and (nq[c].to_numpy() == comb[c].to_numpy()).all(), c
    p0 = FP.blend(q, nq.p0_new.to_numpy(), nq.p0_new.to_numpy() >= 0.01)
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[C.KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[C.KEY4 + ["Phase", "y"]], on=C.KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    labm = X.Phase.notna().to_numpy()
    iters = [116, 258, 453, 227, 107, 310]                  # f76_phase.py score: decoder fold fits (phase_score.log)
    n_est = int(round(np.mean(iters)))
    P = dict(T.BIN_PARAMS, n_jobs=6, bagging_seed=0, feature_fraction_seed=100, seed=0)
    P.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n_est, **P).fit(X.loc[labm, cols2], X.y[labm])
    m.booster_.save_model(str(d / "decode_v3.txt"))
    F75.save_json({"features": cols2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
                   "n_estimators_rule": "mean best iteration of the six note-76 fold fits " + str(iters),
                   "params": {k: v for k, v in P.items() if k != "n_jobs"},
                   "first_stage_input": "note-76 trees bag (3 seeds, phase-order-free features) 0.5 / p3 GRU 0.5, GRU only "
                                        "on candidates with tree p >= .01, six-fold OOF",
                   "trained_on": {"labelled_pair_rows": int(labm.sum())}}, d / "decode_v3.json")
    log(f"decoder: n_estimators {n_est}, {int(labm.sum()):,} labelled rows")


def stage_func(a):
    _func_paths(True)
    F75.stage_func(a)


def stage_lanes(a):
    d = _func_paths(True)
    import ln8_validate as L8
    L8.FUNC_DIR = d
    F75.stage_lanes(a)


def stage_stacker(a):
    sys.path.insert(0, str(rpath.CODE / "evaluation"))
    _func_paths(False)
    F75.stage_stacker(a)
    F75.stage_stacker_single(a)


def stage_setback(a):
    shutil.copytree(DC_WORK / "final_v3_work" / "v3fit" / "setback", OUT / "setback", dirs_exist_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["phase", "decoder", "func", "lanes", "stacker", "setback"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
