"""Note 77: full-data refits for the candidate package after the channel-order fixes.  -> %DC_WORK%/final_v3_work/v3fit77/
(starts as a copy of v3fit76; only the lane model and the stackers change).

    python fit77.py lanes      # lanes D on BOTH pair orientations (f76 arm-c frame + function probabilities, as fit76),
                               # lam = mode of the note-77 per-fold picks
    python fit77.py setback    # setback P50 (sb7 recipe) on f77/sb7/feat.parquet (symmetric travel-time block)
    python fit77.py stacker    # context stacker (3-seed + single-member) on the note-77 context (order-free lanes, pick
                               # inputs, stack health); trees = note-76 arm c
One stage per process.  Run with F76_ARM=c.  locked_v2 asserted absent (frame loaders).  CPU, 6 threads.
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
sys.path.insert(0, str(rpath.CODE / "final76"))
assert os.environ.get("F76_ARM") == "c", "run with F76_ARM=c"
import fit76 as F76  # noqa: E402
import fit75 as F75  # noqa: E402

OUT = DC_WORK / "final_v3_work" / "v3fit77"
if not OUT.exists():
    shutil.copytree(DC_WORK / "final_v3_work" / "v3fit76", OUT)
F75.OUT = OUT
F76.OUT = OUT
log = F75.log


def stage_lanes(a):
    import lightgbm as lgb
    import lane_output as LO
    import ln8_validate as L8
    import of77
    d = F76._func_paths(True)
    L8.FUNC_DIR = d
    dd = OUT / "lanes"
    dd.mkdir(parents=True, exist_ok=True)
    D, Pf, Pr = of77.sym_frames()
    assert not Pf.DeviceId.isin(F75.locked()).any()
    il = np.flatnonzero(Pf.labelled.to_numpy())
    c2 = L8.c2_matrix(Pf, D, il)
    X = np.vstack([np.hstack([Pf.loc[il, LO.FEATURES].to_numpy(np.float32), c2[0]]),
                   np.hstack([Pr.loc[il, LO.FEATURES].to_numpy(np.float32), c2[0]])])
    names = LO.FEATURES + c2[1]
    y = np.concatenate([Pf.same_lane.to_numpy()[il]] * 2).astype(int)
    log(f"lanes D (both orientations): {len(il):,} labelled pair-windows x 2, {X.shape[1]} features")
    files = []
    for s in L8.SEEDS:
        f = dd / f"lane_pair_D_s{s}.txt"
        files.append(f.name)
        lgb.LGBMClassifier(random_state=s, **dict(L8.PARAMS, n_jobs=6)).fit(pd.DataFrame(X, columns=names), y) \
            .booster_.save_model(str(f))
        log(f"  seed {s} done")
    det, ph, T = L8.truth_tabs()
    D9 = D[(D.period == "stg") & D.win.isin(L8.TRAIN_WINS) & D.DeviceId.isin(set(ph.DeviceId))]
    pri = L8.span_priors(det, ph, D9.rename(columns={"Detector": "det"}), set(D9.DeviceId))[0]
    pk = json.load(open(of77.O8 / "decode_pick.json"))["pick"]["D.func"]
    lams = [float(v.split("@")[1]) for v in pk.values()]
    lam = float(pd.Series(lams).mode().iloc[0])
    F75.save_json({"pair_models": files, "features": names, "n_cue_context_features": len(LO.FEATURES),
                   "c2_columns": json.load(open(L8.FUNC_DIR / "cols.json"))["cols"] + [f"P_{c}" for c in LO.C7],
                   "span_prior": {fn: {str(k): v for k, v in x.items()} for fn, x in pri.items()},
                   "lam": lam, "lam_fold_picks": lams, "beta": 0.0, "role_pen": 4.0, "min_on": LO.MIN_ON,
                   "params": {k: v for k, v in L8.PARAMS.items() if k != "n_jobs"},
                   "orientation": "both (note 77): trained on (a, b) and (b, a) of every labelled pair; at inference "
                                  "P(same) = mean of the two orientations; decode in canonical behavioural order",
                   "note": "note 58 joint pair model D: cues + context + predicted-function pair type + function model "
                           "block (229 features + 7 probabilities of both detectors, min / max); decode = note-42 func "
                           "decoder",
                   "trained_on": {"labelled_pairs": int(len(il)), "rows": int(len(y)),
                                  "signals": int(Pf.loc[il, "DeviceId"].nunique()), "same_share": float(y.mean())}},
                  dd / "lane_model.json")
    log(f"lam {lam} (fold picks {lams})")


def stage_stacker(a):
    sys.path.insert(0, str(rpath.CODE / "evaluation"))
    F76._func_paths(False)
    import of77
    import s74 as S74
    import s59_step6 as S59
    import s67_decider as S
    S59.scoring_frame = of77.scoring_frame77
    orig = S74.setup

    def setup(main):
        E = orig(main)
        S._CTX.clear()
        S._CTX["H"] = of77.health_feats77(E["fr"][S59.KEY].reset_index(drop=True)).to_numpy(np.float32)
        S._CTX["ln"] = of77.lane_ctx(E["fr"], of77.O8 / "lanes_D.func.parquet")
        return E
    S74.setup = setup
    for f in (OUT / "stacker").glob("stacker*_s*.txt"):
        f.unlink()
    F75.stage_stacker(a)
    F75.stage_stacker_single(a)


def stage_setback(a):
    """setback sb7 P50 refit (fit75 recipe) on the note-77 features (symmetric same-lane pair model, behavioural
    partner tie-breaks)."""
    import sb7_validate as S7
    S7.OUT = DC_WORK / "final_v3_work" / "f77" / "sb7"
    assert (S7.OUT / "feat.parquet").exists()
    for f in (OUT / "setback").glob("setback_*_q50_s*.txt"):
        f.unlink()
    F75.stage_setback(a)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lanes", "stacker", "setback"])
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
