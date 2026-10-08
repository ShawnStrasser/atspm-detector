"""Note 84: the 'mean3' context stacker for candidate v4b (three full-data siba v4l members averaged).

Recipe = fit83 'single' (v4l target, v4l OOF trees, 47 columns = 15 health columns out, LightGBM PRM0, 150 rounds, seeds
0/1/2) except the net input: the 3-seed MEAN of the siba OOF fold models (x69_siba, seeds 0/1/2), one row per
detector-window (as notes 67 / 69), not the three single-seed versions stacked.

    set F76_ARM=c & python fit84.py stacker3   -> %DC_WORK%/final_v3_work/v3fit84/stacker/stacker_mean3_s{0,1,2}.txt
CPU, LightGBM 4 threads.  locked_v2 asserted absent (fit83.stacker_inputs).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final83"))
import fit83 as F  # noqa: E402

OUT = F.DC_WORK / "final_v3_work" / "v3fit84"


def net_X(s74, S, E, P1):
    """the 47-column stacker matrix for net probabilities P1 (NaN rows -> the trees, as fit83)."""
    import cand64 as C
    fr, Pt = E["fr"], E["Pt"]
    P1 = np.where(np.isnan(P1[:, :1]), Pt, P1)
    X = np.hstack([S.stack_X(fr, Pt, P1), S.ctx_X(E, C.W_TREE * Pt + (1 - C.W_TREE) * P1)])
    assert X.shape[1] == 62, X.shape
    return np.delete(X, F.HEALTH15, axis=1)


def mean3_inputs():
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    Xm = net_X(s74, S, E, s74.net_probs(E["fr"], "x69_siba"))       # 3-seed mean of the fold models
    return s74, S, E, Xs, Xm, y, trm, names


def stage_stacker3():
    import lightgbm as lgb
    d = OUT / "stacker"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("stacker_mean3*"):
        f.unlink()
    s74, S, E, Xs, Xm, y, trm, names = mean3_inputs()
    X, yy = Xm[trm], y[trm]
    F.log(f"stacker mean3 (v4l, no health): X {X.shape}")
    np.save(d / "X_check_mean3.npy", Xm[:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_mean3_s{s}.txt"
        lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=s), lgb.Dataset(X, yy), num_boost_round=150).save_model(str(f))
        files.append(f.name)
        F.log(f"  seed {s} done")
    fr = E["fr"]
    json.dump({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files,
               "params": F.PRM0, "num_boost_round": 150,
               "variant": "mean3: trained on the 3-seed MEAN of the siba OOF fold models (1 x rows, notes 67 / 69)",
               "net": "siba x69_siba 3-seed-mean OOF fold models (v3s-trained); production = 3 full-data siba v4l "
                      "members averaged",
               "labels": "v4l (note 80/81); trees = note-80 v4l OOF",
               "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique())}},
              open(d / "stacker_mean3.json", "w"), indent=1)


if __name__ == "__main__":
    t0 = time.time()
    {"stacker3": stage_stacker3}[sys.argv[1]]()
    F.log(f"== {sys.argv[1]} done ({time.time() - t0:.0f}s)")
