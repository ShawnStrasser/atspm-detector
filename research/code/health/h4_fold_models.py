"""Note 46: the six fold models of the final function recipe (function_v3d = note 45 (a+h3), first.all.wi),
SAVED, so new windows (rolling windows, windows with bad periods removed) can be scored out of fold: a signal is
only ever scored by the fold model that never saw it.  Same code, rows, params and seeds as
v3_retrain.py fit (--frame v6 --min-on 5 --clean --require-validated --allow-not-checkable-high --health3);
check: each refit's predictions vs the saved OOF of run_ad6ea6959f_exclude_min5_clean_valnc_h3.

    python h4_fold_models.py [--seeds 0,1,2] -> %DC_WORK%/health4/func_folds/f{k}_s{s}.txt + check.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackA"))
import a2_model as A2  # noqa: E402
import v3_retrain as V  # noqa: E402

RUN = "run_ad6ea6959f_exclude_min5_clean_valnc_h3"
VARIANT = "first.all.wi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import lightgbm as lgb
    V.set_frame("v6")
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH, V.HEALTH3 = 5, True, True, True, True
    out = V.DCW / "health4" / "func_folds"
    out.mkdir(parents=True, exist_ok=True)
    fr, cols = V.load_feats()
    assert not fr.DeviceId.str.lower().isin(V.locked_signals()).any()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, VARIANT)
    assert V.run_dir("exclude").name == RUN, V.run_dir("exclude").name
    yi = pd.Series(y).map({c: i for i, c in enumerate(V.C7)}).fillna(-1).astype(int).to_numpy()
    folds = fr.fold.to_numpy()
    chk = {}
    for s in [int(x) for x in a.seeds.split(",")]:
        for k in range(A2.N_FOLDS):
            f = out / f"f{k}_s{s}.txt"
            t0 = time.time()
            inner = (k + 1) % A2.N_FOLDS
            trm, vam = ok & (folds != k) & (folds != inner), ok & (folds == inner)
            prm = dict(A2.FUNC_PARAMS, n_jobs=a.threads, num_class=len(V.C7), seed=s, bagging_seed=s + 1,
                       feature_fraction_seed=s + 2, data_random_seed=s + 3)
            n = prm.pop("n_estimators")
            m = lgb.LGBMClassifier(n_estimators=n, **prm)
            m.fit(fr.loc[trm, cols], yi[trm], eval_set=[(fr.loc[vam, cols], yi[vam])], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            m.booster_.save_model(str(f), num_iteration=m.best_iteration_)
            P = m.predict_proba(fr.loc[folds == k, cols])
            ref = np.load(V.OUT / RUN / f"P_{VARIANT}_s{s}_f{k}.npy")
            chk[f"f{k}_s{s}"] = {"trees": int(m.best_iteration_ or 0), "max_abs_diff_vs_saved_oof": float(np.abs(P - ref).max()),
                                 "argmax_agree": float((P.argmax(1) == ref.argmax(1)).mean()), "secs": round(time.time() - t0)}
            V.log(f"{f.name}: {chk[f'f{k}_s{s}']}")
    json.dump({"features": cols, "classes": V.C7, "run": RUN, "variant": VARIANT, "check": chk},
              open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
