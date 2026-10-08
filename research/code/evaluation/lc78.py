"""Note 78: learning curve of the function trees arm (229 features, note-76 arm c frame, note-57 recipe).

For every held-out fold k the training signals (folds != k, != inner) are subsampled to 25 / 50 / 75 % (nested: the
25 % set is inside the 50 % set ...), two subsample draws (78, 79), model seed 0; the inner (early-stopping) fold and
the held-out fold are untouched.  100 % = the existing note-76 arm-c seed-0 fit (same recipe).  Output
%DC_WORK%/err78/lc/P_<frac>_d<draw>_f<k>.npy (held-out fold rows, frame order).  Scoring is in err78.py (lc stage).
locked_v2 asserted absent.  CPU only, 6 threads.

    set F76_ARM=c & python lc78.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["F76_ARM"] = "c"
CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "final76"))
import rpath  # noqa: F401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import f76_function as F  # noqa: E402,F401  (patches V.set_frame / T57F.OUT to the arm-c frame)
import t57_function as T57F  # noqa: E402
import v3_retrain as V  # noqa: E402
import a2_model as A2  # noqa: E402
from common import DC_WORK  # noqa: E402

OUT = DC_WORK / "err78" / "lc"
FRACS = (0.25, 0.5, 0.75)
DRAWS = (78, 79)


def main():
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    T57F.setup("v6e")
    fr, cols = V.load_feats()
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not fr.DeviceId.str.lower().isin(lk).any()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, T57F.VAR)
    yi = pd.Series(y).map({c: i for i, c in enumerate(V.C7)}).fillna(-1).astype(int).to_numpy()
    gs = set(F.CFG[5:].split("+"))
    cols = [c for c in cols if not (T57F.group_of(c) & gs)]
    assert len(cols) == 229, len(cols)
    folds = fr.fold.to_numpy()
    dev = fr.DeviceId.str.lower().to_numpy()
    for d in DRAWS:
        for k in range(A2.N_FOLDS):
            inner = (k + 1) % A2.N_FOLDS
            base = ok & (folds != k) & (folds != inner)
            sig = np.array(sorted(set(dev[base])))
            perm = np.random.default_rng(d * 10 + k).permutation(len(sig))
            vam = ok & (folds == inner)
            for f in FRACS:
                fo = OUT / f"P_{f:.2f}_d{d}_f{k}.npy"
                if fo.exists():
                    continue
                keep = set(sig[perm[:int(round(f * len(sig)))]])
                trm = base & np.isin(dev, list(keep))
                t0 = time.time()
                prm = dict(A2.FUNC_PARAMS, n_jobs=6, num_class=len(V.C7), seed=0, bagging_seed=1,
                           feature_fraction_seed=2, data_random_seed=3)
                n = prm.pop("n_estimators")
                m = lgb.LGBMClassifier(n_estimators=n, **prm)
                m.fit(fr.loc[trm, cols], yi[trm], eval_set=[(fr.loc[vam, cols], yi[vam])], eval_metric="multi_logloss",
                      callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
                np.save(fo, m.predict_proba(fr.loc[folds == k, cols]).astype(np.float32))
                print(f"draw {d} fold {k} frac {f}: {len(keep)}/{len(sig)} signals, {int(trm.sum()):,} rows, "
                      f"{m.best_iteration_} trees, {time.time() - t0:.0f}s", flush=True)
    print("LC DONE", flush=True)


if __name__ == "__main__":
    main()
