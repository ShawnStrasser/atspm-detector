"""Note 95: full-data FUNCTION refits for package v5 on 2026-only data (training rows = Sept-2026 period, labels v4q),
oof88 / oof90 recipes, -> %DC_WORK%/final_v3_work/v3fit95 (a copy of v3fit90; working copies f95/ln8, f95/sb7).

    python pkg95.py pkgfunc      229 trees x 3 seeds, n = mean best iteration of the 18 2026 fold fits (function_c_v4q26)
    python pkg95.py pkglanes     lanes D both orientations (2026 trees OOF function block; lane truth = v4q print lanes)
    python pkg95.py pkgsbfeats   setback sb7 features (2026 trees OOF function)
    python pkg95.py pkgsetback   setback P50 per length group
    python pkg95.py pkgsingle    'single' fallback stacker on the 2026 single-seed siba OOF (run BEFORE pkgstacker)
    python pkg95.py pkgstacker   mean3 stacker on the filtered 3-seed-mean 2026 siba OOF (s95_siba*), Sept-2026 rows only
Run with F76_ARM=c.  CPU 6 threads.  locked_v2 asserted absent by the loaders.  One stage per process.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("F76_ARM", "c")
os.environ["DC_PKG_LABELS"] = "v4q"
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
for _d in ("final84", "final83", "final88", "final86", "final90", "final95"):
    sys.path.insert(0, str(CODE / _d))
import fit84 as F4  # noqa: E402
import fit83 as F  # noqa: E402
import oof88 as O88  # noqa: E402
import func95 as FN  # noqa: E402

DCW = F.DC_WORK
O88.TREES["v4q"] = FN.TREES95
O88.FIT = DCW / "final_v3_work" / "v3fit95"
O88.F88 = DCW / "final_v3_work" / "f95"
F.THREADS = 6
F.F75.THREADS = 6


def _check():
    """replaces oof88.check_truth_same (v4q truth differs from v4l by design): locked_v2 absent only."""
    import pandas as pd
    b = pd.read_parquet(O88.LAB["v4q"], columns=["DeviceId"])
    assert not b.DeviceId.str.lower().isin(F.F75.locked()).any()


O88.check_truth_same = _check


def _mask():
    import v3_retrain as V
    FN.patch_2026(V)


def stage_pkgfunc(a):
    _mask()
    O88.stage_pkgfunc(a)
    m = json.load(open(O88.FIT / "function" / "function229.json"))
    m["recipe"] = m["recipe"] + " | note 95: Sept-2026 rows only (Dec-2024 never trained on), labels v4q"
    m["n_estimators_rule"] = "mean best iteration of the 18 note-95 2026-only six-fold fits (3 seeds); no early stopping"
    F.F75.save_json(m, O88.FIT / "function" / "function229.json")


def stage_pkglanes(a):
    _mask()
    O88.stage_pkglanes(a)


def stage_pkgsbfeats(a):
    _mask()
    O88.stage_pkgsbfeats(a)


def stage_pkgsetback(a):
    O88.stage_pkgsetback(a)


def stage_pkgstacker(a):
    import lightgbm as lgb
    _mask()
    O88._pkg_out()
    O88._v4m_everywhere()
    d = O88.FIT / "stacker"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("stacker_mean3*"):
        f.unlink()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    X = F4.net_X(s74, S, E, s74.net_probs(fr, FN.NET95))
    trm = trm & (fr.period == "stg").to_numpy()
    F.log(f"stacker mean3 (note 95: 2026 trees OOF, filtered {FN.NET95} mean, Sept-2026 rows): X {X[trm].shape}")
    np.save(d / "X_check_mean3.npy", X[:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_mean3_s{s}.txt"
        lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=s), lgb.Dataset(X[trm], y[trm]),
                  num_boost_round=150).save_model(str(f))
        files.append(f.name)
    json.dump({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files,
               "params": F.PRM0, "num_boost_round": 150,
               "variant": f"mean3 (note 95): trained on the 3-seed MEAN of the 2026-only siba OOF fold models ({FN.NET95}), "
                          "filtered (tree p >= .01) OOF, Sept-2026 rows only",
               "net": f"siba {FN.NET95} 3-seed-mean OOF fold models (v4q, 2026-only); production = 3 full-data members",
               "labels": "v4q (note 95); trees = 2026-only OOF (f76/function_c_v4q26)",
               "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique()), "period": "Sept-2026"}},
              open(d / "stacker_mean3.json", "w"), indent=1)


def stage_pkgsingle(a):
    """the 'single' fallback stacker (fit83 recipe: 3 x rows, one per single-seed net) on the 2026 single-seed siba OOF."""
    import lightgbm as lgb
    _mask()
    O88._pkg_out()
    O88._v4m_everywhere()
    d = O88.FIT / "stacker"
    for f in d.glob("stacker_single*"):
        f.unlink()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    trm = trm & (fr.period == "stg").to_numpy()
    Xs = {s: F4.net_X(s74, S, E, s74.net_probs(fr, f"{FN.NET95}:{s}")) for s in (0, 1, 2)}
    X = np.vstack([Xs[s][trm] for s in (0, 1, 2)])
    yy = np.concatenate([y[trm]] * 3)
    np.save(d / "X_check.npy", Xs[0][:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_single_s{s}.txt"
        lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=s), lgb.Dataset(X, yy), num_boost_round=150).save_model(str(f))
        files.append(f.name)
    F.F75.save_json({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files,
                     "params": F.PRM0, "num_boost_round": 150,
                     "variant": "single: trained on the three single-seed versions of every OOF row (3 x rows, note 75)",
                     "net": f"siba {FN.NET95} single-seed 2026-only OOF fold models (v4q)",
                     "labels": "v4q (note 95); trees = 2026-only OOF", "trained_on": {
                         "rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique()), "period": "Sept-2026"}},
                    d / "stacker.json")


def stage_extras(a):
    """note-98 extras for the fast package on v5 (f95/extras/stacker_{nonet,single}): 'nonet' = net columns carry the
    trees' probabilities; 'single' = copy of pkgsingle in the note-98 file layout. Sept-2026 training rows, v4q."""
    import lightgbm as lgb
    import shutil
    _mask()
    O88._pkg_out()
    O88._v4m_everywhere()
    root = DCW / "final_v3_work" / "f95" / "extras"
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr, Pt = E["fr"], E["Pt"]
    trm = trm & (fr.period == "stg").to_numpy()
    d = root / "stacker_nonet"
    d.mkdir(parents=True, exist_ok=True)
    X = F4.net_X(s74, S, E, np.full_like(Pt, np.nan))
    np.save(d / "X_check_nonet.npy", X[:2000].astype(np.float32))
    files = []
    for s in (0, 1, 2):
        f = d / f"stacker_nonet_s{s}.txt"
        lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=s), lgb.Dataset(X[trm], y[trm]), num_boost_round=150).save_model(str(f))
        files.append(f.name)
    json.dump({"classes": E["C7"], "n_features": len(names), "feature_names": names, "files": files, "params": F.PRM0,
               "num_boost_round": 150, "variant": "nonet (note 98 recipe, note 95 data): no function network; net columns = "
               "the 2026-only trees' probabilities; Sept-2026 rows, v4q",
               "trained_on": {"rows": int(trm.sum()), "signals": int(fr.DeviceId[trm].nunique())}},
              open(d / "stacker_nonet.json", "w"), indent=1)
    src = O88.FIT / "stacker"
    if (src / "stacker_single_s0.txt").exists():
        d = root / "stacker_single"
        d.mkdir(parents=True, exist_ok=True)
        for s in (0, 1, 2):
            shutil.copy(src / f"stacker_single_s{s}.txt", d / f"stacker_single_s{s}.txt")
        shutil.copy(src / "X_check.npy", d / "X_check_single.npy")
        m = json.load(open(src / "stacker.json"))
        json.dump(m, open(d / "stacker_single.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pkgfunc", "pkglanes", "pkgsbfeats", "pkgsetback", "pkgsingle", "pkgstacker", "extras"])
    a = ap.parse_args()
    O88.F88.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    F.log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
