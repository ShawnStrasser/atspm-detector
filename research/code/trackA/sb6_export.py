"""Note 48: export the note-41 setback model (sb5_final.pkl) to numpy-runnable files for the final_v3
candidate package, and prove parity.

  * the 3 x 3 LightGBM quantile regressors (P10 / P50 / P90 of log distance) -> LightGBM text models, run by
    the package's lgbm_numpy.NumpyBooster (quantile objective = raw score);
  * the note-30 same-lane pair model (sklearn HistGradientBoostingClassifier, binary) -> a JSON of its tree
    node arrays, evaluated by the package's setback.HGBNumpy;
  * L_fit / CQR width / feature list / confidence cut points -> setback_model.json.

    python sb6_export.py <package_dir>          # writes <package_dir>/weights/setback/, checks parity
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb5_setback as SB  # noqa: E402


def export(pkg: Path) -> None:
    pk = pickle.load(open(SB.default_model_path(), "rb"))
    fit, pm = pk["fit"], pk["pair_model"]
    out = pkg / "weights" / "setback"
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for a, ms in fit["models"].items():
        files[str(a)] = []
        for i, m in enumerate(ms):
            f = f"setback_q{int(round(a * 100)):02d}_s{i}.txt"
            m.booster_.save_model(str(out / f))
            files[str(a)].append(f)
    trees = []
    for it in pm._predictors:
        assert len(it) == 1
        nd = it[0].nodes
        assert not nd["is_categorical"].any()
        trees.append({k: nd[k].tolist() for k in ("value", "feature_idx", "num_threshold",
                                                  "missing_go_to_left", "left", "right", "is_leaf")})
    base = float(np.ravel(pm._baseline_prediction)[0])
    json.dump({"baseline": base, "trees": trees, "n_features": int(pm.n_features_in_),
               "features": SB.PAIR_COLS}, open(out / "setback_pair_hgb.json", "w"))
    meta = {"quantile_models": files, "L_fit": float(fit["L_fit"]), "cqr": float(fit["cqr"]),
            "features": list(fit["feats"]), "conf_cuts": [float(x) for x in fit["conf_cuts"]],
            "median_ft": float(fit["median"]), "hours_trained": pk["hours_trained"],
            "n_train": pk["n_train"], "pair_model": "setback_pair_hgb.json",
            "note": "note 41 (research/notes/41_setback_all_advance.md), exported for note 48"}
    json.dump(meta, open(out / "setback_model.json", "w"), indent=1)
    print("exported to", out)

    # ---- parity on the research feature table (every predicted Advance / Mid, 66 h)
    sys.path.insert(0, str(pkg))
    import setback as PS  # the package module
    F = pd.read_parquet(SB.default_model_path().parent / "sb5_feat_h66.parquet")
    ref = SB.apply_fit(F, fit)
    M = PS.SetbackModel(out)
    got = M.apply(F)
    d = {c: float(np.nanmax(np.abs(ref[c].to_numpy(float) - got[c].to_numpy(float)))) for c in ("p10", "p50", "p90")}
    print("quantile parity (max abs ft):", d)
    Xp = np.random.default_rng(0).normal(size=(5000, len(SB.PAIR_COLS)))
    Xp[np.random.default_rng(1).random(Xp.shape) < 0.2] = np.nan
    dp = float(np.abs(pm.predict_proba(Xp)[:, 1] - M.pair.predict_proba(Xp)).max())
    print("pair model parity on random inputs with NaN (max abs prob):", dp)
    assert max(d.values()) < 1e-6 and dp < 1e-9
    json.dump({"quantile_max_abs_ft": d, "pair_max_abs_prob": dp, "n_rows": len(F)},
              open(out / "export_parity.json", "w"), indent=1)


if __name__ == "__main__":
    export(Path(sys.argv[1]))
