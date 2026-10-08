"""Note 32 -- the final_v3 CANDIDATE function head: note 28's winner `first.all.wi` fitted on
ALL non-locked training rows (no fold held out), 3 seeds, for the candidate package.

    python v3_final_fit.py [--out %DC_WORK%/final_v3_candidate/weights] [--threads 12]
    python v3_final_fit.py --frame v6 --nc-high --out %DC_WORK%/final_v3_work/function_v3b
        (note 28 rerun on label-check v3: frame v6, validated pass + not_checkable high print)
    python v3_final_fit.py --frame v6 --nc-high --ana ana28_lc3c --out %DC_WORK%/final_v3_work/function_v3c
        (2026-09-28: released NEWTEST rows validated; not_checkable rows with a field_issue out)
    python v3_final_fit.py --frame v6 --nc-high --ana ana28_lc4 --out %DC_WORK%/final_v3_work/function_v3d
        (2026-09-29: label-check v4 -- misconfigured / unhealthy out, permissive + right-turn exemptions)

Exactly the note-28 recipe (`v3_retrain.py all --min-on 5 --clean --require-validated`,
variant first.all.wi): label_print_first, rows with >= 5 actuations, no dq_suspect, v3
`validated` == pass, whole-intersection Other rows in, unusual_layout signals out, health not
failed; 442 features (production 388 + 54 expert), 7 classes, the note-14 LightGBM params.
Tree count = mean best iteration of the 18 note-28 fold fits of first.all.wi (stage-11
practice: `function_v4.py --stage ship` used the mean of its fold best iterations).
Writes function_lgbm_v5_s{0,1,2}.txt + function_lgbm_v5.json (counts only, no names).
Locked TEST / NEWTEST signals are asserted absent (inside v3_retrain's loaders).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import v3_retrain as V

VARIANT = "first.all.wi"
RUN_TAG = "_exclude_min5_clean_val"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V.DCW / "final_v3_candidate" / "weights"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--frame", default="v5")
    ap.add_argument("--nc-high", action="store_true", dest="nc")
    ap.add_argument("--ana", default="ana28_lc3", help="v3_analyse output name in the run directory (evidence)")
    ap.add_argument("--lc-feats", action="store_true", dest="lc", help="+ label-check features (note 45 run _lc)")
    ap.add_argument("--health3", action="store_true", help="health-v3 training exclusion (run _h3, note 45)")
    a = ap.parse_args()
    import lightgbm as lgb
    sys.path.insert(0, str(V.DCW / "final_v3_candidate"))
    from lgbm_numpy import NumpyBooster

    V.set_frame(a.frame)
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH, V.LC_FEATS, V.HEALTH3 = 5, True, True, a.nc, a.lc, a.health3
    rd = V.run_dir("exclude")
    assert rd.name.endswith(RUN_TAG + ("nc" if a.nc else "") + ("_h3" if a.health3 else "") + ("_lc" if a.lc else "")) and rd.exists(), \
        f"label table changed since the note-28 run: {rd} does not exist"
    tim = pd.read_json(rd / "timing.jsonl", lines=True)
    tim = tim[tim.variant == VARIANT]
    assert len(tim) == 18, len(tim)
    n_est = int(round(tim.trees.mean()))
    V.log(f"note-28 fold fits: {len(tim)}, best iterations {tim.trees.min()}..{tim.trees.max()}, "
          f"mean -> {n_est} trees")

    fr, cols = V.load_feats()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, VARIANT)
    yi = pd.Series(y).map({c: i for i, c in enumerate(V.C7)}).fillna(-1).astype(int).to_numpy()
    assert (yi[ok] >= 0).all()
    X = fr.loc[ok, cols]
    V.log(f"{VARIANT}: {int(ok.sum()):,} rows, {lab.loc[ok, 'DeviceId'].nunique()} signals, "
          f"{len(cols)} features")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files, iters, parity = [], [], []
    for s in [int(x) for x in a.seeds.split(",")]:
        t0 = time.time()
        prm = dict(V.A2.FUNC_PARAMS, n_jobs=min(a.threads, 12), num_class=len(V.C7), seed=s,
                   bagging_seed=s + 1, feature_fraction_seed=s + 2, data_random_seed=s + 3)
        prm["n_estimators"] = n_est
        m = lgb.LGBMClassifier(**prm)
        m.fit(X, yi[ok])
        f = out / f"function_lgbm_v5_s{s}.txt"
        m.booster_.save_model(str(f))
        sub = X.iloc[np.random.default_rng(s).choice(len(X), 4000, replace=False)]
        d = float(np.abs(np.asarray(m.booster_.predict(sub)) -
                         NumpyBooster(str(f)).predict(sub)).max())
        files.append(f.name)
        parity.append(d)
        V.log(f"seed {s}: {n_est} trees, {time.time()-t0:.0f}s, numpy parity {d:.1e}")

    tr = lab.loc[ok].assign(y=y[ok], wgroup=fr.loc[ok, "wgroup"].to_numpy(),
                            period=fr.loc[ok, "period"].to_numpy())
    det = tr.drop_duplicates(["DeviceId", "Detector"])
    meta = {
        "classes": V.C7, "features": cols, "n_models": len(files), "files": files,
        "averaging": "mean of the seeds' class probabilities",
        "n_estimators": n_est,
        "n_estimators_rule": "mean best iteration of the 18 note-28 six-fold fits (3 seeds) "
                             "of first.all.wi; no early stopping on the final fit",
        "params": {k: v for k, v in dict(V.A2.FUNC_PARAMS, num_class=len(V.C7)).items()
                   if k not in ("n_estimators", "n_jobs")},
        "seeds": [int(x) for x in a.seeds.split(",")],
        "recipe": "note 28 first.all.wi: label_print_first, >= 5 actuations, no dq_suspect, "
                  "validated == pass" + (" + not_checkable rows with a high print label (not dead / "
                  "card-suspect, no field_issue)" if a.nc else "") + ("; health-v3 bad detector-periods and suspect "
                  "bad-period windows out" if a.health3 else "") + ", whole-intersection Other rows in, "
                  "unusual_layout out, health not failed; all non-locked training signals, no fold held out",
        "frame": a.frame, "oof_run": rd.name,
        "label_table_sha1_10": hashlib.sha1(V.LABELS_V3.read_bytes()).hexdigest()[:10],
        "phase_input": "the detector's predicted phase = argmax of the trees-only (ranker bag "
                       "+ joint decoder) phase probability; training rows used out-of-fold "
                       "phase predictions of the same pipeline",
        "trained_on": {"n_rows": int(ok.sum()), "n_signals": int(det.DeviceId.nunique()),
                       "n_detectors": int(len(det)),
                       "rows_by_class": tr.y.value_counts().to_dict(),
                       "detectors_by_class": det.y.value_counts().to_dict(),
                       "detectors_by_label_source": det.source.fillna("none")
                       .value_counts().to_dict(),
                       "signals_by_print_tier": det.drop_duplicates("DeviceId").tier
                       .fillna("no_print").value_counts().to_dict(),
                       "rows_by_period": tr.period.value_counts().to_dict(),
                       "rows_by_window_group": tr.wgroup.value_counts().to_dict(),
                       "windows": "22 windows per period, 5 min .. full span (66-72 h)"},
        "excluded_signals": {"locked": len(V.locked_signals()),
                             "note": "locked hold-outs, asserted absent from frame and labels"},
        "numpy_backend_max_abs_diff": max(parity),
        "oof_evidence": ("note 28: six-fold OOF, 3 seeds, FIXb 7-class .9179 vs .8765 for the "
                         "final_v2 head family (+4.14 pt [+2.62, +5.81]); core-four +2.73 pt")
        if not a.nc else f"note 28 rerun on label-check v3, six-fold OOF, 3 seeds: {rd / (a.ana + '.json')}",
    }
    json.dump(meta, open(out / "function_lgbm_v5.json", "w"), indent=1, default=str)
    V.log(f"wrote {out} ({len(files)} models)")


if __name__ == "__main__":
    main()
