"""Freeze the A2 candidate (base + expert features) under dc_work/trackA/models/.
Nothing is copied into model/ -- this is a candidate, not a release."""
import json
import numpy as np, pandas as pd
import a2_model as A2

DEST = A2.WORK / "models" / "function_v5_expert"
DEST.mkdir(parents=True, exist_ok=True)
fr = A2.load_frame()
cols = A2.feat_cols(fr, A2.ALL_FAM)
y = fr.func5.map({c: i for i, c in enumerate(A2.CLASSES5)}).to_numpy()
ok = (fr.health_flag != "failed").to_numpy()
res = json.load(open(A2.WORK / "a2_full.json"))
n_est = 220        # ~ the mean best_iteration over the 6 folds of the OOF run
import lightgbm as lgb
prm = dict(A2.FUNC_PARAMS, num_class=5, seed=0, bagging_seed=1,
           feature_fraction_seed=2, data_random_seed=3)
prm["n_estimators"] = n_est
m = lgb.LGBMClassifier(**prm)
m.fit(fr.loc[ok, cols], y[ok])
m.booster_.save_model(str(DEST / "function_lgbm_v5.txt"))
imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(
    ascending=False)
imp.to_csv(DEST / "feature_importance.csv", header=["gain"])
meta = {
    "classes": A2.CLASSES5, "features": cols, "n_estimators": n_est,
    "params": {k: v for k, v in prm.items() if k != "n_estimators"},
    "label_source": "research/labels/function_labels_v2.parquet (corrected labels)",
    "new_feature_families": {f: [c for c in cols if A2.family_of(c) == f]
                             for f in ("shape", "demand", "cycle", "pair")},
    "n_rows": int(ok.sum()), "n_signal_periods": int(fr.DeviceId.nunique()),
    "oof_6fold": res["base+expert"]["seed_mean"],
    "oof_6fold_base_only": res["base"]["seed_mean"],
    "paired_vs_base": res["paired"],
    "status": "CANDIDATE -- not shipped; gain over the production feature set is "
              "+0.16 pt all-windows, far below the 1 pt promotion bar in AGENTS.md",
}
json.dump(meta, open(DEST / "function_lgbm_v5.json", "w"), indent=1, default=str)
print("top 20 gain:\n" + imp.head(20).round(0).to_string())
print(f"\nexpert features in the top 40: "
      f"{[c for c in imp.head(40).index if c.startswith('px_')]}")
print(f"wrote {DEST}")
