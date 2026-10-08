"""Note 53 (prep): the tree ranker's per-pair engineered features as a network input.

Builds, once, the 261 phase-ranker features of the phase_v3 pool (`phase_v3_trees.build_pool_v3`,
the exact rows the trees were fitted / scored on: 780 signals x 22 windows) and stores them
standardised for the TCN hybrid (note 53 variant c):

    x -> sign(x) * log1p(|x|)  -> (x - mean) / sd  -> clip +-5  -> NaN = 0
    mean / sd from the rows of folds 1-5 (fold 0 never enters the statistics)

Output `%DC_WORK%/tcn53/feats/`: `F.npy` [rows, 261] float16, `rows.parquet` (DeviceId, Detector,
win, cand_phase, fold), `meta.json` (feature names, the top-k subset by the phase_v3 ranker's
mean gain over its three seeds).  Locked signals are asserted absent (build_pool_v3 does it).

    python tcn53_prep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

OUT = DC_WORK / "tcn53" / "feats"
TOPK = 16


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    import lightgbm as lgb
    import phase_v3_trees as P3
    OUT.mkdir(parents=True, exist_ok=True)
    comb, _, fc = P3.build_pool_v3()
    log(f"pool {comb.shape}, {len(fc)} features")
    rows = comb[["DeviceId", "Detector", "win", "cand_phase", "fold"]].copy()
    rows["DeviceId"] = rows.DeviceId.str.lower()
    X = comb[fc].to_numpy(np.float32)
    del comb
    X = np.sign(X) * np.log1p(np.abs(X))
    tr = (rows.fold != 0).to_numpy()
    mu = np.nanmean(X[tr], axis=0)
    sd = np.nanstd(X[tr], axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    X = (X - mu) / sd
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0), -5, 5)
    np.save(OUT / "F.npy", X.astype(np.float16))
    rows.to_parquet(OUT / "rows.parquet", index=False)
    # top-k by gain of the shipped-candidate ranker (phase_v3, three seeds)
    W = DC_WORK / "final_v3_work" / "phase_v3"
    gain = pd.Series(0.0, index=fc)
    for s in (0, 1, 2):
        b = lgb.Booster(model_file=str(W / f"phase_lgbm_v5_s{s}.txt"))
        g = pd.Series(b.feature_importance("gain"), index=b.feature_name())
        gain = gain.add(g.reindex(fc).fillna(0.0) / g.sum(), fill_value=0.0)
    top = gain.sort_values(ascending=False).index[:TOPK].tolist()
    json.dump({"features": fc, "topk": top, "topk_gain_share": float(
        gain[top].sum() / gain.sum()), "n_rows": int(len(rows))},
        open(OUT / "meta.json", "w"), indent=1)
    log(f"wrote {OUT}: {X.shape}; top-{TOPK} = {top} "
        f"({gain[top].sum() / gain.sum():.1%} of gain)")


if __name__ == "__main__":
    main()
