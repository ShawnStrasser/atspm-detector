"""The control AGENTS.md demands: the same number of PURE NOISE columns added to the
base feature set, over all six folds, so the expert family's gain can be read against
what a meaningless family 'gains'."""
import json
import numpy as np, pandas as pd
import a2_model as A2

fr = A2.load_frame()
y = fr.func5.map({c: i for i, c in enumerate(A2.CLASSES5)}).to_numpy()
yt = fr.func5.to_numpy()
base = [c for c in A2.feat_cols(fr, ("base",))]
n_px = len(A2.feat_cols(fr, tuple(f for f in A2.ALL_FAM if f != "base")))
rng = np.random.default_rng(7)
noise = [f"noise_{i}" for i in range(n_px)]
for c in noise:
    fr[c] = rng.standard_normal(len(fr)).astype(np.float32)
A2.log(f"{len(base)} base + {n_px} noise columns")
res = {}
for s in (0, 1):
    P, _, _ = A2.oof(fr, y, base + noise, seeds=(s,))
    res[f"seed{s}"] = A2.score(fr, P)
    A2.log(f"base+noise seed {s}: allwin {res[f'seed{s}']['acc5_allwin']:.4f} "
           f"full {res[f'seed{s}']['acc5_by_duration'].get('full')}")
    if s == 0:
        np.save(A2.WORK / "a2_oof_base_noise_seed0.npy", P)
json.dump(res, open(A2.WORK / "a2_noise.json", "w"), indent=1, default=str)
