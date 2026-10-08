"""Note 95b: does stochastic weight averaging (SWA) of ONE siba run replace the 3-seed mean? Fold 0 only (the SWA run
s95_sibaswa f0 = seed 0, weights averaged over the last 25 % of its 57 epochs, BatchNorm statistics averaged).
Stacker (47 cols, PRM0, 150 rounds, seeds 0-2) trained on folds 1-5 Sept-2026 rows with one input version, applied to
fold 0 with the arm's input; gate .9 decode; v4q truth; fold-0 Sept-2026 rows.
  seed0   train seed-0 inputs, apply seed-0 fold-0 net        (single run)
  swa     train seed-0 inputs, apply the SWA fold-0 net       (single run, averaged weights)
  mean3   train 3-seed-mean inputs, apply the 3-seed mean     (three runs)
    python swa95.py   -> %DC_WORK%/final_v3_work/f95/oof/swa95.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("F76_ARM", "c")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import func95 as FN  # noqa: E402

F, F4 = FN.fsetup()
import lightgbm as lgb  # noqa: E402
import cand64 as C  # noqa: E402
import of77  # noqa: E402

s74, S, E, Xs, y, trm, names = F.stacker_inputs()
fr = E["fr"]
st = (fr.period == "stg").to_numpy()
fo = fr.fold.to_numpy()
P0 = s74.net_probs(fr, "s95_siba:0")
Pm = s74.net_probs(fr, "s95_siba")
Pw = P0.copy()
import pandas as pd  # noqa: E402
n = pd.read_parquet(C.FPRED / "s95_sibaswa_f0.parquet")
n["DeviceId"] = n.DeviceId.str.lower()
n = n.astype({"Detector": fr.Detector.dtype})
m = fr[s74.KEY].merge(n[s74.KEY + s74.PC], on=s74.KEY, how="left")[s74.PC].to_numpy(float)
Pw[fo == 0] = m[fo == 0]
lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
tr = trm & st & (fo != 0)
te = fo == 0
res = {}
for arm, (Ptr, Pap) in {"seed0": (P0, P0), "swa": (P0, Pw), "mean3": (Pm, Pm)}.items():
    Xt, Xa = F4.net_X(s74, S, E, Ptr), F4.net_X(s74, S, E, Pap)
    R = np.full((len(y), 7), 1 / 7)
    R[te] = 0
    for seed in (0, 1, 2):
        mm = lgb.train(dict(F.PRM0, num_threads=6, seed=seed), lgb.Dataset(Xt[tr], y[tr]), num_boost_round=150)
        R[te] += mm.predict(Xa[te]) / 3
    ok = s74.gate_ok(E, R, lc)
    res[arm] = ok
sig = fr.DeviceId.to_numpy()
out = {}
for stn in ("E", "R"):
    for pool, fams in C.POOLS.items():
        sc = te & st & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce([~np.isnan(res[k][stn]) for k in res])
        r = {"n": int(sc.sum()), "signals": int(len(set(sig[sc])))}
        for k in res:
            r[k] = C.acc_ci(res[k][stn][sc], sig[sc])
        r["swa - seed0"] = C.delta_ci(res["seed0"][stn][sc], res["swa"][stn][sc], sig[sc])
        r["mean3 - seed0"] = C.delta_ci(res["seed0"][stn][sc], res["mean3"][stn][sc], sig[sc])
        r["mean3 - swa"] = C.delta_ci(res["swa"][stn][sc], res["mean3"][stn][sc], sig[sc])
        out[f"{stn}_{pool}"] = r
json.dump(out, open(FN.OOF95 / "swa95.json", "w"), indent=1)
for k, v in out.items():
    print(k, v, flush=True)
