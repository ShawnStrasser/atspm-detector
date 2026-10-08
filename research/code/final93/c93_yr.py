"""Note 93 B: paired comparison of the yr_nc arm (YR-no-Count out of training) vs base (v4o), note-89 recipe seed 0.
Base = %DC_WORK%/x89/v4o/ok_base.npz if present (refit here), else note 89b's x89/v4n/ok_not_checkable.npz (same v4o
training rows).  Scoring with and without the YR-no-Count rows.   -> %DC_WORK%/s93/c93.json"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import rpath  # noqa
import cand64 as C
DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
X = DCW / "x89"
res = {}
B = {"v4n_not_checkable": np.load(X / "v4n" / "ok_not_checkable.npz")}
if (X / "v4o" / "ok_base.npz").exists():
    B["v4o_base"] = np.load(X / "v4o" / "ok_base.npz")
A = np.load(X / "v4o" / "ok_yr_nc.npz")
M = pd.read_parquet(DCW / "s93" / "yr_nc_members.parquet")
mk = set(zip(M.DeviceId.str.lower(), M.detector.astype(int)))
sig = A["sig"]
yrnc = np.array([(s.lower(), int(d)) in mk for s, d in zip(sig, A["det"])])
for bn, b in B.items():
    for k in ("sig", "win", "det", "truth"):
        assert (b[k] == A[k]).all(), k
    if bn == "v4o_base" and "v4n_not_checkable" in B:
        res["base_repro_max_abs_diff_E"] = float(np.nanmax(np.abs(B["v4n_not_checkable"]["E"] - b["E"])))
    r = {}
    for st in ("E", "R"):
        for pool, fams in C.POOLS.items():
            pm = np.isin(A["wgroup"], fams)
            for tag, m in (("all", pm), ("without_yrnc_rows", pm & ~yrnc), ("yrnc_rows", pm & yrnc),
                           ("YR_class", pm & (A["truth"] == "Yellow_Red"))):
                mm = m & ~np.isnan(b[st]) & ~np.isnan(A[st])
                r[f"{st}_{pool}_{tag}"] = {"n": int(mm.sum()), "base": C.acc_ci(b[st][mm], sig[mm]),
                                           "yr_nc_out": C.acc_ci(A[st][mm], sig[mm]),
                                           "delta_pt": C.delta_ci(b[st][mm], A[st][mm], sig[mm]) if mm.sum() else None}
    res[bn] = r
json.dump(res, open(DCW / "s93" / "c93.json", "w"), indent=1, default=str)
for bn in B:
    for k, v in res[bn].items():
        if "_ge30_" in k or "m5_all" in k or "m10_all" in k:
            print(bn, k, v)
print({k: v for k, v in res.items() if k.startswith("base_repro")})
