"""Note 120: bit-identity of two par120 runs.  python cmp120.py <A> <B>  -> prints and writes s120/cmp_<A>_vs_<B>.json"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
OUT = W / "s120" / "out"
A, B = sys.argv[1], sys.argv[2]
res = {"A": A, "B": B}
for k, key in (("det", ["case", "DeviceId", "Detector"]), ("cand", ["case", "DeviceId", "Detector", "cand_phase"]),
               ("ph", ["case", "DeviceId", "phase"])):
    a = pd.read_parquet(OUT / f"{A}_{k}.parquet").sort_values(key).reset_index(drop=True)
    b = pd.read_parquet(OUT / f"{B}_{k}.parquet").sort_values(key).reset_index(drop=True)
    r = {"rows": [len(a), len(b)], "cols_same": list(a.columns) == list(b.columns)}
    assert len(a) == len(b) and (a[key].astype(str).to_numpy() == b[key].astype(str).to_numpy()).all(), k
    diff = {}
    for c in a.columns:
        x, y = a[c], b[c]
        na, nb = x.isna().to_numpy(), y.isna().to_numpy()
        n = int((na != nb).sum())
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            m = ~na & ~nb
            n += int((xv[m] != yv[m]).sum())
        else:
            m = ~na & ~nb
            n += int((x[m].astype(str).to_numpy() != y[m].astype(str).to_numpy()).sum())
        if n:
            diff[c] = n
    r["cells_differing"] = diff
    r["cases"] = int(a.case.nunique())
    res[k] = r
print(json.dumps(res, indent=1))
json.dump(res, open(W / "s120" / f"cmp_{A}_vs_{B}.json", "w"), indent=1)
