"""Note 76: how often does a CHANNEL-number tie-break decide a lag feature (function.add_lag_features 'best' / 'twin'
rows, SQL_LAG top-K cut)?  Read-only measurement on the stored Dec-2024 lag tables (det_lag + det_lag_B)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import FEATURES  # noqa: E402

L = pd.concat([pd.read_parquet(FEATURES / f) for f in ("det_lag.parquet", "det_lag_B.parquet")], ignore_index=True)
K = ["DeviceId", "win", "Detector"]
out = {"rows": len(L), "detwin": int(L[K].drop_duplicates().shape[0])}
for key, vals in (("lag_peak_excess", ["lag_peak", "coinc_05_excess", "n_b", "frac_after"]),
                  ("coinc_05_excess", ["lag_peak", "n_b"])):
    mx = L.groupby(K)[key].transform("max")
    top = L[L[key] == mx]
    n = top.groupby(K).size()
    tied = n[n > 1].index
    tt = top.set_index(K).loc[tied]
    differ = tt.groupby(level=[0, 1, 2])[vals].nunique().max(axis=1) > 1
    out[f"{key}: detwin with tied best"] = int(len(tied))
    out[f"{key}: ... whose tied rows differ in the used values"] = int(differ.sum())
# top-K cut: is the 12th and 13th candidate tied?  (only visible where exactly 12 rows were kept)
c = L.groupby(K).size()
out["detwin at the top-K cap (12 rows)"] = int((c == 12).sum())
print(out)
