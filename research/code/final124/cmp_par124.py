"""Note 124: parity of the package before (final_v7_prod_bak124) and after health v4, note-115 set (132 cases).
Every non-health column must be identical (floats with ==); health / status / review columns: changes counted.
    python cmp_par124.py -> s124/par/cmp_par124.json"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
D = W / "s124" / "par"
HEALTH = {"health_status", "health_score", "health_reason", "health_bad_periods", "health_watch",
          "health_categories", "health_config", "health_signal_note"}
DEP = {"status", "review_flag", "review_reason"}            # text that quotes the health status
a, b = pd.read_parquet(D / "old_det.parquet"), pd.read_parquet(D / "new_det.parquet")
k = ["case", "DeviceId", "Detector"]
m = a.merge(b, on=k, suffixes=("_o", "_n"), how="outer", indicator=True)
res = {"rows_old": len(a), "rows_new": len(b), "both": int((m._merge == "both").sum())}
cols = [c for c in a.columns if c not in k]
diff = {}
for c in cols:
    x, y = m[c + "_o"], m[c + "_n"]
    if x.dtype.kind in "fc" or y.dtype.kind in "fc":
        xx, yy = pd.to_numeric(x, errors="coerce").to_numpy(float), pd.to_numeric(y, errors="coerce").to_numpy(float)
        d = ~((xx == yy) | (np.isnan(xx) & np.isnan(yy)))
    else:
        d = ~((x.astype(str) == y.astype(str)) | (x.isna() & y.isna()))
    diff[c] = int(d.sum())
res["differing_cells_non_health"] = {c: v for c, v in diff.items() if c not in HEALTH | DEP and v}
res["non_health_columns_checked"] = len([c for c in cols if c not in HEALTH | DEP])
res["health_and_dependent"] = {c: v for c, v in diff.items() if c in HEALTH | DEP}
res["new_columns"] = [c for c in b.columns if c not in a.columns]
for nm in ("cand", "ph"):
    x, y = pd.read_parquet(D / f"old_{nm}.parquet"), pd.read_parquet(D / f"new_{nm}.parquet")
    same = x.shape == y.shape and all(((x[c] == y[c]) | (x[c].isna() & y[c].isna())).all() for c in x.columns)
    res[f"{nm}_identical"] = bool(same)
    res[f"{nm}_rows"] = len(x)
st = m[["health_status_o", "health_status_n"]].fillna("NA")
res["status_crosstab"] = pd.crosstab(st.health_status_o, st.health_status_n).to_dict()
res["flagged_old"] = int(st.health_status_o.isin(["suspect", "bad"]).sum())
res["flagged_new"] = int(st.health_status_n.isin(["suspect", "bad"]).sum())
res["n_det"] = len(st)
to = json.load(open(D / "old_time.json")); tn = json.load(open(D / "new_time.json"))
res["time_sum_old_new"] = [round(sum(to.values()), 1), round(sum(tn.values()), 1)]
(D / "cmp_par124.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
