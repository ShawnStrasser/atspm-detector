"""Note 153: compare a predict() output parquet (new) with main's: python cmp153.py new.parquet main.parquet
Health columns within 1e-6 relative (text exact); every other column must be bit-equal."""
import sys, numpy as np, pandas as pd
a = pd.read_parquet(sys.argv[1]); b = pd.read_parquet(sys.argv[2])
key = ["item", "DeviceId", "Detector"] if ("item" in a.columns and "item" in b.columns) else ["DeviceId", "Detector"]
b = b.drop(columns=[c for c in ("item",) if c not in key and c in b.columns]); a = a.drop(columns=[c for c in ("item",) if c not in key and c in a.columns])
m = b.merge(a, on=key, how="outer", suffixes=("_main", "_new"), indicator=True)
print("rows main", len(b), "new", len(a), m._merge.value_counts().to_dict())
H = ["health_status", "health_categories", "health_reason", "health_bad_periods", "health_config", "health_watch",
     "health_signal_note", "health_score"]
for c in [c for c in b.columns if c not in key]:
    x, y = m[c + "_main"], m[c + "_new"]
    if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
        eq = np.isclose(x, y, rtol=1e-6, atol=0) | (x.isna() & y.isna())
        exact = (x == y) | (x.isna() & y.isna())
    else:
        eq = (x.astype(str) == y.astype(str)) | (x.isna() & y.isna())
        exact = eq
    if (~eq).sum() or (c not in H and (~exact).sum()):
        print(f"{c:22s} differ {int((~eq).sum())}  (not bit-equal {int((~exact).sum())})")
        if c in H:
            for _, r in m[~eq].head(5).iterrows():
                print("  ", r.DeviceId, r.Detector, "| main:", r[c + "_main"], "| new:", r[c + "_new"])
print("done")
