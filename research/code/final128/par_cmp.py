"""note-115 set (132 cases): every column of prod vs next identical (detector, candidate and phase tables)"""
import pandas as pd, numpy as np, json
for t in ("det", "cand", "ph"):
    A = pd.read_parquet(f"par/prod_{t}.parquet"); B = pd.read_parquet(f"par/next2_{t}.parquet")
    same = A.equals(B)
    bad = [c for c in A.columns if not A[c].equals(B[c])] if not same else []
    print(t, A.shape, B.shape, "identical" if same else f"DIFF {bad}")
A = pd.read_parquet("par/prod_det.parquet")
print("cases", A.case.nunique(), "detectors", len(A), "health statuses", A.health_status.value_counts().to_dict())
ta, tb = json.load(open("par/prod_time.json")), json.load(open("par/next2_time.json"))
print("sum time prod %.1fs next %.1fs" % (sum(ta.values()), sum(tb.values())))
