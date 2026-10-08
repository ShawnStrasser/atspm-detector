"""Note 134: parity set (132 cases) final_v7_prod_bak134 (old) vs final_v7_prod (new, 4-h persistence rule):
every non-health column identical; health changes counted; bad / suspect per 100 detectors by sample length."""
import os
from pathlib import Path
import pandas as pd
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
D = W / "s134" / "par"
HEALTH = {"health_status", "health_score", "health_reason", "health_bad_periods", "health_watch", "health_categories",
          "health_config", "health_signal_note", "status", "review_flag", "review_reason"}
for t in ("cand", "ph"):
    A, B = pd.read_parquet(D / f"old_{t}.parquet"), pd.read_parquet(D / f"new_{t}.parquet")
    print(t, A.shape, B.shape, "identical" if A.equals(B) else "DIFF " + str([c for c in A if not A[c].equals(B[c])]))
A, B = pd.read_parquet(D / "old_det.parquet"), pd.read_parquet(D / "new_det.parquet")
assert list(A.columns) == list(B.columns) and len(A) == len(B)
non = [c for c in A.columns if c not in HEALTH]
diff = [c for c in non if not A[c].equals(B[c])]
print("cases", A.case.nunique(), "detectors", len(A), "non-health columns", len(non), "differing:", diff or "none")
key = ["phase_pred", "function_pred", "lanes", "distance_ft", "night_speed_mph"]
print("key columns identical:", all(A[c].equals(B[c]) for c in key))
for c in sorted(HEALTH & set(A.columns)):
    a, b = A[c].astype(str), B[c].astype(str)
    print(f"  {c}: {int((a != b).sum())} rows differ")
A["L"] = A.case.str.extract(r"_(m30|h3|h24)$")[0]
B["L"] = A.L
ch = A.health_status != B.health_status
print("status moves:", pd.crosstab(A.health_status[ch], B.health_status[ch]).to_dict())
rows = []
for L in ("m30", "h3", "h24"):
    a, b = A[A.L == L], B[B.L == L]
    rows.append(dict(length=L, detectors=len(a), bad_old=100 * a.health_status.eq("bad").mean(),
                     bad_new=100 * b.health_status.eq("bad").mean(),
                     suspect_old=100 * a.health_status.eq("suspect").mean(),
                     suspect_new=100 * b.health_status.eq("suspect").mean(),
                     flagged_old=100 * a.health_status.isin(["bad", "suspect"]).mean(),
                     flagged_new=100 * b.health_status.isin(["bad", "suspect"]).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
m = ch
print(B.loc[m, ["case", "Detector", "health_status", "health_categories"]].assign(old=A.health_status[m]).to_string(index=False))
