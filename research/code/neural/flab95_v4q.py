"""Note 95: function-net training labels on v4q -> %DC_WORK%/tcn53/func_rows_v4q.parquet (all periods, flab89 recipe) and
func_rows_v4q_2026.parquet (Sept-2026 rows only: the 2026-only siba nets train AND infer on these keys).

    python flab95_v4q.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import pandas as pd  # noqa: E402
import t57_function as T57  # noqa: E402
import v3_retrain as V  # noqa: E402

V.LABEL_SETS["v3s"] = V.LABEL_SETS["v4q"]
T57.OUT = V.DCW / "s95" / "t57_v4q_dummy"
T57.setup("v6e")
print("label table:", V.LABELS_V3)
fr, cols = V.load_feats()
lab = V.load_labels(fr, "exclude")
y, ok = V.variant_target(lab, fr, T57.VAR)
k = fr[V.KEY + ["fold", "wgroup"]].copy()
k["y"] = y
k["ok"] = ok
old = pd.read_parquet(V.DCW / "tcn53" / "func_rows_v4o.parquet")
assert (old[V.KEY].to_numpy() == k[V.KEY].to_numpy()).all()
print(f"rows {len(k):,}; ok {k.ok.mean():.4f} (v4o {old.ok.mean():.4f}); newly ok {int((k.ok & ~old.ok).sum()):,}, "
      f"no longer ok {int((~k.ok & old.ok).sum()):,}")
k.to_parquet(V.DCW / "tcn53" / "func_rows_v4q.parquet", index=False)
s = k[k.period == "stg"].reset_index(drop=True)
print(f"2026 rows {len(s):,} ({s.DeviceId.nunique()} signals), ok {int(s.ok.sum()):,} "
      f"({s[s.ok].DeviceId.nunique()} signals, {s[s.ok][['DeviceId','Detector']].drop_duplicates().shape[0]} detectors)")
s.to_parquet(V.DCW / "tcn53" / "func_rows_v4q_2026.parquet", index=False)
