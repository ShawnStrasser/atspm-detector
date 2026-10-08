"""Note 89: function-net training labels on v4n (v4m + the eight restored exclusion groups) -> %DC_WORK%/tcn53/func_rows_v4n.parquet.
Same recipe as neural/flab74_v4l.py (t57 setup -> v3_retrain.load_labels "exclude" -> variant_target); the siba nets of
notes 74 / 84 / 86 trained on func_rows_v4l (ok 74.8 % of rows here vs 65.5 %). For the final GPU refits.

    python flab89_v4n.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import pandas as pd  # noqa: E402
import t57_function as T57  # noqa: E402
import v3_retrain as V  # noqa: E402

V.LABEL_SETS["v3s"] = V.LABEL_SETS["v4n"]
T57.OUT = V.DCW / "s74" / "t57_v4n_dummy"
T57.setup("v6e")
print("label table:", V.LABELS_V3)
fr, cols = V.load_feats()
lab = V.load_labels(fr, "exclude")
y, ok = V.variant_target(lab, fr, T57.VAR)
k = fr[V.KEY + ["fold", "wgroup"]].copy()
k["y"] = y
k["ok"] = ok
old = pd.read_parquet(V.DCW / "tcn53" / "func_rows_v4l.parquet")
assert (old[V.KEY].to_numpy() == k[V.KEY].to_numpy()).all()
print(f"rows {len(k):,}; ok {k.ok.mean():.4f} (func_rows_v4l {old.ok.mean():.4f}); newly ok {int((k.ok & ~old.ok).sum()):,}, "
      f"no longer ok {int((~k.ok & old.ok).sum()):,}")
k.to_parquet(V.DCW / "tcn53" / "func_rows_v4n.parquet", index=False)
