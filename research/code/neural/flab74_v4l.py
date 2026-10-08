"""Note 74: function-net training labels on the CURRENT label set (v4l, note 81) -> %DC_WORK%/tcn53/func_rows_v4l.parquet.

Same recipe as the 2026-10-01 builder of tcn53/func_rows.parquet (t57_function.setup("v6e") -> v3_retrain.load_labels
"exclude" -> variant_target), which ran on the v3s table; v3_retrain.LABEL_SETS["v3s"] now carries v4l. func_rows.parquet
(the labels every fold run of notes 53-74 trained on) is left unchanged; tcn69_func.py --frows picks the v4l file.
2026-10-03: 456,033 rows, ok .6550 (v3s .6519); 272 ok rows relabelled, 3,314 newly ok, 1,881 no longer ok (329 detectors).

    python flab74_v4l.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import pandas as pd  # noqa: E402
import t57_function as T57  # noqa: E402
import v3_retrain as V  # noqa: E402

T57.OUT = V.DCW / "s74" / "t57_v4l_dummy"      # setup asserts a non-default OUT for a non-BASE_RUN label table
T57.setup("v6e")
print("label table:", V.LABELS_V3)
fr, cols = V.load_feats()
lab = V.load_labels(fr, "exclude")
y, ok = V.variant_target(lab, fr, T57.VAR)
k = fr[V.KEY + ["fold", "wgroup"]].copy()
k["y"] = y
k["ok"] = ok
old = pd.read_parquet(V.DCW / "tcn53" / "func_rows.parquet")
assert (old[V.KEY].to_numpy() == k[V.KEY].to_numpy()).all()
print(f"rows {len(k):,}; ok {k.ok.mean():.4f} (func_rows.parquet {old.ok.mean():.4f})")
k.to_parquet(V.DCW / "tcn53" / "func_rows_v4l.parquet", index=False)
