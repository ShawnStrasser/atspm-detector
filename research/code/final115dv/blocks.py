"""F1 block path: force tiny blocks (every detector its own block) and compare with the one-block path; count blocks on the 7-day file."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.environ["SRC"])
import pandas as pd
from detector_classifier import predict, features as FT
D = f"{DCW}/s115dv"
nb = []; orig = FT._det_blocks
def counting(con):
    b = orig(con); nb.append(1 if b == [None] or b is None or len(b) == 0 else len(b)); return b
FT._det_blocks = counting
def cmp(a, b):
    k = ["DeviceId", "Detector"]; a = a.sort_values(k).reset_index(drop=True); b = b.sort_values(k).reset_index(drop=True)
    out = []
    for c in a.columns:
        if pd.api.types.is_float_dtype(a[c]) and pd.api.types.is_float_dtype(b[c]):
            d = (a[c] - b[c]).abs().max()
            if (d > 1e-9) or (a[c].isna() != b[c].isna()).any(): out.append(f"{c} {d:.2g}")
        elif not (a[c].astype(str) == b[c].astype(str)).all(): out.append(c)
    return out or "identical"
nb.clear(); predict(f"{D}/rob/week_n08.parquet", min_actuations=1, profile="le2h"); print("7-day n08 blocks:", nb, flush=True)
for case in ["n00", "n03", "n05", "n07", "n09"]:
    for s, e in ((None, None), ("2026-09-20 15:00:00", "2026-09-20 18:00:00")):
        FT.J_BLOCK_ROWS = 1_500_000; nb.clear()
        a, pa = predict(f"{D}/ev/{case}.parquet", start=s, end=e, min_actuations=1, return_phases=True); n1 = list(nb)
        FT.J_BLOCK_ROWS = 1; nb.clear()
        b, pb = predict(f"{D}/ev/{case}.parquet", start=s, end=e, min_actuations=1, return_phases=True); n2 = list(nb)
        print(case, "h24" if s is None else "h3", "blocks", n1, "->", n2, cmp(a, b), "phases", pa.astype(str).equals(pb.astype(str)), flush=True)
