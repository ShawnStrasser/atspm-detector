import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import warnings, sys
from detector_classifier import predict
D = f"{DCW}/s115dv/ev/"
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for c in ("n01", "n06", "n08"):
        for p in ("full", "le2h"):
            predict(D + c + ".parquet", profile=p); predict(D + c + ".parquet", start="2026-09-20 07:30", end="2026-09-20 08:00", profile=p)
print("warnings:", len(w), sorted({f"{x.category.__name__}: {str(x.message)[:100]}" for x in w}))
