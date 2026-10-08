import os, sys, time, warnings
for k in ("OMP_NUM_THREADS", "DC_TREE_THREADS", "DC_NET_THREADS"): os.environ[k] = "4"
import pandas as pd
src = sys.argv[1]; sys.path.insert(0, src); warnings.simplefilter("ignore")
from detector_classifier import predict
E = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"s128v\ev")
predict(E + r"\x00_m30.parquet")
for case, win, s, e in [("x05", "m30", "2025-09-20 16:30:00", "2026-09-20 17:00:00"), ("x09", "h3", "2025-09-20 13:00:00", "2026-09-20 16:00:00"),
                        ("x00", "m30", "2000-01-01 00:00:00", "2030-01-01 00:00:00")]:
    t = time.perf_counter(); o = predict(fr"{E}\{case}_{win}.parquet", start=s, end=e)
    print(src, case, win, s, e, f"{time.perf_counter()-t:.2f}s", len(o), flush=True)
