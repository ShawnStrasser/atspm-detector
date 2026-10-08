"""h24_from_06 (start 10 h before the data): new package == old package with the start left out"""
import os
import sys, warnings, json, pandas as pd, numpy as np, pyarrow.dataset as ds
warnings.simplefilter("ignore")
pkg = sys.argv[1]; sys.path.insert(0, pkg)
from detector_classifier import pipeline as P
W = os.environ.get("DC_WORK", os.path.expanduser("~/dc_work"))
dev = next(r["DeviceId"] for r in json.loads(open(W + "/s118c/plot/rows.json").read()) if r["row"] == 2)
w = ds.dataset(W + "/health4/w40_events/DeviceId=" + dev).to_table().to_pandas().assign(DeviceId=dev)
s = None if sys.argv[2] == "none" else "2026-09-26 06:00"
r = P.predict(w, start=s, end="2026-09-27 06:00", min_actuations=1)
r.to_pickle(f"edge_clip_{sys.argv[3]}.pkl")
