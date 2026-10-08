"""df vs path: the window rule.  python robust_d.py (SRC=<src>)"""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.environ["SRC"]) if os.environ.get("SRC") else None
import duckdb, pandas as pd
from detector_classifier import predict
F = f"{DCW}/s115v/ev/v08.parquet"; S, E = "2026-09-20 09:00:00", "2026-09-20 12:00:00"
con = duckdb.connect(); con.execute("set threads=4")
day = con.sql(f"select * from '{F}'").df()
cut = day[(day.Timestamp >= S) & (day.Timestamp < E)]
p = predict(F, start=S, end=E, min_actuations=1).sort_values("Detector").reset_index(drop=True)
a = predict(day, start=S, end=E, min_actuations=1).sort_values("Detector").reset_index(drop=True)
b = predict(cut, start=S, end=E, min_actuations=1).sort_values("Detector").reset_index(drop=True)
c = predict(cut, min_actuations=1).sort_values("Detector").reset_index(drop=True)
def diff(x, y):
    return [k for k in x.columns if not (x[k].astype(str) == y[k].astype(str)).all()]
print("path vs whole-day df + start/end:", diff(p, a) or "identical")
print("path vs cut df + start/end:", diff(p, b) or "identical")
print("path vs cut df, no start/end:", diff(p, c), "first event", cut.Timestamp.min())
print("minutes_of_data", p.minutes_of_data.iloc[0], c.minutes_of_data.iloc[0])
