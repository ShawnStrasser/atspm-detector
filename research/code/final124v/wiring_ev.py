"""124v: why do predict's health events differ from cmp124.prep?  compare as sorted multisets."""
import importlib.util, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, duckdb
warnings.simplefilter("ignore")
W = Path.home() / "dc_work"; O = W / "s124v"
os.environ["DC_PKG"] = str(W / "final_v7_prod" / "src")
spec = importlib.util.spec_from_file_location("cmp124", Path(__file__).resolve().parents[1] / "final124" / "cmp124.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
from detector_classifier import pipeline as PL
cap = []
orig = PL.hv4.assess
PL.hv4.assess = lambda *a, **k: (cap.append(a), orig(*a, **k))[1]
dev = "e21648b5-f2cf-4c1e-aac8-3bd829c143cc"
e = C.load_events(dev)
f = O / "tmp_w40.parquet"
src = next(p for p in C.EVD.iterdir() if p.name.lower() == f"deviceid={dev}")
duckdb.sql(f"COPY (select '{dev}' as DeviceId, Timestamp, EventId, Parameter from read_parquet('{src.as_posix()}/*.parquet')) TO '{f.as_posix()}' (FORMAT parquet)")
for w in ("m30_a", "h24_a"):
    s, h = C.WIN[w]; t0 = pd.Timestamp(s); t1 = t0 + pd.Timedelta(hours=h)
    cap.clear(); PL.predict(str(f), start=str(t0), end=str(t1), min_actuations=1)
    P = cap[0][0]; Q = C.prep(e, t0, t1)
    a = np.lexsort((P.par, P.eid, P.t)); b = np.lexsort((Q.par, Q.eid, Q.t))
    print(w, "n", len(P.t), len(Q.t))
    for x in ("td", "ed", "pd_", "g0", "g1", "ut"):
        print("  ", x, np.array_equal(getattr(P, x), getattr(Q, x)), np.abs(getattr(P, x) - getattr(Q, x)).max() if getattr(P, x).shape == getattr(Q, x).shape and getattr(P, x).size else "")
    print("   sorted t/eid/par equal:", np.array_equal(P.t[a], Q.t[b]), np.array_equal(P.eid[a], Q.eid[b]), np.array_equal(P.par[a], Q.par[b]),
          "max |dt|", np.abs(P.t[a] - Q.t[b]).max() if len(a) == len(b) else "len differ")
