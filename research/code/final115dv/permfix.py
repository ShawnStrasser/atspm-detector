"""115dx: channel + phase bijection on the cases whose health changed with the 115dx fix.   PKGP=<src dir> python permfix.py"""
import os, sys, warnings, re
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, os.environ["PKGP"])
from detector_classifier import predict
import numpy as np, pandas as pd, duckdb
W = (os.environ.get("DC_WORK") or str(Path.home() / "dc_work")).replace("\\", "/")
lk = set(pd.read_csv(f"{W}/official/locked_v2.csv").DeviceId.str.lower())
cases = [("s25", f"{W}/s115/parity/ev/s25.parquet", None, None), ("s32", f"{W}/s115/parity/ev/s32.parquet", None, None),
         ("r8_24", f"{W}/bench71/ev_typical_r8_h24.parquet", None, None), ("r8_3", f"{W}/bench71/ev_typical_r8_h3.parquet", None, None)]
PH = (1, 7, 8, 9, 10, 11, 43, 44)
bad = 0
for name, f, s, e in cases:
    A = duckdb.sql(f"select * from '{f}'").df()
    assert not set(A.DeviceId.astype(str).str.lower()) & lk
    base = predict(A)
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        cm = dict(zip(range(1, 65), (rng.permutation(64) + 1).tolist())); pm = dict(zip(range(1, 17), (rng.permutation(16) + 1).tolist()))
        B = A.copy()
        det = B.EventId.isin([81, 82]) & B.Parameter.between(1, 64); ph = B.EventId.isin(PH) & B.Parameter.between(1, 16)
        B.loc[det, "Parameter"] = B.loc[det, "Parameter"].map(cm); B.loc[ph, "Parameter"] = B.loc[ph, "Parameter"].map(pm)
        o = predict(B)
        o["Detector"] = o.Detector.map({v: k for k, v in cm.items()})
        for c in ("phase_pred", "phase_2nd"):
            o[c] = o[c].map(lambda v: {b: a for a, b in pm.items()}.get(v, v) if pd.notna(v) else v)
        m = base.merge(o, on=["DeviceId", "Detector"], suffixes=("_a", "_b"))
        diffs = []
        for c in ("function_pred", "phase_pred", "health_status", "health_score", "health_reason", "health_bad_periods", "lanes", "distance_ft"):
            x, y = (m[c + "_a"].astype(float).fillna(-1).astype(str), m[c + "_b"].astype(float).fillna(-1).astype(str)) if c in ("phase_pred", "distance_ft", "health_score") else (m[c + "_a"].astype(str), m[c + "_b"].astype(str))
            if c == "health_reason":   # partner channel names follow the renumbering
                y = m[c + "_b"].astype(str).map(lambda t: re.sub(r"d(\d+)", lambda z: f"d{ {v: k for k, v in cm.items()}.get(int(z.group(1)), int(z.group(1)))}", t))
                x = x.map(lambda t: re.sub(r"\s+", " ", t)); y = y.map(lambda t: re.sub(r"\s+", " ", t))
            n = int((x != y).sum())
            if n: diffs.append((c, n))
        bad += bool(diffs)
        print(name, "seed", seed, len(m), "detectors", "diffs:", diffs or "none", flush=True)
print("PERM", "PASS" if not bad else "FAIL")
