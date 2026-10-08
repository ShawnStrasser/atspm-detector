"""Verifier parity runner: python runv.py <name> <flat-dir | installed> <profile>"""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import json, os, sys, time
from pathlib import Path
import pandas as pd
W = Path(DCW); V = W / "s115v"; O = W / "s115d" / "v" / "out"
sig = pd.read_csv(V / "signals_v.csv")
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
assert not set(sig.DeviceId.str.lower()) & lk
CASES = []
for c in sig.case:
    f = str(V / "ev" / f"{c}.parquet")
    CASES += [(f"{c}_m5", f, "2026-09-20 12:00:00", "2026-09-20 12:05:00"),
              (f"{c}_m30", f, "2026-09-20 16:00:00", "2026-09-20 16:30:00"),
              (f"{c}_h3", f, "2026-09-20 09:00:00", "2026-09-20 12:00:00"),
              (f"{c}_h24", f, None, None)]
_, name, pkg, profile = sys.argv[:4]
if pkg == "installed":
    from detector_classifier import pipeline as P
elif pkg.startswith("src:"):
    sys.path.insert(0, pkg[4:]); from detector_classifier import pipeline as P
else:
    sys.path.insert(0, pkg); import predict as P
print("module", P.__file__, flush=True)
cap = {}; orig = P.score
def score(*a, **k):
    r = orig(*a, **k); cap["cand"] = r[["DeviceId", "Detector", "cand_phase", "p0", "prob"]].copy(); return r
P.score = score
dets, cands, phs, tim = [], [], [], {}
for case, f, s, e in CASES:
    cap.clear(); t0 = time.perf_counter()
    out, ph = P.predict(f, start=s, end=e, min_actuations=1, return_phases=True, profile=profile)
    tim[case] = time.perf_counter() - t0
    dets.append(out.assign(case=case)); phs.append(ph.assign(case=case))
    if "cand" in cap: cands.append(cap["cand"].assign(case=case))
    print(f"{case} {tim[case]:.2f}s {len(out)} det", flush=True)
tag = f"{name}_{profile}"
import duckdb
con = duckdb.connect()
for k, L in (("det", dets), ("cand", cands), ("ph", phs)):
    df = pd.concat(L, ignore_index=True)
    for c in df.columns:
        if str(df[c].dtype) in ("Int64",):
            df[c] = df[c].astype("float64")
        elif df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].map(lambda x: None if x is None or (isinstance(x, float) and x != x) or x is pd.NA else str(x))
    con.register("x", df)
    con.execute(f"COPY x TO '{(O/f'{tag}_{k}.parquet').as_posix()}' (FORMAT parquet)")
    con.unregister("x")
json.dump(tim, open(O/f"{tag}_time.json", "w"), indent=1)
