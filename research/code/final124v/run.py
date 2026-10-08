"""124v parity runner: python run.py <name> <src-dir | installed>  -> %DC_WORK%/s124v/out/<name>_{det,cand,ph}.parquet"""
import json, os, sys, time, warnings
from pathlib import Path
import pandas as pd
W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work")); V = W / "s124v"
sig = pd.read_csv(V / "signals_v.csv")
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
assert not set(sig.DeviceId.str.lower()) & lk
CASES = []
for c in sig.case:
    f = str(V / "ev" / f"{c}.parquet")
    CASES += [(f"{c}_m30", f, "2026-09-20 17:00:00", "2026-09-20 17:30:00"),
              (f"{c}_h3", f, "2026-09-20 10:00:00", "2026-09-20 13:00:00"),
              (f"{c}_h24", f, "2026-09-20 06:00:00", "2026-09-21 06:00:00")]
_, name, pkg = sys.argv[:3]
if pkg != "installed":
    sys.path.insert(0, pkg)
from detector_classifier import pipeline as P
print("module", P.__file__, flush=True)
cap = {}; orig = P.score
def score(*a, **k):
    r = orig(*a, **k); cap["cand"] = r[["DeviceId", "Detector", "cand_phase", "p0", "prob"]].copy(); return r
P.score = score
dets, cands, phs, tim, nwarn = [], [], [], {}, {}
for case, f, s, e in CASES:
    cap.clear(); t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        out, ph = P.predict(f, start=s, end=e, min_actuations=1, return_phases=True, threads=4)
    tim[case] = time.perf_counter() - t0; nwarn[case] = [str(w.message)[:200] for w in wl]
    dets.append(out.assign(case=case)); phs.append(ph.assign(case=case))
    if "cand" in cap: cands.append(cap["cand"].assign(case=case))
    print(f"{case} {tim[case]:.2f}s {len(out)} det warn={len(wl)}", flush=True)
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
    con.execute(f"COPY x TO '{(V/'out'/f'{name}_{k}.parquet').as_posix()}' (FORMAT parquet)")
    con.unregister("x")
json.dump({"time": tim, "warnings": nwarn}, open(V/"out"/f"{name}_time.json", "w"), indent=1)
