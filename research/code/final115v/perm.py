"""Random channel-bijection + phase-bijection test on new signals (v7 wheel)."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import sys, json, re, warnings, os
warnings.simplefilter('ignore')
from pathlib import Path
import numpy as np, pandas as pd, duckdb
PKG = os.environ.get("PKG", "installed")
if PKG == "installed":
    from detector_classifier import predict
else:
    sys.path.insert(0, PKG); from predict import predict
W = Path(DCW); V = W / "s115v"; T = V / "perm"; T.mkdir(exist_ok=True)
sig = pd.read_csv(V / "signals_v.csv")
WINS = {"m5": ("2026-09-20 12:00:00", "2026-09-20 12:05:00"), "m30": ("2026-09-20 16:00:00", "2026-09-20 16:30:00"),
        "h3": ("2026-09-20 09:00:00", "2026-09-20 12:00:00"), "h24": (None, None)}
PH_EV = (1, 7, 8, 9, 10, 11, 43, 44)
con = duckdb.connect(); con.execute("set threads=4; set memory_limit='4GB'")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
ONLY = os.environ.get('ONLY')
cases = sys.argv[2].split(",") if len(sys.argv) > 2 else ["v01", "v05", "v08", "v12", "v13", "v14"]
res = []
for cs in cases:
    rng = np.random.default_rng([SEED, int(cs[1:])])
    f = (V / "ev" / f"{cs}.parquet").as_posix()
    chans = [r[0] for r in con.sql(f"select distinct Parameter from '{f}' where EventId in (81,82) and Parameter<=64").fetchall()]
    cm = dict(zip(range(1, 65), (rng.permutation(64) + 1).tolist()))
    pm = dict(zip(range(1, 17), (rng.permutation(16) + 1).tolist()))
    con.execute("create or replace temp table cm as select * from (values " + ",".join(f"({a},{b})" for a, b in cm.items()) + ") t(a,b)")
    con.execute("create or replace temp table pm as select * from (values " + ",".join(f"({a},{b})" for a, b in pm.items()) + ") t(a,b)")
    g = T / f"{cs}_perm.parquet"
    con.execute(f"""COPY (SELECT e.DeviceId, e.Timestamp, e.EventId,
        CASE WHEN e.EventId IN (81,82) AND e.Parameter<=64 THEN cm.b
             WHEN e.EventId IN {PH_EV} AND e.Parameter BETWEEN 1 AND 16 THEN pm.b ELSE e.Parameter END AS Parameter
        FROM '{f}' e LEFT JOIN cm ON cm.a=e.Parameter LEFT JOIN pm ON pm.a=e.Parameter
        ORDER BY random()) TO '{g.as_posix()}' (FORMAT parquet)""")
    ic, ip = {b: a for a, b in cm.items()}, {b: a for a, b in pm.items()}
    for L, (s, e) in WINS.items():
        if ONLY and L not in ONLY.split(','): continue
        for prof in (["full", "le2h"] if L in ("h3", "h24") else ["full"]):
            a, pa = predict(f, start=s, end=e, min_actuations=1, return_phases=True, profile=prof)
            b, pb = predict(str(g), start=s, end=e, min_actuations=1, return_phases=True, profile=prof)
            b = b.copy()
            b["Detector"] = b.Detector.map(ic)
            for c in ("phase_pred", "phase_2nd", "phase_guess"):
                b[c] = b[c].map(lambda x: pd.NA if pd.isna(x) else ip[int(x)]).astype("Int64")
            b = b.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
            a = a.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
            diffs = {}
            assert (a.Detector.to_numpy() == b.Detector.to_numpy()).all()
            for c in a.columns:
                x, y = a[c], b[c]
                if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
                    d = np.nanmax(np.abs(x.to_numpy(float) - y.to_numpy(float))) if x.notna().any() else 0.0
                    nm = int((x.isna() != y.isna()).sum())
                    if (d > 1e-6) or nm:
                        diffs[c] = f"maxdiff {d:.3g} nanmis {nm}"
                else:
                    xs = x.map(lambda v: "<NA>" if v is None or v is pd.NA or (isinstance(v, float) and v != v) else str(v))
                    ys = y.map(lambda v: "<NA>" if v is None or v is pd.NA or (isinstance(v, float) and v != v) else str(v))
                    bad = (xs != ys)
                    if bad.any():
                        diffs[c] = f"{int(bad.sum())} rows e.g. {xs[bad].iloc[0][:80]!r} vs {ys[bad].iloc[0][:80]!r} (det {int(a.Detector[bad].iloc[0])})"
            # phase table
            pb = pb.copy(); pb["phase"] = pb.phase.map(lambda x: ip[int(x)])
            pa2 = pa.sort_values(["DeviceId", "phase"]).reset_index(drop=True); pb = pb.sort_values(["DeviceId", "phase"]).reset_index(drop=True)
            if len(pa2) != len(pb) or not (pa2.astype(object).fillna("<NA>").astype(str).to_numpy() == pb.astype(object).fillna("<NA>").astype(str).to_numpy()).all():
                diffs["phase_table"] = "differs"
            r = dict(case=cs, L=L, prof=prof, n_det=len(a), diffs=diffs)
            tag = os.environ.get("TAG", "v7")
            if diffs:
                a.to_pickle(T / f"{tag}_{cs}_{L}_{prof}_a.pkl"); b.to_pickle(T / f"{tag}_{cs}_{L}_{prof}_b.pkl")
            print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(T / f"perm_{os.environ.get('TAG','v7')}_{SEED}_{'_'.join(cases)}.json", "w"), indent=1)
