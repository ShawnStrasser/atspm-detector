"""124v: random channel (1-64) + phase (1-16) bijection on new signals; every output column, health text translated back.
    python perm.py <seed> v03,v06,v09   (PKG=<src dir> or installed)"""
import json, os, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, duckdb
warnings.simplefilter("ignore")
if os.environ.get("PKG"):
    sys.path.insert(0, os.environ["PKG"])
from detector_classifier import predict
W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work")); V = W / "s124v"; T = V / "perm"; T.mkdir(exist_ok=True)
WINS = {"m30": ("2026-09-20 17:00:00", "2026-09-20 17:30:00"), "h3": ("2026-09-20 10:00:00", "2026-09-20 13:00:00"),
        "h24": ("2026-09-20 06:00:00", "2026-09-21 06:00:00")}
PH_EV = (1, 7, 8, 9, 10, 11, 43, 44)
con = duckdb.connect(); con.execute("set threads=4; set memory_limit='4GB'")
SEED = int(sys.argv[1]); cases = sys.argv[2].split(",")
TXT = ["health_reason", "health_watch", "health_categories", "health_config", "health_signal_note", "status",
       "review_reason", "lanes"]


def tr(s, ic, ip):
    if not isinstance(s, str):
        return s
    s = re.sub(r"\bdet (\d+)", lambda m: f"det {ic[int(m.group(1))]}", s)
    s = re.sub(r"\bdets? ((?:\d+, )*\d+)", lambda m: m.group(0).split(" ")[0] + " " + ", ".join(str(ic[int(x)]) for x in m.group(1).split(", ")), s) if "dets " in s else s
    s = re.sub(r"\bd(\d+)\b", lambda m: f"d{ic[int(m.group(1))]}", s)
    s = re.sub(r"\bP(\d+)\b", lambda m: f"P{ip[int(m.group(1))]}", s)
    s = re.sub(r"\bphase (\d+)\b", lambda m: f"phase {ip[int(m.group(1))]}", s)
    return s


res = []
for cs in cases:
    rng = np.random.default_rng([SEED, int(cs[1:])])
    f = (V / "ev" / f"{cs}.parquet").as_posix()
    cm = dict(zip(range(1, 65), (rng.permutation(64) + 1).tolist()))
    pm = dict(zip(range(1, 17), (rng.permutation(16) + 1).tolist()))
    con.execute("create or replace temp table cm as select * from (values " + ",".join(f"({a},{b})" for a, b in cm.items()) + ") t(a,b)")
    con.execute("create or replace temp table pm as select * from (values " + ",".join(f"({a},{b})" for a, b in pm.items()) + ") t(a,b)")
    g = T / f"{cs}_s{SEED}.parquet"
    con.execute(f"""COPY (SELECT e.DeviceId, e.Timestamp, e.EventId,
        CASE WHEN e.EventId IN (81,82) AND e.Parameter<=64 THEN cm.b
             WHEN e.EventId IN {PH_EV} AND e.Parameter BETWEEN 1 AND 16 THEN pm.b ELSE e.Parameter END AS Parameter
        FROM '{f}' e LEFT JOIN cm ON cm.a=e.Parameter LEFT JOIN pm ON pm.a=e.Parameter
        ORDER BY random()) TO '{g.as_posix()}' (FORMAT parquet)""")
    ic, ip = {b: a for a, b in cm.items()}, {b: a for a, b in pm.items()}
    for L, (s, e) in WINS.items():
        a, pa = predict(f, start=s, end=e, min_actuations=1, return_phases=True)
        b, pb = predict(str(g), start=s, end=e, min_actuations=1, return_phases=True)
        b = b.copy()
        b["Detector"] = b.Detector.map(ic)
        for c in ("phase_pred", "phase_2nd", "phase_guess"):
            b[c] = b[c].map(lambda x: pd.NA if pd.isna(x) else ip[int(x)]).astype("Int64")
        for c in [t for t in TXT if t in b.columns]:
            b[c] = b[c].map(lambda v: tr(v, ic, ip))
        b = b.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
        a = a.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
        assert (a.Detector.to_numpy() == b.Detector.to_numpy()).all()
        diffs, maxd = {}, 0.0
        for c in a.columns:
            x, y = a[c], b[c]
            if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
                xv, yv = x.to_numpy(float), y.to_numpy(float)
                d = float(np.nanmax(np.abs(xv - yv))) if np.isfinite(xv).any() and np.isfinite(yv).any() else 0.0
                nm = int((x.isna() != y.isna()).sum())
                maxd = max(maxd, d) if np.isfinite(d) else maxd
                if d > 1e-6 or nm:
                    diffs[c] = f"maxdiff {d:.3g} nanmis {nm}"
            else:
                fx = lambda v: "<NA>" if v is None or v is pd.NA or (isinstance(v, float) and v != v) else str(v)
                xs, ys = x.map(fx), y.map(fx)
                bad = xs != ys
                if bad.any():
                    diffs[c] = f"{int(bad.sum())} rows e.g. {xs[bad].iloc[0][:120]!r} vs {ys[bad].iloc[0][:120]!r} (det {int(a.Detector[bad].iloc[0])})"
        pb = pb.copy(); pb["phase"] = pb.phase.map(lambda x: ip[int(x)])
        pa2 = pa.sort_values(["DeviceId", "phase"]).reset_index(drop=True); pb = pb.sort_values(["DeviceId", "phase"]).reset_index(drop=True)
        if len(pa2) != len(pb) or not (pa2.astype(object).fillna("<NA>").astype(str).to_numpy() == pb.astype(object).fillna("<NA>").astype(str).to_numpy()).all():
            diffs["phase_table"] = "differs"
        r = dict(case=cs, L=L, n_det=len(a), n_flag=int(a.health_status.isin(["watch", "suspect", "bad"]).sum()),
                 max_float_diff=maxd, diffs=diffs)
        if diffs:
            a.to_pickle(T / f"{cs}_{L}_s{SEED}_a.pkl"); b.to_pickle(T / f"{cs}_{L}_s{SEED}_b.pkl")
        print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(T / f"perm_{SEED}_{'_'.join(cases)}.json", "w"), indent=1)
