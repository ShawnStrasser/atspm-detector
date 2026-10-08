import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import re, sys, warnings, os
warnings.simplefilter("ignore")
P = os.environ["PKGP"]; sys.path.insert(0, P)
if "ref114" in P: from predict import predict
else: from detector_classifier import predict
import duckdb, pandas as pd
ratio = lambda s: max([float(x) for x in re.findall(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)x\b", str(s).replace(",", ""))] or [0])
for case, (s, e), ph in (("n07", ("2026-09-20 15:00", "2026-09-20 18:00"), 6), ("n09", ("2026-09-20 06:00", "2026-09-21 06:00"), 2)):
    A = duckdb.sql(f"select * from '{DCW}/s115dv/ev/{case}.parquet' where Timestamp>='{s}' and Timestamp<'{e}'").df()
    r = predict(A, min_actuations=1); g = r[r.phase_pred == ph]
    busy = int(g.sort_values("n_actuations").Detector.iloc[-1]); others = [int(d) for d in g.Detector if int(d) != busy]
    print(case, "phase", ph, "busy det", busy, "partners", others, dict(zip(g.Detector, g.function_pred)))
    for k in (0, 1, 3, 6, 12, 25, 50, 100):
        d = A.copy()
        keep_on = d[(d.EventId == 82) & d.Parameter.isin(others)].groupby("Parameter").head(k).index
        drop = d.EventId.isin([81, 82]) & d.Parameter.isin(others)
        d = pd.concat([d[~drop], d.loc[keep_on], d.loc[keep_on].assign(EventId=81, Timestamp=lambda x: x.Timestamp + pd.Timedelta(seconds=1))])
        o = predict(d, min_actuations=1).set_index("Detector")
        b = o.loc[busy]
        print(f"  k={k:3d}: busy det {busy}: {b.health_status} {str(b.health_reason)[:70]!r} | partners {o.reindex(others)['health_status'].tolist()}")
