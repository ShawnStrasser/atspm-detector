"""F6 adversarial: make every same-phase partner of one busy detector nearly silent (k ONs kept) and scan health ratios."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import re, sys, warnings, os
warnings.simplefilter("ignore")
sys.path.insert(0, os.environ["SRC"])
import duckdb, pandas as pd
from detector_classifier import predict
ratio = lambda s: max([float(x) for x in re.findall(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)x\b", str(s).replace(",", ""))] or [0])
worst = (0, None)
for case, (s, e) in (("n05", ("2026-09-20 15:00", "2026-09-20 18:00")), ("n07", ("2026-09-20 15:00", "2026-09-20 18:00")),
                     ("n09", ("2026-09-20 06:00", "2026-09-21 06:00"))):
    A = duckdb.sql(f"select * from '{DCW}/s115dv/ev/{case}.parquet' where Timestamp>='{s}' and Timestamp<'{e}'").df()
    r = predict(A, min_actuations=1)
    for ph, g in r.dropna(subset=["phase_pred"]).groupby("phase_pred"):
        if len(g) < 2: continue
        busy = int(g.sort_values("n_actuations").Detector.iloc[-1]); others = [int(d) for d in g.Detector if int(d) != busy]
        for k in (0, 1, 2, 3, 6, 12):
            d = A.copy()
            on = d[(d.EventId == 82) & d.Parameter.isin(others)]
            keep_on = on.groupby("Parameter").head(k).index
            drop = d.EventId.isin([81, 82]) & d.Parameter.isin(others)
            d = pd.concat([d[~drop], d.loc[keep_on], d.loc[keep_on].assign(EventId=81, Timestamp=lambda x: x.Timestamp + pd.Timedelta(seconds=1))])
            o = predict(d, min_actuations=1)
            rr = o.health_reason.map(ratio).fillna(0) + 0 * o.Detector
            mx = float(rr.max())
            if mx > worst[0]: worst = (mx, (case, ph, busy, k, o.loc[rr.idxmax(), ["Detector", "health_status", "health_reason"]].to_dict()))
    print(case, "worst so far", worst, flush=True)
print("FINAL worst ratio", worst)
