"""115dx: phase-mates that die mid-sample (head(k)) vs phase-mates that are just quiet (k spread evenly).
The busy detector must stay ok in both.   PKGP=<dir holding detector_classifier> python f6dead.py"""
import os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, os.environ["PKGP"])
from detector_classifier import predict
import duckdb, pandas as pd
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
for case, (s, e), ph, busy in (("n07", ("2026-09-20 15:00", "2026-09-20 18:00"), 6, 18),
                               ("n09", ("2026-09-20 06:00", "2026-09-21 06:00"), 2, 2)):
    A = duckdb.sql(f"select * from '{(W / 's115dv' / 'ev' / f'{case}.parquet').as_posix()}' "
                   f"where Timestamp>='{s}' and Timestamp<'{e}'").df()
    r = predict(A, min_actuations=1)
    others = [int(d) for d in r[r.phase_pred == ph].Detector if int(d) != busy]
    for mode in ("dead", "spread"):
        for k in (6, 12, 25, 50, 100):
            on = A[(A.EventId == 82) & A.Parameter.isin(others)]
            keep = (on.groupby("Parameter").head(k) if mode == "dead" else
                    on.groupby("Parameter", group_keys=False).apply(lambda g: g.sample(min(k, len(g)), random_state=1))).index
            drop = A.EventId.isin([81, 82]) & A.Parameter.isin(others)
            d = pd.concat([A[~drop], A.loc[keep],
                           A.loc[keep].assign(EventId=81, Timestamp=lambda x: x.Timestamp + pd.Timedelta(seconds=1))])
            o = predict(d, min_actuations=1).set_index("Detector")
            print(case, mode, f"k={k:3d} busy det {busy}: {o.loc[busy].health_status} | partners "
                  f"{o.reindex(others)['health_status'].tolist()}", flush=True)
