import os, sys, time, pickle, warnings
for k in ("OMP_NUM_THREADS", "DC_TREE_THREADS", "DC_NET_THREADS"): os.environ[k] = "4"
from pathlib import Path
import pandas as pd
W = Path(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work"))); E = W / "s128v" / "ev"; O = W / "s128v" / "par"
src, tag = sys.argv[1], sys.argv[2]
sys.path.insert(0, src); warnings.simplefilter("ignore")
from detector_classifier import predict
S = pd.read_csv(E / "signals.csv")
WINS = {"m30": ("2026-09-20 16:30:00", "2026-09-20 17:00:00"), "h3": ("2026-09-20 13:00:00", "2026-09-20 16:00:00")}
res = {}
for case in S.case:
    for win, (a, b) in WINS.items():
        f = str(E / f"{case}_{win}.parquet")
        t = time.perf_counter(); out, ph = predict(f, end=b, return_phases=True)
        res[(case, win, "endonly")] = (out, ph, time.perf_counter() - t, [])
pickle.dump(res, open(O / f"{tag}.pkl", "wb"))
