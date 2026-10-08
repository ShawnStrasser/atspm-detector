"""run one package on the 10 new signals x 30 min / 3 h / 24 h (+ far bounds).  python par.py <src> <tag> [far]"""
import os, sys, time, pickle, warnings
for k in ("OMP_NUM_THREADS", "DC_TREE_THREADS", "DC_NET_THREADS"): os.environ[k] = "4"
from pathlib import Path
import pandas as pd
W = Path(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work"))); E = W / "s128v" / "ev"; O = W / "s128v" / "par"
src, tag = sys.argv[1], sys.argv[2]; far = len(sys.argv) > 3
sys.path.insert(0, src); warnings.simplefilter("ignore")
import detector_classifier
from detector_classifier import predict
assert Path(detector_classifier.__file__).resolve().is_relative_to(Path(src).resolve()), detector_classifier.__file__
lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
S = pd.read_csv(E / "signals.csv"); assert not S.DeviceId.str.lower().isin(lk).any()
WINS = {"m30": ("2026-09-20 16:30:00", "2026-09-20 17:00:00"), "h3": ("2026-09-20 13:00:00", "2026-09-20 16:00:00"), "h24": (None, None)}
def sl(case, win):
    f = E / f"{case}_{win}.parquet"
    if not f.exists():
        e = pd.read_parquet(E / f"{case}.parquet"); a, b = WINS[win]
        if a: e = e[(e.Timestamp >= a) & (e.Timestamp < b)]
        e.to_parquet(f, index=False)
    return str(f)
res = {}
predict(sl("x00", "m30"))  # warm
for case in S.case:
    for win, (a, b) in WINS.items():
        f = sl(case, win)
        calls = {"exact": dict(start=a, end=b), "nobound": {}} if a else {"nobound": {}}
        if far:
            t0 = pd.read_parquet(f, columns=["Timestamp"]).Timestamp.min()
            calls = {"start365": dict(start=str(t0 - pd.Timedelta(days=365)), end=b),
                     "both": dict(start="2000-01-01 00:00:00", end="2030-01-01 00:00:00"),
                     "start1h": dict(start=str(t0.floor("min") - pd.Timedelta(hours=1)), end=b)}
        for k, kw in calls.items():
            t = time.perf_counter()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out, ph = predict(f, return_phases=True, **kw)
            dt = time.perf_counter() - t
            res[(case, win, k)] = (out, ph, dt, sorted({str(x.message)[:100] for x in w}))
            print(case, win, k, f"{dt:.2f}s", len(out), flush=True)
pickle.dump(res, open(O / f"{tag}.pkl", "wb"))
