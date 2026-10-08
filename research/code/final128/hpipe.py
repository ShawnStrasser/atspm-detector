"""health share of a predict() call: time in _health_events + hv4.assess, warm, per bench sample"""
import sys, time, warnings, os
for k in ("OMP_NUM_THREADS", "DC_TREE_THREADS", "DC_NET_THREADS"): os.environ[k] = "4"
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
from detector_classifier import pipeline as P, health_v4 as hv4
acc = {"he": 0.0, "as": 0.0}
o1, o2 = P._health_events, hv4.assess
def he(*a, **k):
    t = time.perf_counter(); r = o1(*a, **k); acc["he"] += time.perf_counter() - t; return r
def as_(*a, **k):
    t = time.perf_counter(); r = o2(*a, **k); acc["as"] += time.perf_counter() - t; return r
P._health_events, hv4.assess = he, as_
for f in sys.argv[2:]:
    P.predict(f, min_actuations=1)
    tt, H = [], []
    for _ in range(3):
        acc.update(he=0.0, **{"as": 0.0}); t = time.perf_counter(); P.predict(f, min_actuations=1)
        tt.append(time.perf_counter() - t); H.append(acc["he"] + acc["as"])
    i = sorted(range(3), key=lambda j: tt[j])[1]
    print(f"{os.path.basename(f):28s} total {tt[i]:.3f}s  health {H[i]:.3f}s", flush=True)
