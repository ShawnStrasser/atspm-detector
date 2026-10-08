"""health stage alone (_health_events + hv4.assess) inside predict(), warm, median of 5.  python ht.py <src> <tag>"""
import sys, time, warnings, os, json
for k in ("OMP_NUM_THREADS", "DC_TREE_THREADS", "DC_NET_THREADS"): os.environ[k] = "4"
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
from detector_classifier import pipeline as P, health_v4 as hv4
acc = {"h": 0.0}
o1, o2 = P._health_events, hv4.assess
def he(*a, **k):
    t = time.perf_counter(); r = o1(*a, **k); acc["h"] += time.perf_counter() - t; return r
def as_(*a, **k):
    t = time.perf_counter(); r = o2(*a, **k); acc["h"] += time.perf_counter() - t; return r
P._health_events, hv4.assess = he, as_
E = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"s128v\ev")
out = []
for s in ("x01", "x05", "x09"):
    for L in ("m30", "h3", "h24"):
        f = fr"{E}\{s}_{L}.parquet"
        P.predict(f)
        tt, H = [], []
        for _ in range(5):
            acc["h"] = 0.0; t = time.perf_counter(); P.predict(f); tt.append(time.perf_counter() - t); H.append(acc["h"])
        r = dict(tag=sys.argv[2], sig=s, L=L, total=sorted(tt)[2], health=sorted(H)[2], health_min=min(H))
        out.append(r); print(json.dumps(r), flush=True)
