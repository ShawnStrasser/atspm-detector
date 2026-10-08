"""capture hv4.assess inputs for bench71 samples -> cap/<label>.pkl"""
import sys, pickle, warnings, os
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
from detector_classifier import pipeline as P, health_v4 as hv4
os.makedirs("cap", exist_ok=True)
orig = hv4.assess
for b in ("typical_r8", "typical_r11", "busiest_ev", "busiest_ch"):
    for L in ("m30", "h3", "h24"):
        cap = {}
        def a(*x, **k):
            cap["a"] = (x, k); return orig(*x, **k)
        hv4.assess = a
        P.predict(os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), rf"bench71\ev_{b}_{L}.parquet"), min_actuations=1)
        pickle.dump(cap["a"], open(f"cap/{b}_{L}.pkl", "wb"))
        print(b, L, flush=True)
