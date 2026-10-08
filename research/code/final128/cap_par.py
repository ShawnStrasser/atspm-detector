"""capture hv4.assess inputs for the note-115 parity cases -> cap/par_<case>.pkl"""
import sys, pickle, warnings, os
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "final124")))
import par124
from detector_classifier import pipeline as P, health_v4 as hv4
orig = hv4.assess
for case, f, s, e in par124.CASES:
    if case.startswith("b_"): continue
    cap = {}
    def a(*x, **k):
        cap["a"] = (x, k); return orig(*x, **k)
    hv4.assess = a
    P.predict(f, start=s, end=e, min_actuations=1)
    if "a" in cap:
        pickle.dump(cap["a"], open(f"cap/par_{case}.pkl", "wb"))
    print(case, flush=True)
