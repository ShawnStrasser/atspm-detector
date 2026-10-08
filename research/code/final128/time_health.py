"""time hv4.assess on captured inputs; optional cProfile of one label"""
import sys, pickle, warnings, time, cProfile, pstats
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
from detector_classifier import health_v4 as hv4
import inspect as _insp  # CONPATCH
if "con" in _insp.signature(hv4.assess).parameters:
    from detector_classifier import pipeline as _pl
    _con = _pl._connect(4, "4GB")
    _orig_assess = hv4.assess
    hv4.assess = lambda *x, **kw: _orig_assess(*x, **kw, con=_con)
from detector_classifier import health_core as hc
labels = sys.argv[2].split(",")
prof = len(sys.argv) > 3
for lb in labels:
    a, k = pickle.load(open(f"cap/{lb}.pkl", "rb"))
    P0 = a[0]
    def fresh():
        return (hc.prep_arrays(P0.t, P0.eid, P0.par),) + tuple(a[1:])
    hv4.assess(*fresh(), **k)
    ts = []
    for _ in range(5):
        t = time.perf_counter(); hv4.assess(*fresh(), **k); ts.append(time.perf_counter() - t)
    print(lb, "median %.3f min %.3f" % (sorted(ts)[2], min(ts)), flush=True)
    if prof:
        pr = cProfile.Profile(); pr.enable()
        for _ in range(3): hv4.assess(*fresh(), **k)
        pr.disable()
        pstats.Stats(pr).sort_stats(sys.argv[3]).print_stats(45)
