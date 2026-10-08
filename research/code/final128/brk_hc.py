import sys, pickle, warnings, time, collections, functools
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
from detector_classifier import health_v4 as hv4
import inspect as _insp  # CONPATCH
if "con" in _insp.signature(hv4.assess).parameters:
    from detector_classifier import pipeline as _pl
    _con = _pl._connect(4, "4GB")
    _orig_assess = hv4.assess
    hv4.assess = lambda *x, **kw: _orig_assess(*x, **kw, con=_con)
from detector_classifier import health_core as hc, health_v4_stats as hs
acc = collections.defaultdict(float); cnt = collections.Counter()
def wrap(mod, name, tag):
    f = getattr(mod, name)
    @functools.wraps(f)
    def g(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); acc[tag] += time.perf_counter() - t; cnt[tag] += 1; return r
    setattr(mod, name, g)
for n in ["events_to_bins","det_stats","shape_stats","act_stats","on_arrays","spike_stats","_partner","on_episodes",
          "rules","partner_notes","twins","_refs","rapid_ratio","_recovery","_flow","health"]:
    wrap(hc, n, n)
N = 5
for lb in sys.argv[2].split(","):
    a, k = pickle.load(open(f"cap/{lb}.pkl", "rb"))
    P0 = a[0]
    def fresh():
        return (hc.prep_arrays(P0.t, P0.eid, P0.par),) + tuple(a[1:])
    hv4.assess(*fresh(), **k); acc.clear(); cnt.clear()
    for _ in range(N): hv4.assess(*fresh(), **k)
    print(f"== {lb}")
    for kk, v in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"  {kk:22s} {v/N:.4f}  n={cnt[kk]//N}")
