"""per-function time inside hv4.assess (monkeypatched wrappers), mean of N runs"""
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
acc = collections.defaultdict(float)
def wrap(mod, name, tag):
    f = getattr(mod, name)
    @functools.wraps(f)
    def g(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); acc[tag] += time.perf_counter() - t; return r
    setattr(mod, name, g)
for n in ["occ_bins","bins_ctx","_median_excl","_ksum","n3_stats","act_stats","slopes","like_corr","occ_hi","shape110","level110","erratic",
          "drop_evidence","act118","episode_overlap","episodes_ctx","colour_bins","colour_stats","dropout_clean","band_of"]:
    if hasattr(hs, n): wrap(hs, n, "hs."+n)
for n in ["events_to_bins","on_episodes"]:
    wrap(hc, n, "hc."+n)
hv4.hc.events_to_bins = hc.events_to_bins; hv4.hc.on_episodes = hc.on_episodes
for n in ["_stage1","_resolve","_time_of_day","_outputs","severity_v4d","_limits","_stuck_stats","_least_strict"]:
    wrap(hv4, n, "hv4."+n)
N = 5
for lb in sys.argv[2].split(","):
    a, k = pickle.load(open(f"cap/{lb}.pkl", "rb"))
    P0 = a[0]
    def fresh():
        return (hc.prep_arrays(P0.t, P0.eid, P0.par),) + tuple(a[1:])
    hv4.assess(*fresh(), **k); acc.clear()
    t = time.perf_counter()
    for _ in range(N): hv4.assess(*fresh(), **k)
    tot = (time.perf_counter() - t) / N
    print(f"== {lb} total {tot:.3f}  ndet {len(a[3]) if a[3] else 0}")
    for kk, v in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"  {kk:22s} {v/N:.4f}")
