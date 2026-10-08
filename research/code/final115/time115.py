"""Note 115 rough timings (NOT the quiet-machine bench): one case per fresh process, 1 cold + 3 warm predict() calls,
stage timers by wrapping functions; peak working set.  python time115.py <flat-dir|src:dir> <case-file> <profile> <out.json>"""
import functools, json, os, sys, time
from pathlib import Path
pkg, f, profile, outp = sys.argv[1:5]
if pkg.startswith("src:"):
    sys.path.insert(0, pkg[4:])
    from detector_classifier import pipeline as P, blend as gb, features as f1, features_partner as f2, similarity as sm
    from detector_classifier import function_stage as fs, setback as sb, health_core as hc, features_expert as fx
    from detector_classifier import lanes as ln, funcnet as fnn
else:
    sys.path.insert(0, pkg)
    import predict as P, gru_blend as gb, features as f1, features_partner as f2, similarity as sm
    import function_stage as fs, setback as sb, health_core as hc, features_expert as fx, lanes as ln, funcnet as fnn
T = {}
def wrap(mod, name, label):
    if not hasattr(mod, name):
        return
    fn0 = getattr(mod, name)
    @functools.wraps(fn0)
    def g(*a, **k):
        t = time.perf_counter()
        try:
            return fn0(*a, **k)
        finally:
            T[label] = T.get(label, 0.0) + time.perf_counter() - t
    setattr(mod, name, g)
for mod, name, lab in [(P, "load_events", "load"), (P, "build_chunk_tables", "tables"), (gb, "streams_for", "streams"),
                       (P, "build_features", "features"), (sm, "build_lead_window", "f.lead"), (f1, "_mask_features", "f.mask"),
                       (P, "score", "phase_score"), (fnn.FuncNet, "both", "network"), (P, "score_function", "function"),
                       (fx, "build", "fn.expert"), (ln, "lanes", "fn.lanes"), (P, "post_outputs", "post"),
                       (sb, "setback", "post.setback"), (hc, "health", "post.health"), (P, "_assemble", "assemble")]:
    wrap(mod, name, lab)
import psutil  # noqa
proc = psutil.Process()
res = {"case": f, "profile": profile, "calls": []}
for i in range(4):
    T.clear()
    t = time.perf_counter()
    P.predict(f, min_actuations=1, profile=profile)
    res["calls"].append({"total": time.perf_counter() - t, **T})
res["peak_ws_mb"] = proc.memory_info().peak_wset / 2**20 if hasattr(proc.memory_info(), "peak_wset") else None
Path(outp).write_text(json.dumps(res, indent=1))
