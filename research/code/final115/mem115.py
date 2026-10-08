"""peak working set growth per stage (first call, then a warm call).  python mem115.py <pkg> <file> <profile>"""
import functools, sys, time, psutil
pkg, f, profile = sys.argv[1:4]
if pkg.startswith("src:"):
    sys.path.insert(0, pkg[4:])
    from detector_classifier import pipeline as P, blend as gb, features as f1, similarity as sm, function_stage as fs
    from detector_classifier import setback as sb, health_core as hc, features_expert as fx, funcnet as fnn
    import detector_classifier.features_lag as fl
else:
    sys.path.insert(0, pkg)
    import predict as P, gru_blend as gb, features as f1, similarity as sm, function_stage as fs, setback as sb
    import health_core as hc, features_expert as fx, funcnet as fnn, features_yellowred as fl
pr = psutil.Process()
log = []
def wrap(mod, name, lab):
    if not hasattr(mod, name):
        return
    f0 = getattr(mod, name)
    @functools.wraps(f0)
    def g(*a, **k):
        p0 = pr.memory_info().peak_wset
        try:
            return f0(*a, **k)
        finally:
            p1 = pr.memory_info().peak_wset
            if p1 > p0:
                log.append((lab, (p1 - p0) / 2**20, p1 / 2**20))
    setattr(mod, name, g)
for mod, name, lab in [(P, "load_events", "load"), (P, "build_chunk_tables", "tables"), (gb, "streams_for", "streams"),
                       (f1, "apply_window", "apply_window"), (f1, "shared_parts", "shared_parts"), (f1, "build_window", "f1.build"),
                       (f1, "_mask_features", "mask"), (f1, "finalise", "finalise"), (sm, "build_window", "sim"),
                       (sm, "build_lead_window", "lead"), (fl, "build", "lag"),
                       (P, "score", "phase_score"), (fnn.FuncNet, "both", "network"), (fx, "build", "expert"), (fs, "run", "fs.run"),
                       (sb, "setback", "setback"), (hc, "health", "health"), (P, "_assemble", "assemble")]:
    wrap(mod, name, lab)
import numpy, pandas, duckdb, onnxruntime  # noqa
print("after import", pr.memory_info().peak_wset / 2**20)
for i in range(2):
    log.clear()
    P.predict(f, min_actuations=1, profile=profile)
    print(f"call {i}: peak {pr.memory_info().peak_wset/2**20:.0f} MB; growth:", [(a, round(b), round(c)) for a, b, c in log])
