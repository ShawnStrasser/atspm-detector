"""Peak working set after each stage (7-day sample). SRC=<src dir> (v7) or PKG=<flat dir> (ref114)."""
import sys, os, time, ctypes, ctypes.wintypes, warnings
warnings.simplefilter("ignore")


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t), ("d", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
ps = ctypes.windll.psapi; ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.wintypes.DWORD]


def mem():
    p = PMC(); p.cb = ctypes.sizeof(PMC); ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.WorkingSetSize / 1e6, p.PeakWorkingSetSize / 1e6


if os.environ.get("SRC"):
    sys.path.insert(0, os.environ["SRC"]); from detector_classifier import pipeline as P
else:
    sys.path.insert(0, os.environ["PKG"]); import predict as P
T0 = time.perf_counter()
for name in ["load_events", "build_chunk_tables", "gate_frame", "build_features", "score", "score_function", "post_outputs", "_assemble", "on_table"]:
    if hasattr(P, name):
        orig = getattr(P, name)
        def wrap(*a, _o=orig, _n=name, **kw):
            r = _o(*a, **kw)
            ws, pk = mem()
            print(f"  after {_n:20s} t={time.perf_counter()-T0:6.1f}s ws={ws:7.0f} peak={pk:7.0f} MB", flush=True)
            return r
        setattr(P, name, wrap)
for mod_name in ("gb",):
    gb = getattr(P, mod_name, None)
    if gb is not None and hasattr(gb, "streams_for"):
        o2 = gb.streams_for
        def sw(*a, _o=o2, **kw):
            r = _o(*a, **kw); ws, pk = mem(); print(f"  after streams_for          ws={ws:7.0f} peak={pk:7.0f} MB", flush=True); return r
        gb.streams_for = sw
for modn, fns in (("f1", ["apply_window", "shared_parts", "build_window", "finalise"]), ("f2", ["build_window", "add_partner_diffs"]),
                  ("sim_mod", ["build_window", "build_lead_window"]), ("f3", ["build"])):
    m = getattr(P, modn, None)
    for fnn in fns:
        if m is not None and hasattr(m, fnn):
            o3 = getattr(m, fnn)
            def w3(*a, _o=o3, _n=f"{modn}.{fnn}", **kw):
                r = _o(*a, **kw); ws, pk = mem(); print(f"    after {_n:22s} t={time.perf_counter()-T0:6.1f}s ws={ws:7.0f} peak={pk:7.0f} MB", flush=True); return r
            setattr(m, fnn, w3)
f = sys.argv[1]
print("start", mem())
P.predict(f, min_actuations=1, profile=sys.argv[2])
print("end", mem(), f"{time.perf_counter()-T0:.1f}s")
