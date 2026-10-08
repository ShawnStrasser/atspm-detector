"""Note 120: the one path on ONE thread (edge device).  One fresh process per sample:
    python edge1t.py <src dir> <events file> <label> [duckdb memory limit]
prints JSON: cold s (first call incl. model load), warm s (second call), peak working set MB, rows."""
import os, sys, time, json, ctypes, ctypes.wintypes, warnings
for k in ("OMP_NUM_THREADS", "DC_TREE_THREADS", "DC_NET_THREADS"):
    os.environ[k] = "1"
warnings.simplefilter("ignore")
src, f, label = sys.argv[1:4]
mem = sys.argv[4] if len(sys.argv) > 4 else "4GB"


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t), ("d", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_mb():
    p = PMC(); p.cb = ctypes.sizeof(PMC)
    k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
    ps = ctypes.windll.psapi; ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.wintypes.DWORD]
    ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.PeakWorkingSetSize / 1e6


sys.path.insert(0, src)
t = time.perf_counter()
from detector_classifier import predict
t_imp = time.perf_counter() - t
m_imp = peak_mb()
t = time.perf_counter()
o = predict(f, min_actuations=1, threads=1, memory=mem)
cold = time.perf_counter() - t
m1 = peak_mb()
t = time.perf_counter()
o2 = predict(f, min_actuations=1, threads=1, memory=mem)
warm = time.perf_counter() - t
if os.environ.get("SAVE"):
    o.to_pickle(os.environ["SAVE"])
print(json.dumps({"label": label, "mem_limit": mem, "import_s": round(t_imp, 2), "cold_s": round(cold, 2),
                  "warm_s": round(warm, 2), "peak_after_import_mb": round(m_imp), "peak_mb_first_call": round(m1),
                  "peak_mb": round(peak_mb()), "rows": len(o), "answered": int(o.phase_pred.notna().sum()),
                  "same_twice": bool(o.equals(o2))}), flush=True)
