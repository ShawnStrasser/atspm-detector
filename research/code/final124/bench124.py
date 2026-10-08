"""Note 124 bench: one fresh process per (package, sample): import, cold call, 3 warm calls; peak working set.
    python bench124.py <src dir> <events file> <label> [threads]   -> prints one JSON line"""
import ctypes, ctypes.wintypes, json, os, sys, time, warnings
thr = sys.argv[4] if len(sys.argv) > 4 else "4"
for k in ("OMP_NUM_THREADS", "DC_TREE_THREADS", "DC_NET_THREADS"):
    os.environ[k] = thr
warnings.simplefilter("ignore")
src, f, label = sys.argv[1:4]


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
from detector_classifier import predict  # noqa: E402
t = time.perf_counter()
predict(f, min_actuations=1, threads=int(thr))
cold = time.perf_counter() - t
w = []
for _ in range(3):
    t = time.perf_counter()
    predict(f, min_actuations=1, threads=int(thr))
    w.append(time.perf_counter() - t)
print(json.dumps({"label": label, "threads": thr, "cold_s": round(cold, 3), "warm_med_s": round(sorted(w)[1], 3),
                  "warm_min_s": round(min(w), 3), "peak_mb": round(peak_mb())}), flush=True)
