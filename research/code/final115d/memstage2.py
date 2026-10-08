"""Peak working set inside post_outputs (7-day sample)."""
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
    return round(p.WorkingSetSize / 1e6), round(p.PeakWorkingSetSize / 1e6)
sys.path.insert(0, os.environ["SRC"])
from detector_classifier import pipeline as P, health_core as hc, setback as sb, night_speed as ns
T0 = time.perf_counter()
def wrap(mod, name):
    o = getattr(mod, name)
    def w(*a, **kw):
        print(f"  before {name:24s} ws/peak={mem()}", flush=True)
        r = o(*a, **kw); print(f"  after  {name:24s} t={time.perf_counter()-T0:6.1f}s ws/peak={mem()}", flush=True); return r
    setattr(mod, name, w)
for m, n in ((sb, "setback"), (sb, "features"), (sb, "pair_probs"), (sb, "cues")):
    wrap(m, n)
P.sb_mod = sb; P.hc = hc; P.ns_mod = ns
P.predict(sys.argv[1], min_actuations=1, profile=sys.argv[2])
print("end", mem())
