"""Note 115d bench: one process per (package, sample, profile): 1 cold + 3 warm predict() calls; peak working set.
    python bench115d.py <ref114|v7d> <sample> <profile>  ->  one JSON line"""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import ctypes, ctypes.wintypes, json, os, sys, time, warnings
warnings.simplefilter("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")


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


W = DCW
SAMPLES = {"v13_h3": (W + r"\s115v\ev\v13.parquet", "2026-09-20 09:00:00", "2026-09-20 12:00:00"),
           "busy_h3": (W + r"\bench71\ev_busiest_ev_h3.parquet", None, None),
           "v13_d7": (W + r"\s115v\rob\week_v13.parquet", None, None)}
pkg, sample, prof = sys.argv[1:4]
if pkg == "ref114":
    sys.path.insert(0, W + r"\s115\ref114"); from predict import predict
else:
    sys.path.insert(0, W + r"\final_v7_prod\src"); from detector_classifier import predict
f, s, e = SAMPLES[sample]
t = time.perf_counter(); o = predict(f, start=s, end=e, min_actuations=1, profile=prof); cold = time.perf_counter() - t
pk1 = peak_mb()
warm = []
for _ in range(3):
    t = time.perf_counter(); predict(f, start=s, end=e, min_actuations=1, profile=prof); warm.append(time.perf_counter() - t)
print(json.dumps({"pkg": pkg, "sample": sample, "profile": prof, "rows": len(o), "cold_s": round(cold, 3),
                  "warm_s": [round(x, 3) for x in warm], "warm_min_s": round(min(warm), 3), "peak_mb_one_call": round(pk1), "peak_mb_4_calls": round(peak_mb())}),
      flush=True)
