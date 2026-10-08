"""7-day sample: one 24-h day of v13 (40 channels) tiled 7 times (+1 day each), via DuckDB only."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import sys, time, ctypes, ctypes.wintypes, warnings, os
from pathlib import Path
import duckdb
warnings.simplefilter("ignore")
V = Path(DCW + r"\s115v")
f = V / "rob" / "week_v13.parquet"
if not f.exists():
    c = duckdb.connect(); c.execute("set threads=4; set memory_limit='4GB'")
    src = (V / "ev" / "v13.parquet").as_posix()
    c.execute(f"""COPY (SELECT DeviceId, Timestamp + INTERVAL (k) DAY AS Timestamp, EventId, Parameter
        FROM '{src}', range(7) t(k)) TO '{f.as_posix()}' (FORMAT parquet)""")
    print("rows", c.sql(f"select count(*), min(Timestamp), max(Timestamp) from '{f.as_posix()}'").fetchall())


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


prof = sys.argv[1]
if os.environ.get("SRC"):
    sys.path.insert(0, os.environ["SRC"]); from detector_classifier import predict
elif os.environ.get("PKG"):
    sys.path.insert(0, os.environ["PKG"]); from predict import predict
else:
    from detector_classifier import predict
t = time.perf_counter()
o, ph = predict(str(f), min_actuations=1, return_phases=True, profile=prof)
print(f"{prof}: {time.perf_counter()-t:.1f}s peak {peak_mb():.0f} MB rows {len(o)} answered {int(o.phase_pred.notna().sum())} "
      f"fn {o.function_pred.value_counts().to_dict()} health {o.health_status.value_counts().to_dict()}", flush=True)
print("status sample:", o.status.str[:90].value_counts().head(5).to_dict())
o.to_csv(V / "rob" / f"week_{os.environ.get('TAG', 'v7')}_{prof}.csv", index=False)
