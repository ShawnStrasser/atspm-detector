"""Note 151: identity + speed harness for predict() before (backup of src) and after (vectorised src).
One job per process (peak memory = the process's peak working set).

    python run151.py <job> --src <package parent> --tag <tag> [--threads 4]
jobs:
  sample          the bundled sample (return_phases)
  alone100        100 h24_b signals (seed 151), each predicted ALONE (events DataFrame of that signal only)
  batch<N>        the first N of the seed-151 order of h24_b signals (N = 1, 10, 100, 755) in ONE predict call on the
                  day parquet (device_ids filter), start / end = the day
  multi           the 4-5 day logs of note 141b, each alone and then all three in one call
Outputs %DC_WORK%/s151/runs/<tag>_<job>_{det,ph}.parquet and <tag>_<job>_times.csv
"""
from __future__ import annotations

import argparse
import ctypes
import functools
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s151" / "runs"
DAY = dict(start="2026-09-28 00:00", end="2026-09-29 00:00")
MULTI = ["ev_typical_r8_d5", "ev_busiest_ev_d5", "ev_asclog_d4"]
T: dict = {}


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_mb() -> float:
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    f = k32.K32GetProcessMemoryInfo
    f.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    f.restype = wintypes.BOOL
    f(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20


def timed(mod, name, key=None):
    if not hasattr(mod, name):
        return
    f = getattr(mod, name)
    key = key or name

    @functools.wraps(f)
    def g(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[key] = T.get(key, 0.0) + time.perf_counter() - t0
    setattr(mod, name, g)


def order_ids():
    import numpy as np
    import pandas as pd
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not {i.lower() for i in ids} & locked
    rng = np.random.default_rng(151)
    return [ids[i] for i in rng.permutation(len(ids))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--kw", default="", help="extra predict kwargs k=v,k=v (ints)")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import pandas as pd
    from atspm_detector import pipeline as PP, health_v4 as hv4
    kwx = {}
    for kv in filter(None, a.kw.split(",")):
        k, v = kv.split("=")
        kwx[k] = int(v)
    # health = the event fetch + the assessment (old: per signal; new: batched helpers if present)
    timed(hv4, "assess", "health")
    timed(PP, "_health_events", "health")
    timed(PP, "_health_events_all", "health")
    OUT.mkdir(parents=True, exist_ok=True)
    day = DCW / "s151" / "h24b_events.parquet"
    rows, dets, phs = [], [], []

    def go(label, events, ids=None, **kw):
        T.clear()
        t0 = time.perf_counter()
        d, p = PP.predict(events, device_ids=ids, threads=a.threads, return_phases=True, **kw, **kwx)
        wall = time.perf_counter() - t0
        rows.append(dict(item=label, wall_s=wall, health_s=T.get("health", 0.0), n_sig=d.DeviceId.nunique(),
                         n_det=len(d), peak_mb=peak_mb()))
        dets.append(d.assign(item=label))
        phs.append(p.assign(item=label))

    # warm-up (model loading) on the bundled sample, not timed
    sample = Path(PP.__file__).parent / "data" / "sample_events.parquet"
    PP.predict(str(sample), threads=a.threads)
    base_mb = peak_mb()
    if a.job == "sample":
        go("sample", str(sample))
    elif a.job == "alone100":
        for dev in order_ids()[:100]:
            ev = pd.read_parquet(day, filters=[("DeviceId", "==", dev)])
            go(dev, ev, **DAY)
    elif a.job.startswith("batch"):
        n = int(a.job[5:])
        go(a.job, str(day), order_ids()[:n], **DAY)
    elif a.job == "multi":
        evs = {m: pd.read_parquet(DCW / "s141p" / "data" / f"{m}.parquet") for m in MULTI}
        for m, ev in evs.items():
            go(m, ev)
        go("multi_all", pd.concat(list(evs.values()), ignore_index=True))
    tm = pd.DataFrame(rows).assign(tag=a.tag, job=a.job, warm_peak_mb=base_mb)
    tm.to_csv(OUT / f"{a.tag}_{a.job}_times.csv", index=False)
    pd.concat(dets, ignore_index=True).to_parquet(OUT / f"{a.tag}_{a.job}_det.parquet")
    pd.concat(phs, ignore_index=True).to_parquet(OUT / f"{a.tag}_{a.job}_ph.parquet")
    print(tm[["item", "wall_s", "health_s", "n_sig", "n_det", "peak_mb"]].tail(5).to_string(), flush=True)
    print(f"total wall {tm.wall_s.sum():.1f}s health {tm.health_s.sum():.1f}s", flush=True)


if __name__ == "__main__":
    main()
