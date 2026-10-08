"""Note 98: speed / peak RAM of the beta (final_v2) vs the final package (v4f) and its simplified variants.

bench84 protocol: ONE signal per predict() call, every (package, signal, length) in a FRESH process, 4 threads;
import = seconds to `import predict`; cold = first predict() (models parsed, sessions created); warm = median of 3
further calls (models kept = a service); peak = process peak working set (psutil).  Signals = the note-71 bench extracts
(non-locked, asserted): typical r8 (22 ch) / r11 (18 ch), busiest by events (31 ch) / channels (43 ch); 30 min / 3 h / 24 h.
Machine load is recorded per case: whole-machine CPU % sampled every 0.5 s while the child runs, and the CPU % of every
other process above 5 % (another agent shares the machine).

    python bench98.py run <tag> <package dir> [ENV=VAL;ENV=VAL]   -> %DC_WORK%/final_v3_work/f98/bench/<tag>/r_<sig>_<L>.json
    python bench98.py summary <tag>[,<tag>...]                      -> print + f98/bench/summary_<tags>.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
B71 = W / "bench71"
OUT = W / "final_v3_work" / "f98" / "bench"
SIGS = os.environ.get("B98_SIGS", "typical_r8,typical_r11,busiest_ev,busiest_ch").split(",")
LENS = os.environ.get("B98_LENS", "m30,h3,h24").split(",")
THREADS = 4


def ev_file(sig, L):
    return B71 / f"ev_{sig}_{L}.parquet"


def one(pkg, sig, L, out):
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    import numpy as np
    import psutil
    proc = psutil.Process()
    t0 = time.perf_counter()
    sys.path.insert(0, str(pkg))
    import predict as P
    t_imp = time.perf_counter() - t0
    rss_imp = proc.memory_info().rss / 2**20
    T = None
    try:                                           # stage timing where the package has the v3 modules
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final75"))
        import bench75 as B75
        T = B75.instrument(P)
    except Exception:
        T = None
    f = str(ev_file(sig, L))
    s = time.perf_counter()
    res = P.predict(f, threads=THREADS)
    cold = time.perf_counter() - s
    peak_cold = proc.memory_info().peak_wset / 2**20
    warm, st_w, cpu = [], [], []
    for _ in range(3):
        if T is not None:
            T.clear()
        c0 = proc.cpu_times()
        s = time.perf_counter()
        P.predict(f, threads=THREADS)
        warm.append(time.perf_counter() - s)
        c1 = proc.cpu_times()
        cpu.append((c1.user - c0.user) + (c1.system - c0.system))
        if T is not None:
            st_w.append(dict(T))
    stages = {}
    if st_w:
        keys = sorted(set().union(*[set(x) for x in st_w]))
        stages = {k: float(np.median([x.get(k, 0.0) for x in st_w])) for k in keys}
    json.dump(dict(sig=sig, L=L, pkg=str(pkg), channels=int(res.Detector.nunique()),
                   answered=int(res.function_pred.notna().sum()), import_s=t_imp, rss_import_mb=rss_imp,
                   cold_s=cold, warm_s=float(np.median(warm)), warm_all=warm, warm_cpu_s=float(np.median(cpu)),
                   profile=os.environ.get("DC_FAST_PROFILE"), peak_cold_mb=peak_cold,
                   peak_all_mb=proc.memory_info().peak_wset / 2**20, stages_warm=stages,
                   env={k: v for k, v in os.environ.items() if k.startswith("DC_")}), open(out, "w"), indent=1)


def _load_monitor(stop, samples, others, me, child):
    import psutil
    psutil.cpu_percent(None)
    procs = {}
    while not stop.is_set():
        time.sleep(0.5)
        samples.append(psutil.cpu_percent(None))
        try:
            mine = set(me) | {c.pid for c in psutil.Process(child).children(recursive=True)}
        except Exception:
            mine = set(me)
        for p in psutil.process_iter(["pid", "name"]):
            if p.pid in mine or p.pid == 0:
                continue
            try:
                if p.pid not in procs:
                    procs[p.pid] = p
                    p.cpu_percent(None)
                    continue
                c = p.cpu_percent(None)
                if c > 5:
                    others.setdefault(f"{p.info['name']}:{p.pid}", []).append(c)
            except Exception:
                pass


def run(tag, pkg, envs=""):
    import duckdb
    import pandas as pd
    o_d = OUT / tag
    o_d.mkdir(parents=True, exist_ok=True)
    lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(THREADS))
    for kv in [x for x in envs.split(";") if x]:
        k, v = kv.split("=", 1)
        env[k] = v
    for sig in SIGS:
        for L in LENS:
            ids = duckdb.sql(f"select distinct lower(DeviceId) d from '{ev_file(sig, L)}'").df().d
            assert not set(ids) & lk
            o = o_d / f"r_{sig}_{L}.json"
            stop, samples, others = threading.Event(), [], {}
            p = subprocess.Popen([sys.executable, "-W", "ignore", __file__, "--one", str(pkg), sig, L, str(o)], env=env)
            th = threading.Thread(target=_load_monitor, args=(stop, samples, others, {os.getpid(), p.pid}, p.pid), daemon=True)
            th.start()
            rc = p.wait()
            stop.set()
            th.join()
            assert rc == 0, (tag, sig, L)
            r = json.load(open(o))
            import numpy as np
            r["machine_cpu_pct_mean"] = float(np.mean(samples)) if samples else None
            r["others_cpu_pct_mean"] = {k: round(float(np.mean(v)), 1) for k, v in others.items() if len(v) >= 2}
            r["tag"], r["env_set"] = tag, envs
            json.dump(r, open(o, "w"), indent=1)
            print(f"{tag} {sig} {L}: import {r['import_s']:.2f} cold {r['cold_s']:.2f}s warm {r['warm_s']:.2f}s "
                  f"peak {r['peak_all_mb']:.0f} MB ({r['channels']} ch) machine cpu {r['machine_cpu_pct_mean']:.0f}%",
                  flush=True)


def summary(tags):
    rows = {}
    for t in tags:
        for f in sorted((OUT / t).glob("r_*.json")):
            r = json.load(open(f))
            rows[f"{t}|{r['sig']}|{r['L']}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                                if k not in ("warm_all", "pkg")}
    json.dump(rows, open(OUT / f"summary_{'_'.join(tags)}.json", "w"), indent=1)
    for k, r in rows.items():
        oth = sum(v for v in (r.get("others_cpu_pct_mean") or {}).values())
        print(f"{k:32s} import {r['import_s']:.2f} cold {r['cold_s']:.2f} warm {r['warm_s']:.2f} "
              f"cpu-s {r.get('warm_cpu_s', float('nan')):.2f} peak {r['peak_all_mb']:.0f} MB machine "
              f"{r.get('machine_cpu_pct_mean'):.0f}% others {oth:.0f}%")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(Path(sys.argv[2]), *sys.argv[3:6])
    elif sys.argv[1] == "run":
        run(sys.argv[2], Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
    else:
        summary(sys.argv[2].split(","))
