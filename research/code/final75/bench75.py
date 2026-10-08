"""Note 75: end-to-end speed and peak RAM of `%DC_WORK%/final_v3_candidate_v3`, ONE signal per predict() call.

Signals = note-71 bench extracts (non-locked, asserted): typical (typical_r8, 19-22 channels; typical_r11), busiest by
channels (busiest_ch, 43), busiest by events (busiest_ev, 31); lengths 30 min / 3 h / 24 h.  Every (signal, length) runs
in a FRESH process, 4 threads (the package default):
  import      seconds to `import predict` (python + numpy / pandas / duckdb / onnxruntime + package)
  cold        first predict() (model files parsed, ONNX sessions created)
  warm        median of 3 further predict() calls in the same process (sessions kept = a service)
  peak        process peak working set (psutil), after the cold call and after all calls
  stages      warm seconds per stage (wrapped functions; the rest = glue)
Also: phase-number invariance on 3 h of the busiest signal (phases renumbered in the raw log -> every answer follows,
lanes / setback / speed / health identical).

    python bench75.py run | inv | summary     -> %DC_WORK%/final_v3_work/bench75/
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PKG = W / "final_v3_candidate_v3"
B71 = W / "bench71"
OUT = W / "final_v3_work" / "bench75"
SIGS = ["typical_r8", "typical_r11", "busiest_ch", "busiest_ev"]
LENS = ["m30", "h3", "h24"]
THREADS = 4


def ev_file(sig, L):
    return B71 / f"ev_{sig}_{L}.parquet"


def instrument(P):
    import functools
    import function_stage as fs
    import funcnet
    import gru_blend as gb
    import lanes as lm
    import pick as pk
    import setback as sb
    import health_core as hc
    import night_speed as ns
    import stacker as st
    import features_expert as fx
    import atspm_decode as ad
    T = {}

    def wrap(obj, name, label):
        f = getattr(obj, name)

        @functools.wraps(f)
        def g(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                T[label] = T.get(label, 0.0) + time.perf_counter() - t
        setattr(obj, name, g)
    wrap(P, "load_events", "events: load")
    wrap(P, "build_chunk_tables", "events: tables")
    wrap(gb, "streams_for", "networks: streams")
    wrap(P, "build_features", "phase features")
    wrap(gb, "phase_probs_kept", "GRU (kept pairs)")
    wrap(P, "score", "[incl] phase total (trees + GRU + decoder)")
    wrap(fx, "build", "function features: expert")
    wrap(P, "_function_frame", "function features: frame")
    wrap(funcnet.FuncNet, "probs", "function net (siba)")
    wrap(lm, "lanes", "lanes D")
    wrap(pk, "span_feats", "pick inputs")
    wrap(pk, "track_feats", "pick inputs")
    wrap(pk, "stack_health", "pick inputs + health ctx (stack)")
    wrap(fs, "_health_context", "health ctx (function-free health_core)")
    wrap(st.Stacker, "predict", "stacker")
    wrap(ad, "decode_signal", "decode")
    wrap(P, "score_function", "[incl] function total")
    wrap(sb, "setback", "setback")
    wrap(hc, "health", "[incl] health_core calls (ctx + output)")
    wrap(ns, "night_speed", "night speed")
    wrap(P, "post_outputs", "[incl] post outputs (setback + health + speed)")
    wrap(P, "_assemble", "output assembly")
    return T


def one(sig, L, out):
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    import psutil
    t0 = time.perf_counter()
    proc = psutil.Process()
    sys.path.insert(0, str(PKG))
    import predict as P
    t_imp = time.perf_counter() - t0
    rss_imp = proc.memory_info().rss / 2**20
    T = instrument(P)
    f = str(ev_file(sig, L))
    s = time.perf_counter()
    res = P.predict(f, threads=THREADS)
    cold = time.perf_counter() - s
    peak_cold = proc.memory_info().peak_wset / 2**20
    stages_cold = dict(T)
    warm, st_w = [], []
    for _ in range(3):
        T.clear()
        s = time.perf_counter()
        P.predict(f, threads=THREADS)
        warm.append(time.perf_counter() - s)
        st_w.append(dict(T))
    import numpy as np
    keys = sorted(set().union(*[set(x) for x in st_w]))
    stages_warm = {k: float(np.median([x.get(k, 0.0) for x in st_w])) for k in keys}
    json.dump(dict(sig=sig, L=L, channels=int(res.Detector.nunique()), answered=int(res.function_pred.notna().sum()),
                   cand=int(res.n_candidate_phases.max()), import_s=t_imp, rss_import_mb=rss_imp, cold_s=cold,
                   warm_s=float(np.median(warm)), warm_all=warm, peak_cold_mb=peak_cold,
                   peak_all_mb=proc.memory_info().peak_wset / 2**20, stages_cold=stages_cold, stages_warm=stages_warm,
                   lanes=int(res.lanes.notna().sum()), setback=int(res.distance_ft.notna().sum()),
                   night_speed=int(res.night_speed_mph.notna().sum())), open(out, "w"), indent=1)


def cmd_run(a):
    import duckdb
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
    for sig in SIGS:
        for L in LENS:
            ids = duckdb.sql(f"select distinct lower(DeviceId) d from '{ev_file(sig, L)}'").df().d
            assert not set(ids) & lk
    for sig in SIGS:
        for L in LENS:
            o = OUT / f"r_{sig}_{L}.json"
            if o.exists() and not a.force:
                continue
            subprocess.run([sys.executable, "-W", "ignore", __file__, "--one", sig, L, str(o)], check=True)
            r = json.load(open(o))
            print(f"{sig} {L}: cold {r['cold_s']:.2f}s warm {r['warm_s']:.2f}s peak {r['peak_all_mb']:.0f} MB "
                  f"({r['channels']} ch)", flush=True)


def inv_one(pkg, sig, L, out):
    """one package, one (signal, length): phases renumbered in the raw log -> what moves."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(pkg))
    import predict as P
    ev = pd.read_parquet(ev_file(sig, L))
    is_ph = ev.EventId.isin([1, 7, 8, 9, 10, 11, 43, 44])
    phases = sorted(ev.loc[is_ph, "Parameter"].unique())
    perm = dict(zip(phases, np.random.default_rng(75).permutation(phases)))
    ev2 = ev.copy()
    ev2.loc[is_ph, "Parameter"] = ev2.loc[is_ph, "Parameter"].map(perm)
    base = P.predict(ev, threads=THREADS, min_actuations=1)
    scr = P.predict(ev2, threads=THREADS, min_actuations=1)
    m = base.merge(scr, on=["DeviceId", "Detector"], suffixes=("", "_s"))
    a_ = m[m.phase_pred.notna() & m.phase_pred_s.notna()]
    r = dict(pkg=pkg.name, sig=sig, L=L, detectors=int(len(m)),
             phase_followed=float((a_.phase_pred.astype(int).map(perm) == a_.phase_pred_s.astype(int)).mean()),
             phase_prob_max_abs=float(np.abs(a_.phase_prob - a_.phase_prob_s).max()),
             function_same=float((m.function_pred.astype(str) == m.function_pred_s.astype(str)).mean()),
             function_prob_max_abs=float(np.nanmax(np.abs(m.function_prob.astype(float) - m.function_prob_s.astype(float)))))
    for c in ("lanes", "distance_ft", "night_speed_mph", "health_status"):
        if c in m:
            r[f"{c}_same"] = float((m[c].astype(str) == m[c + "_s"].astype(str)).mean())
    json.dump(r, open(out, "w"))


def cmd_inv(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    pkgs = [PKG, Path(__file__).resolve().parents[3] / "model"]
    for pkg in pkgs:
        for sig in ("typical_r8", "busiest_ch", "busiest_ev"):
            for L in ("m30", "h3", "h24"):
                o = OUT / f"inv_{pkg.name}_{sig}_{L}.json"
                if not o.exists():
                    subprocess.run([sys.executable, "-W", "ignore", __file__, "--inv", str(pkg), sig, L, str(o)], check=True)
                rows.append(json.load(open(o)))
                print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(OUT / "invariance.json", "w"), indent=1)


def cmd_summary(a):
    rows = []
    for f in sorted(OUT.glob("r_*.json")):
        rows.append(json.load(open(f)))
    keep = {}
    for r in rows:
        keep[f"{r['sig']}|{r['L']}"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                       if k not in ("stages_cold", "stages_warm", "warm_all")}
        keep[f"{r['sig']}|{r['L']}"]["stages_warm"] = {k: round(v, 3) for k, v in r["stages_warm"].items()}
        print(f"{r['sig']:12s} {r['L']:4s} ch {r['channels']:3d} cand {r['cand']}: import {r['import_s']:.2f}s cold "
              f"{r['cold_s']:.2f}s warm {r['warm_s']:.2f}s peak {r['peak_all_mb']:.0f} MB (cold peak "
              f"{r['peak_cold_mb']:.0f})")
        print("      " + " | ".join(f"{k} {v:.2f}" for k, v in sorted(r["stages_warm"].items(), key=lambda x: -x[1])))
    json.dump(keep, open(OUT / "bench75.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(*sys.argv[2:5])
    elif len(sys.argv) > 1 and sys.argv[1] == "--inv":
        inv_one(Path(sys.argv[2]), *sys.argv[3:6])
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["run", "inv", "summary"])
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args()
        {"run": cmd_run, "inv": cmd_inv, "summary": cmd_summary}[a.cmd](a)
