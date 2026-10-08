"""Note 71: per-signal memory / time of final_v2 (model/) vs final_v3_candidate_v2, one signal per process.

Signals = the note-32 bench pool (`final_v3_work/extracts/bench_*.parquet`, non-locked): the one with the most
detector channels, the one with the most detector events, and 4 typical (ranks 5, 8, 11, 15 by channels).
Each (package, signal, length) runs in a fresh process, 4 threads, numpy backend where the package has one:
  rss_import   working set after `import predict` (python + numpy/pandas/duckdb/onnxruntime + package)
  peak_cold    process peak working set after the first predict() on the signal (models load inside it)
  wall_cold / wall_warm   first / second predict() on the same signal (warm = model files in OS cache)
A separate "floor" process predicts the bundled sample only (= model loading + overhead, tiny data).

    python bench71.py            -> %DC_WORK%/bench71/rows.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

W = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
EX = W / "final_v3_work" / "extracts"
OUT = W / "bench71"
REPO = Path(__file__).resolve().parents[3]
PKGS = {"final_v2": REPO / "model", "cand_v2": W / "final_v3_candidate_v2"}
LENS = ["m30", "h3", "h6", "h24"]


def one(pkg, events, out):
    os.environ["OMP_NUM_THREADS"] = "4"
    import psutil
    proc = psutil.Process()
    sys.path.insert(0, str(pkg))
    import predict as P
    if hasattr(P, "set_backend"):
        P.set_backend("numpy")
    pb = int(os.environ.get("BENCH71_PAIR_BATCH", "0"))     # optional: smaller GRU ONNX pair batch
    if pb:
        import functools
        import gru_blend as gb
        gb.phase_probs = functools.partial(gb.phase_probs, pair_batch=pb)
    r = {"rss_import_mb": proc.memory_info().rss / 2**20}
    src = Path(pkg) / "sample_events.parquet" if events == "floor" else events
    kw = {"min_actuations": 1} if events == "floor" else {}
    s = time.perf_counter()
    res = P.predict(src, threads=4, **kw)
    r["wall_cold"] = time.perf_counter() - s
    r["peak_cold_mb"] = proc.memory_info().peak_wset / 2**20
    if events != "floor":
        s = time.perf_counter()
        P.predict(src, threads=4)
        r["wall_warm"] = time.perf_counter() - s
        r["peak_after_2_mb"] = proc.memory_info().peak_wset / 2**20
    r["detectors"] = int(len(res))
    json.dump(r, open(out, "w"))


def main():
    import duckdb
    OUT.mkdir(exist_ok=True)
    import pandas as pd
    lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
    rk = duckdb.sql(f"""select DeviceId, count(distinct Parameter) nd, count(*) n from '{EX / "bench_h24.parquet"}'
                        where EventId in (81,82) and Parameter <= 64 group by 1 order by nd desc, n desc""").df()
    pick = {"busiest_ch": rk.DeviceId[0], "busiest_ev": rk.sort_values("n").DeviceId.iloc[-1]}
    for i in (4, 7, 10, 14):
        pick[f"typical_r{i + 1}"] = rk.DeviceId[i]
    assert not {d.lower() for d in pick.values()} & lk
    rows = []
    for tag, dev in pick.items():
        for L in LENS:
            f = OUT / f"ev_{tag}_{L}.parquet"
            if not f.exists():
                duckdb.sql(f"copy (select * from '{EX / f'bench_{L}.parquet'}' where DeviceId = '{dev}') to '{f}'")
    for name, pkg in PKGS.items():
        jobs = [("floor", "-", "floor")] + [(t, L, str(OUT / f"ev_{t}_{L}.parquet")) for L in LENS for t in pick]
        for tag, L, ev in jobs:
            o = OUT / f"r_{name}_{tag}_{L}.json"
            if not o.exists():
                subprocess.run([sys.executable, "-W", "ignore", __file__, "--one", str(pkg), ev, str(o)], check=True)
            r = json.load(open(o))
            r.update(pkg=name, signal=tag, length=L,
                     channels=int(rk.set_index("DeviceId").nd.get(pick.get(tag), 0)))
            rows.append(r)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(rows, open(OUT / "rows.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(*sys.argv[2:5])
    else:
        main()
