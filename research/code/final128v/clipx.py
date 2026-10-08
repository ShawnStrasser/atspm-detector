"""extra clip edge cases, prod vs next.  python clipx.py <src> <tag>"""
import os, sys, pickle, warnings, time
for k in ("OMP_NUM_THREADS", "DC_TREE_THREADS", "DC_NET_THREADS"): os.environ[k] = "4"
sys.path.insert(0, sys.argv[1]); warnings.simplefilter("ignore")
import pandas as pd
from detector_classifier import predict
E = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"s128v\ev")
f3 = fr"{E}\x05_h3.parquet"; d3 = pd.read_parquet(f3)
g3 = pd.read_parquet(fr"{E}\x02_h3.parquet")
res = {}
def run(name, fn):
    t = time.perf_counter()
    try:
        r = fn()
    except Exception as e:
        r = f"EXC {type(e).__name__}: {e}"
    res[name] = r; print(name, f"{time.perf_counter()-t:.2f}s", r if isinstance(r, str) else (len(r) if not isinstance(r, tuple) else [len(x) for x in r]), flush=True)
P = lambda x, **k: predict(x, return_phases=True, **k)
run("window_after_data", lambda: P(f3, start="2030-01-01", end="2030-01-02"))
run("window_before_data", lambda: P(f3, start="2000-01-01", end="2000-01-02"))
run("start_after_end", lambda: P(f3, start="2026-09-20 16:00", end="2026-09-20 13:00"))
run("start_far_end_mid", lambda: P(f3, start="2000-01-01", end="2026-09-20 14:00"))
run("start_mid_end_far", lambda: P(f3, start="2026-09-20 14:00", end="2030-01-01"))
run("start_29min_early", lambda: P(f3, start="2026-09-20 12:31", end="2026-09-20 16:00"))
run("start_31min_early", lambda: P(f3, start="2026-09-20 12:29", end="2026-09-20 16:00"))
run("end_29min_late", lambda: P(f3, start="2026-09-20 13:00", end="2026-09-20 16:29"))
run("end_31min_late", lambda: P(f3, start="2026-09-20 13:00", end="2026-09-20 16:31"))
run("tz_far_start", lambda: P(d3.assign(Timestamp=d3.Timestamp.dt.tz_localize("America/Indiana/Indianapolis")), start=pd.Timestamp("2025-01-01", tz="UTC")))
# two signals: x05 3 h (13-16) + x02 cut to 14:30-16:00, start 13:00 exact for x05, 1.5 h early for x02
g = g3[g3.Timestamp >= "2026-09-20 14:30"]
run("two_sig_one_late", lambda: P(pd.concat([d3, g]), start="2026-09-20 13:00", end="2026-09-20 16:00"))
run("two_sig_one_late_chunked", lambda: P(pd.concat([d3, g]), start="2026-09-20 13:00", end="2026-09-20 16:00", chunk_signals=1))
run("alone_x05", lambda: P(d3, start="2026-09-20 13:00", end="2026-09-20 16:00"))
run("alone_x02_late_nostart", lambda: P(g, end="2026-09-20 16:00"))
run("alone_x02_late_withstart", lambda: P(g, start="2026-09-20 13:00", end="2026-09-20 16:00"))
# outage inside the window: x05 3 h with the first 45 min removed, start given at 13:00
h = d3[d3.Timestamp >= "2026-09-20 13:45"]
run("outage45_start_given", lambda: P(h, start="2026-09-20 13:00", end="2026-09-20 16:00"))
run("outage45_nostart", lambda: P(h, end="2026-09-20 16:00"))
# 9.998 min sample (status text)
m = pd.read_parquet(fr"{E}\x05_m30.parquet"); m = m[m.Timestamp < pd.Timestamp("2026-09-20 16:40")]
run("m10_short", lambda: P(m))
pickle.dump(res, open(os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), fr"s128v\far\clipx_{sys.argv[2]}.pkl"), "wb"))
