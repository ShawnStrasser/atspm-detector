"""Note 119 (LOCKED EXAM, user-authorised 2026-10-07): run ONE model end to end over the 115 locked signals x 15 windows
(the OOF window anchors of windows_stg: m5 a-d, m30 a-d, h1 a-c, h3 a-b, h24 a-b), one predict() call per signal-window,
raw events in.  Nothing is tuned; models are used as packaged.
    python run119.py v7     # dc_work/final_v7_prod (detector_classifier 7.0.0)
    python run119.py beta   # repo model/ (final_v2)
-> %DC_WORK%/s119/pred/<model>/<DeviceId>.parquet (all windows, + phases for v7) and timing rows.  Resumable per signal.
"""
import os
import sys
import time
import warnings
from pathlib import Path

which = sys.argv[1]
NT = os.environ.get("X119_THREADS", "2")
os.environ["DC_TREE_THREADS"] = NT
os.environ["DC_NET_THREADS"] = NT
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = NT
import pandas as pd  # noqa: E402

W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
REPO = Path(__file__).resolve().parents[3]
OUT = W / "s119" / "pred" / which
OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = [("m5_a", "2026-09-21 07:45:00", 5), ("m5_b", "2026-09-19 12:20:00", 5), ("m5_c", "2026-09-19 22:10:00", 5),
           ("m5_d", "2026-09-18 17:05:00", 5),
           ("m30_a", "2026-09-21 07:30:00", 30), ("m30_b", "2026-09-19 12:00:00", 30),
           ("m30_c", "2026-09-19 21:30:00", 30), ("m30_d", "2026-09-18 17:00:00", 30),
           ("h1_a", "2026-09-20 17:00:00", 60), ("h1_b", "2026-09-20 02:00:00", 60), ("h1_c", "2026-09-21 09:00:00", 60),
           ("h3_a", "2026-09-21 06:00:00", 180), ("h3_b", "2026-09-19 14:00:00", 180),
           ("h24_a", "2026-09-19 00:00:00", 1440), ("h24_b", "2026-09-20 00:00:00", 1440)]

if which == "v7":
    sys.path.insert(0, str(W / "final_v7_prod" / "src"))
    import detector_classifier as M  # noqa: E402
    assert "final_v7_prod" in M.__file__, M.__file__
    fn = M.predict
else:
    sys.path.insert(0, str(REPO / "model"))
    import predict as M  # noqa: E402
    assert str(REPO / "model") in M.__file__, M.__file__
    fn = M.predict
warnings.filterwarnings("ignore")
ids = sorted(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
t_all = time.time()
first = True
for i, d in enumerate(ids):
    dest = OUT / f"{d}.parquet"
    if dest.exists():
        continue
    ev = (W / "s119" / "ev" / f"{d}.parquet").as_posix()
    outs, phs, tim = [], [], []
    for tag, start, mins in WINDOWS:
        end = str(pd.Timestamp(start) + pd.Timedelta(minutes=mins))
        t0 = time.perf_counter()
        if which == "v7":
            o, ph = fn(ev, start=start, end=end, threads=int(NT), memory="2GB", return_phases=True)
            ph = ph.assign(win=tag)
            phs.append(ph)
        else:
            o = fn(ev, start=start, end=end, threads=int(NT), memory="2GB")
        dt = time.perf_counter() - t0
        o = o.assign(win=tag, DeviceId=o.DeviceId.astype(str).str.lower())
        outs.append(o)
        tim.append(dict(DeviceId=d, win=tag, seconds=dt, n_det=len(o), first_call=first))
        first = False
    pd.concat(outs, ignore_index=True).to_parquet(dest, index=False)
    if phs:
        pd.concat(phs, ignore_index=True).to_parquet(OUT / f"{d}.phases.parquet", index=False)
    pd.DataFrame(tim).to_parquet(OUT / f"{d}.time.parquet", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] {which} {i + 1}/115 {d} {sum(t['seconds'] for t in tim):.1f}s", flush=True)
print(f"DONE {which} {time.time() - t_all:.0f}s", flush=True)
