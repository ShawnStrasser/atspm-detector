"""Note 152: cProfile of predict() on k h24_b signals (one call), health functions only (ms per signal, tottime and
cumtime).   python prof152.py <k> --src <package parent> [--batch N]"""
import argparse
import cProfile
import os
import pstats
import sys
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("--src", required=True)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--top", type=int, default=45)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import numpy as np
    import pandas as pd
    from atspm_detector import pipeline as PP
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    rng = np.random.default_rng(151)
    pick = [ids[i] for i in rng.permutation(len(ids))][:a.k]
    day = str(DCW / "s151" / "h24b_events.parquet")
    kw = dict(start="2026-09-28 00:00", end="2026-09-29 00:00", device_ids=pick)
    if a.batch:
        kw["batch_signals"] = a.batch
    PP.predict(str(Path(PP.__file__).parent / "data" / "sample_events.parquet"))
    pr = cProfile.Profile()
    pr.enable()
    PP.predict(day, **kw)
    pr.disable()
    st = pstats.Stats(pr)
    rows = []
    for (f, ln, fn), (cc, nc, tt, ct, callers) in st.stats.items():
        rows.append(dict(file=Path(f).name, func=fn, line=ln, ncalls=nc, tot=1000 * tt / a.k, cum=1000 * ct / a.k))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    H = R[R.file.str.startswith("health")].sort_values("cum", ascending=False)
    print(H.head(a.top).round(2).to_string(index=False))
    print("--- tottime, all files")
    print(R.sort_values("tot", ascending=False).head(a.top).round(2).to_string(index=False))
    for f in ("post_outputs", "_health_events", "assess"):
        x = R[R.func == f]
        if len(x):
            print(f, x.cum.round(1).tolist())


if __name__ == "__main__":
    main()
