"""Drop any locked (TEST / NEWTEST) signal from the expert feature tables.

The event caches contain signals the function frame does not, including the hold-outs.
Nothing downstream joins them -- the frame has no hold-out row -- but the feature files
should not carry them either.
"""
import os
from pathlib import Path
import pandas as pd

# repository root: DC_REPO, else three levels up when run from research/code/trackA
REPO = Path(os.environ.get("DC_REPO") or Path(__file__).resolve().parents[3])
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
WORK = DCW / "trackA"

test = set(pd.read_csv(REPO / "data" / "splits" / "test_config.csv").DeviceId.str.lower())
newtest = set(pd.read_csv(DCW / "official" / "newtest_signals.csv").DeviceId.str.lower())
locked = test | newtest
print(f"{len(locked)} locked signals")

for f in sorted(WORK.glob("feat_expert_*_*.parquet")) + sorted(WORK.glob("pairs_raw_*.parquet")):
    if any(w in f.stem for w in ("_m5_", "_m10_", "_m30_a", "_h1_", "_h3_", "_h6_",
                                 "_h24_", "_full")):
        continue                      # single-window debug files
    d = pd.read_parquet(f)
    n0 = len(d)
    bad = d.DeviceId.str.lower().isin(locked)
    if bad.any():
        d[~bad].to_parquet(f, index=False)
    print(f"{f.name}: {n0:,} -> {n0 - int(bad.sum()):,} rows "
          f"({int(d[bad].DeviceId.nunique())} locked signals removed)")
