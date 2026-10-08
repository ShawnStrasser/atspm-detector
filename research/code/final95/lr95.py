"""Note 95: learning-rate schedules for the 2026-only full-data refits, from the six-fold runs (note 86 rule: epochs =
median best epoch of the fold runs + 1; lr per epoch = median of the replayed ReduceLROnPlateau schedules).

    python final95/lr95.py phase TAG      -> %DC_WORK%/s95/lr_<TAG>_full.json  (tcn53 runs TAG_f0..5)
    python final95/lr95.py siba TAG,TAG_s1,TAG_s2  -> %DC_WORK%/s95/lr_siba95_full.json (tcn69_func runs, all seeds)
Run from research/code/neural (queue PY job) or research/code.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code") / "neural"))
import rpath  # noqa: F401,E402
import numpy as np  # noqa: E402
from common import DC_WORK  # noqa: E402

RUNS = DC_WORK / "tcn53" / "runs"
OUT = DC_WORK / "s95"
OUT.mkdir(parents=True, exist_ok=True)
kind, tags = sys.argv[1], sys.argv[2].split(",")
files = [RUNS / f"{t}_f{k}.done.json" for t in tags for k in range(6)]
assert all(f.exists() for f in files), [str(f) for f in files if not f.exists()]
best = [json.load(open(f))["best_epoch"] for f in files]
n = int(np.median(best)) + 1
if kind == "phase":
    import tcn53 as M
    lrs = M.lr_schedule_from_runs(files, n)
    dest = OUT / f"lr_{tags[0]}_full.json"
else:
    import tcn69_func as M
    lrs = M.lr_schedule_from_runs(files, n)
    dest = OUT / "lr_siba95_full.json"
json.dump(lrs, open(dest, "w"))
print(f"{kind}: best epochs {best} -> {n} epochs, lr {lrs[0]:.1e} -> {lrs[-1]:.1e} -> {dest}")
