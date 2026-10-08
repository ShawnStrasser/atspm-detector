"""Summarise diag77 runs of one tag: moved stages / columns per run, max function-probability move, totals."""
import json
import os
import sys
from pathlib import Path

D = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work") / "final_v3_work" / "f77" / "diag"
tag = sys.argv[1]
agg, n, clean = {}, 0, 0
for f in sorted(D.glob(f"{tag}_*.json")):
    r = json.load(open(f))
    n += 1
    moved = {}
    for k, v in r.items():
        if isinstance(v, dict) and "cols" in v:
            if v["unmatched"]:
                moved[k + ":unmatched"] = v["unmatched"]
            for c, d in v["cols"].items():
                moved[f"{k.split('|')[0]}:{c}"] = (d["n_diff"], d.get("max_abs"))
    fp = max(r.get("function_probs", {"x": 0}).values())
    clean += not moved
    if moved or "-v" in sys.argv:
        print(f"{f.stem:52s} moved={len(moved)} max fprob {fp:.1e}")
        for k, v in moved.items():
            print("     ", k, v)
            agg[k] = agg.get(k, 0) + 1
print(f"{tag}: {n} runs, {clean} with nothing moved (tol 1e-6)")
print(agg)
