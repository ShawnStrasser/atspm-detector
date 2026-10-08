"""Summarise diag76 runs: per run, the stages / columns that move under renumbering, and the max probability moves."""
import os
import json
import sys
from pathlib import Path

D = Path(os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v3_work\diag76"))
tag = sys.argv[1] if len(sys.argv) > 1 else "fix"
agg = {}
for f in sorted(D.glob(f"{tag}_*.json")):
    r = json.load(open(f))
    moved = {}
    for k, v in r.items():
        if isinstance(v, dict) and "cols" in v:
            if v["unmatched"]:
                moved[k + ":unmatched"] = v["unmatched"]
            for c, d in v["cols"].items():
                moved[f"{k}:{c}"] = (d["n_diff"], d.get("max_abs"))
    fp = r.get("function_probs", {})
    mx = {k.split("|")[1]: v for k, v in fp.items()}
    print(f"{f.stem:45s} chan={r['chan']} moved={len(moved)} fprob={ {k: f'{v:.1e}' for k, v in mx.items()} }")
    for k, v in moved.items():
        print("     ", k, v)
        agg.setdefault(k.split(":")[0] + ":" + k.split(":")[-1], 0)
        agg[k.split(":")[0] + ":" + k.split(":")[-1]] += 1
print(agg)
