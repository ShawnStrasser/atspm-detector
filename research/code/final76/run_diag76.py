import os
import subprocess, sys, json
from pathlib import Path
PY = sys.executable
D = str(Path(__file__).resolve().parent / "diag76.py")
OUT = Path(os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v3_work\diag76")); OUT.mkdir(parents=True, exist_ok=True)
pkg = sys.argv[1]; tag = sys.argv[2]
jobs = []
for sig in ("typical_r8", "typical_r11", "typical_r5", "typical_r15", "busiest_ch", "busiest_ev"):
    for L in ("m30", "h3", "h24"):
        for seed in (75, 76):
            jobs.append((sig, L, seed, []))
    jobs.append((sig, "m30", 77, ["--minutes", "10"]))
for sig in ("typical_r8", "typical_r11", "typical_r5", "typical_r15", "busiest_ch", "busiest_ev"):
    for L in ("m30", "h3", "h24"):
        jobs.append((sig, L, 79, ["--chanrev", "--nophase"]))
for sig, L, seed, extra in jobs:
    o = OUT / f"{tag}_{sig}_{L}_{seed}{'_'.join([''] + [e.strip('-') for e in extra])}.json"
    if o.exists():
        continue
    r = subprocess.run([PY, "-W", "ignore", D, pkg, sig, L, str(o), "--seed", str(seed)] + extra, capture_output=True, text=True)
    if r.returncode:
        print("FAIL", o.name, r.stderr[-800:], flush=True)
    else:
        print(o.name, flush=True)
