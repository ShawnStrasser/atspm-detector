"""Note 77: invariance battery of a candidate package (diag77.py): 6 bench signals x 30 min / 3 h / 24 h under a channel
reversal, a channel shift and both (phases kept), plus 2 phase renumberings and the 10-minute short path.

    python run_diag77.py <package dir> <tag>     -> %DC_WORK%/final_v3_work/f77/diag/<tag>_*.json
"""
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable
D = str(Path(__file__).resolve().parent / "diag77.py")
OUT = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work") / "final_v3_work" / "f77" / "diag"
OUT.mkdir(parents=True, exist_ok=True)
pkg, tag = sys.argv[1], sys.argv[2]
SIGS = ("typical_r8", "typical_r11", "typical_r5", "typical_r15", "busiest_ch", "busiest_ev")
jobs = []
for sig in SIGS:
    for L in ("m30", "h3", "h24"):
        for mode in ("--chanrev", "--chanshift", "--chanrevshift"):
            jobs.append((sig, L, 79, [mode, "--nophase"]))
        for seed in (75, 76):
            jobs.append((sig, L, seed, []))
    jobs.append((sig, "m30", 77, ["--minutes", "10"]))
    jobs.append((sig, "m30", 78, ["--minutes", "10", "--chanrev", "--nophase"]))
for sig, L, seed, extra in jobs:
    o = OUT / f"{tag}_{sig}_{L}_{seed}{'_'.join([''] + [e.strip('-') for e in extra])}.json"
    if o.exists():
        continue
    r = subprocess.run([PY, "-W", "ignore", D, pkg, sig, L, str(o), "--seed", str(seed)] + extra, capture_output=True,
                       text=True)
    print(("FAIL " + o.name + " " + r.stderr[-800:]) if r.returncode else o.name, flush=True)
