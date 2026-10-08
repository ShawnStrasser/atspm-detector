"""Note 90: swap the final function-network members into v4f -- the three full-data siba refits on the final labels v4o
(x86_sibafull4o{,_s1,_s2}: tcn53/models/*_full.pt, note-86 GPU queue, x74_sibafull4l recipe on func_rows_v4o) -- with
export84's export (export83 pair + head graphs) and its parity harness (package FuncNet vs torch forward69, bench
extracts, each member unfiltered / filtered + the 3-member average), run against v4f.

    python siba90.py check    -> are the three checkpoints + their runs/*.done.json there? (exit 1 if not)
    python siba90.py export   -> v4f weights/funcnet = the v4o members (v4l member files removed), manifest
    python siba90.py parity   -> %DC_WORK%/final_v3_work/v3fit90/siba_parity90.json
CPU only, 2 threads.  locked_v2 asserted absent (export84 harness).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final84"))
import export84 as X  # noqa: E402

import os  # noqa: E402
# orchestrator 2026-10-05: seed 2 converged poorly (agree90: outlier vs seeds 0 / 1) -> members 0 + 1 + 3
TAGS = os.environ.get("SIBA90_TAGS", "x86_sibafull4o,x86_sibafull4o_s1,x86_sibafull4o_s3").split(",")
X.PKG = X.E.W / "final_v3_candidate_v4f"
X.OUT = X.E.W / "final_v3_work" / "v3fit90"
X.TMP = X.E.W / "final_v3_work" / "f90" / "export_tmp"
X.MEMBERS = [(t, f"{t}_full.pt") for t in TAGS]
MODELS = X.E.W / "tcn53" / "models"
RUNS = X.E.W / "tcn53" / "runs"


def cmd_check():
    miss = [t for t in TAGS if not (MODELS / f"{t}_full.pt").exists() or not (RUNS / f"{t}_full.done.json").exists()]
    print("missing:" if miss else "all three v4o members present", miss, flush=True)
    sys.exit(1 if miss else 0)


def cmd_export():
    X.cmd_export()
    fd = X.PKG / "weights" / "funcnet"
    man = json.load(open(fd / "manifest.json"))
    man["trained"] = (f"note 86 FULL {' / '.join(TAGS)}: all non-locked training signals, FINAL labels v4o "
                      "(tcn53/func_rows_v4o.parquet), 43 epochs on the six-fold-picked schedule (s74/full/"
                      "lr_siba_full.json), research tcn53 / tcn69_func (snap74b, content partner tie-break)")
    for m in man["members"]:
        g = re.search(r"_s(\d+)$", m["tag"])
        m["seed"] = int(g.group(1)) if g else 0
    man["note"] = ("note 90 production net (final package v4f): 3 members trained on the final labels v4o (seeds 0 / 1 / 3; "
                   "seed 2 left out: converged to a higher loss, outlier in member agreement, agree90.json)")
    json.dump(man, open(fd / "manifest.json", "w"), indent=1)


def cmd_parity():
    X.cmd_parity()
    shutil.move(str(X.OUT / "siba_parity84.json"), str(X.OUT / "siba_parity90.json"))


if __name__ == "__main__":
    {"check": cmd_check, "export": cmd_export, "parity": cmd_parity}[sys.argv[1]]()
