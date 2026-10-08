"""Note 80 sweep (v4l store only): an `explained_other` entry may only explain a channel that is NOT a possible
performance-measure detector (whole-intersection rule: such a channel becomes Other). Entries whose reason says the
channel is detection added / renumbered after the print, a probable new radar zone, a radar data / advance channel, or a
relanded loop are removed (logged); derived / fail-transfer / pass copies, bike, ped and classification channels stay.

    python v4l_expl_sweep.py [--apply]      -> %DC_WORK%/lab80/expl_removed.csv (+ <store>/sweep_changes.csv)
"""
from __future__ import annotations

import csv
import json
import re
import sys

import pandas as pd

from cab_common import CAB, DC_WORK, SIG_DIR, STORE

assert STORE, "run with DC_CAB_STORE=cabinet_v4l (never on the original stores)"
POSSIBLE_PM = re.compile(r"added after|renumbered|upgrade|new radar|probably a new|likely (a|the|one|VD)|plausibly one of|"
                         r"radar data channel|radar advance|relanded|replacement|re-?landed|new channel since|template channel", re.I)
KEEP = re.compile(r"bike|ped|truck|\bcars\b|classification|derived|dummy|pass|fail|extend|logic|copy|mirror|call zone|phase-call", re.I)


def main(apply: bool):
    rows = []
    for f in sorted(SIG_DIR.glob("*.json")):
        if f.name.endswith(".extract.json"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        on = {int(d["detector"]) for d in r.get("detectors", [])}
        ex = r.get("explained_other") or {}
        drop = [k for k, v in ex.items() if int(k) not in on and POSSIBLE_PM.search(str(v)) and not KEEP.search(str(v))]
        for k in drop:
            rows.append(dict(DeviceName=r["DeviceName"], detector=k, field="explained_other", old=str(ex[k])[:300], new="",
                             rule="v4l_expl_sweep"))
        if drop and apply:
            r["explained_other"] = {k: v for k, v in ex.items() if k not in drop}
            f.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    out = pd.DataFrame(rows)
    (DC_WORK / "lab80").mkdir(exist_ok=True)
    out.to_csv(DC_WORK / "lab80" / "expl_removed.csv", index=False)
    if apply and rows:
        with open(CAB / "sweep_changes.csv", "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, ["DeviceName", "detector", "field", "old", "new", "rule"]).writerows(rows)
    print(f"{'APPLIED' if apply else 'DRY RUN'}: {len(rows)} entries at {out.DeviceName.nunique() if len(out) else 0} signals")
    if len(out):
        print(out.groupby("DeviceName").size().to_string())


if __name__ == "__main__":
    main("--apply" in sys.argv)
