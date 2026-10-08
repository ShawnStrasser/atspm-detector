"""Orchestrator ruling 2026-09-23: "radar over loops -> radar wins" applies only when the radar zone REPLACES the
loop's function. When every radar zone covering a LIVE loop (>= 5 ONs in the newest window) is itself Other (long
zone, advance-presence, dilemma / flasher, heavy-vehicle, 100-700 ft long presence), the loop gets its own role back
(Advance / Mid / Presence as read from the print, confidence medium) and the radar zone stays Other. Dead loops under
radar stay Other/superseded_by_radar.

    python fix_radar_over_loops.py --batches 1-33            dry run
    python fix_radar_over_loops.py --batches 1-33 --apply    write records + sweep_changes.csv (rule radar_over_loops_refined)

Candidates: subtype superseded_by_radar. The loop's own role = the function the radar_over_loops sweep replaced
(sweep_changes.csv), else the reader's wording ("geometric role = lane-by-lane advance", "advance loop", "mid",
"series", "at the stop bar"). The covering zones = detector numbers the reason names ("2E/49-50", "radar E 49/50",
"zone 57") looked up in the same record; if the reason names none, a wording that describes an Other-type zone
("long advance-presence zone", "100-700 ft", dilemma, flasher, heavy) counts. Anything unresolved is left and listed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import time

import pandas as pd

from cab_common import CAB, SIG_DIR

LOG = CAB / "sweep_changes.csv"
FIELDS = ["DeviceName", "detector", "field", "old", "new", "rule"]
RULE = "radar_over_loops_refined"
FLAG = "under_radar_other_zone"
ACTIVE_MIN = 5
PM = {"Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike"}
SUB = {"Advance": "advance", "Mid": "mid", "Presence": "stopbar_presence"}
ZONE_PATS = [
    re.compile(r"\b\d[A-Z]\s*(?:A\*|CO\*|P\*)?\s*/\s*(\d{1,2})(?:\s*[-,]\s*(\d{1,2}))?"),     # 2E/49-50, 2E A*/49
    re.compile(r"radar\s+[A-Z]\s+(?:A\*\s*/?\s*)?(\d{1,2})(?:\s*/\s*(\d{1,2}))?"),            # radar E 49/50
    re.compile(r"\bzones?\s+(\d{1,2})(?:\s*[-,]\s*(\d{1,2}))?\b(?!\s*ft)"),                    # zone 57, zones 57-60
    re.compile(r"\((\d{2})/(\d{2})\)"),                                                         # (49/50)
]
OTHER_TXT = re.compile(r"long advance-presence|advance-presence zone|long (radar )?zone|long presence|dilemma|flasher|"
                       r"heavy|\b\d{2,3}\s*-\s*\d{3}\s*ft", re.I)
SAME_TXT = re.compile(r"(?<!advance-)presence zone|count zone|advance zone|Count Adv|presence/count zones|A\* zone|radar zone \d[A-Z] A\*",
                      re.I)


ZONE_WORDS = re.compile(r"advance-presence|radar[^;,]*?zones?|count adv zone|a\* zone", re.I)  # the RADAR zone's words


def role(reason: str, swept: str | None) -> str | None:
    """The loop's own role as the reader recorded it (not the covering radar zone's words)."""
    if swept in ("Advance", "Mid", "Presence"):
        return swept
    r = ZONE_WORDS.sub(" ", reason).lower()
    if "geometric role = mid" in r or re.search(r"\bmid\b|series|tied pair across", r):
        return "Mid"
    if "advance" in r or "adv count" in r:
        return "Advance"
    if "stop bar" in r:
        return "Presence"
    return None


def zones(reason: str, own: int) -> set[int]:
    out = set()
    for p in ZONE_PATS:
        for m in p.finditer(reason):
            a = int(m.group(1))
            b = int(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else a
            if a <= b <= a + 4:
                out |= set(range(a, b + 1))
            else:
                out |= {a, b}
    return {z for z in out if 1 <= z <= 64 and z != own}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", required=True, help="e.g. 1-33")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    want = set()
    for part in a.batches.split(","):
        x, _, y = part.partition("-")
        want |= set(range(int(x), int(y or x) + 1))
    dns = []
    for p in sorted((CAB / "batches").glob("batch_[0-9][0-9].txt")):
        if int(p.stem.split("_")[1]) in want:
            dns += [ln.split("\t")[0].strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    swept = {}
    if LOG.exists():
        with open(LOG, encoding="utf-8", newline="") as f:
            for x in csv.DictReader(f):
                if x["rule"] == "radar_over_loops" and x["field"] == "function":
                    swept[(x["DeviceName"], str(x["detector"]))] = x["old"] or None
    act = pd.read_parquet(CAB / "activity.parquet")
    from cab_build import _newest

    rows, res = [], collections.Counter()
    listing = []
    for dn in dns:
        fp = SIG_DIR / f"{dn}.json"
        if not fp.exists():
            continue
        r = json.loads(fp.read_text(encoding="utf-8"))
        if r.get("status") != "visual_done":
            continue
        cand = [d for d in r.get("detectors", []) if d.get("subtype") == "superseded_by_radar"]
        if not cand:
            continue
        _, n_new, _ = _newest(act, dn)
        by_det = {int(d["detector"]): d for d in r["detectors"]}
        out = []
        for d in cand:
            det = int(d["detector"])
            if float(n_new.get(det, 0)) < ACTIVE_MIN:
                res["dead_kept"] += 1
                continue
            reason = d.get("confidence_reason") or ""
            rl = role(reason, swept.get((dn, str(det))))
            ch = next((c for c in r.get("channels", []) if int(c.get("detector") or -1) == det), {})
            rl_cfg = None
            if rl is None and ch.get("func5_label") in ("Advance", "Presence"):
                rl = rl_cfg = ch["func5_label"]  # role from the config label: restored at LOW confidence
            zs = zones(reason, det)
            zf = {z: (by_det[z].get("function") if z in by_det else None) for z in zs}
            known = {z: f for z, f in zf.items() if f is not None}
            ph = d.get("phase_diagram")
            sib = set()
            for s in cand:  # zones named for the other superseded loops on the same phase
                if s is not d and s.get("phase_diagram") == ph and ph is not None:
                    sib |= zones(s.get("confidence_reason") or "", int(s["detector"]))
            sibf = {z: by_det[z].get("function") for z in sib if z in by_det and by_det[z].get("function")}
            radar_ph = {int(x["detector"]): x.get("function") for x in r["detectors"] if ph is not None
                        and x.get("phase_diagram") == ph and (x.get("technology") or "") in ("radar", "radar_or_video")}
            same_role = {"Advance": {"Advance"}, "Mid": {"Advance", "Mid"}, "Presence": {"Presence", "Count"}}.get(rl, PM)
            if known:
                other, lvl = all(f == "Other" for f in known.values()), "named"
                why = "zones " + ",".join(f"{z}={f}" for z, f in sorted(known.items()))
            elif sibf:
                other, lvl = all(f == "Other" for f in sibf.values()), "sibling"
                why = "zones (named for the neighbouring loops) " + ",".join(f"{z}={f}" for z, f in sorted(sibf.items()))
            elif SAME_TXT.search(reason):
                other, lvl, why = False, "text", "text"
            elif OTHER_TXT.search(reason):
                other, lvl, why = True, "text", "zone described as an Other-type zone (long / advance-presence)"
            elif radar_ph and rl:
                other, lvl = not any(f in same_role for f in radar_ph.values()), "phase"
                why = f"radar zones on phase {ph}: {'none' if other else 'at least one'} with the loop's role {rl}"
            else:
                res["live_kept_zone_unresolved"] += 1
                listing.append(f"   kept {dn} d{det}: covering zone not resolved | {reason[:90]}")
                continue
            res[f"decided_by_{lvl}"] += 1
            if not other:
                res["live_kept_radar_replaces"] += 1
                listing.append(f"   kept {dn} d{det}: radar replaces ({lvl}: {why})")
                continue
            if rl is None:
                res["live_kept_role_unknown"] += 1
                listing.append(f"   kept {dn} d{det}: role unknown | {reason[:90]}")
                continue

            def setf(field, new):
                old = d.get(field)
                if old != new:
                    d[field] = new
                    out.append(dict(DeviceName=dn, detector=det, field=field,
                                    old=",".join(old) if isinstance(old, list) else old,
                                    new=",".join(new) if isinstance(new, list) else new, rule=RULE))
            setf("function", rl)
            setf("subtype", SUB[rl])
            setf("stopbar_position", "before" if rl == "Presence" else None)
            setf("flags", sorted(set(d.get("flags") or []) | {FLAG}))
            setf("confidence", "low" if rl_cfg else "medium")
            setf("confidence_reason", reason.rstrip("; ") + f"; orchestrator 2026-09-23: covering radar {why} -> "
                 f"the live loop keeps its own role {rl}" + (" (role from the config label, low)" if rl_cfg else " (medium)"))
            res["restored"] += 1
            res[f"restored_{rl}"] += 1
            listing.append(f"   RESTORED {dn} d{det} -> {rl} ({why})")
        if out:
            rows += out
            if a.apply:
                sw = r.setdefault("sweep", {})
                sw.setdefault("rules", [])
                if RULE not in sw["rules"]:
                    sw["rules"].append(RULE)
                sw["date"] = time.strftime("%Y-%m-%d %H:%M")
                fp.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    if a.apply and rows:
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writerows(rows)
    print(f"{'APPLIED' if a.apply else 'DRY RUN'}: {dict(res)}")
    print("\n".join(listing))


if __name__ == "__main__":
    main()
