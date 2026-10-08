"""Uncoded-zone rules (user decisions 2026-09-24, consistent with the long-presence ruling of 2026-09-23), applied
to finished records like the other fix_* scripts (works in either store; run by cab_final's locked path).

    python fix_uncoded_zones.py --batches 1-10 --locked-labels-only            dry run
    python fix_uncoded_zones.py --batches 1-10 --locked-labels-only --apply    write records + sweep_changes.csv

A. uncoded_long_presence: a radar zone read as Other / long_zone that runs upstream FROM the stop bar with NO code and
   NO written extent (no distance_ft, no range / length > 20 ft on the print; an extent quoted from a sister print does
   not count) -> Presence / stopbar_presence / before, flag uncoded_long_zone, confidence as read. Only a WRITTEN extent
   > 20 ft (or a written code such as 'Phase') keeps a long zone Other.
B. stopbar_by_position: an uncoded radar / video stop-bar zone whose function contradicts its recorded position
   (Count drawn 0-20 ft before the stop bar, or Presence drawn on / past it) -> Presence before / Count on-past.
   A zone with a written code (P / CO / YR) is left as read.
C. undrawn_co_advance: an undrawn radar channel 49-52 (flag position_not_drawn) coded CO in the xlsm / config and read
   as stop-bar Count -> Advance / advance, confidence low (on every other print MT 49-52 are the advance radars, whose
   CO* bar sits far upstream); an undrawn A*/Adv zone on 49-52 of the same signal read as Advance -> Other /
   advance_presence low (the positional A* / CO* rule: CO* upstream = Advance, the A* zone on that lane = Other).
Every field change is logged to sweep_changes.csv (rules above).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import time

from cab_common import CAB, SIG_DIR
from fix_long_presence import OTHER_CODE, SETBACK, batch_dns, extents

LOG = CAB / "sweep_changes.csv"
FIELDS = ["DeviceName", "detector", "field", "old", "new", "rule"]
RADAR = {"radar", "radar_or_video"}
SISTER = re.compile(r"[^;()]*sister print[^;()]*", re.I)
CODE_P = re.compile(r"(?<![A-Za-z])(P|Pres|Presence|CO\*?|Count|YR\*?|Y/R)(?![A-Za-z])")
CO_TXT = re.compile(r"'\s*co\s*'|\bCO\*?\b", re.I)
ADV_TXT = re.compile(r"(?<![a-z])(adv|avd|advance)(?![a-z])|\bA\*", re.I)


def uncoded_long(d) -> bool:
    if d.get("function") != "Other" or d.get("subtype") != "long_zone":
        return False
    if (d.get("technology") or "") not in RADAR or d.get("distance_ft") not in (None, "", "nan"):
        return False
    reason = SISTER.sub(" ", d.get("confidence_reason") or "")
    if extents([reason], None) or SETBACK.search(reason) or OTHER_CODE.search(reason):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    dns = batch_dns(a.batches) or set()
    rows, res, listing = [], collections.Counter(), []
    for dn in sorted(dns):
        fp = SIG_DIR / f"{dn}.json"
        if not fp.exists():
            continue
        r = json.loads(fp.read_text(encoding="utf-8"))
        if r.get("status") != "visual_done":
            continue
        ch = {int(c["detector"]): c for c in r.get("channels", []) if c.get("detector") is not None}
        out = []

        def setf(d, field, new, rule):
            old = d.get(field)
            if old != new:
                d[field] = new
                out.append(dict(DeviceName=dn, detector=int(d["detector"]), field=field,
                                old=",".join(old) if isinstance(old, list) else old,
                                new=",".join(new) if isinstance(new, list) else new, rule=rule))

        co_done = False
        for d in r.get("detectors", []):
            det, reason = int(d["detector"]), d.get("confidence_reason") or ""
            tech, fl = d.get("technology") or "", list(d.get("flags") or [])
            # A
            if uncoded_long(d):
                rule = "uncoded_long_presence"
                setf(d, "function", "Presence", rule)
                setf(d, "subtype", "stopbar_presence", rule)
                setf(d, "stopbar_position", "before", rule)
                setf(d, "flags", sorted(set(fl) | {"uncoded_long_zone"}), rule)
                setf(d, "confidence_reason", reason.rstrip("; ") + "; user 2026-09-24: uncoded long radar zone from the "
                     "stop bar, no written extent -> Presence", rule)
                res[rule] += 1
                listing.append(f"   A {dn} d{det} -> Presence ({d.get('confidence')})")
                continue
            # B
            pos, f = d.get("stopbar_position"), d.get("function")
            if tech in RADAR | {"video"} and not CODE_P.search(reason + " " + str(ch.get(det, {}).get("xlsm") or "")):
                new = ("Presence" if f == "Count" and pos == "before" else
                       "Count" if f == "Presence" and pos in ("on", "past") else None)
                if new:
                    rule = "stopbar_by_position"
                    setf(d, "function", new, rule)
                    setf(d, "subtype", "stopbar_presence" if new == "Presence" else "stopbar_count", rule)
                    setf(d, "confidence_reason", reason.rstrip("; ") + f"; user 2026-09-24: uncoded stop-bar zone by "
                         f"position ({pos}) -> {new}", rule)
                    res[rule] += 1
                    listing.append(f"   B {dn} d{det} {f} -> {new}")
            # C (Count part)
            c = ch.get(det, {})
            code_txt = " ".join([reason, str(c.get("xlsm") or ""), str(c.get("description") or "")])
            if (49 <= det <= 52 and tech in RADAR and "position_not_drawn" in fl and d.get("function") == "Count"
                    and CO_TXT.search(code_txt)):
                rule = "undrawn_co_advance"
                setf(d, "function", "Advance", rule)
                setf(d, "subtype", "advance", rule)
                setf(d, "stopbar_position", None, rule)
                setf(d, "confidence", "low", rule)
                setf(d, "confidence_reason", reason.rstrip("; ") + "; user 2026-09-24: undrawn radar CO channel 49-52 "
                     "-> Advance (low)", rule)
                res[rule] += 1
                co_done = True
                listing.append(f"   C {dn} d{det} Count -> Advance (low)")
        if co_done:  # the A* partners on 49-52 of the same signal
            for d in r.get("detectors", []):
                det, reason = int(d["detector"]), d.get("confidence_reason") or ""
                c = ch.get(det, {})  # the zone's own code: quoted codes in the reason, xlsm, config description
                code_txt = " ".join(re.findall(r"'([^']{1,12})'", reason) + [str(c.get("xlsm") or ""),
                                                                              str(c.get("description") or "")])
                if (49 <= det <= 52 and (d.get("technology") or "") in RADAR and d.get("function") == "Advance"
                        and "position_not_drawn" in (d.get("flags") or []) and ADV_TXT.search(code_txt)
                        and not CO_TXT.search(code_txt)):
                    rule = "undrawn_co_advance"
                    setf(d, "function", "Other", rule)
                    setf(d, "subtype", "advance_presence", rule)
                    setf(d, "confidence", "low", rule)
                    setf(d, "confidence_reason", reason.rstrip("; ") + "; user 2026-09-24: the CO channels on 49-52 are "
                         "the Advance count bars, so this A*/Adv zone is Other/advance_presence (low)", rule)
                    res["undrawn_co_advance_Astar_other"] += 1
                    listing.append(f"   C {dn} d{det} Advance (A*) -> Other/advance_presence (low)")
        if out:
            rows += out
            if a.apply:
                sw = r.setdefault("sweep", {})
                sw.setdefault("rules", [])
                for x in sorted({o["rule"] for o in out}):
                    if x not in sw["rules"]:
                        sw["rules"].append(x)
                sw["date"] = time.strftime("%Y-%m-%d %H:%M")
                fp.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    if a.apply and rows:
        new = not LOG.exists()
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
    print(f"{'APPLIED' if a.apply else 'DRY RUN'}: {dict(res)}, {len(rows)} field changes")
    print("\n".join(listing))


if __name__ == "__main__":
    main()
