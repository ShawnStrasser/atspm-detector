"""User decision 2026-09-23: a radar/video zone the print's table/label calls Presence ("P", "Pres", "Presence") IS
Presence even when drawn long (drawings are not to scale). Only a zone whose length is WRITTEN as > 20 ft
("0-75", "20-75", "0-180 ft", distance_ft > 20) stays Other.

    python fix_long_presence.py                       all visual_done records except batches 20-23 (dry run)
    python fix_long_presence.py --batches 20-35 --apply
    python fix_long_presence.py --batches 13,14,24-26 --apply
    python fix_long_presence.py --batches 01-32 --skip "" --config-desc --apply

--config-desc (orchestrator decision 2026-09-23): the config channel description saying Presence / Pres (or a
"0-20 P" style code) also counts as a Presence code, whether it is quoted in the reason or only in the channel
record. A detector changed on that evidence alone keeps its confidence (never raised to high).

Candidates: function Other with a long-zone subtype (long_zone, long_stopbar_zone, long_presence_*, radar_long_zone,
presence_20_75, ...; advance / set-back / rail zones excluded). Presence evidence = the confidence_reason (with the
quoted "radar Presence is 0-20 ft" rule text and config-description quotes removed) or the xlsm table code; a
detector that was Presence before the earlier d_long_zone sweep but has no code in the record is left, as ambiguous. Written extent = distance_ft, or a range / length > 20 ft in the reason or xlsm code.
Changed -> Presence / stopbar_presence / before, flag drawn_long_no_distance, confidence high when a zone number
and the code are written and timing phase = diagram phase (else unchanged). Every field change is logged to
sweep_changes.csv with rule long_presence_user. Presence records already flagged drawn_long_no_distance are left.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import time

from cab_common import CAB  # %DC_WORK%/cabinet

SIG = CAB / "signals"
LOG = CAB / "sweep_changes.csv"
FIELDS = ["DeviceName", "detector", "field", "old", "new", "rule"]
RULE = "long_presence_user"
FLAG = "drawn_long_no_distance"
NEW_REASON_DESC = ("orchestrator 2026-09-23: config description codes it Presence, drawn long but no written extent "
                   "-> Presence (confidence not raised)")
NEW_REASON = "user 2026-09-23: table calls it Presence, drawn long but no written extent -> Presence (drawings not to scale)"

# the quoted rule text is not a stated extent and not a table code
RULE_TXT = [
    r"user rule:?\s*(radar|video|radar/video)\s+[Pp]resence\s+(is|must be)[^;]*",
    r"(radar|video|radar/video)\s+[Pp]resence\s+(is|must be)\s+(exactly\s+)?0-20\s*ft[^;]*",
    r"a longer zone\s*=\s*Other/long_zone",
    r"not a 0-20 ft presence",
    r"->\s*Other(/long_zone)?",
    r"user rule",
    r"presence-like",  # behaviour, not a code
]
PRES = re.compile(r"(?<![A-Za-z])(Presence|PRESENCE|presence|Pres|PRES|pres|P)(?![A-Za-z/])")
OTHER_CODE = re.compile(r"(?<![A-Za-z])(CO|Count|COUNT|YR|Yellow|Adv|Advance|ADV|advance|A\*|Phase|PHASE|Bike|Mid|ETA|Ext)(?![a-z])")
# set back behind other stop-bar zones = not a stop-bar zone, whatever the code
SETBACK = re.compile(r"behind the stop-bar|upstream of the stop-bar|set ?back|behind the (stop-bar |0-20 )?presence", re.I)
RANGE = re.compile(r"(?<![A-Za-z0-9/.#-])(\d{1,3})\s*'?\s*(?:-|–|to)\s*(\d{1,4})\s*(?:'|ft|feet|\b)")
LEN = re.compile(r"(?<![A-Za-z0-9/.#-])(\d{2,4})\s*(?:'|ft\b|feet\b)")
ZONE_NO = re.compile(r"\b\d{1,2}\s*-?\s*[A-Z]\s*/\s*\d{1,3}\b|\b[A-Z]\s*-\s*\d{1,2}\s*/\s*\d{1,3}\b|\bMT\s*#?\s*\d+|\bzone\s+\d")
UNSURE = re.compile(r"illustrative|representation only|uncertain|unclear|\?|maybe|may be|probabl|not sure|guess|predate|"
                    r"tied? by (phase|description)|\d\s+or\s+\d|swapped|stale", re.I)
DROP_FLAGS = {"label_disagrees"}  # the disagreement was table-P vs drawn-long, which this rule resolves


def is_candidate_subtype(s: str) -> bool:
    s = (s or "").lower()
    if any(x in s for x in ("advance", "setback", "rail", "rr_", "upstream", "departure", "phase")):
        return "long_phase" in s  # 'Phase' call zones are still looked at (they are kept unless coded Presence)
    return "long" in s or "presence" in s


PRES_DESC = re.compile(r"(?i)(?<![a-z])(presence|pres)(?![a-z])|(?<![a-z0-9])\d{1,3}\s*-\s*20\s*'?\s*P(?![a-z0-9])")
OTHER_DESC = re.compile(r"(?i)(?<![a-z])(count|cnt|yr|y/r|yellow|adv|advance|bike|mid|queue|eta|ext|extension|dummy)(?![a-z])")
CONFIG_TXT = re.compile(r"(desc\w*|config\w*|tie[ds]? by description|now)\s*[^;,()]*", re.I)  # config export, not the print


def strip_rule(t: str) -> str:
    for p in RULE_TXT:
        t = re.sub(p, " ", t)
    return t


def extents(texts, dist) -> list[str]:
    out = []
    if dist not in (None, "", "nan"):
        nums = [int(x) for x in re.findall(r"\d+", str(dist))]
        if nums and max(nums) > 20:
            out.append(f"distance_ft={dist}")
    for t in texts:
        if not t:
            continue
        for lo, hi in RANGE.findall(t):
            lo, hi = int(lo), int(hi)
            if lo < hi and 20 < hi <= 1500:
                out.append(f"{lo}-{hi}")
        for n in LEN.findall(t):
            if 20 < int(n) <= 1500 and not any(e.endswith(f"-{n}") for e in out):
                out.append(f"{n} ft")
    return sorted(set(out))


def batch_dns(spec: str | None) -> set[str] | None:
    if not spec:
        return None
    nums = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        nums |= set(range(int(a), int(b or a) + 1))
    dns = set()
    for n in nums:
        p = CAB / "batches" / f"batch_{n:02d}.txt"
        if p.exists():
            dns |= {ln.split("\t")[0].strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    return dns


def validate(dn, r, touched):
    """cab_record.finish's rules; confidence / stop-bar checked on the touched detectors only (old records keep theirs)."""
    errs = []
    for d in r["detectors"]:
        f, t = d.get("function"), d.get("technology") or "loop"
        if f == "Count" and t == "loop":
            errs.append(f"d{d['detector']} loop Count")
        if f == "Yellow_Red" and t == "loop":  # user 2026-09-23: a print-coded YR zone is YR on any non-loop
            errs.append(f"d{d['detector']} YR on {t}")
        if int(d["detector"]) not in touched:
            continue
        if f in ("Presence", "Count", "Yellow_Red") and not d.get("stopbar_position"):
            errs.append(f"d{d['detector']} {f} without stopbar_position")
        if d.get("confidence") not in ("high", "medium", "low"):
            errs.append(f"d{d['detector']} confidence {d.get('confidence')!r}")
    if errs:
        raise SystemExit(f"{dn} NOT saved: {errs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", help="e.g. 20-35 or 13,14,24-26; default = every visual_done record")
    ap.add_argument("--skip", default="20-23", help="batches to leave alone (default 20-23; '' for none)")
    ap.add_argument("--apply", action="store_true", help="write records + log (default: dry run)")
    ap.add_argument("--config-desc", action="store_true", help="config description Presence/Pres counts as a Presence code")
    a = ap.parse_args()
    only, skip = batch_dns(a.batches), batch_dns(a.skip) or set()

    # earlier sweep: detectors whose old function was Presence before d_long_zone
    old_pres = set()
    if LOG.exists():
        with open(LOG, encoding="utf-8", newline="") as f:
            for x in csv.DictReader(f):
                if x["rule"] == "d_long_zone" and x["field"] == "function" and x["old"] == "Presence":
                    old_pres.add((x["DeviceName"], str(x["detector"])))

    changed, kept, notpres, ambig, rows = [], {}, [], [], []
    for fp in sorted(SIG.glob("*.json")):
        dn = fp.stem
        if (only is not None and dn not in only) or dn in skip:
            continue
        r = json.loads(fp.read_text(encoding="utf-8"))
        if r.get("status") != "visual_done":
            continue
        ch = {int(c["detector"]): c for c in r.get("channels", []) if c.get("detector") is not None}
        out = []
        for d in r.get("detectors", []):
            if d.get("function") != "Other" or not is_candidate_subtype(d.get("subtype")):
                continue
            det = int(d["detector"])
            c = ch.get(det, {})
            reason = d.get("confidence_reason") or ""
            rs = CONFIG_TXT.sub(" ", strip_rule(reason))
            xl = str(c.get("xlsm") or "")
            xl_code = re.sub(r"^ph[\d.]+\s+\S+\s*", "", xl)  # 'ph3.0 D P' -> 'P'
            ext = extents([rs, xl_code] + [f"{lo}-{hi} ft" for lo, hi in re.findall(r"(\d+)_(\d+)", d.get("subtype") or "")],
                          d.get("distance_ft"))
            ev_p = bool(PRES.search(rs)) or bool(PRES.search(xl_code))
            ev_o = bool(OTHER_CODE.search(rs)) or bool(OTHER_CODE.search(xl_code))
            by_desc = False
            if a.config_desc and not ev_p:
                cfg = " ".join([str(c.get("description") or "")] + [m.group(0) for m in CONFIG_TXT.finditer(strip_rule(reason))])
                if PRES_DESC.search(cfg) and not OTHER_DESC.search(str(c.get("description") or "")):
                    ev_p = by_desc = True
            tech = d.get("technology") or "loop"
            key = dict(DeviceName=dn, detector=det, subtype=d.get("subtype"), tech=tech, reason=reason[:150], xlsm=xl,
                       desc=str(c.get("description") or ""), by_desc=by_desc)
            if ext:
                kept.setdefault(ext[0], []).append(key)
                continue
            if tech not in ("radar", "video", "radar_or_video"):
                ambig.append({**key, "why": f"technology {tech}"})
                continue
            if SETBACK.search(reason):
                ambig.append({**key, "why": "drawn set back behind other stop-bar zones"})
                continue
            if ev_p and ev_o:
                ambig.append({**key, "why": "Presence and another code both mentioned"})
                continue
            if not ev_p and (dn, str(det)) in old_pres:
                ambig.append({**key, "why": "labelled Presence before the earlier sweep, but no table code in the record"})
                continue
            if not ev_p:
                (notpres if ev_o else ambig).append({**key, "why": "code not Presence" if ev_o else "no Presence code in reason/xlsm"})
                continue
            # --- change
            conf0 = d.get("confidence")
            def setf(field, new):
                old = d.get(field)
                if old != new:
                    d[field] = new
                    out.append(dict(DeviceName=dn, detector=det, field=field,
                                    old=",".join(old) if isinstance(old, list) else old,
                                    new=",".join(new) if isinstance(new, list) else new, rule=RULE))
            setf("function", "Presence")
            setf("subtype", "stopbar_presence")
            setf("stopbar_position", "before")
            fl = [f for f in (d.get("flags") or []) if f not in DROP_FLAGS]
            if FLAG not in fl:
                fl.append(FLAG)
            setf("flags", fl)
            newr = re.sub(r"\s*;\s*;", ";", strip_rule(reason)).strip(" ;,-")
            setf("confidence_reason", (newr + "; " + (NEW_REASON_DESC if by_desc else NEW_REASON)).strip("; "))
            ph_ok = d.get("phase_diagram") is not None and d.get("phase_diagram") in (d.get("phase_timing"), c.get("phase_timing"), c.get("switch_phase"))
            zone_ok = bool(ZONE_NO.search(reason)) or bool(c.get("zone_label"))
            if ph_ok and zone_ok and not by_desc and not UNSURE.search(reason) and "needs_review" not in fl:
                setf("confidence", "high")
            changed.append({**key, "conf": f"{conf0}->{d.get('confidence')}"})
        if out:
            validate(dn, r, {int(o['detector']) for o in out})
            rows += out
            if a.apply:
                sw = r.setdefault("sweep", {})
                sw.setdefault("rules", [])
                if RULE not in sw["rules"]:
                    sw["rules"].append(RULE)
                sw["date"] = time.strftime("%Y-%m-%d %H:%M")
                fp.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    if a.apply and rows:
        new = not LOG.exists()
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)

    print(f"{'APPLIED' if a.apply else 'DRY RUN'}: changed {len(changed)} detectors at "
          f"{len({x['DeviceName'] for x in changed})} signals, {len(rows)} field changes")
    import collections
    print("   confidence:", dict(collections.Counter(x["conf"] for x in changed)))
    print(f"   via config description only: {sum(x['by_desc'] for x in changed)}")
    for x in [x for x in changed if x["by_desc"]][:15]:
        print(f"      {x['DeviceName']} d{x['detector']} desc={x['desc'][:30]!r} {x['reason'][:80]}")
    print(f"kept Other, stated extent > 20 ft: {sum(map(len, kept.values()))}")
    for e, v in sorted(kept.items(), key=lambda kv: -len(kv[1])):
        print(f"   {e:>16}: {len(v)}")
    print(f"kept Other, code not Presence (Phase/Count/...): {len(notpres)}")
    print(f"ambiguous (not changed): {len(ambig)}", dict(collections.Counter(x["why"] for x in ambig)))
    for x in ambig[:20]:
        print(f"   {x['DeviceName']} d{x['detector']} [{x['subtype']}] {x['why']}: {x['reason'][:110]}")


if __name__ == "__main__":
    main()
