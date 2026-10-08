"""User decision 2026-09-23: a zone the print labels / codes YR (YR, YR*, Yellow_Red) IS Yellow_Red whatever the
technology (video, thermal video, ...). "Only radar has Yellow_Red" applies only when there is no YR code. Loops stay
excluded (cab_record.finish refuses Yellow_Red on a loop).

    python fix_video_yr.py                          every fully visual_done batch (dry run)
    python fix_video_yr.py --apply
    python fix_video_yr.py --batches 26-35 --apply  re-run at the end on the late batches

A batch is processed only when ALL its signals are visual_done (a batch still being written is skipped whole,
with a message); inside a processed batch only visual_done records are touched.

Candidates: non-loop detectors that are not Yellow_Red and (a) have a YR subtype (yr_labelled_video,
yr_zone_on_video, ...), or (b) whose confidence_reason records the old conflict ("video cannot speed-filter",
"only radar has Yellow_Red", "video -> Count", ...) next to a YR code, or (c) were changed by sweep rule
b_yr_not_radar in sweep_changes.csv. Changed -> Yellow_Red / yellow_red (the existing subtype convention) /
next_to_count; flags label_disagrees dropped, needs_review dropped unless the reason carries another doubt
(channel map, stale, ...); the conflict text is removed from the reason. Confidence high when a zone number and the
YR code are written and timing phase = diagram phase and no doubt remains; else unchanged. Logged to
sweep_changes.csv with rule yr_label_user.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import time

from cab_common import CAB  # %DC_WORK%/cabinet

SIG = CAB / "signals"
LOG = CAB / "sweep_changes.csv"
FIELDS = ["DeviceName", "detector", "field", "old", "new", "rule"]
RULE = "yr_label_user"
NEW_REASON = "user 2026-09-23: the print codes it YR -> Yellow_Red whatever the technology"

YR_CODE = re.compile(r"(?<![A-Za-z])(YR\*?|Y/R|Yellow[_ ]Red)(?![A-Za-z])", re.I)
# the recorded conflict: YR code overridden because the zone was not radar
CONFLICT = re.compile(r"cannot speed-filter|cannot be Yellow_Red|only radar has Yellow_Red|video\s*->\s*Count|"
                      r"a video zone by the stop bar is Count", re.I)
CONFLICT_TXT = [
    r";?\s*user rule:\s*only radar has Yellow_Red; a video zone by the stop bar is Count \(print code YR\)",
    r",?\s*but video cannot speed-filter \(user rule\)\s*->\s*Count",
    r";?\s*user rule:\s*video cannot speed-filter\s*->\s*Count",
    r";?\s*video cannot speed-filter\s*->\s*Count",
    r";?\s*video cannot be Yellow_Red \(user rule\) so Count",
    r";?\s*video\s*->\s*Count",
    r"\s+but zone type is video \(V\)",
]
ZONE_NO = re.compile(r"\b\d{1,2}\s*-?\s*[A-Z]\s*/\s*\d{1,3}\b|\b[A-Z]\s*-\s*\d{1,2}\s*/\s*\d{1,3}\b|\bMT\s*#?\s*\d+|\bzone\s+\d")
DOUBT = re.compile(r"illustrative|uncertain|unclear|\?|maybe|may be|probabl|not sure|guess|doubt|stale|swapped|"
                   r"not tied|mismatch", re.I)
DOUBT_FLAGS = {"channel_map_mismatch", "suspect_config_or_health"}


def batches(spec: str | None) -> dict[str, list[str]]:
    out = {}
    for p in sorted((CAB / "batches").glob("batch_[0-9][0-9].txt")):
        n = int(p.stem.split("_")[1])
        if spec:
            want = set()
            for part in spec.split(","):
                a, _, b = part.partition("-")
                want |= set(range(int(a), int(b or a) + 1))
            if n not in want:
                continue
        out[p.stem] = [ln.split("\t")[0].strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


def clean_reason(t: str) -> str:
    for p in CONFLICT_TXT:
        t = re.sub(p, "", t, flags=re.I)
    return re.sub(r"\s*;\s*;", ";", t).strip(" ;,-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", help="e.g. 26-35 or 12,13; default = every batch")
    ap.add_argument("--apply", action="store_true", help="write records + log (default: dry run)")
    a = ap.parse_args()

    logged = set()
    if LOG.exists():
        with open(LOG, encoding="utf-8", newline="") as f:
            logged = {(x["DeviceName"], str(x["detector"])) for x in csv.DictReader(f) if x["rule"] == "b_yr_not_radar"}

    rows, changed, seen = [], [], []
    for bn, dns in batches(a.batches).items():
        recs = {}
        for dn in dns:
            fp = SIG / f"{dn}.json"
            recs[dn] = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
        st = collections.Counter(r.get("status") for r in recs.values())
        if set(st) != {"visual_done"}:
            print(f"skip {bn}: not finished {dict(st)}")
            continue
        for dn, r in recs.items():
            ch = {int(c["detector"]): c for c in r.get("channels", []) if c.get("detector") is not None}
            out = []
            for d in r.get("detectors", []):
                det = int(d["detector"])
                tech = d.get("technology") or "loop"
                f0 = d.get("function")
                if f0 == "Yellow_Red" or tech == "loop":
                    continue
                reason = d.get("confidence_reason") or ""
                c = ch.get(det, {})
                xl = str(c.get("xlsm") or "")
                sub = (d.get("subtype") or "").lower()
                by_sub = sub.startswith("yr_") or "_yr" in sub and f0 == "Count"
                by_txt = bool(CONFLICT.search(reason)) and (bool(YR_CODE.search(reason)) or bool(YR_CODE.search(xl)))
                by_log = (dn, str(det)) in logged
                if not (by_sub or by_txt or by_log):
                    continue
                if "presence" in sub:  # e.g. radar 'Y & R actuation' drawn as presence: a reading question, not this rule
                    seen.append(f"{dn} d{det} left ({sub})")
                    continue

                def setf(field, new):
                    old = d.get(field)
                    if old != new:
                        d[field] = new
                        out.append(dict(DeviceName=dn, detector=det, field=field,
                                        old=",".join(old) if isinstance(old, list) else old,
                                        new=",".join(new) if isinstance(new, list) else new, rule=RULE))
                conf0 = d.get("confidence")
                newr = clean_reason(reason)
                doubt = bool(DOUBT.search(newr)) or bool(DOUBT_FLAGS & set(d.get("flags") or []))
                drop = {"label_disagrees"} | (set() if doubt else {"needs_review"})
                setf("function", "Yellow_Red")
                setf("subtype", "yellow_red")
                setf("stopbar_position", "next_to_count")
                setf("flags", [x for x in (d.get("flags") or []) if x not in drop])
                setf("confidence_reason", (newr + "; " + NEW_REASON).strip("; "))
                ph_ok = d.get("phase_diagram") is not None and d.get("phase_diagram") in (
                    d.get("phase_timing"), c.get("phase_timing"), c.get("switch_phase"))
                code_ok = bool(YR_CODE.search(newr)) or bool(YR_CODE.search(xl))
                zone_ok = bool(ZONE_NO.search(reason)) or bool(c.get("zone_label")) or bool(re.match(r"ph[\d.]+\s+[A-Z]\s+YR", xl))
                if ph_ok and code_ok and zone_ok and not doubt and "needs_review" not in (d.get("flags") or []):
                    setf("confidence", "high")
                changed.append(f"{bn} {dn} d{det} {f0}->Yellow_Red conf {conf0}->{d.get('confidence')} flags={d.get('flags')}")
            if out:
                for d in r["detectors"]:  # cab_record.finish rules on the touched detectors
                    if d.get("function") == "Yellow_Red" and (d.get("technology") or "loop") == "loop":
                        raise SystemExit(f"{dn} NOT saved: YR on a loop d{d['detector']}")
                rows += out
                if a.apply:
                    sw = r.setdefault("sweep", {})
                    sw.setdefault("rules", [])
                    if RULE not in sw["rules"]:
                        sw["rules"].append(RULE)
                    sw["date"] = time.strftime("%Y-%m-%d %H:%M")
                    (SIG / f"{dn}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    if a.apply and rows:
        new = not LOG.exists()
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
    print(f"{'APPLIED' if a.apply else 'DRY RUN'}: {len(changed)} detectors, {len(rows)} field changes")
    for x in changed + seen:
        print("  ", x)


if __name__ == "__main__":
    main()
