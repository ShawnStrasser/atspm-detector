"""Visual-pass helper: show a signal's scripted record, crop zones, validate + save the reader's record.

    python cab_record.py show <DN>                         worksheet: files, input file, zones, one line per channel
    python cab_record.py zoom <DN> <page> fx0 fy0 fx1 fy1 [tag]   extra zoom (fractions of render/<DN>/p<page>.png)
    python cab_record.py crop <DN> <page> fx0 fy0 fx1 fy1 <det>   evidence crop -> cabinet/crops/<DN>_d<det>.png
    python cab_record.py finish <DN> <visual.json>         merge the visual fields, validate, save signals/<DN>.json

visual.json (only what the reader decided; scripted fields are kept from the skeleton):
  {"phases": {"2": {"n_lanes": 2, "lanes": "T,TR", "note": ""}, ...},
   "detectors": [{"detector": 4, "phase_diagram": 2, "function": "Presence", "subtype": "stopbar_presence",
                  "stopbar_position": "before", "lane_index": 1, "lanes_spanned": 1, "lane_type": "T",
                  "technology": "loop", "flags": [], "confidence": "high", "confidence_reason": ""}, ...],
   "explained_other": {"33": "why an active channel not on the print is expected"},
   "cabinet_type": "332", "diagram_page": 5, "minutes_visual": 1.5, "note": "",
   "signal_flags": ["unusual_layout"], "signal_flags_reason": "live radar over live loops on the same lanes"}
signal_flags (optional): unusual_layout = rare custom set-up the model cannot read from behaviour (live radar over
live loops, several intersections on one controller, departure / peer-to-peer dominated, heavily rewired print);
such signals are excluded from training and scored separately. Radar zones over loops: label the radar, the loops
underneath are Other / subtype superseded_by_radar.
A detector id not in the skeleton is added (e.g. a zone label the script missed). function null =
left unclassified (give confidence_reason). Count / Yellow_Red need their crop first (label bubble + stop bar
visible; `<DN>_d<det>_approach.png` is accepted when the detector is flagged position_not_drawn).
User rules 2026-09-23 enforced here: a loop is never Count; Yellow_Red never on a loop (radar, or video/any
non-loop zone the print codes YR; without a YR code only radar gets Yellow_Red); flag
series_loop_pending is retired (function Mid, subtype series_loop). Old records are read unchanged.
For a pilot re-read (batch_01) pass only the re-read stop-bar detectors; the rest is kept.
"""
from __future__ import annotations

import json
import sys
import time

from cab_common import CAB, SIG_DIR

CROP_DIR = CAB / "crops"
FUNCS = {"Advance", "Presence", "Count", "Yellow_Red", "Bike", "Mid", "Other", None}
LANES = {"L", "T", "R", "LT", "TR", "LR", "LTR", "bike", "departure", "other", None}
CONF = {"high", "medium", "low"}
POS = {"before", "on", "past", "next_to_count", "upstream", "downstream", None}
TECH = {"loop", "video", "radar", "radar_or_video", None}
SIGNAL_FLAGS = {"unusual_layout"}
VISUAL = ("phase_diagram", "function", "subtype", "stopbar_position", "lane_index", "lanes_spanned", "lane_type",
          "technology", "flags", "confidence", "confidence_reason", "loops", "distance_ft", "crop")


def load(dn: str) -> dict:
    return json.loads((SIG_DIR / f"{dn}.json").read_text(encoding="utf-8"))


def show(dn: str):
    r = load(dn)
    s = r.get("scripted", {})
    print(f"{dn}  status={r.get('status')}  scripted={s.get('status')}  class={s.get('batch_class')}  cabinet={s.get('cabinet_type')}"
          f"  pages={s.get('n_pages')}  diagram_pages={s.get('diagram_pages')}  pdf={s.get('pdf')}  xlsm={s.get('xlsm')}")
    for why in s.get("reasons", []):
        print("  !", why)
    print("  renders:", s.get("render_dir"), {k: (v if isinstance(v, str) else f"{len(v)} files") for k, v in s.get("renders", {}).items()})
    if s.get("contact_sheet"):
        print(f"  ALL pages: {s.get('render_dir')}/{s['contact_sheet']} (+ thumb_p<N>.png) - look at every page first")
    print("  phases (timing):", list(r.get("phases", {})))
    print("  zones:", ", ".join(f"{z['label']}(p{z['page']})" for z in s.get("zones", [])) or "-")
    cols = ["detector", "slot", "input_file", "zone_label", "xlsm", "phase_timing", "switch_phase", "description",
            "config_function", "func5_label", "n_on_staging", "n_on_dec2024", "max5_staging", "dq_score", "dq_flags", "derived"]
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 34)
    ch = pd.DataFrame(r.get("channels", [])).reindex(columns=cols)
    print(ch.to_string(index=False))
    cand = [d["detector"] for d in r.get("detectors", [])]
    print("  detectors pre-listed for the visual pass:", cand)


def crop(dn: str, page: int, fx0, fy0, fx1, fy1, det: int) -> str:
    from cab_pdf import zoom
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    return zoom(dn, page, fx0, fy0, fx1, fy1, dpi=250, out=CROP_DIR / f"{dn}_d{int(det)}.png")


def finish(dn: str, vis: dict) -> dict:
    """Merge the visual fields into signals/<DN>.json after validating them; returns the record."""
    r = load(dn)
    errs = []
    by = {int(d["detector"]): d for d in r.get("detectors", [])}
    for v in vis.get("detectors", []):
        det = int(v["detector"])
        f = v.get("function")
        if f not in FUNCS:
            errs.append(f"d{det}: function {f!r} not in {sorted(x for x in FUNCS if x)}")
        if v.get("lane_type") not in LANES:
            errs.append(f"d{det}: lane_type {v.get('lane_type')!r} not in the vocabulary")
        if v.get("confidence") not in CONF:
            errs.append(f"d{det}: confidence must be high|medium|low")
        if v.get("stopbar_position") not in POS:
            errs.append(f"d{det}: stopbar_position {v.get('stopbar_position')!r} not in {sorted(x for x in POS if x)}")
        if v.get("technology", "loop") not in TECH:
            errs.append(f"d{det}: technology {v.get('technology')!r}")
        if f in ("Presence", "Count", "Yellow_Red") and not v.get("stopbar_position"):
            errs.append(f"d{det}: {f} needs stopbar_position (before / on / past / next_to_count)")
        tech = v.get("technology") or by.get(det, {}).get("technology") or "loop"
        flags = v.get("flags", []) if isinstance(v.get("flags", []), list) else []
        if f == "Count" and tech == "loop":  # user 2026-09-23: loops never count at the stop bar
            errs.append(f"d{det}: a loop cannot be Count -> Advance if upstream, Presence if at the stop bar "
                        "(flag label_disagrees); only radar/video have stop-bar Count")
        if f == "Yellow_Red" and tech == "loop":  # user 2026-09-23: a print-coded YR zone IS Yellow_Red on radar or video
            errs.append(f"d{det}: Yellow_Red not on a loop (got {tech!r}); radar/video zones the print codes YR are "
                        "Yellow_Red, an uncoded video zone by the stop bar is Count, a loop is Presence/Advance")
        if "series_loop_pending" in flags:
            errs.append(f"d{det}: flag series_loop_pending is retired -> function Mid, subtype series_loop "
                        "(the farthest loop in the lane stays Advance)")
        if f in ("Count", "Yellow_Red"):
            c = CROP_DIR / f"{dn}_d{det}.png"
            ca = CROP_DIR / f"{dn}_d{det}_approach.png"
            if c.exists():
                v["crop"] = c.name
            elif "position_not_drawn" in flags and ca.exists():
                v["crop"] = ca.name
            else:
                errs.append(f"d{det}: {f} needs its crop first: python cab_record.py crop {dn} <page> fx0 fy0 fx1 fy1 {det}"
                            + (" (position_not_drawn: <DN>_d<det>_approach.png is accepted)" if "position_not_drawn" in flags else ""))
        if f is None and not v.get("confidence_reason"):
            errs.append(f"d{det}: unclassified needs a confidence_reason")
        if not isinstance(v.get("flags", []), list):
            errs.append(f"d{det}: flags must be a list")
    for f in vis.get("signal_flags", []):
        if f not in SIGNAL_FLAGS:
            errs.append(f"signal flag {f!r} not in {sorted(SIGNAL_FLAGS)}")
    if "unusual_layout" in vis.get("signal_flags", []) and not vis.get("signal_flags_reason"):
        errs.append("signal flag unusual_layout needs signal_flags_reason (why the site is a rare custom set-up)")
    for p, ph in vis.get("phases", {}).items():
        n = ph.get("n_lanes")
        if n is not None and not (isinstance(n, int) and 0 < n <= 6):
            errs.append(f"phase {p}: n_lanes {n!r}")
    if errs:
        raise ValueError(f"{dn}: record NOT saved:\n  " + "\n  ".join(errs))
    for v in vis.get("detectors", []):
        det = int(v["detector"])
        d = by.setdefault(det, {"detector": det})
        d.update({k: v[k] for k in VISUAL if k in v})
    r["detectors"] = sorted(by.values(), key=lambda d: d["detector"])
    for p, ph in vis.get("phases", {}).items():
        r.setdefault("phases", {}).setdefault(str(p), {}).update(ph)
    r.setdefault("explained_other", {}).update({str(k): v for k, v in vis.get("explained_other", {}).items()})
    sf = vis.get("signal_flags", [])
    if sf:  # signal-level flags, e.g. unusual_layout (user 2026-09-23): kept, never dropped by a re-finish
        r["signal_flags"] = sorted(set(r.get("signal_flags", [])) | set(sf))
        if vis.get("signal_flags_reason"):
            r["signal_flags_reason"] = vis["signal_flags_reason"]
    for k in ("cabinet_type", "diagram_page", "note"):
        if k in vis:
            r[k] = vis[k]
    if r.get("status") == "reread_stopbar":
        r["reread"] = dict(done=time.strftime("%Y-%m-%d %H:%M"), detectors=[int(v["detector"]) for v in vis.get("detectors", [])],
                           minutes=vis.get("minutes_visual"))
    else:
        r["minutes_visual"] = vis.get("minutes_visual")
    r["status"] = "visual_done"
    r["visual_saved"] = time.strftime("%Y-%m-%d %H:%M")
    (SIG_DIR / f"{dn}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    return r


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "show":
        for dn in a[1:]:
            show(dn)
    elif a[0] in ("zoom", "crop"):
        from cab_pdf import zoom
        dn, page, box = a[1], int(a[2]), list(map(float, a[3:7]))
        print(crop(dn, page, *box, int(a[7])) if a[0] == "crop" else zoom(dn, page, *box, tag=a[7] if len(a) > 7 else "z"))
    elif a[0] == "finish":
        rec = finish(a[1], json.loads(open(a[2], encoding="utf-8").read()))
        print(f"saved {a[1]}: {len(rec['detectors'])} detectors, status {rec['status']}")
