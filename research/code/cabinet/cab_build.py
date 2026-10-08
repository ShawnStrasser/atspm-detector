"""Per-signal worksheet (for the reader) and the consolidator (guide section 6).

    python cab_build.py sheet 01001      print everything scripted about one signal, one row per channel
    python cab_build.py build            signals/<DN>.json -> %DC_WORK%/cabinet/print_labels.parquet

`signals/<DN>.json` is the reader's record (written after the visual pass), keyed by detector channel:
  {"DeviceName": "...", "cabinet_type": "332", "diagram_page": 5,
   "phases": {"2": {"n_lanes": 2, "lanes": "T,TR", "note": ""}, ...},
   "detectors": [{"detector": 4, "technology": "loop", "loops": "3,4", "phase_diagram": 2,
                  "function": "Mid", "subtype": "mid", "stopbar_position": null, "lane_index": null, "lanes_spanned": 2,
                  "lane_type": "T", "distance_ft": null, "flags": [], "confidence": "high",
                  "confidence_reason": ""}, ...],
   "explained_other": {"33": "why an active channel not on the print is expected"},
   "minutes_visual": 6}
function: Advance | Presence | Count | Yellow_Red | Bike | Mid | Other (Bike / Mid are classes since
2026-09-23); stopbar_position (Presence = before, Count = on / past, Yellow_Red = next_to_count).
Since the scale-up the record also carries `scripted` + `channels` (cab_scale.py) and a `status`:
"scripted" records (visual pass not done) are skipped here; `cab_record.py finish` validates and saves.
Slot, input-file phase, timing phase, activity, dead / unexplained rows and the tier are added here.
Activity, dead flags, whole-intersection rows and tiers use each signal's NEWEST window (see build()); the full
consolidation (fixes, rulings, DQ, label table v3, user lists) is `cab_final.py`, which calls build().
A record's signal-level `signal_flags: ["unusual_layout"]` (or `unusual_layout` written in its `note`) is carried as its own column `unusual_layout` in
print_tiers.csv and print_labels.parquet (tier unchanged): those signals are excluded from training, scored separately.
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from cab_common import CAB, DC_WORK, DET_TO_SLOT, REPO, SIG_DIR, device_id

DERIVED = re.compile(r"(?i)dummy|FYA|pass ?[0-9]|(^|[^a-z])ped([^a-z]|$)|preempt|(^|[^a-z])EV[A-D]?([^a-z]|$)")
COLS = ["DeviceId", "DeviceName", "detector", "cabinet_type", "pdf", "diagram_page", "technology", "loops", "slot",
        "phase_input_file", "phase_diagram", "phase_timing", "function", "subtype", "lane_index", "lanes_spanned",
        "lane_type", "stopbar_position", "n_lanes_phase", "distance_ft", "flags", "dq_score", "confidence", "confidence_reason",
        "crop", "func_label", "n_on", "n_on_new", "n_on_dec2024", "n_on_staging", "window_new", "source"]


def _timing(dn):
    o = pd.read_parquet(DC_WORK / "official/labels_official.parquet")
    o = o[o.DeviceName == dn].copy()
    o["phase_timing"] = o.call_phase.where(o.call_phase > 0, o.call_overlap.map(lambda v: f"O{v}" if v else None))
    return o.set_index("Detector")


def _func(dn):
    f = pd.read_parquet(REPO / "research/labels/function_labels_v2.parquet")
    return f[f.DeviceName == dn].set_index("Detector")


def _act(dn):
    a = pd.read_parquet(CAB / "activity.parquet")
    a = a[a.DeviceName == dn]
    return a.pivot_table(index="Detector", columns="window", values=["n_on", "max_5min", "active_hours"], aggfunc="sum").fillna(0)


def sheet(dn):
    ex = json.loads((SIG_DIR / f"{dn}.extract.json").read_text(encoding="utf-8"))
    cab = ex["cabinet_type"]
    t, f, a = _timing(dn), _func(dn), _act(dn)
    inp = {}
    for r in ex["input_file"]:
        if r.get("detector"):
            inp.setdefault(r["detector"], []).append(f"{r['slot']}[{r['loops']}]ph{r['phase_marker']}"
                                                     + (f"@{r['distance_ft']}" if r.get("distance_ft") else "") + f"/{r['layout']}")
    zones = {z["detector"]: z["label"] for z in ex["zones"]}
    xl = pd.DataFrame()
    if ex.get("xlsm"):
        from cab_xlsm import parse
        xl = parse(ex["xlsm"]).query("used").set_index("mt")
    dets = sorted(set(inp) | set(zones) | set(t[t.phase_timing.notna()].index) | set(a.index) | set(f.index) | set(xl.index))
    rows = []
    for d in dets:
        rows.append(dict(
            det=d, slot=DET_TO_SLOT.get(cab, {}).get(d, ""), input=" ".join(inp.get(d, [])), zone=zones.get(d, ""),
            xlsm=(f"{xl.loc[d, 'phase']}|{xl.loc[d, 'function']}" if d in xl.index else ""),
            timing=t.phase_timing.get(d), desc=str(t.description.get(d, ""))[:24], func=f.func5.get(d, ""),
            cfg=str(f.config_function.get(d, ""))[:14],
            dec=int(a[("n_on", "dec2024")].get(d, 0)) if ("n_on", "dec2024") in a else 0,
            stg=int(a[("n_on", "staging")].get(d, 0)) if ("n_on", "staging") in a else 0,
            max5=int(a["max_5min"].max(axis=1).get(d, 0)) if "max_5min" in a else 0))
    rows = [r for r in rows if r["input"] or r["zone"] or r["xlsm"] or r["dec"] or r["stg"] or r["func"] or r["desc"] not in ("", "None", "nan")]
    print(f"{dn} cabinet {cab} diagram pages {ex['diagram_pages']} pdf {ex['pdf'].split(chr(92))[-1]} xlsm {bool(ex.get('xlsm'))}")
    print(pd.DataFrame(rows).to_string(index=False))


ACTIVE_MIN = 5  # ONs in the newest window for a channel to count as active (note 25)
UNUSUAL_NOTE = re.compile(r"\bunusual_layout\b")
NOT_UNUSUAL = re.compile(r"\bnot\b[^.;]{0,30}\bunusual_layout\b", re.I)


def unusual_flag(s: dict) -> tuple[bool, bool]:
    """(unusual_layout, came from the note only). The flag is set by `signal_flags`, or by the reader writing
    `unusual_layout` in the record's note (6 records did only that); a note saying it is NOT flagged does not count
    (2C023: "not flagged unusual_layout")."""
    note = str(s.get("note") or "")
    by_note = bool(UNUSUAL_NOTE.search(note)) and not NOT_UNUSUAL.search(note)
    by_flag = "unusual_layout" in (s.get("signal_flags") or [])
    return by_flag or by_note, by_note and not by_flag


def _newest(a_all: pd.DataFrame, dn: str):
    """(newest window with any actuation at this signal, ONs per detector there, ONs per detector per window)."""
    a = a_all[a_all.DeviceName == dn]
    per = a.pivot_table(index="Detector", columns="window", values="n_on", aggfunc="sum").fillna(0)
    win = next((w for w in ("staging", "dec2024") if w in per and per[w].sum() > 0), None)
    return win, (per[win] if win else pd.Series(dtype=float)), per


def build(only: set | None = None, keep_print: dict | None = None):
    """Consolidate the visual records. `only` = DeviceNames to include (default: every visual_done record);
    `keep_print` = {DeviceName: {det, ...}} channels that must be print detectors (a ruling; error if not).

    Activity is judged in the signal's NEWEST window (staging if it has any actuation there, else Dec 2024):
      * print detector with 0 ONs there -> flag `dead` (listed for maintenance, not trained on; confidence kept);
      * channel NOT on the print with >= ACTIVE_MIN ONs there -> a row function Other, source data_only, subtype
        `explained` (reader's explained_other, or a derived channel by config description) or `unexplained`.
        A derived-by-description test never applies to a channel that is on the print.
    Tier: complete = the print was read (no print detector with >= ACTIVE_MIN ONs left unclassified) and every
    active channel is on the print or explained; complete_high = complete and every classified print detector high."""
    recs, tiers = [], []
    off = pd.read_parquet(DC_WORK / "official/labels_official.parquet")
    f_all = pd.read_parquet(REPO / "research/labels/function_labels_v2.parquet")
    a_all = pd.read_parquet(CAB / "activity.parquet")
    keep_print = keep_print or {}
    for fj in sorted(SIG_DIR.glob("*.json")):
        if fj.name.endswith(".extract.json"):
            continue
        s = json.loads(fj.read_text(encoding="utf-8"))
        if s.get("status") == "scripted":
            continue
        dn = s["DeviceName"]
        if only is not None and dn not in only:
            continue
        dqs = {int(c["detector"]): c.get("dq_score") for c in s.get("channels", [])}
        cdesc = {int(c["detector"]): c.get("description") for c in s.get("channels", [])}
        dev = device_id(dn)  # refuses locked signals
        ex = json.loads((SIG_DIR / f"{dn}.extract.json").read_text(encoding="utf-8"))
        cab = s.get("cabinet_type") or ex["cabinet_type"]
        t = off[off.DeviceName == dn].copy()
        t["phase_timing"] = t.call_phase.where(t.call_phase > 0, t.call_overlap.map(lambda v: f"O{v}" if v else None))
        t = t.set_index("Detector")
        f = f_all[f_all.DeviceName == dn].set_index("Detector")
        win, n_new, per = _newest(a_all, dn)
        n_dec = per["dec2024"] if "dec2024" in per else pd.Series(dtype=float)
        n_stg = per["staging"] if "staging" in per else pd.Series(dtype=float)
        pif = {r["detector"]: r["phase_marker"] for r in ex["input_file"] if r.get("detector")}
        on_print = set()
        for d in s["detectors"]:
            det = int(d["detector"])
            on_print.add(det)
            flags = list(d.get("flags", []))
            n = float(n_new.get(det, 0))
            if win is None:
                flags.append("no_data")
            elif n == 0:
                flags.append("dead")
            ph_t = t.phase_timing.get(det)
            ph = s.get("phases", {}).get(str(d.get("phase_diagram")), {}) or s.get("phases", {}).get(str(ph_t), {})
            recs.append(dict(DeviceId=dev, DeviceName=dn, detector=det, cabinet_type=cab, pdf=ex["pdf"].split("\\")[-1],
                             diagram_page=s.get("diagram_page"), technology=d.get("technology", "loop"),
                             loops=d.get("loops"), slot=d.get("slot") or DET_TO_SLOT.get(cab, {}).get(det),
                             phase_input_file=pif.get(det), phase_diagram=d.get("phase_diagram"),
                             phase_timing=None if ph_t is None else str(ph_t), function=d.get("function"),
                             subtype=d.get("subtype"), lane_index=d.get("lane_index"),
                             lanes_spanned=d.get("lanes_spanned"), lane_type=d.get("lane_type"),
                             stopbar_position=d.get("stopbar_position"),
                             n_lanes_phase=ph.get("n_lanes"), distance_ft=d.get("distance_ft"),
                             flags=",".join(sorted(set(flags))), dq_score=dqs.get(det), confidence=d.get("confidence"),
                             confidence_reason=d.get("confidence_reason", ""), crop=d.get("crop"),
                             func_label=f.func5.get(det), n_on=float(n_dec.get(det, 0) + n_stg.get(det, 0)),
                             n_on_new=n, n_on_dec2024=float(n_dec.get(det, 0)), n_on_staging=float(n_stg.get(det, 0)),
                             window_new=win, source="print"))
        missing = set(keep_print.get(dn, ())) - on_print
        if missing:
            raise SystemExit(f"{dn}: channels {sorted(missing)} must be print detectors (ruling keep_print) but are not")
        expl = {int(k): v for k, v in s.get("explained_other", {}).items()}
        for det in set(t.index) | set(cdesc):
            desc = t.description.get(det) if det in t.index else cdesc.get(det)
            if det not in expl and det not in on_print and isinstance(desc, str) and DERIVED.search(desc):
                expl[int(det)] = f"derived channel per config description: {desc}"
        act = n_new[n_new >= ACTIVE_MIN]
        for det in sorted(set(act.index) - on_print):
            ph_t = t.phase_timing.get(det)
            recs.append(dict(DeviceId=dev, DeviceName=dn, detector=int(det), cabinet_type=cab, pdf=ex["pdf"].split("\\")[-1],
                             function="Other", subtype="explained" if det in expl else "unexplained",
                             flags="not_on_print", confidence="low", confidence_reason=expl.get(det, "active channel not on the print"),
                             phase_timing=None if ph_t is None else str(ph_t),
                             func_label=f.func5.get(det), n_on=float(n_dec.get(det, 0) + n_stg.get(det, 0)),
                             n_on_new=float(n_new.get(det)), n_on_dec2024=float(n_dec.get(det, 0)),
                             n_on_staging=float(n_stg.get(det, 0)), window_new=win, source="data_only"))
        mine = [r for r in recs if r["DeviceName"] == dn]
        pr = [r for r in mine if r["source"] == "print"]
        unexpl = [r for r in mine if r["subtype"] == "unexplained"]
        unread = [r for r in pr if r["function"] is None and r["n_on_new"] >= ACTIVE_MIN]
        cls = [r for r in pr if r["function"] is not None]
        complete = bool(pr) and not unexpl and not unread and win is not None
        tier = ("complete_high" if complete and cls and all(r["confidence"] == "high" for r in cls)
                else "complete_mixed" if complete else "incomplete")
        unusual, note_unusual = unusual_flag(s)
        tiers.append(dict(DeviceName=dn, DeviceId=dev, tier=tier, unusual_layout=unusual,
                          unusual_reason=(s.get("signal_flags_reason") or (s.get("note") if note_unusual else None))
                          if unusual else None,
                          window_new=win, n_print=len(pr), n_classified=len(cls),
                          n_high=sum(r["confidence"] == "high" for r in cls), n_unread_active=len(unread),
                          n_unexplained=len(unexpl), n_explained=sum(r["subtype"] == "explained" for r in mine),
                          n_dead=sum("dead" in r["flags"] for r in pr), minutes_visual=s.get("minutes_visual")))
    df = pd.DataFrame(recs).reindex(columns=COLS)
    df = df.merge(pd.DataFrame(tiers)[["DeviceName", "tier", "unusual_layout"]], on="DeviceName", how="left")
    for c in ("phase_input_file", "phase_diagram", "lane_index", "lanes_spanned", "n_lanes_phase"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int16")
    for c in ("distance_ft", "loops", "func_label", "crop", "slot"):
        df[c] = df[c].astype("string")
    df.to_parquet(CAB / "print_labels.parquet", index=False)
    pd.DataFrame(tiers).to_csv(CAB / "print_tiers.csv", index=False)
    return df, pd.DataFrame(tiers)


if __name__ == "__main__":
    if sys.argv[1] == "sheet":
        for dn in sys.argv[2:]:
            sheet(dn)
    else:
        df, tiers = build()
        print(tiers.to_string(index=False))
        print(tiers.groupby(["tier", "unusual_layout"]).size())
        print(df.groupby(["confidence"]).size().to_dict(), df.function.value_counts().to_dict())
