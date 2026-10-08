"""Voyage (VD) numbering audit of the cabinet-print records (note 34).

Old prints number video detectors with Voyage "VD" numbers. A VD number is NOT a MaxTime channel: the channel is
decided by the physical input-file slot (SLOT_TO_DET), and the VD labels are a fixed relabelling of the same slots,
VD -> MT below (checked against every config description "VD#n" and against the VD labels drawn per slot on the
cabinet sheets). This script
    python vd_audit.py scan      text layer of every non-locked print: VD tokens, "Voyage", per-page slot check
    python vd_audit.py audit     one row per detector of every VD print: how its channel was derived, checks, verdict
    python vd_audit.py card      card rule: both outputs of one input-file slot dead / erratic together (cleansing only)
Outputs in %DC_WORK%/cabinet/: vd_scan.csv, vd_audit.csv, card_health.parquet (per slot), card_channels.parquet.
Locked hold-out signals are skipped (cab_common.signals().locked). Never a model input.
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd

from cab_common import CAB, DC_WORK, DEFAULT_PHASE, REPO, SIG_DIR, SLOT_TO_DET, DET_TO_SLOT, signals

# Voyage detector number -> MaxTime channel (same physical slot). VD 1-8 = the left-turn / minor phases' pairs,
# VD 9-28 = phases 2, 4, 6, 8 in order, VD 29-40 = identical.
VD2MT = {1: 1, 2: 13, 3: 7, 4: 14, 5: 15, 6: 27, 7: 21, 8: 28, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 14: 8, 15: 9,
         16: 10, 17: 11, 18: 12, 19: 16, 20: 17, 21: 18, 22: 19, 23: 20, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26}
VD2MT.update({i: i for i in range(29, 41)})
MT2VD = {m: v for v, m in VD2MT.items()}
RX_VD = re.compile(r"\bVD ?#? ?(\d{1,2})\b(?![-/]\d)")          # "VD19", "VD 19", "VD#19"; not "VD 00/3", "VD9-27"
SCAN = CAB / "vd_scan.csv"
AUDIT = CAB / "vd_audit.csv"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def records(dns=None):
    lk = signals()
    locked = set(lk[lk.locked].DeviceName)
    for f in sorted(SIG_DIR.glob("*.json")):
        if f.name.endswith(".extract.json"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["DeviceName"] in locked or (dns and r["DeviceName"] not in dns):
            continue
        yield r


# ------------------------------------------------------------------------------------------------ scan
def _page_slot_check(words, cab, page=1):
    """VD tokens on one page vs the physical slot table: each y-band (one input file row) must list its slots in x
    order, the upper output above the lower. Returns (n_tokens_checked, n_violations) or None if not a slot row."""
    if cab not in DET_TO_SLOT:
        return None
    # (1) a table with the slot written on the VD label's own line ("2 | 5U | VD5", "I5U ... VD5")
    slots = [w for w in words if re.fullmatch(r"[IJ]?\d{1,2}[UL]", w[4])]
    n = bad = 0
    for w in words:
        m = re.fullmatch(r"VD(\d{1,2})", w[4])
        if not m or int(m.group(1)) not in VD2MT:
            continue
        row = [s for s in slots if abs((s[1] + s[3]) / 2 - (w[1] + w[3]) / 2) < 3 and 0 < w[0] - s[0] < 120]
        if not row:
            continue
        s = max(row, key=lambda s: s[0])[4]
        s = s if s[0] in "IJ" else "I" + s
        n += 1
        bad += SLOT_TO_DET[cab].get(s) != VD2MT[int(m.group(1))]
    if n >= 4:
        return (n, bad)
    if page > 2:  # the front-view slot rows are on the cabinet sheets (1 for 332, 2 for 332S); later pages = diagrams
        return None
    tok = []
    for w in words:
        m = re.fullmatch(r"VD(\d{1,2})", w[4])
        if m and int(m.group(1)) in VD2MT:
            s = DET_TO_SLOT[cab].get(VD2MT[int(m.group(1))])
            if s:
                tok.append((w[0], w[1], s[0], int(s[1:-1]), s[-1]))
    if len(tok) < 12:
        return None
    best = None
    for ax in (0, 1):  # bands on y (upright) or on x (rotated page)
        t = sorted(set(tok), key=lambda z: z[1 - ax])
        bands, cur = [], [t[0]]
        for z in t[1:]:
            if z[1 - ax] - cur[-1][1 - ax] > 25:
                bands.append(cur)
                cur = [z]
            else:
                cur.append(z)
        bands.append(cur)
        n = bad = 0
        for b in bands:
            files = {z[2] for z in b}
            if len(files) > 1:
                bad += len(b)
                n += len(b)
                continue
            by = {}
            for z in b:
                by.setdefault(z[3], []).append(z)
            nums = sorted(by)
            xs = [np.mean([z[ax] for z in by[k]]) for k in nums]
            if len(nums) > 2:
                rho = pd.Series(xs).rank().corr(pd.Series(range(len(xs))).rank())
                if abs(rho) < 0.999:
                    bad += len(b)
            for k, zs in by.items():  # upper above lower (sign may flip on rotated pages: judged over the page)
                u = [z[1 - ax] for z in zs if z[4] == "U"]
                lo = [z[1 - ax] for z in zs if z[4] == "L"]
                if u and lo and not (max(u) < min(lo)):
                    bad += 1
            n += len(b)
        if best is None or bad < best[1]:
            best = (n, bad)
    return best


def scan():
    import pymupdf as fitz
    rows = []
    for r in records():
        dn, s = r["DeviceName"], r.get("scripted", {})
        cab = r.get("cabinet_type") or s.get("cabinet_type")
        pp = s.get("pdf_path")
        pages, voy, nvd, checks, leg = {}, 0, 0, [], set()
        if pp:
            try:
                d = fitz.open(pp)
            except Exception:
                d = []
            for i, p in enumerate(d):
                t = p.get_text()
                v = RX_VD.findall(t)
                nv = len(re.findall(r"voyage", t, re.I))
                voy += nv
                tt = re.sub(r"\s+", " ", t).lower()
                if re.search(r"max ?tim ?e or voyage", tt):
                    leg.add("mt_or_vd_bubbles")      # radar legend: bubble = "max time or voyage number"
                elif "with voyage detector number" in tt:
                    leg.add("vd_bubbles")            # video/radar legend: bubble = Voyage detector number
                if "voyage detector" in tt and "file slot" in tt:
                    leg.add("vd_slot_table")         # layout table: file slot -> Voyage detector #
                if "distance voyage" in tt:
                    leg.add("loop_table_voyage_col")  # loop wiring table with a Voyage column
                if v:
                    pages[i + 1] = len(set(v))
                    nvd += len(v)
                    c = _page_slot_check(p.get_text("words"), cab, i + 1)
                    if c:
                        checks.append((i + 1, *c))
        desc = sum(1 for c in r.get("channels", []) if RX_VD.search(c.get("description") or ""))
        flag = any("vd_remapped" in (d.get("flags") or []) for d in r.get("detectors", []))
        n1 = max(pages.values()) if pages else 0
        only_cab = bool(pages) and voy == 0 and set(pages) <= {1, 2} and all(
            c[1] >= 12 for c in checks) and len(checks) == len(pages)
        kind = ("vd_layout" if (pages and not only_cab) or voy else "cabinet_template" if only_cab
                else "flag_only" if flag else "desc_only" if desc else "")
        rows.append(dict(DeviceName=dn, status=r.get("status"), cabinet=cab, scripted=s.get("status"), kind=kind,
                         vd_pages=";".join(f"p{k}:{v}" for k, v in sorted(pages.items())), voyage_words=voy,
                         legend=";".join(sorted(leg)), n_vd_tokens=nvd, max_vd_distinct=n1, desc_vd=desc, vd_remapped=flag,
                         slot_checks=";".join(f"p{a}:{b}/{c}bad" for a, b, c in checks),
                         slot_tokens=sum(c[1] for c in checks), slot_bad=sum(c[2] for c in checks)))
    df = pd.DataFrame(rows)
    df.to_csv(SCAN, index=False)
    v = df[df.kind != ""]
    log(f"{len(df)} records; VD prints {len(v)}: {v.kind.value_counts().to_dict()}; slot rows checked on "
        f"{(v.slot_tokens > 0).sum()} prints, {int(v.slot_tokens.sum())} VD labels, {int(v.slot_bad.sum())} off the "
        f"slot table")
    return df


# ------------------------------------------------------------------------------------------------ audit
RX_BUB = [re.compile(r"\b([1-8])[A-H]?/(\d{1,2})\b"),            # "6A/19", "2/9"
          re.compile(r"\b([1-8])[A-H] (\d{1,2})\b"),             # scripted zone label "6A 19"
          re.compile(r"\bbox ([1-8])-(\d{1,2})\b")]              # "box 6-23"


def bubbles(txt: str) -> list[tuple[int, int]]:
    out = []
    for rx in RX_BUB:
        out += [(int(a), int(b)) for a, b in rx.findall(txt)]
    return sorted(set(out))


def method(d, ch):
    """How the reader derived the channel, from the record's own words; plus the VD numbers and bubbles it cites."""
    txt = " ".join(str(d.get(k) or "") for k in ("loops", "confidence_reason", "subtype"))
    low = txt.lower()
    m = []
    if d.get("slot") and ch.get("input_file"):
        m.append("slot_input_file")
    elif re.search(r"\b[IJ]\d{1,2}(-?[DEJK]\b| ?(upper|lower|U\b|L\b))", txt):
        m.append("slot_named")
    vds = [int(x) for x in RX_VD.findall(txt)]
    if vds:
        m.append("vd_number")
    if re.search(r"layout table|vd table|wiring table|vd# ?->|vd ?-> ?slot", low):
        m.append("vd_table")
    if re.search(r"\bmt ?#? ?\d|xlsm|zoneconfiguration|zone table", low):
        m.append("mt_table")
    bub = bubbles(txt)
    if not bub and not m and not re.search(r"\bmt ?#? ?\d|layout table", low):
        bub = bubbles(ch.get("zone_label") or "")  # the scripted label is the reader's only evidence
    if bub or re.search(r"cam ?[a-h] ?\d", low):
        m.append("zone_label")
    if re.search(r"config|description|desc\b", low):
        m.append("config_desc")
    if re.search(r"elimination|counts|volume|pairing|sum of", low):
        m.append("count_matching")
    return m or ["unstated"], vds, bub, txt


def audit():
    sc = pd.read_csv(SCAN, dtype={"DeviceName": str}).fillna({"kind": "", "legend": ""})
    sc = sc[sc.kind != ""].set_index("DeviceName")
    v3 = pd.read_parquet(REPO / "research/labels/function_labels_v3.parquet",
                         columns=["DeviceName", "detector", "validated", "failed_checks"])
    v3 = v3.drop_duplicates(["DeviceName", "detector"]).set_index(["DeviceName", "detector"])
    rows = []
    for r in records(set(sc.index)):
        if r.get("status") != "visual_done":
            continue
        dn = r["DeviceName"]
        kind, legend = sc.at[dn, "kind"], sc.at[dn, "legend"]
        # bubbles on this print are Voyage numbers (legend / VD layout table), not MaxTime channels
        vd_bub_print = "vd_bubbles" in legend or "vd_slot_table" in legend
        cab = r.get("cabinet_type") or r.get("scripted", {}).get("cabinet_type")
        ch = {int(c["detector"]): c for c in r.get("channels", [])}
        on_print = {int(d["detector"]) for d in r.get("detectors", []) if d.get("function")}
        note_mt = bool(re.search(r"\bMT ?#|\bMT\b[^.;]*(table|= ?detector|= ?channel)|[Mm]ax ?[Tt]ime", r.get("note") or ""))

        def live(k):
            return ch.get(k, {}).get("n_on_staging") or ch.get(k, {}).get("n_on_dec2024") or 0

        def tph(k):
            return ch.get(k, {}).get("phase_timing")

        for d in r.get("detectors", []):
            det = int(d["detector"])
            c = ch.get(det, {})
            meth, vds, bub, txt = method(d, c)
            pt, pp = tph(det), d.get("phase_diagram") or d.get("phase_input_file")
            slot_ok = SLOT_TO_DET[cab].get(d["slot"]) == det if d.get("slot") and cab in SLOT_TO_DET else None
            vd_ok = (det in {VD2MT.get(x) for x in vds}) if vds else None
            vd_as_channel = bool(vds) and not vd_ok and det in vds and VD2MT.get(det) != det
            # bubble numbers: does the detector follow the MaxTime reading (channel n) or the Voyage one (VD2MT[n])?
            bub_read, bub_evid = None, ""
            # the reader's explicit derivation (input-file slot, MT / VD table, config text) outranks a bubble number
            explicit = slot_ok or note_mt or any(k in meth for k in ("slot_named", "mt_table", "vd_table", "config_desc"))
            for ph, n in ([] if explicit else bub):
                if n not in VD2MT or VD2MT[n] == n:
                    continue
                alt = VD2MT[n]
                if det == alt or bub_read == "vd" or vd_ok:
                    bub_read = "vd" if det == alt or bub_read == "vd" else bub_read
                elif det == n:
                    # timing phase; for a channel not in the timing, the standard channel -> phase default
                    ev_mt = (tph(n) if tph(n) is not None else DEFAULT_PHASE.get(n)) == ph
                    ev_vd = (tph(alt) if tph(alt) is not None else DEFAULT_PHASE.get(alt)) == ph
                    bub_evid = (f"bubble {ph}/{n}: ch{n} timing ph{tph(n)} n_on {live(n)}; VD{n}=ch{alt} timing "
                                f"ph{tph(alt)} n_on {live(alt)} on_print {alt in on_print}")
                    if vd_bub_print and ev_vd and not ev_mt:
                        bub_read = "mt_wrong"
                    elif ev_vd and not ev_mt:
                        bub_read = "mt_phase_favours_vd"
                    elif vd_bub_print and not ev_mt:
                        bub_read = "mt_unsupported"
                    else:
                        bub_read = "mt"
            alt = VD2MT.get(det)
            red_flag = bool(alt and alt != det and live(det) == 0 and live(alt) > 0 and alt not in on_print
                            and (pp is None or tph(alt) == pp))
            key = (dn, det)
            if note_mt and "mt_table" not in meth:
                meth = meth + ["mt_table_note"]
            rows.append(dict(DeviceName=dn, kind=kind, legend=legend, vd_bubble_print=vd_bub_print, cabinet=cab,
                             detector=det, function=d.get("function"), confidence=d.get("confidence"),
                             technology=d.get("technology"), flags=";".join(d.get("flags") or []), method=";".join(meth),
                             vd_cited=";".join(map(str, vds)), bubbles=";".join(f"{a}/{b}" for a, b in bub),
                             slot=d.get("slot"), slot_ok=slot_ok, vd_ok=vd_ok, vd_as_channel=vd_as_channel,
                             bubble_read=bub_read, bubble_evidence=bub_evid, phase_timing=pt, phase_print=pp,
                             phase_ok=None if (pt is None or pp is None) else int(pt) == int(pp), n_on=live(det),
                             alt_channel=alt if alt != det else None, alt_n_on=live(alt) if alt and alt != det else None,
                             alt_on_print=(alt in on_print) if alt and alt != det else None, red_flag=red_flag,
                             validated=v3.at[key, "validated"] if key in v3.index else None,
                             failed_checks=v3.at[key, "failed_checks"] if key in v3.index else None,
                             reason=txt[:300]))
    a = pd.DataFrame(rows)
    a["verdict"] = a.apply(verdict, axis=1)
    hand = a.apply(lambda x: (x.DeviceName, x.detector) in RULINGS, axis=1)  # reviewed by hand (RULINGS below)
    a.loc[hand, "verdict"] = "suspect"
    a.to_csv(AUDIT, index=False)
    lab = a[a.function.notna()]
    log(f"audit: {a.DeviceName.nunique()} VD prints, {len(a)} detectors ({len(lab)} labelled); verdicts (labelled): "
        f"{lab.verdict.value_counts().to_dict()}")
    return a


def verdict(x):
    """wrong: a VD number / VD bubble taken as the channel, or a slot mapped off the table. suspect: the channel is
    silent while its VD reading is live and unlabelled on the same phase, or the phase favours the VD reading.
    confirmed: derived through slot / VD table / MT table, timing phase = print phase, live, behaviour not failed."""
    if not isinstance(x.function, str):
        return "unlabelled"
    if x.vd_as_channel or x.slot_ok is False or x.bubble_read == "mt_wrong":
        return "wrong"
    if x.red_flag or x.bubble_read in ("mt_phase_favours_vd", "mt_unsupported"):
        return "suspect"
    derived = x.slot_ok is True or x.vd_ok is True or x.bubble_read == "vd" or any(
        m in x.method for m in ("mt_table", "slot_named", "vd_table"))  # "mt_table" also matches mt_table_note
    if x.phase_ok is False and not derived:
        return "suspect"
    if derived and x.phase_ok is True and x.n_on > 0 and str(x.validated) != "fail":
        return "confirmed"
    return "plausible"


# ------------------------------------------------------------------------------------------------ card rule
def card(threads: int = 4, dns=None):
    """Pairs of channels on one physical slot (upper / lower) per the cabinet table; a slot whose two outputs are
    both dead, or both erratic (stuck-on / chatter; no fault events, note 88), is `card_suspect`. Cleansing only, never a model input."""
    import duckdb
    import dq_core
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    out = []
    t0 = time.time()
    recs = [r for r in records(dns) if (r.get("cabinet_type") or r.get("scripted", {}).get("cabinet_type")) in SLOT_TO_DET]
    ids = signals().drop_duplicates("DeviceName").set_index("DeviceName").DeviceId.to_dict()
    for i, r in enumerate(recs):
        r["DeviceId"] = r.get("DeviceId") or ids.get(r["DeviceName"])
        cab = r.get("cabinet_type") or r["scripted"]["cabinet_type"]
        ch = {int(c["detector"]): c for c in r.get("channels", [])}
        pr = {int(d["detector"]): d for d in r.get("detectors", [])}
        dets = sorted(SLOT_TO_DET[cab].values())
        q = pd.DataFrame(dict(DeviceId=r["DeviceId"], detector=dets,
                              phase=[str(ch.get(k, {}).get("phase_timing") or "?") for k in dets],
                              location="other", lane_index=np.nan, lanes_spanned=1.0))
        try:
            h = dq_core.run_signal(con, r["DeviceId"], q, period="auto")
        except Exception as e:  # no events in either window
            log(f"card {r['DeviceName']}: {e}")
            continue
        if h is None or h.empty or "n_on" not in h:
            continue
        h = h.set_index("detector")
        for k in dets:
            c = ch.get(k, {})
            out.append(dict(DeviceName=r["DeviceName"], DeviceId=r["DeviceId"], cabinet=cab, detector=k,
                            slot=DET_TO_SLOT[cab][k], phase_timing=c.get("phase_timing"),
                            on_print=bool(pr.get(k, {}).get("function") or c.get("input_file")),
                            description=c.get("description"),
                            n_on_dec2024=c.get("n_on_dec2024"), source=h.at[k, "source"] if k in h.index else None,
                            n_on=h.at[k, "n_on"] if k in h.index else np.nan,
                            s_health=h.at[k, "s_health"] if k in h.index and "s_health" in h else np.nan,
                            reasons=h.at[k, "reasons"] if k in h.index else None))
        if i % 50 == 0:
            log(f"card {i}/{len(recs)} {time.time() - t0:.0f}s")
    c = pd.DataFrame(out)
    c["phase_timing"] = c.phase_timing.astype("string")  # overlaps are "O9"
    # in use = wired on the print, live in the other (Dec 2024) window, or a specific config description
    # (template texts such as "VD#9", "I3U C1-39", "N/U", "Dummy", "spare" do not count)
    generic = c.description.fillna("").str.strip().str.match(
        r"^$|^VD ?#? ?\d+$|^[IJ]\d{1,2}[UL]\b[^A-Za-z]*(C\d+-\d+)?$|^N/?U$|^(dummy|spare|not used)", case=False)
    c["in_use"] = c.on_print | c.n_on_dec2024.fillna(0).gt(0) | ~generic
    c["dead"] = c.n_on.fillna(0).eq(0)
    c["erratic"] = c.s_health.lt(0.5)
    c["slot_no"] = c.slot.str[:-1]
    c["ul"] = c.slot.str[-1]
    rows = []
    for (dn, s), g in c.groupby(["DeviceName", "slot_no"]):
        if len(g) < 2:
            continue
        u, lo = g[g.ul == "U"].iloc[0], g[g.ul == "L"].iloc[0]
        both_use = bool(u.in_use and lo.in_use)
        both_dead = both_use and u.dead and lo.dead
        both_err = both_use and bool(u.erratic) and bool(lo.erratic)
        was_live = both_dead and (u.n_on_dec2024 or 0) > 0 and (lo.n_on_dec2024 or 0) > 0
        rows.append(dict(DeviceName=dn, DeviceId=u.DeviceId, cabinet=u.cabinet, slot=s, det_upper=int(u.detector),
                         det_lower=int(lo.detector), both_in_use=both_use, both_dead=both_dead, both_erratic=both_err,
                         dead_since_dec2024=was_live, n_on_upper=u.n_on, n_on_lower=lo.n_on,
                         reasons_upper=u.reasons, reasons_lower=lo.reasons,
                         card_suspect=bool(both_dead or both_err)))
    s = pd.DataFrame(rows)
    # a signal with every channel dead is a comms / window problem, not a card: not card_suspect
    alive = c.groupby("DeviceName").dead.apply(lambda x: (~x).any())
    s.loc[~s.DeviceName.map(alive).fillna(False), "card_suspect"] = False
    s.to_parquet(CAB / "card_health.parquet", index=False)
    m = s[s.card_suspect][["DeviceName", "slot", "det_upper", "det_lower"]]
    cc = pd.concat([m.rename(columns={"det_upper": "detector", "det_lower": "card_mate"}),
                    m.rename(columns={"det_lower": "detector", "det_upper": "card_mate"})])
    c = c.merge(cc, on=["DeviceName", "detector"], how="left", suffixes=("", "_card"))
    c["card_suspect"] = c.card_mate.notna()
    c.drop(columns=["slot_card"], errors="ignore").to_parquet(CAB / "card_channels.parquet", index=False)
    log(f"card: {s.DeviceName.nunique()} signals, {len(s)} two-output slots, both in use {int(s.both_in_use.sum())}; "
        f"card_suspect {int(s.card_suspect.sum())} (both dead {int((s.card_suspect & s.both_dead).sum())}, of which "
        f"live in Dec 2024 {int((s.card_suspect & s.dead_since_dec2024).sum())}; both erratic "
        f"{int((s.card_suspect & s.both_erratic).sum())}) on {s[s.card_suspect].DeviceName.nunique()} signals")
    return s


# ------------------------------------------------------------------------------------------------ record changes
# Reviewed by hand after `audit` (note 34). No channel was found wrong; one VD-related suspect gets a flag.
# (DeviceName, detector): (flags to add, confidence to set or None, text appended to confidence_reason)
RULINGS = {
    ("08037", 24): (["vd_suspect"], None, "vd_audit: tied to zone 8A/24 by liveness, but 8A/24 is VD24 = ch22 (dead) "
                                          "and the config calls ch24 VD#26, which has no drawn zone"),
}


def fix():
    """Apply RULINGS through cab_record.finish (validated), back up each record first, log to sweep_changes.csv."""
    import csv
    import shutil
    from cab_record import finish, load
    bak = CAB / "backup_vd_audit"
    bak.mkdir(exist_ok=True)
    log_rows = []
    for (dn, det), (add, conf, why) in RULINGS.items():
        r = load(dn)
        d = next(x for x in r["detectors"] if int(x["detector"]) == det)
        flags = list(d.get("flags") or [])
        new_flags = flags + [f for f in add if f not in flags]
        new_conf = conf or d.get("confidence")
        if new_flags == flags and new_conf == d.get("confidence"):
            continue  # idempotent
        if not (bak / f"{dn}.json").exists():
            shutil.copy2(SIG_DIR / f"{dn}.json", bak / f"{dn}.json")
        v = {k: d.get(k) for k in ("function", "subtype", "stopbar_position", "lane_index", "lanes_spanned",
                                   "lane_type", "technology", "phase_diagram")}
        v.update(detector=det, flags=new_flags, confidence=new_conf,
                 confidence_reason=((d.get("confidence_reason") or "") + "; " + why).strip("; "))
        finish(dn, {"detectors": [v], "minutes_visual": r.get("minutes_visual")})
        if new_flags != flags:
            log_rows.append([dn, det, "flags", ";".join(flags), ";".join(new_flags), "vd_audit"])
        if new_conf != d.get("confidence"):
            log_rows.append([dn, det, "confidence", d.get("confidence"), new_conf, "vd_audit"])
    if log_rows:
        with open(CAB / "sweep_changes.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(log_rows)
    log(f"fix: {len(log_rows)} field changes logged")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "audit"
    {"scan": scan, "audit": audit, "card": card, "fix": fix}[cmd]()
