"""Scale-up of the scripted cabinet-print pass to every non-locked signal (note 23), resumable.

    python cab_scale.py extract [--workers 3]  copy PDF/xlsm + cab_pdf.extract per signal (skips done ones)
    python cab_scale.py activity               cab_data counts for every extracted signal (chunks)
    python cab_scale.py dq                     dq_core on the config-labelled channels -> cabinet/dq_config.parquet
    python cab_scale.py records                signals/<DN>.json scripted skeletons (never touches a visual record)
    python cab_scale.py batches                cabinet/batches/batch_NN.txt
    python cab_scale.py all                    the five steps in order

Outputs (all under %DC_WORK%/cabinet/): signals/<DN>.extract.json (also written for failures, with
`error`), scripted_status.csv (one row per signal: status, reasons, class, difficulty),
scripted_failures.csv, dq_config.parquet, signals/<DN>.json (status "scripted" until the visual pass).
Locked signals are never listed (cab_common.signals().locked) and device_id() refuses them anyway.
With `--locked-labels-only` (cab_common) the same steps run on the locked signals ONLY, into
%DC_WORK%/cabinet_locked/: label extraction + activity counts; the `dq` step is skipped (no dq_core on
locked signals), no pilots, and batches are 25 (ordinary) / 10 (336, complex).
"""
from __future__ import annotations

import json
import re
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

from cab_common import CAB, DC_WORK, DET_TO_SLOT, LOCKED_MODE, REPO, SIG_DIR, in_scope, signals

PILOTS = [] if LOCKED_MODE else ["01001", "01005", "01006", "01007", "01009", "01010", "01011", "01014", "01015", "01017"]
FUNCS = ("Advance", "Presence", "Count", "Yellow_Red", "Bike", "Mid", "Other")
DERIVED = re.compile(r"(?i)dummy|FYA|pass ?[0-9]|(^|[^a-z])ped([^a-z]|$)|preempt|(^|[^a-z])EV[A-D]?([^a-z]|$)")
BATCH_DIR = CAB / "batches"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def todo() -> list[str]:
    s = signals()
    return sorted(s[in_scope(s)].DeviceName)


# ------------------------------------------------------------------ 1. extract
def _extract_one(dn: str) -> str:
    import pymupdf
    import cab_pdf as P
    from cab_common import copy_local, resolve
    f = SIG_DIR / f"{dn}.extract.json"
    if f.exists():
        return "skip"
    t0 = time.time()
    rec = dict(DeviceName=dn)
    try:
        rec = P.extract(dn)
        if rec.get("error"):
            pass
        else:
            doc = pymupdf.open(rec["pdf"])
            rec["text_words"] = sum(p["words"] for p in rec["pages"])
            rec["n_images"] = sum(len(p.get_images()) for p in doc)
            rec["other_pdfs"] = [r["Name"] for r in resolve(dn)["pdf"][1:]]
    except Exception as e:  # noqa: BLE001
        msg = repr(e)
        rec.update(error="unreadable" if re.search(r"(?i)pdf|mupdf|format|damaged|password", msg) else "exception",
                   error_detail=msg[:300], traceback=traceback.format_exc()[-800:])
    rec["extract_seconds"] = round(time.time() - t0, 1)
    f.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec.get("error") or "ok"


def extract_all(workers: int = 3):
    dns = [d for d in todo() if not (SIG_DIR / f"{d}.extract.json").exists()]
    log(f"extract: {len(dns)} signals to do")
    n = 0
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_extract_one, dns, chunksize=2):
            n += 1
            if n % 50 == 0:
                log(f"extract {n}/{len(dns)}")
    log("extract done")


# ------------------------------------------------------------------ 2. activity
def activity_all(chunk: int = 150):
    import cab_data
    have = set(pd.read_parquet(CAB / "activity.parquet").DeviceName) if (CAB / "activity.parquet").exists() else set()
    dns = [d for d in todo() if d not in have]
    log(f"activity: {len(dns)} signals to do")
    for i in range(0, len(dns), chunk):
        cab_data.update(dns[i:i + chunk])
        log(f"activity {min(i + chunk, len(dns))}/{len(dns)}")


# ------------------------------------------------------------------ 3. dq on config labels
def _dq_input() -> pd.DataFrame:
    lab = pd.read_parquet(REPO / "research/labels/function_labels_v2.parquet")
    s = signals()
    lab = lab[lab.DeviceName.isin(set(s[~s.locked].DeviceName)) & lab.func5.notna()
              & ~lab.drop_from_use.fillna(False)].copy()
    cf = lab.config_function.fillna("").str.lower()
    ds = lab.description.fillna("").str.replace("%20", " ").str.lower()
    loc = lab.func5.map({"Advance": "advance", "Presence": "stopbar", "Count": "stopbar",
                         "Yellow_Red": "stopbar"}).fillna("other")
    loc[(lab.func5 == "Other") & (cf.str.contains("bike") | ds.str.contains("bike"))] = "bike"
    loc[(lab.func5 == "Other") & cf.str.contains("mid")] = "mid"
    lab["location"] = loc
    nadv = lab[lab.location == "advance"].groupby(["DeviceId", "cfg_phase"]).size()
    lab["lanes_spanned"] = [max(1, nadv.get((r.DeviceId, r.cfg_phase), 1)) if r.location == "mid" else 1
                            for r in lab.itertuples()]
    lab["lane_index"] = pd.NA
    lab["technology"] = "loop"
    lab["function"] = lab.func5
    return lab.rename(columns={"Detector": "detector", "cfg_phase": "phase"})[
        ["DeviceId", "DeviceName", "detector", "phase", "location", "lane_index", "lanes_spanned", "technology", "function"]]


def dq_all():
    if LOCKED_MODE:
        log("dq: skipped in --locked-labels-only mode (dq_core never runs on locked signals)")
        return
    import dq_core as Q
    f = CAB / "dq_config.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["DeviceId"])
    d = _dq_input()
    d = d[~d.DeviceId.isin(set(old.DeviceId))]
    log(f"dq: {d.DeviceId.nunique()} signals, {len(d)} channels to do")
    con = Q.connect(6)
    outs = [old] if len(old) else []
    for k, (dev, g) in enumerate(d.groupby("DeviceId", sort=False)):
        try:
            o = Q.run_signal(con, dev, g.drop(columns=["DeviceId", "DeviceName"]))
        except Exception as e:  # noqa: BLE001
            o = g.drop(columns=["DeviceId", "DeviceName"]).assign(dq_score=np.nan, flags="dq_error", reasons=repr(e)[:200])
        o.insert(0, "DeviceId", dev)
        o.insert(1, "DeviceName", g.DeviceName.iloc[0])
        o.attrs = {}
        outs.append(o)
        if (k + 1) % 50 == 0:
            pd.concat(outs, ignore_index=True).to_parquet(f, index=False)
            log(f"dq {k + 1}")
    out = pd.concat(outs, ignore_index=True)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    out.to_parquet(f, index=False)
    log(f"dq done: {len(out)} channels")


# ------------------------------------------------------------------ 4. records
HINT = re.compile(r"(?i)\b(VIP\w*|Autoscope|Iteris|Vantage|Vector|Gridsmart|Wavetronix|Matrix|SmartSensor|radar|"
                  r"RAD:\w|SDLC|video|camera|CAM\b|thermal|MVP)")


def _detection_hint(pdf) -> str:
    """Detection-technology words in the print's text (video / radar sites leave the input file blank)."""
    if not pdf:
        return ""
    import pymupdf
    try:
        txt = " ".join(p.get_text() for p in pymupdf.open(pdf))
    except Exception:  # noqa: BLE001
        return ""
    from collections import Counter
    c = Counter(m.group(1).upper().rstrip(":") for m in HINT.finditer(txt))
    return ", ".join(f"{k} x{v}" for k, v in c.most_common(4))


def _status(ex: dict, n_active: int, max_phase: float) -> tuple[str, list, str, float]:
    """(status, reasons, batch class, difficulty) for one extract."""
    reasons = []
    if ex.get("error"):
        return ex["error"].replace(" ", "_"), [ex.get("error_detail", "")], "none", 0
    rows = ex.get("input_file", [])
    loop_rows = [r for r in rows if r.get("detector")]
    if ex.get("text_words", 1) < 50:
        st = "no_text_layer_raster" if ex.get("n_images", 0) >= ex.get("n_pages", 1) else "no_text_layer_outlined"
        reasons.append("no usable text layer: everything visual")
    elif not ex.get("cabinet_type"):
        st = "unknown_cabinet_type"
    elif not loop_rows:  # rows on non-detector slots only (I8L / I13 / I14: EV, ped) count as blank
        st = "input_file_blank_zones" if ex.get("zones") else "input_file_blank"
        hint = _detection_hint(ex.get("pdf"))
        reasons.append("no loops landed in the input file (video/radar/SDLC site or unparsed layout)"
                       + (f"; print mentions: {hint}" if hint else ""))
    else:
        st = "ok"
    # several strong diagrams / duplicate slots / phases > 8 -> multi-intersection or complex
    loops = set(ex.get("loops_in_input_file", []))
    strong = [p for p in ex.get("pages", []) if p.get("diagram_hint") and not p.get("wiring_sheet")]
    dup = pd.Series([r["slot"] for r in rows if r.get("layout") != "looptable"]).duplicated().any() if rows else False
    multi = max_phase > 8 or bool(dup)
    if multi:
        reasons.append("multi-intersection or dual input file (phases > 8 or duplicated slots)")
    if len(strong) >= 2:
        reasons.append(f"{len(strong)} intersection-diagram pages ({[p['page'] for p in strong]}): check each")
    cab = ex.get("cabinet_type")
    cls = ("complex" if multi or st.startswith("no_text") or st == "unknown_cabinet_type" or n_active > 40
           or ex.get("n_pages", 0) > 8 else "336" if cab == "336" else "ordinary")
    diff = (n_active + 0.3 * len(ex.get("zones", [])) + (8 if st.startswith("input_file_blank") else 0)
            + 4 * (len(strong) >= 2) + (10 if st.startswith("no_text") else 0))
    return st, reasons, cls, round(float(diff), 1)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else (int(v) if float(v).is_integer() else round(float(v), 3))
    if v is pd.NA:
        return None
    return v


def records():
    s = signals()
    s = s[in_scope(s)]
    off = pd.read_parquet(DC_WORK / "official/labels_official.parquet")
    off = off[off.DeviceName.isin(set(s.DeviceName))]
    off["phase_timing"] = off.call_phase.where(off.call_phase > 0, off.call_overlap.map(lambda v: f"O{v}" if v else None))
    lab = pd.read_parquet(REPO / "research/labels/function_labels_v2.parquet")
    act = pd.read_parquet(CAB / "activity.parquet")
    dq = pd.read_parquet(CAB / "dq_config.parquet") if (CAB / "dq_config.parquet").exists() else pd.DataFrame(columns=["DeviceName"])
    xl = pd.read_parquet(CAB / "xlsm_zones.parquet") if (CAB / "xlsm_zones.parquet").exists() else         pd.DataFrame(columns=["DeviceName", "used", "mt", "phase", "device", "function", "technology"])
    xl = xl[xl.used.astype(bool)]
    G = {k: dict(tuple(g.groupby("DeviceName"))) for k, g in
         (("off", off), ("lab", lab), ("act", act), ("dq", dq), ("xl", xl))}
    stat = []
    for dn in s.DeviceName:
        fe = SIG_DIR / f"{dn}.extract.json"
        if not fe.exists():
            continue
        ex = json.loads(fe.read_text(encoding="utf-8"))
        o = G["off"].get(dn, pd.DataFrame(columns=off.columns)).set_index("Detector")
        la = G["lab"].get(dn, pd.DataFrame(columns=lab.columns)).set_index("Detector")
        a = G["act"].get(dn, pd.DataFrame(columns=act.columns))
        q = G["dq"].get(dn, pd.DataFrame(columns=["detector"])).set_index("detector")
        x = G["xl"].get(dn, pd.DataFrame(columns=xl.columns)).set_index("mt")
        stg = a[a.window == "staging"].set_index("Detector")
        dec = a[a.window == "dec2024"].set_index("Detector")
        n_active = int((stg.n_on > 0).sum()) if len(stg) else int((dec.n_on > 0).sum()) if len(dec) else 0
        max_phase = float(pd.to_numeric(o.call_phase, errors="coerce").max() or 0) if len(o) else 0
        st, reasons, cls, diff = _status(ex, n_active, max_phase)
        cab = ex.get("cabinet_type")
        rows = ex.get("input_file", [])
        zones = ex.get("zones", [])
        inp = {}
        for r in rows:
            if r.get("detector"):
                inp.setdefault(int(r["detector"]), []).append(r)
        zd = {}
        for z in zones:
            zd.setdefault(int(z["detector"]), []).append(z["label"])
        dets = sorted(set(inp) | set(zd) | set(x.index.astype(int)) | set(o.index[o.phase_timing.notna()].astype(int))
                      | set(stg.index[stg.n_on > 0].astype(int)) | set(dec.index[dec.n_on > 0].astype(int))
                      | set(la.index.astype(int)))
        chans = []
        for d in dets:
            desc = str(o.description.get(d, "") or "").replace("%20", " ")
            ch = dict(
                detector=d, slot=DET_TO_SLOT.get(cab, {}).get(d) if cab else None,
                input_file="; ".join(f"{r['slot']}[{r['loops']}] ph{r['phase_marker']}"
                                     + (f" @{r['distance_ft']}ft" if r.get("distance_ft") else "")
                                     + (" BIKE" if r.get("bike") else "") + f" p{r['page']}/{r['layout']}"
                                     for r in inp.get(d, [])) or None,
                zone_label=", ".join(zd.get(d, [])) or None,
                xlsm=(f"ph{x.phase.get(d)} {x.device.get(d)} {x.function.get(d)}" if d in x.index else None),
                phase_timing=_clean(o.phase_timing.get(d)) if d in o.index else None,
                switch_phase=_clean(o.switch_phase.get(d)) if d in o.index and o.switch_phase.get(d) else None,
                add_phases=(o.additional_call_phases.get(d) or None) if d in o.index else None,
                description=desc or None,
                config_function=_clean(la.config_function.get(d)) if d in la.index else None,
                func5_label=_clean(la.func5.get(d)) if d in la.index else None,
                n_on_staging=_clean(stg.n_on.get(d, 0)) if len(stg) else None,
                n_on_dec2024=_clean(dec.n_on.get(d, 0)) if len(dec) else None,
                max5_staging=_clean(stg.max_5min.get(d, 0)) if len(stg) else None,
                dq_score=_clean(q.dq_score.get(d)) if d in q.index else None,
                dq_flags=(q["flags"].get(d) or None) if d in q.index else None,
                dq_reasons=(q["reasons"].get(d) or None) if d in q.index and "reasons" in q else None,
                derived=bool(DERIVED.search(desc)),
            )
            chans.append({k: _clean(v) for k, v in ch.items()})
        cand = sorted(set(inp) | set(zd) | set(x.index.astype(int)))
        detectors = []
        for d in cand:
            r = (inp.get(d) or [{}])[0]
            tech = "loop" if d in inp else ("radar_or_video" if d in zd else
                                           (x.technology.get(d) if d in x.index else None))
            detectors.append(dict(
                detector=d, technology=tech, loops=r.get("loops"), slot=r.get("slot"),
                phase_input_file=r.get("phase_marker"), phase_timing=_clean(o.phase_timing.get(d)) if d in o.index else None,
                distance_ft=r.get("distance_ft") or None,
                # --- visual pass fills these
                phase_diagram=None, function=None, subtype=None, stopbar_position=None, lane_index=None,
                lanes_spanned=None, lane_type=None, flags=["bike_in_input_file"] if r.get("bike") else [],
                confidence=None, confidence_reason="", crop=None))
        ph = sorted({str(int(v)) for v in pd.to_numeric(o.call_phase, errors="coerce").dropna() if v > 0}, key=int)
        win = {w: [str(g.window_start.min()), str(g.window_end.max())] for w, g in a.groupby("window")} if len(a) else {}
        rec = dict(
            DeviceName=dn, DeviceId=ex.get("DeviceId") or s.set_index("DeviceName").DeviceId[dn], status="scripted",
            scripted=dict(status=st, reasons=reasons, batch_class=cls, difficulty=diff,
                          pdf=(ex.get("pdf") or "").replace("/", "\\").split("\\")[-1] or None,
                          pdf_path=ex.get("pdf"), other_pdfs=ex.get("other_pdfs", []),
                          xlsm=(ex.get("xlsm") or "").split("\\")[-1] or None,
                          cabinet_type=cab, n_pages=ex.get("n_pages"), diagram_pages=ex.get("diagram_pages", []),
                          render_dir=str(CAB / "render" / dn), renders=ex.get("renders", {}),
                          thumbs=ex.get("thumbs", []), contact_sheet=ex.get("contact_sheet"),
                          input_file=[{k: r.get(k) for k in ("slot", "detector", "loops", "phase_marker",
                                                             "distance_ft", "bike", "layout", "page")} for r in rows],
                          zones=[{k: z.get(k) for k in ("label", "detector", "device", "page")} for z in zones],
                          n_active_staging=n_active, windows=win),
            channels=chans,
            # --- visual pass (see batches/VISUAL_PASS_INSTRUCTIONS.md); cab_record.finish() validates + saves
            cabinet_type=cab, diagram_page=ex.get("diagram_page"),
            phases={p: dict(n_lanes=None, lanes=None, note="") for p in ph},
            detectors=detectors, explained_other={}, minutes_visual=None)
        stat.append(dict(DeviceName=dn, status=st, batch_class=cls, difficulty=diff, cabinet_type=cab,
                         n_pages=ex.get("n_pages"), n_input_rows=len(rows), n_input_dets=len(inp), n_zones=len(zones),
                         n_active=n_active, xlsm=bool(ex.get("xlsm")), pdf_versions=ex.get("pdf_versions"),
                         reasons=" | ".join(reasons)))
        fr = SIG_DIR / f"{dn}.json"
        old = json.loads(fr.read_text(encoding="utf-8")) if fr.exists() else None
        if dn in PILOTS and old is not None:
            # pilot records predate the stop-bar position rule: add the scripted context, queue a re-read
            old["scripted"], old["channels"] = rec["scripted"], rec["channels"]
            if "status" not in old:
                old["status"] = "reread_stopbar"
            for d in old.get("detectors", []):  # user decision 2026-09-23: Bike / Mid are classes
                if d.get("function") == "Other" and d.get("subtype") in ("mid", "bike"):
                    d["function"] = d["subtype"].capitalize()
            fr.write_text(json.dumps(old, indent=1, default=str), encoding="utf-8")
            continue
        if old is not None and old.get("status", "visual_done") != "scripted":
            continue  # never overwrite a reader's record
        fr.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    st = pd.DataFrame(stat)
    st.to_csv(CAB / "scripted_status.csv", index=False)
    st[~st.status.isin(["ok"])].to_csv(CAB / "scripted_failures.csv", index=False)
    log(f"records: {len(st)}; status {st.status.value_counts().to_dict()}; class {st.batch_class.value_counts().to_dict()}")
    return st


# ------------------------------------------------------------------ 5. batches
def batches():
    st = pd.read_csv(CAB / "scripted_status.csv", dtype={"DeviceName": str})
    st = st[~st.DeviceName.isin(PILOTS) & ~st.status.isin(["no_pdf", "unreadable", "exception"])]
    BATCH_DIR.mkdir(exist_ok=True)
    for f in BATCH_DIR.glob("batch_*.txt"):
        f.unlink()
    nd = st[st.n_active == 0]  # no actuation in either window: no training use -> held back, last
    (BATCH_DIR / "batch_nodata.txt").write_text("".join(f"{r.DeviceName}\tfull\t{r.batch_class}\t{r.status}\n"
                                                        for r in nd.itertuples()), encoding="utf-8")
    st = st[st.n_active > 0]
    out, k = [], 1

    def deal(df, size, first_extra=0):
        """Snake-deal signals sorted by difficulty so every batch gets a similar mix."""
        nonlocal k
        n = max(1, int(np.ceil((len(df) + first_extra) / size)))
        bins = [[] for _ in range(n)]
        cap = [size - (first_extra if i == 0 else 0) for i in range(n)]
        order = df.sort_values("difficulty", ascending=False)
        i, step = 0, 1
        for r in order.itertuples():
            tries = 0
            while len(bins[i]) >= cap[i] and tries < 2 * n:
                i, step = _next(i, step, n)
                tries += 1
            bins[i].append(r)
            i, step = _next(i, step, n)
        res = []
        for b in bins:
            res.append((k, b))
            k += 1
        return res

    ordinary = deal(st[st.batch_class == "ordinary"], 25, first_extra=len(PILOTS))
    b336 = deal(st[st.batch_class == "336"], 10)
    comp = deal(st[st.batch_class == "complex"], 10)
    for (num, b), cls in [(x, "ordinary") for x in ordinary] + [(x, "336") for x in b336] + [(x, "complex") for x in comp]:
        lines = []
        if num == 1:
            lines += [f"{p}\treread_stopbar\tpilot" for p in PILOTS]
        lines += [f"{r.DeviceName}\tfull\t{cls}\t{r.status}" for r in sorted(b, key=lambda r: r.DeviceName)]
        (BATCH_DIR / f"batch_{num:02d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        out.append(dict(batch=num, cls=cls, n=len(lines), difficulty=round(sum(r.difficulty for r in b), 1)))
    o = pd.DataFrame(out)
    o.to_csv(BATCH_DIR / "batches_index.csv", index=False)
    log(o.groupby("cls").agg(batches=("batch", "size"), sizes=("n", lambda v: sorted(set(v))),
                             diff_min=("difficulty", "min"), diff_max=("difficulty", "max")).to_string())
    return o


def _next(i, step, n):
    j = i + step
    if 0 <= j < n:
        return j, step
    return i, -step  # bounce: snake order


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    steps = {"extract": lambda: extract_all(w), "activity": activity_all, "dq": dq_all, "records": records,
             "batches": batches}
    for c in (list(steps) if cmd == "all" else [cmd]):
        steps[c]()
