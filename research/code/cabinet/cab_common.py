"""Shared paths and lookups for cabinet-print label mining.

Everything agency-specific (share path, listing, PDFs, per-signal records) lives under
%DC_WORK%/cabinet/, never in the repo. The share is read-only: files are copied, never touched.

Locked hold-out signals (TEST + NEWTEST) are refused by default. Label extraction for them (user
approval 2026-09-24: labels only, no model scoring, no metrics) is an explicit opt-in: pass
`--locked-labels-only` on any cab_* command line or set DC_CAB_LOCKED_LABELS_ONLY=1. In that mode
  * every output goes to the SEPARATE store %DC_WORK%/cabinet_locked/ (signals/, pdf/, render/, crops/,
    batches/, activity.parquet ...), never to %DC_WORK%/cabinet/ or research/labels/;
  * only locked signals are accepted (non-locked ones are refused, so the two stores never mix);
  * model-facing scripts (cab_final's training path, label_check) refuse to run (`forbid_locked_mode`);
    the fix_* rules and `cab_final.py --locked-consolidate` (labels only, note 36) do run there.
    The share listing is still read from %DC_WORK%/cabinet/share_listing.csv.
`locked_ids()` stays the 186 ORIGINAL hold-outs (store membership). The hold-out itself is
%DC_WORK%/official/locked_v2.csv since 2026-09-24 (half of NEWTEST released into training, note 36).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DC_WORK = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
LOCKED_FLAG, LOCKED_ENV = "--locked-labels-only", "DC_CAB_LOCKED_LABELS_ONLY"
if LOCKED_FLAG in sys.argv:  # CLI opt-in; the env var carries it into worker processes
    sys.argv.remove(LOCKED_FLAG)
    os.environ[LOCKED_ENV] = "1"
LOCKED_MODE = os.environ.get(LOCKED_ENV, "") == "1"
LISTING = DC_WORK / "cabinet" / "share_listing.csv"  # cached listing of the read-only share (both modes)
# note 80 (label pass v4l): DC_CAB_STORE=<dir under DC_WORK> points the TRAINING tools at a separate copy of the training
# store (e.g. cabinet_v4l) that also holds the 71 released NEWTEST records. In that mode the hold-out is locked_v2.csv
# (115 signals) - the released signals are ordinary training signals - and the outputs are versioned (cab_final writes
# function_labels_<DC_LABELS_TAG>.parquet, user lists go to <store>/lists/, never to review/).
STORE = os.environ.get("DC_CAB_STORE", "")
if STORE and LOCKED_MODE:
    raise PermissionError("DC_CAB_STORE (training-store copy) and the locked-labels mode never mix")
CAB = DC_WORK / ("cabinet_locked" if LOCKED_MODE else (STORE or "cabinet"))
PDF_DIR, XLSM_DIR, SIG_DIR, RENDER_DIR = CAB / "pdf", CAB / "xlsm", CAB / "signals", CAB / "render"
for _d in (PDF_DIR, XLSM_DIR, SIG_DIR, RENDER_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _mapping_tables():
    """DEFAULT_PHASE and SLOT_TO_DET from the python block of standard_detector_mapping.md (one copy)."""
    txt = (REPO / "research/notes/standard_detector_mapping.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", txt, re.S).group(1)
    ns: dict = {}
    exec(code, ns)
    return ns["DEFAULT_PHASE"], ns["SLOT_TO_DET"]


DEFAULT_PHASE, SLOT_TO_DET = _mapping_tables()
DET_TO_SLOT = {cab: {d: s for s, d in m.items()} for cab, m in SLOT_TO_DET.items()}


def locked_ids() -> set[str]:
    test = set(pd.read_csv(DC_WORK / "data/splits/test_config.csv").DeviceId)
    new = set(pd.read_csv(DC_WORK / "official/newtest_signals.csv").DeviceId)
    if STORE:  # v4l store: the hold-out as it stands (43 TEST + 72 NEWTEST); released NEWTEST are training signals
        v2 = set(pd.read_csv(DC_WORK / "official/locked_v2.csv").DeviceId)
        rel = set(pd.read_csv(DC_WORK / "official/newtest_released.csv").DeviceId)
        assert test <= v2 and v2 <= test | new and not v2 & rel and v2 | rel == test | new, "hold-out files inconsistent"
        return v2
    return test | new


def signals() -> pd.DataFrame:
    """DeviceId <-> DeviceName for all signals, with a `locked` flag (TEST or NEWTEST).

    Locked ids missing from the plans table take their name from the plans-download failure list."""
    pl = pd.read_parquet(REPO / "data/detector_plans.parquet", columns=["DeviceId", "DeviceName"]).drop_duplicates()
    lk = locked_ids()
    fail = REPO / "data/detector_plans_failures.csv"
    if fail.exists():
        f = pd.read_csv(fail, usecols=["DeviceId", "DeviceName"], dtype=str).drop_duplicates()
        f = f[f.DeviceId.isin(lk - set(pl.DeviceId))]
        pl = pd.concat([pl, f], ignore_index=True)
    pl["locked"] = pl.DeviceId.isin(lk)
    return pl.reset_index(drop=True)


def in_scope(s: pd.DataFrame) -> pd.Series:
    """Rows this mode may process: non-locked by default, locked only with --locked-labels-only."""
    return s.locked if LOCKED_MODE else ~s.locked


def device_id(dn: str) -> str:
    s = signals()
    r = s[s.DeviceName == dn]
    if r.empty:
        raise KeyError(dn)
    if r.locked.iloc[0] and not LOCKED_MODE:
        raise PermissionError(f"{dn} is a locked hold-out signal: never processed "
                              f"(label extraction only: {LOCKED_FLAG}, separate store)")
    if LOCKED_MODE and not r.locked.iloc[0]:
        raise PermissionError(f"{dn} is not a locked signal: {LOCKED_FLAG} processes locked signals only")
    return r.DeviceId.iloc[0]


def forbid_locked_mode(what: str):
    """Scripts that feed training labels or models never run in locked-label mode."""
    if LOCKED_MODE:
        raise PermissionError(f"{what} never runs with {LOCKED_FLAG} / {LOCKED_ENV}=1 (labels-only store)")


def listing() -> pd.DataFrame:
    L = pd.read_csv(LISTING, encoding="utf-8-sig")
    L["ext"] = L.Name.str.extract(r"\.([A-Za-z0-9]+)$")[0].str.lower()
    L["dn"] = L.Name.str.extract(r"^\s*([0-9A-Za-z]{5})")[0].str.upper()
    L["mtime"] = pd.to_datetime(L.LastWriteTime, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    return L


def resolve(dn: str, L: pd.DataFrame | None = None) -> dict:
    """Candidate PDF / xlsm files for a DeviceName, newest first. Falls back to a dropped leading zero."""
    L = listing() if L is None else L
    out = {}
    for ext in ("pdf", "xlsm"):
        c = L[(L.ext == ext) & (L.dn == dn.upper())]
        if c.empty and dn.startswith("0"):
            c = L[(L.ext == ext) & L.Name.str.upper().str.match(re.escape(dn[1:].upper()) + r"[^0-9]")]
        out[ext] = c.sort_values("mtime", ascending=False)[["Name", "FullName", "mtime", "Length"]].to_dict("records")
    return out


def copy_local(dn: str, L: pd.DataFrame | None = None) -> dict:
    """Copy the newest PDF (and xlsm) of a signal from the share to %DC_WORK%/cabinet (read-only use)."""
    device_id(dn)  # refuses locked signals
    res = resolve(dn, L)
    got = {"pdf": None, "xlsm": None, "pdf_versions": len(res["pdf"]), "xlsm_versions": len(res["xlsm"])}
    for ext, d in (("pdf", PDF_DIR), ("xlsm", XLSM_DIR)):
        if not res[ext]:
            continue
        r = res[ext][0]
        dst = d / r["Name"]
        if not dst.exists() or dst.stat().st_size != int(r["Length"]):
            shutil.copy2(r["FullName"], dst)
        got[ext] = str(dst)
    return got
