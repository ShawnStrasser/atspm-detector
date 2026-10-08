"""Parse the "Detection Configuration" workbooks (.xlsm) that sit next to some cabinet prints.

Sheet `ZoneConfigurationTable`, one row per controller detector channel ("MT #"):
  A  'x' = channel used by an SDLC device (radar / video zone)   G  'x' = input-file slot used (loops)
  H  slot reference (I4U ...)   I  device letter (A..H = radar/camera unit) or 'Loop'
  J  phase   L  function / loop text ("Loop 3,4", "Bike Loop 5", "P 0-20'", "CO*", "YR*", "A*", "Count")
  M  MT # = detector channel   N  BIU number or input slot   O  software zone (sometimes lane text)
  B/C/D  flags: device / phase / function differ from the template default.
Rows with neither A nor G marked are unused template rows.

    python cab_xlsm.py            -> copies every non-locked signal's newest workbook, writes
                                     %DC_WORK%/cabinet/xlsm_zones.parquet and prints an agreement summary
"""
from __future__ import annotations

import re
import sys

import openpyxl
import pandas as pd

from cab_common import CAB, DC_WORK, copy_local, in_scope, listing, signals

COLS = dict(used_mt=0, dev_diff=1, phase_diff=2, func_diff=3, default_phase=4, default_func=5, used_slot=6,
            slot=7, device=8, phase=9, phase_device=10, function=11, mt=12, biu=13, zone=14, channel=15)


def parse(path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = next((w for w in wb.worksheets if w.title.lower().startswith("zoneconfig")), None)
    if ws is None:
        return pd.DataFrame()
    rows, started = [], False
    for r in ws.iter_rows(values_only=True):
        r = list(r) + [None] * 16
        if not started:
            started = isinstance(r[12], str) and r[12].strip().upper().startswith("MT")
            continue
        if not isinstance(r[12], (int, float)):
            continue
        rows.append({k: r[i] for k, i in COLS.items()})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["mt"] = d.mt.astype(int)
    for c in ("used_mt", "used_slot"):
        d[c] = d[c].astype(str).str.strip().str.lower().eq("x")
    d["used"] = d.used_mt | d.used_slot
    d["function"] = d.function.astype("string").str.strip()
    d["loops"] = d.function.str.extract(r"(?i)loops?\s*([\d ,&-]+)")[0].str.replace(r"\s", "", regex=True)
    d["bike"] = d.function.str.contains("bike", case=False, na=False)
    d["technology"] = d.device.astype(str).str.lower().map(lambda v: "loop" if v.startswith("loop") else None)
    d.loc[d.technology.isna() & d.used_mt, "technology"] = "radar_or_video"
    d.loc[d.technology.isna() & d.used_slot, "technology"] = "loop"
    d["zone_text"] = d.zone.where(d.zone.map(lambda v: isinstance(v, str)))
    return d


def run_all():
    s = signals()
    s = s[in_scope(s)]  # non-locked; locked only with --locked-labels-only (separate store)
    L = listing()
    out = []
    for dn in s.DeviceName:
        if not (L[(L.ext == "xlsm") & (L.dn == dn.upper())]).shape[0]:
            continue
        got = copy_local(dn, L)
        if not got["xlsm"]:
            continue
        try:
            d = parse(got["xlsm"])
        except Exception as e:  # noqa: BLE001
            print(dn, "unreadable", e)
            continue
        if d.empty:
            print(dn, "no ZoneConfigurationTable")
            continue
        d.insert(0, "DeviceName", dn)
        d["xlsm"] = got["xlsm"].split("\\")[-1].split("/")[-1]
        out.append(d)
    if not out:
        print("no readable workbook")
        return pd.DataFrame()
    z = pd.concat(out, ignore_index=True)
    for c in z.columns:
        if z[c].dtype == object:
            z[c] = z[c].map(lambda v: None if v is None else str(v))
    z.to_parquet(CAB / "xlsm_zones.parquet", index=False)
    return z


def agreement(z: pd.DataFrame) -> pd.DataFrame:
    """Per workbook: used channels vs the controller timing and vs hi-res activity."""
    off = pd.read_parquet(DC_WORK / "official/labels_official.parquet")
    off["phase_timing"] = off.call_phase.where(off.call_phase > 0, off.call_overlap)
    off["active"] = (off.n_on_dec2024 + off.n_on_staging) > 0
    u = z[z.used].merge(off[["DeviceName", "Detector", "phase_timing", "active", "description"]],
                        left_on=["DeviceName", "mt"], right_on=["DeviceName", "Detector"], how="left")
    u["phase"] = pd.to_numeric(u.phase, errors="coerce")
    u["phase_ok"] = u.phase == u.phase_timing
    act = off[off.active].groupby("DeviceName").Detector.apply(set)
    per = u.groupby("DeviceName").agg(n_used=("mt", "size"), phase_ok=("phase_ok", "mean"), active=("active", "mean"))
    per["active_not_in_xlsm"] = [len(act.get(dn, set()) - set(u[u.DeviceName == dn].mt)) for dn in per.index]
    return per, u


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            print(parse(f).query("used").to_string())
    else:
        z = run_all()
        per, u = agreement(z)
        print(per.describe().to_string())
        print("signals", len(per), "| used channels", len(u), "| phase==timing", round(u.phase_ok.mean(), 3),
              "| used channel active", round(u.active.mean(), 3),
              "| signals with every active channel in the workbook", int((per.active_not_in_xlsm == 0).sum()))
        per.to_csv(CAB / "xlsm_agreement.csv")
