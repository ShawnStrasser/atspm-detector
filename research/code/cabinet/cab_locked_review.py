"""Low-priority review list for the user (note 36): locked-store print readings flagged by the reader.

    python cab_locked_review.py      -> review/locked_print_readings_to_check.xlsx

Labels only (no model output exists for these signals and none is read): every detector of
research/labels/function_labels_locked_v1.parquet whose print reading carries needs_review, label_disagrees or
phase_print_vs_timing. One row per detector, grouped by signal (signals that stay locked first, then the released
ones); plain columns; the crop is the reader's evidence image in %DC_WORK%/cabinet_locked/crops/.
"""
from __future__ import annotations

import re

import pandas as pd

from cab_common import DC_WORK, REPO

FLAGS = {"needs_review": "reader unsure", "label_disagrees": "print and config disagree",
         "phase_print_vs_timing": "print phase differs from the controller timing"}
OUT = REPO / "review" / "locked_print_readings_to_check.xlsx"
CROPS = DC_WORK / "cabinet_locked" / "crops"


def phase_txt(r) -> str:
    pr = "" if pd.isna(r.phase_diagram) else f"P{int(r.phase_diagram)}"
    tm = "" if pd.isna(r.phase_target) or r.phase_target is None else str(r.phase_target)
    tm = re.sub(r"^(\d+)$", r"P\1", tm)
    if pr and tm and pr != tm:
        return f"print {pr} / timing {tm}"
    return pr or tm or "?"


def main():
    lk = pd.read_parquet(REPO / "research/labels/function_labels_locked_v1.parquet")
    fl = lk.print_flags.fillna("")
    m = fl.str.contains("|".join(rf"\b{f}\b" for f in FLAGS))
    x = lk[m].copy()
    why = []
    for f, reason in zip(x.print_flags.fillna(""), x.confidence_reason.fillna("")):
        w = [FLAGS[k] for k in FLAGS if re.search(rf"\b{k}\b", f)]
        why.append("; ".join(w) + (f" — reader: {reason[:200]}" if reason else ""))
    crop = []
    for dn, d, c in zip(x.DeviceName, x.detector, x.crop):
        name = c.split("\\")[-1].split("/")[-1] if isinstance(c, str) and c else f"{dn}_d{d}.png"
        crop.append(name if (CROPS / name).exists() else "")
    status = x.released_from_newtest.map({True: "released to training", False: "stays locked"})
    out = pd.DataFrame({
        "status": status, "signal": x.DeviceName, "detector": x.detector,
        "phase (print vs timing)": [phase_txt(r) for r in x.itertuples()],
        "label": x.function.fillna("(none)") + " (" + x.print_confidence.fillna("?") + ")",
        "config said": x.func7_v2.fillna(""), "technology": x.technology.fillna(""),
        "why flagged": why, "crop": crop, "correct label (fill in)": "", "comment": ""})
    out["_o"] = out.status.eq("released to training")
    out = out.sort_values(["_o", "signal", "detector"]).drop(columns="_o").reset_index(drop=True)
    sig = (out.groupby(["status", "signal"]).size().rename("flagged detectors").reset_index()
           .sort_values(["status", "signal"], ascending=[False, True]))
    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="detectors", index=False)
        ws = xw.sheets["detectors"]
        for col, wd in zip("ABCDEFGHIJK", (20, 9, 9, 24, 20, 12, 12, 90, 22, 20, 30)):
            ws.column_dimensions[col].width = wd
        ws.freeze_panes = "C2"
        ws.auto_filter.ref = ws.dimensions
        sig.to_excel(xw, sheet_name="signals", index=False)
        xw.sheets["signals"].column_dimensions["A"].width = 22
    print(f"{len(out)} detectors on {out.signal.nunique()} signals -> {OUT}")
    print(out.groupby("status").agg(detectors=("detector", "size"), signals=("signal", "nunique")).to_string())
    print({k: int(fl[m].str.contains(rf"\b{k}\b").sum()) for k in FLAGS})


if __name__ == "__main__":
    main()
