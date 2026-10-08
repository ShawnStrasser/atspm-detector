"""A1 (part 3) -- the round-2 confident function-disagreement list, ALL labelled signals.

Truth = `research/labels/function_labels_v2.parquet` (the user's corrections already
folded in, dropped rows removed).  Model opinion:
  * DEV / NEWTRAIN signals -> the stage-11 6-fold out-of-fold prediction (honest)
  * TEST / NEWTEST signals -> the frozen final_v2 predictions already on disk
Rows the user reviewed in round 1 are excluded.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

# repository root: DC_REPO, else three levels up when run from research/code/trackA
REPO = Path(os.environ.get("DC_REPO") or Path(__file__).resolve().parents[3])
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PRED = DCW / "preds" / "final_test_DO_NOT_USE" / "v2"
CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
PCOLS = [f"p_{c.lower()}" for c in CLASSES5]
TH = 0.80
OUT = REPO / "review" / "function_disagreements_round2.xlsx"


def oof_part() -> pd.DataFrame:
    o = pd.read_parquet(DCW / "function_v4" / "function_oof_v4_full.parquet")
    o["DeviceId"] = o.DeviceId.str.lower()
    o["Detector"] = o.Detector.astype(int)
    # one row per detector: prefer the Sept-2026 sample (contemporaneous with the config)
    o["ord"] = np.where(o.period == "stg", 0, 1)
    o = o.sort_values(["DeviceId", "Detector", "ord"]).groupby(
        ["DeviceId", "Detector"], as_index=False).first()
    o = o.rename(columns={"pred5": "model_function", "det_n_on": "n_actuations",
                          "pred_phase": "model_phase", "top_prob": "model_phase_prob"})
    o["model_function_prob"] = o[PCOLS].max(1)
    o["src"] = np.where(o.split == "NEWTRAIN", "NEWTRAIN", "DEV") + " (out-of-fold)"
    return o[["DeviceId", "Detector", "model_function", "model_function_prob",
              "model_phase", "model_phase_prob", "n_actuations", "src"] + PCOLS]


def locked_part() -> pd.DataFrame:
    parts = []
    for tag, setname in (("NEWTEST", "NEWTEST"), ("TEST", "TESTSTG")):
        f = PRED / f"{setname}_v2_full.parquet"
        p = pd.read_parquet(f)
        p["DeviceId"] = p.DeviceId.str.lower()
        p["Detector"] = p.Detector.astype(int)
        p = p.rename(columns={"function_guess": "model_function",
                              "function_guess_prob": "model_function_prob",
                              "phase_guess": "model_phase",
                              "phase_guess_prob": "model_phase_prob"})
        p["src"] = f"{tag} locked (frozen final_v2)"
        parts.append(p[["DeviceId", "Detector", "model_function",
                        "model_function_prob", "model_phase", "model_phase_prob",
                        "n_actuations", "src"] + PCOLS])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    lab = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v2.parquet")
    lab = lab[lab.func5.notna()]

    pred = pd.concat([oof_part(), locked_part()], ignore_index=True)
    pred = pred.drop_duplicates(["DeviceId", "Detector"])

    off = pd.read_parquet(DCW / "official" / "labels_official.parquet")
    off = off[off.target_type == "phase"].copy()
    off["DeviceId"] = off.DeviceId.str.lower()
    off["Detector"] = off.Detector.astype(int)
    off = off[["DeviceId", "Detector", "target_num", "delay", "extend",
               "description"]].rename(columns={"target_num": "timing_phase",
                                               "description": "timing_description"})

    d = lab.merge(pred, on=["DeviceId", "Detector"], how="inner").merge(
        off.drop_duplicates(["DeviceId", "Detector"]), on=["DeviceId", "Detector"],
        how="left")
    d = d[d.model_function.notna() & d.model_function_prob.notna()]
    print(f"{len(d):,} labelled detectors with an opinion "
          f"({d.DeviceId.nunique()} signals)")

    bad = d[(d.model_function != d.func5) & (d.model_function_prob >= TH) &
            (~d.reviewed)].copy()
    print(f"{len(bad):,} confident (p>={TH}) disagreements on "
          f"{bad.DeviceId.nunique()} signals")
    print(bad.src.value_counts().to_string())
    print(pd.crosstab(bad.func5, bad.model_function).to_string())

    pmap = {c: f"p_{c.lower()}" for c in CLASSES5}
    bad["label_prob"] = [r[pmap[c]] for c, (_, r) in zip(bad.func5, bad.iterrows())]
    bad["notes"] = ["label class scored " + f"{lp:.2f}" +
                    ("" if not isinstance(td, str) or not td.strip()
                     else f"; tech description: '{td}'") +
                    ("" if not ex or np.isnan(ex) or ex == 0 else f"; extend {ex:.0f}s")
                    for lp, td, ex in zip(bad.label_prob, bad.timing_description,
                                          bad.extend.fillna(0))]
    bad["signal_max_prob"] = bad.groupby("DeviceId").model_function_prob.transform("max")
    bad["correct_function"] = ""
    bad["comment"] = ""
    bad["signal_set"] = np.where(bad.locked == "", "train/dev", bad.locked + " (locked)")

    out = bad.rename(columns={"config_function": "label_raw", "func5": "label_function",
                              "cfg_phase": "config_phase"})[
        ["DeviceName", "DeviceId", "Detector", "signal_set", "label_raw",
         "label_function", "model_function", "model_function_prob", "signal_max_prob",
         "p_advance", "p_presence", "p_count", "p_yellow_red", "p_other",
         "config_phase", "timing_phase", "model_phase", "model_phase_prob",
         "n_actuations", "delay", "extend", "timing_description", "notes",
         "correct_function", "comment"]]
    out = out.sort_values(["signal_max_prob", "DeviceName", "model_function_prob"],
                          ascending=[False, True, False]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="disagreements", index=False)
        ws = xw.sheets["disagreements"]
        ws.freeze_panes = "A2"
        for i, c in enumerate(out.columns, 1):
            w = min(42, max(10, int(out[c].astype(str).str.len().quantile(0.95)) + 2,
                            len(c) + 2))
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    print(f"wrote {OUT}: {len(out)} rows, {out.DeviceId.nunique()} signals")


if __name__ == "__main__":
    main()
