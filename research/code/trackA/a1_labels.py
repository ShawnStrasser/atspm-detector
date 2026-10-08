"""A1 (part 1) -- fold the user's review into the function label table.

Rules agreed with the user:
  correct_function blank        -> keep the config label
  correct_function '?'          -> drop the detector from function scoring AND training
  'Bike' / 'Bike Loop' /
  'Departure'                   -> Other  (not classified classes, behave like Other)
  anything else                 -> that class is the truth

Output: research/labels/function_labels_v2.parquet (+ README.md)
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# repository root: DC_REPO, else three levels up when run from research/code/trackA
REPO = Path(os.environ.get("DC_REPO") or Path(__file__).resolve().parents[3])
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
WORK = DCW / "trackA"
CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]

REVIEW = REPO / "review" / "function_misclassified_locked143_REVIEWED.xlsx"
CFG = DCW / "data" / "labels" / "detector_config_current.parquet"
LMAP = DCW / "data" / "labels" / "function_label_map_v2.csv"
OUT = REPO / "research" / "labels" / "function_labels_v2.parquet"

# correct_function strings that are real classes, and the ones that map to Other
TO_OTHER = {"bike", "bike loop", "departure"}


def base_labels() -> pd.DataFrame:
    cfg = pd.read_parquet(CFG)
    cfg["DeviceId"] = cfg.DeviceId.str.lower()
    cfg["Detector"] = cfg.Detector.astype(int)
    mp = pd.read_csv(LMAP)
    m = dict(zip(mp.raw_key.astype(str), mp.std_function))
    cfg["func5_config"] = cfg.Function.astype(str).str.strip().str.lower().map(m)
    bad = cfg[cfg.func5_config.isna()].Function.unique()
    if len(bad):
        raise ValueError(f"unmapped function strings: {bad[:20]}")
    cfg = cfg[cfg.Detector.between(1, 64)]
    return (cfg[["DeviceId", "Detector", "Phase", "Function", "func5_config"]]
            .rename(columns={"Phase": "cfg_phase", "Function": "config_function"})
            .reset_index(drop=True))


def main() -> None:
    lab = base_labels()

    off = pd.read_parquet(DCW / "official" / "labels_official.parquet")[
        ["DeviceId", "DeviceName", "Detector", "description"]]
    off["DeviceId"] = off.DeviceId.str.lower()
    off["Detector"] = off.Detector.astype(int)
    lab = lab.merge(off.drop_duplicates(["DeviceId", "Detector"]),
                    on=["DeviceId", "Detector"], how="left")

    test = set(pd.read_csv(REPO / "data" / "splits" / "test_config.csv")
               .DeviceId.str.lower())
    newtest = set(pd.read_csv(DCW / "official" / "newtest_signals.csv")
                  .DeviceId.str.lower())
    lab["locked"] = np.where(lab.DeviceId.isin(newtest), "NEWTEST",
                             np.where(lab.DeviceId.isin(test), "TEST", ""))

    rv = pd.read_excel(REVIEW)
    rv["DeviceId"] = rv.DeviceId.str.lower()
    rv["Detector"] = rv.Detector.astype(int)
    rv["cf"] = rv.correct_function.astype(str).str.strip()
    rv.loc[rv.correct_function.isna(), "cf"] = ""

    def resolve(row):
        c = row.cf
        if c == "":
            return pd.Series({"action": "keep", "func5": np.nan})
        if c == "?":
            return pd.Series({"action": "drop", "func5": np.nan})
        if c.lower() in TO_OTHER:
            return pd.Series({"action": "to_other", "func5": "Other"})
        if c not in CLASSES5:
            raise ValueError(f"unexpected correct_function {c!r}")
        return pd.Series({"action": "class", "func5": c})

    rv = pd.concat([rv, rv.apply(resolve, axis=1)], axis=1)
    rv_small = rv[["DeviceId", "Detector", "cf", "action", "func5", "comment",
                   "label_function", "model_function"]].rename(
        columns={"func5": "func5_review", "cf": "review_raw",
                 "comment": "review_comment"})

    lab = lab.merge(rv_small, on=["DeviceId", "Detector"], how="left")
    lab["reviewed"] = lab.action.notna()
    lab["drop_from_use"] = lab.action.eq("drop")
    lab["func5"] = np.where(lab.func5_review.notna(), lab.func5_review,
                            lab.func5_config)
    lab.loc[lab.drop_from_use, "func5"] = np.nan
    lab["label_source"] = np.select(
        [lab.action.eq("drop"), lab.action.isna(), lab.action.eq("keep")],
        ["review_dropped", "config", "review_confirmed_config"],
        default="review_corrected")
    # a correction that happens to reproduce the config label is a confirmation
    same = lab.reviewed & lab.func5.notna() & (lab.func5 == lab.func5_config)
    lab.loc[same & lab.label_source.eq("review_corrected"),
            "label_source"] = "review_confirmed"

    cols = ["DeviceId", "DeviceName", "Detector", "cfg_phase", "config_function",
            "func5_config", "func5", "label_source", "reviewed", "drop_from_use",
            "review_raw", "review_comment", "description", "locked"]
    lab = lab[cols].sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(OUT, index=False)

    n_rev = int(lab.reviewed.sum())
    changed = lab.reviewed & lab.func5.notna() & (lab.func5 != lab.func5_config)
    summ = {
        "n_rows": int(len(lab)), "n_signals": int(lab.DeviceId.nunique()),
        "n_reviewed": n_rev,
        "n_changed": int(changed.sum()),
        "n_dropped": int(lab.drop_from_use.sum()),
        "n_confirmed": int((lab.reviewed & ~lab.drop_from_use & ~changed).sum()),
        "label_source_counts": lab.label_source.value_counts().to_dict(),
        "class_counts_v2": lab.func5.value_counts().to_dict(),
        "class_counts_config": lab.func5_config.value_counts().to_dict(),
        "changed_confusion": pd.crosstab(lab.loc[changed, "func5_config"],
                                         lab.loc[changed, "func5"]).to_dict(),
        "reviewed_by_locked": lab[lab.reviewed].locked.value_counts().to_dict(),
    }
    json.dump(summ, open(WORK / "a1_labels_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))
    print(f"\nwrote {OUT}  {lab.shape}")


if __name__ == "__main__":
    main()
