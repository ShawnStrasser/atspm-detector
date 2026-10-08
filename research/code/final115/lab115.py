"""Note 115: v6b vs v7 function answers on the parity cases (in-sample training signals: a sanity check of the size of
the note-114 change, NOT an accuracy estimate).  ATSPM score without stack credit: error = labelled ATSPM class not
predicted, or a non-ATSPM label predicted as an ATSPM class."""
import json, os, sys
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
import pandas as pd
OUT = str(W / r"s115\parity\out")
ATSPM = {"Advance", "Presence", "Count", "Yellow_Red"}
lab = pd.read_parquet(str(Path(__file__).resolve().parents[2] / "labels" / "function_labels_v4q.parquet"),
                      columns=["DeviceId", "detector", "function"])
lab = lab.assign(DeviceId=lab.DeviceId.str.lower(), Detector=lab.detector.astype(int)).dropna(subset=["function"])
lab = lab[lab.function != "?"].drop_duplicates(["DeviceId", "Detector"])
res = {}
for prof in ("full", "le2h"):
    a = pd.read_parquet(f"{OUT}/v6b_{prof}_det.parquet")
    b = pd.read_parquet(f"{OUT}/{sys.argv[1]}_{prof}_det.parquet")
    m = a.merge(b, on=["case", "DeviceId", "Detector"], suffixes=("_6b", "_7"))
    m = m.assign(dl=m.DeviceId.str.lower()).merge(lab.rename(columns={"DeviceId": "dl"}), on=["dl", "Detector"])
    m = m[m.function_pred_6b.notna()]

    def ok(p, f):
        return ((f.isin(ATSPM) & (p == f)) | (~f.isin(ATSPM) & ~p.isin(ATSPM)))
    for L in ("m30", "h3", "h24"):
        s = m[m.case.str.endswith(L)]
        o6, o7 = ok(s.function_pred_6b, s.function), ok(s.function_pred_7, s.function)
        res[f"{prof}_{L}"] = {"n": int(len(s)), "changed": int((s.function_pred_6b != s.function_pred_7).sum()),
                              "acc_v6b": round(float(o6.mean()), 4), "acc_v7": round(float(o7.mean()), 4),
                              "v7_fixes": int((~o6 & o7).sum()), "v7_breaks": int((o6 & ~o7).sum())}
print(json.dumps(res, indent=1))
json.dump(res, open(str(W / rf"s115\parity\lab115_{sys.argv[1]}.json"), "w"), indent=1)
