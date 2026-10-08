"""Lanes redo -- oracle: the per-lane decode with the TRUE lanes (RL/CL/LL text) on the
phases where every detector is lane-tagged and one sensor unit (or none) is named.
If perfect lanes do not help here, no lane inference can.  Full window, all folds.

Needs `a3_lanes.py` (the A3 stage) next to it.

    python lr5_oracle.py
"""
import json
import numpy as np, pandas as pd
import lr_common as C
import a3_lanes as A3

CL5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
keys = pd.read_parquet(C.WORK / "a2_oof_keys.parquet")
Pr = np.load(C.WORK / "a2_oof_base_expert_mean.npy")
m = (keys.wgroup == "full").to_numpy()
K, PK = keys[m].reset_index(drop=True), Pr[m]
K["DeviceId"] = K.DeviceId.str.lower()
lab = C.labelled_detectors()[["DeviceId", "Detector", "lane", "unit"]]
lab["Detector"] = lab.Detector.astype(K.Detector.dtype)
K = K.merge(lab.drop_duplicates(["DeviceId", "Detector"]), on=["DeviceId", "Detector"],
            how="left")
g = K.groupby(["DeviceId", "period", "pred_phase"])
ok = (g.lane.transform(lambda s: s.notna().all()) & g.Detector.transform("size").gt(1)
      & g.unit.transform("nunique").le(1)).to_numpy()
idx = np.flatnonzero(ok)
S, PS = K.iloc[idx].reset_index(drop=True), PK[idx]
yt = S.func5.to_numpy()
apc = np.isin(yt, C.APC)
res = {"n_det": int(len(S)), "n_phases": int(S.groupby(["DeviceId", "period",
                                                         "pred_phase"]).ngroups)}
for tag, pred in [("argmax", PS.argmax(1))] + [
        (f"true_lane/{mode}", None) for mode in ("lane", "cap")]:
    if pred is None:
        mode = tag.split("/")[1]
        pred = PS.argmax(1).copy()
        for _, ii in S.groupby(["DeviceId", "period", "pred_phase"]).indices.items():
            pred[ii] = A3.assign(PS[ii], pd.factorize(S.lane.to_numpy()[ii])[0], mode)
    p = np.array(CL5)[pred]
    res[tag] = {"acc5": round(float((yt == p).mean()), 4),
                "accAPC": round(float((yt[apc] == p[apc]).mean()), 4)}
# do the labels themselves obey <= 1 per true lane here?
v = S.groupby(["DeviceId", "period", "pred_phase", "lane"]).func5.agg(
    lambda s: any((s == c).sum() > 1 for c in C.APC))
res["label_violates_per_true_lane"] = round(float(v.mean()), 4)
print(json.dumps(res, indent=1))
json.dump(res, open(C.WORK / "lr5_oracle.json", "w"), indent=1)
