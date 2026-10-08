"""Are there real lane labels hiding in the technician's channel descriptions?

Some agencies name the lane: "Loop 1 - RL Advance", "Rad A - CL Count", "RADAR D THRU
z4".  RL / CL / LL (right / centre / left lane) is a genuine lane label the model never
sees -- a validation set for the A3 lane grouper.  This script extracts it, measures how
well the current similarity graph reproduces it, and tries a few alternative graphs.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
import a3_lanes as A3

DCW = A3.DCW
WORK = A3.WORK

keys = pd.read_parquet(WORK / "a2_oof_keys.parquet")
off = pd.read_parquet(DCW / "official" / "labels_official.parquet")[
    ["DeviceId", "DeviceName", "Detector", "description"]]
off["DeviceId"] = off.DeviceId.str.lower()
off["Detector"] = off.Detector.astype(keys.Detector.dtype)
k = keys[keys.wgroup == "full"].merge(off.drop_duplicates(["DeviceId", "Detector"]),
                                      on=["DeviceId", "Detector"], how="left")
d = k.description.astype(str).str.upper()
lane = pd.Series(pd.NA, index=k.index, dtype="object")
# right / centre / left lane, as a standalone token
for tag in ("RL", "CL", "LL"):
    lane = lane.mask(d.str.contains(rf"\b{tag}\b", regex=True, na=False), tag)
k["lane_true"] = lane
have = k[k.lane_true.notna()]
print(f"{len(have)} detector-rows carry an RL/CL/LL lane token, "
      f"{have.DeviceId.nunique()} signals")
g = have.groupby(["DeviceId", "period", "pred_phase"])
ok = g.Detector.transform("size").gt(1)
have = have[ok]
print(f"{len(have)} on {have.groupby(['DeviceId','period','pred_phase']).ngroups} "
      f"phases with >=2 labelled lanes; letters per phase: "
      f"{g.lane_true.nunique().value_counts().sort_index().to_dict()}")

links = A3.link_table()
lk = {}
for (did, per, win), gg in links.groupby(["DeviceId", "period", "win"], sort=False):
    lk[(did, per, win)] = dict(zip(zip(gg.lo.to_numpy(), gg.hi.to_numpy()),
                                   zip(gg.s.to_numpy(), gg.lag_excess.fillna(0).to_numpy(),
                                       gg.c_off.fillna(gg.c_all).fillna(0.5).to_numpy(),
                                       gg.coinc.fillna(0).to_numpy(),
                                       gg.bal.to_numpy())))

VARIANTS = {
    "s = bal*corr*(0.3+0.7*ex)": lambda s, ex, c, co, b: s,
    "ex alone":                  lambda s, ex, c, co, b: ex,
    "corr alone":                lambda s, ex, c, co, b: c,
    "bal alone":                 lambda s, ex, c, co, b: b,
    "max(ex,coinc)":             lambda s, ex, c, co, b: max(ex, co),
    "bal*corr":                  lambda s, ex, c, co, b: b * c,
    "corr^2*bal":                lambda s, ex, c, co, b: c * c * b,
}
rows = []
pairs = []
for (did, per, ph), gg in have.groupby(["DeviceId", "period", "pred_phase"]):
    win = gg.win.iloc[0]
    sub = lk.get((did, per, win), {})
    dets = gg.Detector.to_numpy()
    lanes = gg.lane_true.to_numpy()
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = (dets[i], dets[j]) if dets[i] < dets[j] else (dets[j], dets[i])
            v = sub.get((a, b))
            if v is None:
                continue
            pairs.append((lanes[i] == lanes[j],) + v)
P = pd.DataFrame(pairs, columns=["same_lane", "s", "ex", "corr", "coinc", "bal"])
print(f"\n{len(P)} pairs with a similarity value, {P.same_lane.mean():.3f} same-lane")
print(P.groupby("same_lane")[["s", "ex", "corr", "coinc", "bal"]].median().round(3)
      .to_string())
from sklearn.metrics import roc_auc_score
res = {}
for name, f in VARIANTS.items():
    v = np.array([f(*r) for r in P[["s", "ex", "corr", "coinc", "bal"]].to_numpy()])
    keep = ~np.isnan(v)
    auc = roc_auc_score(P.same_lane[keep], v[keep])
    res[name] = round(float(auc), 4)
    print(f"  AUC same-lane, {name:28s} {auc:.4f}")
json.dump({"n_pairs": int(len(P)), "share_same_lane": float(P.same_lane.mean()),
           "auc": res}, open(WORK / "a3_lanelabels.json", "w"), indent=1)
