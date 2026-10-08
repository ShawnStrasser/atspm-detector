"""Lanes from cabinet prints (note 30), step 4 -- per-phase n_lanes from the pair model,
scored against the print n_lanes (nlanes_labels.parquet).  Timing phase grouping (isolates
the lane model from phase errors).  Every P(same lane) is out-of-fold (lp3).

Detector sets on a phase:
  all   every detector with >= 20 actuations (production-like: includes bike / departure /
        unlisted channels)
  pred  those whose note-25 b7 OOF function is Advance / Presence / Count / Yellow_Red
        (only signals in that model's frame)
  print the print's high-confidence vehicle-lane detectors (oracle membership: measures the
        grouping alone)
Estimators:
  const1          always 1 lane
  lmin_pred       max(#A, #P, #C) of the predicted functions (no pair model)
  clus_{n17,print}  average-linkage clusters on P(same lane), cut at th
  mis_{n17,print}   size of the largest set of detectors that are pairwise different-lane
                    (P < th) -- spanning detectors drop out naturally
  th chosen per fold on the other five folds (exact accuracy).

    python lp4_nlanes.py
"""
from __future__ import annotations
import itertools
import json
import numpy as np
import pandas as pd
import lr_common as C
import lp1_labels as LP1
import a3_lanes as A3

THS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def max_diff_set(S, th):
    n = len(S)
    if n <= 1:
        return n
    D = S < th
    best = 1
    for k in range(2, min(n, 6) + 1):
        found = False
        for c in itertools.combinations(range(n), k):
            if all(D[i, j] for i, j in itertools.combinations(c, 2)):
                found = True
                break
        if not found:
            break
        best = k
    return best


def main():
    NL = pd.read_parquet(C.WORK / "nlanes_labels.parquet")
    PR = pd.read_parquet(C.WORK / "lp3_pairs_oof.parquet")
    f = pd.read_csv(C.DCW / "folds_v3.csv")
    f["DeviceId"] = f.DeviceId.str.lower()
    NL = NL.merge(f[["DeviceId", "fold"]], on="DeviceId")
    d = LP1.load()
    act = pd.concat([PR[["DeviceId", "target", "da"]].rename(columns={"da": "det"}),
                     PR[["DeviceId", "target", "db"]].rename(columns={"db": "det"})]
                    ).drop_duplicates()
    # single-detector phases have no pair: take actuation from lp2's detector list
    v = d[["DeviceId", "phase_target", "detector", "oof_pred", "print_confidence",
           "lane_type", "lane_index"]].rename(columns={"phase_target": "target",
                                                       "detector": "det"})
    v["det"] = v.det.astype("int16")
    act["det"] = act.det.astype("int16")
    n_on = pd.read_parquet(C.WORK / "lp2_pairs.parquet", columns=["DeviceId", "da", "n_a"])
    V = v.merge(act.assign(inpair=True), on=["DeviceId", "target", "det"], how="left")
    V["inpair"] = V.inpair.fillna(False).astype(bool)
    lk = {}
    for r in PR.itertuples(index=False):
        lk[(r.DeviceId, int(r.da), int(r.db))] = (r.p_n17, r.p_print)

    sets = {
        "all": V.inpair,
        "pred": V.inpair & V.oof_pred.isin(["Advance", "Presence", "Count", "Yellow_Red"]),
        "print": V.inpair & (V.print_confidence == "high") & V.lane_index.notna()
                 & ~V.lane_type.isin(LP1.NONVEH)}
    groups = {k: V[m].groupby(["DeviceId", "target"]) for k, m in sets.items()}
    rows = []
    NLi = NL.set_index(["DeviceId", "target"])
    has_pred = set(V[V.oof_pred.notna()].DeviceId)
    for sname, G in groups.items():
        gi = G.groups
        for key, r in NLi.iterrows():
            if sname == "pred" and key[0] not in has_pred:
                continue
            dets = np.sort(V.loc[gi[key], "det"].to_numpy()) if key in gi else np.array([])
            if len(dets) == 0:
                continue
            ofp = V.loc[gi[key], "oof_pred"] if key in gi else pd.Series(dtype=object)
            rec = {"set": sname, "DeviceId": key[0], "target": key[1], "fold": r.fold,
                   "n_true": int(r.n_lanes), "n_det": len(dets),
                   "lmin_pred": int(max(1, max((ofp == c).sum() for c in C.APC)))
                   if ofp.notna().all() else np.nan}
            for mi, mname in ((0, "n17"), (1, "print")):
                S = np.eye(len(dets))
                for i, j in itertools.combinations(range(len(dets)), 2):
                    p = lk.get((key[0], int(dets[i]), int(dets[j])), (np.nan, np.nan))[mi]
                    S[i, j] = S[j, i] = 0.0 if np.isnan(p) else p
                for th in THS:
                    rec[f"clus_{mname}_{th}"] = len(np.unique(
                        A3.cluster(np.arange(len(dets)), S, th)))
                    rec[f"mis_{mname}_{th}"] = max_diff_set(S, th)
            rows.append(rec)
    R = pd.DataFrame(rows)
    res = {}
    for sname, g in R.groupby("set"):
        out = {"n_phases": int(len(g)), "n_signals": int(g.DeviceId.nunique()),
               "true_dist": g.n_true.value_counts().sort_index().to_dict()}
        y = g.n_true.to_numpy()

        def score(pred):
            pred = np.asarray(pred, float)
            ok = ~np.isnan(pred)
            return {"exact": round(float((pred[ok] == y[ok]).mean()), 4),
                    "pm1": round(float((np.abs(pred[ok] - y[ok]) <= 1).mean()), 4),
                    "n": int(ok.sum()),
                    "exact_multi": round(float((pred[ok & (y >= 2)] == y[ok & (y >= 2)]).mean()), 4)}
        out["const1"] = score(np.ones(len(g)))
        if g.lmin_pred.notna().any():
            out["lmin_pred"] = score(g.lmin_pred)
        for est in ("clus_n17", "mis_n17", "clus_print", "mis_print"):
            pred = np.full(len(g), np.nan)
            chosen = {}
            for fo in sorted(g.fold.unique()):
                tr = (g.fold != fo).to_numpy()
                th = max(THS, key=lambda t: (g[f"{est}_{t}"].to_numpy()[tr] == y[tr]).mean())
                chosen[int(fo)] = th
                pred[~tr] = g[f"{est}_{th}"].to_numpy()[~tr]
            out[est] = score(pred)
            out[est]["th_by_fold"] = chosen
            if est == "mis_print":
                ct = pd.crosstab(pd.Series(y, name="true"),
                                 pd.Series(pred.astype(int), name="pred"))
                out[est]["confusion"] = {int(k): {int(c): int(v) for c, v in row.items()}
                                         for k, row in ct.iterrows()}
        res[sname] = out
        C.log(f"{sname}: {json.dumps(out)}")
    R.to_parquet(C.WORK / "lp4_nlanes.parquet", index=False)
    json.dump(res, open(C.WORK / "lp4_nlanes.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
