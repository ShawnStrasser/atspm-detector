"""Where does the v12 permutation difference enter?  Capture ranker p0_tree, keep set, network function probs."""
import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import warnings, numpy as np, pandas as pd
warnings.simplefilter("ignore")
from detector_classifier import pipeline as P
from detector_classifier import function_stage as FS
V = DCW + r"\s115v"
rng = np.random.default_rng([11, 12])
cm = dict(zip(range(1, 65), (rng.permutation(64) + 1).tolist())); pm = dict(zip(range(1, 17), (rng.permutation(16) + 1).tolist()))
ic, ip = {b: a for a, b in cm.items()}, {b: a for a, b in pm.items()}
cap = {}
o_score = P.score
def score(*a, **k):
    r = o_score(*a, **k); cap["sc"] = r[["DeviceId", "Detector", "cand_phase", "p0_tree", "p0", "prob"]].copy(); return r
P.score = score
o_feat = P.build_features
def bf(*a, **k):
    r = o_feat(*a, **k); cap["feat"] = r[0].copy(); return r
P.build_features = bf
S, E = "2026-09-20 12:00:00", "2026-09-20 12:05:00"
res = {}
for tag, f in (("orig", rf"{V}\ev\v12.parquet"), ("perm", rf"{V}\perm\v12_perm.parquet")):
    out = P.predict(f, start=S, end=E, min_actuations=1, profile="full")
    sc, ft = cap["sc"].copy(), cap["feat"].copy()
    if tag == "perm":
        for d in (sc, ft, out):
            d["Detector"] = d.Detector.astype(int).map(ic)
        for d in (sc, ft):
            d["cand_phase"] = d.cand_phase.astype(int).map(ip)
    res[tag] = (sc.sort_values(["Detector", "cand_phase"]).reset_index(drop=True),
                ft.sort_values(["Detector", "cand_phase"]).reset_index(drop=True), out.sort_values("Detector").reset_index(drop=True))
a, b = res["orig"], res["perm"]
num = [c for c in a[1].columns if c not in ("DeviceId", "Detector", "cand_phase", "win") and pd.api.types.is_numeric_dtype(a[1][c])]
X, Y = a[1][num].to_numpy(float), b[1][num].to_numpy(float)
d = np.abs(X - Y); nanm = np.isnan(X) != np.isnan(Y)
bad = [(num[j], float(np.nanmax(d[:, j])), int(nanm[:, j].sum())) for j in range(len(num)) if np.nanmax(np.nan_to_num(d[:, j])) > 1e-9 or nanm[:, j].any()]
print("pair features differing (name, max|diff|, nan mismatches):", len(bad), "of", len(num)); print(sorted(bad, key=lambda t: -t[1])[:15])
dp = np.abs(a[0].p0_tree.to_numpy() - b[0].p0_tree.to_numpy()); print("ranker p0_tree max diff", dp.max())
ka, kb = a[0].p0_tree >= .01, b[0].p0_tree >= .01
print("keep set flips:", int((ka != kb).sum()))
print("final prob max diff", np.abs(a[0].prob.to_numpy() - b[0].prob.to_numpy()).max())
print("p_advance max diff", np.nanmax(np.abs(a[2].p_advance.to_numpy(float) - b[2].p_advance.to_numpy(float))))
