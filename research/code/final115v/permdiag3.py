import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import warnings, numpy as np, pandas as pd
warnings.simplefilter("ignore")
from detector_classifier import pipeline as P, stacker as ST
V = DCW + r"\s115v"
rng = np.random.default_rng([11, 12])
cm = dict(zip(range(1, 65), (rng.permutation(64) + 1).tolist())); pm = dict(zip(range(1, 17), (rng.permutation(16) + 1).tolist()))
ic, ip = {b: a for a, b in cm.items()}, {b: a for a, b in pm.items()}
cap = {}
o_pred = ST.Stacker.predict
def pred(self, minutes, Pt, Pn, fr, *a, **k):
    r = o_pred(self, minutes, Pt, Pn, fr, *a, **k)
    cap["st"] = (minutes, Pt.copy(), Pn.copy(), fr.copy(), np.asarray(r).copy() if not isinstance(r, tuple) else r)
    return r
ST.Stacker.predict = pred
o_ctx = ST.ctx_X
def ctx(fr, Pb):
    X = o_ctx(fr, Pb); cap["ctx"] = (fr.copy(), X.copy()); return X
ST.ctx_X = ctx
S, E = "2026-09-20 12:00:00", "2026-09-20 12:05:00"
res = {}
for tag, f in (("orig", rf"{V}\ev\v12.parquet"), ("perm", rf"{V}\perm\v12_perm.parquet")):
    cap.clear()
    P.predict(f, start=S, end=E, min_actuations=1, profile="full")
    res[tag] = dict(cap)
fa, Xa = res["orig"]["ctx"]; fb, Xb = res["perm"]["ctx"]
da = fa.Detector.astype(int).tolist(); db = [ic[int(d)] for d in fb.Detector]
order = [db.index(d) for d in da]
print("ctx columns", Xa.shape)
d = np.abs(Xa - Xb[order]); nm = np.isnan(Xa) != np.isnan(Xb[order])
names = ST.ALL_NAMES if hasattr(ST, "ALL_NAMES") else None
for j in np.where((np.nan_to_num(d) > 1e-6).any(0) | nm.any(0))[0]:
    i = int(np.nanargmax(np.nan_to_num(d[:, j])))
    print("  ctx col", j, names[j] if names and len(names) == Xa.shape[1] else "", "max", np.nanmax(np.nan_to_num(d[:, j])), "det", da[i], Xa[i, j], Xb[order][i, j])
_, Pta, Pna, fra, _ = res["orig"]["st"]; _, Ptb, Pnb, frb, _ = res["perm"]["st"]
print("Pt max diff", np.abs(Pta - Ptb[order]).max(), " Pn max diff", np.abs(Pna - Pnb[order]).max())
for c in fra.columns:
    if c in ("Detector",): continue
    x, y = fra[c].reset_index(drop=True), frb[c].iloc[order].reset_index(drop=True)
    if c == "pred_phase": y = y.map(lambda v: ip.get(int(v), v) if v == v else v)
    if (x.astype(str) != y.astype(str)).any():
        i = int(np.flatnonzero((x.astype(str) != y.astype(str)).to_numpy())[0])
        print("  fr col differs:", c, "det", da[i], repr(x[i]), repr(y[i]))
