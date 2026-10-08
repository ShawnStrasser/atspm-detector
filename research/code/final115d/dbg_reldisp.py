import os as _os
from pathlib import Path as _P
DCW = _os.environ.get("DC_WORK", str(_P.home() / "dc_work"))
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from detector_classifier import pipeline as P, health_core as hc
orig = hc.rel_disp
def rd(x, r, ok, h):
    D, nb = orig(x, r, ok, h)
    xx = np.where(ok, x, 0.0).astype(float); okf = ok.astype(float)
    X = hc._movsum(xx, h) - xx
    if r is not None:
        rr = np.where(ok, r, 0.0).astype(float); R = hc._movsum(rr, h) - rr
        s = X / np.maximum(R, 1e-9); e = s*rr; v = e*rr*0+s*rr*(1+s)
        m = ok & (R > 0) & (X + xx > 0); m &= (e + xx) > 0
        if np.isfinite(D) and (v[m].sum() < 1 or D > 1000):
            print("D", D, "nb", nb, "vsum", v[m].sum(), "esum", e[m].sum(), "xsum", xx[m].sum(), "x>0 bins", int((xx>0).sum()), "R>0 bins", int((R>0).sum()), "Xsum", X[m].sum(), "r", rr.sum())
    return D, nb
hc.rel_disp = rd
o = P.predict(DCW + r"\s115v\ev\v08.parquet", start="2026-09-20 09:00:00", end="2026-09-20 12:00:00", min_actuations=1)
print(o[o.Detector.isin([16,17])][["Detector","function_pred","health_status","health_reason"]].to_string())
