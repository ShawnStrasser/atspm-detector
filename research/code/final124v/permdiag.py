"""124v: where does the v08 seed-11 h24 function difference come from? capture the stacker inputs (Pt, Pn, fr)."""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
sys.path.insert(0, os.environ.get("PKG", str(Path.home() / "dc_work/final_v7_prod/src")))
from detector_classifier import predict, stacker
W = Path.home() / "dc_work"; V = W / "s124v"
cs, SEED = sys.argv[1], int(sys.argv[2]); s, e = sys.argv[3], sys.argv[4]
rng = np.random.default_rng([SEED, int(cs[1:])])
cm = dict(zip(range(1, 65), (rng.permutation(64) + 1).tolist())); pm = dict(zip(range(1, 17), (rng.permutation(16) + 1).tolist()))
ic, ip = {b: a for a, b in cm.items()}, {b: a for a, b in pm.items()}
cap = []
orig = stacker.Stacker.predict
def pr(self, minutes, Pt, Pn, fr):
    r = orig(self, minutes, Pt, Pn, fr); cap.append((Pt.copy(), Pn.copy(), fr.copy(), r.copy())); return r
stacker.Stacker.predict = pr
predict(str(V / "ev" / f"{cs}.parquet"), start=s, end=e, min_actuations=1)
predict(str(V / "perm" / f"{cs}_s{SEED}.parquet"), start=s, end=e, min_actuations=1)
(a_pt, a_pn, a_fr, a_ps), (b_pt, b_pn, b_fr, b_ps) = cap
bd = b_fr.Detector.map(ic).to_numpy(); order = np.argsort(bd)
b_pt, b_pn, b_ps = b_pt[order], b_pn[order], b_ps[order]; b_fr = b_fr.iloc[order].reset_index(drop=True)
b_fr["Detector"] = b_fr.Detector.map(ic); b_fr["pred_phase"] = b_fr.pred_phase.map(lambda x: ip[int(x)] if np.isfinite(x) else x)
assert (a_fr.Detector.to_numpy() == b_fr.Detector.to_numpy()).all()
print("Pt maxdiff", np.nanmax(np.abs(a_pt - b_pt)), "Pn maxdiff", np.nanmax(np.abs(a_pn - b_pn)), "Ps maxdiff", np.nanmax(np.abs(a_ps - b_ps)))
dets = a_fr.Detector.to_numpy()
for c in a_fr.columns:
    x, y = a_fr[c].astype(str).to_numpy(), b_fr[c].astype(str).to_numpy()
    if (x != y).any():
        i = np.flatnonzero(x != y)
        print("fr col", c, "differs on dets", dets[i].tolist()[:10], "e.g.", x[i[0]][:80], "|", y[i[0]][:80])
for nm, A, B in (("Pt", a_pt, b_pt), ("Pn", a_pn, b_pn), ("Ps", a_ps, b_ps)):
    d = np.abs(A - B).max(1); i = np.flatnonzero(d > 1e-6)
    print(nm, "dets differing >1e-6:", list(zip(dets[i].tolist(), np.round(d[i], 4).tolist()))[:12])
j = [int(np.flatnonzero(dets == d)[0]) for d in (23, 26)]
np.set_printoptions(precision=9, suppress=True)
print("Pt 23/26", a_pt[j]); print("Pn A 23/26", a_pn[j]); print("Pn B 23/26", b_pn[j])
print("Ps A", a_ps[j].round(4)); print("Ps B", b_ps[j].round(4))
print(a_fr.iloc[j].T.to_string())
from detector_classifier.stacker import ctx_X, W_TREE
xa = ctx_X(a_fr, W_TREE * a_pt + (1 - W_TREE) * a_pn); xb = ctx_X(b_fr, W_TREE * b_pt + (1 - W_TREE) * b_pn)
print("ctx cols differing:", np.flatnonzero(np.abs(xa - xb).max(0) > 1e-6).tolist(), "rows", np.flatnonzero(np.abs(xa - xb).max(1) > 1e-6).tolist())
import duckdb
f = (V / "ev" / f"{cs}.parquet").as_posix()
for d in (23, 26):
    print(d, duckdb.sql(f"select EventId, count(*) n, min(Timestamp), max(Timestamp) from '{f}' where EventId in (81,82) and Parameter={d} and Timestamp>='{s}' and Timestamp<'{e}' group by 1 order by 1").fetchall())
x = duckdb.sql(f"""with a as (select Timestamp t, EventId from '{f}' where EventId in (81,82) and Parameter=23), b as (select Timestamp t, EventId from '{f}' where EventId in (81,82) and Parameter=26)
 select (select count(*) from a join b using (t, EventId)) same_ts, (select count(*) from a) na, (select count(*) from b) nb""").fetchall()
print("same timestamp+event 23 vs 26:", x)
