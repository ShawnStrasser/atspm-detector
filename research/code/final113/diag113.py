"""Note 113 step 1: what does channel adjacency capture that phi-neighbours do not?  Seed-mean per-row correctness of
the v6 / trees decoder with vs without adj_* (q112), stratified by window, volume, phi-neighbour coverage, adjacency
purity (truth; evaluation only) and odd/even truth phase (left-turn proxy; evaluation only)."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final112"))
import f112  # noqa
F, P, P109 = f112._setup()
C = P.C
comb, simc = P.pool(P.KEY4 + ["fold"])
q = pd.read_parquet(C.DC_WORK / "s112" / "phase" / "q112.parquet")
rows = P.rows_().copy()
DET = P.DET
def okE(col):
    d = comb[P.KEY4].copy(); d["p"] = q[col].to_numpy()
    d = d.dropna(subset=["p"]).sort_values(DET + ["p", "cand_phase"], ascending=[True, True, True, False, True])
    t = d.groupby(DET, sort=False).first().reset_index()
    pr = rows[DET].merge(t[DET + ["cand_phase"]], on=DET, how="left").cand_phase.to_numpy()
    return (pr == rows.Phase.to_numpy()).astype(float)
for arm in ("v6", "trees"):
    rows[f"{arm}_adj"] = np.mean([okE(f"p2_{arm}_adj_s{s}") for s in range(3)], 0)
    rows[f"{arm}_noadj"] = np.mean([okE(f"p2_{arm}_noadj_s{s}") for s in range(3)], 0)
    rows[f"{arm}_d"] = rows[f"{arm}_adj"] - rows[f"{arm}_noadj"]
# phi-neighbour coverage
s = simc.copy(); s = s[s.phi > 0]
nphi = s.groupby(DET).size().rename("n_phi").reset_index()
topphi = s.sort_values("phi", ascending=False).groupby(DET).phi.first().rename("phi_top").reset_index()
rows = rows.merge(nphi, on=DET, how="left").merge(topphi, on=DET, how="left")
rows["n_phi"] = rows.n_phi.fillna(0)
# adjacency purity (truth, evaluation only): share of labelled |dch|<=2 neighbours with the same truth phase
lab = rows[DET + ["Phase"]].dropna()
j = lab.merge(lab.rename(columns={"Detector": "o", "Phase": "Po"}), on=["DeviceId", "win"])
dd = (j.Detector.astype(int) - j.o.astype(int)).abs(); j = j[(dd >= 1) & (dd <= 2)]
j["same"] = (j.Phase == j.Po).astype(float)
pur = j.groupby(DET).agg(adj_pur=("same", "mean"), adj_nlab=("same", "size")).reset_index()
rows = rows.merge(pur, on=DET, how="left")
# phi purity: share of phi-neighbours (top-8) with same truth phase
js = s.merge(lab.rename(columns={"Detector": "other", "Phase": "Po"}), on=["DeviceId", "win", "other"]).merge(lab, on=DET)
js["same"] = (js.Phase == js.Po).astype(float)
pp = js.groupby(DET).same.mean().rename("phi_pur").reset_index()
rows = rows.merge(pp, on=DET, how="left")
m = rows.everything.to_numpy()
r = rows[m].copy()
r["ge30"] = r.fam.isin(["m30", "h1", "h3", "h6", "h24", "full"])
r["lenb"] = np.where(r.ge30, "ge30", r.fam)
r["vol"] = pd.cut(r.det_n_on, [-1, 5, 20, 100, 1000, 1e12], labels=["<=5", "6-20", "21-100", "101-1000", ">1000"])
r["odd"] = np.where(r.Phase % 2 == 1, "odd(LT)", "even(thru)")
r["phib"] = pd.cut(r.n_phi, [-1, 0, 2, 5, 8], labels=["0", "1-2", "3-5", "6-8"])
r["purb"] = pd.cut(r.adj_pur, [-0.01, 0.0, 0.5, 0.99, 1.0], labels=["0", "<=.5", ".5-.99", "1"]).cat.add_categories("none").fillna("none")
r["phipur"] = pd.cut(r.phi_pur, [-0.01, 0.0, 0.5, 0.99, 1.0], labels=["0", "<=.5", ".5-.99", "1"]).cat.add_categories("none").fillna("none")
out = {}
for by in ("lenb", "vol", "odd", "phib", "purb", "phipur"):
    for L in ("ge30", "m5"):
        g = r[r.lenb == L].groupby(by, observed=True).agg(n=("v6_d", "size"), v6_adj=("v6_adj", "mean"), v6_d=("v6_d", "mean"),
                                                         tr_adj=("trees_adj", "mean"), tr_d=("trees_d", "mean"))
        g["v6_share_gain"] = (r[r.lenb == L].groupby(by, observed=True).v6_d.sum() / r[r.lenb == L].v6_d.sum())
        g["tr_share_gain"] = (r[r.lenb == L].groupby(by, observed=True).trees_d.sum() / r[r.lenb == L].trees_d.sum())
        print(f"--- {by} {L}\n{g.round(4).to_string()}")
        out[f"{by}_{L}"] = g.round(5).reset_index().astype({by: str}).to_dict("records")
# rows that flip: adj right, noadj wrong (v6, all 3 seeds)
fl = r[(r.v6_adj == 1) & (r.v6_noadj == 0)]
fr = r[(r.v6_adj == 0) & (r.v6_noadj == 1)]
print("v6 flips adj-only right", len(fl), "noadj-only right", len(fr))
print(fl.groupby("lenb").size())
print("flip rows: median n_on", fl.det_n_on.median(), "all", r.det_n_on.median(), "phi_top med", fl.phi_top.median(), r.phi_top.median())
o = C.DC_WORK / "s113"; o.mkdir(exist_ok=True)
json.dump(out, open(o / "diag113.json", "w"), indent=1, default=str)
