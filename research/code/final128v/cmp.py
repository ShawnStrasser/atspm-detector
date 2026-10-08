import os
import pickle, sys, pandas as pd, numpy as np
O = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"s128v\par")
P = pickle.load(open(O + r"\prod.pkl", "rb")); N = pickle.load(open(O + r"\next.pkl", "rb")); F = pickle.load(open(O + r"\nextfar.pkl", "rb"))
def diff(a, b):
    if a.equals(b): return []
    if list(a.columns) != list(b.columns) or len(a) != len(b): return [f"shape {a.shape} vs {b.shape}"]
    return [c for c in a.columns if not a[c].equals(b[c])]
nd = 0; ncol = None; ndet = 0; nph = 0
for k in P:
    po, pp, pt, pw = P[k]; no, np_, nt, nw = N[k]
    d1, d2 = diff(po, no), diff(pp, np_)
    ncol = len(po.columns); ndet += len(po); nph += len(pp)
    if d1 or d2 or pw != nw:
        nd += 1; print("DIFF", k, d1, d2, pw, nw)
        for c in d1:
            m = ~((po[c] == no[c]) | (po[c].isna() & no[c].isna()))
            print(pd.DataFrame({"det": po.Detector[m], "prod": po[c][m], "next": no[c][m]}).to_string()[:1500])
print(f"prod vs next: {len(P)} cases, {ncol} cols, {ndet} detector rows, {nph} phase rows, cases differing {nd}")
fd = 0
for (case, win, k), (fo, fp, ft, fw) in F.items():
    ref = P[(case, win, "nobound")]; refn = N[(case, win, "nobound")]
    d1, d2 = diff(ref[0], fo), diff(ref[1], fp)
    e1 = diff(refn[0], fo)
    if d1 or d2:
        fd += 1; print("FARDIFF", case, win, k, d1, d2, "vs next-nobound:", e1)
print(f"far: {len(F)} cases vs prod with bound left out, differing {fd}")
print("times far next: max %.2f s" % max(v[2] for v in F.values()))
print("status samples:", N[("x00","m30","nobound")][0].status.unique()[:5])
