"""note 50: function head with / without the permissive-phase features, overall and on permissive subsets.

Sets as note 45: A honest (every labelled detector, >= 5 actuations), C clean-label (note-28 FIX). Subsets by the
detector's LABELLED phase (evaluation only): FYA = protected-permissive left-turn phase (label_check_pplt: events,
timing or behaviour), FYAet = events or timing only (not the behaviour rule, which resembles the new features),
RT = print lane R, PRM = FYA | RT, rest = none of these. Paired signal bootstrap (2000), 3-seed mean OOF.

    python pm_eval.py --models h3=RUN,pm=RUN,pmshuf=RUN --ref h3
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import v3_retrain as V

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--ref", default="h3")
ap.add_argument("--variant", default="first.all.wi")
ap.add_argument("--out", default="pm_eval")
A = ap.parse_args()
V.set_frame("v6")
V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH = 5, True, True, True
fr, _ = V.load_feats()
lab = V.load_labels(fr, "exclude")
es = V.eval_sets(fr, lab)
C7 = np.array(V.C7, object)
wg, sig = fr.wgroup.to_numpy(), fr.DeviceId.to_numpy()

v3 = pd.read_parquet(V.LABELS_V3)
v3["DeviceId"] = v3.DeviceId.str.lower()
pf, src = v3.print_function, v3.source.astype("string")
user = src.eq("user_ruling").fillna(False)
high = v3.print_source.eq("print") & v3.print_confidence.eq("high") & pf.isin(V.C7)
noprint = ~v3.print_source.eq("print").fillna(False) | pf.isna()
v3["truth"] = np.where(user, v3.function, np.where(high, pf, np.where(noprint & v3.func7_v2.notna(), v3.func7_v2,
                                                                      None)))
pp = pd.read_parquet(V.DCW / "cabinet" / "label_check_pplt.parquet")
pp["sid"] = pp.sid.str.lower()
allp = set(zip(pp.sid, "P" + pp.p.astype(str)))
etp = set(zip(pp.sid[pp.by_events | pp.by_timing], "P" + pp.p[pp.by_events | pp.by_timing].astype(str)))
v3["fya"] = [(a, str(b)) in allp for a, b in zip(v3.DeviceId, v3.phase_target)]
v3["fyaet"] = [(a, str(b)) in etp for a, b in zip(v3.DeviceId, v3.phase_target)]
v3["rt"] = v3.lane_type.astype("string").eq("R").fillna(False)
t = v3.rename(columns={"detector": "Detector"}).astype({"Detector": fr.Detector.dtype})
T = fr[V.KEY].merge(t[["DeviceId", "Detector", "truth", "fya", "fyaet", "rt"]], on=["DeviceId", "Detector"],
                    how="left")
assert len(T) == len(fr)
yA = T.truth.to_numpy(object)
mA = (pd.notna(yA) & np.isin(yA, V.C7) & (fr.det_n_on >= 5).to_numpy()).astype(bool)
mC = np.asarray(es["FIX"][0], bool)
yC = lab.print_function.to_numpy(object)
fya, fyaet, rt = (T[c].eq(True).to_numpy(dtype=bool, na_value=False) for c in ("fya", "fyaet", "rt"))
subs = {"all": np.ones(len(fr), bool), "FYA": fya, "FYAet": fyaet, "RT": rt, "PRM": fya | rt, "rest": ~(fya | rt)}

preds, have = {}, np.ones(len(fr), bool)
for spec in A.models.split(","):
    name, rd = spec.split("=")
    Ps = []
    for s in (0, 1, 2):
        P, hv, got = V.load_oof(fr, V.OUT / rd, A.variant, s, range(6))
        assert len(got) == 6, (rd, s, got)
        Ps.append(P)
        have &= hv
    preds[name] = C7[np.mean(Ps, 0).argmax(1)]


def boot(ok_a, ok_b, m, n_boot=2000, seed=0):
    s = sig[m]; a = ok_a[m].astype(float); b = ok_b[m].astype(float)
    u, inv = np.unique(s, return_inverse=True); n = len(u)
    sa, sb, c = np.bincount(inv, a, n), np.bincount(inv, b, n), np.bincount(inv, minlength=n)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    d = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return f"{(b.mean() - a.mean()) * 100:+.2f} [{np.quantile(d, .025) * 100:+.2f}, {np.quantile(d, .975) * 100:+.2f}]"


res = {}
for sn, m0, y in (("A", mA, yA), ("C", mC, yC)):
    m0 = m0 & have
    for bn, b in subs.items():
        m = m0 & b
        r = {"n": int(m.sum()), "sig": int(pd.unique(sig[m]).size)}
        for v, p in preds.items():
            r[v] = float((p == y)[m].mean())
        oref = preds[A.ref] == y
        for v, p in preds.items():
            if v != A.ref:
                r[f"{v}-{A.ref}"] = boot(oref, p == y, m)
                for g in ("m30", "full"):
                    r[f"{v}-{A.ref} {g}"] = boot(oref, p == y, m & (wg == g))
        res[f"{sn}|{bn}"] = r
json.dump(res, open(V.OUT / f"{A.out}.json", "w"), indent=1)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(res).T.to_string(float_format=lambda x: f"{x:.4f}"))
