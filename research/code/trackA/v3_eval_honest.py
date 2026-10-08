"""Honest evaluation of the function head (orchestrator / user, 2026-09-29): the label-cleansing filters are for
TRAINING only; scoring must not quietly drop hard rows.

  A   ALL labelled detectors, truth = the user's ruling, else a high-confidence print label, else the config label
      where the signal has no print reading of the detector; NO behaviour-validation, health, field-issue, dq or
      unusual-layout filter -- only the production rule (>= 5 actuations in the window). The honest number.
  B   A restricted to detector-periods the production health scorer (hi-res only, health_v3) does not call `bad`
      (production can compute this too). Coverage reported.
  C   the note-28 filtered FIX set (validated, not dq, ...): "clean-label subset (optimistic)".
All out-of-fold (six signal-grouped folds, 3-seed mean). Baseline = note-25 b7 OOF, compared on the rows it has.

    python v3_eval_honest.py --run RUN_DIR_NAME [--also name=RUN_DIR_NAME,...] --out NAME
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import v3_retrain as V

ap = argparse.ArgumentParser()
ap.add_argument("--frame", default="v6")
ap.add_argument("--variant", default="first.all.wi")
ap.add_argument("--models", required=True, help="name=RUN_DIR_NAME,... (first = reference for the lc comparison)")
ap.add_argument("--pairs", default="", help="a:b,... extra paired bootstraps (b minus a) on A / B / C")
ap.add_argument("--out", default="honest")
A = ap.parse_args()
V.set_frame(A.frame)
V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH = 5, True, True, True
fr, _ = V.load_feats()
lab = V.load_labels(fr, "exclude")            # the filtered (C) state
es = V.eval_sets(fr, lab)
C7 = np.array(V.C7, object)
wg, sig = fr.wgroup.to_numpy(), fr.DeviceId.to_numpy()

# ---- truth for A
v3 = pd.read_parquet(V.LABELS_V3)
v3["DeviceId"] = v3.DeviceId.str.lower()
pf, src = v3.print_function, v3.source.astype("string")
user = src.eq("user_ruling").fillna(False)
high = v3.print_source.eq("print") & v3.print_confidence.eq("high") & pf.isin(V.C7)
noprint = ~v3.print_source.eq("print").fillna(False) | pf.isna()
truth = pd.Series(np.where(user, v3.function, np.where(high, pf, np.where(noprint & v3.func7_v2.notna(), v3.func7_v2,
                                                                           None))), index=v3.index)
kind = np.where(user, "user", np.where(high, "print_high", np.where(noprint & v3.func7_v2.notna(), "config", "none")))
t = v3.assign(truth=truth, kind=kind)[["DeviceId", "detector", "truth", "kind", "unusual_layout"]]
t = t.rename(columns={"detector": "Detector"}).astype({"Detector": fr.Detector.dtype})
T = fr[V.KEY].merge(t, on=["DeviceId", "Detector"], how="left")
assert len(T) == len(fr)
yt = T.truth.to_numpy(object)
mA = pd.notna(yt) & np.isin(yt, V.C7) & (fr.det_n_on >= 5).to_numpy()
# ---- B: production health (health_v3, per detector and period) not bad
h = pd.read_parquet(V.DCW / "health" / "health_v3.parquet", columns=["period", "DeviceId", "detector", "status"])
h = h.assign(DeviceId=h.DeviceId.str.lower()).rename(columns={"detector": "Detector"}).astype(
    {"Detector": fr.Detector.dtype})
hs = fr[V.KEY].merge(h, on=["DeviceId", "Detector", "period"], how="left").status.to_numpy(object)
mB = mA & (hs != "bad")
mC = es["FIX"][0]
yC = lab.print_function.to_numpy(object)

Pb, hb = V.load_baseline(fr)
preds = {"baseline": C7[Pb.argmax(1)]}
have = np.ones(len(fr), bool)
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


res, lines = {}, []
for sn, m0, y in (("A", mA, yt), ("B", mB, yt), ("C", mC, yC)):
    m0 = m0 & have
    core = np.isin(y, V.CORE4)
    for rows, mm in (("all", m0), ("b7rows", m0 & hb)):
        for v, p in preds.items():
            if v == "baseline" and rows == "all":
                continue
            ok = p == y
            r = {"n": int(mm.sum()), "sig": int(pd.unique(sig[mm]).size), "acc7": ok[mm].mean(),
                 "core4": ok[mm & core].mean()}
            for g in ("m30", "h6", "full"):
                r[g] = ok[mm & (wg == g)].mean()
            if v != "baseline" and rows == "b7rows":
                ob = preds["baseline"] == y
                r["vs_b7"] = boot(ob, ok, mm)
                r["vs_b7_core4"] = boot(ob, ok, mm & core)
                for g in ("m30", "h6", "full"):
                    r[f"vs_b7_{g}"] = boot(ob, ok, mm & (wg == g))
            res[f"{sn}|{rows}|{v}"] = r
    for pr in [x for x in A.pairs.split(",") if x]:
        a, b = pr.split(":")
        oa, ob_ = preds[a] == y, preds[b] == y
        res[f"{sn}|pair|{b}-{a}"] = {"acc7": boot(oa, ob_, m0), "core4": boot(oa, ob_, m0 & core),
                                      **{g: boot(oa, ob_, m0 & (wg == g)) for g in ("m30", "h6", "full")}}
cov = {"A_rows": int((mA & have).sum()), "A_signals": int(pd.unique(sig[mA & have]).size),
       "A_by_truth_kind": pd.Series(T.kind.to_numpy()[mA & have]).value_counts().to_dict(),
       "A_unusual_rows": int((mA & have & T.unusual_layout.eq(True).to_numpy()).sum()),
       "B_rows": int((mB & have).sum()), "B_share_of_A": float((mB & have).sum() / max((mA & have).sum(), 1)),
       "C_rows": int((mC & have).sum()), "C_signals": int(pd.unique(sig[mC & have]).size),
       "A_rows_labelled_but_lt5": int((pd.notna(yt) & np.isin(yt, V.C7) & (fr.det_n_on < 5).to_numpy()).sum())}
json.dump({"coverage": cov, "results": res}, open(V.OUT / f"{A.out}.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(json.dumps(cov, indent=1, default=float))
df = pd.DataFrame(res).T
print(df.to_string(float_format=lambda x: f"{x:.4f}"))
