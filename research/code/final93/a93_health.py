"""Note 93 A: is the final model less accurate on detectors the package's health output flags?

Joins the v4f six-fold OOF rows (s93/func_rows, s93/phase_rows; >= 30 min, v4l truth, E and R sets) with the package
health output on the same windows (s93/health_rows, h93_health.py).  Accuracy with signal-bootstrap 95 % CIs by status,
by the check that fired, and by health_score threshold (keep rows with score >= t).  -> %DC_WORK%/s93/a93.json

    python a93_health.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import rpath  # noqa: F401,E402
import cand64 as C  # noqa: E402

DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
S93 = DCW / "s93"
K = ["dkey", "period", "Detector", "win"]
CHECKS = None
STATUS = ["ok", "suspect", "bad", "not_enough_data", "no_row"]
THR = [0.0, 0.25, 0.5, 0.7, 0.9, 0.99]


def load():
    H = pd.read_parquet(S93 / "health_rows.parquet").rename(columns={"detector": "Detector"})
    H["dkey"] = H.DeviceId.str.lower()
    global CHECKS
    CHECKS = [c for c in H.columns if c.startswith("s_")]
    s = H[CHECKS].fillna(0).to_numpy()
    H["primary"] = np.where(s.max(1) > 0, np.array([c[2:] for c in CHECKS])[s.argmax(1)], "")
    H = H.drop(columns="DeviceId")
    f = pd.read_parquet(S93 / "func_rows.parquet")
    f = f[f.wgroup.isin(C.GE30)].copy()
    f["dkey"] = f.DeviceId.str.lower()
    f["sig"] = f.dkey
    f = f.merge(H.astype({"Detector": f.Detector.dtype}), on=K, how="left")
    p = pd.read_parquet(S93 / "phase_rows.parquet")
    p = p[p.fam.isin(C.GE30)].copy()
    p["period"] = np.where(p.DeviceId.str.endswith("@stg"), "stg", "dec")
    p["dkey"] = p.DeviceId.str.replace("@stg", "", regex=False).str.lower()
    p["sig"] = p.dev_plain.str.lower()
    p = p.merge(H.astype({"Detector": p.Detector.dtype}), on=K, how="left")
    for d in (f, p):
        d["status"] = d.status.fillna("no_row")
    return f, p


def acc(d, st, m):
    v = d[f"ok{st}"].to_numpy(float)
    mm = m & ~np.isnan(v)
    r = C.acc_ci(v[mm], d.sig.to_numpy()[mm])
    return [None if x is None else round(100 * x, 2) for x in r] + [int(mm.sum()), int(d.sig[mm].nunique())]


def table(d):
    out = {}
    for st in ("E", "R"):
        base = np.ones(len(d), bool)
        o = {"all": acc(d, st, base)}
        for s in STATUS:
            o[s] = acc(d, st, d.status.eq(s).to_numpy())
        o["suspect_or_bad"] = acc(d, st, d.status.isin(["suspect", "bad"]).to_numpy())
        o["ok_only"] = o["ok"]
        o["all_but_suspect_bad"] = acc(d, st, ~d.status.isin(["suspect", "bad"]).to_numpy())
        # primary check of a flagged detector (largest rule score)
        fl = d.status.isin(["suspect", "bad"]).to_numpy()
        o["by_primary_check_flagged"] = {c: acc(d, st, fl & d.primary.eq(c).to_numpy())
                                         for c in sorted(d.primary[fl].dropna().unique()) if c}
        # any rule score > 0 (fired at all, any status)
        o["by_check_fired_any_status"] = {c[2:]: acc(d, st, d[c].fillna(0).gt(0).to_numpy()) for c in CHECKS}
        # health_score thresholds: keep rows with score >= t (NaN score = not judged: kept, counted apart)
        hs = d.health_score.to_numpy(float)
        o["keep_score_ge"] = {}
        for t in THR:
            keep = ~(hs < t)
            o["keep_score_ge"][str(t)] = {"kept": acc(d, st, keep), "dropped": acc(d, st, ~keep)}
        o["score_band"] = {b: acc(d, st, m) for b, m in {
            "<0.25": hs < .25, "0.25-0.7": (hs >= .25) & (hs < .7), "0.7-0.9": (hs >= .7) & (hs < .9),
            "0.9-0.99": (hs >= .9) & (hs < .99), ">=0.99": hs >= .99, "NaN": np.isnan(hs)}.items()}
        out[st] = o
    return out


def main():
    f, p = load()
    res = {"function": table(f), "phase": table(p),
           "status_share_function_E": f[f.okE.notna()].status.value_counts(normalize=True).round(4).to_dict(),
           "status_share_phase_E": p[p.okE.notna()].status.value_counts(normalize=True).round(4).to_dict(),
           "checks": CHECKS}
    # flagged vs ok difference (unpaired: different rows) via signal bootstrap of the difference of means
    for nm, d in (("function", f), ("phase", p)):
        for st in ("E", "R"):
            v = d[f"ok{st}"].to_numpy(float)
            ok_m = d.status.eq("ok").to_numpy() & ~np.isnan(v)
            fl_m = d.status.isin(["suspect", "bad"]).to_numpy() & ~np.isnan(v)
            sig = d.sig.to_numpy()
            u = np.unique(sig[ok_m | fl_m])
            rng = np.random.default_rng(93)
            ix = {s: i for i, s in enumerate(u)}
            si = np.array([ix.get(s, -1) for s in sig])
            a_ok = np.bincount(si[ok_m], v[ok_m], len(u)), np.bincount(si[ok_m], minlength=len(u))
            a_fl = np.bincount(si[fl_m], v[fl_m], len(u)), np.bincount(si[fl_m], minlength=len(u))
            bs = []
            for _ in range(2000):
                w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
                bs.append((w @ a_fl[0]) / max(w @ a_fl[1], 1) - (w @ a_ok[0]) / max(w @ a_ok[1], 1))
            dlt = a_fl[0].sum() / a_fl[1].sum() - a_ok[0].sum() / a_ok[1].sum()
            res[f"{nm}_{st}_flagged_minus_ok_pt"] = [round(100 * dlt, 2), round(100 * np.percentile(bs, 2.5), 2),
                                                    round(100 * np.percentile(bs, 97.5), 2)]
    json.dump(res, open(S93 / "a93.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
