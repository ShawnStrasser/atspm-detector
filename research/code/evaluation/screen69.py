"""Note 69: screen function-network variants through the cand64 function pipeline (trees 0.6 + net 0.4 before the decode).

For every variant tag, the fold files `tcn53/fpreds/<tag>_f{k}.parquet` that exist are copied (as fj_f{k}.parquet) into a
temp folder and cand64's `stage_function` is run on them (side outputs / final_v2 arms skipped: they do not touch the
scores) -> `%DC_WORK%/s69/<tag>/function_rows.parquet`.  The comparison is on the folds the variant AND the reference
(default fj seed 0 = `fj_f{k}`) both have: accuracy of trees / reference / variant, the paired signal-bootstrap delta
variant - reference (pt, 95 % CI), per pool (>= 30 min headline, 5 / 10 min, all) and set (E / R), and at >= 30 min
the ATSPM-score accuracy by TRUE class (Advance, Presence, Count, Yellow_Red, non-ATSPM = Other / Mid / Bike).
Locked_v2 asserted absent (cand64).  Every variant's OOF predictions stay in fpreds (never overwritten).

    python screen69.py --tags fj,fj_s1,fj_s2,x69_r01s --folds 0,3 [--ref fj]   -> %DC_WORK%/s69/screen.json
    python screen69.py --ref fj+fj_s1+fj_s2 --tags x69_siba+x69_siba_s1+x69_siba_s2 --folds 0,1,2,3,4,5 --out six.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cand64 as C  # noqa: E402

ROOT = C.DC_WORK / "s69"
FP = C.DC_WORK / "tcn53" / "fpreds"
CLS = ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]


def build(tag: str, folds: list[int], force: bool) -> Path | None:
    out = ROOT / tag.replace("+", "~")
    parts = tag.split("+")                      # "a+b+c" = seeds averaged per fold (cand64 averages fj_f / fj_sN_f)
    have = [k for k in folds if all((FP / f"{t}_f{k}.parquet").exists() for t in parts)]
    if not have:
        return None
    meta = out / "function_meta.json"
    if meta.exists() and not force:
        m = json.load(open(meta))
        if sorted(h["fold"] for h in m["fj_folds"]) == have:
            return out
    tmp = C.DC_WORK / "tmp" / "s69" / tag.replace("+", "~")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for k in have:
        for i, t in enumerate(parts):
            shutil.copy(FP / f"{t}_f{k}.parquet", tmp / (f"fj_f{k}.parquet" if i == 0 else f"fj_s{i}_f{k}.parquet"))
    out.mkdir(parents=True, exist_ok=True)
    C.FPRED, C.OUT = tmp, out
    C.attach_side = lambda o: o                 # side outputs do not change any score
    C.final_v2_function = lambda o: o
    C.stage_function(None)
    return out


def load(tag: str) -> pd.DataFrame:
    f = pd.read_parquet(ROOT / tag.replace("+", "~") / "function_rows.parquet",
                        columns=["DeviceId", "Detector", "period", "win", "wgroup", "fold", "truth_v3s", "has_fj",
                                 "ok_E_trees", "ok_E_fj", "ok_R_trees", "ok_R_fj", "pred_fj"])
    return f


def compare(ref: pd.DataFrame, var: pd.DataFrame, folds: list[int]) -> dict:
    key = ["DeviceId", "Detector", "period", "win"]
    m = ref.merge(var[key + ["ok_E_fj", "ok_R_fj", "has_fj", "pred_fj"]], on=key, suffixes=("", "_v"))
    m = m[m.fold.isin(folds) & m.has_fj & m.has_fj_v]
    sig = m.DeviceId.to_numpy()
    res = {"folds": folds}
    for s in ("E", "R"):
        sc = ~np.isnan(m[f"ok_{s}_trees"].to_numpy())
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            k = sc & m.wgroup.isin(fams).to_numpy()
            t, r, v = (m[c].to_numpy()[k].astype(float) for c in (f"ok_{s}_trees", f"ok_{s}_fj", f"ok_{s}_fj_v"))
            res[f"{s}_{pool}"] = {"n": int(k.sum()), "signals": int(len(set(sig[k]))),
                                  "trees": C.acc_ci(t, sig[k]), "ref": C.acc_ci(r, sig[k]), "var": C.acc_ci(v, sig[k]),
                                  "ref_minus_trees": C.delta_ci(t, r, sig[k]), "var_minus_trees": C.delta_ci(t, v, sig[k]),
                                  "var_minus_ref": C.delta_ci(r, v, sig[k])}
        k = sc & m.wgroup.isin(C.GE30).to_numpy()
        tr = m.truth_v3s.to_numpy(object)
        cl = np.where(np.isin(tr, CLS[:4]), tr, "nonATSPM")
        bc = {}
        for c in CLS:
            kk = k & (cl == c)
            r, v = (m[col].to_numpy()[kk].astype(float) for col in (f"ok_{s}_fj", f"ok_{s}_fj_v"))
            bc[c] = {"n": int(kk.sum()), "ref": round(float(r.mean()), 4) if kk.any() else None,
                     "var": round(float(v.mean()), 4) if kk.any() else None,
                     "var_minus_ref": C.delta_ci(r, v, sig[kk]) if kk.any() else None}
        res[f"{s}_ge30_by_class"] = bc
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--folds", default="0,3")
    ap.add_argument("--ref", default="fj")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="screen.json")
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    tags = [a.ref] + [t for t in a.tags.split(",") if t != a.ref]
    built = {t: build(t, folds, a.force) for t in tags}
    ref = load(a.ref)
    js = ROOT / a.out
    allres = json.load(open(js)) if js.exists() else {}
    for t in tags[1:]:
        if built[t] is None:
            C.log(f"{t}: no fold files"); continue
        var = load(t)
        fv = sorted(set(var.fold[var.has_fj]) & set(ref.fold[ref.has_fj]) & set(folds))
        r = compare(ref, var, fv)
        allres[t] = r
        e, rr = r["E_ge30"], r["R_ge30"]
        C.log(f"{t} folds {fv}: >=30 E trees {e['trees'][0]} ref {e['ref'][0]} var {e['var'][0]} "
              f"var-ref {e['var_minus_ref']} | R var-ref {rr['var_minus_ref']} | m5 E {r['E_m5']['var_minus_ref']} "
              f"m10 E {r['E_m10']['var_minus_ref']}")
        C.log("   by class >=30 E (ref -> var, pt): " + "  ".join(
            f"{c} {v['ref']}->{v['var']} ({v['var_minus_ref'][0]:+.2f} [{v['var_minus_ref'][1]:+.2f},"
            f"{v['var_minus_ref'][2]:+.2f}], n {v['n']})" for c, v in r["E_ge30_by_class"].items() if v["n"]))
    json.dump(allres, open(js, "w"), indent=1)


if __name__ == "__main__":
    main()
