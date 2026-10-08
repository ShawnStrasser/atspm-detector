"""Note 76: patch the phase pool (note 57 / cand64) and the function frame (v6e) with the phase-order-free features.

Inputs: `excl76.py` outputs in %DC_WORK%/final_v3_work/f76/ (new excl_partner_diff per pair; partner-tie rows with the
new partner quantities).  Only the affected columns are rebuilt:
  phase pool   excl_partner_diff; every cross-candidate companion (__rank / __z / __mgap / __argmax of the 28 RANK_FEATS,
               recomputed from the stored float32 values = the new rule); where the features_partner partner was a tie:
               cogreen_secs / _frac, excl_secs_p / _q / _min, partner_lead_start / _end, pex_* (7), every __pdiff (20).
  function     excl_partner_diff of the detector's predicted phase (the only affected column among the 229 the
  frame v6e    function trees use).
Counts of changed values are written to f76/changes.json.

    python f76_pool.py          -> f76/phase/pool.parquet (+ pool_meta.json, pool_sim.parquet), f76/frame_v6e/feat_frame.parquet
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import features as F  # noqa: E402
import features_partner as FP  # noqa: E402

F76 = DC_WORK / "final_v3_work" / "f76"
T57P = DC_WORK / "trees57" / "phase"
FRAME = DC_WORK / "trackA" / "v3" / "frame_v6e"
K4 = ["DeviceId", "Detector", "cand_phase", "win"]
DETK = ["DeviceId", "win", "Detector"]
PDIFF = list(dict.fromkeys(FP.PDIFF_FEATS + ["on_lift_green", "occ_lift_green", "f_on_green", "excl_diff_min",
                                             "release_frac_long", "call43_fwd_lift"]))
PMETA = ["cogreen_secs", "cogreen_frac", "excl_secs_p", "excl_secs_q", "excl_secs_min", "partner_lead_start",
         "partner_lead_end"]
PEX = ["pex_lift_p", "pex_lift_q", "pex_diff", "pex_occ_diff", "pex_diff_shrunk", "pex_n_on", "pex_share_p"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def changed(old, new):
    old, new = np.asarray(old, float), np.asarray(new, float)
    nm = np.isnan(old) != np.isnan(new)
    d = np.abs(np.nan_to_num(old) - np.nan_to_num(new)) > 1e-6 * np.maximum(1.0, np.abs(np.nan_to_num(old)))
    return nm | (d & ~np.isnan(old) & ~np.isnan(new))


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower()


def companions(df: pd.DataFrame, f: str) -> pd.DataFrame:
    """features.add_rank_features for one column (same maths, float64 in, float32 out)."""
    v = df[f].astype(np.float32).astype(np.float64)
    g = v.groupby([df[c] for c in DETK], sort=False)
    mu, sd = g.transform("mean"), g.transform("std")
    mx = g.transform("max")
    out = pd.DataFrame({f"{f}__rank": g.rank(pct=True, method="average"),
                        f"{f}__z": (v - mu) / sd.replace(0, np.nan),
                        f"{f}__mgap": v - mx,
                        f"{f}__argmax": (v >= mx).astype("float32")}, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def load_new() -> tuple[pd.DataFrame, pd.DataFrame]:
    ex = pd.concat([pd.read_parquet(f) for f in sorted(F76.glob("excl_*.parquet"))], ignore_index=True)
    pt = [pd.read_parquet(f) for f in sorted(F76.glob("ptie_*.parquet"))]
    pt = pd.concat(pt, ignore_index=True) if pt else pd.DataFrame()
    for d in (ex, pt):
        if len(d):
            d["key_id"] = norm_id(d.DeviceId)
    return ex, pt


def main():
    ex, pt = load_new()
    ex = ex.drop_duplicates(["key_id", "Detector", "cand_phase", "win"])
    res = {}
    # ------------------------------------------------------------------ phase pool
    out = F76 / "phase"
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(T57P / "pool.parquet")
    meta = json.load(open(T57P / "pool_meta.json"))
    fc = meta["features"]
    pool["key_id"] = norm_id(pool.DeviceId)
    e = pool[["key_id", "Detector", "cand_phase", "win"]].merge(
        ex.astype({"Detector": pool.Detector.dtype, "cand_phase": pool.cand_phase.dtype})[
            ["key_id", "Detector", "cand_phase", "win", "epd_new", "n_tied"]],
        on=["key_id", "Detector", "cand_phase", "win"], how="left")
    miss = e.n_tied.isna() & e.epd_new.isna()
    res["pool_rows"] = int(len(pool))
    res["pool_rows_not_rescanned"] = int(miss.sum())
    log(f"pool {len(pool):,} rows; not found in the scan {int(miss.sum()):,}")
    new = np.where(miss, pool.excl_partner_diff.to_numpy(float), e.epd_new.to_numpy(float))
    ch = changed(pool.excl_partner_diff.to_numpy(), new)
    lab = pool.Phase.notna().to_numpy()
    res["excl_partner_diff"] = {"changed_rows": int(ch.sum()), "share_rows": float(ch.mean()),
                                "changed_labelled_rows": int((ch & lab).sum()),
                                "tie_rows (n_tied>=2)": int((e.n_tied >= 2).sum()),
                                "by_n_tied": {str(int(k)): int(v) for k, v in
                                              pd.Series(ch).groupby(e.n_tied.fillna(-1).to_numpy()).sum().items()}}
    det_ch = pd.Series(ch).groupby([pool[c].to_numpy() for c in DETK]).transform("max").to_numpy().astype(bool)
    res["excl_partner_diff"]["detector_windows_touched_share"] = float(
        pool.loc[det_ch, DETK].drop_duplicates().shape[0] / pool[DETK].drop_duplicates().shape[0])
    log(f"excl_partner_diff: {res['excl_partner_diff']}")
    pool["excl_partner_diff"] = new.astype(np.float32)
    # every cross-candidate companion recomputed from the stored float32 values (= the new production rule, which
    # ranks the float32-rounded value so ulp-level near-ties are true ties)
    for f in F.RANK_FEATS:
        if f not in pool.columns or f + "__rank" not in fc:
            continue
        cn = companions(pool, f)
        for c in cn.columns:
            o = pool[c].to_numpy(float)
            chc = changed(o, cn[c].to_numpy(float))
            res.setdefault("companions_changed", {})[c] = int(chc.sum())
            if f != "excl_partner_diff":
                res.setdefault("companions_changed_not_epd_rows_in_epd_detwin", {})[c] = int((chc & det_ch).sum())
            pool[c] = cn[c].to_numpy()
    log(f"companions changed: {res['companions_changed']}")
    # partner ties (features_partner)
    if len(pt):
        pt = pt.astype({"Detector": pool.Detector.dtype}) if "Detector" in pt else pt
        pt = pt.rename(columns={"det": "Detector", "p": "cand_phase"})
        pt = pt.astype({"Detector": pool.Detector.dtype, "cand_phase": pool.cand_phase.dtype})
        pt = pt.drop_duplicates(["key_id", "Detector", "cand_phase", "win"])
        j = pool[["key_id", "Detector", "cand_phase", "win"]].reset_index().merge(
            pt[["key_id", "Detector", "cand_phase", "win", "partner_set"] + PMETA + PEX],
            on=["key_id", "Detector", "cand_phase", "win"], how="inner")
        idx = j["index"].to_numpy()
        res["partner_tie_rows"] = int(len(idx))
        res["partner_tie_labelled_rows"] = int(lab[idx].sum())
        for c in PMETA + PEX:
            if c not in pool.columns:
                continue
            o = pool.loc[idx, c].to_numpy(float)
            res.setdefault("partner_cols_changed", {})[c] = int(changed(o, j[c].to_numpy(float)).sum())
            pool.loc[idx, c] = j[c].to_numpy().astype(np.float32)
        # pdiff = f(p) - mean over the tied partners of f(q)
        lft = j[["index", "key_id", "Detector", "win", "partner_set"]].copy()
        lft["q"] = lft.partner_set.str.split(",")
        lft = lft.explode("q")
        lft["q"] = lft.q.astype(int).astype(pool.cand_phase.dtype)
        right = pool[["key_id", "Detector", "win", "cand_phase"] + PDIFF].rename(columns={"cand_phase": "q"})
        mm = lft.merge(right, on=["key_id", "Detector", "win", "q"], how="left").groupby("index")[PDIFF].mean()
        ii = mm.index.to_numpy()
        for f in PDIFF:
            c = f + "__pdiff"
            if c not in pool.columns:
                continue
            nv = pool.loc[ii, f].to_numpy(np.float32) - mm[f].to_numpy(np.float32)
            o = pool.loc[ii, c].to_numpy(float)
            res.setdefault("pdiff_changed", {})[c] = int(changed(o, nv).sum())
            pool.loc[ii, c] = nv
    else:
        res["partner_tie_rows"] = 0
    pool = pool.drop(columns=["key_id"])
    for c in fc:
        if pool[c].dtype != np.float32:
            pool[c] = pool[c].astype(np.float32)
    pool.to_parquet(out / "pool.parquet", index=False)
    shutil.copy(T57P / "pool_sim.parquet", out / "pool_sim.parquet")
    json.dump(meta, open(out / "pool_meta.json", "w"), indent=1)
    shutil.copy(T57P / "score_rows.parquet", out / "score_rows.parquet")
    log(f"pool written -> {out}")
    del pool
    # ------------------------------------------------------------------ function frame v6e
    fo = F76 / "frame_v6e"
    fo.mkdir(parents=True, exist_ok=True)
    fr = pd.read_parquet(FRAME / "feat_frame.parquet")
    k = fr[["DeviceId", "period", "Detector", "win", "pred_phase"]].copy()
    k["key_id"] = np.where(k.period == "stg", norm_id(k.DeviceId) + "@stg", norm_id(k.DeviceId))
    exf = ex.rename(columns={"cand_phase": "pred_phase"})
    exf = exf.astype({"Detector": fr.Detector.dtype})
    exf["pred_phase"] = exf.pred_phase.astype(float)
    k["pred_phase"] = k.pred_phase.astype(float)
    exf["_hit"] = True
    m = k.merge(exf[["key_id", "Detector", "pred_phase", "win", "epd_new", "_hit"]],
                on=["key_id", "Detector", "pred_phase", "win"], how="left")
    hit = m._hit.eq(True).to_numpy()
    newf = np.where(hit, m.epd_new.to_numpy(float), fr.excl_partner_diff.to_numpy(float))
    chf = changed(fr.excl_partner_diff.to_numpy(), newf)
    res["frame"] = {"rows": int(len(fr)), "rescanned": int(hit.sum()), "rescanned_nan": int((hit & m.epd_new.isna().to_numpy()).sum()),
                    "old_nan": int(fr.excl_partner_diff.isna().sum()),
                    "pred_phase_missing": int(fr.pred_phase.isna().sum()),
                    "changed_rows": int(chf.sum()), "share": float(chf.mean())}
    log(f"frame v6e: {res['frame']}")
    fr["excl_partner_diff"] = newf.astype(fr.excl_partner_diff.dtype)
    fr.to_parquet(fo / "feat_frame.parquet", index=False)
    shutil.copy(FRAME / "feat_meta.json", fo / "feat_meta.json")
    json.dump(res, open(F76 / "changes.json", "w"), indent=1)
    log(f"-> {F76 / 'changes.json'}")


if __name__ == "__main__":
    main()
