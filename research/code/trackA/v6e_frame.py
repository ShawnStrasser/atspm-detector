"""Note 49: funcframe_v6e = funcframe_v6 with every PHASE-DERIVED column rebuilt from the phase_v3 blend OOF
(`fv3e_phase.py`, the phase input the final_v3 candidate ships) instead of the older tree-only phase OOF.

Same rows, same order, same code: `v5_frame.frame_rows` (-> function_v4.load_pairs + function_v3 shape /
sibling / lag features) with the new probabilities, then `a2_features.build` (expert det / cyc / pair) on the
new predicted phases.  Bookkeeping columns (labels, fold, split, health_flag, period, wgroup) are copied from
v6 unchanged.  YR / lag inputs = the union of the tables the v4 / v5 / v6 rows were built from.
Check: rows whose whole signal-window keeps its predicted phases must come out identical except top_prob.

    python v6e_frame.py rows       # -> %DC_WORK%/function_v4/funcframe_v6e.parquet
    python v6e_frame.py expert     # -> %DC_WORK%/trackA/feat_expert_{det,cyc,pair}_{dec,stg}_v6e.parquet
    python v6e_frame.py check      # identity on unchanged signal-windows -> function_v3e/frame_check.json
    add --tag v6t for the trees-only phase_v3 OOF variant (funcframe_v6t, *_v6t expert tables)
Locked_v2 asserted absent.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v5_frame as V5  # noqa: E402
import v6_frame as V6  # noqa: E402  (10 GB / 8-thread DuckDB, locked_v2 hold-out patch)

DCW = V5.DCW
FRAME_V6 = V6.FRAME_V6
FRAME_V6E = DCW / "function_v4" / "funcframe_v6e.parquet"
TAG, PROBS = "v6e", "v3blend"    # --tag v6t: the trees-only phase_v3 OOF (what candidate v2 feeds its function model)
W = DCW / "final_v3_work" / "function_v3e"
PH = W / "phase"
KEY = V5.KEY
BOOK = ["func5", "Function", "cfg_phase", "fold", "split", "health_flag", "period", "wgroup"]
log, P = V5.log, V5.P
CHUNK = 120


def _union(files, key, dest) -> Path:
    if dest.exists():
        return dest
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n = len(d)
    d = d.drop_duplicates(key, keep="first")
    d.to_parquet(dest, index=False)
    log(f"{dest.name}: {len(d):,} rows ({n - len(d):,} duplicate rows of re-built verify signals dropped)")
    return dest


def inputs(per: str):
    import function_v4 as fv4
    if per == "dec":
        return fv4._files("dec", "yr"), fv4._files("dec", "lag")
    yr = [fv4.V4_SFEAT / "func_yr_extra_stg.parquet", V5.OUT / "func_yr_extra_stg_new.parquet",
          V6.OUT / "func_yr_extra_stg_new.parquet"]
    lag = [fv4.V4_SFEAT / "det_lag_stg.parquet", V5.OUT / "det_lag_stg_new.parquet",
           V6.OUT / "det_lag_stg_new.parquet"]
    return ([_union(yr, ["DeviceId", "Detector", "cand_phase", "win"], W / "yr_stg_union.parquet")],
            [_union(lag, ["DeviceId", "Detector", "other", "win"], W / "lag_stg_union.parquet")])


def stage_rows(a) -> None:
    V6._patch_holdouts()
    t0 = time.time()
    old = pd.read_parquet(FRAME_V6)
    assert not old.DeviceId.str.lower().isin(V6.locked()).any()
    cols, dt = list(old.columns), old.dtypes.to_dict()
    empty = pd.read_parquet(V6._patch_holdouts().LABELS_V4).iloc[:0]
    parts = []
    for per in ("dec", "stg"):
        probs = pd.read_parquet(PH / f"phase_pred_{PROBS}_{per}.parquet")
        yr, lag = inputs(per)
        sigs = sorted(old.loc[old.period == per, "DeviceId"].unique())
        for i in range(0, len(sigs), CHUNK):
            keep = set(sigs[i:i + CHUNK])
            top = V5.frame_rows(per, keep, probs[probs.DeviceId.isin(keep)], yr, lag, empty, {}, {})
            top = top[[c for c in cols if c not in BOOK or c in ("period", "wgroup")]]
            parts.append(top)
            log(f"[{per}] signals {i}-{i + len(keep)}: {top.shape} ({time.time()-t0:.0f}s)")
    new = pd.concat(parts, ignore_index=True).drop(columns=["wgroup"])
    assert not new.duplicated(KEY).any()
    new["Detector"] = new.Detector.astype(old.Detector.dtype)
    new = old[KEY].merge(new, on=KEY, how="left")          # old row order, one row per old row
    assert len(new) == len(old), (len(new), len(old))
    miss = new.pred_phase.isna()
    assert not miss.any(), f"{int(miss.sum())} frame rows got no phase_v3 prediction"
    for c in BOOK:
        new[c] = old[c].to_numpy()
    new = new[cols]
    for c in cols:
        if new[c].dtype != dt[c]:
            new[c] = new[c].astype(dt[c])
    assert (new[KEY].astype(str).values == old[KEY].astype(str).values).all()
    new.to_parquet(FRAME_V6E, index=False)
    ch = (new.pred_phase.to_numpy() != old.pred_phase.to_numpy())
    log(f"funcframe_v6e {new.shape} -> {FRAME_V6E}; predicted phase changed on {ch.mean():.2%} of rows "
        f"({time.time()-t0:.0f}s)")


def stage_expert(a) -> None:
    import a2_features as A2F
    fr = pd.read_parquet(FRAME_V6E, columns=["DeviceId", "Detector", "win", "period", "pred_phase"])
    tfile = W / f"expert_targets_{TAG}.parquet"
    fr.to_parquet(tfile, index=False)
    A2F.FRAME = tfile
    A2F.WORK = W / ("expert" if TAG == "v6e" else f"expert_{TAG}")
    A2F.WORK.mkdir(parents=True, exist_ok=True)
    A2F.connect = _a2_connect
    for per in ("dec", "stg"):
        if all((A2F.WORK / f"feat_expert_{t}_{per}.parquet").exists() for t in ("det", "cyc", "pair")):
            continue
        A2F.build(per)
    lk = V6.locked()
    for per in ("dec", "stg"):
        for tag in ("det", "cyc", "pair"):
            ref = pd.read_parquet(V5.EXPERT_OLD / f"feat_expert_{tag}_{per}_v6.parquet").iloc[:0]
            e = pd.read_parquet(A2F.WORK / f"feat_expert_{tag}_{per}.parquet")
            # the det / pair SQL runs over every signal of the event cache: keep the frame's signals only
            e = e[e.DeviceId.isin(set(fr.DeviceId))]
            e = e[list(ref.columns)].astype(ref.dtypes.to_dict())
            assert not e.DeviceId.str.lower().isin(lk).any() and not e.duplicated(KEY).any()
            e.to_parquet(V5.EXPERT_OLD / f"feat_expert_{tag}_{per}_{TAG}.parquet", index=False)
            log(f"feat_expert_{tag}_{per}_{TAG}: {e.shape}")


def _a2_connect():
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=8")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


def stage_check(a) -> None:
    """Rows of signal-windows whose predicted phases are all unchanged must be identical except top_prob."""
    old = pd.read_parquet(FRAME_V6)
    new = pd.read_parquet(FRAME_V6E)
    assert (old[KEY].values == new[KEY].values).all()
    ch = pd.Series(old.pred_phase.to_numpy() != new.pred_phase.to_numpy())
    sw = ch.groupby([old.DeviceId, old.period, old.win]).transform("max").to_numpy()
    same = ~sw
    res = {"rows": len(old), "pred_phase_changed_rows": float(ch.mean()),
           "signal_windows_changed_share_of_rows": float(sw.mean()),
           "changed_by_wgroup": ch.groupby(old.wgroup.to_numpy()).mean().round(4).to_dict(),
           "changed_by_period": ch.groupby(old.period.to_numpy()).mean().round(4).to_dict()}
    diff = {}
    for c in old.columns:
        if c in KEY or c == "top_prob":
            continue
        x, y = old.loc[same, c], new.loc[same, c]
        if pd.api.types.is_numeric_dtype(x):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            eq = (np.isnan(xv) & np.isnan(yv)) | (np.abs(xv - yv) <= 1e-5 * (1 + np.abs(yv)))
        else:
            eq = (x.isna() & y.isna()) | (x.astype(str) == y.astype(str))
        if not np.all(eq):
            diff[c] = int((~np.asarray(eq)).sum())
    res["frame_columns_differing_on_unchanged_rows"] = diff
    ex = {}
    for per in ("dec", "stg"):
        m = (old.period == per).to_numpy() & same
        k = old.loc[m, KEY]
        for tag in ("det", "cyc", "pair"):
            a_ = pd.read_parquet(V5.EXPERT_OLD / f"feat_expert_{tag}_{per}_v6.parquet")
            b_ = pd.read_parquet(V5.EXPERT_OLD / f"feat_expert_{tag}_{per}_{TAG}.parquet")
            ja = k.merge(a_, on=KEY, how="left")
            jb = k.merge(b_, on=KEY, how="left")
            bad = {}
            for c in a_.columns:
                if c in KEY:
                    continue
                xv, yv = ja[c].to_numpy(float), jb[c].to_numpy(float)
                eq = (np.isnan(xv) & np.isnan(yv)) | (np.abs(xv - yv) <= 1e-5 * (1 + np.abs(yv)))
                if not eq.all():
                    bad[c] = int((~eq).sum())
            ex[f"{tag}_{per}"] = {"rows": len(k), "columns_differing": bad}
    res["expert_on_unchanged_rows"] = ex
    json.dump(res, open(W / ("frame_check.json" if TAG == "v6e" else f"frame_check_{TAG}.json"), "w"), indent=1)
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rows", "expert", "check"])
    ap.add_argument("--tag", default="v6e", choices=["v6e", "v6t"])
    a = ap.parse_args()
    if a.tag == "v6t":
        TAG, PROBS = "v6t", "v3trees"
        FRAME_V6E = DCW / "function_v4" / "funcframe_v6t.parquet"
    globals()[f"stage_{a.stage}"](a)
