"""Note 36 -- extend the function feature frame to the RELEASED half of NEWTEST (user decision 2026-09-24):
`funcframe_v6` = funcframe_v5 + the released signals' rows (v5 rows byte-identical), expert tables `_v6`,
`dc_work/folds_v4.csv` (folds_v3 unchanged + the released signals). No training here.

Nothing here defines a feature: rows, YR / lag and expert features come from the code that built v5
(`v5_frame.frame_rows` -> function_v4 / function_v3 / model/, build_features 'yrlag', a2_features).

Phase inputs (out-of-sample): the released signals were never trained on, so every model fitted without them is
out-of-sample for them. Used: the final_v1 LightGBM phase pipeline -- 3-seed ranker bag + joint decoder,
`model/weights/phase_lgbm_v4_s{0,1,2}.txt` + `decode_lgbm_v4.txt` (= the trees of the shipped final_v2), fitted on
the 701-signal pool (DEV + NEWTRAIN). It is the same pipeline whose six-fold OOF gives the v5 rows' phase column.
Every detector with pair features is scored (as in production), not only the ones with a timing target.
Only the 71 released signals are ever loaded; the locked half (locked_v2.csv) is asserted absent everywhere.

    python v6_frame.py folds      # dc_work/folds_v4.csv (existing assignments unchanged)
    python v6_frame.py phase      # final_v1 trees -> phase probabilities for the released signals, 22 windows
    python v6_frame.py yrlag      # YR + lag tables (Sept-2026 pull; own process, DC_WORK=stg)
    python v6_frame.py rows       # frame rows
    python v6_frame.py expert     # expert det / cyc / pair
    python v6_frame.py merge      # funcframe_v6 + feat_expert_*_v6 (v5 rows byte-identical)
    python v6_frame.py check      # coverage (+ top-1 vs official timing on the released rows: a pipeline check)
Outputs under %DC_WORK%/trackA/v6/ and %DC_WORK%/function_v4/funcframe_v6.parquet.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v5_frame as V5  # noqa: E402

REPO, DCW = V5.REPO, V5.DCW
OUT = DCW / "trackA" / "v6"
FRAME_V5 = V5.FRAME_V5
FRAME_V6 = DCW / "function_v4" / "funcframe_v6.parquet"
FOLDS_V3, FOLDS_V4 = V5.FOLDS_V3, DCW / "folds_v4.csv"
RELEASED = DCW / "official" / "newtest_released.csv"
LOCKED_V2 = DCW / "official" / "locked_v2.csv"
WEIGHTS = REPO / "model" / "weights"
STG, EXPERT_OLD, KEY = V5.STG, V5.EXPERT_OLD, V5.KEY
SPLIT = "NEWTEST_RELEASED"
log, P = V5.log, V5.P


def duck():
    """DuckDB capped for this task (10 GB, 8 threads), temp files in dc_work/tmp."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=8; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


V5.duck = duck
V5._a2_connect = duck


def released() -> list[str]:
    r = sorted(pd.read_csv(RELEASED).DeviceId.astype(str).str.lower())
    assert len(r) == 71 and not set(r) & locked()
    return r


def locked() -> set:
    return set(pd.read_csv(LOCKED_V2).DeviceId.astype(str).str.lower())


def _patch_holdouts():
    """function_v4's hold-out assertion, pointed at locked_v2 (43 TEST + the NEWTEST half that stays locked)."""
    sys.path.insert(0, str(HERE.parent))
    import rpath  # noqa: F401
    import function_v4 as fv4
    fv4.TEST_CFG = V5.LOCKED[0]
    lk = pd.read_csv(LOCKED_V2)
    t = set(lk[lk.set == "TEST"].DeviceId.str.lower())
    n = set(lk[lk.set == "NEWTEST"].DeviceId.str.lower())
    fv4._holdouts = lambda: (t, n)
    return fv4


def stg_ids(ids: list[str]) -> list[str]:
    """The ids as spelled in the Sept-2026 feature tables."""
    con = V5.duck()
    con.register("keep", pd.DataFrame({"d": ids}))
    got = con.execute(f"SELECT DISTINCT DeviceId FROM '{P(STG / 'features' / 'pair_features_stg.parquet')}' "
                      f"WHERE lower(DeviceId) IN (SELECT d FROM keep)").df().DeviceId.tolist()
    return sorted(got)


# ------------------------------------------------------------------ folds
def stage_folds(a) -> None:
    """folds_v4 = folds_v3 (unchanged) + the released signals, spread over folds 1-5 (fold 0 stays the 2025 model's
    own hold-out, as for the v5 newcomers), evenly by a greedy balance of their detector count, ties by a seed-0 draw."""
    f3 = pd.read_csv(FOLDS_V3)
    f3["DeviceId"] = f3.DeviceId.str.lower()
    rel = released()
    assert not set(rel) & set(f3.DeviceId)
    con = V5.duck()
    con.register("keep", pd.DataFrame({"d": rel}))
    nd = con.execute(f"SELECT lower(DeviceId) DeviceId, count(DISTINCT Detector) n FROM "
                     f"'{P(STG / 'features' / 'pair_features_stg.parquet')}' WHERE lower(DeviceId) IN "
                     f"(SELECT d FROM keep) GROUP BY 1").df()
    nd = pd.DataFrame({"DeviceId": rel}).merge(nd, how="left").fillna({"n": 0})
    load = {k: 0.0 for k in range(1, V5.N_FOLDS)}  # spread the released half evenly (v5_frame's rule)
    rng = np.random.default_rng(0)
    nd = nd.assign(r=rng.random(len(nd))).sort_values(["n", "r"], ascending=[False, True])
    rows = []
    for d, n in zip(nd.DeviceId, nd.n):
        k = min(load, key=lambda x: (load[x], x))
        load[k] += n
        rows.append({"DeviceId": d, "fold": k, "src": "newtest_released"})
    out = pd.concat([f3, pd.DataFrame(rows)], ignore_index=True)
    assert not out.DeviceId.duplicated().any() and not out.DeviceId.isin(locked()).any()
    chk = f3.merge(out, on="DeviceId")
    assert (chk.fold_x == chk.fold_y).all()
    out[["DeviceId", "fold", "src"]].to_csv(FOLDS_V4, index=False)
    log(f"folds_v4: {len(out)} signals; released by fold "
        f"{pd.DataFrame(rows).fold.value_counts().sort_index().to_dict()} -> {FOLDS_V4}")


def fold_map() -> dict:
    f = pd.read_csv(FOLDS_V4)
    return dict(zip(f.DeviceId.str.lower(), f.fold.astype(int)))


# ------------------------------------------------------------------ phase
def stage_phase(a) -> None:
    sys.path.insert(0, str(HERE.parent))
    import rpath  # noqa: F401
    import lightgbm as lgb
    import train_official as T
    import decode_train as dec
    rel = released()
    ids = stg_ids(rel)
    assert len(ids) == len(rel)
    t0 = time.time()
    df, sim = T.load_stg(keep=set(ids))
    assert not df.DeviceId.str.replace("@stg", "").str.lower().isin(locked()).any()
    rmeta = json.load(open(WEIGHTS / "phase_lgbm_v4.json"))
    dmeta = json.load(open(WEIGHTS / "decode_lgbm_v4.json"))
    fc = rmeta["features"]
    miss = [c for c in fc if c not in df.columns]
    assert not miss, miss
    S = [lgb.Booster(model_file=str(WEIGHTS / f"phase_lgbm_v4_s{j}.txt")).predict(df[fc])
         for j in range(rmeta["n_models"])]
    pr = df[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    pr["p0"] = np.mean([T.to_prob(df, s) for s in S], axis=0)
    X = dec.assemble(pr, pairs=df, sim=sim)
    s2 = lgb.Booster(model_file=str(WEIGHTS / "decode_lgbm_v4.txt")).predict(X[dmeta["features"]])
    X["prob"] = T.norm_prob(X, s2)
    o = pr.merge(X[["DeviceId", "Detector", "win", "cand_phase", "prob"]],
                 on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    assert o.prob.notna().all()
    o["DeviceId"] = o.DeviceId.str.replace("@stg", "", regex=False).str.lower()
    o["Detector"] = o.Detector.astype(np.int16)
    o["cand_phase"] = o.cand_phase.astype(np.int16)
    OUT.mkdir(parents=True, exist_ok=True)
    o[["DeviceId", "Detector", "win", "cand_phase", "prob", "p0"]].to_parquet(OUT / "phase_pred_new_stg.parquet",
                                                                              index=False)
    json.dump({"model": "final_v1 trees (model/weights phase_lgbm_v4_s0-2 + decode_lgbm_v4), fitted on the 701-signal "
                        "DEV + NEWTRAIN pool; never trained on NEWTEST", "signals": int(o.DeviceId.nunique()),
               "rows": len(o), "windows": sorted(o.win.unique().tolist()),
               "built": time.strftime("%Y-%m-%d %H:%M")}, open(OUT / "phase_pred_meta.json", "w"), indent=1)
    log(f"phase: {o.DeviceId.nunique()} signals, {len(o):,} pair rows, {o.win.nunique()} windows "
        f"({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ YR + lag
def stage_yrlag(a) -> None:
    ids = released()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "yrlag_signals_stg.txt").write_text("\n".join(ids))
    env = dict(os.environ, DC_WORK=str(STG), DC_WORK_MAIN=str(DCW))
    subprocess.run([sys.executable, __file__, "_yrlag_child", "--threads", str(a.threads)], env=env, check=True)


def stage__yrlag_child(a) -> None:
    sys.path.insert(0, str(HERE.parent))
    import rpath  # noqa: F401
    import build_features as bf
    import windows_stg  # noqa: F401  -- registers 'stgall'
    out = Path(os.environ["DC_WORK_MAIN"]) / "trackA" / "v6"
    devs = set((out / "yrlag_signals_stg.txt").read_text().split())
    con = bf.connect(threads=a.threads)
    meta = con.sql(f"SELECT DeviceId FROM read_parquet('{P(bf.CACHE / 'signal_meta.parquet')}')"
                   " ORDER BY DeviceId").df().DeviceId.tolist()
    keep = [d for d in meta if d.lower() in devs]
    con.close()
    assert len(keep) == len(devs), (len(keep), len(devs))
    bf._signals = lambda con, limit: keep            # only which signals, not how
    bf.build("yrlag", str(out / "func_yr_extra_stg_new.parquet"),
             str(out / "det_lag_stg_new.parquet"), "stgall", a.threads, 8, 0)


# ------------------------------------------------------------------ rows
def stage_rows(a) -> None:
    fv4 = _patch_holdouts()
    t0 = time.time()
    cols, dt = V5._fr_schema()
    rel = released()
    fold_of = {d: f for d, f in fold_map().items() if d in set(rel)}
    split_of = {d: SPLIT for d in rel}
    probs = pd.read_parquet(OUT / "phase_pred_new_stg.parquet")[["DeviceId", "Detector", "win", "cand_phase", "prob"]]
    empty = pd.read_parquet(fv4.LABELS_V4).iloc[:0]
    keep = set(stg_ids(rel))
    probs["DeviceId"] = probs.DeviceId.map({k.lower(): k for k in keep})
    top = V5.frame_rows("stg", keep, probs, [OUT / "func_yr_extra_stg_new.parquet"],
                        [OUT / "det_lag_stg_new.parquet"], empty, fold_of, split_of)  # maps take lower-case ids
    new = V5.to_v4_schema(top, cols, dt)
    assert new.fold.notna().all() and new.split.eq(SPLIT).all()
    new.to_parquet(OUT / "frame_new.parquet", index=False)
    log(f"rows: {new.shape}, {new.DeviceId.nunique()} signals ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ expert
def stage_expert(a) -> None:
    import a2_features as A2F
    new = pd.read_parquet(OUT / "frame_new.parquet", columns=["DeviceId", "Detector", "win", "period", "pred_phase"])
    con = V5.duck()
    devs = sorted(set(new.DeviceId))
    cdir = OUT / "cache_stg"
    cdir.mkdir(parents=True, exist_ok=True)
    con.register("keep", pd.DataFrame({"d": [d.lower() for d in devs]}))
    for f in ("det_intervals", "phase_cycles", "coord_state"):
        con.execute(f"COPY (SELECT * FROM '{P(V5.CACHE['stg'] / (f + '.parquet'))}' WHERE "
                    f"lower(DeviceId) IN (SELECT d FROM keep)) TO '{P(cdir / (f + '.parquet'))}' (FORMAT parquet)")
    tfile = OUT / "expert_targets_stg.parquet"
    new.to_parquet(tfile, index=False)
    A2F.CACHE = {**A2F.CACHE, "stg": cdir}
    A2F.FRAME = tfile
    A2F.WORK = OUT / "expert_stg"
    A2F.WORK.mkdir(parents=True, exist_ok=True)
    A2F.connect = V5._a2_connect
    log(f"[stg] expert features for {len(devs)} signals")
    A2F.build("stg")


# ------------------------------------------------------------------ merge
def stage_merge(a) -> None:
    t0 = time.time()
    lk = locked()
    v5 = pd.read_parquet(FRAME_V5)
    new = pd.read_parquet(OUT / "frame_new.parquet")
    assert list(new.columns) == list(v5.columns)
    assert not (set(new.DeviceId.str.lower()) & set(v5.DeviceId.str.lower()))
    v6 = pd.concat([v5, new], ignore_index=True)
    for c in v5.columns:
        assert v6[c].dtype == v5[c].dtype, c
    assert not v6.DeviceId.str.lower().isin(lk).any()
    assert not v6.duplicated(KEY).any()
    v6.to_parquet(FRAME_V6, index=False)
    back = pd.read_parquet(FRAME_V6)
    assert back.iloc[:len(v5)].reset_index(drop=True).equals(v5), "v5 rows changed"
    newsig = set(new.DeviceId)
    ex = {}
    for per in ("dec", "stg"):
        for tag in ("det", "cyc", "pair"):
            old = pd.read_parquet(EXPERT_OLD / f"feat_expert_{tag}_{per}_v5.parquet")
            f = OUT / f"expert_{per}" / f"feat_expert_{tag}_{per}.parquet"
            fresh = pd.read_parquet(f) if f.exists() else old.iloc[:0]
            fresh = fresh[fresh.DeviceId.isin(newsig)][list(old.columns)].astype(old.dtypes.to_dict())
            assert not set(fresh.DeviceId) & set(old.DeviceId)
            e = pd.concat([old, fresh], ignore_index=True)
            assert not e.DeviceId.str.lower().isin(lk).any() and not e.duplicated(KEY).any()
            e.to_parquet(EXPERT_OLD / f"feat_expert_{tag}_{per}_v6.parquet", index=False)
            assert pd.read_parquet(EXPERT_OLD / f"feat_expert_{tag}_{per}_v6.parquet").iloc[:len(old)] \
                .reset_index(drop=True).equals(old.reset_index(drop=True)), f"{tag}_{per} v5 rows changed"
            ex[f"{tag}_{per}"] = {"v5_rows": len(old), "fresh": len(fresh)}
    summ = {"v5_rows": len(v5), "new_rows": len(new), "v6_rows": len(v6), "v6_signals": int(v6.DeviceId.nunique()),
            "new_signals": len(newsig), "expert_v6": ex, "v5_rows_identical": True,
            "built": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(summ, open(OUT / "merge.json", "w"), indent=1, default=str)
    log(json.dumps(summ, indent=1, default=str) + f"\nmerge done in {time.time()-t0:.0f}s -> {FRAME_V6}")


# ------------------------------------------------------------------ check
def stage_check(a) -> None:
    rel = set(released())
    fr = pd.read_parquet(FRAME_V6, columns=["DeviceId", "Detector", "win", "wgroup", "pred_phase", "fold", "split"])
    fr["DeviceId"] = fr.DeviceId.str.lower()
    new = fr[fr.DeviceId.isin(rel)]
    v3 = pd.read_parquet(REPO / "research/labels/function_labels_v3.parquet")
    v3["DeviceId"] = v3.DeviceId.str.lower()
    r = v3[v3.released_from_newtest.astype(bool)]
    full = new[new.wgroup == "full"]
    j = r.merge(full.rename(columns={"Detector": "detector"}), on=["DeviceId", "detector"], how="left")
    lab = j[j.function.notna()]
    off = pd.read_parquet(DCW / "official/labels_official.parquet",
                          columns=["DeviceId", "Detector", "target_type", "target_num", "switch_phase"])
    off = off[off.target_type == "phase"]
    off["DeviceId"] = off.DeviceId.str.lower()
    t = full.merge(off, on=["DeviceId", "Detector"], how="inner")
    summ = {"released_signals": len(rel), "signals_in_frame": int(new.DeviceId.nunique()),
            "missing_signals": sorted(rel - set(new.DeviceId)), "rows": len(new),
            "rows_by_wgroup": new.wgroup.value_counts().to_dict(),
            "detectors_full_window": len(full), "folds": new.groupby("fold").DeviceId.nunique().to_dict(),
            "labelled_rows_v3": len(lab), "labelled_in_frame": int(lab.win.notna().sum()),
            "train_use_rows": int(r.train_use.sum()),
            "train_use_in_frame": int(j[j.train_use.astype(bool)].win.notna().sum()),
            "labelled_not_in_frame_why": {
                "dead_or_no_data": int((lab.win.isna() & lab.dead.astype(bool)).sum()),
                "other": int((lab.win.isna() & ~lab.dead.astype(bool)).sum())},
            "pipeline_check_top1_vs_timing_full_window": round(float(
                (t.pred_phase.astype(int) == t.target_num.astype(int)).mean()), 4), "n_timing": len(t)}
    json.dump(summ, open(OUT / "check.json", "w"), indent=1, default=str)
    log(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["folds", "phase", "yrlag", "_yrlag_child", "rows", "expert", "merge", "check"])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
