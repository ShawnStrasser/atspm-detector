"""Note 28 add-on -- extend the function feature frame (funcframe_v4) to every signal of the
print-derived label table v3 that has usable data: `funcframe_v5` + expert tables v5 + folds_v3.

Nothing here defines a feature. Every feature comes from the code that built the originals:
    base / v2 / sim pair tables   already exist for the Sept-2026 pull (build_stg_features.py)
    YR + lag tables               build_features.build(what="yrlag") -> model/features_yellowred.py
    frame rows                    function_v4.load_pairs + function_v3 add_shape / add_sibling /
                                  add_lag (-> model/function.py), mirroring function_v4.stage_frame
    expert det / cyc / pair       a2_features.build (its SQL), targets = the new rows' pred_phase
Phase inputs are out-of-fold: the stage-12 LightGBM ranker-bag + decoder OOF
(`official/final_v1/oof_bywindow.parquet`, 701 signals; DEV Dec-2024 + NEWTRAIN Sept-2026), which
is the tree pipeline the frame's own phase column comes from (function_v4.stage_phase, not saved
as fold models). A signal-period with no such OOF is left out, never scored in-sample.

    python v5_frame.py audit        # why each v3 signal is missing from funcframe_v4
    python v5_frame.py folds        # dc_work/folds_v3.csv (superset; existing folds unchanged)
    python v5_frame.py phase        # OOF phase probabilities for the new signal-periods
    python v5_frame.py yrlag        # YR + lag tables (Sept-2026 pull; own process, DC_WORK=stg)
    python v5_frame.py rows         # frame rows for the new signal-periods (+ verify signals)
    python v5_frame.py expert       # expert det / cyc / pair for the new rows (+ verify signals)
    python v5_frame.py verify       # recomputed rows / YR / lag / expert == stored, 5 signals
    python v5_frame.py merge        # funcframe_v5 + feat_expert_*_v5 (old rows byte-identical)
Outputs under %DC_WORK%/trackA/v5/ and %DC_WORK%/function_v4/funcframe_v5.parquet.
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
REPO = Path(os.environ.get("DC_REPO") or HERE.parents[2])
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "trackA" / "v5"
LABELS_V3 = REPO / "research" / "labels" / "function_labels_v3.parquet"
FRAME_V4 = DCW / "function_v4" / "funcframe_v4.parquet"
FRAME_V5 = DCW / "function_v4" / "funcframe_v5.parquet"
FOLDS = DCW / "folds.csv"
FOLDS_V4 = DCW / "function_v4" / "folds_v4.csv"
FOLDS_NT = DCW / "official" / "newtrain_folds.csv"      # the stage-12/13 NEWTRAIN fold map
FOLDS_V3 = DCW / "folds_v3.csv"
PHASE_OOF = DCW / "official" / "final_v1" / "oof_bywindow.parquet"
PP_V4 = {p: DCW / "function_v4" / f"phase_pred_{p}.parquet" for p in ("dec", "stg")}
STG = DCW / "official" / "stg"
CACHE = {"dec": DCW / "cache", "stg": STG / "cache"}
EXPERT_OLD = DCW / "trackA"
LOCKED = [DCW / "data" / "splits" / "test_config.csv", DCW / "official" / "newtest_signals.csv"]
KEY = ["DeviceId", "Detector", "period", "win"]
N_FOLDS = 6
VERIFY_N = 5                                   # covered signals recomputed as a check


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    s = set()
    for f in LOCKED:
        s |= set(pd.read_csv(f).DeviceId.astype(str).str.lower())
    return s


def duck():
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=10; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


def sigs(con, sql) -> set:
    return {str(x).lower() for x in con.execute(sql).df().iloc[:, 0]}


def P(p: Path) -> str:
    return p.as_posix()


# ------------------------------------------------------------------ audit
def stage_audit(a) -> None:
    con = duck()
    v3 = pd.read_parquet(LABELS_V3, columns=["DeviceId", "DeviceName", "detector", "tier"])
    v3["DeviceId"] = v3.DeviceId.str.lower()
    lk = locked()
    assert not v3.DeviceId.isin(lk).any()
    fs = DCW / "features"
    S = {
        "frame": sigs(con, f"SELECT DISTINCT DeviceId FROM '{P(FRAME_V4)}'"),
        "labels_v4": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                               f"'{P(DCW / 'function_v4' / 'labels_v4.parquet')}'"),
        "folds_csv": set(pd.read_csv(FOLDS).DeviceId.str.lower()),
        "newtrain": set(pd.read_csv(FOLDS_NT).DeviceId.str.lower()),
        "events_dec": {p.name.split("=", 1)[1].lower()
                       for p in os.scandir(DCW / "cache" / "events") if p.is_dir()},
        "events_stg": set(pd.read_csv(STG / "signals.csv").DeviceId.str.lower()),
        "base_dec": sigs(con, f"SELECT DISTINCT DeviceId FROM read_parquet(["
                              f"'{P(fs / 'pair_features_windows.parquet')}',"
                              f"'{P(fs / 'pair_features_windows_B.parquet')}'])"),
        "base_stg": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                              f"'{P(STG / 'features' / 'pair_features_stg.parquet')}'"),
        "yrlag_dec": sigs(con, f"SELECT DISTINCT DeviceId FROM read_parquet(["
                               f"'{P(fs / 'det_lag.parquet')}','{P(fs / 'det_lag_B.parquet')}'])"),
        "yrlag_stg": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                               f"'{P(DCW / 'function_v4' / 'feat_stg' / 'det_lag_stg.parquet')}'"),
        "ex_det_stg": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                                f"'{P(EXPERT_OLD / 'feat_expert_det_stg.parquet')}'"),
        "ex_cyc_stg": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                                f"'{P(EXPERT_OLD / 'feat_expert_cyc_stg.parquet')}'"),
        "ex_det_dec": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                                f"'{P(EXPERT_OLD / 'feat_expert_det_dec.parquet')}'"),
        "ex_cyc_dec": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                                f"'{P(EXPERT_OLD / 'feat_expert_cyc_dec.parquet')}'"),
        "pp_v4": sigs(con, f"SELECT DISTINCT DeviceId FROM read_parquet(["
                           f"'{P(PP_V4['dec'])}','{P(PP_V4['stg'])}'])"),
        "oof_dec": sigs(con, f"SELECT DISTINCT DeviceId FROM '{P(PHASE_OOF)}' WHERE src='DEC'"),
        "oof_stg": sigs(con, f"SELECT DISTINCT replace(DeviceId,'@stg','') FROM '{P(PHASE_OOF)}'"
                             f" WHERE src='STG'"),
        "dec_cycles": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                                f"'{P(DCW / 'cache' / 'phase_cycles.parquet')}'"),
        "stg_cycles": sigs(con, f"SELECT DISTINCT DeviceId FROM "
                                f"'{P(STG / 'cache' / 'phase_cycles.parquet')}'"),
    }
    g = v3.groupby("DeviceId").agg(DeviceName=("DeviceName", "first"), tier=("tier", "first"),
                                   n_labels=("detector", "size")).reset_index()
    g = g[~g.DeviceId.isin(S["frame"])].reset_index(drop=True)
    for k, s in S.items():
        g[k] = g.DeviceId.isin(s)

    def why(r):
        if not r.events_dec and not r.events_stg:
            return "no_hi_res_data"
        if not r.base_dec and not r.base_stg:
            return ("no_green_events_no_candidate_phase" if not (r.stg_cycles or r.dec_cycles)
                    else "no_detector_events_no_pair_features")
        parts = []
        if r.base_stg:
            parts.append("stg:" + ("oof_phase" if r.oof_stg else "no_oof_phase"))
        if r.base_dec:
            parts.append("dec:" + ("oof_phase" if r.oof_dec else "no_oof_phase"))
        return "not_in_config_table_v4;" + ",".join(parts)
    g["why_missing"] = g.apply(why, axis=1)
    g["in_phase_training_709"] = g.folds_csv | g.newtrain
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "missing_audit.csv", index=False)
    summ = {"v3_signals": int(v3.DeviceId.nunique()), "missing": int(len(g)),
            "why": g.why_missing.value_counts().to_dict(),
            "why_by_tier": g.groupby([g.tier.fillna("none"), "why_missing"]).size()
            .unstack(fill_value=0).to_dict(),
            "missing_with_base_stg": {k: int(g[g.base_stg][k].sum())
                                      for k in ("ex_det_stg", "ex_cyc_stg", "yrlag_stg",
                                                "pp_v4", "oof_stg", "newtrain", "labels_v4")},
            "in_phase_training_709": int(g.in_phase_training_709.sum())}
    json.dump(summ, open(OUT / "missing_audit.json", "w"), indent=1, default=str)
    log(json.dumps(summ, indent=1, default=str))


def audit() -> pd.DataFrame:
    return pd.read_csv(OUT / "missing_audit.csv")


# ------------------------------------------------------------------ folds
def stage_folds(a) -> None:
    """folds_v3 = folds.csv (375) + folds_v4 extras (the frame's own 29) + the NEWTRAIN map the
    stage-12/13 phase OOF used (so a new signal's function fold == the fold its phase features
    were held out in) + a balanced draw (folds 1-5, greedy by detector count, seed 0) for any
    signal with rows that is in none of them. Existing assignments are never changed."""
    lk = locked()
    f = pd.read_csv(FOLDS).assign(src="folds.csv")
    f4 = pd.read_csv(FOLDS_V4).assign(src="folds_v4")
    nt = pd.read_csv(FOLDS_NT).assign(src="newtrain_folds")
    for d in (f, f4, nt):
        d["DeviceId"] = d.DeviceId.str.lower()
    m = f4.merge(f, on="DeviceId")
    assert (m.fold_x == m.fold_y).all()
    out = pd.concat([f, f4[~f4.DeviceId.isin(f.DeviceId)]], ignore_index=True)
    out = pd.concat([out, nt[~nt.DeviceId.isin(out.DeviceId)]], ignore_index=True)
    au = audit()
    need = au[au.why_missing.str.startswith("not_in_config") & ~au.DeviceId.isin(out.DeviceId)]
    if len(need):
        con = duck()
        con.register("need", need[["DeviceId"]])
        nd = con.execute(f"SELECT lower(DeviceId) DeviceId, count(DISTINCT Detector) n FROM "
                         f"'{P(STG / 'features' / 'pair_features_stg.parquet')}' WHERE "
                         f"lower(DeviceId) IN (SELECT DeviceId FROM need) GROUP BY 1").df()
        rng = np.random.default_rng(0)
        nd = nd.assign(r=rng.random(len(nd))).sort_values(["n", "r"], ascending=[False, True])
        load = {k: 0 for k in range(1, N_FOLDS)}
        rows = []
        for d, n in zip(nd.DeviceId, nd.n):
            k = min(load, key=lambda x: (load[x], x))
            load[k] += n
            rows.append({"DeviceId": d, "fold": k, "src": "new_balanced_seed0"})
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
    assert not out.DeviceId.duplicated().any()
    assert not out.DeviceId.isin(lk).any(), "locked signal in folds_v3"
    out["fold"] = out.fold.astype(int)
    # the old assignments are unchanged
    for d in (f, f4):
        chk = d.merge(out, on="DeviceId")
        assert (chk.fold_x == chk.fold_y).all()
    out[["DeviceId", "fold", "src"]].to_csv(FOLDS_V3, index=False)
    log(f"folds_v3: {len(out)} signals, by source {out.src.value_counts().to_dict()}, "
        f"by fold {out.fold.value_counts().sort_index().to_dict()} -> {FOLDS_V3}")


def folds_v3() -> dict:
    f = pd.read_csv(FOLDS_V3)
    return dict(zip(f.DeviceId.str.lower(), f.fold.astype(int)))


# ------------------------------------------------------------------ targets
def targets() -> dict[str, list[str]]:
    """New signal-periods: missing from the frame, with pair features and an OOF phase."""
    au = audit()
    t = {"stg": sorted(au[au.base_stg & au.oof_stg].DeviceId),
         "dec": sorted(au[au.base_dec & au.oof_dec].DeviceId)}
    return t


def verify_signals() -> dict[str, list[str]]:
    """Five covered signals: 3 Sept-2026 (2 DEV + 1 NEWTRAIN-in-frame) and 2 Dec-2024."""
    fr = pd.read_parquet(FRAME_V4, columns=["DeviceId", "period", "split"])
    fr = fr.drop_duplicates()
    rng = np.random.default_rng(28)
    pick = lambda s: sorted(rng.choice(sorted(s), 1 if len(s) else 0, replace=False).tolist())
    stg_dev = sorted(set(fr[(fr.period == "stg") & (fr.split == "DEV")].DeviceId))
    stg_nt = sorted(set(fr[(fr.period == "stg") & (fr.split == "NEWTRAIN")].DeviceId))
    dec = sorted(set(fr[fr.period == "dec"].DeviceId))
    v = {"stg": sorted(rng.choice(stg_dev, 2, replace=False).tolist() + pick(stg_nt)),
         "dec": sorted(rng.choice(dec, 2, replace=False).tolist())}
    return v


# ------------------------------------------------------------------ phase
def stage_phase(a) -> None:
    """OOF phase probabilities for the new signal-periods, from the stage-12 ranker-bag +
    decoder OOF. Its fold of each signal is asserted equal to folds_v3."""
    con = duck()
    t = targets()
    fv = folds_v3()
    for per, src in (("stg", "STG"), ("dec", "DEC")):
        con.register("keep", pd.DataFrame({"d": t[per]}))
        o = con.execute(f"""SELECT lower(replace(DeviceId,'@stg','')) AS DeviceId, Detector,
                win, cand_phase, prob FROM '{P(PHASE_OOF)}'
                WHERE src='{src}' AND lower(replace(DeviceId,'@stg','')) IN
                (SELECT d FROM keep)""").df()
        o["Detector"] = o.Detector.astype(np.int16)
        o["cand_phase"] = o.cand_phase.astype(np.int16)
        assert not o.DeviceId.isin(locked()).any()
        missing_fold = set(o.DeviceId) - set(fv)
        assert not missing_fold, missing_fold
        o.to_parquet(OUT / f"phase_pred_new_{per}.parquet", index=False)
        log(f"[{per}] OOF phase for {o.DeviceId.nunique()} signals, {len(o):,} pair rows")
    # the stage-12 NEWTRAIN fold map is exactly newtrain_folds.csv (fit_final_v1.build_pool:
    # rng(0) over the sorted NEWTRAIN ids) -- re-derive and assert
    ids = np.array(sorted(con.execute(f"SELECT DISTINCT DeviceId FROM '{P(PHASE_OOF)}' "
                                      f"WHERE src='STG'").df().DeviceId))
    fm = dict(zip([i.replace("@stg", "").lower() for i in ids],
                  np.random.default_rng(0).integers(0, N_FOLDS, len(ids))))
    bad = [d for d in t["stg"] if fm.get(d, -1) != fv[d]]
    assert not bad, f"{len(bad)} signals: phase-OOF fold != folds_v3"
    log("phase-OOF fold == folds_v3 fold for every new Sept-2026 signal")


# ------------------------------------------------------------------ YR + lag
def stage_yrlag(a) -> None:
    """Runs build_features.build(what='yrlag') in a child process with DC_WORK pointed at the
    Sept-2026 work dir (exactly how the stg feature tables were built), on the new + verify
    signals only. Dec-2024 YR/lag already exist for every new Dec-2024 signal."""
    t, v = targets(), verify_signals()
    devs = sorted(set(t["stg"]) | set(v["stg"]))
    au = audit()
    have = set(au[au.yrlag_dec].DeviceId)
    assert set(t["dec"]) <= have, "a new Dec-2024 signal lacks YR/lag"
    lst = OUT / "yrlag_signals_stg.txt"
    lst.write_text("\n".join(devs))
    env = dict(os.environ, DC_WORK=str(STG), DC_WORK_MAIN=str(DCW))
    subprocess.run([sys.executable, __file__, "_yrlag_child", "--threads", str(a.threads)],
                   env=env, check=True)


def stage__yrlag_child(a) -> None:
    sys.path.insert(0, str(HERE.parent))
    import rpath  # noqa: F401
    import build_features as bf
    import windows_stg  # noqa: F401  -- registers 'stgall'
    out = Path(os.environ["DC_WORK_MAIN"]) / "trackA" / "v5"
    devs = (out / "yrlag_signals_stg.txt").read_text().split()
    # the cache's own spelling of the ids
    con = bf.connect(threads=a.threads)
    meta = con.sql(f"SELECT DeviceId FROM read_parquet('{P(bf.CACHE / 'signal_meta.parquet')}')"
                   " ORDER BY DeviceId").df().DeviceId.tolist()
    keep = [d for d in meta if d.lower() in set(devs)]
    con.close()
    assert len(keep) == len(devs), (len(keep), len(devs))
    bf._signals = lambda con, limit: keep            # only which signals, not how
    bf.build("yrlag", str(out / "func_yr_extra_stg_new.parquet"),
             str(out / "det_lag_stg_new.parquet"), "stgall", a.threads, 8, 0)


# ------------------------------------------------------------------ frame rows
def frame_rows(period: str, keep: set, probs: pd.DataFrame, yr_files, lag_files,
               lab: pd.DataFrame, fold_of: dict, split_of: dict) -> pd.DataFrame:
    """function_v4.stage_frame's per-period body, for `keep` signals, with the phase
    probabilities, YR and lag tables passed in."""
    import function_v4 as fv4
    import function_v3 as fv3
    orig = fv4._files
    fv4._files = lambda p, k: (list(yr_files) if k == "yr" else orig(p, k))
    try:
        df, _ = fv4.load_pairs(period, keep, with_yr=True)
    finally:
        fv4._files = orig
    probs = probs.copy()
    probs["Detector"] = probs.Detector.astype(df.Detector.dtype)
    probs["cand_phase"] = probs.cand_phase.astype(df.cand_phase.dtype)
    p = df.merge(probs, on=fv4.PAIR_KEY, how="inner")
    del df
    i = p.groupby(["DeviceId", "Detector", "win"], sort=False)["prob"].idxmax()
    top = p.loc[i].copy()
    del p
    top = top.rename(columns={"cand_phase": "pred_phase", "prob": "top_prob"})
    top = fv3.add_shape_features(top)
    top = fv3.add_sibling_features(top)
    top = top.reset_index(drop=True)
    lg = fv4._read(list(lag_files), keep)
    o3 = fv3._cat
    fv3._cat = lambda files, _lg=lg: _lg
    try:
        top = fv3.add_lag_features(top)
    finally:
        fv3._cat = o3
    top["period"] = period
    top["Detector"] = top.Detector.astype(int)
    top = top.merge(lab[["DeviceId", "Detector", "func5", "Function", "cfg_phase", "fold",
                         "split"]], on=["DeviceId", "Detector"], how="left")
    if fold_of:                                       # new signals: signal-level fold / split
        top["fold"] = top.DeviceId.str.lower().map(fold_of).astype(float)
        top["split"] = top.DeviceId.str.lower().map(split_of)
    if period == "dec":
        from train_lgbm_v2 import load_health
        h = load_health()
        h["DeviceId"] = h.DeviceId.str.lower()
        h["Detector"] = h.Detector.astype(int)
        top = top.merge(h, on=["DeviceId", "Detector"], how="left")
        top["health_flag"] = top.health_flag.fillna("unknown")
    else:
        top["health_flag"] = "unknown"
    top["wgroup"] = fv4.wgroup(top.win)
    for c in top.columns:
        if top[c].dtype == np.float64:
            top[c] = top[c].astype(np.float32)
    fv4.assert_no_holdout(top, f"v5 rows {period}")
    return top


def to_v4_schema(top: pd.DataFrame, cols, dtypes) -> pd.DataFrame:
    top = top[cols].copy()
    for c in cols:
        if top[c].dtype != dtypes[c]:
            top[c] = top[c].astype(dtypes[c])
    return top


def _fr_schema():
    s = pd.read_parquet(FRAME_V4).iloc[:0]
    return list(s.columns), s.dtypes.to_dict()


def stage_rows(a) -> None:
    sys.path.insert(0, str(HERE.parent))
    import rpath  # noqa: F401
    import function_v4 as fv4
    fv4.TEST_CFG = LOCKED[0]          # the ported module's repo-relative path no longer exists
    t0 = time.time()
    cols, dt = _fr_schema()
    t, v = targets(), verify_signals()
    fold_of = folds_v3()
    nt = set(pd.read_csv(FOLDS_NT).DeviceId.str.lower())
    split_of = {d: ("NEWTRAIN" if d in nt else "DEV") for d in fold_of}
    lab4 = pd.read_parquet(fv4.LABELS_V4)
    empty = lab4.iloc[:0]
    parts = []
    for per in ("stg", "dec"):
        if not t[per]:
            continue
        probs = pd.read_parquet(OUT / f"phase_pred_new_{per}.parquet")
        yr = ([OUT / "func_yr_extra_stg_new.parquet"] if per == "stg"
              else fv4._files("dec", "yr"))
        lag = ([OUT / "det_lag_stg_new.parquet"] if per == "stg" else fv4._files("dec", "lag"))
        top = frame_rows(per, set(t[per]), probs, yr, lag, empty, fold_of, split_of)
        log(f"[{per}] new rows {top.shape}, {top.DeviceId.nunique()} signals "
            f"({time.time()-t0:.0f}s)")
        parts.append(to_v4_schema(top, cols, dt))
    new = pd.concat(parts, ignore_index=True)
    new.to_parquet(OUT / "frame_new.parquet", index=False)
    # verification rows: covered signals, stored phase_pred + stored YR/lag + labels_v4
    vparts = []
    for per in ("stg", "dec"):
        probs = pd.read_parquet(PP_V4[per], filters=[("DeviceId", "in", v[per])])
        top = frame_rows(per, set(v[per]), probs, fv4._files(per, "yr"),
                         fv4._files(per, "lag"), lab4, {}, {})
        vparts.append(to_v4_schema(top, cols, dt))
    pd.concat(vparts, ignore_index=True).to_parquet(OUT / "frame_verify.parquet", index=False)
    log(f"rows done in {time.time()-t0:.0f}s: new {new.shape}")


# ------------------------------------------------------------------ expert features
def stage_expert(a) -> None:
    """a2_features.build per period on a signal-filtered copy of the cache (new + verify
    signals; every expert feature is per signal), targets = the rows' pred_phase."""
    import duckdb  # noqa: F401
    sys.path.insert(0, str(HERE))
    import a2_features as A2F
    t, v = targets(), verify_signals()
    new = pd.read_parquet(OUT / "frame_new.parquet", columns=["DeviceId", "Detector", "win",
                                                              "period", "pred_phase"])
    old = pd.read_parquet(FRAME_V4, columns=["DeviceId", "Detector", "win", "period",
                                             "pred_phase"])
    con = duck()
    for per in ("stg", "dec"):
        vs = set(v[per])
        tg = pd.concat([new[new.period == per],
                        old[(old.period == per) & old.DeviceId.isin(vs)]], ignore_index=True)
        devs = sorted(set(tg.DeviceId))
        if not devs:
            continue
        cdir = OUT / f"cache_{per}"
        cdir.mkdir(parents=True, exist_ok=True)
        con.register("keep", pd.DataFrame({"d": devs}))
        for f in ("det_intervals", "phase_cycles", "coord_state"):
            con.execute(f"COPY (SELECT * FROM '{P(CACHE[per] / (f + '.parquet'))}' WHERE "
                        f"lower(DeviceId) IN (SELECT d FROM keep)) TO "
                        f"'{P(cdir / (f + '.parquet'))}' (FORMAT parquet)")
        tfile = OUT / f"expert_targets_{per}.parquet"
        tg.to_parquet(tfile, index=False)
        A2F.CACHE = {**A2F.CACHE, per: cdir}
        A2F.FRAME = tfile
        A2F.WORK = OUT / f"expert_{per}"
        A2F.WORK.mkdir(parents=True, exist_ok=True)
        A2F.connect = _a2_connect                   # 10 GB cap (AGENTS.md), same settings else
        log(f"[{per}] expert features for {len(devs)} signals ({len(vs)} verify)")
        A2F.build(per)


def _a2_connect():
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=12")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


# ------------------------------------------------------------------ verification
def _same(a: pd.DataFrame, b: pd.DataFrame, key, tol=0.0) -> dict:
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    r = {"rows_new": len(a), "rows_stored": len(b)}
    if len(a) != len(b) or not (a[key].astype(str).values == b[key].astype(str).values).all():
        m = a[key].astype(str).merge(b[key].astype(str), how="outer", indicator=True)
        r["key_mismatch"] = m._merge.value_counts().to_dict()
        return r
    bad = {}
    for c in a.columns:
        if c in key:
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            eq = (np.isnan(xv) & np.isnan(yv)) | (np.abs(xv - yv) <= tol * (1 + np.abs(yv)))
        else:
            eq = (x.isna() & y.isna()) | (x.astype(str) == y.astype(str))
        if not np.all(eq):
            bad[c] = int((~np.asarray(eq)).sum())
    r["columns_differing"] = bad
    r["identical"] = not bad
    return r


def stage_verify(a) -> None:
    con = duck()
    v = verify_signals()
    res = {"signals": v}
    fr = pd.read_parquet(FRAME_V4)
    vf = pd.read_parquet(OUT / "frame_verify.parquet")
    st = fr[fr.set_index(["DeviceId", "period"]).index.isin(
        [(d, p) for p in v for d in v[p]])]
    res["frame_rows"] = _same(vf, st, KEY)
    # YR / lag, recomputed for the 3 Sept-2026 verify signals
    vs = "(" + ",".join(f"'{d}'" for d in v["stg"]) + ")"
    for kind, new, old, key in (
            ("yr", "func_yr_extra_stg_new", "func_yr_extra_stg",
             ["DeviceId", "Detector", "cand_phase", "win"]),
            ("lag", "det_lag_stg_new", "det_lag_stg", ["DeviceId", "Detector", "other", "win"])):
        n = con.execute(f"SELECT * FROM '{P(OUT / (new + '.parquet'))}' WHERE lower(DeviceId) "
                        f"IN {vs}").df()
        o = con.execute(f"SELECT * FROM '{P(DCW / 'function_v4' / 'feat_stg' / (old + '.parquet'))}'"
                        f" WHERE lower(DeviceId) IN {vs}").df()
        res[kind] = _same(n, o, key)
    # expert tables
    for per in ("stg", "dec"):
        vv = "(" + ",".join(f"'{d}'" for d in v[per]) + ")"
        for tag in ("det", "cyc", "pair"):
            n = con.execute(f"SELECT * FROM '{P(OUT / f'expert_{per}' / f'feat_expert_{tag}_{per}.parquet')}'"
                            f" WHERE DeviceId IN {vv}").df()
            o = con.execute(f"SELECT * FROM '{P(EXPERT_OLD / f'feat_expert_{tag}_{per}.parquet')}'"
                            f" WHERE DeviceId IN {vv}").df()
            res[f"expert_{tag}_{per}"] = _same(n, o, KEY)
    # fresh det features of the NEW signals vs the stale stored rows (det has no phase input)
    t = targets()
    ns = "(" + ",".join(f"'{d}'" for d in t["stg"]) + ")"
    n = con.execute(f"SELECT * FROM '{P(OUT / 'expert_stg' / 'feat_expert_det_stg.parquet')}' "
                    f"WHERE DeviceId IN {ns}").df()
    o = con.execute(f"SELECT * FROM '{P(EXPERT_OLD / 'feat_expert_det_stg.parquet')}' "
                    f"WHERE DeviceId IN {ns}").df()
    res["expert_det_stg_new_signals_vs_stale_stored"] = _same(n, o, KEY)
    json.dump(res, open(OUT / "verify.json", "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))


# ------------------------------------------------------------------ merge
def stage_merge(a) -> None:
    t0 = time.time()
    lk = locked()
    v4 = pd.read_parquet(FRAME_V4)
    new = pd.read_parquet(OUT / "frame_new.parquet")
    assert list(new.columns) == list(v4.columns)
    assert not (set(new.DeviceId) & set(v4.DeviceId))
    v5 = pd.concat([v4, new], ignore_index=True)
    for c in v4.columns:
        assert v5[c].dtype == v4[c].dtype, c
    assert not v5.DeviceId.str.lower().isin(lk).any()
    assert not v5.duplicated(KEY).any()
    v5.to_parquet(FRAME_V5, index=False)
    back = pd.read_parquet(FRAME_V5)
    assert back.iloc[:len(v4)].reset_index(drop=True).equals(v4), "old rows changed"
    newsig = set(new.DeviceId)
    ex = {}
    for per in ("dec", "stg"):
        for tag in ("det", "cyc", "pair"):
            old = pd.read_parquet(EXPERT_OLD / f"feat_expert_{tag}_{per}.parquet")
            f = OUT / f"expert_{per}" / f"feat_expert_{tag}_{per}.parquet"
            fresh = pd.read_parquet(f) if f.exists() else old.iloc[:0]
            fresh = fresh[fresh.DeviceId.isin(newsig)]
            keep = old[~old.DeviceId.isin(newsig)]        # stale rows of new signals replaced
            fresh = fresh[list(old.columns)].astype(old.dtypes.to_dict())
            e = pd.concat([keep, fresh], ignore_index=True)
            assert not e.DeviceId.str.lower().isin(lk).any()
            assert not e.duplicated(KEY).any()
            e.to_parquet(EXPERT_OLD / f"feat_expert_{tag}_{per}_v5.parquet", index=False)
            ex[f"{tag}_{per}"] = {"stored_kept": len(keep), "stale_replaced":
                                  int(len(old) - len(keep)), "fresh": len(fresh)}
    fv = pd.read_csv(FOLDS_V3)
    per_fold = v5.assign(f=v5.DeviceId.str.lower().map(dict(zip(fv.DeviceId, fv.fold))))
    pf = per_fold[per_fold.wgroup == "full"].groupby("f").agg(
        signals=("DeviceId", "nunique"), detectors=("Detector", "size"))
    summ = {"v4_rows": len(v4), "new_rows": len(new), "v5_rows": len(v5),
            "v5_signals": int(v5.DeviceId.nunique()),
            "new_signals_by_period": new.groupby("period").DeviceId.nunique().to_dict(),
            "expert_v5": ex, "full_window_rows_by_fold": pf.to_dict(),
            "old_rows_identical": True, "built": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(summ, open(OUT / "merge.json", "w"), indent=1, default=str)
    log(json.dumps(summ, indent=1, default=str))
    log(f"merge done in {time.time()-t0:.0f}s -> {FRAME_V5}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "folds", "phase", "yrlag", "_yrlag_child",
                                      "rows", "expert", "verify", "merge"])
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
