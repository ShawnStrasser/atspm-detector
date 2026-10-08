"""Note 66: does MORE DATA per detector help function?  New WEEKDAY windows (Mon 28 - Wed 30 Sept 2026, daily pull
`data/staging_2026_w40`) of the same training signals are added to TRAINING ONLY; scoring stays on exactly the existing
OOF evaluation rows (note-64 integrated scorer, trees arm, no fj).

New windows (period "wkd") mirror the Sept-18 "stg" mix (22 windows, same lengths; weekday anchors, peak / midday /
night spread like the original): see WKD below.  Features = the final_v3 candidate package's own code path
(`final_v3_candidate_v2`: load_events -> GRU pieces -> build_features -> score (phase blend + decoder) ->
_function_frame -> features_expert), i.e. the production feature definitions (notes 32 / 49 checked their parity with
frame v6e; `parity` re-checks it here with the frame's own phase input).  APPROXIMATION: the phase input of the new
rows is the package's FULL-FIT phase blend, not six-fold OOF (training rows only; the evaluation rows keep v6e).
Labels = v3s per detector (same tables / masks as the note-57 arm); health: stg health3 + w40 health `bad` out.
Lanes / pick inputs are evaluation-side only (unchanged rows) -> unchanged.

    python s66_moredata.py cache                 # -> %DC_WORK%/s66/events/DeviceId=*/  (Sept 28 00:00 - Oct 1 00:00)
    python s66_moredata.py parity                # package vs frame v6e on a few stg signal-windows (229 columns)
    python s66_moredata.py feat --procs 4        # -> %DC_WORK%/s66/feat/<dev>.parquet
    python s66_moredata.py frame                 # -> %DC_WORK%/s66/wkd_frame.parquet (+ ok / y)
    python s66_moredata.py fit --cfg plus --seeds 0      (cfg base_re = base refit, same code, no new rows)
    python s66_moredata.py score --cfgs plus@s0 --base base@s0 --tag screen
    python s66_moredata.py wkd                   # secondary: base model on weekday windows vs Sept windows
locked_v2 asserted absent everywhere. CPU, <= 8 threads, DuckDB 10 GB total.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("DC_GRU_THREADS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import v3_retrain as V  # noqa: E402

DCW = V.DCW
OUT = DCW / "s66"
SRC = DCW / "data" / "staging_2026_w40"
EVC = OUT / "events"
FEAT = OUT / "feat"
PKG = DCW / "final_v3_candidate_v2"
SPAN = ("2026-09-28 00:00:00", "2026-10-01 00:00:00")
FRAME_V6E = DCW / "function_v4" / "funcframe_v6e.parquet"
KEY = ["DeviceId", "Detector", "period", "win"]
PER = "wkd"
VAR = "first.all.wi"
NTHREAD = 8


def _w(n, t, s):
    return (n, pd.Timestamp(t), int(s))


# weekday mirror of windows_stg (same names / lengths); Mon 28, Tue 29, Wed 30 Sept 2026
WKD = [
    _w("m5_a", "2026-09-29 07:45", 300), _w("m5_b", "2026-09-30 12:20", 300),
    _w("m5_c", "2026-09-28 22:10", 300), _w("m5_d", "2026-09-30 17:05", 300),
    _w("m10_a", "2026-09-28 08:05", 600), _w("m10_b", "2026-09-29 13:00", 600),
    _w("m10_c", "2026-09-30 02:30", 600), _w("m10_d", "2026-09-29 17:20", 600),
    _w("m30_a", "2026-09-30 07:30", 1800), _w("m30_b", "2026-09-28 12:00", 1800),
    _w("m30_c", "2026-09-29 21:30", 1800), _w("m30_d", "2026-09-28 17:00", 1800),
    _w("h1_a", "2026-09-29 17:00", 3600), _w("h1_b", "2026-09-28 02:00", 3600), _w("h1_c", "2026-09-30 09:00", 3600),
    _w("h3_a", "2026-09-28 06:00", 3 * 3600), _w("h3_b", "2026-09-29 14:00", 3 * 3600),
    _w("h6_a", "2026-09-30 06:00", 6 * 3600), _w("h6_b", "2026-09-29 12:00", 6 * 3600),
    _w("h24_a", "2026-09-28 00:00", 24 * 3600), _w("h24_b", "2026-09-29 00:00", 24 * 3600),
    _w("full66", "2026-09-28 06:00", 66 * 3600),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())


def frame_stg_devs() -> list:
    f = pd.read_parquet(FRAME_V6E, columns=["DeviceId", "period"])
    d = sorted(set(f.loc[f.period == "stg", "DeviceId"].str.lower()))
    assert not set(d) & locked()
    return d


def fam_of(w: str) -> str:
    return "full" if w.startswith("full") else w.split("_")[0]


# ------------------------------------------------------------------------------------------------ cache
def stage_cache(a):
    import duckdb
    EVC.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='10GB'; SET threads={NTHREAD}; SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    keep = frame_stg_devs()
    con.register("keep", pd.DataFrame({"d": keep}))
    codes = "1,7,8,9,10,11,43,44,81,82,131,150,173"
    t0 = time.time()
    con.execute(f"""COPY (SELECT DISTINCT lower(DeviceId) AS DeviceId, Timestamp, EventId, Parameter
        FROM read_parquet('{SRC.as_posix()}/date=*/*.parquet', hive_partitioning=false)
        WHERE lower(DeviceId) IN (SELECT d FROM keep) AND EventId IN ({codes})
          AND NOT (EventId IN (81, 82) AND Parameter > 64)
          AND Timestamp >= TIMESTAMP '{SPAN[0]}' AND Timestamp < TIMESTAMP '{SPAN[1]}')
        TO '{EVC.as_posix()}' (FORMAT parquet, PARTITION_BY (DeviceId), OVERWRITE_OR_IGNORE true)""")
    n = len(list(EVC.glob("DeviceId=*")))
    log(f"{n} of {len(keep)} frame stg signals cached ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------------------------ package features
_P = {}


def _pkg():
    if not _P:
        sys.path.insert(0, str(PKG))
        import predict as P
        import gru_blend as gb
        import features_expert as fx
        assert Path(P.__file__).resolve().parent == PKG.resolve(), P.__file__
        _P.update(P=P, gb=gb, fx=fx)
    return _P["P"], _P["gb"], _P["fx"]


def _load(con, evdir: Path, dev: str, t0, t1) -> pd.DataFrame:
    d = [p for p in evdir.iterdir() if p.name.lower() == f"deviceid={dev}"]
    if len(d) != 1:
        return pd.DataFrame()
    return con.sql(f"SELECT '{dev}' AS DeviceId, Timestamp, EventId, Parameter FROM read_parquet("
                   f"'{(d[0] / '*.parquet').as_posix()}', hive_partitioning=false) WHERE Timestamp >= "
                   f"TIMESTAMP '{t0}' AND Timestamp < TIMESTAMP '{t1}'").df()


def package_signal(evdir: Path, dev: str, span, windows, threads=2, mem="2500MB", override=None) -> dict:
    """The research-frame definition through the package code: tables (intervals, cycles, coordination, calls) are
    built ONCE from the signal's whole log over `span` (so ONs / cycles crossing a window edge are known, as in the
    research builder), then every window is cut with build_features(con, t0, t1) (apply_window) and the expert
    features are binned on the requested window.  -> {window name: function design frame}."""
    P, gb, fx = _pkg()
    con = P._connect(threads, mem)
    out = {}
    try:
        ev = _load(con, evdir, dev, span[0], span[1])
        if not len(ev):
            return out
        _, _, info = P.load_events(con, ev)
        if not info.get("n_events"):
            return out
        P.build_chunk_tables(con)
        cfg = gb.config(P.DEFAULT_MODEL_DIR)
        ep = pd.Timestamp("1970-01-01")
        for name, t0, secs in windows:
            try:
                w0 = (pd.Timestamp(t0) - ep).total_seconds()
                w1 = w0 + float(secs)
                gru = None
                minutes = round(secs / 60.0)
                if override is None and gb.runs_on(cfg, minutes):
                    gru = gb.phase_probs(con, cfg["weights_path"], int(round(w0 * 1000)), int(round(w1 * 1000)),
                                         **gb.piece_plan(cfg, minutes))
                df, sim = P.build_features(con, w0, w1)
                if df is None or not len(df):
                    continue
                if override is None:
                    df = P.score(df, sim, P.DEFAULT_MODEL_DIR, gru, cfg)
                else:
                    o = override(name)
                    o["Detector"] = o.Detector.astype(df.Detector.dtype)
                    o["cand_phase"] = o.cand_phase.astype(df.cand_phase.dtype)
                    df = df.merge(o, on=["DeviceId", "Detector", "cand_phase"], how="inner")
                top = P._function_frame(df)
                if not len(top):
                    continue
                ex = fx.build(con, w0, w1, top)
                ex["Detector"] = ex.Detector.astype(top.Detector.dtype)
                out[name] = top.merge(ex, on=["DeviceId", "Detector"], how="left")
            except Exception as e:  # noqa: BLE001
                print(f"{dev} {name}: {type(e).__name__} {e}", flush=True)
        return out
    finally:
        con.close()


def _feat_one(dev):
    f = FEAT / f"{dev}.parquet"
    if f.exists():
        return dev, -1, 0.0
    feats = json.load(open(PKG / "weights" / "function_lgbm_v5.json"))["features"]
    t = time.time()
    parts = []
    for name, top in package_signal(EVC, dev, SPAN, WKD, int(os.environ.get("S66_DT", "2")),
                                    os.environ.get("S66_MEM", "2500MB")).items():
        x = top[list(dict.fromkeys(["DeviceId", "Detector", "pred_phase", "top_prob"]
                                   + [c for c in feats if c in top.columns]))].copy()
        for c in feats:
            if c not in x.columns:
                x[c] = np.nan
        x["win"] = name
        parts.append(x)
    if parts:
        d = pd.concat(parts, ignore_index=True)
        d["DeviceId"] = d.DeviceId.str.lower()
        d["period"] = PER
        d.to_parquet(f, index=False)
        n = len(d)
    else:
        pd.DataFrame({"DeviceId": []}).to_parquet(f, index=False)
        n = 0
    return dev, n, time.time() - t


def stage_feat(a):
    FEAT.mkdir(parents=True, exist_ok=True)
    os.environ["S66_DT"], os.environ["S66_MEM"] = str(a.dthreads), a.dmem
    devs = [d for d in frame_stg_devs() if (EVC / f"DeviceId={d}").is_dir()]
    if a.limit:
        devs = devs[:a.limit]
    t0 = time.time()
    done = 0
    with Pool(a.procs) as p:
        for dev, n, s in p.imap_unordered(_feat_one, devs, chunksize=1):
            done += 1
            if n >= 0 and (done % 10 == 0 or done <= 4):
                log(f"{done}/{len(devs)} {dev[:8]}: {n} rows, {s:.0f}s (elapsed {time.time()-t0:.0f}s)")
    log(f"feat done: {done} signals ({time.time()-t0:.0f}s)")


def stage_parity(a):
    """package features (with frame v6e's own OOF phase input) vs frame v6e, 229 arm columns, stg windows."""
    import a2_features as A2F
    cols = json.load(open(DCW / "trees57" / "function" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx" /
                          "cols.json"))["cols"]
    import t57_function as T57
    T57.setup("v6e")
    fe = pd.read_parquet(V.FEATS, columns=KEY + ["pred_phase"] + cols)
    fe = fe[fe.period == "stg"]
    rng = np.random.default_rng(66)
    s = fe.groupby("DeviceId").Detector.nunique()
    devs = list(rng.choice(sorted(s[(s >= 8) & (s <= 24)].index), a.n, replace=False))
    oof = pd.read_parquet(DCW / "final_v3_work" / "function_v3e" / "phase" / "phase_pred_v3blend_stg.parquet")
    evdir = DCW / "official" / "stg" / "cache" / "events"
    res, diffs = {}, {}
    nrow = 0
    span = ("2026-09-18 00:00:00", "2026-09-22 00:00:00")
    wins = [w for w in A2F.WINDOWS["stg"] if w[0] in a.wins.split(",")]
    rel = {}
    for dev in devs:
        oo = oof[oof.DeviceId == dev].assign(DeviceId=dev.lower())
        t = time.time()
        tops = package_signal(evdir, dev.lower(), span, wins, 4, "4GB",
                              override=lambda n: oo[oo.win == n].drop(columns="win").copy())
        for name, top in tops.items():
            ref = fe[(fe.DeviceId == dev) & (fe.win == name)].assign(DeviceId=dev.lower())
            top["Detector"] = top.Detector.astype(ref.Detector.dtype)
            top = top.reindex(columns=list(dict.fromkeys(list(top.columns) + cols)))
            m = ref.merge(top, on=["DeviceId", "Detector"], suffixes=("_ref", ""))
            nrow += len(m)
            bad = {}
            for c in cols:
                x, y = m[c + "_ref"].to_numpy(float), m[c].to_numpy(float)
                eq = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-4 * (1 + np.abs(x)))
                if not eq.all():
                    bad[c] = int((~eq).sum())
                    diffs[c] = diffs.get(c, 0) + int((~eq).sum())
                    r = np.abs(x - y) / (1e-9 + np.abs(x))
                    rel.setdefault(c, []).extend([float(v) for v in r[~eq] if np.isfinite(v)])
            res[f"{dev[:8]}|{name}"] = {"ref": len(ref), "matched": len(m),
                                        "pred_phase_diff": int((m.pred_phase != m.pred_phase_ref).sum()),
                                        "n_cols_differing": len(bad), "cols_differing": bad}
            log(f"{dev[:8]} {name}: matched {len(m)}/{len(ref)}, pred diff "
                f"{res[f'{dev[:8]}|{name}']['pred_phase_diff']}, cols differing {len(bad)} {list(bad)[:8]}")
        log(f"{dev[:8]}: {time.time()-t:.0f}s")
    diffs = {c: {"rows": n, "median_rel": round(float(np.median(rel.get(c, [0]))), 5)} for c, n in diffs.items()}
    out = {"rows": nrow, "columns_differing_total": diffs, "windows": res}
    json.dump(out, open(OUT / "parity.json", "w"), indent=1)
    log(f"parity rows {nrow}; differing columns {diffs}")


# ------------------------------------------------------------------------------------------------ new-row frame
def stage_frame(a):
    import t57_function as T57
    T57.setup("v6e")
    cols = json.load(open(DCW / "trees57" / "function" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx" /
                          "cols.json"))["cols"]
    fs = sorted(FEAT.glob("*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs if pd.read_parquet(f).shape[0]], ignore_index=True)
    lk = locked()
    assert not d.DeviceId.isin(lk).any()
    fr = pd.read_parquet(FRAME_V6E, columns=["DeviceId", "Detector", "fold", "period"])
    fold = fr.assign(d=fr.DeviceId.str.lower()).groupby("d").fold.first()
    d["fold"] = d.DeviceId.map(fold)
    assert d.fold.notna().all()
    d["fold"] = d.fold.astype(int)
    d["wgroup"] = d.win.map(fam_of)
    d["health_flag"] = "unknown"
    d["Detector"] = d.Detector.astype(fr.Detector.dtype)
    lab = V.load_labels(d, "exclude")
    y, ok = V.variant_target(lab, d, VAR)
    # health: detectors `bad` in the stg health3 table or in the late-Sept (w40) health run stay out of training
    h3 = pd.read_parquet(V.HEALTH3_FILE, columns=["period", "DeviceId", "detector", "status"])
    h3 = h3[h3.period == "stg"]
    hw = pd.read_parquet(DCW / "health4" / "health_w40.parquet", columns=["DeviceId", "detector", "status"])
    bad = set(zip(h3.loc[h3.status == "bad", "DeviceId"].str.lower(), h3.loc[h3.status == "bad", "detector"].astype(int)))
    bad |= set(zip(hw.loc[hw.status == "bad", "DeviceId"].str.lower(), hw.loc[hw.status == "bad", "detector"].astype(int)))
    hb = np.fromiter(((x, int(z)) in bad for x, z in zip(d.DeviceId, d.Detector)), bool, len(d))
    ok &= ~hb
    d["y"] = y
    d["ok"] = ok
    d = d[["DeviceId", "Detector", "period", "win", "wgroup", "fold", "pred_phase", "top_prob", "health_flag", "y", "ok"]
          + cols]
    d.to_parquet(OUT / "wkd_frame.parquet", index=False)
    info = {"rows": len(d), "signals": int(d.DeviceId.nunique()), "ok_rows": int(ok.sum()),
            "ok_signals": int(d.loc[ok, "DeviceId"].nunique()), "health_bad_rows_out": int((hb & pd.notna(y)).sum()),
            "ok_by_class": pd.Series(y[ok]).value_counts().to_dict(),
            "ok_by_wgroup": d.loc[ok].wgroup.value_counts().to_dict()}
    json.dump(info, open(OUT / "wkd_frame.json", "w"), indent=1, default=str)
    log(json.dumps(info, default=str))


# ------------------------------------------------------------------------------------------------ fit
def stage_fit(a):
    import s59_step6 as S59
    fr, cols, lab, y, ok = S59.load_train()
    fo = fr.fold.to_numpy()
    yi_map = {c: i for i, c in enumerate(S59.C7)}
    yi = pd.Series(y).map(yi_map).fillna(-1).astype(int).to_numpy()
    X0 = fr[cols].to_numpy(np.float32)
    nw = pd.read_parquet(OUT / "wkd_frame.parquet")
    assert not nw.DeviceId.isin(locked()).any()
    Xn = nw[cols].to_numpy(np.float32)
    yn = nw.y.map(yi_map).fillna(-1).astype(int).to_numpy()
    okn = nw.ok.to_numpy(bool)
    fn_ = nw.fold.to_numpy()
    cfg = a.cfg
    assert cfg in ("plus", "base_re", "plus_long", "plus_short", "plus_op", "plus_op_shuf"), cfg
    if cfg.startswith("plus_op"):
        Ef = S59.offpeak_feats(fr).to_numpy(np.float32)
        F = pd.read_parquet(OUT / "offpeak_wkd.parquet").astype({"Detector": nw.Detector.dtype})
        En = nw[KEY].merge(F[KEY + S59.OP_COLS], on=KEY, how="left")[S59.OP_COLS].to_numpy(np.float32)
        if cfg.endswith("_shuf"):
            E = np.vstack([Ef, En])[np.random.default_rng(66).permutation(len(Ef) + len(En))]
            Ef, En = E[:len(Ef)], E[len(Ef):]
        X0 = np.hstack([X0, Ef])
        Xn = np.hstack([Xn, En])
        log(f"off-peak columns: frame {np.isfinite(Ef[:, 0]).mean():.2f} / new {np.isfinite(En[:, 0]).mean():.2f} non-NaN")
    if cfg == "plus_long":
        okn = okn & ~nw.wgroup.isin(["m5", "m10"]).to_numpy()
    if cfg == "plus_short":
        okn = okn & nw.wgroup.isin(["m5", "m10"]).to_numpy()
    d = OUT / "fit" / cfg
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"cfg": cfg, "n_cols": len(cols), "new_ok_rows": int(okn.sum()) if cfg != "base_re" else 0},
              open(d / "cols.json", "w"), indent=1)
    for s in [int(x) for x in a.seeds.split(",")]:
        for k in range(6):
            f = d / f"P_{VAR}_s{s}_f{k}.npy"
            if f.exists():
                continue
            t0 = time.time()
            inner = (k + 1) % 6
            trm = ok & (fo != k) & (fo != inner)
            X, Y, W = X0, yi, np.ones(len(fr))
            tem = fo == k
            if cfg != "base_re":
                addm = okn & (fn_ != k) & (fn_ != inner)
                X = np.vstack([X0, Xn[addm]])
                Y = np.concatenate([yi, yn[addm]])
                W = np.ones(len(X))
                trm = np.concatenate([trm, np.ones(int(addm.sum()), bool)])
                vam = np.concatenate([ok & (fo == inner), np.zeros(int(addm.sum()), bool)])
                tem = np.concatenate([fo == k, np.zeros(int(addm.sum()), bool)])
            else:
                vam = ok & (fo == inner)
                addm = np.zeros(len(nw), bool)
            P, nt, m = lgb_fit(X, Y, W, trm, vam, tem, s, a.threads)
            np.save(f, P)
            # the same model on the NEW weekday rows of fold k (secondary analysis)
            np.save(d / f"Pwkd_s{s}_f{k}.npy", m.predict_proba(Xn[fn_ == k]).astype(np.float32))
            log(f"{cfg} s{s} f{k}: {nt} trees, train {int(trm.sum()):,} (+{int(addm.sum()):,} new), "
                f"{time.time()-t0:.0f}s")


def lgb_fit(X, yi, w, trm, vam, tem, seed, threads):
    import lightgbm as lgb
    import a2_model as A2
    prm = dict(A2.FUNC_PARAMS, n_jobs=threads, num_class=7, seed=seed, bagging_seed=seed + 1,
               feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    n = prm.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n, **prm)
    m.fit(X[trm], yi[trm], sample_weight=w[trm], eval_set=[(X[vam], yi[vam])], eval_sample_weight=[w[vam]],
          eval_metric="multi_logloss", callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    return m.predict_proba(X[tem]).astype(np.float32), int(m.best_iteration_ or 0), m


# ------------------------------------------------------------------------------------------------ score (cand64 path)
_S = {}


def _scorer():
    if _S:
        return _S["x"]
    import s59_step6 as S59
    import s62_short as S62
    import cand64 as C
    fr, rows = S59.scoring_frame(S59.PICK_TAG)
    fr = fr.copy()
    assert not fr.DeviceId.isin(locked()).any()
    pairs = pd.read_parquet(S62.PAIRS_F)
    short = fr.wgroup.isin(S62.SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy()
    tw = S62.twin_tokens(fr, pairs, C.TWIN_THR, 1.0, 0)
    _S["x"] = (fr, rows, short, tw)
    return _S["x"]


def probs(spec: str, fr: pd.DataFrame):
    import s59_step6 as S59
    parts = spec.split("@")
    cfg = parts[0]
    seeds = [int(o[1:]) for o in parts[1:] if o.startswith("s") and o[1:].isdigit()]
    if cfg == "base":
        return S59.spec_probs(spec, fr)
    d = OUT / "fit" / cfg
    if not seeds:
        seeds = sorted({int(p.name.split("_s")[1].split("_")[0]) for p in d.glob(f"P_{VAR}_s*_f5.npy")})
    k = S59.frame_keys()
    P = S59.std_probs(k, seeds, d)
    k["_i"] = np.arange(len(k))
    k["Detector"] = k.Detector.astype(fr.Detector.dtype)
    idx = fr[S59.KEY].merge(k[S59.KEY + ["_i"]], on=S59.KEY, how="left")._i.to_numpy()
    return P[idx.astype(int)], seeds


def run_spec(spec: str):
    """cand64.stage_function's trees arm: D-lane stack-pick decode + cy twin decode (thr .4) on 5 / 10 min."""
    import s59_step6 as S59
    import s62_short as S62
    import atspm_score as S
    fr, rows, short, tw = _scorer()
    P, seeds = probs(spec, fr)
    for i, c in enumerate(S59.C7):
        fr[f"P_{c}"] = P[:, i]
    pr = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
    pr = S62.twin_decode(P, pr, tw, short, ("Count", "Yellow_Red"), "cy")
    out = {}
    for s, r in rows.items():
        d = S.credit(fr, pr, "truth_v3s", r, True)
        out[s] = pd.DataFrame({"ok": d.ok_a.to_numpy(float), "wgroup": fr.wgroup.to_numpy()[r],
                               "sig": fr.DeviceId.to_numpy()[r], "fold": fr.fold.to_numpy()[r]})
    return out, seeds


def stage_score(a):
    import cand64 as C
    base, bs = run_spec(a.base)
    res = {"base": a.base, "base_seeds": bs, "results": {}}
    pools = {**C.POOLS, "all": C.FAMS}
    for sname in ("everything", "realistic"):
        b = base[sname]
        res.setdefault("base_acc", {})[sname] = {p: C.acc_ci(b.ok.to_numpy()[b.wgroup.isin(f)],
                                                             b.sig.to_numpy()[b.wgroup.isin(f)]) for p, f in pools.items()}
    log(f"base {a.base}: " + json.dumps(res["base_acc"]))
    for spec in [s for s in a.cfgs.split(",") if s]:
        D, seeds = run_spec(spec)
        r = {"seeds": seeds}
        for sname in ("everything", "realistic"):
            b, x = base[sname], D[sname]
            assert (b.sig.to_numpy() == x.sig.to_numpy()).all()
            r[sname] = {}
            for p, f in {**pools, **{g: [g] for g in C.FAMS}}.items():
                m = b.wgroup.isin(f).to_numpy()
                r[sname][p] = {"acc": C.acc_ci(x.ok.to_numpy()[m], x.sig.to_numpy()[m]),
                               "delta_pt": C.delta_ci(b.ok.to_numpy()[m], x.ok.to_numpy()[m], b.sig.to_numpy()[m])}
            r[sname]["by_fold_ge30_delta_pt"] = {
                int(k): round(100 * float(x.ok[(x.fold == k) & x.wgroup.isin(C.GE30)].mean()
                                          - b.ok[(b.fold == k) & b.wgroup.isin(C.GE30)].mean()), 3) for k in range(6)}
        res["results"][spec] = r
        for sname in ("everything", "realistic"):
            log(f"{spec} {sname[:4]}: " + " | ".join(f"{p} {r[sname][p]['acc'][0]} d {r[sname][p]['delta_pt']}"
                                                    for p in ("ge30", "m5", "m10", "all")))
    json.dump(res, open(OUT / f"score_{a.tag}.json", "w"), indent=1, default=str)
    log(f"-> {OUT / f'score_{a.tag}.json'}")


# ------------------------------------------------------------------------------------------------ secondary
def atspm_ok(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    ats = {"Advance", "Presence", "Count", "Yellow_Red"}
    ya = np.isin(y, list(ats))
    pa = np.isin(p, list(ats))
    return np.where(ya, p == y, ~pa)


def stage_wkd(a):
    """base (no new rows) fold models: argmax ATSPM accuracy (no decode) on the weekday windows vs the same detectors'
    Sept windows (truth = v3s training target; trained-on mask = scored mask; >= 5 actuations), paired by detector-window
    family."""
    import s59_step6 as S59
    import cand64 as C
    cfg = a.cfg
    d = OUT / "fit" / cfg
    nw = pd.read_parquet(OUT / "wkd_frame.parquet", columns=["DeviceId", "Detector", "win", "wgroup", "fold", "y", "ok",
                                                             "det_n_on", "pred_phase"])
    seeds = [int(x) for x in a.seeds.split(",")]
    Pn = np.zeros((len(nw), 7), np.float32)
    fo = nw.fold.to_numpy()
    for k in range(6):
        Pn[fo == k] = np.mean([np.load(d / f"Pwkd_s{s}_f{k}.npy") for s in seeds], 0)
    nw["pred"] = np.array(S59.C7, object)[Pn.argmax(1)]
    fr, cols, lab, y, ok = S59.load_train()
    k = S59.frame_keys()
    Pf = S59.std_probs(k, seeds, d)
    st = pd.DataFrame({"DeviceId": fr.DeviceId.str.lower(), "Detector": fr.Detector, "period": fr.period, "win": fr.win,
                       "wgroup": fr.wgroup, "y": y, "ok": ok, "pred": np.array(S59.C7, object)[Pf.argmax(1)]})
    st = st[(st.period == "stg") & st.ok]
    nw = nw[nw.ok]
    nw["Detector"] = nw.Detector.astype(st.Detector.dtype)
    # same detectors in both
    both = set(zip(st.DeviceId, st.Detector)) & set(zip(nw.DeviceId, nw.Detector))
    st = st[[x in both for x in zip(st.DeviceId, st.Detector)]]
    nw = nw[[x in both for x in zip(nw.DeviceId, nw.Detector)]]
    st["okA"] = atspm_ok(st.y.to_numpy(object), st.pred.to_numpy(object)).astype(float)
    nw["okA"] = atspm_ok(nw.y.to_numpy(object), nw.pred.to_numpy(object)).astype(float)
    res = {"cfg": cfg, "seeds": seeds, "detectors": len(both)}
    for p, f in {**C.POOLS, "all": C.FAMS, **{g: [g] for g in C.FAMS}}.items():
        a_ = st[st.wgroup.isin(f)]
        b_ = nw[nw.wgroup.isin(f)]
        # paired by detector: per-detector mean in each period, detectors present in both
        ga = a_.groupby(["DeviceId", "Detector"]).okA.mean()
        gb_ = b_.groupby(["DeviceId", "Detector"]).okA.mean()
        j = pd.concat([ga.rename("sept"), gb_.rename("wkd")], axis=1).dropna().reset_index()
        res[p] = {"sept_rows": len(a_), "wkd_rows": len(b_), "sept": C.acc_ci(a_.okA.to_numpy(), a_.DeviceId.to_numpy()),
                  "wkd": C.acc_ci(b_.okA.to_numpy(), b_.DeviceId.to_numpy()),
                  "paired_det_delta_pt": C.delta_ci(j.sept.to_numpy(), j.wkd.to_numpy(), j.DeviceId.to_numpy()),
                  "paired_dets": len(j)}
        log(f"{p}: {res[p]}")
    json.dump(res, open(OUT / f"wkd_{cfg}.json", "w"), indent=1, default=str)

# ------------------------------------------------------------------------------------------------ off-peak (note 59 d)
def stage_offpeak(a):
    """note-59 off-peak features for the weekday windows: derived tables from the s66 event cache, then note 59's own
    query (source re-used, period / windows / targets swapped) -> %DC_WORK%/s66/offpeak_wkd.parquet."""
    import duckdb
    import inspect
    import s59_step6 as S59
    import a2_features as A2F
    C = OUT / "cache"
    C.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='10GB'; SET threads={NTHREAD}; SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    ev = f"read_parquet('{EVC.as_posix()}/*/*.parquet', hive_partitioning=true)"
    if not (C / "det_intervals.parquet").exists():
        con.execute(f"""COPY (WITH e AS (SELECT DeviceId, Parameter AS det, Timestamp AS ts, EventId FROM {ev}
                                          WHERE EventId IN (81, 82)),
              d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev FROM e
                    WINDOW w AS (PARTITION BY DeviceId, det ORDER BY ts, CASE WHEN EventId = 82 THEN 0 ELSE 1 END))
            SELECT DeviceId, det::UTINYINT AS Detector, ts AS t_on, nts AS t_off, epoch_ms(nts - ts) / 1000.0 AS dur
            FROM d WHERE EventId = 82 AND nev = 81 AND nts IS NOT NULL)
            TO '{(C / "det_intervals.parquet").as_posix()}' (FORMAT parquet)""")
        con.execute(f"""COPY (WITH g AS (
              SELECT DeviceId, Parameter AS p, Timestamp AS t, EventId,
                     SUM(CASE WHEN EventId = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY DeviceId, Parameter ORDER BY Timestamp,
                         CASE EventId WHEN 1 THEN 0 WHEN 7 THEN 1 WHEN 8 THEN 2 WHEN 9 THEN 3 WHEN 10 THEN 4 ELSE 5 END
                         ROWS UNBOUNDED PRECEDING) AS cyc
              FROM {ev} WHERE EventId IN (1, 7, 8, 9, 10, 11)),
            c AS (SELECT DeviceId, p, cyc, min(t) FILTER (EventId = 1) AS green_start, min(t) FILTER (EventId = 8) AS ye,
                         min(t) FILTER (EventId = 7) AS gt, min(t) FILTER (EventId = 10) AS re,
                         min(t) FILTER (EventId = 9) AS yend
                  FROM g WHERE cyc > 0 AND p BETWEEN 1 AND 16 GROUP BY 1, 2, 3)
            SELECT DeviceId, p::UTINYINT AS Phase, green_start, coalesce(ye, gt) AS yellow_start,
                   coalesce(re, yend) AS red_start,
                   LEAD(green_start) OVER (PARTITION BY DeviceId, p ORDER BY green_start) AS next_green
            FROM c WHERE green_start IS NOT NULL)
            TO '{(C / "phase_cycles.parquet").as_posix()}' (FORMAT parquet)""")
        log("derived tables written")
    con.close()
    src = inspect.getsource(S59.stage_offpeak)
    src = src.replace('cache = {"dec": DCW / "cache", "stg": DCW / "official" / "stg" / "cache"}', 'cache = {"wkd": S66C}')
    src = src.replace('for period in ("dec", "stg"):', 'for period in ("wkd",):')
    src = src.replace('F.to_parquet(OUT / "offpeak.parquet", index=False)', 'F.to_parquet(S66OUT, index=False)')
    assert src.count("S66") == 2 and 'for period in ("wkd",)' in src
    nw = pd.read_parquet(OUT / "wkd_frame.parquet", columns=["DeviceId", "Detector", "period", "win", "pred_phase",
                                                             "det_n_on"])
    A2F.WINDOWS["wkd"] = [(n, str(t), s_) for n, t, s_ in WKD]
    g = dict(vars(S59))
    g.update(S66C=C, S66OUT=OUT / "offpeak_wkd.parquet", frame_keys=lambda: nw)
    exec(src, g)
    g["stage_offpeak"](a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dthreads", type=int, default=2)
    ap.add_argument("--dmem", default="2500MB")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--wins", default="m5_a,m30_a,h6_a,full66")
    ap.add_argument("--cfg", default="plus")
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--base", default="base@s0")
    ap.add_argument("--tag", default="screen")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)
