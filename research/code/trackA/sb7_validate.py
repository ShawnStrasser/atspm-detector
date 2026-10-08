"""Note 58 (validation plan step 5) -- SETBACK: separate model vs outputs of the main model, by sample length, on
current labels / inputs.

Truth: printed single-number distance_ft of print Advance / Mid (print_labels: training + the released NEWTEST half),
role and unusual flag from function_labels_v3s (stack relabels, radar-over-loop signals re-admitted), rows with
exclude_train_score out. locked_v2 asserted absent everywhere.
Inputs: frame v6e predicted phase + note-57 function arm (229 features, 3-seed OOF mean; argmax + P(Mid)); the
travel-time block's same-lane model = note-30 pair model per fold (sb5_eval.pair_model). Features = note 41's SQL
(sb5_setback.features) on each Sept-2026 window: m30 a-d, h6 a/b, h24 a/b, full66; phase cycles restricted to the
window (note 41 ran 66 h only).
Estimators (LightGBM P50 of log distance, 3 seeds, one model per window group pooled over its windows, OOF six folds
of folds_v4):
  all      note 41's 60 features (as is)
  nofunc   no function / lane input: minus phase make-up by predicted function, P(Mid), stop-bar volume ratio and
           the travel-time block (needs predicted stop-bar zones + lanes)
  nophys   all minus the travel-time block            std    phase timing + make-up (note 41's "agency standard")
  timing   phase timing only                          own    the detector's own behaviour only (no phase timing)
  physics  note-39 rule (d_user) where a same-lane predicted stop-bar partner exists; fold median; shuffled control

    python sb7_validate.py feats | eval
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb5_setback as SB  # noqa: E402
import sb5_eval as S5  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rpath  # noqa: E402,F401

DCW = S5.DCW
OUT = DCW / "trackA" / "setback" / "sb7"
REPO = Path(__file__).resolve().parents[3]
V3S = rpath.LABELS_CURRENT   # note 81: v4l (was function_labels_v3s)
FUNC_DIR = DCW / "trees57" / "function" / "v6e" / "drop_pp_xcand-pp_pdiff-pp_v2-yr-ratio-phctx"
WINS = {"m30_a": ("2026-09-21 07:30:00", 1800), "m30_b": ("2026-09-19 12:00:00", 1800),
        "m30_c": ("2026-09-19 21:30:00", 1800), "m30_d": ("2026-09-18 17:00:00", 1800),
        "h6_a": ("2026-09-20 06:00:00", 6 * 3600), "h6_b": ("2026-09-19 12:00:00", 6 * 3600),
        "h24_a": ("2026-09-19 00:00:00", 24 * 3600), "h24_b": ("2026-09-20 00:00:00", 24 * 3600),
        "full66": ("2026-09-18 16:15:00", 66 * 3600)}
WGS = ["m30", "h6", "h24", "full"]
C7 = S5.C7
TT = ["tau", "same", "p_same", "n_in", "pk_strength", "v_self", "tau_any", "pk_any", "p_same_any", "n_cand", "n_acc",
      "tau_rank", "d_user", "d_lfit", "d_any_lfit", "role_c"]
MAKEUP = ["ph_n_Advance", "ph_n_Mid", "ph_n_Presence", "ph_n_Count", "ph_n_Yellow_Red", "ph_n_Other", "p_mid"]
TIMING = ["green_mean", "green_q90", "red_mean", "cycle_mean", "green_share", "n_cyc"]
VARS = {
    "all": SB.FEATS,
    "nofunc": [f for f in SB.FEATS if f not in set(TT + MAKEUP + ["vol_vs_sb", "v_phase_count", "v_phase_pres"])],
    "nophys": [f for f in SB.FEATS if f not in set(TT)],
    "std": SB.STD_FEATS,
    "timing": TIMING,
    "own": [f for f in SB.FEATS if f not in set(TT + MAKEUP + TIMING + ["vol_vs_sb", "v_phase_count", "v_phase_pres",
                                                                          "n_det_phase"])],
}
log = SB.log


def wgroup(w):
    return "full" if w.startswith("full") else w.split("_")[0]


def truth_all() -> pd.DataFrame:
    """print rows incl. those the print table flagged unusual (v3s re-admitted 33 radar-over-loop signals)."""
    import duckdb  # noqa: F401
    con = S5.connect()
    cols = "DeviceId, detector, function, technology, distance_ft, unusual_layout"
    rel = pd.read_csv(DCW / "official" / "newtest_released.csv")
    rel["dev"] = rel.DeviceId.str.lower()
    con.register("rel", rel[["dev"]])
    a = con.sql(f"SELECT {cols}, 'train' AS src FROM read_parquet("
                f"'{(DCW / 'cabinet' / 'print_labels.parquet').as_posix()}')").df()
    b = con.sql(f"SELECT {cols}, 'released' AS src FROM read_parquet("
                f"'{(DCW / 'cabinet_locked' / 'print_labels.parquet').as_posix()}') "
                "WHERE lower(DeviceId) IN (SELECT dev FROM rel)").df()
    p = pd.concat([a, b], ignore_index=True)
    p["dev"] = p.DeviceId.str.lower()
    assert not p.dev.isin(S5.locked()).any()
    d = p.distance_ft.map(S5.parse_dist)
    p["dist"], p["dist_kind"] = [x[0] for x in d], [x[1] for x in d]
    p = p.rename(columns={"detector": "det", "function": "print_fn_raw"})
    p["det"] = p.det.astype("int16")
    v = pd.read_parquet(V3S, columns=["DeviceId", "detector", "print_function", "unusual_layout",
                                      "exclude_train_score"])
    v["dev"] = v.DeviceId.str.lower()
    v["det"] = v.detector.astype("int16")
    assert not v.dev.isin(S5.locked()).any()
    p = p.merge(v[["dev", "det", "print_function", "unusual_layout", "exclude_train_score"]].rename(
        columns={"unusual_layout": "unusual_v3s"}), on=["dev", "det"], how="left")
    p["print_fn"] = p.print_function.fillna(p.print_fn_raw)
    unus = p.unusual_v3s.where(p.unusual_v3s.notna(), p.unusual_layout)
    p = p[~unus.eq(True) & ~p.exclude_train_score.eq(True)]
    return p[["dev", "det", "print_fn", "technology", "dist", "dist_kind", "src"]].drop_duplicates(["dev", "det"])


def predictions() -> pd.DataFrame:
    import v3_retrain as V
    V.set_frame("v6e")
    fr = pd.read_parquet(V.FEATS, columns=V.KEY + ["fold", "pred_phase", "det_n_on"])
    P = np.zeros((len(fr), 7), np.float32)
    fo = fr.fold.to_numpy()
    for s in (0, 1, 2):
        for k in range(6):
            P[fo == k] += np.load(FUNC_DIR / f"P_first.all.wi_s{s}_f{k}.npy") / 3
    m = ((fr.period == "stg") & fr.win.isin(WINS)).to_numpy()
    d = fr[m].reset_index(drop=True)
    P = P[m]
    d["dev"] = d.DeviceId.str.lower()
    d["fn"] = np.array(C7)[P.argmax(1)]
    d["p_mid"] = P[:, 4]
    d = d.rename(columns={"Detector": "det", "pred_phase": "phase", "det_n_on": "n_on"})
    d["det"] = d.det.astype("int16")
    assert not d.dev.isin(S5.locked()).any()
    return d[["dev", "det", "win", "phase", "fn", "p_mid", "n_on", "fold"]]


def stage_feats(chunk: int = 80):
    OUT.mkdir(parents=True, exist_ok=True)
    tr = truth_all()
    pr = predictions()
    lab = tr[tr.print_fn.isin(SB.TARGET) & (tr.dist_kind == "single")]
    pr = pr.merge(lab[["dev", "det"]].assign(lab=True), on=["dev", "det"], how="left")
    pr["is_t"] = pr.fn.isin(SB.TARGET) | pr.lab.eq(True)
    log(f"{pr.dev.nunique()} signals; labelled A/Mid with distance {len(lab)}")
    pms = {k: S5.pair_model(k) for k in range(6)}
    parts = []
    for win, (ts, secs) in WINS.items():
        T0 = pd.Timestamp(ts)
        hours = secs / 3600.0
        e0 = (T0 - pd.Timestamp("1970-01-01")).total_seconds()
        t1 = T0 + pd.Timedelta(seconds=secs)
        pw = pr[pr.win == win]
        t_w = time.time()
        for k in range(6):
            devs = sorted(pw[pw.fold == k].dev.unique())
            for i in range(0, len(devs), chunk):
                ds = devs[i:i + chunk]
                d = pw[pw.dev.isin(ds)]
                con = S5.connect()
                con.execute("SET threads=6")
                con.register("dv", pd.DataFrame({"dev": ds}))
                con.execute(f"""CREATE TEMP TABLE iv AS
                    SELECT lower(DeviceId) AS dev, Detector::SMALLINT AS det,
                           epoch_ms(t_on) / 1000.0 - {e0} AS t, dur,
                           (hour(t_on) >= 22 OR hour(t_on) < 6) AS quiet
                    FROM read_parquet('{(S5.CACHE / 'det_intervals.parquet').as_posix()}')
                    WHERE lower(DeviceId) IN (SELECT dev FROM dv)
                      AND t_on >= TIMESTAMP '{T0}' AND t_on < TIMESTAMP '{t1}'""")
                con.execute(f"""CREATE TEMP TABLE cyc AS
                    SELECT lower(DeviceId) AS dev, Phase::INT AS phase,
                           epoch_ms(green_start) / 1000.0 - {e0} AS t_g,
                           epoch_ms(coalesce(yellow_start, red_start, next_green)) / 1000.0 - {e0} AS t_y,
                           epoch_ms(coalesce(red_start, yellow_start, next_green)) / 1000.0 - {e0} AS t_red,
                           epoch_ms(next_green) / 1000.0 - {e0} AS t_next
                    FROM read_parquet('{(S5.CACHE / 'phase_cycles.parquet').as_posix()}')
                    WHERE next_green IS NOT NULL AND lower(DeviceId) IN (SELECT dev FROM dv)
                      AND next_green > TIMESTAMP '{T0}' AND green_start < TIMESTAMP '{t1}'""")
                F = SB.features(con, d[["dev", "det", "phase", "fn", "n_on", "p_mid", "is_t"]], hours, pms[k])
                con.close()
                F["fold"] = k
                F["win"] = win
                parts.append(F)
        log(f"{win}: {sum(len(p) for p in parts if p.win.iloc[0] == win)} targets ({time.time()-t_w:.0f}s)")
    F = pd.concat(parts, ignore_index=True)
    F = F.merge(tr, on=["dev", "det"], how="left")
    F.to_parquet(OUT / "feat.parquet", index=False)
    log(f"wrote feat.parquet {F.shape}")


# ------------------------------------------------------------------------------- eval
def fit_p50(tr, te, feats, seeds=(0, 1, 2), y=None):
    import lightgbm as lgb
    L = SB.fit_L(tr)
    tr, te = SB.add_physics(tr, L), SB.add_physics(te, L)
    yy = SB._lg(tr.dist.to_numpy() if y is None else y)
    prm = dict(SB.LGB, n_jobs=4)
    p = np.mean([lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=s, **prm)
                 .fit(tr[feats].astype(float), yy).predict(te[feats].astype(float)) for s in seeds], 0)
    p = np.exp(p)
    return np.where(p < SB.CLIP0, 0.0, p), te


def stage_eval():
    F = pd.read_parquet(OUT / "feat.parquet")
    F["wg"] = F.win.map(wgroup)
    L = F.dist_kind.eq("single") & F.print_fn.isin(SB.TARGET) & (F.n_on > 0)
    E = pd.DataFrame(index=F.index)
    for wg in WGS:
        for k in range(6):
            trm = L & (F.wg == wg) & (F.fold != k)
            te = (F.wg == wg) & (F.fold == k)
            tr = F[trm]
            E.loc[te, "median"] = tr.dist.median()
            for v, fs in VARS.items():
                for si, seeds in enumerate([(0, 1, 2)] + ([(3,), (4,), (5,)] if v in ("all", "nofunc") else [])):
                    p, tep = fit_p50(tr, F[te], fs, seeds)
                    E.loc[te, v if si == 0 else f"{v}_s{seeds[0]}"] = p
            E.loc[te, "d_user"] = np.where(tep.d_user < SB.CLIP0, 0.0, tep.d_user)
            E.loc[te, "covered"] = tep.covered.to_numpy()
            sh = np.random.default_rng(k).permutation(tr.dist.to_numpy())
            E.loc[te, "shuffled"] = fit_p50(tr, F[te], VARS["all"], (0,), y=sh)[0]
        log(f"{wg} done")
    F = pd.concat([F.drop(columns=[c for c in ("d_user", "covered") if c in F]), E], axis=1)
    F.to_parquet(OUT / "oof.parquet", index=False)
    report(F)


def _m(y, p):
    ok = ~np.isnan(p)
    e = np.abs(p[ok] - y[ok])
    return {"n": int(ok.sum()), "medAE": round(float(np.median(e)), 1) if len(e) else None,
            "w50": round(100 * float(np.mean(e <= 50)), 1) if len(e) else None,
            "w100": round(100 * float(np.mean(e <= 100)), 1) if len(e) else None}


def _boot(F, a, b, reps=1000, seed=58):
    """paired signal bootstrap of (b - a): medAE ft and within-50 / within-100 pt (rows where both answer)."""
    m = F[a].notna() & F[b].notna()
    x = F[m]
    ea, eb = np.abs(x[a] - x.dist).to_numpy(), np.abs(x[b] - x.dist).to_numpy()
    sig = x.dev.to_numpy()
    u, inv = np.unique(sig, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(u))]
    rng = np.random.default_rng(seed)
    out = {"medAE": [], "w50": [], "w100": []}
    for _ in range(reps):
        s = np.concatenate([idx[i] for i in rng.integers(0, len(u), len(u))])
        out["medAE"].append(np.median(eb[s]) - np.median(ea[s]))
        out["w50"].append(100 * ((eb[s] <= 50).mean() - (ea[s] <= 50).mean()))
        out["w100"].append(100 * ((eb[s] <= 100).mean() - (ea[s] <= 100).mean()))
    pt = {"medAE": np.median(eb) - np.median(ea), "w50": 100 * ((eb <= 50).mean() - (ea <= 50).mean()),
          "w100": 100 * ((eb <= 100).mean() - (ea <= 100).mean())}
    return {k: [round(float(pt[k]), 1), round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)]
            for k, v in out.items()}


def report(F):
    adv = F.dist_kind.eq("single") & (F.print_fn == "Advance") & (F.n_on > 0)
    ests = ["all", "all_s3", "all_s4", "all_s5", "nofunc", "nofunc_s3", "nofunc_s4", "nofunc_s5", "nophys", "std",
            "timing", "own", "d_user", "median", "shuffled"]
    for v in ("all", "nofunc"):
        F[f"prod_{v}"] = np.where(F.fn.isin(SB.TARGET), F[v], np.where(F.fn.isin(SB.STOPBAR), 0.0, np.nan))
    ests += ["prod_all", "prod_nofunc"]
    res = {"n_adv": {wg: int((adv & (F.wg == wg)).sum()) for wg in WGS},
           "signals": {wg: int(F[adv & (F.wg == wg)].dev.nunique()) for wg in WGS}, "metrics": {}, "ci_vs_all": {},
           "ci_vs_nofunc": {}}
    for e in ests:
        res["metrics"][e] = {wg: _m(F.dist[adv & (F.wg == wg)].to_numpy(), F[e][adv & (F.wg == wg)].to_numpy(float))
                             for wg in WGS}
        cov = adv & F.covered.eq(True)
        res["metrics"][e]["full_covered"] = _m(F.dist[cov & (F.wg == "full")].to_numpy(),
                                               F[e][cov & (F.wg == "full")].to_numpy(float))
        for t in ("loop", "radar", "video"):
            mm = adv & (F.wg == "full") & (F.technology == t)
            res["metrics"][e][f"full_{t}"] = _m(F.dist[mm].to_numpy(), F[e][mm].to_numpy(float))
        mid = F.dist_kind.eq("single") & (F.print_fn == "Mid") & (F.n_on > 0) & (F.wg == "full")
        res["metrics"][e]["full_mid"] = _m(F.dist[mid].to_numpy(), F[e][mid].to_numpy(float))
        log(f"{e:12s} " + " | ".join(f"{wg} {res['metrics'][e][wg]['medAE']} ft {res['metrics'][e][wg]['w50']}% "
                                      f"{res['metrics'][e][wg]['w100']}%" for wg in WGS))
    for ref in ("all", "nofunc"):
        for e in ests:
            if e == ref or e.startswith("prod") or e == "d_user":
                continue
            res[f"ci_vs_{ref}"][e] = {wg: _boot(F[adv & (F.wg == wg)], ref, e) for wg in WGS}
        log(f"vs {ref}: " + json.dumps({e: res[f'ci_vs_{ref}'][e] for e in ('nofunc', 'all', 'nophys', 'own', 'std')
                                        if e in res[f'ci_vs_{ref}']}))
    res["pred_fn_of_printed_adv_full"] = F[adv & (F.wg == "full")].fn.value_counts().to_dict()
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    st = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"feats": stage_feats, "eval": stage_eval, "report": lambda: report(pd.read_parquet(OUT / "oof.parquet"))}[st]()
