"""A2 -- does the expert feature family buy anything?  6 signal-grouped folds.

    python a2_model.py --stage screen     # one fold, the cheap screening run
    python a2_model.py --stage full       # 6-fold OOF, base vs base+expert, seeds
    python a2_model.py --stage ablate     # leave-one-expert-family-out, one fold
"""
from __future__ import annotations
import argparse
import json
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd

# repository root: DC_REPO, else three levels up when run from research/code/trackA
REPO = Path(os.environ.get("DC_REPO") or Path(__file__).resolve().parents[3])
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
WORK = DCW / "trackA"
FRAME = DCW / "function_v4" / "funcframe_v4.parquet"
LABELS = REPO / "research" / "labels" / "function_labels_v2.parquet"

N_FOLDS = 6
CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CLASSES3 = ["Advance", "Presence", "Count"]
DUR_GROUPS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]

KEY_EXCLUDE = {"DeviceId", "Detector", "cand_phase", "win", "dev", "Phase", "Function",
               "fold", "y", "cyc", "partner_phase", "health_flag", "std_phase",
               "period", "func5", "cfg_phase", "src", "label_phase", "other_phase",
               "prob", "pred_phase", "split", "has_dec", "has_stg", "wgroup",
               "top_prob_x", "sib_n_x", "is_full", "func5_config", "label_source",
               "reviewed", "drop_from_use", "px_n_on"}

FUNC_PARAMS = dict(objective="multiclass", learning_rate=0.05, num_leaves=31,
                   min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=1.0, n_estimators=1200, n_jobs=12,
                   verbose=-1)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ the frame
def family_of(c: str) -> str:
    if not c.startswith("px_"):
        return "base"
    if c.startswith(("px_corr", "px_n_sib", "px_n_any", "px_ratiodiv", "px_rpeak",
                     "px_roff")):
        return "pair"
    if c.startswith(("px_occ_red", "px_red_late", "px_span_to_green", "px_frac_on_first5",
                     "px_on_first5", "px_hold_through_red", "px_g_first", "px_cyc_hit",
                     "px_dur_med_red", "px_dur_med_green", "px_dur_red_over_green")):
        return "cycle"
    if c.startswith(("px_disp", "px_cnt_cv", "px_peak_over_med", "px_occ_bin",
                     "px_saturation", "px_occ_per_on")):
        return "demand"
    return "shape"


def load_frame(with_expert: bool = True) -> pd.DataFrame:
    fr = pd.read_parquet(FRAME)
    fr = fr[fr.func5.notna()].reset_index(drop=True)
    # corrected labels (a no-op inside the folds -- every correction is on a locked
    # signal -- but this is the one truth table from now on)
    lab = pd.read_parquet(LABELS)[["DeviceId", "Detector", "func5", "drop_from_use"]]
    lab["Detector"] = lab.Detector.astype(fr.Detector.dtype)
    fr = fr.drop(columns=["func5"]).merge(lab, on=["DeviceId", "Detector"], how="left")
    n0 = len(fr)
    fr = fr[fr.func5.notna() & ~fr.drop_from_use.fillna(False)].reset_index(drop=True)
    log(f"frame {n0:,} -> {len(fr):,} labelled rows after the corrected label table")
    if with_expert:
        for tag in ("det", "cyc", "pair"):
            parts = []
            for per in ("dec", "stg"):
                f = WORK / f"feat_expert_{tag}_{per}.parquet"
                parts.append(pd.read_parquet(f))
            e = pd.concat(parts, ignore_index=True)
            e["Detector"] = e.Detector.astype(fr.Detector.dtype)
            before = fr.shape[1]
            fr = fr.merge(e, on=["DeviceId", "Detector", "win", "period"], how="left")
            log(f"  + {tag}: {fr.shape[1]-before} columns, "
                f"{fr[e.columns[2]].notna().mean():.3f} matched")
    fr["is_full"] = fr.wgroup == "full"
    return fr


def feat_cols(fr: pd.DataFrame, families=("base",)) -> list[str]:
    return [c for c in fr.columns
            if c not in KEY_EXCLUDE and pd.api.types.is_numeric_dtype(fr[c])
            and family_of(c) in families]


# ------------------------------------------------------------------- training
def fit_fold(fr, y, cols, k, seed=0, train_mask=None, params=FUNC_PARAMS,
             classes=CLASSES5):
    import lightgbm as lgb
    folds = fr.fold.to_numpy()
    ok = (fr.health_flag != "failed").to_numpy()
    if train_mask is not None:
        ok = ok & train_mask
    inner = (k + 1) % N_FOLDS
    trm = ok & (folds != k) & (folds != inner)
    vam = ok & (folds != k) & (folds == inner)
    prm = dict(params, num_class=len(classes), seed=seed, bagging_seed=seed + 1,
               feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    n = prm.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n, **prm)
    m.fit(fr.loc[trm, cols], y[trm],
          eval_set=[(fr.loc[vam, cols], y[vam])], eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    return m


def oof(fr, y, cols, seeds=(0,), classes=CLASSES5):
    P = np.zeros((len(fr), len(classes)))
    folds = fr.fold.to_numpy()
    iters, models = [], []
    for k in range(N_FOLDS):
        acc = np.zeros((int((folds == k).sum()), len(classes)))
        for s in seeds:
            m = fit_fold(fr, y, cols, k, seed=s, classes=classes)
            acc += m.predict_proba(fr.loc[folds == k, cols])
            iters.append(int(m.best_iteration_ or 0))
            models.append(m)
        P[folds == k] = acc / len(seeds)
        log(f"  fold {k}: {iters[-1]} trees")
    return P, models, iters


# --------------------------------------------------------------------- scoring
def score(fr, P, classes=CLASSES5, mask=None) -> dict:
    m = np.ones(len(fr), bool) if mask is None else mask
    yt = fr.func5.to_numpy()[m]
    pred = np.array(classes)[P[m].argmax(1)]
    wg = fr.wgroup.to_numpy()[m]
    apc = np.isin(yt, CLASSES3)
    d = {"n": int(m.sum()), "acc5_allwin": round(float((yt == pred).mean()), 4),
         "accAPC_allwin": round(float((yt[apc] == pred[apc]).mean()), 4)}
    by, byapc = {}, {}
    for g in DUR_GROUPS:
        s = wg == g
        if not s.any():
            continue
        by[g] = round(float((yt[s] == pred[s]).mean()), 4)
        sa = s & apc
        byapc[g] = round(float((yt[sa] == pred[sa]).mean()), 4) if sa.any() else None
    d["acc5_by_duration"] = by
    d["accAPC_by_duration"] = byapc
    full = wg == "full"
    per = {}
    for c in classes:
        tp = int(((yt[full] == c) & (pred[full] == c)).sum())
        fp = int(((yt[full] != c) & (pred[full] == c)).sum())
        fn = int(((yt[full] == c) & (pred[full] != c)).sum())
        per[c] = {"n": tp + fn,
                  "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                  "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    d["per_class_full"] = per
    d["other_recall_allwin"] = round(float((pred[yt == "Other"] == "Other").mean()), 4)
    return d


def paired_bootstrap(fr, mask, ok_a, ok_b, n_boot=2000, seed=0) -> dict:
    sig = fr.DeviceId.to_numpy()[mask]
    a, b = ok_a[mask].astype(float), ok_b[mask].astype(float)
    uniq, inv = np.unique(sig, return_inverse=True)
    n = len(uniq)
    sa = np.bincount(inv, weights=a, minlength=n)
    sb = np.bincount(inv, weights=b, minlength=n)
    cnt = np.bincount(inv, minlength=n)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (sb[idx].sum() - sa[idx].sum()) / cnt[idx].sum()
    return {"delta_pt": round(float((b.mean() - a.mean()) * 100), 3),
            "lo90_pt": round(float(np.quantile(diffs, 0.05) * 100), 3),
            "hi90_pt": round(float(np.quantile(diffs, 0.95) * 100), 3),
            "p_better": round(float((diffs > 0).mean()), 3)}


# ---------------------------------------------------------------------- stages
ALL_FAM = ("base", "shape", "demand", "cycle", "pair")


def stage_screen(a) -> None:
    fr = load_frame()
    y = fr.func5.map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    res = {}
    te = (fr.fold == 0).to_numpy()
    # noise control: the same number of pure-noise columns as the expert family has
    n_px = len(feat_cols(fr, tuple(f for f in ALL_FAM if f != "base")))
    rng = np.random.default_rng(7)
    noise_cols = [f"noise_{i}" for i in range(n_px)]
    for i, c in enumerate(noise_cols):
        fr[c] = rng.standard_normal(len(fr)).astype(np.float32)
    for tag, fams in (("base", ("base",)), ("base+expert", ALL_FAM),
                      ("base+noise", ("base",))):
        cols = feat_cols(fr, fams)
        if tag == "base+noise":
            cols = [c for c in cols if not c.startswith("noise_")] + noise_cols
        else:
            cols = [c for c in cols if not c.startswith("noise_")]
        t0 = time.time()
        m = fit_fold(fr, y, cols, 0, seed=0)
        P = np.zeros((len(fr), 5))
        P[te] = m.predict_proba(fr.loc[te, cols])
        res[tag] = score(fr, P, mask=te)
        res[tag]["n_features"] = len(cols)
        res[tag]["trees"] = int(m.best_iteration_ or 0)
        res[tag]["fit_secs"] = round(time.time() - t0, 1)
        np.save(WORK / f"a2_screen_P_{tag.replace('+','_')}.npy", P[te])
        log(f"{tag}: {len(cols)} feats, {res[tag]['acc5_allwin']:.4f} allwin, "
            f"full {res[tag]['acc5_by_duration'].get('full')}, "
            f"{res[tag]['fit_secs']:.0f}s")
    yt = fr.func5.to_numpy()[te]
    sub = fr[te].reset_index(drop=True)
    allm = np.ones(len(sub), bool)
    ok = {t: np.array(CLASSES5)[np.load(
        WORK / f"a2_screen_P_{t.replace('+', '_')}.npy").argmax(1)] == yt
        for t in ("base", "base+expert", "base+noise")}
    res["paired_fold0"] = {
        "expert_vs_base": paired_bootstrap(sub, allm, ok["base"], ok["base+expert"]),
        "noise_vs_base": paired_bootstrap(sub, allm, ok["base"], ok["base+noise"])}
    json.dump(res, open(WORK / "a2_screen.json", "w"), indent=1, default=str)
    log(json.dumps({k: (v if k == "paired_fold0" else v["acc5_by_duration"])
                    for k, v in res.items()}, indent=1))


def stage_ablate(a) -> None:
    fr = load_frame()
    y = fr.func5.map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    te = (fr.fold == 0).to_numpy()
    yt = fr.func5.to_numpy()[te]
    res = {}
    variants = [("full expert", ALL_FAM)]
    for drop in ("shape", "demand", "cycle", "pair"):
        variants.append((f"- {drop}", tuple(f for f in ALL_FAM if f != drop)))
    variants.append(("only expert (no base)", tuple(f for f in ALL_FAM if f != "base")))
    for tag, fams in variants:
        cols = feat_cols(fr, fams)
        m = fit_fold(fr, y, cols, 0, seed=0)
        P = np.zeros((len(fr), 5))
        P[te] = m.predict_proba(fr.loc[te, cols])
        res[tag] = score(fr, P, mask=te)
        res[tag]["n_features"] = len(cols)
        log(f"{tag}: {len(cols)} feats allwin {res[tag]['acc5_allwin']:.4f} "
            f"full {res[tag]['acc5_by_duration'].get('full')}")
    json.dump(res, open(WORK / "a2_ablate.json", "w"), indent=1, default=str)


def stage_full(a) -> None:
    fr = load_frame()
    y = fr.func5.map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    yt = fr.func5.to_numpy()
    seeds = [int(s) for s in str(a.seeds).split(",") if s != ""]
    res, P = {}, {}
    for tag, fams in (("base", ("base",)), ("base+expert", ALL_FAM)):
        cols = feat_cols(fr, fams)
        per_seed = []
        acc = None
        for s in seeds:
            Ps, models, iters = oof(fr, y, cols, seeds=(s,))
            per_seed.append(score(fr, Ps))
            acc = Ps if acc is None else acc + Ps
            log(f"{tag} seed {s}: allwin {per_seed[-1]['acc5_allwin']:.4f} "
                f"full {per_seed[-1]['acc5_by_duration'].get('full')}")
            if s == seeds[0]:
                np.save(WORK / f"a2_oof_{tag.replace('+','_')}_seed{s}.npy", Ps)
                imp = pd.Series(np.mean([m.booster_.feature_importance("gain")
                                         for m in models], axis=0), index=cols)
                imp.sort_values(ascending=False).to_csv(
                    WORK / f"a2_importance_{tag.replace('+','_')}.csv", header=["gain"])
        P[tag] = acc / len(seeds)
        res[tag] = {"n_features": len(cols), "per_seed": per_seed,
                    "sd_acc5_allwin_pt": round(float(np.std(
                        [p["acc5_allwin"] for p in per_seed], ddof=1) * 100), 3)
                    if len(per_seed) > 1 else None,
                    "seed_mean": score(fr, P[tag])}
    oka = np.array(CLASSES5)[P["base"].argmax(1)] == yt
    okb = np.array(CLASSES5)[P["base+expert"].argmax(1)] == yt
    full = fr.is_full.to_numpy()
    m30 = (fr.wgroup == "m30").to_numpy()
    res["paired"] = {
        "allwin": paired_bootstrap(fr, np.ones(len(fr), bool), oka, okb),
        "full": paired_bootstrap(fr, full, oka, okb),
        "m30": paired_bootstrap(fr, m30, oka, okb)}
    np.save(WORK / "a2_oof_base_expert_mean.npy", P["base+expert"])
    np.save(WORK / "a2_oof_base_mean.npy", P["base"])
    fr[["DeviceId", "Detector", "period", "win", "wgroup", "fold", "func5",
        "Function", "split", "pred_phase", "det_n_on", "top_prob"]].to_parquet(
        WORK / "a2_oof_keys.parquet", index=False)
    json.dump(res, open(WORK / "a2_full.json", "w"), indent=1, default=str)
    log(json.dumps(res["paired"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["screen", "full", "ablate"])
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
