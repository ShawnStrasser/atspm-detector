"""Note 25 -- the whole-intersection problem: active channels that carry no function label.

In production the model sees every active channel of a signal; training only ever showed it
the channels the config export labels.  This script measures the size of that gap and runs one
cheap experiment on the six signal-grouped folds (training / dev signals only).

    python w1_whole_intersection.py --stage audit          # counts, no model
    python w1_whole_intersection.py --stage exp --variants a,b,c --seeds 0
    python w1_whole_intersection.py --stage report         # tables from the saved OOF

Variants (all on the note-14 feature set: production 388 + 54 expert columns):
  a  T head, Other trained as one class (note 14 baseline)
  b  Other split into subtypes as training classes (mid, bike, advance-presence, other),
     summed back to Other for output and scoring
  b7 7-class head (A, P, C, YR, Mid, Bike, Other): Mid and Bike are REPORTED outputs (user
     decision 2026-09-23); collapsed to Other only to compare with (a)
  c  (b) + every UNLABELLED active channel at a labelled signal added as an extra
     "unexplained" Other class -- a sensitivity run only: without a cabinet print we do not
     know that the config export is complete at those signals.
Unlabelled rows are always scored out-of-fold (their signal's fold), never used by a/b.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

import a2_model as A2

DCW = A2.DCW
WORK = DCW / "trackA"
OUT = WORK / "w1"
LABELS = A2.LABELS
OFFICIAL = DCW / "official" / "labels_official.parquet"
META = {"dec": DCW / "cache" / "detector_meta.parquet",
        "stg": DCW / "official" / "stg" / "cache" / "detector_meta.parquet"}
LOCKED = [DCW / "data" / "splits" / "test_config.csv",
          DCW / "official" / "newtest_signals.csv"]
MIN_ON = 5
CLASSES5 = A2.CLASSES5
REAL4 = ["Advance", "Presence", "Count", "Yellow_Red"]
SUB = ["O_mid", "O_bike", "O_advpres", "O_other"]
PARAMS = dict(A2.FUNC_PARAMS, n_jobs=6)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked_signals() -> set:
    s = set()
    for f in LOCKED:
        d = pd.read_csv(f)
        s |= set(d.DeviceId.astype(str).str.lower())
    return s


def subtype(raw: str) -> str:
    r = str(raw).strip().lower()
    if "mid" in r:
        return "O_mid"
    if "bike" in r:
        return "O_bike"
    if r == "advance presence":
        return "O_advpres"
    return "O_other"


def active_channels() -> pd.DataFrame:
    parts = []
    for per, f in META.items():
        m = pd.read_parquet(f, columns=["DeviceId", "Detector", "n_on"])
        m["DeviceId"] = m.DeviceId.str.lower()
        m["Detector"] = m.Detector.astype(int)
        m = m[(m.Detector >= 1) & (m.Detector <= 64)]
        m["period"] = per
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------ frame
def load_all() -> pd.DataFrame:
    """The function frame WITH the unlabelled rows, corrected labels, expert features."""
    fr = pd.read_parquet(A2.FRAME)
    fr["DeviceId"] = fr.DeviceId.str.lower()
    lock = locked_signals()
    assert not fr.DeviceId.isin(lock).any(), "locked signal in the frame"
    fold_of = fr[fr.fold.notna()].groupby("DeviceId").fold.first()
    fr["fold"] = fr.DeviceId.map(fold_of)
    fr = fr[fr.fold.notna()].reset_index(drop=True)
    fr["fold"] = fr.fold.astype(int)
    lab = pd.read_parquet(LABELS)[["DeviceId", "Detector", "func5", "drop_from_use",
                                   "config_function"]]
    lab["DeviceId"] = lab.DeviceId.str.lower()
    lab["Detector"] = lab.Detector.astype(fr.Detector.dtype)
    fr = fr.drop(columns=["func5"]).merge(lab, on=["DeviceId", "Detector"], how="left",
                                          indicator="_lab")
    fr["in_label_table"] = fr._lab == "both"
    fr = fr.drop(columns="_lab")
    fr = fr[~fr.drop_from_use.fillna(False).astype(bool)]
    fr["labelled"] = fr.func5.notna()
    fr = fr[fr.labelled | ~fr.in_label_table].reset_index(drop=True)   # unmapped: out
    act = active_channels()
    act["Detector"] = act.Detector.astype(fr.Detector.dtype)
    fr = fr.merge(act.rename(columns={"n_on": "period_n_on"}),
                  on=["DeviceId", "Detector", "period"], how="left")
    fr["active"] = fr.period_n_on.fillna(0) >= MIN_ON
    for tag in ("det", "cyc", "pair"):
        e = pd.concat([pd.read_parquet(WORK / f"feat_expert_{tag}_{p}.parquet")
                       for p in ("dec", "stg")], ignore_index=True)
        e["DeviceId"] = e.DeviceId.str.lower()
        e["Detector"] = e.Detector.astype(fr.Detector.dtype)
        fr = fr.merge(e, on=["DeviceId", "Detector", "win", "period"], how="left")
    fr["is_full"] = fr.wgroup == "full"
    fr["sub"] = np.where(fr.func5 == "Other", fr.config_function.map(subtype), fr.func5)
    log(f"frame {len(fr):,} rows: labelled {int(fr.labelled.sum()):,}, unlabelled "
        f"{int((~fr.labelled).sum()):,} (active {int((~fr.labelled & fr.active).sum()):,})"
        f", {fr.DeviceId.nunique()} signals")
    return fr


# ------------------------------------------------------------------------ audit
def stage_audit(a) -> None:
    lock = locked_signals()
    lab = pd.read_parquet(LABELS)
    lab["DeviceId"] = lab.DeviceId.str.lower()
    lab = lab[~lab.DeviceId.isin(lock) & (lab.locked == "")]
    folds = pd.read_parquet(DCW / "function_v4" / "labels_v4.parquet",
                            columns=["DeviceId", "fold"]).drop_duplicates()
    folds["DeviceId"] = folds.DeviceId.str.lower()
    sig = set(folds.DeviceId) & set(lab.DeviceId)
    lab = lab[lab.DeviceId.isin(sig)]
    labd = lab[lab.func5.notna()][["DeviceId", "Detector", "func5", "config_function"]]
    act = active_channels()
    act = act[act.DeviceId.isin(sig)]
    act["active"] = act.n_on >= MIN_ON
    off = pd.read_parquet(OFFICIAL)
    off["DeviceId"] = off.DeviceId.str.lower()
    off["Detector"] = off.Detector.astype(int)
    labd["Detector"] = labd.Detector.astype(int)
    m = act.merge(labd, on=["DeviceId", "Detector"], how="left").merge(
        off[["DeviceId", "Detector", "target_type", "target_num", "call_ped",
             "description"]], on=["DeviceId", "Detector"], how="left")
    m["labelled"] = m.func5.notna()
    # coarse, generic reading of the timing's channel text for the unlabelled active ones
    u = m[m.active & ~m.labelled].copy()
    d = u.description.fillna("").str.lower()
    u["cat"] = np.select(
        [d.str.contains("dummy|fail"), d.str.contains("pass"), d.str.contains("fya"),
         d.str.contains("bike"), d.str.contains("adv|count|cnt|yr|y/r|pres|stop|sb"),
         d.str.contains("ped|preempt|pe "), d.str.strip() == ""],
        ["dummy/fail", "pass-through", "fya", "bike", "names a PM role", "ped/preempt",
         "empty"], "other text")
    OUT.mkdir(parents=True, exist_ok=True)
    u[["DeviceId", "Detector", "period", "cat"]].to_parquet(OUT / "unl_cat.parquet",
                                                            index=False)
    res = {"signals": len(sig),
           "unl_cat": u.groupby("period").cat.value_counts().unstack(0).to_dict()}
    for per, g in m.groupby("period"):
        ga = g[g.active]
        per_sig = ga.groupby("DeviceId").agg(n_act=("Detector", "size"),
                                             n_lab=("labelled", "sum"))
        per_sig["n_unl"] = per_sig.n_act - per_sig.n_lab
        u = ga[~ga.labelled]
        res[per] = {
            "signals_with_data": int(per_sig.shape[0]),
            "active_channels": int(len(ga)), "labelled": int(ga.labelled.sum()),
            "unlabelled": int((~ga.labelled).sum()),
            "unl_share": round(float((~ga.labelled).mean()), 4),
            "per_signal_active_median": float(per_sig.n_act.median()),
            "per_signal_unl_median": float(per_sig.n_unl.median()),
            "per_signal_unl_mean": round(float(per_sig.n_unl.mean()), 2),
            "signals_with_any_unl": int((per_sig.n_unl > 0).sum()),
            "signals_unl_ge3": int((per_sig.n_unl >= 3).sum()),
            "unl_target_type": u.target_type.fillna("not in timing").value_counts().to_dict(),
            "unl_call_ped": int((u.call_ped.fillna(0) > 0).sum()),
            "lab_target_type": ga[ga.labelled].target_type.fillna(
                "not in timing").value_counts().to_dict(),
            "unl_nOn_median": float(u.n_on.median()),
            "lab_nOn_median": float(ga[ga.labelled].n_on.median()),
            "labelled_but_inactive": int(g[~g.active & g.labelled].shape[0]),
            "unl_desc_empty_share": round(float(
                u.description.fillna("").str.strip().eq("").mean()), 3),
        }
        toks = (u.description.fillna("").str.lower()
                .str.replace(r"[^a-z ]", " ", regex=True).str.split().explode())
        res[per]["unl_desc_top_tokens"] = toks.value_counts().head(25).to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "audit.json", "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))


# ---------------------------------------------------------------------- models
def fit_predict(fr, y, cols, k, classes, train_mask, seed):
    import lightgbm as lgb
    folds = fr.fold.to_numpy()
    ok = (fr.health_flag != "failed").to_numpy() & train_mask
    inner = (k + 1) % A2.N_FOLDS
    trm = ok & (folds != k) & (folds != inner)
    vam = ok & (folds == inner)
    prm = dict(PARAMS, num_class=len(classes), seed=seed, bagging_seed=seed + 1,
               feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    n = prm.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n, **prm)
    m.fit(fr.loc[trm, cols], y[trm], eval_set=[(fr.loc[vam, cols], y[vam])],
          eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    te = folds == k
    return m.predict_proba(fr.loc[te, cols]), int(m.best_iteration_ or 0)


def collapse(P, classes) -> np.ndarray:
    """Sum the Other subtype columns back into one Other column, CLASSES5 order."""
    out = np.zeros((len(P), 5))
    for j, c in enumerate(classes):
        out[:, CLASSES5.index(c if c in REAL4 else "Other")] += P[:, j]
    return out


def stage_exp(a) -> None:
    fr = load_all()
    cols = A2.feat_cols(fr, A2.ALL_FAM)
    # bookkeeping columns added here must never be features (booleans pass the numeric test)
    cols = [c for c in cols if c not in ("period_n_on", "fold", "labelled", "active",
                                         "in_label_table", "is_full", "drop_from_use")]
    assert len(cols) == 442, len(cols)
    log(f"{len(cols)} features")
    lab = fr.labelled.to_numpy()
    unl_act = (~fr.labelled & fr.active).to_numpy()
    folds = fr.fold.to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    fr[["DeviceId", "Detector", "period", "win", "wgroup", "fold", "func5", "sub",
        "labelled", "active", "pred_phase", "top_prob", "det_n_on", "health_flag"]
       ].to_parquet(OUT / "keys.parquet", index=False)
    for v in a.variants.split(","):
        if v == "a":
            classes, target, tmask = CLASSES5, fr.func5.to_numpy(), lab
        elif v == "b":
            classes, target, tmask = REAL4 + SUB, fr["sub"].to_numpy(), lab
        elif v == "b7":                      # user decision: Mid and Bike are outputs
            classes = REAL4 + ["Mid", "Bike", "Other"]
            target = fr["sub"].map({"O_mid": "Mid", "O_bike": "Bike", "O_advpres": "Other",
                                    "O_other": "Other"}).fillna(fr["sub"]).to_numpy()
            tmask = lab
        elif v == "c":
            classes = REAL4 + SUB + ["O_unlab"]
            target = np.where(unl_act, "O_unlab", fr["sub"].to_numpy())
            tmask = lab | unl_act
        else:
            raise ValueError(v)
        yi = pd.Series(target).map({c: i for i, c in enumerate(classes)}).fillna(-1
                                                                                ).to_numpy()
        yi = yi.astype(int)
        assert (yi[tmask] >= 0).all()
        for s in [int(x) for x in a.seeds.split(",")]:
            f = OUT / f"P_{v}_s{s}.npy"
            if f.exists():
                log(f"{f.name} exists, skipped")
                continue
            P = np.zeros((len(fr), 5))
            Praw = np.zeros((len(fr), len(classes)))
            t0 = time.time()
            for k in range(A2.N_FOLDS):
                Pk, it = fit_predict(fr, yi, cols, k, classes, tmask, s)
                P[folds == k] = collapse(Pk, classes)
                Praw[folds == k] = Pk
                log(f"  {v} s{s} fold {k}: {it} trees")
            np.save(f, P.astype(np.float32))
            np.save(OUT / f"Praw_{v}_s{s}.npy", Praw.astype(np.float32))
            sc = score(fr, P)
            log(f"{v} seed {s} ({time.time()-t0:.0f}s): {json.dumps(sc)}")


# -------------------------------------------------------------------- scoring
def score(k: pd.DataFrame, P: np.ndarray) -> dict:
    pred = np.array(CLASSES5)[P.argmax(1)]
    lab = k.labelled.to_numpy()
    yt = k.func5.to_numpy()
    full = (k.wgroup == "full").to_numpy()
    apc = lab & np.isin(yt, A2.CLASSES3)
    oth = lab & (yt == "Other")
    ua = (~k.labelled & k.active).to_numpy()
    d = {"acc5": (yt[lab] == pred[lab]).mean(), "acc5_full": (yt[lab & full] ==
                                                              pred[lab & full]).mean(),
         "accAPC": (yt[apc] == pred[apc]).mean(),
         "other_R": (pred[oth] == "Other").mean(),
         "other_P": (yt[lab & (pred == "Other")] == "Other").mean(),
         "unl_pred_other": (pred[ua] == "Other").mean(),
         "unl_pred_other_full": (pred[ua & full] == "Other").mean(),
         "n_lab": int(lab.sum()), "n_unl": int(ua.sum())}
    return {x: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for x, v in d.items()}


def stage_report(a) -> None:
    k = pd.read_parquet(OUT / "keys.parquet")
    res = {}
    for v in ("a", "b", "b7", "c"):
        fs = sorted(OUT.glob(f"P_{v}_s*.npy"))
        if not fs:
            continue
        per = [score(k, np.load(f)) for f in fs]
        res[v] = {"seeds": len(fs), "per_seed": per,
                  "mean": {x: round(float(np.mean([p[x] for p in per])), 4)
                           for x in per[0]}}
    # b7: Mid and Bike as reported classes, precision / recall (seed-mean probabilities)
    fs = sorted(OUT.glob("Praw_b7_s*.npy"))
    if fs:
        C7 = np.array(REAL4 + ["Mid", "Bike", "Other"])
        P7 = np.mean([np.load(f) for f in fs], axis=0)
        p7 = C7[P7.argmax(1)]
        sub7 = k["sub"].map({"O_mid": "Mid", "O_bike": "Bike", "O_advpres": "Other",
                             "O_other": "Other"}).fillna(k["sub"]).to_numpy()
        labm = k.labelled.to_numpy()
        fullm = (k.wgroup == "full").to_numpy()
        ua7 = (~k.labelled & k.active).to_numpy()
        out = {}
        for tag, msk in (("allwin", labm), ("full", labm & fullm)):
            for c in ("Mid", "Bike"):
                tp = int(((sub7 == c) & (p7 == c) & msk).sum())
                out[f"{c}_{tag}"] = {
                    "n": int(((sub7 == c) & msk).sum()),
                    "precision": round(tp / max(int(((p7 == c) & msk).sum()), 1), 4),
                    "recall": round(tp / max(int(((sub7 == c) & msk).sum()), 1), 4)}
            out[f"acc7_{tag}"] = round(float((sub7[msk] == p7[msk]).mean()), 4)
        out["unl_pred7_share"] = pd.Series(p7[ua7]).value_counts(
            normalize=True).round(4).to_dict()
        res["b7_classes"] = out
    # what the baseline does with the unlabelled active channels (seed 0)
    P = np.load(OUT / "P_a_s0.npy")
    pred = np.array(CLASSES5)[P.argmax(1)]
    conf = P.max(1)
    ua = (~k.labelled & k.active).to_numpy()
    lab = k.labelled.to_numpy()
    full = (k.wgroup == "full").to_numpy()
    d = {}
    for tag, msk in (("unl_all", ua), ("unl_full", ua & full), ("lab_full", lab & full)):
        vc = pd.Series(pred[msk]).value_counts(normalize=True).round(4).to_dict()
        d[tag] = {"n": int(msk.sum()), "pred_share": vc,
                  "conf_median": round(float(np.median(conf[msk])), 3),
                  "conf_ge_080": round(float((conf[msk] >= 0.8).mean()), 3),
                  "conf_ge_080_nonOther": round(float(((conf[msk] >= 0.8) &
                                                       (pred[msk] != "Other")).mean()), 3)}
    # per-signal: share of predicted A/P/C/YR among unlabelled channels at the full window
    g = k[ua & full].assign(pred=pred[ua & full])
    d["unl_full_signals"] = int(g.DeviceId.nunique())
    d["unl_full_per_signal_nonOther_median"] = float(
        g.assign(no=g.pred != "Other").groupby(["DeviceId", "period"]).no.sum().median())
    cat = pd.read_parquet(OUT / "unl_cat.parquet")
    cat["Detector"] = cat.Detector.astype(k.Detector.dtype)
    g = g.merge(cat, on=["DeviceId", "Detector", "period"], how="left")
    d["unl_full_by_text"] = {c: {"n": int(len(x)), "pred_other": round(float(
        (x.pred == "Other").mean()), 3)} for c, x in g.groupby("cat")}
    res["baseline_on_unlabelled"] = d
    for v in ("b", "c"):
        fs = sorted(OUT.glob(f"P_{v}_s*.npy"))
        if fs:
            Pv = np.mean([np.load(f) for f in fs], axis=0)
            pv = np.array(CLASSES5)[Pv.argmax(1)][ua & full]
            gg = k[ua & full][["DeviceId", "Detector", "period"]].assign(pred=pv).merge(
                cat, on=["DeviceId", "Detector", "period"], how="left")
            res[f"{v}_unl_full_by_text"] = {c: round(float((x.pred == "Other").mean()), 3)
                                           for c, x in gg.groupby("cat")}
    # phase side: out-of-fold decoded phase vs official target, full window
    off = pd.read_parquet(OFFICIAL)
    off["DeviceId"] = off.DeviceId.str.lower()
    off = off[off.target_type == "phase"][["DeviceId", "Detector", "target_num",
                                           "switch_phase", "additional_call_phases"]]
    off["Detector"] = off.Detector.astype(k.Detector.dtype)
    kf = k[full & (ua | lab)].merge(off, on=["DeviceId", "Detector"], how="left")
    ph = {}
    for tag, msk in (("unl", ~kf.labelled.to_numpy()), ("lab", kf.labelled.to_numpy())):
        s = kf[msk]
        has = s.target_num.notna()
        ph[tag] = {"n": int(len(s)), "has_phase_target": round(float(has.mean()), 4),
                   "top1_vs_official": round(float(
                       (s[has].pred_phase == s[has].target_num).mean()), 4)}
    res["phase_full_window"] = ph
    json.dump(res, open(OUT / "report.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "exp", "report"])
    ap.add_argument("--variants", default="a,b,c")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
