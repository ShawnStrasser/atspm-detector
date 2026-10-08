"""Lanes from cabinet prints (note 30), step 5 -- does the print-trained lane model help
FUNCTION?  Screen on the note-25 b7 out-of-fold predictions (w1/Praw_b7_s{0,1,2}.npy,
Sept-2026 full window), print-labelled signals only.

Truth for this screen: the v3 print label at high confidence (`source` print_high,
label_print_first) -- the function labels are being revalidated, so this is a screen, not a
headline.  Scored on complete_high signals (asked) and, for size, complete_mixed.
Mid / Bike -> Other for 5-class; core-four = rows whose truth is A / P / C / YR.

Lanes: P(same lane) = lp3's print-trained model, out-of-fold.  Pairs are the lp2 pairs
(same timing phase) restricted to detectors whose b7-frame predicted phase is that phase,
so the grouping is predicted-phase within timing phase.
  argmax      b7 seed-mean argmax (the baseline)
  decode      per-lane constraint decode (A3.assign, <= 1 A / P / C per lane) on lanes
              from average linkage at th, th in {.5, .6, .7}
  stack0      LightGBM on the 7 b7 probabilities only (stacking control)
  stack_lane  the same + lane features: n same-lane partners, sum P(same), phase n_lanes
              (largest pairwise-different set), lanes spanned, upstream score in its lane,
              lane-mates' b7 class mass (7), best hp_off partner
Stackers: out-of-fold over the six folds (folds_v3), trained on print_high rows at
complete_high + complete_mixed; one run per b7 seed (3 seeds).

    python lp5_function.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import itertools
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import lr_common as C
import lp1_labels as LP1
import lp4_nlanes as LP4
import a3_lanes as A3

W1 = C.WORK / "w1"
CL7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
CL5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CORE = CL5[:4]


def frame():
    k = pd.read_parquet(W1 / "keys.parquet")
    m = ((k.period == "stg") & (k.wgroup == "full")).to_numpy()
    K = k[m].reset_index(drop=True)
    K["DeviceId"] = K.DeviceId.str.lower()
    Praw = [np.load(W1 / f"Praw_b7_s{s}.npy")[m] for s in range(3)]
    v = pd.read_parquet(LP1.V3)
    v["DeviceId"] = v.DeviceId.str.lower()
    v = v[v.tier.isin(["complete_high", "complete_mixed"]) & ~v.unusual_layout
          & (v.source == "print_high")]
    v = v[["DeviceId", "detector", "phase_target", "label_print_first", "tier"]].rename(
        columns={"detector": "Detector", "label_print_first": "y7"})
    K["Detector"] = K.Detector.astype(int)
    K = K.reset_index().merge(v, on=["DeviceId", "Detector"], how="inner")
    assert not K.DeviceId.isin(C.locked_ids()).any()
    idx = K["index"].to_numpy()
    Praw = [p[idx] for p in Praw]
    K["y5"] = K.y7.replace({"Mid": "Other", "Bike": "Other"})
    K = K[K.y5.isin(CL5)].reset_index(drop=True)
    Praw = [p[K.index.to_numpy()] for p in Praw]
    return K, Praw


def lane_features(K, P7):
    """Per detector lane features from OOF P(same lane); P7 = b7 probs (for lane-mates)."""
    PR = pd.read_parquet(C.WORK / "lp3_pairs_oof.parquet")
    L2 = pd.read_parquet(C.WORK / "lp2_pairs.parquet",
                         columns=["DeviceId", "da", "db", "lead_asym_all", "hp_off"])
    PR = PR.merge(L2, on=["DeviceId", "da", "db"], how="left")
    lk = {(r.DeviceId, int(r.da), int(r.db)): (r.p_print, r.lead_asym_all, r.hp_off)
          for r in PR.itertuples(index=False)}
    F = np.full((len(K), 7 + 7), np.nan)
    lanes = np.zeros(len(K), int)
    K = K.copy()
    K["pp"] = "P" + K.pred_phase.astype(int).astype(str)
    grp = K[K.pp == K.phase_target].groupby(["DeviceId", "phase_target"]).indices
    for (dev, tg), rows in grp.items():
        dets = K.Detector.to_numpy()[rows]
        n = len(rows)
        S = np.eye(n)
        LA = np.zeros((n, n))
        HP = np.full((n, n), np.nan)
        for i, j in itertools.combinations(range(n), 2):
            a, b = (i, j) if dets[i] < dets[j] else (j, i)
            p, la, hp = lk.get((dev, int(dets[a]), int(dets[b])), (np.nan, np.nan, np.nan))
            p = 0.0 if np.isnan(p) else p
            S[i, j] = S[j, i] = p
            la = 0.0 if np.isnan(la) else la
            LA[a, b], LA[b, a] = la, -la        # >0: row detector is upstream of column
            HP[i, j] = HP[j, i] = hp
        lab = A3.cluster(np.arange(n), S, 0.6)
        lanes[rows] = lab
        nl = LP4.max_diff_set(S, 0.3)
        for ii, r in enumerate(rows):
            others = np.arange(n) != ii
            ps = S[ii, others]
            span = sum(S[ii, lab == L].mean() >= 0.5 for L in np.unique(lab[others])
                       if (lab[others] == L).any() and L != lab[ii]) + 1
            w = S[ii] * others
            mate = (w[:, None] * P7[rows]).sum(0) / max(w.sum(), 1e-9)
            F[r, :7] = [float((ps >= 0.5).sum()), float(ps.sum()) if n > 1 else 0.0, nl,
                        span, float((w * LA[ii]).sum() / max(w.sum(), 1e-9)),
                        float(np.nanmax(HP[ii])) if n > 1 and np.isfinite(HP[ii]).any()
                        else np.nan, n]
            F[r, 7:] = mate
    names = ["n_same", "sum_psame", "phase_nlanes", "span_k", "upstream", "best_hp_off",
             "n_det_phase"] + [f"mate_{c}" for c in CL7]
    return pd.DataFrame(F, columns=names), lanes


def stack(K, X, y, seed):
    P = np.zeros((len(K), len(CL7)))
    yi = pd.Series(y).map({c: i for i, c in enumerate(CL7)}).to_numpy()
    prm = dict(objective="multiclass", num_class=len(CL7), learning_rate=0.05, num_leaves=15,
               min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=1.0, seed=seed, num_threads=4, verbose=-1)
    for f in sorted(K.fold.unique()):
        te = (K.fold == f).to_numpy()
        m = lgb.train(prm, lgb.Dataset(X[~te], yi[~te]), num_boost_round=200)
        P[te] = m.predict(X[te])
    return P


def to5(P7):
    return np.column_stack([P7[:, :4], P7[:, 4:].sum(1)])


def score(K, pred5, mask):
    y = K.y5.to_numpy()
    p = np.array(CL5)[pred5]
    core = mask & np.isin(y, CORE)
    return {"acc5": round(float((p[mask] == y[mask]).mean()), 4),
            "core4": round(float((p[core] == y[core]).mean()), 4)}


def main():
    K, Praw = frame()
    sets = {"complete_high": (K.tier == "complete_high").to_numpy(),
            "complete_mixed": (K.tier == "complete_mixed").to_numpy(),
            "both": np.ones(len(K), bool)}
    res = {"n_rows": {k: int(m.sum()) for k, m in sets.items()},
           "n_signals": {k: int(K[m].DeviceId.nunique()) for k, m in sets.items()},
           "pred_phase_eq_timing": round(float(("P" + K.pred_phase.astype(int).astype(str)
                                               == K.phase_target).mean()), 4)}
    C.log(json.dumps(res))
    out = {}
    for s in range(3):
        P7 = Praw[s]
        F, _ = lane_features(K, P7)
        r = {"argmax": {k: score(K, to5(P7).argmax(1), m) for k, m in sets.items()}}
        P5 = to5(P7)
        for th in (0.5, 0.6, 0.7):
            pred = P5.argmax(1).copy()
            Kp = K.assign(pp="P" + K.pred_phase.astype(int).astype(str))
            _, lanes = _lanes_at(K, th)
            for key, rows in Kp[Kp.pp == Kp.phase_target].groupby(
                    ["DeviceId", "phase_target"]).indices.items():
                if len(rows) > 1:
                    pred[rows] = A3.assign(P5[rows], lanes[rows], "lane")
            r[f"decode_th{th}"] = {k: score(K, pred, m) for k, m in sets.items()}
        X0 = P7
        X1 = np.column_stack([P7, F.to_numpy()])
        for name, X in (("stack0", X0), ("stack_lane", X1)):
            Ps = stack(K, X, K.y7.to_numpy(), seed=s)
            r[name] = {k: score(K, to5(Ps).argmax(1), m) for k, m in sets.items()}
        # noise control on the lane stacker: shuffle lane features within fold
        Fs = F.copy()
        rng = np.random.default_rng(s)
        for f in K.fold.unique():
            ii = np.flatnonzero((K.fold == f).to_numpy())
            Fs.iloc[ii] = Fs.iloc[rng.permutation(ii)].to_numpy()
        Ps = stack(K, np.column_stack([P7, Fs.to_numpy()]), K.y7.to_numpy(), seed=s)
        r["stack_lane_shuffled"] = {k: score(K, to5(Ps).argmax(1), m) for k, m in sets.items()}
        out[f"seed{s}"] = r
        C.log(f"seed {s}: {json.dumps(r)}")
    # seed means
    mean = {}
    for v in out["seed0"]:
        mean[v] = {k: {mt: round(float(np.mean([out[f'seed{s}'][v][k][mt] for s in range(3)])), 4)
                       for mt in ("acc5", "core4")} for k in sets}
    res["by_seed"] = out
    res["seed_mean"] = mean
    C.log(json.dumps(mean, indent=1))
    json.dump(res, open(C.WORK / "lp5_function.json", "w"), indent=1, default=str)


_LK = None


def _lanes_at(K, th):
    """Lane ids (average linkage at th) on predicted-phase-within-timing-phase groups."""
    global _LK
    if _LK is None:
        PR = pd.read_parquet(C.WORK / "lp3_pairs_oof.parquet")
        _LK = {(r.DeviceId, int(r.da), int(r.db)): r.p_print for r in PR.itertuples(index=False)}
    lanes = np.zeros(len(K), int)
    pp = "P" + K.pred_phase.astype(int).astype(str)
    for (dev, tg), rows in K[pp == K.phase_target].groupby(["DeviceId", "phase_target"]).indices.items():
        dets = K.Detector.to_numpy()[rows]
        n = len(rows)
        S = np.eye(n)
        for i, j in itertools.combinations(range(n), 2):
            p = _LK.get((dev, int(min(dets[i], dets[j])), int(max(dets[i], dets[j]))), np.nan)
            S[i, j] = S[j, i] = 0.0 if np.isnan(p) else p
        lanes[rows] = A3.cluster(np.arange(n), S, th)
    return None, lanes


if __name__ == "__main__":
    main()
