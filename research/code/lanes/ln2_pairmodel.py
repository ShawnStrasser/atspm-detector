"""Lane output (note 42), step 2 -- the same-lane pair model, out of fold (folds_v4, six folds).

Training / truth pairs: two print high-confidence vehicle-lane detectors on the same timing phase
of a kept phase (ln1_truth_*), both >= MIN_ON actuations in the window; same lane = lane intervals
[lane_index, lane_index + span - 1] overlap.  One model over all nine Sept-2026 windows
(win_hours is a feature).  LightGBM binary, 3 seeds, fixed 300 trees, 4 threads.
Writes ln2_pairs_oof.parquet (OOF P(same) for EVERY ln1 pair: seed mean), ln2_auc.json.
Controls: + noise column; labels shuffled within fold.  Ablations: - pair type, - context.

    python ln2_pairmodel.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import json
import os
import time
import numpy as np
import pandas as pd
import lane_output as LO
import ln1_cues as L1

OUT = L1.OUT
PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15,
              min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=4, verbose=-1)
WG = {"m30_a": "m30", "m30_b": "m30", "m30_c": "m30", "m30_d": "m30", "h6_a": "h6",
      "h6_b": "h6", "h24_a": "h24", "h24_b": "h24", "full66": "full"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def truth_pairs(det: pd.DataFrame, ph: pd.DataFrame) -> pd.DataFrame:
    h = det[det.high_veh & det.phase_kept][["DeviceId", "target", "det", "lane_index", "end",
                                            "span", "print_function"]]
    P = h.merge(h, on=["DeviceId", "target"], suffixes=("_a", "_b"))
    P = P[P.det_a < P.det_b]
    P["same_lane"] = ((P.lane_index_a <= P.end_b) & (P.lane_index_b <= P.end_a)).astype(int)
    P = P.merge(ph, on=["DeviceId", "target"])
    return P.rename(columns={"det_a": "da", "det_b": "db"})


def load():
    ph = pd.read_parquet(OUT / "ln1_truth_phase.parquet")
    det = pd.read_parquet(OUT / "ln1_truth_det.parquet")
    P = pd.read_parquet(OUT / "ln1_pairs.parquet")
    D = pd.read_parquet(OUT / "ln1_dets.parquet")
    fold = D.groupby("DeviceId").fold.first()
    P["fold"] = P.DeviceId.map(fold).astype(int)
    P["wg"] = P.win.map(WG)
    T = truth_pairs(det, ph)
    P = P.merge(T[["DeviceId", "da", "db", "same_lane", "n_lanes", "span_a", "span_b",
                   "print_function_a", "print_function_b"]],
                on=["DeviceId", "da", "db"], how="left")
    P["labelled"] = P.same_lane.notna() & P.same_true
    return P, T, D


def auc(y, s):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) < 20 or y.min() == y.max():
        return None
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    # ties: average ranks
    ss = s[o]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1 = y.sum()
    return round(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))), 4)


def fit(X, y, seed):
    import lightgbm as lgb
    return lgb.LGBMClassifier(random_state=seed, **PARAMS).fit(X, y)


def oof(P, lab, cols, seeds=(0, 1, 2), ycol="same_lane"):
    """OOF P(same) for every row of P, models fit on the labelled rows of the other folds."""
    out = np.zeros(len(P))
    for s in seeds:
        for f in range(6):
            tr = lab & (P.fold != f)
            te = (P.fold == f).to_numpy()
            m = fit(P.loc[tr, cols], P.loc[tr, ycol].astype(int), s)
            out[te] += m.predict_proba(P.loc[te, cols])[:, 1]
    return out / len(seeds)


def summary(P, lab, p):
    y = P.same_lane.to_numpy()
    r = {}
    for wg in ["m30", "h6", "h24", "full"]:
        m = (lab & (P.wg == wg)).to_numpy()
        mm = m & (P.n_lanes >= 2).to_numpy()
        r[wg] = {"n": int(m.sum()), "all": auc(y[m], p[m]), "multi": auc(y[mm], p[mm]),
                 "n_multi": int(mm.sum())}
    return r


def main():
    P, T, D = load()
    lab = P.labelled.to_numpy()
    res = {"pairs_total": int(len(P)), "labelled_pair_windows": int(lab.sum()),
           "truth_pairs": int(len(T)), "truth_same_share": round(float(T.same_lane.mean()), 4),
           "labelled_by_window": P[lab].groupby("wg").size().to_dict(),
           "signals": int(P.DeviceId.nunique())}
    log(json.dumps(res))
    feats = LO.FEATURES
    variants = {"full": feats,
                "no_pairtype": [c for c in feats if not c.startswith("pt_")],
                "cues_only": LO.CUE_COLS + ["win_hours"]}
    for name, cols in variants.items():
        t0 = time.time()
        p = oof(P, lab, cols)
        res[name] = summary(P, lab, p)
        log(f"{name}: {json.dumps(res[name])} ({time.time()-t0:.0f}s)")
        if name == "full":
            P["p_same"] = p
            # seed spread on the full set
            res["full_seed_multi"] = {}
            for s in (0, 1, 2):
                ps = oof(P, lab, cols, seeds=(s,))
                res["full_seed_multi"][s] = {k: v["multi"] for k, v in summary(P, lab, ps).items()}
            log(f"seeds {res['full_seed_multi']}")
    rng = np.random.default_rng(0)
    P["noise"] = rng.normal(size=len(P))
    res["ctrl_noise"] = summary(P, lab, oof(P, lab, feats + ["noise"], seeds=(0,)))
    Ps = P.copy()
    Ps["y_shuf"] = Ps.same_lane
    for f in range(6):
        m = lab & (Ps.fold == f).to_numpy()
        Ps.loc[m, "y_shuf"] = rng.permutation(Ps.loc[m, "same_lane"].to_numpy())
    res["ctrl_shuffled"] = summary(P, lab, oof(Ps, lab, feats, seeds=(0,), ycol="y_shuf"))
    log(f"controls noise {res['ctrl_noise']} shuffled {res['ctrl_shuffled']}")
    P.drop(columns=["noise"]).to_parquet(OUT / "ln2_pairs_oof.parquet", index=False)
    json.dump(res, open(OUT / "ln2_auc.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
