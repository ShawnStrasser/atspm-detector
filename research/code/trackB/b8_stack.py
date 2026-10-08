"""Track B step B8: stack LightGBM and the network with a small learner -- ONE test.

Inputs are out-of-fold predictions that already exist for all six folds: the shipped trees
(`final_v1` OOF: ranker p0 and decoded prob) and the stage-13 GRU (seed 0, `gru2_oof_f*`).
Per (detector, window, candidate) the stacker sees
    p_tree, p_net, their ranks among the detector's candidates, the top probability of each
    model on that detector, n candidates, log window hours, log(1 + actuations)
and predicts "this candidate is the phase"; scores are renormalised over the candidates.
Fixed, untuned learner: LightGBM binary, 15 leaves, lr 0.05, min_child_samples 200, no bagging,
early stopping (100) on fold 1, trained on folds 2-5.  **No fold-0 row is ever in training**;
folds are signal-grouped.  Two places, each against the fixed 0.5 average:
  after   stack(decoded trees, GRU)             vs 0.5 * decoded trees + 0.5 * GRU
  before  stack(ranker p0, GRU) -> joint decoder vs 0.5 * p0 + 0.5 * GRU -> decoder (shipped)
For `before` the decoder is trained on folds 2-5, so their stacked inputs must be out-of-fold
too: fold k in 1..5 is predicted by a stacker trained on the other four of folds 1-5 (same
tree count as the fold-0 stacker).

    python research/code/trackB/b8_stack.py      ->  dc_work/trackB/b8/b8.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import blend_v2 as B  # noqa: E402
import trackb_eval as TE  # noqa: E402
import b6_blend3 as B6  # noqa: E402

KEY, DET, GRP = B.KEY, B.DET, B6.GRP
OUT = DC_WORK / "trackB" / "b8"
FEATS = ["p_t", "p_n", "r_t", "r_n", "max_t", "max_n", "n_cand", "log_win_hours", "log_act"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=200,
              lambda_l2=1.0, feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
              n_jobs=B6.NJ, seed=0, verbose=-1)
log = TE.log


def features(d: pd.DataFrame, ctx_det: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(DET, sort=False)
    d["r_t"] = g.p_t.rank(ascending=False, method="first")
    d["r_n"] = g.p_n.rank(ascending=False, method="first")
    d["max_t"] = g.p_t.transform("max")
    d["max_n"] = g.p_n.transform("max")
    d["n_cand"] = g.p_t.transform("size")
    d = d.merge(ctx_det, on=DET, how="left")
    d["log_act"] = np.log1p(d.n_act.astype(float))
    return d


def fit(tr: pd.DataFrame, va: pd.DataFrame | None, n: int | None = None):
    m = lgb.LGBMClassifier(n_estimators=n or 2000, **PARAMS)
    if va is None:
        m.fit(tr[FEATS], tr.y.astype(int))
    else:
        m.fit(tr[FEATS], tr.y.astype(int), eval_set=[(va[FEATS], va.y.astype(int))],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return m


def renorm(d: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    s = pd.Series(np.clip(s, 1e-9, None), index=d.index)
    return (s / s.groupby([d[c] for c in DET]).transform("sum")).to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fr = TE.Frame()
    ctx_det = fr.ctx.groupby(DET, as_index=False).log_win_hours.first()
    res = {"features": FEATS, "params": PARAMS}

    # ------------------------------------------------ after the decoder (scorable rows)
    a = fr.base.rename(columns={"p_lg": "p_t", "p_nn": "p_n"}).copy()
    a["y"] = (a.cand_phase == a.Phase).astype(int)
    a = features(a, ctx_det)
    tr, va, te = a[a.fold.between(2, 5)], a[a.fold == 1], a[a.fold == 0].copy()
    m = fit(tr, va)
    te["p_s"] = renorm(te, m.predict_proba(te[FEATS])[:, 1])
    te["p_b"] = 0.5 * te.p_t + 0.5 * te.p_n
    res["after"] = {"trees": m.best_iteration_,
                    "fixed_avg": TE.flat(B.acc_by_fam(te, "p_b")),
                    "stack": TE.flat(B.acc_by_fam(te, "p_s")),
                    "importance_gain": dict(zip(FEATS, np.round(
                        m.booster_.feature_importance("gain") /
                        m.booster_.feature_importance("gain").sum(), 3).tolist()))}
    log(f"after: {m.best_iteration_} trees  avg {res['after']['fixed_avg']}")
    log(f"after: stack {res['after']['stack']}")
    json.dump(res, open(OUT / "b8.json", "w"), indent=1, default=str)

    # ------------------------------------------------ before the decoder (all rows)
    lg = fr.lg.merge(B6.TE._norm(fr.gru_all)[KEY + ["prob", "n_act"]]
                     .rename(columns={"prob": "p_n"}), on=KEY, how="left")
    tot = lg.groupby(GRP)["p_n"].transform("sum")
    lg["p_n"] = np.where(tot > 0, lg.p_n / tot.replace(0, np.nan), np.nan)
    lg["p_t"] = lg.p0
    has = lg.p_n.notna()
    b = features(lg[has].copy(), ctx_det)
    lab = b.y.notna()
    trb = b[lab & b.fold.between(2, 5)]
    vab = b[lab & (b.fold == 1)]
    mb = fit(trb, vab)
    n = mb.best_iteration_
    s = np.full(len(b), np.nan)
    f0 = (b.fold == 0).to_numpy()
    s[f0] = mb.predict_proba(b.loc[f0, FEATS])[:, 1]
    for k in range(1, 6):
        trk = b[lab & b.fold.between(1, 5) & (b.fold != k)]
        mk = fit(trk, None, n)
        ik = (b.fold == k).to_numpy()
        s[ik] = mk.predict_proba(b.loc[ik, FEATS])[:, 1]
        log(f"  before: cross-fit fold {k} ({len(trk):,} training rows)")
    b["p_s"] = renorm(b, s)
    pr = lg[KEY + ["p0"]].merge(b[KEY + ["p_s"]], on=KEY, how="left")
    pr["p0"] = np.where(pr.p_s.notna(), pr.p_s, pr.p0)
    pr = pr[KEY + ["p0"]]
    pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
    D = B6.Decoder(fr, p0_all=pr)
    q = D.score_fold0(pr[pr.DeviceId.isin(set(fr.keep.DeviceId))])
    q["fam"] = q.win.map(B.fam_of)
    D0 = B6.Decoder(fr)  # the shipped 0.5 mixture with the stage-13 GRU, same code path
    g0 = fr.lg.loc[fr.lg.fold == 0, KEY].merge(fr.gru_all.pipe(TE._norm)[KEY + ["prob"]],
                                                on=KEY, how="left")
    t0g = g0.groupby(GRP)["prob"].transform("sum")
    g0["prob"] = np.where(t0g > 0, g0.prob / t0g.replace(0, np.nan), np.nan)
    q0 = D0.score_fold0(B6.mix_fold0(fr, {"gru0": g0}, 0.5, {"gru0": 0.5}))
    q0["fam"] = q0.win.map(B.fam_of)
    res["before"] = {"stacker_trees": n, "decoder_trees": D.m.best_iteration_,
                     "fixed_avg": TE.flat(B.acc_by_fam(q0, "p2")),
                     "stack": TE.flat(B.acc_by_fam(q, "p2")),
                     "diag_m30_fixed": B.diagnostics(q0, "p2", "m30"),
                     "diag_m30_stack": B.diagnostics(q, "p2", "m30"),
                     "importance_gain": dict(zip(FEATS, np.round(
                         mb.booster_.feature_importance("gain") /
                         mb.booster_.feature_importance("gain").sum(), 3).tolist()))}
    log(f"before: avg   {res['before']['fixed_avg']}")
    log(f"before: stack {res['before']['stack']}")
    json.dump(res, open(OUT / "b8.json", "w"), indent=1, default=str)
    log(f"wrote {OUT/'b8.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
