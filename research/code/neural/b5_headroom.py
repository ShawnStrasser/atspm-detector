"""Track B, B5 pre-check: where does the fold-0 blend (before the decoder) still fail, and
is the network alone right on any of those rows?  Reads existing predictions only.

For each TCN seed (and the stage-13 GRU for reference) the shipped arrangement is rebuilt
exactly as `trackb_eval.Frame.before` does, but the per-row decoded probabilities are
kept, so the blend's errors can be crossed with the net alone, the trees alone (decoded
LightGBM) and the ranker alone.  Also: for each error, are there other detectors on the
same signal-window with the same TRUE phase that the blend gets right (evaluation-only use
of labels -- the "sibling could have told it" count).

    python research/code/neural/b5_headroom.py                      # the baselines
    python research/code/neural/b5_headroom.py --out b5 b5a=tb_b5a_f0_oof b5b=...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import CONCURRENT_PAIRS, DC_WORK  # noqa: E402
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
import blend_v2 as B  # noqa: E402
import trackb_eval as TE  # noqa: E402

KEY, DET = B.KEY, B.DET
OUT = DC_WORK / "trackB" / "b5"
FAMS = ("m5", "m10", "m30", "h1")


def decoded(fr: TE.Frame, cand: pd.DataFrame, w: float = 0.5) -> pd.DataFrame:
    """`Frame.before` returning the fold-0 rows with p2 instead of accuracies."""
    nn = pd.concat([fr.nn_rest, cand[KEY + ["prob"]]], ignore_index=True)
    nn = nn.rename(columns={"prob": "p_nn"})
    lg = fr.lg.merge(nn, on=KEY, how="left")
    tot = lg.groupby(TE.GRP)["p_nn"].transform("sum")
    lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
    pr = lg[KEY].copy()
    pr["p0"] = np.where(lg.p_nn.notna(), w * lg.p0 + (1.0 - w) * lg.p_nn, lg.p0)
    pr["p0"] = pr.p0 / pr.groupby(TE.GRP)["p0"].transform("sum")
    X = dec.assemble(pr, pairs=fr.ctx, sim=fr.sim).merge(fr.meta, on=KEY, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    te = (X.fold == 0).to_numpy()
    lab = X.y.notna().to_numpy()
    base = (~te) & lab
    tr = X[base & (X.fold != 1).to_numpy()]
    va = X[base & (X.fold == 1).to_numpy()]
    P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0, n_jobs=8)
    n = P.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n, **P)
    m.fit(tr[cols], tr.y.astype(int), eval_set=[(va[cols], va.y.astype(int))],
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    s2 = np.zeros(len(X))
    s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
    X["p2"] = T.norm_prob(X, s2)
    return fr.keep.merge(X[KEY + ["p2"]], on=KEY, how="left").fillna({"p2": 0.0})


def argmax_of(df: pd.DataFrame, col: str) -> pd.Series:
    t = B.top1(df, col)
    return t.set_index(DET)["cand_phase"].rename(col)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="headroom")
    ap.add_argument("cands", nargs="*", help="name=prediction file")
    opt = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fr = TE.Frame()
    f0 = fr.f0.drop(columns=["p_nn"])
    # trees alone after the decoder = LightGBM decoded prob (p_lg); ranker = p0
    res = {}
    cands = {"tcn_s0": "tb_tcn_f0_oof", "tcn_s1": "tb_b2s1_f0_oof",
             "tcn_s2": "tb_b2s2_f0_oof", "gru13": "gru2_oof_f0_bywindow.parquet"}
    if opt.cands:
        cands = dict(c.split("=", 1) for c in opt.cands)
    per_det = None
    for name, path in cands.items():
        cand = TE.read_pred(path)
        m = f0.merge(cand[KEY + ["prob"]].rename(columns={"prob": "p_nn"}), on=KEY,
                     how="inner")
        m["p_nn"] = m.p_nn / m.groupby(DET)["p_nn"].transform("sum")
        q = decoded(fr, cand)
        m = m.merge(q[KEY + ["p2"]], on=KEY, how="left")
        t = B.top1(m, "p2")[DET + ["Phase", "cand_phase", "fam", "ok"]].rename(
            columns={"cand_phase": "pred_blend", "ok": "ok_blend"})
        for col, tag in (("p_nn", "net"), ("p_lg", "trees"), ("p0", "ranker")):
            a = argmax_of(m, col).rename(f"pred_{tag}").reset_index()
            t = t.merge(a, on=DET, how="left")
            t[f"ok_{tag}"] = (t[f"pred_{tag}"] == t.Phase).astype(int)
        t["conc"] = [frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                     for a, b in zip(t.Phase, t.pred_blend)]
        # sibling test (labels used for evaluation only): same window, same true phase,
        # a different detector the blend gets right
        g = t.groupby(["DeviceId", "win", "Phase"]).agg(n_sib=("ok_blend", "size"),
                                                         n_sib_ok=("ok_blend", "sum"))
        t = t.merge(g.reset_index(), on=["DeviceId", "win", "Phase"], how="left")
        t["sib_ok"] = (t.n_sib_ok - t.ok_blend) > 0
        # does the blend's WRONG phase have a correctly-labelled sibling on this window?
        r = {}
        for fam in FAMS:
            s = t[t.fam == fam]
            e = s[s.ok_blend == 0]
            nw = s.win.nunique()
            r[fam] = dict(
                acc_blend=round(float(s.groupby("win").ok_blend.mean().mean()), 5),
                n=int(len(s)), errors=int(len(e)), per_window=round(len(e) / nw, 1),
                concurrent=int(e.conc.sum()),
                net_right=int(e.ok_net.sum()), trees_right=int(e.ok_trees.sum()),
                ranker_right=int(e.ok_ranker.sum()),
                net_right_conc=int(e[e.conc].ok_net.sum()),
                net_only_right=int(((e.ok_net == 1) & (e.ok_trees == 0)).sum()),
                any_right=int(((e.ok_net + e.ok_trees + e.ok_ranker) > 0).sum()),
                with_correct_sibling=int(e.sib_ok.sum()),
                conc_with_correct_sibling=int(e[e.conc].sib_ok.sum()),
                no_sibling=int((e.n_sib == 1).sum()),
                # the other direction: rows the blend gets right but the net alone gets wrong
                blend_right_net_wrong=int(((s.ok_blend == 1) & (s.ok_net == 0)).sum()),
                net_err=int((s.ok_net == 0).sum()),
                net_err_conc=int(sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                                     for a, b in zip(s[s.ok_net == 0].Phase,
                                                     s[s.ok_net == 0].pred_net))),
            )
        res[name] = r
        print(name, json.dumps(r["m30"]), flush=True)
        t["cand"] = name
        per_det = t if per_det is None else pd.concat([per_det, t], ignore_index=True)
    per_det.to_parquet(OUT / f"{opt.out}_rows.parquet", index=False)
    json.dump(res, open(OUT / f"{opt.out}.json", "w"), indent=1)
    if opt.cands:
        return
    # errors shared by all three TCN seeds (the stable core)
    tc = per_det[per_det.cand.str.startswith("tcn")]
    core = tc.pivot_table(index=DET + ["fam", "Phase"], columns="cand", values="ok_blend")
    for fam in FAMS:
        c = core.xs(fam, level="fam")
        print(fam, "errors in all 3 seeds", int((c.sum(1) == 0).sum()),
              "in any seed", int((c.min(1) == 0).sum()), flush=True)


if __name__ == "__main__":
    main()
