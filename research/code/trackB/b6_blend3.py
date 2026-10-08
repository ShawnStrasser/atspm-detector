"""Track B step B6: trees + GRU + TCN, a three-way blend, from predictions that already exist.

Fold 0 only (the TCN has out-of-fold predictions for fold 0 alone), on exactly the stage-13
rows of `research/code/neural/trackb_eval.py`.  Weights are FIXED before looking:
    w3  = 0.50 trees / 0.25 GRU / 0.25 TCN      (primary)
    w13 = 1/3 each                               (the one alternative)
and every TCN seed on disk is swapped in (seed 0, 1, 2), so a gain has to survive a seed swap.
Controls with the same 0.5/0.25/0.25 shape: two GRU seeds, two TCN seeds -- is a gain from
architecture diversity or from averaging any two nets?

Blend BEFORE the decoder (shipped arrangement): folds 1-5 keep the stage-13 GRU mixture, so the
decoder is fitted ONCE (train folds 2-5, stop on 1, n_jobs 6) and only fold 0's inputs change;
`build_second_stage` works within (signal, window), so fold 0 is re-assembled on its own.
The GRU-seed-0 candidate must reproduce the stage-13 `_fold0` entry (.97449 at 30 min).

    python research/code/trackB/b6_blend3.py
Writes dc_work/trackB/b6/b6.json.
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
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
import blend_v2 as B  # noqa: E402
import trackb_eval as TE  # noqa: E402

KEY, DET, GRP = B.KEY, B.DET, ["DeviceId", "Detector", "win"]
OUT = DC_WORK / "trackB" / "b6"
FAMS = TE.FAMS
NJ = 6
NETS = {"gru0": "gru2_oof_f0_bywindow.parquet", "gru1": "gru2_oof_f0_s1_bywindow.parquet",
        "gru2": "gru2_oof_f0_s2_bywindow.parquet", "tcn0": "tb_tcn_f0_oof",
        "tcn1": "tb_b2s1_f0_oof", "tcn2": "tb_b2s2_f0_oof", "b3a": "tb_b3a_f0_oof",
        "b3b": "tb_b3b_f0_oof"}
log = TE.log


def load_nets(fr: TE.Frame) -> dict[str, pd.DataFrame]:
    """Each net's fold-0 probability, renormalised over the tree frame's candidates."""
    f0keys = fr.lg.loc[fr.lg.fold == 0, KEY]
    out = {}
    for k, f in NETS.items():
        d = f0keys.merge(TE.read_pred(f)[KEY + ["prob"]], on=KEY, how="left")
        tot = d.groupby(GRP)["prob"].transform("sum")
        d["prob"] = np.where(tot > 0, d.prob / tot.replace(0, np.nan), np.nan)
        out[k] = d
    return out


class Decoder:
    """The stage-13 decoder, fitted once on folds 1-5 (stage-13 GRU mixture there)."""

    COLS = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS

    def __init__(self, fr: TE.Frame, p0_all: pd.DataFrame | None = None) -> None:
        self.fr = fr
        pr = p0_all if p0_all is not None else self.stage13_mix()
        X = dec.assemble(pr, pairs=fr.ctx, sim=fr.sim).merge(fr.meta, on=KEY, how="left")
        X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
        base = (X.fold != 0).to_numpy() & X.y.notna().to_numpy()
        tr = X[base & (X.fold != 1).to_numpy()]
        va = X[base & (X.fold == 1).to_numpy()]
        P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0, n_jobs=NJ)
        n = P.pop("n_estimators")
        self.m = lgb.LGBMClassifier(n_estimators=n, **P)
        self.m.fit(tr[self.COLS], tr.y.astype(int),
                   eval_set=[(va[self.COLS], va.y.astype(int))], eval_metric="binary_logloss",
                   callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        log(f"    decoder fitted: {self.m.best_iteration_} trees on {len(tr):,} rows")

    def stage13_mix(self) -> pd.DataFrame:
        lg = self.fr.lg.merge(self.fr.nn_rest.rename(columns={"prob": "p_nn"}), on=KEY,
                              how="left")
        tot = lg.groupby(GRP)["p_nn"].transform("sum")
        lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
        pr = lg[KEY].copy()
        pr["p0"] = np.where(lg.p_nn.notna(), 0.5 * lg.p0 + 0.5 * lg.p_nn, lg.p0)
        pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
        return pr

    def score_fold0(self, p0_f0: pd.DataFrame) -> dict:
        """p0_f0: KEY + p0 for every fold-0 tree-frame row (labelled or not)."""
        X = dec.assemble(p0_f0, pairs=self.fr.ctx, sim=self.fr.sim)
        X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
        s = self.m.predict_proba(X[self.COLS])[:, 1]
        X["p2"] = T.norm_prob(X, s)
        q = self.fr.keep.merge(X[KEY + ["p2"]], on=KEY, how="left").merge(
            self.fr.lg[KEY + ["Phase"]], on=KEY, how="left")
        q["p2"] = q.p2.fillna(0.0)
        return q


def mix_fold0(fr: TE.Frame, nets: dict, w_tree: float, w: dict[str, float]) -> pd.DataFrame:
    """w_tree * ranker p0 + sum w_i * net_i, over the nets present on a row, renormalised."""
    lg = fr.lg[fr.lg.fold == 0][KEY + ["p0"]].reset_index(drop=True)
    num = w_tree * lg.p0.to_numpy()
    den = np.full(len(lg), w_tree)
    for k, wk in w.items():
        p = nets[k].prob.to_numpy()
        ok = ~np.isnan(p)
        num = num + np.where(ok, wk * np.nan_to_num(p), 0.0)
        den = den + np.where(ok, wk, 0.0)
    pr = lg[KEY].copy()
    pr["p0"] = num / den
    pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
    return pr


def after_mix(fr: TE.Frame, nets: dict, w_tree: float, w: dict[str, float]) -> pd.DataFrame:
    m = fr.f0[KEY + ["p_lg", "Phase", "fam"]].copy()
    num = w_tree * m.p_lg.to_numpy()
    for k, wk in w.items():
        p = m[KEY].merge(nets[k], on=KEY, how="left").prob.fillna(0.0).to_numpy()
        p = p / np.maximum(pd.Series(p).groupby([m[c] for c in DET]).transform("sum")
                           .to_numpy(), 1e-12)
        num = num + wk * p
    m["p_b"] = num
    return m


def f0_net(fr: TE.Frame, nets: dict, k: str) -> pd.DataFrame:
    m = fr.f0[KEY + ["Phase", "fam"]].merge(nets[k], on=KEY, how="left")
    m["prob"] = m.prob.fillna(0.0)
    m["prob"] = m.prob / m.groupby(DET)["prob"].transform("sum").replace(0, 1)
    return m


def diversity(fr: TE.Frame, nets: dict) -> dict:
    """Correlation of top-1 errors between two models on the same detector-windows."""
    ok = {}
    for k in ("gru0", "gru1", "gru2", "tcn0", "tcn1", "tcn2", "b3b"):
        t = B.top1(f0_net(fr, nets, k), "prob").set_index(DET)
        ok[k] = t[["ok", "fam"]]
    t = B.top1(fr.f0, "p_lg").set_index(DET)
    ok["trees"] = t[["ok", "fam"]]
    pairs = [("gru0", "tcn0"), ("gru0", "tcn1"), ("gru0", "tcn2"), ("gru1", "tcn0"),
             ("tcn0", "tcn1"), ("tcn0", "tcn2"), ("tcn1", "tcn2"),
             ("gru0", "gru1"), ("gru0", "gru2"), ("gru1", "gru2"),
             ("trees", "gru0"), ("trees", "tcn0"), ("trees", "tcn1")]
    res = {}
    for fam in ("m5", "m10", "m30", "h1"):
        r = {}
        for a, b in pairs:
            ea = 1 - ok[a].loc[ok[a].fam == fam, "ok"]
            eb = 1 - ok[b].loc[ea.index, "ok"]
            phi = float(np.corrcoef(ea, eb)[0, 1])
            both, either = int(((ea == 1) & (eb == 1)).sum()), int(((ea == 1) | (eb == 1)).sum())
            r[f"{a}~{b}"] = {"phi": round(phi, 3), "err_a": int(ea.sum()),
                             "err_b": int(eb.sum()), "both": both,
                             "jaccard": round(both / max(either, 1), 3)}
        res[fam] = r
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fr = TE.Frame()
    nets = load_nets(fr)
    for k, d in nets.items():
        log(f"  {k}: covers {d.prob.notna().mean():.1%} of fold-0 tree rows")
    D = Decoder(fr)
    res = {"n_rows_fold0": int(len(fr.keep)), "trees_fold0": TE.flat(B.acc_by_fam(fr.f0, "p_lg")),
           "net_alone": {}, "before": {}, "after": {}}
    for k in NETS:
        res["net_alone"][k] = TE.flat(B.acc_by_fam(f0_net(fr, nets, k), "prob"))
    cands = {"gru0_2way": (0.5, {"gru0": 0.5})}
    for s in ("tcn0", "tcn1", "tcn2"):
        cands[f"{s}_2way"] = (0.5, {s: 0.5})
    for s in ("tcn0", "tcn1", "tcn2"):
        cands[f"w3_gru0_{s}"] = (0.5, {"gru0": 0.25, s: 0.25})
        cands[f"w13_gru0_{s}"] = (1 / 3, {"gru0": 1 / 3, s: 1 / 3})
    cands["ctl_w3_gru0_gru1"] = (0.5, {"gru0": 0.25, "gru1": 0.25})
    cands["ctl_w3_gru0_gru2"] = (0.5, {"gru0": 0.25, "gru2": 0.25})
    cands["ctl_w3_tcn0_tcn1"] = (0.5, {"tcn0": 0.25, "tcn1": 0.25})
    cands["ctl_w3_tcn0_tcn2"] = (0.5, {"tcn0": 0.25, "tcn2": 0.25})
    cands["xtra_w3_gru0_b3b"] = (0.5, {"gru0": 0.25, "b3b": 0.25})
    for name, (wt, w) in cands.items():
        q = D.score_fold0(mix_fold0(fr, nets, wt, w))
        q["fam"] = q.win.map(B.fam_of)
        res["before"][name] = TE.flat(B.acc_by_fam(q, "p2"))
        res["after"][name] = TE.flat(B.acc_by_fam(after_mix(fr, nets, wt, w), "p_b"))
        if name in ("gru0_2way", "w3_gru0_tcn0", "w3_gru0_tcn1", "w3_gru0_tcn2"):
            res.setdefault("diag_m30_before", {})[name] = B.diagnostics(q, "p2", "m30")
        log(f"{name:20s} before {res['before'][name]}")
        log(f"{'':20s} after  {res['after'][name]}")
        json.dump(res, open(OUT / "b6.json", "w"), indent=1, default=str)
    res["diversity"] = diversity(fr, nets)
    res["check_gru0_before_m30_vs_stage13"] = [res["before"]["gru0_2way"]["m30"], 0.97449]
    json.dump(res, open(OUT / "b6.json", "w"), indent=1, default=str)
    log(f"wrote {OUT/'b6.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
