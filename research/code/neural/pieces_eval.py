"""Stage 33: six-fold OOF blend accuracy when the GRU sees only K pieces of a long sample.

Same rows, labels, folds and 0.5 before-decoder blend as `trackb_eval_full.py` /
`final6_gru13.json`.  The stage-13 GRU OOF files are used unchanged for m5..h1; for the
long windows (h3, h6, h24, full) the network probability is replaced by a variant from
`pieces_infer.py` (`lp_k4` = mean log-prob over 4 evenly spaced 30-min pieces, ...).

Two decoder modes:
  frozen  the six OOF decoders are fitted ONCE on the reference inputs (stored stage-13
          GRU, all pieces) and applied to each variant -- what production would do
          (the decoder is not re-fitted when the runtime changes the piece count);
  refit   the decoder is re-fitted out-of-fold on the variant's own inputs.

    python research/code/neural/pieces_eval.py --out pieces
"""
from __future__ import annotations

import argparse
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
from common import DC_WORK, N_FOLDS  # noqa: E402
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
import blend_v2 as B  # noqa: E402
import blend_v2_predecode as BP  # noqa: E402
import trackb_eval as TE  # noqa: E402

PIECES = DC_WORK / "trackB" / "pieces"
EVALDIR = DC_WORK / "trackB" / "eval"
KEY, DET, GRP = B.KEY, B.DET, BP.GRP
LONGF = ("h3", "h6", "h24", "full")
COLS = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def score(q: pd.DataFrame, col: str) -> dict:
    """{fam: {acc, err, n}} pooled, plus per fold."""
    t = B.top1(q, col)
    t["fam"] = t.win.map(B.fam_of)
    out = {}
    for scope, g0 in [("all", t)] + [(int(k), t[t.fold == k]) for k in range(N_FOLDS)]:
        r = {}
        for fam in LONGF + ("m30",):
            g = g0[g0.fam == fam]
            if len(g):
                r[fam] = {"acc": round(float(g.groupby("win").ok.mean().mean()), 5),
                          "err": int((1 - g.ok).sum()), "n": int(len(g))}
        out[str(scope)] = r
    return out


class Harness:
    def __init__(self) -> None:
        self.ref_nn = B.gru_oof()
        self.ref_nn = TE._norm(self.ref_nn)
        self.keep = B.common_frame(self.ref_nn)[KEY].copy()
        lg = TE._norm(pd.read_parquet(B.LG_OOF))
        lg = lg.merge(B.folds(), on="DeviceId", how="inner")
        lab = B.labels()
        lg["dev_plain"] = lg.DeviceId.str.replace("@stg", "", regex=False)
        lg = lg.merge(lab.rename(columns={"DeviceId": "dev_plain"}),
                      on=["dev_plain", "Detector"], how="left")
        lg["y"] = np.where(lg.Phase.notna(), (lg.cand_phase == lg.Phase).astype(float),
                           np.nan)
        self.lg = lg
        self.meta = lg[KEY + ["Phase", "fold", "y"]]
        self.ctx, self.sim = BP.load_context(set(lg.DeviceId.unique()))
        self.models: dict = {}

    def X_for(self, nn: pd.DataFrame) -> pd.DataFrame:
        lg = self.lg.merge(nn[KEY + ["prob"]].rename(columns={"prob": "p_nn"}),
                           on=KEY, how="left")
        tot = lg.groupby(GRP)["p_nn"].transform("sum")
        lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
        pr = lg[KEY].copy()
        pr["p0"] = np.where(lg.p_nn.notna(), 0.5 * lg.p0 + 0.5 * lg.p_nn, lg.p0)
        pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
        X = dec.assemble(pr, pairs=self.ctx, sim=self.sim).merge(self.meta, on=KEY,
                                                                 how="left")
        return X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(
            drop=True)

    def fit(self, X: pd.DataFrame) -> dict:
        ms = {}
        lab = X.y.notna().to_numpy()
        for k in range(N_FOLDS):
            te = (X.fold == k).to_numpy()
            inner = (k + 1) % N_FOLDS
            base = (~te) & lab
            tr = X[base & (X.fold != inner).to_numpy()]
            va = X[base & (X.fold == inner).to_numpy()]
            P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0,
                     n_jobs=12)
            n = P.pop("n_estimators")
            m = lgb.LGBMClassifier(n_estimators=n, **P)
            m.fit(tr[COLS], tr.y.astype(int), eval_set=[(va[COLS], va.y.astype(int))],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
            ms[k] = m
        return ms

    def decode(self, X: pd.DataFrame, ms: dict) -> pd.DataFrame:
        s2 = np.zeros(len(X))
        for k, m in ms.items():
            te = (X.fold == k).to_numpy()
            s2[te] = m.predict_proba(X.loc[te, COLS])[:, 1]
        X = X.copy()
        X["p2"] = T.norm_prob(X, s2)
        q = self.keep.merge(X[KEY + ["p2"]], on=KEY, how="left").merge(
            self.lg[KEY + ["Phase", "fold", "prob"]], on=KEY, how="left")
        q["p2"] = q.p2.fillna(0.0)
        return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pieces")
    ap.add_argument("--refit", default="lp_k2,lp_k4,lp_k6,lp_all")
    a = ap.parse_args()
    H = Harness()
    pcs = TE._norm(pd.concat([pd.read_parquet(f) for f in sorted(PIECES.glob(
        "gru2_pieces_f*.parquet"))], ignore_index=True))
    variants = sorted(pcs.variant.unique())
    log(f"pieces: {len(pcs):,} rows, variants {variants}")
    res: dict = {"n_rows": int(len(H.keep)), "variants": {}}

    # reference = the stored stage-13 GRU (all pieces, max 32) -> final6_gru13 numbers
    Xr = H.X_for(H.ref_nn)
    H.models = H.fit(Xr)
    q = H.decode(Xr, H.models)
    res["trees"] = score(q, "prob")
    res["reference_stored"] = score(q, "p2")
    log("trees  " + json.dumps({f: v["acc"] for f, v in res["trees"]["all"].items()}))
    log("ref    " + json.dumps({f: v["acc"] for f, v in res["reference_stored"]["all"]
                                .items()}))
    json.dump(res, open(EVALDIR / f"{a.out}.json", "w"), indent=1)

    is_long = H.ref_nn.win.map(B.fam_of).isin(LONGF)
    short = H.ref_nn[~is_long][KEY + ["prob"]]
    refit = set(a.refit.split(","))
    for v in variants:
        t0 = time.time()
        pv = pcs[pcs.variant == v][KEY + ["prob"]]
        nn = pd.concat([short, pv], ignore_index=True)
        X = H.X_for(nn)
        e = {"frozen": score(H.decode(X, H.models), "p2")}
        if v in refit:
            e["refit"] = score(H.decode(X, H.fit(X)), "p2")
        res["variants"][v] = e
        log(f"{v:8s} frozen " + json.dumps({f: x["acc"] for f, x in
                                            e["frozen"]["all"].items()})
            + (" refit " + json.dumps({f: x["acc"] for f, x in e["refit"]["all"].items()})
               if "refit" in e else "") + f" ({time.time()-t0:.0f}s)")
        json.dump(res, open(EVALDIR / f"{a.out}.json", "w"), indent=1)
    log(f"wrote {EVALDIR / f'{a.out}.json'}")


if __name__ == "__main__":
    main()
