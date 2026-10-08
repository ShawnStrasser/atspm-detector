"""Track B step B9: score ranker variants on fold 0, on exactly stage 13's rows.

Reuses `research/code/neural/trackb_eval.py` (same rows, same stage-13 GRU, same decoder
recipe: trained on folds 2-5, early-stopped on fold 1, scored on fold 0).  A variant
replaces the ranker's first-stage probability `p0` by a refitted one for every fold it
has a file for (b9_fit.py); the decoder is then refitted on those inputs.  Columns:

  ranker        p0 alone
  trees         p0 -> joint decoder                    (no network)
  blend_raw     0.5 p0 + 0.5 GRU, not decoded
  blend_before  0.5 p0 + 0.5 GRU -> joint decoder      (the shipped arrangement)
  blend_after   0.5 x trees + 0.5 x GRU

"stored" = the shipped 3-seed ranker bag (the stage-13 reference; its blend_before
reproduces `blend_before.json`'s `_fold0`).

    python research/code/trackB/b9_eval.py --variants stored,base_s0,nb_s0 --out b9_f0
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
from common import CONCURRENT_PAIRS, DC_WORK  # noqa: E402
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
import blend_v2 as B  # noqa: E402
import trackb_eval as E  # noqa: E402

B9 = DC_WORK / "trackB" / "b9"
KEY, DET, GRP = B.KEY, B.DET, E.GRP
FAMS = E.FAMS


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_variant(name: str) -> pd.DataFrame | None:
    if name == "stored":
        return None
    fs = sorted(B9.glob(f"p0_{name}_f*.parquet"))
    if not fs:
        raise SystemExit(f"no p0 files for {name}")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    log(f"{name}: {len(fs)} fold files, {len(d):,} rows")
    return E._norm(d)


def decode(fr: E.Frame, p0: pd.Series, p_nn: pd.Series | None, w: float) -> pd.DataFrame:
    """Decoder (stage-13 recipe) on w*p0 + (1-w)*nn; returns fold-0 keep rows with p2."""
    lg = fr.lg
    pr = lg[KEY].copy()
    if p_nn is None:
        pr["p0"] = p0.to_numpy()
    else:
        pr["p0"] = np.where(p_nn.notna(), w * p0 + (1.0 - w) * p_nn, p0)
    pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
    X = dec.assemble(pr, pairs=fr.ctx, sim=fr.sim).merge(fr.meta, on=KEY, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    te = (X.fold == 0).to_numpy()
    base = (~te) & X.y.notna().to_numpy()
    tr = X[base & (X.fold != 1).to_numpy()]
    va = X[base & (X.fold == 1).to_numpy()]
    P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0, n_jobs=6)
    n = P.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n, **P)
    m.fit(tr[cols], tr.y.astype(int), eval_set=[(va[cols], va.y.astype(int))],
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    s2 = np.zeros(len(X))
    s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
    X["p2"] = T.norm_prob(X, s2)
    q = fr.keep.merge(X[KEY + ["p2"]], on=KEY, how="left")
    q["p2"] = q.p2.fillna(0.0)
    return q


def conc_errors(df: pd.DataFrame, col: str, fam: str = "m30") -> dict:
    t = B.top1(df[df.fam == fam], col)
    err = t[t.ok == 0]
    conc = sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
               for a, b in zip(err.Phase, err.cand_phase))
    return {"n": int(len(t)), "errors": int(len(err)), "concurrent": int(conc)}


def evaluate(fr: E.Frame, name: str) -> dict:
    t0 = time.time()
    v = load_variant(name)
    lg = fr.lg[KEY + ["p0", "fold"]].copy()
    if v is not None:
        lg = lg.merge(v[KEY + ["p0"]].rename(columns={"p0": "p0n"}), on=KEY, how="left")
        miss = lg.p0n.isna() & lg.fold.isin(v.merge(lg[KEY + ["fold"]], on=KEY).fold.unique())
        if miss.any():
            log(f"  WARNING {int(miss.sum())} rows of refitted folds have no new p0")
        lg["p0"] = np.where(lg.p0n.notna(), lg.p0n, lg.p0)
    # stage-13 GRU on the tree frame, renormalised over the tree candidates
    nn = E._norm(fr.gru_all)[KEY + ["prob"]].rename(columns={"prob": "p_nn"})
    lg = lg.merge(nn, on=KEY, how="left")
    tot = lg.groupby(GRP)["p_nn"].transform("sum")
    lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
    assert (lg[KEY].to_numpy() == fr.lg[KEY].to_numpy()).all()

    trees = decode(fr, lg.p0, None, 1.0).rename(columns={"p2": "p_trees"})
    bb = decode(fr, lg.p0, lg.p_nn, 0.5).rename(columns={"p2": "p_bb"})
    f = fr.f0[KEY + ["Phase", "fam", "p_nn"]].merge(lg[KEY + ["p0"]], on=KEY, how="left")
    f = f.merge(trees[KEY + ["p_trees"]], on=KEY).merge(bb[KEY + ["p_bb"]], on=KEY)
    for c in ("p0", "p_trees", "p_bb"):
        f[c] = f[c] / f.groupby(DET)[c].transform("sum")
    f["p_raw"] = 0.5 * f.p0 + 0.5 * f.p_nn
    f["p_after"] = 0.5 * f.p_trees + 0.5 * f.p_nn
    cols = {"ranker": "p0", "trees": "p_trees", "blend_raw": "p_raw",
            "blend_before": "p_bb", "blend_after": "p_after"}
    res = {k: E.flat(B.acc_by_fam(f, c)) for k, c in cols.items()}
    res["m30_errors"] = {k: conc_errors(f, c) for k, c in cols.items()}
    res["m5_errors"] = {k: conc_errors(f, c, "m5") for k, c in cols.items()}
    res["secs"] = round(time.time() - t0)
    f[KEY + ["Phase", "fam", "p0", "p_trees", "p_bb", "p_nn"]].to_parquet(
        B9 / f"eval_rows_{name}.parquet", index=False)
    log(f"{name}: " + json.dumps({k: res[k] for k in cols}))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fr = E.Frame()
    outp = B9 / f"{a.out}.json"
    res = json.load(open(outp)) if outp.exists() else {}
    res["n_rows_fold0"] = int(len(fr.keep))
    res["n_detector_windows_fold0"] = int(fr.f0.groupby(DET).ngroups)
    for name in a.variants.split(","):
        res[name] = evaluate(fr, name)
        json.dump(res, open(outp, "w"), indent=1)
    log(f"wrote {outp}")


if __name__ == "__main__":
    main()
