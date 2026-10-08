"""Note 53: score a TCN variant on fold 0 -- the network alone, the trees alone, and the blend.

Harness = note 37 (`phase_v3_eval.py`) restricted to fold 0, i.e. the CURRENT trees:
  trees   phase_v3 ranker bag + joint decoder, OOF (`phase_v3/work/trees_oof_bywindow.parquet`)
  blend   0.5 x ranker bag p0 + 0.5 x net, BEFORE the joint decoder; the decoder is fitted on
          folds 2-5 (fold 1 = early stopping) with the phase_v3 GRU OOF (`p3_oof_f1..5`) as the
          net of the training folds, so every candidate is decoded by the same recipe and only
          fold 0's net changes -- with `p3_oof_f0` itself as the candidate this is exactly
          note 37's fold-0 decode (same fit as `decode_oof`'s fold 0).
Rows: stage 13's fold-0 rows (`blend_v2.common_frame`: labelled phase greens in the window, the
detector actuated), all 22 windows, per-family accuracy = mean over the family's windows.
Also a paired, signal-grouped bootstrap (2,000 draws) of every candidate against `--ref`.

    python tcn53_eval.py --cands base,r05,... [--ref base] [--out eval_f0]
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
from common import DC_WORK  # noqa: E402
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
import blend_v2 as B  # noqa: E402
import blend_v2_predecode as BP  # noqa: E402
import trackb_eval as TE  # noqa: E402
import phase_v3_eval as PE  # noqa: E402

ROOT = DC_WORK / "tcn53"
KEY, DET, GRP = B.KEY, B.DET, BP.GRP
FAMS = TE.FAMS
W3 = DC_WORK / "final_v3_work" / "phase_v3" / "work"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_cand(name: str) -> pd.DataFrame:
    for p in (ROOT / "preds" / f"{name}_f0_bywindow.parquet", ROOT / "preds" / name,
              W3 / "gru_preds" / f"{name}_bywindow.parquet", Path(name)):
        if p.exists():
            return TE._norm(pd.read_parquet(p))[KEY + ["prob"]]
    raise FileNotFoundError(name)


class Frame:
    def __init__(self, fold: int = 0) -> None:
        self.fold = fold
        keep = B.common_frame(TE._norm(B.gru_oof()))
        self.keep = keep.loc[keep.fold == fold, KEY].reset_index(drop=True)
        folds = PE.fold_map()
        lk = set(pd.read_csv(PE.LOCKED).DeviceId.str.lower())
        tn = PE.trees_new()
        assert not PE.plain(tn.DeviceId).isin(lk).any()
        lab = B.labels()
        lg = tn[KEY + ["prob", "p0"]].merge(folds, on="DeviceId", how="inner")
        lg["dev_plain"] = PE.plain(lg.DeviceId)
        lg = lg.merge(lab.rename(columns={"DeviceId": "dev_plain"}),
                      on=["dev_plain", "Detector"], how="left")
        lg["y"] = np.where(lg.Phase.notna(), (lg.cand_phase == lg.Phase).astype(float), np.nan)
        self.lg = lg
        self.meta = lg[KEY + ["Phase", "fold", "y"]]
        self.ctx, self.sim = BP.load_context(set(folds.DeviceId))
        nn = PE.net_new().merge(folds, on="DeviceId", how="inner")
        self.nn_rest = nn[nn.fold != fold][KEY + ["prob"]]
        log(f"fold-{fold} rows {len(self.keep):,}, det-windows "
            f"{self.keep.groupby(DET).ngroups:,}")

    def run(self, cand: pd.DataFrame, w: float = 0.5) -> pd.DataFrame:
        nn = pd.concat([self.nn_rest, cand], ignore_index=True).rename(
            columns={"prob": "p_nn"})
        lg = self.lg.merge(nn, on=KEY, how="left")
        tot = lg.groupby(GRP)["p_nn"].transform("sum")
        lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
        pr = lg[KEY].copy()
        pr["p0"] = np.where(lg.p_nn.notna(), w * lg.p0 + (1 - w) * lg.p_nn, lg.p0)
        pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
        X = dec.assemble(pr, pairs=self.ctx, sim=self.sim).merge(self.meta, on=KEY, how="left")
        X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
        cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
        k = self.fold
        te = (X.fold == k).to_numpy()
        inner = (k + 1) % 6
        base = (~te) & X.y.notna().to_numpy()
        tr = X[base & (X.fold != inner).to_numpy()]
        va = X[base & (X.fold == inner).to_numpy()]
        P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0, n_jobs=8)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(tr[cols], tr.y.astype(int), eval_set=[(va[cols], va.y.astype(int))],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2 = np.zeros(len(X))
        s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
        X["p2"] = T.norm_prob(X, s2)
        q = self.keep.merge(lg[KEY + ["Phase", "prob", "p_nn"]], on=KEY, how="left") \
            .merge(X[KEY + ["p2"]], on=KEY, how="left")
        q["p2"] = q.p2.fillna(0.0)
        return q


def score(q: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    d = q.copy()
    for c in ("prob", "p_nn"):
        d[c] = d[c] / d.groupby(DET)[c].transform("sum")
    res, oks = {}, {}
    for col, nm in (("prob", "trees"), ("p_nn", "net"), ("p2", "blend")):
        dd = d if col != "p_nn" else d[d.groupby(DET)["p_nn"].transform("count") > 0]
        r = B.acc_by_fam(dd, col)
        res[nm] = TE.flat(r)
        res[nm + "_n"] = {f: r[f]["n"] for f in FAMS if f in r}
        oks[nm] = B.top1(dd, col).set_index(DET).ok
    return res, pd.DataFrame(oks)


def boot(ok_ref, ok_new, n=2000, seed=0) -> dict:
    out = {}
    for m in ("net", "blend"):
        j = pd.concat([ok_ref[m].rename("o"), ok_new[m].rename("n")], axis=1).dropna()
        j["fam"] = j.index.get_level_values("win").map(B.fam_of)
        j["sig"] = j.index.get_level_values("DeviceId")
        out[m] = {}
        for fam in FAMS:
            g = j[j.fam == fam]
            if not len(g):
                continue
            s = g.groupby("sig").agg(d=("n", "sum"), o=("o", "sum"), k=("n", "size"))
            dd, kk = (s.d - s.o).to_numpy(float), s.k.to_numpy(float)
            idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
            bs = dd[idx].sum(1) / kk[idx].sum(1)
            out[m][fam] = [round(100 * dd.sum() / kk.sum(), 3),
                           round(100 * float(np.quantile(bs, .025)), 3),
                           round(100 * float(np.quantile(bs, .975)), 3)]
    return out


def six(names: list[str], ref: str, out: str) -> None:
    """Six-fold OOF (note-37 harness, stage-13 rows): every fold's net = the candidate, decoder
    re-fitted out of fold on the blended inputs (`phase_v3_eval.run_arm`)."""
    dest = ROOT / f"{out}.json"
    res = json.load(open(dest)) if dest.exists() else {}
    keep = B.common_frame(TE._norm(B.gru_oof()))[KEY].copy()
    folds = PE.fold_map()
    lab = B.labels()
    ctx, sim = BP.load_context(set(folds.DeviceId))
    tn = PE.trees_new()[KEY + ["prob", "p0"]]
    arms = {}
    for nm in names:
        if nm == "gru_p3":
            arms[nm] = PE.net_new()
            continue
        fs = [ROOT / "preds" / f"{nm}_f{k}_bywindow.parquet" for k in range(6)]
        if not all(f.exists() for f in fs):
            log(f"{nm}: not all six folds predicted"); continue
        arms[nm] = pd.concat([TE._norm(pd.read_parquet(f))[KEY + ["prob"]] for f in fs],
                             ignore_index=True)
    oks = {}
    for nm, nn in arms.items():
        q = PE.run_arm(nm, tn, nn, folds, lab, ctx, sim, set())
        r, ok = PE.score(q, keep)
        oks[nm] = ok
        res[nm] = r
        log(f"{nm}: " + json.dumps(r))
        json.dump(res, open(dest, "w"), indent=1)
    for nm in oks:
        if nm != ref and ref in oks:
            res[nm][f"boot_vs_{ref}"] = PE.boot(oks[ref], oks[nm])
    json.dump(res, open(dest, "w"), indent=1)
    log(f"wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--six", action="store_true")
    ap.add_argument("--cands", required=True)
    ap.add_argument("--ref", default="base")
    ap.add_argument("--out", default="eval_f0")
    a = ap.parse_args()
    if a.six:
        six(a.cands.split(","), a.ref, a.out)
        return
    dest = ROOT / f"{a.out}.json"
    res = json.load(open(dest)) if dest.exists() else {}
    fr = Frame(0)
    names = a.cands.split(",")
    if a.ref not in names:
        names = [a.ref] + names
    oks = {}
    for nm in names:
        t0 = time.time()
        try:
            cand = read_cand(nm)
        except FileNotFoundError:
            log(f"{nm}: no predictions yet"); continue
        q = fr.run(cand)
        r, ok = score(q)
        oks[nm] = ok
        res[nm] = r
        log(f"{nm}: net {json.dumps(r['net'])}")
        log(f"{nm}: blend {json.dumps(r['blend'])}  ({time.time()-t0:.0f}s)")
        json.dump(res, open(dest, "w"), indent=1)
    if a.ref in oks:
        for nm, ok in oks.items():
            if nm != a.ref:
                res[nm][f"boot_vs_{a.ref}"] = boot(oks[a.ref], ok)
    json.dump(res, open(dest, "w"), indent=1)
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
