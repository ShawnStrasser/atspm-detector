"""Note 112: drop the 5 channel-adjacency neighbour aggregates (adj_*, |channel difference| <= 2) from the phase decoders
(AGENTS.md: the model never sees a detector channel number).  Same OOF inputs and recipe as note 111 (fit111 `dec` /
`dectrees`, p90_phase.decode_iters: folds_v4, inner early stopping on fold k+1, BIN_PARAMS), with / without adj_*, 3 seeds.

    python f112.py oof [--arms v6,trees] [--seeds 0,1,2]   six-fold decoder OOF -> %DC_WORK%/s112/phase/q112.parquet
                                                           (+ dec_iters112.json); columns p2_{arm}_{adj|noadj}_s{seed}
    python f112.py score                                   phase E / R by length, paired signal bootstrap -> score112.json
    python f112.py fit                                     full-data noadj refits (seed 0, n = mean fold iterations)
                                                           -> final_v3_work/v3fit112/{decoder,decode_trees}
Arms: v6 = ranker s0 0.5 / siba w32 x3 phase head 0.5 (net on ranker p >= .01) = q109 p2_w32m3_rs0; trees = ranker s0
alone = q111 p2_trees_s0.  Seed s -> bagging_seed s, feature_fraction_seed s + 100, seed s (seed 0 = the note-111 fits).
CPU, LightGBM 4 threads.  locked_v2 asserted absent by the pool loader.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
for _d in ("final111", "final95", "final109", "final103", "final100", "final90"):
    sys.path.insert(0, str(CODE / _d))
sys.path.insert(0, str(CODE))
import rpath  # noqa: E402,F401

NJ = 4
FAMS = {"m5": ["m5"], "m10": ["m10"], "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h24": ["h24"],
        "ge30": ["m30", "h1", "h3", "h6", "h24", "full"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _setup():
    import fit111 as F
    F.NJ = NJ
    P, P109 = F._phase()
    P.NJ = P.C.NJ = P.T57.NJ = P109.NJ = NJ
    return F, P, P109


def cols(adj: bool):
    import decode_train as dec
    return dec.BASE_COLS + dec.SIM_COLS + (dec.ADJ_COLS if adj else [])


def prm(seed: int):
    import train_official as T
    p = dict(T.BIN_PARAMS, n_jobs=NJ, bagging_seed=seed, feature_fraction_seed=seed + 100, seed=seed)
    return p, p.pop("n_estimators")


def design(P, comb, simc, p0):
    import decode_train as dec
    pr = comb[P.KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[P.KEY4 + ["Phase", "fold", "y"]], on=P.KEY4, how="left")
    return X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)


def first_stage(F, P, P109, comb, arm):
    p0 = F.p0_s0(P, comb)
    return P.blend(comb, p0, P109.net_col(comb, "w32m3")) if arm == "v6" else p0


def stage_oof(a):
    import lightgbm as lgb
    import train_official as T
    F, P, P109 = _setup()
    comb, simc = P.pool()
    out = P.C.DC_WORK / "s112" / "phase"
    out.mkdir(parents=True, exist_ok=True)
    fq, fi = out / "q112.parquet", out / "dec_iters112.json"
    q = pd.read_parquet(fq) if fq.exists() else comb[P.KEY4].copy()
    its = json.load(open(fi)) if fi.exists() else {}
    for arm in a.arms.split(","):
        X = design(P, comb, simc, first_stage(F, P, P109, comb, arm))
        lab, fx = X.Phase.notna().to_numpy(), X.fold.to_numpy()
        for sd in [int(s) for s in a.seeds.split(",")]:
            for adj in (False, True):
                nm = f"{arm}_{'adj' if adj else 'noadj'}_s{sd}"
                if f"p2_{nm}" in q.columns:
                    continue
                c2 = cols(adj)
                s2, ii = np.zeros(len(X)), []
                t0 = time.time()
                for k in range(6):
                    te, inner = fx == k, (k + 1) % 6
                    base = (~te) & lab
                    tr, va = X[base & (fx != inner)], X[base & (fx == inner)]
                    p, n = prm(sd)
                    m = lgb.LGBMClassifier(n_estimators=n, **p).fit(
                        tr[c2], tr.y, eval_set=[(va[c2], va.y)], eval_metric="binary_logloss",
                        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
                    s2[te] = m.predict_proba(X.loc[te, c2])[:, 1]
                    ii.append(int(m.best_iteration_))
                Xo = X[P.KEY4].copy()
                Xo["p2"] = T.norm_prob(X, s2)
                q[f"p2_{nm}"] = comb[P.KEY4].merge(Xo, on=P.KEY4, how="left").p2.to_numpy()
                its[nm] = ii
                q.to_parquet(fq, index=False)
                json.dump(its, open(fi, "w"), indent=1)
                log(f"{nm}: iterations {ii} ({time.time() - t0:.0f}s)")


def stage_score(a):
    F, P, P109 = _setup()
    C = P.C
    comb, _ = P.pool(P.KEY4 + ["fold"])
    q = pd.read_parquet(C.DC_WORK / "s112" / "phase" / "q112.parquet")
    q9 = pd.read_parquet(P109.OUT / "q109.parquet")
    q11 = pd.read_parquet(C.DC_WORK / "s111" / "phase" / "q111.parquet")
    for t in (q, q9, q11):
        for c in P.KEY4:
            assert (t[c].to_numpy() == comb[c].to_numpy()).all()
    arms = {"v6_ref": q9.p2_w32m3_rs0.to_numpy(), "trees_ref": q11.p2_trees_s0.to_numpy()}
    arms.update({c[3:]: q[c].to_numpy() for c in q.columns if c.startswith("p2_")})
    rows = P.rows_()
    alt, sig, DET = rows.alt.to_numpy(), rows.sig.to_numpy(), P.DET
    ok = {}
    for nm, p in arms.items():
        d = comb[P.KEY4].copy()
        d["p"] = p
        d = d.dropna(subset=["p"]).sort_values(DET + ["p", "cand_phase"], ascending=[True, True, True, False, True])
        t = d.groupby(DET, sort=False).first().reset_index()
        pr = rows[DET].merge(t[DET + ["cand_phase"]], on=DET, how="left").cand_phase.to_numpy()
        okE = (pr == rows.Phase.to_numpy()).astype(float)
        ok[nm] = {"E": okE, "R": np.maximum(okE, np.array([x in s for x, s in zip(pr, alt)], float))}
    # comparisons: each noadj vs the same-seed adj (paired), and the seed-mean of noadj vs the v6 package reference
    seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in arms if "_s" in k and k.split("_")[0] in ("v6", "trees")})
    res = {"seeds": seeds, "n": {}, "acc": {}, "delta": {}, "repro_seed0": {}}
    for arm, ref in (("v6", "v6_ref"), ("trees", "trees_ref")):
        if f"{arm}_adj_s0" in ok:
            res["repro_seed0"][arm] = float(np.mean(ok[f"{arm}_adj_s0"]["E"] == ok[ref]["E"]))
    for sname, pre in (("everything", "E"), ("realistic", "R")):
        mset = rows[sname].to_numpy()
        for pn, fams in FAMS.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            res["n"].setdefault(pre, {})[pn] = [int(m.sum()), int(len(set(sig[m])))]
            for nm in ok:
                res["acc"].setdefault(nm, {}).setdefault(pre, {})[pn] = C.acc_ci(ok[nm][pre][m], sig[m])
            for arm, ref in (("v6", "v6_ref"), ("trees", "trees_ref")):
                for sd in seeds:
                    a_, b_ = f"{arm}_adj_s{sd}", f"{arm}_noadj_s{sd}"
                    if a_ in ok and b_ in ok:
                        res["delta"].setdefault(f"{arm} noadj-adj s{sd}", {}).setdefault(pre, {})[pn] = C.delta_ci(
                            ok[a_][pre][m], ok[b_][pre][m], sig[m])
                    if b_ in ok:
                        res["delta"].setdefault(f"{arm} noadj s{sd} - pkg", {}).setdefault(pre, {})[pn] = C.delta_ci(
                            ok[ref][pre][m], ok[b_][pre][m], sig[m])
                na = [ok[f"{arm}_noadj_s{s}"][pre][m] for s in seeds if f"{arm}_noadj_s{s}" in ok]
                aa = [ok[f"{arm}_adj_s{s}"][pre][m] for s in seeds if f"{arm}_adj_s{s}" in ok]
                if len(na) > 1 and len(na) == len(aa):   # seed-averaged per-row correctness, paired bootstrap
                    res["delta"].setdefault(f"{arm} noadj-adj seedmean", {}).setdefault(pre, {})[pn] = C.delta_ci(
                        np.mean(aa, 0), np.mean(na, 0), sig[m])
    o = C.DC_WORK / "s112" / "score112.json"
    json.dump(res, open(o, "w"), indent=1)
    log(f"repro seed0 (row agreement with package OOF): {res['repro_seed0']}")
    for nm in ok:
        log(f"{nm:18s} E " + " ".join(f"{p} {res['acc'][nm]['E'][p][0]}" for p in FAMS))
    for nm, v in res["delta"].items():
        log(f"{nm:26s} dE " + " ".join(f"{p} {v['E'][p]}" for p in FAMS) + f" | dR ge30 {v['R']['ge30']}")


def stage_fit(a):
    import lightgbm as lgb
    F, P, P109 = _setup()
    comb, simc = P.pool()
    its = json.load(open(P.C.DC_WORK / "s112" / "phase" / "dec_iters112.json"))
    c2 = cols(False)
    for arm, sub, stem in (("v6", "decoder", "decode_v3"), ("trees", "decode_trees", "decode_v3")):
        X = design(P, comb, simc, first_stage(F, P, P109, comb, arm))
        lab = X.Phase.notna().to_numpy()
        ii = its[f"{arm}_noadj_s0"]
        n_est = int(round(np.mean(ii)))
        p, _ = prm(0)
        out = P.C.DC_WORK / "final_v3_work" / "v3fit112" / sub
        out.mkdir(parents=True, exist_ok=True)
        m = lgb.LGBMClassifier(n_estimators=n_est, **p).fit(X.loc[lab, c2], X.y[lab])
        m.booster_.save_model(str(out / f"{stem}.txt"))
        np.save(out / "X_check.npy", X.loc[lab, c2].to_numpy(np.float64)[:2000])
        json.dump({"features": c2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
                   "params": {k: v for k, v in p.items() if k != "n_jobs"},
                   "trained_on": {"labelled_pair_rows": int(lab.sum()), "period": "Sept-2026 only"},
                   "n_estimators_rule": f"mean best iteration of the six note-112 fold fits (arm {arm}_noadj_s0) {ii}",
                   "first_stage_input": ("ranker s0 0.5 / siba w32 x3 phase head 0.5 (net on ranker p >= .01)"
                                         if arm == "v6" else "ranker s0 alone (fast profiles)"),
                   "note": "note 112: channel-adjacency aggregates (adj_*) removed -- no channel-number input"},
                  open(out / f"{stem}.json", "w"), indent=1)
        log(f"{arm} noadj full fit -> {out}: {n_est} trees, {int(lab.sum()):,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["oof", "score", "fit"])
    ap.add_argument("--arms", default="v6,trees")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
