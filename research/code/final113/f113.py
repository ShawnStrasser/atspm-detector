"""Note 113: behaviour-only neighbour aggregates for the phase decoders, to replace the channel-adjacency block (adj_*).

Same OOF inputs, folds and recipe as note 112 (f112: fit111 dec / dectrees, folds_v4, inner early stopping on fold k+1,
BIN_PARAMS, seeds 0/1/2).  Every arm = BASE_COLS + SIM_COLS (no adj_*) + one or more new neighbour families, each the
note-04 `_agg_neighbours` block (mean / max / top1 / wsum / n of the neighbours' first-stage probability for the
candidate) over a different behaviour-only neighbour graph built from pairs113.parquet:
  co   stratified same-second co-onset excess  r0 = (O0 - E0) / sqrt(na nb)   (green-state-stratified expectation)
  lag  stratified 1-8 s lead excess (either direction) rl = max(OL_ab - EL_ab, OL_ba - EL_ba) / sqrt(na nb)
  phi1 plain same-second phi (1-s bins; the shipped phi uses 2-s bins)  -- bin-size control
  nn2  second-order phi votes: weight(a, c) = sum_b w(a, b) w(b, c) over the shipped phi graph, c != a
Graph weights = clip(r, 0)^2, top-8 per detector (as similarity.py).  No channel number enters any column.

    python f113.py oof --arms co,lag,colag,phi1,nn2 [--fam trees,v6] [--seeds 0]
    python f113.py score                      -> %DC_WORK%/s113/score113.json (paired vs v6 adj and noadj, E / R, length)
    python f113.py shuf --arm X               shuffled-feature control (new block permuted across rows, seeds 0-2)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final112"))
import f112  # noqa: E402

NJ = 4
TOPK = 8
FAMS = f112.FAMS
AGG = ["mean", "max", "top1", "wsum", "n"]
ARMS = {"co": ["co"], "lag": ["lag"], "colag": ["co", "lag"], "phi1": ["phi1"], "nn2": ["nn2"],
        "colagnn2": ["co", "lag", "nn2"], "all": ["co", "lag", "phi1", "nn2"],
        "lagz": ["lagz"], "lag16": ["lag16"], "lagraw": ["lagraw"], "lagadj": ["lag", "ADJ"], "lagzlag": ["lag", "lagz"],
        "lagt": ["lagt"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def graphs(simc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Neighbour graphs (DeviceId, win, Detector, other, w), top-8 per detector."""
    import cand64 as C
    pr = pd.read_parquet(C.DC_WORK / "s113" / "pairs113.parquet")
    pr = pr.rename(columns={"a": "Detector", "c": "other"})
    g = np.sqrt(np.maximum(pr.na.to_numpy(np.float64) * pr.nb.to_numpy(np.float64), 1.0))
    # reverse-direction lag for the symmetric lead measure
    rev = pr[["DeviceId", "win", "Detector", "other", "OL", "EL"]].rename(
        columns={"Detector": "other", "other": "Detector", "OL": "OLr", "EL": "ELr"})
    pr = pr.merge(rev, on=["DeviceId", "win", "Detector", "other"], how="left")
    N, na, nb, O0 = (pr[c].to_numpy(np.float64) for c in ("N", "na", "nb", "O0"))
    sc = {"co": (O0 - pr.E0.to_numpy(np.float64)) / g,
          "lag": np.maximum(pr.OL.to_numpy(np.float64) - pr.EL.to_numpy(np.float64),
                            pr.OLr.fillna(0).to_numpy(np.float64) - pr.ELr.fillna(0).to_numpy(np.float64)) / g,
          "phi1": (O0 * N - na * nb) / np.sqrt(np.clip(na * (N - na) * nb * (N - nb), 1e-9, None))}
    OLs = pr.OL.to_numpy(np.float64) + pr.OLr.fillna(0).to_numpy(np.float64)
    ELs = pr.EL.to_numpy(np.float64) + pr.ELr.fillna(0).to_numpy(np.float64)
    sc["lagz"] = (OLs - ELs) / np.sqrt(ELs + 1.0)                       # significance, both directions pooled
    sc["lagraw"] = (OLs - 2 * 8 * na * nb / N) / g                      # unstratified lead excess
    sc["lag16"] = sc["lag"]
    sc["lagt"] = sc["lag"]      # the shipped form: top-8 with ties kept (rank 'min'), so no tie is broken by row order
    out = {}
    base = pr[["DeviceId", "win", "Detector", "other"]]
    for k, r in sc.items():
        d = base.copy()
        d["r"] = r
        d = d[d.r > 0].sort_values(["DeviceId", "win", "Detector", "r"], ascending=[True, True, True, False])
        if k == "lagt":
            d = d[d.groupby(["DeviceId", "win", "Detector"], sort=False).r.rank(method="min", ascending=False) <= TOPK]
        else:
            d = d.groupby(["DeviceId", "win", "Detector"], sort=False).head(16 if k == "lag16" else TOPK)
        d["w"] = d.r ** 2
        out[k] = d.drop(columns="r")
    # second-order votes over the shipped phi graph
    s = simc[simc.phi > 0][["DeviceId", "win", "Detector", "other", "phi"]].copy()
    s["w1"] = s.phi.astype(np.float64) ** 2
    j = s[["DeviceId", "win", "Detector", "other", "w1"]].merge(
        s[["DeviceId", "win", "Detector", "other", "w1"]].rename(columns={"Detector": "other", "other": "o2", "w1": "w2"}),
        on=["DeviceId", "win", "other"])
    j = j[j.o2 != j.Detector]
    j["w"] = j.w1 * j.w2
    j = j.groupby(["DeviceId", "win", "Detector", "o2"], as_index=False).w.sum().rename(columns={"o2": "other"})
    j = j.sort_values(["DeviceId", "win", "Detector", "w"], ascending=[True, True, True, False])
    out["nn2"] = j.groupby(["DeviceId", "win", "Detector"], sort=False).head(TOPK)
    for k, d in out.items():
        log(f"graph {k}: {len(d):,} edges, {d.groupby(['DeviceId', 'win', 'Detector']).ngroups:,} detectors with a neighbour")
    return out


def new_feats(X: pd.DataFrame, gr: dict, fams: list[str], P) -> tuple[pd.DataFrame, list[str]]:
    import decode as D
    slim = X[["DeviceId", "win", "Detector", "cand_phase", "p0"]].copy()
    mx = slim.groupby(["DeviceId", "win", "Detector"], sort=False).p0.transform("max")
    slim["is_top1"] = (slim.p0 >= mx).astype(np.float32)
    cols = []
    for f in fams:
        if f == "ADJ":
            cols += ["adj_mean", "adj_max", "adj_top1", "adj_wsum", "adj_n"]
            continue
        a = D._agg_neighbours(slim, gr[f][["DeviceId", "win", "Detector", "other", "w"]], f)
        c = [f"{f}_{x}" for x in AGG]
        X = X.drop(columns=[x for x in c if x in X.columns]).merge(
            a[["DeviceId", "win", "Detector", "cand_phase"] + c], on=["DeviceId", "win", "Detector", "cand_phase"], how="left")
        cols += c
    return X, cols


def run_arm(X, cols, sd, fx, lab):
    import lightgbm as lgb
    s2, ii = np.zeros(len(X)), []
    for k in range(6):
        te, inner = fx == k, (k + 1) % 6
        base = (~te) & lab
        tr, va = X[base & (fx != inner)], X[base & (fx == inner)]
        p, n = f112.prm(sd)
        m = lgb.LGBMClassifier(n_estimators=n, **p).fit(
            tr[cols], tr.y, eval_set=[(va[cols], va.y)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
        ii.append(int(m.best_iteration_))
    return s2, ii


def stage_oof(a, shuf=False):
    import train_official as T
    F, P, P109 = f112._setup()
    comb, simc = P.pool()
    out = P.C.DC_WORK / "s113" / "phase"
    out.mkdir(parents=True, exist_ok=True)
    fq, fi = out / "q113.parquet", out / "dec_iters113.json"
    q = pd.read_parquet(fq) if fq.exists() else comb[P.KEY4].copy()
    its = json.load(open(fi)) if fi.exists() else {}
    gr = graphs(simc)
    arms = a.arms.split(",")
    allf = sorted({f for x in arms for f in ARMS[x] if f != "ADJ"})
    for fam in a.fam.split(","):
        X = f112.design(P, comb, simc, f112.first_stage(F, P, P109, comb, fam))
        X, _ = new_feats(X, gr, allf, P)
        lab, fx = X.Phase.notna().to_numpy(), X.fold.to_numpy()
        for arm in arms:
            newc = [f"{f}_{x}" if f != "ADJ" else f"adj_{x}" for f in ARMS[arm] for x in AGG]
            for sd in [int(s) for s in a.seeds.split(",")]:
                nm = f"{fam}_{arm}{'_shuf' if shuf else ''}_s{sd}"
                if f"p2_{nm}" in q.columns:
                    continue
                Xa = X
                if shuf:   # permute the whole new block across rows (keeps its marginal, kills its link to the row)
                    rng = np.random.default_rng(1000 + sd)
                    Xa = X.copy()
                    Xa[newc] = X[newc].to_numpy()[rng.permutation(len(X))]
                t0 = time.time()
                s2, ii = run_arm(Xa, f112.cols(False) + newc, sd, fx, lab)
                Xo = X[P.KEY4].copy()
                Xo["p2"] = T.norm_prob(X, s2)
                q[f"p2_{nm}"] = comb[P.KEY4].merge(Xo, on=P.KEY4, how="left").p2.to_numpy()
                its[nm] = ii
                q.to_parquet(fq, index=False)
                json.dump(its, open(fi, "w"), indent=1)
                log(f"{nm}: iterations {ii} ({time.time() - t0:.0f}s)")


def stage_score(a):
    F, P, P109 = f112._setup()
    C = P.C
    comb, _ = P.pool(P.KEY4 + ["fold"])
    q = pd.read_parquet(C.DC_WORK / "s113" / "phase" / "q113.parquet")
    q12 = pd.read_parquet(C.DC_WORK / "s112" / "phase" / "q112.parquet")
    for t in (q, q12):
        for c in P.KEY4:
            assert (t[c].to_numpy() == comb[c].to_numpy()).all()
    arms = {c[3:]: q12[c].to_numpy() for c in q12.columns if c.startswith("p2_")}
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
    seeds = lambda stem: sorted(int(k.rsplit("_s", 1)[1]) for k in ok if k.rsplit("_s", 1)[0] == stem)  # noqa: E731
    stems = sorted({k.rsplit("_s", 1)[0] for k in ok if k.split("_")[0] in ("v6", "trees")})
    res = {"n": {}, "acc": {}, "delta": {}, "seeds": {s: seeds(s) for s in stems}}
    for sname, pre in (("everything", "E"), ("realistic", "R")):
        mset = rows[sname].to_numpy()
        for pn, fams in FAMS.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            res["n"].setdefault(pre, {})[pn] = [int(m.sum()), int(len(set(sig[m])))]
            for st in stems:
                ss = seeds(st)
                v = np.mean([ok[f"{st}_s{s}"][pre][m] for s in ss], 0)
                res["acc"].setdefault(st, {}).setdefault(pre, {})[pn] = C.acc_ci(v, sig[m])
            for st in stems:
                fam = st.split("_")[0]
                if st in (f"{fam}_adj", f"{fam}_noadj"):
                    continue
                ss = seeds(st)
                for ref in (f"{fam}_adj", f"{fam}_noadj"):
                    rs = [s for s in ss if f"{ref}_s{s}" in ok]
                    if not rs:
                        continue
                    A = np.mean([ok[f"{ref}_s{s}"][pre][m] for s in rs], 0)
                    B = np.mean([ok[f"{st}_s{s}"][pre][m] for s in rs], 0)
                    res["delta"].setdefault(f"{st} - {ref} (seeds {rs})", {}).setdefault(pre, {})[pn] = C.delta_ci(A, B, sig[m])
    o = C.DC_WORK / "s113" / "score113.json"
    json.dump(res, open(o, "w"), indent=1)
    for st in stems:
        log(f"{st:22s} {res['seeds'][st]} E " + " ".join(f"{p} {res['acc'][st]['E'][p][0]}" for p in FAMS))
    for nm, v in res["delta"].items():
        log(f"{nm:44s} dE " + " ".join(f"{p} {v['E'][p]}" for p in ("m5", "m10", "ge30")) + f" | dR ge30 {v['R']['ge30']}")


LEAD_COLS = ["lag_mean", "lag_max", "lag_top1", "lag_wsum", "lag_n"]


def stage_fit(a):
    """Full-data refits of both decoders with the shipped behaviour-only arm 'lagt' (columns renamed lag_*, the names
    the package's decode.py builds): BASE + SIM + lag_*, seed 0, n = mean best iteration of the six lagt_s0 folds."""
    import lightgbm as lgb
    F, P, P109 = f112._setup()
    comb, simc = P.pool()
    its = json.load(open(P.C.DC_WORK / "s113" / "phase" / "dec_iters113.json"))
    gr = graphs(simc)
    c2 = f112.cols(False) + LEAD_COLS
    for fam, sub in (("v6", "decoder"), ("trees", "decode_trees")):
        X = f112.design(P, comb, simc, f112.first_stage(F, P, P109, comb, fam))
        X, nc = new_feats(X, gr, ["lagt"], P)
        X = X.rename(columns={f"lagt_{x}": f"lag_{x}" for x in AGG})
        lab = X.Phase.notna().to_numpy()
        ii = its[f"{fam}_lagt_s0"]
        n_est = int(round(np.mean(ii)))
        p, _ = f112.prm(0)
        out = P.C.DC_WORK / "final_v3_work" / "v3fit113" / sub
        out.mkdir(parents=True, exist_ok=True)
        m = lgb.LGBMClassifier(n_estimators=n_est, **p).fit(X.loc[lab, c2], X.y[lab])
        m.booster_.save_model(str(out / "decode_v3.txt"))
        np.save(out / "X_check.npy", X.loc[lab, c2].to_numpy(np.float64)[:2000])
        json.dump({"features": c2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
                   "params": {k: v for k, v in p.items() if k != "n_jobs"},
                   "trained_on": {"labelled_pair_rows": int(lab.sum()), "period": "Sept-2026 only"},
                   "n_estimators_rule": f"mean best iteration of the six note-113 fold fits ({fam}_lagt_s0) {ii}",
                   "first_stage_input": ("ranker s0 0.5 / siba w32 x3 phase head 0.5 (net on ranker p >= .01)"
                                         if fam == "v6" else "ranker s0 alone (fast profiles)"),
                   "note": "note 113: channel-adjacency aggregates (adj_*) replaced by LEAD neighbours (lag_*, "
                           "similarity.build_lead_window) -- no channel number in any form"},
                  open(out / "decode_v3.json", "w"), indent=1)
        log(f"{fam} lagt full fit -> {out}: {n_est} trees, {int(lab.sum()):,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["oof", "score", "shuf", "fit"])
    ap.add_argument("--arms", default="co,lag,colag,phi1,nn2")
    ap.add_argument("--fam", default="trees,v6")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "shuf":
        stage_oof(a, shuf=True)
    else:
        globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
