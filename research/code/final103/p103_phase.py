"""Note 103: can ONE LightGBM per task replace scorer -> (TCN) -> decider?  PHASE side, 2026-only pool (note 95).

    python p103_phase.py ctx                 build context columns aligned to the pool -> s103/phase/ctx_*.parquet
    python p103_phase.py fit --arm A1 [--seeds 0,1,2]   one LambdaRank on ranker features + context, six folds
    python p103_phase.py score               per-length E / R, CIs vs full v5b, all arms + references

Arms (all: note-57 ranker recipe, inner fold (k+1)%6 early stopping, folds_v4, 3 seeds bagged, argmax):
  A1   ranker features (261) + PREDICTION-FREE context: for 8 key per-candidate features, what the OTHER detectors at
       the signal show for the same candidate (leave-one-out mean / max, own rank among detectors, similarity- and
       channel-adjacency-weighted neighbour means), 2 "claims" counts (other detectors whose best candidate by the
       feature is this one), graph sizes.  Needs no other model.  ONE LightGBM (3 seeds).
  A1o  ranker features + the decoder's own context columns built from the six-fold OOF ranker probabilities
       (standard OOF, as the current decoder).  Needs the ranker's prediction -> still TWO models in production.
  A2   A1 + TCN output (p95_ad six-fold OOF, all candidates, no tree filter) as features + the decoder's context
       columns computed on the TCN probabilities (no tree model needed).  TCN + ONE LightGBM.
References (saved OOF, same rows): full v5b (= v5 phase, p2_p95_ad), no networks (trees bag + decoder, p2_trees),
fast le2h (p2_le2h), trees alone (ranker argmax p0t).  locked_v2 asserted absent (pool loader).  CPU <= 6 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
import phase95 as P  # noqa: E402  (sets T57.OUT = s95/phase, NJ)

C, T57 = P.C, P.T57
NJ = int(os.environ.get("P103_THREADS", "6"))
OUT = C.DC_WORK / "s103" / "phase"
KEY4, DET = P.KEY4, P.DET
F8 = ["call43_fwd_lift__z", "call43_fwd_035", "on_lift_green__z", "call43_rev_frac__mgap", "f_on_green__z",
      "call43_rev_onnow__z", "release_frac__z", "call44_fwd_1"]
CLAIM = ["call43_fwd_lift__rank", "on_lift_green__rank"]
FAMS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]
POOLS = {"m5": ["m5"], "m10": ["m10"], "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h24": ["h24"],
         "ge30": ["m30", "h1", "h3", "h6", "h24", "full"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def loo_max(v, g):
    """max of v over the OTHER members of group g (NaN for singletons / all-NaN)."""
    v2 = np.where(np.isnan(v), -np.inf, v)
    o = np.lexsort((-v2, g))
    gs, vs = g[o], v2[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    sp = np.minimum(start + 1, len(gs) - 1)
    sec = np.where((start + 1 < len(gs)) & (gs[sp] == gs), vs[sp], -np.inf)
    out = np.where(np.arange(len(gs)) == start, sec, vs[start])
    r = np.empty(len(v))
    r[o] = out
    return np.where(np.isinf(r), np.nan, r)


def nb_mean(comb, nb, cols, prefix):
    """weighted mean over neighbours `other` (DeviceId, win, Detector, other, w) of comb[cols] at the same candidate."""
    src = comb[["DeviceId", "win", "Detector", "cand_phase"] + cols].rename(columns={"Detector": "other"})
    j = nb.merge(src, on=["DeviceId", "win", "other"], how="inner")
    out = {}
    for c in cols:
        v = j[c].to_numpy(np.float64)
        ok = ~np.isnan(v)
        j["_wv"] = np.where(ok, j.w * np.nan_to_num(v), 0.0)
        j["_ww"] = np.where(ok, j.w, 0.0)
        g = j.groupby(["DeviceId", "win", "Detector", "cand_phase"], sort=False)[["_wv", "_ww"]].sum()
        out[f"{prefix}_{c}"] = g._wv / g._ww.replace(0, np.nan)
    o = pd.DataFrame(out).reset_index()
    return comb[KEY4].merge(o, on=KEY4, how="left").drop(columns=KEY4)


def stage_ctx(a):
    OUT.mkdir(parents=True, exist_ok=True)
    comb, simc, fc = T57.load_pool()
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    assert all(c in fc for c in F8 + CLAIM)
    t0 = time.time()
    # ---------------- A1: prediction-free context
    g = comb.groupby(["DeviceId", "win", "cand_phase"], sort=False).ngroup().to_numpy()
    ctx = {}
    gs = pd.Series(g)
    for c in F8:
        v = comb[c].to_numpy(np.float64)
        s = pd.Series(np.nan_to_num(v)).groupby(g).transform("sum").to_numpy()
        n = pd.Series((~np.isnan(v)).astype(float)).groupby(g).transform("sum").to_numpy()
        own = np.nan_to_num(v)
        ownn = (~np.isnan(v)).astype(float)
        ctx[f"sx_mean_o_{c}"] = np.where(n - ownn > 0, (s - own) / np.maximum(n - ownn, 1), np.nan)
        ctx[f"sx_max_o_{c}"] = loo_max(v, g)
        ctx[f"sx_rank_{c}"] = pd.Series(v).groupby(g).rank(ascending=False, pct=True).to_numpy()
    for c in CLAIM:
        top = (comb[c].to_numpy() == 1).astype(float)
        ctx[f"sx_claims_o_{c}"] = pd.Series(top).groupby(g).transform("sum").to_numpy() - top
    ctx["sx_ndet"] = gs.map(gs.value_counts()).to_numpy(float)
    X = pd.DataFrame(ctx)
    # similarity / channel-adjacency neighbours (decoder graphs, behaviour weights)
    sim = simc.copy()
    sim["w"] = sim.phi.clip(lower=0) ** 2
    sim = sim[sim.w > 0][["DeviceId", "win", "Detector", "other", "w"]]
    dets = comb[["DeviceId", "win", "Detector"]].drop_duplicates()
    adj = dets.merge(dets.rename(columns={"Detector": "other"}), on=["DeviceId", "win"])
    d = (adj.Detector.astype(int) - adj.other.astype(int)).abs()
    adj = adj[(d >= 1) & (d <= 2)].copy()
    adj["w"] = np.where((adj.Detector.astype(int) - adj.other.astype(int)).abs() == 1, 1.0, 0.5)
    for c in CLAIM:
        comb[f"top_{c}"] = (comb[c].to_numpy() == 1).astype(float)
    nbc = F8 + [f"top_{c}" for c in CLAIM]
    X = pd.concat([X, nb_mean(comb, sim, nbc, "sim"), nb_mean(comb, adj, nbc, "adj")], axis=1)
    for nm, nb in (("sim", sim), ("adj", adj)):
        st = nb.groupby(["DeviceId", "win", "Detector"]).agg(**{f"{nm}_wsum": ("w", "sum"), f"{nm}_n": ("w", "size")})
        X = pd.concat([X, comb[["DeviceId", "win", "Detector"]].merge(st.reset_index(), on=["DeviceId", "win", "Detector"],
                                                                       how="left").drop(columns=["DeviceId", "win", "Detector"])],
                      axis=1)
    X = X.astype(np.float32)
    X.to_parquet(OUT / "ctx_A1.parquet", index=False)
    log(f"A1 context {X.shape} ({time.time() - t0:.0f}s)")
    # ---------------- decoder columns from a probability (OOF ranker bag -> A1o; TCN -> A2)
    import decode_train as dec
    dcols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    dcols = [c for c in dcols if c not in ("cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours")]
    t = P.trees_bag(comb)
    nn = P.tcn_probs(comb, "p95_ad")
    # TCN missing on 1.2 % of pairs: uniform over the detector's candidates where missing, then renormalised
    grp = [comb[c] for c in DET]
    ncand = pd.Series(np.ones(len(comb))).groupby(grp).transform("sum").to_numpy()
    nnf = np.where(np.isnan(nn), 1.0 / ncand, nn)
    nnf = nnf / pd.Series(nnf).groupby(grp).transform("sum").to_numpy()
    for tag, p0 in (("A1o", t.p0_bag.to_numpy()), ("A2", nnf)):
        pr = comb[KEY4].copy()
        pr["p0"] = p0
        D = dec.assemble(pr, pairs=comb, sim=simc)
        D = comb[KEY4].merge(D[KEY4 + dcols], on=KEY4, how="left")
        pre = "d_" if tag == "A1o" else "tn_"
        D = D[dcols].rename(columns={c: pre + c for c in dcols}).astype(np.float32)
        if tag == "A2":
            D["tn_missing"] = np.isnan(nn).astype(np.float32)
        D.to_parquet(OUT / f"ctx_{tag}_dec.parquet", index=False)
        log(f"{tag} decoder-style columns {D.shape} ({time.time() - t0:.0f}s)")


def arm_cols(arm, comb, fc):
    add = []
    A1 = pd.read_parquet(OUT / "ctx_A1.parquet")
    if arm in ("A1", "A2"):
        add.append(A1)
    if arm == "A1o":
        add.append(pd.read_parquet(OUT / "ctx_A1o_dec.parquet"))
    if arm == "A2":
        add.append(pd.read_parquet(OUT / "ctx_A2_dec.parquet"))
    new = []
    for D in add:
        assert len(D) == len(comb)
        for c in D.columns:
            comb[c] = D[c].to_numpy()
            new.append(c)
    return list(fc) + new


def stage_fit(a):
    import train_official as T
    comb, simc, fc = T57.load_pool()
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    cols = arm_cols(a.arm, comb, fc)
    d = OUT / a.arm
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"arm": a.arm, "n_cols": len(cols), "cols": cols}, open(d / "cols.json", "w"), indent=1)
    log(f"arm {a.arm}: {len(cols)} features")
    lab = comb.Phase.notna().to_numpy()
    fold = comb.fold.to_numpy()
    t0 = time.time()
    for sd in [int(x) for x in a.seeds.split(",")]:
        for k in range(6):
            f = d / f"S_f{k}_s{sd}.npy"
            if f.exists():
                continue
            t1 = time.time()
            inner = (k + 1) % 6
            base = (fold != k) & lab
            P_ = T57.rp(sd)
            P_["n_jobs"] = NJ
            m = T._fit_rank(comb[base & (fold != inner)], comb[base & (fold == inner)], cols, P_)
            np.save(f, m.predict(comb.loc[fold == k, cols], num_threads=NJ))
            with open(d / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"fold": k, "seed": sd, "trees": int(m.best_iteration_ or 0),
                                     "secs": round(time.time() - t1, 1)}) + "\n")
            log(f"  {a.arm} fold {k} seed {sd}: {m.best_iteration_} trees ({time.time() - t1:.0f}s, total {time.time() - t0:.0f}s)")
            if k == 0 and sd == 0:
                imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
                (imp / imp.sum()).sort_values(ascending=False).head(40).to_json(d / "gain_f0_s0.json", indent=1)


def arm_prob(arm, comb):
    import train_official as T
    d = OUT / arm
    fold = comb.fold.to_numpy()
    seeds = sorted({int(p.stem.split("_s")[1]) for p in d.glob("S_f5_s*.npy")})
    Ps = []
    for sd in seeds:
        S = np.zeros(len(comb))
        for k in range(6):
            S[fold == k] = np.load(d / f"S_f{k}_s{sd}.npy")
        Ps.append(T.to_prob(comb, S))
    return np.mean(Ps, 0), seeds


def stage_score(a):
    comb, _ = P.pool(KEY4 + ["fold"])
    q = pd.read_parquet(P.P95 / "q95.parquet")
    for c in KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all()
    arms = {"full_v5b": q.p2_p95_ad.to_numpy(), "no_nets": q.p2_trees.to_numpy(), "fast_le2h": q.p2_le2h.to_numpy(),
            "trees_alone": q.p0t.to_numpy()}
    seeds = {}
    for arm in a.arms.split(","):
        if (OUT / arm / "S_f5_s0.npy").exists():
            arms[arm], seeds[arm] = arm_prob(arm, comb)
            for sd in seeds[arm]:
                if len(seeds[arm]) > 1:
                    arms[f"{arm}@s{sd}"], _ = arm_prob_seed(arm, comb, sd)
    rows = P.rows_()
    alt = rows.alt.to_numpy()
    sig = rows.sig.to_numpy()
    ok = {}
    for nm, p in arms.items():
        d = comb[KEY4].copy()
        d["p"] = p
        d = d.dropna(subset=["p"]).sort_values(DET + ["p", "cand_phase"], ascending=[True, True, True, False, True])
        t = d.groupby(DET, sort=False).first().reset_index()
        pred = rows[DET].merge(t[DET + ["cand_phase"]], on=DET, how="left").cand_phase.to_numpy()
        okE = (pred == rows.Phase.to_numpy()).astype(float)
        okR = np.maximum(okE, np.array([p_ in s for p_, s in zip(pred, alt)], float))
        ok[nm] = {"E": okE, "R": okR}
    res = {"seeds": seeds, "n": {}, "acc": {}, "delta_vs_full": {}, "delta_vs_no_nets": {}}
    for sname, pre in (("everything", "E"), ("realistic", "R")):
        mset = rows[sname].to_numpy()
        for pn, fams in POOLS.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            res["n"].setdefault(pre, {})[pn] = [int(m.sum()), int(len(set(sig[m])))]
            for nm in ok:
                res["acc"].setdefault(nm, {}).setdefault(pre, {})[pn] = C.acc_ci(ok[nm][pre][m], sig[m])
                if nm != "full_v5b":
                    res["delta_vs_full"].setdefault(nm, {}).setdefault(pre, {})[pn] = C.delta_ci(
                        ok["full_v5b"][pre][m], ok[nm][pre][m], sig[m])
                if nm not in ("no_nets",):
                    res["delta_vs_no_nets"].setdefault(nm, {}).setdefault(pre, {})[pn] = C.delta_ci(
                        ok["no_nets"][pre][m], ok[nm][pre][m], sig[m])
    json.dump(res, open(OUT / "score103.json", "w"), indent=1)
    for nm in ok:
        log(f"{nm:14s} E " + " ".join(f"{pn} {res['acc'][nm]['E'][pn][0]}" for pn in POOLS))
        if nm != "full_v5b":
            log(f"{'':14s} dE vs full " + " ".join(f"{pn} {res['delta_vs_full'][nm]['E'][pn]}" for pn in POOLS))


def arm_prob_seed(arm, comb, sd):
    import train_official as T
    d = OUT / arm
    fold = comb.fold.to_numpy()
    S = np.zeros(len(comb))
    for k in range(6):
        S[fold == k] = np.load(d / f"S_f{k}_s{sd}.npy")
    return T.to_prob(comb, S), [sd]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ctx", "fit", "score"])
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="A1,A1o,A2")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
