"""Note 95: the PHASE side on 2026-only data (pool %DC_WORK%/s95/phase, pool95.py).

    python phase95.py fit                         note-57 ranker recipe, 3 seeds x 6 folds (folds_v4) -> s95/phase/full/oof.parquet
    python phase95.py decode --tcn TAG            trees bag 0.5 / TCN 0.5 on candidates with tree p >= .01, decoder six-fold OOF
                                                  (note-57 recipe, fold best iterations kept) -> s95/phase/q95.parquet
    python phase95.py v4f                         v4f's saved OOF (s90/p87_q p2_tcn_ad76) on the same 2026 rows (where it has them)
    python phase95.py score                       E / R, >= 30 min / 10 / 5 min, CIs, paired vs v4f on shared rows
    python phase95.py fitdec --tcn TAG            full-data decoder, n = mean of the six fold iterations -> v3fit95/decoder
    python phase95.py fitranker                   full-data ranker, 3 seeds, n = mean fold iterations (fit76 recipe) -> v3fit95/phase
Scoring rows = t57 score_rows on this pool (det_n_on >= 5 'everything'; 'realistic' drops label-check fails and print-phase
disagreements, v4q table), alt phases (switch / additional call) count as right in R. locked_v2 asserted absent.
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

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import cand64 as C  # noqa: E402
import t57_phase as T57  # noqa: E402
import trackb_eval as TE  # noqa: E402

NJ = int(os.environ.get("P95_THREADS", "6"))
C.NJ = NJ
T57.NJ = NJ
P95 = C.DC_WORK / "s95" / "phase"
T57.OUT = P95
FIT = C.DC_WORK / "final_v3_work" / "v3fit95"
PRED = C.DC_WORK / "tcn53" / "preds"
KEY4, DET = C.KEY4, C.DET
POOLS = {"ge30": ["m30", "h1", "h3", "h6", "h24", "full"], "m10": ["m10"], "m5": ["m5"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pool(cols=None):
    need = cols or KEY4 + ["fold", "src", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(P95 / "pool.parquet", columns=need)
    simc = pd.read_parquet(P95 / "pool_sim.parquet")
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    return comb, simc


def stage_fit(a):
    T57.stage_run(argparse.Namespace(cfg="full", seeds="0,1,2", n_noise=0, decode_seeds=False))


def tcn_probs(comb, tag) -> np.ndarray:
    fs = [PRED / f"{tag}_f{k}_bywindow.parquet" for k in range(6)]
    t = pd.concat([TE._norm(pd.read_parquet(f))[KEY4 + ["prob"]] for f in fs], ignore_index=True)
    assert not t.duplicated(KEY4).any()
    k = comb[KEY4].copy()
    k["DeviceId"] = k.DeviceId.str.lower()
    t = t.astype({"Detector": k.Detector.dtype, "cand_phase": k.cand_phase.dtype})
    p = k.merge(t, on=KEY4, how="left").prob.to_numpy()
    log(f"TCN {tag}: coverage {np.mean(~np.isnan(p)):.4f} of pool pairs")
    return p


def blend(q, p0, nn, thr=0.01):
    grp = [q[c] for c in DET]
    nn = np.where(p0 >= thr, nn, np.nan)
    tot = pd.Series(np.nan_to_num(nn)).groupby(grp).transform("sum").to_numpy()
    nn = np.where(~np.isnan(nn) & (tot > 0), nn / np.where(tot > 0, tot, 1.0), np.nan)
    p = np.where(np.isnan(nn), p0, 0.5 * p0 + 0.5 * nn)
    return p / pd.Series(p).groupby(grp).transform("sum").to_numpy()


def trees_bag(comb):
    t = pd.read_parquet(P95 / "full" / "oof.parquet", columns=KEY4 + ["p0_bag", "p2_bag"])
    for c in KEY4:
        assert (t[c].to_numpy() == comb[c].to_numpy()).all(), c
    return t


def stage_decode(a):
    import p90_phase as P90
    P90.C.NJ = NJ
    comb, simc = pool()
    t = trees_bag(comb)
    fq, fi = P95 / "q95.parquet", P95 / "dec_iters.json"
    done = pd.read_parquet(fq) if fq.exists() else comb[KEY4].copy()
    its = json.load(open(fi)) if fi.exists() else {}
    done["p0t"] = t.p0_bag.to_numpy()
    done["p2_trees"] = t.p2_bag.to_numpy()
    nn = tcn_probs(comb, a.tcn)
    done[f"pn_{a.tcn}"] = nn
    done[f"p2_{a.tcn}"], its[a.tcn] = P90.decode_iters(comb, simc, blend(comb, t.p0_bag.to_numpy(), nn))
    done.to_parquet(fq, index=False)
    json.dump(its, open(fi, "w"), indent=1)
    log(f"decoded {a.tcn}: iterations {its[a.tcn]}")


def stage_trees(a):
    """q95 with the trees-only arm (decoded bag) -- before the TCN folds exist."""
    comb, _ = pool(KEY4)
    t = trees_bag(comb)
    fq = P95 / "q95.parquet"
    done = pd.read_parquet(fq) if fq.exists() else comb[KEY4].copy()
    done["p0t"] = t.p0_bag.to_numpy()
    done["p2_trees"] = t.p2_bag.to_numpy()
    done.to_parquet(fq, index=False)


def stage_v4f(a):
    """v4f's saved six-fold OOF on the 2026 rows it has (STG + REL; the Dec-pool signals had no 2026 phase rows)."""
    comb, _ = pool(KEY4)
    q = pd.read_parquet(C.DC_WORK / "s90" / "p87_q.parquet", columns=KEY4 + ["p2_tcn_ad76"])
    q = q[q.DeviceId.str.endswith("@stg")]
    m = comb[KEY4].merge(q.astype({"Detector": comb.Detector.dtype, "cand_phase": comb.cand_phase.dtype}),
                         on=KEY4, how="left")
    out = pd.read_parquet(P95 / "q95.parquet")
    out["p2_v4f"] = m.p2_tcn_ad76.to_numpy()
    t = pd.read_parquet(C.DC_WORK / "final_v3_work" / "f76" / "phase" / "full" / "oof.parquet", columns=KEY4 + ["p2_bag"])
    t = t[t.DeviceId.str.endswith("@stg")]
    out["p2_v4ftrees"] = comb[KEY4].merge(t.astype({"Detector": comb.Detector.dtype, "cand_phase": comb.cand_phase.dtype}),
                                         on=KEY4, how="left").p2_bag.to_numpy()
    out.to_parquet(P95 / "q95.parquet", index=False)
    log(f"v4f rows on the 2026 pool: {np.mean(~np.isnan(out.p2_v4f)):.4f} of pairs")


def rows_():
    r = T57.score_rows().copy()
    r["sig"] = C.plain(r.DeviceId)
    return r


def stage_score(a):
    q = pd.read_parquet(P95 / "q95.parquet")
    rows = rows_()
    alt = rows.alt.to_numpy()
    arms = [c for c in q.columns if c.startswith("p2_")]
    for col in arms:
        d = q[KEY4 + [col]].dropna()
        d = d.sort_values(DET + [col, "cand_phase"], ascending=[True, True, True, False, True])
        t = d.groupby(DET, sort=False).first().reset_index()
        pred = rows[DET].merge(t[DET + ["cand_phase"]], on=DET, how="left").cand_phase.to_numpy()
        has = ~pd.isna(pred)
        okE = pred == rows.Phase.to_numpy()
        okR = okE | np.array([p in s for p, s in zip(pred, alt)])
        rows[f"E_{col}"] = np.where(has, okE, np.nan)
        rows[f"R_{col}"] = np.where(has, okR, np.nan)
    rows["src"] = rows.src.astype(str)
    res = {"acc": {}, "delta": {}, "n": {}}
    sig = rows.sig.to_numpy()
    main = a.main or [c for c in arms if c.startswith("p2_") and c not in ("p2_trees", "p2_v4f", "p2_v4ftrees")][0]
    for sname, pre in (("everything", "E"), ("realistic", "R")):
        mset = rows[sname].to_numpy()
        for pn, fams in POOLS.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            res["n"].setdefault(sname, {})[pn] = [int(m.sum()), int(len(set(sig[m])))]
            for col in arms:
                v = rows[f"{pre}_{col}"].to_numpy()
                mm = m & ~np.isnan(v)
                res["acc"].setdefault(col, {}).setdefault(sname, {})[pn] = C.acc_ci(v[mm], sig[mm])
            # paired vs v4f on the rows v4f has (STG + REL), and by src
            if "p2_v4f" in arms:
                v0 = rows[f"{pre}_p2_v4f"].to_numpy()
                mm = m & ~np.isnan(v0)
                res["delta"].setdefault("new_vs_v4f_shared", {}).setdefault(sname, {})[pn] = C.delta_ci(
                    v0[mm], rows[f"{pre}_{main}"].to_numpy()[mm], sig[mm])
            if "p2_v4ftrees" in arms and "p2_trees" in arms:
                v0 = rows[f"{pre}_p2_v4ftrees"].to_numpy()
                mm = m & ~np.isnan(v0)
                res["delta"].setdefault("trees2026_vs_v4ftrees_shared", {}).setdefault(sname, {})[pn] = C.delta_ci(
                    v0[mm], rows[f"{pre}_p2_trees"].to_numpy()[mm], sig[mm])
            for s in ("STG", "REL", "STG2", "STG3"):
                mm = m & (rows.src.to_numpy() == s)
                if mm.any():
                    res.setdefault("by_src", {}).setdefault(sname, {}).setdefault(pn, {})[s] = \
                        C.acc_ci(rows[f"{pre}_{main}"].to_numpy()[mm], sig[mm])
    json.dump(res, open(P95 / "score95.json", "w"), indent=1)
    rows.drop(columns=["alt"]).to_parquet(P95 / "rows95.parquet", index=False)
    for col in arms:
        log(f"{col:28s} " + " | ".join(f"{s[0]} {p} {res['acc'][col][s][p]}" for s in ("everything", "realistic")
                                       for p in POOLS))
    log(json.dumps(res["delta"]))
    log(json.dumps(res.get("by_src", {}).get("everything", {}).get("ge30", {})))


def stage_fitdec(a):
    import lightgbm as lgb
    import decode_train as dec
    import train_official as T
    comb, simc = pool()
    t = trees_bag(comb)
    its = json.load(open(P95 / "dec_iters.json"))[a.tcn]
    p0 = blend(comb, t.p0_bag.to_numpy(), tcn_probs(comb, a.tcn))
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[KEY4 + ["Phase", "y"]], on=KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    labm = X.Phase.notna().to_numpy()
    n_est = int(round(np.mean(its)))
    P = dict(T.BIN_PARAMS, n_jobs=NJ, bagging_seed=0, feature_fraction_seed=100, seed=0)
    P.pop("n_estimators")
    d = FIT / "decoder"
    d.mkdir(parents=True, exist_ok=True)
    m = lgb.LGBMClassifier(n_estimators=n_est, **P).fit(X.loc[labm, cols2], X.y[labm])
    m.booster_.save_model(str(d / "decode_v3.txt"))
    np.save(d / "X_check.npy", X.loc[labm, cols2].to_numpy(np.float64)[:2000])
    json.dump({"features": cols2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
               "n_estimators_rule": f"mean best iteration of the six note-95 fold fits {its}",
               "params": {k: v for k, v in P.items() if k != "n_jobs"},
               "first_stage_input": "note-95 2026-only trees bag (3 seeds) 0.5 / TCN (2026-only six-fold OOF, "
                                    f"{a.tcn}) 0.5, network only on candidates with tree p >= .01",
               "arm": a.tcn, "trained_on": {"labelled_pair_rows": int(labm.sum()), "period": "Sept-2026 only"}},
              open(d / "decode_v3.json", "w"), indent=1)
    log(f"decoder: n_estimators {n_est}, {int(labm.sum()):,} labelled rows -> {d}")


def stage_fitdec_trees(a):
    """note-98 extra for the fast package on v5: the joint decoder on the 2026 trees bag ALONE (no phase network);
    six-fold OOF fits for the iteration count, then one full-data fit -> f95/extras/decoder_trees_dec."""
    import lightgbm as lgb
    import decode_train as dec
    import train_official as T
    import p90_phase as P90
    P90.C.NJ = NJ
    comb, simc = pool()
    t = trees_bag(comb)
    _, its = P90.decode_iters(comb, simc, t.p0_bag.to_numpy())
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[KEY4].copy()
    pr["p0"] = t.p0_bag.to_numpy()
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[KEY4 + ["Phase", "y"]], on=KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    labm = X.Phase.notna().to_numpy()
    n_est = int(round(np.mean(its)))
    P = dict(T.BIN_PARAMS, n_jobs=NJ, bagging_seed=0, feature_fraction_seed=100, seed=0)
    P.pop("n_estimators")
    d = C.DC_WORK / "final_v3_work" / "f95" / "extras" / "decoder_trees_dec"
    d.mkdir(parents=True, exist_ok=True)
    m = lgb.LGBMClassifier(n_estimators=n_est, **P).fit(X.loc[labm, cols2], X.y[labm])
    m.booster_.save_model(str(d / "decode_v3.txt"))
    np.save(d / "X_check.npy", X.loc[labm, cols2].to_numpy(np.float64)[:2000])
    json.dump({"features": cols2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
               "n_estimators_rule": f"mean best iteration of the six note-95 trees-only fold fits {its}",
               "params": {k: v for k, v in P.items() if k != "n_jobs"},
               "first_stage_input": "note-95 2026-only trees bag (3 seeds) alone, no phase network",
               "arm": "trees_dec", "trained_on": {"labelled_pair_rows": int(labm.sum()), "period": "Sept-2026 only"}},
              open(d / "decode_v3.json", "w"), indent=1)
    log(f"trees-only decoder: n_estimators {n_est} (folds {its}) -> {d}")


def stage_fitranker(a):
    """fit76.stage_phase on the 2026 pool: 3 seeds, n_estimators = mean best iteration of the 18 fold fits."""
    import lightgbm as lgb
    import train_official as T
    comb, simc, fc = T57.load_pool()
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    rec = [json.loads(x) for x in open(P95 / "full" / "timing.jsonl")]
    assert len(rec) == 18
    n_est = int(round(np.mean([r["trees"] for r in rec])))
    lab = comb[comb.Phase.notna()].sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    d = FIT / "phase"
    d.mkdir(parents=True, exist_ok=True)
    log(f"phase ranker: {len(fc)} features, {len(lab):,} labelled pair rows, n_estimators {n_est}")
    for s in (0, 1, 2):
        f = d / f"phase_lgbm_v5_s{s}.txt"
        if f.exists():
            continue
        P = T57.rp(s)
        P.pop("n_estimators")
        m = lgb.LGBMRanker(n_estimators=n_est, **P).fit(lab[fc], lab["y"], group=T._groups(lab))
        m.booster_.save_model(str(f))
        log(f"  seed {s} done")
    json.dump({"features": fc, "n_estimators": n_est, "rule": "mean best iteration of the 18 note-95 fold fits",
               "trained_on": {"labelled_pair_rows": int(len(lab)), "signals": int(lab.DeviceId.nunique()),
                              "period": "Sept-2026 only"}}, open(d / "phase_fit95.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "trees", "decode", "v4f", "score", "fitdec", "fitdec_trees", "fitranker"])
    ap.add_argument("--tcn", default="p95_ad")
    ap.add_argument("--main", default="")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
