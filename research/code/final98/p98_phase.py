"""Note 98: phase arms of the simplified (fast) variants, six-fold OOF from saved predictions only (CPU, no network run).

Same pipeline / pool / scorer as note 90 (p90_phase / p87_phase): f76 trees (3 seeds) -> [TCN ad_all on candidates with
tree p >= .01, 0.5 / 0.5] -> joint decoder re-fitted six-fold OOF on folds_v4 (note-57 recipe).  Arms:
  tcn_ad76   the v4f pipeline (copied from s90, = the package)                               <- full
  trees_dec  no phase network at all: the decoder re-fitted on the trees' probabilities alone
  tcn_le2h   the network only on samples <= 2 h (families m5 / m10 / m30 / h1, as the beta); trees alone above; ONE
             decoder re-fitted on that mixed input
The stage `fitdec` writes the full-data decoder of an arm (fit76 recipe, n = mean of the six fold iterations).

    python p98_phase.py decode   -> %DC_WORK%/s98/p87_q.parquet, s98/dec_iters.json
    python p98_phase.py score    -> %DC_WORK%/s98/p98_phase.json (+ p87_rows.parquet: ok columns per arm)
    python p98_phase.py fitdec --arm trees_dec|tcn_le2h -> %DC_WORK%/final_v3_work/f98/decoder_<arm>/decode_v3.{txt,json}
locked_v2 asserted absent (p90 _pool).  LightGBM 4 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("P90_THREADS", "4")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final90"))
import p90_phase as P90  # noqa: E402

C, P87 = P90.C, P90.P87
OUT = C.DC_WORK / "s98"
SHORT = ("m5", "m10", "m30", "h1")            # <= 2 h: where the beta ran its network (cand64.V2_NET)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def arm_p0(q, arm):
    p0t = q.p0t.to_numpy()
    if arm == "trees_dec":
        return p0t
    nn = P87.tcn_probs(q, "tcn_ad76")
    full = P87.blend(q, p0t, nn)
    if arm == "tcn_le2h":
        short = q.win.map(C.fam_of).isin(SHORT).to_numpy()
        return np.where(short, full, p0t)
    raise ValueError(arm)


def stage_decode(a):
    comb, simc = P90._pool()
    q = P87.base()
    OUT.mkdir(parents=True, exist_ok=True)
    fq, fi = OUT / "p87_q.parquet", OUT / "dec_iters.json"
    if fq.exists():
        done = pd.read_parquet(fq)
        its = json.load(open(fi))
    else:
        s90 = pd.read_parquet(C.DC_WORK / "s90" / "p87_q.parquet",
                              columns=C.KEY4 + ["p2_gru", "pn_gru", "p2_tcn_ad76", "pn_tcn_ad76"])
        for c in C.KEY4:
            assert (s90[c].to_numpy() == q[c].to_numpy()).all(), c
        done = s90
        its = {"tcn_ad76": json.load(open(C.DC_WORK / "s90" / "dec_iters.json"))["tcn_ad76"]}
    for arm in a.arms.split(","):
        if f"p2_{arm}" in done and not a.force:
            log(f"{arm} cached")
            continue
        t0 = time.time()
        done[f"p2_{arm}"], its[arm] = P90.decode_iters(comb, simc, arm_p0(q, arm))
        done.to_parquet(fq, index=False)
        json.dump(its, open(fi, "w"), indent=1)
        log(f"{arm} decoded in {time.time() - t0:.0f}s, iterations {its[arm]}")


def stage_score(a):
    q = pd.read_parquet(OUT / "p87_q.parquet")
    if "p2_le2h" not in q and "p2_trees_dec" in q:
        # what the fast package does: blend decoder where the network ran (<= 2 h), trees-only decoder above
        short = q.win.map(C.fam_of).isin(SHORT).to_numpy()
        q["p2_le2h"] = np.where(short, q.p2_tcn_ad76.to_numpy(), q.p2_trees_dec.to_numpy())
        q.to_parquet(OUT / "p87_q.parquet", index=False)
    P87.OUT = OUT
    P87.stage_score(a)                         # writes p87_phase.json + p87_rows.parquet (okE_/okR_ per arm)
    rows = pd.read_parquet(OUT / "p87_rows.parquet")
    sig = rows.dev_plain.to_numpy()
    res = {}
    arms = [c[4:] for c in rows.columns if c.startswith("okE_")]
    for sname, okp in (("everything", "okE"), ("realistic", "okR")):
        mset = rows[sname].to_numpy()
        for pool, fams in dict(C.POOLS, **{f: [f] for f in P87.FAMS}).items():
            m = mset & rows.fam.isin(fams).to_numpy()
            ref = rows[f"{okp}_tcn_ad76"].to_numpy()
            for nm in arms:
                v = rows[f"{okp}_{nm}"].to_numpy()
                mm = m & ~np.isnan(v)
                r = res.setdefault(nm, {}).setdefault(sname, {})
                r[pool] = {"acc": C.acc_ci(v[mm], sig[mm]), "n": int(mm.sum())}
                if not nm.startswith("net_") and nm != "tcn_ad76":
                    mm2 = mm & ~np.isnan(ref)
                    r[pool]["d_vs_full"] = C.delta_ci(ref[mm2], v[mm2], sig[mm2])
    q = pd.read_parquet(OUT / "p87_q.parquet")
    tops = {}
    for col in [c for c in q.columns if c.startswith("p2_")]:
        d = q[C.KEY4 + [col]].sort_values(C.DET + [col], ascending=[True, True, True, False])
        tops[col[3:]] = d.groupby(C.DET, sort=True).first().cand_phase
    fam = tops["tcn_ad76"].index.get_level_values("win").map(C.fam_of)
    res["top_change_vs_full"] = {nm: {p: round(float((tops[nm] != tops["tcn_ad76"]).to_numpy()[fam.isin(f)].mean()), 5)
                                      for p, f in C.POOLS.items()} for nm in tops if nm != "tcn_ad76"}
    json.dump(res, open(OUT / "p98_phase.json", "w"), indent=1)
    for nm in res:
        if nm == "top_change_vs_full":
            log(f"top change vs full: {res[nm]}")
            continue
        log(f"{nm:18s} " + " | ".join(f"{s[:1]} {p} {res[nm][s][p]['acc']} d {res[nm][s][p].get('d_vs_full')}"
                                      for s in ("everything", "realistic") for p in C.POOLS))


def stage_fitdec(a):
    """full-data decoder for an arm's first-stage input (p90 fitdec recipe)."""
    import lightgbm as lgb
    import decode_train as dec
    import train_official as T
    comb, simc = P90._pool()
    q = P87.base()
    its = json.load(open(OUT / "dec_iters.json"))[a.arm]
    p0 = arm_p0(q, a.arm)
    cols2 = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    pr = comb[C.KEY4].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[C.KEY4 + ["Phase", "y"]], on=C.KEY4, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    labm = X.Phase.notna().to_numpy()
    n_est = int(round(np.mean(its)))
    P = dict(T.BIN_PARAMS, n_jobs=4, bagging_seed=0, feature_fraction_seed=100, seed=0)
    P.pop("n_estimators")
    fit = C.DC_WORK / "final_v3_work" / "f98" / f"decoder_{a.arm}"
    fit.mkdir(parents=True, exist_ok=True)
    m = lgb.LGBMClassifier(n_estimators=n_est, **P).fit(X.loc[labm, cols2], X.y[labm])
    m.booster_.save_model(str(fit / "decode_v3.txt"))
    np.save(fit / "X_check.npy", X.loc[labm, cols2].to_numpy(np.float64)[:2000])
    first = {"trees_dec": "note-76 trees bag (3 seeds, phase-order-free features) alone, no phase network",
             "tcn_le2h": "note-76 trees bag 0.5 / TCN ad_all 0.5 (tree p >= .01) on samples <= 2 h; trees alone above"}
    json.dump({"features": cols2, "mode": "binary", "temperature": 1.0, "n_estimators": n_est,
               "n_estimators_rule": f"mean best iteration of the six note-98 fold fits {its}",
               "params": {k: v for k, v in P.items() if k != "n_jobs"}, "first_stage_input": first[a.arm],
               "arm": a.arm, "trained_on": {"labelled_pair_rows": int(labm.sum())}},
              open(fit / "decode_v3.json", "w"), indent=1)
    log(f"decoder ({a.arm}): n_estimators {n_est}, {int(labm.sum()):,} labelled rows -> {fit}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["decode", "score", "fitdec"])
    ap.add_argument("--arms", default="trees_dec,tcn_le2h")
    ap.add_argument("--arm", default="trees_dec")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
