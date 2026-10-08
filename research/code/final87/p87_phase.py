"""Note 87b A: TCN vs GRU as the phase network inside the CURRENT phase pipeline (saved OOF only, CPU).

Pipeline = note 76 champion (f76_phase.py 'new' arm): note-57 ranker on the phase-order-free pool (f76/phase, 3 seeds,
p0_bag) 0.5 / 0.5 with a network BEFORE the joint decoder, network used only on candidates with tree p >= .01 (note 73b,
renormalised over the kept candidates), decoder re-fitted OOF on folds_v4 (cand64.decode, threads 4).
Networks (all six-fold OOF on the note-37 pool / fold map, K = 4 pieces past 2 h, identical pair coverage):
  gru      p3 (current; = f76_q.p2_new, reused; 'gru_redecode' re-decodes it at 4 threads as a check)
  tcn_ad   tcn53 ad_all_e100 (1 s bins, 15 channels), seed 0
  tcn_r05  tcn53 r05_all (0.5 s bins, 15 channels), seed 0
  *_s1f0   fold 0's net replaced by its seed-1 twin (ad_all_s1 / r05_all_s1; ad_all s0 cap 50 == e100) -> seed check
Scored with cand64 / t57 rows (everything / realistic, v4l row sets), pools >= 30 min, 5, 10 min, per family (pooled
rows), per fold (>= 30 min), paired signal bootstrap vs gru.  Locked_v2 asserted absent.

    python p87_phase.py decode [--arms tcn_ad,tcn_r05,...]   -> %DC_WORK%/s87/p87_q.parquet
    python p87_phase.py score                                -> %DC_WORK%/s87/p87_phase.json
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

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import cand64 as C  # noqa: E402
import t57_phase as T57  # noqa: E402
import trackb_eval as TE  # noqa: E402

C.NJ = 4
F76P = C.DC_WORK / "final_v3_work" / "f76" / "phase"
OUT = C.DC_WORK / "s87"
PRED = C.DC_WORK / "tcn53" / "preds"
TCN = {"tcn_ad": ("ad_all_e100", "ad_all_s1"), "tcn_r05": ("r05_all", "r05_all_s1"),
       "tcn_ad76": ("ad_all_e100n76", "ad_all_s1")}   # note 86: same fold models re-inferred with the note-76 tcn53.py
FAMS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def base():
    q = pd.read_parquet(C.OUT / "phase_oof.parquet", columns=C.KEY4 + ["p_nn"])
    new = pd.read_parquet(F76P / "full" / "oof.parquet", columns=C.KEY4 + ["p0_bag"])
    new = q[C.KEY4].merge(new, on=C.KEY4, how="left")
    assert new.p0_bag.notna().all()
    q["p0t"] = new.p0_bag.to_numpy()
    return q


def tcn_probs(q, arm: str) -> np.ndarray:
    root = arm.replace("_s1f0", "")
    nm, s1 = TCN[root]
    fs = []
    for k in range(6):
        tag = s1 if (k == 0 and arm.endswith("_s1f0")) else nm
        fs.append(PRED / f"{tag}_f{k}_bywindow.parquet")
    t = pd.concat([TE._norm(pd.read_parquet(f))[C.KEY4 + ["prob"]] for f in fs], ignore_index=True)
    assert not t.duplicated(C.KEY4).any()
    k = q[C.KEY4].copy()
    k["DeviceId"] = k.DeviceId.str.lower()
    t = t.astype({"Detector": k.Detector.dtype, "cand_phase": k.cand_phase.dtype})
    p = k.merge(t, on=C.KEY4, how="left").prob.to_numpy()
    assert (np.isnan(p) == q.p_nn.isna().to_numpy()).all(), "coverage differs from the GRU"
    return p


def blend(q, p0, nn, thr=0.01):
    grp = [q[c] for c in C.DET]
    nn = np.where(p0 >= thr, nn, np.nan)
    tot = pd.Series(np.nan_to_num(nn)).groupby(grp).transform("sum").to_numpy()
    nn = np.where(~np.isnan(nn) & (tot > 0), nn / np.where(tot > 0, tot, 1.0), np.nan)
    p = np.where(np.isnan(nn), p0, 0.5 * p0 + 0.5 * nn)
    return p / pd.Series(p).groupby(grp).transform("sum").to_numpy()


def stage_decode(a):
    need = C.KEY4 + ["fold", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(F76P / "pool.parquet", columns=need)
    simc = pd.read_parquet(F76P / "pool_sim.parquet")
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    q = base()
    for c in C.KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all(), c
    OUT.mkdir(parents=True, exist_ok=True)
    fq = OUT / "p87_q.parquet"
    done = pd.read_parquet(fq) if fq.exists() else q[C.KEY4].copy()
    if "p2_gru" not in done:
        f76 = pd.read_parquet(F76P / "f76_q.parquet")
        for c in C.KEY4:
            assert (f76[c].to_numpy() == q[c].to_numpy()).all(), c
        done["p2_gru"] = f76.p2_new.to_numpy()
        done["pn_gru"] = q.p_nn.to_numpy()
        done.to_parquet(fq, index=False)
    for arm in a.arms.split(","):
        if f"p2_{arm}" in done:
            log(f"{arm} cached"); continue
        t0 = time.time()
        if arm == "gru_redecode":
            done["p2_gru_redecode"] = C.decode(comb, simc, blend(q, q.p0t.to_numpy(), q.p_nn.to_numpy()))
        else:
            nn = tcn_probs(q, arm)
            done[f"pn_{arm}"] = nn
            done[f"p2_{arm}"] = C.decode(comb, simc, blend(q, q.p0t.to_numpy(), nn))
        done.to_parquet(fq, index=False)
        log(f"{arm} decoded in {time.time()-t0:.0f}s")


def stage_score(a):
    q = base()
    done = pd.read_parquet(OUT / "p87_q.parquet")
    for c in C.KEY4:
        assert (done[c].to_numpy() == q[c].to_numpy()).all(), c
    q = q.join(done.drop(columns=C.KEY4))
    q["p0t_n"] = q.p0t
    for c in [c for c in q.columns if c.startswith("pn_")]:          # net alone, renormalised per det-window
        s = q.groupby(C.DET)[c].transform("sum")
        q[c] = np.where(s > 0, q[c] / s, np.nan)
    rows = C.phase_rows()
    rows = rows[[c for c in rows.columns if not c.startswith(("pred_", "okE_", "okR_", "p_"))]].copy()
    sig = rows.dev_plain.to_numpy()
    alt = T57.score_rows().set_index(C.DET).alt.reindex(rows.set_index(C.DET).index).to_numpy()
    arms = {c[3:]: c for c in q.columns if c.startswith("p2_")}
    arms.update({"net_" + c[3:]: c for c in q.columns if c.startswith("pn_")})
    arms["trees_pre_decode"] = "p0t_n"
    for nm, col in arms.items():
        pred, pr = C.top1(q, col, rows)
        has = ~pd.isna(pr) if nm.startswith("net_") else np.ones(len(rows), bool)
        okE = pred == rows.Phase.to_numpy()
        okR = okE | np.array([p in s_ for p, s_ in zip(pred, alt)])
        rows[f"okE_{nm}"] = np.where(has, okE, np.nan)
        rows[f"okR_{nm}"] = np.where(has, okR, np.nan)
    res = {"acc": {}, "delta_vs_gru": {}, "by_fold_ge30": {}, "n": {}}
    pools = dict(C.POOLS, **{f: [f] for f in FAMS})
    dec_arms = [nm for nm in arms if not nm.startswith(("net_", "trees"))]
    for sname, okp in (("everything", "okE"), ("realistic", "okR")):
        mset = rows[sname].to_numpy()
        for pool, fams in pools.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            res["n"].setdefault(sname, {})[pool] = [int(m.sum()), int(len(set(sig[m])))]
            for nm in arms:
                v = rows[f"{okp}_{nm}"].to_numpy()
                mm = m & ~np.isnan(v)
                res["acc"].setdefault(nm, {}).setdefault(sname, {})[pool] = C.acc_ci(v[mm], sig[mm])
            g = rows[f"{okp}_gru"].to_numpy()
            for nm in dec_arms:
                if nm != "gru":
                    res["delta_vs_gru"].setdefault(nm, {}).setdefault(sname, {})[pool] = C.delta_ci(
                        g[m], rows[f"{okp}_{nm}"].to_numpy()[m], sig[m])
            ng = rows[f"{okp}_net_gru"].to_numpy()
            mm = m & ~np.isnan(ng)
            for nm in [n for n in arms if n.startswith("net_tcn")]:
                res["delta_vs_gru"].setdefault(nm + "_vs_net_gru", {}).setdefault(sname, {})[pool] = C.delta_ci(
                    ng[mm], rows[f"{okp}_{nm}"].to_numpy()[mm], sig[mm])
        m30 = mset & rows.fam.isin(C.GE30).to_numpy()
        for k in range(6):
            m = m30 & (rows.fold.to_numpy() == k)
            res["by_fold_ge30"].setdefault(sname, {})[k] = {
                "n": int(m.sum()), **{nm: round(float(np.nanmean(rows[f"{okp}_{nm}"].to_numpy()[m])), 4) for nm in arms}}
            for nm in dec_arms:
                if nm != "gru":
                    res["by_fold_ge30"][sname][k][f"d_{nm}"] = C.delta_ci(
                        rows[f"{okp}_gru"].to_numpy()[m], rows[f"{okp}_{nm}"].to_numpy()[m], sig[m])
    tops = {}
    for nm in dec_arms:
        col = arms[nm]
        d = q[C.KEY4 + [col]].sort_values(C.DET + [col], ascending=[True, True, True, False])
        tops[nm] = d.groupby(C.DET, sort=True).first().cand_phase
    fam = tops["gru"].index.get_level_values("win").map(C.fam_of)
    res["top_change_vs_gru"] = {nm: {p: round(float((tops[nm] != tops["gru"]).to_numpy()[fam.isin(f)].mean()), 5)
                                     for p, f in C.POOLS.items()} for nm in tops if nm != "gru"}
    json.dump(res, open(OUT / "p87_phase.json", "w"), indent=1)
    rows[[c for c in rows.columns if c.startswith(("okE_", "okR_"))] + C.DET + ["fold", "fam", "dev_plain",
                                                                              "everything", "realistic"]] \
        .to_parquet(OUT / "p87_rows.parquet", index=False)
    for nm in arms:
        log(f"{nm:22s} " + " | ".join(f"{s[:1]} {p} {res['acc'][nm][s][p]}" for s in ("everything", "realistic")
                                      for p in C.POOLS))
    for nm, v in res["delta_vs_gru"].items():
        log(f"d {nm}: " + json.dumps({s: {p: v[s][p] for p in C.POOLS} for s in v}))
    log(json.dumps(res["by_fold_ge30"]["everything"]))
    log(json.dumps(res["top_change_vs_gru"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["decode", "score"])
    ap.add_argument("--arms", default="tcn_ad,tcn_r05,tcn_ad_s1f0,tcn_r05_s1f0")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
