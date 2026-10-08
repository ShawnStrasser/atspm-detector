"""Note 76: phase refit on the phase-order-free pool and scoring inside the current champion.

    python f76_phase.py fit      # note-57 ranker recipe, 3 seeds x 6 folds (t57_phase.stage_run, cfg full, OUT -> f76/phase)
    python f76_phase.py score    # champion pipeline on the new trees: 0.5 / 0.5 GRU p3 blend on candidates with tree
                                 # p >= .01 (note 73b), decoder re-fitted OOF (cand64.decode); vs the current champion
                                 # (prof73/oof73b_q.parquet p2_thr0.01), >= 30 min / 5 / 10 min, E and R, paired signal
                                 # bootstrap; + trees alone vs trees alone; + share of detector-windows whose top phase moves.
Locked_v2 asserted absent (pool + cand64).  CPU, 6 threads.
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

F76P = C.DC_WORK / "final_v3_work" / "f76" / "phase"
C.NJ = 6
T57.NJ = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stage_fit(a):
    T57.OUT = F76P
    comb, simc, fc = T57.load_pool()
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    del comb, simc
    ns = argparse.Namespace(cfg="full", seeds="0,1,2", n_noise=0, decode_seeds=False)
    T57.stage_run(ns)


def blend(q: pd.DataFrame, p0: np.ndarray, keep: np.ndarray) -> np.ndarray:
    grp = [q[c] for c in C.DET]
    nn = np.where(keep, q.p_nn.to_numpy(), np.nan)
    tot = pd.Series(np.nan_to_num(nn)).groupby(grp).transform("sum").to_numpy()
    nn = np.where(~np.isnan(nn) & (tot > 0), nn / np.where(tot > 0, tot, 1.0), np.nan)
    p = np.where(np.isnan(nn), p0, 0.5 * p0 + 0.5 * nn)
    return p / pd.Series(p).groupby(grp).transform("sum").to_numpy()


def stage_score(a):
    need = C.KEY4 + ["fold", "Phase", "y", "cand_green_share", "call43_per_cycle", "det_n_on", "log_win_hours"]
    comb = pd.read_parquet(F76P / "pool.parquet", columns=need)
    simc = pd.read_parquet(F76P / "pool_sim.parquet")
    assert not C.plain(comb.DeviceId).isin(C.locked()).any()
    q = pd.read_parquet(C.OUT / "phase_oof.parquet")               # cand64: p0_bag (old trees), p_nn (GRU p3), p2_cand
    for c in C.KEY4:
        assert (q[c].to_numpy() == comb[c].to_numpy()).all(), c
    ref = pd.read_parquet(C.DC_WORK / "prof73" / "oof73b_q.parquet")
    for c in C.KEY4:
        assert (ref[c].to_numpy() == q[c].to_numpy()).all(), c
    new = pd.read_parquet(F76P / "full" / "oof.parquet")
    new = q[C.KEY4].merge(new, on=C.KEY4, how="left")
    assert new.p0_bag.notna().all()
    fq = F76P / "f76_q.parquet"
    done = pd.read_parquet(fq) if fq.exists() else q[C.KEY4].copy()
    if "p2_new" not in done:
        keep = new.p0_bag.to_numpy() >= 0.01
        done["p0_new"] = new.p0_bag.to_numpy()
        done["p2_new"] = C.decode(comb, simc, blend(q, new.p0_bag.to_numpy(), keep))
        done.to_parquet(fq, index=False)
    q["p2_champ"] = ref["p2_thr0.01"].to_numpy()
    q["p2_new"] = done.p2_new.to_numpy()
    q["p0t_old"] = q.p0_bag.to_numpy()
    q["p0t_new"] = new.p0_bag.to_numpy()
    q["p2t_new"] = new.p2_bag.to_numpy()
    old_t = pd.read_parquet(C.T57P / "full" / "oof.parquet")
    old_t = q[C.KEY4].merge(old_t, on=C.KEY4, how="left")
    q["p2t_old"] = old_t.p2_bag.to_numpy()
    rows = C.phase_rows()
    sig = rows.dev_plain.to_numpy()
    alt = T57.score_rows().set_index(C.DET).alt.reindex(rows.set_index(C.DET).index).to_numpy()
    arms = {"champ": "p2_champ", "new": "p2_new", "trees_old": "p2t_old", "trees_new": "p2t_new"}
    for nm, col in arms.items():
        rows[f"pred_{nm}"], _ = C.top1(q, col, rows)
        rows[f"okE_{nm}"] = (rows[f"pred_{nm}"] == rows.Phase).astype(float)
        rows[f"okR_{nm}"] = (rows[f"okE_{nm}"].astype(bool) |
                             np.array([p in s_ for p, s_ in zip(rows[f"pred_{nm}"], alt)])).astype(float)
    res = {"acc": {}, "delta": {}}
    for sname, okp in (("everything", "okE"), ("realistic", "okR")):
        mset = rows[sname].to_numpy()
        for pool, fams in C.POOLS.items():
            m = mset & rows.fam.isin(fams).to_numpy()
            for nm in arms:
                res["acc"].setdefault(nm, {}).setdefault(sname, {})[pool] = C.acc_ci(rows[f"{okp}_{nm}"].to_numpy()[m], sig[m])
            res["delta"].setdefault("new_vs_champ", {}).setdefault(sname, {})[pool] = C.delta_ci(
                rows[f"{okp}_champ"].to_numpy()[m], rows[f"{okp}_new"].to_numpy()[m], sig[m])
            res["delta"].setdefault("trees_new_vs_old", {}).setdefault(sname, {})[pool] = C.delta_ci(
                rows[f"{okp}_trees_old"].to_numpy()[m], rows[f"{okp}_trees_new"].to_numpy()[m], sig[m])
            res.setdefault("pred_changed_share", {}).setdefault(sname, {})[pool] = float(
                (rows.pred_new.to_numpy()[m] != rows.pred_champ.to_numpy()[m]).mean())
    # top-phase change on every detector-window (the function's phase input)
    tops = {}
    for nm, col in (("champ", "p2_champ"), ("new", "p2_new")):
        d = q[C.KEY4 + [col]].sort_values(C.DET + [col], ascending=[True, True, True, False])
        tops[nm] = d.groupby(C.DET, sort=True).first().cand_phase
    fam = tops["champ"].index.get_level_values("win").map(C.fam_of)
    res["top_change_share_all_detwin"] = {pool: float((tops["new"] != tops["champ"]).to_numpy()[fam.isin(f)].mean())
                                          for pool, f in C.POOLS.items()}
    res["max_abs_prob_change"] = float(np.nanmax(np.abs(q.p2_new - q.p2_champ)))
    json.dump(res, open(F76P / "score76.json", "w"), indent=1)
    for nm in arms:
        log(f"{nm:10s} " + " | ".join(f"{s[:4]} {p} {res['acc'][nm][s][p]}" for s in ("everything", "realistic")
                                      for p in C.POOLS))
    for k, v in res["delta"].items():
        log(f"{k}: {json.dumps(v)}")
    log(f"pred changed: {res['pred_changed_share']}; top change all: {res['top_change_share_all_detwin']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "score"])
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
