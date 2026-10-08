"""Note 83: six-fold OOF headline of the candidate-v4 function recipe (v4l truth), against the v4l champion.

v4 recipe, as far as OOF models exist:
  trees    229 features x 3 seeds trained on v4l (note 80, f76/function_c_v4l)                    = production recipe
  net      ONE siba member: each of the three single-seed siba OOF fold models in turn (x69_siba, v3s-trained;
           production = one full-data siba trained on v4l -- no six-fold v4l siba exists)
  stacker  'single' variant (trained on the three single-seed net versions of every row, 3 x rows), 47 columns =
           the 15 health columns removed, v4l target, fold k's stacker never sees fold k, stacker seeds 0/1/2 averaged
  decode   note-77 lanes D (OOF), gate .9, stack pick (pk_unhealthy = OOF ln7 stack health), twin decode < 30 min
Arms (each scored per net seed; headline = mean over the three net seeds of the per-row correctness):
  v4      the recipe above
  v4h     same with the 62 columns (health kept) -> isolates the health-column drop
Baselines: champion = note-81 v4l headline (stacker 3-seed-mean siba, 62 columns, trees v3s; lab80/ok_v4l_champ);
           v4l-refit champion (note 80 B: trees + stacker on v4l, 62 columns, 3-seed-mean siba; lab80/ok_v4l_v4l).
Paired signal bootstrap (2,000 draws), pools >= 30 min / 10 / 5 min, sets E and R.

    set F76_ARM=c & python oof83.py stack     -> %DC_WORK%/final_v3_work/f83/oof/P_<arm>_net<s>.npy
    set F76_ARM=c & python oof83.py score     -> %DC_WORK%/final_v3_work/f83/oof/oof83.json
CPU, LightGBM 4 threads.  locked_v2 asserted absent (frame loaders + fit83.stacker_inputs).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit83 as F  # noqa: E402

OUT = F.F83 / "oof"
LAB80 = F.DC_WORK / "lab80"


def _inputs(with_health: bool):
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    if with_health:                                     # rebuild the 62-column version (same rows / order)
        import cand64 as C
        fr, Pt = E["fr"], E["Pt"]
        Xs = {}
        for s in (0, 1, 2):
            P1 = s74.net_probs(fr, f"x69_siba:{s}")
            P1 = np.where(np.isnan(P1[:, :1]), Pt, P1)
            Xs[s] = np.hstack([S.stack_X(fr, Pt, P1), S.ctx_X(E, C.W_TREE * Pt + (1 - C.W_TREE) * P1)])
    return s74, S, E, Xs, y, trm


def check_trees(E):
    """the stacker's tree input is the note-80 v4l OOF (3-seed mean), not the v3s one."""
    import s59_step6 as S59
    k = S59.frame_keys()
    P = S59.std_probs(k, (0, 1, 2), F.CFGD)
    k["_i"] = np.arange(len(k))
    fr = E["fr"]
    idx = fr[S59.KEY].merge(k.astype({"Detector": fr.Detector.dtype})[S59.KEY + ["_i"]], on=S59.KEY, how="left")._i
    d = float(np.abs(P[idx.to_numpy(int)] - E["Pt"]).max())
    assert d < 1e-6, f"stacker tree input is not the v4l OOF (max |diff| {d})"
    return d


def stage_stack():
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, wh in (("v4", False), ("v4h", True)):
        if all((OUT / f"P_{arm}_net{s}.npy").exists() for s in range(3)):
            continue
        s74, S, E, Xs, y, trm = _inputs(wh)
        F.log(f"{arm}: tree input = v4l OOF (max |diff| {check_trees(E):.1e}); X {Xs[0].shape}")
        fo = E["fr"].fold.to_numpy()
        P = {s: np.zeros((len(y), 7)) for s in range(3)}
        t0 = time.time()
        for seed in (0, 1, 2):
            for k in range(6):
                tr, te = trm & (fo != k), fo == k
                X = np.vstack([Xs[s][tr] for s in range(3)])
                yy = np.concatenate([y[tr]] * 3)
                m = lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=seed), lgb.Dataset(X, yy), num_boost_round=150)
                for s in range(3):
                    P[s][te] += m.predict(Xs[s][te]) / 3
            F.log(f"  {arm} stacker seed {seed} done ({time.time() - t0:.0f}s)")
        for s in range(3):
            np.save(OUT / f"P_{arm}_net{s}.npy", P[s].astype(np.float32))


def stage_score():
    import cand64 as C
    import of77
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    base = {k: np.load(LAB80 / f"ok_{k}.npz") for k in ("v4l_champ", "v4l_v4l")}
    for z in base.values():
        assert (z["sig"] == fr.DeviceId.to_numpy(str)).all() and (z["win"] == fr.win.to_numpy(str)).all()
        assert (z["det"] == fr.Detector.to_numpy(int)).all()
    ok = {}
    for arm in ("v4", "v4h"):
        per = [s74.gate_ok(E, np.load(OUT / f"P_{arm}_net{s}.npy").astype(float), lc) for s in range(3)]
        ok[arm] = {st: np.mean([p[st] for p in per], 0) for st in ("E", "R")}
        ok[arm]["seeds"] = per
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    res = {"caveats": "net = single-seed v3s-trained siba fold models (production: full-data v4l siba); lanes = note-77 "
                      "OOF (v3s lane truth); headline = mean over the 3 net seeds of per-row correctness"}
    for st in ("E", "R"):
        for pool, fams in C.POOLS.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            r = {}
            for arm in ("v4", "v4h"):
                a = ok[arm][st]
                m = pm & ~np.isnan(a)
                r[arm] = C.acc_ci(a[m], sig[m])
                r[f"{arm}_per_net_seed"] = [round(float(np.nanmean(p[st][pm])), 4) for p in ok[arm]["seeds"]]
            for bn, z in base.items():
                b0 = z[st].astype(float)
                m = pm & ~np.isnan(b0)
                r[bn] = C.acc_ci(b0[m], sig[m])
                for arm in ("v4", "v4h"):
                    a = ok[arm][st]
                    mm = m & ~np.isnan(a)
                    r[f"{arm}_minus_{bn}"] = C.delta_ci(b0[mm], a[mm], sig[mm])
            a, b = ok["v4h"][st], ok["v4"][st]
            mm = pm & ~np.isnan(a) & ~np.isnan(b)
            r["v4_minus_v4h (health drop)"] = C.delta_ci(a[mm], b[mm], sig[mm])
            res[f"{st}_{pool}"] = r
        m = fr.wgroup.isin(C.GE30).to_numpy()
        b0 = base["v4l_champ"][st].astype(float)
        a = ok["v4"][st]
        res[f"{st}_ge30_by_class_v4_minus_champ"] = {
            c: [int((m & (cl == c) & ~np.isnan(a) & ~np.isnan(b0)).sum())] +
            C.delta_ci(b0[m & (cl == c) & ~np.isnan(a) & ~np.isnan(b0)], a[m & (cl == c) & ~np.isnan(a) & ~np.isnan(b0)],
                       sig[m & (cl == c) & ~np.isnan(a) & ~np.isnan(b0)])
            for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
    json.dump(res, open(OUT / "oof83.json", "w"), indent=1, default=str)
    for k, v in res.items():
        F.log(f"{k}: {v}")


if __name__ == "__main__":
    {"stack": stage_stack, "score": stage_score}[sys.argv[1]]()
