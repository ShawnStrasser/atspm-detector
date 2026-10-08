"""Note 84: six-fold OOF headline, v4l truth, of the two candidate-v4b function recipes WITH the siba candidate filter:
  single + filter   ONE siba member, stacker 'single' (fit83 recipe: 3 x rows of the single-seed OOF versions)
  mean3  + filter   THREE members averaged, stacker 'mean3' (fit84 recipe: 1 x rows of the 3-seed-mean OOF)
Both stackers: v4l target, v4l OOF trees, 47 columns, fold k's stacker never sees fold k, stacker seeds 0/1/2 averaged;
decode = note-77 lanes D (OOF), gate .9, stack pick, twin decode < 30 min (s74.gate_ok, as oof83).

Filtered OOF nets exist only for seed 0 (x74_sibaflt = x69_siba seed-0 fold models with the pair net only on candidates
with OOF tree p >= .01; x74_sibanf = the same models, same post-note-76 inputs, no filter; note 74b).  The GPU is not
used here, so seeds 1 / 2 are unfiltered (x69_siba_s1 / _s2).  Arms (stackers trained on the unfiltered x69_siba OOF,
as the package: production applies them to filtered net output):
  single_s{0,1,2}  single stacker, net = x69_siba seed s                         (= note 83 v4 arm)
  single_nf0       single stacker, net = x74_sibanf seed 0
  single_flt0      single stacker, net = x74_sibaflt seed 0                      <- single + filter
  mean3            mean3 stacker, net = mean(x69_siba seeds 0, 1, 2)
  mean3_nf0        mean3 stacker, net = mean(x74_sibanf 0, x69_siba 1, 2)
  mean3_flt0       mean3 stacker, net = mean(x74_sibaflt 0, x69_siba 1, 2)       <- mean3 + filter (1 of 3 filtered)
Note 84b (2026-10-04): the seed-1 / seed-2 fold models were re-inferred with the filter (x74_sibaflt_s1 / _s2, same
snapshot / inputs as seed 0) -> arms single_flt1 / single_flt2, 'single_flt' (= mean over the three seeds' correctness) and
  mean3_flt3       mean3 stacker, net = mean(x74_sibaflt 0, 1, 2)                <- mean3 + filter, all three filtered
Paired signal bootstrap (2,000 draws), pools >= 30 min / 10 / 5 min, sets E and R.

    set F76_ARM=c & python oof84.py stack    -> %DC_WORK%/final_v3_work/f84/oof/P_<arm>.npy
    set F76_ARM=c & python oof84.py score    -> %DC_WORK%/final_v3_work/f84/oof/oof84.json
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final83"))
import fit84 as F4  # noqa: E402
import fit83 as F  # noqa: E402

OUT = F.DC_WORK / "final_v3_work" / "f84" / "oof"
LAB80 = F.DC_WORK / "lab80"
SINGLE = ["single_s0", "single_s1", "single_s2", "single_nf0", "single_flt0", "single_flt1", "single_flt2"]
MEAN3 = ["mean3", "mean3_nf0", "mean3_flt0", "mean3_flt3"]


def stage_stack():
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    s74, S, E, Xs, Xm, y, trm, names = F4.mean3_inputs()
    fr = E["fr"]
    P = {s: s74.net_probs(fr, f"x69_siba:{s}") for s in range(3)}
    nf0, flt0 = s74.net_probs(fr, "x74_sibanf:0"), s74.net_probs(fr, "x74_sibaflt:0")
    flt1, flt2 = s74.net_probs(fr, "x74_sibaflt:1"), s74.net_probs(fr, "x74_sibaflt:2")
    F.log(f"coverage flt1 {np.mean(~np.isnan(flt1[:, 0])):.4f} flt2 {np.mean(~np.isnan(flt2[:, 0])):.4f}")
    F.log(f"coverage nf0 {np.mean(~np.isnan(nf0[:, 0])):.4f} flt0 {np.mean(~np.isnan(flt0[:, 0])):.4f}")
    assert np.allclose(np.nanmean(np.abs((P[0] + P[1] + P[2]) / 3 - s74.net_probs(fr, "x69_siba"))), 0, atol=1e-7)
    Xa = {"single_s0": Xs[0], "single_s1": Xs[1], "single_s2": Xs[2],
          "single_nf0": F4.net_X(s74, S, E, nf0), "single_flt0": F4.net_X(s74, S, E, flt0),
          "mean3": Xm,
          "mean3_nf0": F4.net_X(s74, S, E, (nf0 + P[1] + P[2]) / 3),
          "mean3_flt0": F4.net_X(s74, S, E, (flt0 + P[1] + P[2]) / 3),
          "single_flt1": F4.net_X(s74, S, E, flt1), "single_flt2": F4.net_X(s74, S, E, flt2),
          "mean3_flt3": F4.net_X(s74, S, E, (flt0 + flt1 + flt2) / 3)}
    fo = fr.fold.to_numpy()
    R = {a: np.zeros((len(y), 7)) for a in Xa}
    t0 = time.time()
    for seed in (0, 1, 2):
        for k in range(6):
            tr, te = trm & (fo != k), fo == k
            prm = dict(F.PRM0, num_threads=F.THREADS, seed=seed)
            ms = lgb.train(prm, lgb.Dataset(np.vstack([Xs[s][tr] for s in range(3)]), np.concatenate([y[tr]] * 3)),
                           num_boost_round=150)
            mm = lgb.train(prm, lgb.Dataset(Xm[tr], y[tr]), num_boost_round=150)
            for a, X in Xa.items():
                R[a][te] += (ms if a in SINGLE else mm).predict(X[te]) / 3
        F.log(f"  stacker seed {seed} done ({time.time() - t0:.0f}s)")
    old = np.load(F.F83 / "oof" / "P_v4_net0.npy")
    F.log(f"single_s0 vs note-83 P_v4_net0 max |diff| {np.abs(R['single_s0'] - old).max():.2e}")
    for a, v in R.items():
        if (OUT / f"P_{a}.npy").exists():
            F.log(f"{a} vs previous run max |diff| {np.abs(np.load(OUT / f'P_{a}.npy') - v).max():.2e}")
        np.save(OUT / f"P_{a}.npy", v.astype(np.float32))


def stage_score():
    import cand64 as C
    import of77
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    champ = np.load(LAB80 / "ok_v4l_champ.npz")
    assert (champ["sig"] == fr.DeviceId.to_numpy(str)).all() and (champ["win"] == fr.win.to_numpy(str)).all()
    ok = {a: s74.gate_ok(E, np.load(OUT / f"P_{a}.npy").astype(float), lc) for a in SINGLE + MEAN3}
    ok["single"] = {st: np.mean([ok[f"single_s{s}"][st] for s in range(3)], 0) for st in ("E", "R")}
    ok["single_flt"] = {st: np.mean([ok[f"single_flt{s}"][st] for s in range(3)], 0) for st in ("E", "R")}
    ok["champion"] = {st: champ[st].astype(float) for st in ("E", "R")}
    arms = ["single", "single_nf0", "single_flt0", "single_flt", "mean3", "mean3_nf0", "mean3_flt0", "mean3_flt3",
            "champion"]
    contrasts = [("single_flt0", "mean3_flt0"), ("single", "mean3"), ("single_nf0", "mean3_nf0"),
                 ("single_nf0", "single_flt0"), ("mean3_nf0", "mean3_flt0"), ("champion", "mean3_flt0"),
                 ("champion", "single_flt0"), ("champion", "mean3"),
                 ("single_flt0", "mean3_flt3"), ("single_flt", "mean3_flt3"), ("mean3_flt0", "mean3_flt3"),
                 ("mean3", "mean3_flt3"), ("champion", "mean3_flt3"), ("single", "single_flt")]
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    res = {"caveats": "mean3_flt0: seed 0 filtered only; mean3_flt3: all three filtered (note 84b); OOF nets v3s-trained "
                      "(production: full-data v4l); stackers trained on unfiltered OOF; lanes = note-77 OOF"}
    for st in ("E", "R"):
        for pool, fams in C.POOLS.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            sc = pm & np.logical_and.reduce([~np.isnan(ok[a][st]) for a in arms])
            r = {"n": int(sc.sum())}
            for a in arms:
                r[a] = C.acc_ci(ok[a][st][sc], sig[sc])
            for a, b in contrasts:
                r[f"{b} - {a}"] = C.delta_ci(ok[a][st][sc], ok[b][st][sc], sig[sc])
            res[f"{st}_{pool}"] = r
        m = fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok["mean3_flt0"][st]) & ~np.isnan(ok["single_flt0"][st])
        res[f"{st}_ge30_by_class_mean3flt0_minus_singleflt0"] = {
            c: [int((m & (cl == c)).sum())] + C.delta_ci(ok["single_flt0"][st][m & (cl == c)],
                                                        ok["mean3_flt0"][st][m & (cl == c)], sig[m & (cl == c)])
            for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        for ref in ("single_flt0", "single_flt", "champion"):
            m = fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok["mean3_flt3"][st]) & ~np.isnan(ok[ref][st])
            res[f"{st}_ge30_by_class_mean3flt3_minus_{ref}"] = {
                c: [int((m & (cl == c)).sum())] + C.delta_ci(ok[ref][st][m & (cl == c)],
                                                            ok["mean3_flt3"][st][m & (cl == c)], sig[m & (cl == c)])
                for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
    json.dump(res, open(OUT / "oof84.json", "w"), indent=1, default=str)
    for k, v in res.items():
        F.log(f"{k}: {v}")


if __name__ == "__main__":
    {"stack": stage_stack, "score": stage_score}[sys.argv[1]]()
