"""Note 79b: is detector health useful for CLASSIFICATION?  Ablations inside the note-77 champion
(trees arm c + siba -> context stacker f77, 3 seeds -> D lanes, gate .9, stack pick, twin decode; truth_v3s).

Health enters the champion in two places:
  1. stacker context: 15 health columns (health_core_ff: score, status, n_families, 7 family scores; stack-relative
     chi / surge / drop / chat / rel_chi) + pk_unhealthy (the pick's health flag).  hf_clus_n (stack size) is
     structure, not health, and is kept.
  2. stack pick rule: an unhealthy stack member loses the ATSPM claim (fr.pk_unhealthy).
Arms (six folds OOF, stacker seeds 0/1/2 averaged, scored with the champion decode, paired vs the saved f77 stack):
  a   stacker without the 16 health columns (all other context kept)
  b   stacker with the 16 health columns shuffled jointly within fold
  c   champion stacker, pick without the health key (pk_unhealthy = False in the decode)
  ac  a + c (health removed everywhere)
CPU, LightGBM threads 2.  locked_v2 asserted absent (setup).  Out %DC_WORK%/final_v3_work/f77/hc79b/.

    set F76_ARM=c & python hc79b.py stack|score
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import of77  # noqa: E402

OUT = of77.F77 / "hc79b"
N_STACKX = 20                     # s67_decider.stack_X columns; ctx_X follows
HF = ["hf_score", "hf_status", "hf_nfam", "hf_s_dropout", "hf_s_stuck", "hf_s_chatter", "hf_s_rapid", "hf_s_volume",
      "hf_s_level", "hf_s_choppy", "hf_chi", "hf_surge", "hf_drop", "hf_chat", "hf_rel_chi", "hf_clus_n"]
HEALTH_IX = [N_STACKX + i for i, c in enumerate(HF) if c != "hf_clus_n"] + [N_STACKX + len(HF) + 1]  # + pk_unhealthy


def build():
    s74, S, E = of77.setup_new()
    S.THREADS = 2
    import cand64 as C
    Pt, Pn = E["Pt"], E["Pn"]
    Pb = C.W_TREE * Pt + (1 - C.W_TREE) * Pn
    X = np.hstack([S.stack_X(E["fr"], Pt, Pn), S.ctx_X(E, Pb)])
    spec = json.load(open(of77.F77 / "s74" / "f77" / "spec.json"))
    assert X.shape[1] == spec["ncol"] == 62, (X.shape, spec)
    fr = E["fr"]
    assert np.array_equal(X[:, N_STACKX + len(HF) + 1], fr.pk_unhealthy.to_numpy(float))
    return s74, S, E, X


def stage_stack():
    OUT.mkdir(parents=True, exist_ok=True)
    s74, S, E, X = build()
    keep = np.setdiff1d(np.arange(X.shape[1]), HEALTH_IX)
    Xb = X.copy()
    fo = E["fr"].fold.to_numpy()
    rng = np.random.default_rng(79)
    for k in range(6):
        ix = np.flatnonzero(fo == k)
        Xb[np.ix_(ix, HEALTH_IX)] = X[np.ix_(rng.permutation(ix), HEALTH_IX)]
    arms = {"a": X[:, keep], "b": Xb, "x": X}      # x = full champion X, seed 0: 2-thread refit vs the saved 4-thread one
    for nm, XX in arms.items():
        for seed in ((0,) if nm == "x" else (0, 1, 2)):
            f = OUT / f"stack_{nm}_s{seed}.npy"
            if f.exists():
                continue
            t0 = time.time()
            np.save(f, S.oof_stack(E, XX, seed).astype(np.float32))
            S.log(f"{nm} seed {seed}: X {XX.shape} ({time.time() - t0:.0f}s)")


def stage_score():
    import cand64 as C
    s74, S, E, X = build()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    s74.OUT = of77.F77 / "s74"
    P0 = s74.load_stack("f77")
    unh0 = fr.pk_unhealthy.copy()

    def score(P, nopick_health=False):
        if nopick_health:
            fr["pk_unhealthy"] = False
        try:
            return s74.gate_ok(E, P, lc)
        finally:
            fr["pk_unhealthy"] = unh0
    ld = lambda nm, s: np.load(OUT / f"stack_{nm}_s{s}.npy").astype(float)  # noqa: E731
    ok0 = score(P0)
    oks = {"a": score(np.mean([ld("a", s) for s in range(3)], 0)),
           "b": score(np.mean([ld("b", s) for s in range(3)], 0)),
           "c": score(P0, True),
           "ac": score(np.mean([ld("a", s) for s in range(3)], 0), True)}
    per_seed = {nm: [score(ld(nm, s)) for s in range(3)] for nm in ("a", "b")}
    seeds0 = [score(np.load(of77.F77 / "s74" / "f77" / f"stack_ctx_s{s}.npy").astype(float)) for s in range(3)]
    okx = score(ld("x", 0))
    sig = fr.DeviceId.to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    hst = S._CTX["H"][:, HF.index("hf_status")]
    flag = (np.nan_to_num(hst) > 0) | fr.pk_unhealthy.to_numpy(bool)
    ge = fr.wgroup.isin(C.GE30).to_numpy()
    res = {"n_health_cols": len(HEALTH_IX),
           "flagged_share_ge30": round(float(flag[ge].mean()), 4),
           "pk_unhealthy_share_ge30": round(float(fr.pk_unhealthy.to_numpy(bool)[ge].mean()), 4)}
    for s in ("E", "R"):
        A = ok0[s]
        sc0 = ~np.isnan(A)
        res[f"{s}_champion"] = {p: C.acc_ci(A[sc0 & fr.wgroup.isin(f).to_numpy()], sig[sc0 & fr.wgroup.isin(f).to_numpy()])
                                for p, f in C.POOLS.items()}
        res[f"{s}_champion_seeds_ge30"] = [round(float(np.nanmean(o[s][ge])), 4) for o in seeds0]
        mm = ge & ~np.isnan(okx[s]) & ~np.isnan(seeds0[0][s])
        res[f"{s}_threads2_refit_seed0_minus_saved"] = [int((okx[s][mm] != seeds0[0][s][mm]).sum())] +             C.delta_ci(seeds0[0][s][mm], okx[s][mm], sig[mm])
        for nm, ok in oks.items():
            B = ok[s]
            sc = sc0 & ~np.isnan(B)
            r = {}
            for pool, fams in C.POOLS.items():
                m = sc & fr.wgroup.isin(fams).to_numpy()
                r[pool] = C.delta_ci(A[m], B[m], sig[m])
            m = sc & ge
            r["acc_ge30"] = round(float(B[m].mean()), 4)
            r["ge30_by_class"] = {c: [int((m & (cl == c)).sum())] + C.delta_ci(A[m & (cl == c)], B[m & (cl == c)],
                                                                               sig[m & (cl == c)])
                                  for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
            r["ge30_flagged"] = [int((m & flag).sum())] + C.delta_ci(A[m & flag], B[m & flag], sig[m & flag])
            r["ge30_unflagged"] = [int((m & ~flag).sum())] + C.delta_ci(A[m & ~flag], B[m & ~flag], sig[m & ~flag])
            r["ge30_by_fold"] = [round(100 * float(B[m & (fr.fold == k).to_numpy()].mean()
                                                   - A[m & (fr.fold == k).to_numpy()].mean()), 2) for k in range(6)]
            r["ge30_rows_changed"] = int((A[m] != B[m]).sum())
            if nm in per_seed:
                r["ge30_per_seed_vs_same_seed_champion"] = [
                    round(100 * float(np.nanmean(per_seed[nm][i][s][m]) - np.nanmean(seeds0[i][s][m])), 3)
                    for i in range(3)]
            res[f"{s}_{nm}"] = r
    json.dump(res, open(OUT / "hc79b.json", "w"), indent=1, default=str)
    for k, v in res.items():
        S.log(f"{k}: {v}")


if __name__ == "__main__":
    t0 = time.time()
    {"stack": stage_stack, "score": stage_score}[sys.argv[1]]()
    print(f"== {sys.argv[1]} done ({time.time() - t0:.0f}s)", flush=True)
