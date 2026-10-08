"""Note 114: health fully out of classification -- the v6b function stacker WITHOUT pk_unhealthy (stack-relative chatter
flag, pick.py) and the per-lane decode stack pick WITHOUT the healthy-first key.  Same recipe as note 111
(fit111 stack / oofnonet; f109 stack_seeds): 1-seed (s0) function-trees OOF, Sept-2026 training rows, v4q truth, six
folds folds_v4, PRM0 x 150 rounds; three stacker seeds for measurement discipline.

    set F76_ARM=c & python f114.py oof      six-fold OOF, variants mean3 / single / nonet x arms base / nou x seeds 0-2
                                            -> %DC_WORK%/s114/oof/P_{variant}_{arm}_s{seed}[_m{member}].npy
    set F76_ARM=c & python f114.py score    gate .9 lane decode (base: pk_unhealthy key on; nou: key off; stk: stacker
                                            without the column, key on; dec: stacker with it, key off) -> score114.json
    set F76_ARM=c & python f114.py refit    full-data refit (seed 0) of the three variants without pk_unhealthy -> ONNX +
                                            stacker.json in %DC_WORK%/s114/weights_stacker (parity < 1e-9)
locked_v2 asserted absent by the loaders.  CPU, 4 threads.
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

CODE = Path(__file__).resolve().parents[1]
for _d in ("final95", "final109", "final103", "final100", "final111"):
    sys.path.insert(0, str(CODE / _d))
sys.path.insert(0, str(CODE))
import rpath  # noqa: E402,F401

NJ = 4
SEEDS = (0, 1, 2)
LE = ["m5", "m10", "m30", "h1"]
FAMS = {"m5": ["m5"], "m10": ["m10"], "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h24": ["h24"],
        "ge30": ["m30", "h1", "h3", "h6", "h24", "full"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def DCW():
    import cand64 as C
    return C.DC_WORK


def setup():
    import func95 as G
    import f109_func as F109
    import cand64 as C
    F, F4 = G.fsetup()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    assert not fr.DeviceId.str.lower().isin(C.locked()).any()
    E = dict(E)
    E["Pt"] = F109.trees_seeds(fr, [0])
    trm = trm & (fr.period == "stg").to_numpy()
    X = {"mean3": F4.net_X(s74, S, E, s74.net_probs(fr, "x100_w32")),
         "nonet": F4.net_X(s74, S, E, np.full_like(E["Pt"], np.nan))}
    for m in (0, 1, 2):
        X[f"single_m{m}"] = F4.net_X(s74, S, E, s74.net_probs(fr, f"x100_w32:{m}"))
    j = names.index("pk_unhealthy")
    assert np.array_equal(X["mean3"][:, j], fr.pk_unhealthy.to_numpy(float))
    return F, s74, S, E, X, y, trm, names, j


def stage_oof():
    import lightgbm as lgb
    F, s74, S, E, X, y, trm, names, j = setup()
    fr = E["fr"]
    log(f"pk_unhealthy True share: all rows {fr.pk_unhealthy.mean():.4f}, train rows {fr.pk_unhealthy[trm].mean():.4f}")
    out = DCW() / "s114" / "oof"
    out.mkdir(parents=True, exist_ok=True)
    fo = fr.fold.to_numpy()
    keep = {"base": np.arange(len(names)), "nou": np.array([i for i in range(len(names)) if i != j])}
    for arm, cols in keep.items():
        for seed in SEEDS:
            prm = dict(F.PRM0, num_threads=NJ, seed=seed)
            for var in ("mean3", "nonet", "single"):
                f = out / (f"P_{var}_{arm}_s{seed}.npy" if var != "single" else f"P_single_{arm}_s{seed}_m2.npy")
                if f.exists():
                    continue
                if var != "single":
                    Xv = X[var][:, cols]
                    R = np.zeros((len(y), 7))
                    for k in range(6):
                        tr, te = trm & (fo != k), fo == k
                        R[te] = lgb.train(prm, lgb.Dataset(Xv[tr], y[tr]), num_boost_round=150).predict(Xv[te])
                    np.save(f, R.astype(np.float32))
                else:
                    Xm = [X[f"single_m{m}"][:, cols] for m in (0, 1, 2)]
                    R = [np.zeros((len(y), 7)) for _ in range(3)]
                    for k in range(6):
                        tr, te = trm & (fo != k), fo == k
                        b = lgb.train(prm, lgb.Dataset(np.vstack([x[tr] for x in Xm]), np.concatenate([y[tr]] * 3)),
                                      num_boost_round=150)
                        for m in range(3):
                            R[m][te] = b.predict(Xm[m][te])
                    for m in range(3):
                        np.save(out / f"P_single_{arm}_s{seed}_m{m}.npy", R[m].astype(np.float32))
                log(f"{var} {arm} s{seed} done")
    # sanity: base mean3 s0 must reproduce note 111's v6 function OOF (s109/func P_m3_t0_s0)
    ref = np.load(DCW() / "s109" / "func" / "P_m3_t0_s0.npy")
    got = np.load(out / "P_mean3_base_s0.npy")
    log(f"repro vs P_m3_t0_s0: max abs {np.abs(ref - got).max():.2e}")


def stage_score():
    import cand64 as C
    import of77
    F, s74, S, E, X, y, trm, names, j = setup()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    st = (fr.period == "stg").to_numpy()
    d = DCW() / "s114" / "oof"
    unh0 = fr.pk_unhealthy.copy()

    def ok_of(P, key_on):
        fr["pk_unhealthy"] = unh0 if key_on else False
        try:
            return s74.gate_ok(E, P, lc)
        finally:
            fr["pk_unhealthy"] = unh0
    # arms: (stacker arm, decode key on)
    ARMS = {"base": ("base", True), "nou": ("nou", False), "stk": ("nou", True), "dec": ("base", False)}
    ok = {}
    for var in ("mean3", "single", "nonet"):
        for seed in SEEDS:
            for arm, (sa, key) in ARMS.items():
                if var == "single":
                    oks = [ok_of(np.load(d / f"P_single_{sa}_s{seed}_m{m}.npy").astype(float), key) for m in range(3)]
                    ok[(var, arm, seed)] = {s: np.mean([o[s] for o in oks], 0) for s in ("E", "R")}
                else:
                    ok[(var, arm, seed)] = ok_of(np.load(d / f"P_{var}_{sa}_s{seed}.npy").astype(float), key)
            log(f"decoded {var} s{seed}")
    # fast setting le2h = mean3 <= 2 h, nonet above (as score111 v6_le2h)
    le = fr.wgroup.isin(LE).to_numpy()
    for seed in SEEDS:
        for arm in ARMS:
            ok[("le2h", arm, seed)] = {s: np.where(le, ok[("mean3", arm, seed)][s], ok[("nonet", arm, seed)][s])
                                       for s in ("E", "R")}
    res = {"pk_unhealthy_share": {"all": float(unh0.mean()), "train": float(unh0[trm].mean())},
           "n": {}, "acc": {}, "delta": {}, "delta_per_seed": {}}
    for var in ("mean3", "single", "nonet", "le2h"):
        for stn in ("E", "R"):
            for pool, fams in FAMS.items():
                sc = st & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce(
                    [~np.isnan(ok[(var, a, s)][stn]) for a in ARMS for s in SEEDS])
                res["n"].setdefault(var, {}).setdefault(stn, {})[pool] = [int(sc.sum()), int(len(set(sig[sc])))]
                mean = {a: np.mean([ok[(var, a, s)][stn] for s in SEEDS], 0) for a in ARMS}
                for a in ARMS:
                    res["acc"].setdefault(var, {}).setdefault(a, {}).setdefault(stn, {})[pool] = C.acc_ci(
                        mean[a][sc], sig[sc])
                    if a == "base":
                        continue
                    res["delta"].setdefault(var, {}).setdefault(a, {}).setdefault(stn, {})[pool] = C.delta_ci(
                        mean["base"][sc], mean[a][sc], sig[sc])
                    res["delta_per_seed"].setdefault(var, {}).setdefault(a, {}).setdefault(stn, {})[pool] = [
                        round(100 * float(np.mean(ok[(var, a, s)][stn][sc]) - np.mean(ok[(var, "base", s)][stn][sc])), 3)
                        for s in SEEDS]
            log(f"{var}: nou dE " + " ".join(f"{p} {res['delta'][var]['nou']['E'][p]}" for p in FAMS))
    json.dump(res, open(DCW() / "s114" / "score114.json", "w"), indent=1)


def stage_refit():
    """full-data seed-0 refits without pk_unhealthy, fit111 _stack_setup inputs; ONNX + parity; stacker.json."""
    import lightgbm as lgb
    import fit111
    fit111.NJ = NJ
    PK, s74, S, E, y, trm, names, fr = fit111._stack_setup()
    F, F4 = PK.F, PK.F4
    j = names.index("pk_unhealthy")
    cols = [i for i in range(len(names)) if i != j]
    names2 = [("stack_n" if names[i] == "hf_clus_n" else names[i]) for i in cols]   # patched stacker.py name
    d = DCW() / "s114" / "weights_stacker"
    d.mkdir(parents=True, exist_ok=True)
    src = DCW() / "final_v3_candidate_v6b" / "weights" / "stacker" / "stacker.json"
    sm = json.load(open(src))
    assert sm["feature_names"] == names, "v6b stacker columns differ from the fit111 inputs"
    prm = dict(F.PRM0, num_threads=NJ, seed=0)
    Xk = {"mean3": [F4.net_X(s74, S, E, s74.net_probs(fr, "x100_w32"))[:, cols]],
          "single": [F4.net_X(s74, S, E, s74.net_probs(fr, f"x100_w32:{m}"))[:, cols] for m in (0, 1, 2)],
          "nonet": [F4.net_X(s74, S, E, np.full_like(E["Pt"], np.nan))[:, cols]]}
    sys.path.insert(0, str(rpath.CODE / "final75"))
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(DCW() / "final_v3_candidate_v6b"))
    import trees_onnx as TO
    par = {}
    for kind, xs in Xk.items():
        Xtr = np.vstack([x[trm] for x in xs])
        ytr = np.concatenate([y[trm]] * len(xs))
        txt = d / f"stacker_{kind}_s0.txt"
        lgb.train(prm, lgb.Dataset(Xtr, ytr), num_boost_round=150).save_model(str(txt))
        Xc = xs[0][:2000].astype(np.float32).astype(np.float64)
        nb = NumpyBooster(txt)
        f = d / f"stacker_{kind}_s0.onnx"
        OT.to_onnx_te5(nb, f)
        m = TO.OnnxTrees(f, 4)
        par[f.name] = {"rows": len(Xc), "max_abs_out": float(np.abs(np.asarray(nb.predict(Xc)) -
                                                                      np.asarray(m.predict(Xc))).max())}
        assert par[f.name]["max_abs_out"] < 1e-9, par
        log(f"{kind}: {Xtr.shape} -> {f.name} parity {par[f.name]}")
    sm["feature_names"], sm["n_features"] = names2, len(names2)
    sm["variant"] = ("three variants at ONE seed each (note 114 = note 111 recipe without pk_unhealthy): mean3 (default), "
                     "single, nonet; 1-seed trees OOF inputs")
    sm["note114"] = ("pk_unhealthy (stack-relative health flag) removed: health is not a classification input; "
                     "hf_clus_n renamed stack_n (same values: hi-res stack size, pick.stack_sizes)")
    for k in sm["variants"]:
        sm["variants"][k]["trained"] = sm["variants"][k]["trained"] + "; note 114: without pk_unhealthy"
    json.dump(sm, open(d / "stacker.json", "w"), indent=1)
    json.dump(par, open(d / "parity114.json", "w"), indent=1)
    log(f"stacker.json: {len(names2)} columns")


if __name__ == "__main__":
    t0 = time.time()
    globals()[f"stage_{sys.argv[1]}"]()
    log(f"== {sys.argv[1]} done ({time.time() - t0:.0f}s)")
