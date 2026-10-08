"""Note 109: a SIMPLER architecture -- FUNCTION side (note-95 / 100b setup: 2026 trees OOF + siba OOF -> v5 context
stacker (3 LightGBM seeds, Sept-2026 rows) -> gate .9 lane decode; v4q truth; six folds folds_v4).

    set F76_ARM=c & python f109_func.py stack NAME NETSPEC   = f100.stack into s109/func (NETSPEC as s74.net_probs)
    set F76_ARM=c & python f109_func.py score                every arm per length (5 / 10 / 30 min, 1 h, 3 h, 24 h,
                                                             >= 30 min), E / R, CI vs full v5c, + 'no lane rule'
                                                             (argmax, stack credit) for the main arms -> score109.json
Arms: full = P_w32m3 (s100/b; = v5b / v5c function OOF); w32s0 / w32s1 (s100/b) / w32s2 (here) = one w32 member;
nf_* = the same checkpoints without the tree candidate filter (x109_w32nf, local A1000); A2 (s103/func) = one LightGBM.
locked_v2 asserted absent by the loaders.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "5")
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
sys.path.insert(0, str(CODE / "final100"))
sys.path.insert(0, str(CODE / "final103"))
import func95 as G  # noqa: E402
import f100  # noqa: E402

OUT = G.DC_WORK / "s109" / "func"
B100 = G.DC_WORK / "s100" / "b"
POOLS = {"m5": ["m5"], "m10": ["m10"], "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h24": ["h24"],
         "ge30": ["m30", "h1", "h3", "h6", "h24", "full"]}
LE = ["m5", "m10", "m30", "h1"]
NOLANE = ["full", "w32s0", "nf_m3", "nf_s0"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stack(name, spec):
    f100.OUT = OUT
    f100.stack(name, spec)


def trees_seeds(fr, seeds):
    """the 2026 function-trees OOF (func95 fit) for the given seeds, aligned to the stacker frame (err78 / tree106 loader)."""
    import pandas as pd
    import v3_retrain as V
    import t57_function as T57F
    T57F.setup("v6e")
    d = G.TREES95 / G.CFG_SUB
    key = ["DeviceId", "Detector", "period", "win"]
    fk = pd.read_parquet(V.FEATS, columns=key + ["fold"])
    fk["DeviceId"] = fk.DeviceId.str.lower()
    P = np.zeros((len(fk), 7), np.float32)
    for s in seeds:
        for k in range(6):
            P[fk.fold.to_numpy() == k] += np.load(d / f"P_first.all.wi_s{s}_f{k}.npy") / len(seeds)
    fk["_i"] = np.arange(len(fk))
    fk["Detector"] = fk.Detector.astype(fr.Detector.dtype)
    idx = fr[key].assign(DeviceId=fr.DeviceId.str.lower()).merge(fk[key + ["_i"]], on=key, how="left")._i.to_numpy()
    assert not np.isnan(idx).any()
    return P[idx.astype(int)].astype(float)


def stack_seeds(name, spec, tseeds, sseeds):
    """f100.stack with the trees OOF from `tseeds` only and the stacker averaged over `sseeds` only (1-seed designs)."""
    import lightgbm as lgb
    F, F4 = G.fsetup()
    OUT.mkdir(parents=True, exist_ok=True)
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    Pt3 = E["Pt"]
    E = dict(E)
    E["Pt"] = trees_seeds(fr, [int(x) for x in tseeds.split(",")])
    if tseeds == "0,1,2":
        assert np.allclose(E["Pt"], Pt3, atol=1e-5)
    Pf = s74.net_probs(fr, spec)
    X = F4.net_X(s74, S, E, Pf)
    trm = trm & (fr.period == "stg").to_numpy()
    fo = fr.fold.to_numpy()
    ss = [int(x) for x in sseeds.split(",")]
    R = np.zeros((len(y), 7))
    for seed in ss:
        for k in range(6):
            tr, te = trm & (fo != k), fo == k
            m = lgb.train(dict(F.PRM0, num_threads=F.THREADS, seed=seed), lgb.Dataset(X[tr], y[tr]), num_boost_round=150)
            R[te] += m.predict(X[te]) / len(ss)
    np.save(OUT / f"P_{name}.npy", R.astype(np.float32))
    json.dump({"spec": spec, "tree_seeds": tseeds, "stack_seeds": sseeds}, open(OUT / f"P_{name}.json", "w"))
    log(f"wrote P_{name}")


def stage_score():
    import cand64 as C
    import of77
    import atspm_score as AS
    F, F4 = G.fsetup()
    s74, S, E, Xs, y, trm, _ = F.stacker_inputs()
    fr = E["fr"]
    assert not fr.DeviceId.str.lower().isin(C.locked()).any()
    C7 = list(E["C7"])
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    sig = fr.DeviceId.to_numpy()
    st = (fr.period == "stg").to_numpy()
    le = fr.wgroup.isin(LE).to_numpy()[:, None]
    nonet = np.load(G.OOF95 / "P_v5_nonet.npy").astype(float)
    P = {"full": np.load(B100 / "P_w32m3.npy").astype(float), "no_nets": nonet, "trees_alone": E["Pt"].astype(float)}
    P["fast_le2h"] = np.where(le, P["full"], nonet)
    for n in ("w32s0", "w32s1"):
        P[n] = np.load(B100 / f"P_{n}.npy").astype(float)
    for f in sorted(OUT.glob("P_*.npy")):
        P[f.stem[2:]] = np.load(f).astype(float)
    A2d = G.DC_WORK / "s103" / "func" / "A2"
    if (A2d / "P_s2_f5.npy").exists():
        import f103_func as F103
        P["A2_m3net"], _ = F103.arm_P("A2", fr)
    for d in sorted(OUT.glob("A2_*")):
        if (d / "P_s0_f5.npy").exists():
            import f103_func as F103
            F103.OUT = OUT
            P[d.name], _ = F103.arm_P(d.name, fr)
    ok = {}
    for k_, v in P.items():
        ok[k_] = s74.gate_ok(E, v, lc)
        log(f"decoded {k_}")
    key = ["DeviceId", "Detector", "period", "win"]
    base = fr[key + ["wgroup", "truth_v3s", "stack_group", "stack_role"]].copy()

    def argmax_ok(Pm):
        b = base.copy()
        for i, c in enumerate(C7):
            b[f"P_{c}"] = Pm[:, i]
        pred = np.nan_to_num(Pm, nan=-1).argmax(1)
        out = {}
        for stn in ("E", "R"):
            r = np.flatnonzero(~np.isnan(ok["full"][stn]))
            d = AS.credit(b, pred, "truth_v3s", r, True)
            o = np.full(len(b), np.nan)
            o[r] = d.ok_a.to_numpy(float)
            out[stn] = o
        return out
    for k_ in NOLANE:
        if k_ in P:
            ok[f"{k_}_nolane"] = argmax_ok(P[k_])
            log(f"argmax {k_}")
    res = {"n": {}, "acc": {}, "delta_vs_full": {}}
    for stn in ("E", "R"):
        for pool, fams in POOLS.items():
            sc = st & fr.wgroup.isin(fams).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k_][stn]) for k_ in ok])
            res["n"].setdefault(stn, {})[pool] = [int(sc.sum()), int(len(set(sig[sc])))]
            for k_ in ok:
                res["acc"].setdefault(k_, {}).setdefault(stn, {})[pool] = C.acc_ci(ok[k_][stn][sc], sig[sc])
                if k_ != "full":
                    res["delta_vs_full"].setdefault(k_, {}).setdefault(stn, {})[pool] = C.delta_ci(
                        ok["full"][stn][sc], ok[k_][stn][sc], sig[sc])
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    m = st & fr.wgroup.isin(POOLS["ge30"]).to_numpy() & np.logical_and.reduce([~np.isnan(ok[k_]["E"]) for k_ in ok])
    res["E_ge30_by_class_vs_full"] = {k_: {c: C.delta_ci(ok["full"]["E"][m & (cl == c)], ok[k_]["E"][m & (cl == c)],
                                                         sig[m & (cl == c)]) for c in ["Advance", "Presence", "Count",
                                                                                       "Yellow_Red", "nonATSPM"]}
                                      for k_ in ok if k_ != "full"}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "score109.json", "w"), indent=1)
    for k_ in ok:
        log(f"{k_:16s} E " + " ".join(f"{p} {res['acc'][k_]['E'][p][0]}" for p in POOLS))
        if k_ != "full":
            log(f"{'':16s} dE " + " ".join(f"{p} {res['delta_vs_full'][k_]['E'][p]}" for p in POOLS))
            log(f"{'':16s} dR ge30 {res['delta_vs_full'][k_]['R']['ge30']}")


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1] == "stack":
        stack(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "stackseeds":
        stack_seeds(*sys.argv[2:6])
    else:
        stage_score()
    log(f"== {sys.argv[1]} done ({time.time() - t0:.0f}s)")
