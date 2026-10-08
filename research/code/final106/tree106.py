"""Note 106: function trees (note-95 2026 recipe: 229 features, A2.FUNC_PARAMS, inner-fold early stopping, Sept-2026
training rows, v4q labels, six folds) + extra feature blocks from s106/feat_*.parquet; then the v5b stacker on top.
(Fit code = final103/f103_func.stage_fit with a different column source.)

    set F76_ARM=c & python tree106.py fit --arm NAME --blocks vp[+...|none] [--seeds 0] [--shuf]
    set F76_ARM=c & python tree106.py stack --arm NAME [--seeds 0] [--sseeds 0]   trees OOF (stacker frame order) ->
                         stacker on X0 with the tree columns recomputed (siba w32 mean3 fixed) -> s106/P_T<NAME>.npy
locked_v2 asserted absent by the loaders.  CPU <= 6 threads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
sys.path.insert(0, str(CODE / "final106"))
import func95 as G  # noqa: E402
import s106 as B  # noqa: E402

OUT = B.OUT / "trees"
KEY = B.KEY
THREADS = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tree_frame():
    import t57_function as T57F
    import f76_function  # noqa: F401
    import v3_retrain as V
    assert Path(V.LABEL_SETS["v3s"][0]).name == "function_labels_v4q.parquet"
    G.patch_2026(V)
    T57F.OUT = G.TREES95
    T57F.setup("v6e")
    fr, _ = V.load_feats()
    lab = V.load_labels(fr, "exclude")
    y, ok = V.variant_target(lab, fr, T57F.VAR)
    yi = pd.Series(y).map({c: i for i, c in enumerate(V.C7)}).fillna(-1).astype(int).to_numpy()
    return fr, ok, yi, V


def stage_fit(a):
    import lightgbm as lgb
    import a2_model as A2
    fr, ok, yi, V = tree_frame()
    cols = json.load(open(G.TREES95 / G.CFG_SUB / "cols.json"))["cols"]
    assert len(cols) == 229
    k = fr[KEY].copy()
    k["DeviceId"] = k.DeviceId.str.lower()
    new = []
    if a.blocks != "none":
        for b in a.blocks.split("+"):
            D = pd.read_parquet(B.OUT / f"feat_{b}.parquet")
            D["DeviceId"] = D.DeviceId.str.lower()
            D["Detector"] = D.Detector.astype(k.Detector.dtype)
            m = k.merge(D, on=KEY, how="left")
            assert len(m) == len(fr)
            c2 = [c for c in D.columns if c not in KEY]
            Z = m[c2].to_numpy(np.float32)
            if a.shuf:
                rng = np.random.default_rng(1061)
                for ix in k.groupby(["period", "win"]).indices.values():
                    Z[ix] = Z[rng.permutation(ix)]
            for i, c in enumerate(c2):
                fr[c] = Z[:, i]
            new += c2
            log(f"block {b}: {len(c2)} cols, coverage {np.mean(~np.isnan(Z[:, 0])):.3f}")
    cols = cols + new
    d = OUT / a.arm
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"arm": a.arm, "blocks": a.blocks, "shuf": a.shuf, "n_cols": len(cols), "cols": cols,
               "train_rows": int(ok.sum())}, open(d / "cols.json", "w"), indent=1)
    log(f"{a.arm}: {len(cols)} features, {int(ok.sum()):,} training rows (Sept-2026)")
    folds = fr.fold.to_numpy()
    k.to_parquet(d / "keys.parquet", index=False)
    np.save(d / "folds.npy", folds)
    for s in [int(x) for x in a.seeds.split(",")]:
        for kf in range(6):
            f = d / f"P_s{s}_f{kf}.npy"
            if f.exists():
                continue
            t0 = time.time()
            inner = (kf + 1) % 6
            trm = ok & (folds != kf) & (folds != inner)
            vam = ok & (folds == inner)
            prm = dict(A2.FUNC_PARAMS, n_jobs=THREADS, num_class=7, seed=s, bagging_seed=s + 1,
                       feature_fraction_seed=s + 2, data_random_seed=s + 3)
            n = prm.pop("n_estimators")
            mdl = lgb.LGBMClassifier(n_estimators=n, **prm)
            mdl.fit(fr.loc[trm, cols], yi[trm], eval_set=[(fr.loc[vam, cols], yi[vam])], eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            np.save(f, mdl.predict_proba(fr.loc[folds == kf, cols]).astype(np.float32))
            imp = pd.Series(mdl.booster_.feature_importance("gain"), index=cols)
            with open(d / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"seed": s, "fold": kf, "trees": int(mdl.best_iteration_ or 0),
                                     "secs": round(time.time() - t0, 1),
                                     "new_gain_share": float(imp[new].sum() / imp.sum()) if new else 0.0}) + "\n")
            log(f"  {a.arm} s{s} f{kf}: {mdl.best_iteration_} trees, {time.time() - t0:.0f}s")
            if kf == 0:
                (imp / imp.sum()).sort_values(ascending=False).head(40).to_json(d / f"gain_f0_s{s}.json", indent=1)


def arm_P(arm, seeds):
    """trees OOF of an arm, mean over seeds, in stacker-frame (s106/keys.parquet) order."""
    d = OUT / arm
    fk = pd.read_parquet(d / "keys.parquet")
    folds = np.load(d / "folds.npy")
    Ps = []
    for s in seeds:
        P = np.zeros((len(fk), 7), np.float32)
        for k in range(6):
            P[folds == k] = np.load(d / f"P_s{s}_f{k}.npy")
        Ps.append(P)
    P = np.mean(Ps, 0)
    fk["_i"] = np.arange(len(fk))
    kk = pd.read_parquet(B.OUT / "keys.parquet", columns=KEY)
    fk["Detector"] = fk.Detector.astype(kk.Detector.dtype)
    idx = kk.assign(DeviceId=kk.DeviceId.str.lower()).merge(fk, on=KEY, how="left")._i.to_numpy()
    assert not np.isnan(idx).any()
    return P[idx.astype(int)].astype(float)


def base_trees(seeds):
    """the 2026 trees (func95 fit) OOF for the given seeds, stacker-frame order."""
    d = G.TREES95 / G.CFG_SUB
    import v3_retrain as V
    import t57_function as T57F
    T57F.setup("v6e")
    fk = pd.read_parquet(V.FEATS, columns=KEY + ["fold"])
    fk["DeviceId"] = fk.DeviceId.str.lower()
    Ps = []
    for s in seeds:
        P = np.zeros((len(fk), 7), np.float32)
        for k in range(6):
            P[fk.fold.to_numpy() == k] = np.load(d / f"P_first.all.wi_s{s}_f{k}.npy")
        Ps.append(P)
    P = np.mean(Ps, 0)
    fk["_i"] = np.arange(len(fk))
    kk = pd.read_parquet(B.OUT / "keys.parquet", columns=KEY)
    fk["Detector"] = fk.Detector.astype(kk.Detector.dtype)
    idx = kk.assign(DeviceId=kk.DeviceId.str.lower()).merge(fk[KEY + ["_i"]], on=KEY, how="left")._i.to_numpy()
    assert not np.isnan(idx).any()
    return P[idx.astype(int)].astype(float)


def stacker_X(Pt, lanes_ctx=None):
    """47 stacker columns with a different trees OOF (siba w32 mean3 fixed)."""
    d = B.load()
    E, S, F4, s74 = d["E"], d["S"], d["F4"], d["s74"]
    Pn = np.load(B.OUT / "Pn.npy").astype(float)
    E2 = dict(E)
    E2["Pt"] = Pt
    return F4.net_X(s74, S, E2, Pn)


def stage_stack(a):
    seeds = [int(x) for x in a.seeds.split(",")]
    Pt = base_trees(seeds) if a.arm == "BASE" else arm_P(a.arm, seeds)
    X = stacker_X(Pt)
    name = f"T{a.arm}_t{''.join(map(str, seeds))}_s{''.join(a.sseeds.split(','))}"
    B.stack(name, a.blocks or "none", [int(x) for x in a.sseeds.split(",")], X0=X)
    np.save(B.OUT / f"Pt_{name}.npy", Pt.astype(np.float32))
    log(f"wrote P_{name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "stack"])
    ap.add_argument("--arm", default="vp")
    ap.add_argument("--blocks", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--sseeds", default="0")
    ap.add_argument("--shuf", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
