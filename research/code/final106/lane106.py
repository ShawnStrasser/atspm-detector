"""Note 106, idea 2: iterate function <-> lanes.  The lanes D step (note-102 2026 recipe: pair model = cues + context +
pair type + function block, both orientations, six folds x 3 seeds, lam per held-out fold) normally takes its function
input (pair type, the 7 probabilities of the c2 block, decoder anchors) from the 2026 function TREES.  Here it takes
them from a stacker OOF (the full function model); the new lanes then feed the stacker's lane context and the gate /
per-lane decode, and the stacker is refitted.

    set F76_ARM=c & python lane106.py lanes --P NAME --tag TAG      lanes D with function = s106/P_NAME -> s106/ln/TAG/
    set F76_ARM=c & python lane106.py func --lanes TAG --name NAME [--sseeds 0] [--blocks none]
                       stacker on X0 with the lane context rebuilt from lanes TAG ('f102' = the 2026 trees lanes)
                       -> s106/P_NAME.npy (+ lanes5g / lane_conf of TAG saved for scoring)
    set F76_ARM=c & python lane106.py score NAME1 NAME2 ...         gate decode on each NAME's own lanes, paired vs first
    set F76_ARM=c & python lane106.py laneeval TAG1 TAG2 ...        n_lanes exact (14 Sept windows >= 30 min), CIs,
                       note-102 review phases (rev102/rows102.parquet), vs the first tag
OOF caveat (as the rest of the chain): the stacker probabilities given to the lane step are six-fold OOF, but the
stacker that produced fold j's probabilities saw fold k's labels -- standard stacking, not nested.
locked_v2 asserted absent by the loaders.  CPU <= 6 workers.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("F76_ARM", "c")
import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
for _d in ("final106", "lanes", "final77", "evaluation", "trackA", ""):
    sys.path.insert(0, str(CODE / _d) if _d else str(CODE))
import s106 as B  # noqa: E402

LN = B.OUT / "ln"
F102 = B.G.DC_WORK / "final_v3_work" / "f102" / "ln8"
KEY = B.KEY
REVIEW_WINS = ["m30_a", "m30_b", "m30_c", "m30_d", "h1_a", "h1_b", "h1_c", "h3_a", "h3_b", "h6_a", "h6_b",
               "h24_a", "h24_b", "full66"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def lanes_path(tag) -> Path:
    return F102 / "lanes_base.parquet" if tag == "f102" else (
        B.G.DC_WORK / "final_v3_work" / "f77" / "ln8" / "lanes_D.func.parquet" if tag == "f77" else LN / tag / "lanes_base.parquet")


def stage_lanes(a):
    import ln102_wide as L102
    import ln8_validate as L8
    d = LN / a.tag
    d.mkdir(parents=True, exist_ok=True)
    for f in ("cues.parquet", "cues_rev.parquet", "dets.parquet", "keys.parquet", "truth_det.parquet",
              "truth_phase.parquet", "truth_text_pairs.parquet"):
        if not (d / f).exists():
            shutil.copy(F102 / f, d / f)
    L102.OUT = d
    P = np.load(B.OUT / f"P_{a.P}.npy").astype(np.float32)
    kk = pd.read_parquet(B.OUT / "keys.parquet", columns=KEY)
    kk["DeviceId"] = kk.DeviceId.str.lower()
    kk["_i"] = np.arange(len(kk))

    def func_probs(k):
        x = k[KEY].merge(kk.astype({"Detector": k.Detector.dtype}), on=KEY, how="left")
        assert len(x) == len(k) and x._i.notna().all()
        return P[x._i.to_numpy(int)]
    L8.func_probs = func_probs
    ns = argparse.Namespace
    t0 = time.time()
    if not (d / "p_base.parquet").exists():
        L102.stage_fit(ns(var="base", workers=a.workers))
        log(f"fit done ({time.time()-t0:.0f}s)")
    if not (d / "pick_base.json").exists():
        L102.stage_decode(ns(var="base", notau=True, advs="none", suffix="", workers=a.workers))
        log(f"decode grid done ({time.time()-t0:.0f}s)")
    L102.stage_full(ns(var="base", suffix="", notau=False, workers=a.workers))
    log(f"lanes {a.tag} done ({time.time()-t0:.0f}s)")


def lane_inputs(tag):
    """frame-aligned lanes5g (stg rows from TAG, dec rows unchanged) and lane ctx (3 cols)."""
    import of77
    d = B.load()
    fr = d["E"]["fr"]
    st = (fr.period == "stg").to_numpy()
    L = pd.read_parquet(lanes_path(tag), columns=KEY + ["phase", "lanes"])
    L = L[(L.period == "stg") & (L.lanes != "")].astype({"Detector": fr.Detector.dtype})
    x = fr[KEY].merge(L, on=KEY, how="left")
    assert len(x) == len(fr)
    new = x.lanes.where(x.phase.eq(fr.pred_phase.to_numpy()), None).where(~fr.wgroup.isin(["m5", "m10"]), None)
    lanes = np.where(st, new.to_numpy(object), fr.lanes5g.to_numpy(object))
    lc0 = np.load(B.OUT / "lane_ctx.npy")
    lc = of77.lane_ctx(fr, lanes_path(tag))
    lc = np.where(st[:, None], lc, lc0)
    return lanes, lc


def stage_func(a):
    d = B.load()
    E, S, F4, s74 = d["E"], d["S"], d["F4"], d["s74"]
    fr = E["fr"]
    lanes, lc = lane_inputs(a.lanes)
    old_l, old_c = fr.lanes5g.copy(), S._CTX.get("ln")
    fr["lanes5g"] = lanes
    S._CTX["ln"] = lc
    try:
        Pn = np.load(B.OUT / "Pn.npy").astype(float)
        X = F4.net_X(s74, S, E, Pn)
    finally:
        fr["lanes5g"] = old_l
        S._CTX["ln"] = old_c
    np.save(B.OUT / f"lanes5g_{a.name}.npy", np.asarray(lanes, object), allow_pickle=True)
    np.save(B.OUT / f"lanectx_{a.name}.npy", lc)
    B.stack(a.name, a.blocks, [int(x) for x in a.sseeds.split(",")], X0=X)
    log(f"P_{a.name} written (lanes {a.lanes})")


def ok_for(name):
    P = np.load(B.OUT / f"P_{name}.npy").astype(float)
    fl = B.OUT / f"lanes5g_{name}.npy"
    if fl.exists():
        lanes = np.load(fl, allow_pickle=True)
        lc = np.load(B.OUT / f"lanectx_{name}.npy")[:, 2]
        return B.ok_arrays(P, lconf=lc, lanes=lanes)
    return B.ok_arrays(P)


def stage_score(a):
    names = a.names
    ok = {n: ok_for(n) for n in names}
    B.score_many(ok, names[0], "lanes:" + ",".join(names))


# ================================================================================================ lane evaluation
def sample_rows(tag):
    ph = pd.read_parquet(F102 / "truth_phase.parquet")
    ph["ph_num"] = ph.target.str[1:].astype(int)
    f = str(lanes_path(tag)).replace(".parquet", "_ph.parquet")
    P = pd.read_parquet(f)
    P = P[(P.period == "stg") & P.win.isin(REVIEW_WINS)]
    x = ph.merge(P[["DeviceId", "phase", "win", "n_lanes", "n_lanes_conf"]], left_on=["DeviceId", "ph_num"],
                 right_on=["DeviceId", "phase"], suffixes=("", "_p"))
    x["ok"] = (x.n_lanes == x.n_lanes_p).astype(float)
    return x


def boot_delta(d0, d1, key="DeviceId", reps=1000, seed=106):
    j = d0[[key, "target", "win", "ok"]].merge(d1[[key, "target", "win", "ok"]], on=[key, "target", "win"],
                                               suffixes=("_0", "_1"))
    g = j.groupby(key).agg(a=("ok_0", "sum"), b=("ok_1", "sum"), n=("ok_0", "size"))
    rng = np.random.default_rng(seed)
    A, Bv, N = g.a.to_numpy(), g.b.to_numpy(), g.n.to_numpy()
    out = []
    for _ in range(reps):
        i = rng.integers(0, len(g), len(g))
        out.append((Bv[i].sum() - A[i].sum()) / N[i].sum())
    return [round(100 * (Bv.sum() - A.sum()) / N.sum(), 2), round(100 * np.quantile(out, .025), 2),
            round(100 * np.quantile(out, .975), 2)]


def laneeval(tags, extra_rows: dict | None = None):
    R = {t: sample_rows(t) for t in tags}
    if extra_rows:
        R.update(extra_rows)
        tags = list(tags) + list(extra_rows)
    key = ["DeviceId", "target", "win"]
    common = None
    for t in tags:
        s = R[t][key]
        common = s if common is None else common.merge(s, on=key)
    rv = pd.read_parquet(B.G.DC_WORK / "rev102" / "rows102.parquet", columns=["DeviceId", "target", "cause", "truth", "maj"])
    res = {"review102_causes": rv.cause.value_counts().to_dict()}
    for t in tags:
        x = R[t].merge(common, on=key)
        R[t] = x
        r = {"n": int(len(x)), "exact": round(float(x.ok.mean()), 4),
             "over": round(float((x.n_lanes_p > x.n_lanes).mean()), 4),
             "under": round(float((x.n_lanes_p < x.n_lanes).mean()), 4),
             "by_truth": {int(k): round(float(v), 4) for k, v in x.groupby("n_lanes").ok.mean().items()}}
        # note-102 review phases: phase majority right?
        y = x.merge(rv, on=["DeviceId", "target"], suffixes=("", "_rv"))
        maj = y.groupby(["DeviceId", "target", "cause"]).apply(
            lambda g: int(g.n_lanes_p.mode().iloc[0] == g.n_lanes.iloc[0]), include_groups=False).reset_index(name="right")
        r["review102_phase_majority_right"] = {c: f"{int(g.right.sum())}/{len(g)}" for c, g in maj.groupby("cause")}
        r["review102_sample_exact"] = {c: round(float(g.ok.mean()), 3) for c, g in y.groupby("cause")}
        res[t] = r
        log(f"{t:14s} exact {r['exact']} (over {r['over']}, under {r['under']}) by truth {r['by_truth']} | "
            f"rev102 right {r['review102_phase_majority_right']}")
    for t in tags[1:]:
        c = {"all": boot_delta(R[tags[0]], R[t])}
        for nm, m in (("truth1", 1), ("truth2", 2)):
            c[nm] = boot_delta(R[tags[0]][R[tags[0]].n_lanes == m], R[t][R[t].n_lanes == m])
        c["truth3+"] = boot_delta(R[tags[0]][R[tags[0]].n_lanes >= 3], R[t][R[t].n_lanes >= 3])
        res[f"{t} vs {tags[0]}"] = c
        log(f"{t} vs {tags[0]}: {c}")
    return res, R


def stage_laneeval(a):
    res, _ = laneeval(a.names)
    f = B.OUT / "laneeval106.json"
    old = json.load(open(f)) if f.exists() else {}
    old[",".join(a.names)] = res
    json.dump(old, open(f, "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lanes", "func", "score", "laneeval"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--P", default="base_m3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lanes", default="f102")
    ap.add_argument("--name", default="")
    ap.add_argument("--sseeds", default="0")
    ap.add_argument("--blocks", default="none")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    globals()[f"stage_{a.stage}"](a)
    log(f"== {a.stage} done ({time.time() - t0:.0f}s)")
