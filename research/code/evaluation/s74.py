"""Note 74: function-net variants inside the current champion (note 67: context stacker 3 seeds + D-lane gate .9).

Generalises `s69_on67.py`: any set of 3-seed function nets (tag prefixes in %DC_WORK%/tcn53/fpreds:
PREFIX_f{k}, PREFIX_s1_f{k}, PREFIX_s2_f{k}) can be the MAIN net (fixed 0.6 / 0.4 blend input of the context and the
stacker's net columns, exactly as note 67 / 69), and further nets can be added as EXTRA stacker columns (the
"specialist decider", user idea: the stacker sees every specialist's probabilities and learns whom to trust).

    python s74.py stack --name sibm --main x69_sibm
    python s74.py stack --name spec --main x69_siba --extra x69_sibm,x69_sibacw2
    python s74.py compare --a siba --b sibm          # paired, gate .9, pools / classes / folds, E and R
    python s74.py check                              # stack --name siba must reproduce s69/on67 (note 69) exactly
    python s74.py stack --name siba0 --main x69_siba:0   # single seed (seed list after ':', e.g. :0+1)

Stacker OOF over the six folds (fold k's stacker never sees a fold-k row), seeds 0/1/2, same fixed LightGBM as note 67.
Output %DC_WORK%/s74/<name>/stack_ctx_s*.npy, %DC_WORK%/s74/cmp_<a>_vs_<b>.json. locked_v2 asserted absent (setup).
CPU only, 4 threads (s67_decider.THREADS).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cand64 as C  # noqa: E402
import s67_decider as S  # noqa: E402

OUT = C.DC_WORK / "s74"
GATE = 0.9
PC = [f"P_{c}" for c in ("Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other")]
KEY = ["DeviceId", "Detector", "period", "win"]
REF = {"siba": C.DC_WORK / "s69" / "on67", "fj": C.DC_WORK / "s67"}      # stacks computed by notes 69 / 67


def net_probs(fr: pd.DataFrame, prefix: str) -> np.ndarray:
    """frame-aligned probabilities of a 3-seed net set (NaN where missing); asserts all 18 files exist.
    `prefix:0` = seed 0 only (PREFIX_f{k}; used for the single-seed GRU-vs-TCN comparison)."""
    Pn = np.full((len(fr), 7), np.nan)
    sufs = ("", "_s1", "_s2")
    if ":" in prefix:
        prefix, s = prefix.split(":")
        sufs = tuple("" if x == "0" else f"_s{x}" for x in s.split("+"))
    for k in range(6):
        fs = [C.FPRED / f"{prefix}{s}_f{k}.parquet" for s in sufs]
        miss = [f.name for f in fs if not f.exists()]
        assert not miss, miss
        parts = []
        for f in fs:
            n = pd.read_parquet(f)
            n["DeviceId"] = n.DeviceId.str.lower()
            parts.append(n.astype({"Detector": fr.Detector.dtype}).set_index(KEY)[PC])
        n = sum(parts) / len(parts)
        m = fr[KEY].merge(n.reset_index(), on=KEY, how="left")[PC].to_numpy(float)
        fk = fr.fold.to_numpy() == k
        Pn[fk] = m[fk]
    return Pn


def setup(main: str):
    """S.setup() with the main net read from `main` instead of the fj files."""
    import s59_step6 as S59
    import s62_short as S62
    fr, rows = S59.scoring_frame(S59.PICK_TAG)
    fr = fr.copy()
    assert not fr.DeviceId.isin(C.locked()).any()
    Pt, seeds = S59.spec_probs("base", fr)
    assert seeds == [0, 1, 2]
    Pn = net_probs(fr, main)
    hasn = ~np.isnan(Pn[:, 0])
    S.log(f"main {main}: coverage {hasn.mean():.4f}")
    Pn = np.where(hasn[:, None], Pn, Pt)
    pairs = pd.read_parquet(S62.PAIRS_F)
    short = fr.wgroup.isin(S62.SHORT).to_numpy() & (fr.det_n_on >= 5).to_numpy()
    tw = S62.twin_tokens(fr, pairs, C.TWIN_THR, 1.0, 0)
    return dict(fr=fr, rows=rows, Pt=Pt, Pn=Pn, hasn=hasn, short=short, tw=tw, C7=S59.C7)


def extra_X(E, prefix):
    """specialist columns of one extra net: log probs (7) + margin + entropy + agrees-with-main-argmax."""
    P = net_probs(E["fr"], prefix)
    has = ~np.isnan(P[:, 0])
    S.log(f"extra {prefix}: coverage {has.mean():.4f}")
    P = np.where(has[:, None], P, E["Pn"])
    return np.column_stack([np.log(np.clip(P, 1e-6, 1)), S.margin(P), S.ent(P),
                            (P.argmax(1) == E["Pn"].argmax(1)).astype(float)])


def cmd_stack(a):
    d = OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    E = setup(a.main)
    Pt, Pn = E["Pt"], E["Pn"]
    Pb = C.W_TREE * Pt + (1 - C.W_TREE) * Pn
    X = np.hstack([S.stack_X(E["fr"], Pt, Pn), S.ctx_X(E, Pb)])
    for p in [x for x in a.extra.split(",") if x]:
        X = np.hstack([X, extra_X(E, p)])
    S.log(f"{a.name}: X {X.shape}")
    json.dump(dict(main=a.main, extra=a.extra, ncol=int(X.shape[1])), open(d / "spec.json", "w"))
    for seed in (0, 1, 2):
        f = d / f"stack_ctx_s{seed}.npy"
        if f.exists():
            continue
        t0 = time.time()
        np.save(f, S.oof_stack(E, X, seed).astype(np.float32))
        S.log(f"  seed {seed} ({time.time() - t0:.0f}s)")


def load_stack(name):
    d = REF.get(name, OUT / name)
    if not (d / "stack_ctx_s2.npy").exists():
        d = OUT / name
    return np.mean([np.load(d / f"stack_ctx_s{s}.npy") for s in (0, 1, 2)], 0).astype(float)


def gate_ok(E, P, lconf):
    fr = E["fr"]
    lanes0 = fr.lanes5g.copy()
    fr["lanes5g"] = lanes0.where(~(lconf < GATE), None)
    _, cr = S.run_decode(E, P)
    fr["lanes5g"] = lanes0
    return S.ok_cols(E, cr)


def cmd_compare(a):
    E = setup("x69_siba")                       # frame / rows / twin tokens only; the main net does not enter decode
    fr = E["fr"]
    lc = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["DeviceId", "win", "lane_conf"])
    assert (lc.DeviceId.to_numpy() == fr.DeviceId.to_numpy()).all() and (lc.win.to_numpy() == fr.win.to_numpy()).all()
    lconf = lc.lane_conf.to_numpy(float)
    ok_a, ok_b = gate_ok(E, load_stack(a.a), lconf), gate_ok(E, load_stack(a.b), lconf)
    sig = fr.DeviceId.to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    res = {}
    for s in ("E", "R"):
        A, B = ok_a[s], ok_b[s]
        sc = ~np.isnan(A) & ~np.isnan(B)
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            m = sc & fr.wgroup.isin(fams).to_numpy()
            res[f"{s}_{pool}"] = {"n": int(m.sum()), a.a: C.acc_ci(A[m], sig[m]), a.b: C.acc_ci(B[m], sig[m]),
                                  "b_minus_a": C.delta_ci(A[m], B[m], sig[m])}
        m = sc & fr.wgroup.isin(C.GE30).to_numpy()
        res[f"{s}_ge30_by_class"] = {c: {"n": int((m & (cl == c)).sum()),
                                         "acc_a": round(float(A[m & (cl == c)].mean()), 4),
                                         "acc_b": round(float(B[m & (cl == c)].mean()), 4),
                                         "d": C.delta_ci(A[m & (cl == c)], B[m & (cl == c)], sig[m & (cl == c)])}
                                     for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        res[f"{s}_ge30_by_fold"] = {int(k): round(100 * float(np.mean(B[m & (fr.fold == k).to_numpy()])
                                                               - np.mean(A[m & (fr.fold == k).to_numpy()])), 3)
                                    for k in range(6)}
    f = OUT / f"cmp_{a.a}_vs_{a.b}.json"
    json.dump(res, open(f, "w"), indent=1, default=str)
    for k, v in res.items():
        S.log(f"{k}: {v}")
    S.log(f"-> {f}")


def cmd_check(a):
    """recompute the siba stack (seed 0) through this script and compare with note 69's saved one."""
    d = OUT / "siba_check"
    d.mkdir(parents=True, exist_ok=True)
    E = setup("x69_siba")
    Pt, Pn = E["Pt"], E["Pn"]
    X = np.hstack([S.stack_X(E["fr"], Pt, Pn), S.ctx_X(E, C.W_TREE * Pt + (1 - C.W_TREE) * Pn)])
    P = S.oof_stack(E, X, 0)
    ref = np.load(REF["siba"] / "stack_ctx_s0.npy")
    S.log(f"check vs s69/on67 seed 0: max |diff| {np.abs(P - ref).max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stack", "compare", "check"])
    ap.add_argument("--name", default="")
    ap.add_argument("--main", default="x69_siba")
    ap.add_argument("--extra", default="")
    ap.add_argument("--a", default="siba")
    ap.add_argument("--b", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
