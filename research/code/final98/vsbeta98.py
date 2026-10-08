"""Note 98: every fast variant against the GitHub beta (final_v2) on paired rows, the note-85 protocol (saved OOF only).
  phase     s98/p87_rows.parquet ok columns (tcn_ad76 = v4f, trees_dec, le2h) vs cand64.phase_rows v2 (note-48 OOF;
            the beta's GRU runs <= 2 h, trees + decoder above)
  function  f98/oof ok_*.npz arms (+ the length-switched composites of f98_func.score) vs the beta function OOF
            (note-25 b7 -> 5 classes, argmax, stack-aware credit = monday85.function_part)
>= 30 min pool and the 3-h family alone; E and R; paired signal bootstrap.

    python vsbeta98.py   -> %DC_WORK%/final_v3_work/f98/oof/vsbeta98.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f98_func as FF  # noqa: E402

POOLS = {"ge30": ["m30", "h1", "h3", "h6", "h24", "full"], "h3": ["h3"], "gt2h": ["h3", "h6", "h24", "full"]}


def phase(res):
    import cand64 as C
    rows = C.phase_rows()
    r98 = pd.read_parquet(FF.S98 / "p87_rows.parquet")
    keep = [c for c in r98.columns if c.startswith(("okE_", "okR_")) and
            c.split("_", 1)[1] in ("tcn_ad76", "trees_dec", "le2h")]
    m = rows[C.DET + ["okE_v2", "okR_v2", "has_v2", "everything", "realistic", "fam", "dev_plain"]].merge(
        r98[C.DET + keep], on=C.DET, how="inner")
    assert len(m) == len(rows)
    sig = m.dev_plain.to_numpy()
    assert not pd.Series(sig).isin(C.locked()).any()
    out = {}
    for st, sname in (("E", "everything"), ("R", "realistic")):
        for pool, fams in POOLS.items():
            mm = m[sname].to_numpy() & m.fam.isin(fams).to_numpy() & m.has_v2.to_numpy()
            b = m[f"ok{st}_v2"].to_numpy()[mm]
            blk = {"n": int(mm.sum()), "beta": C.acc_ci(b, sig[mm])}
            for arm in ("tcn_ad76", "trees_dec", "le2h"):
                a = m[f"ok{st}_{arm}"].to_numpy()[mm]
                blk[arm] = {"acc": C.acc_ci(a, sig[mm]), "d_vs_beta": C.delta_ci(b, a, sig[mm])}
            out[f"{st}_{pool}"] = blk
            FF.F.log(f"phase {st} {pool}: {blk}")
    res["phase"] = out


def function(res):
    import cand64 as C
    import atspm_score as AS
    s74, S, E, y, trm, names = FF._setup()
    fr = E["fr"]
    assert not fr.DeviceId.str.lower().isin(C.locked()).any()
    ok = {}
    for f in sorted(FF.OUT.glob("ok_*.npz")):
        z = np.load(f)
        ph = f.stem[3:].rsplit("_", 1)[0]
        for k in z.files:
            nm, st = k.rsplit("_", 1)
            ok.setdefault(f"{ph}|{nm}", {})[st] = z[k]
    ks = [k for k in ok if k.startswith("tcn_ad76|single_s")]
    ok["tcn_ad76|single_avg"] = {st: np.mean([ok[k][st] for k in ks], axis=0) for st in ("E", "R")}
    short = fr.wgroup.isin(["m5", "m10", "m30", "h1"]).to_numpy()
    for nm, (s_, l_) in {"le2h_both|mean3>nonet": ("tcn_ad76|mean3", "trees_dec|nonet"),
                         "le2h_phase|mean3": ("tcn_ad76|mean3", "trees_dec|mean3"),
                         "le2h_both|single>nonet": ("tcn_ad76|single_avg", "trees_dec|nonet")}.items():
        ok[nm] = {st: np.where(short, ok[s_][st], ok[l_][st]) for st in ("E", "R")}
    key = ["DeviceId", "Detector", "period", "win"]
    base = fr[key + ["wgroup", "truth_v3s", "stack_group", "stack_role"]].copy()
    P5 = C.final_v2_probs(base)["v2"]
    has = ~np.isnan(P5[:, 0])
    C7 = list(AS.C7)
    P7 = np.zeros((len(base), 7))
    for j, c in enumerate(C.C5):
        P7[:, C7.index(c)] = np.nan_to_num(P5[:, j])
    pred = np.where(has, np.array([C7.index(c) for c in C.C5])[np.nan_to_num(P5, nan=-1).argmax(1)], C7.index("Other"))
    for i, c in enumerate(C7):
        base[f"P_{c}"] = P7[:, i]
    ref = ok["tcn_ad76|mean3"]
    okb = {}
    for st in ("E", "R"):
        r = np.flatnonzero(~np.isnan(ref[st]) & has)
        d = AS.credit(base, pred, "truth_v3s", r, True)
        o = np.full(len(base), np.nan)
        o[r] = d.ok_a.to_numpy(float)
        okb[st] = o
    sig = fr.DeviceId.to_numpy()
    out = {}
    for st in ("E", "R"):
        for pool, fams in POOLS.items():
            m = fr.wgroup.isin(fams).to_numpy() & ~np.isnan(okb[st]) & np.logical_and.reduce(
                [~np.isnan(v[st]) for v in ok.values()])
            blk = {"n": int(m.sum()), "signals": int(len(set(sig[m]))), "beta": C.acc_ci(okb[st][m], sig[m])}
            for k, v in ok.items():
                blk[k] = {"acc": C.acc_ci(v[st][m], sig[m]), "d_vs_beta": C.delta_ci(okb[st][m], v[st][m], sig[m])}
            out[f"{st}_{pool}"] = blk
            FF.F.log(f"function {st} {pool}: n {blk['n']} beta {blk['beta']}")
    res["function"] = out


if __name__ == "__main__":
    res = {}
    phase(res)
    function(res)
    json.dump(res, open(FF.OUT / "vsbeta98.json", "w"), indent=1, default=str)
    FF.F.log("== vsbeta done")
