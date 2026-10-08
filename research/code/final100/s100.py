"""Note 100: distilled siba students vs the 3-member teacher (x86_siba4l fold 0 seeds 0/1/2), fold 0 held-out.

Same scorer as note 97 B (b97.py): net alone (argmax of 7, labelled rows, v4l truth) and the fixed blend (v4o trees OOF
0.6 + net 0.4 -> gate .9 decode, ATSPM E / R), paired signal bootstrap, on the rows where EVERY compared net has a
prediction. Reference nets: single seeds s0 / s1 / s2, their mean ('single' = the average of the three single-seed
scores is reported too) and mean3 (= average of the three probability tables, the teacher).
    set F76_ARM=c & python s100.py TAG [TAG ...]       -> %DC_WORK%/s100/s100.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
os.environ["DC_PKG_LABELS"] = "v4o"
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final90"))
import oof90 as O  # noqa: E402

F = O.F
OUT = F.DC_WORK / "s100"
FOLD = 0
SEEDS = ["x86_siba4l_f0", "x86_siba4l_s1_f0", "x86_siba4l_s2_f0"]


def main():
    import cand64 as C
    O.use_v4o()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    import of77
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    Pt = np.load(O.OOF / "Pt_v4o.npy").astype(float)
    f0 = fr.fold.to_numpy() == FOLD
    fp = Path(os.environ.get("S100_FPRED", str(C.FPRED)))

    def net(tag):
        d = C.FPRED if (C.FPRED / f"{tag}.parquet").exists() else fp
        n = pd.read_parquet(d / f"{tag}.parquet")
        n["DeviceId"] = n.DeviceId.str.lower()
        n = n.astype({"Detector": fr.Detector.dtype}).set_index(s74.KEY)[s74.PC]
        return fr[s74.KEY].merge(n.reset_index(), on=s74.KEY, how="left")[s74.PC].to_numpy(float)

    Pn = {t: net(t) for t in SEEDS}
    Pn["mean3"] = (Pn[SEEDS[0]] + Pn[SEEDS[1]] + Pn[SEEDS[2]]) / 3
    extra = sys.argv[1:]
    for t in extra:
        Pn[t] = net(t)
    tags = list(Pn)
    have = f0 & np.logical_and.reduce([~np.isnan(v[:, 0]) for v in Pn.values()])
    F.log(f"fold {FOLD}: {f0.sum()} rows, all nets present on {have.sum()}")
    tr = fr.truth_v3s.to_numpy(object)
    C7 = np.array(E["C7"])
    lab = np.array([t in E["C7"] for t in tr])
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    sig = fr.DeviceId.to_numpy()
    ok, alone = {}, {}
    for t, P in Pn.items():
        alone[t] = (C7[np.nan_to_num(P, nan=-1).argmax(1)] == tr).astype(float)
        B = np.where(have[:, None], 0.6 * Pt + 0.4 * np.nan_to_num(P), Pt)
        ok[t] = s74.gate_ok(E, B, lc)
    refs = ["mean3", SEEDS[0]]
    res = {"rows_fold0": int(f0.sum()), "rows_all_nets": int(have.sum()), "tags": tags}
    for pool, fams in C.POOLS.items():
        pm = have & fr.wgroup.isin(fams).to_numpy()
        r = {}
        m = pm & lab
        r["n_labelled"] = int(m.sum())
        r["net_alone"] = {t: C.acc_ci(alone[t][m], sig[m]) for t in tags}
        r["net_alone_single_mean"] = float(np.mean([alone[t][m].mean() for t in SEEDS]))
        for ref in refs:
            r[f"net_alone_delta_vs_{ref}"] = {t: C.delta_ci(alone[ref][m], alone[t][m], sig[m]) for t in tags if t != ref}
        for st in ("E", "R"):
            mm = pm & np.logical_and.reduce([~np.isnan(ok[t][st]) for t in tags])
            r[f"n_{st}"] = int(mm.sum())
            r[f"blend_{st}"] = {t: C.acc_ci(ok[t][st][mm], sig[mm]) for t in tags}
            r[f"blend_{st}_single_mean"] = float(np.mean([ok[t][st][mm].mean() for t in SEEDS]))
            for ref in refs:
                r[f"blend_{st}_delta_vs_{ref}"] = {t: C.delta_ci(ok[ref][st][mm], ok[t][st][mm], sig[mm])
                                                   for t in tags if t != ref}
        res[pool] = r
    # arms = seed groups (tag minus _sN): per-row correctness averaged over the arm's seeds, delta vs mean3 and vs the
    # seed-averaged single teacher member ('single'), paired signal bootstrap
    import re
    groups = {"single": SEEDS}
    for t in extra:
        groups.setdefault(re.sub(r"(_s\d+)?_f0$", "", t), []).append(t)
    arms = {}
    for pool, fams in C.POOLS.items():
        pm = have & fr.wgroup.isin(fams).to_numpy()
        m = pm & lab
        mmE = pm & np.logical_and.reduce([~np.isnan(ok[t]["E"]) for t in tags])
        mmR = pm & np.logical_and.reduce([~np.isnan(ok[t]["R"]) for t in tags])
        for g, ts in groups.items():
            a_ = np.mean([alone[t] for t in ts], 0)
            e_ = np.mean([ok[t]["E"] for t in ts], 0)
            r_ = np.mean([ok[t]["R"] for t in ts], 0)
            sE = np.mean([ok[t]["E"] for t in SEEDS], 0)
            sA = np.mean([alone[t] for t in SEEDS], 0)
            arms.setdefault(g, {"seeds": ts})[pool] = dict(
                alone=round(float(a_[m].mean()), 4), alone_seed_range=[round(float(alone[t][m].mean()), 4) for t in ts],
                E=round(float(e_[mmE].mean()), 4), E_seed_range=[round(float(ok[t]["E"][mmE].mean()), 4) for t in ts],
                R=round(float(r_[mmR].mean()), 4),
                dE_vs_mean3=C.delta_ci(ok["mean3"]["E"][mmE], e_[mmE], sig[mmE]),
                dE_vs_single=C.delta_ci(sE[mmE], e_[mmE], sig[mmE]),
                dR_vs_mean3=C.delta_ci(ok["mean3"]["R"][mmR], r_[mmR], sig[mmR]),
                dalone_vs_mean3=C.delta_ci(alone["mean3"][m], a_[m], sig[m]),
                dalone_vs_single=C.delta_ci(sA[m], a_[m], sig[m]))
    res["arms"] = arms
    m = have & fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok["mean3"]["E"])
    res["ge30_E_by_class_delta_vs_mean3"] = {t: {c: [int((m & (cl == c)).sum())] + C.delta_ci(
        ok["mean3"]["E"][m & (cl == c)], ok[t]["E"][m & (cl == c)], sig[m & (cl == c)])
        for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]} for t in tags if t != "mean3"}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "s100.json", "w"), indent=1, default=str)
    for kk, v in res.items():
        F.log(f"{kk}: {v}")


if __name__ == "__main__":
    main()
