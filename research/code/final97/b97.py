"""Note 97 B: siba given the decided phase (tcn97_dec.py, x97_dec) vs x86_siba4l, fold 0 seed 0, same rows.

Net alone (argmax of 7, labelled rows, v4l truth) and a fixed blend (v4o trees OOF 0.6 + net 0.4 -> gate .9 decode, the
ATSPM E / R score), paired signal bootstrap.  Noise reference: x86_siba4l seed 1 vs seed 0 on the same fold.
    set F76_ARM=c & python b97.py [VARIANT ...]     -> %DC_WORK%/s97/b97.json
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
OUT = F.DC_WORK / "s97"
REF = "x86_siba4l_f0"
FOLD = 0


def main():
    import cand64 as C
    import of77
    O.use_v4o()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    Pt = np.load(O.OOF / "Pt_v4o.npy").astype(float)
    f0 = fr.fold.to_numpy() == FOLD

    def net(tag):
        n = pd.read_parquet(C.FPRED / f"{tag}.parquet")
        n["DeviceId"] = n.DeviceId.str.lower()
        n = n.astype({"Detector": fr.Detector.dtype}).set_index(s74.KEY)[s74.PC]
        return fr[s74.KEY].merge(n.reset_index(), on=s74.KEY, how="left")[s74.PC].to_numpy(float)

    tags = [REF, "x86_siba4l_s1_f0"] + (sys.argv[1:] or ["x97_dec_f0"])
    tags = [t for t in tags if (C.FPRED / f"{t}.parquet").exists()]
    Pn = {t: net(t) for t in tags}
    have = f0 & np.logical_and.reduce([~np.isnan(v[:, 0]) for v in Pn.values()])
    F.log(f"fold {FOLD}: {f0.sum()} rows, all nets present on {have.sum()}")
    # decoder-flag coverage (signal-periods in the phase pool)
    q = pd.read_parquet(O.DCW / "s90" / "p87_q.parquet", columns=["DeviceId", "win"]).drop_duplicates()
    q["period"] = np.where(q.DeviceId.str.endswith("@stg"), "stg", "dec")
    q["DeviceId"] = q.DeviceId.str.replace("@stg", "", regex=False).str.lower()
    k = fr[["DeviceId", "period", "win"]].assign(DeviceId=fr.DeviceId.str.lower())
    cov = k.merge(q.assign(c=1), on=["DeviceId", "period", "win"], how="left").c.notna().to_numpy()
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
    res = {"rows_fold0": int(f0.sum()), "rows_all_nets": int(have.sum()), "tags": tags}
    for pool, fams in C.POOLS.items():
        pm = have & fr.wgroup.isin(fams).to_numpy()
        for sub, sm in (("all", pm), ("flagged", pm & cov), ("unflagged", pm & ~cov)):
            r = {}
            m = sm & lab
            r["net_alone"] = {t: C.acc_ci(alone[t][m], sig[m]) for t in tags}
            r["net_alone_delta_vs_ref"] = {t: C.delta_ci(alone[REF][m], alone[t][m], sig[m]) for t in tags[1:]}
            r["n_labelled"] = int(m.sum())
            for st in ("E", "R"):
                mm = sm & np.logical_and.reduce([~np.isnan(ok[t][st]) for t in tags])
                r[f"blend_{st}"] = {t: C.acc_ci(ok[t][st][mm], sig[mm]) for t in tags}
                r[f"blend_{st}_delta_vs_ref"] = {t: C.delta_ci(ok[REF][st][mm], ok[t][st][mm], sig[mm]) for t in tags[1:]}
                r[f"n_{st}"] = int(mm.sum())
            res[f"{pool}_{sub}"] = r
    m = have & fr.wgroup.isin(C.GE30).to_numpy() & ~np.isnan(ok[REF]["E"])
    res["ge30_E_by_class_delta_vs_ref"] = {t: {c: [int((m & (cl == c)).sum())] + C.delta_ci(
        ok[REF]["E"][m & (cl == c)], ok[t]["E"][m & (cl == c)], sig[m & (cl == c)])
        for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]} for t in tags[1:]}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "b97.json", "w"), indent=1, default=str)
    for kk, v in res.items():
        F.log(f"{kk}: {v}")


if __name__ == "__main__":
    main()
