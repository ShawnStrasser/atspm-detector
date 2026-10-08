"""Note 69: does the sibling-attention function net (siba, 3 seeds) still pay on top of the CURRENT champion
(note 67: context stacker, 3 seeds, + D-lane confidence gate .9)?

Re-runs note 67's `stack` stage (ctx arm only, seeds 0/1/2, OOF over folds) with siba in place of fj (cand64 FPRED pointed
at the siba fold files), then decodes the seed-averaged stacker output with the gate at .9 and compares it, paired, with
the same recipe on fj (note 67's saved `stack_ctx_s*.npy`).  Output %DC_WORK%/s69/on67/ (siba stack OOF + on67.json).

    python s69_on67.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cand64 as C  # noqa: E402
import s67_decider as S  # noqa: E402

OUT = C.DC_WORK / "s69" / "on67"
FJ67 = C.DC_WORK / "s67"
SIBA = ["x69_siba", "x69_siba_s1", "x69_siba_s2"]
GATE = 0.9


def gate_ok(E, P, lconf):
    fr = E["fr"]
    lanes0 = fr.lanes5g.copy()
    fr["lanes5g"] = lanes0.where(~(lconf < GATE), None)
    _, cr = S.run_decode(E, P)
    fr["lanes5g"] = lanes0
    return S.ok_cols(E, cr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = C.DC_WORK / "tmp" / "s69" / "on67_fp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for k in range(6):
        for i, t in enumerate(SIBA):
            shutil.copy(C.DC_WORK / "tcn53" / "fpreds" / f"{t}_f{k}.parquet",
                        tmp / (f"fj_f{k}.parquet" if i == 0 else f"fj_s{i}_f{k}.parquet"))
    lc = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["DeviceId", "win", "lane_conf"])  # cand64 rows
    C.FPRED = tmp
    S.OUT = OUT
    if not all((OUT / f"stack_ctx_s{s}.npy").exists() for s in (0, 1, 2)):
        S.stage_stack(type("A", (), {"arms": "ctx", "seeds": [0, 1, 2]})())
    E = S.setup()
    fr = E["fr"]
    assert (lc.DeviceId.to_numpy() == fr.DeviceId.to_numpy()).all() and (lc.win.to_numpy() == fr.win.to_numpy()).all()
    lconf = lc.lane_conf.to_numpy(float)
    P_siba = np.mean([np.load(OUT / f"stack_ctx_s{s}.npy") for s in (0, 1, 2)], 0).astype(float)
    P_fj = np.mean([np.load(FJ67 / f"stack_ctx_s{s}.npy") for s in (0, 1, 2)], 0).astype(float)
    ok_s, ok_f = gate_ok(E, P_siba, lconf), gate_ok(E, P_fj, lconf)
    sig = fr.DeviceId.to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    res = {}
    for s in ("E", "R"):
        a, b = ok_f[s], ok_s[s]
        sc = ~np.isnan(a) & ~np.isnan(b)
        for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
            m = sc & fr.wgroup.isin(fams).to_numpy()
            res[f"{s}_{pool}"] = {"n": int(m.sum()), "champion_fj": C.acc_ci(a[m], sig[m]),
                                  "champion_siba": C.acc_ci(b[m], sig[m]), "siba_minus_fj": C.delta_ci(a[m], b[m], sig[m])}
        m = sc & fr.wgroup.isin(C.GE30).to_numpy()
        res[f"{s}_ge30_by_class"] = {c: {"n": int((m & (cl == c)).sum()),
                                         "d": C.delta_ci(a[m & (cl == c)], b[m & (cl == c)], sig[m & (cl == c)])}
                                     for c in ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]}
        res[f"{s}_ge30_by_fold"] = {int(k): round(100 * float(np.mean(b[m & (fr.fold == k).to_numpy()])
                                                               - np.mean(a[m & (fr.fold == k).to_numpy()])), 3)
                                    for k in range(6)}
    json.dump(res, open(OUT / "on67.json", "w"), indent=1, default=str)
    for k, v in res.items():
        C.log(f"{k}: {v}")


if __name__ == "__main__":
    main()
