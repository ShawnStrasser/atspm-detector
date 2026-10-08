"""Note 109: simpler SETBACK -- one P50 LightGBM pooled over every sample length (log hours as a feature) instead of one
per length group (m30 / h6 / h24 / full = 4 x 3 seeds in the package), and 1 seed instead of 3.
Data = note-77 features (f77/sb7/feat.parquet), six folds folds_v4, printed Advance truth; same metric / bootstrap as
notes 58 / 77 (sb7_validate._m / _boot).  locked_v2 absent (sb7 loaders).  CPU 4 threads.

    python sb109.py      -> %DC_WORK%/s109/setback/eval109.json
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import sb7_validate as S7  # noqa: E402
import sb5_setback as SB  # noqa: E402

F77 = S7.DCW / "final_v3_work" / "f77" / "sb7"
OUT = S7.DCW / "s109" / "setback"
HOURS = {"m30": 0.5, "h6": 6.0, "h24": 24.0, "full": 66.0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = pd.read_parquet(F77 / "feat.parquet")
    lk = set(pd.read_csv(S7.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not F.dev.astype(str).str.lower().isin(lk).any()
    F["wg"] = F.win.map(S7.wgroup)
    F["log_hours"] = np.log(F.wg.map(HOURS).astype(float))
    L = F.dist_kind.eq("single") & F.print_fn.isin(SB.TARGET) & (F.n_on > 0)
    feats = S7.VARS["all"]
    for c in ("per_group", "per_group_s0", "pooled", "pooled_s0", "pooled_nolen"):
        F[c] = np.nan
    for k in range(6):
        for wg in S7.WGS:
            trm = L & (F.wg == wg) & (F.fold != k)
            te = (F.wg == wg) & (F.fold == k)
            F.loc[te, "per_group"] = S7.fit_p50(F[trm], F[te], feats, (0, 1, 2))[0]
            F.loc[te, "per_group_s0"] = S7.fit_p50(F[trm], F[te], feats, (0,))[0]
        trm = L & (F.fold != k)
        te = F.fold == k
        F.loc[te, "pooled"] = S7.fit_p50(F[trm], F[te], feats + ["log_hours"], (0, 1, 2))[0]
        F.loc[te, "pooled_s0"] = S7.fit_p50(F[trm], F[te], feats + ["log_hours"], (0,))[0]
        F.loc[te, "pooled_nolen"] = S7.fit_p50(F[trm], F[te], feats, (0, 1, 2))[0]
        S7.log(f"fold {k} done")
    adv = F.dist_kind.eq("single") & (F.print_fn == "Advance") & (F.n_on > 0)
    ests = ["per_group", "per_group_s0", "pooled", "pooled_s0", "pooled_nolen"]
    res = {"n": {wg: int((adv & (F.wg == wg)).sum()) for wg in S7.WGS}, "metrics": {}, "ci_vs_per_group": {}}
    for e in ests:
        res["metrics"][e] = {wg: S7._m(F.dist[adv & (F.wg == wg)].to_numpy(), F[e][adv & (F.wg == wg)].to_numpy(float))
                             for wg in S7.WGS}
        res["metrics"][e]["pooled_windows"] = S7._m(F.dist[adv].to_numpy(), F[e][adv].to_numpy(float))
        if e != "per_group":
            res["ci_vs_per_group"][e] = {wg: S7._boot(F[adv & (F.wg == wg)], "per_group", e) for wg in S7.WGS}
            res["ci_vs_per_group"][e]["pooled_windows"] = S7._boot(F[adv], "per_group", e)
        S7.log(f"{e:14s} " + " | ".join(f"{wg} {res['metrics'][e][wg]['medAE']} ft {res['metrics'][e][wg]['w50']}%"
                                        for wg in S7.WGS + ["pooled_windows"]))
    json.dump(res, open(OUT / "eval109.json", "w"), indent=1, default=str)
    S7.log(json.dumps(res["ci_vs_per_group"]))


if __name__ == "__main__":
    main()
