"""Note 77: setback (sb7, note 58) with the channel-order-free travel-time block: the same-lane pair model scored on
both orientations and averaged, behavioural tie-breaks in the partner pick (trackA/sb5_setback.py, package setback.py).

    python sb77.py feats    # sb7_validate.stage_feats with the fixed code      -> f77/sb7/feat.parquet
    python sb77.py eval     # 'all' P50 (3 seeds, six folds, per window group) on the new features vs note 58's OOF
                            # 'all' (sb7/oof.parquet); paired signal bootstrap  -> f77/sb7/eval.json
locked_v2 asserted absent (sb7 loaders).  CPU only.
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

OLD = S7.OUT
NEW = S7.DCW / "final_v3_work" / "f77" / "sb7"


def feats():
    S7.OUT = NEW
    NEW.mkdir(parents=True, exist_ok=True)
    S7.stage_feats()


def evaluate():
    F = pd.read_parquet(NEW / "feat.parquet")
    F["wg"] = F.win.map(S7.wgroup)
    L = F.dist_kind.eq("single") & F.print_fn.isin(SB.TARGET) & (F.n_on > 0)
    F["new"] = np.nan
    for wg in S7.WGS:
        for k in range(6):
            trm = L & (F.wg == wg) & (F.fold != k)
            te = (F.wg == wg) & (F.fold == k)
            p, _ = S7.fit_p50(F[trm], F[te], S7.VARS["all"], (0, 1, 2))
            F.loc[te, "new"] = p
        S7.log(f"{wg} done")
    O = pd.read_parquet(OLD / "oof.parquet", columns=["dev", "det", "win", "all", "all_s3", "all_s4", "all_s5"])
    k = ["dev", "det", "win"]
    X = F.merge(O, on=k, how="inner")
    assert len(X) == len(F), (len(X), len(F))
    adv = X.dist_kind.eq("single") & (X.print_fn == "Advance") & (X.n_on > 0)
    res = {"metrics": {}, "ci_new_vs_all": {}, "ci_s3_vs_all": {},
           "inputs_changed": {c: int(((X[c] if c in X else np.nan) != (X[c] if c in X else np.nan)).sum())
                              for c in ()}}
    oldF = pd.read_parquet(OLD / "feat.parquet")
    m = oldF.merge(F, on=k, suffixes=("_o", "_n"))
    for c in ("p_same", "tau", "same", "d_user", "p_same_any", "tau_any"):
        a, b = m[c + "_o"].astype(float), m[c + "_n"].astype(float)
        res["inputs_changed"][c] = int((((a - b).abs() > 1e-9) | (a.isna() != b.isna())).sum())
    res["inputs_rows"] = int(len(m))
    for e in ("all", "new", "all_s3", "all_s4", "all_s5"):
        res["metrics"][e] = {wg: S7._m(X.dist[adv & (X.wg == wg)].to_numpy(), X[e][adv & (X.wg == wg)].to_numpy(float))
                             for wg in S7.WGS}
    for wg in S7.WGS:
        res["ci_new_vs_all"][wg] = S7._boot(X[adv & (X.wg == wg)], "all", "new")
        res["ci_s3_vs_all"][wg] = S7._boot(X[adv & (X.wg == wg)], "all", "all_s3")
    res["pooled_new_vs_all"] = S7._boot(X[adv], "all", "new")
    json.dump(res, open(NEW / "eval.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"feats": feats, "eval": evaluate}[sys.argv[1]]()
