"""Note 75: does the stacker (trained on the 3-seed mean of the siba fold models) lose accuracy when production feeds it
ONE siba model instead?  Six folds OOF: stacker for fold k trained exactly as s74 / note 69 (3 seeds, 3-seed siba mean
as its net input), applied to fold k with the net input = 3-seed mean (reference = note 69) or seed 0 / 1 / 2 alone;
gate .9 decode, ATSPM stack-aware score, paired signal bootstrap.

    python member75.py      -> %DC_WORK%/final_v3_work/v3fit/member75.json
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import cand64 as C  # noqa: E402
import s67_decider as S  # noqa: E402
import s74 as S74  # noqa: E402

OUT = C.DC_WORK / "final_v3_work" / "v3fit"


def main():
    import lightgbm as lgb
    import pandas as pd
    E = S74.setup("x69_siba")
    fr, C7 = E["fr"], E["C7"]
    Pt, Pn = E["Pt"], E["Pn"]
    X = np.hstack([S.stack_X(fr, Pt, Pn), S.ctx_X(E, C.W_TREE * Pt + (1 - C.W_TREE) * Pn)])
    alt = {}
    for s in (0, 1, 2):
        P1 = S74.net_probs(fr, f"x69_siba:{s}")
        P1 = np.where(np.isnan(P1[:, :1]), Pt, P1)
        alt[s] = np.hstack([S.stack_X(fr, Pt, P1), S.ctx_X(E, C.W_TREE * Pt + (1 - C.W_TREE) * P1)])
    y = np.array([C7.index(t) if isinstance(t, str) and t in C7 else -1 for t in fr.truth_v3s.to_numpy(object)])
    trm = (y >= 0) & ~fr.exclude_train_score.fillna(False).to_numpy(bool) & (fr.det_n_on >= 5).to_numpy()
    fo = fr.fold.to_numpy()
    Pref = np.zeros((len(fr), 7))
    Palt = {s: np.zeros((len(fr), 7)) for s in alt}          # stacker trained on the 3-seed mean, fed one seed
    Pmix = {s: np.zeros((len(fr), 7)) for s in alt}          # stacker trained on the three single seeds (3 x rows)
    Psame = {s: np.zeros((len(fr), 7)) for s in alt}         # stacker trained on seed s alone, fed seed s
    for seed in (0, 1, 2):
        prm = dict(objective="multiclass", num_class=7, learning_rate=0.05, num_leaves=7, max_depth=3,
                   min_data_in_leaf=300, lambda_l2=10.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   num_threads=6, verbose=-1, seed=seed)
        for k in range(6):
            tr = trm & (fo != k)
            te = fo == k
            m = lgb.train(prm, lgb.Dataset(X[tr], y[tr]), num_boost_round=150)
            Pref[te] += m.predict(X[te]) / 3
            for s in alt:
                Palt[s][te] += m.predict(alt[s][te]) / 3
            mx = lgb.train(prm, lgb.Dataset(np.vstack([alt[s][tr] for s in alt]), np.concatenate([y[tr]] * len(alt))),
                           num_boost_round=150)
            for s in alt:
                Pmix[s][te] += mx.predict(alt[s][te]) / 3
                ms = lgb.train(prm, lgb.Dataset(alt[s][tr], y[tr]), num_boost_round=150)
                Psame[s][te] += ms.predict(alt[s][te]) / 3
    lc = pd.read_parquet(C.OUT / "function_rows.parquet", columns=["lane_conf"]).lane_conf.to_numpy(float)
    ok_ref = S74.gate_ok(E, Pref, lc)
    sig = fr.DeviceId.to_numpy()
    res = {"ref_check_vs_note69_max_abs": float(np.abs(Pref - S74.load_stack("siba")).max())}
    for arm, PP in (("asis", Palt), ("mix", Pmix), ("same", Psame)):
        for s in alt:
            ok = S74.gate_ok(E, PP[s], lc)
            for st in ("E", "R"):
                a, b = ok_ref[st], ok[st]
                sc = ~np.isnan(a) & ~np.isnan(b)
                for pool, fams in {**C.POOLS, "all": C.FAMS}.items():
                    mm = sc & fr.wgroup.isin(fams).to_numpy()
                    res[f"{arm}_seed{s}_{st}_{pool}"] = {"ref3": C.acc_ci(a[mm], sig[mm]), "one": C.acc_ci(b[mm], sig[mm]),
                                                         "one_minus_ref3": C.delta_ci(a[mm], b[mm], sig[mm])}
            S.log(f"{arm} seed {s}: " + json.dumps({k: v["one_minus_ref3"] for k, v in res.items()
                                                    if k.startswith(f"{arm}_seed{s}_E")}))
    json.dump(res, open(OUT / "member75.json", "w"), indent=1)


if __name__ == "__main__":
    main()
