"""Monday headline table: GitHub beta (final_v2) vs candidate v4b (mean3 + siba filter), saved OOF only, nothing trained.
v4l truth, six folds folds_v4, >= 30 min pool, paired rows (only rows both arms score), E and R, signal bootstrap.
  function v4b  = note-84 arm mean3_flt0 (P_mean3_flt0.npy) through gate .9 decode (s74.gate_ok), as oof84 score
  function beta = note-25 b7 OOF -> 5 classes, argmax, no decode, stack-aware ATSPM credit (cand64.final_v2_function)
  phase v4b     = note-76 'new' arm (order-free trees 3 seeds + GRU p3 .01 filter + decoder; f76_q.parquet p2_new)
  phase beta    = note-48 final_v2 OOF on cand64 rows (cand64.phase_rows v2_pred)
    set F76_ARM=c & python monday85.py   -> %DC_WORK%/final_v3_work/monday_table_core.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("F76_ARM", "c")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final83"))
import fit83 as F  # noqa: E402

OUT = F.DC_WORK / "final_v3_work" / "monday_table_core.json"
OOF = F.DC_WORK / "final_v3_work" / "f84" / "oof"
CLASSES = ["Advance", "Presence", "Count", "Yellow_Red", "nonATSPM"]


def function_part(res):
    import cand64 as C
    import of77
    import atspm_score as AS
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    assert not fr.DeviceId.str.lower().isin(C.locked()).any()
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    ok4 = s74.gate_ok(E, np.load(OOF / "P_mean3_flt0.npy").astype(float), lc)
    champ = np.load(F.DC_WORK / "lab80" / "ok_v4l_champ.npz")
    assert (champ["sig"] == fr.DeviceId.to_numpy(str)).all()
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
    okb = {}
    for st in ("E", "R"):
        r = np.flatnonzero(~np.isnan(ok4[st]) & has)
        d = AS.credit(base, pred, "truth_v3s", r, True)
        o = np.full(len(base), np.nan)
        o[r] = d.ok_a.to_numpy(float)
        okb[st] = o
    sig = fr.DeviceId.to_numpy()
    tr = fr.truth_v3s.to_numpy(object)
    cl = np.where(np.isin(tr, S.ATS), tr, "nonATSPM")
    ge30 = fr.wgroup.isin(C.GE30).to_numpy()
    out = {}
    for st in ("E", "R"):
        all4 = ge30 & ~np.isnan(ok4[st])
        m = all4 & ~np.isnan(okb[st])
        blk = {"n_rows_paired": int(m.sum()), "signals": int(len(set(sig[m]))),
               "coverage_of_v4b_rows": round(float(m.sum() / all4.sum()), 4),
               "v4b_all_rows_unpaired": C.acc_ci(ok4[st][all4], sig[all4]),
               "overall": {"beta": C.acc_ci(okb[st][m], sig[m]), "v4b": C.acc_ci(ok4[st][m], sig[m]),
                           "delta_pt": C.delta_ci(okb[st][m], ok4[st][m], sig[m])}}
        for c in CLASSES:
            mc = m & (cl == c)
            blk[c] = {"n": int(mc.sum()), "beta": C.acc_ci(okb[st][mc], sig[mc]), "v4b": C.acc_ci(ok4[st][mc], sig[mc]),
                      "delta_pt": C.delta_ci(okb[st][mc], ok4[st][mc], sig[mc])}
        out[st] = blk
        F.log(f"function {st}: {blk}")
    res["function"] = out


def phase_part(res):
    import cand64 as C
    rows = C.phase_rows()
    q = pd.read_parquet(C.OUT / "phase_oof.parquet")
    f = pd.read_parquet(F.DC_WORK / "final_v3_work" / "f76" / "phase" / "f76_q.parquet")
    for c in C.KEY4:
        assert (q[c].to_numpy() == f[c].to_numpy()).all(), c
    q["p2_new"] = f.p2_new.to_numpy()
    rows["pred_new"], _ = C.top1(q, "p2_new", rows)
    import t57_phase as T57
    alt = T57.score_rows().set_index(C.DET).alt.reindex(rows.set_index(C.DET).index).to_numpy()
    okE = (rows.pred_new == rows.Phase).to_numpy()
    okR = okE | np.array([p in s for p, s in zip(rows.pred_new, alt)])
    rows["okE_new"], rows["okR_new"] = okE.astype(float), okR.astype(float)
    sig = rows.dev_plain.to_numpy()
    assert not pd.Series(sig).isin(C.locked()).any()
    out = {}
    for sname, okp, st in (("everything", "okE", "E"), ("realistic", "okR", "R")):
        m0 = rows[sname].to_numpy() & rows.fam.isin(C.GE30).to_numpy()
        m = m0 & rows.has_v2.to_numpy()
        a, b = rows[f"{okp}_v2"].to_numpy()[m], rows[f"{okp}_new"].to_numpy()[m]
        out[st] = {"n_rows_paired": int(m.sum()), "signals": int(len(set(sig[m]))),
                   "coverage": round(float(m.sum() / m0.sum()), 4),
                   "beta": C.acc_ci(a, sig[m]), "v4b": C.acc_ci(b, sig[m]), "delta_pt": C.delta_ci(a, b, sig[m])}
        F.log(f"phase {st}: {out[st]}")
    res["phase"] = out


if __name__ == "__main__":
    res = {}
    phase_part(res)
    function_part(res)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    F.log(f"-> {OUT}")
