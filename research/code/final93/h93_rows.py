"""Note 93: per-row OOF outcomes of the FINAL recipe (v4f), saved once so the health and Yellow_Red questions can be
answered without re-running the chain.

    python h93_rows.py func    -> %DC_WORK%/s93/func_rows.parquet   (v4f function OOF: decoded pred, probs, okE / okR)
    python h93_rows.py phase   -> %DC_WORK%/s93/phase_rows.parquet  (v4f phase OOF: TCN-blend decode top, okE / okR)

Function = oof90 stage_score's v4f arm exactly (P_v4f, gate .9 decode, v4l truth). Phase = p87 scorer on
s90/p87_q.parquet column p2_tcn_ad76 (the v4f decoder blend). Training signals only (locked_v2 asserted absent).
CPU 4 threads. No training.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final90"))
sys.path.insert(0, str(CODE / "final87"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import oof90 as O  # noqa: E402  (sets env DC_PKG_LABELS=v4o, F76_ARM=c, paths)

OUT = O.DCW / "s93"


def stage_func():
    import of77
    import s67_decider as S67
    O.use_v4o()
    s74, S, E, Xs, y, trm, names = O.F.stacker_inputs()
    fr = E["fr"]
    lc = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    P = np.load(O.OOF / "P_v4f.npy").astype(float)
    lanes0 = fr.lanes5g.copy()
    fr["lanes5g"] = lanes0.where(~(lc < s74.GATE), None)
    pr2, cr = S67.run_decode(E, P)
    fr["lanes5g"] = lanes0
    ok = S67.ok_cols(E, cr)
    C7 = list(E["C7"])
    keep = ["DeviceId", "Detector", "period", "win", "wgroup", "fold", "pred_phase", "det_n_on", "truth_v3s",
            "lanes5g", "exclude_train_score"]
    d = fr[[c for c in keep if c in fr.columns]].copy()
    d["lane_conf"] = lc
    pr = np.asarray(pr2)
    d["pred_func"] = [C7[int(i)] if np.issubdtype(type(i), np.integer) else i for i in pr] if pr.dtype != object \
        else pr
    for i, c in enumerate(C7):
        d[f"p_{c}"] = P[:, i].astype(np.float32)
    d["okE"], d["okR"] = ok["E"], ok["R"]
    import cand64 as C0
    assert not C0.plain(d.DeviceId).isin(C0.locked()).any()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "func_rows.parquet", index=False)
    import cand64 as C
    m = d.wgroup.isin(C.GE30).to_numpy()
    for st in ("okE", "okR"):
        v = d[st].to_numpy()
        mm = m & ~np.isnan(v)
        print(st, int(mm.sum()), C.acc_ci(v[mm], d.DeviceId.to_numpy()[mm]), flush=True)


def stage_phase():
    import cand64 as C
    import p87_phase as P87
    import t57_phase as T57
    q = P87.base()
    done = pd.read_parquet(O.DCW / "s90" / "p87_q.parquet", columns=C.KEY4 + ["p2_tcn_ad76"])
    for c in C.KEY4:
        assert (done[c].to_numpy() == q[c].to_numpy()).all(), c
    rows = C.phase_rows()
    rows = rows[[c for c in rows.columns if not c.startswith(("pred_", "okE_", "okR_", "p_"))]].copy()
    alt = T57.score_rows().set_index(C.DET).alt.reindex(rows.set_index(C.DET).index).to_numpy()
    pred, pr = C.top1(done, "p2_tcn_ad76", rows)
    okE = pred == rows.Phase.to_numpy()
    okR = okE | np.array([p in s_ for p, s_ in zip(pred, alt)])
    rows["pred_phase_v4f"], rows["p_phase_v4f"] = pred, pr
    rows["okE_all"], rows["okR_all"] = okE.astype(float), okR.astype(float)
    rows["okE"] = np.where(rows.everything.to_numpy(bool), okE, np.nan)
    rows["okR"] = np.where(rows.realistic.to_numpy(bool), okR, np.nan)
    assert not C.plain(rows.DeviceId).isin(C.locked()).any()
    keep = [c for c in rows.columns if c not in ("alt",) and rows[c].dtype != object or c in
            ("DeviceId", "win", "fam", "dev_plain")]
    OUT.mkdir(parents=True, exist_ok=True)
    rows[keep].to_parquet(OUT / "phase_rows.parquet", index=False)
    sig = rows.dev_plain.to_numpy()
    m = rows.fam.isin(C.GE30).to_numpy()
    for st in ("okE", "okR"):
        v = rows[st].to_numpy(float)
        mm = m & ~np.isnan(v)
        print(st, int(mm.sum()), C.acc_ci(v[mm], sig[mm]), flush=True)


if __name__ == "__main__":
    {"func": stage_func, "phase": stage_phase}[sys.argv[1]]()
