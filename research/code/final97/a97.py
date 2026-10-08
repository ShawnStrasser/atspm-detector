"""Note 97 A: function side given the DECIDED phase (decoder top) instead of the phase-blend argmax.

Reuses note 90's stage_tcnphase arms (f90/oof/P_v4f.npy = frame phase = blend argmax; P_v4f_tcnphase.npy = stacker
context / lanes / gate decode re-built on the v4f decoder top, p2_tcn_ad76).  Adds what that stage did not report:
on the rows whose phase input changes, is the decoder's phase right more often, and what is the most the function
trees could gain if they were rebuilt on it (ceiling: function accuracy of changed rows lifted to the accuracy of
unchanged rows with the same phase-correctness).  CPU, no refit.

    set F76_ARM=c & python a97.py      -> %DC_WORK%/s97/a97.json
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


def main():
    import cand64 as C
    import of77
    import v3_retrain as V
    O.use_v4o()
    s74, S, E, Xs, y, trm, names = F.stacker_inputs()
    fr = E["fr"]
    F.log(f"frame columns: {list(fr.columns)[:60]}")
    lc0 = of77.lane_ctx(fr, of77.O8 / "lanes_D.func.parquet")[:, 2]
    P0 = np.load(O.OOF / "P_v4f.npy").astype(float)
    P1 = np.load(O.OOF / "P_v4f_tcnphase.npy").astype(float)
    # decoder top per row (as stage_tcnphase)
    q = pd.read_parquet(O.DCW / "s90" / "p87_q.parquet", columns=["DeviceId", "Detector", "win", "cand_phase", "p2_tcn_ad76"])
    k = ["DeviceId", "Detector", "win"]
    t = q.sort_values(k + ["p2_tcn_ad76"], ascending=[True, True, True, False]).groupby(k, sort=False).first()
    t = t.cand_phase.rename("dec_top").reset_index()
    t["period"] = np.where(t.DeviceId.str.endswith("@stg"), "stg", "dec")
    t["DeviceId"] = t.DeviceId.str.replace("@stg", "", regex=False).str.lower()
    key = fr[["DeviceId", "period", "Detector", "win"]].copy()
    key["DeviceId"] = key.DeviceId.str.lower()
    m = key.merge(t.astype({"Detector": fr.Detector.dtype}), on=["DeviceId", "period", "Detector", "win"], how="left")
    assert len(m) == len(fr)
    old = fr.pred_phase.to_numpy(float)
    new = np.where(m.dec_top.notna(), m.dec_top.to_numpy(float), old)
    ch = ~np.isnan(old) & (new != old)
    lc1 = np.where(ch, np.nan, lc0)
    ok0, ok1 = s74.gate_ok(E, P0, lc0), s74.gate_ok(E, P1, lc1)
    # phase correctness from the note-87/90 phase rows (E rule): decoder top (= new) and the GRU-pipeline top; the frame
    # phase (old) is judged by the GRU row only where old == GRU top (else unknown)
    rw = pd.read_parquet(O.DCW / "s90" / "p87_rows.parquet", columns=k + ["okE_tcn_ad76", "okE_gru"])
    g = pd.read_parquet(O.DCW / "s90" / "p87_q.parquet", columns=k + ["cand_phase", "p2_gru"])
    gt = g.sort_values(k + ["p2_gru"], ascending=[True, True, True, False]).groupby(k, sort=False).first()
    rw = rw.merge(gt.cand_phase.rename("gru_top").reset_index(), on=k, how="left")
    rw["period"] = np.where(rw.DeviceId.str.endswith("@stg"), "stg", "dec")
    rw["DeviceId"] = rw.DeviceId.str.replace("@stg", "", regex=False).str.lower()
    m2 = key.merge(rw.astype({"Detector": fr.Detector.dtype}), on=["DeviceId", "period", "Detector", "win"], how="left")
    assert len(m2) == len(fr)
    new_ok = m2.okE_tcn_ad76.to_numpy(float)
    old_ok = np.where(old == m2.gru_top.to_numpy(float), m2.okE_gru.to_numpy(float), np.nan)
    old_ok = np.where(~ch, new_ok, old_ok)
    tcol = "p87_rows okE"
    sig = fr.DeviceId.to_numpy()
    fo = fr.fold.to_numpy()
    res = {"rows": int(len(fr)), "changed": int(ch.sum()), "phase_truth_col": tcol}
    for st in ("E", "R"):
        for pool, fams in C.POOLS.items():
            pm = fr.wgroup.isin(fams).to_numpy()
            sc = pm & ~np.isnan(ok0[st]) & ~np.isnan(ok1[st])
            r = {"n": int(sc.sum()), "changed_scored": int((sc & ch).sum())}
            for nm, msk in (("all_folds", sc), ("fold0", sc & (fo == 0))):
                r[nm] = {"cur": C.acc_ci(ok0[st][msk], sig[msk]), "dec": C.acc_ci(ok1[st][msk], sig[msk]),
                         "dec - cur": C.delta_ci(ok0[st][msk], ok1[st][msk], sig[msk]), "n": int(msk.sum())}
            cm = sc & ch
            r["changed_rows_acc"] = {"cur": float(np.mean(ok0[st][cm])) if cm.any() else None,
                                     "dec": float(np.mean(ok1[st][cm])) if cm.any() else None}
            if True:
                oc, nc = old_ok == 1, new_ok == 1
                un = sc & ~ch & ~np.isnan(new_ok)
                a_right = float(np.mean(ok0[st][un & oc])); a_wrong = float(np.mean(ok0[st][un & (old_ok == 0)]))
                w2r, r2w = int((cm & (old_ok == 0) & nc).sum()), int((cm & oc & (new_ok == 0)).sum())
                r["phase"] = {"changed_old_right": int((cm & oc).sum()), "changed_new_right": int((cm & nc).sum()),
                              "changed_old_unknown": int((cm & np.isnan(old_ok)).sum()),
                              "changed_new_unknown": int((cm & np.isnan(new_ok)).sum()),
                              "wrong_to_right": w2r, "right_to_wrong": r2w,
                              "func_acc_unchanged_phase_right": round(a_right, 4),
                              "func_acc_unchanged_phase_wrong": round(a_wrong, 4),
                              "ceiling_gain_pt": round(100 * (w2r - r2w) * (a_right - a_wrong) / sc.sum(), 3),
                              "upper_bound_pt": round(100 * w2r / sc.sum(), 3)}
            res[f"{st}_{pool}"] = r
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "a97.json", "w"), indent=1, default=str)
    for kk, v in res.items():
        F.log(f"{kk}: {v}")


if __name__ == "__main__":
    main()
