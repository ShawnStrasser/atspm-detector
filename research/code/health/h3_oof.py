"""Note 43: out-of-fold phase + function predictions per (signal, detector, window group), Sept 2026.

Both periods (Dec 2024 `dec`, Sept 2026 `stg`).  Function = first.all.wi OOF (note 28, frame_v6 run on the current label table, mean of 3 seeds);
phase = the frame's pred_phase (stage-12 ranker-bag + decoder OOF).  Nothing in-sample.
    python h3_oof.py -> %DC_WORK%/health3/oof_pf.parquet
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
FR = DCW / "trackA" / "v3" / "frame_v6"
RUN = FR / "run_d3f4e7ccbc_exclude_min5_clean_valnc"
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]


def main():
    fr = pd.read_parquet(FR / "feat_frame.parquet", columns=["DeviceId", "Detector", "period", "win", "wgroup",
                                                             "fold", "pred_phase"])
    P = np.zeros((len(fr), 7), np.float32)
    fo = fr.fold.to_numpy()
    for k in range(6):
        P[fo == k] = np.mean([np.load(RUN / f"P_first.all.wi_s{s}_f{k}.npy") for s in range(3)], axis=0)
    for i, c in enumerate(C7):
        fr[f"p_{c}"] = P[:, i]
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not fr.DeviceId.str.lower().isin(locked).any()
    fr["DeviceId"] = fr.DeviceId.str.lower()
    pc = [f"p_{c}" for c in C7]
    g = fr.groupby(["period", "DeviceId", "Detector", "wgroup"])
    out = g[pc].mean()
    out["pred_phase"] = g.pred_phase.agg(lambda s: s.mode().iat[0])
    out = out.reset_index().rename(columns={"Detector": "detector"})
    out["pred_function"] = np.array(C7)[out[pc].to_numpy().argmax(1)]
    out.to_parquet(DCW / "health3" / "oof_pf.parquet")
    print(out.shape, out.DeviceId.nunique(), out.wgroup.value_counts().to_dict())
    print(out[out.wgroup == "full"].pred_function.value_counts())


if __name__ == "__main__":
    main()
