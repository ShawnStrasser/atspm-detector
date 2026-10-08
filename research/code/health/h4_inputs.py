"""Note 46: the model inputs health v4 may use, per (period, signal, detector), all out of fold:
predicted phase + its probability (frame v6 `pred_phase` / `top_prob`: stage-12 trees OOF), function class
probabilities (the function_v3d recipe's six-fold OOF, run_ad6ea6959f_exclude_min5_clean_valnc_h3, 3-seed mean),
lanes spanned (h4_lanes.py).  One row per window group of the frame (full = the site answer).

    python h4_inputs.py -> %DC_WORK%/health4/inputs.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackA"))
import v3_retrain as V  # noqa: E402

RUN = "run_ad6ea6959f_exclude_min5_clean_valnc_h3"


def main():
    V.set_frame("v6")
    fr = pd.read_parquet(V.FEATS, columns=V.KEY + ["wgroup", "fold", "det_n_on", "pred_phase", "top_prob"])
    assert not fr.DeviceId.str.lower().isin(V.locked_signals()).any()
    P = np.mean([V.load_oof(fr, V.OUT / RUN, "first.all.wi", s, range(6))[0] for s in range(3)], 0)
    for i, c in enumerate(V.C7):
        fr[f"p_{c}"] = P[:, i]
    fr["DeviceId"] = fr.DeviceId.str.lower()
    pc = [f"p_{c}" for c in V.C7]
    g = fr.groupby(["period", "DeviceId", "Detector", "wgroup"])
    out = g[pc + ["top_prob"]].mean()
    out["pred_phase"] = g.pred_phase.agg(lambda s: s.mode().iat[0])
    out = out.reset_index().rename(columns={"Detector": "detector"})
    out["pred_function"] = np.array(V.C7)[out[pc].to_numpy().argmax(1)]
    L = pd.read_parquet(V.DCW / "health4" / "lanes_full.parquet")
    L = L.rename(columns={"Detector": "detector"})[["period", "DeviceId", "detector", "n_lanes_spanned"]]
    out = out.merge(L, on=["period", "DeviceId", "detector"], how="left")
    out.to_parquet(V.DCW / "health4" / "inputs.parquet", index=False)
    print(out.shape, out.groupby("wgroup").size().to_dict())


if __name__ == "__main__":
    main()
