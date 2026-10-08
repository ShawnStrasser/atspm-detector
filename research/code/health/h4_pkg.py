"""Note 46: score the function head on ARBITRARY windows of a training signal, out of fold.

Features come from the final_v3 candidate package's own inference path (predict.load_events ->
build_chunk_tables -> build_features -> _function_frame -> features_expert.build; note 32 checked it
against the training frame), but the PHASE is not re-predicted: every detector keeps the phase (and phase
probability, `top_prob`) the out-of-fold pipeline gave it - so only the function side is re-scored, by the
fold model that never saw the signal (h4_fold_models.py).  Research only (lightgbm).

    from h4_pkg import FunctionScorer
    fs = FunctionScorer(fold)                       # loads f{fold}_s{0,1,2}.txt
    fs.score(events_df, t0, t1, phase={det: p}, phase_prob={det: q}) -> DataFrame(Detector, p_*, n_on)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

PKG = H.DCW / "final_v3_candidate"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
import features_expert as fx  # noqa: E402
import predict as P  # noqa: E402

FOLDS = H.DCW / "health4" / "func_folds"
P._VERBOSE = False
P.log = lambda m: None


class FunctionScorer:
    def __init__(self, fold: int, seeds=(0, 1, 2)):
        import lightgbm as lgb
        fm = H.DCW / "trackA" / "v3" / "frame_v6" / "feat_meta.json"      # the same 442 columns (checked in meta.json)
        self.features = json.load(open(fm))["features"]
        self.classes = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
        self.models = [lgb.Booster(model_file=str(FOLDS / f"f{fold}_s{s}.txt")) for s in seeds]
        self.fold = fold

    def features_for(self, ev: pd.DataFrame, t0, t1, phase: dict, phase_prob: dict):
        """-> the function design matrix (one row per detector whose given phase greens in the window)."""
        t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
        e = ev[(ev.Timestamp >= t0) & (ev.Timestamp < t1)]
        if not len(e):
            return None
        con = P._connect(2, "3GB")
        try:
            w0, w1, info = P.load_events(con, e)
            if not info["n_events"]:
                return None
            P.build_chunk_tables(con)
            df, sim = P.build_features(con, w0, w1)
            if df is None or not len(df):
                return None
            det = df.Detector.astype(int).to_numpy()
            want = np.array([phase.get(int(d), -1) for d in det])
            df["prob"] = (df.cand_phase.astype(int).to_numpy() == want).astype(float)
            ok = df.groupby(["DeviceId", "Detector"])["prob"].transform("max") > 0
            df = df[ok].copy()
            if not len(df):
                return None
            df["prob_lgbm"] = df["prob"]
            top = P._function_frame(df)
            top["top_prob"] = top.Detector.astype(int).map(phase_prob).astype(float)
            ex = fx.build(con, *P._bin_window(t0, t1, w0, w1), top)
            ex["Detector"] = ex.Detector.astype(top.Detector.dtype)
            top = top.merge(ex, on=["DeviceId", "Detector"], how="left")
        finally:
            con.close()
        for c in self.features:
            if c not in top.columns:
                top[c] = np.nan
        return top

    def score(self, ev, t0, t1, phase: dict, phase_prob: dict) -> pd.DataFrame | None:
        top = self.features_for(ev, t0, t1, phase, phase_prob)
        if top is None or not len(top):
            return None
        X = top[self.features]
        Q = np.mean([m.predict(X) for m in self.models], axis=0)
        out = pd.DataFrame({"Detector": top.Detector.astype(int).to_numpy(),
                            "det_n_on": top.det_n_on.to_numpy() if "det_n_on" in top else np.nan})
        for i, c in enumerate(self.classes):
            out[f"p_{c}"] = Q[:, i]
        return out
