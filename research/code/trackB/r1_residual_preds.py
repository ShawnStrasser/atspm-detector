"""Stage 22, step 1: the stage-13 out-of-fold blend + joint decoder, kept per row.

Same arithmetic as `research/code/neural/trackb_eval_full.py` with the stage-13 GRU
(`preds/gru2/gru2_oof_f{0..5}_bywindow.parquet`): 0.5 x LightGBM ranker + 0.5 x GRU,
BEFORE the six-fold joint decoder, scored on `blend_v2.common_frame()` rows (701 training
signals, 22 windows, labelled phase greens in the window, >= 1 actuation).  Unlike the eval
script it keeps every row's decoded probability p2 next to the trees-alone (p_lg), the
network-alone (p_nn) and the ranker (p0), so the residual errors can be dissected.

    python research/code/trackB/r1_residual_preds.py
Output: DC_WORK/trackB/resid/blend13_rows.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import blend_v2 as B  # noqa: E402
import blend_v2_predecode as BP  # noqa: E402
import trackb_eval as TE  # noqa: E402

OUT = DC_WORK / "trackB" / "resid"
KEY, DET, GRP = B.KEY, B.DET, BP.GRP


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gru = B.gru_oof()                                    # stage-13 GRU, six folds, seed 0
    base = B.common_frame(gru)                           # p_lg, p_nn, p0, Phase, fold, n_act
    keep = base[KEY].copy()

    nn = TE._norm(gru)
    lg = TE._norm(pd.read_parquet(B.LG_OOF)).merge(B.folds(), on="DeviceId", how="inner")
    lab = B.labels()
    lg["dev_plain"] = lg.DeviceId.str.replace("@stg", "", regex=False)
    lg = lg.merge(lab.rename(columns={"DeviceId": "dev_plain"}),
                  on=["dev_plain", "Detector"], how="left")
    lg["y"] = np.where(lg.Phase.notna(), (lg.cand_phase == lg.Phase).astype(float), np.nan)
    lg = lg.merge(nn[KEY + ["prob"]].rename(columns={"prob": "p_nn"}), on=KEY, how="left")
    tot = lg.groupby(GRP)["p_nn"].transform("sum")
    lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
    meta = lg[KEY + ["Phase", "fold", "y"]]
    ctx, sim = BP.load_context(set(lg.DeviceId.unique()))

    pr = lg[KEY].copy()
    pr["p0"] = np.where(lg.p_nn.notna(), 0.5 * lg.p0 + 0.5 * lg.p_nn, lg.p0)
    pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
    out = BP.decode_oof(pr, ctx, sim, meta, n_jobs=6)

    q = base.merge(out, on=KEY, how="left")
    q["p2"] = q.p2.fillna(0.0)
    q.to_parquet(OUT / "blend13_rows.parquet", index=False)
    acc = {f: round(v["acc"], 5) for f, v in B.acc_by_fam(q, "p2").items()}
    json.dump({"blend_before": acc, "n_rows": int(len(q))},
              open(OUT / "blend13_check.json", "w"), indent=1)
    B.log(f"blend before (check vs stage 13 blend_before.json w=0.5): {acc}")


if __name__ == "__main__":
    main()
