"""Note 75: assemble the weights of `%DC_WORK%/final_v3_candidate_v3` (copy of candidate v2, extended).

Sources (text LightGBM models are kept OUTSIDE the package, in %DC_WORK%/final_v3_work/v3fit/; the package holds only
the compiled ONNX tree ensembles + their json metadata):
  phase ranker   candidate v2 `phase_lgbm_v5_s{0,1,2}.txt` (note 37 full fit = note-57 recipe; NOT refitted)
  decoder        v3fit/decoder/decode_v3.txt               (refit, fit75.py)
  function       v3fit/function/function229_s{0,1,2}.txt   (refit)
  lanes D        v3fit/lanes/lane_pair_D_s{0,1,2}.txt      (refit)
  stacker        v3fit/stacker/stacker_s{0,1,2}.txt        (refit)
  setback        v3fit/setback/setback_{m30,h6,h24,full}_q50_s*.txt (refit, sb7) + candidate v2 note-41 q10 / q90 (band)
  GRU            candidate v2 gru.onnx (p3_final, unchanged); funcnet/ = siba ONNX (export75.py siba)

    python assemble75.py          -> converts, writes metas, verifies each ONNX file vs the numpy evaluation of its text
                                     model on real rows of its own frame (where cheap) -> v3fit/trees_parity_frames.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

V2 = DC_WORK / "final_v3_candidate_v2" / "weights"
FIT = DC_WORK / "final_v3_work" / "v3fit"
PKG = DC_WORK / "final_v3_candidate_v3"
WTS = PKG / "weights"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def plan() -> list:
    """(source txt, destination onnx relative to weights/)."""
    out = [(V2 / f"phase_lgbm_v5_s{s}.txt", f"phase_lgbm_v5_s{s}.onnx") for s in range(3)]
    out.append((FIT / "decoder" / "decode_v3.txt", "decode_v3.onnx"))
    out += [(FIT / "function" / f"function229_s{s}.txt", f"function/function229_s{s}.onnx") for s in range(3)]
    out += [(FIT / "lanes" / f"lane_pair_D_s{s}.txt", f"lanes/lane_pair_D_s{s}.onnx") for s in range(3)]
    out += [(FIT / "stacker" / f"stacker_s{s}.txt", f"stacker/stacker_s{s}.onnx") for s in range(3)]
    out += [(FIT / "stacker" / f"stacker_single_s{s}.txt", f"stacker/stacker_single_s{s}.onnx") for s in range(3)]
    for g in ("m30", "h6", "h24", "full"):
        out += [(FIT / "setback" / f"setback_{g}_q50_s{s}.txt", f"setback/setback_{g}_q50_s{s}.onnx") for s in range(3)]
    for q in ("q10", "q90"):
        out += [(V2 / "setback" / f"setback_{q}_s{s}.txt", f"setback/setback_{q}_s{s}.onnx") for s in range(3)]
    return out


def main():
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(PKG))
    import trees_onnx as TO
    assert Path(TO.__file__).resolve().parent == PKG.resolve()
    # ---- clean the copied v2 weights: text models and superseded files go
    for f in list(WTS.rglob("*.txt")):
        f.unlink()
    for f in ("function_lgbm_v5.json", "decode_lgbm_v5.json", "lanes/lane_model.json"):
        (WTS / f).unlink(missing_ok=True)
    nbs = {}
    for src, rel in plan():
        dst = WTS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        nb = NumpyBooster(src)
        OT.to_onnx_te5(nb, dst)
        nbs[rel] = (src, nb)
    log(f"compiled {len(nbs)} tree models")
    # ---- metadata
    fm = json.load(open(FIT / "function" / "function229.json"))
    fm["onnx_files"] = [f"function229_s{s}.onnx" for s in range(3)]
    json.dump(fm, open(WTS / "function" / "function229.json", "w"), indent=1)
    dm = json.load(open(FIT / "decoder" / "decode_v3.json"))
    json.dump(dm, open(WTS / "decode_v3.json", "w"), indent=1)
    lm = json.load(open(FIT / "lanes" / "lane_model.json"))
    lm["onnx_files"] = [f"lane_pair_D_s{s}.onnx" for s in range(3)]
    json.dump(lm, open(WTS / "lanes" / "lane_model.json", "w"), indent=1)
    sm = json.load(open(FIT / "stacker" / "stacker.json"))
    sm["onnx_files"] = [f"stacker_s{s}.onnx" for s in range(3)]
    sm["variants"] = {"mean3": {"files": [f"stacker_s{s}.onnx" for s in range(3)],
                                "use_when": "the function network has >= 2 members (trained on the 3-seed mean of the "
                                            "siba fold models, note 69)"},
                      "single": {"files": [f"stacker_single_s{s}.onnx" for s in range(3)],
                                 "use_when": "the function network has 1 member (trained on the three single-seed "
                                             "versions of every OOF row, note 75)"}}
    json.dump(sm, open(WTS / "stacker" / "stacker.json", "w"), indent=1)
    sb = json.load(open(V2 / "setback" / "setback_model.json"))
    sb["band_models"] = {"0.1": [f"setback_q10_s{s}.onnx" for s in range(3)],
                         "0.9": [f"setback_q90_s{s}.onnx" for s in range(3)]}
    sb.pop("quantile_models", None)
    sb["note"] = ("band (P10 / P90, confidence only) = note 41 (66 h, cqr); distance = setback_p50.json (sb7, note 58, "
                  "one P50 model per sample-length group)")
    json.dump(sb, open(WTS / "setback" / "setback_model.json", "w"), indent=1)
    p50 = json.load(open(FIT / "setback" / "setback_p50.json"))
    for g, d in p50["groups"].items():
        d["onnx_files"] = [f"setback_{g}_q50_s{s}.onnx" for s in range(3)]
    json.dump(p50, open(WTS / "setback" / "setback_p50.json", "w"), indent=1)
    shutil.copy(V2 / "setback" / "setback_pair_hgb.json", WTS / "setback" / "setback_pair_hgb.json")
    (WTS / "setback" / "export_parity.json").unlink(missing_ok=True)
    # ---- parity on real rows of each model's own frame
    res = {}
    rows = frames(fm, sm, p50)
    for rel, (src, nb) in nbs.items():
        X = rows.get(rel.split("/")[0] if "/" in rel else rel.split("_s")[0])
        if X is None:
            res[rel] = {"checked_here": False}
            continue
        if isinstance(X, pd.DataFrame):
            X = X[nb.feature_names].to_numpy(np.float64, na_value=np.nan)
        m = TO.OnnxTrees(WTS / rel, 6)
        r0, r1 = OT.raw_numpy(nb, X), m.raw(X)
        p0, p1 = nb.predict(X), m.predict(X)
        res[rel] = {"rows": int(len(X)), "max_abs_raw": float(np.abs(r0 - r1).max()),
                    "max_abs_out": float(np.abs(np.asarray(p0) - np.asarray(p1)).max())}
        log(f"{rel}: {res[rel]}")
    json.dump(res, open(FIT / "trees_parity_frames.json", "w"), indent=1)


def frames(fm, sm, p50) -> dict:
    """real rows: phase pool, function frame v6e, stacker training X, setback sb7 features (all non-locked)."""
    import pyarrow.parquet as pq
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    out = {}
    feats = json.load(open(V2 / "phase_lgbm_v5.json"))["features"]
    t = pq.read_table(DC_WORK / "trees57" / "phase" / "pool.parquet", columns=["DeviceId"] + feats)
    rng = np.random.default_rng(75)
    idx = np.sort(rng.choice(t.num_rows, 30000, replace=False))
    ph = t.take(idx).to_pandas()
    assert not ph.DeviceId.str.replace("@stg", "").str.lower().isin(lk).any()
    out["phase_lgbm_v5"] = ph[feats]
    import v3_retrain as V
    V.set_frame("v6e")
    t = pq.read_table(V.FEATS, columns=["DeviceId"] + fm["features"])
    idx = np.sort(rng.choice(t.num_rows, 30000, replace=False))
    fr = t.take(idx).to_pandas()
    assert not fr.DeviceId.str.lower().isin(lk).any()
    out["function"] = fr[fm["features"]]
    out["stacker"] = np.load(FIT / "stacker" / "X_check.npy").astype(np.float64)
    import sb5_setback as SB
    import sb7_validate as S7
    F = pd.read_parquet(S7.OUT / "feat.parquet")
    assert not F.dev.isin(lk).any()
    out["setback"] = SB.add_physics(F, 50.0).astype({c: float for c in SB.FEATS})
    return out


if __name__ == "__main__":
    main()
