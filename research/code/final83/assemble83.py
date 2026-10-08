"""Note 83: compile the fit83 refits into `%DC_WORK%/final_v3_candidate_v4/weights` (copy of candidate v3).

Replaced: function trees (v4l), lanes D (v4l lane truth + v4l function block), context stacker ('single' only, 47
columns, no health), setback P50 per length group (v4l store / v4l OOF function).  Unchanged (v3 files kept): phase
ranker, decoder, GRU, setback P10 / P90 band, setback pair model.  The function network is exported by export83.py.
Each ONNX file is checked against the numpy evaluation of its text model on real rows (function frame, stacker
training X, setback features): raw score and transformed output.

    python assemble83.py      -> weights rewritten + v3fit83/trees_parity_frames.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
from common import DC_WORK  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
FIT = DC_WORK / "final_v3_work" / "v3fit83"
PKG = DC_WORK / "final_v3_candidate_v4"
WTS = PKG / "weights"
GROUPS = ("m30", "h6", "h24", "full")


def plan() -> list:
    out = [(FIT / "function" / f"function229_s{s}.txt", f"function/function229_s{s}.onnx") for s in range(3)]
    out += [(FIT / "lanes" / f"lane_pair_D_s{s}.txt", f"lanes/lane_pair_D_s{s}.onnx") for s in range(3)]
    out += [(FIT / "stacker" / f"stacker_single_s{s}.txt", f"stacker/stacker_single_s{s}.onnx") for s in range(3)]
    for g in GROUPS:
        out += [(FIT / "setback" / f"setback_{g}_q50_s{s}.txt", f"setback/setback_{g}_q50_s{s}.onnx") for s in range(3)]
    return out


def main():
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(PKG))
    import trees_onnx as TO
    assert Path(TO.__file__).resolve().parent == PKG.resolve()
    for f in (WTS / "stacker").glob("stacker_s*.onnx"):          # the 'mean3' variant is not shipped in v4
        f.unlink()
    nbs = {}
    for src, rel in plan():
        dst = WTS / rel
        nb = NumpyBooster(src)
        OT.to_onnx_te5(nb, dst)
        nbs[rel] = nb
    print(f"compiled {len(nbs)} tree models", flush=True)
    fm = json.load(open(FIT / "function" / "function229.json"))
    fm["onnx_files"] = [f"function229_s{s}.onnx" for s in range(3)]
    json.dump(fm, open(WTS / "function" / "function229.json", "w"), indent=1)
    lm = json.load(open(FIT / "lanes" / "lane_model.json"))
    lm["onnx_files"] = [f"lane_pair_D_s{s}.onnx" for s in range(3)]
    json.dump(lm, open(WTS / "lanes" / "lane_model.json", "w"), indent=1)
    sm = json.load(open(FIT / "stacker" / "stacker.json"))
    files = [f"stacker_single_s{s}.onnx" for s in range(3)]
    sm["onnx_files"] = files
    sm["variants"] = {"single": {"files": files, "use_when": "the function network has 1 member (v4 ships one "
                                                             "full-data siba); no 'mean3' variant in v4"}}
    json.dump(sm, open(WTS / "stacker" / "stacker.json", "w"), indent=1)
    p50 = json.load(open(FIT / "setback" / "setback_p50.json"))
    for g, d in p50["groups"].items():
        d["onnx_files"] = [f"setback_{g}_q50_s{s}.onnx" for s in range(3)]
    p50["labels"] = "note 83: sb7 features on the v4l print store (cabinet_v4l) and the v4l OOF function"
    json.dump(p50, open(WTS / "setback" / "setback_p50.json", "w"), indent=1)
    # ---- parity on real rows
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    rng = np.random.default_rng(83)
    import pyarrow.parquet as pq
    t = pq.read_table(DC_WORK / "final_v3_work" / "f76" / "frame_v6e_c" / "feat_frame.parquet",
                      columns=["DeviceId"] + fm["features"])
    idx = np.sort(rng.choice(t.num_rows, 30000, replace=False))
    fr = t.take(idx).to_pandas()
    assert not fr.DeviceId.str.lower().isin(lk).any()
    rows = {"function": fr[fm["features"]], "stacker": np.load(FIT / "stacker" / "X_check.npy").astype(np.float64)}
    sys.path.insert(0, str(rpath.CODE / "trackA"))
    import sb5_setback as SB
    F = pd.read_parquet(DC_WORK / "final_v3_work" / "f83" / "sb7" / "feat.parquet")
    assert not F.dev.isin(lk).any()
    rows["setback"] = SB.add_physics(F, 50.0).astype({c: float for c in SB.FEATS})
    res = {}
    for rel, nb in nbs.items():
        X = rows.get(rel.split("/")[0])
        if X is None:
            res[rel] = {"checked_here": False}
            continue
        if isinstance(X, pd.DataFrame):
            X = X[nb.feature_names].to_numpy(np.float64, na_value=np.nan)
        m = TO.OnnxTrees(WTS / rel, 4)
        r0, r1 = OT.raw_numpy(nb, X), m.raw(X)
        p0, p1 = nb.predict(X), m.predict(X)
        res[rel] = {"rows": int(len(X)), "max_abs_raw": float(np.abs(r0 - r1).max()),
                    "max_abs_out": float(np.abs(np.asarray(p0) - np.asarray(p1)).max())}
    json.dump(res, open(FIT / "trees_parity_frames.json", "w"), indent=1)
    mx = max(v.get("max_abs_out", 0.0) for v in res.values())
    print(f"parity: {sum(1 for v in res.values() if 'rows' in v)} checked, max |out diff| {mx:.2e}", flush=True)
    assert mx < 1e-9


if __name__ == "__main__":
    main()
