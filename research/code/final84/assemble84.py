"""Note 84: add the 'mean3' context stacker (fit84) to `%DC_WORK%/final_v3_candidate_v4b/weights/stacker` next to the
'single' variant (kept, used when the function network has one member).  Each ONNX file is checked against the numpy
evaluation of its text model on 2,000 real stacker rows (raw score and probabilities).

    python assemble84.py   -> weights/stacker/stacker_mean3_s{0,1,2}.onnx + stacker.json variants; v3fit84/trees_parity84.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
from common import DC_WORK  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
FIT = DC_WORK / "final_v3_work" / "v3fit84" / "stacker"
PKG = DC_WORK / "final_v3_candidate_v4b"
WTS = PKG / "weights" / "stacker"


def main():
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(PKG))
    import trees_onnx as TO
    assert Path(TO.__file__).resolve().parent == PKG.resolve()
    sm = json.load(open(WTS / "stacker.json"))
    m3 = json.load(open(FIT / "stacker_mean3.json"))
    assert m3["feature_names"] == sm["feature_names"], "mean3 and single must read the same columns"
    X = np.load(FIT / "X_check_mean3.npy").astype(np.float64)
    res, files = {}, []
    for s in range(3):
        nb = NumpyBooster(FIT / f"stacker_mean3_s{s}.txt")
        dst = WTS / f"stacker_mean3_s{s}.onnx"
        OT.to_onnx_te5(nb, dst)
        m = TO.OnnxTrees(dst, 4)
        res[dst.name] = {"rows": int(len(X)), "max_abs_raw": float(np.abs(OT.raw_numpy(nb, X) - m.raw(X)).max()),
                         "max_abs_out": float(np.abs(np.asarray(nb.predict(X)) - np.asarray(m.predict(X))).max())}
        files.append(dst.name)
    single = sm["variants"]["single"]["files"]
    sm["variants"] = {
        "single": {"files": single, "use_when": "the function network has 1 member present (fallback)",
                   "trained": sm["variant"]},
        "mean3": {"files": files, "use_when": "the function network has >= 2 members (v4b ships 3 full-data members)",
                  "trained": m3["variant"], "net": m3["net"], "trained_on": m3["trained_on"]}}
    sm["onnx_files"] = files
    sm["files"] = {"single": sm["files"] if isinstance(sm["files"], list) else sm["files"].get("single"),
                   "mean3": m3["files"]}
    sm["variant"] = "two variants (note 84): 'mean3' (default, 3 members) and 'single' (1 member)"
    json.dump(sm, open(WTS / "stacker.json", "w"), indent=1)
    (DC_WORK / "final_v3_work" / "v3fit84").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(DC_WORK / "final_v3_work" / "v3fit84" / "trees_parity84.json", "w"), indent=1)
    mx = max(v["max_abs_out"] for v in res.values())
    print(f"mean3 stacker compiled, parity max |out diff| {mx:.2e}", flush=True)
    assert mx < 1e-9


if __name__ == "__main__":
    main()
