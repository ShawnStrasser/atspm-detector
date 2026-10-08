"""Note 86: candidate v4e = copy of v4d (note 88, v4m labels) with the 'mean3' context stacker replaced by the note-86 refit (trained on the
filtered 3-seed-mean OOF of the v4l siba fold nets x86_siba4l).  Networks, trees, 'single' fallback stacker and every
other file unchanged.  Each ONNX file is checked against the numpy evaluation of its text model on 2,000 real rows.

    python assemble86.py copy   -> %DC_WORK%/final_v3_candidate_v4e (fresh copy of v4d; refuses to overwrite)
    python assemble86.py stack  -> v4e weights/stacker/stacker_mean3_s{0,1,2}.onnx + stacker.json mean3 entry;
                                   %DC_WORK%/final_v3_work/v3fit86/trees_parity86.json
Then: python %DC_WORK%/final_v3_candidate_v4e/check.py --freeze ; python .../check.py ; python assemble86.py card
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
from common import DC_WORK  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
FIT = DC_WORK / "final_v3_work" / "v3fit86" / "stacker"
SRC = DC_WORK / "final_v3_candidate_v4d"
PKG = DC_WORK / "final_v3_candidate_v4e"
WTS = PKG / "weights" / "stacker"


def cmd_copy():
    assert not PKG.exists(), f"{PKG} exists"
    shutil.copytree(SRC, PKG, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"copied {SRC} -> {PKG}", flush=True)


def cmd_stack():
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(PKG))
    import trees_onnx as TO
    assert Path(TO.__file__).resolve().parent == PKG.resolve()
    sm = json.load(open(WTS / "stacker.json"))
    m3 = json.load(open(FIT / "stacker_mean3.json"))
    assert m3["feature_names"] == sm["feature_names"], "mean3 must read the same columns as v4d"
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
    v = sm["variants"]["mean3"]
    assert v["files"] == files
    v.update(trained=m3["variant"], net=m3["net"], trained_on=m3["trained_on"],
             use_when="the function network has >= 2 members (v4e ships 3 full-data members)")
    sm["variant"] = ("two variants: 'mean3' (default, 3 members; note 86 refit on the v4l siba fold-net OOF + v4m trees) and 'single' "
                     "(1 member; note 88, unchanged)")
    json.dump(sm, open(WTS / "stacker.json", "w"), indent=1)
    json.dump(res, open(DC_WORK / "final_v3_work" / "v3fit86" / "trees_parity86.json", "w"), indent=1)
    mx = max(r["max_abs_out"] for r in res.values())
    print(f"mean3 stacker (note 86) compiled, parity max |out diff| {mx:.2e}", flush=True)
    assert mx < 1e-9


def cmd_card():
    """model card entry + sha256 of every weights file (as assemble88)."""
    import hashlib
    wts = PKG / "weights"
    card = json.load(open(wts / "model_card.json"))
    oof = json.load(open(DC_WORK / "final_v3_work" / "f86" / "oof" / "oof86.json"))
    card["candidate_v4e_note86"] = {
        "changes": ["context stacker 'mean3' re-fitted on the FILTERED 3-seed-mean OOF of siba fold nets trained on v4l "
                    "labels (x86_siba4l, six folds x seeds 0/1/2, post-note-76 inputs) instead of the v3s-trained x69_siba "
                    "unfiltered OOF; trees = v4m OOF (as v4d); everything else = v4d"],
        "oof_ge30": {"E": oof["E_ge30"]["n86"], "R": oof["R_ge30"]["n86"],
                     "vs_v4d_E_pt": oof["E_ge30"]["n86 - v4d"], "vs_v4d_R_pt": oof["R_ge30"]["n86 - v4d"]},
        "oof_5min_E": oof["E_m5"]["n86"], "oof_10min_E": oof["E_m10"]["n86"]}
    card["file_sha256"] = {str(p.relative_to(wts)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(wts.rglob("*")) if p.is_file() and p.name != "model_card.json"}
    card["total_model_bytes"] = sum(p.stat().st_size for p in wts.rglob("*") if p.is_file() and p.name != "model_card.json")
    json.dump(card, open(wts / "model_card.json", "w"), indent=1, default=str)
    print("model card updated", flush=True)


if __name__ == "__main__":
    {"copy": cmd_copy, "stack": cmd_stack, "card": cmd_card}[sys.argv[1]]()
