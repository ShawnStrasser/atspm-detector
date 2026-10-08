"""Note 98: the fast package `%DC_WORK%/final_v3_candidate_v5_fast` = a copy of v4f (+ the note-95 health_score fix)
with fast profiles (weights_<base>/fast.json, code: predict / function_stage / stacker / funcnet) and the two extra
models the profiles need, added to the weight folder named by --base (default weights_v4f):
  decode_trees.{onnx,json}         joint decoder re-fitted on the trees alone (p98_phase.py fitdec --arm trees_dec)
  stacker/stacker_nonet_s*.onnx    stacker 'nonet' (no function network; f98_func.py pkgstacker --kind nonet)
  stacker/stacker_single_s*.onnx   stacker 'single' re-fitted on the x86_siba4l single-seed OOF (pkgstacker --kind single)
Each ONNX tree file is checked against the numpy evaluation of its text model on the fit's own X_check rows.

    python assemble98.py extras [--base weights_v4f] [--src <f98 dir>]   -> weights + f98/parity98.json
    python assemble98.py default <profile> [--base ...]                   -> fast.json "default"
    python assemble98.py card [--base ...]                                -> model card entry + sha256
Then: python <pkg>/check.py --freeze ; python <pkg>/check.py
Re-base onto v5: copy the v5 weights into weights_v5, refit the three extras on the v5 OOF (same recipes, inputs of the
v5 pipeline), run `extras --base weights_v5 --src <v5 extras dir>`, set package.json "weights": "weights_v5", freeze, check.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402,F401
from common import DC_WORK  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
PKG = DC_WORK / "final_v3_candidate_v5_fast"
F98 = DC_WORK / "final_v3_work" / "f98"


def _tools():
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(PKG))
    import trees_onnx as TO
    assert Path(TO.__file__).resolve().parent == PKG.resolve()
    return OT, NumpyBooster, TO


def _conv(OT, NumpyBooster, TO, src, dst, X):
    nb = NumpyBooster(src)
    OT.to_onnx_te5(nb, dst)
    m = TO.OnnxTrees(dst, 4)
    r = {"rows": int(len(X)), "max_abs_raw": float(np.abs(OT.raw_numpy(nb, X) - m.raw(X)).max()),
         "max_abs_out": float(np.abs(np.asarray(nb.predict(X)) - np.asarray(m.predict(X))).max())}
    assert r["max_abs_out"] < 1e-9, (dst, r)
    return r


def cmd_extras(a):
    OT, NumpyBooster, TO = _tools()
    W = PKG / a.base
    src = Path(a.src)
    par = {}
    d = src / "decoder_trees_dec"
    X = np.load(d / "X_check.npy").astype(np.float64)
    par["decode_trees.onnx"] = _conv(OT, NumpyBooster, TO, d / "decode_v3.txt", W / "decode_trees.onnx", X)
    dm = json.load(open(d / "decode_v3.json"))
    old = json.load(open(W / "decode_v3.json"))
    assert dm["features"] == old["features"] and dm["mode"] == old["mode"], "decoder columns changed"
    json.dump(dm, open(W / "decode_trees.json", "w"), indent=1)
    sm = json.load(open(W / "stacker" / "stacker.json"))
    for kind in ("nonet", "single"):
        ds = src / f"stacker_{kind}"
        meta = json.load(open(ds / f"stacker_{kind}.json"))
        assert meta["feature_names"] == sm["feature_names"], f"{kind}: stacker columns changed"
        X = np.load(ds / f"X_check_{kind}.npy").astype(np.float64)
        files = []
        for s in range(3):
            f = f"stacker_{kind}_s{s}.onnx"
            par[f] = _conv(OT, NumpyBooster, TO, ds / f"stacker_{kind}_s{s}.txt", W / "stacker" / f, X)
            files.append(f)
        sm["variants"][kind] = {
            "files": files, "trained": meta["variant"], "trained_on": meta["trained_on"],
            "use_when": ("the function network did not run (fast profile above its length limit)" if kind == "nonet"
                         else "the function network has 1 member (fallback / fast profile siba1)")}
        sm["files"][kind] = meta["files"]
    sm["variant"] = ("three variants: 'mean3' (3 members), 'single' (1 member; note 98 refit on the x86_siba4l single-"
                     "seed OOF), 'nonet' (no function network; note 98)")
    json.dump(sm, open(W / "stacker" / "stacker.json", "w"), indent=1)
    json.dump(par, open(F98 / f"parity98_{a.base}.json", "w"), indent=1)
    print(json.dumps(par, indent=1))


def cmd_default(a):
    f = PKG / a.base / "fast.json"
    c = json.load(open(f))
    assert a.profile in c["profiles"], a.profile
    c["default"] = a.profile
    json.dump(c, open(f, "w"), indent=1)
    print(f"default profile -> {a.profile}")


def cmd_card(a):
    W = PKG / a.base
    card = json.load(open(W / "model_card.json"))
    card["fast_note98"] = json.load(open(F98 / "card98.json"))
    card["file_sha256"] = {str(p.relative_to(W)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(W.rglob("*")) if p.is_file() and p.name != "model_card.json"}
    card["total_model_bytes"] = sum(p.stat().st_size for p in W.rglob("*") if p.is_file() and p.name != "model_card.json")
    json.dump(card, open(W / "model_card.json", "w"), indent=1, default=str)
    print("model card updated")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extras", "default", "card"])
    ap.add_argument("profile", nargs="?")
    ap.add_argument("--base", default="weights_v4f")
    ap.add_argument("--src", default=str(F98))
    a = ap.parse_args()
    assert PKG.exists(), PKG
    {"extras": cmd_extras, "default": cmd_default, "card": cmd_card}[a.cmd](a)
