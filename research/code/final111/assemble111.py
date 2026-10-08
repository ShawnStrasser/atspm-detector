"""Note 111: package `%DC_WORK%/final_v3_candidate_v6` = note-109 design C, built from a copy of v5c (v5c untouched) with
the fast-profile code of v5b_fast (= v5c's code + profiles; check.py 5th check) and these changes:
  phase     the phase TCN is gone: the three w32 siba members' pair phase head is the phase network (predict.score /
            funcnet.FuncNet.both, one memoised pass shared with the function stage); ranker ONE seed (s0); decoder refit
            on ranker-s0 0.5 / siba head 0.5 (fit111 dec); fast profiles: decode_trees refit on ranker s0 alone
  function  trees ONE seed (s0); stackers mean3 / single / nonet at ONE seed on the 1-seed trees OOF (fit111 stack)
  setback   P50 ONE seed per length group (s0); band P10 / P90 and pair model unchanged
  kept      lanes (v5c, note 105), health (v5c), night speed, decode rules
Ranker s0 / trees s0 / setback s0 = the existing full-data seed-0 fits (= the 1-seed refit of the same recipe).

    python assemble111.py init      copy v5c -> v6 + fast code (done by hand on 2026-10-06; asserts the layout)
    python assemble111.py weights   1-seed files, new decoder / decode_trees / stackers (ONNX + parity), blend.json, fast.json
    python assemble111.py card      model card entry final_v6_note111 + sha256
Then: python <v6>/check.py --freeze ; python <v6>/check.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402,F401
from common import DC_WORK  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
PKG = DC_WORK / "final_v3_candidate_v6"
WTS = PKG / "weights"
FIT = DC_WORK / "final_v3_work" / "v3fit111"
OUT = DC_WORK / "s111"


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


def _rm(p: Path):
    if p.exists():
        p.unlink()


def cmd_init():
    assert (PKG / "predict.py").exists() and (WTS / "phase_lgbm_v5.json").exists()
    assert "fast_profile" in (PKG / "predict.py").read_text(encoding="utf-8")
    print("layout ok")


def cmd_weights():
    OT, NumpyBooster, TO = _tools()
    par = {}
    # ---- phase ranker: ONE seed
    pm = json.load(open(WTS / "phase_lgbm_v5.json"))
    for s in (1, 2):
        _rm(WTS / f"phase_lgbm_v5_s{s}.onnx")
    pm["n_models"], pm["seeds"] = 1, [0]
    pm["note111"] = ("ONE seed (s0 = the note-95 full-data seed-0 fit; a 1-seed refit of the same recipe): note 109 "
                     "measured tree seeds as free to drop (design C)")
    json.dump(pm, open(WTS / "phase_lgbm_v5.json", "w"), indent=1)
    # ---- phase TCN out; the siba members' pair phase head is the phase network
    _rm(WTS / "phase_tcn.onnx")
    b = json.load(open(WTS / "blend.json"))
    b["weights_file"] = "funcnet/manifest.json"
    b["network_kind"] = "siba_phase_head"
    b["network"] = {"arch": "the function network's pair phase head (siba w32, 3 members, weights/funcnet): per member a "
                            "softmax over the kept candidates of the piece-mean pair logit, members averaged",
                    "candidate_filter": "kept = ranker p >= .01 (the same pairs the function network runs on)",
                    "note": "note 111 (design C of note 109): one network kind for both tasks; the separate phase TCN is "
                            "gone"}
    b["note111"] = "weight_lightgbm = 0.5 on the 1-seed ranker, 0.5 on the siba phase head (note-109 OOF recipe)"
    json.dump(b, open(WTS / "blend.json", "w"), indent=1)
    # ---- decoder refit on that blend
    d = FIT / "decoder"
    X = np.load(d / "X_check.npy").astype(np.float64)
    old = json.load(open(WTS / "decode_v3.json"))
    dm = json.load(open(d / "decode_v3.json"))
    assert dm["features"] == old["features"] and dm["mode"] == old["mode"], "decoder columns changed"
    par["decode_v3.onnx"] = _conv(OT, NumpyBooster, TO, d / "decode_v3.txt", WTS / "decode_v3.onnx", X)
    json.dump(dm, open(WTS / "decode_v3.json", "w"), indent=1)
    d = FIT / "decode_trees"
    X = np.load(d / "X_check.npy").astype(np.float64)
    dm = json.load(open(d / "decode_v3.json"))
    assert dm["features"] == old["features"]
    par["decode_trees.onnx"] = _conv(OT, NumpyBooster, TO, d / "decode_v3.txt", WTS / "decode_trees.onnx", X)
    json.dump(dm, open(WTS / "decode_trees.json", "w"), indent=1)
    # ---- function trees: ONE seed
    fm = json.load(open(WTS / "function" / "function229.json"))
    for s in (1, 2):
        _rm(WTS / "function" / f"function229_s{s}.onnx")
    fm["n_models"], fm["files"], fm["onnx_files"] = 1, ["function229_s0.txt"], ["function229_s0.onnx"]
    fm["note111"] = "ONE seed (s0 = the note-95 full-data seed-0 fit)"
    json.dump(fm, open(WTS / "function" / "function229.json", "w"), indent=1)
    # ---- stackers: mean3 / single / nonet at ONE seed
    sd = WTS / "stacker"
    sm = json.load(open(sd / "stacker.json"))
    for f in sd.glob("stacker_*.onnx"):
        f.unlink()
    sm["variants"], sm["files"] = {}, {}
    use = {"mean3": "the function network ran with >= 2 members (default)",
           "single": "the function network has 1 member (fast profile siba1)",
           "nonet": "the function network did not run (fast profile above its length limit)"}
    for kind in ("mean3", "single", "nonet"):
        meta = json.load(open(FIT / "stacker" / f"stacker_{kind}.json"))
        assert meta["feature_names"] == sm["feature_names"], f"{kind}: stacker columns changed"
        X = np.load(FIT / "stacker" / f"X_check_{kind}.npy").astype(np.float64)
        f = f"stacker_{kind}_s0.onnx"
        par[f] = _conv(OT, NumpyBooster, TO, FIT / "stacker" / f"stacker_{kind}_s0.txt", sd / f, X)
        sm["variants"][kind] = {"files": [f], "use_when": use[kind], "trained": meta["variant"],
                                "trained_on": meta["trained_on"]}
        sm["files"][kind] = meta["files"]
    sm["onnx_files"] = sm["variants"]["mean3"]["files"]
    sm["variant"] = "three variants at ONE seed each (note 111): mean3 (default), single, nonet; 1-seed trees OOF inputs"
    sm["labels"] = "v4q (note 95); trees = 2026-only OOF of seed 0 only (note 111)"
    sm["net"] = "siba x100_w32 six-fold OOF (3-member mean for mean3; each member for single)"
    json.dump(sm, open(sd / "stacker.json", "w"), indent=1)
    # ---- setback P50: ONE seed per length group
    sp = json.load(open(WTS / "setback" / "setback_p50.json"))
    for g, gd in sp["groups"].items():
        for s in (1, 2):
            _rm(WTS / "setback" / f"setback_{g}_q50_s{s}.onnx")
        gd["files"], gd["onnx_files"] = [f"setback_{g}_q50_s0.txt"], [f"setback_{g}_q50_s0.onnx"]
    sp["note111"] = "ONE seed per length group (s0 = the note-95 seed-0 fit); note 109: tie (+0.3 [-0.3,+0.8] w50 pt)"
    json.dump(sp, open(WTS / "setback" / "setback_p50.json", "w"), indent=1)
    # ---- function network manifest: it is the phase network too
    mf = json.load(open(WTS / "funcnet" / "manifest.json"))
    mf["phase_head"] = ("note 111: the pair graph's output s (pair phase logit) is the package's phase network: per member "
                        "softmax over the kept candidates of the piece-mean logit, members averaged; blended 0.5 / 0.5 "
                        "with the 1-seed ranker before the joint decoder (blend.json network_kind siba_phase_head)")
    json.dump(mf, open(WTS / "funcnet" / "manifest.json", "w"), indent=1)
    # ---- fast profiles (one network now: phase and function limits move together)
    fast = {"default": "full",
            "note": "note 111 (v6): ONE network (the siba members) serves phase AND function, so phase_net_max_minutes and "
                    "func_net_max_minutes move together. null = always, 0 = never; siba_members: members used (manifest "
                    "order; null = all). Where the network did not run: decode_trees (ranker s0 alone) + stacker 'nonet'. "
                    "Pick with predict(..., profile=), --profile or DC_FAST_PROFILE. 'le2h' = the fast setting (network "
                    "only up to 2 h).",
            "profiles": {"full": {"phase_net_max_minutes": None, "func_net_max_minutes": None, "siba_members": None},
                         "siba1": {"phase_net_max_minutes": None, "func_net_max_minutes": None, "siba_members": 1},
                         "trees": {"phase_net_max_minutes": 0, "func_net_max_minutes": 0, "siba_members": 0},
                         "le2h": {"phase_net_max_minutes": 120, "func_net_max_minutes": 120, "siba_members": None},
                         "le2h_siba1": {"phase_net_max_minutes": 120, "func_net_max_minutes": 120, "siba_members": 1}}}
    json.dump(fast, open(WTS / "fast.json", "w"), indent=1)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(par, open(OUT / "trees_parity111.json", "w"), indent=1)
    print(json.dumps(par, indent=1))


def cmd_card():
    card = json.load(open(WTS / "model_card.json"))
    card["final_v6_note111"] = json.load(open(OUT / "card111.json"))
    card["file_sha256"] = {str(q.relative_to(WTS)).replace("\\", "/"): hashlib.sha256(q.read_bytes()).hexdigest()
                           for q in sorted(WTS.rglob("*")) if q.is_file() and q.name != "model_card.json"}
    card["total_model_bytes"] = sum(q.stat().st_size for q in WTS.rglob("*") if q.is_file() and q.name != "model_card.json")
    json.dump(card, open(WTS / "model_card.json", "w"), indent=1, default=str)
    print("model card updated")


if __name__ == "__main__":
    {"init": cmd_init, "weights": cmd_weights, "card": cmd_card}[sys.argv[1]]()
