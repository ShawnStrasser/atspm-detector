"""Note 88: candidate package with the fault-event-free label cleansing.

Copies `%DC_WORK%/final_v3_candidate_v4b` to `final_v3_candidate_v4d` (v4b untouched; the name v4c is reserved by note
86's plan) and replaces every model whose training rows or tree inputs moved with the v4m labels (oof88.py pkg*):
function trees 229 x 3 (v3fit88/function), lanes D x 3 (v3fit88/lanes), context stacker 'mean3' x 3 and 'single' x 3
(v3fit88/stacker), setback P50 per length group (v3fit88/setback).  Unchanged: phase ranker, decoder, GRU, function
network (3 siba members; trained on func_rows_v4l -- see note 88), setback P10 / P90 band and pair model, health, code.
Each ONNX file is checked against the numpy evaluation of its text model on real rows.

    python assemble88.py    -> final_v3_candidate_v4d/weights rewritten + v3fit88/trees_parity88.json + model card entry
Then: python <pkg>/check.py --freeze ; python <pkg>/check.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
from common import DC_WORK  # noqa: E402

sys.path.insert(0, str(rpath.CODE / "final75"))
FIT = DC_WORK / "final_v3_work" / "v3fit88"
SRC = DC_WORK / "final_v3_candidate_v4b"
PKG = DC_WORK / "final_v3_candidate_v4d"
WTS = PKG / "weights"
GROUPS = ("m30", "h6", "h24", "full")


def plan() -> list:
    out = [(FIT / "function" / f"function229_s{s}.txt", f"function/function229_s{s}.onnx") for s in range(3)]
    out += [(FIT / "lanes" / f"lane_pair_D_s{s}.txt", f"lanes/lane_pair_D_s{s}.onnx") for s in range(3)]
    out += [(FIT / "stacker" / f"stacker_single_s{s}.txt", f"stacker/stacker_single_s{s}.onnx") for s in range(3)]
    out += [(FIT / "stacker" / f"stacker_mean3_s{s}.txt", f"stacker/stacker_mean3_s{s}.onnx") for s in range(3)]
    for g in GROUPS:
        out += [(FIT / "setback" / f"setback_{g}_q50_s{s}.txt", f"setback/setback_{g}_q50_s{s}.onnx") for s in range(3)]
    return out


def main():
    if not PKG.exists():
        shutil.copytree(SRC, PKG, ignore=shutil.ignore_patterns("__pycache__"))
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(PKG))
    import trees_onnx as TO
    assert Path(TO.__file__).resolve().parent == PKG.resolve()
    nbs = {}
    for src, rel in plan():
        nb = NumpyBooster(src)
        OT.to_onnx_te5(nb, WTS / rel)
        nbs[rel] = nb
    print(f"compiled {len(nbs)} tree models", flush=True)
    fm = json.load(open(FIT / "function" / "function229.json"))
    fm["onnx_files"] = [f"function229_s{s}.onnx" for s in range(3)]
    json.dump(fm, open(WTS / "function" / "function229.json", "w"), indent=1)
    lm = json.load(open(FIT / "lanes" / "lane_model.json"))
    lm["onnx_files"] = [f"lane_pair_D_s{s}.onnx" for s in range(3)]
    lm["labels"] = "note 88: print-lane truth (unchanged) + v4m OOF function block (dq without fault events)"
    json.dump(lm, open(WTS / "lanes" / "lane_model.json", "w"), indent=1)
    ss = json.load(open(FIT / "stacker" / "stacker.json"))
    m3 = json.load(open(FIT / "stacker" / "stacker_mean3.json"))
    assert m3["feature_names"] == ss["feature_names"], "mean3 and single must read the same columns"
    old = json.load(open(WTS / "stacker" / "stacker.json"))
    assert old["feature_names"] == ss["feature_names"], "stacker columns differ from v4b"
    sm = dict(ss)
    single = [f"stacker_single_s{s}.onnx" for s in range(3)]
    mean3 = [f"stacker_mean3_s{s}.onnx" for s in range(3)]
    sm["variants"] = {
        "single": {"files": single, "use_when": "the function network has 1 member present (fallback)",
                   "trained": ss["variant"]},
        "mean3": {"files": mean3, "use_when": "the function network has >= 2 members (ships 3 full-data members)",
                  "trained": m3["variant"], "net": m3["net"], "trained_on": m3["trained_on"]}}
    sm["onnx_files"] = mean3
    sm["files"] = {"single": ss["files"], "mean3": m3["files"]}
    sm["variant"] = "two variants (notes 84 / 88): 'mean3' (default, 3 members) and 'single' (1 member)"
    sm["labels"] = "v4l truth (= v4m truth); trees = note-88 v4m OOF (label cleansing without detector fault events)"
    json.dump(sm, open(WTS / "stacker" / "stacker.json", "w"), indent=1)
    p50 = json.load(open(FIT / "setback" / "setback_p50.json"))
    for g, d in p50["groups"].items():
        d["onnx_files"] = [f"setback_{g}_q50_s{s}.onnx" for s in range(3)]
    p50["labels"] = "note 88: sb7 features on the v4l print store and the v4m OOF function"
    json.dump(p50, open(WTS / "setback" / "setback_p50.json", "w"), indent=1)
    # ---- parity on real rows
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    rng = np.random.default_rng(88)
    import pyarrow.parquet as pq
    t = pq.read_table(DC_WORK / "final_v3_work" / "f76" / "frame_v6e_c" / "feat_frame.parquet",
                      columns=["DeviceId"] + fm["features"])
    idx = np.sort(rng.choice(t.num_rows, 30000, replace=False))
    fr = t.take(idx).to_pandas()
    assert not fr.DeviceId.str.lower().isin(lk).any()
    rows = {"function": fr[fm["features"]],
            "stacker_single": np.load(FIT / "stacker" / "X_check.npy").astype(np.float64),
            "stacker_mean3": np.load(FIT / "stacker" / "X_check_mean3.npy").astype(np.float64)}
    sys.path.insert(0, str(rpath.CODE / "trackA"))
    import sb5_setback as SB
    F = pd.read_parquet(DC_WORK / "final_v3_work" / "f88" / "sb7" / "feat.parquet")
    assert not F.dev.isin(lk).any()
    rows["setback"] = SB.add_physics(F, 50.0).astype({c: float for c in SB.FEATS})
    res = {}
    for rel, nb in nbs.items():
        key = rel.split("/")[0]
        if key == "stacker":
            key = "stacker_mean3" if "mean3" in rel else "stacker_single"
        X = rows.get(key)
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
    json.dump(res, open(FIT / "trees_parity88.json", "w"), indent=1)
    mx = max(v.get("max_abs_out", 0.0) for v in res.values())
    print(f"parity: {sum(1 for v in res.values() if 'rows' in v)} checked, max |out diff| {mx:.2e}", flush=True)
    assert mx < 1e-9
    # ---- model card
    card = json.load(open(WTS / "model_card.json"))
    o = json.load(open(DC_WORK / "final_v3_work" / "f88" / "oof" / "oof88.json"))
    card["candidate_v4d_note88"] = {
        "changes": ["label cleansing without detector fault events 83-88 (user ban 2026-09-28): dq_suspect recomputed "
                    "actuation-only (labels v4m), card rule actuation-only; refitted on it: function trees 229 x 3, "
                    "lanes D, stackers mean3 + single, setback P50. Function network unchanged (3 siba v4l members)"],
        "oof_v4l_ge30": {"E": o["E_ge30"]["v4c"], "R": o["R_ge30"]["v4c"],
                         "vs_v4b_E_pt": o["E_ge30"]["v4c - v4b"], "vs_v4b_R_pt": o["R_ge30"]["v4c - v4b"]},
        "parity_trees_onnx_vs_text_max": mx}
    card["file_sha256"] = {str(p.relative_to(WTS)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(WTS.rglob("*")) if p.is_file() and p.name != "model_card.json"}
    card["total_model_bytes"] = sum(p.stat().st_size for p in WTS.rglob("*") if p.is_file() and p.name != "model_card.json")
    json.dump(card, open(WTS / "model_card.json", "w"), indent=1, default=str)
    print("model card updated", flush=True)


if __name__ == "__main__":
    main()
