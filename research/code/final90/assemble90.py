"""Note 90: FINAL package `%DC_WORK%/final_v3_candidate_v4f` = candidate v4e (note 86) with
  phase   network = TCN ad_all full-data refit (`%DC_WORK%/s86/phase/ad_all76_full.onnx` -> weights/phase_tcn.onnx; the
          GRU graph gru.onnx is removed), blend.json names it; decoder re-fitted on the trees + TCN blend
          (p90_phase.py fitdec -> v3fit90/decoder)
  function  trees 229 x 3, lanes D, setback P50, stackers mean3 + single re-fitted on the final labels v4o (oof90.py pkg*
          -> v3fit90)
  siba    members = the three full-data v4o refits x86_sibafull4o{,_s1,_s2} once they exist (`siba` stage); until then
          the v4l members of v4e
Package code (gru_onnx / gru_blend / funcnet) was edited in place in v4f (15-channel raster = funcnet.render, cached
per piece and shared with the function network).  Each ONNX tree file is checked against the numpy evaluation of its
text model on real rows.

    python assemble90.py phase   -> phase_tcn.onnx + blend.json + decode_v3.{onnx,json}; v3fit90/parity_phase90.json
    python assemble90.py func    -> function / lanes / stacker / setback ONNX + metadata; v3fit90/trees_parity90.json
    python assemble90.py siba    -> funcnet members = x86_sibafull4o (export83 graphs) + manifest
    python assemble90.py card    -> model card entry + sha256
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
FIT = DC_WORK / "final_v3_work" / "v3fit90"
PKG = DC_WORK / "final_v3_candidate_v4f"
WTS = PKG / "weights"
TCN_ONNX = DC_WORK / "s86" / "phase" / "ad_all76_full.onnx"
GROUPS = ("m30", "h6", "h24", "full")


def _onnx_tools():
    import onnx_trees75 as OT
    from lgbm_numpy import NumpyBooster
    sys.path.insert(0, str(PKG))
    import trees_onnx as TO
    assert Path(TO.__file__).resolve().parent == PKG.resolve()
    return OT, NumpyBooster, TO


def _check(OT, TO, nb, dst, X):
    m = TO.OnnxTrees(dst, 4)
    return {"rows": int(len(X)), "max_abs_raw": float(np.abs(OT.raw_numpy(nb, X) - m.raw(X)).max()),
            "max_abs_out": float(np.abs(np.asarray(nb.predict(X)) - np.asarray(m.predict(X))).max())}


def cmd_phase():
    OT, NumpyBooster, TO = _onnx_tools()
    par = json.load(open(TCN_ONNX.parent / "parity86_phase.json"))
    assert par["pass"] and par["cfg"]["chans"] == ["occ", "onrate", "g", "y", "rc", "call", "og", "oc", "coord", "onE",
                                                    "offE", "cOn", "cOff", "pg", "pc"], par
    shutil.copy(TCN_ONNX, WTS / "phase_tcn.onnx")
    (WTS / "gru.onnx").unlink(missing_ok=True)
    b = json.load(open(WTS / "blend.json"))
    b["weights_file"] = "phase_tcn.onnx"
    b["network"] = {"arch": "TCN ad_all (note 53 recipe: 7 dilated blocks, 96 channels, attention pooling)",
                    "bin_ms": 1000, "channels": par["cfg"]["chans"],
                    "trained": "full-data refit on the 780-signal pool (note 86, 31 epochs, seed 0), locked never used",
                    "onnx_parity_vs_torch": {"random": par["random"], "real_max_abs_logp": par["real"]["max_abs_logp"]},
                    "why": "note 87b: ties the GRU p3 (>= 30 min E .9816 vs .9818, CI spans 0), ~2.8x faster; user rule "
                           "tie -> faster (2026-10-04)"}
    b["note"] = b["note"].replace("the GRU", "the phase network").replace("(gru_onnx.py)", "(gru_onnx.py; the TCN reads "
                                                                                          "the 15-channel funcnet raster)")
    json.dump(b, open(WTS / "blend.json", "w"), indent=1)
    nb = NumpyBooster(FIT / "decoder" / "decode_v3.txt")
    OT.to_onnx_te5(nb, WTS / "decode_v3.onnx")
    dm = json.load(open(FIT / "decoder" / "decode_v3.json"))
    json.dump(dm, open(WTS / "decode_v3.json", "w"), indent=1)
    X = np.load(FIT / "decoder" / "X_check.npy").astype(np.float64)
    res = {"decode_v3.onnx": _check(OT, TO, nb, WTS / "decode_v3.onnx", X), "decoder_arm": dm["arm"],
           "n_estimators": dm["n_estimators"]}
    json.dump(res, open(FIT / "parity_phase90.json", "w"), indent=1)
    print(f"phase: TCN copied, blend.json updated, decoder ({dm['arm']}, {dm['n_estimators']} trees) parity {res}",
          flush=True)
    assert res["decode_v3.onnx"]["max_abs_out"] < 1e-9


def plan_func() -> list:
    out = [(FIT / "function" / f"function229_s{s}.txt", f"function/function229_s{s}.onnx") for s in range(3)]
    out += [(FIT / "lanes" / f"lane_pair_D_s{s}.txt", f"lanes/lane_pair_D_s{s}.onnx") for s in range(3)]
    out += [(FIT / "stacker" / f"stacker_single_s{s}.txt", f"stacker/stacker_single_s{s}.onnx") for s in range(3)]
    out += [(FIT / "stacker" / f"stacker_mean3_s{s}.txt", f"stacker/stacker_mean3_s{s}.onnx") for s in range(3)]
    for g in GROUPS:
        out += [(FIT / "setback" / f"setback_{g}_q50_s{s}.txt", f"setback/setback_{g}_q50_s{s}.onnx") for s in range(3)]
    return out


def cmd_func():
    OT, NumpyBooster, TO = _onnx_tools()
    nbs = {}
    for src, rel in plan_func():
        nb = NumpyBooster(src)
        OT.to_onnx_te5(nb, WTS / rel)
        nbs[rel] = nb
    fm = json.load(open(FIT / "function" / "function229.json"))
    fm["onnx_files"] = [f"function229_s{s}.onnx" for s in range(3)]
    json.dump(fm, open(WTS / "function" / "function229.json", "w"), indent=1)
    lm = json.load(open(FIT / "lanes" / "lane_model.json"))
    lm["onnx_files"] = [f"lane_pair_D_s{s}.onnx" for s in range(3)]
    lm["labels"] = "note 90: print-lane truth (unchanged) + v4o OOF function block (final training labels)"
    json.dump(lm, open(WTS / "lanes" / "lane_model.json", "w"), indent=1)
    ss = json.load(open(FIT / "stacker" / "stacker.json"))
    m3 = json.load(open(FIT / "stacker" / "stacker_mean3.json"))
    old = json.load(open(WTS / "stacker" / "stacker.json"))
    assert m3["feature_names"] == ss["feature_names"] == old["feature_names"], "stacker columns changed"
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
    sm["variant"] = ("two variants: 'mean3' (default, 3 members; note 90 = note-86 recipe on the v4o trees OOF) and "
                     "'single' (1 member, fallback)")
    sm["labels"] = "v4l truth (= v4o plain columns); trees = v4o OOF (final training labels, note 89b)"
    json.dump(sm, open(WTS / "stacker" / "stacker.json", "w"), indent=1)
    p50 = json.load(open(FIT / "setback" / "setback_p50.json"))
    for g, d in p50["groups"].items():
        d["onnx_files"] = [f"setback_{g}_q50_s{s}.onnx" for s in range(3)]
    p50["labels"] = "note 90: sb7 features on the v4l print store and the v4o OOF function"
    json.dump(p50, open(WTS / "setback" / "setback_p50.json", "w"), indent=1)
    # parity on real rows
    lk = set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())
    rng = np.random.default_rng(90)
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
    F = pd.read_parquet(DC_WORK / "final_v3_work" / "f90" / "sb7" / "feat.parquet")
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
        res[rel] = _check(OT, TO, nb, WTS / rel, X)
    json.dump(res, open(FIT / "trees_parity90.json", "w"), indent=1)
    mx = max(v.get("max_abs_out", 0.0) for v in res.values())
    print(f"func: {len(nbs)} compiled, {sum(1 for v in res.values() if 'rows' in v)} checked, max |out diff| {mx:.2e}",
          flush=True)
    assert mx < 1e-9


def cmd_siba():
    """swap the function-network members to the final v4o refits (export83 graphs: pair + head, head masks filtered
    candidates), then re-check parity vs torch with export84's harness (run separately: parity90 siba)."""
    sys.path.insert(0, str(rpath.CODE / "final83"))
    import export83 as E83
    tags = ["x86_sibafull4o", "x86_sibafull4o_s1", "x86_sibafull4o_s2"]
    md = WTS / "funcnet"
    man = json.load(open(md / "manifest.json"))
    E83.PKG = PKG
    new = []
    for t in tags:
        pair, head = E83.export_member(t, md) if hasattr(E83, "export_member") else (None, None)
        assert pair is not None, "export83.export_member missing -- see parity90.py"
        new.append({"tag": t, "pair": pair, "head": head})
    for m in man["members"]:
        for k in ("pair", "head"):
            (md / m[k]).unlink(missing_ok=True)
    man["members"] = new
    json.dump(man, open(md / "manifest.json", "w"), indent=1)


def cmd_card():
    card = json.load(open(WTS / "model_card.json"))
    ent = json.load(open(FIT / "card90.json"))
    card["final_v4f_note90"] = ent
    card["file_sha256"] = {str(p.relative_to(WTS)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(WTS.rglob("*")) if p.is_file() and p.name != "model_card.json"}
    card["total_model_bytes"] = sum(p.stat().st_size for p in WTS.rglob("*") if p.is_file() and p.name != "model_card.json")
    json.dump(card, open(WTS / "model_card.json", "w"), indent=1, default=str)
    print("model card updated", flush=True)


if __name__ == "__main__":
    assert PKG.exists(), PKG
    {"phase": cmd_phase, "func": cmd_func, "siba": cmd_siba, "card": cmd_card}[sys.argv[1]]()
