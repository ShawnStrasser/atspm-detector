"""Note 113: package `%DC_WORK%/final_v3_candidate_v6b` = v6 (copied by hand 2026-10-07, v6 untouched) with
  decoders   decode_v3 and decode_trees refitted on full data with LEAD neighbours (lag_*) in place of the channel-
             adjacency block (f113.py fit; ONNX via onnx_trees75 + parity, as assemble111)
  code       decode.py (no channel arithmetic), similarity.build_lead_window, predict.py (lead graph wired in, dead yr_*
             block removed, docstrings / CLI help: default profile full), check.py (joint channel-bijection + phase
             renumbering test), gru_blend / function_stage / stacker docstrings, blend.json note
  card       model_card.json: a fresh top block (v6b) + entry final_v6b_note113; the old v4b top block kept under
             'history_top_block_v4b'

    python assemble113.py weights     ONNX + parity for both decoders, json metas
    python assemble113.py card        model card (reads s113/card113.json) + sha256
Then: python <v6b>/check.py --freeze ; python <v6b>/check.py
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
PKG = DC_WORK / "final_v3_candidate_v6b"
WTS = PKG / "weights"
FIT = DC_WORK / "final_v3_work" / "v3fit113"
OUT = DC_WORK / "s113"
LEAD_COLS = ["lag_mean", "lag_max", "lag_top1", "lag_wsum", "lag_n"]


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


def cmd_weights():
    OT, NumpyBooster, TO = _tools()
    par = {}
    for sub, stem in (("decoder", "decode_v3"), ("decode_trees", "decode_trees")):
        d = FIT / sub
        X = np.load(d / "X_check.npy").astype(np.float64)
        old = json.load(open(WTS / f"{stem}.json"))
        dm = json.load(open(d / "decode_v3.json"))
        assert [c for c in old["features"] if not c.startswith("adj_")] + LEAD_COLS == dm["features"], stem
        assert not any(c.startswith("adj_") for c in dm["features"])
        par[f"{stem}.onnx"] = _conv(OT, NumpyBooster, TO, d / "decode_v3.txt", WTS / f"{stem}.onnx", X)
        json.dump(dm, open(WTS / f"{stem}.json", "w"), indent=1)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(par, open(OUT / "trees_parity113.json", "w"), indent=1)
    print(json.dumps(par, indent=1))


def cmd_card():
    card = json.load(open(WTS / "model_card.json"))
    top = ["name", "date", "status", "what_it_does", "phase_anonymous", "pipeline", "runtime", "models", "parity",
           "bench_note75", "held_out_never_used", "label_sources", "thresholds"]
    if "history_top_block_v4b" not in card:
        card["history_top_block_v4b"] = {k: card[k] for k in top if k in card}
    c113 = json.load(open(OUT / "card113.json"))
    new = {"name": "final_v3 candidate v6b",
           "date": "2026-10-07",
           "status": "candidate (note 113), not shipped; model/ = final_v2",
           "what_it_does": card["history_top_block_v4b"].get("what_it_does"),
           "phase_anonymous": ("no phase number and no detector channel number -- nor channel order or channel "
                               "difference -- is ever a model input (v6b: the decoder's channel-adjacency neighbours "
                               "are replaced by behaviour-only lead neighbours); candidates are every phase with a Begin "
                               "Green in the sample; check.py proves it with random channel bijections + phase "
                               "renumberings. No channel-to-phase wiring table is consulted anywhere."),
           "pipeline": [
               "phase: LightGBM pair ranker, 261 features, ONE seed (note 95 full fit, seed 0)",
               "phase: the function network's pair phase head (siba w32, 3 members) on candidates with ranker p >= .01, "
               "0.5 / 0.5 before the decoder (note 111)",
               "phase: joint decoder decode_v3 (20 features + 5 lead-neighbour aggregates, note 113) refitted on full "
               "data on that blend; fast profiles above their length limit: decode_trees (same features) on the ranker",
               "function: 229-feature trees ONE seed; siba members x3; context stacker mean3 / single / nonet ONE seed",
               "lanes model D (v5c, note 105); setback P50 one seed per length group + P10 / P90 band; health; night "
               "speed",
               "default profile 'full' (network always); 'le2h' = network only up to 2 h (edge / fast switch)"],
           "runtime": card["history_top_block_v4b"].get("runtime"),
           "held_out_never_used": card["history_top_block_v4b"].get("held_out_never_used"),
           "label_sources": card["history_top_block_v4b"].get("label_sources"),
           "history_note": "keys final_v6_note111, final_v5b_note100, ... below are the earlier candidates' entries, "
                           "kept as history; history_top_block_v4b is the old (v4b) top block"}
    rest = {k: v for k, v in card.items() if k not in top}
    card = {**new, **rest}
    card["final_v6b_note113"] = c113
    card["file_sha256"] = {str(q.relative_to(WTS)).replace("\\", "/"): hashlib.sha256(q.read_bytes()).hexdigest()
                           for q in sorted(WTS.rglob("*")) if q.is_file() and q.name != "model_card.json"}
    card["total_model_bytes"] = sum(q.stat().st_size for q in WTS.rglob("*") if q.is_file() and q.name != "model_card.json")
    json.dump(card, open(WTS / "model_card.json", "w"), indent=1, default=str)
    print("model card updated")


if __name__ == "__main__":
    {"weights": cmd_weights, "card": cmd_card}[sys.argv[1]]()
