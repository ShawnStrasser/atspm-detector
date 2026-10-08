"""Note 75: model card of `%DC_WORK%/final_v3_candidate_v3` (weights/model_card.json): what each model is, what it was
trained on, how it was verified, file hashes.  Reads the fit metadata, the parity / bench outputs of note 75.

    python card75.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PKG = W / "final_v3_candidate_v3"
WTS = PKG / "weights"
FIT = W / "final_v3_work" / "v3fit"


def jl(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


def main():
    old = json.load(open(W / "final_v3_candidate_v2" / "weights" / "model_card.json"))
    fm, dm, lm = jl(WTS / "function" / "function229.json"), jl(WTS / "decode_v3.json"), jl(WTS / "lanes" / "lane_model.json")
    sm, sb, p50 = jl(WTS / "stacker" / "stacker.json"), jl(WTS / "setback" / "setback_model.json"), \
        jl(WTS / "setback" / "setback_p50.json")
    net = jl(WTS / "funcnet" / "manifest.json")
    card = {
        "name": "final_v3_candidate_v3",
        "date": time.strftime("%Y-%m-%d"),
        "status": "CANDIDATE, not shipped (model/ stays final_v2 until the user says ship); decided on six-fold OOF evidence; "
                  "locked_v2 signals never used",
        "what_it_does": old["what_it_does"],
        "phase_anonymous": old["phase_anonymous"],
        "pipeline": [
            "phase: LightGBM pair ranker, 261 features, 3 seeds (note 37 full fit = note-57 recipe, 772 signals)",
            "phase: GRU pair network p3 (all 780 signals) on candidates with ranker probability >= .01 only (note 73b), "
            "pair batch 64, 4 pieces of a 32 grid above 120 min; 0.5 / 0.5 before the decoder",
            "phase: joint decoder refitted on that filtered blend (six-fold OOF inputs, all labelled rows; note 75)",
            "function: 229-feature trees, 3 seeds (note 57 arm, refitted on all rows; note 75)",
            "function: siba TCN with sibling attention (note 69) -- ONNX; members in funcnet/manifest.json",
            "function: context stacker, 3 seeds (notes 67 / 69; refitted on all OOF rows; note 75)",
            "function: per-lane decode, lane-confidence gate .9, stack pick (loser -> best non-ATSPM for Advance / "
            "Presence), samples < 30 min: Count / Yellow_Red twin decode (thr .4)",
            "lanes: model D (note 58; refitted), from 30 min; setback: sb7 P50 by sample-length group (note 58; refitted) + "
            "note-41 P10 / P90 band for the confidence; health v5 (note 47); night speed (note 60)",
            "every LightGBM model compiled to an ONNX TreeEnsemble (ai.onnx.ml opset 5, double precision)"],
        "runtime": "CPU, onnxruntime only (+ numpy / pandas / duckdb / pyarrow); no torch / lightgbm / scipy / sklearn",
        "models": {
            "ranker": {k: old["models"]["ranker"][k] for k in ("n_models", "n_features", "best_iterations", "params")} |
                      {"files": [f"phase_lgbm_v5_s{s}.onnx" for s in range(3)], "refit": "no (already the full fit)"},
            "gru": {"file": "gru.onnx", "source": "p3_final (note 37), unchanged", "keep_min_tree_p": 0.01,
                    "pair_batch": 64},
            "decoder": dm,
            "function_trees": {k: fm[k] for k in fm if k != "features"} | {"n_features": len(fm["features"])},
            "function_net": net | {"refit_status": "fold-0 seed-0 research model (trained on folds 1-5, ~5/6 of the "
                                                   "signals) until the GPU full-data refit; one member keeps 3 h warm "
                                                   "time in budget; OOF single seed vs 3 seeds <= ~0.1 pt (note 69)"},
            "stacker": {k: sm[k] for k in sm if k != "feature_names"} | {"feature_names": sm["feature_names"]},
            "lanes": {k: lm[k] for k in lm if k not in ("features", "c2_columns")} |
                     {"n_features": len(lm["features"])},
            "setback": {"band": sb, "p50": p50},
            "health": {"file": "health_core.py", "version": "v5 (note 47) + note-56 short-window fix"},
            "night_speed": {"file": "night_speed.py", "version": "note 60 sp1 v2, deterministic"},
        },
        "parity": {"trees_onnx_vs_text_frames": jl(FIT / "trees_parity_frames.json"),
                   "siba_onnx_vs_torch": {k: v for k, v in (jl(FIT / "siba_parity.json") or {}).items() if k != "pieces"},
                   "note75_vs_oof": jl(W / "final_v3_work" / "parity75" / "parity75.json")},
        "bench_note75": jl(W / "final_v3_work" / "bench75" / "bench75.json"),
        "held_out_never_used": old.get("held_out_never_used_in_training"),
        "label_sources": old["label_sources"],
        "thresholds": {k: old["thresholds"][k] for k in ("min_actuations", "min_prob", "low_evidence_flag_below_actuations")},
        "file_sha256": {str(p.relative_to(WTS)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(WTS.rglob("*")) if p.is_file() and p.name != "model_card.json"},
        "total_model_bytes": sum(p.stat().st_size for p in WTS.rglob("*") if p.is_file() and p.name != "model_card.json"),
    }
    json.dump(card, open(WTS / "model_card.json", "w"), indent=1, default=str)
    print("card written", card["total_model_bytes"])


if __name__ == "__main__":
    main()
