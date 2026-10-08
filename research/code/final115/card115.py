"""Note 115: write the v7 model card (weights/model_card.json): what it is, models with sha256, accuracy with sources."""
import hashlib, json
import os
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
D = Path(str(W / r"final_v7_prod\src\detector_classifier\weights"))
files = sorted(p for p in D.rglob("*") if p.is_file() and p.name != "model_card.json")
sha = {str(p.relative_to(D)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
size = sum(p.stat().st_size for p in files)
card = {
 "name": "detector_classifier", "version": "7.0.0", "date": "2026-10-07",
 "status": "production build (note 115) of candidate v6b (note 113) + note 114 (health out of classification); "
           "the repo's model/ is still final_v2 until the user decides",
 "what_it_does": "per detector from the hi-res controller log alone: phase, function (Advance / Presence / Count / "
                 "Yellow_Red / Mid / Bike / Other), lanes, setback distance, night speed; detector health is a "
                 "separate output computed after classification and never read by it",
 "phase_anonymous": "no phase number, channel number, channel order or channel difference is a model input; checked by "
                    "check.py (phase renumbering, channel reversal / shift, random channel bijection + phase renumbering)",
 "pipeline": {
  "phase": "pair ranker (phase_lgbm_v5_s0, 261 features) -> network pair phase head (funcnet x100_w32full{,_s1,_s2}, "
           "kept pairs = ranker p >= .01) blended 0.5 / 0.5 -> joint decoder decode_v3 (25 features: first-stage "
           "probabilities, candidate claims, similarity + lead neighbours); no network -> decode_trees",
  "function": "function229_s0 trees (229 features) + network function head (same pass) -> stacker (46 columns; "
              "mean3 / single / nonet variants) -> per-lane ATSPM decode (lane confidence >= .9); < 30 min: twin decode",
  "lanes": "lane_pair_D_s0..s2 pair model + constrained decode (>= 30 min)",
  "setback": "P50 per length group (m30 / h6 / h24 / full) + P10 / P90 band (3 + 3 seeds) + same-lane pair HGB",
  "health": "rules (health_core.py), after classification; not a model input",
  "night_speed": "deterministic, from the answers above"},
 "profiles": {"full": "default; network at any length",
              "le2h": "fast / edge: network only up to 120 min; above it decode_trees + stacker nonet, no network loaded"},
 "trained_models_default_path": 21,
 "accuracy_oof_six_fold": {
  "phase_E_ge30min": 0.9846, "phase_E_5min": 0.9634,
  "phase_source": "note 113 (s113/score113.json; v6b decoder, 3 seeds, 761 signals)",
  "function_ATSPM_E_ge30min": 0.9304, "function_ATSPM_E_5min": 0.9022,
  "function_le2h_E_ge30min": 0.9268,
  "function_source": "note 114 (s114/score114.json, mean3 'nou' = stacker without pk_unhealthy, 3 seeds, 642 signals)"},
 "runtime": "CPU, onnxruntime (ORT_ENABLE_ALL, sequential, 4 intra-op threads, no spinning; network sessions without "
            "memory arena); DuckDB in-memory, 4 threads; numpy, pandas, duckdb, onnxruntime only",
 "held_out_never_used": "dc_work/official/locked_v2.csv (115 signals): no training, tuning or scoring",
 "labels": "phase = official controller timing; function = research/labels/function_labels_v4q.parquet",
 "total_model_bytes": size,
 "file_sha256": sha,
}
(D / "model_card.json").write_text(json.dumps(card, indent=1))
print(len(sha), size)
