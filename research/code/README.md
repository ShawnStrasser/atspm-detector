# Research code — index

Kept for posterity. None of it is needed to run the model: that is the pip package in
`../../src/atspm_detector/`.
It reads and writes a local work directory (`%DC_WORK%`, default `~/dc_work`) that is
not in this repository, so most of it cannot be re-run as is.

Every script starts by importing `rpath`, which puts every folder here and `../../src/` on
`sys.path`. For scripts written against the earlier model package, see the last section.

The stage notes in `../notes/` refer to the old `src/…` paths. `src/predict.py` and the
modules it imported became the earlier model package (git tag `beta-final_v2`); `src/official/*`, `src/tune/*`, `src/statewide/*`,
`src/contrast/*`, `src/neural/*`, `src/old_baseline/*` are the folders below.

| file | what it did |
|---|---|
| **`common.py`** | work-dir paths, event codes, fold count, the ODOT wiring table (evaluation only), capped DuckDB connection |
| **`rpath.py`** | the import-path helper every script uses |
| **`features/`** | |
| `build_cache.py` | raw events → folds, label tables, event cache, derived tables (`--step labels\|events\|derived\|check`) |
| `build_features.py` | the training feature tables: window sets, cache loader, CLI over the shipped builders (`--what base\|extra\|sim\|yrlag`) |
| `build_detector_health.py` | the per-detector health table from raw events |
| `health_sql.py` | the DuckDB form of the health rules, bad-period masking, the cached-table reader |
| `build_stg_cache.py` / `build_stg_features.py` / `build_stg_flat.py` | the same, for the Sept-2026 pull |
| `windows_stg.py` | the Sept-2026 window anchors (registers them with `build_features`) |
| **`labels/`** | |
| `labels_official.py` | official controller timing → the phase truth table |
| `make_label_map.py` | free-text function strings → the five classes |
| `make_split.py` / `split_counts.py` | the locked TEST / NEWTEST splits and their counts |
| `analyse_labels.py` | hand config vs controller timing: how far they agree, and how stably |
| `disagreements.py` / `make_disagreements.py` | confident model-vs-label disagreements, with an evidence sentence each |
| `resolve_review.py` | folds returned review files back into the label tables |
| `trust.py` | per-signal label-trust summary |
| **`lightgbm/`** | |
| `train_lgbm.py` | stage 01: the first pair ranker, ablations, the scoring API |
| `train_lgbm_v2.py` | stage 04: ranker with the partner features, ablations, decoder, function head |
| `decode_train.py` | training and calibration of the joint per-signal decoder |
| `function_v2.py` / `function_v3.py` / `function_v4.py` | the three generations of the function head; v4 is what ships |
| `train_official.py` / `run_train.py` / `score_variants.py` | stage 10: refitting on the official timing labels, variant comparison |
| `fit_final.py` / `fit_final_v1.py` | the candidate and the final LightGBM fit (`--stage oof\|models\|card`) |
| `ship_final_v2.py` | assembled the earlier model's weights folder and model card |
| **`neural/`** | |
| `raster.py` / `ncache.py` / `ncache2.py` | the 1 s, 9-channel raster and its cache (two generations) |
| `data.py` / `data2.py` | datasets and samplers |
| `models.py` | GRU, TCN, conv-GRU, transformer, 2-D cycle CNN |
| `train.py` / `train2.py` | training (stage 03, then the stage-13 refit) |
| `infer.py` / `infer2.py` / `score_neural.py` / `score_minact.py` | out-of-fold and held-out scoring |
| `arch_select.py` / `pick_best.py` / `summarize.py` / `concat_oof.py` | the architecture comparison and its bookkeeping |
| `export_gru.py` / `bench_gru.py` / `gru_speedup.py` | ONNX export, the runtime benchmark, the research-only GPU forward pass |
| `blend_v2.py` / `blend_v2_predecode.py` / `blend_check.py` / `gru_old_vs_new.py` | the blend: weight, where to apply it, cut-off, and the refit comparison |
| `trackb_*.py` / `b5_*.py` | Track B (notes 15, 16, 19): TCN backbone, seeds, B3/B4 chain; B5 sibling-context net (`b5_net`, `b5_train`, `b5_infer`, `b5_chain`, `b5_headroom`) |
| **`evaluation/`** | |
| `evaluate.py` | the scoring harness: top-1, non-standard wiring, concurrent pairs, coverage curves, function metrics |
| `error_forensics.py` | what the remaining errors actually are |
| `score_final.py` / `score_final_v2.py` / `score_official.py` | the scorings of the locked signals |
| `curve_final.py` / `plot_final_accuracy.py` | the accuracy-vs-sample-length curve and its chart |
| `analyze_detector_health.py` | how much of the population is suspect or failed, and how much it costs |
| `eval48_phase.py` / `eval48_function.py` | note 48: final_v3 candidate v2 vs the earlier model, six folds OOF, everything and realistic sets, by health status |
| **`experiments/`** (things that did not ship) | |
| `old_baseline/` | re-running and scoring the 2025 BiLSTM on the same hold-out |
| `tune/` | Optuna, seed bagging, XGBoost/CatBoost/forests, feature pruning |
| `statewide/` | the six-event-code statewide pull and the Other / Yellow_Red classes |
| `contrast/` | peak vs off-peak demand-contrast features |
| `health2/` | detector-health masking and a learned trust score |
| `overlaps/` | overlaps as a predictable output class |
| `delay_extend.py` | whether programmed delay / extend settings help |
| **`lanes/`** (note 42: lane output, research only) | |
| `lane_output.py` | `lanes(events, predictions) -> (phase table, detector table)`: pair cues (numpy), pair model (numpy booster), constrained lane decode |
| `ln1_cues.py` / `ln2_pairmodel.py` / `ln3_decode.py` / `ln4_final.py` | print lane truth + cues per window; OOF pair model + controls; decode tuning + OOF scoring; final fit + smoke |
| `ln5_extend.py` / `atspm_decode.py` | note 54: six fold pair models + OOF lanes for every signal / period / window; per-lane ATSPM decode (one A/P/C/YR per lane) |
| `../evaluation/atspm_score.py`, `../cabinet/stack_labels_v3s.py` | note 54: ATSPM-only stack-aware scoring (+ old acc7); label variant v3s (stack relabel, YR==Count out, radar-over-loops re-admitted, Dec-role function/phase rule) |
| **`trackA/`** (function search; paths from `DC_WORK` / `DC_REPO`, run from this folder) | |
| `a1_labels.py` / `a1_rescore.py` / `a1_disagree.py` | note 14, A1: corrected function labels, phase-scorer fix and re-score, round-2 disagreement list |
| `a2_features.py` / `a2_filter.py` / `a2_model.py` / `a2_noise.py` / `a2_candidate.py` | note 14, A2: expert-shaped features, locked-signal filter, function retrain + ablation, noise-column control, freezing the candidate |
| `a3_lanes.py` / `a3_pairs.py` / `a3_lanelabels.py` | note 14, A3: lane grouping + per-lane role decoding, raw pair table, RL/CL/LL validation |
| `a4_reject.py` | note 14, A4: Other as rejection, leave-one-Other-subtype-out |
| `v5_frame.py` / `v3_retrain.py` | note 28: funcframe_v5 + folds_v3 (every v3 signal with data), the v3-label retrain harness |
| `v6_frame.py` | note 36: funcframe_v6 + folds_v4 (v5 + the released NEWTEST half, final_v1-tree phase inputs) |
| `sb6_export.py` | note 48: exports the note-41 setback model to numpy-runnable files for the candidate package, with parity |
| `v3_final_fit.py` / `v3_candidate_verify.py` | note 32: the final_v3 candidate's 3-seed function head (all training rows) and the package-vs-training-frame feature parity check |
| `w1_whole_intersection.py` | note 25: unlabelled active channels (audit), Other subtypes / Mid+Bike 7-class head / unlabelled-as-Other sensitivity |
| `lr_common.py` / `lr1_labels.py` … `lr5_oracle.py` | note 17: the lanes redo (imports `a3_lanes`) |
| **note 57** (trees alone, validation step 2) | |
| `lightgbm/t57_phase.py` | phase pool on folds_v4, ranker per feature-group config + decoder OOF, noise-column control, two-number scoring |
| `trackA/t57_function.py` | function head per feature-group / weighting / phase-input config; ATSPM pick-decode scoring vs the v3s baseline |
| `trackA/s59_step6.py` | note 59: step-6 additions (nested lanes as inputs, ETA class, function-free health, off-peak features / weighting) vs the current set-up; rebuilds pick inputs on D lanes |
| `trackA/sp1_night_speed.py` | note 60: deterministic night-time free-flow speed per detector / phase (numpy only; ported to the package as `night_speed.py`) |
| `trackA/s62_short.py` | note 62: short-window (5/10 min) function: twin decode on co-actuating stop-bar zones, lane-gate check, short-weighted head routed by sample length, non-ATSPM prior |
| `trackA/sp1_eval.py` | note 60: runs it on the Sept windows with OOF inputs; coverage, distribution, setback sensitivity, shifted control |
| **`download/`** | agency-specific pull scripts — git-ignored, never committed |
| **`final75/`** (note 75: the integrated package `final_v3_candidate_v3`) | |
| `fit75.py` | CPU full-data refits: function trees (229), decoder on the filtered blend, lanes D, context stacker (3-seed and single-member variants), setback sb7 P50 |
| `export75.py` / `onnx_trees75.py` / `assemble75.py` | siba -> two ONNX graphs + parity vs torch; LightGBM -> ONNX TreeEnsemble (opset ml 5); weights assembly + parity on real frame rows |
| `parity75.py` / `member75.py` / `bench75.py` / `card75.py` | parity vs OOF (injected + production), one-vs-three siba members into the stacker, speed / RAM / invariance per signal, model card |
| **`phasefree/`** (note 76) | phase- and channel-order-free versions of the earlier model's `features.py`, `features_partner.py`, `features_yellowred.py`, `similarity.py`, `function.py` (ties averaged / kept, never broken by phase or channel number; ranks on float32-rounded values). `rpath.py` puts this folder ahead of the others, so research builds use them |
| **`final76/`** (note 76) | |
| `diag76.py` / `run_diag76.py` / `sum_diag76.py` | stage-by-stage renumbering diff of a package (phases permuted in the raw log; `--chanrev` = channel order reversed, adjacency kept) |
| `excl76.py` / `f76_pool.py` | new partner features on the phase pool and the function frame (mask stage only), patched pool / frame + change counts |
| `lag76.py` / `lagtie76.py` / `topk76.py` | channel-order part: lag tables with tie-keeping top-K, tie-averaged best / twin lag features; tie frequencies |
| `f76_phase.py` / `f76_function.py` | refits on the patched pool / frame (note-57 recipes) and scoring inside the champion vs the current one |
| **`final77/`** (note 77) | channel-order-free lanes D / pick inputs / decodes / setback pair model (fixes live in `lanes/lane_output.py`, `ln6_pick.py`, `ln7_stackhealth.py`, `atspm_decode.py`, `trackA/s62_short.py`, `trackA/sb5_setback.py`, `health/health_core.py`) |
| `of77.py` | OOF rebuild: reversed-orientation cues + content keys, lanes D on both orientations, canonical decode, pick / stack health / twins, context stacker, compare vs the note-76 champion |
| `sb77.py` / `fit77.py` / `assemble77.py` | setback features + eval; full-data refits (lanes, setback P50, stackers) and package assembly |
| `diag77.py` / `run_diag77.py` / `sum_diag77.py` / `bench77.py` | invariance battery (channel reversal / shift / both + phase renumbering) and 3-h speed bench |

## Scripts that need the earlier model package

Until the pip package shipped, `rpath.py` also put the earlier model package `model/` on `sys.path`, and many
scripts here import its modules. That folder was removed from the tree at 1.0.0 and is kept in git tag
`beta-final_v2`. Without it:

* `decode` and `health` resolve to `atspm_detector.decode` / `atspm_detector.health` (same names, the final
  model's definitions; a warning says so), so scripts that need only those still import;
* `predict` (46 scripts), `lgbm_numpy` (23), `gru_blend` (17), `gru_input` (9) and `gru_onnx` (4) have no
  counterpart: 68 scripts import one of them directly and about 180 of the ~620 fail to import through them;
* a few scripts read `REPO/model/...` paths directly: `evaluation/bench71.py`, `evaluation/curve_final.py`,
  `evaluation/score_final_v2.py`, `final119/run119.py`, `final75/bench75.py`, `lanes/lane_output.py` (fallback),
  `lightgbm/ship_final_v2.py`, `neural/trackb_export.py`, `trackA/v6_frame.py`.

To run one as its note describes, check the tag out next to the repository and set `DC_BETA_MODEL`
(the aliases above are then off):

    git worktree add ../dc-beta beta-final_v2
    set DC_BETA_MODEL=..\dc-beta\model          (Windows; elsewhere: export DC_BETA_MODEL=../dc-beta/model)

Scripts with hard-coded `REPO/model` paths are simplest to run from inside that worktree.
