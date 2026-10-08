# 49 — Function head on the phase_v3 phase input (closes note 48's open item) (2026-09-29)
Question: function_v3d (note 45) was trained and scored with frame v6's phase columns (older tree-only phase OOF:
stage-11 Dec-only ranker for the v4 rows, stage-12 OOF / final_v1 for v5 / v6 rows), but candidate v2 runs it on
phase_v3. Code: `trackA/fv3e_phase.py` (phase inputs), `trackA/v6e_frame.py` (frame), `v3_retrain.py --frame
v6e|v6t`, `evaluation/eval49_function.py` -> `final_v3_work/function_v3e/eval49.json`. locked_v2 asserted absent; nothing run on it.
**Finding while wiring it up:** candidate v2's `predict.py` hands the function model the TREES-ONLY phase
(`prob_lgbm`: phase_v3 ranker bag + decoder, no GRU), not the blend. So two frames were built:
* **v6t** = phase_v3 trees-only OOF (what the package feeds today); **v6e** = phase_v3 blend OOF (note 37, K = 4).
## Phase inputs (six-fold OOF for every frame row)
* The note-37 OOF covers 62 % of frame detector-windows (Dec DEV, Sept NEWTRAIN + released, timing-labelled channels).
  The rest (171,354: Sept-2026 rows of 354 DEV signals, Dec rows of 6 NEWTRAIN signals, 530 unlabelled channels) were
  scored by phase_v3 FOLD models of their own phase fold: trees re-fitted per fold with the note-37 recipe (6 threads;
  pool predictions = saved OOF to 1e-10 on 8 of 18 fits, corr >= .998 on the rest: thread-count float drift);
  GRU = saved `p3_f{k}.pt`, K = 4 (Sept DEV rasters built in a private folder); decoder = note 37's `run_arm`.
* Pool rows keep note 37's values exactly (top-1 = note-48 candidate blend / trees on 100 % of note-48 rows).
  Extras top-1 vs timing: blend .965, trees-only .953 (168,350 det-windows). Frame rows with timing label, top-1:
  v6 .9610, v6t .9626, v6e .9714 (5 min .938 / .959 blend). Predicted phase changes on 1.7 % (v6t) / 3.0 % (v6e) of rows.
* Frame rebuilt with the same code (`v5_frame.frame_rows`, `a2_features.build`); bookkeeping columns copied.
  Check: signal-windows whose phases did not change are identical except top_prob (v6t: 17 values of one sibling
  column differ; expert pair: <= 12 of 288 k rows, tie order).
## Function, note-45 recipe (first.all.wi, h3 cleansing, 283,894 training rows, 6 folds x 3 seeds); same rows everywhere
| set | v6 (note 45) | v6t (package today) | v6e (blend) | v6t - v6 [95 % CI] | v6e - v6 | v6e - v6t |
|---|---|---|---|---|---|---|
| everything acc7 (268,861 rows, 652 sig.) | .8714 | .8712 | **.8724** | -0.01 [-0.09, +0.07] | **+0.11 [+0.01, +0.21]** | +0.12 [+0.03, +0.20] |
| everything acc5 / core-four | .8759 / .9015 | .8758 / .9007 | .8768 / .9016 | -0.02 / -0.08 [-0.15, -0.01] | +0.09 / +0.02 | +0.11 / +0.10 |
| realistic acc7 (261,622, 651) | .8834 | .8833 | **.8845** | -0.01 [-0.09, +0.07] | **+0.11 [+0.01, +0.22]** | +0.12 [+0.04, +0.20] |
| realistic acc5 / core-four | .8879 / .9158 | .8878 / .9150 | .8889 / .9160 | -0.01 / -0.08 [-0.15, -0.01] | +0.10 / +0.02 | +0.11 / +0.10 |
By window, everything acc7 v6 / v6t / v6e: 5 min .8423/.8428/.8455 (v6e - v6 +0.32 [+0.12, +0.51]); 30 min .8727/.8730/.8738;
1 h .8719/.8704/.8720; 6 h .8871/.8877/.8883; 24 h .8880/.8883/.8886; full .8844/.8839/.8839.
Seed sd of acc7 .04-.05 pt (single-seed pairs differ by up to 0.10 pt). vs final_v2 head (b7 rows, 204,889 / 199,542):
everything +2.70 (v6) / +2.72 (v6t) / **+2.78 [+2.17, +3.43]** (v6e); realistic +2.87 / +2.89 / **+2.96 [+2.38, +3.61]**.
## Verdict
* The open item closes: on the input the package feeds today (trees-only phase_v3) the function head scores the same as
  note 45 (-0.01 pt, CI across 0); function_v3d's numbers stand for candidate v2 as wired.
* Feeding the BLEND to the function head is +0.11-0.12 pt (above seed noise, CI clear of 0; mostly 5 min, +0.3 pt), far
  below the 1-pt screening bar. Refit on all training signals (frame v6e, 3 seeds x 189 trees, numpy parity 0.0):
  `final_v3_work/function_v3e/`. Using it needs BOTH the weights and `predict.py` `_function_frame` reading `prob`
  (blend) instead of `prob_lgbm` (then the second, trees-only decode is no longer needed), plus a check.py re-freeze.
  Package, `model/` and STATUS.md untouched.
## Addendum: swapped into candidate v2 (orchestrator decision, 2026-09-29)
`final_v3_candidate_v2/`: function weights = function_v3e; `predict.py` `score` / `_function_frame` feed the blend `prob` (second trees-only decode removed); backup `final_v3_work/candidate_v2_backup_pre_v3e/`.
Parity (`trackA/v3e_package_verify.py`, 5 non-locked signals x 5 min / 30 min / 6 h / 24 h, 317 det-windows): package `_function_frame` on v6e's own phase input = v6e pred_phase / top_prob on 315 (<= 3e-8); 2 = window-edge candidate sets (phase 5 has no green inside the window). Package's own blend: frame phase = reported phase 317/317; = v6e 314, = v6t 307.
check.py re-frozen (`final_v3_work/freeze_refs_v3e.py`; sample: phase identical, function_pred identical, prob drift <= .096, distance_ft moved on 5/17), 3/3 PASS; card `update_card_v3e.py`.
Runtime (`bench_v2.py`, 20 signals, numpy): 30 min 0.79 s/signal, 6 h 2.44, 24 h 2.97 (was 0.82 / 2.48 / 3.08); peak 1.56 / 1.64 / 1.85 GB.
