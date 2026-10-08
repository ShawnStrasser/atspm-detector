# 95 — Labels v4q + final model on 2026 data only: package v5 (2026-10-05/06) — not shipped, locked untouched
Code `research/code/cabinet/v4q_rules.py`, `research/code/final95/` (pool95, phase95, func95, pkg95, lr95, export95_phase,
assemble95, swa95, le2h95), `neural/flab95_v4q.py`, `neural/phase_v3_net.py` (DC_SIG2026), `neural/tcn69_func.py` (--swa).
Work `%DC_WORK%/s95/`, `final_v3_work/{f95,v3fit95}`, `lab95/`; GPU queue `tcn53/q95/` (snap95); RunPod pod dc95a (siba).
## 1. Labels v4q (labels only; pointers flipped: rpath.LABELS_CURRENT / DEC_ROLE_CURRENT, v3_retrain "v3s", oof88 default)
* `function_labels_v4q.parquet` = v4p + rulings of 2026-10-05 16:20 + note-93 items + 3 prints; changes
  `function_label_changes_v4q.csv` (39 rows, 28 ATSPM truth). "Model" = note-77 champion OOF majority (share >= .6, >= 2 win).
* Q1 05999 d23 out. Q2 2B061: only P4 out (asserted). Q3 110 loops with hand "advance presence": hand label := Advance;
  user-named 5 -> Advance; 8 relabelled, 67 already Advance; print disagrees -> N1 (22 decided, 2 out).
* Q4 channel text vs config Function (633 parsed; bare "CO" no longer read as Count; rows decided by a print skipped):
  text 6, restored from N1 exclusion 7, Function kept 16, out 7 (04034 d9 / d24, model .50). Q5 04016 d23 -> Count (model
  1.00); 04040 d28 stays out (model YR .54). Q6 prints: 04156 complete_high 10/10 agree; 05166 incomplete, d23 -> Mid,
  d24 -> Presence (model), d21 / d22 unmatched (no whole-intersection Other); 05069 skipped (2019 print, no channel maps).
* Truth rows 10,902 (ATSPM 8,852); n1_model_decided truth rows 624 (413 ATSPM). Locked key v3
  `function_labels_locked_v3.parquet` (labels only): 19 loops -> Advance without model; pending exam 168 -> 149.
## 2. 2026-only data (Sept 18-21 2026 staging period; Dec-2024 dropped everywhere; no w40 period added)
| | before (Dec-2024 + Sept-2026) | 2026 only |
|---|---|---|
| function frame signals / detectors | 772 / 15,540 | 756 / 14,914 |
| function training rows / signals / detectors | 347,784 / 735 / 13,385 | 234,807 / 719 / 13,116 |
| function scored truth rows / signals / detectors | 272,798 / 659 / 10,131 | 177,106 / 642 / 9,796 |
| phase pool signals (labelled pair rows) | 772 (367 on Dec-2024 only) | 761 (1,837,561) |
* Phase pool `s95/phase/pool.parquet`: STG 334 + REL 71 + the Dec-pool signals' own Sept-2026 rows (352) + 4, unlabelled
  detectors KEPT as rows; labelled STG/REL rows identical to f76 (0 of 271 columns differ). Folds = folds_v4 everywhere
  (TCN too). Held fixed (as notes 83-90): the function frame's pred_phase input and the lanes / pick / twin OOF chain
  (older OOF that saw Dec-2024); setback P10/P90 band (note 41, Sept-2026). Rebuilt: everything else.
## 3. Six-fold OOF headline, Sept-2026 windows (v4f = its own saved OOF on the same rows)
| phase (761 signals) | v5 | v4f on shared rows (405 sig) | v5 - v4f shared |
|---|---|---|---|
| >= 30 min E (189,509) | .9843 [.9811,.9870] | .9871 | +0.01 [-0.08,+0.10] |
| >= 30 min R | .9863 [.9835,.9888] | .9892 | +0.00 [-0.10,+0.09] |
| 10 / 5 min E | .9739 / .9640 | .9761 / .9674 | -0.05 / -0.07 n.s. |
Trees alone 2026 vs v4f trees (shared) -0.06 [-0.15,+0.03]. By src: STG .9862, REL .9919, Dec-pool signals' 2026 .9818.
| function, v4q truth (642 signals) | v5 | v4f | v5 - v4f |
|---|---|---|---|
| >= 30 min E (126,505) | .9305 [.9214,.9388] | .9316 [.9226,.9396] | -0.11 [-0.33,+0.09] |
| >= 30 min R | .9395 [.9307,.9474] | .9396 | -0.00 [-0.20,+0.19] |
| >= 30 min E without n1_model_decided (118,592) | .9309 | .9304 | +0.05 [-0.14,+0.25] |
| n1_model_decided rows only (7,913) | .9252 | .9497 | -2.45 [-4.18,-0.96] |
| 10 / 5 min E | .9122 / .9047 | .9122 / .9039 | +0.00 / +0.08 n.s. |
By class >= 30 E (v5 - v4f): Advance -0.24, Presence -0.06, Count -0.46, YR -0.34, non-ATSPM +0.47, all n.s.; folds -0.32..+0.25.
v5 with seed-0 nets only: .9296 (-0.20 vs v4f). Reading: 2026-only = TIE with v4f on both outputs; the n1 rows favour v4f
because their truth was chosen by agreement with the older model (report both numbers).
## 4. Recipe (unchanged from v4f except data)
Phase: ranker 3 seeds (n 565), TCN ad_all 6 folds (local A1000, ~40 min each) + full refit p95_ad_full (31 epochs, lr
replayed), decoder 396 trees on trees + TCN blend. Function: trees 229 x 3 (Sept-2026 rows), siba 6 folds x seeds 0/1/2
(RunPod RTX PRO 6000, 7 jobs at once, ~30-45 min each, filtered by the 2026 tree OOF), full members s95_sibafull seeds
0/1/2 (42 epochs; train loss .491 / .523 / .496; member disagreement on bench pieces .103 vs v4f trio .118 -> no outlier),
stacker mean3 (+ single), lanes D, setback P50 refit. Cloud outputs md5-verified locally (60 files).
## 5. Package `%DC_WORK%/final_v3_candidate_v5` (copy of v4f code + note-98 no-spin onnxruntime + health fix)
* health_score = NaN when health_status = not_enough_data (orchestrator). ONNX parity: trees <= 2e-14, TCN 9.5e-6, siba
  1.4e-5 (filtered 8.4e-5, pass 1e-4). check.py --freeze then 4/4 PASS. Model card entry final_v5_note95 + sha256.
* Speed (bench98 protocol, 2 passes, warm s / peak MB): 30 min 0.74-1.37 / 502-752; 3 h 1.57-2.90 / 524-760; 24 h
  1.75-4.16 / 562-889. v4f same protocol (spin on, loaded machine) 3 h 1.97-3.25. Beta 3 h 0.40-0.57 s / 259-296 MB (note 98).
* Fast package `final_v3_candidate_v5_fast` re-based on weights_v5 (note 98 layout; extras decode_trees / stacker nonet /
  single refit on the v5 OOF, `f95/extras`), check 5/5; default le2h 3 h 0.57-0.94 s / 292-322 MB; accuracy note 95b.
Locked signals: not used (locked_v2 asserted absent in every loader; locked key v3 labels only).
