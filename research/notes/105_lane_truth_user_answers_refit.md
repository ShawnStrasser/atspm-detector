# 105 — User lane answers -> corrected lane truth; lane model refit; package copy v5c (2026-10-06)
Brief (orchestrator): turn every answer / comment in review/lane_count_review.xlsx (v1 rows 1-10), lane_count_review_v2.xlsx
and tr_r_lane_examples.xlsx into lane-truth corrections; refit lanes (note-102 2026 'base' recipe); v5c only if not worse.
CPU 4 workers, GPU unused, locked_v2 asserted absent, review files read only. Code `research/code/lanes/ln105_lanefix.py`
(build / fit / decode / full / eval / atspm), `research/code/final105/pkg105.py` (fit / assemble); work `%DC_WORK%/final_v3_work/f105`.
## Rules (labels only; the model still sees only the log)
* U - user answers exactly: n_lanes = his preferred answer; 'X also acceptable' kept as an accepted alternative.
  A right-turn lane he does not count: its detectors get NO lane (phase and function unchanged).
* A - arrow rule (TR + R split): lanes = movement arrows drawn on the approach. All 6 TR+R answers follow it (checked on
  the prints: 01072 P8 / 2B337 P8 / 04013 P4 one LTR arrow = 1; 01063 P8 / 03022 P4 / 03054 P4 two arrows = 2). Applied by eye
  to the other 8 TR+R phases of note 102b in the truth: 07047 P4, 07047 P8, 07050 P8, 08040 P8, 08080 P4 -> 1 lane;
  03042 P4, 2B003 P4 stay 2; 09040 P4 unchecked (print has no layout sheet). 03021 P8 / 08033 P4 are not in the lane truth.
* O - overlap lanes (user 'fine either way'; consistent option = his explicit 04064 answer, OLA / OLB lanes not counted):
  a lane whose detectors are all described OLA-OLD in the timing and typed R is not counted on any phase (no lane); the
  count with it is accepted. Generalised to 6 more phases (03008 P1 / P6, 03009 P4, 03072 P1, 2B019 P5, 2B036 P1).
  05025 P8 (user '2, 1 acceptable'): lane 2 is the OLA lane -> stays 1, 2 accepted.
* Lenient score also accepts n_lanes minus the R-only (right-turn) lanes still counted (01063-type answers say 2, the
  2B001-type say 'either'; the user's own answers are not one rule, so both counts are accepted).
* No phase or function label changed (user's 03024 'det 21 on P8' contradicts the timing, P2 kept; 03046 det 15 already P5;
  03057 comments = model errors). Label table stays v4q (no v4r).
* 09037 / 2B421 '?' ("no print, where do the lanes come from?"): both are released NEWTEST signals; their prints were read in
  the locked label store and the local copies sit in `%DC_WORK%/cabinet_locked/pdf/` - the sheets only looked in cabinet/pdf.
## Corrected truth (base = v4q print lanes, f95/ln8)
* `research/labels/lane_truth_corrections_user_v2.csv` (98 rows, supersedes v1) and `lane_truth_changes_v2.csv` (old -> new).
* 30 phases' n_lanes changed (U 19, A 5, O 6), 61 detector rows (U 34, A 15, O 12); 347 phases carry an accepted alternative.
  High-confidence training pairs 10,139 -> 10,088: 2 labels flipped, 56 dropped, 5 added (most fixed rows are medium-confidence).
## Refit (lanes D both orientations, 2026 function block, folds_v4 x 3 seeds; lam 3 picked on every fold, tau off)
14 Sept windows >= 30 min, 32,048 phase-samples / 95,948 high-confidence detector-samples, BOTH on the corrected truth;
before = note-102 base OOF (v4q + v1 answers), after = refit. Paired signal bootstrap.
| | n_lanes exact | lenient | 484 changed-phase samples | detector-lane exact (covered / all) |
|---|---|---|---|---|
| before | .8505 | .8842 | .4979 | .9322 / .8409 |
| after | .8509 | .8850 | .5124 | .9320 / .8407 |
| delta | +0.04 [-0.07,+0.15] | +0.09 [-0.01,+0.19] | +1.45 [+0.21,+3.25] | -0.02 [-0.09,+0.05] (all) |
* Truth effect alone: the before model scores .8481 on the old truth -> .8505 on the corrected one (+0.24).
* By length (after): m30 .8229, h1 .7950, h3 .8766, h6 .8812, h24 .8879, full .8949 (= before within 0.2).
* ATSPM function through the lane step (P_v5 fixed, ln102 recipe): >= 30 min E .9308 -> .9307 (-0.002 pt [-0.018,+0.016]),
  R .9398 = .9398; 5 / 10 min unchanged (no lanes). = tie.
* Verdict: not worse (overall tie, corrected phases +1.45*), so the package copy was built; no shuffled control (no new
  feature family, label change only).
## Package `%DC_WORK%/final_v3_candidate_v5c` (v5b untouched)
* = v5b + lanes D refit full-data on the corrected truth (pkg95 recipe: 78,028 labelled Sept pair-windows x 2, 495 signals,
  3 seeds, lam 3), ONNX parity 7.8e-16 (v3fit105/lanes_parity105.json), lane_model.json labels text + model_card note105.
* check.py --freeze, then check.py: 4/4 PASS. Sample answers identical to v5b (lanes, n_lanes, functions); only lane_conf /
  n_lanes_conf and function probabilities moved slightly (e.g. a Mid lane_conf .84 -> .66).
* Rebuild: ln105_lanefix build -> fit -> decode --notau -> full -> eval -> atspm; pkg105 fit -> assemble -> check.py --freeze / check.
## Open
* Caveat: v5b's lane model was fitted on v4q WITHOUT the v1 answers; the research 'before' had them, so v5c vs v5b moves a
  little more than the table (no package-level OOF).
* The arrow rule could re-read every 2+-lane print phase (the user says prints are often misread); only TR+R phases done.
