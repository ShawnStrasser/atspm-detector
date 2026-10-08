# 88 — Detector fault events 83-88 removed from the label-cleansing chain; refit; candidate v4d (2026-10-04)
Brief (orchestrator, after note 87 finding 1): find every use of fault events 83-88 (user ban 2026-09-28), remove them,
refit what depended on them, score vs v4b, package. CPU <= 4 threads, GPU (note 86) untouched, locked_v2 never read
(asserted). Code `research/code/final88/`; work `%DC_WORK%/final_v3_work/{f88,v3fit88}/`; labels `function_labels_v4m`.
## 1. Where fault events were read (research/code grep; model/ = final_v2 not touched)
LIVE, reached the current models (fixed): `cabinet/dq_core.py` health part (FAULT_N / FAULT_FRAC) and ALLOWED (83-88
counted as comms coverage) -> dq_score / dq_suspect -> `v3_retrain --clean` dropped training rows (trees AND the siba
labels func_rows_v4l, via t57 setup); dq_core s_health -> `vd_audit.card` card_suspect ("erratic" on faults) ->
`v3_retrain.nc_high_mask` (not-checkable-high admission); `health/hb_data.weak_labels` (s_health, card erratic,
dq_score) -> presumed_healthy -> note-79 health evalset negatives -> note-83 chatter / rapid limit calibration.
Already clean: label_check's own health and card_faults (actuation-only), health_core (research + package), features
(package check asserts 83-88 never read; 229-feature parity), the v4b package code.
LEGACY, patched: `common.ALLOWED_EVENTS` (caches), `build_cache` (detector_meta.unhealthy), `health_sql` ('suspect' rule;
'failed', the one v3_retrain filters on, never used faults). History, untouched: experiments/*, health h3-h5 / hb_* studies,
build_detector_health counts, review87. model/ final_v2 still reports `controller_fault_events` -> gone at the next ship.
## 2. Recompute (dq88)
* dq_core fault-free on the 14,440 cabinet_v4l print rows (746 signals, 96 s): dq_suspect 1,971 -> 1,644; 328 true ->
  false, every one with a fault-event health reason; 1 false -> true and 7 tiny score moves from the coverage change.
* label_check health, 16,775 rows: 0 changes. Cards: 302 -> 281 suspect slots (21 dropped, all fault-only).
* v4m = v4l with the new dq columns only (truth columns asserted identical). Trees training rows (current v4l, new card
  rule) 298,713 -> 306,079: +7,379 rows / 228 detectors (Advance 60, YR 57, Mid 47, Presence 38, Count 12, Other 12,
  Bike 3 of the 229 changed training rows), -13 (the one new flag). func_rows_v4m (siba labels): ok .6550 -> .6712.
## 3. Six-fold OOF (oof88; v4l truth, gate .9 decode, stacker mean3 trained on unfiltered x69_siba mean, applied to
mean(x74_sibaflt 0,1,2) = v4b recipe; paired signal bootstrap). v4l-trees run reproduces f84 P_mean3_flt3 to 3.0e-8.
ctrl81 = trees refit on the CURRENT v4l (note-81 R5/R6 rules were applied after the note-80 fit v4b uses) -> pure dq effect.
| pool | v4b | v4c (v4m trees) | v4c - v4b | ctrl81 | v4c - ctrl81 |
|---|---|---|---|---|---|
| >= 30 min E (187,444) | .9192 [.9081,.9292] | .9193 [.9082,.9293] | +0.01 [-0.04,+0.06] | .9190 | +0.04 [-0.01,+0.10] |
| >= 30 min R (182,828) | .9291 | .9292 [.9185,.9387] | +0.01 [-0.04,+0.06] | .9289 | +0.03 [-0.02,+0.09] |
| 10 min E | .9034 | .9032 | -0.02 [-0.11,+0.07] | .9028 | +0.04 [-0.07,+0.15] |
| 5 min E | .8933 | .8937 | +0.04 [-0.07,+0.15] | .8933 | +0.04 [-0.06,+0.14] |
By class (>= 30 E, v4c - ctrl81): Count +0.12 [-0.03,+0.29], others within +-0.06; trees alone acc7 .8767 -> .8775.
Folds v4c - v4b -0.02..+0.05. Scored rows of the 228 re-admitted detectors (4,186): v4c - ctrl81 -0.29 [-0.58,-0.03]
(worth a look in the review; no reason to keep a banned filter). ctrl81 - v4b (R5/R6) -0.03 [-0.07,+0.01]. Neutral; adopt.
## 4. Package `%DC_WORK%/final_v3_candidate_v4d` (copy of v4b; v4b untouched; NAME v4d because note 86's plan reserves v4c)
Refitted on v4m (oof88 pkg*): function trees 229 x 3 (n_est = mean of the 18 v4m fold fits), lanes D (v4m OOF function
block), setback P50 (sb7 on v4m OOF, 42,374 rows), stackers mean3 + single. Unchanged: phase, decoder, GRU, siba members
(func_rows_v4l-trained), health code. ONNX vs text 1.95e-14 (21 checked). check.py --freeze, then 4 / 4 PASS (stored
answers 0.0; phase renumbering 1.6e-7; channel order 0.0; torch / lightgbm / scipy / sklearn blocked). Model card entry
candidate_v4d_note88 + sha256. Speed, 3 h warm, paired on the same (busy: note-86 GPU + another agent) machine, v4b / v4d:
typical 3.84 / 3.94, 3.49 / 3.44 s; busiest 5.59 / 5.66, 4.35 / 4.40 s; peak 913 / 907 MB -> unchanged (absolute
numbers ~+35 % vs note 84 because of the load).
## 5. Health reference (health88; writes only f88)
Presumed healthy 8,796 -> 9,020; evalset healthy-N 8,630 -> 8,974; note-83 calib: chatter .2632 -> .2639, rapid short /
mid moved <= .0008, rest same -> package limits NOT changed. (health83/calib.json, truncated by a first run, regenerated: identical.)
## Open for the orchestrator
* Adopt v4m as the training label set (LABEL_SETS / rpath) and v4d as the champion package; note 86's v4c build should
  start from v4d / v4m trees (its oof86 stack reads the v4l trees OOF - point it at f76/function_c_v4m).
* siba members and note-86 fold nets train on func_rows_v4l (fault-filtered); func_rows_v4m is ready for the next GPU run.
* Health limits of notes 38/43/79 were set on the fault-shaped presumed-healthy set; only note 83's re-checked (no move).
