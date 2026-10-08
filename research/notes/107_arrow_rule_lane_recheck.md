# 107 — Arrow-rule re-check of every multi-lane phase against the prints; lane truth v3; v5c OOF re-scored (2026-10-06)
Brief (orchestrator): the user says print lane counts were often misread. Re-check every truth phase with n_lanes >= 2 (and
1-lane phases where the model confidently says 2+) with the note-105 arrow rule (lanes = movement arrows drawn on the
phase's approaches; overlap-only right-turn lanes not counted, counting accepted). High-confidence fixes -> truth v3;
re-score the existing v5c lane OOF (no retrain). Labels only. locked_v2 asserted absent; prints from cabinet/pdf plus
cabinet_locked/pdf for the 71 released NEWTEST signals only. Code `research/code/lanes/ln107_arrowcheck.py`
(worklist / collect / build / score); work `%DC_WORK%/rev107` (briefs, 150-dpi renders, per-signal readings `read/*.csv`)
and `%DC_WORK%/final_v3_work/f107/ln8` (truth v3, score107*.json).
## Reading
* Worklist 1,114 phases / 452 signals: 1,091 with truth >= 2 lanes + 23 one-lane phases with the v5c OOF majority >= 2
  (>= 9 samples, conf >= .8). Read by eye by 8 parallel readers (one CSV per signal = checkpoint) against a fixed protocol
  (`rev107/PROTOCOL.txt`); I re-read 01001, 01072, 08080 and 07002 myself: all readings correct.
* All 452 readable (13 needed another page of the same PDF). Confidence: 1,017 high, 97 medium, 0 low.
* Arrows = truth on 1,019 phases (91.5 %). The 23 one-lane "model says 2+" phases: all 23 confirmed 1 lane.
## Disagreements (95 phases: 72 high, 23 medium)
| kind | high | medium | outcome |
|---|---|---|---|
| right-turn lane drawn with an overlap arrow (OLA-OLD) counted in the truth | 58 | 8 | high -> corrected (old count accepted, rule O) |
| wide lane: one arrow over two loop columns side by side (01030 / 01072 type) | 10 | 2 | high -> corrected |
| exclusive phase right-turn lane the truth does not count | 4 | 4 | arrows accepted as alternative (user: RT 'either') |
| other (unlabelled slip lanes, a second approach marked 'No Work', OLA left lane) | 0 | 9 | listed |
* 3 user-answered phases (2B001, 2B047, 2B089 P6) kept as answered; 04035 P8 has no phase arrow at all (both detectors on
  the OLD dual-right lanes) -> listed, phase kept.
* Corrections (high only): 67 phases, 42 x 2->1, 23 x 3->2, 4->2 1, 5->4 1. Files: `research/labels/lane_truth_changes_v3.csv`
  (67 rows, one-line reason each), `research/labels/lane_truth_arrow_list_v3.csv` (28 rows: medium / rt-only / kept).
  Wide-lane fixes: 01062 P8, 04031 P8, 04071 P8, 07002 P4 / P8, 07025 P8, 08045 P8, 08148 P8, 2C046 P8, 2C054 P8.
* Detector lanes in v3: a phase corrected to 1 lane puts its laned detectors on lane 1; other corrected phases (and every
  overlap fix) keep their detectors but drop them from the high-confidence lane pairs (lane positions now unknown).
## Re-score of the existing v5c lane OOF (note-105 'after', 14 Sept windows >= 30 min, 32,048 phase-samples; no retrain)
| truth | n_lanes exact | lenient | 67 corrected phases (899 samples) |
|---|---|---|---|
| v2 (note 105) | .8509 | .8850 | .6974 |
| v3 (this note) | .8381 | .8844 | .2425 |
| v3 - v2 | -1.28 [-1.92,-0.70] | -0.06 [-0.33,+0.21] | |
* By kind: overlap fixes 777 samples, model = old count 70.4 % (exact v3 23.4 %); wide-lane fixes 122 samples, model = old
  count 65.6 % (v3 29.5 %). Unchanged phases .8553.
* So the corrected phases are NOT where the model was wrong: it reproduces the old counts. Corrected labels do not lower
  the lane error, they raise it (exact error 14.9 % -> 16.2 %); lenient is flat because overlap fixes keep the old count
  as accepted. Expected for a model fitted on v2: in 53 of the 57 overlap fixes the right-turn lane's detectors call the
  phase in the timing, so the log shows them as a separate lane of that phase; the wide-lane loop columns act like 2 lanes.
## Verdict / open
* The label misread the user suspects is real but small: 10 wide-lane phases (0.9 % of checked). The bigger difference is a
  convention: 57 overlap right-turn lanes the v2 truth counted. Lane-count error is not mostly label error.
* Open (orchestrator / user): should an overlap right-turn lane whose detectors call the phase count as a lane of that
  phase? v3 says no (old count accepted); the model, trained on v2, says yes. A lanes refit on v3 (ln105 fit/decode recipe on
  f107/ln8) would show whether the model can learn it; not run (labels-only brief). Unlabelled channelised slips were read
  inconsistently by the readers (none changed at high confidence).
* No package, phase or function label touched; truth v3 not yet used by any model.
