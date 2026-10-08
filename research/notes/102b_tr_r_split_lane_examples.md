# 102b — Through/right stop-bar split ("TR loop + extra R loop"): how the print-lane truth counts it (2026-10-06)
Brief (orchestrator, user question "I thought you had already labelled those as one lane?"): count how the current print-lane
truth treats a loop approach whose stop bar has a TR loop plus an extra loop on the right (corner widening); ~10 examples.
Labels / analysis only. Code `research/code/evaluation/tr_r_lane_examples.py`; truth = `%DC_WORK%/final_v3_work/f102/ln8`
(v4q print lanes + `research/labels/lane_truth_corrections_user_v1.csv`); locked_v2 asserted absent; CPU, single process.
## Rule (loop stop-bar detectors only; stacked radar / video zones left out; single-lane rows)
* counted 1 lane: one print lane holds >= 2 loop stop-bar detectors and its lane types carry both T and R
  (mostly the extra loop labelled Other / stopbar_secondary).
* counted 2 lanes: two adjacent lanes with loop stop-bar detectors, the right one typed R / TR, the left one's lane carries an
  R movement (often only through the single upstream advance loop typed LTR / TR), and no advance loop of its own on the
  right lane. Left lane with no T (L + R on a side-street stem) reported separately.
* Checked by eye on the 01072 print (sheet 6): P8 approach = one LTR arrow, stop-bar loops 23-24 and 25-26 side by side =
  exactly the P4 layout the user called one lane, but P8 is still truth 2 lanes.
## Counts (phases)
| how counted | high | medium | low | all | signals |
|---|---|---|---|---|---|
| 1 lane | 0 | 13 | 2 | 15 (13 as read + 2 user answers: 01030 P8, 01072 P4) | 13 |
| 2 lanes | 7 | 9 | 0 | 16 | 15 |
| 2 lanes, L/R split on a stem (01062 P8, 04104 P1, 07025 P8) | 0 | 3 | 0 | 3 | 3 |
* TR + R core: 31 phases / 27 signals. 5 of the 34 phases are not in the lane truth (phase dropped): 3 one-lane, 2 two-lane.
* Confidence = all stop-bar rows of the pair high -> high; any low -> low; else medium.
* 2-lane list: 01063 P8, 01072 P8, 03021 P8 (extra loop given its own lane 3), 03022 P4, 03042 P4, 03054 P4, 04013 P4,
  07047 P4, 07047 P8, 07050 P8, 08033 P4, 08040 P8, 08080 P4, 09040 P4, 2B003 P4, 2B337 P8.
* 1-lane list: 01017 P8, 01030 P8*, 01072 P4*, 01074 P4, 03043 P4, 03043 P8, 03044 P4, 04043 P8, 04059 P8, 04073 P4,
  04073 P8, 07005 P4, 07041 P4, 07051 P8, 08020 P8 (* = user answers).
## Sheet `review/tr_r_lane_examples.xlsx` (user asked)
* 2-line answer on top; 10 rows (4 counted 1 lane: 04059 P8, 01074 P4, 01017 P8, 03044 P4; 6 counted 2 lanes: 01072 P8,
  01063 P8, 03022 P4, 2B337 P8, 03054 P4, 04013 P4); detectors with print lane type, role, loop numbers and our lane;
  one-line reason; print link (local copy); question "Is this one lane or two? 1 / 2 / ?".
## Open
* If the user says these are one lane, the 16 (+3 stem?) two-lane phases should be re-labelled like 01030 / 01072 P4 and the
  lane step re-fitted (note 102 found features alone cannot fix it while the truth says 2). Nothing changed yet.
