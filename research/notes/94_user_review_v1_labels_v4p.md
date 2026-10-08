# 94 — User answers on review v1 (rows 1-17, 36, 37) + rulings of 2026-10-05 -> labels v4p, locked key v2 (2026-10-05)
Labels only: nothing trained, scored or packaged. Sheet `review/function_label_review_v1.xlsx` only read (md5 a96520df...
unchanged). Code `research/code/cabinet/v4p_user_v1.py` (corrections | build | desc | dec), `locked_v2_labels.py`
(pairs | build); work `%DC_WORK%/lab93/` (05999 reading, detpred, lists). locked_v2 asserted absent from v4p.
## 1. Answers -> `research/labels/function_label_corrections_user_v1.csv` (37 lines, answers verbatim)
* Out of training AND scoring: 2B091 whole signal (24 rows, already out by R2; truth now cleared too); 2B061 every P4
  channel (8, 9, 10, 11 labelled + 12, 34) - the user adds "or potentially the entire signal" (not done; det 50 P2 of the
  same row unanswered); 2B054 d8 / 9 / 23 (changepoint, loops probably replaced by radar); 10073 d61 (extends P1, does
  not call); 05048 d14 / 28 (no call / extend, not bike).
* Relabelled (source user_ruling, truth = training label, validated pass): 05999 d15 / 23 Advance (d25 = config Advance);
  2B095 d7 Advance, d14 Presence; 05166 d9 Advance, d10 Presence; 2B331 d1 / 2 / 16 / 17 / 23 Advance; 2B551 d22 / 24
  Advance, d23 / 25 Presence; 04034 d8 / 22 / 23 Presence; 06048 d24 Advance (was a print_high Presence), d25 Presence
  (was print_low Bike); 01067 d21 Presence + `signal_note` "unusual but valid (rail crossing, two stop bars)", stays in
  training. 2C009 d5 / d19 phase confirmed (already in v4l). 21 rows changed (05999 d25 unchanged).
* 04034 "why was the print reading wrong": it was not a print reading. d8 / 22 / 23 never had a print label (re-read 80:
  ph4 / ph8 re-equipped with radar after the print, left unread). The label was the config export's Function column
  (Advance, in both the Dec-2024 and Sept-2026 exports); the user read the channel TEXT ("Rad B - Presence", "Rad D -
  RL / LL Presence"). Same signal, not answered: d9 text "Rad B - Count" / d24 "Rad D - RL Count" vs Function Presence.
  Across v4p 598 rows have an ATSPM class in the channel text that contradicts the Function column; the model sides with
  the text on 325, the Function on 220 (`lab93/desc_conflicts.csv`, regex parse, noisy). Not changed.
* 05999 d23 CONFLICT: the print (input file J6-J/K = loops 21,22, config text agrees) puts d23 at the stop bar of the
  P8 right-turn lane (Presence); the user says Advance. User ruling applied, flagged for the orchestrator.
* Same pattern, unanswered, unchanged: loop channels with hand "advance presence" still labelled Other while the model
  says Advance: 2B331 d22, 2B417 d8, 04026 d8 / 9 / 21 (109 such loop rows in v4p, most already Advance / Mid by print).
## 2. 05999 print (share file named by its TSSU id 5CE055, sheet 1 rev 05/17/24) - `lab93/05999_print_reading.json`
332 cabinet, 7 sheets, loops only, BNSF rail crossing on the P4 leg. Input file matches the config text loop for loop
(except d15: loop 5 only; 6-7 are d27). All 16 active channels on the print; d8 (loop 8, "broken") dead -> tier
complete_mixed. P1 1 lane L (d1 Advance, d13 Presence); P2 2 T (d3 / d2 Advance lane 1 / 2, d4 = loops 3,4 Mid
spanning); P5 L (d15 Advance, d27 Presence); P6 2 T (d17 / d16 Advance, d18 = loops 14,15 Mid); P8 LT + R (d25 / d22
Advance, d24 / d23 Presence); P4 LTR (d8 Advance dead, d9 Presence, d10 Other / stopbar_secondary medium). All high
except d10. Under N1: d4, d18 Presence -> Mid (model Mid 1.00); d10 (model Presence .55) and d8 left out.
## 3. Rulings 2026-10-05 (USER_INPUT q1 / q4 / q5) on the training table
* N1 print vs hand (func7_v2), every confidence, not on user rulings / round-1 / R1 / stacked relabels: model = note-77
  champion OOF majority class over >= 30-min windows, agree if share >= .6 and >= 2 windows. 974 conflicts: hand kept 160
  (print high 71 / medium 82 / low 7), print used 499 (157 changed), left out 315 (no OOF 50, model third class 87,
  unsure 203; dead 52). The 71 high rows are task (b): print readings that overwrote a hand label the model agreed with,
  listed in `lab93/print_overrode_hand_model_agreed.csv` (64 signals); N1 already restores them.
* N2: channels 40 / 41 with no timing phase, text or config function -> out: 57 rows, 14 had a label. Not limited to
  Region 2: most are District 2B / 2C (Region 1); the 4 nameless-signal config rows on 40 / 41 (Count / YR) kept.
* Caution: 659 scored truth rows (465 ATSPM) are now chosen by agreement with the model (`n1_model_decided`); a
  headline on v4p is biased upward on them - report it with and without those rows.
## 4. v4p = `research/labels/function_labels_v4p.parquet` (v4o schema + rule_v4p, n1_*, user_review_v1, signal_note)
Change list `function_label_changes_v4p.csv`: 687 rows / 186 signals (N1 632, U 41, N2 14); ATSPM truth changed 415.
Trainable labelled detectors 14,266 -> 13,986; scored truth rows 10,810 -> 10,892 (ATSPM 8,746 -> 8,831). Slot
`v3_retrain.LABEL_SETS["v4p"]` (Dec table `lab93/dec_role_changed_v4p.parquet`: 515 rows recomputed, truth flags
517 -> 460). Default pointers NOT flipped (still v4o).
## 5. Locked key v2 = `research/labels/function_labels_locked_v2.parquet` (113 of 115 locked_v2 signals have rows)
Training rules, labels only (no model, no prediction, no metric; ON intervals read for the two actuation rules, TEST
Dec-2024 / NEWTEST Sept-2026): 5 stacked groups -> 7 relabels (unreviewed), YR identical 1, R1 4 (2 print, 2 out),
R5 1, R6 0, sweep 0, N2 10, radar-over-loops re-admitted 5 signals. N1 not applicable: 168 conflict rows flagged
`n1_pending_exam` (70 with ATSPM truth) and left out of `truth_v2` (old-rule truth in `truth_v2_oldrule`). Truth rows
1,455 -> 1,454 (old rule) -> 1,369. Changes `function_label_changes_locked_v2.csv` (99). Dec-role not applied.
## 6. (a) Prints under another file name (cached listing; candidates by city prefix + digits, then input file checked)
05999 -> 5CE055 (user); 05166 -> 5CE166 (all 14 loop texts match: same signal); 04156 -> 04CA156 (9 of 9 parsed loop
texts match); 05069 -> 5CE069 (Beltline WB off-ramp @ Delta Hwy; radar units A / B / C / E / F and phases match, MT
numbers renumbered since the 2019 print: stale). Street-name matches to other signals' files were all neighbouring
intersections. 20 signals (356 label rows, config only) have no DeviceName in the timing export: not searchable.
## For the orchestrator: 05999 d23 (user vs print); 2B061 whole signal?; 04034 d9 / d24 + the 598 text-vs-Function rows;
read 5CE166 / 04CA156 / 5CE069 into the store; flip pointers to v4p; AGENTS.md still has the old print rule.
