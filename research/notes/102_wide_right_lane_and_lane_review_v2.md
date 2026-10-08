# 102 — Wide right lane (extra stop-bar loop) fix attempt + lane review sheet v2, one row per signal (2026-10-06)
Brief (orchestrator, user feedback on review/lane_count_review.xlsx): try to fix with hi-res data the recurring error "extra
stop-bar loop on the right of a through / right lane = still ONE lane"; retrain lanes six-fold OOF in the 2026 set-up; then a
v2 sheet, one row per signal, reviewed phases left out. CPU <= 6 workers, GPU not used, locked_v2 asserted absent.
Code `research/code/lanes/ln102_wide.py` (setup / wcues / tri / fit / decode / full / eval / atspm),
`research/code/evaluation/lane_review102.py` (rows / data / write); work `%DC_WORK%/final_v3_work/f102/ln8/`, `%DC_WORK%/rev102/`.
## User answers of review v1 used (rows 1-10 answered, 11-25 blank)
* 01030 P8 and 01072 P4: print lane count was MISREAD - 1 wide lane, not 2 (det 24 / det 9 = loop to the right in the same
  lane). Lane truth corrected: `research/labels/lane_truth_corrections_user_v1.csv` (2 phases, 4 detector rows), applied in
  ln102 setup only (label table untouched). 03044 P4 / 04059 P8 = the pattern, model wrong (over-count). 03010 / 03021 / 04064 P4
  "either is acceptable"; 03024 / 03046 / 03057 confirm the print.
* Finding: the extra loop is mostly medium / low print confidence -> not in the lane model's labelled pairs (high only), and
  several "2-lane" prints with a TR + R stop-bar split may be the same misread (01030-type) teaching "different lane".
## Set-up (= package v5 lane recipe, pkg95.pkglanes): lanes D both orientations, cues + context + pair type + function block of
the 2026 tree OOF (function_c_v4q26), v4q print-lane truth + the corrections, Sept-2026 rows only, folds_v4 x 3 seeds, lam
picked per held-out fold. Pattern set (eval only): 83 print phases / 56 signals with >= 2 single-lane stop-bar zones in one
print lane (same function, or stop-bar + 'Other' drawn R / TR) + the 4 user-confirmed phases; 886 phase-samples.
## Variants (pair AUC >= 2-lane phases m30 / h6 / full; n_lanes exact per sample, 14 Sept windows >= 30 min, 32,048 phase-samples)
| variant | AUC m30 / h6 / full | exact all | pattern (886) | vs base all [CI] | vs base pattern [CI] |
|---|---|---|---|---|---|
| base (2026 retrain) | .9634 / .9747 / .9783 | .8481 | .8183 | - | - |
| W: + 16 interval pair features (co-occupancy, nested ONs, same-vehicle, lift, night) | .9667 / .9756 / .9786 | .8483 | .8228 | +0.02 [-0.15,+0.18] | +0.45 [-0.74,+1.85] |
| Wm: W + medium-confidence print rows as training pairs | .9662 / .9754 / .9784 | .8491 | .8149 | +0.10 [-0.16,+0.37] | -0.34 [-2.2,+1.25] |
| T: + 2nd-stage shared-neighbour features (max_k min P(a,k), P(b,k)) | .9685 / .9773 / .9799 | .8471 | .8273 | -0.10 [-0.28,+0.08] | +0.90 [-1.0,+2.49] |
| ref: note-77 OOF (review-v1 model) | - | .8445 | .8149 | -0.36 [-0.78,+0.06] | -0.34 |
* Decoder rules, grid per held-out fold (objective nl_exact + pair_acc_multi, 9 Sept windows): 'stack exemption' (same-role pair
  with P(same) >= .90 / .95 decoded as one lane, weaker = Other) never picked on any fold (base / W / Wm); 'Advance may not
  span' (log prior -3 / -6 / -10) lowers the objective at every lam (10.70 -> 10.67 / 10.62 / 10.55), never picked.
* User-confirmed 4: 01030 P8 and 01072 P4 right in every variant; 04059 P8 1 / 13 samples right, 03044 P4 0 / 14 in every variant.
  03044: P(det 11, det 12) = .01 in all models while the advance (det 8) is "same lane" with both (.91 / .94) -> decoder splits.
* Phase majority (base): pattern phases over-counted 9 of 70 (1->2 3, 2->3 5, 3->4 1); all phases wrong 272 vs ref 288.
* Wm: truth-1 +0.33 [+0.09,+0.65] but truth-3+ -0.94 [-2.0,0.0], pattern -0.34: mixed, not adopted.
* No variant clears the noise floor overall or on the pattern -> per the screening rule nothing adopted, no shuffled control
  run (needed only for an apparent gain). The 2026 retrain 'base' is +0.36 vs the note-77 OOF (CI touches 0).
## ATSPM function through the lane step (P_v5 fixed; lanes5g + lane_conf of the gate swapped; pick inputs / stacker context fixed)
>= 30 min E (126,505 rows): base .9308, W .9306 (-0.01 [-0.04,+0.01]), T .9304 (-0.03 [-0.07,+0.00]), ref77 .9305 (= v5 headline,
sanity) ; R base .9398, W -0.02, T -0.03 n.s.; 5 / 10 min unchanged (no lanes). As expected ~0.
## Sheet `review/lane_count_review_v2.xlsx` (+ `_charts/`, 19 PNG) - model = 2026 base lanes (no fix adopted)
* Phase majority (note-99 rules), left out: the 8 phases the user answered in v1, 121 blind under-counts, how-sure < .70.
  43 eligible phases / 38 signals; 17 signals with the highest how-sure + 2 NEW signals (wrong now, right for the v1 model;
  how-sure >= .60; only 1 confident new phase exists) = 19 rows / 23 phases; NEW rows shaded yellow, phase marked NEW.
* Detectors: one per line per phase, "det 15: print P2 Presence lane 1 | model P2 Presence lane 1", flags "<- wrong phase",
  "<- wrong function"; model function = v5 OOF decoded through these lanes; "no lane (the lane step saw it as Bike)" when the
  lane step's tree function differs. Likely cause: wrong function 12, wrong phase 8, lane grouping 3 (none 'wide right lane':
  the pattern errors left are low-confidence or reviewed). One question per phase; print link + chart (main phase).
* Remaining errors are mostly the function head calling a lane loop Mid / Bike / Other (06040, 08022, 2B001, 2B089, 2C039,
  10093) or phase errors (04064 overlap lanes, 2B464, 2C009, 2C070).
## Open / caveats
* Real fix probably needs labels, not features: the extra loops are unlabelled or labelled as a second lane; a check of the
  "TR + R split" 2-lane prints (01030-type misreads) would clean the truth. Ask only if the orchestrator wants it.
* Pick inputs (ln6 / ln7) not rebuilt for the variants; package lanes unchanged (model/ = final_v2, v5 candidate untouched).
