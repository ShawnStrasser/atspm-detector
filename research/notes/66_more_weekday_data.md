# 66 — More data per detector for function: weekday windows added to training (2026-10-01)
Question: does MORE data per detector help function? Code `trackA/s66_moredata.py` (cache / parity / feat / frame / fit /
score / wkd / offpeak); work `%DC_WORK%/s66/`. CPU only (8 workers x 1 DuckDB thread, LightGBM 8 threads). locked_v2
asserted absent everywhere. Nothing shipped. GPU folders untouched.
## Data
* Daily pull `data/staging_2026_w40`: Sat 26 Sept 16:15 .. Wed 30 Sept 24:00 (Oct 1 arrives with the 06:30 pull of Oct 2).
  Used Mon 28 - Wed 30 (72 h, all weekday). The current frame's "stg" period is the EARLIER pull, Fri 18 Sept 16:15 -
  Mon 21 10:25 (two-thirds weekend); none of the w40 days were in any frame before.
* New period `wkd`: same 22 windows / lengths as stg (4 x m5, 4 x m10, 4 x m30, 3 x h1, 2 x h3, 2 x h6, 2 x h24, full66),
  weekday anchors mirroring the stg mix (AM / PM peak, midday, night; full66 = Mon 06:00 - Thu 00:00). 756 frame stg signals.
* Features = the final_v3 candidate package's code (`final_v3_candidate_v2`): tables built once from the signal's whole
  72-h log, each window cut with build_features(t0, t1) (apply_window), expert features binned on the window -> the research
  builder's definition. PARITY: with frame v6e's own OOF phase input, all 229 arm columns identical on 953 rows (11 signals,
  m5 .. full66). (Window-only events = the production path differ on ~25 % of values: edge effects; not used.)
* APPROXIMATIONS: phase input of the new rows = the package's full-fit phase blend (not six-fold OOF; training rows only);
  health = detector `bad` in stg health3 or in the w40 health run (note 46) -> out. Labels = v3s per detector, same masks.
  Lanes / pick inputs are evaluation-side and the evaluation rows are unchanged, so nothing to rebuild.
* New rows 284,876 (756 signals); trainable 190,522 (684 signals; +67 % on top of the 284k-row training set; health-bad
  out 6,174). Each new row inherits its signal's fold; fold k's model trains on new rows of folds not in {k, inner} only;
  early stopping on the original inner-fold rows; prediction on exactly the existing OOF evaluation rows.
## Result (note-64 scorer, trees arm, no fj: D-lane stack pick + cy twin decode; paired signal bootstrap, pt)
* Control: `base_re` (same code, no new rows) reproduces base seed 0 exactly (delta 0.000 everywhere); base 3 seeds
  through this scorer = cand64 headline exactly (.9009 E / .9137 R >= 30 min).
| arm vs base (same seeds) | >= 30 min E | >= 30 min R | 5 min E | 10 min E | all E |
|---|---|---|---|---|---|
| plus s0 | .9009, +0.09 [-0.00,+0.19] | +0.08 [-0.01,+0.18] | +0.05 [-0.13,+0.23] | -0.03 [-0.22,+0.16] | +0.07 [-0.02,+0.16] |
| plus s1 | +0.03 [-0.06,+0.12] | +0.04 [-0.06,+0.13] | -0.01 | -0.05 | +0.01 |
| plus s2 | +0.05 [-0.04,+0.14] | +0.05 [-0.04,+0.14] | +0.20 [+0.02,+0.38] | +0.04 | +0.07 |
| **plus 3 seeds** | **.9013, +0.04 [-0.04,+0.13]** | **.9141, +0.04 [-0.04,+0.13]** | .8679, +0.18 [+0.00,+0.34] | +0.02 | +0.06 [-0.02,+0.14] |
3 seeds by window E: m30 +0.07, h1 +0.08, h3 +0.04, h6 +0.04, h24 -0.05, full +0.05 (all CIs span 0); per fold >= 30 min
-0.13..+0.18 (two folds negative). Seed 0 screen sat at the CI edge, so 3 seeds were run: the gain is noise-sized.
## Secondary: is a weekday window harder? (argmax, no decode; trained-label rows; 11,439 detectors present in both)
| model | Sept stg >= 30 min | weekday >= 30 min | paired per detector (weekday - Sept) |
|---|---|---|---|
| base (no weekday rows), s0 | .8930 | .8888 | **-0.36 [-0.54,-0.19]**; full -0.80 [-1.17,-0.43], m30 -0.36, h1 -0.37, h3 -0.32 |
| plus (weekday rows of other folds), s0 | .8939 | .8902 | -0.29 [-0.46,-0.12] |
5 / 10 min: weekday +0.17 / +0.14 n.s. Weekday rows even had the more favourable (full-fit) phase input, so the gap is not
a phase-input artefact. Reading: weekday samples are slightly HARDER, not easier (peak queues blur Presence / Count /
Advance timing), and training on weekday data recovers only ~0.1 pt of it.
## Off-peak features re-test (note 59 d; cheap, on top of plus s0)
12 note-59 columns (outside weekday 06:30-09:00 / 15:30-18:30); now non-NaN on 56 % of Sept and 53 % of weekday rows, with
real peak / off-peak contrast in the weekday training rows. Sept eval >= 30 min: -0.02 [-0.11,+0.06] vs plus s0; shuffled
control -0.03; 5 / 10 min +0.06 / +0.06 (control +0.10 / +0.04). Weekday eval >= 30 min .8905 vs plus .8902 vs shuffled
.8898. Dropped again: off-peak behaviour adds nothing the whole-window features lack, even with weekday data.
## Verdict
* More data per detector (3 more weekdays, +67 % training rows): +0.04 pt at >= 30 min (3 seeds, CI spans 0), nothing at
  10 min, +0.18 at 5 min (CI touching 0). Far under the 1-pt bar -> function is NOT data-limited per detector; its gap is
  labels / ambiguity (notes 61, 70), consistent with the curves flattening from 24 h to full.
* Not worth adding to the final fit for accuracy; harmless if wanted for robustness (weekday coverage). The daily pull
  needs no change for function's sake.
* Weekday samples score ~0.3-0.4 pt lower than the Sept (weekend-heavy) ones at >= 30 min: the Sept OOF headline is, if
  anything, slightly optimistic for weekday use. Worth one sentence to the user with the final numbers.
## Files
`%DC_WORK%/s66/`: events/ (w40 Mon-Wed cache), feat/ (per-signal package features), wkd_frame.parquet (+json), parity.json,
fit/{plus,base_re,plus_op,plus_op_shuf}/ (P_* = frame-aligned OOF, Pwkd_* = fold-k weekday rows), score_{screen,seeds,
seed1,seed2,op}.json, wkd_*.json, offpeak_wkd.parquet, cache/ (weekday det_intervals / phase_cycles), logs/.
