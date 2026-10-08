# 93 — Accuracy by detector health; Yellow_Red zones with no Count on the phase (2026-10-05)
USER_INPUT items 2 / 3. v4f six-fold OOF (note 90; >= 30 min, v4l truth, E / R; reproduced .9197 / .9291, phase .9816 / .9841),
labels v4o, one seed-0 tree refit for B; CPU <= 4; locked_v2 asserted absent. Code `research/code/final93/`, work `%DC_WORK%/s93/`.
## A. Health vs accuracy
Health = the v4f package `health_core.health` called as predict.post_outputs does, on the same 304,432 OOF detector-windows
(events of the window only, de-duplicated; inputs = the OOF's own phase / phase prob / function probs / lanes): 305,295 rows, ok
248,591, not_enough_data 49,666, suspect 4,681, bad 2,357 (547 s, 4 procs). Accuracy %, signal-bootstrap 95 % CI, rows:
| status (share of function E rows) | function E | function R | phase E | phase R |
|---|---|---|---|---|
| ok (86.8 %) | 92.40 [91.33,93.37] 162,719 | 93.35 | 98.32 [97.98,98.65] 149,122 | 98.58 |
| not_enough_data (11.2 %) | 89.79 [88.48,90.95] 20,930 | 90.70 | 97.21 [96.72,97.68] 19,734 | 97.36 |
| suspect (1.4 %) | 89.77 [86.66,92.53] 2,629 | 90.89 | 96.72 [95.28,97.99] 3,052 | 96.88 |
| bad (0.6 %) | **75.04 [67.24,83.33] 1,166** | 75.82 | 97.25 [95.22,98.85] 1,419 | 97.19 |
| flagged (suspect + bad) minus ok, pt | -7.16 [-10.91,-3.38] | -7.06 [-11.27,-3.18] | -1.42 [-2.70,-0.25] | -1.60 [-2.91,-0.40] |
* Function E by the flagged detector's main check: rapid 77.8 [67.9,87.4] (936), corr 77.1 (157), night_day 78.4 (51), chatter
  81.2 (101), night_drop 85.3, choppy 86.5 (401), stuck 89.0 (1,384), dropout 89.0 (564), level 92.1, volume 96.2 (52); rule fired
  at all (any status): chatter 55.4 [39.1,74.3] (334), volume 58.5 (159). Phase by main check 94.0 (night_drop) .. 100.
* health_score is 1.0 for ok AND not_enough_data; suspect .25-.65, bad < .25; nothing in (.65, 1). Inside the flagged range it
  ranks: function E by score 0-.05 71.3 [61.4,81.9] (853), .05-.25 85.3 (313), .25-.5 87.5 (566), .5-.65 90.4 (2,063); phase flat (96-98).
* Filters (function E, accuracy / coverage): all 91.97 / 100 %; drop bad (score >= .25) 92.07 / 99.4 %; drop flagged (>= .7)
  92.11 / 98.0 %; status ok only 92.40 / 86.8 %. Phase 98.16 / 98.17 / 98.19 / 98.32.
* v4f output per detector (every call): phase_pred / phase_prob (+ 2nd), function_pred / function_prob, p_<class> x 7, status /
  review_flag, health_status, health_score, health_reason, health_bad_periods, health_watch, review_reason = the user's ask. Caveat:
  not_enough_data has score 1.0 -> filter on health_status, or give it NaN (package change, not made).
## B. Yellow_Red on a phase with no Count (77 detectors / 49 phases / 17 signals = user's 42 training-label phases + current truth)
* Matters? Scored rows (truth YR) 445 of 187,444 E (0.24 %, 10 signals): 33.9 % [12.2,98.4] vs 88.4 [84.5,91.9] for YR with a
  Count (9,046). Radar 88.3 [67.4,100] (163 rows, 9 signals) = normal; video = 12052 only, 282 rows at 2.5 %: model says Count
  (52 % of YR-no-Count rows; YR 34 %). Out of scoring: headline .9197 -> .9210 (+0.13), YR class 85.8 -> 88.4.
* Out of TRAINING (seed-0 trees, note-89 recipe; base v4o = note 89b's arm exactly; -1,177 rows / 44 detectors): >= 30 min E
  -0.02 [-0.08,+0.05], R -0.01 [-0.08,+0.05], 10 / 5 min +0.05 / +0.02 n.s., YR class +0.12 [-0.29,+0.49], own rows 34.2 -> 34.4 %.
* Behaviour (per-detector medians, >= 30 min; YR-no-Count / YR with Count / Count): first ON after begin-green 6.1 s [5.2,6.7] /
  4.65 / 4.75; greens whose first ON is >= 6 s in .47 / .25 / .26; share of ONs in red 1.1 % / 3.7 % / 7.9 % (Advance 48, Presence
  57); count vs the phase's presence total .75 / .94 / .74. Pulse .99 on radar, ~0 on video (all video zones). = speed-filtered
  stop-bar zones (late first car, off in red), not presence or advance. Without a Count lane-mate the model has no contrast and
  says Count on video; training on them does not fix it (accepted limitation).
* Per phase: REAL set-ups (print high + YR-like behaviour): 12052 (6 phases, the 282-row loss; P1 / P5 / some P4 / P8 zones ON
  in the opposing green = permissive turns), 2B341 P3, 2B353 P1, 2B432 P4, 2C023 P3 / P7, and the YR zones of 2B048 / 2C026 /
  2C040 P4. NOT really without a Count (a Count zone exists, its label was dropped from truth): 10007 P2, 10050 P6, 2B054 P5,
  2C045 P6, 2C074 P6 (model Count .78-.98). LABEL ERRORS on the partner: 04016 P8 d23 "Rad D - Count" (+ unlabelled d25 "RT
  Count") and 04040 P7 d28 "Rad B - Count" are labelled Presence but are pulse zones with 0-16 % of ONs in red (Presence 57 %),
  model Count .97 / .82 / .42. NO DATA: 12060, 12067, 12072 and 04016 d48 (YR channels silent in every window).
* CAN'T TELL -> review/yr_without_count_review.xlsx (+ _charts/, 10 rows, charts from saved Sat 19 Sep 12-15 h events):
  08CM406 d41 / 42 / 44 (print low; 55-83 % of ONs in red); 12053 d33+39 (advance-like pulse) and d37 / 38 / 43 / 45 (25-87 % in
  red; its other 7 zones look like 12052's); 2B382 d5 (config only; unlabelled pulse d31 / 36); possible Count partners labelled
  Advance: 2C023 d7 + d21 ("count z6 / z4", print high, 4-9 % in red, model Count .94), 2B048 d8 + 9 ("CO", 0 % in red),
  2C026 d8, 2C040 d8.
## Answers / for the orchestrator
* A: yes for function (flagged -7 pt, bad 75 vs ok 92 %), slightly for phase (-1.4 pt); dropping "bad" = 0.6 % of answers, +0.1 pt.
* B: real set-ups or dropped / wrong Count labels; keep them (training-neutral; scoring +0.13 pt = 12052). Sheet optional. Partner
  fixes 04016 d23, 04040 d28 (Presence -> Count) not applied (fold into the next label pass; v4p = note 94, not used here).
