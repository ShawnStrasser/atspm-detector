# 119 — Locked exam of the production package v7 vs the beta (2026-10-07) — user-authorised, run ONCE
Authorised by the user on 2026-10-07 ~23:15 Pacific ("Run the exam now": one-time score, results reported as they come out, nothing tuned after). **The exam signals have now been opened again** (the 2025 beta / final_v1 / final_v2 were
scored on them in notes 12 / 13), so this is a confirmation, not an independent estimate. Nothing in any model, label or weight was changed after the numbers were seen. Code `research/code/final119/` (ex119 extract, run119 predict, score119
score, verify119 independent re-score); outputs `%DC_WORK%/s119/` (pred/, score119.json, rows_*.parquet, n1_decided.csv).
## Set-up
* Models: v7 = `dc_work/final_v7_prod` (detector_classifier 7.0.0, one path, health v4; phase / function answers bit-identical to v6b + note 114). Beta = repo `model/` (final_v2). Both run end to end from raw events, one `predict()` call per
  signal-window, package defaults (min_actuations 5), 2 DuckDB / ORT threads each, the two runs side by side.
* Data: 115 locked_v2 signals (43 TEST + 72 NEWTEST), Sept-2026 staging pull (all 115 have the full 66 h span). Windows = the OOF anchors (windows_stg): m5 x4, m30 x4, h1 x3, h3 x2, h24 x2. Headline >= 30 min = m30+h1+h3+h24 pooled.
* Phase truth: official timing (labels_official, all 4,102 locked rows are phase targets). E exact; R also accepts switch / additional call phases and drops high-confidence print-phase disagreements. Scorable = >= 5 actuations AND the labelled
  phase turns green in the window. Answer = phase_pred, else phase_guess (answered share .9994 at >= 30 min, both models).
* Function truth: locked key **v3** (`function_labels_locked_v3`, truth_v3). The 149 n1_pending_exam rows decided now with the v7 exam answers (print-vs-hand rule; v7 majority over the 11 >= 30-min windows, share >= .6, >= 2 windows): print used
  112, hand kept 17, out 20 (no model 13, unsure 4, third class 3); 107 ATSPM truths. ATSPM-only score with stack credit (atspm_score rule), >= 5 actuations, exclude_score out. Function R = E here (no validated fail rows in the locked key).
## Coverage (>= 30 min)
Phase: 45,122 labelled detector-windows; unscorable: 22,115 zero actuations (unused channels), 1,381 with 1-4, 285 labelled phase never green; scored E 21,341 / R 21,268, 113 signals. Function: 14,311 rows / 87 signals (1,303 n1 rows).
## Headline, v7 vs beta, same rows (acc [95 % signal bootstrap]; delta pt, paired)
| length | phase E v7 | phase E beta | d E | phase R v7 | d R | function v7 | function beta | d function |
|---|---|---|---|---|---|---|---|---|
| 5 min | .9405 [.926,.955] | .9466 | -0.61 [-1.44,+0.18] | .9423 | -0.57 n.s. | .8642 [.833,.892] | .8012 | +6.30 [+4.40,+8.29] |
| 30 min | .9771 [.967,.986] | .9733 | +0.38 [-0.09,+0.82] | .9780 | +0.37 n.s. | .9173 [.889,.940] | .8525 | +6.49 [+4.35,+8.65] |
| 1 h | .9802 | .9766 | +0.37 [-0.09,+0.85] | .9817 | +0.42 n.s. | .9180 | .8443 | +7.37 [+4.87,+9.94] |
| 3 h | .9823 | .9747 | +0.76 [+0.30,+1.26] | .9835 | +0.79* | .9291 | .8640 | +6.51 [+3.97,+9.04] |
| 24 h | .9838 | .9773 | +0.66 [+0.25,+1.07] | .9847 | +0.66* | .9249 | .8658 | +5.92 [+3.75,+8.14] |
| **>= 30 min** | **.9802 [.9704,.9891]** | .9752 [.9646,.9849] | **+0.51 [+0.12,+0.90]** | .9813 [.9728,.9893] | +0.52 [+0.13,+0.92] | **.9213 [.8927,.9434]** | .8554 [.8224,.8851] | **+6.60 [+4.35,+8.81]** |
Phase errors >= 30: v7 422 / beta 530. By set: TEST .9844 / .9748 (+0.96*), NEWTEST .9771 / .9754 (+0.17 n.s.).
Function without the n1 rows (they flatter v7, truth chosen by v7): >= 30 min v7 .9165 [.8869,.9403] / beta .8621, +5.44 [+3.24,+7.63]; 5 min .8636 / .8092. The n1 rows alone: v7 .9693, beta .7882. Locked key v2 truth (traceability,
its pending rows out): v7 .9153 / beta .8613. By set: TEST v7 .9443 / beta .8923; NEWTEST .8974 / .8169.
## Function by class, >= 30 min (truth class; v7 / beta / d pt [CI])
Advance (3,700) .9395 / .9146 / +2.49 [-0.11,+5.50]; Presence (4,304) .9517 / .9338 / +1.79 [+0.81,+3.03]; Count (2,713) .9661 / .8492 / +11.68 [+5.17,+19.16]; Yellow_Red (679) .8954 / .8085 / +8.69 [+4.31,+13.63]; non-ATSPM (2,915) .8178 /
.6810 / +13.69 [+7.20,+20.14]. v7 errors >= 30 min 1,126 (beta 2,070): non-ATSPM called ATSPM 531, wrong ATSPM 357, ATSPM called non-ATSPM 238; stack extra 53. Non-ATSPM mix-ups (secondary, not in the score): v7 .9857 of 2,384.
## Lanes and setback (v7 only; the beta has neither)
Lanes, locked print truth with the ln1 rules (298 phases / 62 signals): n_lanes exact >= 30 min .8251 [.791,.861] (m30 .797, h1 .791, h3 .866, h24 .871), within 1 .975. Loose truth (every printed phase, 560): .7823. OOF (note 99, per sample) .844.
Setback, printed Advance on the locked prints (153 detectors / 30 signals), >= 30 min: all rows medAE 42 ft, within 50 ft 54.7 %; predicted Advance + usual layout (995 rows / 25 signals) 36 ft / 59.9 % (h24 32 ft / 60.1 %); + high-confidence
print 33 ft / 62.9 %. OOF production path (note 58) 23.9 ft / 65.8 %: the exam is weaker, few signals (25).
## Exam vs OOF (v7 recipe OOF: phase v6_lagt note 113, function mean3 'nou' note 114; same lengths pooled)
Phase E >= 30: exam .9802 vs OOF .9839 (-0.37, inside the exam CI); R .9813 vs .9857 (-0.44, inside). Function >= 30: .9213 vs .9285 (-0.72, inside). Every single length 30 min - 24 h inside the CI, all 0.3-1.1 pt below OOF. **5 min is
outside: phase .9405 vs .9634 (-2.3), function .8642 vs .9022 (-3.8)**; the beta is also low at 5 min here (.9466), so the locked signals look harder in short windows; the v7 - beta 5-min phase delta is -0.6 n.s. (OOF had v7 ahead at 5 min).
## Run time (the user asked why the exam is slow)
Extract 31 s; v7 predict 1,725 calls 1,571 s wall (mean per call m5 .61, m30 .77, h1 .84, h3 1.15, h24 1.67 s; max 4.5 s); beta 1,725 calls 1,523 s (m5 .57, m30 .99, h1 1.40, h3 .53, h24 .87; first call 4.6 s),
the two in parallel on 2 threads each on a shared machine: ~26 min wall. Scoring 6 s; the independent re-score (verify119) minutes. The time is the number of calls (115 signals x 15 windows x 2 models = 3,450 end-to-end runs, each reading
the signal's events and loading its tables), not the scoring.
Verification: second script `verify119.py`, written by a separate agent from the written rules only (did not read score119 or its outputs), DuckDB SQL implementation -> `s119/verify119.json`. **Identical to 4 decimals** on every number checked:
phase E / R both models all lengths and row counts, unscorable counts (22,115 / 1,381 / 285), n1 decisions (112 / 17 / 13 / 4 / 3), function all lengths both models, without n1 (.9165 / .8621, 13,008 rows), all 5 classes, lanes loose .7823.
Caveats: function truth covers 87 of the 115 signals; Yellow_Red CI is wide (679 rows). Exam signals were opened before (notes 12/13);
labels of the locked key were built with training rules but never reviewed by the user. Nothing changed after scoring.
