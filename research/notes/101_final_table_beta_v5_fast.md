# 101 — Final comparison table for the morning page: beta vs v5 full vs v5 fast (le2h) (2026-10-05) — saved OOF only
Code `research/code/final101/table101.py` (phase / func / bench / assemble); output `%DC_WORK%/final_v3_work/final_table101.json`
(parts in `final_v3_work/f101/`). Nothing trained, CPU 4 threads, GPU not touched, locked_v2 asserted absent.
## Scope
* Six-fold OOF, Sept-2026 windows, >= 30 min, paired rows (beta, v5 and v5 fast all score the row), signal bootstrap 95 % CI.
  Phase truth = timing (R accepts switch / additional call phases); function truth v4q, ATSPM score with stack credit.
* beta = note-48 final_v2 phase OOF (GRU <= 2 h, trees above) and note-25 b7 function OOF (5 classes, argmax; monday85
  protocol). v5 = note-95 OOF (phase p2_p95_ad; function P_v5 + gate .9 decode). fast = note-95b le2h (phase p2_le2h;
  function P_v5 <= 1 h families, P_v5_nonet above).
* Coverage: beta OOF exists only on the STG + REL signals; the Dec-pool signals' own Sept-2026 rows (v5 pool only) have none.
  Paired = 45 % of v5 phase rows (85,695 E / 405 sig) and 63 % of v5 function rows (80,280 E / 373 sig). v5 on all its rows
  reproduces note 95 (.9843 / .9863 phase, .9305 / .9395 function).
## Accuracy (>= 30 min, paired)
| row | n E | beta E | v5 E | fast E | v5-beta pt [CI] | fast-beta pt [CI] | beta R | v5 R | fast R |
|---|---|---|---|---|---|---|---|---|---|
| Phase | 85,695 | .9856 [.9813,.9897] | .9872 [.9827,.9913] | .9849 [.9805,.9890] | +0.16 [+0.01,+0.28] | -0.07 [-0.16,+0.01] | .9879 | .9892 | .9871 |
| Function | 80,280 | .8852 [.8719,.8984] | .9370 [.9262,.9469] | .9338 [.9230,.9437] | +5.18 [+4.30,+6.10] | +4.86 [+4.03,+5.71] | .8952 | .9454 | .9421 |
| Advance | 24,338 | .9043 | .9440 | .9434 | +3.97 [+2.53,+5.59] | +3.90 [+2.52,+5.47] | .9167 | .9531 | .9520 |
| Presence | 23,855 | .9512 | .9609 | .9585 | +0.97 [+0.35,+1.59] | +0.73 [+0.11,+1.34] | .9609 | .9711 | .9688 |
| Count | 14,699 | .9204 | .9433 | .9410 | +2.29 [+0.10,+4.57] | +2.06 [-0.08,+4.23] | .9370 | .9568 | .9544 |
| Yellow_Red | 4,327 | .6600 | .8706 | .8655 | +21.1 [+16.5,+26.0] | +20.5 [+16.1,+25.4] | .6600 | .8706 | .8655 |
| Non-ATSPM | 13,061 | .7637 | .8952 | .8854 | +13.1 [+10.0,+16.3] | +12.2 [+9.1,+15.2] | .7680 | .8965 | .8872 |
fast - v5: phase -0.23 [-0.33,-0.11] E; function -0.32 [-0.46,-0.19] E (= note 95b). R deltas: phase v5-beta +0.14
[-0.01,+0.26], fast-beta -0.08 [-0.18,-0.00]; function +5.02 / +4.70. Class CIs in the json (R n 77,905).
## v5 per stage (all v5 rows >= 30 min, E / R; TCN covers 99.9 % phase / 100 % function rows)
| phase (189,277 E) | trees alone | TCN alone | average 0.5/0.5 | trees + decoder | decider (trees+TCN+decoder) | fast le2h |
|---|---|---|---|---|---|---|
| E | .9679 | .9738 | .9808 | .9783 | .9843 [.9814,.9869] | .9822 |
| R | .9703 | .9756 | .9828 | .9805 | .9863 | .9843 |
Decider - average +0.34 [+0.26,+0.43]; average - trees +1.29 [+1.09,+1.48] (trees alone = ranker argmax, no decoder).
| function (126,505 E) | trees alone | TCN (siba 3-seed) alone | fixed 0.6/0.4 | decider (stacker) | after lane rule |
|---|---|---|---|---|---|
| E | .9140 | .9047 | .9247 | .9307 | .9305 [.9214,.9388] |
| R | .9238 | .9146 | .9346 | .9395 | .9395 |
Stages before the lane rule = argmax with stack credit; lane rule = gate .9 D-lane decode + stack pick (adds ~0 at >= 30 min).
## Lanes / setback (no v5 OOF: note 95 held the lanes / pick / twin OOF chain fixed) — latest measured
* Lanes D (note-77 OOF): n_lanes exact .844 per sample on v4q print lanes (note 99; m30 .813 .. full66 .884); Sept scope
  m30 + full .854 (within 1 .991), detector lane set exact .922 (note 77/85). Beta has no lane output.
* Setback (note 77, printed Advance): median error 25.0 / 22.5 / 21.7 / 20.4 ft, within 50 ft 69.1 / 68.5 / 70.4 / 70.2 % at
  m30 / h6 / h24 / full66. Beta has no setback output.
## Speed / peak RAM (f98/bench run files, bench98 protocol; beta 4 passes, v5 2, v5 fast 1; warm s (peak MB))
| case | beta 30m | v5 30m | fast 30m | beta 3h | v5 3h | fast 3h | beta 24h | v5 24h | fast 24h |
|---|---|---|---|---|---|---|---|---|---|
| typical r8 22 ch | 1.02 (834) | 0.93 (682) | 0.92 (685) | 0.47 (266) | 2.12 (668) | 0.68 (304) | 0.84 (346) | 2.51 (606) | 1.35 (390) |
| typical r11 18 ch | 0.75 (582) | 0.75 (506) | 0.74 (504) | 0.40 (259) | 1.62 (526) | 0.57 (292) | 0.59 (286) | 1.80 (563) | 0.89 (334) |
| busiest events 31 ch | 1.46 (1338) | 1.18 (738) | 1.10 (730) | 0.57 (296) | 2.27 (637) | 0.94 (322) | 1.19 (390) | 3.47 (685) | 2.21 (455) |
| busiest channels 43 ch | 1.83 (1560) | 1.33 (746) | 1.27 (739) | 0.53 (277) | 2.81 (756) | 0.94 (308) | 1.05 (358) | 4.05 (883) | 1.94 (409) |
## Health (note-93 health rows = v4f package health_core on the same Sept-2026 windows, joined to v5 rows; still valid)
| status (E rows) | ok 110,302 | not_enough_data 14,052 | suspect 1,402 | bad 749 | flagged (s+b) 2,151 |
|---|---|---|---|---|---|
| v5 function E | 93.49 [92.60,94.32] | 90.78 | 90.80 [87.65,93.42] | 75.17 [67.93,83.10] | 85.36 [81.05,89.12] |
| v5 fast E | 93.18 | 90.67 | 89.66 | 74.10 | |
R: ok 94.41, ned 91.56, suspect 91.91, bad 75.63. Same picture as note 93 on v4f (bad 75.0 vs ok 92.4).
## Caveats
* Paired set = STG + REL signals only; on it every model is ~0.3-0.6 pt higher than on all v5 rows.
* Beta function = research OOF of its head (no lane decode), not the shipped booster (note 64: similar gap).
* Health statuses come from v4f-era inputs (phase / function / lanes); events dominate, so the split is still meaningful.
