# 85 — Monday headline table: GitHub beta (final_v2) vs candidate v4b (2026-10-04) — nothing trained, locked untouched
Code `research/code/final84/monday85.py` (scoring, saved OOF only), `bench_beta85.py` (beta speed); output
`%DC_WORK%/final_v3_work/monday_table.json` (+ `monday_table_core.json`, `f84/bench_beta/`). CPU 4 threads, no GPU.
## Arms and scope
* Six folds folds_v4 OOF, samples >= 30 min pooled, paired rows (both arms score the row), signal bootstrap 95 % CI.
* Function truth v4l, ATSPM-only scoring with stack credit; phase truth = official timing.
* beta: phase = note-48 final_v2 OOF (cand64 v2ref); function = note-25 b7 OOF -> 5 classes, argmax, no lane decode (note 64
  reference arm; read the older tree-only phase). Covers 75.3 % of v4b function rows (382 / 652 signals); phase 99.96 %.
* v4b: phase = note-76 'new' arm (order-free trees + GRU .01 filter + decoder = package phase); function = note-84 arm
  mean3_flt0 (v4b on all its rows .9191 E / .9290 R reproduced exactly).
## Accuracy (>= 30 min, paired; delta = v4b - beta, pt)
| row | n E | beta E | v4b E | delta E [CI] | beta R | v4b R | delta R [CI] |
|---|---|---|---|---|---|---|---|
| Phase | 173,817 | .9802 | .9818 | +0.16 [+0.06,+0.26] | .9828 | .9843 | +0.15 [+0.06,+0.26] |
| Function overall | 141,163 | .8761 | .9195 | +4.34 [+3.53,+5.16] | .8855 | .9293 | +4.38 [+3.58,+5.14] |
| Advance | 43,935 | .9113 | .9434 | +3.21 [+1.82,+4.70] | .9181 | .9508 | +3.27 [+1.95,+4.81] |
| Presence | 42,559 | .9391 | .9458 | +0.67 [-0.12,+1.41] | .9536 | .9614 | +0.78 [-0.02,+1.50] |
| Count | 25,880 | .9033 | .9277 | +2.44 [+0.86,+4.09] | .9217 | .9425 | +2.07 [+0.36,+3.88] |
| Yellow_Red | 7,335 | .6416 | .8438 | +20.2 [+15.4,+25.1] | .6416 | .8438 | +20.2 [+15.4,+25.1] |
| Non-ATSPM | 21,454 | .7263 | .8345 | +10.8 [+8.0,+13.6] | .7289 | .8375 | +10.9 [+8.1,+13.7] |
Class rows = rows whose v4l truth is that class (Non-ATSPM: correct = not called an ATSPM class). Function R n 137,895.
## Side outputs (v4b only; final_v2 has no lane or setback output) — REUSED, not re-measured
* Lanes (note 77, f77/laneeval.log; OOF, print-lane truth of that time, Sept scope m30 + full): n_lanes exact .854
  (within 1 .991; multi-lane phases .726), detector lane set exact .922, same-lane pair acc (multi) .924. The v4l lane refit
  (note 83) has no OOF evaluation.
* Setback (note 77, f77/sb_eval.log 'new'; printed Advance, OOF): median error 25.0 / 22.5 / 21.7 / 20.4 ft, within 50 ft
  69.1 / 68.5 / 70.4 / 70.2 % at m30 / h6 / h24 / full66. v4l refit (note 83) not OOF-evaluated.
## Speed / peak RAM (one signal per call, fresh process, 4 threads; warm = median of 3; peak working set MB)
| 3 h warm s (peak MB) | typical r8 22 ch | typical r11 18 ch | busiest events 31 ch | busiest channels 43 ch |
|---|---|---|---|---|
| beta final_v2 (new bench) | 0.47 (263) | 0.37 (252) | 0.55 (279) | 0.51 (279) |
| v4b (note 84) | 2.99 (691) | 2.47 (628) | 3.42 (733) | 4.06 (893) |
Peak over all 12 cases: beta 1,546 MB (busiest 30 min: its GRU runs only <= 120 min), v4b 1,012 MB (busiest 24 h). 24 h
busiest warm: beta 1.04-1.12 s, v4b 4.50-5.41 s. Beta cold at 3 h 1.2-1.5 s, v4b 3.5-5.5 s.
## Caveats
* Function beta is a research OOF of its head on 75 % of rows, not the shipped booster (note 64: shipped-booster OOF gives a
  similar gap, +2.5 vs +2.7 pt on v3s-era truth with the trees-only candidate).
* v4b function OOF nets are v3s-trained and only seed 0 filtered (note 84); production members are full-data v4l.
* Beta is ~6-8x faster at 3 h because it runs no network above 2 h; v4b stays inside the 4-5 s target.
