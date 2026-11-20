# 159c — Shockwave features: retrain on the shipped recipe, cost in the package path, lanes (study only, 2026-11-16)
User questions after 159b: can the model be retrained with it, is it worth shipping, what does it cost in compute /
size, and does it help the lane count. Nothing in `src/` changed; nothing committed. Code `research/code/shock159/`:
`sw5_pkgfeat.py` (features as package SQL + SQL timing), `sw6_tiers.py fit|report|stage` (tiers, retrain),
`sw7_bench.py` (end-to-end timing), `sw8_lanes.py` (lane screen). Outputs `%DC_WORK%/s159/` (pkgfeat, tiers_oof,
tiers_results.json, sw6.log, staged/, lane_pairs, sw8.log).
**Features the way the package would compute them.** SQL on `setback.py`'s own `sb_onc` (package SQL_ONC run as is),
held event = ON that began outside green, > 6 s before the next green, still ON at it, released < 60 s after. Shares
and a capped count instead of raw counts (the "full" group can be longer than 66 h in production). Tiers by cost:
* A (12): share of greens held, capped count, release median / q10 / q25 / q75 / IQR, follower share, 3 s and 10 s
  hold variants (share + median). One GROUP BY, same shape as the existing `rel` step.
* B (3): queue count before the held ON (median, q75), follower ON time. One window pass over `sb_onc`.
* C (4): share / median / q25 of same-green release minus the first predicted stop-bar zone released, that zone's
  own release. One self-join. (The "any detector" anchor of 159b has no source in `sb_onc`: dropped. Mates here = the
  note-57 OOF function arm; the note-95 per-detector function of non-targets is not on disk.)
Package-shaped vs 159b research extraction, release median: corr .998, median difference 0.00 s.
**Frame = the shipped model's own training table** (note 95: `final_v3_work/f95/sb7/feat.parquet`, 127 signals, labels
v4q; recipe `fit75.stage_setback`; the staged refit reproduces the shipped `L_fit` and `n_train` of all four groups
exactly). OOF six folds, 15 single seeds, permuted control. Printed Advance 5,551 detector-windows, Mid 973.
| pooled over lengths, single seed | base | A0 (4) | A (12) | **A+C (16)** | A+B+C (19) | permuted A+C |
|---|---|---|---|---|---|---|
| Advance medAE ft | 23.2 | 22.8 | 22.5 | **22.4** | 22.2 | 23.3 |
| Advance within 50 ft % | 69.2 | 69.9 | 70.3 | **70.5** | 70.8 | 69.3 |
| Mid medAE ft | 24.1 | 19.8 | 18.1 | **18.2** | 17.4 | 24.5 |
| Mid within 25 ft % | 51.4 | 60.1 | 62.5 | **62.6** | 63.7 | 50.8 |
Signal bootstrap, A+C - base (15-seed mean): Advance within-50 **+1.6 [+0.5, +2.9]**, medAE -0.5 [-2.1, +0.6] (not
clear), capped mean error -1.2 [-2.1, -0.5]; Advance > 200 ft medAE **-4.8 [-7.6, -1.3]**, within-50 +3.9 [+1.6, +6.5];
<= 100 ft unchanged; loop within-50 +1.5 [+0.4, +2.8]; video nothing; radar within-50 +4.4 [+1.3, +8.2] (39 -> 42 %,
still poor). **Mid medAE -6.6 [-9.3, -2.6]**, within-25 +12.2 [+6.6, +18.6]. By length (A+C, base -> new): Advance
within-50 68.5 -> 69.4 / 68.6 -> 70.3 / 70.8 -> 71.5 / 70.4 -> 73.5 (m30 / h6 / h24 / full); Mid medAE 23.3 -> 20.5 /
24.7 -> 18.6 / 26.4 -> 16.6 / 22.6 -> 15.0. Production path (predicted Advance / Mid) within-50 +1.7 [+0.6, +2.9].
**Regressions re-checked.** Advance loops over 2+ lanes (18 detectors): A+C 16.8 -> 17.7 ft, within-50 80.3 -> 80.2
(CIs cover 0) — fine; A+B+C 16.8 -> 22.7 ft, within-25 60.5 -> 52.5: the queue-count tier was the cause in 159b. No
held cycle: Advance A+C +0.9 ft [-0.7, +2.1], within-50 -0.3 [-1.1, +0.6] (noise); Mid within-25 57.9 -> 56.0
(-3.7 [-7.1, -0.6], 242 detector-windows) — small, real, the one cost. Mid over 2+ lanes -6.1 [-8.6, -2.2].
**Cost.** SQL alone, 127 signals x 24 h in one call: existing SQL_ONC 4.3-7.5 s + SQL_BEH 0.4-0.8 s; A 0.04-0.08 s,
C 0.08-0.10 s, B 2.4-3.4 s. Real package path (`predict` end to end, warm, `setback.features` wrapped to run the extra
SQL on the package's `sb_onc`; 24-h logs; median of 7 / 5 runs):
| | none | A | A+C | A+B+C |
|---|---|---|---|---|
| 1 signal, wall s (run-to-run 1.67-1.74) | 1.692 | 1.705 | 1.718 | 1.772 |
| ... extra SQL | - | 6 ms | **10 ms (0.6 %)** | 52 ms (3.1 %) |
| 10 signals, wall s (5.67-6.04) | 5.879 | 5.927 | 5.956 | 6.082 |
| ... extra SQL | - | 9 ms | **17 ms (0.3 %)** | 123 ms (2.1 %) |
Model size: same 300 trees x 15 leaves per group; LightGBM text 460-469 KB vs 459-468 KB shipped (ONNX ~144 KB each,
unchanged); 59 -> 75 input columns; no new model file, no new dependency.
**A or A+C?** A+C - A, signal bootstrap: Advance within-50 +0.4 [-0.1, +1.0], > 200 ft medAE -1.3 [-2.7, +0.6] (14 / 15 seeds better), Mid -0.2 [-1.2, +1.2]: nothing clears 0, and C leans on other detectors' predicted function. **Tier A (12) is the pick.** A alone: Advance within-50 +1.2 [+0.1, +2.4], > 200 ft -3.5 ft [-6.4, -0.7], Mid -6.4 ft [-9.3, -2.4], within-25 +11.7 [+6.4, +17.7].
**Retrain (staged, not shipped).** `sw6_tiers.py stage A` -> `%DC_WORK%/s159/staged/setback_{m30,h6,h24,full}_q50_s0.txt` + json (seed 0, full data, shipped recipe + 12 columns; LightGBM text, ONNX export + parity still to do). P10 / P90 band untouched.
**Lanes: no.** Screen on note 58's lane pair table (77,102 labelled pair-windows, 485 signals, locked absent, join
coverage asserted): 10 pair cues from greens both detectors were held at (co-held share, phi, same-green release
difference / spread / correlation, hold difference), stacked on logit of model D's OOF P(same lane). AUC on 2+ lane
phases: stack without cues .9709, with cues .9710, permuted .9708; difference +0.0001 [-0.0003, +0.0004] (note 58 seed
spread .0008); pair accuracy 92.49 -> 92.52. The cues alone reach AUC .67 (.84 where co-held >= 5 greens): real signal,
but model D already has it. Not pursued.
**Verdict.** Ship tier A at the next release if the user agrees: Mid 24 -> 18 ft (clear; 109 of 737 detectors here), Advance beyond 200 ft 54 -> 51 ft (255 of 628) and within-50 +1.2 pt overall (small, clear), ~6 ms per signal-day, no size change. Advance alone would not justify a release; Mid plus zero cost does. B dropped (cost, hurts multi-lane loops); C dropped (no proven gain over A). Build = one SQL block in `setback.features`, 12 names in `setback_p50.json`, four ONNX re-exports + parity, check --freeze, tests, CHANGELOG (distance_ft changes for Advance / Mid -> minor version). Caveats as 159b: mostly-weekend pull; Mid sits at 110 / 160 ft so part of its gain may be the agency standard. WAITING on the user's go-ahead.
