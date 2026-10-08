# 131 — atspm-detector output as atspm's detector configuration (2026-10-07)
Brief: make the package's output feed the user's `atspm` (PyPI 2.4.0 installed; source 2.5.2 at
a local checkout of `ATSPM_Aggregation_package`, read-only, copied to `%DC_WORK%/s131/atspm_src`). Model = read-only copy
of `final_v7_prod/src` (7.0.0) in `s131/_pkg`. No model change, no locked labels, nothing trained. CPU, minutes.
Files: `%DC_WORK%/s131/atspm_export.py` (exporter), `test_atspm_export.py` (7 tests), logs `e2e_pypi.log`,
`e2e_src.log`, `atspm_selftest.log`, probe `probe_atspm_issues.py`.
## 1. What atspm reads (data_loader.py + queries/*.sql)
`detector_config` = 4 columns by name: `DeviceId` (int or text; DuckDB casts on join), `Phase` (INT16), `Parameter`
(= detector channel, INT16), `Function` (text, exact case-sensitive match). Extra columns ignored. One row per
(detector, phase). No lanes, distance, approach or direction. Only 3 Function values are read:
| Function | atspm measures | our class |
|---|---|---|
| Advance | arrival_on_green, platoon_ratio (2.5.x only) | Advance |
| Presence | split_failures (by approach = OR of a phase's zones, or per detector) | Presence |
| Yellow_Red | yellow_red | Yellow_Red |
| anything else (samples: 'Stopbar Count', 'stop bar count') | none, carried | Count -> 'Stopbar Count'; Mid/Bike/Other not exported by default |
actuations, has_data, timeline, terminations, ped, phase_wait, coordination, detector_health use raw events only.
## 2. Exporter `to_atspm_config(results, ...)` -> DeviceId, Phase, Parameter, Function
Drops (with a reason in the optional report): no answer (< 5 actuations), non-ATSPM class, `health_status` in
`exclude_health` (default "bad"), optional min phase / function probability, and `one_per_lane` (default on): at most
one detector per (signal, phase, Function) per lane, lane-by-lane before spanning, then higher function_prob; no-lane
detectors kept. Options: `include`, `function_names`, `device_id_dtype` (predict returns DeviceId as text even for
integer input), `extra_columns` (Lanes, DistanceFromStopBar ft, PhaseProb, FunctionProb, HealthStatus; atspm ignores
them), `return_report`. Plus `atspm_coverage(config)`: which measures each phase gets. numpy/pandas only.
## 3. End to end (test_atspm_export.py): 7/7 pass on atspm 2.4.0 AND on source 2.5.2
Bundled sample (1 signal, 30 min, 17 detectors): 12 rows exported (8 Advance, 4 Presence); dropped det 3 (spans the
lanes of Advance 2 and 5), det 6 (1 actuation), 18/19 Mid, 20 Bike. AOG phases 2/4/5/6, split failures 2/4/5, no
Yellow_Red on this signal (yellow_red empty, no error). Checks: atspm AOG = by-hand count exactly (P2 348/413, P4 43/83,
P5 8/38, P6 364/595); occupancies in [0,1]; atspm `actuations` per detector = predict `n_actuations` for all 17;
platoon_ratio runs (2.5.2). Extra columns, empty config, text DeviceId vs integer events: all run.
atspm's own samples vs their shipped configs (sanity only; their DeviceIds cannot be mapped to our signal list, so
overlap with locked signals is unknown — no accuracy claim): 1136 (2 h): 13 of 14 shipped A/P/YR rows same phase +
function (det 25 Presence -> Other); 227/452/454 (3 h): 37 of 49. Of those 12: 4 Yellow_Red called Count (227 det 42 =
twin of det 31, identical counts; 452 det 41/42/47), 1 Presence called Count (454 det 11), 2 Mid (227 det 4, 18), 2
dropped by one_per_lane (454 det 50/52, spanning advance zones the agency used instead of the lane loops), 1 Other
(452 det 13), 1 no answer (454 det 41, 2 act.), 1 Yellow_Red on phase 1 vs 6 (454 det 46).
Measures, shipped vs ours: AOG same on 7 of 12 phases, max gap 0.08 (454 P2 .870/.792); red occupancy same on 17 of 19
shared phases (1136 P8 .047/.052, 454 P2 .902/.957); split failures per bin differ on 1136 P8 (.125/0) and 454 P2
(0.17/5.4: det 10 vs 11 called Presence).
## 4. Gaps (atspm needs what we cannot give) — none blocking
- Approach / direction / movement: atspm does not use them. (UDOT ATSPM would; least-bad default there: NEMA
  convention, odd = left, even = through, not inferable from the log.)
- Overlap-assigned detectors: atspm config is phase-only too; we output phase only (accepted limit).
- Yellow_Red where the signal has none: no fallback to Count (no speed filter -> stopped cars counted as red arrivals).
## 5. atspm-side issues for the user (all reproduced, probe_atspm_issues.py, both versions)
1. yellow_red ignores `bin_size`: hard-coded `time_bucket(interval '15 minutes')` (bin_size 60 -> 15-min rows).
2. arrival_on_green inner-joins green and total counts: a bin with actuations but none on green is dropped instead of
   0 % (1-min bins, 3 test signals: 253 of 1,404 bins missing; 0 of 96 at 15 min).
3. Function match is exact and case-sensitive with no warning ('advance' -> AOG empty); allowed values undocumented;
   the two samples spell Count differently.
4. Duplicate config rows double-count (one Advance row repeated: AOG total 21,187 vs 18,962). Exporter dedupes.
5. tests/test_timeline_data_gap_incremental.py (untracked in git) imports installed `atspm`, not `src.atspm`: 2 of 99
   fail against PyPI 2.4.0; with the import fixed (in the s131 copy) 99/99 pass on 2.5.2.
6. AOG counts arrivals at the advance loop, not the stop bar (README "future plans"): our `distance_ft` (+ night speed
   when the sample has nights) could feed optional DistanceFromStopBar / Speed columns.
## 6. Model-side observation (not changed)
predict() output still holds same-phase, same-class, same-lane duplicates on 5 of 144 detectors over these 5 signals
(sample det 3, 1136 det 24, 452 det 24, 454 det 50/52) although the decoder enforces one per lane — worth a look; the
exporter removes them. Proposed packaging: `atspm_detector.to_atspm_config` exported from `__init__` and a CLI
`--out-atspm config.csv`; needs nothing beyond numpy/pandas.
