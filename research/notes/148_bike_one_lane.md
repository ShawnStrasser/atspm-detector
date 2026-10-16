# 148 Bike zones cover one lane (2026-10-15, CPU, package fix)

User (2026-10-15): bike detectors never span lanes; never more than one bike lane (two bike zones on the
same lane can be tied together, never two lanes).

Cause. The lane decode (lanes.py) runs on the TREES' function (func_t), before the stacker / per-lane
decode settle the final function. A zone decoded as Other / Mid (may span) or as an anchor A / P / C / YR
(may make a lane) that ends up Bike kept its decoded lane set. The public `lanes` column was already
blank for Bike (pipeline.post_outputs), but n_lanes_spanned went unfiltered into health (span "2+" type,
too-fast gap per lane) and the phase table kept the lane the bike had made (phase_n_lanes one too many).

Fix (src/atspm_detector, after the function decision, so phase / function cannot move):
* lanes.bike_one_lane, called from function_stage for the final-Bike detectors of a signal:
  - a phase where some lane's single-lane anchors are all bikes is decoded again without its bikes, from
    the same pair probabilities (decode_groups; other phases untouched);
  - each bike gets ONE lane: the lane with the largest sum of pair log-odds "same lane" (+ beta, the
    decoder's own non-anchor placement rule) over the phase's non-bike detectors on it, among the lanes the
    decode gave it (all lanes of a re-decoded phase); ties -> busier lane. Bike-only phase -> 1 lane.
* pipeline.post_outputs: health span of a final Bike = 1 (guard). Public `lanes` stays blank for Bike.

Numbers (package, h24_b 2026-09-28 00-24 h, all 755 signals, 14,937 detectors; locked absent; harness
research/code/lanes148/run148.py + cmp148.py, outputs dc_work/s148/base, new):
* package Bike calls 659; 69 had a decoded lane set, 5 spanning 2 lanes -> health span 2+ -> now 0.
  (out144_h24b.parquet, research classifier inputs, showed 10 of 596.)
* 22 of 4,083 phases re-decoded (a bike had made a lane): 21 lose one lane (3->2: 15, 2->1: 6), 1 keeps
  its count; n_detectors -1 on all 22. Non-bike detectors: lanes string changed 7, phase_n_lanes 68,
  lane_conf 68. Phase, function, setback, night speed, health status / score / reason: 0 changes.
* Bundled sample: no change (atspm-detector-check 7/7 PASS, no re-freeze). pytest 18 passed, 4 skipped
  (new tests/test_lanes_bike.py, 3 tests).
* Health limits for the "Bike 2+" type are now unreachable (health limits files not touched: note 147 agent).
