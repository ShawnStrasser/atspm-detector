# 160 - travel time to the stop bar exported; the atspm config carries Lanes and TravelTime

2026-11-18, done from the ATSPM_Aggregation_package session (user request: make atspm-detector and atspm work together).
Production change, src only; nothing committed.

## Why
atspm 3.0.1 (arrival_on_green, queue_model=True) reads two optional detector_config columns: `Lanes` (lane numbers as
this package writes them, "1,2"; atspm counts them) and `TravelTime` (seconds a free-flowing car takes from the advance
detector to the stop line). The setback step already measured that time (`tau`, same-lane stop-bar partner) and threw it
away after turning it into a distance.

## What changed
- setback.py: new output `travel_time_s` = tau + edge / v_user where `covered` (same-lane stop-bar partner), rounded to
  0.1 s; NaN for stop-bar zones, uncovered detectors, and where distance_ft / travel_time_s is outside 10-60 mph (a wrong
  partner: 0.6 s over 367 ft).
- pipeline.py: column carried (POST_COLS, EXTRA_COLS after setback_confidence, refused rows NaN).
- check.py: INVARIANT + consistency asserts. References re-frozen: every existing column identical except the m10
  reference's status text ("10 min" -> "9.9 min"), which predates this change.
- atspm_export.py: `Lanes` always exported (no longer only with extra_columns); new `TravelTime` (Advance rows only);
  docstrings now list every column atspm reads.
- tests/test_atspm_export.py: contract adjusted, new test for Lanes / TravelTime.

## Validation (stop-bar truth lanes of the atspm study, raw events of 2026-11-12)
- 319 lanes, 284 with a travel time before the speed check. Median |exported - lane's own free-flow time| 0.1 s
  (fitted line max(d at 35 mph, 2 s + d / 70) 0.57 s); within 0.5 s at 86 % (line 44 %). Under 100 ft only 16 of 40
  lanes get one.
- The 10-60 mph check drops 11 of the 25 times off by more than 1 s and 1 good one.
- atspm queue model on 277 lanes, wrong calls per 1,000 cars counted on green: none 74.1, fitted line 41.2, this export
  40.6 (lanes worse than no model 24 -> 19), the study's own measured times 40.6.

## Checks
atspm-detector-check: all pass after freeze. pytest: see STATUS.

## Not done
Distance and speed models untouched (note 159 is paused on the user). Speed from ON time needs zone lengths; at the
atspm study lanes only 3 of 174 presence zones have a size in the ODOT config, and the slowing physics with assumed
lengths missed the measured time by 1.3-3.3 s (median).
