# 157 Busy at night vs long zones (2026-11-09)

Trigger: a signal timer (Region 5, US 395) reported "busy at night" on long phase-call zones (one input
spanning the approach, SDLC 1-8 / 9-12). Window = the Nov 9 report's Nov 6-8; health re-run with the report's
own path (its detector-health runner) -> 261 busy-at-night rows = 86 detectors.
Code: research/code/busy157 (extract.py, simulate.py, build_review.py); sample data %DC_WORK%/tmp/busy157.

## Finding
* The count ratio (night / busiest 4 h) is invalid on a zone that is ON much of the day: vehicles merge
  into long ONs, so daytime counts are capped and the ratio inflates. Affects long zones AND stop-bar
  presence zones holding a queue. Night ONs on these zones match vehicles on the phase mates (+-8 s).
* 12065 det 6, 12055 det 2 (user's examples): this cap; false positives. The 12065 zones (inputs 1-8)
  give 0.1-s blips by day and night alike (sensor property, not a night fault).
* 12041 det 47: Advance, occupancy < 10 %, a 46-in-15-min burst at 2 am vs ~3 usual -> true positive.

## Proposed gate (simulated only, NOT in the package)
* sat = busiest-4h mean occupancy (b4o in tod.sql) >= 0.25.
* For sat zones flag only if sqrt(night occ / b4o) - mate_r >= 0.30 (mate_r = the count-based mate ratio
  already used, sqrt space) OR night ONs >= 50 % 0.1-s blips while daytime ONs < 10 % blips.
* Gate only removes flags; function-agnostic (physical: occupancy), so it works whatever the model calls the zone.
* Nov 6-8: clears 38 / 86 detectors, 30 of them have no other finding. Sensitivity: b4o 0.30 -> 34,
  0.20 -> 41. The occupancy z of the type model was not re-derived; the gate uses the mate delta only.
* "No daily pattern" checked too: of 39, 14 are saturating zones and most have night occupancy near the
  day level (real) -> no change proposed.

## Review
review/busy_at_night_review.xlsx (10 rows, charts in review/busy157_charts). Pending user OK; then
implement in hq/sql/judge/tod.sql (tdl) + verdict.sql (pv), add limits to health_limits.json, re-freeze.
