# 157b Busy at night v2: occupancy + evening reference (2026-11-09)

Follows 157 after the user asked whether occupancy was already used and whether a correlation would do better.
Code: research/code/busy157 (profile_corr.py, evening_ratio.py, v2.py, build_review.py). Simulated only.

## How the shipped rule uses occupancy (tod.sql)
* Type-model z: expected night count share from b0 + b1*asin(sqrt(peak occ)) + b2*log volume (occupancy is a
  covariate, small slope), and for Mid / Other / Presence a second, occupancy z; z = greatest(count z, occ z),
  so occupancy can only ADD a flag.
* Mate test (tod_dph = r - median mate r) is counts only. No saturation excuse (the qzone excuse in
  assemble.sql flatm exists for no-daily-pattern only).

## Profile correlation (flatm's hour-of-day profiles): rejected
* corr(occupancy profile, mates' count profile) is 0.9-1.0 for real faults too (12041 det 47 0.97, 2B098 det 24
  0.98, 12065 det 6 1.00): 20 normal day hours dominate 4 night hours. Measures shape, not the night level.

## Evening reference
* k = (night counts / mates' night counts) / (same for 20-24 h), +1 slack. Same detector at similar low volume,
  so zone length / saturation cancel. Healthy saturating zones: p95 1.65, p99 2.23; > 3.5 for 0.1 %.
* Blind to faults present all day (2B049 det 51 k 1.1): the occupancy test (157) covers those.

## v2 (replaces 157's 0.1-s-blip clause)
* Saturating = busiest-4h occupancy >= 0.25. Keep the flag if A) sqrt(night occ share) - mate r >= 0.30, or
  B) k >= 3.5, or k not computable. Non-saturating zones unchanged.
* Nov 6-8: clears 33 / 86 (28 with no other finding). vs v1: re-keeps 06016 d24, 2B465 d9, 2B368 d13 (phantom
  0.1-s pulses, 4-6 x evening) and 03096 d15; 12065 d6 / 12055 d2 still cleared, 12041 d47 unchanged.

## 12041 det 47 (user question)
* One night only (Nov 8 01-05: 36 / h vs 5 / h on Nov 7). Spiky counts misses it: it needs >= 2 bursts a day
  averaged over the window (here 1 in 3 days). Candidate: report a one-night burst as spiky counts; not done.

Review: review/busy_at_night_review.xlsx (rebuilt, 10 rows, A/B numbers + limits per row).
