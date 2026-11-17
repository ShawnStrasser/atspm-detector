# 157c Busy at night: occupancy fix, closer look at the unsure rows (2026-11-10)

User verdicts on review/busy_at_night_review.xlsx: agree on 12065 d6, 12055 d2, 2C070 d26, 03057 d24, 12041 d47,
2B337 d4, 2B368 d13 (tentative); unsure / disagree on 10004 d2, 2B049 d51, 2B098 d24. Asked for data through Nov 9.
Code: research/code/busy157 (occ.py, rebuild.py, fetch_day.py, look_*.py, build_review2.py).

## Occupancy bug (research only)
* 157/157b binned each ON's full duration into its start bin -> > 100 % bins. occ.py splits ON->OFF intervals across
  bins and drops intervals touching a bin with no event at the signal. The package was already right (split5, unk_ns).
* Re-run on fixed bins: v2 clears 28 / 87 (24 with no other finding); every reviewed row keeps its outcome.

## Data
* Nov 9 pulled once for the 4 signals (one query).
* The report's local raw store misses night hours: Nov 4 19-24, Nov 5 00-03, Nov 6 01-03 + 23, Nov 7 00-03 (empty
  0000_0215 / 0000_0300 files). Busy at night needs 24 covered hours, so only Nov 8 was judged in the Nov 6-8 window.
  Overlapping fetch windows store events twice; the package de-duplicates.

## Findings
* 10004 d2 + d4 (advance, lanes 1 / 2): 1-Hz ON/OFF in lockstep at night (each a Ph2 call), held ON for hours by day,
  worse daily (3.4k -> 14k ONs/day Nov 4 -> 8). "Misses vehicles at night" came from det 4 as yardstick:
  clean_yardsticks keeps two non-traffic partners as each other's yardstick. Too fast missed it (limit 0.6 s / lanes;
  gaps ~0.5 s; det 4 labelled 2 lanes -> 0.3 s). v2 cleared it (its evening is broken too).
* 2B049 d51: flat ~640/h day and night, occ ~33 %. No daily pattern blocked by mate d52 (flat ~4,000/h, itself
  "no daily pattern"): mates' rest_n 4,452 vs rest_d 7,930. Without d52: 372 vs 3,962, own_n / own_d 1.05,
  prof_corr -0.38 -> no daily pattern.
* 2B098 d24: night 0.1-s pulses every ~13 s, 90 % in red, 32 % matched (+-8 s) by a Ph8 mate; 1 of 284 night greens
  from d24 alone (Nov 8-9 01-05). 2B368 d13: 8 % matched; 33 of 132 Ph5 night greens from d13 alone.

## Proposed (pending user, review/busy_at_night_review_2.xlsx)
1. v3 = v2 + no long-zone excuse for a detector with a stuck-on finding: clears 25 / 87 (24 with no other finding);
   restores 10004 d2 / d4, 06002 d22.
2. Night drop (misses vehicles at night) not judged when its yardstick does not look like traffic.
3. No daily pattern: mates' day / night levels without detectors that do not look like traffic (ntraf).
4. Triage (the report): phantom greens (greens only this detector asked for) as the urgency evidence for false calls.
Network effect of 2-3 not measured yet (needs the package run).
