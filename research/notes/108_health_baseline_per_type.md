# 108 — Health: what NORMAL looks like per detector type, then per-type limits + time-of-day profile (2026-10-06)
User: limits were set before knowing what normal looks like; break detectors down by type first. Corrections: (a) at high
traffic a zone's time ON can keep rising while its counts fall (queue) - test it as a measured pattern; (b) lane span is
where a different rapid-firing limit belongs. Later: no "long zone" type anywhere; types = model function x lane span (+
volume band); time-of-day profile check (new idea); spot-check sheet. Data = note-96 w40 (Sat 26 - Mon 28 Sep, 763 training
signals, locked_v2 asserted absent, 115,518 detector-windows with a function); hi-res log + classifier outputs only (technology
joined for analysis only). CPU <= 6 procs. Code `research/code/health/h108_{events,base,profile,resolve,answers,review}.py`,
work `%DC_WORK%/health108/` (act, base, prof, resolved108_q*, pct_table.csv, limits108_q*.csv, plots/type_*.png - 14 types).
## Types and "presumed healthy"
Model: function (7 classes) x lanes spanned (1 / 2+). Log: volume band (< 20 / 20-100 / > 100 ONs per hour); inside Count only,
ON mode measured on clean ONs (pulse = median <= 0.25 s): 2,760 pulse / 359 normal (24 h); radar 2,721 / 244, loop 5 / 31.
Healthy for statistic k = no finding from any check OUTSIDE k's family, >= 20 ONs (leave-own-check-out: a limit is not cut
off at the old limit of the same check).
## Normal per type (24 h, healthy; p50 [p99.5]; full table pct_table.csv)
| type | ONs/h | median ON s | mean % ON | erratic chop15 | ON->ON < 1 s | re-ON < .3 s | max / 5 min | night/day vs signal |
|---|---|---|---|---|---|---|---|---|
| Advance 1 / 2+ | 99 / 261 | 1.0 / 0.2 | 3.9 | 1.09 [6.2] / .67 [4.8] | .001 [.13] / .056 [.20] | [.18] / [.11] | 29 [105] / 57 [130] | .44 [2.5] / .63 [4.4] |
| Count 1 / 2+ | 70 / 188 | 0.2 | 0.5 | .90 [5.1] / .47 [2.4] | .004 [.10] / .023 [.26] | [.07] / [.10] | 23 [93] / 51 [124] | .40 [1.9] |
| Yellow_Red 1 / 2+ | 37 / 233 | 0.2 | 0.4 | .68 [4.1] / .42 [2.9] | .021 [.23] / .057 [.35] | [.14] / [.15] | 15 [70] / 59 [145] | .28 [2.6] |
| Mid 1 / 2+ | 216 / 318 | 0.5 | 7.8 | 1.36 [6.7] / .20 [3.6] | [.13] / [.17] | [.13] / [.09] | 54 [122] / 71 [137] | .50 [1.8] |
| Presence 1 / 2+ | 44 / 96 | 2.9 / 2.1 | 15.5 | 1.02 [5.8] / .83 [4.1] | [.12] / [.14] | [.18] / [.19] | 16 [83] / 28 [87] | .47 [4.9] / .70 [4.4] |
| Other 1 / 2+ | 38 / 133 | 3.8 / 3.0 | 11.1 | 1.10 [6.6] / 1.02 [4.9] | [.12] / [.13] | [.16] / [.13] | 14 [100] / 33 [123] | .46 [3.9] / .74 [5.6] |
| Bike 1 | 3.5 | 0.8 | 0.1 | 1.24 [12.2] | [.26] | [.20] | 4 [33] | .38 [7.8] |
* Lane span matters most for ON->ON < 1 s (side-by-side vehicles): 2+ lanes 1.5-2.6x the 1-lane p99.5 (Advance .13 -> .20, Count
  .10 -> .26, YR .23 -> .35); re-ON < 0.3 s after an OFF (chatter) is NOT higher on 2+ lanes except Count. Volume p99.5 is far
  below the old 150 for most types (Presence 83, YR-1 70, Bike 33), above it only for 2+ lane YR / Mid (137-145).
* Counts vs traffic (plots, panel 1-2): at the median, counts keep RISING with phase traffic in every class but flatten at the
  top in Presence / Other (busier-half elasticity counts .86 / .81 vs time ON 1.05 / 1.09; Count .96 vs .54). Counts actually
  FALLING while time ON rises (the user's queue pattern) is measured in 2.8 % Presence-1, 15.5 % Presence-2+, 4.3 % Other-1,
  7.9 % Other-2+, <= 1.6 % Advance / Count / YR / Mid samples (more on radar / video): real, but a minority in this weekend.
* Longest ON p99.5 (24 h): Count-1 4 min, YR 2-4, Advance-2+ 6; Presence / Other / Advance-1 run past 15 (queues + faults).
  "ON logged again without an OFF" is a class + technology habit (24 h p95: loop Other .45, loop Count .22, radar Count 0).
## Rules v108 (h108_resolve docstring); limits = p99.8 of healthy per cell (type [x Count mode] x sample length; fallback lane span)
Per-type limits for erratic counts, chatter, rapid (lane span in the cell), volume, stuck (clip 5-15 min, bad 60), N3 (all
classes now). Profile P1 at p99.5 REPLACES doesn't-follow-traffic, busier-at-night and N5. Q1 = stuck is a queue only with
healthy phase mates ON 1.5x usual + tracking them (R1b dropped: 08CM405 d37 -> suspect). Q2 = erratic / count-drop / N3 cleared
only with the measured queue pattern AND time ON tracking healthy same-class phase mates r >= .8 (like with like; replaces R2 /
R3). Y = no healthy phase mate after R9 -> watch (12032 d41). D1 = >= 70 % of time ON in ON-again-without-OFF chains -> data note
only (2B068 d19, 2B058 d46). Silent / count drops / night drop unchanged. Nothing keys on a "long zone".
## Time-of-day profile (24 h; hourly share of the day vs type + band median, same day; Sun / Mon separate)
Fires 2.10 / 100 (counts 1.26, time ON 1.37); catches 81 % of busier-at-night, 74 % of doesn't-follow, 57 % of N5; 0.69 / 100 new. Misses are borderline (profile 0.80-1.0x its limit); their v108 status 49 ok /
21 flagged of 77. By band low 1.7 / medium 2.3 / high 2.1; video 4.6 vs loop 1.4 / radar 0.9 (analysis only). 3-h version:
1.26 / 100, only 19 % of 24-h profile fires also fire at 3 h -> 24 h only. All 8 user-Y busier-at-night / doesn't-follow rows
(v1 46-50, 52, 54, 55) fire on the profile at p99.5 (row 50 not at p99.8).
## Fire rates, suspect + bad per 100 (package / v104 / v108)
30 min .60 / .47 / .59; 3 h 1.80 / 1.51 / 1.48; 24 h 5.05 / 3.85 / 3.96. Flags vs v104: 3 h +.43 / -.45, 24 h +.84 / -.73 per
100. Alternatives: all limits p99.5 -> 1.10 / 2.55 / 5.02 (per-type p99.5 multiplies fires: 0.5 % per statistic per type);
p99.9 -> .37 / 1.10 / 3.15. Per type 24 h v104 -> v108: Count-1 1.46 -> 2.38, Presence-2+ 2.83 -> 7.55 (n 212), YR-2+ 7.58 ->
2.25, Advance-1 4.52 -> 4.09, Bike-1 5.66 -> 4.37. Note-79 evalset (circular positives, comparison only) 24 h positives 16.67 ->
17.93 %, presumed healthy 2.36 -> 2.50; 3 h 9.45 -> 9.67 / .94 -> .94; 30 min 5.35 -> 4.50 / .26 -> .41.
## User's earlier answers (sanity only, nothing fitted)
v1 Y flagged 36/37 (v104) -> 34/37 (+ watch 36/37); lost: row 11 chatter (27 % vs Advance-1 30-min limit 31 %), row 50 watch.
N 0/3 stays 0. v2 OK'd rows keeping the OK'd result 8 -> 4 of 8: row 8 now watch (profile), row 9 04035 d53 bad (192 / 5 min >
Advance-2+ limit 186; he doubted its night counts), row 10 Mid-2+ 155 / 5 min > 150, row 11 ok (was watch). v2 row 3 'No!'
stays suspect. v3 row 1 (2B334 d13) -> ok: erratic 6.2 < Presence-1 limit 8.8; its time ON tracks healthy mates r=.67, inside
normal (Presence-1 p5 .44) - his "follows neither" is not unusual for its type by this data.
## Review sheet (review/health_review_v3.xlsx, 30 rows; backup review/_backup/health_review_v3_20261006_153355.xlsx)
2-3 examples per v108 flag + Q1 / Q2 / Y / D1 / N1; profile: 2 new-only, 1 both, 2 old-only; plus 11 earlier-reviewed dets.
Carried by (signal, det): row-1 answer + comment verbatim, 9 / 11 earlier comments (04090 d5, 12015 d36 only in the backup).
## For the orchestrator: nothing adopted; quantile (p99.8 + profile p99.5) is a proposal - p99.5 everywhere is +40-65 % flags.
