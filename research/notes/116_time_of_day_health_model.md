# 116 — Health: generalised time-of-day model (user feedback on health_review_v3, Oct 7)
User: group ALL detectors by type, learn each type's normal 24 h from counts AND time ON, spot outliers without labels;
phase comparison optional. Must catch 12032 d6, 04035 d52 / d53, 2B069 (d23, "all"), 2B502 d4; leave 2B531 d35, 2B530
d60, 14003 d4 alone. Data: w40 full days Sun 27 / Mon 28, 760 training signals (locked_v2 asserted absent), 29,609
detector-days, 94 % scorable; stg OOF function / lanes / phase; hi-res log only. Code `research/code/health116/`
(h116_data / study / methods / build, **tod116.py = numpy-only scorer**); work `%DC_WORK%/s116/`. Nothing adopted.
## Method comparison (2-fold by signal, references never from the scored signal; s116/method_table.csv)
Groups = function x lane span x volume band (< 20 / 20-100 / > 100 per h), merged to fn|*|band, fn|span|*, fn if < 150.
| method | must-bad 5 | fine 3 | other bad 12 | OK'd 12 | 2B069 Mon | flags /100 Sun / Mon | Sun-Mon Jaccard |
|---|---|---|---|---|---|---|---|
| hourly Mahalanobis (shares + % ON + diffs) @2/100 | 2 | 0 | 4 | 1 | 2/37 | 1.84 / 2.16 | .30 |
| hourly kNN (k 10) @2/100 | 1 | 0 | 3 | 1 | 0 | 1.84 / 2.17 | .38 |
| hourly k-means (4) nearest centroid @2/100 | 2 | 1 | 5 | 1 | 0 | 1.89 / 2.12 | .32 |
| hourly one-class SVM (nu .05) @2/100 | 3 | 0 | 2 | 2 | 3 | 1.74 / 2.26 | .28 |
| v110 profile (prof_x >= 1) | 2 | 1 | 6 | 1 | 1 | 1.92 / 2.51 | .27 |
| night level, type layer only (z >= 3.5) | 5 | 0 | 2 | 1 | 9 | 2.17 / 2.46 | .40 |
| night level, phase layer only (dph >= .3) | 4 | 2 | 1 | 0 | 1 | 0.62 / 0.76 | .46 |
| **night level, type + phase (chosen)** | **5** | **0** | 2 | **0** | 7 | **0.96 / 1.09** | **.47** |
Also worse or equal (compare.csv): z-RMS (0 / 5), 7 clock blocks, counts-only / %ON-only / 30-min, min or mean of
counts and %ON models, block z-max, min/max contrast, log night ratio (squashes the tail; sqrt kept). Whole-profile distances spread over 48-120 numbers,
so a night that does not go down (one feature at z 3-4) drowns in ordinary noise (04035 d52 = 92nd pct of Other 2+
high) and their flags are day-specific (Jaccard .26-.38). k-means per type (clusters.parquet): main shapes differ by
peak hour (13 vs 16 h) and night level; the only small clusters (<= 2 %) are night-peaked.
## Chosen: night level vs the type, phase as context (tod116.py)
r = sqrt((count/h 01-05 + .5) / (busiest 4-h count/h + .5)). Per group, r predicted from busiest 4-h % ON (saturated
long zones keep day counts flat: min/max 4-h ratio p98 .16-.18 at <= 5 % ON, .47 at > 80 %) and volume (trimmed LS,
one-sided robust z); Presence / Other / Mid add absolute night % ON vs volume (scale floor .05); z calibrated per group
on out-of-fold scores. Phase layer dph = r minus median r of phase mates (same predicted phase, >= 200 / day; population
p99 .28). Finding: night actuations >= 20 AND (z >= 6 OR (z >= 3.5 AND dph >= .3)); bad = z >= 6 AND dph >= .3.
* Cases (Mon, ratio = r^2 vs type expected): 12032 d6 1.00 vs .19 (mates .04) z 7.7 bad; 04035 d52 .84 / .13 z 4.6;
  d53 .44 / .09 z 4.2; 2B502 d4 .60 / .07 z 4.1; 2B069 d23 .76 / .04 z 7.9 (mates equally flat -> type layer). Fine:
  2B531 d35 .66 / .14 z 3.49 (EDGE, 0.6 z under 2B502 d4), 2B530 d60 z 2.8, 14003 d4 z 0.1. Same verdicts on Sun.
  2B069: 7-9 of 37 flagged (low-volume flat ones); its busy detectors have a normal night ("all bad" not reproduced).
* Grid (grid.csv): Z_LO 3.5-4 x Z_HI 5-6 x D .2-.4 all keep 5 / 5 and 0 / 3 (.87-1.49 / 100). The phase layer EXCUSES:
  halves the flags (2.3 -> 1.0 / 100) by clearing detectors whose mates are equally busy at night (real night
  traffic), drops the one OK'd false alarm (08019 d16, no mate); alone it flags 2 of 3 fine cases.
* Day type NOT needed as a group dimension (same-day / pooled / cross-day refs: same catches); final = pooled. 43 % of
  flags are new vs v110 (v110 ok 123 / 286). Per 100 by fn: Presence 1.6, Bike 1.6, Mid 1.5, Other 1.3, Advance .6,
  Count .3, YR .1; video 1.6 vs loop .4 (analysis only).
* For the resolver: silent all day after 04:00 (01074 d3 / d4) puts the busiest 4 h at night -> also flagged here;
  prefer 'goes silent'. >= 3 flags at one signal in a day: 20 signal-days (signal-wide note?).
## Partial samples (production scorer, held-out refs; partial.csv)
Group model rebuilt on the fly from stored reference profiles (<= 1,500 / group) on the same hours, split-half
calibration; needs 01-05 + a whole peak (07-10 or 15-19) + >= 10 h. 00-12: scored 90 %, 1.12 / 100, finds 83 % of the
24-h flags (79 % of its flags are 24-h flags), 5 / 5, fine 0 / 3; 00-10: 85 %, 68 %, 4 / 5. Under 10 h or no night:
not scored (108b: the hourly-shape version found 26 % at 3 h).
## Deliverables (`%DC_WORK%/s116/`): tod116_ref.npz (57 groups: coefficients, scales, calibration, typical day for
charts, reference profiles; 5.9 MB), oof_final, scores_review.csv (36 review signals), plot_ready (own / type band /
phase mates), clusters, method_table, grid, partial, final_summary.json. Scorer runs with sklearn / scipy / pandas /
torch / lightgbm blocked; prod = OOF flags 99.94 %. Caveats: 2 days, one weekend; 8 user cases shaped the rule.
