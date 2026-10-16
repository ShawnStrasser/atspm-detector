# 147e User-approved changes on top of 147d (2026-10-15)
Research only (src/ untouched). 147d kept where not listed (Old-prior allocation R_s 797 / R_b 10 not re-forced;
stuck-on bad limits; Bike = 1 lane). Code research/code/health147/h147e_feat.py (per-signal features from the
h24_b events, 2 min), h147e_alloc.py (ModelE, scoring all_scores_e), h147e_sheet.py. Work dc_work/s147:
feat147e.parquet, holds147e.parquet, det147e.parquet, knobs147e.json, prop147e.csv, flagged147e.csv,
stuck147e.csv. review/thresholds_review_v12.xlsx rebuilt in place (backup review/_backup/thresholds_review_v12_147d.xlsx).
## Changes
1 Too fast pairs: short gap = ON-to-ON < 0.8 s (stop-bar) / 0.6 s, / lanes; pair_share = share of short gaps
  whose next gap is not short; >= .9 -> reason "counts each vehicle twice". 86 % of detectors with short gaps
  are >= .9; of the 26 too-fast flags only 1 (the rest chatter in runs). Text only.
2 Holds >= 300 s overlapping other detectors' holds (1,800): a common failure (all start AND end within 5 s) 228
  -> not excused, score 1 (bad) once >= the suspect limit; 94 detectors at 35 signals; signal note "N detectors
  held ON together" N = 2 (26 signals), 3 (2), 4 (5), 6 (1), 10 (1). b queue (q1 / q1b) 665 -> excused.
  c unclear 907 -> suspect at most (existing shared cap).
3 Bike (596): no goes silent / count drops / night misses.
4 Count drops: occupancy confirms when own occupancy after / before < 0.8 x the yardstick mates' same ratio;
  otherwise excused (occupancy moved with the mates = congestion). GEH 10 / 15 for every type. Of 244 non-Bike
  detectors at GEH >= 10: 139 confirmed, 105 excused. 04022 det 49 (GEH 14.1, own occ x1.2 vs mates x5.9) -> bad.
5 Night misses: GEH 8 / 12. A night silence (silent run >= 80 % at night) on a detector expecting < 50 vehicles
  at night is capped at score .5 (suspect at most): 3 detectors.
6 Busy at night: any finding needs the phase-mate delta >= .30 (D_PH), not only bad.
7 Too fast 2+ lanes: limit = the 1-lane limit (New per-1,000) of the same function (two lanes give real close
  ONs); bad 1.27x. 10052 det 42 (Yellow_Red 2+, 42 per 1,000 vs 192) -> ok. No 2+ lane flags left.
8 Too many 2,100 / 2,520 veh/h/lane. 9 Erratic removed.
10 Undercounts all day (new, status family 'quiet'): own hourly counts vs the best-correlated same-phase ATSPM
  mate (Advance / Presence / Count, corr >= .8, mate x own lanes / mate lanes); measure = median signed hourly
  GEH 07-19 (+ = under); only function probability >= .7. Data (4,823 eligible): p95 9.9, p99 20.4, p99.5 24.6;
  GEH >= 10 would flag 1.6 % of all detectors (mates are not truly same-lane), so suspect 15, bad 25 (0.59 / 0.16 %
  by the check). 04023 det 24 (Count, f_top .52) skipped by the .7 rule; its best-correlated mate (det 38) counts
  less than it (ratio 1.5), so it would not flag anyway.
## Final status % (ok / suspect / bad / not enough data)
Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 | Proposed 94.02 / 2.13 / 2.19 / 1.66 (4.3 %).
Flagged at final status (suspect + bad) Old / New / Proposed: stuck 1.93 / .52 / 2.34; counts fall .67 / .22 / .67;
night misses .62 / .34 / .30; busy at night .91 / .81 / .64; too fast + too many .26 / .17 / .20; erratic .20 /
.27 / 0; undercounts all day - / - / .53.
## Caveats
- Lanes are not known per detector, so 'same lane' = best-correlated same-phase ATSPM mate; noisy (median |GEH| 4).
- All features research-side; the stuck-on allocation curves were re-read with the new scoring, knobs unchanged.
