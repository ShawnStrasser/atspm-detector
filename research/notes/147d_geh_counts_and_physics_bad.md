# 147d GEH for count checks, too many 1,980 / 2,520, too fast bad 1.27x (2026-10-15)
User decisions 2026-10-15. Research only (src/ untouched). v12 machinery of 147c (Old-prior allocation alpha 3x
excess, user stuck-on bad limits, Presence / Other bad 60, Bike = 1 lane). Code research/code/health147/
h147d_geh.py (hourly GEH from the h24_b events, 1 m 55 s, 8 procs), h147d_alloc.py, h147d_sheet.py; h147_lib.
all_scores2(geh=True). Work dc_work/s147: geh147d.parquet, grid147d.csv, knobs147d.json, prop147d.csv,
stuck147d.csv, flagged147d.csv. review/thresholds_review_v12.xlsx rebuilt in place (backup of the 147c version:
review/_backup/thresholds_review_v12_147c.xlsx).
## GEH measures (hourly, clock hours fully covered)
GEH = sqrt(2 (M - C)^2 / (M + C)); M = observed hourly count; C = share x yardstick hourly count. Yardstick = the
package's night one (tracked sibling, corr >= .7; else phase mates >= 2; else rest of signal). Share = median M/R
over day hours 07-19 with R >= 30 (>= 4 h, else totals). Signed: + when M < C.
- Count drops: drop hour = largest step up in signed GEH (two-sample, >= 3 h each side); share re-estimated
  before it; measure = median signed GEH of the hours after (only if > 0).
- Misses at night: signed GEH of the average night hour 21-05 (mean M vs mean C, >= 4 night hours).
- Erratic: number of hours with |GEH| >= 10 (either direction; hours with M + C >= 20). No 15-min scaling needed.
Data (all detectors): night GEH p99 4.1, >= 5 0.51 %, >= 10 0.05 %; drops >= 5 5.1 %, >= 10 1.7 % (high volume
makes small share changes large GEH); erratic hours >= 1: 5.5 %, >= 3: 1.3 %, >= 6: 0.5 %.
## Limits
- Too many: suspect 1,980 (1.1 x 1,800), bad 2,520 (1.4 x 1,800) veh/h/lane, hard. Too fast: New per-1,000 cuts,
  bad = 1.4/1.1 = 1.27 x suspect (the generic 2x rule rejected by the user). Erratic (fixed): 3 / 6 hours GEH >= 10.
- Drops and night through the allocation on the GEH measure, suspect never below 5 (rule of thumb: < 5 good),
  bad = max(10, suspect + 5). Night lands at 5 / 10 for almost every type (Mid 1 5.6, Advance 2+ 14.7); drops
  suspect 5-29 by type (Advance 1 12.6, Presence 1 15.3, Count 1 7.1: busy detectors reach GEH 5 on small changes).
- Stuck on (min, suspect / bad): Advance 11.6 / 30 (2+ 15.7), Presence 11.6 / 60 (2+ 15.7), Count 5.7 / 20,
  Yellow_Red 5.5 / 20 (2+ 15.7), Mid 11.6 / 35 (2+ 15.7), Bike 5.5 / 10, Other 14.7 / 60 (2+ 15.7).
## Result (R_s 797, R_b 10; 5 % was a starting point, not forced)
Final status ok / suspect / bad / n.e.d.: Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 |
Proposed 94.14 / 2.15 / 2.07 / 1.65 (4.2 % flagged). More flags only by raising busy at night and drops (bad grows
faster than suspect: 5.0 % total gives 2.24 / 2.78).
Flagged at final status (suspect + bad) Old / New / Proposed: stuck 1.93 / .52 / 2.14; counts fall .67 / .22 / .65;
night misses .62 / .35 / .54; busy at night .91 / .81 / 1.05; too fast + too many .26 / .17 / .26; erratic .19 /
.27 / .13.
## Caveats
- The prior is the Old flag mix on the Old (ratio) measures; the excess evidence for drops / night is the GEH fit.
- Yardstick here ignores the package's pass-1 'bad' exclusion; GEH measures are research-side only (src unchanged).
- Small cells (2+ lanes) still move on 1-2 detectors.
