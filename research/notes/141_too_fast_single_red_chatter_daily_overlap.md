# 141 Too-fast single gaps, chattering in red, daily-pattern overlap - 24-h samples only (2026-10-14)
Orchestrator task. Research only: nothing in src/ touched, no fault events, locked_v2 asserted absent. Sample = h24_b
of note 140 (Mon 2026-09-28 00-24 h, the same 755 training signals, 14,806 detectors with a predicted function).
Code research/code/health141 (h141_events, h141_rerun, h141_analyze [diag], h141_cases); work dc_work/s141.
Checks: events pass N / v4i runs / implausible gaps = parity_h24b 0 differences; engine re-run (h140 rules, + raw
daily-pattern internals) status and findings 14,806 / 14,806 equal to parity_h24b.
Cut rule (1, 2): 99.8th pct of ALL detectors per function x span (1 / 2+); cell < 500 -> same v4i type and span pool
(stop-bar / moving / other), then item 1 all functions of that span, item 2 all stop-bar; floor 1 / 1,000; >= 3
events; suspect at cut, bad at 2x. Package categories = parity_h24b_out (suspect / bad as reported).
## 1 Too-fast, single gaps (every ON-to-ON gap < 0.8 s / lanes stop-bar, 0.6 s / lanes others, per 1,000 ONs)
Base 14,322 detectors >= 50 ONs. Cuts per 1,000 ONs, 1 lane: Advance 188, Presence 197, Count 159, Yellow_Red 192
(stop-bar 1 pool; own cell 318), Mid 170 (moving 1), Bike 208 (other 1), Other 190; 2+ lanes: moving 31.7, stop-bar
36.2 (all 2+ pool), Other / Bike 16.8. The tail is long: 1-in-500 sits at 16-20 % of gaps implausible.
Flagged 37 = 0.26 % (suspect 30 = 0.21 %, bad 7 = 0.05 %); per-cell cuts without pooling 35. By function 0.17-0.24 %,
Bike 0.64 %, Yellow_Red 1.13 % (pooled cut below its own tail).
Of the 37 also flagged: chattering 15 (41 %), v4i runs 15 (41 %), erratic counts 8 (22 %), too many 8 (22 %); any of
these / any category 22 (59 %); new 15 (41 %). Reverse: v4i runs 15 of 20 caught (75 %), chattering 15 of 72 (21 %).
Flicker cases: none caught. 11021 d2 / d3 Advance 33 / 44 per 1,000 (cut 188; 97th / 98th pct), d4 Mid 2+ 0.3;
asclog d16 Presence 27 (93rd pct), integrator d16 Presence 85 (99.1 pct; would need about 1-in-100), d2 1.2.
Plot review/thresholds_all/crit16_too_fast_single.png.
## 2 Chattering in red at stop-bar zones (OFF -> next ON < 3 ticks, both while the predicted phase is red)
Red = phase's last colour event at or before t is 9 / 10 / 11 (end yellow / red clearance / end red clearance) until
the next 1; 1 / 7 green, 8 yellow; unknown before the phase's first colour event. Same-tick order 7, 8, 9, 10, 11, 1.
Rate per 1,000 red ONs (chosen: a share like the package's chatter; per red hour ranks by volume and red length -
it flags 13, only 4 the same). Needs >= 50 red ONs: 6,292 of 7,262 stop-bar detectors; 57 % have >= 1 red re-trigger;
red re-triggers = 17.4 % of all their re-triggers.
Cuts: Count 1 lane 134, Presence 1 lane 284, Count / Presence 2+ and Yellow_Red 266-269 (stop-bar pools).
Flagged 12 = 0.19 % (suspect 12, bad 0); Count 3 (0.20 %), Presence 9 (0.20 %), Yellow_Red 0.
Also flagged by: chattering 8, v4i runs 5, item 1 7, any package category 9 (75 %, same incl. watch), any or item 1
10. Flagged by NO other check: 2 of 12 (17 %).
Cases: asclog d16 81 / 1,000 (94th pct), integrator d16 102 (96th), d2 2+ 102: ok. 11021 d2-d4 are Advance / Mid
(not in scope) but sit at the 98.6 / 99.6 / 99.5 pct of their function's red rate (170 / 256 / 203).
Plot review/thresholds_all/crit17_red_dropout.png.
## 3 Unusual daily pattern RAW (bd_flag and run >= 3 h, before any suppression) vs the four checks that switch it off
Raw daily flags 77 (reported 28). Reported category -> also raw daily:
  busy at night 133 -> 37 (28 %); misses vehicles at night 91 -> 8 (9 %); goes silent 75 -> 14 (19 %);
  count drops 46 -> 11 (24 %).
Reverse: of 77 raw, 24 (31 %) have none of the four (19 alone, 18 down / 6 up; 15 bad, 9 suspect); also busy at
night 37, silent 14, count drops 11, night misses 8. Every raw flag has some reported category.
Misses (missed / no run / run 1-2 h / run >= 3 h cleared by mates / congestion / not scored):
  busy at night 96 / 22 / 55 / 18 / 2 / 0 - looks at counts not time ON (75 of 96 are Presence / Other / Mid) and
    night counts are small (median 106 01-05 h vs 622 caught): the night share barely moves the hourly shape.
  misses at night 83 / 47 / 32 / 3 / 1 / 0 - busy detectors (median 2,589 a day vs 1,039 caught): night hours are a
    tiny share of the day, so missing night cars does not reach z 4 for 3 h.
  goes silent 61 / 24 / 34 / 3 / 0 / 6 - silence too short (34 < 3 h) or at night (24 start 21-05 h) where the
    expected counts are already near zero.
  count drops 35 / 7 / 18 / 9 / 1 / 1 - drop late in the day (26 from 17 h or before 05 h): the hours after it are
    low-count; mates drop too (phase-wide) on 9.
Tables dc_work/s141/item1.csv, item2.csv, item3.csv, item3_diag.csv, cuts141_item1.csv, cuts141_item2.csv,
cases141.csv; per detector det141.parquet.
## Open
- Item 1 as specified catches no known flicker case; 1-in-500 of all detectors is far above them.
- Pooling choice for small cells (Yellow_Red 1 lane, 2+ cells) moves flags; per-cell p99.8 of < 500 is near the max.
