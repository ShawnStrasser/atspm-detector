# 139 Too-fast = physically impossible actuations only; package back to note 135 (2026-10-12)
User (Oct 12): drop the random-arrival yardstick; too-fast only when traffic cannot produce the ONs. Research only -
the package keeps the note-135 rule until the user approves. Code research/code/health139, out dc_work/s139.
## Step 0 - package restored
Note 138 had put its rule into src/ without review. src/atspm_detector/health_v4.py, health_v4_stats.py and
weights/health/health_v4_refs.json restored from dc_work/s138b/bak_src_pre138 (= s135 build + the two narrowing fixes
ref_held / rule F on Presence-Other-Mid); whole src tree now equals that backup; CHANGELOG 1.0.1 lines for note 138
(headway rule, chatter extension guard) removed. 138 state kept in s139/bak_src_138. check 7 / 7 PASS;
cmp135 research review: 4,180 / 4,180 status and rule equal to score_v4e.
## Rule (h139_events, score_v4g = score_v4e with s8_rapid replaced)
Data = w40, 763 training signals, 8 windows, locked_v2 asserted absent, no fault events; phase / function / lanes =
model (lanes = detector lane span). Every 82 is an ON (also an ON logged again without an OFF).
(a) impossible runs: 5 ONs in a row, each < 0.6 s / lanes after the previous. 0.6 s per lane: 1-lane gap distribution
    has no knee (share of gaps < 0.4 / 0.6 / 0.8 / 1.0 s = .55 / 1.0 / 1.6 / 2.4 %), so physics + the OK heavy rows:
    0.6 s = 40 ft front to front at 45 mph; at 0.8 s the OK heavy row 2B039 d18 has 19 runs a day, at 0.6 s 2.
    Limit L_a = max(3, 0.5 per 1,000 ONs) (above the 1-in-500 of the population at 30 min / 3 h).
(b) over-capacity cycles: ONs in green + yellow (Count / Presence / Yellow_Red) or the whole cycle (Advance / Mid /
    Bike / Other) > secs x 1800 veh/h/lane x lanes x 1.25. Population: 0 in 99.7 % (24 h) of detector-windows; 99.8th
    percentile = 2 cycles a day -> limit L_b = 3 cycles.
x = max(runs / L_a, n_over / L_b); suspect from 1, bad from 2 (>= 50 ONs). Chattering unchanged (score_v4e has no
extension guard); rule F and every other check unchanged.
## Findings (v4g)
- Too-fast per 100 (1.0.0 -> v4g): 30 min .134 -> .056, 3 h .311 -> .229, 24 h .554 -> .615; cases as v4h below
  except 08052 d10 (28 runs) and the user flicker cases, none caught (runs 0-1 at 0.6 s).
## Revision v4h (user, Oct 12) - h139h_events, score_v4h, h139h_sheet; out s139/*139h*
Supersedes v4g above (v4g kept for comparison).  Implausible gap = ON-to-ON < 1.6 s / lanes for stop-bar zones
(Count, Presence, Yellow_Red; 1800 veh/h x 1.25 per lane, any colour), < 0.6 s / lanes for Advance / Mid (Bike, Other
too).  Measure = runs of 5 ONs (4 such gaps in a row) per 1,000 ONs; plain gap counts rejected: OK extension zones
2B068 d19 / 2B058 d46 have 119 / 209 per 1,000 at 1.6 s and the stop-bar 1-in-500 is 376-463 per 1,000.
Cycle = Begin Green to next Begin Green, nothing skipped; capacity = green s (1 -> 8) x 0.5 x lanes x 1.25; count =
green + yellow ONs (stop-bar) or whole cycle (others).  Cut-offs = 99.8th pct per type x length (>= 50 ONs), floors
1 run / 1,000 and 3 cycles, and >= 3 events; suspect at the cut, bad at 2x (cuts139h.csv):
  runs / 1,000 ONs  stop-bar 34.1 / 23.3 / 20.7, moving 11.2 / 11.4 / 7.4, other 1.0 / 7.1 / 8.0 (30 min / 3 h / 24 h)
  over-cap cycles   stop-bar 3 / 3 / 5.6, moving 16 / 77 / 483, other 10.7 / 45 / 227.
Green capacity is far too small for advance / other zones (43 % of advance days have an over-capacity cycle: arrivals
in red, permissive / short protected greens), so their cut-off is huge; flags with most ONs outside the phase's own
green + yellow (permissive hint): 22 / 36 (30 min), 22 / 43 (3 h), 24 / 59 (24 h).
Too-fast per 100 (1.0.0 / v4g / v4h): 30 min .134 / .056 / .117; 3 h .311 / .229 / .267; 24 h .554 / .615 / .334.
v4h status bad / suspect per 100: .081 / .290, .482 / .612, 2.010 / 1.770.  Cases as v4i below except 08052 d10
(1,125 over-cap cycles) and integrator d16 (17 runs / 1,000 vs 20.7).  v4h sheet (13 rows) superseded by v4i.
## Revision v4i (orchestrator, Oct 14) - h139i_events, score_v4i, h139i_sheet, h139iv_plots; out s139/v4i
Stop-bar limit 0.8 s / lanes (was 1.6): 1-lane stop-bar gap histogram (thr_v6/det_hist) has a separate bump peaking
at 0.4 s; excess over a log-linear fit of the main mode's left flank (1.0-1.6 s) = +20-140 % at 0.3-0.6 s, +4-8 %
at 0.7 s, ~0 from 0.8 s -> the bump ends at 0.8 s (pooled minimum 0.5-0.6 s; 0.9 s would cut into the main flank).
0.79 % of 1-lane stop-bar gaps are under it (was 5.04 %).  Advance / Mid / Bike / Other stay 0.6 s / lanes.
Over-capacity cycle check DROPPED.  Runs measure and 1-in-500 cuts as v4h (>= 3 runs; floor 1 / 1,000), cuts139i:
  runs / 1,000 ONs  stop-bar 12.1 / 7.8 / 7.9, moving 11.2 / 11.4 / 7.4, other 1.0 / 7.1 / 8.0 (30 min / 3 h / 24 h).
Too-fast per 100, 1.0.0 -> v4i: 30 min .134 -> .042 (-68 %), 3 h .311 -> .113 (-64 %), 24 h .554 -> .169 (-70 %).
Bad per 100: .062 -> .053 (-14 %), .533 -> .434 (-19 %), 2.175 -> 1.993 (-8 %); suspect .304 -> .288 (-5 %),
.629 -> .608 (-3 %), 1.736 -> 1.760 (+1 %).  254 status moves vs 1.0.0; 23 newly flagged, 119 cleared by too-fast.
Cases (h24_b): 08073 d7 bad -> bad (14.4 / 1,000); 08052 d10 bad -> bad (not too-fast: 1.9 / 1,000; prof / occspk);
10055 d5 bad -> bad (level / night); 11021 d2 ok, d3 bad -> suspect (chatter), d4 suspect -> ok; user OK 2B039 d18,
04028 d53, 2C028 d37, 01062 d2 ok; extension 2B068 d19, 2B058 d46 ok; integrator d2 / d16 bad -> ok (0 / 1 run);
asclog d16 bad -> ok (0 runs), d57 bad -> bad (night), d54 / d58 watch -> ok.
Sheet: health_review_v6.xlsx was open in Excel -> written as review/health_review_v6b.xlsx, 11 rows, charts =
30-min raw ON/OFF bars + 3-min zoom, phase green shaded, too-close gaps marked; all 11 opened.
## Open
User flicker cases (11021, asclog d16, integrator) stay ok / chatter-only under a physics-only rule.
