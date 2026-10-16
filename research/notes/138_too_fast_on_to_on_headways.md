# 138 Too-fast judged on ON-to-ON headways; chattering extension guard; in the package (2026-10-12)
User (Oct 12): judge too-fast by the gap between consecutive ONs. Data = w40 (763 training signals, 8 windows,
locked_v2 asserted absent), hi-res log + model phase / function / lanes, no fault events. Study code health138b
(h138b_events, _ext, _lib, _study, _rule), out dc_work/s138b. s138 / health138 (aborted run) not used. Nothing committed.
## Study
Every 82 is an ON (ON logged again without an OFF included; re-ONs < 0.3 s after an OFF no longer split off). Fast = < 1
s after the previous ON. Expected = random arrivals per 5-min bin and color state at the detector's own rate (ONs >= 1 s
apart, inverted for the 1-s cut, capped at the centred 1-h median). Short headways over all colors: OK rows 2B039 d18
h3_b, 2B058 d46 m30_c exceed (green platoons beat random arrivals) - rejected; red only adopted. 1-lane Presence / Count
< 1 s ~0; 2-lane green 20-30 %; heavy 1-lane Advance ~10 %: no plain share limit. Night excess: OK 04028 d53 = bad 08052
d10 - not used. 1-in-200 / 1-in-100 limits flag OK rows - 1-in-500 kept.
## Rule (as implemented)
Too fast if any: red z (fast ONs in red after its first 2 s beyond chance, SD) > L_z; red bursts (5-min red bins >= 4 SD
with >= 5 fast; not on Advance) > L_b; impossible runs (5 ONs in a row each < 1 s / (lanes + 1): 0.5 / 0.35 / 0.25 s)
>= max(3, 0.5 per 1,000 ONs). x = largest value / limit; score(x, 1, 2): suspect above 1, bad from 2; >= 50 ONs.
L_z / L_b = max(1-in-500 of type x sample length, floor 3 SD / 2 bins); type = fn x span, Count pulse / normal; cells <
300 take the span pool if higher. Red z: population everywhere; bursts: population at 24 h, floor 2 at 30 min and most
2+ / Count at 3 h; runs physical only (population would allow 6-21 at 24 h: faulty tail). Own type only, as studied
(the model-unsure loosening of other checks moved 25 rows, e.g. cleared 08052 d10 m30_d - not applied).
Chattering kept; not scored when the 10th-percentile ON lasts >= 1 s (extension time): 5 flags removed (30 min / 3 h).
Rule F (note 135 even flicker) unchanged, still on the old fast starts.
## Package (src, version 1.0.1)
health_v4_stats.headway_stats = h138b det_stats (per-bin red fast / expected / ONs, burst and run bins); act_stats adds
dur_p10; _colour_timelines shared with colour_bins (memo). health_v4: _hw_limits / _hw_score replace s8_rapid in the
v4c pass (v4 pass for the dropout yardstick unchanged, as research); chatter guard there. Limits = refs key fast138
(h138b_refs.py --ship: 47 type rows = rule_lim.csv, + 6 span-pool rows for a type the population lacks, e.g. Bike 2+ 30
min). Reason: "in red, 31 % of ONs under 1 s apart; type limit 27 %" / "47 red 5-min periods with bursts of ONs under
1 s apart; type normal at most 14" / "19 runs of 5 ONs each under 0.5 s apart, faster than traffic; normal at most 2";
rule F: "in red, 34 % ... ~1.5 % by chance; spread evenly with traffic - ...". Bad periods "too-fast actuations": run
bins, red-burst bins, hours with red excess >= 3 SD, for each measure over its limit. CHANGELOG 1.0.1 updated.
## Research scorer score_v4f (health138b) = score_v4e + this rule
s8_rapid flags = study rule_rows s_new exactly (too-fast per 100 .134 / .311 / .554 -> .113 / .287 / .581).
Status per 100 detector-windows (bad / suspect), 1.0.1-before -> after: 30 min .062 / .304 -> .071 / .293; 3 h .516 /
.642 -> .506 / .639; 24 h 2.145 / 1.770 -> 2.185 / 1.776; 361 status moves. Reviewed rows (203): 8 move - 04083 d2
h24_b bad -> suspect, h24_a suspect -> ok (old too-fast); 11021 d3 h24_b bad -> suspect (chattering only, as the
study said); 11021 d4 h24_b suspect -> ok; 08019 d3 h24_a suspect -> bad (Count, 8 red bursts vs 4).
## Checks
- Package = research (cmp138, 25 signals = 24 review + note-138 cases, 8 windows, 4,348 rows): research mode status and
  rule 4,348 / 4,348; s8_rapid, s8_chatter, dur_p10, runs, bursts, red ONs / fast ONs exact (s8_rapid max rel 2e-9);
  expected red fast max rel 8e-4 = research's float seconds since 26 Sep (h138b det_stats fed the package's times:
  identical). Package mode 4,345 / 4,348: 2 known night cases + 10055 d2 h24_b ok vs watch (ON longer than its kind,
  stage-1 difference; was bad on old too-fast).
- Parity (note-115 set, 132 cases, 2,627 detectors) vs 1.0.0 and 1.0.1-before: candidates 15,705 and phases 688
  identical; 32 non-health columns identical (phase, function, lanes, setback, night speed); 8 health status moves.
  Time median 30 min / 3 h / 24 h .62 / .86 / 1.16 s (before .63 / .87 / 1.17).
- check 7 / 7 (no --freeze); wheel 5.7 MB, twine check --strict PASSED (wheel + sdist); fresh venv (pandas 3.0.6):
  smoke OK, wheel = src, CLI, check 7 / 7. pytest not installed: not run.
- Outside logs (package): integrator d2 / d16 suspect (rule F); asclog full: 16 suspect, copy 57 bad, 54 / 58 ok.
## Sheet review/health_review_v6.xlsx (h138b_sheet.py run / build; package 1.0.0 vs repo, own classification)
14 rows: newly flagged 08073 d8, 04101 d15 (runs 24 / 14), 2B095 d25 (7 red bursts, 3 h), 2B058 d50 (2-lane Advance,
13 % vs 11 %); cleared 10107 d21, 08072 d11, 01009 d51 (extension), 2B048 d9 (chatter guard); stays bad 08073 d7,
08052 d10, 10055 d5; heavy ok 01062 d2, 2B039 d18, 04028 d53. Headway chart (red share per period vs chance and limit +
gap histogram; chatter row: ON durations) + day chart; all 28 opened. Package classification differs from research on
08052 d10 m30_d (Presence: not flagged) and 08019 d23 m30_c (Advance) - not used.
## Open
24-h limits loose (faulty tail): cleared 10107 d21 / 08072 d11 / 01009 d51 have red fast 3-15x chance in some hours.
Rule F evenness not recomputed on the new fast ONs.
