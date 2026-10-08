# 118b — Health v4: one research scorer, one chart per category, fresh example sheet (2026-10-07)
Task after notes 116 / 117 / 118a (user review of health_review_v3, Oct 7). Same w40 data (763 training signals,
locked_v2 absent - asserted upstream), hi-res log + classifier outputs only, CPU. Nothing adopted into a package.
Code `research/code/health118/{score_v4,h118b_plotdata,health_charts,h118b_sheet}.py`; work `%DC_WORK%/s118b/`.
## v4 = v110 + 118a G1-G4 + 117 fast / volume + 116 time of day (score_v4.py, 33 s on all 115,518 detector-windows)
* h118_resolve.run (all G) with the score table patched: rapid = score(new_fast_x, 1, 2) (zf excess over random
  arrivals or burst bins, n >= 50); volume = score(new_vol_x, 1, 2) (busiest 5-min flow per lane vs max(p99.8, 1800),
  green time for stop-bar types); prof = note-116 OOF level (suspect .50, bad .80; not scored -> NaN) replacing the
  24-h profile AND the 3-h profile watch (prof3_x = NaN); 'goes silent' wins over 'busy at night' (prof -> 0).
* Categories (explicit, CATEGORIES in score_v4): findings stuck, dropout (goes silent), level (count drops),
  night_drop, choppy (erratic counts), rapid (too-fast), volume (too many for the traffic), chatter, occspk (erratic
  time ON), prof (busy at night); watch occ_hi (ON longer than its kind); config notes C1 (extension time on a count
  zone), C2 (set to pulse but holds ON). D1 data note and Y / Y1 no-yardstick watches are not categories (no fault).
## Rates per 100 detector-windows (rates_v4.csv), suspect + bad v118 -> v4 (bad)
24 h 3.789 -> 3.739 (1.682 -> 1.625), newly +.645 / cleared -.696 (mostly profile -> night-level swap);
3 h 1.319 -> 1.172 (.540 -> .560); 30 min .563 -> .450 (.109 -> .117). Category rates at 24 h: stuck 1.41, silent
.84, busy at night .89, too-fast .67, chatter .44, erratic counts .43, C2 .59, occ_hi .59, night_drop .27, drops .26,
volume .10, C1 .09, occspk .05. Review signals (36): 5,965 detector-windows, per-category counts cats_v4.csv.
## User cases through v4 (review_v4.parquet)
* Time of day: 12032 d6 bad (Mon), 04035 d52 / d53, 2B069 d23, 2B502 d4 suspect; 2B531 d35, 2B530 d60, 14003 d4 ok.
* Heavy traffic 2B039 d18, 04028 d53, 2C028 d37, 01062 d2: ok. 11021 phase 2: d2, d3, d4 now flagged too-fast
  (d2 h3 bad, d3 / d4 Mon bad) = closer to the user than 118a alone. 08052 d10 bad (fast + night + erratic ON).
* Lost vs v118: 10028 d16 h24 bad -> watch (its bad came from the old profile; the night-level check gives z 1.9-2.7,
  its 'goes silent' is borderline -> R7 watch). 2B019 d44 Mon suspect -> ok (profile only). 08154 d1 stays watch.
* 2B058 d46 h3_b gains 'too-fast' suspect (863 fast starts vs 654 random-arrival expectation, 1.24x limit) although
  re-logged ONs are excluded - against the user's 'extension time, not too fast'; not on the sheet, open.
## Charts (health_charts.py, matplotlib; draws only from s118b/plot/*, built by h118b_plotdata.py)
One chart per detector x category: title = the problem with the check's numbers, subtitle 'det 15: P5 Presence ·
signal · sample · status'; base = the day chart (detector thick, phase mates thin, real units) + ONE evidence layer:
stuck spans with lengths (queue spans grey); silent stretch; drop line + expected; night expected (zoomed);
erratic band + outside points; fast ONs green / red vs expected + burst marks (15-min bars on a day, 5-min check);
flow per lane vs limit line; chatter bars; % ON vs ON explained by counts; hourly counts vs busiest-4-h bar and the
type's normal night level; % ON vs similar-zone limit; starts vs re-logged ONs; ONs >= 5 s on a pulse zone.
QA: every title number recomputed from saved data = resolver (rows.json *_chk: fast ONs, bursts, flow, chatter,
night, erratic periods, N3 minutes, re-logs, long ONs all equal); all 33 category PNGs opened; fixed: int16 bin
index overflow (bins117 b), package dropout bins are 5 min, label collisions, night zoom.
## Sheet review/health_review_v4.xlsx (fresh, 33 rows, 13 categories x 2-5; 6 healthy look-alikes; 3 header lines)
Dropped as example: 12032 d40 'goes silent 01:00-02:05, ~101 expected' - the yardstick includes night-busy faulty
mates (d6, d39) -> a false 'silent'; dropout yardstick should exclude detectors flagged busy at night (open).
## Open (orchestrator)
own-roughness erratic (118a); Bike-1 lenient erratic limit; F5 plausible-type limits not applied to the 116 / 117
checks; 2B058 d46 fast; dropout yardstick polluted by night-busy mates; 10028 d16 lost.
