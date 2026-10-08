# 118c — Health v4 fixed after the adversarial check (2026-10-07)
Open items of 118b + user feedback on health_review_v3. Same w40 data (763 training signals, locked_v2 absent), hi-res
log + classifier outputs only, no fault events, CPU. Nothing adopted. Work `%DC_WORK%/s118c/`. Code `research/code/
health118/`: h118c_band, h118c_dropout, h118c_fast (+ h117_events: non-chatter fast columns), score_v4c, h118c_plotdata,
health_charts (two-panel time-of-day chart), h118c_sheet.
## 1 Time of day (h118c_band, 2 folds by signal, groups fn x span x volume band x DAY, >= 150)
* Per-hour robust z vs the type's band, counts (sqrt share) AND % ON (arcsin sqrt) + first differences, run of K hours:
  any % ON or difference variant that keeps 5/5 must-bad flags 2-3 of the 3 fine cases and 6-17 / 100 (band_grid.csv);
  % ON alone flags user-OK'd 12061 d41, 14003 d4 and the extension zones -> not used to flag (shown on the chart).
* Adopted on counts only: |z| >= 4 for >= 3 consecutive hours (+ Poisson noise in the scale, >= 30 actuations gap),
  against the TYPE band or against its PHASE MATES' own day (>= 2 mates, >= 200 / day). Excuses: mates outside the
  type band the same way (traffic), quiet while its own % ON >= 50 % / 1.5x its day (queue), busier while its mates are
  >= 50 % ON (their counts flatten). Model unsure: shortest run over the plausible types. Suspect 3 h, bad ~6 h.
  Same quiet stretch reported once (stronger of 'goes silent' / this). Night busy stays with the 116 night check.
* 0.50 / 100 detector-days (type .37, mates .27), +0.23 beyond the night check; must-bad 4/5, fine 0/3, OK'd 0/12;
  Sun-Mon Jaccard .49 (night check .47). Visual check of 35 new-only flags: ~25 real (3-10 h quiet while mates count,
  c585e635 d36 11-17 h both days, 10028 d16), rest traffic-like; the queued-mates excuse was added after that check.
* Signal-wide note: >= 20 % (and >= 5) of a signal's detectors outside their type's day: 26 of 1,484 signal-days,
  incl. 2B069 both days (11 of 37), 11021 both days, 01074 Mon.
## 2 Stuck on
* No queue excuse for Advance / Count / Yellow_Red (unless a queue-able type is plausible): +38 Advance findings at 24 h
  (single 18-48 min ONs, suspect). h3 / h24 made consistent: 'mates >= 1.5x their usual' used the sample mean, which a
  3-h peak sample already is; mates above their type's p95 % ON through >= 75 % of the ON now also count as queued
  (2B530 d60, 14003 d4 3 h: bad / suspect -> ok; 10037 d22 bad in both).
* 'Count zone held ON' (finding): 15-min bins over max(pulse-Count p99.5, actuations x 4 s (p90 normal-mode Count),
  20 % ON) while traffic < 40 % of the day's peak, >= 30 min; not when ON-again-without-OFF explains it (C1 / D1).
  2B316 d6 watch -> bad (210 min, up to 54 % ON); .02 / 100 (a version without the 20 % / light-traffic terms: .44).
## 3 Goes silent
* Expected count drops mates with any v4 finding outside the silence family (recomputed with the package code: 91 %
  exact reproduction; change applied as a ratio). 172 candidates had a flagged mate, 80 lose the finding (12032 d40).
* A silent stretch >= 80 % inside 21:00-05:00 = 'misses vehicles at night' (01062 d57).
## 4 Investigations
* Too fast: re-ONs < 0.3 s after an OFF (chatter check's job) no longer count; 2B058 d46 3 h: 863 -> 486 fast vs 496
  expected -> ok. Costs 08073 d7 (bad -> suspect; 278 of 328 fast were re-triggers, chatter 19.4 % < 21.1 %) and
  11021 d2 Sun (suspect -> ok). Floors: excess >= 3 SD, bursts >= 2 (Mid m30 limit was .54 SD), erratic share >= 2 %.
* Erratic cells need >= 300 healthy (p99.8 of fewer = the worst one): Bike-1 3 h .38 -> .16. Own-roughness erratic not
  adopted (+.3 / 100; its catches 11021 d3 / d4, 08073 d7 are flagged anyway).
* 10028 d16: back via 'unusual daily pattern' (Sun 74 vs ~495 at 11-15 h; Mon silent + pattern) -> suspect both days.
  2B019 d44 Mon ok is right (counts track mates); Sun (the reviewed day) stays suspect (stuck).
* Model unsure (top < 70 %): least strict over plausible types for night check (286 -> 268 flags), fast, volume, band.
## Rates per 100 detector-windows, suspect + bad v4 -> v4c (bad)
24 h 3.74 -> 3.90 (1.63 -> 1.64), newly +.40 / cleared -.23; 3 h 1.17 -> 1.16 (.56 -> .52); 30 min .45 -> .37
(.12 -> .06). 24 h categories: stuck 1.41 -> 1.52, silent .84 -> .45, misses at night .27 -> .50, too fast .67 -> .52,
busy at night .89 -> .84, unusual day +.18, Count held ON +.02. User rows: bad 19 / 20 flagged somewhere (08154 d1
watch), fine 0 / 12 (v4: 4).
## Sheet review/health_review_v4.xlsx (fresh, 40 rows: 8 healthy look-alikes, 5 config notes; 3 header lines)
stuck 3, silent 3, drops 2, night 3, erratic 3, too fast 4, volume 3, chatter 2, erratic ON 2, busy night 4, unusual
day 3, Count held ON 2, ON longer 2, C1 2, C2 2. All 40 charts opened; titles = saved numbers (fast / bursts / held-ON
recomputed = scorer). Time-of-day rows: 24 h counts AND % ON vs the type band + phase mates.
## Open: only 2 days; Advance no-queue rule adds 38 suspects; re-trigger rule vs user's 'bad' on 08073 d7.
