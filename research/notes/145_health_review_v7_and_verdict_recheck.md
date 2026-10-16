# 145 Health 1.1.0: final review sheet v7 (new examples + near-misses) and the README verdict re-check (2026-10-15)
Orchestrator task after note 144.  Code research/code/health145 (h145_verdict, h145_pick, h145_charts, h145_sheet);
work dc_work/s145.  No src/ change.  No fault events.  locked_v2 asserted absent (h24_b set and the 26 verdict signals).
## README verdict re-check (h145_verdict)
The 32 detectors behind "19 of 20 faulty, 0 of 12 fine" = s118c/cases_v4c.csv (note 118c), judged on the h118 windows
h24_a (Sun 27 Sep 24 h), h24_b (Mon 28 Sep 24 h), h3_a (Sun 12-15 h), h3_b (Mon 06-09 h); a detector counts as caught
when flagged (suspect / bad) in any of them.  Same harness as h144_run (package mode, shipped references, classifier
inputs health4/inputs.parquet stg h24), serial.  Harness check: the 1.0.1 backup (H144_SRC=dc_work/bak_1_0_1_pre110)
reproduces 19 / 20 and 0 / 12 exactly (verdict145_v101.csv).
1.1.0 (verdict145.csv): faulty 14 / 20 flagged, fine 2 / 12 flagged.
- Faulty lost (1.0.1 finding -> 1.1.0): 11021 d2 / d3 / d4 (chattering, too-fast 5-in-a-row; both rules gone),
  08073 d7 (too many 2,250 + erratic -> watch), 2B368 d27 (stuck on 24 h -> ok), 10028 d16 kept only via h3_b
  (silent; unusual daily pattern gone), 10037 d22 / 2B557 d24 kept via h3_a only.
- Fine now flagged: 04028 d53 h3_b too-fast suspect; 12032 d41 h24_b misses vehicles at night suspect.
README line set to 14 of 20 / 2 of 12 and 0.8 / 1.1 / 2.2 per 100 (note 144 rates).  docs/embedding.md `what` list =
the 1.1.0 values in health_v4._periods (stuck on, count zone held ON, silent, silent at night, counts dropped, busy at
night, too-fast actuations, too many for the traffic, erratic counts).
## Sheet review/health_review_v7.xlsx (no earlier file; no backup needed)
- Flagged: note-144 table (24 h per check suspect / bad / total vs 1.0.1 total; removed checks .76 %; status split;
  3 h / 30 min any) + the verdict line above.
- Examples: 32 rows, 24 h h24_b, out144_h24b.  Per check: 2 bad, up to 2 suspect, 2 near-misses (status ok, measure
  80-99.7 % of the suspect limit).  Counts: stuck 6 (2/2/2), counts fall 5 (2/1/2: only 1 single-check suspect not on
  a reviewed signal), night misses 6, busy at night 4 (2 bad / 0 suspect: none exist / 2 near), too fast 5 (1 bad:
  only 1 exists / 2 / 2; no 'too many' flag off a reviewed signal, one 'too many' near-miss), erratic 6.
  New = signal not in any earlier review sheet (v1-v6b, spotcheck_health; asclog sheet has no signal column); one
  example per signal; seed 145; spread over functions and sub-checks.
- Measure vs limit: package numbers (stuck_x / st holds vs lim_stuck, drop_lam vs lim_dropout, lv_ratio110 vs
  exp(-lim_level), night_ratio vs exp(-lim_night_drop), tod_z vs lim_prof (x 3.5 / 6 when 30 points over the mates),
  fast_rate vs lim_rapid, q5 vs lim_volume, exc_15 vs lim_choppy); near-miss % of limit = measure / limit (falling
  measures: limit % / measure %).
- Charts health_review_v7_charts/ (32, all opened): stuck / too fast = raw ON bars (detector + up to 3 phase mates,
  phase green / yellow / red band; holds > 50 min = start and end panels; too-fast = red tick at each ON within the gap,
  window trimmed to the burst); the rest = 24-h 15-min counts vs phase mates scaled to the detector over the unflagged
  day (no phase mate: the signal's other healthy detectors), flagged periods shaded red, night 21-05 grey for night
  misses (with the package night numbers), busy at night adds % ON vs mates.  The dashed line is a chart yardstick, not
  the package's expected series.
- Limits: 24-h suspect / bad per check x type (stuck = median applied limit of confident detectors, f_top >= .7;
  others from health_limits.json with the 1.0.1 floor), and 1.0.1 (stuck / erratic / too many = median applied).
## Seen while charting (for the user's verdicts, not acted on)
- 2C041 d17 (suspect, shared hold): the whole signal goes quiet at 15:34 (phase red 8 h, mates silent) - log or
  cabinet event rather than one detector.
- 2B113 d2 erratic suspect: the shaded periods are where its only phase mate goes silent (yardstick fault).
- 2B011 d8 near-miss (count drops, 85 %): silent 15:45-17:45 and 18:30-20:00 - looks faulty but under both limits.
## Open (orchestrator)
- 1.1.0 catches 14 / 20 user-faulty vs 19 / 20 (chattering / old too-fast losses = 11021 x3 by design of the agreed
  six checks); 2 / 12 fine now flagged.  README states the new numbers.
