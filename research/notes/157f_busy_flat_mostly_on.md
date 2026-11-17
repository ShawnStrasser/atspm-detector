# 157f Sheet-3 follow-ups: no daily pattern on one burst, busier-at-night naming, mostly ON built (2026-11-10)

User's sheet-3 review (review/busy_at_night_review_3.xlsx): busy-at-night clears, 06016 kept, 2B049 d51, 08300 d24 agreed;
09007 d17 "clearly has a daily pattern, busy only Sunday morning"; 2B365/2C010 "busy at night may fit better"; 10004 d2
"is it an average washing out?"; mostly ON: 01066 d41, 03020 d54, 11009 d18 agreed, 03007 d47 unsure, 11012 d2 "suspect".
Code: src (options below), research/code/busy157 (flat_night.py, mo_list.py, build_review4.py; debug_steps.py OPT updated).
Review: review/busy_at_night_review_4.xlsx.  NB the report's detector config changed during the day (823 -> 899 signals
from the workbook): every comparison below re-ran its baseline the same hour (off, on1 = 157d options, on2, on3).

## Package (health_duckdb.OPTIONS, all OFF = byte-identical; snapshots 7/7, atspm-detector-check passes, pytest 250)
* flat_two_hours (assemble flatm/flat): a busy zone (own_d >= 30) needs own_n2 = its second-busiest night hour >= 0.8 x
  day as well.  First tried the median night hour: dropped 12 existing flats, mostly low-volume zones whose one covered
  night is a single hour (08300 d24, which the user called right) -> rejected.  Two-hours on busy zones: only 09007 d17
  changes (night hours 724, -, 12, 26; Nov 6/7 nights missing in the local store, 1-1.5 nights covered typically).
* flat_busier_busy (assemble xx flat_nb, periods, outputs): flat and own_n > own_d -> named busy at night, whole window,
  text "busier at night than by day: ~N an hour at night, ~D by day, while the rest drops from A to B".  27 relabelled,
  no status change (no daily pattern 213 -> 129 detector-days at conf >= 0.5).  10004 d2: 514 vs 413 an hour.
* mostly_on (held_on mo10 / mo10l / mo_runs / mon, assemble xx0, verdict v_count_on, periods, outputs; limits
  health_limits.json "mostly_on"): 10-min bins >= occ 0.5 in a row (>= 3), run mean >= 3 x median over the run of
  (own 20-120-min level x mates' rise), mates' mean < 0.5 x own (else congestion), mates exclude bad ones and bins held
  >= 95 % (03007 d47: mate d24 stuck at the same time hid it); runs half covered by holds >= 5 min are left to stuck on
  (else one event counted twice: 25 suspect -> bad); score sc(minutes, 30, 60), no mates -> <= 0.5 (suspect).  Folded
  into key count_on; reason "held ON x % of the time for ..." so findings / the ATSPM report file it as stuck on (no
  report change).  Per-bin hot test fragmented runs (mates' level jitter) -> run-level test.
* Network (on2 -> on3): 63 detectors with mostly ON, 41 change status (26 ok -> suspect, 5 ok -> bad, 10 suspect -> bad),
  14 without mates.  Side effect: 03020 d8/d10 lose silent (d54 left their yardstick); both still bad on stuck on.
* off -> on1 with today's config: 29 detectors (as 157d minus 2B098 d24, now config-dependent).

Pending user on sheet 4; on approval all six options True + re-freeze + CHANGELOG.
