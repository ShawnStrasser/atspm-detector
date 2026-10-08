# 82c — Health review sheet v1: charts redrawn again as line charts with context (2026-10-05)
Trigger (user via orchestrator, after 82b): no bar charts - line charts (this detector blue, partner orange); show
enough context before and after the flagged period that normal behaviour is visible; same context rule for every
check; plain one-line titles kept. Charts only; answers, links, rows, sentences unchanged. No model / package /
locked data touched. Code `research/code/health/h82_charts3.py` (reuses h82_charts2's row loop, partner choice and
titles); titles in `%DC_WORK%/health82/titles_charts3.csv`.
## Form per check (no bars, no step plots)
* Context: saved w40 events run Sat 26 16:15 - Mon 28 24:00 per signal, so charts reach past the sample the check
  saw. Flagged stretch shaded; sample start / end as dotted lines when the chart reaches past them (whole-sample
  checks shade the sample itself).
* Goes silent: 5-min (15-min if > 6 h shown) count lines, >= 2 h either side of the silent stretch.
* Too many in 5 min: 5-min count lines, busiest 5 min shaded, +-2 h, limit 150 dashed.
* Count drops / erratic / partner / almost none: count lines over the sample +- max(1.5 h, half the sample).
* Night checks + doesn't follow traffic: hourly lines over the whole Sat-Mon period, every night shaded; busier-at-
  night and corr as % of each line's own total (this detector vs all other detectors combined).
* Stuck on: ON/OFF timeline with >= 1 h either side and green / not-green background when <= 3 h is shown; longer
  episodes (rows 3, 4) = line of "% of each 5 / 15 min ON".  ON almost all the time: same % ON line, limit 90 %.
* Chattering: line of "% of ONs back within 0.3 s" per 5 min / hour over the sample and its surroundings (limit 30 %)
  - a sub-second ON/OFF timeline cannot show hours of context, so the share over time replaces it.
* Too-short ONs / too-fast actuations: the distribution drawn as two lines (this detector vs partner, same sample),
  limit marked.
## Workbook
* Backups: `review/_backup/health_review_v1_20261005_153624.xlsx` (state before this pass) and
  `review/_backup/health_review_v1_charts_v2bars_20261005_153624/` (82b charts); 82b's originals still in
  `*_20261005_152234*`.
* Same 70 PNG names, links unchanged (70 / 70 resolve). Only A2 (intro) rewritten. Checked cell by cell against both
  backups: only A2 differs; the 4 answers (K4 K5 K6 K9 = Y, rows 1 2 3 6) identical; comments still empty.
## Visual checks (opened the PNGs)
Rows 1, 2, 3, 4, 6, 8, 10, 11, 15, 16, 21, 24, 27, 31, 36, 41, 44, 46, 56, 57, 66.
What the context shows (for the orchestrator, not a verdict): row 6 det 13 is a 0-4 per 5 min detector, so 20 silent
min looks ordinary; row 8 det 4 was already silent from 05:15, before the sample; rows 3 / 4 were ON 100 % for hours
either side of the flagged episode; row 15's partner also chatters 40-50 % all weekend (limit 30 % may be low for
that kind of detector); row 2's partner det 8 was held ON over the same 30 min.
