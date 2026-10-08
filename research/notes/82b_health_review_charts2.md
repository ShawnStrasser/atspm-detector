# 82b — Health review sheet v1: charts redrawn in a simpler form (2026-10-05)
Trigger (user via orchestrator): "why is a step function used, it's confusing - give me something simpler". Charts
only; rows, columns, sentences, links and the user's answers unchanged. No model, package or locked data touched.
Code `research/code/health/h82_charts2.py` (imports h82_review: same chosen rows, same saved w40 events, same
note-79 inputs, same partner choice); work dir `%DC_WORK%/health82/` (titles_charts2.csv = the 70 chart titles).
## What changed
* One panel per chart (was two), no step plots, fonts 12-16 pt, a one-line bold title stating the finding with the
  check's own number (e.g. "Det 12 stayed ON 29 min without a break (Mon 07:30-08:00)"), a grey line with signal /
  model / check and at most one short note. Blue = this detector, orange = its partner (check's own partner, else
  best-correlated same-phase detector, else busiest other detector - the legend says which).
* Form per check: stuck on / chattering / ON almost all the time = timeline of ON periods (thick bars), phase green /
  not-green as a light background when the window is <= 2 h; stuck = red dashed box on the long ON, partner row
  dropped when the window is > 2 h (its ON count given instead); chatter = busiest 30 s with a red marker on every
  re-ON within 0.3 s. Goes silent / too many in 5 min / count drops / erratic / partner / almost none / misses at
  night = side-by-side bars per 1 min (30-min samples), 5 min (silent zoom, volume) , 15 min or hour, with the
  stretch shaded and the 150 limit as a dashed line. Busier at night / doesn't follow traffic = hourly bars as % of
  each one's day (this detector vs all other detectors combined). Too-short ONs / too-fast actuations = plain
  grouped histogram (% of each detector's own total) with the limit as a dashed line.
* Wording fixes found while drawing: a silent stretch where the shown partner was silent too says "where about N
  were expected" (row 9); a night check judged against the rest of the signal says so (row 41).
## Workbook
* Backups first: `review/_backup/health_review_v1_20261005_152234.xlsx` + `..._charts_20261005_152234/` (70 PNG).
* PNG names unchanged, so the 70 links needed no change (70 / 70 resolve). Only cell A2 (the intro line) was
  rewritten to describe the new charts. Not locked; saved in place.
* Verified against the backup cell by cell, both sheets: 1 cell differs (A2); all 4 answers (rows 1, 2, 3, 6 = Y)
  and their empty comments are identical.
## Visual checks (opened the PNGs)
Rows 1, 2, 4, 6, 9, 11, 15, 16, 19, 24, 26, 31, 36, 38, 41, 44, 46, 48, 51, 57, 61, 66, 69: titles match the
"what the check saw" column; bar sums agree with the counts quoted (e.g. row 16: under-0.5 + 0.5-1 = 38 %).
Known limits: row 4 (5.1 h stuck in a 24 h sample) - the ON bar also covers time before / after the box because its
breaks there are shorter than a pixel; the note says so (ON 54 % of the time outside the box). Row 69's partner is
itself ON most of the time (both look like long zones) - shown as is.
