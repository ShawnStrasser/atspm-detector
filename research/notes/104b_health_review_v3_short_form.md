# 104b — Health review v3 rebuilt in a short form (2026-10-06; sheet + charts only)
User feedback on the note-104 sheet: far too wordy, the same explanation on every row, charts unclear. No rule, limit,
row choice or status changed: same 25 rows, same order (h104_review.select, seed 104), resolver v104 outputs as in note 104.
Code `research/code/health/h104b_review.py` (imports h104_review; run with the backup as argument so the user's entries are
read from it); CPU 4 threads; saved w40 events only; hi-res log + classifier outputs (phase, function, lanes); no locked signal.
## Backup and the user's entries
* Backup first: `review/_backup/health_review_v3_20261006_145801.xlsx` (+ `_charts_` / `_day_charts_` folders, same stamp).
* The old sheet held ONE user entry: row 1 (2B334 det 13), Answer "?", Comment "ok I see that it is not following the rest of
  the phase either with actuations OR with occupancy. So why then is it being considered "ok"?". No cell comments, one sheet.
* Carried over by (signal, detector), verified cell by cell after the rebuild: 25/25 rows match on Answer, Comment and "Your
  earlier comment"; 0 mismatches; the one answered row kept both cells word for word.
## Sheet
* Header block (8 short lines, once): what Old / New / Old -> New mean; how to read the charts; what "expected" means; one
  line per check; one line per rule (R1', R1b, R2, R3, R4, R7, R9, N1', N3, N5).
* Columns: # | Signal | Det ("det 13 · P8 Presence · lane 1") | Check (+ sample) | Old -> New ("suspect -> ok") | Why it
  changed (one line: rule + deciding number, e.g. "R2: long zone whose time ON follows phase traffic (r=0.83); erratic 6.2
  vs limit 6.1") | Chart | Day chart | Your earlier comment | Answer | Comment.
* Lane = research `lanes/lane_output.py` (pair model `dc_work/lanes/model`) run on the sample window with the resolver's
  phase / function; lane 1 = busiest lane of the phase. A label, not a check input.
## Charts (`review/health_review_v3_charts/`, `_v3_day_charts/`; all 25 + 25 redrawn)
* Every detector on the flagged detector's predicted phase is its own line, "det 12 P4 Advance L1"; flagged = thick dark
  blue; a phase mate itself flagged bad = dotted and "(flagged bad)". Top = real counts per 15 min, bottom = real % of each
  15 min ON. Nothing scaled. Flagged period shaded (the long ON / the 3 bins the erratic check objects to most / after the
  drop / busiest 5 min / night 00-05 / N3 spike bins / 15 min with a > 1 s ON on pulse zones).
* Expected (dashed, legend says how): erratic rows = the check's own expected (traffic x share over the surrounding 2 h;
  01064 uses the whole signal as its reference); doesn't follow traffic = phase traffic x share over the day; count drop =
  share before the drop; R9 = the old yardstick (phase mates incl. the bad ones); N5 = daytime share; N3 / N1' rows = expected
  time ON = own count x usual ON length (bottom panel). R4 row: 5-min counts with the 150 and 200 limits. An expected line
  more than 1.5x the real lines is clipped and its legend says "runs off the top, up to N" (2B502: up to 2620).
* N1' / N3 rows keep a third panel: length of every ON (missing-OFF chains as red x). Chart = the sample (3-h: +-3 h);
  day chart = all saved data (Sat 16:15 - Mon 24:00).
* Opened: charts 1, 2, 3, 4 (twice), 5, 6, 7, 9, 11, 12, 14, 17, 20, 22, 24 + day chart 10.
## What the new charts show (for the orchestrator, not decided here)
* Row 1 (user's v3 question): R2 clears it because its time ON correlates .83 with the phase's Advance/Count COUNTS; the chart
  shows its counts track det 9 (Mid, same lane) closely but its time ON departs from the other zones' time ON. R2 never checks
  occupancy against the peers' occupancy - that is what he is asking about.
* Row 7 (12032 det 41): ALL four phase mates are flagged bad, so after R9 there is no phase yardstick at all; the det counts
  ~0-4 per 5 min all night against an old expected ~10.
* Row 14 (08CM405 det 37, R1b -> watch): two 100 %-ON stretches with ZERO counts (11-12 h, 18:30-19 h) while the same-lane
  Advance keeps counting 50-80 per 15 min - looks more like stuck than congestion.
* Row 17 (2B068 det 19, normal mode, ok -> ok): its "median ON 7.6 s" is mostly ON events logged again without an OFF (red x
  throughout), i.e. the normal-mode label partly rests on the logging pattern of note 104's finding.
* Row 4 (2B502 det 4): counts flat ~100 per 15 min all day while its phase rises ~10x; the "drop" is a night-share artefact.
## For the orchestrator
Nothing adopted. The sheet waits for the user's v3 answers, as in note 104. Row 1's question (R2 uses traffic counts, not the
peers' occupancy) may be worth a reply; rows 14 and 17 may draw "No" answers on R1b and N1'.
