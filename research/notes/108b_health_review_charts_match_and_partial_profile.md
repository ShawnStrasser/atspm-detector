# 108b — Health review v3 rebuilt so every chart shows its explanation; profile check on partial samples (2026-10-06)
User (stopped at row 8 of the note-108 sheet): "your charts are shit and don't match the explanations, I can't follow this".
Same 30 rows (dc_work/health108/review_rows108.csv, none broken), new code `research/code/health/h108c_review.py`;
study code `h108b_profile_partial.py`. Work `%DC_WORK%/health108/v3c/`. Hi-res log + classifier outputs only; training
signals only; CPU <= 4 threads. Nothing adopted; resolver v108 unchanged.
## What the user's comments showed (rows 1-8 answered; 12 entries incl. earlier comments)
* Row 2: 'no yardstick' unexplained; det 6 looked bad at night and seemed unflagged (it IS suspect by the profile, but the
  legend showed only 'bad' mates). Row 3: stuck several times, only one flagged. Row 5: complaint about ON-again-without-
  OFF, while the 'watch' came from too-fast actuations that were not drawn. Row 6: 'erratic' contradicted the chart.
  Row 7: 'expected' off the chart, no drop visible. Row 8: 'L1+2' unclear, too-fast ONs not drawn.
* Real explanation bugs found while rebuilding (old sheet): row 3 said the healthy mates 'were not queued' - they were
  2.3x busier; it failed Q1 on r = 0.60 < 0.70. Row 8 cited ON->ON < 1 s; the score was driven by bursts (7.7 % vs 4.5 %).
  Row 24's change point was drawn at the wrong time (package level_b counts covered 15-min bins only; fixed in the chart,
  the package display `_hm(level_b)` has the same offset when a sample has comms gaps). Row 2's finding is pass 1
  (~50 expected from its bad mates); pass 2 on the rest of the signal expects ~26 (< 30) - both now drawn.
* Check behaviour the charts now expose (for the orchestrator, not fixed): erratic counts falls back to the WHOLE signal
  when phase mates are twins (row 6: det 2 / 3 carry det 42's counts) - the misfit is the signal's; count drops uses the
  whole signal as reference, so a detector busy at night 'drops' at dawn (rows 7, 26: 140 % / 2,198 % of all others
  before 04:15 / 05:15); stuck scores only the longest ON (row 3: 56, 43, 30 min); row 9 (121-min ON, mates 1.7x busier,
  r = 0.99) stays bad only because Q1 clears <= 60 min; row 20 'Presence' counts exactly like its Advance/Count mates, so
  the Presence volume limit (83) fires - a function error becomes a health flag; row 17 mates chatter as much as det 2.
## The rebuild
* Every statistic behind every deciding finding is RE-COMPUTED from the saved events with the package's own functions
  (health_core.health pass 1 / pass 2 with the note-96 inputs, same references, twins, bins) and the explanation is
  written from those numbers; 91 recomputed values checked against the resolver: 91 / 91 match (v3c/auto_check.csv).
* One evidence panel per finding (stuck: one stem per ON >= 1 min, limit + bad lines, over-limit ONs labelled; silent:
  5-min zoom +-2 h with expected per 5 min; count drops: share of the rest of the signal with before / after means;
  erratic: actual vs the check's own expected with the chance band, worst 3 periods labelled; volume: 5-min counts,
  limit + old 150, peak labelled; too fast / chatter: per-hour shares with each driving limit; night: expected from the
  partner; erratic time ON: actual vs count x usual ON, every flagged period marked; N1: per-traffic-level limit as a
  step line; profile: hourly share vs the type + volume band (95 % range), out-of-band hours circled and named in the
  text; Q1 / Q2: healthy mates' average % ON on the same axis with r printed; Q2 queue-pattern scatter with slopes;
  D1: minutes ON per 15 min split normal / logged-again). Then every phase detector, 'det 12 P4 Advance L1', this row
  thick, mates' own new status in [ ] (bad dotted, suspect dashed); 15-min counts and % ON; 3 h context either side.
* Header: each check once in plain words with its threshold rule. New column 'Our reply' answers rows 1-9, 11, 12.
* User entries: 12 / 12 carried, verified cell by cell (backup review/_backup/health_review_v3_20261006_172751.xlsx).
* QA: all 30 PNGs opened; every number / time in each 'Why' found on its chart -> dc_work/health108/qa_v3.csv (109
  claims, all verified yes). Fixed during QA: invisible 20-min silence (zoom panel), N1 y-scale, crowded stem labels,
  N3 marks (all 32 periods), share axis cut (stated), empty x label, change-point offset.
## Profile check on shorter samples (2-fold split by signal, references + limits fitted on the other half)
Presaved type + band profile restricted to the available clock hours and renormalised; limit p99.5 per cell; >= 50
actuations in the partial sample; volume band from the partial rate. Baseline = the 24-h profile flags (f_prof).
| sample | windows | scored | recall per window | recall any window | flag rate | healthy flagged | flags also 24 h |
|---|---|---|---|---|---|---|---|
| 3 h | 8 | 67 % | 26 % (00-03 h 18 %, 18-21 h 34 %) | 70 % | 2.0 / 100 | 1.4 % | 27 % |
| 6 h | 4 | 79 % | 36 % (00-06 h 22 %, 18-24 h 44 %) | 75 % | 2.1 / 100 | 1.4 % | 36 % |
| 12 h | 2 | 92 % | 60 % | 83 % | 2.3 / 100 | 1.4 % | 54 % |
| 24 h (held out) | 1 | 95 % | 93 % | 93 % | 2.5 / 100 | 1.3 % | 83 % |
* It CAN run on partial samples: the false-flag rate on presumed-healthy detectors stays ~1.4 % at every length (held
  out). But it is a different, weaker check: a 3-h sample finds a quarter of the 24-h profile problems, 12 h about 60 %,
  and most short-sample flags are detectors the 24-h check does not flag. Night windows score least (few actuations).
* Suggestion (orchestrator): run it from 12 h as a finding; 3-6 h at most as 'watch'. Not tested: a 24-h presaved
  profile of the same detector (needs history), which would be the stronger short-sample yardstick.
