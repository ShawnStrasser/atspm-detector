# 82 — Health review sheet v1: every check the package runs, judged by examples (2026-10-03)
Goal (orchestrator): let the user judge each health heuristic by ~5 examples where it fired. No model change; `model/`
and the candidate package untouched. Code `research/code/health/h82_gate.py` (gate metrics, DuckDB 4 threads) and
`h82_review.py` (selection, charts, workbook); work dir `%DC_WORK%/health82/` (gate_w40, chosen, rows, rates).
Deliverable `review/health_review_v1.xlsx` + `review/health_review_v1_charts/` (70 PNG). Data = note-79 w40 windows
(Sat 26 - Mon 28 Sep 2026: 4 x 30 min, 2 x 3 h, 2 x 24 h), 764 training signals, locked_v2 asserted absent.
Inputs = the saved note-79 inputs (stg OOF phase / function / lanes of the matching length): hi-res log + the
classifier's own outputs only; no prints, no config, no fault events. Sensor type (records) used only to spread the
examples and shown as a column for the user.
## Checks listed (16; plain names in the sheet)
health_core rules that set suspect / bad: stuck on, goes silent, chattering, too-fast actuations (rapid), too many in
5 min (volume), count drops (level), erratic counts (choppy), misses vehicles at night (night_drop), busier at night
(night_day), doesn't follow traffic (corr). Note only: too-short ONs (short_on). Classifiability gate (health.py, the
answer is withheld): almost no actuations (< 20/day), ON almost all the time (>= 90 %), chatter storm (>= 600/min).
Not run: dead channel (needs a channel list; prod passes none). Proposed, NOT in the package: undercounts vs partner
(note 79 c1, limit p99.5 of presumed-healthy per (own, partner) predicted function and length, here also requiring
>= 20 ONs on the detector, as health's own data rule; limits fitted on all w40 signals, display only).
## Rows: 70 = 14 checks x 5 (chatter storm 0 fired anywhere in w40; dead never runs)
Per check: 3 clear (top half of rule score) + 2 borderline (lowest score above the limit), greedy spread over signals
(at most one row per signal per check; reuse across checks penalised), sample lengths and sensor types; seed 81.
Columns: #, check, signal, det, model (phase + function), sample, sensor, what the check saw (one sentence with the
number, one decimal when within 2 pt of the limit), result (package status / no answer), chart link, answer, comment.
Question on top: "Is this detector really faulty in the way the check says? Y / N / ?". Second sheet "Checks": plain
name, what it looks for, effect, fires per 100 detectors at 3 h and 24 h, examples.
Charts (saved events only): stuck = counts + length of every ON; silent = counts + running totals; chatter / rapid /
short / volume = ON-OFF bars over the busiest 2 min with the predicted phase's green + histogram (OFF->ON gap, ON->ON
interval, ON length) or 5-min counts; level / choppy / partner / low = counts + running totals; night / corr = hourly
counts + share-of-day vs the rest of the signal (night_drop: % of its partner). Comparison = the check's own partner
where it has one (partner / night / recovery), else the best-correlated same-phase detector, else the busiest.
## Fires per 100 detectors (w40 prod, every detector with an actuation; 3 h / 24 h)
stuck .46 / 1.82; silent .20 / 1.73; chatter .17 / .16; rapid .66 / .81; volume .05 / .09; short-ON note .31 / .65;
level .08 / .53; choppy .82 / 1.04; night_drop 0 / .28; night_day 0 / .32; corr 0 / .77; partner (proposed) .52 / 1.30;
gate low 1.20 / 2.50; gate ON-all-the-time .03 / 0; storm 0 / 0; any suspect / bad 1.82 / 5.20.
(Population = all detectors, not note 79's presumed-healthy set, so these are fire rates, not false-alarm rates.)
## Checks done
* Reproduction: for all 70 rows the package health_core re-run on the saved events with the saved inputs gives the
  same status and the same rule score as the note-79 run (0 mismatches).
* Spot-check from raw events, independent code (6 rows): #8 2B424 d4 0 ONs 06:00-07:10 while d3 / d2 counted 654 / 534
  (first ON 07:12) ok; #24 14035 d38 280 ONs in 17:10-17:15 ok; #44 2B099 d1 night 1 vs 60 x 655 / 1020 = 38.5 ok;
  #57 03101 d15 21 vs d17 1911 ok; #70 10015 d15 ON 96.4 % ok; #15 01028 d8 chatter 68.1 % exact vs 72 % reported.
* FINDING (package, not fixed): chatter's 0.3-s limit sits on the 0.1-s log tick. health_core takes gaps as float
  seconds from the window start, so an OFF->ON gap of exactly 0.3 s is counted as "< 0.3" only sometimes (float
  rounding; depends on the window start). 01028 d8: exact < 0.3 = 68.1 %, float = 71.7 %, <= 0.3 = 78.2 %. The same
  applies to the 1.0-s / 0.5-s rapid limits (act_stats). Fix = compare in integer ticks (or limit 0.25 / 0.95 s);
  then recalibrate the chatter / rapid limits. Orchestrator to decide; changes fire rates, not the review question.
## For the orchestrator
* Answers will say, per check, how often a firing is a real fault (precision by check) - the number note 79 could not
  measure. The partner check's rows are the evidence for or against adding it as a note.
* Several borderlines sit exactly at their limit (chatter 30.0 %, rapid 1.00x, level 14.7 %): useful for the user to
  see what the edge looks like.
