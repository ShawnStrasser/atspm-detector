# 135 atspm-detector 1.0.1: long-span crash fix + three narrow health rules + bad periods (2026-10-08)
Input: 4 integrator cases (s135/integrator_cases.parquet), user: "don't overcorrect". Backup dc_work/bak_1_0_1. Code
research/code/health135 (h135_lift/green/fast, score_v4e, h135_sheet2/2b), final135 (cmp135, par135, bench135); out
dc_work/s135. Nothing committed or published; no fault events; no locked signal (asserted).
## 1 Crash (> 24.8 days)
streams._ms stored ms offsets as int32: past 2^31 ms they wrapped negative -> np.bincount error, or silently empty
network pieces. Now int64 (pointers stay int32). Grep: no other whole-window int32 / INT offsets (health hours, SQL
floor(t/3600)::INT, lag differences are small). Tests tests/test_long_span.py: int64 unit test + 38-day gap case fail
on 1.0.0, pass now; 40-day and 60-day spans run, phases agree with the sample alone, first / last 30-min pieces equal it.
Integrator crash case: runs; whole-call phases = each 2-h piece except det 41, and det 10 is withheld "stuck ON" (an ON
left open across the 38-day hole). Not changed (outside this release): scoring each piece separately is the right use.
## 2 Stuck on, queue excuse (score_v4e / health_v4._lifted)
Mate OFF counts cannot separate: det 4's mate 42 turned OFF 32 times in 57 greens, but in the user's OK'd queue 2B530
det 60 the mates turned OFF 5-7 times per green. Rule used instead: a hold >= 1 h excused by Q1b is not excused when
>= 60 min of other 15-min periods (hold +-1 period left out) had phase traffic >= the hold's mean and the detector's
median % ON there was < 50 %. It is then stuck on, score capped .35 (suspect; bad only if other findings are bad on
their own). Holds < 1 h unchanged (applying it to them lifted 397 of 2,192 excused holds incl. the user's 06013 d8).
Fix A (after verification): no lift when a detector of the traffic yardstick (the 'ref' members: the phase's other
Advance / Count, else the signal's) was >= 95 % ON in a 15-min bin of the hold (ref_held_e) - an Advance buried under
the queue under-measures the hold's traffic. Population: 11 Q1b holds, 2 lifted (08079 Bike 2 h 9 min, ON 1 % at the
same traffic; 10004 Presence 1 h 6 min, 46 %, its Advance max 61 %); 08150 d3 (Other 1 h 45 min, Advance d2 100 % ON
in 3 of 8 bins) excused again. Det 4: ON 25 % -> suspect. 2B530 d60 (99 %), 14003 d4, 06013 d8 stay excused.
## 3 Count zone held ON (occ_hi)
A held-ON 15-min period also needs the zone ON >= 20 % of its phase's green after the first 5 s of each green, with
>= 60 s of such green. Det 22 / 23: green-ON 5-14 % in light traffic -> not held (375 / 150 min -> 0). 2B316 d6
210 -> 210 min (bad, kept); 08019 d3 180 -> 150 and 165 -> 120 min (rule C suspect, kept). 2 detector-windows move.
## 4 Too fast, even flicker (rule F)
phi = sum (fast - p base)^2 / (p (1 - p) base) / (k - 1) over 15-min periods with starts (fast = starts < 1 s after
the previous start that are not re-triggers; base = starts that are not re-triggers). Rule F: status bad, the only
finding is too-fast, phi <= 2 over >= 8 periods, fn Presence / Other / Mid (fix B: on Count / Yellow_Red / Advance /
Bike steady extra ONs inflate counts -> stay bad; 10 of the 24 first-draft moves undone) -> suspect. Reason (package)
"N extra fast ONs, spread evenly with traffic - flickers as vehicles pass, check zone setup", N = fo_use - fem_use,
whatever drove the score (the draft appended the note to "38 bursts ...", a contradiction). Bad rapid-only phi:
11021 3.5-5.8, 10055 d5 3.1, 08073 d21 8-10 (stay); 04083 d2 0.4-0.9 (Presence, bad 3 h -> suspect).
## 5 Bad periods (package only; same JSON)
chattering (15-min bins share > limit, >= 10 ONs), too-fast (burst 5-min bins, else hours with excess >= 3 SD), too
many (5-min bins over the limit), busy at night (01-05 h of the day), erratic counts (periods outside the range),
erratic time ON (n3 bins), Count zone held ON (held bins). docs/embedding.md lists every `what`.
## Rates per 100 detector-windows, research population (bad / suspect), 1.0.0 -> 1.0.1
30 min .062 / .304 -> .062 / .304; 3 h .533 / .629 -> .516 / .642; 24 h 2.175 / 1.736 -> 2.145 / 1.770. 18 moves: 14
flicker bad -> suspect (all Presence / Other), 2 not-a-queue ok -> suspect, 2 Count held suspect -> watch (first
draft: 29 moves, 3 h .499 / .660, 24 h 2.128 / 1.790; fixes undid 10 counting-zone flickers + 08150 d3). Reviewed rows (v3 36, v4 40,
32 verdict detectors x 4 windows = 203): 0 change. README numbers (0.4 / 1.2 / 3.9, 19 of 20, 0 of 12) still hold.
## Checks
- Package = research (cmp135, research mode): review set 4,180 / 4,180, changed set 7,632 / 7,632 statuses and rules;
  hi_min_c, n_lift, fe_n, fe_phi (max rel 6e-16), stuck / count_on / rapid scores 0 diffs; all 18 moves reproduced
  (after fixes A / B; 'changed' = first-draft + final moves).
  Package mode 4,178 / 4,180 and 7,628 / 7,632: only the known all-data night-model cases (note 124).
- Parity (note-115 set, 132 cases, 2,627 detectors): candidate 15,705 and phase 688 rows identical; 32 non-health
  columns identical; health status 0 moves; 23 rows gain bad periods; 0 reasons (the draft's s25 d24 Bike 10-h
  overnight hold is excused again by fix A: its Advances d23 / d25 were stuck ON too; bad either way).
- After fixes A / B (re-run): check 7 / 7, no --freeze. Wheel 5.7 MB = src (70 files); twine --strict PASSED; fresh
  venv (pandas 3.0.6, numpy 2.5.3, duckdb 1.5.6, ort 1.30): smoke OK, check 7 / 7, pytest 11 passed / 4 skipped.
- Time: 1.0.1 - 1.0.0 within +-0.02 s (3 h, 24 h).
Verify (2nd agent): 34 signals x 4 windows, 1.0.0 vs 1.0.1: 5 moves, all change 3 / 4; 40 / 60-day repeats run.
After fixes: same 34 x 4 vs draft 0 status moves, 2 reason texts (671cf1c8 d2). Integrator: d4 suspect, d2 / d16
Presence suspect with the new wording, d22 / d23 ok, crash case runs. 08150 d3 end-to-end (package classifier):
suspect in 1.0.0 and 1.0.1 alike - its queue test fails before the lift (corr with the only healthy mate d2 .53 <
Q1_CORR; Bike d4 bad) - so the 'saturated queue stays ok' example exists only with research inputs: not in the sheet.
Sheet v5: 14 rows (+ 10078 d16 Advance, steady flicker, stays bad); 04010 d24 is Presence -> stays suspect.
