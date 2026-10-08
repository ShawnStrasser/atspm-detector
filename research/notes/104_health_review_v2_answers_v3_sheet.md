# 104 — Health review v2 answers -> resolver changes -> review sheet v3 (2026-10-06; analysis only)
Data = note 96 (w40, 763 training signals, locked_v2 absent, 116,873 detector-windows; hi-res log + classifier outputs only). CPU 4
procs; no package / model/ touched. Code `research/code/health/h104_{resolve,review}.py`; work `%DC_WORK%/health104/`. v2 answers only
compared, never used to set a limit; v2 workbook read-only (backup `review/_backup/health_review_v2_20261006_104423.xlsx`).
## User's v2 answers (19 rows)
* Confirmed: R1 rows 1, 2 ("see how the occupancy correlates better than the counts? This is good."); R3 row 8 (business driveway,
  plausible); R4 rows 9, 10 (9 with a doubt, below); R5 row 11; R7 row 13; R9 row 14. Rejected: R1 row 3 (2B368 d27) "No!".
* Could not judge (why): rows 4, 5, 7 (R2 / R3) - the "jumps 6x / 18x more than the others explain" and "11 % of its earlier share"
  did not match a chart where the lines track; rows 6, 12 - the traffic line was SCALED ("I'm not sure what you mean by scaled");
  row 15 (R9) - the chart never showed the silence the old check saw; rows 16, 18 (N1) - normal-mode count zone; row 19 - unsure.
* Rule ideas, verbatim: row 3 "I think you're getting a false result here because you're comparing it to other detectors that are
  also bad? I am seeing them get stuck on during nightimte hours, so that's not good. also the occupancy between 27 and the other
  detectors does NOT look corelated. You need to look at this example and rehone the rules more carefully" | row 16 "there are two
  differen types of stopbar count zone, usually they are set to pulse so they can only be on for 0.2 seconds, but sometimes they are
  normal and are on as long as there is a vehicle over them ... it would be better if you actually put those into separate
  categories" | row 17 "those occupancy spikes are suspicious, especially when they happen at night. We already have a check for very
  spikey or high variance actuations right? What about something similar for occupancy? That could be a new check ... the coutns look
  fine but the occupancy looks bad, so it is suspect, not healthy." | row 19 "I see that occupance does correlate, so this could just
  be high congestion right? Look closer at this one." | rows 6 / 9 (questions) "at night its actuations are higher than traffic on its
  phase. Or.. Am I misreading because you scaled it???" / "nightime counts seem pretty high comarably and idk if its due to your scaling".
## Implemented (resolver v104 = v96 + these; details in the h104_resolve docstring)
* R1' (row 3): queue judged on HEALTHY peers only (not bad in pass 1, not themselves held ON); own 15-min occupancy must correlate
  >= .7 with them; never >= 90 % ON elsewhere in the sample while counts were light AND peers not queued (in a queue counts fall too).
* R1b (row 19): stuck on a long zone not cleared by R1', any length, occupancy follows traffic (>= .8), already >= 80 % ON the hour
  before, traffic >= its usual, never held ON in light traffic -> WATCH (possible congestion), not ok.
* N1' (rows 16 / 18): mode by median ON (pulse <= 0.25 s). The old split (share of 1-tick ONs >= .7) put the usual 0.2-s pulse zones
  in "normal", so true normal zones were judged against pulse ones. Limits now per (type, mode).
* N3 (row 17) NEW check "erratic time ON", count-type zones only: 15-min bins ON >= 3x (and >= 5 pt above) what the detector's own
  count x its usual ON length explains, healthy peers not busier than their +-1 h median; statistic = minutes of unexplained ON
  (>= 2 bins); suspect at p99.5 of ok count-type detectors (30 min 2.1, 3 h 6.6, 24 h 60.8 min), watch p99-p99.5 (N3w). Long zones
  excluded (their ON length follows the queue: on them it fired on user-OK'd row 1).
* N5 (rows 6 / 9) NEW watch: count-type zone, night/day rate >= 3x the signal's although night < day (package check needs night >
  day), >= 30 actuations 00-05. Long zones excluded (day counts saturate: would fire on .83 % of ok long zones vs .19 % count-type).
## Fire rates, suspect + bad per 100 detectors (package / v96 proposal / v104)
| sample | package | v96 | v104 | bad v96 / v104 | watch v96 / v104 |
|---|---|---|---|---|---|
| 30 min | 0.60 | 0.40 | 0.47 | .16 / .18 | .23 / .29 |
| 3 h | 1.83 | 1.40 | 1.54 | .73 / .75 | .47 / .58 |
| 24 h | 5.20 | 3.73 | 4.01 | 1.80 / 1.83 | 1.12 / 1.41 |
Added vs v96 (3 h / 24 h): R1' keeps .024 / .053, N3 .115 / .240; R1b removes .003 / .013. Fires: R1 .01 / .33 (v96 .03 / .41),
R1b 0 / .02, N1 .16 / .37 (v96 .18 / .38), N3 .16 / .36, N3w .14 / .32, N5 0 / .14 (30 min: N3 .075).
## Checks
* v2 rows (sanity only): the 8 OK'd rows keep the result he OK'd; row 3 now suspect (stuck) = his "No!"; row 17 N3 50 min vs limit 61
  -> watch (N3w + N1), not the suspect he asked for (limit not moved for it); ? rows: 4 5 6 7 15 16 ok, 12 18 19 watch.
* Note-79 evalset (circular positives, comparison only): presumed healthy flagged 3 h .83 -> .95 (+.12 [+.07,+.17]), 24 h 2.21 ->
  2.36 (+.15 [+.08,+.21]); positives 9.59 -> 9.80 / 16.98 -> 16.77. Positives among N3 fires 13.0 % (3 h, n 23) / 21.9 % (24 h, n 32)
  vs base 2.6 / 2.7 %; N5 5.6 % (n 18); N1 1.7-3.8 %; R1b 1 fire, a positive.
* FINDING: of 292 N3 / N3w detectors (3 h + 24 h) 66 % are mostly real long ONs, 26 % mostly ON events logged again without an OFF
  (package: ON -> ON = extension, ON until the next OFF; 2B058 d46, a 0.2-s pulse zone: 167 min, all missing-OFF). Charts mark both.
* FINDING (rows 4 / 5): chop15 is a Poisson dispersion; at 30-70 counts per 15 min a 2.5x-typical miss already reaches the limit
  (2B334 d13: 6.2 vs 6.1), so the lines look close. Not changed (not his idea). Row 9 (04035 d53 Advance): 4631 actuations 00-06
  vs 368 on the same-phase Count zone - his night doubt is real, not scaling.
## Review sheet `review/health_review_v3.xlsx` (25 rows; charts `_v3_charts/`, `_v3_day_charts/`)
Rows 1-10 = v2 rows 4 5 6 7 9 12 15 16 18 19 redrawn; 11-13 R1' (incl. v2 row 3); 14-16 R1b; 17-19 N1'; 20-23 N3 (one borderline);
24-25 N5. Dropped: v2 rows 1 2 8 10 11 13 14 (OK'd), 17 (judged). "Your earlier comment" = his v2 answer + comment. Charts: nothing
scaled; orange = one named healthy same-phase detector (or healthy-peer average, as the legend says); erratic rows: the check's
expected count + red circles on the 3 biggest misses; R9 row: 5-min zoom on the silent stretch with the old yardstick; ON-length
panel (missing-OFF ONs as red x) on N1' / N3 rows. Opened: 1, 4, 5, 7, 9, 11-14, 17, 20, 22, 24 + day chart 10.
## For the orchestrator: nothing adopted. After the v3 answers: R1' / R1b / N1' / N3 / N5 into the next package or not; and
whether "ON logged again without an OFF" on count zones should be its own data-quality note instead of part of N3.
