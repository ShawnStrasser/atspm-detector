# 47 — Detector health v5: the user's 2026-09-29 review (continuous ON, short ONs, grading) (2026-09-29)
Code `research/code/health/` (`health_core` + `h5_explore/final/final_eval/spotcheck`; `h3_spotcheck.on_table` fixed);
artefacts `%DC_WORK%/health5/` (v4 core backed up there). Training signals only (locked_v2 absent); no fault events;
note 38's harness, note 46's out-of-fold inputs. v4 = `health4/final_eval_final.parquet`. `model/` untouched.
## (a) "ON with no OFF" is NOT a fault (user: extension re-calls) -> removed everywhere
* A continuous ON now runs from the ON that starts it to the next OFF; ONs inside it are normal re-calls
  (`continuous_on`, used by bins/occupancy, episodes, charts). The no-OFF gate, `ep_no_off`, "(no OFF logged)" texts
  and bad-period labels are gone; stuck-on = a continuous ON >= 15 min, reason quotes its length and expected traffic.
* In practice the old no-OFF episodes (ON, long gap, ON, OFF) are still long continuous ONs: of 568 v4 calls that
  quoted a missing OFF, statuses v4 -> v5: bad->bad 132, suspect->suspect 428, suspect->bad 7, bad->suspect 1; no
  reason quotes a missing OFF now. 10090 d22 stays bad (96 min ON Sat and Sun, 36 min Mon; 35 % share after vs d23).
* Cost: the old gate skipped no-OFF gaps when < 30 ONs were expected; now every continuous ON counts. Healthy flagged
  at 66 h 3.88 -> 4.68 % (continuous ON alone). A uniform "expected traffic" gate on ALL continuous ONs is worse:
  gate none / 5 / 10 / 30 ONs expected: healthy flagged 66 h 5.30 / 5.07 / 4.95 / 4.49 %, dq health caught
  98.9 / 88.5 / 85.1 / 63.2 % (24 h: 64.4 / 57.5 / 50.6 / 39.1; short-ON score on). Real stuck loops stick in quiet
  hours too -> no gate (`STUCK_LAM = None`).
## (b) new checks
* **Short ONs** (`short_on`): non-pulse detector (mode read from its MOST normal 3-h clock block: < 70 % ONs <= 0.2 s)
  with too many ONs <= 0.2 s overall or in its worst 3-h block (limits .44 / .53 = p99.5 of presumed-healthy non-pulse,
  24 h + 66 h). Text gives the implied speed (15-ft car, 6-ft loop / 20-ft zone). Result: +46 healthy flagged at 66 h
  (+0.53 pt), +39-70 at other windows, **0 new catches**; s>0 on positives 4 vs 1 shuffled (66 h) - correlated with
  real problems but always already caught. **Scored OFF** (`OPTS min_on=False`); reported as a note above the limit.
  Many hits are signals whose Count loops log 0.1-0.3 s mixed ONs (10041 d35-d48): pulse set-up or fault - asked.
  The user's 10018 d2 is NOT separated by it: d4 (its partner) also has 0.3-s night ONs (~48 mph free flow).
* **Night drop** (`night_drop`, kept): night (21-05) ONs / expected from the day share of the detector it tracks
  (best 15-min corr >= .7, same phase, twins in), else phase (>= 2), else signal; needs >= 2 h night and day and >= 30
  expected. Suspect < .19 (p0.5 healthy, 24 h + 66 h), bad at 1/3 of it. +7 healthy at 66 h (+0.08 pt), +8 at 24 h;
  no new catch in the weak-label groups (s>0 on positives 2 vs 0 shuffled at 66 h, 4 vs 1 at 24 h); catches
  10018 d2 (14 vs 185 expected from d4 = 8 %) -> BAD, as the user said.
## (c) grading: when do several suspects add up?
Score = prod(1 - s): each finding at its suspect limit leaves .65, bad < .25 -> four at-limit findings or two well
past their limits. Tested "two different kinds of finding (uncapped, each >= suspect) = bad" (`OPTS grade2`): % bad on
healthy 1.07 -> 1.11 at 66 h, no catch or answer changed -> OFF. 07035 d19: one finding (rapid 1.3x its group limit,
bad needs ~1.6x); short ONs 14 % (healthy up to 44 %) -> suspect is consistent. Explained in the sheet.
## Result, % flagged 2 h / 6 h / 24 h / 66 h (v5 = shipped defaults)
| group | v4 | **v5** |
|---|---|---|
| presumed healthy = FA (8,666 at 66 h) | 0.95 / 1.06 / 2.28 / 3.88 | **1.01 / 1.15 / 2.69 / 4.77** |
| % bad on healthy | .37 / .39 / .71 / .70 | .38 / .42 / .79 / .84 |
| dq health (87) | 24.1 / 27.0 / 63.2 / 97.7 | 26.8 / 28.7 / 64.4 / 98.9 |
| label-check health (74) | 21.2 / 24.3 / 62.2 / 97.3 | 23.9 / 26.4 / 62.8 / 98.7 |
| card erratic, dead groups, share fell > 5x | | identical |
1 detector = 1.1-1.4 pt of a positive group. The FA rise is the price of (a), which the user's rule requires.
User answers (22, all sheets, later answer wins): v4 21/22 (10018 d2 suspect, wanted bad), **v5 22/22**; remarks
3/5 -> 4/5 (11042 d22 now suspect: 48-min continuous ON shared with 10 detectors; 2B422 d8 still ok - lane pairing).
66 h v4 -> v5: 195 status changes (ok->suspect 153, ok->bad 15, suspect->bad 19, suspect->ok 5; healthy 93).
Runtime unchanged. Deliverable `%DC_WORK%/health/health_v5.parquet` (h5_deliver.py): 33,936 rows, stg 18-21 Sep ok 13,722 /
bad 2,181 / suspect 849 / n.e.d. 217; w40 26-29 Sep 13,729 / 2,159 / 810 / 269. FA rise accepted (coordinator, 2026-09-29).
## Spot-check `review/spotcheck_health.xlsx`: 6 rows (10018 d2, 10090 d22, 07035 d19; new 10041 d41 short-ON
question, 11042 d22, 04017 d9 = cost of (a)); charts `v5_*.png`; the v4 sheet's 8 answers moved to "earlier answers" (33 rows; backup `health5/spotcheck_health_v4_backup.xlsx`).
