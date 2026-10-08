# 121 - Health v4d: lane span on charts, wording, severity rule (2026-10-07)
Input: user comments on review/health_review_v4.xlsx (all 40 rows 'Y'; 9 lean stricter, 2 lean looser; "cautious,
don't overcorrect"). Research scorer only, nothing in a package. Same w40 data, no fault events, numpy / DuckDB.
Code research/code/health118/{score_v4d.py, h121_charts.py} + edits in health_charts.py. Work %DC_WORK%/s121/.
## (a) Lane span and wording
* Every chart subtitle and the day chart now say "covers N lane(s)" (model lane span, s117 ln); titles / legends
  of span-dependent limits name the type, e.g. "limit 21.1 % for 1-lane Advance". Reasons of stuck, erratic, too
  fast, too many, chatter, erratic ON, busy at night, unusual day end in "; covers N lane(s)".
* Separate multi-lane limits were already in place, all from healthy detectors of the same type x lane span
  (p99.5 / p99.8, >= 100 per cell, fallback lane span x length, never span-free for these): chatter, too fast
  (excess + bursts), erratic counts, stuck count, erratic ON; too many = per lane (busiest 5 min / model lanes,
  floor 1,800). Nothing added. Data says healthy 2-lane zones do NOT chatter more: 24 h limit Advance 1-lane
  21.1 % vs 2-lane 10.8 %, Presence 20.2 / 19.4 %; too-fast excess Advance 9.3 vs 11.3 SD (stats in s118c).
* 11021 det 3 is a 1-lane Advance (model lanes 1), so the 1-lane limits are the right ones.
* 'restart' gone everywhere (charts said "restart < 0.3 s"): now "turn on again" / "turning on again".
* Chatter vs extension: chatter = share of OFF -> ON gaps < 0.3 s; an ON logged again without an OFF is in neither
  numerator nor denominator, so extension cannot inflate it (extension stretches the OFF, it can only hide
  chatter). Check: chatter flags 0.19 % among detectors with >= 5 % ONs logged again vs 0.22 % among the rest.
  10055 det 5 (row 22): 0 of 168 ONs logged again in that 3 h; median gap 0.8 s - flicker while vehicles cross.
## (b) Severity rule (score_v4d.severity_v4d, applied to the final status)
* A: two independent findings -> bad. v4c needed two FAMILIES; now any two different checks (chatter + too fast,
  erratic + too many). Not counted, as before: stuck / silent exactly at the limit (.35). One quiet stretch seen by
  several checks (silent, count drop, misses at night, quiet unusual day) counts once.
* C: a Count zone held ON >= 30 min in light traffic is a counting fault even when extension (ONs logged again
  without OFF) explains it: watch -> suspect (2 detector-days in all).
* Modest = bad rate up <= 30 % and flagged rate up <= 5 % at every length. A+C: bad per 100 30 min .060 -> .062
  (+2.9 %), 3 h .516 -> .533 (+3.3 %), 24 h 1.638 -> 1.648 (+0.6 %); flagged unchanged (24 h 3.904 -> 3.911,
  +0.2 %). ADOPTED in the research scorer.
* Earlier user verdicts (20 bad / 12 fine detectors, s118c/cases_v4c.csv): bad 10 -> 12 (08073 d7, 11021 d3),
  fine still 0 flagged.
* Tested, not adopted: lowering the single-finding bad cut (bad at score .5: 24 h bad +53 %, 30 min +103 %;
  at .6: +14 % but moves row 16, which the user OK'd as suspect, and none of his rows).
* PROPOSED (P, switch in score_v4d): a finding that lasts >= 4 h of the day -> bad (busy at night 01-05 h, unusual
  day run >= 4 h, silent >= 4 h, unexplained ON in >= 16 periods). 24 h bad 1.638 -> 2.175 (+33 %, over the 30 %
  cutoff; 30 min / 3 h unchanged; flagged unchanged). User verdicts: bad 12 -> 16 of 20, fine 0. Moves rows 25,
  27, 28, 30 too. Applying it to 'misses at night' as well (+46 %) would go against his rows 9 / 10.
## Rows (s121/rows121.csv; old = sheet status)
* A: 12, 15, 20, 23 suspect -> bad (08073 det 7: P3 Advance; 11021 det 3: P2 Advance). C: 34 watch -> suspect
  (08019 det 3: P2 Count). Others unchanged.
* User leanings met: 15, 20, 23, 34 (and 12, same detector as 20). Not met: 22 (1 finding just past its limit,
  and his own extension doubt; extension ruled out above), 25 / 27 / 28 / 30 (only with P), 9 / 10 looser (no
  general rule found: a night-only cap would also move row 11, which he confirmed bad).
## Charts
40 + 40 PNGs re-drawn from s118c/plot (no re-scoring) -> s121/charts, s121/day_charts; review/ untouched, no new
sheet. Opened rows 15, 19, 22, 23, 34. Rerun: score_v4d.py then h121_charts.py.
