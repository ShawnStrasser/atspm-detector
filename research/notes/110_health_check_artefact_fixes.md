# 110 — Health v110: six check artefacts from the 108b charts fixed; review sheet rebuilt (2026-10-06)
Orchestrator task after note 108b. Base = v108 (per-type p99.8 limits, profile p99.5; h108_resolve). Same w40 data (763
training signals, locked_v2 asserted absent), hi-res log + classifier outputs only, CPU <= 4 procs. Code
`research/code/health/h110_{events,resolve,review}.py`; work `%DC_WORK%/health110/`. Nothing adopted into a package.
## Checks first
New event pass (`h110_events`, 251 s): package chop15 recomputed = saved 100 %, level ratio 100 %, longest episode >= 5 min
= saved 97.5 %, per-episode queue context (mates' x, r) = saved 100 %. v110 code with every fix off reproduces v108
status on 115,518 / 115,518 detector-windows.
## Fixes (each switchable; `fix_rates110.csv`)
F1 erratic yardstick: >= 2 non-twin phase mates (as before), else a non-twin phase mate of the same function, else >= 2
   same-function detectors elsewhere on the signal, else none -> not scored; watch 'no yardstick' if the old whole-signal
   value fires. Reference mix (24 h): phase 73 %, same-fn mate 2.3 %, same-fn signal 22 %, none 2.4 %. Limits re-fitted.
F2 count drops, time-of-day aware: expected = rest of signal x the hourly relative share of healthy detectors of the
   same type + band + day (measured, `tod_w.parquet`); cut needs >= 2 h each side (1/4 of a short sample) AND the
   detector's OWN counts per 15 min at least halving. Package LLR > 50 / .15 / .05 kept. The type weights alone did not
   clear the night-busy rows (2B502 d4 ratio .11 -> .46 only with a 10 %-support guard that also lost the real evening
   failure 2B316 d3); the own-count condition separates them (2B316 .07 kept, 2B502 / 08052 / 2B069 cleared).
F3 stuck: every continuous ON over the per-type limit counts: total time (suspect at the limit, bad at 60 min) or
   number of such ONs > p99.8 of healthy per type (h24: Presence-1 2, Other-2+ 29, most others 0-1); shared (>= 3 others)
   ONs dropped; 'recovered -> suspect at most' only for a single ON. ONs still ON at the window end now count (the v108
   longest-ON statistic missed them: m30 +.04 / 100).
F4 queue per episode; > 60 min cleared (Q1b) when Q1 holds AND a healthy mate is queued (15-min % ON >= p95 of healthy
   detectors of the mate's type: Presence-1 67 %, Advance-1 30 %, Count-1 6 % ...) in >= 75 % of the ON. Orchestrator
   (after the first sheet): EVERY queue clear also needs traffic - the phase's Advance / Count (else other mates) count
   >= 0.5x their usual in the 15-min periods fully inside the ON; a shared ON that is not a queue holds at suspect.
   It blocks 108 of 2,331 otherwise queue-qualified ONs (22 of 64 over 60 min); Q1b clears 9 detector-windows (24 h).
F5 function probability < .7 (14-21 % of detector-windows): every limit (incl. profile, stuck episodes) = least strict
   over classes with p >= .15, same lane span; the sheet names the type used and the model's own-type limit.
F6 profile: finding from 12 h; 3-12 h watch only (3-h partial profile, hours renormalised, p99.5 per type x band x day).
   On w40 the 3-h windows (Sun 12-15, Mon 06-09) are scored for 86 %; 1.38 / 100 get the watch. No 6-12 h windows here.
## Fire rates per 100 detector-windows, suspect + bad (bad) [watch]: v108 -> v110
30 min .588 (.113) [.127] -> .554 (.104) [.125]; 3 h 1.483 (.629) [.458] -> 1.404 (.598) [1.456]; 24 h 3.958 (1.692)
[1.685] -> 3.796 (1.648) [1.662] (with the traffic rule; 3.691 before it). Newly flagged / unflagged: 24 h +.105 / -.267,
3 h +.048 / -.126, 30 min +.039 / -.072. Per fix (24 h, + / -): F1 .020 / .020 (erratic fires .675 -> .605; Y1 watch
.068); F2 0 / .088 (count-drop fires .534 -> .314; 3 h .082 -> .062); F3 .084 / .003 (bad from stuck 44 -> 88
detector-windows: several ONs add up); F4 0 / .017; F5 0 / .135 (3 h .089, 30 min .069); F6 watch only (3 h +.93).
## User's answered rows (sheet rows 1-12) and earlier answers
Changed: row 3 08CM405 d37 suspect -> bad (30 + 56 + 43 = 129 min; he said 'stuck several times, flagged once' but
'suspect' label correct - ask); row 5 watch -> ok (Count 57 / YR 42 %: YR limit 36.6 % > 27.4 %); row 6 erratic gone
(same-function yardstick 12.2 < 15.0), still suspect on volume; row 7 suspect -> ok (own counts 63 -> 91 per 15 min);
row 9 bad -> suspect (121-min ON = Q1b queue: mates queued 100 %, their Count zones 1.84x busier; profile remains); row 12 suspect -> bad (23 + 26 min,
two families). Rows 1, 2, 4, 8, 10, 11 unchanged. v1 answers: Y flagged 34 / 37 unchanged, N 0 / 3, '?' 7 -> 6 / 16;
v2 OK'd keep the OK'd status 4 / 8 unchanged; v2 row 3 'No!' suspect -> bad (his direction). Costs seen: v1 row 31
(Y, count drop) loses the count-drop finding, stays suspect on stuck; row 17 11021 d2 becomes bad because the re-fitted
erratic limit (13.0) is just under its 13.1.
## Review sheet (review/health_review_v3.xlsx, 36 rows; backup review/_backup/health_review_v3_20261006_175608.xlsx)
Same 30 rows + 6 new examples (31 Y1 no yardstick, 32 F2 night-busy cleared, 33 F3 four ONs = 109 min -> bad, 34 a
112-min ON NOT cleared: whole phase ON, Advance det 25 at 0.04x traffic -> suspect (shared), 35 F5, 36 F6 3-h watch). Header rewritten for the changed checks; rows that changed say 'CHANGED since
the last sheet (was ...)'; 'Our reply' updated for rows 3 (incl. why bad: ONs summed), 5, 6, 7, 9, 12. Method of 108b: 127 / 127
recomputed values = resolver (auto_check.csv), 36 / 36 PNGs opened (stuck rows re-opened after the traffic rule), 160 claims logged `%DC_WORK%/health110/qa_v110.csv`; fixed during QA:
stem labels numbered + listed, own-type F5 limit line, mixed-source F5 limits not relabelled, clipped mark label,
'first pass' wording, numbers not on the chart removed. User entries 12 / 12 carried, verified cell by cell.
## For the orchestrator
* Decided (orchestrator): queue clears need traffic (done, row 34 now suspect); 3-h profile stays watch-only.
* F2 misses partial failures that start at night (no qualifying 'before'); total silences are still caught by 'goes silent'.
* Nothing adopted; health rules still wait on the user's answers (now rows 3, 9, 12, 13-36).
