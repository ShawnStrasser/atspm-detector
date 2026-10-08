# 96 — Context-aware health: occupancy, traffic reference, multi-flag resolver (2026-10-05; data study + proposal)
User feedback on health_review_v1 (4 ideas). Data = note-79 w40 windows (Sat 26 - Mon 28 Sep; 4 x 30 min, 2 x 3 h, 2 x 24 h),
763 training signals (folds_v4 minus locked_v2, asserted), 116,873 detector-windows. Health = the CURRENT package (v4f
health_core, tick fix) re-run with the note-79 inputs (stg OOF phase / function / lanes); hi-res log + classifier outputs only,
no prints / config / fault events. CPU 4 procs; model/ and packages untouched. Code `research/code/health/h96_{occ,health,
study,resolve,review}.py`; work `%DC_WORK%/health96/`. User answers (v1 rows 1-55, 60: 37 Y / 3 N / 16 ?) used ONLY to compare.
## Normal patterns per predicted type (24 h, status ok, >= 96 ONs; 15-min bins; traffic = phase's Advance + Count counts)
| type | median occ | occ at peak traffic (median / p95) | ON s median | corr counts / occ with traffic (median) | max 5-min p99.5 (1 / 2 lanes) |
|---|---|---|---|---|---|
| Presence | 16 % | 42 / 83 % | 11.7 | .94 / .91 | 83 / 87 |
| Other | 13 % | 33 / 82 % | 6.3 | .93 / .93 | 100 / 119 |
| Mid | 8 % | 20 / 56 % | 0.9 | .99 / .94 | 122 / 134 |
| Advance | 4 % | 10 / 50 % | 1.4 | .93 / .87 | 104 / 119 |
| Yellow_Red / Count / Bike | 0.5 / 0.6 / 0.8 % | 2 / 1 / 1 (p95 33 / 14 / 23) | 0.2 / 0.2 / 2.9 | .98 / .93 / .72 | 71 / 93 / 28 (YR 2-lane 139) |
* Occupancy of long zones (Presence / Other) climbs with signal congestion (index 1 -> 3+: 1 -> 33 %); count-type zones plateau ~1 %.
* Count-corr < .5 is rare (1-2 % of ok long zones); of those 40 % still follow traffic in OCCUPANCY (Advance / Count 3-5 %).
  The current "doesn't follow traffic" check fires mostly on long zones (155 of 219 h24 rows Presence / Other).
* Lanes raise the busiest 5 min by 15-40 %, not x2 (healthy 2-lane max ~140): a per-lane x2 limit is not supported.
* Idea 2 as a LEVEL check fails: own count / phase traffic varies over 2 orders of magnitude (p5 = .03-.06 x the type median for
  Advance / Count / Presence; a 10 %-of-typical rule fired on 12 per 100). Traffic works as a SHAPE reference only.
## Does the context separate the user's Y from N / ? (56 answered rows; tiny n, and the rules were shaped by these comments)
* Multi-flag (idea 4): >= 2 independent findings on 25 / 37 Y, 5 / 16 ?, 0 / 3 N.
* Occupancy follows traffic (ideas 2 / 3) on corr / erratic rows: the 3 ? (ETA / long zones, rows 37, 51, 53) .95-.98; the 7 Y -.29-.71.
* Stuck (idea 1): the ? (row 5, radar presence 21 min) had phase peers 3.5x their usual occupancy and traffic 1.9x usual; the 4 Y
  had peers ~1x or traffic stopped (row 2: reference 0 - a shared outage, not a queue).
* Type alone does not separate (long zones 18 / 37 Y vs 9 / 16 ?); lanes >= 2: 5 / 37 Y vs 5 / 16 ?; borderline (< 10 % over the
  limit): 3 Y vs 8 ? / N.
## Proposed logic (stage 1 = every package finding; stage 2 = resolver; then the package's score / grade2 on what is left)
R1 stuck on a long zone < 60 min while phase peers >= 1.5x their usual occupancy AND traffic >= .5x usual -> ok (queue).
R2 corr / erratic on a long zone whose 15-min occupancy correlates >= .8 with the phase traffic -> ok.
R3 count drops on a long zone whose occupancy share (vs traffic) stayed >= .5 of before -> ok.
R4 too many in 5 min, predicted >= 2 lanes, < 200, counts follow traffic (>= .5) -> ok.
R5 too-fast actuations with < 100 ONs while it still sees >= .1x its type's usual share of traffic -> watch.
R6 goes silent in a sample < 1 h -> watch.   R7 a single statistical finding < 10 % over its limit (score < .42) -> watch.
R8 too-short ONs note kept only on Presence (watch).   R9 two passes: detectors called bad in pass 1 are removed from their
phase mates' yardstick (found while spot-checking: 10062 d42 "silent" vs three broken Advance loops at 228 / 5 min all night).
N1 NEW watch note: count-type zone >= 30 min above its type + pulse/normal-mode p99.5 occupancy at that traffic level, signal
and phase not congested (peers < 40 % ON), occupancy not following traffic. WATCH = information, no status change.
## Fire rates per 100 detectors (suspect + bad; w40, 116,873 detector-windows; N1 alone .18 / .38 at 3 / 24 h)
| sample | before | after | bad before / after | + watch | cleared by (per 100) |
|---|---|---|---|---|---|
| 30 min | 0.60 | 0.40 | .24 / .16 | .23 | R5 .10, R7 .06, R6 .03 |
| 3 h | 1.83 | 1.40 | .81 / .73 | .47 | R7 .20, R2 .09, R5 .06, R9 .04, R1 .03 |
| 24 h | 5.20 | 3.73 | 1.91 / 1.80 | 1.12 | R7 .72, R1 .37, R9 .16, R2 .16, R5 .03, R3 .02, R4 .01 |
## Checks
* User answers (comparison only): flagged Y 37 -> 36 (row 9, borderline 20-min silence, -> watch), ? 15 -> 5, N 0 -> 0.
* note-79 evalset (positives mostly CIRCULAR tier C stuck / chatter): presumed-healthy flagged 3.38 -> 2.21 % at 24 h
  (-1.17 [-1.40,-0.97]), 1.18 -> 0.83 at 3 h; positives 19.9 -> 17.0 % at 24 h (-2.9 [-5.8,-0.9]), 10.7 -> 9.6 at 3 h.
  Per rule (share of positives among cleared vs kept flags): R7 2.6 vs 18.9 % (supported); R1 16.5 vs 21.9; R2 21.1 vs 21.5
  (shuffled-occupancy control 27.5 %); R3 17.4 vs 3.0 % (adverse, n 23). R1 clears 10 of the 14 lost positives (stuck class).
* Spot checks from raw events: 04035 d53 192 ONs in one 5 min; 10041 d8 no OFF 07:39-08:05; 2B535 d4 58 ONs, 14 % < 1 s;
  10062 d42 0 ONs 03:45-03:50. Charts opened (final sheet): rows 1, 6, 7, 9, 14, 17, 19 + day chart 2.
## Review sheet (`review/health_review_v2.xlsx`, 19 rows, charts `_v2_charts/` + `_v2_day_charts/`)
None of the 70 v1 detectors except row 19 (= v1 row 53, occupancy asked for). R1 x3, R2 x3, R3 x2, R4 x2, R5, R7 x2, R9 x2,
N1 x3; seed 96, one per signal. Chart = counts vs phase traffic (scaled, bad detectors left out) + % time ON vs phase peers.
## For the orchestrator
* Nothing changed in any package. ADOPTED (orchestrator): R7 + R9 for the next package build. R1 / R2 / R4 / N1 wait for
  the v2 answers; R3 dropped unless v2 supports it. Unanswered v1 rows (56-59, 61-70) = not reviewed (no evidence either way);
  no number above used them. Open: the gate's 'ON almost all the time' also hits presence zones (rows 66-70), not changed.
