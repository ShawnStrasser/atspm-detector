# 136 asclog (1).db: every detector's labels justified, full log vs last 12 h, too-fast verdicts (2026-10-12)
Input: the user's Econolite ASC SQLite log (copied to dc_work/s136/raw; original untouched), 5 day tables, local =
Pacific (event_2026_09_26 starts 00:00:08 PDT), Timestamp = epoch + Tick/10. 1,247,419 events, Fri 25 Sep 11:22 - Tue
29 Sep 13:54 (98.5 h); last 12 h = 01:55-13:54 Tue (160,715). No config table in the db. Code research/code/asclog136
(a136_events, a136_run, a136_evidence, a136_fast, a136_fastbins, a136_sheet); out dc_work/s136. Fault events (83) only as
a cross-check / chart mark. Packages: 1.0.0 = dc_work/bak_1_0_1/src, 1.0.1 = repo src. Nothing committed.
## Runs (1.0.0 vs 1.0.1)
Phase, function, lanes, setback identical between versions on both samples. Only change: det 16 full log bad -> suspect
(note-135 rule F, steady flicker on a Presence zone). 39 detector rows full / 30 in 12 h (9 unused inputs absent).
## Signal layout found in the raw log (evidence: zero-lag ON starts, ON-time overlap, calls 43 at the same tick)
P1/P5 lefts, P2/P6 coordinated main street (green 93 % of 00-05 h), P4/P8 side street 2 lanes each.
- Stop-bar Presence + Count pairs: P1 1+13, P4 8+9 / 10+11, P8 24+25 / 22+23, P2 2+3, P6 16+17; Advance 49 (P2), 50 (P6).
- Channels 53-60 are 20-s splits of 2/49/16/50: 54/56/58/60 = parent from 20 s into green to end (min 20.1 s, never in
  red); 53/55/57/59 = parent at all other times (max 20.0 s into green), these place the phase calls (43). Union = parent
  (99.8 % of ON time); copy starts 0.1 s after the parent. Model: all Other. Correct as stacked copies.
- 32 = copy of det 1 (99 % ON time inside), places P2 calls (49 % of its red ONs; det 1 2 %) -> model P2 (.87); 31 = det 1
  only from 18 s into P2 green; 36 = copy of det 15, also calls P6 (52 %); 35 = det 15 only during P2/P6 green.
- det 15/36: 5,700 ONs/day vs ~300/day on P5 count det 27; 80 % of det 15 ONs are followed by a P2 count (det 3) within
  0.5-4 s (chance 32 %), 2 % by det 27. P5 greens median 6 s, 88 % with no det 27 pulse -> zone sees P2 through traffic;
  likely phantom P5 calls. Flagged for the user, not a model question.
## Full vs last 12 h (1.0.1) - every difference and why
- det 15 Other -> Presence, det 35 Presence -> Other: each detector's OWN argmax flips (per-lane decode did not act;
  function_guess = argmax p_*). Full: det 15 p_presence .13 / p_other .82, det 35 .54. 12 h: det 15 .58, det 35 7 ONs.
- det 16 (1.0.0 only): full bad (1,938 back-to-back ONs vs limit 1,014) vs 12 h suspect (302 vs 282): limits scale with
  sample and traffic; 1.0.1 gives suspect on both.
- det 17 watch (24 ONs > 5 s on a pulse zone), 53 watch (night), 54 / 58 watch (too-fast): borderline only with 4 days of
  evidence; ok in 12 h. det 57 bad on both (full: bad findings; 12 h: two suspect findings agree).
- det 31 / 35 too few ONs in 12 h (15 / 7). Setback 49 / 50: 252 / 245 ft -> 222 / 216 ft (within setback error).
- det 9 / 11 stuck on (Tue 12:44) is inside both samples. 12 h has no whole day: time-of-day checks cannot run (none fired
  on the full log either).
## Too-fast verdicts (fast = ON < 1 s after previous ON, re-trigger < 0.3 s after OFF excluded, as the package)
- det 16 P6 Presence: REAL zone/setup problem. Red 1,136 vs 233 expected (4.9x), green 798 vs 413; twin det 2 (P2) red
  363 vs 196, green 1,716 vs 2,050. Fast share in red 27 % vs det 2 10 %; green 9.1 vs 8.5 %. Prev ON 0.3 s, gap 0.5 s;
  88 % of fast ONs have no new det 17 pulse (NOT diagnostic: det 2's fast red ONs 95 %, slow red ONs 72 %), 98 % advance;
  night share as day; steady every day (14.4-15.2 %); 2 ON-without-OFF in 4 days. Also counts only ~70 % of det 17's cars
  by day, ~25 % at night (det 2 / det 3 ~94 %). Not machine-like, not heavy traffic. 1.0.1 suspect wording fits; field check.
- det 57 (copy of 16): inherits the flicker (red 1,061 vs 207). Its 'misses vehicles at night' is the 20-s split: P6
  rests in green, so det 16's night ONs go to det 58. No own fault; stays bad in 1.0.1 (two findings; rule F needs the
  too-fast finding alone).
- det 58 / 54: the 20-s split, NOT det 16's flicker (verify pass): yardstick on green > 20 s only gives 58: 156 vs ~197,
  54: 937 vs ~1,403 (below chance); det 58 fast share 5.0 %, det 16 in late green 5.4 % (162 vs 180). No fault.
- Cabinet note (cross-check): 'detector restored' for every input in groups of 4, Tue 12:41-12:47 and 13:15-13:19; dets
  8-11 came ON within 0.1 s at 12:44:11 -> det 9 / 11 'stuck on' = card restarts. Fast share after 12:51 lower (16: 9.5 %,
  326 ONs) - too short to say the zone was changed.
## Run time (1.0.1, warm, threads 4; scale.csv)
1 h .72-.88 s, 3 h .99-1.2, 6 h 1.05-1.17, 12 h 1.18-1.33, 24 h 1.78-1.87, 48 h 2.3-2.7, 98.5 h 3.4-3.6 s; cold first call
+0.4-0.7 s. 8x the data (12 -> 98.5 h) = 2.9x the time. Features 0.29 -> 1.32 s (~linear in events), model scoring
.30-.40 -> .41-.48 s (flat: network runs on pieces), function .15 -> .42, health .13 -> .55 s (sub-linear, fixed part).
1.0.0 same within 0.1 s.
## Sheet
review/asclog_review.xlsx, 39 rows (every detector), Why <= 20 words, Doubt where needed; review/asclog_charts: 30 day
charts (Mon 28 Sep, detector + phase mates, copies with their parent), 4 too-fast charts (by hour, red vs green, vs the
package's random-arrival yardstick, twin approach below), 2 stuck-on timelines. Every chart opened and checked vs its row.
Doubtful labels: det 35 (should be Other), det 15 (P5 Presence; zone sees through traffic), det 32 (P1 zone, P2 calls),
det 36 (calls P6), det 27 (phase .84).
