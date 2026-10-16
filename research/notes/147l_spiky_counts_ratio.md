# 147l Spiky counts as a ratio to the detector's own level (2026-10-16)
User: spike counts with GEH did not separate his verdicts (GEH inflates normal high-volume jitter). Research only;
replaces 147k, everything else = 147j. Code research/code/health147/h147l_spike.py, h147l_alloc.py (ModelL),
h147l_sheet.py (bins from h147k_feat). Work dc_work/s147: knobs147l.json, prop147l.csv, flagged147l.csv,
gallery/l0-l1.png. v12 rebuilt in place (backup _backup/thresholds_review_v12_147k.xlsx).
## Rule
10-min counts. Own level = median of the bins within +-1 h, leaving out the bin and its two neighbours (a burst does
not raise its own baseline). Expected = max(level, 1) x the phase mates' own rise at that bin (their count / their
level, only when they count >= 10, never below 1: mates rising together raise the bar, not excuse). Spike bin:
count >= 3.5 x expected AND count - expected >= 10 vehicles. Consecutive spike bins = one burst. Suspect >= 2
bursts a day, bad >= 3. (A single huge burst as bad did not help the verdicts; not used.)
## Fit (17 verdicts, spiky alone): grid 10 / 15 min x ratio 3-5 x excess 10-20 x counts x baseline floor 1-5
Best 15 / 17 at 1.46 % (10-min, 3.5, 10, 2 / 3). Excess 15 -> 14 / 17 at 0.92 %; a higher baseline floor drops the
low-volume bad (05032) and suspect (06033) cases. Misses: 2B497 d3 (no burst, user suspect), 2B348 d64 (7 bursts:
bad, user suspect). 15-min bins were worse.
## Final status vs verdicts (14 / 17)
BAD all 6 bad (2B028 d9 8 bursts, 03031 d28 4, 2B087 d11 7, 08097 d20 3, 10068 d25 3, 05032 d2 3).
SUSPECT: 10004 d23 2, 06033 d10 2 -> suspect; 2B497 d3 0 -> ok; 2B348 d64 7 -> bad.
OK: 03087, 08048, 2B062, 03049, 01072, 13005 -> ok; 10041 d48 -> suspect (too many, not spiky; user 'possibly
suspect').
## Flags
Spiky 1.47 % (suspect .75, bad .72); with too fast 13 / 26, too many 4 / 12, alone 202.
Gallery l (12 random flagged, not in the verdicts): 9 visibly spiky (ba04cbbd d15, 7cf0de1f d8, e4eb5afe d15,
81d590e1 d1, 936c0524 d13, e30e8317 d27, d0406b1c d8, 741ff056 d22, 7613b724 d58); 3 marginal low-volume
(e0917852 d1 night blips, 2ea53aba d20, 5d9c8e40 d2). Tightening (excess 15 or floor 3) removes them but also the
user's low-volume verdicts 05032 / 06033, so not tightened.
## Final status % (ok / suspect / bad / n.e.d.)
Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 | Proposed 93.54 / 2.45 / 2.34 / 1.66.
Flagged (suspect + bad) Old / New / Proposed: stuck 1.93 / .52 / 2.34; counts fall .67 / .22 / .72; night .62 /
.34 / .32; busy .91 / .81 / .64; too fast + many .26 / .17 / .21; erratic .20 / .27 / 0; spiky - / - / 1.47.
