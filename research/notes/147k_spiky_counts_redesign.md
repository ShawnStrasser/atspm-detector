# 147k Spiky counts redesigned from the user's day-chart verdicts (2026-10-15)
Research only (src/ untouched). Replaces the 147f-j step-z measure; everything else = 147j. Code
research/code/health147/h147k_feat.py (10/15-min counts own + phase mates, 2 min), h147k_spike.py, h147k_alloc.py,
h147k_sheet.py. Work dc_work/s147: bins147k.npz, binkeys147k.parquet, knobs147k.json, prop147k.csv,
flagged147k.csv, gallery/ (g, h, k pngs). v12 rebuilt in place (backup _backup/thresholds_review_v12_147j.xlsx).
## Rule
10-min counts. Spike at bin t: count >= 1.3 x expected, GEH(count, expected) >= 4 and count >= 10, where expected =
the larger of the two neighbouring bins x the phase mates' own rise at t (their count / the larger of their
neighbours, >= 1, only when the mates count >= 10). = a one-bin up-and-down excursion above BOTH sides that the
mates do not explain; a jump shared with the mates counts only beyond the mates' rise. Suspect >= 2 spikes a day,
bad >= 4.
## How it was fitted (10 verdicts; h147k_spike)
Tried 10 / 15-min bins, GEH 3-6, ratio 1.3-2.5, minimum count 10-20, three baselines: (1) median of 4 neighbours
flagged ramps and gap edges (gallery g: 5 of 12 false), (2) max of both neighbours + 1.5x bar when mates spike
still flagged shared jumps (gallery h), (3) neighbours x mates' rise (chosen). 15-min bins matched 7 / 10, 10-min
8 / 10 at the fewest flags (>= 2 spikes 1.47 % vs 15-min 1.14 % with fewer matches).
## Verdict match (spiky alone -> final)
BAD: 2B028 d9 12 spikes bad -> bad; 03031 d28 4 bad -> bad; 2B087 d11 3 suspect -> bad (+ too fast);
08097 d20 4 bad -> bad; 10041 d48 1 none -> suspect (too many only) MISS.
SUSPECT: 2B497 d3 2 -> suspect; 2B348 d64 2 -> suspect; 2B062 d23 1 -> ok MISS.  OK: 03087 d53 1 -> ok;
08048 d15 0 -> ok.  8 / 10.
## Flags
Spiky 1.48 % (suspect 1.09, bad 0.38); with too fast 15 / 26, too many 4 / 12, on its own 201.
Gallery k (12 random flagged, not in the verdicts): 8 clearly spiky (night bursts 80dfed92 d26 / d48, 51ab9823 d25,
9feb43f4 d23, ba04cbbd d1, e4eb5afe d23, 2f75782b d2, 741ff056 d1); 4 mild (5c5fb21a d56, a51a0dda d58, d782763f d2,
07e51479 d10 low volume): a minimum count of 15 would drop some but loses 2B348 (suspect verdict).
## Final status % (ok / suspect / bad / n.e.d.)
Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 | Proposed 93.47 / 2.69 / 2.17 / 1.66.
Flagged (suspect + bad) Old / New / Proposed: stuck 1.93 / .52 / 2.34; counts fall .67 / .22 / .70; night .62 /
.34 / .32; busy .91 / .81 / .64; too fast + many .26 / .17 / .21; erratic .20 / .27 / 0; spiky - / - / 1.48.
