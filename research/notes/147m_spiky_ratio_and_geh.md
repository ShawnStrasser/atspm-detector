# 147m Spiky counts = ratio AND GEH, low-volume capped (2026-10-16)
147l (ratio only) too sensitive at low volume, 147k (GEH only) to high-volume jitter. Research only; replaces 147l,
everything else = 147j. Code research/code/health147/h147l_spike.events2, h147m_alloc.py (ModelM), h147m_sheet.py.
Work dc_work/s147: knobs147m.json, prop147m.csv, flagged147m.csv, gallery/m0-m1.png. v12 rebuilt in place
(backup _backup/thresholds_review_v12_147l.xlsx).
## Rule
10-min counts; own level = median of the bins within +-1 h leaving out the bin and its neighbours; expected =
max(level, 1) x the phase mates' own rise (mates >= 10, never below 1). Spike bin: count >= 3 x expected AND
GEH(count, expected) >= 5; consecutive spike bins = one burst. Suspect >= 2 bursts a day, bad >= 3.
Low volume: < 200 vehicles in the day -> never flagged; < 500 -> suspect at most unless a burst reaches GEH >= 10
(unmistakable bursts, e.g. night spikes on a flat signal). A 'mates flat' test (mates < 10 or rise < 1.3) was true
for almost every burst, so it could not separate; GEH >= 10 is used as the exception instead.
## Fit (22 verdicts + 10041 either; grid ratio 2.5-5 x GEH 3-6 x 1-4 bursts x low-volume rules)
Ratio 3 + GEH 5 + 2 / 3 bursts: 0.95 % flagged; ratio 3.5 + GEH 3 matched one more on spiky alone but 2.5 %.
## Final status vs verdicts: 22 / 23
BAD: 2B028 d9 (8 bursts), 03031 d28 (5), 2B087 d11 (4, + too fast), 08097 d20 (3), 10068 d25 (3), 08170 d15 (3)
-> bad; 05032 d2 (1 burst: night blips GEH < 5) -> bad through goes silent + stuck, not spiky.
SUSPECT: 10004 d23, 08117 d2, 2B348 d64 (423 / day, capped), 08052 d1 (2 bursts each) -> suspect; 06033 d10
(Presence 2 lanes, 302 / day, 1 burst) -> ok: LOST (its blips are GEH < 5).
OK: all 10 ok (03123 d58 74 / day not flagged; 05026 d1 0 bursts); 10041 d48 suspect (too many) = allowed.
## Flags
Spiky 0.94 % (suspect .51, bad .43); with too fast 13 / 26, too many 3 / 12, alone 124.
Gallery m (12 random new flags): 10 visibly spiky (faf66ec4 d11 Bike bursts, 91ad0128 d9, f206459b d27, 258c4613 d1,
768b7be8 d8 night, cd8fc58c d13, d75e023f d7, 0c45d4b2 d8, 4da13b62 d21, 53576864 d23); 2 marginal (53576864 d8
jittery Presence, 3aaeb193 d15 peak on the morning ramp). Not tightened (next step would lose 08117 / 08052).
## Final status % (ok / suspect / bad / n.e.d.)
Old 94.37 / 2.07 / 1.93 / 1.63 | New 96.18 / .83 / 1.34 / 1.65 | Proposed 93.96 / 2.28 / 2.11 / 1.66.
Flagged (suspect + bad) Old / New / Proposed: stuck 1.93 / .52 / 2.34; counts fall .67 / .22 / .70; night .62 /
.34 / .32; busy .91 / .81 / .64; too fast + many .26 / .17 / .21; erratic .20 / .27 / 0; spiky - / - / .94.
