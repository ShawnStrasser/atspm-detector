# 159 — Queue-discharge (shockwave) timing as a setback estimate (study only, 2026-11-16)
User idea: a car standing on an advance detector at green start moves when the discharge wave reaches it, so
`release = lost + d / wave` and `d = wave * (release - lost)`, no approach speed needed. Nothing in `src/` changed.
Code `research/code/shock159/sw1_extract.py`, `sw2_analyze.py` -> `%DC_WORK%/s159/` (cycles, dets, per_det, results.json, sw2.log).
**Data.** Note-58 truth frame, full 66 h (Fri 16:15 - Mon 10:24, mostly weekend): printed single distance, print Advance 628
(loop 417, video 140, radar 71) + Mid 109, 129 signals, locked_v2 asserted absent. Phase = the frame's PREDICTED phase.
Cycle = ON > 6 s before the green and still ON at it; release = green begin -> OFF. Kept: ON spans one green only, began
after the previous green ended, release <= 60 s (drops 1.5 % / 1.0 % / 0.1 %). 79,817 cycles, 581 detectors. Per detector:
median release. Fit = Theil-Sen on single-lane loop Advance with >= 10 cycles, on five folds; estimate on the sixth.
"Current" = note 58's OOF P50 `all` (the shipped recipe; the package itself is in-sample on these signals).
**Fit (211 loops).** release = **3.3 s + d / 27.4 ft/s** (95 % CI 25.8-29.2; folds 26.5-28.7 ft/s, 3.0-3.5 s), r = .86,
residual MAD 0.9 s. The 3.3 s includes clearing the loop, so it is not pure start-up lost time; with 2 s forced the
distance groups imply 19-25 ft/s. Video: r = -.04 (wide zones stay ON while the queue passes; cycle IQR 7.4 s vs 2.5 s
for loops). Radar: 13 detectors qualify, releases ~8 s early. **The line exists for loops only.**
**Held-out error, ft** — medAE / middle half of signed error / % within 50 ft (n)
| set | shockwave alone | current | blend |
|---|---|---|---|
| loop, 1 lane, >= 10 cycles (211) | 25 / -24..+27 / 83 | 11 / -18..+7 / 80 | 9.5 / -14..+5 / 86 |
| ... <= 100 ft (95) | 20 / 95 | 1.7 / 99 | 1.7 / 100 |
| ... 101-200 ft (59) | 26 / 86 | 20 / 81 | 12 / 88 |
| ... > 200 ft (57) | 40, bias +7 / 61 | 53, bias -53 / 46 | 32 / 60 |
| video >= 10 cycles (129) | 70 / 40 | 30 / 68 | 31 / 66 |
| radar >= 5 cycles (14) | 232 / 7 | 96 / 29 | 105 / 29 |
| all 628 printed Advance | (473 answered) 37 / 62 | 20.4 / 70 | 18.3 / 72 |
| Mid, all 109 | 23 / 73 | 23.9 / 94 (within 25: 50) | 14.2 / 94 (within 25: 74) |
Blend = behaviour-only rule (no technology): >= 10 cycles AND cycle IQR <= 4 s AND current > 100 ft -> mean of the two,
else current (thresholds picked on all folds: disclosed). It fires on 121 of 628 (103 loop, 9 video, 9 radar).
Paired signal bootstrap, blend - current: all 628 medAE -2.1 [-4.7, 0.0], within-50 +1.6 [-0.5, +4.0]; where it fires
-6.6 [-17.5, -0.4]; > 200 ft -4.2 [-17.9, +1.7]; **Mid -9.6 [-14.8, -3.6]**. Shockwave alone vs current on loops +14 ft worse.
**Inside the learned model** (note-58 recipe + 5 features: n, median, q25, IQR of release, share with a follower <= 4 s;
OOF, 3 seeds): all 628 20.4 -> 19.6 ft, within-50 70 -> 73 (+3.0 [+0.3, +5.8]; permuted-feature control +0.3 [-1.2,
+1.9]; single seeds 3-5: base 69/71/69, with 73/73/72); >= 10 cycles within-50 +4.4 [+1.3, +7.7]; > 200 ft 51 -> 46
(seeds 53/51/55 -> 45/48/48; CI -15.6..+3.3); <= 100 ft unchanged; < 10 cycles 27 -> 32 medAE (within-50 same, noise);
Mid 23.9 -> 16.1. The existing `rel_med` feature (release without the 6 s hold) has rank corr .11 with the new median.
**Cycles needed** (112 loops with >= 50, k drawn at random): spread of the estimate 49 ft at 1 cycle, 25 at 5, 17 at 10,
11 at 20; error vs print 39 / 27 / 24 / 23 ft -> **10 cycles**. Share of printed Advance reaching 10: 66 h loops 53 %
(video 92 %, radar 18 %, all 58 %); Saturday 24 h loops 38 %; weekday 18 h (Fri eve + Mon morning) loops 39 %. Loops by
distance: <= 100 ft 68 %, 101-200 59 %, 201-300 47 %, > 300 ft 30 % — the far detectors, which need it most, get it least.
**Is the wave speed stable?** Per-detector d / (release - 3.3): median 27.4, middle half 23-34 ft/s. By printed distance
class (the agency sets distance by approach speed = the only speed-limit proxy): 26.9 / 29.3 / 26.2 / 28.2 / 26.9, flat.
Lanes on the phase 1 / 2 / 3: 30.7 / 25.5 / 28.3, no trend. Left-turn lanes release ~0.6 s earlier than through. Free-flow
speed terciles (own ON time): no trend. Weekday peak vs other hours, same detector: +0.3 s. Between-site share of the
residual variance 26 % (within-site sd 1.16 s vs 1.35 total): one constant serves every site. **Posted speed and grade
are not in any table here — not tested.**
**Zones over more than one lane.** Advance loops over 2+ lanes (10): residual +0.1 s, MAD 0.7 s (1 lane -0.1 / 0.9):
same line, no extra scatter, 20 ft. Mid loops over 2+ lanes (56): -0.2 s, MAD 0.4 s, shockwave alone 12 ft vs current
23. Radar over 2+ lanes (9): 8.5 s early, 220 ft. Video 2+ lanes: 2 detectors, nothing to say.
**Not a queue / part discharge.** Per cycle on loops: 58 % within 2 s of the line, 81 % within 4 s, 14 % more than 4 s
early (moved before the wave could arrive: creeping, right on red, no standing queue ahead), 4 % more than 4 s late
(blocked, part discharge); 2 % release after the green ended. Having no follower within 4 s changes nothing. The median
absorbs all of it. Detectors > 5 s off the line: 4 of 211 — two at one signal printed 460 ft that release like ~80 ft
(the model also says ~146 ft and calls them Presence: print distance suspect), one 75 ft left-turn at 18 s (released by
another phase than the predicted one), one 225 ft with exactly 10 cycles. Doubles as a print / label check.
**Verdict.** Not a replacement (25 ft vs 11 ft on loops; useless on video and radar; the current model already nails
the 75 ft standard). Worth adding as FEATURES to the setback model at its next refit: clear gain for Mid (24 -> 14-16
ft), a consistent but small one for Advance (+3 pt within 50 ft, mostly beyond 100 ft where the current estimate reads
~50 ft short), no harm elsewhere, one extra aggregate over tables setback.py already builds. It is also the only part
that carries to another agency unchanged. Open before a build: weekday data, the m30 / h6 / h24 length groups (few
detectors reach 10 cycles under 24 h), radar gate. Package untouched; nothing committed.
