# 111 — Package v6 = note-109 design C (2026-10-06) — candidate package, nothing shipped (model/ = final_v2)
Brief (orchestrator, after note 109): build design C as a package, prove parity, score it from saved OOF, bench it vs beta and
v5c. CPU <= 4-6 threads, no new GPU training, locked_v2 asserted absent everywhere. Code `research/code/final111/` (fit111,
assemble111, parity111, score111); work `%DC_WORK%/s111/`, fits `final_v3_work/v3fit111/`, package `final_v3_candidate_v6`.
## Design C (copy of v5c + v5b_fast profile code; v5c untouched)
* ONE network kind: the three w32 siba members (x100_w32full) give phase (their pair phase head, on candidates with ranker
  p >= .01) AND function, in one shared pass. The separate phase TCN is gone.
* Every tree stage ONE seed: ranker s0, function trees s0, stacker s0 (mean3 / single / nonet), setback P50 s0 per length group
  (the existing note-95 seed-0 full-data fits = 1-seed refits of the same recipe). Decoder refit on ranker-s0 0.5 / siba head
  0.5 (351 trees); fast-profile decoder `decode_trees` on ranker s0 alone (501 trees, six-fold OOF in s111/phase/q111).
* Kept: lanes D (v5c, note 105), health, setback band / pair models, decode rules.
* Trained models on the default path 21 vs v5c 36 (net 3 + ranker 1 + decoder 1 + trees 1 + stacker 1 + lanes 3 + setback 11);
  fast-profile fallbacks (decode_trees, stacker single / nonet) on top. Weights 9.2 MB vs 19 MB.
## Parity
siba phase head (package ONNX) vs torch checkpoints, 4 bench signals x 30 min / 3 h, 120 pieces, unfiltered and random filter:
max |diff| 9.5e-6 (logits), phase probabilities 8.5e-7 (member) / 4.4e-7 (3-member, filtered), function 1.0e-6; 0 winners differ
-> PASS (1e-4). Tree ONNX parity <= 7.5e-16. check.py 5/5 PASS after --freeze (stored answers, phase / channel renumbering,
no torch / lightgbm / scipy / sklearn, no-network profile).
## OOF (six folds, 2026 pool, Sept-2026 rows; phase 761 signals / function 642 at >= 30 min). dE pt vs v5c [CI], * = CI excludes 0
| | 5 min | 10 min | 30 min | 1 h | 3 h | 24 h | >= 30 E | >= 30 R |
|---|---|---|---|---|---|---|---|---|
| phase v6 (acc) | .9676 | .9763 | .9812 | .9837 | .9859 | .9872 | .9846 [.9815,.9872] | .9864 |
| phase v6 vs v5c | +0.36* | +0.24* | +0.03 | +0.00 | +0.04 | +0.06 | +0.03 [-0.03,+0.09] | +0.01 |
| function v6 (acc) | .9024 | .9102 | .9250 | .9246 | .9336 | .9365 | .9307 [.9216,.9389] | .9397 |
| function v6 vs v5c | -0.16* | +0.00 | +0.02 | -0.03 | -0.04 | +0.02 | -0.01 [-0.06,+0.05] | -0.01 |
Fast setting le2h (network only up to 2 h; above: decode_trees + stacker nonet): phase >= 30 E .9827, vs v5c le2h +0.04
[+0.00,+0.09]; function .9267, vs v5c le2h -0.08 [-0.15,-0.02]* (3 h -0.23*, 24 h -0.16*: the 1-seed no-network stacker is
weaker than v5c's 3-seed one; <= 2 h identical to v6 full). vs v5c full: phase -0.16*, function -0.40*.
## Bench (bench98: fresh process per case, 4 threads, warm median of 3, 2 passes; machine 7-20 % busy). r8 / r11 / busiest ev / ch
| warm s, peak MB | 30 min | 3 h | 24 h |
|---|---|---|---|
| beta (final_v2) | 0.96 / 0.72 / 1.42 / 1.70 ; 582-1560 | 0.45 / 0.39 / 0.56 / 0.52 ; 259-299 | 0.81 / 0.56 / 1.11 / 1.01 ; 296-396 |
| v5c | 0.72 / 0.61 / 0.87 / 0.99 ; 402-541 | 1.34 / 1.01 / 1.60 / 1.78 ; 427-581 | 1.93 / 1.32 / 2.91 / 2.96 ; 455-686 |
| v6 full | 0.60 / 0.54 / 0.73 / 0.82 ; 310-392 | 0.99 / 0.76 / 1.28 / 1.41 ; 310-414 | 1.72 / 1.10 / 2.65 / 2.47 ; 355-490 |
| v6 le2h (fast) | = v6 full | 0.68 / 0.56 / 0.93 / 0.89 ; 249-279 | 1.38 / 0.87 / 2.25 / 1.93 ; 292-411 |
3 h: v6 vs v5c time -20..-26 %, peak -23..-29 %; v6 = 1.95-2.71x beta time, 1.20-1.46x peak; le2h = 1.44-1.71x beta time,
0.93-1.01x peak (within the user's 2x rule; v6 full is not, at 3 of 4 signals). 30 min: v6 faster and leaner than beta.
24 h: v6 1.96-2.45x beta, le2h 1.55-2.03x. (Note 109's -27..-30 % was on a v5b_fast copy with profile switches; the real
package saves a little less at 3 h, more RAM.)
## Verdict
v6 ties v5c on accuracy (phase +0.03, function -0.01 at >= 30 min; 5 min phase +0.36*, function -0.16*) with 21 instead of 36
trained models, one network kind, 20-26 % less time and ~25 % less RAM at 3 h; parity and check.py pass -> v6 is the candidate
in place of v5c. Fast setting le2h is within 2x of beta at every 3 h signal at -0.40 function / -0.16 phase vs v5c full (>= 30 min).
Open (orchestrator / user): full vs le2h as the default profile. Exam signals not opened.
