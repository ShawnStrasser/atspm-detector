# 67 — Six-fold decider (stacker) re-test; Count / Presence regression vs final_v2; decode demotions (2026-10-01)
Base = current function (cand64 + note 65): trees 229 x 3 seeds, fj 3 seeds averaged, tree weight 0.6 -> D-lane stack-pick
decode -> short twin decode. Reproduced exactly (cand64 ok_*_fj identical on all 261,528 E rows; >= 30 min .9048 E).
Six folds folds_v4, CPU 4 threads, locked_v2 asserted absent. Headline = >= 30-min pool (187,980 E rows, 652 signals); pt
[95 % paired signal-bootstrap CI]. Code `evaluation/s67_decider.py` (base / diag / stack / fix / demote); work `%DC_WORK%/s67/`.
## A. Decider, six folds (stacker for fold k trained only on folds != k OOF rows; fixed shallow LGB as note 63, 3 seeds)
| arm (vs fixed 0.6 blend) | >= 30 min E | >= 30 min R | 5 min E | 10 min E | folds >= 30 E |
|---|---|---|---|---|---|
| probs-only stacker (seed avg) | +0.26 [+0.10,+0.43] | +0.16 [+0.01,+0.32] | +0.25 | +0.31 | +.24 +.26 +.70 +.06 -.08 +.48 |
| **context stacker** (seed avg) | **+0.48 [+0.28,+0.68]** | **+0.35 [+0.16,+0.54]** | +0.57 | +0.53 | +.59 +.72 +.70 +.20 +.24 +.52 |
| control: context shuffled in window group | +0.23 [+0.06,+0.40] | +0.13 [-0.03,+0.28] | +0.20 | +0.33 | = probs-only |
* Seeds: context +.48/+.48/+.50, probs +.24/+.26/+.26, shuffled +.22/+.23/+.22; context - shuffled +0.26 [+0.13,+0.38] / +0.25 / +0.28 E.
* Context = 42 hi-res columns: health core + stack-relative chi (function-free, note 59 c), pick / stack-membership inputs
  (span, unhealthy, track, number of span / co-located peers), D lanes (lanes spanned, lane id, phase n_lanes + conf, lane
  conf), phase-mates' best ATSPM probs, own rank in the phase, lane-mates' max ATSPM probs, log actuations. No label input.
* Caveat (standard OOF stacking): base OOF probs of fold j came from models trained on folds that include k. No nested TCN
  OOF exists. Note 63's fold-0 loss (-0.3 pt for a learned stacker on ~73 signals) is gone with ~540 training signals.
## B1. Count / Presence vs final_v2 (paired rows >= 30 min: 141,842 E, 383 signals; class accuracy, stack-aware)
| class (E) | final_v2 | trees | cand (trees+fj) | cand - v2 |
|---|---|---|---|---|
| Count (26,231) | .8941 | .8820 | .8862 | -0.79 [-2.54,+1.10] |
| Presence (42,559) | .9391 | .9296 | .9361 | -0.30 [-1.12,+0.50] |
| Advance / Yellow_Red / non-ATSPM | .9092 / .6416 / .7263 | | .9160 / .8352 / .8634 | +0.67 / +19.4 / +13.7 |
Neither regression is outside noise. R: Count -0.55, Presence -0.19.
* Count rows v2 right / cand wrong 1,197 (cand right / v2 wrong 991): called YR 606, Other 312, Presence 125, Advance 118;
  radar 1,074 (Count is 93 % radar); pulse 705 / normal 417; none stacked or twin; print_high 1,134. Step: lane decode 723,
  argmax (trees wrong too) 405, fj flips argmax 48, stack pick 21. Loop / video Count are weak for both (loop .30 v2 / .11).
* Presence: 986 lost / 857 won: called Other 852 (radar 645); normal output; lane decode 601, argmax 341, pick 11.
* The decode is the trade: argmax -> decode at >= 30 min Count .9127 -> .8872, Presence .9512 -> .9379, Advance .9287 -> .9177,
  YR .7087 -> .8464, non-ATSPM .7976 -> .8515 (net +0.34). final_v2 has no decode, so it keeps Count / Presence.
* Lane-decode Count losers lost to a lane-mate whose truth is Count in 548 / 1,029 (a lane error) and to a true YR in 331.
## B2. Who demotes true ATSPM to non-ATSPM (orchestrator add-on, note 70; instrumented copy of the decode, identical output)
Realistic, >= 30 min, blend: 6,583 true-ATSPM rows called non-ATSPM, 2,206 (33.5 %, 1.43 pt of true ATSPM) with argmax right
(Advance 1,038, Presence 963, Count 190). Mechanism: one-lane conflict, next free class non-ATSPM 1,093; stack-loser rule 56b
598; loser spans > 1 lane 435; pick span extension 80. Winner has the SAME truth in 55 % (D merged two lanes); winner a stack
peer 29 %; winner P > .9 in 54 %, < .7 in 14 %.
## B3. Fixes, nested (per held-out fold the grid point best on the other folds' E >= 30 rows), vs shipped decode
| fix (on blend) | >= 30 E | >= 30 R | Count / Presence / Advance | YR / non-ATSPM |
|---|---|---|---|---|
| class prior weights on P(Count), P(Presence) 1-1.5 | -0.00 [-0.04,+0.03] | -0.01 | +0.36 / +0.01 / -0.06 | -0.38 / -0.17 |
| stack loser -> next free class (drop 56b) | +0.04 [+0.01,+0.09] | +0.03 | +0.23 / 0 / -0.01 | +0.19 / -0.03 |
| unconfident winner does not demote (tau .7) | +0.06 [+0.00,+0.12] | +0.06 | +0.16 / +0.21 / +0.21 | -0.61 / -0.46 |
| **lane-confidence gate** (only D lane_conf >= .8 binds) | **+0.25 [+0.09,+0.41]** | +0.25 [+0.09,+0.40] | +1.37 / +0.69 / +0.74 | -2.33 / -1.99 |
| all three | +0.27 [+0.10,+0.43] | +0.26 | +1.58 / +0.73 / +0.78 | -2.68 / -2.19 |
* Gate chose .8 in 6/6 folds; fold 0 is -0.41, the other five +0.10..+0.63. Count / Presence go above final_v2
  (.9011 / .9422 vs .8941 / .9391). Part of the YR / non-ATSPM gain is given back (YR .835 -> .812, still +17 pt over v2).
* On the context stacker: gate (.9 in 5/6 folds, .8 in one; .9 is the top of the grid) +0.42 [+0.25,+0.60] E / +0.40 R, all
  folds positive. Tau .7 alone +0.10 [+0.06,+0.15]. Gate + tau +0.45 [+0.27,+0.63]. Stack-loser rule: no effect (-0.01).
## Combined: context stacker + lane gate (nested), vs current fixed 0.6 blend
>= 30 min .9048 -> .9134 E, **+0.86 [+0.59,+1.13]** / .9174 -> .9246 R, +0.72 [+0.46,+0.98]; 5 min +0.52, 10 min +0.50
(stacker only, the gate does not act below 30 min). Per fold +0.50..+1.37, all positive. vs trees alone +1.25 [+0.94,+1.57].
vs final_v2 (paired, >= 30 E): .8739 -> .9137, +3.98 [+3.21,+4.74]. Class cand -> fixed (v2): Count .8862 -> .9124 (.8941),
Presence .9361 -> .9432 (.9391), Advance .9160 -> .9383, YR .8352 -> .8389, non-ATSPM .8634 -> .8313 (.7263).
## Verdict / open for the orchestrator
* A: context stacker clears the rule (CI > 0 at >= 30 min, both sets, 3 seeds, beats its shuffled control by +0.25 [+0.13,+0.38])
  -> adopt instead of the fixed 0.6 blend. Cost: one small LightGBM (62 columns; needs health, pick inputs, D lanes).
* B: the Count / Presence "regression" is within noise. Its cause is the per-lane decode merging lanes. Lane-confidence gate:
  +0.42 on the stacker, CI > 0, wins Count / Presence back above final_v2, but gives back 2.3 pt non-ATSPM / 1 pt YR. Net
  positive -> adopt. Gate .9 = grid edge (1.0 = no lane decode). Class weights drop; tau, stack-loser under noise, drop.
* Combined +0.86 [+0.59,+1.13] E is just under the 1-pt bar. Each piece is beyond noise; the decision is the orchestrator's.
