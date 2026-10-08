# 103 — Simpler architecture: ONE LightGBM per task (decider folded into the scorer), with / without the TCN (2026-10-06)
User question: can a simpler design do as well as v5b (phase ranker -> TCN -> decoder; function trees -> siba w32 x3 ->
context stacker -> lane rule)? CPU only (<= 6 threads), GPU untouched, saved TCN / siba OOF reused, nothing shipped,
locked_v2 asserted absent. Code `research/code/final103/` (p103_phase.py, f103_func.py); work `%DC_WORK%/s103/`
(score103.json per task). Setup = note 95 (2026-only, six folds folds_v4, timing truth / v4q truth, Sept-2026 rows).
## Arms (each single model = the scorer's own recipe, inner-fold early stopping, 3 seeds bagged, argmax)
* A1 (one LightGBM, no network). Phase: 261 ranker features + 51 PREDICTION-FREE context columns (for 8 key per-candidate
  features: what the OTHER detectors show for the same candidate = leave-one-out mean / max, own rank, similarity- and
  adjacency-weighted neighbour means; 2 "claims" counts; graph sizes). Function: 229 + the stacker's 16 prediction-free
  columns (pick / span / track, peers, D-lane columns, phase-mate and lane-mate counts, log minutes / actuations).
  The decider's other context (neighbours' / phase-mates' / lane-mates' PROBABILITIES, own rank) needs a model's prediction,
  so it cannot be folded into a single model without a first stage -> tested separately as A1o.
* A1o (two stages): A1 + the decider's prediction-based columns built from the six-fold OOF scorer (standard OOF, like
  the current decider). = "decider that also sees the raw features"; same model count as today's no-network design.
* A2 (network + one LightGBM): A1 + the network output as features + the decider-style context computed on the NETWORK's
  probabilities (no tree model needed). Phase TCN p95_ad OOF on ALL candidates (no tree filter); siba = v5b w32 mean3 OOF.
* References on the same rows (saved OOF): full v5b; no networks (trees + decoder / nonet stacker); fast le2h; trees alone.
Fixed for every arm: lanes / pick inputs (lane chain), the function frame's phase input; siba OOF filtered by phase-tree OOF.
## Accuracy, E (everything), pt vs full v5b [95 % paired signal-bootstrap CI]; R within 0.05 pt of E everywhere (json)
| PHASE (761 sig) | 5 min | 10 min | 30 min | 1 h | 3 h | 24 h | >= 30 min |
|---|---|---|---|---|---|---|---|
| full v5b | .9640 | .9739 | .9809 | .9836 | .9856 | .9866 | .9843 |
| A2 TCN + 1 LGB | +0.15 [+0.02,+0.29] | +0.11 [-0.01,+0.22] | +0.08 [-0.02,+0.17] | +0.05 [-0.08,+0.16] | +0.07 [-0.06,+0.18] | -0.02 [-0.16,+0.11] | +0.05 [-0.05,+0.14] |
| fast le2h | +0.09* | -0.00 | +0.01 | -0.05* | -0.48* | -0.22* | -0.20 [-0.27,-0.14] |
| no networks (2 stages) | -1.91* | -1.45* | -0.91* | -1.00* | -0.42* | -0.18* | -0.59 [-0.72,-0.46] |
| A1o (2 stages) | -1.66* | -1.18* | -0.74* | -0.88* | -0.33* | -0.14 [-0.30,+0.04] | -0.49 [-0.63,-0.35] |
| A1 one LGB | -2.18* | -1.65* | -1.20* | -1.07* | -0.60* | -0.29* | -0.75 [-0.95,-0.56] |
| trees alone (ranker argmax) | -4.17* | -3.48* | -2.44* | -2.48* | -1.19* | -0.68* | -1.63* |
| FUNCTION (642 sig) | 5 min | 10 min | 30 min | 1 h | 3 h | 24 h | >= 30 min |
|---|---|---|---|---|---|---|---|
| full v5b | .9040 | .9102 | .9247 | .9249 | .9340 | .9363 | .9307 |
| A2 siba x3 + 1 LGB | -0.58 [-0.89,-0.29] | -0.21 [-0.44,+0.02] | -0.33 [-0.54,-0.10] | -0.21 [-0.46,+0.04] | +0.00 [-0.23,+0.23] | -0.20 [-0.48,+0.10] | -0.22 [-0.39,-0.04] |
| fast le2h | 0 | 0 | 0 | 0 | -0.60* | -0.46* | -0.32 [-0.43,-0.22] |
| no networks (2 stages) | -1.93* | -1.47* | -1.00* | -1.12* | -0.60* | -0.46* | -0.81 [-1.01,-0.63] |
| A1o (2 stages) | -2.02* | -1.50* | -1.06* | -1.16* | -0.79* | -0.68* | -0.92 [-1.18,-0.67] |
| A1 one LGB | -2.03* | -1.76* | -1.28* | -1.49* | -1.00* | -0.79* | -1.14 [-1.41,-0.87] |
| trees alone (+ lane rule) | -2.01* | -1.40* | -1.49* | -1.66* | -1.13* | -0.94* | -1.31* |
* Each single seed within 0.1 pt of its 3-seed mean. vs no networks >= 30: A1 phase -0.15 [-0.27,-0.04], function -0.33
  [-0.54,-0.11]; A1o phase +0.10 [+0.04,+0.16], function -0.11 n.s.; A2 phase +0.64*, function +0.59*.
## Models (classification only; seeds counted; setback / health excluded) and CPU at 3 h (warm s, typical r8 / r11 / busiest ev / ch)
| design | models phase + function (+ lanes D x3) | 3 h warm s | basis |
|---|---|---|---|
| full v5b | 3 ranker + TCN + decoder; 3 trees + 3 siba + 3 stacker (+3) = 17 | 1.33 / 1.02 / 1.59 / 1.78 | bench (note 100c) |
| fast le2h (both sets) | 6; 12 (+3) = 21 (at 3 h runs the no-network path) | 0.68 / 0.57 / 0.95 / 0.93 | bench |
| no networks | 3 + decoder; 3 trees + 3 stacker (+3) = 13 | = fast at 3 h | bench |
| A1 one LGB | 3; 3 (+3) = 9 | ~0.67 / 0.56 / 0.94 / 0.92 | no-net bench - decoder / stacker (<= 0.02 s) |
| A1o | 6; 6 (+3) = 15 | ~ = no networks | estimate |
| A2 net + 1 LGB | TCN + 3; 3 siba + 3 (+3) = 13 | ~2.2 / 1.7 / 2.4 / 3.0 | v5b - 0.02 + TCN on all candidates |
Bench stage timings: decoder + stacker <= 0.02 s per 3-h call; phase TCN on kept pairs 0.24-0.44 s. A2 runs it on every
candidate (median 3.7x the kept pairs at 3 h, p90 6x): +0.7-1.2 s; keeping the tree filter brings the ranker back.
## Verdict
* Removing the decider never pays for itself in time (decoder + stacker are ~1 % of a 3-h call); it only cuts model count.
* Without networks, one LightGBM (A1) is real but small below the current two-stage no-network design (phase -0.15,
  function -0.33 at >= 30 min); the decider's value is the OTHER detectors' predictions, which a single model cannot see.
* With the network, one LightGBM matches the phase side at every length (A2 +0.05 n.s.; +0.15* at 5 min) and is close on
  function (-0.22* at >= 30 min, tie at 3 h, -0.58* at 5 min). But phase A2 needs the TCN on every candidate (~+0.9 s at 3 h)
  unless the ranker filter is kept.
* The networks matter most on short samples: no-network cost phase -1.9 / -1.0 / -0.4 / -0.2 pt and function -1.9 / -1.1 /
  -0.6 / -0.5 pt at 5 min / 1 h / 3 h / 24 h. le2h already exploits that.
* Not tested: A2 with a cheap rule filter for the TCN, a nested-OOF A1o (standard OOF only, as today's decider).
