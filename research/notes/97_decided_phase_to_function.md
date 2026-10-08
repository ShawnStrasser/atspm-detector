# 97 — Does the function side gain from the DECIDED phase (step-1 decoder output)? Fast test, fold 0 / seed 0 (2026-10-05)
Orchestrator brief: FAST TEST ONLY (one fold, one seed). Code `research/code/final97/{a97,b97}.py`, `neural/tcn97_dec.py`
(snapshot `tcn53/snap97` = snap74b + tcn97_dec). Work `%DC_WORK%/s97/` (a97.json, b97.json, logs/). locked_v2 absent (func_table /
cand64 asserts). Truth v4l, ATSPM E / R score, paired signal bootstrap.
## A. Trees + context stacker on the decoder's top phase (CPU, no refit)
* Premise check: the function frame is ALREADY built on a decided phase. Package v4f `_function_frame` takes the argmax of `prob`
  = trees + net + **decoder** output; the research frame's pred_phase = the stage-12 ranker-bag + decoder OOF (v5_frame).
  The only open swap is the old decoder -> the v4f decoder top (p2_tcn_ad76, s90/p87_q), = note 90's `oof90 tcnphase` arm
  (stacker context / lanes / gate redone; 229 tree features held at the frame phase).
* Changed rows: 3,597 of 456,033 (covered 62 %, whole signal-periods); scored >= 30 min E only 563. On those, the new decoder
  top is not more often right: new right 220, old right 182, old unknown 190 (wrong -> right 149, right -> wrong 182).
  Ceiling for a full tree-feature rebuild (changed rows lifted to the accuracy of same-correctness unchanged rows): -0.004 pt.
| >= 30 min | current | decided (v4f decoder) | delta pt [95 % CI] |
|---|---|---|---|
| E all folds (187,444) | .9197 | .9197 | -0.00 [-0.02,+0.02] |
| E fold 0 (19,757) | .9103 | .9105 | +0.02 [-0.02,+0.05] |
| R all / fold 0 | .9291 / .9143 | .9291 / .9144 | +0.00 / +0.02 |
5 min fold 0 -0.14 [-0.26,-0.03] (all folds +0.03 n.s.); 10 min +0.03 / -0.03 n.s. -> **neutral, nothing to gain** (the trees
already see a decided phase).
## B. siba given the decided phase (GPU, x97_dec fold 0 seed 0, 51 epochs, best 43, ~65 min + 8 min infer)
x86_siba4l recipe (--sib attn --max_det 32 --accum 2, func_rows_v4l, seed 0, filtered infer tree p >= .01) + one input channel
per (detector, candidate), constant in time: 1 = this candidate is the decoder's top phase (six-fold OOF p2_tcn_ad76; training
windows use the top of a same-signal evaluation window of the nearest length m5 / m10 / m30), all 0 outside the phase pool.
Flags set on 65 % of detector-pieces at inference (73,594 / 113,850). Coverage is per signal-period (not by function class),
so no label leak; residual: training-signal flags come from decoders that saw fold-0 PHASE labels (2nd order).
Same rows (fold 0, 49,402 frame rows with all nets); blend = fixed trees 0.6 + net 0.4 -> gate .9 decode.
| fold 0 | x86 s0 (ref) | x86 s1 | x86 s2 | **x97_dec** | x97 - s0 [95 % CI] |
|---|---|---|---|---|---|
| net alone >= 30 (21,556 lab.) | .7723 | .8177 | .8425 | **.8509** | +7.86 [+4.74,+11.06] |
| net alone 5 / 10 min | .7322 / .7526 | .7732 / .7875 | .7979 / .8150 | .8073 / .8180 | +7.51 / +6.55 |
| blend >= 30 E (19,757) | .8909 | .9005 | .9037 | .9035 | +1.26 [+0.53,+2.08] |
| blend >= 30 R | .8963 | .9053 | .9083 | .9082 | +1.18 [+0.48,+2.03] |
| blend 5 / 10 min E | .8558 / .8675 | .8688 / .8800 | .8785 / .8884 | .8777 / .8848 | +2.19 / +1.73 |
* The gain vs seed 0 is mostly seed 0 being the weak run (note 86: 77.2 net alone). Seeds 1 / 2 of the SAME recipe land at
  +0.96 / +1.28 E over seed 0; x97 vs them: >= 30 E +0.30 / -0.02, 5 min +0.89 / -0.08, 10 min +0.48 / -0.36.
* Flag-specific check: on unflagged signal-periods (no decoder input) x97 gains as much over s0 (+1.33 E) as on flagged ones
  (+1.22); s1 does the same (+1.20 / +0.84). By class >= 30 E vs s0: Adv +2.25*, Pres +1.36*, Count +2.20*, YR +0.57, non-ATSPM
  -1.49* (s1: +1.59 / +0.87 / +1.94 / +0.92 / -0.96).
* Net alone: x97 is above all three reference seeds (+0.84 / +0.94 / +0.30 pt vs the best, s2, at >= 30 / 5 / 10 min) —
  a single seed, within the 3-seed spread of ~7 pt on this fold; not separable from seed luck without more seeds.
## Verdict
* A: neutral by construction — function trees / stacker already consume the decided phase; the v4f decoder does not pick a
  better phase on the 0.3 % of rows where it differs. Dropped.
* B: inside the blend x97 = the better baseline seeds (-0.02 .. +0.30 pt >= 30 min vs s1 / s2); far below the 1-pt function bar.
  Net alone looks slightly better but one seed on one fold cannot show it. **Not promising; not promoted.** If ever revisited:
  3 seeds on fold 0 vs x86 s0-2 (3 GPU-h) is the cheapest decisive test. GPU released to the q95 retrain queue.
