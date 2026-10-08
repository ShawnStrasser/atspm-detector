# 19_trackB_b5_sibling_net — sibling-detector context inside the network (B5): **dropped**

Track B step B5 of `AGENTS.md`, TCN backbone, screened on **fold 0**, on exactly the stage-13 rows (22 windows,
210,038 rows, 34,552 detector-windows), same harness as notes 15/16 (`trackb_eval.py`, folds 1–5 keep the stage-13 GRU).
Metric: **blend before the joint decoder** (weight 0.5). Code `research/code/neural/b5_{headroom,net,train,infer,chain}.py`,
artefacts `dc_work/trackB/{b5,runs,preds,eval}/`, logs `dc_work/logs/trackB_b5*.log`, `trackB_tb_b5?_f0.log`.

## Headroom first (existing predictions, `b5_headroom.py`)
Where the fold-0 blend still fails at 30 min (TCN seed 0; seeds 1/2 and the stage-13 GRU look the same):
**170 errors** of 6,489 detectors, **90 (53 %) concurrent pairs**. The network alone is right on only **18** of them
(10 concurrent), the decoded trees on 47, the ranker on 39; *some* model is right on 73. The network alone makes 277
errors, and the blend fixes 125 of them. Across the three TCN seeds, 121 errors are shared by all three and 205 occur in at least one.
So the net-side ceiling at 30 min is ~17 rows (**~0.26 pt**) if the blend flipped exactly those. On the other hand,
**139 of 170 errors (82 %)** have another detector of the same true phase on that window which the blend gets right
(76 of the 90 concurrent ones): the sibling information exists, so the question is whether the net can use it
where the decoder does not.

## Design (two variants fixed before launch)
The TCN pair scorer is unchanged up to its 128-d pair embedding z(d,c) and logit l1(d,c). One set-attention layer
across the detectors of the same signal-window: for each candidate c, (d,c) attends over (d',c) of every **other**
detector (4 heads, a learned null slot, self masked). The attention bias and part of the value come from 6 pairwise
trace features: zero-lag occupancy correlation (2-s bins), max correlation with d' lagging / leading by 2–10 s, 30-s
count correlation, relative activity, d' active. A zero-initialised MLP on [z, context] adds a residual, and the score is
l1 + delta, softmaxed over candidates as before. **b5a** uses embeddings only (+84 k weights, 785 k in total). **b5b** is b5a plus the
siblings' first-pass distribution p1(d',·) at c (p, log p, is-argmax) in keys and values, plus an auxiliary 0.5×
loss on l1 ("the joint decoder inside the net"). Both are phase-anonymous and permutation-equivariant, checked numerically
(detector and candidate permutations: max logit change 6e-8). Training is `trackb_train` unchanged (12 sampled
detectors per signal act as siblings); at inference every labelled detector of the window is in one forward pass.
Labelled channels average 20.9 per signal against 21.2 active, so this sibling set is close to what production would see.

## Fold 0 (gate: best variant seed 0 vs TCN seed 0 at 30 min, ≥ +0.3 pt, then a seed 1)
| fold 0 | 5 min | 10 min | 30 min | 1 h | 3 h |
|---|---|---|---|---|---|
| blend: TCN seed 0 / three-seed mean | .9492 / .9506 | .9622 / .9639 | .9737 / .9750 | .9710 / .9734 | .9808 / .9814 |
| **blend: b5a / b5b** | .9511 / .9508 | .9629 / .9627 | **.9731 / .9729** | .9751 / .9746 | .9821 / .9817 |
| b5a − TCN seed 0 / − three-seed mean | +0.19 / +0.05 | +0.07 / −0.10 | **−0.07 / −0.19** | +0.41 / +0.17 | +0.12 / +0.07 |
| net alone: TCN mean / b5a / b5b | .9275 / .9304 / .9317 | .9478 / .9473 / .9496 | .9611 / .9636 / .9645 | .9588 / .9650 / .9635 | .9658 / .9689 / .9668 |
| concurrent-pair errors at 30 min, blend (net alone) | TCN 90 / 90 / 72 (140 / 127 / 116); **b5a 96 (124), b5b 94 (119)** | | | | |

**Verdict: dropped.** The screen failed at 30 min (−0.07 pt vs seed 0, −0.19 pt vs the three-seed mean), so no
seed 1 and no six folds, per the rule. It is the same pattern as B2, B3 and B9: the **network alone** improves
(+0.24 / +0.34 pt at 30 min, +0.3 / +0.4 pt at 5 min, ~1 net-seed sd, one seed each), and its concurrent-pair errors
drop 7–15 % against the seed mean, but the blend absorbs all of it. Blend concurrent-pair errors are *not* lower (96 / 94 against 72–90). The
decoder already feeds the neighbours' probabilities back in, and the net's gain lands on rows the trees and decoder already get right: b5b
turns the net's own errors from 277 (TCN seed 0; mean 252) to 230, but only 12 of the blend's 175 errors are net-right. The +0.41 pt at 1 h
is against the worst seed; against the mean it is +0.17 pt (under 1 sd), and the pre-registered gate is 30 min.

## ONNX (for a later ship decision)
It exports and runs, **with a contract change**. The `SibExport` wrapper takes **one signal-window**, `x [D,K,9,T]` → `score [D,K]`
(the detector block is read back from channels 0–1, so no second input). Opset 17 with dynamic D/K/T, and onnxruntime matches torch to
2.4e-7. `torch.eye` had to go (no ORT kernel for EyeLike on bool). `model/gru_onnx.py` would have to stop chunking
pairs across detectors and feed every detector × candidate of a signal in one call. Memory grows as D²·K for the
attention, which is negligible at D ≤ 64.

## Cost, caveats
GPU: b5a 21 min + b5b 24 min of training, 2 × 4.3 min of inference, ~1 min smoke test: **0.9 h** (Track B total 3.2 h).
The CPU evaluations took ~5 min each. The raster preload was not needed: an epoch took 31 s, the same as the plain TCN.
Locked signals were never read. Not tried: siblings attending across *other* candidates (c′ ≠ c); more than 12 siblings in training;
a GRU backbone. Given the headroom (~0.26 pt net-side ceiling at 30 min) none is likely to clear 0.3 pt.
