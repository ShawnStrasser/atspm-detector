# 20_trackB_b6b8_blend_stack — TCN added to the blend (B6), stacking trees ⇄ net (B8): **both dropped**

Steps B6 and B8 of `AGENTS.md`, run on predictions already on disk (no network trained), CPU (6 threads), ~2 min wall.
Fold 0, the stage-13 rows (22 windows, 210,038 rows, 34,552 detector-windows; 6,489 at 30 min). Blend *before* the
decoder = shipped arrangement; folds 1–5 keep the stage-13 GRU mixture, so the decoder is fitted once and only fold
0's inputs change. The GRU-seed-0 candidate reproduces stage 13's `_fold0` exactly (**.97449**). Code
`research/code/trackB/{b6_blend3,b8_stack}.py`; results `dc_work/trackB/{b6/b6,b8/b8}.json`. Locked signals unread.

## B6 — trees + GRU + TCN
Weights fixed before looking: **w3** = 0.5 trees / 0.25 GRU / 0.25 TCN; **w13** = ⅓ each. None were picked on folds
1–5: the TCN has out-of-fold predictions for fold 0 only. GRU = stage-13 seed 0; each TCN seed swapped in. Controls
(w3 shape, two nets of one kind) separate *architecture* diversity from merely averaging *two nets*.

| fold 0, 30 min | before decoder | Δ vs .97449 | after decoder | Δ vs .97345 |
|---|---|---|---|---|
| trees + TCN s0 / s1 / s2 (2-way) | .97372 / .97388 / .97745 | −0.08 / −0.06 / +0.30 | .97157 / .97280 / .97530 | −0.19 / −0.07 / +0.19 |
| **w3** GRU s0 + TCN s0 / s1 / s2 | .97510 / .97510 / .97652 | **+0.06 / +0.06 / +0.20** | .97389 / .97528 / .97438 | +0.04 / +0.18 / +0.09 |
| w13 GRU s0 + TCN s0 / s1 / s2 | .97340 / .97370 / .97497 | −0.11 / −0.08 / +0.05 | .97235 / .97250 / .97300 | −0.11 / −0.10 / −0.05 |
| control w3 GRU s0 + GRU s1 / s2 | .97482 / .97361 | +0.03 / −0.09 | .97424 / .97468 | +0.08 / +0.12 |
| control w3 TCN s0 + TCN s1 / s2 | .97494 / .97651 | +0.05 / +0.20 | .97373 / .97468 | +0.03 / +0.12 |

Other lengths (before, w3 mean over the 3 TCN seeds vs shipped): 5 min .9541 vs .9522 (+0.18), 10 min .9647 vs .9625
(+0.22), 1 h .9745 vs .9741 (+0.04); the GRU-seed-pair control gets +0.10 / +0.20 / +0.12 — same size, inside the
0.12 pt blend seed sd. At 30 min w3 removes 4 / 4 / 13 of 165 errors (concurrent-pair 85 → 83 / 84 / 79).
**Diversity** (nets alone, 30 min, top-1 errors; phi / Jaccard = both-wrong ÷ either-wrong): GRU~TCN .735–.749 /
.59–.61; TCN seeds .741–.768 / .60–.64; GRU seeds .764–.805 / .63–.68; trees~net .42–.44 / .29–.30 (same order at
5, 10, 60 min). A TCN errs with the GRU almost as often as another GRU seed does; the trees are the diverse partner.
**Verdict: dropped.** w3 = +0.06 / +0.06 / +0.20 pt at 30 min before the decoder (mean +0.11), +0.04 / +0.18 / +0.09
after: never ≥ 0.3 pt, and no better than two seeds of one architecture. w13 is worse than shipped. B6 = B2 again.

## The 120-min network cut-off — blend before the decoder minus trees alone, points
| | 1 h | 3 h | 6 h | 24 h | full |
|---|---|---|---|---|---|
| GRU s0 2-way, fold 0 | +1.36 | +0.15 | −0.06 | −0.20 | −0.12 |
| TCN 2-way, mean of 3 seeds, fold 0 | +1.29 | +0.33 | +0.26 | +0.07 | +0.20 |
| w3, mean of 3 TCN seeds, fold 0 | +1.39 | +0.35 | +0.19 | −0.02 | +0.13 |
| GRU 2-way, **all six folds** (stage 13) | +1.26 | +0.35 | +0.31 | +0.20 | +0.20 |

Fold 0 has ~1,700 detectors × 2 windows per long family (1 error ≈ 0.03 pt); the OOF set has no 2-, 4- or 12-h
window. Past 2 h the net is worth **~+0.3 pt up to 6 h** (six folds, GRU; fold-0 TCN agrees), ≤ +0.2 pt beyond.
At the TCN's 0.18 s per signal per 30 min (note 15) a 6-h sample costs ~2 s, about what a 2-h GRU sample costs
now (1.5 s). **Recommendation:** if the TCN ships, raise the cut-off to **360 min**; with the GRU keep 120. Keep the
net off past 6 h. A cost/benefit call, not a search result: fix it at the final re-fit and confirm on six folds.

## B8 — stacking LightGBM ⇄ network (one test)
Fixed learner: LightGBM binary, 15 leaves, lr 0.05, min_child_samples 200, no bagging, early stop on fold 1, train
folds 2–5. Features: p_tree, p_net, both ranks among the candidates, each model's top probability, n candidates, log
window hours, log(1 + actuations); renormalised over candidates. Stage-13 GRU OOF, signal-grouped folds, **no fold-0
row in training**. *Before*: stacked (ranker p0, GRU) replaces the 0.5 mixture into the decoder; folds 1–5 are
cross-fitted (each from a stacker trained on the other four), so the decoder never sees in-sample stacker output.

| fold 0 | 5 min | 10 min | 30 min | 1 h | 3 h |
|---|---|---|---|---|---|
| after decoder: fixed 0.5 / stack | .9495 / .9502 | .9608 / .9609 | .97345 / **.97359** | .9733 / .9740 | .9814 / .9805 |
| before decoder: fixed 0.5 (shipped) / stack | .9522 / .9534 | .9625 / .9639 | .97449 / **.97327** | .9741 / .9746 | .9797 / .9788 |

**Verdict: dropped.** +0.01 pt after the decoder, −0.12 pt before it at 30 min (errors 165 → 173). Window length
and actuation count got 0 % of the gain: nothing a fixed average misses. After the decoder the stacker leans on the
decoded tree probability (88 % of gain), before it on the net (94 %). **Carry forward:** the 0.5 two-way blend
stays; B2, B6, B8 each give ≤ +0.1 pt — only B5 (changing what the net *sees*) has headroom left in Track B.
