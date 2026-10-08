# 112 — Phase decoder without the channel-adjacency features (2026-10-07) — analysis only, v6b NOT built
Brief (orchestrator): the v6 decoder (decode_v3, 25 features) and the fast-profile decode_trees read 5 neighbour aggregates
over channels with |channel number difference| <= 2 (`decode.py:71-77`; adj_mean / max / top1 / wsum / n). AGENTS.md: the
model never sees a channel number. Measure the cost of dropping them; build v6b only if it costs < 0.1 pt at >= 30 min
(or within seed noise). Also time the dead yr_* block. CPU 4 threads, saved OOF only, locked_v2 asserted absent.
Code `research/code/final112/` (f112.py oof / score / fit, yrtime112.py); work `%DC_WORK%/s112/`.
## Set-up
Both decoders carry the 5 adj_* columns (decode_v3.json and decode_trees.json `features`, 25 each). Same OOF inputs and
recipe as fit111 `dec` / `dectrees` (p90 decode_iters: folds_v4, inner early stopping on fold k+1, BIN_PARAMS):
v6 = ranker s0 0.5 / siba w32 x3 phase head 0.5 (net on ranker p >= .01); trees = ranker s0 alone. Arms 20 features
(noadj) vs 25 (adj), seeds 0 / 1 / 2 (bagging / feature-fraction seeds). Seed-0 adj reproduces the package OOF exactly
(q109 p2_w32m3_rs0 and q111 p2_trees_s0: same fold iterations, 100 % row agreement). score111 conventions: timing truth,
E / R, per length, paired signal bootstrap; >= 30 min 189,509 rows / 761 signals.
## Phase accuracy, decoder noadj vs adj (pt, same seed paired; * = CI excludes 0)
| decoder | 5 min | 10 min | 30 min | 1 h | 3 h | 24 h | >= 30 E | >= 30 R |
|---|---|---|---|---|---|---|---|---|
| v6 (full) s0 | -0.67* | -0.46* | -0.26* | -0.22* | -0.17* | -0.13* | -0.19 [-0.26,-0.12]* | -0.20* |
| v6 s1 / s2 | -0.67* / -0.57* | -0.44* / -0.43* | -0.29* / -0.27* | -0.18* / -0.23* | -0.14* / -0.15* | -0.14* / -0.16* | -0.19* / -0.21* | -0.20* / -0.22* |
| v6 seed mean | -0.64 [-0.77,-0.51]* | -0.44* | -0.27* | -0.21* | -0.16* | -0.15* | -0.19 [-0.26,-0.13]* | -0.21* |
| trees (fast) seed mean | -1.55 [-1.80,-1.32]* | -1.26* | -0.91* | -0.81* | -0.40* | -0.23* | -0.57 [-0.68,-0.47]* | -0.58* |
Accuracy >= 30 E: v6 adj .9846 (seeds .9846 / .9846 / .9846), noadj .9827 / .9827 / .9826 [.9795,.9855]; trees adj .9780,
noadj .9724 / .9721 / .9723. Seed spread <= 0.01 pt at >= 30 min, <= 0.07 pt at 5 min: the loss is 15-20x the seed noise.
Function not re-scored (its OOF frame holds pred_phase fixed); a worse phase can only cost function too.
## yr_* (dead) feature block, v6 full, 3 h, 4 threads, median of 3 warm calls (machine ~0 % else)
r8 / r11 / busiest ev / busiest ch: 0.021 / 0.016 / 0.027 / 0.022 s of 0.96 / 0.73 / 1.24 / 1.31 s = 1.7-2.2 % of the call.
No v6 model reads a yr_* column (grep of every weights JSON); function.SIB_FEATS lists 6 yr_* names but their sibling
columns are not in the 229-feature list. Removing it is safe and saves ~0.02 s per 3-h call.
## Verdict
Dropping channel adjacency costs -0.19 pt phase at >= 30 min (-0.64 at 5 min) with the v6 decoder and -0.57 (-1.55) with
the fast decoder -> over the 0.1-pt limit and far outside seed noise -> v6b NOT built, packages untouched. Channel
wiring in blocks carries real phase information. The feature reads channel DIFFERENCES only (check.py's reversal / shift
tests pass), but it is a channel-number input in the sense of AGENTS.md: a decision for the orchestrator / user
(keep and state it plainly, or accept the loss). Untested alternative: a behaviour-only substitute (e.g. co-location /
lead-lag neighbours) might recover part of it. The yr_* removal alone (+ the docstring fixes) can go into any later
package with zero accuracy change. Exam signals not opened.
