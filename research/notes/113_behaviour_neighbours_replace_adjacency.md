# 113 — Behaviour-only "lead" neighbours replace channel adjacency in the phase decoders; package v6b (2026-10-07)
Brief: find decoder features without channel numbers that recover note 112's adjacency gain; user decision (Oct 7):
adjacency goes regardless -> build v6b with the best arm. CPU 4 threads, saved OOF (f112 recipe, folds_v4, seeds 0-2),
locked_v2 asserted absent, exam signals not opened. Code `research/code/final113/`; work `%DC_WORK%/s113/`.
## 1. What adjacency captured (diag113.py; seed-mean v6, adj minus noadj)
All of it on EVEN (through) truth phases (>= 30 min +0.28 pt, odd/left-turn -0.07) = the concurrent-through problem
(2<->6, 4<->8). Gain sits where the phi neighbours fail: detectors whose top-8 phi neighbours are mostly another phase
(phi purity <= .5: 81 % of the gain) or absent, and low volume (6-20 actuations: 30 % of the gain at >= 30 min, 55 % at
5 min). Flipped rows: median 24 actuations (all 81), top phi .13 (all .20).
## 2. Pair table (pairs113.py, 761 signals x 22 windows, 2 min)
Onsets in 1-s bins; stratum = green-state mask x time since last change (0-2/3-7/8-19/20+ s; mask only as equality key).
co = same-second excess over the stratified expectation; lag = 1-8 s lead excess max(O-E) either direction / sqrt(na nb);
phi1 = plain 1-s phi; nn2 = second-order phi votes. Each -> the decoder's 5 neighbour aggregates, top-8, w = r^2.
## 3. Screen, seed 0, phase E vs v6 (with adjacency) / vs noadj, pt
| arm | v6 >= 30 | v6 5 min | trees >= 30 | trees 5 min |
|---|---|---|---|---|
| co (same second) | -0.17* / +0.02 | -0.63* / +0.04 | -0.54* / +0.02 | -1.50* / -0.03 |
| phi1 (1-s bins) | -0.17* / +0.02 | -0.65* / +0.03 | -0.54* / +0.02 | -1.45* / +0.03 |
| nn2 (2nd order) | -0.19* / +0.01 | -0.62* / +0.05 | -0.53* / +0.03 | -1.44* / +0.04 |
| **lag (1-8 s lead)** | **+0.01 / +0.20*** | **-0.41* / +0.26*** | **+0.31* / +0.87*** | **-0.28 / +1.20*** |
| lag variants: 16 nb, +co, +z-weighted lag | = lag (+-0.03) | = lag (+-0.03) | = lag (+-0.02) | = lag (+-0.07) |
| z-weighted only / unstratified | -0.03 / -0.07 | -0.48* / -0.57* | +0.24* / +0.09 | -0.34* / -0.68* |
| lag + adj (diagnostic only) | +0.14* / +0.33* | +0.19* / +0.86* | +0.58* / +1.14* | +0.98* / +2.45* |
Only the advance -> stop-bar travel signal carries the adjacency information; co-timing at the same second does not
(both concurrent throughs discharge together). Adjacency still adds on top of lag (wiring blocks are extra information).
## 4. Shipped arm 'lagt' (= lag, top-8 with ties kept: no row-order tie-break), 3 seeds, paired, seed-mean
| decoder | vs | >= 30 E | >= 30 R | 5 min E | 10 min | 30 m / 1 h / 3 h / 24 h E |
|---|---|---|---|---|---|---|
| v6 (full) | v6 adj | +0.00 [-0.08,+0.09] | -0.01 | -0.42 [-0.58,-0.26]* | -0.13* | -0.00 / -0.01 / +0.02 / -0.02 |
| v6 (full) | noadj | +0.20 [+0.15,+0.25]* | +0.20* | +0.22 [+0.12,+0.32]* | +0.32* | +0.27* / +0.20* / +0.18* / +0.12* |
| trees (fast) | v6 adj | +0.32 [+0.18,+0.45]* | +0.31* | -0.31 [-0.59,-0.02]* | +0.18 | +0.39* / +0.60* / +0.21* / +0.16* |
| trees (fast) | noadj | +0.89 [+0.75,+1.03]* | +0.89* | +1.24 [+1.00,+1.48]* | +1.45* | +1.30* / +1.41* / +0.61* / +0.40* |
Accuracy >= 30 E: v6b .9846 (= v6 .9846; noadj .9827); 5 min .9634 (v6 .9676, noadj .9612). Fast decode_trees .9811
(v6 .9780), 5 min .9407 (v6 .9439). Shuffled control (lag block permuted across rows, 3 seeds) = noadj at every length
(|d| <= 0.03 pt, both decoders) -> real signal. Function not re-scored (its OOF frame holds pred_phase fixed).
Verdict: replaces adjacency fully at >= 30 min (fast path is better than v6's); 5 min keeps -0.42 (fast -0.31).
## 5. Package dc_work/final_v3_candidate_v6b (copy of v6; v6 untouched; nothing in model/)
decode_v3 / decode_trees refit full data (f113 fit: 355 / 516 trees = mean fold iterations; ONNX parity 4e-16).
decode.py: adjacency code gone, lead aggregates (lag_*); similarity.build_lead_window (SQL in package; parity113 vs
training graph on 12 signals x 22 windows: 99.7 % of edges identical, rest float32 rounding at r ~ 0, w rel diff p99 2e-6).
predict.py: lead graph wired in, dead yr_* block (+ window_cycles) removed; docstrings / CLI help / blend.json note /
model_card top block updated (default profile full; v4b top block kept as history_top_block_v4b).
check.py: new check 3b, 3 random channel bijections onto 1..64 x phase renumberings (+1 at 10 min, +1 in the no-network
check). On v6 it FAILS (decoded prob moves 0.04-0.17 per seed; p0 passes = ranker + network clean). On v6b it passes
(max 2.7e-8; also under 'trees' and at 10 min): no other leak (network sibling order, ties) on the sample.
check.py --freeze then 6/6 PASS. Sample answers: phase and function identical to v6 (17/17; probs move <= .09).
Bench (bench98, 3 h warm r8/r11/ev/ch, 4 thr): v6 2.25/1.13/2.91/3.04 s, v6b 2.00/1.22/2.85/3.18 s; cpu-s within
+-4 %, peak 305-402 vs 319-391 MB = tie. Machine shared (others up to 82 % CPU): times ~2x note 111's, compare paired only.
Rebuild: pairs113 -> f113 oof/shuf/score -> f113 fit -> assemble113 weights -> check.py --freeze -> assemble113 card.
