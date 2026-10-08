# 109 — A simpler architecture: one network for both tasks, single-seed trees (2026-10-06) — analysis only, nothing shipped
Brief (orchestrator / user "not sold on 2 TCNs + ~6 LightGBMs"). Setup = note 95/100b 2026 six-fold OOF (folds_v4, timing truth,
v4q truth, Sept-2026 rows); reference v5c (= v5b classification OOF; v5c lane refit is an ATSPM tie, note 105). CPU <= 5 threads,
local A1000 only (unfiltered siba re-inference, 18 x ~5 min), no RunPod, locked_v2 asserted absent. Code `research/code/final109/`
(p109_phase, f109_func, sb109); work `%DC_WORK%/s109/` (phase/score*.json, func/score*.json, setback/eval109.json, pkg = timing copy).
## 1. The siba net's own phase head replaces the phase TCN (saved x100_w32 ppreds, tree-filtered as packaged)
Trees bag 0.5 / net 0.5 -> decoder refit six-fold, exactly the v5 recipe with the siba phase head instead of p95_ad. dE pt vs v5c [CI]:
| phase | 5 min | 10 min | 30 min | 1 h | 3 h | 24 h | >= 30 E | >= 30 R |
|---|---|---|---|---|---|---|---|---|
| v5c (acc) | .9640 | .9739 | .9809 | .9836 | .9856 | .9866 | .9843 | .9863 |
| siba w32 x3 phase head | +0.38* | +0.17* | +0.04 | +0.03 | +0.03 | +0.06 | +0.04 [-0.02,+0.09] | +0.03 |
| one w32 member (s0 / s1 / s2) | +0.18*/+0.22*/+0.21* | +0.14/+0.11/+0.16 | | | | | +0.02 / -0.01 / +0.01 | +0.01 / -0.03 / 0.00 |
| + ranker 1 seed (s0, net x3) | +0.36* | +0.24* | +0.03 | 0.00 | +0.04 | +0.06 | +0.03 | +0.01 |
| + ranker 1 seed + 1 member | +0.10 | +0.15* | -0.04 | +0.01 | +0.06 | +0.06 | +0.02 [-0.05,+0.07] | -0.00 |
| no decoder (0.5/0.5 average, net x3) | -0.42* | -0.29* | -0.33* | -0.22* | -0.17* | -0.18* | -0.23* | -0.25* |
Net alone: siba phase head .9740 vs phase TCN .9726 (>= 30). The decoder must stay (-0.23* without it); the phase TCN can go.
## 2. Function: seeds (stacker refit, v5 recipe; gate .9 lane rule)
| function | 5 min | 10 min | 30 min | 1 h | 3 h | 24 h | >= 30 E | >= 30 R |
|---|---|---|---|---|---|---|---|---|
| v5c (acc) | .9040 | .9102 | .9247 | .9249 | .9340 | .9363 | .9307 | .9397 |
| trees 1 seed + stacker 1 seed (net x3) | -0.16* | 0.00 | +0.02 | -0.03 | -0.04 | +0.02 | -0.01 [-0.06,+0.05] | -0.01 |
| 1 net member, 3/3 seeds (s0 / s1 / s2) | -0.52*/-0.40*/-0.64* | | | | | | -0.09 / -0.13* / -0.28* | -0.13* / -0.14* / -0.27* |
| 1 member + 1-seed trees + 1-seed stacker (s0 / s1) | -0.61* / -0.44* | -0.20 / -0.37* | -0.18* / -0.33* | -0.15 / -0.15 | -0.09 / -0.12 | 0.00 / -0.02 | -0.12 / -0.17* | -0.15* / -0.17* |
| no lane rule (argmax + stack credit, net x3) | -0.17* | -0.09 | +0.07 | -0.02 | -0.09 | -0.10 | -0.04 [-0.20,+0.11] | -0.05 |
| one LightGBM instead of trees + stacker (note 103 A2) | -0.59* | -0.22 | -0.33* | -0.21 | 0.00 | -0.20 | -0.22* | -0.17 |
Seed averaging pays only for the NETWORK (as notes 95b / 100b); tree / stacker seeds are free to drop. Lane rule ~0 at >= 30 min
but -0.17* at 5 min, and the lane model stays anyway (lane output + stacker input) -> keep the rule (no model saved by dropping it).
## 3. Setback (19 models in the package: 4 length groups x 3 P50 seeds + P10/P90 x 3 + pair model). Printed Advance, w50 pt vs as-is
1 seed per group: +0.3 [-0.3,+0.8] (medAE -0.1 ft) = tie. One pooled model (3 seeds): -0.6 [-1.8,+0.6] (m30 -1.5*); pooled no length
feature -0.2 n.s.; pooled 1 seed -1.0 n.s. -> 1 seed per group (12 -> 4 P50 models). Band models not tested (confidence only).
## 4. Speed / RAM (bench98 on a COPY of v5b_fast with design profiles, 4 threads, warm, 2 passes; machine ~20-25 % busy, GPU job running)
| 3 h warm s (typical r8 / r11 / busiest ev / ch), peak MB | |
|---|---|
| v5c-equivalent (full) | 1.51 / 1.22 / 1.78 / 2.20 ; 424-593 |
| no phase TCN, siba x3 (design B/C) | 1.11 / 0.87 / 1.37 / 1.55 ; 370-453 (-27..-30 %) |
| no phase TCN, siba x1 (design D) | 0.89 / 0.72 / 1.19 / 1.22 ; 323-379 (-41..-45 %) |
| trees only (no network) | 0.73 / 0.70 / 1.05 / 1.08 ; 295-328 |
| siba x3 / x1 WITHOUT the tree filter | 1.19-2.66 / 0.79-1.59 (filter-off is costly; needed only for a ranker-free phase model) |
30 min: full 0.71-1.10, B 0.60-0.95, D 0.57-0.86 s; 24 h: full 1.68-3.60, B 1.23-2.99, D 1.12-2.84 s. Beta 3 h 0.39-0.57 s (note 100c).
Seed count does not change time (1-seed vs 3-seed trees: ONNX trees are ~0.05 s of a call; not benched separately).
## 5. Not finished at hand-back
Ranker-free phase (net on ALL candidates + one LightGBM, note-103 A2 recipe on the siba head): unfiltered inference x109_w32nf
done/running (tcn53/fpreds + ppreds), chain `s109/chain_nf.sh` (decode / stack / A2 fits / score2-3) still running; numbers will land in
s109/phase/score2-3.log and func/score2.log. It cannot be cheaper than B (filter off costs +0.3-1.1 s at 3 h).
## Verdict
* Design C = ONE network kind (siba w32, 3 members) + ranker 1 + decoder 1 + function trees 1 + stacker 1 + lanes 3 + setback 11:
  phase +0.03 n.s. (5 min +0.36*), function -0.01 n.s., 27-30 % faster, ~20 % less RAM; 21 trained models vs v5c's 36.
* Design D = C with one network member: phase +0.02, function -0.12..-0.17 at >= 30 (5 min -0.4..-0.6*), 41-45 % faster than v5c,
  1.6-2.1x beta at 3 h; 19 models. Within the 0.1-0.2 band only at >= 30 min.
* Recommend C (ties v5c everywhere, simpler and faster); D if the user values speed over short-sample function accuracy.
  To ship C: full-data refit of nothing new on the network side (x100_w32full already has the phase head) + export its phase
  logits, decoder refit on trees-s0 + siba-head blend, 1-seed full-data ranker / trees / stacker, setback 1 seed per group.
