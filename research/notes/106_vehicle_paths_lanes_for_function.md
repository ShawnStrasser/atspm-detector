# 106 — Vehicle paths / lane structure for the function model: three architecture ideas (2026-10-06)
User idea: the function model does not "see" which detectors are upstream / downstream of which, travel times, lane
groups, lanes per phase; a vehicle-path step should help function and stop lane-inconsistent answers. Set-up = v5b
2026 six-fold OOF (2026 trees + siba w32 mean3 -> context stacker -> gate .9 lane decode; v4q truth; Sept-2026 rows).
Reference reproduced: >= 30 min E .9308 / R .9398 (v5b .9307 / .9397). Lane truth = f102 (v4q + user v1 answers; notes
105 / 107 truth not used). CPU 6 threads, no GPU / pod. locked_v2 asserted absent. Code `research/code/final106/`
(s106 cache / stack / score, path106, tree106, lane106, joint106); work `%DC_WORK%/s106/` (scores.json, laneeval106.json,
joint/eval106.json). CI = paired signal bootstrap, pt.
## 1. Vehicle-path features per detector (path106; prediction-free, hi-res only, windows >= 30 min)
ON-time correlogram (0.5 s bins, +-15 s) for every pair on the same predicted phase: zero-lag excess (co-location),
forward / backward peak 1-12 s (lag = travel time, excess / own ONs = share of actuations that propagate). 26 columns:
best downstream / upstream partner (share, lag, sharpness, volume ratio, partner-side share), # strong down / up edges,
chain position / depth, farthest travel time, any-mate propagation, co-location, D lane-mates. Coverage 39 % of rows.
| arm (seed 0 unless noted) | >= 30 E delta [CI] | R | note |
|---|---|---|---|
| stacker + 26 path cols | +0.03 [-0.04,+0.09] | +0.02 | gain share 0.8 %; shuffled +0.01 |
| trees + 26 path cols -> stacker | +0.01 [-0.09,+0.11] | -0.01 | gain share 3.8 %/fold; 5 / 10 min -0.01 / -0.03 |
| trees alone (+ lane rule), with vs without path cols | **+0.21 [+0.09,+0.33]** | +0.20 [+0.08,+0.33] | shuffled control +0.01 / -0.00 |
Reading: the path signal is real at the tree level (beats its control), but the full model already has it — the trees
already carry sibling lead-lag features (lagsib_best_lag, lagany_*, call43_*: top-10 gain) and the stacker gets the
lanes D context + siba net. Nothing left for the headline. Not promoted (CI covers 0 on the full model).
## 2. Iterate function <-> lanes (lane106)
Lanes D (note-102 2026 recipe, 6 folds x 3 seeds, lam per held-out fold) with its function input (pair type, the 7
probabilities of the c2 block, decoder anchors) taken from the STACKER OOF instead of the trees; new lanes -> stacker
lane context + gate / per-lane decode -> stacker refit. R0 = 2026-trees lanes (f102), R1 = lanes from R0's stacker,
R2 = lanes from R1's stacker. Pair AUC m30 / h6 / full: .963/.975/.978 -> .966/.977/.980 (R1, R2 same).
| 3 stacker seeds, >= 30 min | E | R | vs R0 E [CI] | vs v5b (f77 lanes) E [CI] | R vs v5b |
|---|---|---|---|---|---|
| R0 (2026 trees lanes) | .9304 | .9394 | - | -0.04 [-0.11,+0.03] | -0.04 |
| R1 | .9312 | .9403 | +0.08 [+0.00,+0.16] | +0.04 [-0.05,+0.14] | +0.04 [-0.05,+0.12] |
| R2 | .9311 | .9401 | +0.07 [-0.00,+0.14] | +0.03 [-0.06,+0.12] | +0.02 n.s. |
Seed-0 screens vs R0: R1 +0.08 [0.00,+0.16], R2 +0.08 [+0.01,+0.16] -> 3 seeds above. R1 vs v5b by window (E): m30 +0.09, h1 +0.00, h3 +0.09, h6 +0.09, h24 -0.03, full -0.12 (all n.s.); 5 / 10 min 0 (no lanes). By class: Advance
+0.40*, Presence +0.13*, YR +0.25*, Count -0.13, non-ATSPM -0.72* (trade, as notes 67 / 67b). Lanes: n_lanes exact (14
Sept windows) f102 .8481, L1 .8466 (-0.15 n.s.), L2 .8482 (+0.02 [-0.34,+0.37]); f77 (v5b's lanes) .8445.
Not nested (standard OOF stacking: fold j's stacker saw fold k); nested check skipped: optimistic gain already n.s.
## 3. Structured per-phase decode (joint106)
Per (signal, window >= 30 min, predicted phase): roles {A, P, C, YR, non-ATSPM} and lane sets (<= 4 lanes, spanning)
chosen jointly: sum log P(role) + w_p * pair LLR log(p/.5) / log((1-p)/.5) (lanes D P(same), f77) - mu * extra spans -
lam * n_lanes - nu * [Advance shares a lane with a stop-bar zone that leads it in the ON-time lead-lag]; hard <= 1 per
ATSPM class per lane (spanning holds it on every lane; two Count => two lanes), every lane anchored (single-lane ATSPM).
Coordinate ascent with exact deltas from two starts (current decode; argmax + greedy fill); weights nested per fold.
| grid (nested pick) | >= 30 E [CI] | R | n_lanes exact (v5b lanes .8445) | phase-windows with > lanes A/P/C (v5b 6.1 %) |
|---|---|---|---|---|
| strong lanes (w_p .5-1, lam 1-2) | -0.19 [-0.34,-0.05] | -0.19* | .8431 (-0.14 n.s.) | 4.7 % |
| wide (w_p .1-.5, lam 0-1, mu, nu) -> picks w_p .1 | +0.01 [-0.13,+0.14] | +0.00 | .8283 (-1.62 [-2.38,-0.76]) | 6.8 % |
Best lane count of any of 40 configs .8432 (< .8445); best function E .9315 at w_p .1 (= almost no lane constraint). The
upstream rule (nu) never helps. Strong lanes cost Count -0.52* / Presence -0.46*, win YR +0.48* / non-ATSPM +0.60*.
## Note-102 review phases (23; phase majority right after each idea)
v5b lanes (f77): wrong function 2/12, wrong phase 1/8, lane grouping 0/3; 2026 lanes f102 0/12, 0/8, 0/3; iterate L1 /
L2 1/12, 0/8, 0/3; joint strong 1/12, 1/8, 0/3; joint wide 1/12, 0/8, 1/3. No idea fixes the "wrong function" cases:
the lane loops there are called Mid / Bike / Other by every function model (trees, stacker), so feeding function back
into lanes or deciding jointly keeps them lane-less. Lane-inconsistent outputs: argmax 9.1 %, v5b 6.1 %, R1 7.0 %.
## Verdict
No idea clears the noise floor vs v5b; nothing adopted, packages untouched. The model already sees vehicle paths (tree
lead-lag features + lanes D context); harder lane constraints trade Count / Presence for YR / non-ATSPM (notes 54 / 67b).
Recommended (only if the orchestrator wants lane outputs + function from one place, accuracy-neutral): (a) give lanes D
the stacker's probabilities instead of the trees' (iterate once; ties v5b, +0.08 vs its own 2026-trees baseline); (b) keep
the greedy gate-.9 decode — the joint decode is not better for function or lane count. Real gains on the 12 "wrong
function" phases need labels for low-volume lane loops (Bike / Mid confusion), not architecture.
