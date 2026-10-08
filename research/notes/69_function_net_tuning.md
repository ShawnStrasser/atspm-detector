# 69 — Function network ("fj") tuned: 13 variants screened, sibling attention wins (2026-10-01)
Code `neural/tcn69_func.py` (one script, defaults = fj: 1 s, 15 ad_all channels, TCN 96 x 7, joint phase + function
loss), cloud wrappers `neural/cloud69/{run69,worker69,setup69}.sh`, scorers `evaluation/screen69.py` (cand64 function
pipeline: trees 0.6 + net 0.4 before the D-lane decode; paired signal bootstrap vs a reference net) and
`evaluation/s69_on67.py` (same nets inside the note-67 champion: context stacker 3 seeds + lane gate .9). Work
`%DC_WORK%/s69/` (screen.json, noise.json, six3.json, on67/), preds `tcn53/fpreds|ppreds/x69_*` (44 files, all kept for a
specialist / combiner study), checkpoints + logs `tcn53/cloud69/`. locked_v2 asserted absent (cand64). Headline = >= 30 min.
## Screen: folds 0 + 3, seed 0, vs fj seed 0 (blend .9061 E, trees .9006); >= 30 min E pt [95 % CI]; class deltas >= 30 E
Noise floor: fj seeds 1 / 2 vs seed 0 = -0.06 [-0.26,+0.13] / +0.00 [-0.17,+0.19]; per class up to +-0.6 (YR, Presence).
| variant | >= 30 E | 5 / 10 min E | by class (Adv / Pres / Count / YR / nonATSPM) |
|---|---|---|---|
| base (fj re-run in tcn69, cloud) | -0.04 [-0.15,+0.07] | -0.16 / -0.10 | +.04 / -.11 / -.04 / +.41 / -.23 |
| base, K = 12 pieces past 2 h (infer only) | -0.01 | same as base | +.04 / -.11 / -.01 / +.54 / -.11 |
| r05: 0.5-s raster | +0.10 [-0.05,+0.25] | +0.03 / +0.18 | +.11 / -.09 / +.35* / +.75* / -.02 |
| r02: 0.2-s raster (receptive field 1/5) | -0.09 [-0.35,+0.20] | -0.14 / -0.10 | +.06 / -.69* / -.07 / +.79 / +.49 |
| r02s: 0.2 s + learned stride-5 pre-stem | -0.05 [-0.19,+0.09] | -0.06 / -0.01 | -.10 / -.24* / -.25 / +.25 / +.52* |
| r01s: 0.1 s + learned stride-10 pre-stem | -0.01 [-0.15,+0.13] | +0.07 / +0.07 | +.05 / -.14 / -.06 / +.17 / +.12 |
| md32: 32 detectors per training window (control) | +0.07 [-0.11,+0.24] | +0.08 / +0.08 | +.08 / -.21* / +.16 / +.37 / +.39 |
| **sibm**: sibling mean-pool on same candidate (+md32) | **+0.31 [+0.04,+0.61]** | +0.66 / +0.65 | +.06 / +.23 / +.20 / +1.25* / +.83 |
| **siba**: sibling attention, log p(c) bias (+md32) | **+0.21 [-0.03,+0.49]** | +0.74 / +0.88 | +.06 / +.14 / +.39 / +1.29* / +.17 |
| cw2: ATSPM class weight 2 | +0.16 [-0.02,+0.35] | -0.18 / -0.10 | **+.49*** / +.23 / +.01 / +.67* / **-.64*** |
| ls: label smoothing .1 | +0.00 | -0.01 / -0.10 | +.14 / -.18 / -.04 / +.83* / -.13 |
| aug: dropout .2 + context-channel dropout .3 | -0.16 [-0.36,+0.07] | -0.18 / -0.26 | -.05 / -.48* / -.25 / +.12 / +.25 |
| cap: width 128, 8 blocks | +0.06 [-0.07,+0.20] | -0.15 / +0.10 | +.18 / -.15 / +.07 / +.33 / +.13 |
| long: 5-120 min training windows | -0.07 [-0.21,+0.07] | -0.15 / -0.09 | -.02 / -.18* / -.20 / +.46 / -.01 |
(* = class CI excludes 0.) Inner-val function accuracy: siblings .82-.88 vs .75-.82 for every other arm.
## Promotion: siba, six folds x 3 seeds (vs fj six folds x 3 seeds; trees .9009 E)
| pool | trees | fj blend | siba blend | siba - fj | siba - trees |
|---|---|---|---|---|---|
| >= 30 min E | .9009 | .9048 | **.9077** | **+0.29 [+0.17,+0.41]** | +0.68 [+0.51,+0.86] |
| >= 30 min R | .9137 | .9175 | .9202 | +0.27 [+0.15,+0.39] | +0.65 [+0.49,+0.82] |
| 5 / 10 min E | .8662 / .8776 | .8775 / .8863 | .8861 / .8953 | +0.86 / +0.90 [+0.63..+1.11] | +1.98 / +1.77 |
| all windows E | .8928 | .8984 | .9029 | +0.45 [+0.34,+0.57] | +1.02 |
Per seed (each alone vs 3-seed fj): s0 +0.30 (vs fj s0) / s1 +0.26 / s2 +0.19, all CI > 0. Per fold +0.62 / +0.07 / +0.30 /
+0.05 / +0.35 / +0.44 (all positive). By class >= 30 E: Advance +0.45 [+0.23,+0.70], Count +0.29 [+0.07,+0.52], nonATSPM
+0.37 [+0.01,+0.73], YR +0.43 [-0.10,+0.94], Presence +0.06 (flat).
## Inside the current champion (note 67: context stacker + gate .9, everything else unchanged)
Champion with fj .9143 E (this script's fixed-gate reproduction; 67b reported .9136) -> with siba **.9169 E, +0.26
[+0.10,+0.42]**; R .9255 -> .9281, +0.26 [+0.10,+0.41]; 5 / 10 min +0.90 / +0.92 E [+0.64..+1.20]; all +0.44. Folds +0.48 /
**-0.42** / +0.33 / +0.20 / +0.39 / +0.61. Class: Advance +0.33*, Count +0.32*, nonATSPM +0.49, Presence +0.08, YR -0.01.
## Verdict (bars: function 1 pt; noise floor ~0.06 pt on 2 folds, 3-seed CIs above)
* **Sibling context is the one lever that pays**: the head sees a p(c)-weighted summary of the other detectors' embeddings
  on the same candidate phase (phase-anonymous). Real on six folds x 3 seeds: +0.29 pt >= 30 min over fj in the fixed
  blend, +0.26 inside the champion stacker; ~+0.9 pt at 5 / 10 min. Under the 1-pt bar at >= 30 min, clear of noise.
  sibm (mean-pool, cheaper) screened at least as well (+0.31) but was NOT promoted (credit); siba / sibm seeds would
  tell them apart.
* Finer time resolution (0.5 / 0.2 / 0.1 s, with or without a learned pooling stem) does not help function: the 1-s
  raster's occupancy / ON-rate / edge-impulse channels already carry the pulse signature. Dropped, as are label
  smoothing, augmentation, more capacity, longer training windows and more inference pieces (all within +-0.1 pt or worse).
* Specialist flag (user idea): **cw2** is clearly better on Advance (+0.49 [+0.21,+0.76]) and worse on non-ATSPM
  (-0.64); siba / sibm / r05 / ls are better on Yellow_Red (+0.75..+1.3, 2 folds). OOF preds kept for a combiner test.
* Cloud: 1 H100 SXM ($3.49/h) + 1 RTX PRO 6000 ($2.09/h) + a 10-min MIG pod to copy checkpoints; 42 runs; one H100 pod
  never started (terminated). **Spent $14.05** (balance $17.56 -> $3.51; volume kept, no pods). All runs finished.
