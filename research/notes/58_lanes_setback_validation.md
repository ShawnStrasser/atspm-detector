# 58 — Lanes and setback: separate models vs outputs of the main model (validation plan step 5) (2026-10-01)
CPU, six folds of folds_v4, all OOF; inputs = frame v6e predicted phase + note-57 function arm (229 feat, 3 seeds). locked_v2
absent everywhere; nothing shipped. Code `lanes/ln8_validate.py`, `trackA/sb7_validate.py`. CI = paired signal bootstrap, pt.
## Lanes
Truth: v3s print lanes (ln1 rules; 2,455 phases / 504 signals incl. the 33 re-admitted radar-over-loop signals; note 42:
2,303 / 476); 77,102 labelled pair-windows, nine Sept windows, 63 % same lane. Validation-only second source: RL/CL/LL text,
2,477 pair-windows / 44 signals. Pair models (note-42 recipe, 3 seeds): A = cues + context + predicted-function pair type
(refit on v3s); A_old = note 54's fold models; B = no function input; C2 = function model only (its 229 features + 7 probs of
both detectors, min/max, no cues); D = A + C2 (joint). Decodes: func = note 42 (function anchors, <= 1 A/P/C/YR per lane);
free = no function at all; C1 = no pair model (n_lanes = max #A/#P/#C predicted, k-th busiest per role = lane k). lam per fold.
Pair AUC, >= 2-lane phases, m30/h6/full: A_old .936/.958/.967, A .937/.959/.967, B .874/.935/.956, C2 .932/.944/.945,
**D .963/.975/.979**; seed spread <= .0008; control D with the C2 block permuted across pairs .935/.958/.967 (= A).
| Sept scope (m30 / full) | n_lanes exact | exact >= 2 lanes | lane set exact | pair acc >= 2 lanes | text pairs |
|---|---|---|---|---|---|
| A_old.func (as was) | .848 (.803 / .884) | .721 | .910 | .900 | .874 |
| A.func (refit v3s) | .846 (.801 / .882) | .716 | .911 | .901 | .872 |
| B.func (no pair type) | .840 (.789 / .881) | .703 | .904 | .882 | .870 |
| B.free (pure behaviour) | .715 (.644 / .774) | .481 | .841 | .816 | .742 |
| C2.func (function model only) | .837 (.797 / .863) | .701 | .899 | .880 | .823 |
| C1 rule (no pair model) | .793 (.759 / .809) | .684 | .813 | .700 | .815 |
| **D.func (joint)** | **.854 (.811 / .886)** | **.727** | **.922** | **.923** | .871 |
A.free .776. Phase found .935, always-1 .536, coverage .933. vs A.func: B.func n_lanes -0.63 [-1.04,-0.26], lane -0.72;
C2 -0.92 [-1.57,-0.28], lane -1.22; C1 -5.3, lane -9.8; **D +0.80 [+0.34,+1.27] n_lanes, +1.08 [+0.77,+1.39] lane set,
+2.28 [+1.77,+2.86] pairs** (full n_lanes +0.41 n.s.); text D -0.08 [-2.6,+1.9]. A_old vs A: all within +-0.2, CIs cover 0.
ATSPM through the per-lane decode (stack pick, stack_loser nonatspm_ap, ln6 pick inputs fixed; step-4 rows, 261,528 E):
| lanes | ATSPM E / R | vs ln5 as was | vs A.func | full E | stack_extra |
|---|---|---|---|---|---|
| ln5_lanes_v3s (as was; 442-feat function input) | .8897 / .9027 | - | -0.07 [-0.12,-0.03] | .9111 | 383 |
| A.func | .8905 / .9034 | +0.07 [+0.03,+0.12] | - | .9115 | 381 |
| **D.func** | **.8914 / .9042** | **+0.17 [+0.07,+0.27]** | +0.09 [+0.00,+0.18] (R [-0.01,+0.18]) | .9135 | 350 |
| B.func · B.free | .8863 · .8561 | -0.34 · -3.36 | -0.41 [-0.52,-0.32] · -3.43 | .9116 · .8915 | 377 · 259 |
| C1 rule | .8891 / .9018 | -0.07 n.s. | -0.14 [-0.36,+0.05] (full -0.40) | .9075 | 590 |
| argmax (no lanes) | .8893 / .9020 | -0.05 n.s. | -0.12 [-0.36,+0.11] (full -0.69 [-1.19,-0.23]) | .9046 | 632 |
**Verdict.** Function-free grouping fails (n_lanes -13 pt, ATSPM -3.4): the per-lane role rule is what counts lanes, so
lanes cannot be separate from the function head. Lanes from the function model alone (C2, C1) lose to the cue model.
Joint **D** beats as-is on every print measure (CI clear) and ATSPM vs as was (+0.17); vs refit A the ATSPM gain is at the
noise edge (lower bound 0.00). Keep rule (CI clear of 0 on the lane outputs) -> **D kept**; cost = function frame features
at lane time (already computed). Text pairs (44 signals, CI +-2 pt) neither confirm nor contradict D.
## Setback
Truth: printed single distance of print Advance / Mid (training + released prints), role / unusual from v3s, exclude rows out.
Windows m30 a-d, h6 a/b, h24 a/b, full66 (cycles cut to the window; note 41: 66 h only). Printed Advance, >= 1 actuation:
2,465 / 1,250 / 1,253 / 628 det-windows (m30/h6/h24/full), 129 signals. P50, 3 seeds, per window group; medAE / %<=50 / %<=100 ft:
| estimator | m30 | h6 | h24 | full |
|---|---|---|---|---|
| **all 60 features (as is)** | **25.7 / 68.9 / 86.9** | 22.4 / 69.3 / 86.6 | 22.6 / 70.3 / 87.8 | **20.4 / 70.2 / 88.9** |
| nofunc (no function / lane input) | 25.1 / 67.2 / 85.8 | 25.1 / 66.2 / 84.9 | 22.7 / 68.7 / 86.0 | 21.9 / 69.1 / 87.7 |
| nophys (no travel-time block) | 25.3 / 67.6 / 86.3 | 23.3 / 67.4 / 85.8 | 23.4 / 69.0 / 86.1 | 22.2 / 67.8 / 88.4 |
| own behaviour only | 26.7 / 66.1 / 86.3 | 25.0 / 68.4 / 85.8 | 24.2 / 69.6 / 86.6 | 23.3 / 70.4 / 88.5 |
| physics (note 39 rule, covered rows only) | 75.1 / 34.9 / 62.3 | 58.8 / 45.2 / 67.4 | 53.6 / 48.5 / 72.0 | 43.6 / 54.2 / 73.1 (n 275) |
std (timing + make-up) m30 25.8 / 65.6 / 82.4, full 34.6 / 61.0 / 78.3; timing only ~43 / 54 / 74; fold median 85 / 26 / 73;
shuffled 75-79 ft / 28-33 %. Seeds 3-5: all w50 68.5-69.1 (m30) / 68.9-70.7 (full); nofunc 66.7-66.9 / 67.7-69.1. Learned on
the physics-covered full rows 22.4 / 68.4 / 87.6. Production path (560 of 628 printed Advance predicted Advance; stop-bar -> 0):
all 23.9 / 65.8 / 87.0, nofunc 26.1 / 64.5 / 85.8 (full). Full, all: loop 14.7 ft (417), video 29.6 (140), radar 65.2 (71).
Pooled over windows: all vs nofunc w50 **+1.9 [+0.3,+3.7]**, w100 +1.4 [+0.1,+2.7], medAE -0.8 [-2.8,+1.0] ft; travel-time
block +1.6 [+0.5,+2.7] w50; own vs nofunc +0.3 [-0.8,+1.6] (phase timing adds nothing over own behaviour). Per group only h6 clears.
**Verdict.** Keep as is: function / lane inputs and the travel-time block pay ~2 pt within 50 ft (CI clear, small). Simplest
function-free fallback = own behaviour only (-1.6 pt). New: works from 30 min (25.7 ft vs 20.4 at 66 h); radar poor (~65 ft).
## OOF for steps 4 / 7, caveats (ln6 pick inputs built from the old lanes, as in note 57; free-decode lam at grid floor 0)
* `%DC_WORK%/lanes/ln8/lanes_<cfg>.parquet` (ln5 format; every frame signal / period / window >= 30 min; A.func, D.func, B.func,
  B.free, C1.rule), `p_<var>.parquet`, `eval.json`, `atspm.json`; `%DC_WORK%/trackA/setback/sb7/oof.parquet`, `results.json`.
