# 62 — Function on short samples (5 / 10 min): twin decode + short-window weighted head (2026-10-01)
Baseline = note 59 set-up (229-feature function arm, D lanes, stack-pick decode gated >= 30 min, stack_loser nonatspm_ap):
ATSPM .8914 E / .9042 R (3 seeds); m5 .8607, m10 .8732. Six folds OOF, note-55 step-4 rows, paired signal bootstrap (pt).
Screen = seed 0 vs base seed 0 (base@s0 .8905 / .9034); keep if overall CI > 0, then 3 seeds. CPU, <= 6 threads.
Code `trackA/s62_short.py` (pairs / twin / gate / bias / fit / score / combo / twinlong); work `%DC_WORK%/s62/` (logs/,
*.json, fit/). locked_v2 asserted absent. Decision rules use hi-res event times only.
## Diagnosis (realistic rows, note-61 OOF): short vs >= 30 min error 12.0 vs 8.6 pt
A->wrongA 5.1 vs 2.7 (YR->Count 1.42 vs .54, Adv->Pres 1.01 vs .52), nonA->A 4.0 vs 2.4, A->nonA 2.6 vs 3.5. Short windows
over-call ATSPM and swap stop-bar twins; < 10 actuations: .861 acc vs .89-.90 above 40.
## 1. Short-window twin decode (no lanes, no refit)
Pairs on the same predicted phase, 5 / 10-min windows (>= 3 ONs each; 227,498 pairs, 26 s): chance-corrected share of ONs
matched within +-0.5 s at the best lag (+-2 s, ln6 match_frac), both ways. Twin = min(both ways) >= thr, |lag| <= 1 s.
Within twins at most one detector per constrained class; (detector, class) pairs taken by probability. Seed 0, delta pt E [CI]:
| variant | thr .3 | .4 | .5 | .6 | .7 | .9 |
|---|---|---|---|---|---|---|
| Count only, loser -> next class | +0.04 | +0.09 [.05,.13] | +0.09 | +0.07 | +0.05 | +0.02 |
| Count + YR, next class | +0.07 | +0.12 [.08,.16] | +0.12 | +0.09 | | |
| all four ATSPM, next class | +0.02 | +0.12 [.06,.18] | +0.12 | +0.10 | +0.08 | |
| **Count + YR, loser -> the other of Count/YR, else keep (cy)** | +0.09 | **+0.135 [.10,.18]** | +0.13 | | | |
thr .2 hurts (-0.14 / -0.36 all4). Nested threshold (picked on the other five folds) chooses .4 or .5 in every fold and
reproduces the fixed-.4 numbers (C+YR +0.12, all4 +0.12, Count +0.085). Variants at .7: yr-only / joint pair / lag 2 s /
volume ratio 1.5 all +0.05 (= Count next). Control (same number of NON-twin same-class pairs, min match < .2, all4):
**-0.37 [-0.41,-0.33]** -> the co-actuation is the signal, not "fewer duplicates". Moves (C+YR next, s0, R): Count->YR 441
right / 161 wrong, YR->Count 129 / 40; Count->Advance mostly wrong (51) -> cy keeps the loser inside Count/YR.
Seeds (C+YR next .4): s1 +0.12, s2 +0.13 (CI > 0). all4 adds nothing over C+YR -> C+YR kept. Same rule on >= 30-min rows
without a lane (18,748): +0.01 [0.00,+0.02] at every threshold -> not worth it.
## Gate (lanes below 30 min, measured)
Pair same-lane accuracy vs print lanes (stg): ln5 lanes m5 .712, m10 .848, m30 .920, h1 .931, full .954; D m30 .935 (D has no
m5 / m10 cues). Lane decode at m10 with ln5 lanes -0.21 [-0.28,-0.14] overall (m10 -1.44); m5+m10 -0.77 (m5 -4.08).
No 15 / 20-min windows exist. **Gate stays at 30 min.**
## 2. Training mix (refit, seed 0)
Short rows are 28 % of training (already = their scoring share). Head with short rows weighted k, or a head trained on
short rows only; `route:` = its probabilities on 5 / 10-min rows, base everywhere else.
| cfg @s0 | overall E [CI] | short pooled | m5 / m10 |
|---|---|---|---|
| shortw2 (one head for all) | -0.04 [-0.10,+0.03] | +0.11 | +0.18 / +0.03 |
| shortw4 (one head) | -0.01 [-0.09,+0.08] | +0.34 | +0.50 / +0.18 |
| route shortw2 | +0.03 [-0.00,+0.07] | +0.11 | |
| **route shortw4** | **+0.095 [+0.05,+0.14]** | +0.34 [+0.19,+0.49] | +0.50 / +0.18 |
| route shortw8 | +0.091 [+0.05,+0.13] | +0.32 | +0.47 / +0.19 |
| route shorthead (5/10 rows only) | +0.06 [+0.01,+0.12] | +0.22 | |
| route shorthead30 (5/10/30) | +0.06 [+0.02,+0.11] | +0.23 | |
| control: route base@s1 / @s2 | -0.01 / -0.02 n.s. | +0.03 / -0.07 | |
One weighted head hurts long windows; routing keeps them untouched. Weight 4 chosen among {2,4,8} on seed 0 (8 = same).
Seeds: route shortw4 s1 +0.04 [+0.01,+0.08], s2 +0.09 [+0.06,+0.13]. 3. non-ATSPM prior x alpha on short rows: nested
alpha .8-1.0, -0.00 n.s. (all alpha >= 1.25 lose) -> dropped.
## Combined, 3 seeds (vs .8914 E / .9042 R)
| set-up | ATSPM E | dE [CI] | ATSPM R | dR [CI] | m5 E | m10 E |
|---|---|---|---|---|---|---|
| base + cy .4 twin decode | .8928 | +0.14 [+0.10,+0.18] | .9057 | +0.14 [+0.10,+0.19] | .8662 | .8776 |
| route shortw4 | .8922 | +0.08 [+0.05,+0.12] | .9050 | +0.08 [+0.04,+0.11] | .8655 | .8744 |
| **route shortw4 + cy .4** | **.8936** | **+0.22 [+0.17,+0.27]** | **.9064** | **+0.22 [+0.17,+0.27]** | **.8710** | **.8788** |
Combined: short pooled +0.79 [+0.61,+0.99] (m5 +1.03, m10 +0.57), >= 30 min unchanged by construction. Per seed: s0 +0.23,
s1 +0.19, s2 +0.23 (all CI > 0). YR<->C errors 1.33 -> 1.19 pt of rows. Under the 1-pt overall bar; clears "CI > 0".
## Verdict / for the orchestrator
* Keep (by the brief's rule): the cy twin decode on 5 / 10-min windows (numpy, hi-res, no model) and a second, short-window-
  weighted function head (weight 4) routed for samples <= 10 min. Gain +0.22 pt overall, ~+0.8 pt on 5 / 10 min.
* Cost: one more LightGBM head in production (same 229 features). The twin decode alone (+0.14) needs no new model.
* Caveats: thr .4 and weight 4 picked on OOF (nested threshold agrees; w8 equal); the twin decode partly overlaps the open
  user question on scoring radar YR<->Count twin swaps (if those become "correct", part of this gain moves into scoring).
