# 79b — Does detector health help CLASSIFICATION? Ablations inside the champion (2026-10-03)
Question (user): health enters the function champion twice: (1) the context stacker (health_core_ff columns + stack-relative
chi, 16 of the 42 context columns) and (2) the stack pick rule (an unhealthy stack member loses its ATSPM claim). Is it useful?
Champion = note-77 OOF (trees arm c + siba -> context stacker f77, 3 seeds -> D lanes, gate .9, stack pick, twin decode;
truth_v3s; ATSPM stack-aware score). Reproduced: >= 30 min .9171 E / .9282 R. Code `final77/hc79b.py` (stack | score), out
`%DC_WORK%/final_v3_work/f77/hc79b/hc79b.json`. Six folds folds_v4 OOF, stacker seeds 0/1/2 averaged, same fixed LightGBM as
note 67, CPU 2 threads; locked_v2 asserted absent. Paired signal bootstrap, pt [95 % CI], arm minus champion.
Health columns (16): hf_score, hf_status, hf_nfam, 7 family scores (dropout, stuck, chatter, rapid, volume, level, choppy),
stack-relative hf_chi / surge / drop / chat / rel_chi, and pk_unhealthy. hf_clus_n (stack size = structure) kept.
Control: refit of the full 62-column stacker, seed 0, at 2 threads = saved 4-thread seed 0 exactly (0 rows differ).
## Result (positive = the arm WITHOUT health is better)
| arm | >= 30 E | >= 30 R | 5 min E | 10 min E | rows changed >= 30 |
|---|---|---|---|---|---|
| (a) stacker without health columns | -0.003 [-0.030,+0.024] | -0.005 [-0.032,+0.022] | -0.036 [-0.092,+0.020] | -0.056 [-0.102,-0.008] | 460 |
| (b) health columns shuffled within fold | +0.003 [-0.016,+0.024] | -0.003 [-0.024,+0.016] | -0.017 [-0.051,+0.017] | -0.008 [-0.039,+0.023] | 284 |
| (c) stack pick without the health key | +0.001 [-0.001,+0.003] | +0.001 [-0.001,+0.003] | 0 | 0 | 6 |
| (a)+(c) health removed everywhere | +0.000 [-0.027,+0.027] | -0.002 [-0.029,+0.025] | -0.036 | -0.056 [-0.102,-0.008] | 462 |
* Per seed (a) vs same-seed champion +.021 / -.009 / +.012; (b) -.001 / +.003 / +.013 (champion seed spread +-0.02).
* Per fold (a) +.05 .00 +.01 +.02 -.04 -.03; (b) -.02 -.02 +.03 -.02 +.01 +.04. No fold moves more than 0.05 pt.
* 10-min (a) -0.056 has CI < 0, but the shuffled control (b) at 10 min is -0.008 [-0.039,+0.023]: the loss comes from
  removing 16 columns (different splits), not from health information. Not counted as a health effect.
## By class, >= 30 min E (n rows; a / b / c)
| class | n | (a) no health cols | (b) shuffled | (c) no pick key |
|---|---|---|---|---|
| Advance | 59,544 | +0.015 [-0.030,+0.057] | +0.017 [-0.017,+0.046] | +0.005 [0.000,+0.012] |
| Presence | 57,704 | +0.023 [-0.013,+0.060] | +0.010 [-0.023,+0.043] | 0 |
| Count | 31,990 | -0.028 [-0.088,+0.030] | 0.000 [-0.054,+0.054] | 0 |
| Yellow_Red | 9,477 | +0.032 [-0.142,+0.198] | -0.063 [-0.196,+0.065] | 0 |
| non-ATSPM | 29,265 | -0.075 [-0.162,+0.003] | -0.014 [-0.065,+0.033] | -0.003 [-0.011,0.000] |
R is the same within 0.02 pt. Every class CI covers 0.
## Where: detectors the health scorer flags (status suspect / bad or pick-unhealthy; 2.75 % of >= 30 rows, 4,756 E)
* Flagged rows: (a) +0.11 [-0.06,+0.27], (b) +0.21 [+0.02,+0.42], (a)+(c) +0.23 [+0.08,+0.40]: on flagged detectors the
  classifier is slightly BETTER without health. Unflagged rows (183,224): (a) -0.006 [-0.034,+0.021], (b) -0.002.
* Post-hoc subgroup, and worth ~0.006 pt of the headline (0.2 pt x 2.75 %). Read it as "health does not help where it
  should", not as a finding.
* Pick key: pk_unhealthy fires on 0.41 % of >= 30 rows; removing it changes 6 rows (net +1 correct). It never acts below
  30 min (the lane gate / pick only bind there).
## Verdict
* Health does not help classification: removing it from the stacker, shuffling it, or removing it from the pick rule all
  stay inside +-0.03 pt at >= 30 min (both sets, CIs cover 0); no class and no fold moves beyond noise.
* The only CI-excluding effect (10 min, (a) -0.06 pt) fails its shuffled control. On flagged detectors health slightly hurts.
* For the orchestrator: health can come out of the classifier (16 stacker columns + the pick key) at no measurable cost,
  which removes the health -> function dependency in the package; it stays a separate output. Not done here (package and
  model/ untouched); if adopted, refit the 46-column stacker (arm a is that refit, OOF) and re-run check.py.
