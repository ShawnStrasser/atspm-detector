# 89 — Label-exclusion groups decided with data, not by row review (2026-10-04)
Brief (orchestrator): the user will not review note 87's 1,001 rows. For each exclusion GROUP, refit the function trees
with that group's TRAINING rows restored (labels as originally given, every other exclusion kept) and score the whole
champion recipe on six folds against ONE fixed truth; restore a group unless that hurts beyond noise (simpler = fewer
exclusions). Never per row by model agreement (circular). CPU 4 threads, GPU untouched, locked_v2 asserted absent.
Code `evaluation/g89_groups.py` (fit | stack | score | report | labels); work `%DC_WORK%/x89/` (tables, trees, stack*,
ok*_*.npz, g89_s0.json / g89_s3.json, labels_v4n.json; queue run89.sh).
## Set-up
* Baseline = note 88's labels v4m (dq_suspect recomputed without fault events, actuation-only card rule). The base trees
  reproduce note 88's `f76/function_c_v4m` seed 0 bit for bit (max |dP| 0); its seeds 1 / 2 copied as base seeds.
* Recipe = s80's: 229-feature trees arm c -> context stacker (x69_siba, 3 stacker seeds) -> gate .9 lane decode; lanes,
  pick / health inputs, siba OOF held fixed. Truth = v4l truth_v3s, asserted identical in every arm (>= 30 min 187,444 E /
  182,828 R rows); paired signal-bootstrap 95 % CI (pt). The stacker trains on truth: only the trees' rows change.
* Screen = trees seed 0 (single-seed trees into the stacker, every arm alike); 3 seeds for the combined arm and check_fail.
* Not tested (user rules, not discretion): R1 loop config Count, YR identical to Count, stacked-group relabel, R2 / R4.
## Result (>= 30 min; delta vs base, pt [95 % CI]; base seed 0 .9192 E / .9291 R, base 3 seeds .9195 / .9294)
| group | label rows | + training detectors / rows | E | R | 5 min E | decision |
|---|---|---|---|---|---|---|
| Dec-2024 role disagrees | 621 | +5 / +8,277 | +0.02 [-0.03,+0.07] | +0.02 [-0.03,+0.07] | +0.05 | restore |
| behaviour check failed | 405 | +353 / +9,537 | +0.01 [-0.06,+0.07] | -0.01 [-0.08,+0.05] | -0.11 [-0.22,+0.01] | restore |
|   same, 3 seeds | | | -0.03 [-0.09,+0.04] | -0.05 [-0.11,+0.01] | -0.10 [-0.21,+0.02] | (within noise) |
| unhealthy by actuations | 166 | +70 / +1,290 | +0.01 [-0.04,+0.07] | +0.02 [-0.04,+0.08] | +0.04 | restore |
| data quality (actuation-only, all left in v4m) | 1,644 | +94 / +1,732 | +0.02 [-0.03,+0.07] | +0.02 [-0.03,+0.07] | -0.05 | restore |
| misconfigured (label kept) | 70 | +67 / +1,978 | +0.05 [-0.02,+0.12] | +0.03 [-0.03,+0.10] | +0.04 | restore |
| R5 low-conf print tied by behaviour | 68 | +58 / +1,161 | +0.01 [-0.04,+0.05] | 0.00 [-0.04,+0.05] | -0.02 | restore |
| unusual-layout signals (21) | 438 | +307 / +7,682 | +0.05 [-0.01,+0.11] | +0.04 [-0.01,+0.11] | +0.04 | restore |
| small: R6 noise + sweep + re-read flips | 45 | +32 / +464 | -0.02 [-0.07,+0.03] | -0.02 [-0.08,+0.03] | -0.03 | restore |
| ALL eight together, 3 seeds | | +1,098 / +34,840 | +0.05 [-0.04,+0.14] | +0.03 [-0.05,+0.11] | -0.01 [-0.14,+0.13] | restore all |
No group hurts beyond noise at >= 30 min (every CI spans 0). Single-window flags, not decisive: check_fail 5 min -0.10
(3 seeds, CI touches 0); small 10 min -0.11 [-0.22,-0.01] at seed 0. Combined arm .9200 E / .9297 R, 10 min +0.09.
## Training labels v4n (`research/labels/function_labels_v4n.parquet`; Dec table `%DC_WORK%/x89/dec_role_changed_v4n.parquet`)
* v4m with all eight groups' TRAINING values in `<col>_train` (validated -> pass, dq off, R5 / R6 / unusual back in,
  25 sweep rows' old Other label, 7 re-read flips' old reading; user_ruling flip kept; Dec-role train flag cleared);
  plain columns = v4m, so every scoring set is unchanged; `g89_restored` names the group per row (final layout, 89b).
* `v3_retrain.LABEL_SETS["v4n"]` added; CLEAN no longer re-flags the 4 restored unusual signals in EXTRA_UNUSUAL when
  `g89_restored` says so. Census via v3_retrain = 340,919 training rows = the combined arm's exact count.
* Labelled detectors 15,278 -> 15,303 (sweep). With frame rows: 14,089 -> 14,114; trained 11,893 -> 12,991;
  **still excluded (have data, never trained): 2,196 -> 1,123**: not_checkable not re-admitted 675, window-level only
  (min-on / health3 / health_flag) 171, dead 167 (147 first mis-filed as not train_use, see 89b), no validation data 31,
  validated fail / misconfigured / unhealthy left by another gate 20 / 13 / 3, R1 17, YR identical 10, user rulings 16.
## Draft 20-row sample (superseded by 89b sheet; of the 1,107 still excluded after v4n, user rulings out)
10090 d17 A win | 03070 d2 A nc | 03085 d51 A nc | 05050 d56 O win | 03099 d11 C dead | 05050 d54 O win | 2B357 d2 A nc |
06079 d1 A no_data | 03093 d3 Mid no_data | 2B099 d1 A nc | 2B368 d4 Mid win | 2B081 d3 Mid nc | 08093 d10 Mid not_train_use
| 2C023 d18 A nc | 03099 d37 C dead | a87d162a d4 Mid nc | 2B129 d9 A nc | 2B053 d23 A no_data | 01026 d22 A win | a87d162a
d10 P nc (A/P/C/O = Advance/Presence/Count/Other; nc = not_checkable; win = window-level only).
## For the orchestrator
* Adopted (orchestrator). not_checkable then tested too and restored -> v4o (note 89b), now the pointer everywhere.
