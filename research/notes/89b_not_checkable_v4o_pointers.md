# 89b — not_checkable group tested (restored -> labels v4o), pointers flipped, 20-row sanity sheet (2026-10-04)
Brief (orchestrator, after adopting note 89): (1) flip every label pointer to the final table and build the siba
func_rows (no GPU job); (2) same group test for "not_checkable, not re-admitted"; restore if not worse beyond noise ->
v4o; (3) 20 random still-excluded labels (user rulings out) as a tiny sheet. CPU 4 threads; locked_v2 asserted absent.
## Label-table layout (changed from note 89's first build)
* Restored TRAINING values live in `<col>_train` (validated, dq_suspect, exclude_train_score, train_use, unusual_layout,
  label_print_first, tier); the plain columns stay = v4m (= v4l on every scoring column, asserted by oof88
  check_truth_same). Reason: atspm_score's realistic set reads `validated`; writing pass into it would have moved the
  scoring set. `v3_retrain.load_labels` copies `<col>_train` over the plain columns whenever `g89_restored` exists
  (G89_TRAIN_COLS). Rebuilt v4n reproduces note 89 exactly (340,919 training rows).
## not_checkable (base = v4n, trees 3 seeds; `G89_SRC=v4n`, work `%DC_WORK%/x89/v4n/`)
* Group = every labelled row with validated not_checkable -> trained like a pass (1,560 label rows; the ones already
  re-admitted by the print-high rule change nothing): +657 detectors / +16,835 training rows (357,754).
* >= 30 min, paired vs base, pt [95 % CI]: seed 0 E -0.00 [-0.09,+0.08], R -0.01 [-0.10,+0.07]; **3 seeds E .9205
  +0.05 [-0.01,+0.11], R .9300 +0.04 [-0.02,+0.09]**; 10 min -0.02 [-0.12,+0.09]; 5 min E -0.05 [-0.16,+0.07],
  R -0.05 [-0.17,+0.06]. Base 3 seeds .9200 / .9297 (= note 89's combined arm). Not worse beyond noise -> RESTORED.
* **v4o** = v4n + not_checkable: `research/labels/function_labels_v4o.parquet`, Dec table
  `%DC_WORK%/x89/dec_role_changed_v4o.parquet` (= v4n's). Training rows 306,079 (v4m) -> 340,919 (v4n) -> 357,754;
  trained labelled detectors 11,893 -> 12,991 -> 13,648.
* Still excluded with data: 2,196 (v4m) -> 1,123 (v4n) -> **466**: every sample cut by a per-window gate (min-on 5 /
  health3 / health_flag) 202, dead 167, no validation data 31, behaviour fail / misconfigured / unhealthy kept out by a
  second gate 20 / 13 / 3, user rulings 16, YR identical to Count 10, R1 4. (Note 89's "not train_use 147" was dead
  rows mis-filed by the reason order; fixed.) List `x89/v4n/excluded_v4o.parquet`.
## Pointers (all now v4o; v4l / v4m / v4n kept for comparisons)
* `rpath.LABELS_CURRENT` = v4o, `DEC_ROLE_CURRENT` = x89/dec_role_changed_v4o; `v3_retrain.LABEL_SETS["v3s"]` (the default
  slot every trainer asks for) = v4o; named slots "v4l", "v4m", "v4n", "v4o" added.
* `final88/oof88.py` package stages: `PKG` = env DC_PKG_LABELS, default v4o (labels, atspm V3S, fit83 LAB_V4L, trees
  `f76/function_c_v4o` = the 3-seed not_checkable arm, copied; `function_c_v4n` = note 89's combined arm). Smoke test:
  check_truth_same + use_trees pass. Package not rebuilt.
* siba func_rows: `neural/flab89_v4o.py` -> `%DC_WORK%/tcn53/func_rows_v4o.parquet` (ok 78.5 % of 456,033 rows vs 65.5 %
  in func_rows_v4l; nothing newly dropped); v4n twin `flab89_v4n.py` -> func_rows_v4n. No GPU job started.
* review81 / review87 assert LABELS_CURRENT is v4l (stage-specific; left as they are).
## Sanity sheet `review/label_sanity_20.xlsx` + `label_sanity_20_charts/` (`evaluation/sanity89.py`)
* Pool = the 466 minus user rules / rulings (R1, YR identical, user_ruling) = 436; 20 at random (seed 89). Columns: #,
  signal, detector, label, why it is left out (one plain sentence), chart (review87 layout, saved events), print, answer;
  question "Should this label be used? Y / N / ?". 19 of 20 have a print link.
* Rows: dead 9 (2B343 d51/53/59, 2B346 d56, 2B530 d29, 2B063 d16, 10078 d15, 2B358 d3, 2C071 d8), window-level 7 (10028
  d16, 08150 d2, 2B069 d25, 11021 d19, 10018 d2, 08130 d22, 13025 d39), misconfigured 2 (03099 d17 / d28 pulse presence),
  no validation data 1 (2B050 d17), fail 1 (03085 d18 Bike). Spot-check 10028 d16: chart drawn, 2,888 actuations, cut
  by the health3 / min-on gates only.
## For the orchestrator
* Final training labels = v4o. The last refits (siba x3 on func_rows_v4o, phase TCN, package via oof88 pkg*) read it by
  default. If the user answers N on many window-level rows, the per-window health3 gate is the next group to test.
