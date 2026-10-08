# Function labels, v2 — provenance

`function_labels_v2.parquet` is the detector-**function** truth from 2026-09-22 onward.
It replaces the raw config mapping for every purpose: training, ablation and scoring.

## Where it comes from

1. **Base** — the hand-maintained config export
   `%DC_WORK%/data/labels/detector_config_current.parquet` (8,659 channels / 466 signals,
   pulled 2026-09-21), free-text `Function` strings mapped to the five model classes by
   `%DC_WORK%/data/labels/function_label_map_v2.csv`. Channels outside 1–64 dropped →
   **8,633 rows**.
2. **Corrections** — the traffic engineer's review of the 79 function misses on the 143
   locked signals, `review/function_misclassified_locked143_REVIEWED.xlsx`. His rules,
   applied by `dc_work/trackA/a1_labels.py`:

   | `correct_function` cell | what happens |
   |---|---|
   | blank | keep the config label (10 rows) |
   | `?` | **drop** the detector from function training *and* scoring (4 rows) |
   | `Bike`, `Bike Loop`, `Departure` | → `Other` (7 rows; not classified classes yet) |
   | a class name | that class is the truth |

   Result: **37 labels changed, 38 confirmed, 4 dropped**. The changes are
   Presence→Count 13, Presence→Advance 8, Other→Advance 6, Advance→Other 3,
   Count→Presence 2, Presence→Other 2, Other→Presence 2, Other→Count 2, Count→Other 1.
   Every correction is on a NEWTEST (locked) signal, so **no training label moved**;
   what moved is the exam's truth.

## Columns

`DeviceId, DeviceName, Detector, cfg_phase, config_function` (raw string),
`func5_config` (the old mapped label), **`func5`** (the truth — NaN when dropped),
`label_source` ∈ {config, review_confirmed, review_confirmed_config, review_corrected,
review_dropped}, `reviewed`, `drop_from_use`, `review_raw`, `review_comment`,
`description` (the technician's channel text), `locked` ∈ {"", TEST, NEWTEST}.

**Use `func5`, and skip rows where it is null.**

## Effect (locked 143 signals, frozen `final_v2` predictions, full 66 h window)

| truth | 5-class | A/P/C |
|---|---|---|
| config label | .7539 | .7797 |
| corrected label | **.8675** | **.9156** |

## Next round

`review/function_disagreements_round2.xlsx` — 586 confident (p ≥ 0.80) model-vs-label
disagreements on 193 signals, all labelled signals this time, rows already reviewed in
round 1 removed. Sorted so the signals with the most confident disagreement come first.
Filling in `correct_function` / `comment` there and re-running `a1_labels.py` with the
second workbook extends this table.

---

# Function labels, v3 — cabinet prints + whole intersection

`function_labels_v3.parquet` (git-ignored, like every parquet) is built by
`research/code/cabinet/cab_final.py` from the cabinet-print readings
(`%DC_WORK%/cabinet/print_labels.parquet`, guide `cabinet_print_guide.md`) and v2. One row per
(`DeviceId`, `detector`) on **training/dev signals only** — the locked TEST and NEWTEST signals are
filtered out and asserted absent. The phase target is unchanged (official timing: `phase_target`,
`phase_target_type`, `switch_phase`, `additional_call_phases`).

## Classes and sources
Seven function classes: Advance, Presence, Count, Yellow_Red, Mid, Bike, Other. The v2 label is
carried as `func5_v2` and, in seven classes, `func7_v2` (a v2 Other whose config text names mid /
bike becomes Mid / Bike). `source` says where the chosen label came from: `print_high`,
`print_medium`, `print_low`, `config`, `review_round1`, `whole_intersection_other`.

## Which label wins — two variants
* `label_print_first` (= `function`, `source`): print label at high confidence > v2 > print label at
  medium / low confidence > whole-intersection Other.
* `label_print_agree` (`source_agree`): the AGENTS.md rule — a high-confidence print label replaces a
  *different* v2 label only where the six-fold out-of-fold 7-class model prediction agrees with the
  print (`oof_pred`, `oof_p`: seed-mean, full window, note 25 variant b7); otherwise v2 stays. Where
  v2 has no label the print label is used in both variants.
* `print_overrides_v2` marks rows where a high print label differs from v2 (the user's review list).

## Whole-intersection Other
At a signal whose print was read completely (`tier` complete_high / complete_mixed), every active
channel (>= 5 actuations in the newest window) that is not on the print and has no v2 label is
labelled Other (`source` whole_intersection_other). Active channels not on the print at an
`incomplete` signal stay unlabelled (the print may be stale).

## Tiers and exclusions
* `tier`: complete_high (print read, every active channel accounted for, every classified print
  detector high), complete_mixed (the same with some medium / low), incomplete; null = no print.
* `unusual_layout`: rare custom set-ups (live radar over live loops, several intersections on one
  controller, rewired since the print) — excluded from training, scored separately.
* `dead`: a print detector with no actuations in the newest window (listed for maintenance).
* `dq_score`, `dq_flags`, `dq_suspect`: data-quality checks run with the print-derived location
  (`dq_core.py`, note 24). Information only; not an exclusion by themselves.
* `train_use` = a label, not unusual_layout, not dead.
* `validated` (pass | fail | not_checkable | no_data), `failed_checks`, `validation_reason` (plain English),
  `validation_numbers`: does the detector behave like its label on the hi-res data (engineer's rules, thresholds set
  once on trusted rows; `label_check.py`, note 29). `train_use_validated` = `train_use` and `validated == pass`.
  `unusual_layout` is also read from a record's note (6 signals had it only there).
* `pending_user`: rows whose label depends on an open question to the user. An answer is one
  `relabel` line in `%DC_WORK%/cabinet/final_rulings.csv` and a re-run of `cab_final.py`.

Print-reading columns are kept for analysis: `print_function`, `print_subtype`,
`print_confidence`, `confidence_reason`, `technology`, `lane_index`, `lanes_spanned`, `lane_type`,
`n_lanes_phase`, `print_flags`.

## Released NEWTEST rows in v3 (note 36, 2026-09-24)
A seed-fixed, region-stratified random half of NEWTEST (71 signals, `%DC_WORK%/official/newtest_released.csv`)
was moved into training / dev; their rows come from `function_labels_locked_v1` (same label rule) and are
appended by `cab_final.py` (step 9, or `--append-released`), marked `released_from_newtest` = True, with
`validated` = pending and `train_use_validated` = False until the user's review of the behaviour checks.
They carry no OOF / DQ columns. The hold-out is now `%DC_WORK%/official/locked_v2.csv` (43 TEST + 72 NEWTEST).

---

# Function labels, locked v1 — the answer key for the hold-out signals

`function_labels_locked_v1.parquet` (git-ignored) is built by `cab_final.py --locked-consolidate
--locked-labels-only` from the SEPARATE locked store `%DC_WORK%/cabinet_locked/` (the 186 original hold-out
signals: 43 TEST + 143 NEWTEST). **Labels only: no model prediction, DQ or metric was computed for it.**
Same schema and label rule as v3 where applicable (`function` = `label_print_first`; `label_print_agree`
has no model input here — where a high print label differs from v2, v2 stays), plus `set` (TEST / NEWTEST),
`released_from_newtest`, `func5_config` (the raw config label before the round-1 review), `label_use`
(labelled, not unusual_layout, not dead; v3's `train_use`) and `validated` = pending (the behaviour checks
wait for the user's review). Cleansing rules are the training ones (guide + rulings seeded from the
training store) plus `fix_uncoded_zones.py`: uncoded long radar zones from the stop bar -> Presence,
uncoded stop-bar zones by position, undrawn radar CO channels 49–52 -> Advance (low).

---

# Function labels, v4l — label pass of note 80 (2026-10-03)

`function_labels_v4l.parquet` (git-ignored) has the v3s schema and column names (`truth_v3s` = THIS table's scoring
truth, so every scorer takes the path) plus `rule_v4l` and `label_version`. It is built from a separate copy of the
training print store, `%DC_WORK%/cabinet_v4l/` (`research/code/cabinet/v4l_store.py`), never from `cabinet/`:
1. `v4l_store.py setup` — copy of the training store + ONLY the 71 released NEWTEST records (batch 36; 2 had no print);
   in this mode (`DC_CAB_STORE=cabinet_v4l`) the hold-out is `locked_v2.csv` and the tools refuse those 115 signals.
2. Print re-reads of incomplete frame signals (130, note 80) + `v4l_expl_sweep.py` (explained_other entries that
   could hide a performance-measure detector removed).
3. `DC_CAB_STORE=cabinet_v4l cab_final.py` -> `function_labels_v4l_base.parquet` (lists in `cabinet_v4l/lists/`, never
   review/); then `DC_CAB_STORE=cabinet_v4l stack_labels_v3s.py` -> `function_labels_v4l.parquet`,
   `cabinet_v4l/dec_role_changed_v4l.parquet`; then `v4l_rules.py apply` (R1 config Count / YR on a print loop -> print
   reading or excluded; R2 2B091 excluded on the user's word; R3 re-read YR zones identical to a Count zone excluded).
4. `v4l_rules.py diff` -> `function_label_changes_v4l.csv` here: every label that differs from v3s (training label
   or scoring truth), with cause, priority-sorted (ATSPM truth changes first).

## v4l, note 81 (2026-10-03): adopted as THE label table; three more rules and the user's v1 answers
* v4l is the scoring truth and the training labels for all research from now on: `rpath.LABELS_CURRENT`,
  `atspm_score.V3S` and `v3_retrain.LABEL_SETS["v3s"]` point at it (the old table is `LABEL_SETS["v3s_orig"]`, kept
  unchanged for before / after comparisons only).
* `v4l_rules.py apply` adds (all: `exclude_train_score` = True, truth cleared, tag in `rule_v4l`):
  R4 the user's answers on `review/function_label_review_v1.xlsx`, listed in `function_label_corrections_v4l.csv`
  (function rows answered "label wrong" -> out; phase rows answered "label right" -> `phase_user_confirmed` = True, the
  timing phase stays the truth); column `user_review` holds the answer. R5 low-confidence print readings whose channel
  tie or class was decided from hi-res behaviour (68 rows, 24 signals; regex + 7 hand-checked exemptions in the
  script). R6 unused inputs (whole-intersection Other, no config function) with < 15 ONs = noise (10 rows).
* Effect: training rows 11,390 -> 11,322; scoring truth unchanged (R5 rows are print_low, never truth; 2B091 already out).

# Function labels, v4m — label cleansing without detector fault events (note 88, 2026-10-04)

`function_labels_v4m.parquet` (git-ignored) = v4l with `dq_score` / `dq_flags` / `dq_suspect` recomputed by the
fault-free `cabinet/dq_core.py` (detector fault events 83-88 never read: user ban 2026-09-28) on the same cabinet_v4l
print rows (`research/code/final88/dq88.py dq labels`); every other column, including all truth columns, is identical
to v4l (asserted by `final88/oof88.py`). dq_suspect 1,971 -> 1,644 (328 true -> false, all had a fault-event health
reason; 1 false -> true from the coverage change); 229 of the changed rows are training rows. Trainer label set
`v3_retrain.LABEL_SETS["v4m"]`. v3_retrain's not-checkable-high card rule is actuation-only as well (note 88).

# Function labels, v4p — the user's review-v1 answers and the rulings of 2026-10-05 (note 94)

`function_labels_v4p.parquet` (git-ignored) = v4o + (built by `research/code/cabinet/v4p_user_v1.py build`):
* **U** the user's answers on `review/function_label_review_v1.xlsx` rows 1-17, 36, 37, verbatim in
  `function_label_corrections_user_v1.csv`: relabels -> `source` user_ruling (label = truth, validated pass); exclusions ->
  `exclude_train_score` (+ `_train`) True, truth cleared; `user_review_v1`, `signal_note` (01067: unusual but valid).
* **P** the 05999 print (filed on the share under its TSSU id 5CE055), read in full (`%DC_WORK%/lab93/05999_print_reading.json`).
* **N1** print vs hand (user 2026-10-05, train and test): where the print reading and `func7_v2` disagree, keep the hand label
  if the note-77 champion's out-of-fold majority class agrees with it, use the print if it agrees with the print, else
  leave the row out. Columns `n1_flag`, `n1_model_pred / _share / _nwin / _p`, `n1_model_decided` (scored truth chosen by
  model agreement: report headlines with and without these rows).
* **N2** channels 40 / 41 with no timing phase, text or config function: custom dummies, out.
`rule_v4p` names the rule per row; `function_label_changes_v4p.csv` lists every change (old / new training label, truth,
reason). Slot `v3_retrain.LABEL_SETS["v4p"]` (Dec table `%DC_WORK%/lab93/dec_role_changed_v4p.parquet`).

# Function labels, locked v2 — the exam answer key under the training rules (note 94)

`function_labels_locked_v2.parquet` (git-ignored; `research/code/cabinet/locked_v2_labels.py`) = locked v1 restricted to
the 115 locked_v2 signals + R1, stacked-group relabel, YR identical to Count, R3, R5, R6, explained-other sweep, N2 and
the radar-over-loops re-admission. **Labels only**: no model output was read or computed for a locked signal; the
actuation rules read ON intervals only. N1 needs model agreement, so its conflicts are flagged `n1_pending_exam` and left
out of `truth_v2` (old-rule truth in `truth_v2_oldrule`). Use `truth_v2` + `label_use_v2`.

# Function labels, v4q — the user's rulings of 2026-10-05 16:20 (note 95) — CURRENT (rpath.LABELS_CURRENT)

`function_labels_v4q.parquet` (git-ignored) = v4p + (`research/code/cabinet/v4q_rules.py build`; column `rule_v4q`, `q_flag`):
* **Q1** 05999 d23 out (user). **Q2** 2B061: only the phase-4 channels out (asserted).
* **Q3** loops whose hand label (config Function) was "advance presence": hand label = Advance (`func7_v2`; old value in
  `func7_v2_v4p`, flag `advpres_loop`); the 5 user-named channels are user rulings; where a print reading disagrees the N1
  rule decides (model agreement).
* **Q4** channel text (`desc_class`, URL-decoded description; bare "CO" not read as Count) vs config Function, where the
  Function alone decides the label: keep whichever the model agrees with, else out; rows N1 left out are restored when the
  model agrees with the text; rows a print or user decided are untouched.
* **Q5** 04016 d23 / 04040 d28 (Count vs Presence) by model agreement. **Q6** prints filed as 5CE166 (05166), 04CA156 (04156);
  5CE069 (05069) skipped (`%DC_WORK%/lab95/*_print_reading.json`).
Model-decided rows carry `n1_model_decided`. Changes: `function_label_changes_v4q.csv`. Dec table `%DC_WORK%/lab95/dec_role_changed_v4q.parquet`.

# Function labels, locked v3 (note 95)
`function_labels_locked_v3.parquet` = locked v2 + Q3 (loops -> Advance where no print disagrees) + Q4 (text vs Function rows
-> `n1_pending_exam`, reason in `pending_reason`). Labels only. Use `truth_v3` + `label_use_v3`. Changes `function_label_changes_locked_v3.csv`.
