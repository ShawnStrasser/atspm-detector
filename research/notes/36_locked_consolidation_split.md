# 36 — Locked-store consolidation, NEWTEST split, released signals prepared (2026-09-24)
User decisions: (a) the locked prints are read (note 35, 10 batches); (b) keep the 43 TEST signals locked, move a RANDOM
half of NEWTEST into training / dev; (c) cleanse labels with the training rules. Signals that stay locked: labels only —
no model prediction, no metric, no scoring of any kind was computed on them here.

## 1. Locked-store consolidation (labels only)
`cab_final.py --locked-consolidate --locked-labels-only` (new path; refuses without the flag; no DQ, OOF, label_check or
user lists). Rulings: `cabinet_locked/final_rulings.csv`, seeded with the training file's 5 general ('*') rulings.
Fixes on the 10 batches (182 visual_done; 4 signals have no PDF and no v2 row): long presence / video YR / radar over
loops: 0 changes (readers applied them). New `fix_uncoded_zones.py`: **31 uncoded long radar zones from the stop bar ->
Presence** (14015, 14029, 14033, 14036, 14075 d3, 2B012), **2C011 d50/d52 undrawn CO -> Advance low** (+ their A* partners
d49/d51 -> Other/advance_presence low, the positional A*/CO* rule); uncoded stop-bar zones by position: 0 conflicts left.
Records backed up to `cabinet_locked/backup_pre_final/`; changes in `cabinet_locked/sweep_changes.csv`.
`research/labels/function_labels_locked_v1.parquet`: v3 schema where applicable + `set`, `released_from_newtest`,
`func5_config`, `label_use` (= v3 train_use), `validated` = pending. 3,677 rows / 182 signals.
* Tiers (182 print signals): complete_high 34 · complete_mixed 95 · incomplete 53; unusual_layout 15.
* Labelled 3,220 (label_use 2,560; dead 347): print_high 2,035 · print_medium 559 · config 247 · print_low 159 ·
  whole-intersection Other 122 · user_ruling 88 · review_round1 10. Presence 957 · Advance 918 · Other 516 · Count 421 ·
  Mid 151 · YR 134 · Bike 123. High print label differs from v2 on 101 rows.
* By group (labelled / label_use / signals with rows): TEST 983 / 858 / 43 · NEWTEST locked 1,142 / 853 / 70 ·
  NEWTEST released 1,095 / 849 / 69.

## 2. Split (`research/code/cabinet/cab_split_newtest.py`, seed 20260924)
Stratified by DeviceName region prefix (13 regions; odd regions settled by one coin each), names / ids only.
**Released 71** (`official/newtest_released.csv`); **locked_v2 115** = 43 TEST + 72 NEWTEST (`official/locked_v2.csv`,
column `set`). Old split files untouched. AGENTS.md "Locked hold-outs" and `research/labels/README.md` updated;
`cab_common.locked_ids()` stays the 186 (locked-store membership).

## 3. Released signals prepared for training (no training)
* Labels: `cab_final.py` step 9 / `--append-released` appends their locked_v1 rows to v3 (idempotent; also re-run at
  the end of every training-path cab_final run): **1,280 rows / 69 signals, 1,095 labelled, 849 train_use**, sources
  print_high 696 · medium 262 · low 74 · config 31 · user_ruling 23 · WI Other 5 · round-1 4; tiers (signals) complete_high
  12 · mixed 37 · incomplete 20. `validated` = pending, `train_use_validated` = False. v3 = 16,759 rows; the 15,479 old
  rows are identical to the backup (`cabinet/backup_v3_pre_release_20260924.parquet`).
* Folds: `dc_work/folds_v4.csv` = folds_v3 (709, unchanged) + 71 released over folds 1–5 (14/14/15/14/14; fold 0 stays
  the 2025 model's hold-out), greedy on detector count, seed 0.
* Phase inputs (out-of-sample): **final_v1 trees** (`model/weights/phase_lgbm_v4_s0-2` + `decode_lgbm_v4` = shipped
  final_v2's trees), fitted on the 701-signal DEV + NEWTRAIN pool, never on NEWTEST — the same pipeline whose six-fold
  OOF feeds the v5 rows. All detectors with pair features scored (production-like), 22 windows. Pipeline check on the
  released signals (no longer locked): full-window top-1 vs timing .9856 on 1,112 detectors (final_v1's usual level).
* Frame: `research/code/trackA/v6_frame.py` (v5_frame's code path; YR/lag via build_features, expert via a2_features).
  `funcframe_v6` = 456,033 rows / 772 signals: **v5's 432,890 rows byte-identical** + 23,143 new rows (71 signals, split
  `NEWTEST_RELEASED`); expert tables `feat_expert_*_{dec,stg}_v6` (v5 rows identical, +23,143 det / 22,184 cyc / 15,497
  pair). Coverage: 71/71 signals, 1,143 full-window detectors; labelled rows in the frame 960 / 1,095 (132 dead or no
  data, 3 other), train_use 846 / 849.
* `v3_retrain.py`: `--frame v6` added; its locked assertion now reads locked_v2.csv. Run with `--require-validated`
  (otherwise the pending released rows would be trained on). Older scripts (a1/a2/w1/lr/sb, v5_frame) still read the
  two original hold-out files, i.e. they conservatively exclude the released half.

## 4. Review list (low priority, labels only)
`review/locked_print_readings_to_check.xlsx` (`cab_locked_review.py`): **557 detectors / 100 signals** flagged
needs_review 375 · label_disagrees 107 · phase_print_vs_timing 100; stays locked 352 / 65 signals, released 205 / 35.
Columns: status, signal, detector, phase print vs timing, label (confidence), config label, technology, why flagged,
crop (41 have one), two blank columns for the user.

## Open (orchestrator)
* The training store has the same uncoded patterns (dry run, not applied): 32 long radar zones -> Presence (03066,
  12061, 14030, 14032, 14034, 14035, 2C055) and CO 49–52 at 2B132 / 2B382 (3 -> Advance, 3 A* -> Other). Applying them
  = `fix_uncoded_zones.py --batches 1-35 --apply` + a cab_final re-run.
* Behaviour validation of the released rows waits for the user's review of the label checks (note 29).
