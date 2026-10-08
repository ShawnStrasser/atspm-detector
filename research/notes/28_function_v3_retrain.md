# 28 — Function head retrained on the cabinet-print labels (v3) — harness (2026-09-23)
**Status: DONE. 2026-09-23: first.all.wi = function candidate (+4.1 pt). 2026-09-28 label-check v3: +3.9 pt (`function_v3b/`, `v3c/`). 2026-09-29 (note 45): label-check v4 + health v3, HONEST set +2.70 pt, refit `function_v3d/`.**
`research/code/trackA/v3_retrain.py` → `%DC_WORK%/trackA/v3/`: `prep` (feature cache), `fit --variants --seeds --folds`,
`report`, or `all`. Predictions under `run_<sha1 of v3 table>_<pending>[_min5][_clean][_val]/` (never reused across tables).
* Model = note 14 T head (442 features, same params), 7 classes, all windows, both periods, signal-grouped folds, inner fold (k+1)%6 for early stopping; locked absent.
* Variant `label.train.wi`: label first | agree | v2 (func7_v2, same-code control); train high (complete_high) | mixed
  (+complete_mixed) | all (+incomplete, config); wi = whole-intersection Other rows in / out. agree rejected (circular).
* OOF sets: **FIX** = complete_high, high print label (FILTERED: clean-label, optimistic - see the honest set at the end); **V2** = note-14 rows,
  5 classes; **WI** = whole-intersection Other (share non-PM); **UNU** = unusual_layout. Baseline = note-25 b7 OOF, same rows;
  paired signal bootstrap. Also AGR (print = config) / DIS (print ≠ config), in `v3_analyse.py`.
## Coverage extension — funcframe_v5 (2026-09-23), `research/code/trackA/v5_frame.py`
* 316 v3 signals were missing from frame v4 (301 NEWTRAIN Sept-2026 never in the stage-11 config table; 2B073; 14 with no usable data; audit `trackA/v5/missing_audit.csv`). **folds_v3.csv** (709 signals) = folds.csv + folds_v4 + `official/newtrain_folds.csv` (a signal's function fold = the fold its phase was held out in). New rows' phase input = stage-12 ranker-bag + decoder OOF (stronger than v4's Dec-only OOF: a small known mismatch); 1.1 % of new detector-windows lack an OOF prob and are dropped.
* Features by the original code (build_features yrlag, function_v4/v3 rows, a2_features expert); funcframe_v5 = 432,890 rows / 701 signals, v4 rows byte-identical; 5-signal end-to-end recompute identical (≤ 4e-6). `--frame v5` → `trackA/v3/frame_v5/`; in frame by tier: complete_high 124/124, mixed 383/383, incomplete 175/181.
## Result — validated print labels (2026-09-23 night; run `..._min5_clean_val`, frame v5, 6 folds × 3 seeds, 51 min)
Labels = `label_print_first`; >= 5 actuations, no dq_suspect, `validated` == pass (label-check v2); unusual never trained.
FIX 49,548 rows / 122 sig; **FIXb** (rows b7 has) 37,941 / 75; first.all.wi trained on 275,316 rows. Output `frame_v5/ana28_val.txt`.
| FIXb | acc7 | acc5 | core4 | mF1-7 | 30 min | 6 h | full | sd |
|---|---|---|---|---|---|---|---|---|
| baseline (note-25 b7) | .8765 | .8785 | .9031 | .829 | .8809 | .8904 | .8864 | – |
| v2.all.nowi (control) | .8748 | .8761 | .9049 | .823 | .8776 | .8892 | .8882 | .15 |
| first.high.wi / .nowi | .8653 / .8698 | .868 / .872 | .884 / .893 | .80 | .865 / .867 | .880 / .886 | .880 / .884 | .17 / .08 |
| first.mixed.wi / .nowi | .9129 / .9159 | .914 / .917 | .924 / .929 | .884 / .890 | .915 / .917 | .930 / .929 | .928 / .928 | .06 / .02 |
| **first.all.wi** | **.9179** | **.9187** | **.9303** | **.890** | **.9190** | **.9321** | **.9298** | .03 |
* first.all.wi vs b7 (paired signal bootstrap, 95 %): acc7 **+4.14 [+2.62, +5.81]**, 30 min +3.81, 6 h +4.17, full +4.34, core-four
  +2.73 [+1.22, +4.44]; all 6 folds positive; v2 control −0.17 [−0.96, +0.60] ⇒ the gain is the labels. AGR (print = config) +2.49
  [+1.17, +4.09]; DIS +10.6; V2 continuity .8384 → .8474. Recalibration-only stacker on b7 probs: +0.85 ⇒ ~3.3 pt is the retrain.
* WI rows called non-PM: b7 64.7 %, first.all.wi 93.7 %; nowi variants stay ~66 % ⇒ WI rows teach "not a PM detector".
  UNU .8004 (b7 signals .8047 vs .7650). Bike .88/.88, Mid .97/.88 (b7 .81/.85, .99/.74).
* **Verdict:** first.all.wi = new function candidate; mixed.nowi within 0.2 pt but WI 66 % ⇒ rejected. Not shipped; locked untouched.
## Rerun on label-check v3 (2026-09-28; note 29 rebuild, frame v6 / folds_v4, run `…fdc9e921fd_…_min5_clean_valnc`)
Rows = `validated` pass + `not_checkable` with a print_high label, not dead / card-suspect (`v3_retrain --allow-not-checkable-high`,
822 dets, 661 Advance, 16 with a field_issue); fail / unhealthy / no_data / < 5 act / dq out; unusual scored only. All 1,280 released
NEWTEST rows (69 signals) are still `pending` ⇒ excluded (train and score). first.all.wi 275,241 rows / 602 sig; 6 folds × 3 seeds, 35 min.
Analysis `v3_analyse.py --frame v6 --nc-high --extra first.all.wi` → `frame_v6/ana28_lc3.txt`. FIXb = 75 sig / 39,177 rows (baseline b7 rows).
| FIXb acc7 (95 % CI vs b7) | acc7 | acc5 | core4 | mF1-7 | 30 min | 6 h | full | WI non-PM | seed sd |
|---|---|---|---|---|---|---|---|---|---|
| baseline b7 | .8710 | .8729 | .9022 | .823 | .8761 | .8867 | .8798 | 72.6 % | – |
| v2.all.nowi (control) | .8664 (−0.46 [−1.51, +0.46]) | .8682 | .9038 | .812 | .8681 | .8812 | .8794 | 71.9 % | .06 |
| first.mixed.wi | .9072 (+3.62 [+2.21, +5.25]) | .9085 | .9216 | .877 | .9089 | .9234 | .9245 | 93.4 % | .03 |
| **first.all.wi** | **.9102 (+3.92 [+2.51, +5.56])** | .9111 | .9267 | .881 | .9111 | .9268 | .9227 | 92.8 % | .06 |
| first.all.wi pass-only | .9017 (+3.07 [+1.54, +4.77]) | .9033 | .9193 | .872 | .9029 | .9193 | .9169 | 93.4 % | .07 |
* first.all.wi vs b7: 30 min +3.50, 6 h +4.01, full +4.29, acc5 +3.82, core-four +2.45 [+0.89, +4.25]; all 6 folds positive (+2.5 … +5.6). Note 28's +4.14 was on
  other rows (v2 check, frame v5): same size. AGR +2.28 [+1.00, +3.81] (core4 +1.13, n.s.); DIS +10.4. V2 set .8397 vs .8303. UNU .7913 (b7 .7540).
* Per class P/R (FIXb, all windows) b7 → first.all.wi: Adv .90/.91 → .93/.95 · Pres .85/.94 → .91/.94 · Cnt .89/.88 → .92/.91 · YR .97/.65 → .92/.77 ·
  Mid .99/.69 → .99/.83 · Bike .81/.85 → .88/.89 · Other .66/.65 → .76/.78.
* **FYA phases** (FIX 2,820 rows / 32 sig; b7 rows 1,789 / 14): b7 .840 → .864 (+2.5 [−2.4, +9.4], n.s.) vs .912 elsewhere ⇒ ~5 pt harder.
* **not_checkable in vs pass-only**: +0.85 [+0.60, +1.12] on FIXb, +0.27 on pass rows (FIXpb); the gain is on the 7,007 nc rows (.932 vs .890) ⇒ rule kept.
* **Refit** (`v3_final_fit.py --frame v6 --nc-high`) → `%DC_WORK%/final_v3_work/function_v3b/`: 3 seeds × 196 trees (mean fold best iteration),
  275,241 rows / 10,166 dets / 602 signals, numpy parity 0. Candidate package and `model/` untouched; locked_v2 (115) asserted absent.
## Released rows validated + field-issue rule (2026-09-28; run `run_d3f4e7ccbc_…_valnc`, `frame_v6/ana28_lc3c.txt`)
* Released NEWTEST rows now go through the label check (cab_final step 9 before 7): 1,095 labelled, pass 787 · fail 52 ·
  unhealthy 18 · nc 103 · no_data 135 (no card table / DQ for them); old 15,479 rows byte-identical.
* Orchestrator rule: `--allow-not-checkable-high` no longer admits not_checkable rows with a `field_issue` (19 rows: 14 pulse-set Presence, 5 Adv).
* first.all.wi 288,448 rows / 659 sig (6 folds × 3 seeds, 17 min). FIXb (75 sig / 39,137 rows): **.9099 vs b7 .8714, +3.85 [+2.41, +5.49]**;
  30 min +3.24, 6 h +4.04, full +4.29, core4 +2.44 [+0.87, +4.24]; AGR +2.13 [+0.80, +3.64]; DIS +10.7; FIX (133 sig) .9081; WI 92.6 %; seed sd .06 ⇒ holds.
* Refit (`v3_final_fit.py --frame v6 --nc-high --ana ana28_lc3c`) → `final_v3_work/function_v3c/`: 3 × 209 trees, 10,904 dets, parity 0.
## Honest evaluation (2026-09-29, note 45; label-check v4 + health-v3 training cleansing; filters are TRAINING-only)
first.all.wi vs b7 on b7's rows, 95 % CI: **A all labelled (print high > config, >= 5 actuations only, no other filter;
268,861 rows / 652 sig): .8693 vs .8423, +2.70 [+2.10, +3.33]**, core-four +1.34; **B = A minus health-v3 `bad` (98.7 %
of A): +2.72 [+2.11, +3.34]**; C = the FIXb rows above, "clean-label subset (optimistic)": .9100 vs .8710, +3.90.
