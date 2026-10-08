# 130 — Is the model more accurate than the hand labels? Evidence for a public claim (2026-10-07)
Brief: find honest numbers for "it turned out to be more accurate than our hand-labelled training data". Hand labels = the
2025 config export (`labels_dev.parquet` phase, `func7_v2` function in v4q: source config 5,215 / round-1 39). Non-locked OOF
only; locked figures are note 119's exam outputs, nothing new computed on locked signals. Nothing trained. CPU, minutes.
Code `research/code/final130/p130_phase.py`, `p130_func.py`; outputs `%DC_WORK%/s130/p130_{phase,func}.json`.
## 1. Phase vs the controllers' timing (independent truth) — the cleanest comparison
Same detectors, Sept-2026 windows >= 30 min, >= 5 actuations, signal-grouped OOF (v7 phase recipe = note 113 v6_lagt,
3 seeds; reproduces note 113's .9846 E on all 189,509 rows). 4,636 detectors with a hand label / 355 signals / 60,891 rows:
| | hand label | v7 model | model - hand [95 % signal bootstrap] |
|---|---|---|---|
| matches timing, rows | .9741 | **.9872** | +1.31 pt [+0.53, +2.13] |
| full window only | .9743 | .9906 | |
| detectors wrong | 119 (2.6 %) | 39 (model majority) / 1.3 % mean | |
Hand and model disagree on 134 detectors: timing backs the **model 104, the hand label 24**, both 2 (alt phase), neither 4.
Drift is not the cause: hand Dec-2024 vs Feb-2025 agree 99.76 % and timing agrees equally with both (note 10 §2).
A 2025-style model trained ON the hand labels (note 10 variant A, Dec-2024 full window, 4,654 dets / 352 signals) only
ties them: .9789 vs .9768 (+0.2 pt [-0.5, +0.9], n.s.); disagreements 61 model / 51 hand / 11 neither.
So the gain needed the timing labels (note 10: +0.27 pt) plus later model work, not "learning past the noise" alone.
Earlier: note 10 review list (beta_v0, 155 rows, selected): model right 61, hand right 41; whole-signal swaps 16-0.
## 2. Function vs the cabinet print (high confidence), ATSPM level (A / P / C / YR vs non-ATSPM)
Same non-locked detectors, >= 30 min, >= 5 act., not dead; print_high reading AND a hand label: 5,254 dets / 346 signals /
67,976 rows. v7 = note 114 mean3 'nou' stacker OOF, 3-seed mean, argmax BEFORE the lane / stack decode (decode adds ~0,
note 78); champ = note-77 champion decoded (rev81), for reference.
| | hand label | v7 OOF | note-77 champ |
|---|---|---|---|
| matches print, rows | .9026 | **.9371** (+3.45 pt [+1.64, +5.54]) | .9331 (+3.05 [+1.17, +5.17]) |
| detectors wrong | 508 (9.7 %) | 287 majority / 6.3 % mean | 310 / 6.7 % |
| disagreements: model right / hand right / neither | | **398 / 177 / 39** (614) | 393 / 195 / 37 |
Without unusual_layout signals: .9031 vs .9383. Print high+medium (6,264 dets): .8688 vs .9097, disagreements 579 / 280 / 75.
User's own rulings as truth (224 dets / 68 signals; rows were picked from model-label disagreements, so biased): hand
.8002 vs v7 .8743, CI includes 0 [-0.9, +16.6]; disagreements 36 model / 16 hand / 7 neither.
Caveat (important): the model is trained on print-first labels of OTHER signals, so it partly learned the prints'
conventions; prints can be stale (note 27: 1,181 dead print detectors, stale/renumbered sites). This is "agrees with the
print more often than the hand label does", not proof against field truth.
## 3. Other sources, as found
* Note 14 (final_v2, locked-143, 79 function misses, user-reviewed): 37 label wrong, 38 model wrong, 4 '?' -> a TIE on
  the misses; the honest use is "nearly half of the apparent misses were label errors" (5-class .754 -> .868 full window).
* Note 94 (N1, note-77 champion OOF, training signals): 974 print-vs-hand conflicts -> print used 499, hand kept 160, out 315.
* Note 119 (v7 exam, 115 locked signals, never trained on): 149 print-vs-hand conflicts -> model sided with print 112,
  hand 17, out 20. Agreement only: those rows' truth was then chosen by v7, so their .9693 score must not be quoted.
* Note 27 (note-25 b7 model): 594 high print labels overriding the v2 label; model agrees with the print on 311 of 565.
* Note 78 §2: config-only labels err 12.2 % vs 6.3 % on print_high rows; ~1-2 % of today's scoring labels wrong.
## 4. Defensible one-liners (model = final v7 unless said)
1. Phase: against the controllers' own timing, the hand labels were wrong on 2.6 % of detectors, the model (on signals it
   never trained on) on 1.3 %; where they disagreed, the timing backed the model 104 times and the hand label 24 (§1).
2. Function: against the cabinet prints, the hand labels disagreed on 9.7 % of detectors, the model on about 6 %; where
   the model and the hand label disagreed, the print backed the model about 2 to 1 (398 vs 177) (§2).
3. On the 115 held-out exam signals, where print and hand label disagreed, the model sided with the print 112 to 17 (n. 119).
## 5. Avoid
* "More accurate than the training data" without saying WHICH: the v7 model trains on timing / print labels, not on the
  hand labels; a model trained on the hand labels only tied them (§1).
* "47 % of model errors were label errors" for v7: that was final_v2 on locked-143, and it was a 37-38 tie.
* "61-41" for v7: beta_v0, on a selected review list. "Hand labels are 10 % wrong": vs prints, mixed with convention /
  stale prints. Any accuracy on n1 / model-decided rows. The user-ruling comparison (biased selection, CI includes 0).
