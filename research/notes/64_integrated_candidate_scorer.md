# 64 — One integrated OOF scorer for the current candidate + builder for the final error review (2026-10-01)
Code `research/code/evaluation/cand64.py` (scorer), `review64.py` (review builder); work `%DC_WORK%/cand64/` (phase_oof,
phase_rows, function_rows = per-row OOF incl. side outputs; headline.json; review_items, review_dryrun.json, review_preview/).
One command: `python cand64.py all` (phase cached after the first run; function + score redo in ~3 min, picking up every
`tcn53/fpreds/fj[_sN]_f{k}.parquet` present, seeds averaged). Six folds folds_v4; locked_v2 asserted absent; CPU, 6 threads;
nothing shipped; review/ untouched.
## Candidate as scored
* Phase: note-57 trees (261 feat, 3-seed bag) 0.5 / 0.5 with the GRU p3 OOF (all lengths, K = 4; note 37) BEFORE the joint
  decoder, decoder re-fitted OOF on folds_v4. GRU covers 97 % of pair rows (rest trees alone). Check: re-decoding the trees
  alone reproduces note 57 exactly (30 min .9696 E, mean of windows); candidate 30 min .9798 = note 48's candidate.
* Function: 229-feature arm 3 seeds (frame v6e phase input) -> D-lane stack-pick decode (stack_loser nonatspm_ap) -> cy twin
  decode thr .4 on 5/10 min; no routed short head (as briefed). All rows .8928 E / .9057 R = note 62's twin-only row exactly.
  Optional fj blend (tree weight 0.6, before the decode) where fj exists: fold 0 only today.
* Side outputs attached per row as their notes left them: lanes D (88 % of Sept >= 30-min rows get a lane), setback sb7 P50
  (42 % of predicted Advance rows: sb7 OOF covers m30 / h6 / h24 / full66 only), health v5 (Sept: ok 92.4 %, suspect 5.7,
  bad 1.6), night speed sp1 (19 % of predicted Advance at full66). They read the 229-arm function, not the fj blend.
* Function's phase input (v6e) = candidate phase on 99.76 % of matched det-windows (99.84 % >= 30 min) -> not rebuilt.
## Headline (rows pooled; 95 % CI = signal bootstrap; E = everything, R = realistic)
| | >= 30 min | 5 min | 10 min |
|---|---|---|---|
| phase E (173,878 / 32,213 / 34,294 det-windows, 757 signals) | **.9818 [.9787,.9849]** | .9646 [.9591,.9695] | .9726 [.9680,.9766] |
| phase R | **.9842 [.9813,.9870]** | .9680 [.9632,.9724] | .9761 [.9720,.9797] |
| phase trees alone E / R | .9755 / .9781 | .9440 / .9474 | .9571 / .9607 |
| GRU blend vs trees alone, E (pt) | +0.63 [+0.47,+0.80] | +2.06 [+1.66,+2.46] | +1.56 [+1.25,+1.87] |
| function ATSPM E (187,980 / 35,917 / 37,631 rows, 652 signals) | **.9009 [.8894,.9114]** | .8662 [.8552,.8772] | .8776 [.8653,.8887] |
| function ATSPM R | **.9137 [.9030,.9236]** | .8792 [.8687,.8899] | .8908 [.8794,.9014] |
Phase by window (mean of windows, E): m30 .9798, h1 .9810, h3 .9827, h6 .9832, h24 .9829, full .9831. Function by window
(E): m30 .8907, h1 .8908, h3 .9062, h6 .9080, h24 .9122, full .9135. Function errors >= 30 min (pt of rows, E):
A->nonA 4.00, A->wrongA 3.47, nonA->A 2.38, stacked extra lane 0.08.
## fj blend on the folds where it exists (fold 0, 91 signals; with vs without, same rows)
| | >= 30 min | 5 min | 10 min | all |
|---|---|---|---|---|
| E without -> with | .8922 -> .9011, **+0.89 [+0.18,+1.71]** | .8498 -> .8625, +1.27 [+0.50,+2.07] | .8586 -> .8698, +1.12 [+0.36,+1.88] | +0.97 [+0.35,+1.69] |
| R without -> with | .8987 -> .9062, +0.76 [+0.15,+1.49] | +1.22 [+0.48,+1.98] | +1.07 [+0.35,+1.86] | +0.86 [+0.33,+1.54] |
Matches note 63. Caveat: w 0.6 was chosen inside fold 0 (nested there), so fold 0 is not an independent check of the
weight; folds 1-5 are. They add themselves on the next `all`.
## Error-review builder (`review64.py`; dry run only, nothing in review/)
Scope: Sept log, >= 30-min samples. Item = one detector, listed when wrong after exclusions in >= half of its samples;
phase errors take precedence over the same detector's function errors. Excluded (det-windows): phase - additional / switch
call phase 91, label check fail / misconfigured 77, print phase differs (known mislabel) 40, phase of a called overlap 0 left
(the 14 wrong windows on overlap-calling detectors were all already explained); function - label check fail / misconfigured
2,003, YR <-> Count twin swaps 837 (open USER_INPUT question: not asked twice), known field issue 251, phase wrong 149.
Kept wrong det-windows: phase 864 of 1,072, function 9,344 of 12,584.
**Dry run: 651 rows on 250 signals** (49 phase, 238 A->nonA, 216 nonA->A, 141 A->wrongA, 7 stacked extra lane; radar 293,
loop 190, video 133); by the model's confidence >= .5 / .6 / .7 / .8 / .9: 508 / 420 / 329 / 234 / 134 rows (118 signals at
>= .8). Top types: advance called other 84, other called presence 82, other called advance 81, presence called other 80.
Sheet: one question at the top, 10 narrow columns (#, signal, det, label, model says, how sure, why listed, chart, print,
answer L / M / ?); signals ordered by their most confident item. Chart per item from saved data (`review_data/`): phase item
= share of ONs starting in each phase's green (label / model highlighted); function item = 15-min counts vs the labelled
phase's detectors; both + 10 busy minutes of ONs vs green and an ON-length histogram. Preview of 4 checked
(`review_preview/`). Signal 03104 alone has 11 radar detectors labelled Other that the model calls ATSPM (likely one label
question, not 11).
## For the orchestrator
* 651 rows is too many for the user (memory: few rows). Suggest `--min-conf 0.8` (234 rows / 118 signals) or 0.9 (134 / 68),
  or a per-signal cap; the fj arm (`--arm fj`) only once all six folds exist.
* Scorer reproduces notes 57 / 48 / 62 / 63 exactly; use `headline.json` as the step-7 reference.
## Addendum: final_v2 reference arm (published beta; saved OOF, nothing re-trained; `cand64.py v2ref`; E pt [CI], cand - v2)
Phase = note 48's final_v2 OOF on cand64 rows (99.96 %, 756 sig.): >= 30 min .9802 vs .9818, +0.16 [+0.07,+0.26] (R +0.14 [+0.05,+0.24]); 5 min +0.05 [-0.09,+0.18]; 10 min -0.01 [-0.14,+0.11]. Function = b7 -> 5 classes, argmax, no decode (note 48's head; 75.5 % of rows, 383 sig.): >= 30 min .8739 vs trees .9011, +2.72 [+1.93,+3.51] (R +2.91 [+2.13,+3.70]); 5 min +3.64 [+2.96,+4.34]; 10 min +3.25 [+2.58,+3.91].
fj folds 0,1,3,4,5 (316 sig.): cand+fj vs v2 >= 30 min .9056 vs .8760, +2.96 [+2.10,+3.82] (R +3.13); 5 min +4.54 [+3.77,+5.30]; 10 min +4.12 [+3.40,+4.86].
Shipped stage-11 booster OOF instead of b7 (69 % of rows, labelled channels only): >= 30 min +2.52 [+1.76,+3.23], 5 min +3.84, 10 min +3.29. Caveat: v2 heads read older tree-only OOF phase.
