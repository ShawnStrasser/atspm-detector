# 22_phase_residual_errors — what the phase model still gets wrong (training/dev, six folds, out-of-fold)

Stage-13 shipped arrangement (0.5 × LightGBM ranker + 0.5 × GRU, then the six-fold joint decoder), rebuilt per row by
`research/code/trackB/r1_residual_preds.py` — reproduces `blend_before.json` (w = 0.5) to five decimals on all eight lengths.
Stage-13 rows: 701 signals, labelled phase green in the window, >= 1 actuation. Classifier `r2_residual_classify.py`;
artefacts `dc_work/trackB/resid/`. Locked signals never read or scored (ID lists only assert none is in the rows).

## Bins (assigned in this order; one count per detector-window)
* **a0** the model's phase is the timing's `switch_phase` / an `additional_call_phases` entry (A1 scorer accepts; stage 13 did not).
* **a** likely label error: the detector's full-window answer is the same phase at p >= .80, top-1 in >= 75 % of its windows,
  AND outside support — >= 2 detectors of the signal make the same label→model move, or the hand config disagrees with the
  timing; if the pair is green together >= 90 % of the full window, only hand = model counts. **Strong** subset: hand = model,
  or >= 3 detectors, or a swap (the signal also has model→label errors).
* **b** concurrent: label and model phase green together >= 90 % of their joint green time (Jaccard, `green_state`) in THAT window.
* **c** low data: < 10 actuations in the window. **d** the rest.

| bin | 30 min errors (4 anchors, 47,399 det-windows) | share | no model right¹ | full-window errors (12,712 det.) | share | no model right¹ |
|---|---|---|---|---|---|---|
| a0 switch / additional | 36 | 3.0 % | 13 | 6 | 2.6 % | 1 |
| **a label** (strong) | **224** (151) | **18.7 %** | 200 | **59** (40) | **26.0 %** | 56 |
| b concurrent | 164 | 13.7 % | 81 | 34 | 15.0 % | 15 |
| c low data | 350 | 29.2 % | 230 | 8 | 3.5 % | 4 |
| d model | 426 | 35.5 % | 274 | 120 | 52.9 % | 68 |
| **total** | **1,200** (acc .9747) | | 798 | **227** (.9821 pooled / .9828 per window) | | 144 |

¹ none of decoded trees / net alone / ranker has the labelled phase. Fold 0 alone: 165 errors at 30 min (note 19: 170 with TCN seed 0).
Sensitivity: b at J >= .80/.90/.95 = 202/164/144 (30 min), 35/34/30 (full); c at < 5/10/20 act. = 256/350/432 and 5/8/16;
16 % of 30-min detector-windows have < 10 actuations and score .946 (vs .980 for the rest).

## Implied ceiling (pooled; a0 and a counted correct once the label is fixed)
| | now | + a0 | + a0 + strong a | **+ a0 + all a** | … and b removed as unresolvable |
|---|---|---|---|---|---|
| 30 min | .9747 | .9754 | .9786 | **.9802 (+0.55 pt)** | .9836 |
| full | .9821 | .9826 | .9858 | **.9873 (+0.52 pt)** | .9899 |

## What each bin looks like
* **a** — 59 full-window cases on 27 signals, mostly signal-wide (one signal: 4→2 and 8→6 on 3 detectors each; two signals
  with 1↔8 swaps). The hand config labels 113 of the 227 full-window errors and agrees with the timing on only 63 % of them
  (97.7 % on all 6,550 hand-labelled scored detectors: 16× enrichment); it backs the model on 31. 225 of the 1,200 30-min
  errors sit on these 59 detectors; 89 % of 30-min a-errors have no model right (the models agree with each other against the
  label). **Caution:** where the channel description names a phase it sides with the *label* 4×, the model 1× (full window);
  stage 10's review found the model right ~3 : 2. Category a is a suspect list, not proven errors.
* **b** — mostly 2↔6 and 4↔8 (median J .93 / 1.0); NEMA pairs 2↔5, 1↔6 are *not* (median J ≈ .02). **c** — 30-min only.
* **d, 30 min** (median 54 actuations): 118 confident + consistent but without outside evidence (median p .97 — likely more
  unconfirmed label errors), 122 partial-concurrent NEMA pairs with J < .9 (2↔5, 1↔6), 43 ring neighbours (1↔2, never green
  together), 143 other, mostly cross-street (2→8, 4→6, 1→8; median p .61). Trees alone right on 73 of 426, net alone on 64.
* **d, full** (median 312 actuations, median p .74): 46 confident-unconfirmed, 41 other, 18 partial-concurrent, 15 ring
  neighbours; top moves 2→1 (16), 2→5 (12), 6→1 (11) — a through detector read as a left-turn phase. Hand config present on
  61 of the 120 and backs the model on only 6, so d is mostly real model error.
## Verdict
Not at the ceiling on the scorer's terms. A fifth (30 min) to a quarter (full) of the residual looks like label error, worth
~+0.5 pt if confirmed — more than any Track B idea delivered. Truly unresolvable concurrency is only 14–15 % of errors. At
30 min the largest bin is thin data (29 %), which no architecture fixes. Genuine model error is ~0.9 pt at both lengths, and
its recurring shape is **through ↔ left turn** (2→1, 6→1, 2→5), not concurrent pairs.

## Deliverables, caveats
`review/phase_label_suspects_round1.xlsx` (git-ignored): all **59** full-window a-detectors (< ~60 cap), strong first then by
confidence (40 strong / 19 moderate; 48 Dec-2024, 11 Sept-2026; 27 signals), with timing/hand phase, top-2 + probs, actuations,
reasons, `on_earlier_list` (48 were on the unanswered `official_vs_model_disagreements.csv`), empty `user_verdict`; a0 omitted.
Thresholds (.80 / 75 % / J .90 / 10 act.) set once, not tuned; "2 detectors" can be the decoder coupling siblings (hence "strong").
