# 45 — Function head on label-check v4 + health v3, honest evaluation, label-check features (2026-09-29)
Continues note 28 (full). Label table = note 44 (sha1 `ad6ea6959f`). Frame v6, folds_v4, 6 folds x 3 seeds, first.all.wi,
`v3_retrain.py all --min-on 5 --clean --require-validated --allow-not-checkable-high` (run script
`%DC_WORK%/trackA/v3/frame_v6/run_lc4.sh`); analysis `v3_analyse.py ... --out ana28_lc4` (-> `ana28_lc4.txt`) and
`v3_eval_honest.py` (-> `frame_v6/honest_lc4.json`). Locked_v2 asserted absent throughout; nothing shipped.
## Training cleansing (training only, never scoring)
* (a) label-check v4 rows: 288,099 training rows / 659 signals (v3c 288,448).
* (a+h3) orchestrator rule: health v3 (note 43, `health_v3.parquet`, per detector and period): `bad` detector-periods
  (7,811 frame rows) and `suspect` windows overlapping a listed bad period (2,116) out of training (`--health3`,
  run suffix `_h3`): 283,894 rows. Scoring rows unchanged.
## Evaluation sets (orchestrator / user 2026-09-29: cleansing filters must not quietly drop hard rows)
* **A (honest)**: every labelled detector, truth = user ruling > high-confidence print > config label where there is
  no print reading; only the production rule >= 5 actuations; no validation / health / field-issue / dq / unusual
  filter. 268,861 rows / 652 signals (print_high 240,983, config 16,967, user 10,911; unusual 21,500).
* **B**: A minus detector-periods health v3 calls `bad` (production can compute it): 265,404 rows = 98.7 % of A.
* **C**: note 28's filtered FIX = "clean-label subset (optimistic)": 54,386 rows / 133 signals.
Baseline b7 (note 25) is compared on the rows it has (A: 204,889 / 383 signals; C: FIXb 39,029 / 75).
| set (b7 rows) | b7 | (a) | (a+h3) | (a+h3) - b7, 95 % CI | core-four b7 -> a+h3 | 30 min / 6 h / full, a+h3 - b7 |
|---|---|---|---|---|---|---|
| **A honest** | .8423 | .8685 | .8693 | **+2.70 [+2.10, +3.33]** | .8846 -> .8980 (+1.34) | +2.57 / +2.67 / +2.78 |
| **B health-ok** | .8438 | .8700 | .8709 | **+2.72 [+2.11, +3.34]** | .8857 -> .8993 (+1.36) | +2.59 / +2.66 / +2.79 |
| C clean-label (optimistic) | .8710 | .9096 | .9100 | +3.90 [+2.46, +5.54] | .9024 -> .9272 (+2.49) | +3.25 / +4.06 / +4.49 |
All rows of A (not only b7's): (a) .8708, (a+h3) .8714; B .8725 / .8731; C .9078 / .9079.
* The filtered set overstates the gain by ~1.2 pt (C +3.9 vs A +2.7) and the level by ~4 pt: C is the easy rows.
* Health v3 cleansing: (a+h3) - (a) on A +0.05 [-0.00, +0.11], core-four +0.11 [+0.06, +0.17]; on C +0.01 (n.s.) -> kept
  (costs nothing, small core-four gain on the honest set).
* C details (ana28_lc4): AGR +2.19 [+0.92, +3.69]; WI non-PM 92.3 % (b7 72.6 %); FYA phases (events / timing /
  behaviour; b7 rows 26 sig) +2.30 [-0.48, +5.54] vs b7; permissive incl. right-turn lanes (PRM, 81 sig) .866 vs .908
  on all of C: still ~4 pt harder. Seed sd .04-.07 pt. v3c (note 28, same recipe, v3 labels) was .9099 / +3.85 on C.
* **Refit** (a+h3): `v3_final_fit.py --frame v6 --nc-high --health3 --ana ana28_lc4` -> `final_v3_work/function_v3d/`,
  3 seeds x 207 trees (mean fold best iteration), 283,894 rows / 659 signals. `model/` untouched.
## (b) Label-check statistics as model features - the user's question "do the expert checks feed the model?"
They did not. `trackA/lc_features.py`: 12 per-window features from the log only, grouped by the PREDICTED phase
(never print / timing / lane / technology), NaN when the window lacks the condition: red-start ONs held to green,
ON at green start, arrivals on red vs chance, red waits that leave before green (own and phase max: the permissive
clue), occupancy and count relative to same-phase siblings, share of siblings out-occupied with fewer counts (and the
reverse), lead out / lead in (1.5-9 s minus chance), order position. `%DC_WORK%/trackA/lc/feat_lc_{stg,dec}.parquet`
(456,033 rows = the frame; build 53 + 16 min). Run on top of (a+h3), `--lc-feats` (_lc) and `--lc-shuffle` (control).
| lc minus (a+h3), 95 % CI | acc7 | core-four | 30 min | 6 h | FYA phases | permissive (PRM) |
|---|---|---|---|---|---|---|
| A honest | +0.04 [-0.02, +0.11] | -0.00 | -0.01 | +0.10 | | |
| C clean-label | +0.07 [-0.05, +0.21] | +0.02 | +0.17 | -0.03 | **+0.31 [+0.07, +0.59]** | **+0.24 [+0.03, +0.47]** |
| shuffled control (C) | -0.01 [-0.10, +0.10] | -0.02 | -0.06 | -0.12 | +0.05 [-0.23, +0.36] | +0.12 [-0.11, +0.39] |
* **Verdict: not kept.** Far below the 1-pt function bar on every set; nothing on the honest set. The one signal is
  on permissive phases (+0.3 pt, CI clear of 0, shuffled control +0.05 / +0.12): real but small - a hint that a
  dedicated permissive feature could help there, not worth 12 columns now. Code and tables kept (`--lc-feats`).
  Why so little: the 442 features already carry most of it (px_span_to_green, px_occ_red_*, px_short_frac, the pair
  correlation block); the checks separate labels because they were calibrated on the labels, the trees get there too.
