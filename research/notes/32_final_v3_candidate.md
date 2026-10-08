# 32 — final_v3 CANDIDATE package (2026-09-23) — assembled, not shipped, locked signals untouched

`%DC_WORK%/final_v3_candidate/` = a copy of `model/` (final_v2, untouched) with two changes; phase trees, decoder,
GRU weights and blend weight are byte-identical. Code: `research/code/trackA/v3_final_fit.py` (fit),
`v3_candidate_verify.py` (feature parity); helper scripts in `%DC_WORK%/final_v3_work/` (bench, smoke, card, refs).

## What changed vs final_v2, and the out-of-fold evidence for each
1. **Function head = note 28's `first.all.wi`**, fitted once on all non-locked training rows: 7 classes (Advance,
   Presence, Count, Yellow_Red, Mid, Bike, Other), `label_print_first`, validated, >= 5 actuations, no dq_suspect,
   whole-intersection Other in, unusual_layout out. 275,316 detector-window rows / 10,083 detectors / 609 signals
   (detectors: Presence 2,848, Advance 2,734, Other 1,900, Count 1,485, Yellow_Red 423, Bike 417, Mid 276). 3 seeds ×
   189 trees (= mean best iteration of note 28's 18 fold fits; stage-11 rule), averaged in probability space;
   numpy evaluator = lightgbm to 0.0. Evidence (note 28, six folds × 3 seeds): FIXb acc7 .8765 → .9179, **+4.14 pt
   [+2.62, +5.81]**, core-four +2.73, label-neutral AGR +2.49, all folds positive, seed sd .03 pt; WI rows called
   non-PM 65 % → 94 %. Output: `function_pred` may be Mid/Bike; `p_mid`, `p_bike` added after `p_other`.
   54 expert features computed in the package: `features_expert.py` (the `a2_features.py` SQL verbatim).
2. **Network at any length (no cut-off; final_v2 stopped at 120 min), GRU kept; above 120 min it reads `max_chunks=4`
   evenly spaced 30-min pieces** (orchestrator decision after note 33). `blend.json`: `cutoff_minutes null`,
   `long_minutes 120`, `long_max_chunks 4`, `long_piece_grid 32` (`gru_blend.runs_on/piece_plan`, `split_range(grid=)`).
   Placement = note 33's `pieces_infer.py` exactly (32-piece linspace grid, then `linspace(0, n-1, 4).round()`):
   0 mismatches over every span 121 min–100 h (`final_v3_work/placement_check.py`); check.py asserts 24 h → pieces
   0/15/32/47. ≤ 120 min path unchanged. OOF (note 33): K=4 within 0.06 pt of all pieces; blend − trees +0.35 / +0.32 /
   +0.14 / +0.18 pt at 3 h / 6 h / 24 h / full. GRU leads the TCN by 0.13 pt at 30 min (notes 13/20/21).

## Feature parity (5 training signals: 3 Sept-2026, 2 Dec-2024 × m30 / h6 / full = 15 signal-windows)
* **A — definitions**: package `features_expert` on package tables built from each signal's whole cached log, research
  window → max relative difference to the training frame **3.3e-16** over all 54 columns, 0 NaN mismatches.
* **B — real inference path** (window-only events, all 442 features): 0 missing. Share within 1e-3, production (388)
  vs expert (54): m30 .70–.80 vs .68–.83; h6 .83–.88 vs .82–.95; full .95–1.00 vs .97–1.00 — the window-edge effect
  every production feature already has (training frame cut from a multi-day cache).
* **Bin anchoring fix:** expert ~15-min bins anchored at the first event instead of the window's clock start moved
  binned features by up to 30 %; the package lays them on `start`/`end` when given, else the event span.

## check.py (re-frozen on the bundled 30-min sample, a non-locked training signal; re-run after the K=4 change)
All three PASS: reproduces references (max drift 8.9e-07; phase unchanged vs final_v2 and vs the pre-K=4 candidate,
function checked on probability too, < 1e-6); phase-number invariance (17/17, phase **and** function);
torch/lightgbm/scipy/sklearn blocked. No absolute paths, no research imports. Model card: counts only, blend section
carries the long-sample rule + note 33 evidence. Weights 32 MB (was 21 MB).

## Runtime / memory, 20 non-locked training signals, one predict() call, predict defaults (4 threads)
| s / signal (network part) · peak RAM | 30 min | 3 h | 6 h | 24 h |
|---|---|---|---|---|
| final_v2 | 0.55 (0.49) · 1.79 GB | 0.11 (off) · 0.68 GB | 0.13 (off) · 0.76 GB | 0.36 (off) · 1.57 GB |
| **candidate (K=4, session freed)** | 0.57 (0.50) · 1.63 GB | **2.12 (2.01) · 1.68 GB** | **2.13 (1.99) · 1.70 GB** | **2.45 (2.06) · 1.92 GB** |
Function +0.01–0.03 s/signal; network flat ~2 s/signal > 2 h (pre-K=4: 3.05 / 6.04 at 3 h / 6 h). 24 h was 2.82 GB:
onnxruntime's CPU arena (~1.2 GB, kept for the process's life) under the tree features, not the interval build (0.1 GB).
Now the session is dropped after the network (re-made per call, 0.02 s), intervals fetched only near the pieces; p_gru +
predict() bit-identical 24 h/6 h/3 h/30 min (a 30-min tree prob flips 1.8e-5 run-to-run, old too). No arena: 1.68 GB, +0.6 s.

## Smoke test — 30 fold-0 Sept-2026 training signals, IN-SAMPLE (not an accuracy claim), re-run with K=4
Runs at 30 min / 3 h / 6 h / 66 h; 7-class probabilities sum to 1 (≤ 4e-16); answered counts identical to final_v2;
30 min identical to final_v2. Function acc7 in-sample .89–.94 (369–393 labelled), unchanged. Phase (in-sample trees,
GRU also fitted on these): 3 h .9502 vs v2 .9617 (−6 det. of ~522), 6 h .9713 vs .9694 (+1), 66 h .9718 vs .9774
(−3, the network now runs there). Blending in-sample trees with a network dilutes their memorised answers, so this is
not evidence against note 33's six-fold gains; worth one line in the locked confirmation run.

## Open issues for the orchestrator
* Locked confirmation NOT run; `model/` untouched. Backup of the pre-K=4 package: `final_v3_work/candidate_backup_pre34/`.
* Accepted edge effects: without `start`/`end` the expert bins sit on the event span; a window with no 131 event
  leaves `px_disp_coord/free` NaN (training knew the pre-window state).
