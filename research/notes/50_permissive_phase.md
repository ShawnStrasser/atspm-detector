# 50 — Permissive-phase recognition from the log alone, and does it help the function head? (2026-09-29)
Continues notes 44 / 45. Can the model recognise a permissive phase from ALLOWED codes only, and does it help
function (FYA / right-turn rows ~4 pt harder)? Locked_v2 asserted absent; nothing shipped; `model/` untouched.
## Code (research/code/trackA)
* `pm_features.py` -> `%DC_WORK%/trackA/pm/pm_{det,ph}_{stg,dec}.parquet` (build 16 + 7 min, 6 DuckDB threads). Per
  window and candidate phase, codes 81/82, 1/8/10, 43/44 only; a phase's detectors are those PREDICTED on it (frame
  `pred_phase`), no phase number, print, timing or technology. Phase: red-wait leave share (note 44) on the busiest-
  wait detector / on the behaviour stop-bar pick (max n_wait x share of greens starting with the zone ON) / wait-
  weighted mean / max; red waits per cycle; share of phase calls (43) placed in red that are withdrawn (44) >= 1 s
  before green; green share; cycle ratio. Detector: own leave, own - stop-bar leave, is-the-pick, share of waits.
* `pm_score.py` -> `pm_score.parquet`: small LightGBM per (signal, period, window, phase), target = >= 5 FYA
  events 32 on the phase in that period OR timing additional-call to its through (label_check_pplt by_timing).
  Events are the TARGET only, never an input. Trained only at signals that log 32 (179 signals, 26,330 phase rows,
  6,160 positive); folds_v4, 3 seeds, every row out-of-fold. `pm_labels.parquet` = event-32 counts (eval only).
* `v3_retrain.py --pm-feats [--pm-shuffle]` (run suffix _pm / _pmshuf): + 9 columns (score of the detector's
  predicted phase, largest score of the other phases, 3 phase leave / drop features, 4 detector relations).
  `pm_eval.py`: note-45 sets A / C by LABELLED phase: FYA (pplt), FYAet (events / timing only), RT (lane R),
  PRM = FYA | RT, rest; paired signal bootstrap, 3-seed mean OOF.
## (1)+(2) Permissive-phase score: yes, the log shows it (OOF AUC)
| phase set | all windows | m5 | m30 | h6 | full |
|---|---|---|---|---|---|
| logging signals, left phases (1/3/5/7; 79 % positive) | .819 | .663 | .806 | .876 | .881 |
| logging signals, all phases | .958 | .908 | .958 | .973 | .980 |
| non-logging signals, left: timing-positive vs rest | .844 | .649 | .817 | .921 | .936 |
Single features (left, logging, full): stop-bar-pick leave .873 (coverage 96 %), call drop 43/44 .720 (note 44
.66-.75; its label-picked stop bar .95 on a narrower negative set). Behaviour-flagged left phases at non-logging signals (note 44 rule): median
score .87 vs .19 for the rest. Short samples are weak: m5 has almost no red waits (leave coverage 1 %).
## (3) Function head, first.all.wi, frame v6, (a+h3) recipe, 6 folds x 3 seeds (~60-110 s / fit at 6 threads)
| accuracy (rows / signals) | h3 ref | pm | pmshuf | pm - h3, 95 % CI | pm - pmshuf, 95 % CI |
|---|---|---|---|---|---|
| A honest, all (268,861 / 652) | .8714 | .8714 | .8713 | +0.00 [-0.06, +0.07] | +0.01 [-0.04, +0.07] |
| A FYA (22,510 / 256) | .8659 | .8677 | .8663 | +0.18 [-0.04, +0.40] | +0.14 [-0.06, +0.36] |
| A FYAet (11,225 / 150) | .8450 | .8479 | .8459 | +0.29 [-0.06, +0.67] | +0.20 [-0.13, +0.58] |
| A RT (14,310 / 247) | .8088 | .8098 | .8092 | +0.10 [-0.18, +0.36] | +0.06 [-0.26, +0.39] |
| A PRM (36,573 / 405) | .8445 | .8461 | .8449 | +0.16 [-0.01, +0.32] | +0.11 [-0.06, +0.30] |
| C clean-label, all (54,386 / 133) | .9079 | .9083 | .9079 | +0.04 [-0.09, +0.17] | +0.04 [-0.08, +0.16] |
| C FYA (5,523 / 57) | .8943 | .8991 | .8973 | +0.49 [+0.11, +0.89] | +0.18 [-0.17, +0.59] |
| C FYAet (3,046 / 35) | .8500 | .8575 | .8549 | +0.76 [+0.22, +1.30] | +0.26 [-0.31, +0.86] |
| C RT (2,754 / 46) | .8083 | .8065 | .8101 | -0.18 [-0.61, +0.22] | -0.36 [-0.85, +0.11] |
A = realistic / everything scored, C = clean subset: both flat. `frame_v6/pm_eval.txt`, `pm_eval_vs_shuf.txt`.
## Verdict: NOT kept
* Overall: nothing (bar 1 pt). Permissive subsets: pm beats h3 on C FYA (+0.49, CI clear of 0), but the SHUFFLED
  control beats h3 there too (+0.31 [+0.09, +0.56]; FYAet +0.49): on 35-57 signals a re-fit alone moves the subgroup
  ~0.3-0.5 pt. Against its own control the gain is +0.1-0.3 pt, no CI clear of 0 except A FYAet at m30 (+0.51
  [+0.05, +1.01], one cell of many). Right-turn lanes: no gain anywhere.
* Caution for note 45: its lc +0.31 on C FYA equals this refit noise; compare subgroups to their own control.
* Why: the 442 features already carry per-detector red-time behaviour (px_occ_red_*, px_span_to_green, release);
  knowing the PHASE is permissive changes few argmaxes. Hard FYA rows look more like label / layout issues.
* Worth keeping as a by-product: the permissive score itself (OOF AUC .88 on left turns, .98 among all phases with
  >= 6 h) could be a reported per-phase output like n_lanes, or replace the label-picked stop bar in the label check.
