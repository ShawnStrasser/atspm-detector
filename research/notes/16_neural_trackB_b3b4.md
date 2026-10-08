# 16_neural_trackB_b3b4 — dropout + augmentation (B3), mixed sample lengths (B4)

Track B of `AGENTS.md`, TCN backbone (`--arch tcn`, note 15), screened on **fold 0** on exactly the stage-13 rows
(22 out-of-fold windows, 210,038 rows, 34,552 detector-windows), same harness as note 15 (`trackb_eval.py`, folds
1–5 keep the stage-13 GRU mixture). Metric: **blend before the joint decoder** (weight 0.5). The window set has no
15-min or 2-h family, so B4 is read at 1 h and 3 h. Driver `research/code/neural/trackb_b3b4.py` (one resumable
Python chain, native paths, no Git Bash), artefacts `dc_work/trackB/{runs,preds,eval}/`, trail in `state/trail.json`.
**Gate, fixed before launch:** screen = candidate seed 0 vs TCN seed 0, needs ≥ +0.3 pt at 30 min (B4 may pass on
1 h instead if 30 min drops ≤ 0.2 pt); only a pass earns a seed-1 fit, and *kept* = two-seed mean ≥ three-seed TCN
mean + 0.3 pt. Baseline seeds: `tb_tcn_f0_oof`, `tb_b2s1_f0_oof`, `tb_b2s2_f0_oof` (read from `eval/b2.json`).

## B3 — dropout + augmentation: **dropped**
Two settings fixed in advance (`trackb_train.py` knobs): **b3a** dropout 0.10 (inside every TCN block and on the head),
context-channel dropout 0.15 (a training sample has its three signal-wide channels — phases green, phases called,
coordinated — zeroed), start jitter ±300 s; **b3b** the same at 0.20 / 0.30 / ±300 s.

| fold 0, blend before the decoder | 5 min | 10 min | 30 min | 1 h | 3 h |
|---|---|---|---|---|---|
| TCN seed 0 / three-seed mean | .9492 / .9506 | .9622 / .9639 | .9737 / .9750 | .9710 / .9734 | .9808 / .9814 |
| b3a (0.10 / 0.15) | .9522 | .9660 | .9750 | .9722 | .9809 |
| **b3b (0.20 / 0.30)** | **.9536** | .9645 | **.9759** | .9716 | .9806 |
| b3b − TCN seed 0 | +0.44 pt | +0.23 pt | **+0.21 pt** | +0.06 pt | −0.03 pt |
| b3b − three-seed mean | +0.30 pt | +0.06 pt | +0.08 pt | −0.18 pt | −0.09 pt |
| net alone: TCN mean / b3a / b3b | .9275 / .9334 / .9337 | .9478 / .9526 / .9524 | .9611 / .9654 / .9637 | .9588 / .9632 / .9626 | .9658 / .9704 / .9671 |

**Verdict.** The better setting clears TCN seed 0 by **+0.21 pt at 30 min**, under the 0.3 pt screen, so no seed 1 was
fitted; against the three-seed mean it is +0.08 pt — noise. **Dropped, not tuned.** It does make the *network alone*
better (+0.43 pt at 30 min, ~+0.6 pt at 5 min over the three-seed net mean; inner validation .9579 / .9556 against
.9501–.9535), ~1.3 net-seed sd on one seed each, which the trees absorb in the blend exactly as with B2. Both runs
early-stopped (44 / 42 epochs, ~23 min each). **Caveat:** `--jitter` shifts a start that `data2.sample_plan` already
draws at random every epoch, so B3 was effectively *dropout + context-channel dropout*; no real time-shift was tested.

## B4 — mixed sample lengths, 5 min … 6 h: **dropped**
`--minutes 5,10,15,30,30,60,120,360 --det-budget 21600` (detectors × seconds per sample, so a 6-h sample carries 2
detectors, a 2-h one 3, ≤ 30 min the usual 12), on the plain TCN because B3 was not kept.

| fold 0, blend before the decoder | 5 min | 10 min | 30 min | 1 h | 3 h | 6 h | 24 h | full |
|---|---|---|---|---|---|---|---|---|
| TCN three-seed mean | .9506 | .9639 | .9750 | .9734 | .9814 | .9804 | .9818 | .9809 |
| **B4** | .9456 | .9567 | **.9711** | **.9700** | .9808 | .9807 | .9826 | .9845 |
| B4 − TCN seed 0 | −0.36 pt | −0.54 pt | **−0.26 pt** | **−0.10 pt** | −0.00 pt | +0.21 pt | +0.03 pt | +0.40 pt |
| B4 − three-seed mean | −0.50 pt | −0.72 pt | −0.39 pt | −0.34 pt | −0.06 pt | +0.03 pt | +0.08 pt | +0.36 pt |
| net alone − three-seed mean | −0.94 pt | −1.38 pt | −0.68 pt | −0.36 pt | −0.33 pt | −0.43 pt | +0.11 pt | +0.20 pt |

**Verdict.** Worse at every length production uses the network for (≤ 120 min): −0.26 pt at 30 min and −0.10 pt at
1 h against seed 0, failing both gates. **Dropped.** The only gain is on the full 66–72 h span (+0.36 pt over the
mean), where production switches the network off. Early stopping picked epoch 22 of 30 on an inner validation of
.9427 (TCN .9501). Not tuned, but three things stacked against it and should be read before any retry: inner
validation scores only 5/15/30-min windows, so it selects for short-window skill; the 30-min share of training fell
from 3/7 to 2/8; long samples carry only 2–3 detectors, so a step at 6 h teaches little. The long-span gain suggests
the network *can* use long context — a separate model for > 2 h, if ever wanted, not a mixed one.

## Six folds — **not run**, neither step was kept
`state/best_args.txt` stays `--arch tcn`, `best_seed0.txt` stays `tb_tcn_f0_oof`; the stage-13 six-fold rows (note 13 §2)
remain the reference for the shipped GRU blend.

## GPU hours, bugs
Train 22.8 + 23.2 + 19.9 min, inference 3 × ~4.4 min: **1.3 h** (Track B total 2.3 h). No crashes. `run_trackb.sh`
was not reused (its gate re-appends flags to `best_args.txt` on a re-run; its `spec()` had the note-15 path bug):
`trackb_b3b4.py` is idempotent and passes only native paths. The locked signals were not touched.
