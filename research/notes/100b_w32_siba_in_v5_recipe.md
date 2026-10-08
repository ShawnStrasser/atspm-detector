# 100b — Plain width-32 siba in the v5 function recipe (2026-only, six folds x 3 seeds) (2026-10-06)
Follow-up of note 100 (fold-0 screen: a width-32 siba net matched the 3-member mean3 at 1/15 of the network cost). Orchestrator
brief: the note-95 2026-only setup, six folds, seeds 0/1/2; refit the v5 context stacker on the w32 OOF; score vs v5; CPU time.
Code: `final100/f100.py` (stack = func95.stage_stack with another net input / output folder; score = func95 scorer vs v5,
paired signal bootstrap). Training = `tcn69_func.py` (snapshot snap95) `--sib attn --max_det 32 --width 32 --accum 2
--frows func_rows_v4q_2026`, inference filtered (2026 tree OOF s95/phase/full/oof.parquet, p >= .01) = the s95_siba recipe
with width 32 instead of 96. Seed 0: local A1000, queue `tcn53/q100` (29-52 min per fold). Seeds 1/2: RunPod pod (the
orchestrator ran the network side from the bundle `s100/cloud_b`: data md5-checked against local (DATA_OK), 12/12 jobs ok,
outputs md5-verified locally in `s100/cloud_b_out`). Work `%DC_WORK%/s100/b/` (P_*.npy, f100.json). Locked signals: never
used (func_table / frame asserts; v4q truth, Sept-2026 rows only).
## Check
`f100.py stack v5re s95_siba` reproduces P_v5.npy exactly (max |diff| 0.0); seed-0 siba alone .9296 (= note 95's .9296).
## Accuracy (v4q truth, Sept-2026 rows, gate .9 decode; >= 30 min E 126,505 rows / 642 signals)
| net input to the v5 stacker | >= 30 E | vs v5 [95 % CI] | >= 30 R (vs v5) | 5 min E (vs v5) | 10 min E (vs v5) |
|---|---|---|---|---|---|
| v5 = siba width 96, 3 seeds | .9305 | — | .9395 | .9047 | .9122 |
| siba width 96, seed 0 only | .9296 | -0.09 [-0.20,+0.02] | -0.12* | -0.39* | -0.43* |
| siba width 96, seed 1 only | .9301 | -0.05 [-0.13,+0.05] | -0.05 | -0.43* | -0.07 |
| **w32, 3 seeds (mean3)** | **.9307** | **+0.02 [-0.09,+0.14]** | .9397 (+0.02 [-0.09,+0.13]) | -0.07 [-0.30,+0.16] | -0.20 [-0.40,-0.01] |
| w32, seeds 1+2 | .9295 | -0.10 [-0.22,+0.01] | -0.09 | -0.46* | -0.27* |
| w32, seed 0 only | .9299 | -0.06 [-0.22,+0.10] | -0.11 | -0.59* | -0.40* |
| w32, seed 1 only | .9294 | -0.11 [-0.25,+0.03] | -0.12 | -0.47* | -0.48* |
w32 mean3 by class >= 30 E vs v5: Advance +0.11, Presence -0.04, Count +0.01, YR -0.17, non-ATSPM +0.06 (all n.s.); folds
-0.20 / +0.06 / +0.10 / +0.06 / +0.04 / +0.01. 10 min R -0.17 [-0.37,+0.02].
## CPU (bench98 protocol: fresh process per signal, 4 threads, warm median of 3, 2 passes, idle machine)
Timing copies of v5_fast with the siba members swapped (`s100/pkg_w32` = one w32 net, `s100/pkg_w32m3` = three; fold-0
checkpoints, export83 pair + head graphs; ACCURACY OF THESE COPIES IS MEANINGLESS: their stacker is v5's). Profile 'full'.
| warm s (peak MB) | v5 full | w32 x 3 | w32 x 1 | (v5 siba1, loaded machine) |
|---|---|---|---|---|
| 3 h, typical r8 / r11 | 2.01-2.03 / 1.51-1.54 (655-668 / 517-524) | 1.32-1.34 / 1.03 (494 / 408-416) | 1.13-1.14 / 0.89-0.90 | 1.47 / 1.15 |
| 3 h, busiest ev / ch | 2.17 / 2.64-2.65 (638-641 / 789-794) | 1.59-1.62 / 1.78 (522 / 566-583) | 1.40-1.41 / 1.53-1.58 | 1.71 / 1.93 |
| 30 min, range | 0.74-1.28 (498-748) | 0.60-0.99 (397-528) | 0.58-0.91 (350-469) | 0.64-1.04 |
w32 x 3 vs v5: -26..-34 % warm time at 3 h, -20..-23 % at 30 min, -18..-27 % peak RAM at 3 h. (Beta at 3 h: 0.40-0.57 s, note 98.)
## Verdict
* **Three width-32 nets = v5 on accuracy (+0.02 E / +0.02 R at >= 30 min, CIs +-0.12) at about 70 % of the time and
  about 80 % of the memory at 3 h.** Short samples: 5 min tie (-0.07), 10 min -0.20 E (CI just below 0; R n.s.). That is the
  only cost, and is smaller than any single-member variant.
* A single w32 net = a single full member (.9299 vs .9296 / .9301) but loses ~0.5 pt at 5 / 10 min, like any single member:
  averaging seeds is what pays (note 95b), width does not.
* Note 100's fold-0 hint that plain small nets BEAT mean3 (+0.4) does not survive six folds and the stacker: it is a tie.
* To ship (orchestrator / user): 3 full-data w32 refits (s95_sibafull recipe, --width 32: 42 epochs, lr schedule replayed
  from these 18 fold runs; ~25-35 min each locally), export with assemble95 siba (export83 graphs), stacker refit on this
  w32 mean3 OOF (f100.py stack = the v5 recipe), check.py --freeze / check, bench98. Not done here (screen only; nothing
  shipped, model/ untouched).
