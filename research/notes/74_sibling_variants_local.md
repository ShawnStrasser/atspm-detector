# 74 — Function-net follow-ups on the local A1000: sibm vs siba, siba + cw2, specialist decider, GRU backbone (2026-10-02)
Queue (orchestrator, note 69 follow-up), local RTX A1000 8 GB only, detached resumable runner `%DC_WORK%/tcn53/q74/queue74b.ps1`
(jobs `q74/queue74.txt`, markers `q74/done/`, logs `tcn53/logs74/`, code snapshot `tcn53/snap74`). Training = `neural/tcn69_func.py`
(note-69 recipe; new knobs `--arch gru`, `--accum N`, `--full`, `--keep`, `--ckdir`; defaults unchanged). Scoring =
`evaluation/s74.py` = note-69 `s69_on67` generalised: context stacker (3 LightGBM seeds, OOF over the six folds) on trees + the
net(s), D-lane gate .9, paired signal bootstrap; `s74.py check` reproduces note 69's siba stacker to 3e-8. locked_v2 absent
(asserted in setup and in func_table). Headline = >= 30-min pool (187,980 E / 183,024 R rows, 652 signals).
## Memory on the 8 GB card (smoke = one full epoch, fold 0)
* TCN siba / sibm (max_det 32), no accumulation: peak 3.6 GiB allocated but ~1.8 GB spilled to shared system memory
  (WDDM), 85-100 s/epoch. **`--accum 2`** (each batch's windows split in 2 micro-batches, losses weighted by labelled-row
  share -> the same full-batch gradient; only BatchNorm statistics are per half batch): 2.5 GiB, no spill, **69 s/epoch**.
  All local TCN runs use it; note 69's cloud runs (siba, sibm f0/f3 seed 0) did not.
* GRU backbone (fp32, note 68): 12 GiB peak without accumulation (368 s/epoch, spilling); accum 4 still spills;
  **`--accum 8`** (one window per micro-batch) 4.2 GiB, 116 s/epoch; inference needs `--ichunk 128`.
* A full TCN run = 35-57 epochs, ~66 min incl. ~9 min inference. One transient DataLoader I/O error (sibm_s2 f0) was
  retried by the runner from last.pt.
## 1. sibm (sibling mean-pool) six folds x 3 seeds vs siba, inside the champion (`s74.py compare --a siba --b sibm`)
| pool | siba | sibm | sibm - siba [95 % CI] |
|---|---|---|---|
| >= 30 min E | .9169 | .9165 | -0.04 [-0.16,+0.09] |
| >= 30 min R | .9281 | .9278 | -0.03 [-0.16,+0.09] |
| 5 / 10 min E | .8921 / .9009 | .8900 / .8990 | **-0.21 [-0.42,-0.01]** / -0.19 [-0.41,+0.02] |
| all windows E | .9112 | .9103 | -0.08 [-0.21,+0.04] |
By class >= 30 E: Advance -0.19 [-0.45,+0.03], Presence -0.04, Count -0.08, YR +0.19, nonATSPM +0.27 [-0.19,+0.76]. Folds -0.10 /
+0.19 / -0.16 / +0.16 / -0.17 / -0.14. Seed 0 only (siba0 vs sibm0 stackers): -0.03 [-0.20,+0.13], 5 / 10 min -0.20 / -0.34.
CPU inference per signal (torch fp32, 4 threads, warm, fold-0 nets; `neural/time74.py`; shared machine, +-10 %): typical (18 det,
4 cand) 30 min / 3 h: siba .20 / .64 s, sibm .17 / .68 s, no-sibling base .19 / .80 s; busiest (44 det, 6 cand): siba .58 / 2.76,
sibm .63 / 3.06, base .65 / 3.06. The sibling block is a few % of the pair TCN: **no speed difference**.
Verdict: sibm is not cheaper and is worse on 5-min samples (CI < 0) -> **keep siba**; sibm dropped.
## 2. siba + cw2 (ATSPM classes weight 2, six folds x 3 seeds, local) vs siba, inside the champion
>= 30 min .9169 -> .9166 E, -0.02 [-0.13,+0.09]; R -0.03 [-0.13,+0.07]; 5 / 10 min -0.11 / -0.12 (n.s.); all -0.05. By class
>= 30 E: Advance -0.03, Presence -0.11, Count +0.13, YR +0.30 [0.00,+0.64], nonATSPM -0.11. Folds -0.06 / +0.22 / +0.01 / +0.07 /
-0.01 / -0.34. Seed 0 alone -0.01 [-0.16,+0.15]. Note 69's Advance gain of cw2 (+0.49 on 2 folds, no siblings) does not
survive on top of sibling attention. -> **dropped**.
## 3. Specialist decider (user idea): context stacker fed trees + siba + sibm + siba_cw2 (+ same context), OOF as note 67
Extra columns per specialist (3-seed average): 7 log-probs, margin, entropy, agrees-with-siba (82 columns in all).
| arm vs siba stacker | >= 30 E | >= 30 R | 5 min E | 10 min E |
|---|---|---|---|---|
| spec (siba + sibm + cw2, 3 seeds each) | +0.03 [-0.03,+0.08] | +0.03 [-0.02,+0.09] | +0.12 [-0.01,+0.25] | -0.02 |
By class >= 30 E: Advance +0.07 [-0.00,+0.16], Presence -0.02, Count +0.04, YR 0.00, nonATSPM +0.04; folds +0.11 / +0.03 / 0 /
+0.02 / 0 / +0.06. Control, single seed each (3 nets either way): spec0 (siba0 + sibm0 + cw2_0) vs seedx0 (siba seeds 0 / 1 / 2
as three column sets) -0.03 [-0.13,+0.06] >= 30, +0.04 / +0.01 at 5 / 10 min -> what the specialists add is seed diversity,
not specialism. -> **not adopted** (CI includes 0, and it triples network inference).
## 4. GRU backbone of the final head — queued
## Orchestrator add-ons (2026-10-02), queued after 4
* Full-data siba refit `x74_sibafull` (seed 0, every function-table training signal, no inner-val; 43 epochs = median best
  epoch 42 of the six siba seed-0 fold runs + 1; lr per epoch = median of those runs' ReduceLROnPlateau schedules, replayed
  from their logged scores - replay matches the logged lr). Output `tcn53/models/x74_sibafull_full.pt` (final weights).
* ONNX: `neural/export74.py` exports one graph per signal x piece (inputs = kept pairs + their detector / candidate index,
  so the candidate filter is native). Parity on real pieces, with and without a random filter: max |dlogp| 1.1e-5 -> PASS.
* siba candidate filter (tree p >= .01, OOF trees `cand64/phase_oof.parquet`), seed-0 fold models, inference only.
* Note 76 changed `neural/tcn53.py` (partner-channel overlap ties broken by behaviour, not candidate order). The refit and the
  filter inference run from snapshot `tcn53/snap74b` (new code); an unfiltered siba inference on the same new inputs
  (x74_sibanf) is the filter's reference. sibm, cw2 and GRU runs keep the old inputs (same as the siba they are compared to).
