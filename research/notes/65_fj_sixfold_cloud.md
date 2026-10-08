# 65 — TCN function head fj on six folds x 3 seeds (RunPod) + cand64 headline (2026-10-01)
Goal: settle the note-63 fold-0 result (fj joint head blended with the trees at fixed tree weight 0.6) on all six
folds_v4 folds, seeds 0 / 1 / 2. Training recipe unchanged (`neural/tcn53_func.py`, mode joint, cloud package
`%DC_WORK%/cloud_pkg`, `run.sh TAG joint FOLD SEED`). Locked_v2 absent (cand64 asserts it). CPU scoring 6 threads.
## Cloud run
* One RunPod community RTX 5090 (32 GB, 128 vCPU, $0.69/h; no 4090 stock). Image runpod/pytorch 2.4 + torch 2.11 cu128
  (Blackwell needs CUDA 12.8; local is torch 2.14 cu126). Pod up 11:28-14:01 PDT, about 2.5 h. **Spend $1.78**
  (balance $20.00 -> $18.22). Pod terminated; the API lists no pods afterwards.
* Upload is the bottleneck. The office uplink gives ~90 KB/s per TCP stream, so the 2.1 GB package went up as 26 chunks
  over parallel scp: ~55 min, md5-checked. The pod's sshd/proxy resets connections above ~20 at once. A second pod would
  have needed the same upload, so a single pod finished sooner.
* Speed: 12 s/epoch with 2 jobs at once and 19 s/epoch with 4 (GPU 97 %, 23 GB), against ~60 s locally. A full run took
  ~10-18 min. 18 runs in total: 2 validation runs, then 16 queue runs (fj f1-f5 / fj_s1 f1-f5 / fj_s2 f0-f5), 4 at a time,
  ~75 min. Local fj fold 1 was cut off by the orchestrator and re-run on the cloud. Local fj_f0 (seed 0) kept.
* Package bug: `tcn53_func.py train` writes `tcn53/runs/<tag>.done.json` but does not create `runs/`. The two validation
  runs crashed after their last epoch; their done.json was rebuilt from last.pt without retraining, then inference ran.
  `runs/` was created before the queue started. Fix for the package: mkdir runs.
* Outputs: `%DC_WORK%/tcn53/fpreds/fj[_s1|_s2]_f{0..5}.parquet` (18 files, the names cand64 expects); copies, logs and
  best checkpoints are in `%DC_WORK%/tcn53/cloud65/` (fjv_f0 = the cloud fold-0 seed-0 validation, kept out of fpreds).
## Validation (fold 0, `tcn53_feval.py --cands fj,fjv,fj_s1 --out feval_val65`, 0.5/0.5 blend, ATSPM all windows E)
| run | inner-val best | net alone vs trees | 0.5 blend vs trees |
|---|---|---|---|
| fj local seed 0 | .8884 @ep45/53 | -2.67 | +0.77 [+0.04,+1.63] |
| fjv cloud seed 0 | .8851 @ep37/45 | -4.63 | +0.34 [-0.40,+1.22] |
| fj_s1 cloud seed 1 | .8873 @ep46/54 | -3.72 | +0.21 [-0.40,+0.91] |
* Top-class agreement between runs: local fj vs fjv .851, fj vs fj_s1 .857, fjv vs fj_s1 .845, local fj vs local ff .871.
  The cloud runs differ from local by about as much as two local runs differ from each other, so the environment is
  sound. But single-run function spread is large: about 2 pt for the net alone and about 0.5 pt in the blend. The
  note-63 fold-0 number (+0.89, one local seed) was at the lucky end.
## Six-fold headline (`cand64.py all`; fj = 3 seeds averaged per fold, coverage 1.000 (fold 1 .9991); signal bootstrap)
| pool | trees (E) | trees + fj (E) | delta E | delta R |
|---|---|---|---|---|
| >= 30 min (652 sig, 187,980 rows) | .9009 [.8894,.9114] | .9048 [.8930,.9152] | **+0.39 [+0.23,+0.56]** | +0.38 [+0.22,+0.54] |
| 5 min | .8662 | .8774 | +1.13 [+0.88,+1.39] | +1.15 [+0.92,+1.42] |
| 10 min | .8776 | .8863 | +0.87 [+0.63,+1.12] | +0.89 [+0.66,+1.14] |
| all windows | .8928 | .8984 | +0.56 [+0.40,+0.71] | +0.56 [+0.41,+0.70] |
* Realistic set >= 30 min: .9137 -> .9174.
* Per seed (each alone, `evaluation/cand65_seeds.py`), >= 30 min E: s0 +0.36 [+0.18,+0.53], s1 +0.32 [+0.16,+0.48],
  s2 +0.35 [+0.18,+0.52]. At 5 min: +1.02 / +0.91 / +1.00. Averaging the 3 seeds adds ~0.04 pt.
* Per fold, >= 30 min E (3 seeds): f0 +0.67, f1 +0.25, f2 +0.22, f3 +0.52, f4 +0.45, f5 +0.32. All six are positive.
* Same run, end to end vs final_v2 (realistic set, all windows, rows final_v2 covers): candidate +2.95 pt, candidate+fj
  +3.47.
## Verdict
The gain is real: CI > 0, every fold and every seed positive, and the tree-tree control in note 63 was ~0. At >= 30 min
it is +0.39 pt, well under the 1-pt function bar. On short samples it is about +1 pt (5 min +1.13, 10 min +0.87).
Note 63's fold-0 +0.89 shrinks to +0.67 with three seeds, and to +0.39 over all six folds. Cost of shipping: a GPU-trained
TCN function head (CPU inference) next to the trees. Open for the orchestrator: adopt (overall CI > 0, ~+1 pt on 5/10 min)
or drop by the >= 30-min bar. A middle option is to route fj to <= 10-min samples only, but that needs its own nested
check. The 18 fold models make the note-63 context stacker testable on six-fold rows now.
