# 100 — Distil the 3-member siba ensemble into one small TCN? Fast screen, fold 0 (2026-10-06)
Orchestrator brief (user: "did you try distilling"; then: report the accuracy-vs-speed trade-off at several sizes). FAST SCREEN
ONLY: fold 0, no locked signal anywhere (func_table asserts; teachers and students are fold-0 models, scored on fold 0 held-out).
Code `neural/tcn100_distill.py` (student trainer), `neural/time100.py` (ONNX CPU timing), `final100/s100.py` (scorer = note 97
b97 + seed-group arms). Work `%DC_WORK%/s100/` (out/ = 18 checkpoints + fold-0 preds + logs, md5-verified copy of the pod;
podlogs/, onnx/, s100.json, time100.json). GPU: RunPod dc95a (RTX PRO 6000, shared with the note-95 agent, then alone).
## Set-up
* Teacher = x86_siba4l fold-0 seeds 0/1/2 (note 86; v4l rows, post-76 inputs). mean3 = mean of their probabilities.
* Distillation loss (ONLINE: the 3 teachers run on every training batch, the same windows the student sees): 0.5 x true-label CE
  (function + phase) + 0.5 x T^2 x KL(teacher || student) at T = 2, for the function head AND the phase head, on every actuated
  detector. Otherwise the note-69 siba recipe (sib attn, max_det 32, inner-val stopping). Student = the same Net69 with a narrower TCN
  (width 96 -> 56 / 32 / 16, 7 blocks kept: same receptive field). Params 813 k -> 366 k / 206 k / 143 k.
* Controls: the same small nets trained with NO teacher ("plain", note-69 recipe), and plain / distilled at width 96.
  The pod runs use accum 1 (the local teachers used --accum 2); the plain width-96 runs measure that difference.
* Inputs are the same as for the teachers: the pod's inference pair counts match local x86 exactly (374,932 of 681,904
  pairs run, 73,594 filtered detector-pieces). ONNX export with export74 --check: PASS for all students (max |dlogp| 7.6e-6).
## Accuracy, fold 0 (mean over seeds; the delta CI is a paired signal bootstrap of the seed-averaged correctness)
Blend = fixed trees 0.6 + net 0.4 -> gate .9 decode, ATSPM E (19,757 rows >= 30 min); net alone = argmax of 7, labelled.
| arm (seeds) | net alone >= 30 | blend >= 30 E (seed range) | vs mean3 [95 % CI] | 5 / 10 min E | 5 / 10 vs mean3 |
|---|---|---|---|---|---|
| teacher member, single (3) | .8108 (.772-.843) | .8984 (.8909-.9037) | -0.15 [-0.28,-0.04] | .8677 / .8786 | -0.14 / -0.42 |
| **teacher mean3** | .8339 | **.8999** | — | .8691 / .8828 | — |
| distilled w96 (1) | **.8412** (+0.73*) | .9018 | +0.19 [-0.02,+0.37] | .8716 / .8843 | +0.25 / +0.15 |
| distilled w56 (3) | .8313 | .9007 (.9004-.9014) | +0.08 [-0.10,+0.26] | .8728 / .8824 | +0.37 / -0.03 |
| distilled w32 (3) | .8301 | .9010 (.8991-.9031) | +0.11 [-0.09,+0.34] | .8734 / .8808 | +0.42 / -0.20 |
| distilled w16 (2) | .8023 (-3.2*) | .8979 (.8955-.9003) | -0.20 [-0.55,+0.14] | .8677 / .8782 | -0.14 / -0.46 |
| plain w96, pod (2) | .8232 (-1.1*) | .9011 (.8990-.9032) | +0.12 [-0.13,+0.39] | .8664 / .8770 | -0.28 / -0.57* |
| plain w56 (3) | .8319 | .9038 (.9019-.9058) | **+0.40 [+0.10,+0.69]** | .8768 / .8838 | +0.77* / +0.10 |
| plain w32 (3) | .8312 | .9043 (.8997-.9071) | **+0.44 [+0.13,+0.80]** | .8764 / .8835 | +0.73* / +0.07 |
| plain w16 (1) | .8224 | .9006 | +0.08 [-0.35,+0.50] | .8719 / .8820 | +0.28 / -0.08 |
R tracks E within 0.05 pt (e.g. plain w32 R .9089, +0.42*; distilled w32 .9057, +0.10). * = CI excludes 0.
## CPU cost of the network per signal (ONNX Runtime, 4 threads, spinning off, warm median of 7, filtered pairs; seconds)
| net | typical 30 min / 3 h | busiest 30 min / 3 h | 3 h vs mean3 |
|---|---|---|---|
| mean3 (3 x w96) | .158 / .517 | 1.14 / 4.80 | 1x |
| one member (w96) | .052 / .171 | .380 / 1.53 | 3.0-3.1x faster |
| w56 | .022 / .074 | .170 / .700 | 6.9-7.0x |
| w32 | .011 / .035 | .077 / .312 | 15x |
| w16 | .005 / .017 | .032 / .130 | 30-37x |
Typical = 18 detectors / 4 candidates (filter keeps 128 of 288 pairs at 3 h); "busiest" (44 det / 6 cand) is not in the tree
pair table, so the filter keeps every pair there: an upper bound. Network time only (the raster is shared with the phase TCN).
## Verdict
* **Yes, one small network matches the 3-member ensemble in this screen, at a fraction of the cost.** Width 32 (15x cheaper than
  mean3 at 3 h, 5x cheaper than one member) scores >= mean3 at >= 30 min and 5 min, with or without distillation; width 56
  likewise. Width 16 starts to lose (net alone -3 pt distilled; blend at single-member level).
* **Distillation itself is not what does it.** Plain small nets do as well, and in the blend +0.3 better than distilled ones
  (w32 .9043 vs .9010; w56 .9038 vs .9007). Distillation shrinks the seed spread (w56 blend .9004-.9014 vs .9019-.9058) and
  lifts net-alone accuracy at full width (w96 distilled .8412 = +0.7* over mean3), but this does not carry into the blend.
  Likely reason: the teachers are over-sized for this data, and a student copies their mistakes, which the trees already share.
* Caveats: one fold, the fixed 0.6/0.4 blend (not the mean3 context stacker in the package; note 84 found mean3 +0.17 over a
  single member there), and the plain arms were trained on the pod with accum 1. Plain w96 on the pod is +0.27 [+0.02,+0.57]
  over the local members (local seed 0 is a known weak run), so part of the plain-small gain over mean3 may be the set-up. The
  small-vs-w96 comparison inside the pod runs (.9038-.9043 vs .9011, 3 vs 2 seeds) stays positive but is inside seed noise.
* Not a shipping decision. Next if wanted (orchestrator): plain w32 (and w56) on all six folds x 3 seeds -> refit the mean3 /
  single stacker on that OOF -> compare with v4f inside the package recipe. On a 96-GB card that is ~2 h of GPU (3 jobs in
  parallel; ~15 min per run alone); on the local A1000 about a day. Expected gain: ~0.45 s per typical 3-h call (v4f warm ~1.8 s).
* Cloud: 18 runs + 0 failures; the pod was shared until ~06:25 UTC. Pod outputs copied back and md5-verified (109 files);
  nothing of mine left running.
* Follow-up note 100b (six folds x 3 seeds, v5 stacker): w32 x 3 = v5 (+0.02 E), not better; 3 h package time -26..-34 %.
