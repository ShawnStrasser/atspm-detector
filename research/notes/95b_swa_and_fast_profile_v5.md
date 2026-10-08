# 95b — SWA instead of three seeds? and the fast profile on v5 (2026-10-05) — CPU scoring of saved OOF
Code `research/code/final95/swa95.py`, `le2h95.py`; outputs `%DC_WORK%/final_v3_work/f95/oof/{swa95,le2h95_func}.json`,
`s95/phase/score95.json` (arm p2_le2h). Truth v4q, Sept-2026 rows, gate .9 decode. Locked untouched.
## 1. Stochastic weight averaging (user: "why train 3 copies?")
One extra siba run s95_sibaswa fold 0 seed 0 (local A1000, 57 epochs, `tcn69_func --swa`: weights averaged over the last
25 % of epochs = 42..56, BatchNorm statistics averaged too, an approximation of an update_bn pass), filtered inference.
Stacker trained on folds 1-5 (seed-0 inputs; 3-seed mean for mean3), applied to fold 0 (69 signals):
| fold 0, >= 30 min E (13,752) | seed 0 | SWA | 3-seed mean |
|---|---|---|---|
| accuracy | .9277 | .9259 | .9295 |
| vs seed 0 | | -0.18 [-0.60,+0.21] | +0.18 [-0.22,+0.58] |
| mean3 - SWA | | | +0.36 [-0.04,+0.86] |
5 min: SWA +0.55, mean3 +0.59 vs seed 0 (n.s.); 10 min all within 0.15. -> SWA of one run does not replace the 3-seed
mean (one fold, 69 signals: indicative only). Note 97 saw large seed-to-seed variance on fold 0; averaging runs is what
removes it, averaging the late epochs of one run does not.
## 2. Fast profile le2h (note 98) on the v5 OOF
Phase: TCN only on m5..h1, trees alone above, one decoder re-fitted six-fold on the mixed input (iterations 196..762).
Function: v5 mean3 stack <= h1, a 'nonet' stacker OOF (net columns = trees' probabilities, Sept-2026 rows) above.
| >= 30 min | full v5 | le2h | delta |
|---|---|---|---|
| phase E | .9843 | .9823 [.9792,.9850] | -0.20 |
| phase R | .9863 | .9844 | -0.19 |
| function E | .9305 | .9275 [.9183,.9358] | -0.30 [-0.42,-0.19] |
| function R | .9395 | .9365 | -0.31 |
5 / 10 min identical by construction (phase 5 min .9649 vs .9640: one shared re-fitted decoder). Function > 2 h only:
-0.57 [-0.79,-0.36]. Same size as note 98 on v4f (-0.16 phase, -0.33 function).
## 3. Package v5_fast
weights_v5 = v5 weights + fast.json (from weights_v4f) + extras refit on the v5 OOF: decode_trees (2026 trees alone, 496
trees = mean of folds 315..800), stacker nonet and single (Sept-2026 rows). ONNX parity <= 1e-14. package.json -> weights_v5;
references re-frozen; check.py 5/5 PASS. Bench (bench98, warm s / peak MB): 3 h 0.57-0.94 / 292-322; 24 h 0.89-2.21 /
334-455; 30 min = full (0.74-1.27 / 504-739).
