# 98 — Final model vs beta on speed / RAM; simplified variants; package v5_fast (2026-10-05) — not shipped, locked untouched
User rule: if the final model needs > 2x the beta's time OR memory, also build a simplified version that trades accuracy for
efficiency. Code `research/code/final98/` (bench98, p98_phase, f98_func, vsbeta98, assemble98); work `%DC_WORK%/s98/`,
`%DC_WORK%/final_v3_work/f98/` (bench/, oof/, logs/). CPU only (LightGBM 4 threads), GPU not used, locked_v2 asserted absent.
## 1. Beta (final_v2) vs v4f — bench98 = note-84 protocol (one signal per call, fresh process, 4 threads; warm = median of 3)
Note-71 extracts typical r8 / r11, busiest by events / channels. Beta 4 passes, v4f 2 passes, means. Machine load recorded per
case (other agents' processes < 1 core; the 45-60 % "machine CPU" in the v4f runs was v4f itself, see 3).
| v4f / beta | 30 min | 3 h | 24 h |
|---|---|---|---|
| warm time | 0.9-1.4x (beta runs its GRU <= 2 h) | **4.7-6.2x** (2.2-3.3 s vs 0.40-0.57 s) | 3.0-4.4x |
| cold time | 0.8-1.6x | 2.7-3.3x | 1.8-3.1x |
| peak RAM | 0.49-0.92x (beta 0.58-1.56 GB) | **2.1-2.7x** (551-755 vs 259-296 MB) | 1.8-2.5x |
-> over 2x at 3 h and 24 h on both time and memory: simplified variants built.
## 2. Accuracy vs cost of the variants (six-fold OOF from saved predictions; nothing trained on GPU)
Phase: note-90 pipeline, decoder re-fitted six-fold for each first-stage input (trees-only decoder: iterations 116..798).
Function: note-90 recipe (v4o trees, x86_siba4l OOF filtered, 47-col stacker, gate .9 decode, v4l truth); frame phase input
replaced by each phase arm's top (oof90 tcnphase method, 62 % coverage; tree features held). nonet = stacker refit with no
net inputs; single = 'single' recipe on single-seed inputs (mean of seeds 0-2). >= 30 min E (Δ pt vs full [95 % CI]).
Time / peak at 3 h, warm, typical-busiest, x beta.
| variant | phase E | function E | 3 h warm s (x beta) | 3 h peak MB (x) |
|---|---|---|---|---|
| (a) full = v4f (spin off) | .9816 | .9197 | 1.77-2.98 (4.2-5.7x) | 545-746 (2.1-2.7) |
| (b) 1 siba member | .9816 | .9180 (-0.17 [-0.24,-0.11]) | 1.22-2.03 (3.1-3.9x) | 426-585 (1.6-2.1) |
| (c) no function net | .9816 | .9112 (-0.85 [-1.07,-0.63]) | 0.90-1.48 (2.3-2.8x) | 372-479 (1.3-1.7) |
| (d) no phase net | .9757 (-0.59 [-0.74,-0.44]) | .9195 (-0.02 n.s.) | 1.47-2.58 (3.7-4.9x) | 486-659 (1.8-2.4) |
| (e) no networks | .9757 (-0.59) | .9111 (-0.87 [-1.09,-0.64]) | 0.58-1.01 (1.5-1.8x) | 292-323 (1.1-1.2) |
| (f) phase net <= 2 h only | .9799 (-0.16 [-0.24,-0.08]) | .9197 (+0.00) | = (d) | = (d) |
| **(g) both nets <= 2 h (le2h)** | **.9799 (-0.16)** | **.9165 (-0.33 [-0.45,-0.20])** | **= (e)** | **= (e)** |
| (h) le2h + 1 siba member | .9799 | .9153 (-0.44) | = (e) | = (e) |
R: le2h phase .9825 (-0.16), function .9257 (-0.35). 5 / 10 min: le2h = full (identical answers <= 2 h); (e) phase -1.93 /
-1.41, function -2.49 / -1.95. (d): the phase net adds nothing to FUNCTION (tree features held at the frame phase).
vs beta, paired rows (note-85 protocol, vsbeta98): le2h phase .9800 vs .9802 (-0.02 [-0.07,+0.02]; > 2 h -0.00; 3 h +0.02);
function .9174 vs .8761 (+4.13 [+3.39,+4.87]; 3 h +3.66); full v4f +0.14* / +4.43. So le2h gives up the phase gain over the
beta (all of it was > 2 h, +0.30 there) and keeps ~93 % of the function gain.
## 3. Cheap engineering
* onnxruntime pool threads spinning (41 sessions x 4 threads): v4f burned 27-39 CPU-s per 3-h call (~11 cores). Spin off
  (session.intra_op.allow_spinning 0): same answers (check.py drift 0.0), CPU 4.6-7.6 s, wall -0.3..-0.8 s, cold -1..-2 s.
  The le2h trees path: 0.8-1.5 CPU-s at 3 h vs beta 1.3-1.7.
* Not done: fewer pieces > 2 h (no per-piece OOF for the TCN / siba: needs GPU re-inference); int8 / fp16 ONNX (note 73: int8
  no gain for the GRU, 18x slower for the function TCN); health_core (0.1-0.5 s) / setback (0.1-0.3 s) are the next costs.
## 4. Pick and package
Within 2x on both: (e), (g), (h) at 3 h / 24 h; (g) le2h is the most accurate. **Recommend le2h.** Package
`%DC_WORK%/final_v3_candidate_v5_fast` = v4f copy + note-95 health_score fix + fast profiles (`weights_v4f/fast.json`: full,
siba1, no_funcnet, no_phasenet, trees, le2h_phase, le2h, le2h_siba1; default le2h; predict(profile=) / --profile /
DC_FAST_PROFILE) + `package.json` weight-set switch (weights_v4f now, weights_v5 later; DC_WEIGHTS) + decode_trees.onnx +
stacker nonet / single (ONNX vs text <= 1e-15). Networks load lazily (never loaded when a call does not need them).
check.py 5 / 5 (new 5th: no-network path under profile 'trees': stored answers, phase renumbering, channel reversal);
references per weight set (reference/weights_v4f; 30-min answers = v4f except health_score NaN for not_enough_data).
Final bench (default le2h, 3 passes, beta 4 passes), x beta:
| v5_fast le2h | 30 min | 3 h | 24 h |
|---|---|---|---|
| warm | 0.74-1.09x (0.82-1.36 s) | **1.45-1.79x** (0.58-0.95 s) | 1.53-2.00x (0.90-2.38 s; busiest_ev 2.00) |
| cold | 0.44-0.79x | 0.82-1.08x | 0.93-1.39x |
| peak | 0.48-0.90x | **1.09-1.15x** (291-322 MB) | 1.12-1.16x |
## Open for the orchestrator
* Re-base on v5: the three extras must be refit on the v5 OOF inputs (trees-only decoder on the 2026 phase pool; nonet / single
  stackers on the 2026 function OOF) — same recipes, scripts are wired to the v4f inputs; then `assemble98.py extras --base
  weights_v5`, package.json -> weights_v5, check.py --freeze, check.py, bench98.
* Choice for the user: le2h (within 2x, phase = beta, function +4.1) vs full (phase +0.14 over beta, function +4.4, 4-6x time).
* 24 h busiest by events sits at 2.00x: within noise of the bar; health_core / setback engineering would give margin.
