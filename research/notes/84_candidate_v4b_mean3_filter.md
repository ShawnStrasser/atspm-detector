# 84 — candidate v4b: three siba members (mean3 stacker) + siba filter ON (2026-10-04) — not shipped, locked untouched
Package `%DC_WORK%/final_v3_candidate_v4b/` (copy of v4; v4 untouched; `model/` = final_v2). Code `research/code/final84/`
(fit84 mean3 stacker, oof84, export84, assemble84, bench84); work `%DC_WORK%/final_v3_work/{v3fit84,f84}/`. CPU <= 6 threads
(4 + 2), GPU not used, locked_v2 asserted absent in every fit / parity / bench.
## What changed vs v4
* Function net = the three full-data siba v4l refits (x74_sibafull4l, _s1, _s2; tcn53/models/*_full.pt = the s74/full ONNX
  twins), re-exported as package pair + head graphs (export83 graphs: head masks filtered candidates), averaged in
  probability space. FuncNet now loads only members whose two files are present (assert >= 1); function_stage passes
  len(members) -> stacker 'mean3' with >= 2, 'single' (v4 files, kept) with 1.
* Stacker 'mean3' = fit83 recipe (v4l target, v4l OOF trees, 47 columns, PRM0, 150 rounds, seeds 0/1/2) trained on the
  3-seed-MEAN siba OOF, 1 x rows (268,766 rows / 652 signals), as notes 67 / 69. ONNX vs text 6.7e-16.
* SIBA_FILTER default ON (tree p >= .01); off via predict(siba_filter=False), DC_SIBA_FILTER=0, CLI --no-siba-filter.
## Parity (export84 parity: package FuncNet vs torch forward69 fp32, 4 bench signals x 30 min / 3 h, 120 pieces)
Each member: |dlogp| 2.1e-5, logits 1.5e-5, filtered (random keep mask) 5.5e-5, member prob 1.4e-6; 3-member average vs
torch average |dprob| 5.4e-7, filtered 1.2e-6 -> PASS 1e-4. Members disagree on argmax for 5-29 % of detectors per case.
## OOF headline (oof84: six folds, v4l truth, gate .9 decode, paired signal bootstrap; n >= 30 min E 187,444)
Filtered OOF nets exist for seed 0 only (x74_sibaflt; x74_sibanf = same models / inputs unfiltered); seeds 1 / 2 are
unfiltered. "+filter" = single on flt0; mean3 on mean(flt0, s1, s2) (1 of 3 members filtered). Stackers trained on
unfiltered OOF (as the package). single_s0 reproduces note 83's arm (2.98e-8).
| v4l | single+filter | mean3+filter | mean3 - single [95 % CI] | unfiltered: single (3 seeds) / mean3, diff |
|---|---|---|---|---|
| >= 30 min E | .9174 [.9062,.9273] | .9191 [.9079,.9291] | +0.17 [+0.06,+0.29] | .9181 / .9191, +0.10 [+0.03,+0.18] |
| >= 30 min R | .9273 | .9290 [.9183,.9384] | +0.17 [+0.07,+0.28] | .9279 / .9290, +0.11 [+0.04,+0.18] |
| 10 min E | .8998 | .9036 | +0.38 [+0.19,+0.57] | .8993 / .9034, +0.40 |
| 5 min E | .8909 | .8940 | +0.31 [+0.12,+0.50] | .8907 / .8941, +0.35 |
Filter: single nf0 -> flt0 -0.05 [-0.11,0.00] E; mean3 nf0 -> flt0 -0.00 [-0.03,+0.02] (5 / 10 min -0.01 / +0.02).
vs champion (note 81, .9190): mean3+filter +0.01 [-0.05,+0.07] E (5 / 10 min +0.04 / +0.05) = the single-member cost
of v4 is won back. By class >= 30 E (mean3+f - single+f): Advance +0.34*, Count +0.41*, Presence +0.07, YR -0.04,
non-ATSPM -0.17 [-0.46,+0.14]. Caveats: OOF nets v3s-trained (production v4l full-data); filtered member seed 0 only.
## Speed / RAM (bench84: one signal per call, fresh process, 4 threads, filter ON; warm median of 3 / cold; peak MB)
| mean3 + filter (v4b default) | 30 min | 3 h | 24 h |
|---|---|---|---|
| typical r8 22 ch | 1.53 / 2.86 (845) | **2.99** / 4.05 (691) | 2.99 / 4.64 (689) |
| typical r11 18 ch | 1.19 / 2.48 (643) | **2.47** / 3.53 (628) | 2.66 / 3.99 (635) |
| busiest events 31 ch | 1.88 / 3.14 (890) | **3.42** / 4.61 (733) | 4.50 / 5.87 (788) |
| busiest channels 43 ch | 2.11 / 3.12 (915) | **4.06** / 5.52 (893) | 5.41 / 7.00 (1012) |
single + filter (same package, members 1/2 removed): 3 h warm 2.21 / 1.76 typical, 2.47 / 2.74 busiest; 24 h busiest
3.63 / 4.02; peak <= 793 MB. mean3 costs ~+0.7 s typical / +1.0-1.3 s busiest at 3 h (siba 1.74 vs .58 s busiest 3 h)
and ~+0.2 GB peak. Import time not captured here (bench84 imports before timing; note 83: .6-.8 s).
## Pick (rule: mean3 if busiest 3 h warm <= 5.0 s and typical <= 4 s)
Busiest 4.06 / 3.42 s, typical 2.99 / 2.47 s -> **mean3 + filter = v4b default**. check.py --freeze then check.py
4 / 4 PASS (stored answers, phase renumbering 1.6e-7, channel order 0.0, torch / lightgbm / scipy / sklearn blocked).
model_card.json: candidate_v4b_note84 entry + sha256 refreshed.
## Open for the orchestrator
* Nearly the whole margin to the 5 s target is used at busiest 3 h (4.06 s); 24 h busiest 5.41 s warm, peak 1.0 GB.
* A filtered OOF for seeds 1 / 2 (2 x 6 GPU inferences, ~2 h) would make the mean3+filter number fully clean.
* Swap into model/ only at the end of the search (copy v4b, run check.py).
