# 73 — Per-component timing profile of the champion, one signal (groundwork for the speed step; no model change) (2026-10-01)
Code `research/code/evaluation/prof73.py` (export | pb | profile | summary | ideas); work `%DC_WORK%/prof73/` (prof.json, r_*.json,
fj_parity.json, pb64.json, ideas*.json, fj_f0.onnx). CPU, 4 threads, numpy boosters, one signal per process; note-71 bench
extracts (non-locked, asserted): typical (19 ch at 30 min / 22 at 3 h), busiest (43 ch), 8 candidate phases; daytime windows.
Warm = models / sessions kept, median of 3 calls; cold = fresh process, first call. Shared machine (one other agent's process):
+-10 % between warm calls. Package parts = `final_v3_candidate_v2`, wrapped, not edited. Research-only parts = PROXIES on the
package's own answers: fj = `fj_f0` exported to ONNX here + torch-free render53 on the GRU's own streams (shared, no extra SQL);
D lanes = package lane step (same tree shape as D: 300 x 15 leaves x 3) + the D feature block; stack-pick inputs = `ln6_pick` +
`ln7_stackhealth` work(); stacker = `s67.stack_X / ctx_X` + an s67-shaped booster (random data, timing only); decode =
`s67.decode_x` (gate .9); night speed = `sp1` (also forced night). Function trees = package 442-feature head for the 229 arm;
setback = package note-41 model for sb7. Twin decode runs on 5 / 10-min samples only (not timed).
## Seconds per component, warm
| component | typ 30 min | typ 3 h | busiest 30 min | busiest 3 h |
|---|---|---|---|---|
| event load / DuckDB tables | .03 | .04 | .03 | .05 |
| phase features (f1 / f2 / similarity / f3) | .17 | .23 | .19 | .28 |
| phase trees, 3 seeds (each .08-.10) | .26 | .26 | .29 | .28 |
| GRU streams (SQL) + raster | .03 | .05 | .03 | .07 |
| **GRU ONNX inference** (1 / 4 pieces) | .61 | **2.78** | 1.32 | **5.19** |
| GRU mix + joint decoder | .08 | .08 | .08 | .08 |
| function features (frame + expert) / function trees, 3 seeds (each .12) | .07 / .36 | .07 / .36 | .07 / .38 | .08 / .36 |
| fj raster (15 ch) / **fj ONNX inference** | .01 / .22 | .04 / **1.01** | .02 / .51 | .07 / **1.89** |
| D lanes (cues + pair model + decode + D block) | .07 | .08 | .16 | .18 |
| health v5 / setback | .04 / .27 | .07 / .28 | .04 / .22 | .09 / .29 |
| stack-pick inputs / context stacker / pick + lane decode / night speed | .01 / .10 / 0 / 0 | .02 / .11 / 0 / 0 | .03 / .11 / 0 / 0 | .04 / .11 / 0 / 0 |
| output assembly + glue | .03 | .08 | .03 | .04 |
| **total warm** | **2.36** | **5.56** | **3.51** | **9.10** |
| **total cold** (+ imports .60-.69, model parsing .42, first-call research imports ~.15) | 3.72 | 6.89 | 4.88 | 10.52 |
Busiest-by-events (31 ch) 3 h: 7.20 warm / 8.46 cold. Target 4-5 s at 3 h: typical misses by ~0.6-1.9 s, busiest by ~4-5.5 s; the two networks are 66 / 78 % of the 3-h time, all the rest ~1.8-2.0 s.
## Checks
GRU pair batch 64 vs 512: p_gru max |diff| 0, predict() output identical (`DataFrame.equals`), 3 signals x 30 min / 3 h (6 / 6).
fj ONNX (one graph: pair net + candidate pooling + head; opset 17; dynamic D / K / T) vs the research torch forward (fp32), real
inputs, 2 signals x 30 min / 3 h, 10 pieces: |log-prob| <= 1.2e-5, |logit| <= 1.3e-5, pooled prob <= 1.3e-6 -> PASS (1e-4);
raster copy and numpy channel assembly = `tcn53.render53 / assemble53` exactly.
## Three biggest costs (busiest 3 h, warm) and one concrete idea each (`prof73.py ideas`, measured on both 3-h signals)
1. GRU inference 5.19 s (57 %). Score only candidates with TREE probability >= .01 (19-27 % of pairs): 5.23 -> 1.04 s busiest,
   2.88 -> 0.80 typical (**-4.2 / -2.1 s**); pre-decoder blended top phase unchanged on all 65 detectors (max prob change .006 /
   .16). Needs a six-fold OOF check from saved trees + GRU OOF (CPU, no retrain). int8 dynamic quantisation: no gain (5.7 s);
   K = 2 pieces halves it (accuracy unmeasured).
2. fj inference 1.89 s (21 %). Same candidate filter (cost scales with pairs): expect **-1.4 / -0.75 s**, but it changes fj's
   max-pool over candidates -> needs fj re-inference OOF (one cloud run). int8 is 18x SLOWER (ConvInteger): dropped. Later:
   fj's own phase head instead of the GRU would remove 2.8-5.2 s (needs fj phase-head OOF).
3. Tree ensembles run tree-by-tree in numpy, ~0.85 s (phase .28, function .36, decoder .05, lanes .07, stacker .10; + setback's
   9 quantile boosters). Compile each LightGBM text model to an ONNX TreeEnsemble (onnxruntime already shipped; converter
   `lgbm_to_onnx` in prof73, double thresholds): phase s0 .093 -> .0017 s, function s0 .127 -> .0018, decoder .049 -> .0007,
   max |diff| 2.4e-7 -> **~-0.8 s**, faster cold load too. (All-trees-at-once numpy: function only, 7-13x.)
Projection, warm: 1 + 3 (measured) busiest 3 h 9.1 -> ~4.0 s, typical 5.6 -> ~2.6 s; + 2 -> ~2.6 / ~1.8 s. Cold adds ~1.1 s:
a service should keep models loaded. Note 71's fj estimate (+0.9 / +2.7 s) is now measured 1.0 / 1.9 s.
Open for the orchestrator: OOF check of idea 1 (CPU) before any speed decision; idea 3 is float-noise only; idea 2 needs a cloud
re-inference. Nothing shipped; model/, review/, USER_INPUT and locked_v2 untouched.
