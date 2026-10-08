# 75 — Integrated champion as a package: final_v3 CANDIDATE v3 (2026-10-02) — not shipped, locked untouched
Package `%DC_WORK%/final_v3_candidate_v3/` (copy of candidate v2 + new modules; `model/` = final_v2, untouched). Code
`research/code/final75/`: `fit75.py` (CPU refits), `export75.py` (siba ONNX), `onnx_trees75.py` + `assemble75.py` (trees ->
ONNX, weights), `parity75.py`, `member75.py`, `bench75.py`, `card75.py`; work `%DC_WORK%/final_v3_work/{v3fit,parity75,bench75}/`.
CPU only (<= 6 threads); GPU queue q74 not touched. locked_v2 asserted absent in every fit / parity / bench.
## What is in it (runtime = onnxruntime + numpy / pandas / duckdb; no torch / lightgbm / scipy / sklearn)
* phase: ranker 261 feat x 3 seeds (candidate v2 files = note-37 full fit of the note-57 recipe: NOT refitted) -> p3 GRU only
  on candidates with tree p >= .01, pair batch 64 -> 0.5 blend -> joint decoder REFITTED on that filtered blend (all labelled
  rows of the cand64 OOF pool, 203 rounds = mean of the six 73b fold fits).
* function: 229-feat trees x 3 seeds REFITTED (all 297,280 note-55 rows, 233 rounds) + siba (two ONNX graphs: pair TCN in pair
  batches of 64, head with sibling attention per piece) -> context stacker (62 cols) REFITTED -> D-lane decode, gate .9, stack
  pick (nonatspm_ap); < 30 min: no lanes, cy twin decode .4.  New modules funcnet, function_stage, pick, stacker, trees_onnx.
* lanes D REFITTED (77,102 labelled pairs, lam 3 = 4/6 fold picks, span prior all print signals), from 30 min only;
  setback = sb7 P50 per length group (m30 / h6 / h24 / full, nearest on a log scale) REFITTED + note-41 P10/P90 band for the
  confidence; health v5 (+ note-56 short-window fix); night speed (note 60) per Advance detector and per phase (new columns
  night_speed_mph / _vehicles). Setback and night speed read the FINAL decoded function (research: the 229-arm argmax).
* All 34 LightGBM models -> ai.onnx.ml opset-5 TreeEnsemble (double): max |diff| vs the text models 2.3e-14 on 30k real frame
  rows each (phase, function, stacker, setback) and on every input captured in the parity runs (decoder, lanes D: 2.2e-14).
* siba ONNX vs the research torch forward (fp32 CPU, package streams, 6 bench signal-lengths, every piece): |dlogp| 3.1e-5,
  |dprob| 1.7e-6 -> PASS 1e-4.
## Function net: one member, and which stacker (member75.py, six folds OOF, gate .9 decode, paired signal bootstrap)
Shipped member = research fold-0 seed-0 siba (trained on folds 1-5) until the GPU full-data refit; swap via
`export75.py siba --members ...` + `check.py --freeze` (1 member -> stacker 'single', >= 2 -> 'mean3', both in weights).
One model instead of the 3-seed mean costs (>= 30 min E, pt vs .9169): stacker as trained -0.20 (seeds -0.26 / -0.11 /
-0.22, 2 of 3 CI < 0); stacker refitted on single-seed inputs ('single', 3x rows) -0.12 (-0.13 / -0.09 / -0.13; R -0.12);
5 / 10 min -0.39 / -0.34 (as trained -0.55 / -0.49). -> "single" adopted. Three members recover it for ~+5 s (busiest 3 h).
## Parity vs the OOF pipeline (parity75.py: 6 non-locked fold-0 signals x m5 / m10 / m30 / h3 / h24 / full66, frame mode)
A, same models + same inputs injected (OOF phase input, OOF trees / siba probs, OOF D pair probs + fold-0 lane settings, a
fold-0 stacker refitted exactly as s74), 741 detector-windows: 229 features 169,689 values identical; lanes / lane conf /
n_lanes identical; pick inputs identical; function-free health context identical except 1 chatter value (.015); stacker
probs <= .0064; FINAL FUNCTION 741 / 741 identical.
B, production models: decoded top phase = OOF on 99.6 % (239 det-windows; 1 miss at 5 min); siba (= its own OOF model) argmax
99.9 %, |dprob| <= .05 (OOF ran bf16 on GPU); tree argmax 93.0 %; final function agrees with OOF on 96.0 % (full refits vs fold models).
## check.py (package): 3 / 3 PASS — references (GRU, siba, 30-min + 10-min short path), renumbered phases, torch / lightgbm /
scipy / sklearn blocked. FINDING (bench75 inv): on 3 bench signals x 30 min / 3 h / 24 h every ANSWER follows a phase
renumbering (top phase, function, lanes, setback, speed, health identical), but probabilities move (phase <= .0055, function
<= 5e-4) on 7 of 9 cases: features.py `excl_partner_diff` breaks a co-green tie by phase order (argmax of equal values).
Same code in shipped final_v2 (model/): there phase <= .031, function <= .015, answers identical. Fix = phase-free tie-break
+ phase-feature refit (pool rebuild) — not done here.
## Speed / RAM per signal (bench75.py: one signal per call, fresh process, 4 threads; warm = median of 3, models kept)
| s warm / cold (peak GB) | 30 min | 3 h | 24 h |
|---|---|---|---|
| typical 19-22 ch | 1.28 / 2.49 (.64) | **3.00** / 3.92 (.60) | 3.79 / 5.16 (.63) |
| typical 17-19 ch | 1.13 / 2.28 (.54) | **2.47** / 3.37 (.57) | 2.73 / 3.82 (.57) |
| busiest by events, 31 ch | 1.76 / 2.77 (.68) | **4.02** / 5.00 (.63) | 5.49 / 6.91 (.72) |
| busiest by channels, 43 ch | 2.20 / 3.18 (.67) | **5.09** / 6.33 (.71) | 6.41 / 7.70 (.79) |
Import 0.5 s extra cold. 3 h busiest: siba 2.5 s (49 %), GRU 1.3, phase features .3, lanes .2, setback .16, rest < .1 each.
Target <= 4-5 s at 3 h: typical met (2.5-3.0), busiest 4.0 / 5.1 (edge). Pair batch 128-512 for siba: no gain.
## Open for the orchestrator
* Single siba member: -0.12 pt (>= 30 min) / ~-0.35 pt (5 / 10 min) vs the OOF champion, for ~2.5 s at busiest 3 h. GPU full
  refit of siba (1 model) then swap members; or 3 members (+~5 s busiest 3 h). The siba candidate filter (note-73 idea 2)
  would cut siba ~-60 % but needs one re-inference.
* Invariance tie-break above (pre-existing; all answers invariant).
* Not measured end-to-end: setback / night speed / health on the package's own answers (code paths = candidate v2's).
