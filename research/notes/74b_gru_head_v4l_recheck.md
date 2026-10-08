# 74b — GRU backbone of the final function head; steps 1-3 re-checked on v4l labels; add-ons (2026-10-03)
Continuation of note 74 (same queue, scorer `evaluation/s74.py`, champion = context stacker 3 LightGBM seeds + gate .9, paired
signal bootstrap, >= 30-min headline). locked_v2 absent (asserted).
## Labels changed mid-run
Note 81 adopted label set v4l (scoring truth + stacker training labels) on 2026-10-03, between steps 3 and 4. Steps 1-3 in
note 74 were scored on v3s (187,980 E rows >= 30 min); step 4 is on v4l (187,444). Every v4l arm below has its stacker re-fit on
v4l. Stackers from different label sets are never compared with each other.
## Steps 1-3 re-checked on v4l (three seeds each; vs siba stacker .9187 E / .9287 R)
| arm | >= 30 E | >= 30 R | 5 min E | 10 min E |
|---|---|---|---|---|
| sibm | -0.02 [-0.15,+0.11] | -0.03 [-0.15,+0.10] | **-0.27 [-0.48,-0.05]** | -0.19 [-0.40,+0.02] |
| siba + cw2 | -0.03 [-0.14,+0.08] | -0.04 [-0.15,+0.07] | -0.12 [-0.31,+0.06] | -0.10 [-0.30,+0.08] |
| specialist decider | +0.04 [-0.01,+0.10] | +0.05 [-0.01,+0.10] | +0.10 [-0.02,+0.22] | +0.02 [-0.10,+0.14] |
Same verdicts as on v3s: sibm and cw2 dropped. The specialist decider is not adopted: its CI includes 0 at >= 30 min on both
sets, the single-seed control in note 74 says the small gain is seed diversity, and it needs 3x the network inference.
## 4. GRU backbone (p3 GRUAttnD, fp32, grad-accum 8) of the siba head, six folds x seed 0, vs the TCN siba seed 0
Both single-seed stackers re-fit on v4l (`stack --name siba0n --main x69_siba:0` / `--name gru0 --main x74_sibagru:0`).
| pool | TCN siba0 | GRU | GRU - TCN [95 % CI] |
|---|---|---|---|
| >= 30 min E | .9176 | .9159 | -0.17 [-0.34,0.00] |
| >= 30 min R | .9275 | .9261 | -0.15 [-0.30,+0.02] |
| 5 / 10 min E | .8910 / .9007 | .8872 / .8966 | **-0.39 [-0.64,-0.13] / -0.41 [-0.68,-0.15]** |
| all windows E | .9115 | .9092 | -0.23 [-0.41,-0.06] |
By class >= 30 E: Advance -0.02, Presence -0.21, Count **-0.36 [-0.72,-0.05]**, YR -0.11, nonATSPM -0.21. Folds +0.06 / -0.10 /
-0.22 / -0.02 / -0.44 / -0.17 (5 of 6 negative).
CPU inference per signal (torch fp32, 4 threads, warm, fold-0 nets, `neural/time74.py`): at 3 h, typical (18 det, 4 cand) GRU
.84 s vs TCN .63 s; busiest (44 det, 6 cand) 2.85 vs 2.62 s; at 30 min .26 vs .15 / .78 vs .59 s. Training on the A1000: GRU
needs grad-accum 8 to fit 8 GB, ~110-195 s/epoch (2.1-2.9 h per fold) vs ~70 s (~66 min) for the TCN.
Verdict: **keep the TCN backbone**. The GRU is worse (beyond noise on short samples and on Count), slower to run on CPU, and
about 2.5x slower to train. This closes the AGENTS.md end-of-search "re-fit with a GRU backbone and compare" check for the
function head. One seed only, as the orchestrator specified.
## Add-on A: full-data siba refit (the single production network for final_v3_candidate_v3)
Seed 0, every function-table signal-period (no fold, no inner-val; locked_v2 asserted absent), 43 epochs (median best epoch 42
of the six siba seed-0 fold runs + 1), lr per epoch = median of those runs' replayed ReduceLROnPlateau schedules, grad-accum 2,
final weights, post-note-76 inputs (snap74b). Two twins: `tcn53/models/x74_sibafull_full.pt` (v3s labels = the fold runs'
`func_rows.parquet`, 1,031 signal-periods, final loss .450) and **`x74_sibafull4l_full.pt`** (v4l labels, `func_rows_v4l.parquet`
from `neural/flab74_v4l.py`, 1,027, loss .430; v4l vs v3s: 272 rows relabelled, +3,314 / -1,881 usable rows, 329 detectors).
Recommended: the v4l twin (matches the v4l stacker). ONNX `%DC_WORK%/s74/full/*.onnx` (`neural/export74.py`), parity 1.1e-5 PASS.
No held-out score exists for a full refit; the fold runs with the same recipe are its estimate.
Orchestrator 2026-10-04: two more v4l members, seeds 1 / 2, same recipe: `x74_sibafull4l_s1_full.pt` / `_s2_full.pt` (final loss
.447 / .445 at epoch 42), ONNX parity 1.3e-5 / 1.1e-5 PASS -> 3 members available for the 3-member + filter speed test.
## Add-on B: siba candidate filter (pair net only on candidates with OOF tree p >= .01), seed-0 fold models, inference only
Reference = the same six models without the filter on the same post-note-76 inputs (x74_sibanf); single-model stackers on v4l.
>= 30 min -0.01 [-0.07,+0.05] E / -0.02 [-0.07,+0.04] R; 5 / 10 min 0.00 / +0.02 E; all windows -0.00. By class >= 30 E: Advance
+0.02, Presence -0.01, Count +0.04, YR +0.10, nonATSPM -0.16 [-0.37,+0.02]; folds -0.09 .. +0.12. Argmax of the net alone changes
on 3.8 % of fold-0 rows; the stacker absorbs it. -> **accuracy-neutral**, adopt with the GRU filter (note 73b).
Coverage caveat: only 59 % of detector-pieces had a tree row in `cand64/phase_oof` (the rest kept every candidate), so 58 % of
pairs ran here; where tree rows exist the filter keeps 32 % of pairs (27 % at 3 h, 41 % at 5 min). Speed: pair-net time scales
with pairs (bench, torch CPU, typical 3 h: 1.28 -> .55 s at 44 % of pairs); expected in the package at 3 h ~-70 % of the siba
ONNX time (note 73: 1.0 s typical / 1.9 s busiest -> ~-0.7 / ~-1.3 s).
Side result: old vs post-note-76 inputs, same models (siba0 vs x74_sibanf): >= 30 min -0.00 [-0.02,+0.01], 5 min -0.05 [-0.09,-0.01].
