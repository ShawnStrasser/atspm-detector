# 48 — final_v3 CANDIDATE v2: assembled + six-fold OOF evaluation (2026-09-29) — not shipped, locked untouched
Package `%DC_WORK%/final_v3_candidate_v2/` (copy of note 32's candidate; `model/` = final_v2, untouched). Build/bench/card
scripts `%DC_WORK%/final_v3_work/{freeze_refs_v2,bench_v2,update_card_v2,invariance_24h_v2}.py`; setback export
`research/code/trackA/sb6_export.py`; evaluation `research/code/evaluation/eval48_{phase,function}.py` ->
`final_v3_work/eval48/{phase,function}.json`. locked_v2 asserted absent everywhere; nothing run on it.
## Components (every booster numpy at run time; torch / lightgbm / scipy / sklearn blocked in check.py)
| part | shipped model | OOF evidence used here | numpy parity |
|---|---|---|---|
| phase | phase_v3 trees + decoder + GRU ONNX, 780 signals; net at all lengths, K=4 pieces > 120 min | note-37 fold models (phase_v3 OOF) | trees/decoder 0.0; ONNX vs torch 5e-6 |
| function | function_v3d, 7 classes, 442 features | note-45 run `..._valnc_h3` (same recipe, OOF) | 0.0 |
| lanes | note-42 pair model + decode (`lanes.py`) | note 42 (frame-v6 OOF phase / function inputs) | = research `lane_output` exactly (6 signals, 66 h) |
| setback | note-41 quantile LGBM + HGB pair model (`setback.py`) | note 41 (66 h, OOF) | export 0.0; end to end = research `setback()` 0.0 ft |
| health | health_core v5 (`health_core.py`, byte copy) | note 47 | same code |
New columns: `lanes`, `lane_conf`, `phase_n_lanes`, `phase_n_lanes_conf`, `distance_ft`, `setback_confidence`, `health_status`,
`health_score`, `health_reason`, `health_bad_periods` (JSON), `health_watch`; `predict(..., return_phases=True)` / `--out-phases`
= per-phase n_lanes table. Answer columns (incl. lanes / setback) blank where the answer is withheld; health always given.
Removed: `health_flag` with `controller_fault_events` and the old advisory rules (`health.py` is now only the gate: no
actuations / stuck ON / chatter); fault events 83-88 dropped from `ALLOWED_EVENTS`. **watch**: rolling function-consistency
NOT computed (one extra function pass per 2 h of data: ~+6 s/signal at 24 h, AUC .58-.65 in note 46); `health_watch` = the
informational notes only (short ONs above limit, unsure classifier). Bug found: research `sb5_setback.setback()` took
quiet hours from `hour(to_timestamp())` = session time zone (PDT here): fixed there and in the package (note-41 OOF unaffected).
**check.py** re-frozen, 3/3 PASS; new assertions: stop-bar = 0 ft, Other/Bike no setback, no lane for Bike, highest lane =
phase n_lanes, health in {ok, suspect, bad, not_enough_data}; renumbering leaves lanes / n_lanes / setback / health identical
(also on 3 signals x 24 h: all 11 new columns + phase table identical). Weights 40 MB (32).
## Runtime, 20 non-locked signals, one predict(), 4 threads, numpy backend (s/signal; GRU; lanes / setback / health; peak RAM)
30 min 0.82 (0.51; .09/.02/.03) 1.56 GB · 6 h 2.48 (2.04; .10/.03/.08) 1.64 GB · 24 h 3.08 (2.15; .11/.05/.26) 1.86 GB.
Candidate v1 (note 32, lightgbm backend): 0.57 / 2.13 / 2.45 s, 1.63 / 1.70 / 1.92 GB; v2 with lightgbm 0.83 / 2.51 / 3.50 (loaded box).
## Phase, six folds OOF (note-37 harness; final_v2 = final_v1 trees + stage-13 GRU <= 1 h, trees only from 3 h)
everything = timing label, >= 5 actuations (240,308 det-windows). realistic = minus label-check fail/misconfigured (9,366) and
high-confidence print phase != timing (634), switch/additional call phase credited (529 detectors): 230,308.
| window (n everything) | everything cand / v2, Δ pt [95 % CI] | realistic cand / v2, Δ pt [95 % CI] |
|---|---|---|
| 5 min (32,207) | .9653 / .9645, +0.05 [-0.09, +0.19] | .9687 / .9683, +0.02 [-0.12, +0.16] |
| 30 min (47,247) | .9798 / .9801, -0.03 [-0.11, +0.05] | .9827 / .9830, -0.04 [-0.12, +0.04] |
| 6 h (26,763) | **.9831 / .9792, +0.39 [+0.23, +0.58]** | **.9856 / .9819, +0.37 [+0.20, +0.56]** |
| full 66-72 h (13,787) | .9829 / .9816, +0.13 [-0.07, +0.34] | .9849 / .9839, +0.10 [-0.09, +0.31] |
3 h +0.43 [+0.25, +0.64], 24 h +0.14 [-0.07, +0.36] (everything). The gain is the network past 2 h (note 33); the 780-signal
refit is neutral (note 37). By health v5 (Sept 2026 rows, everything), cand / v2: 30 min ok .9857/.9857 (21,706), suspect
.9731/.9797 (1,525), bad .9678/.9678 (435); 6 h ok .9909/.9857, suspect .9855/.9904 (830), bad .9751/.9680 (281).
## Function, six folds OOF (frame v6; phase input = the frame's OOF phase, not phase_v3; v2 head = note-25 b7, its rows)
| set | candidate, all rows: acc7 / acc5 / core-four | b7 rows: v2 -> cand acc7, Δ [CI] | acc5 Δ | core-four Δ | 30 min / 6 h / full Δ |
|---|---|---|---|---|---|
| everything (268,861 rows, 652 sig.) | .8714 / .8759 / .9015 | .8423 -> .8693, **+2.70 [+2.10, +3.33]** | +2.63 [+2.06, +3.26] | +1.34 [+0.79, +1.87] | +2.57 / +2.67 / +2.78 |
| realistic (261,622, 651) | .8834 / .8879 / .9157 | .8525 -> .8812, **+2.87 [+2.30, +3.53]** | +2.79 [+2.22, +3.41] | +1.50 [+0.95, +2.07] | +2.74 / +2.81 / +2.98 |
b7 rows = 204,889 / 199,542 on 383 signals. acc5 = Mid, Bike -> Other on both sides (final_v2 ships 5 classes). By health v5
(Sept, everything) acc7: ok .883 (164,651) / suspect .849 (8,004) / bad .739 (1,995); b7 .851 / .843 / .709.
## Lanes, setback, health (from their notes; inputs there = research OOF predictions, not this package's own)
* Lanes (note 42): n_lanes exact .80 / .87 / .88 / .88 at 30 min / 6 h / 24 h / full, within ±1 .99; detector lane set exact
  .88-.93; same-lane precision / recall .96 / .94 (full). Lane 1 = busiest lane, not geometry.
* Setback (note 41, 66 h): medAE 20 ft, 71 % within ±25 %, 72 % within 50 ft (579 printed Advance; loop 17 ft, video 23,
  radar 47, > 200 ft 55); production path 23 ft / 66 %; answered 96.5 %; high confidence 3 ft. Agency-specific.
* Health (note 47, v5): presumed-healthy flagged 1.0 / 1.2 / 2.7 / 4.8 % at 2 h / 6 h / 24 h / 66 h; known problems caught
  27 / 29 / 64 / 99 % (dq set); user spot-check answers 22/22.
## Open
* Locked confirmation not run (the stop rule's one run). Suspect-health rows lose ~0.5-0.7 pt phase vs v2 (small n, no CI): watch.
* Setback trained and measured at 66 h only; at 30 min most answers carry low confidence - not measured below 66 h.
* No end-to-end OOF of lanes / setback / health on the candidate's own phase_v3 / function_v3d answers (would need fold
  packages); function OOF uses the stage-12 phase input; phase fold map vs folds_v4 differ on 22 signals (note 37).
* Production has no channel list: a detector silent all window is not in the log, so `dead` cannot be reported.
* health_core copied at md5 bbef7b2f; if the health agent changes it, re-copy and re-freeze check.py.
