# 77 — Channel-order-free lanes D, pick inputs, decodes and setback; refits; channel invariance in check.py (2026-10-02)
Binding rule: no result may depend on a channel number or channel order. Note 76 left lanes D / pick channel-order
dependent (lane_conf <= .15, function prob <= .04 under reversal). Code `research/code/final77/`; work
`%DC_WORK%/final_v3_work/f77/`, refits `v3fit77/`. CPU <= 6 workers; q74 GPU queue untouched; locked_v2 absent.
## Leaks found and fixes (research + package code identical in the shared parts)
* Lanes D pair model: cues / context are not symmetric in (a, b) (log_na / log_nb, quiet weight of a's ONs, lag-bin
  edges at exact half seconds, fwd/bwd tie) and a = the LOWER channel. Reversed cues differ on 488,170 / 516,915 pairs;
  OOF orientation gap mean .011, p99 .15. FIX: train on both orientations, P(same) = mean of the two.
* Lanes decode (lane_output / lanes.py): detectors in channel order -> coordinate ascent, non-anchor argmax, linkage
  and top-4 partition ties depended on it. FIX: canonical behavioural order `content_key` = (actuations, sum and a
  hash of the ON offsets in 0.1 s), busiest first; stable top-4 sort. Ties only for identical ON series.
* Pick inputs (ln6 span, ln7 stack health, package pick.py): covered peers greedily kept in -M order, ties (often
  M = 1.0) by channel. FIX: every per-phase loop in canonical order (`order_dets`). Same for short-sample twin pairs
  (orientation -> |lag| asymmetric: 82,485 of 227,498 pairs had a different |lag|).
* Per-lane / twin decodes (atspm_decode, s62): stable flat argsort / first-of-ties = row order. FIX: ties by a content
  rank of the probability row (`_pair_order`, `_crank`) in decode_group, pick order, clusters, exact, twin_decode.
* Setback (found by the battery): same-lane pair model scored on (min ch, max ch) only, partner pick sort ties ->
  FIX: both orientations averaged, complete stable tie-break keys. Night speed: partner ONs sorted by unstable argsort
  -> lexsort (ON, OFF). health_core `_partner`: first of equal correlations -> tie by the sibling's own counts.
  health_reason names a partner ("relative to d12"): equivariant text, mapped back in the tests.
* Not changed (by design, note 76): phase decoder's channel ADJACENCY uses channel differences, so only renumberings
  that keep |differences| (reversal, shift) are invariances; an arbitrary permutation is not.
## OOF rebuild (six folds folds_v4, 3 seeds; inputs as note 58 / 76: frame v6e phase, note-57 arm for lanes / pick)
* Lanes D (both orientations): pair AUC >= 2-lane m30 / h6 / h24 / full .9630 -> .9635 / .9751 -> .9758 / .9779 -> .9786
  / .9785 -> .9790 (seed spread <= .0004). Same lam picks per fold. Print-lane eval (Sept scope, CI pt): n_lanes exact
  .8540 -> .8538 (-0.02 [-0.14,+0.09]), lane set .9215 -> .9224 (+0.08 [+0.01,+0.16]), pair acc multi .9233 -> .9237
  (+0.05 [-0.07,+0.16]). Full decode (304,432 det-windows): lanes string changed 2.0 %, lane_conf |d| > .001 42 %,
  gate (.9) side flips 2.2 %.
* Pick inputs: span flag 23 / 304,432 changed, span peers 111, track 2,015 (lanes changed); ln7 chi / clusters 0; twin
  pairs (thr .4): 389 lost / 1,079 gained of ~19k. Function-free stack health (stacker context): unchanged.
| inside the champion (context stacker 3 seeds, gate .9) | note-76 champion f76c | note 77 | delta pt [95 % CI] |
|---|---|---|---|
| function >= 30 min E (187,980 rows) | .9170 | .9171 | +0.012 [-0.015,+0.041] |
| function >= 30 min R | .9281 | .9282 | +0.009 [-0.018,+0.037] |
| 5 / 10 min E | .8916 / .9009 | .8916 / .9007 | +0.003 [-0.025,+0.033] / -0.016 [-0.043,+0.013] |
Per class >= 30 E: Advance -0.02, Presence +0.02, Count +0.03, YR +0.01, non-ATSPM +0.03 (all CI cover 0); per fold
-.09..+.05; new stack per seed .9172 / .9169 / .9168. NEUTRAL -> ADOPTED (enforces the binding rule).
* Setback sb7 'all' P50 on rebuilt features (p_same changed on 10,189 / 42,287 rows, chosen tau 292): printed Advance
  within 50 ft m30 / h6 / h24 / full 68.9 / 69.3 / 70.3 / 70.2 -> 69.1 / 68.5 / 70.4 / 70.2; pooled w50 -0.1 [-0.6,+0.4],
  medAE -0.2 [-1.0,+0.5] ft (seed-to-seed spread larger). Neutral. q10 / q90 band models (note 41) not refit.
* Not rebuilt: health_core_ff (status / scores of the stacker context): only exact-equal partner correlations change.
## Package `%DC_WORK%/final_v3_candidate_v3` (pre-fix copy `final_v3_candidate_v3_pre77`; not shipped, model/ = final_v2)
* Code: lanes.py, pick.py, function_stage.py, atspm_decode.py, setback.py, night_speed.py, health_core.py. Refits
  (fit77.py): lanes D both orientations (154,204 rows, lam 3), setback P50 per group, stacker mean3 + single; phase,
  decoder, function trees, siba, GRU unchanged (v3fit76). assemble77 -> ONNX (parity <= 2.4e-14); references re-frozen.
* check.py: new check 3 "channel-order invariance": reversal, shift, reversal + shift on 30 min + reversal on the
  10-min path; every candidate probability and EVERY output column (detector-name text mapped back), tol 1e-6.
  4 / 4 PASS (channel max |diff| 0.0). The same check FAILS on the pre-77 package (function_prob, reversal).
* Battery (diag77, 6 bench signals x 30 min / 3 h / 24 h x reversal / shift / both + 2 phase renumberings + 10-min
  path, 102 runs): 100 runs nothing moved; 2 = decoder logit0 2.6e-6 at p0 near 0 / 1 (note 76 float noise), outputs
  identical. Before the fixes: lane_conf <= .15, function prob <= .024, setback 6 ft, night speed 1.2 mph moved.
## Speed (bench75 recipe, 3 h, fresh process, 4 threads; after notes 76 + 77)
| 3 h | warm s | cold s | peak MB |
|---|---|---|---|
| typical r8 (22 ch) / r11 (18 ch) | 3.18 / 2.40 | 4.27 / 3.34 | 595 / 545 |
| busiest by events (31 ch) / channels (43 ch) | 4.10 / 4.96 | 5.43 / 6.33 | 615 / 699 |
= note 75 within noise (3.00 / 2.47, 4.02 / 5.09). Busiest 43 ch: siba 2.5 s, GRU 1.2, lanes D .19 (2 orientations).
## Open
* model/ (final_v2) keeps its channel / phase tie-breaks until the candidate replaces it.
* GPU refit of siba unchanged from note 76 (use research tcn53.py).
