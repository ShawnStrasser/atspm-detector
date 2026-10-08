# 76 — Phase-order-free (and channel-order-free) features; refits; invariance on probabilities (2026-10-02)
Binding rule: the model never sees a phase / channel number, so no tie may be broken by one. Code `research/code/final76/`,
research copies of the fixed model modules in `research/code/phasefree/` (rpath puts them BEFORE model/; model/ = final_v2
untouched); work `%DC_WORK%/final_v3_work/f76/`, refits `v3fit76/`. CPU <= 6 threads; q74 GPU queue untouched; locked_v2 absent.
## Leaks found (diag76.py: every intermediate of the package diffed under a phase renumbering / channel-order reversal)
* features.py `excl_partner_diff`: argmax of co-green time -> first phase on ties (ties: 0 s co-green = no concurrent
  phase). FIX: mean over all tied partners. features_partner.py partner: (co-green, green time) tie -> first phase.
  FIX: every partner quantity / `__pdiff` averaged over the tied partners (`partner_set` key column).
* Cross-candidate companions (`__rank/__z/__mgap/__argmax`): values equal in exact arithmetic differ by ulps with DuckDB's
  summation order (which moves with the numbering) -> rank / argmax flip (release_frac). FIX: computed on the float32 value.
* funcnet (siba) partner channel: argmax of green overlap -> first candidate. FIX: overlap ties broken by the partner's own
  green / call traces (content); tied identical traces give identical inputs. Same in research `neural/tcn53.py`.
* Channel order: function.py lag 'best' / 'twin' neighbour (sort by value, then CHANNEL) -> FIX mean over tied rows; SQL_LAG
  top-12 cut (row_number by channel) and similarity top-8 -> FIX keep every tie at the cut; features_expert px '*_at_best_*'
  (DuckDB arg_max, arbitrary on ties) -> FIX mean over tied neighbours (also research trackA/a2_features.py).
## How much moved on the training data (f76/changes.json, lag76.json)
* Phase pool (1,718,334 rows): excl_partner_diff changed on 393,061 rows (22.9 %; 48.5 % of detector-windows; 514,693
  rows sit on a tie). Partner ties 68,391 rows: partner meta / pex unchanged (tied partners identical), pdiff changed
  (call43_b0 36,104, call43_fwd_lift 15,710, call43_red_lift 11,986, rest <= 353). Companions: excl_partner_diff 30-660k
  each; float32 rule elsewhere: release_frac__rank 25,425, queue_occ_pre_green__rank 16,156, others <= 2,112.
* Function frame v6e (456,033 rows): excl_partner_diff changed 24.0 %. Lag columns (tables rebuilt with the tie-keeping
  cut, recipe check = stored frame exactly): rows with any lag column changed 23.7 % (dec) / 23.9 % (stg).
* Channel-number tie-breaks that decided a value: lag best neighbour 10,902 / 168,989 Dec det-windows (6.5 %), twin 4,053
  (2.4 %); top-K cut inside a tie on 42 / 462 bench detector-samples (lag; mostly 30 min), 3 / 441 (similarity).
## Refit (note-57 recipes, six folds folds_v4, 3 seeds) inside the current champion; paired signal bootstrap
| >= 30 min unless stated | current | new | delta pt [95 % CI] |
|---|---|---|---|
| phase E (trees + GRU .01 filter + decoder) | .9818 | .9818 | -0.004 [-0.035,+0.025] |
| phase R | .9843 | .9843 | +0.005 [-0.024,+0.032] |
| phase 5 / 10 min E | .9647 / .9726 | .9648 / .9722 | +0.009 [-0.070,+0.083] / -0.038 [-0.104,+0.029] |
| phase trees alone E | .9755 | .9757 | +0.019 [-0.015,+0.052] |
| function E, arm a (phase-order fixes only) | .9169 | .9167 | -0.018 [-0.052,+0.015] (10 min -0.104 [-0.185,-0.029]) |
| **function E, arm c (+ channel-order fixes)** | .9169 | .9170 | **+0.011 [-0.025,+0.047]** |
| function R, arm c | .9281 | .9281 | +0.002 [-0.035,+0.039] |
| function 5 / 10 min E, arm c | .8921 / .9009 | .8916 / .9009 | -0.058 [-0.142,+0.019] / +0.003 [-0.093,+0.094] |
Function = 229 trees refit -> siba (unchanged OOF) -> context stacker 3 seeds -> gate .9 decode (s74 compare). Held fixed as
in note 57: the frame's predicted phase (top phase moves on 0.34 % of >= 30-min det-windows), lanes D / pick / health inputs,
the decoder's similarity tables (old top-8 cut, ties < 1 %). Every delta is inside seed noise (trees ~0.07 pt): NEUTRAL.
Adopted arm c (enforces the binding rule). siba is NOT retrained (GPU): its partner fix is inference-only (ties rare).
## Package (`%DC_WORK%/final_v3_candidate_v3`; pre-fix copy kept as `final_v3_candidate_v3_pre76`)
* Code fixes above; full-data refits (fit76.py, rounds = mean best iteration of the f76 folds): phase ranker 3 seeds (579),
  decoder (245), function 229 x 3 (arm c frame), lanes D (new features + OOF probs), stacker mean3 + single; setback copied.
  assemble76.py -> ONNX (parity <= 2e-14); references re-frozen (`check.py --freeze`, deliberate model change).
* check.py: the invariance test now compares EVERY probability (each (detector, candidate) p0 and decoded prob, all output
  probability columns) for 3 renumberings of the 30-min sample + 1 of the 10-min path, tol 1e-6; 3 / 3 PASS (max |diff|
  1.6e-7). The same test FAILS on the pre-76 package (candidate p0 moved 1.85e-3) -> it detects the old leak.
* diag76 battery (6 bench signals x 30 min / 3 h / 24 h x 2 renumberings + 10 min): 42 / 42 renumbered runs identical: every phase feature,
  tree / GRU / decoder probability, function feature, Pt / Ps and output (max: siba 4.6e-7 batch-order noise; decoder logit0
  2.6e-6 at p0 near 0/1). Before the fix: busiest 3 h p0 1.7e-3, prob 3.3e-4; typical_r5 24 h / r8 3 h moved too.
* Channel-order reversal (|channel differences| kept, order reversed; 18 runs): phase features / probabilities identical;
  answers identical; remaining order dependence in lanes D decode + pick peers (lane_conf up to .15, function probability up
  to .04 at 30 min) -> channel-order in lanes.py / pick.py greedy steps NOT fixed here (open, below).
## Open
* model/ (final_v2) still has the excl_partner_diff tie-break (note 75: probabilities <= .031 / .015, answers invariant);
  fixed when the candidate replaces it. GPU refit of siba should use research tcn53.py (content partner tie-break).
* lanes D decode / pick: channel-order dependence of lane_conf (no answer change seen); a later lane task.
