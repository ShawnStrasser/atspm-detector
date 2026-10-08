# 53 — TCN optimised on its own (validation plan step 3): resolution, channels, tree features, function head (2026-09-30 / 10-01)
Code `research/code/neural/tcn53{,_prep,_eval,_chain,_func,_feval}.py`; work `%DC_WORK%/tcn53/` (eval_f0.json, eval_six.json,
feval_f0.json, logs/). Pool = note 37 (780 signals, phase map), TCN = note 15 (+chans), trained from a local snapshot. Eval = note 37
harness: phase_v3 trees OOF, 0.5 blend BEFORE the decoder; fold 0 = decoder fit on folds 2-5 with the p3 GRU (p3_oof_f0 as
candidate reproduces note 37 fold 0 exactly, .9745 at 30 min); six folds = `run_arm` (reproduces note 37 "same" row). Stage-13
rows (fold 0: 34,552 det-windows). locked_v2 asserted absent everywhere; nothing shipped; `model/` untouched.
## Fold 0, net alone / blend (base and promoted variants = mean of seeds 0+1; others seed 0) — m5 / m30 / h1 / h6
| arm | net alone | blend (before decoder) |
|---|---|---|
| trees alone (phase_v3) | .9239 / .9642 / .9600 / .9771 | - |
| GRU p3 (current net) | .9290 / .9604 / .9648 / .9578 | .9514 / .9745 / .9742 / .9798 |
| base TCN, 1 s, 9 ch (s0 / s1 net m30 .9601 / .9629) | .9253 / .9615 / .9630 / .9607 | .9507 / .9738 / .9729 / .9801 |
| r05 = 0.5 s steps | .9344 / .9650 / .9639 / .9606 | .9525 / .9747 / .9744 / .9807 |
| ad_all = + 6 channels (below) | .9334 / .9666 / .9662 / .9634 | .9503 / .9762 / .9749 / .9833 |
| r05_all = both | **.9407 / .9675** / .9660 / .9663 | .9550 / .9753 / .9735 / .9819 |
| ad_edge / ad_cdrop / ad_partner (s0) | m5 .9286 / .9363 / .9322, m30 .9647 / .9645 / .9645 | m30 .9753 / .9746 / .9761 |
| fx0 / fxA / fxK (hybrid, s0) | .9223/.9602/.9596/.9558 · .9299/.9617/.9614/.9716 · .9219/.9538/.9526/.9625 | .9511/.9740 · .9473/.9686 · .9428/.9670 (m5/m30) |
New channels: onE/offE = detector ON/OFF edge impulses split between the two nearest bin centres by sub-bin position (keeps 0.1-s
timing at 1-s bins); cOn/cOff = 43 placed / 44 dropped impulses; pg/pc = partner phase green / call (partner = candidate whose
green overlaps most in the window, behaviour only). Yellow and red clearance were ALREADY separate channels (ch 3 / 4).
## Six folds, one seed each (1,532,182 rows) — net alone / blend, delta vs base TCN pt [95 % signal-bootstrap CI]
| | m5 | m30 | h1 | h6 | full |
|---|---|---|---|---|---|
| trees alone | .9292 | .9623 | .9623 | .9774 | .9804 |
| base TCN net / blend | .9263 / .9509 | .9614 / .9738 | .9634 / .9733 | .9616 / .9807 | .9323 / .9828 |
| ad_all net Δ | +0.53 [+.33,+.73] | +0.17 [+.02,+.33] | +0.33 [+.19,+.47] | +0.25 [+.06,+.44] | -0.08 |
| ad_all blend Δ (= .9522 / .9745 / .9742 / .9805) | +0.13 [-.01,+.27] | +0.07 [-.03,+.17] | +0.08 | -0.01 | +0.01 |
| r05_all net Δ | +0.64 [+.41,+.87] | +0.27 [+.11,+.43] | +0.24 | +0.32 [+.09,+.58] | -0.70 [-1.11,-.28] |
| r05_all blend Δ (= .9520 / .9740 / .9731 / .9801) | +0.12 [-.03,+.26] | +0.02 [-.07,+.11] | -0.02 | -0.05 | +0.01 |
| GRU p3 net / blend | .9280 / .9527 | .9637 / .9745 | .9647 / .9742 | .9632 / .9808 | .9316 / .9821 |
## Other findings
* **Epoch cap**: ad_all f0 "hit" cap 50 (best ep 43); cap 100, same seed: bit-identical, stopped by patience at 50 -> NOT under-
  trained (its seed-1 re-run skipped: ad_all_s1 had plateaued at ep 18). Six-fold runs used cap 100; all plateaued.
* **Hybrid (c)**: fx0 / fxA / fxK all plateaued (13-19 min, not under-trained). fxA had the best inner-val so far (.958 vs .9525
  base) and its net gains where the features carry whole-window evidence (h6 .9716, full .9720 vs .9176 base) yet its blend is
  WORSE (-0.42 pt m30 [-0.96,-0.02], -0.56 h1): it is less complementary to the trees. Error overlap with the trees at 30 min
  (phi): fxA .52, fxK .51 vs base .44, fx0 .38, ad_all .42. fx0 (fixed windows, no features) = base: the window set is not the cause.
* **Ablation** (fold 0, s0, net m30 / m5 vs base mean .9615 / .9253): drop occ -0.55 / -1.58 (pays); onrate -0.19 / +0.21, g -0.02 /
  +0.62, y +0.02 / +0.18 (noise-sized, single seed). rc, call, og, oc, coord NOT run (stopped by the orchestrator: six folds showed
  no network gain, so they cannot change a decision).
* **0.2 s steps**: r02_all killed - no epoch finished in 25 min (0.5 s: 55 s/epoch), GPU memory saturated at 7.97 GB on the
  8 GB card. Not feasible here without a smaller batch / more memory. 0.5 s costs ~1.7x train, ~1.1x inference.
## Function head on the TCN (fold 0 of folds_v4, note 59 harness: ATSPM stack-aware, D lanes, pick D229; trees fold 0 .8810 E / .8870 R)
fj = joint phase+function loss, ad_all channels, 1 s; function representation = [sum_c p(c) z(d,c), max_c z(d,c)] -> 7 classes.
Training labels = frame v6e target (v3s), 1,031 signal-periods (355 Sept rasters newly built in `neural/sig_stg/`, ncache2, 3 s).
| fj, fold 0 | overall E (R) | m5 | m10 | m30 | h1 | h6 | full |
|---|---|---|---|---|---|---|---|
| net alone | .8543 (.8581), **-2.67 [-4.05,-1.34]** | -4.34 | -4.49 | -1.91 | -1.52 | -1.15 | -4.08 |
| 0.5/0.5 blend with trees | .8886 (.8934), **+0.77 [+0.04,+1.63]** (R +0.63) | +0.58 | +0.38 | +0.84 | +0.98 | +0.92 | +0.76 |
ff (function loss only) trained (inner-val func .80 vs fj .81); its scoring was blocked by a tool-permission denial - score with
`python tcn53_feval.py --cands fj,ff` once `fpreds/ff_f0.parquet` exists.
## Verdict (bars: phase 0.3 pt at 30 min, function 1 pt)
* Network alone, phase: 0.5 s + 6 channels is the best TCN (six folds +0.27 pt at 30 min, +0.64 at 5 min, real; ~= GRU p3 at 30 min,
  +0.47 at 5 min) - under the 30-min bar. In the blend nothing moves (<= 0.07 pt, CIs over 0): the trees + decoder absorb it, as in
  notes 15-21. **Keep the current net (GRU p3); none of a/b/c is promoted.** Hybrid features make the net less useful in the blend.
* Network vs trees alone (six folds): r05_all net .9327 / .9641 vs trees .9292 / .9623 at 5 / 30 min (net ahead by 0.35 / 0.18 pt),
  tied at 1 h, behind past 3 h (h6 .9648 vs .9774, full .9262 vs .9804; K = 4 pieces).
* Function: the net alone is far below the trees (-2.7 pt, worst at 5-10 min, so no short-window rescue); the blend +0.77 pt is
  under the 1-pt bar on one fold / one seed (CI 0.04-1.63); a 2nd seed + six folds would settle it (`%DC_WORK%/cloud_pkg/`,
  2.1 GB, run.sh TAG MODE FOLD SEED). Fold 0 overstated every phase gain (r05_all net m30 +0.60 -> +0.27 six-fold). GPU ~23 h.
