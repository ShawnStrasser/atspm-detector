# 87b — Phase net TCN vs GRU in the current pipeline; one joint phase + function decider (2026-10-04) — locked untouched
Code `research/code/final87/` (p87_phase.py, t87_speed.py, j87_joint.py); work `%DC_WORK%/s87/` (p87_phase.json, t87_speed.json,
joint/j87.json, logs). Saved OOF only, CPU 4 threads, GPU not touched (note 86 running), locked_v2 asserted absent, nothing shipped.
## A. TCN vs GRU as the phase network (note-76 pipeline: f76 trees 3 seeds, 0.5 / 0.5 net on tree p >= .01, decoder OOF refit)
Is the TCN OOF fair? Yes for the swap. GRU p3 and both TCNs (note 53: ad_all_e100 = 1 s, 15 ch; r05_all = 0.5 s, 15 ch) are
six-fold OOF of the same era: note-37 pool (780 signals), same fold map, same timing labels, K = 4 pieces past 2 h, identical pair
coverage (97.2 % of 1,718,334 pool rows, asserted). Nets do not read tree features, so note 76's feature fixes do not concern them.
One gap: the TCN's partner channel (pg / pc) was inferred before note 76's content tie-break (first-candidate tie). Check: the GRU
arm reproduces the champion exactly (.9818; re-decode 0 rows differ). Seed 0 per fold; fold 0 also scored with seed 1.
| >= 30 min unless stated | GRU p3 (champion) | TCN ad_all | TCN r05_all | ad - GRU pt [95 % CI] | r05 - GRU |
|---|---|---|---|---|---|
| E (173,878 det-windows, 757 sig.) | .9818 [.9785,.9848] | .9816 [.9784,.9846] | .9814 [.9784,.9844] | -0.02 [-0.07,+0.04] | -0.03 [-0.09,+0.03] |
| R (166,711) | .9843 [.9814,.9870] | .9841 [.9811,.9869] | .9839 | -0.03 [-0.07,+0.03] | -0.04 [-0.09,+0.02] |
| 5 min E (32,213) | .9648 | .9637 | .9637 | -0.11 [-0.27,+0.05] | -0.11 [-0.30,+0.07] |
| 10 min E (34,294) | .9722 | .9718 | .9712 | -0.04 [-0.18,+0.09] | -0.10 [-0.22,+0.02] |
By length E (GRU / ad / r05; ad - GRU): m30 .9799 / .9798 / .9793 (-0.01); h1 .9814 / .9804 / .9802 (-0.10 [-0.19,-0.01]); h3 .9827 /
.9829 / .9830 (+0.02); h6 .9830 / .9828 / .9826 (-0.02); h24 .9828 / .9829 / .9832 (+0.01); full .9826 / .9830 / .9833 (+0.04).
By fold >= 30 E (GRU / ad): .9791/.9813, .9799/.9790, .9867/.9868, .9804/.9799, .9826/.9829, .9809/.9796 (ad ahead 3, behind 3; fold 5
-0.13 [-0.24,-0.05], fold 0 +0.22 [0.00,+0.51]). Fold 0 with seed-1 TCN: ad -0.01 overall (vs -0.02), r05 -0.03: seed-stable.
Net alone >= 30 E: GRU .9685, ad .9692 (+0.06 [-0.03,+0.16]), r05 .9686; 5 min .9458 / .9475 / .9483 (r05 +0.25 [+0.03,+0.48]).
Decoded top phase differs from the GRU arm on 0.7 % of >= 30-min det-windows (2.4 % at 5 min).
Speed (onnxruntime CPU, 4 threads, pair batch 64, 4 x 30-min pieces, kept pairs only; shape-only timing on random inputs of the
production shape; fold-0 checkpoints exported here, ONNX parity 5.7e-6 / 6.3e-6; raster building not timed):
| 3 h, kept pairs | typical (24 pairs) | 90th pct (59) | busiest (131) |
|---|---|---|---|
| GRU p3 (package gru.onnx) | 0.38 s | 0.91 s | 2.08 s |
| TCN ad_all | 0.13 s (2.9x faster) | 0.37 s (2.5x) | 0.74 s (2.8x) |
| TCN r05_all | 0.30 s (1.3x) | 0.75 s (1.2x) | 1.67 s (1.2x) |
Verdict A: tie (every >= 30-min CI contains 0, |delta| <= 0.03 pt, inside the 0.07-pt tree seed noise). User rule: on a tie take
the faster -> TCN ad_all (1 s bins), ~-0.25 s typical / -1.3 s busiest per 3-h signal. Caveats: 5 min leans GRU (-0.11, n.s.); h1
-0.10 (CI just < 0, one length of eight); single seed per net.
To swap the package's phase net to TCN ad_all (nothing started):
1. GPU (~1.5-2 h): full-data refit of ad_all (780 signals, tcn53.py as it is now = content partner tie-break; 1 s, 15 ch, cap 100
   / patience 7) -> ONNX export (this note's export path, parity check).
2. GPU optional (~1 h, 6 x ~10 min): re-infer the six ad_all fold models with the fixed partner tie-break -> OOF without the leak;
   re-run `p87_phase.py decode/score` to confirm the tie (expected: no change; ties are rare).
3. CPU: package raster gets the 6 extra channels (onE / offE tent impulses, cOn / cOff 43 / 44 impulses, pg / pc partner green /
   call with the funcnet content tie-break) in gru_input.render / assemble; gru_blend unchanged (filter .01, K = 4 of 32, 0.5 / 0.5);
   decoder refit on the TCN-blend OOF (this note's tcn_ad decode = the recipe); check.py --freeze (invariance incl. partner); bench.
## B. One joint decider (iterate once; nested)
Round 1 = current deciders (phase p1 = champion decoder; function f1 = mean3 stacker on mean of 3 filtered siba, oof84 mean3_flt3).
Round 2: phase decoder + 13 columns per (detector, candidate) = own f1 (7) + candidate's soft class mix S_c(k) = sum over other
detectors of p1(d', k) f1(d', c) (A / P / C / Y / non-ATSPM) + expected phase-mates; function stacker + 11 columns = p1 top / margin
/ entropy / n candidates / agrees with frame phase + expected phase-mate class mass sum_k p1(d, k) S_c(k). Nested: for held-out
fold k every round-1 input on the training folds was recomputed without fold k (inner decoder / stacker on folds not in {k, j});
base learners standard OOF as notes 64-84. Recipes reproduce round 1 (stacker 2.98e-8, decoder 0). Control: joint columns shuffled
within window family. 99.6 % of pool rows have a function row; only 62.4 % of function rows have phase-pool rows (rest: NaN).
| delta vs current separate deciders, pt [95 % CI] | >= 30 min E | >= 30 min R | 10 min E | 5 min E |
|---|---|---|---|---|
| phase joint (.9822 [.9791,.9852] vs .9818) | **+0.05 [+0.02,+0.08]** | +0.04 [+0.01,+0.07] | +0.12 [+0.06,+0.18] | +0.07 [0.00,+0.14] |
| phase shuffled control | -0.01 [-0.02,+0.01] | -0.01 | +0.02 | -0.04 |
| function joint (.9188 [.9078,.9286] vs .9192) | -0.05 [-0.09,0.00] | -0.03 [-0.08,+0.01] | -0.09 [-0.18,+0.02] | -0.10 [-0.20,-0.01] |
| function shuffled control | -0.01 [-0.04,+0.01] | -0.01 | +0.02 | +0.01 |
Phase by fold >= 30 E: -0.02, 0.00, +0.05, +0.14, +0.06, +0.03. Function by class >= 30 E: Advance -0.08*, Presence -0.05, Count
-0.07, YR +0.05, non-ATSPM +0.02 (n.s.).
Verdict B: function information helps the phase decoder a little and for real (beats its control, CI > 0, 5 / 6 folds >= 0), but
+0.05 pt is far under the 0.3-pt phase bar and inside tree seed noise -> not adopted (decoder seed 0 only; more seeds not run, they
cannot lift it over the bar). Phase information does not help the function stacker (it already sees the frame phase, phase-mates
and lanes) -> dropped. Keep the separate deciders.
