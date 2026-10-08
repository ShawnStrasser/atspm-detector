# 86 — siba fold nets on v4l -> mean3 stacker refit (candidate v4e); phase TCN full refit; v4o siba refits (2026-10-04/05)
Brief: the v4b stacker learned on OOF of x69_siba fold nets trained on OLD v3s labels (cloud, pre-76 inputs); make it match
production. Add-ons (orchestrator): phase TCN full refit, ad_all fold re-inference, v4o siba refits. Local A1000, detached queue
`tcn53/q74/queue74d.ps1` (v4: PHASEFULL / PHASEINFER lines; snapshots snap74b, snap86); CPU 4 threads; locked_v2 asserted absent.
## 1. Fold nets x86_siba4l (six folds x seeds 0/1/2; code research/code/final86/)
note-69 siba recipe, `--frows func_rows_v4l` (same rows / folds as v3s), `--accum 2`, snap74b (post-76 inputs); each run's own
inference FILTERED (tree p >= .01; keep masks = x74_sibaflt). 18/18 ok first try, 06:02 -> 03:21, 35-76 epochs, ~80 min a run.
Net alone (argmax, labelled >= 30 min, 3-seed mean): 86.64 vs old filtered 87.01 (-0.37), mostly fold 0 (83.4 vs 85.3; seed-0
fold 0 alone 77.2 = one weak run); 5 / 10 min 82.99 / 84.43 vs 83.37 / 84.89.
## 2. Stacker (oof86.py; after note 88: v4m trees OOF, truth v4m = v4l, label pointers pinned to v4m in-process)
mean3 recipe (47 cols, PRM0 150 rounds, seeds 0/1/2, fold k never seen), trained AND applied on the filtered 3-seed-mean OOF.
Check: old recipe on old nets reproduces v4d (f88 P_v4m) exactly.
| gate .9 | v4d (package) | n86 (new) | n86 - v4d [95 % CI] | n86 - champion |
|---|---|---|---|---|
| >= 30 min E (187,444) | .9193 [.9082,.9293] | .9193 [.9082,.9290] | -0.00 [-0.11,+0.11] | +0.03 [-0.10,+0.17] |
| >= 30 min R (182,828) | .9292 | .9291 [.9184,.9385] | -0.01 [-0.11,+0.09] | +0.03 |
| 10 min E / R | .9032 / .9140 | .9024 / .9131 | -0.08 [-0.26,+0.11] / -0.09 | -0.06 |
| 5 min E / R | .8937 / .9043 | .8946 / .9052 | +0.09 [-0.12,+0.30] / +0.09 | +0.10 |
By class >= 30 E: Advance -0.09 [-0.27,+0.09], Presence -0.01, Count **+0.24 [+0.02,+0.50]**, YR +0.32 [-0.13,+0.74], non-ATSPM
-0.17 [-0.52,+0.16]. Folds -0.22 / +0.31 / +0.05 / +0.09 / -0.18 / -0.10. Side result (old nets): stacker trained on FILTERED
instead of unfiltered OOF +0.04 [+0.00,+0.08]. Interims, new / old nets: seed 0 -0.16 / -0.13 (vs v4b), seeds 0+1 -0.04 / +0.00.
Verdict: **neutral** -> package (rule "wins or neutral"). The gain is consistency (train = production labels), not accuracy.
## 3. Package `%DC_WORK%/final_v3_candidate_v4e` = v4d + new mean3 stacker (v4d untouched)
`oof86.py fit` (268,766 rows) -> `assemble86.py copy / stack / card`; ONNX vs text 8.3e-16; networks (3 x x74_sibafull4l), trees,
lanes, setback, 'single' fallback = v4d. check.py --freeze, then 4/4 PASS (stored 0.0, renumbering 1.6e-7, channel order 0.0,
torch / lightgbm / scipy / sklearn blocked); 17/17 sample answers = v4d. Speed 3 h warm, paired, busy machine (bench86, 2 passes):
v4d / v4e typical 3.10-3.23 / 3.22-3.24 and 2.84-2.88 / 2.74-2.80 s, busiest 4.39-4.59 / 4.33-4.40 and 3.68 / 3.73-3.85 s, peak
<= 907 / 908 MB -> unchanged.
## 4. Phase TCN full refit (replaces the GRU, note 87b: tie -> faster)
`tcn53.py train --full` (new knob, defaults unchanged; + `infer --otag`): ad_all (1 s, 15 ch, seed 0), every training signal of
the 780-pool, no inner-val, 31 epochs (median best epoch 30 of the six ad_all_e100 fold runs + 1), lr = per-epoch median of their
replayed schedules made non-increasing (two short runs made the raw median rise at ep 25), final weights, snap86; 21 min.
`tcn53/models/ad_all76_full.pt` -> `s86/phase/ad_all76_full.onnx` (export86_phase.py; x [pairs, 15, T] -> logit). Parity vs
torch fp32: random 1.1e-5; real pairs (6 signals x m5 / m30 / h3 piece, 1,797) logit 7.6e-6, log-prob 9.5e-6 -> PASS.
## 5. ad_all fold models re-inferred with the note-76 tcn53.py (note 87b check)
`ad_all_e100n76` f0-5 -> `final87/p87_phase.py decode --arms tcn_ad76; score` (arm added; 87b json kept as p87_phase_note87b.json).
>= 30 min E: GRU .9818, TCN .9816 on old and new inputs, new - GRU **-0.02 [-0.07,+0.04]** (R -0.02); 5 min -0.12 [-0.28,+0.05];
10 min -0.03; h1 -0.09 [-0.18,0.00]; full +0.06; fold 5 -0.11*. Net alone .9692 (= old). Tie holds; TCN stays the pick.
## 6. Final production function networks: full-data siba on v4o
`FULL x86_sibafull4o{,_s1,_s2,_s3}`, x74_sibafull4l recipe (43 epochs, replayed lr, snap74b, --accum 2), --frows func_rows_v4o
(1,089 signal-periods; ok .785 of rows vs .655 v4l), ~61 min each. ONNX `s74/full/x86_sibafull4o*_full.onnx` (export74.py --check,
6 real pieces, random filter on half): flogp 2.9e-5 / 9.5e-6 / 3.6e-5 / 1.4e-5 -> PASS. Final train loss .560 / .556 / **.653** /
.586 (seed 2 ~0.09 higher all run; v4l twins .430 / .427 / .447) -> seed 3 added on the orchestrator's word; the package agent
picks three by member agreement. No held-out estimate exists for a full refit.
## Open for the orchestrator
* v4e is neutral (consistency only); its 'single' fallback still learned on x69_siba OOF. Stacker OOF = v4l fold nets, final
  members = v4o: a v4o fold re-run (~24 h GPU) would close that gap; this note says such changes stay inside noise.
* Optional unfiltered x86 infers: only f0 ran; the rest are commented out (#OFF) in q74/queue74.txt.
