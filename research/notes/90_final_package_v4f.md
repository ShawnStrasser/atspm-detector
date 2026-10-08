# 90 — FINAL package v4f: phase net -> TCN ad_all, function refits on v4o (2026-10-05) — not shipped, locked untouched
Package `%DC_WORK%/final_v3_candidate_v4f` (copy of v4e; v4e untouched; `model/` = final_v2). Code `research/code/final90/`
(p90_phase, oof90, assemble90, parity90, siba90, agree90, bench90); work `%DC_WORK%/s90/`, `.../final_v3_work/{f90,v3fit90}/`. CPU <= 6 threads, GPU queue not touched, locked_v2 asserted absent.
## Phase: TCN ad_all replaces the GRU (note 87b tie, user rule tie -> faster)
* Network = `s86/phase/ad_all76_full.onnx` (note-86 full-data refit, 31 epochs; ONNX vs torch 1.1e-5) -> weights/phase_tcn.onnx;
  gru.onnx removed; blend.json `weights_file` names it. Unchanged: filter tree p >= .01, 0.5 / 0.5 before the decoder, K = 4 pieces
  past 2 h, mean log-prob pooling.
* Raster: gru_onnx reads the graph's input width; 15 channels = funcnet.render / assemble (identical code to tcn53 render53 /
  assemble53 incl. the content partner tie-break), memoised per piece on the stream bundle -> built once, shared with siba.
* Parity on real signals (parity90: 4 bench signals x 30 min / 3 h / 24 h, package streams from raw events, 7,861 pair-pieces):
  raster vs research render53+assemble53 **0.0** (every pair / channel / bin); logits torch vs ONNX 6.5e-6; pooled probs 1.1e-6,
  with a random candidate filter 9.6e-7 -> PASS.
* Decoder refit on the trees + TCN blend, six-fold OOF of the ad_all_e100 fold models. First on the note-53 OOF (reproduces 87b
  exactly, iterations 112..346 -> 207), then on the note-86 re-inference with the note-76 tcn53.py (`ad_all_e100n76`; iterations
  [108, 242, 349, 221, 107, 266] -> **216 trees**, the shipped decoder; ONNX vs text 4.4e-16).
| phase, six-fold OOF (757 signals) | GRU p3 (v4e) | TCN ad76 (v4f) | v4f - v4e pt [95 % CI] |
|---|---|---|---|
| >= 30 min E (173,878) | .9818 [.9785,.9848] | .9816 [.9784,.9846] | -0.02 [-0.07,+0.04] |
| >= 30 min R | .9843 [.9814,.9870] | .9841 [.9811,.9869] | -0.02 [-0.07,+0.03] |
| 10 min E | .9722 | .9719 | -0.03 [-0.16,+0.10] |
| 5 min E | .9648 | .9637 | -0.12 [-0.28,+0.05] |
Note-53 vs re-inferred OOF: same to 0.0001 (partner ties are rare); top phase changes vs GRU 0.7 % (>= 30) / 2.4 % (5 min). TIE.
## Function: refits on the final training labels v4o (oof90; truth = v4l, plain columns of v4o, asserted)
* Package fits (oof88 recipes, PKG = v4o, -> v3fit90): trees 229 x 3 (357,754 rows / 735 signals, n = 222 = mean of the 18
  v4o fold fits), lanes D (78,423 pair-windows, lam 3), setback P50 (sb7 on the v4o OOF function, 43,853 rows), stacker 'single'
  (fit83 recipe) and 'mean3' (note-86 recipe: filtered 3-seed-mean x86_siba4l OOF, all 268,766 training rows). ONNX vs text
  2.0e-14 (21 checked).
* OOF of the full recipe (stacker trained + applied on the filtered x86_siba4l mean, gate .9 decode; v4e = note 86 P_n86):
| function, v4l truth (652 signals) | v4e | v4f | v4f - v4e pt [95 % CI] |
|---|---|---|---|
| >= 30 min E (187,444) | .9193 [.9082,.9290] | **.9197 [.9087,.9293]** | +0.03 [-0.06,+0.12] |
| >= 30 min R (182,828) | .9291 [.9184,.9385] | **.9291 [.9183,.9385]** | -0.00 [-0.09,+0.08] |
| 10 min E | .9024 | .9028 | +0.03 [-0.13,+0.19] |
| 5 min E | .8946 | .8939 | -0.07 [-0.21,+0.08] |
By class >= 30 E: Advance +0.07, Presence -0.02, Count +0.05, YR -0.14 [-0.62,+0.33], non-ATSPM +0.10 (all n.s.); folds -0.09..+0.13;
trees alone acc7 .8775 -> .8798. Neutral, as notes 89 / 89b. vs champion (note 81) +0.06 [-0.09,+0.22].
## Siba members: final v4o refits, seeds 0 / 1 / 3 (swapped in 09:00)
* x86_sibafull4o{,_s1,_s2,_s3} (note-86 GPU queue; s1 resumed after a CUDA OOM at epoch 42). Train loss .560 / .556 / .653 / .586.
  Member agreement (agree90: 339 bench + 2,426 pool detector-windows, unfiltered; pairwise argmax agreement / mean TV):
  v4o 0~1 .913 / .096, 0~3 .924 / .090, 1~3 .914 / .099; seed 2 vs others .868-.901 / .123-.141 = outlier; v4l trio .919-.926 /
  .085-.094. -> members 0 + 1 + 3 (orchestrator rule); seed 2 left out. Stacker mean3 unchanged (x86_siba4l fold-net OOF).
* Parity in v4f (export84 harness, 4 bench signals x 30 min / 3 h, 120 pieces): member prob 3.3e-6, 3-member avg 1.1e-6 (filtered
  7.9e-7), logits 1.0e-5; one piece |dlogp| 1.7e-4 on a near-zero class (busiest_ev 3 h; > the 1e-4 log tolerance, prob 3e-6).
## check.py / speed / RAM
* check.py --freeze then 4 / 4 PASS (re-run after the member swap) (stored answers 0.0; phase renumbering 8.3e-8; channel order 0.0; torch / lightgbm / scipy /
  sklearn blocked). Model card entry final_v4f_note90 + sha256.
* bench90 (bench84 recipe: fresh process per signal, 4 threads, filter ON; warm median of 3 / cold; peak MB), v4f and v4e paired
  on the same load (note-86 GPU job running):
| v4f (v4e) | 30 min | 3 h | 24 h |
|---|---|---|---|
| typical r8 22 ch | 1.43 / 2.84 (708) [1.58] | **2.80** / 4.07 (623) [3.38] | 3.05 / 4.64 (626) [3.34] |
| typical r11 18 ch | 1.23 / 2.27 (527) [1.41] | **2.40** / 3.57 (536) [2.83] | 2.49 / 3.81 (570) [2.74] |
| busiest events 31 ch | 1.80 / 3.03 (740) [2.09] | **3.19** / 4.61 (603) [3.71] | 4.37 / 5.81 (693) [4.80] |
| busiest channels 43 ch | 2.04 / 3.02 (763) [2.20] | **3.78** / 5.24 (806) [4.38] | 5.32 / 6.65 (886) [6.12] |
Final members (0/1/3), 3 h warm: 2.80 / 2.46 typical, 3.33 / 3.73 busiest; 24 h busiest 5.04 s; peak <= 0.89 GB.
Phase-net stage at 3 h warm 0.45-0.70 s vs GRU 0.84-1.22 s; whole call -0.4..-0.8 s; peak -70..-140 MB (<= 0.89 GB).
## Function re-scored on the v4f phase input (oof90 tcnphase). Locked exam: not run.
* Frame pred_phase := top of the TCN-blend decode where the phase pool covers the row (62 %): 3,597 rows changed (1,555 >= 30 min).
  Changed rows lose lane string / context; stacker context + six-fold mean3 + gate decode redone; tree features / pick inputs held.
  >= 30 min E .9197, +0.01 [-0.02,+0.03] vs v4f; R .9292, +0.01 [-0.01,+0.03]; 5 / 10 min +0.03 / -0.03 n.s. Same with GRU top (control) +0.00 -> ~0.
