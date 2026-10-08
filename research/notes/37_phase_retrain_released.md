# 37 — Phase retrain with the 71 released NEWTEST signals (2026-09-24/25)

Code `research/code/lightgbm/phase_v3_trees.py`, `research/code/neural/phase_v3_{net,chain,eval}.py`; results
`dc_work/final_v3_work/phase_v3/work/eval.json`. Official timing labels only. No `locked_v2` signal in any pool/prediction (asserted).
## Set-up (stage-13 recipe, nothing tuned)
* Trees: `fit_final_v1` recipe (3-seed LambdaRank bag → joint decoder, 22 windows, same params), on the 701-signal pool
  plus the 71 released signals: 772 signals, 1,670,485 labelled rows. The released rows are built like NEWTRAIN
  (`build_extra`: labelled detectors only). Non-recipe changes: n_jobs 8, float32 features, per-fold checkpoints.
* GRU: `train2.run` unchanged (seed 0, 45-epoch cap, patience 7), 780-signal map. Rasters were built for the 71
  (`ncache2`). The integrity check flags 01032 (cf208ff8): its raw log has only 2 call events in 66 h. That is real
  data, not a truncated cache.
* **Folds:** the 709 keep the stage-13 phase fold map; the 71 take their `folds_v4` fold (1–5). `folds_v4` itself
  differs from the phase map on 22 old NEWTRAIN signals (inherited function folds). I kept the phase map so that only
  the training data changes. Flag for the orchestrator: phase OOF and `folds_v4` disagree on those 22 signals.
* Evaluation (`phase_v3_eval.py`): 0.5 blend before the decoder, with the decoder re-fitted out of fold on the blended
  inputs. Above 120 min the net sees K=4 pieces from the 32-piece grid (the final_v3 rule, note 33). Old arm = final_v1 trees OOF +
  stage-13 GRU OOF (lp_k4 pieces for long windows). On the released rows the old arm uses the final_v1 trees and the
  stage-13 final GRU; those rows are scored but excluded from its decoder's training. It matches note 33's lp_k4 refit exactly.
  Paired signal-grouped bootstrap (2,000 draws) of new − old.

## Results (six-fold OOF; blend = what ships)
| rows | model | 5 min | 10 min | 30 min | 1 h | 3 h | 6 h | 24 h | full |
|---|---|---|---|---|---|---|---|---|---|
| same (stage 13's 1,532,182 rows, 685 sig.) | trees old/new | .9287/.9292 | .9423/.9418 | .9623/.9623 | .9621/.9623 | .9762/.9767 | .9770/.9774 | .9797/.9793 | .9808/.9804 |
| | net old/new | .9273/.9280 | .9475/.9451 | .9640/.9637 | .9658/.9647 | .9689/.9674 | .9647/.9632 | .9542/.9539 | .9313/.9316 |
| | **blend old/new** | .9521/.9527 | .9617/.9608 | **.9747/.9745** | .9746/.9742 | .9799/.9802 | .9808/.9808 | .9812/.9811 | .9832/.9821 |
| | blend Δ pt [95 % CI] | +.05 [−.07,+.18] | −.07 [−.18,+.04] | **−.01 [−.10,+.07]** | −.03 [−.12,+.05] | +.03 [−.05,+.11] | −.00 [−.11,+.10] | −.01 [−.10,+.08] | −.09 [−.20,+.01] |
| all (+130,612 released rows, 71 sig.) | trees old/new | .9303/.9306 | .9430/.9425 | .9634/.9634 | .9628/.9630 | .9769/.9774 | .9782/.9785 | .9807/.9804 | .9813/.9810 |
| | net old/new | .9283/.9299 | .9483/.9462 | .9651/.9649 | .9673/.9660 | .9699/.9686 | .9662/.9646 | .9559/.9554 | .9318/.9318 |
| | **blend old/new** | .9535/.9541 | .9626/.9619 | **.9757/.9757** | .9758/.9753 | .9808/.9811 | .9818/.9818 | .9822/.9821 | .9834/.9825 |
| | blend Δ pt [95 % CI] | +.05 [−.07,+.17] | −.05 [−.15,+.05] | **−.00 [−.09,+.08]** | −.04 [−.12,+.04] | +.03 [−.05,+.10] | −.00 [−.11,+.09] | −.01 [−.09,+.06] | −.09 [−.19,+.01] |
| released only (old = models that never saw them) | blend old/new | .9699/.9696 | .9725/.9738 | .9875/.9882 | .9884/.9873 | .9909/.9909 | .9932/.9932 | .9936/.9932 | .9928/.9928 |

(Net above 120 min = K=4 pieces, weaker alone than all pieces, e.g. .9668 → .9313 at full.) Per-fold blend at 30 min, old → new: .9747/.9745, .9720/.9729,
.9778/.9797, .9744/.9719, .9775/.9773, .9715/.9710.
## Noise floor and verdict
* Tree recipe-noise control (same new code, 709 old signals only, `trees_oof_control_bywindow.parquet`): control −
  final_v1, trees alone, ranges −0.05 … +0.05 pt. The trees' new − old (−0.04 … +0.05) is the same size, so the 71
  signals do nothing measurable to the trees on the old rows.
* No blend difference beats the noise: every 95 % CI contains 0. The largest is the full window, −0.09 pt (12 more
  errors out of 12.7 k). That is inside the blend seed sd at full length (0.15 pt, note 13) and in line with the tree sd (~0.07 pt).
* The net alone (one seed per fold) moves −0.2 … +0.1 pt, inside its 0.1–0.45 pt seed sd. Its 10-min / 3 h / 6 h
  CIs exclude 0 on the same rows, but a single-seed refit cannot separate data from seed. The blend absorbs it (note 20).
* **Verdict: neutral.** +10 % signals neither helps nor hurts the phase OOF at any length. The rule "more data should
  not hurt" holds, so the final models were fitted on all 780.
## Final models (`dc_work/final_v3_work/phase_v3/`; candidate package and `model/` untouched)
* `phase_lgbm_v5_s{0,1,2}.txt` (+ `.json`, 261 features, 600/561/619 trees), `decode_lgbm_v5.txt` (+ `.json`, 595
  trees; trained on the new OOF first-stage probabilities, as final_v1). Recipe as final_v1: the rankers train on
  folds 1–5 with fold 0 as the early-stopping set; the decoder trains on the OOF tree-only p0 (final_v2 also ships the
  tree-only decoder). The numpy backend reproduces both exactly (0.0). `trees_fit_info.json`.
* `gru.onnx` from the `p3_final` GRU (all 780 signals, `work/gru_models/p3_final.pt`). `gru_onnx_check.json` compares
  ONNX with torch on real rasters. Drop-in names differ from the candidate's (`_v5`), so rename or update the `.json`
  pointers when assembling final_v3. Also re-run `check.py` and re-freeze its references.

## Cost and incidents
GPU ~11.5 h wall (six folds 78–110 min training + ~4 min K=4 inference each; final fit; 9 min stage-13 final GRU on the
71). Fold 1 hit the 45-epoch cap (best at epoch 44), exactly as stage 13's fold 1 did; left at the recipe.
CPU ~1.5 h at 8 threads (trees OOF 30 min, control 30 min, final 8 min, evals). The first chain stalled at epoch 14 of
fold 0: DataLoader workers spawned from the network share took ~5 min each. Fixed by running from a local snapshot
(`phase_v3/repo_snapshot/`) and resuming from the checkpoint: launch GPU chains from local disk.
