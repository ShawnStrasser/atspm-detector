# 41 — Setback distance for every advance detector (predicted inputs, one number per detector)
`trackA/sb5_setback.py` (SQL features, estimators, `setback(events, predictions) -> table`), `sb5_eval.py feats|eval|final`
→ `%DC_WORK%/trackA/setback/sb5_feat_h66.parquet`, `sb5_oof_h66.parquet`, `sb5_results.csv`, `sb5_final.pkl`.
**Inputs:** hi-res only, staging 66 h. Phase = frame-v6 `pred_phase` (OOF), function = note-28 first.all.wi OOF (3 seeds);
no print fact is an input. Truth = printed single `distance_ft` of print Advance (Mid apart), unusual out: training prints +
the released NEWTEST half (cabinet_locked filtered to newtest_released.csv); locked_v2 asserted absent. folds_v4, everything OOF.
**Output (user decision 2026-09-29):** ONE number, `distance_ft` = P50; predicted Presence/Count/Yellow_Red → 0; Other/Bike →
none; < 15 ft → 0. P10/P90 are internal (`internal=True`) and only drive `setback_confidence` high/medium/low (terciles of
log P90/P10 over all predicted A/Mid, no truth used).
**Model.** LightGBM quantile (P10/P50/P90 of log distance, 3 seeds), 60 features: phase green/red/cycle/green share, own volume,
ON-time quantiles, pulse share, arrival shares (first 5/10 s of green, last 10 s, red), start-wave release after green, first-ON
lag, held-in-red timing, phase make-up by predicted function, P(Mid); travel-time block = note-39 tau to the same-lane
(note-30 pair model, per fold) predicted stop-bar zone, speeds, physics estimates. CQR widening from inner folds (80 % target).
**Advance, OOF** — medAE ft / % within ±25 % / % within ±50 ft (n = 579 printed Advance in the frame; 655 printed in all)
| estimator | all 579 | loop 385 | radar 54 | video 140 | <=100 ft 202 | 101-200 159 | > 200 218 |
|---|---|---|---|---|---|---|---|
| **learned P50 (shipped estimate)** | **20 / 71 / 72** | 17 / 78 / 76 | 47 / 54 / 54 | 23 / 57 / 66 | 2 / 82 / 90 | 23 / 71 / 81 | 55 / 60 / 48 |
| seeds 3-5 · no travel-time block | 19 / 69 / 71 · 20 / 69 / 70 | | | | | | |
| physics (user rule, pred. inputs; 42 % covered) | 46 / 42 / 53 | 34 / 48 / 61 | | | | | |
| physics, own-ON speed, L fitted (49-65 ft) | 60 / 30 / 46 | | | | | | |
| hybrid (physics where covered, else P50) | 31 / 54 / 62 | | | | | | |
| **agency standard:** fold median · green-share-tercile median | 85 / 27 / 28 · 40 / 54 / 57 | | | | | | |
| **agency standard:** LGBM on phase timing + make-up only | 34 / 58 / 60 | 29 / 62 / 62 | 64 / 37 / 32 | 34 / 56 / 63 | 1 / 81 / 90 | 29 / 56 / 65 | 73 / 39 / 28 |
| shuffled-label control | 74 / 25 / 33 | | | | | | |
Mid (104): P50 24 / 74 / 91. P10-P90 coverage 80.3 % (target 80), median width 122 ft (48 ft <= 100, 197 ft > 200).
By confidence (523 predicted A/Mid): high 146: 3 / 95 / 95 · medium 158: 19 / 66 / 76 · low 219: 49 / 58 / 52. Per fold medAE 12-32 ft.
**Coverage.** Of 655 printed Advance: 68 had zero actuations in the window (unscorable), 8 lack a classifier row; of 579 scored,
523 predicted Advance (estimate), 36 predicted stop-bar (→ 0, wrong for them), 20 Other/Bike (no answer) ⇒ answered 96.5 %
of scored, production-path medAE 23 / 66 % / 67 %, P10-P90 75 %. Coverage of the estimate itself: 100 % of predicted A/Mid.
Printed stop-bar zones the classifier calls Advance (73) get a median 127 ft — function errors propagate.
**What is agency-specific (report honestly).** 32 % of printed advance sit at 75 ft and the rest cluster on 140/180/220/320 ft:
the agency's design standard. Phase timing + phase make-up alone (the standard, recognised through minor/major approach)
reach 34 ft / 58 %; the full model's extra 14 ft comes from the detector's own behaviour (volume, ON time, arrival profile,
release timing) — which also partly encodes the standard (a short-green minor approach with a 75 ft loop behaves typically).
Dropping the travel-time block costs nothing (20 vs 20 ft): with PREDICTED phase/function and all technologies, the physics
is weaker than note 39's (46 vs 27 ft covered) and the learned model ignores it. Only the physics transfers to another agency
unchanged; the learned model needs per-agency retraining on that agency's prints. > 200 ft and radar stay poor (~50 ft).
**Decisions.** Ship-candidate estimator = learned P50 (not the hybrid). Physics kept as features only. Not in `model/`
(LightGBM + sklearn pair model; a port needs numpy tree evaluation). 66 h only; shorter windows not measured here (note 31:
~6 h enough for the learned model). `setback()` smoke-tested on the bundled sample (no metric read). Locked signals untouched.
