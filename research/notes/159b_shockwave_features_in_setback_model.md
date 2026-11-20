# 159b — Shockwave features inside the setback model: do they improve it? (study only, 2026-11-16)
User correction to note 159: the question is whether queue-discharge timing, as ENGINEERED FEATURES, improves the
distance the model outputs — not a standalone estimate. Nothing in `src/` changed; nothing committed.
Code `research/code/shock159/sw3_features.py` (features, every window), `sw4_model.py fit|report` (arms, seeds) ->
`%DC_WORK%/s159/swfeat.parquet`, `model_oof.parquet`, `model_gain.json`, `model_results.json`, `sw4.log`.
**Set-up.** Note-58 recipe unchanged (LightGBM P50 of log distance, 60 features, one model per length group, six
signal-grouped folds, OOF; the base refit reproduces note 58 exactly: 20.4 ft / 70.2 %). Windows m30 a-d, h6 a/b, h24 a/b,
full66; printed Advance 628 detectors (5,596 detector-windows), Mid 109 (973); locked_v2 asserted absent. Event = an ON
that began in the red of the PREDICTED phase, lasted > 6 s before the green, still ON at it; release = green -> OFF.
29 features in four families (hi-res only; phase = grouping key; no channel / phase number):
* core (13): count, share of greens, release median / q10 / q25 / q75 / IQR, follower share, late share, 3 s / 10 s holds
* reach (3): when in the red the queue reached the detector (s, share of red), hold
* qcount (4): own actuations between end of green and the held ON, actuations in the 10 s after release, follower ON time
* anchor (9): same-green difference to the first detector released on the same predicted phase (predicted stop-bar
  zones; any detector), the mates' own release, release minus the phase minimum
Arms: base; + each family; + all (PRIMARY, fixed before looking); drop-one; 3-feature minimum; + all with the block
permuted across detectors inside each window (control). Seeds 0-14 on base / all / control, 0-5 elsewhere.
**Result, single seed (the package ships one), mean of 15; base -> + all** (medAE ft; % within 50 ft; Mid: % within 25)
| | 30 min | 6 h | 24 h | 66 h | pooled |
|---|---|---|---|---|---|
| Advance medAE | 25.0 -> 24.3 | 22.5 -> 21.2 | 22.3 -> 21.1 | 20.9 -> 20.8 | 23.4 -> 22.6 |
| Advance within 50 | 68.9 -> 69.9 | 69.0 -> 70.4 | 70.1 -> 71.4 | 70.0 -> 72.8 | 69.3 -> 70.7 |
| Mid medAE | 24.1 -> 20.8 | 26.3 -> 17.8 | 27.3 -> 15.6 | 23.2 -> 15.3 | 25.2 -> 17.9 |
| Mid within 25 | 51 -> 57 | 48 -> 64 | 46 -> 67 | 54 -> 72 | 50 -> 62 |
Seeds better than base: Advance pooled 15 / 15 on every measure (within-50 +1.4 ± 0.3); Mid pooled 15 / 15 (medAE
-7.3 ± 0.4). Control: Advance within-50 -0.1 ± 0.3, Mid medAE +0.2 ± 0.4 — flat.
Paired signal bootstrap, 15-seed-averaged predictions: **Advance pooled** medAE -0.8 [-2.3, +0.5] (not clear), mean error
capped at 300 ft -1.3 [-2.1, -0.5], within-50 **+1.4 [+0.5, +2.3]**, within-100 +1.0 [+0.4, +1.6]; per length within-50
+1.1 [+0.3, +1.9] / +1.6 [0.0, +3.3] / +1.1 [-0.5, +2.7] / +2.7 [-0.2, +5.4] (note 159's "+3.0 [+0.3, +5.8]" at 66 h was
3 seeds). **Mid pooled** medAE **-8.0 [-11.2, -3.1]**, within-25 +13.5 [+7.8, +20.0]; per length -3.2 [-7.3, +0.8] /
-8.8 [-13.6, -3.1] / -13.7 [-16.9, -5.5] / -9.3 [-13.3, -1.6].
**Where (pooled, base -> + all).** Advance > 200 ft 52.6 -> 49.5 ft, -3.2 [-5.7, -0.3], within-50 +2.2 [+0.6, +3.8]
(bias -46 -> -44: still reads short); 101-200 ft 26.0 -> 24.2 (n.s.); <= 100 ft 1.6 -> 1.9 (unchanged). Loop within-50
+1.6 [+0.6, +2.7]; video and radar: nothing (CIs cover 0). Production path (predicted Advance / Mid) within-50 +1.6
[+0.7, +2.4]. Mid loop 23.1 -> 15.6, -7.5 [-12.2, -2.7]; Mid video nothing.
**Held cycles needed.** The gain starts at ONE held cycle: Advance within-50 +3.1 [+1.3, +5.0] with 1-9, +2.9 [+0.6,
+5.3] with >= 10; Mid medAE -12.2 [-15.0, -7.6] with >= 10. With none: Advance medAE +1.5 [-0.4, +2.7], capped mean
+0.5 [0.0, +1.0] — a hair worse, at the edge of noise. Share of printed Advance with >= 1 held cycle: 37 % at 30 min,
57 % at 6 h, 66 % at 24 h, 75 % at 66 h (>= 10: 4 / 29 / 41 / 58 %); Mid 60 / 81 / 88 / 96 %.
**Zones over more than one lane.** Mid over 2+ lanes (78 detectors): 25.2 -> 18.5, -6.7 [-10.5, -2.5]. Advance loops over
2+ lanes (18 detectors): 17.8 -> 22.1 (+4.4 [-2.0, +14.0]), within-50 80 -> 76, **-3.7 [-6.7, -0.7] — worse**. Advance
radar over 2+ lanes (39): unchanged. Video 2+ lanes: 2 detectors.
**Which features.** Pooled gain by family alone (Advance within-50 / Mid medAE): core +0.9 / -6.3, anchor +0.8 / -3.8,
qcount +0.9 / -1.8, reach -0.3 / +0.3 (nothing). All four +1.4 / -7.7; without reach +1.4 / -7.2; 3-feature minimum
+0.6 / -4.5. Shockwave share of model gain 18 % (core 8, anchor 5, qcount 3, reach 2). Keep core + qcount + anchor (26).
**Caveats.** (1) Advance gain is consistent across seeds but small against signal sampling: within-50 and within-100
clear, median error not. (2) 82 of 109 Mid sit at 110 or 160 ft: part of the Mid gain may be telling the agency's two
standard distances apart rather than measuring — untested on another agency. (3) 66 h Fri evening - Mon morning, mostly
weekend. (4) Arms chosen on the same folds; primary arm was fixed first. (5) Only the distance model was tried, not
function or phase. (6) Posted speed / grade are in no table here.
**Verdict.** Yes, it improves the model: Mid clearly (25 -> 18 ft, from 6 h up), Advance modestly (within 50 ft +1.4 pt
pooled, +2-3 pt beyond 200 ft and wherever a held cycle exists), nothing at <= 100 ft, video or radar. Costs to fix in a
build: Advance loops over 2+ lanes (18 detectors) and detectors with no held cycle drift slightly worse. Recommend
adding core + qcount + anchor at the next setback refit; build = one more SQL block on tables `setback.py` already
makes, refit of the four P50 groups, check + re-freeze. Waiting on the user.
