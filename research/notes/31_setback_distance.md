# 31 — Setback distance of advance detectors (first look, time-boxed)

`trackA/sb1_setback.py`, `sb2_eval.py` → `%DC_WORK%/trackA/setback/sb2_results.csv`. No locked
signal read; folds_v3; every constant / model fitted on the other folds.
**Truth.** Print `distance_ft`, unusual_layout excluded, phase detectors: 697 Advance/Mid on 115
signals; scored (single numbers) **572 Advance + 104 Mid** (14 ranges/lists not scored). Advance
is standardised: 75 ft = 35 %, then 140/180/220/320 (loop 392, video 131, radar 56). Staging
events, 66 h (Fri 16:15 to Mon 10:24). Phase = timing phase (predicted phase in production).
**Estimators.** (a) Travel time: ON cross-correlogram advance → stop-bar zone on the same phase
(v3 Presence/Count/Yellow_Red, same lane_index preferred); advance ONs in green >= 8 s after
start, dur <= 2 s; tau = smoothed peak lag; × 40 mph, × fold-fitted speed, or × 22 ft / ON time.
A partner peak exists for 312 / 572. (b) Queue: ONs on the loop from red − 8 s to the first held
ON in red × fitted spacing. (c) LightGBM, 27 features, log target, 3 seeds; shuffled control.
**Advance, 66 h, out-of-fold** — medAE ft / % within ±25 % / % within ±50 ft

| estimator | all 572 (fallback = fold median) | loop, same-lane partner (174) |
|---|---|---|
| fold median (no data) | 84 / 23 / 27 | 85 / 16 / 17 |
| (a) tau × 40 mph | 61 / 26 / 42 | 56 / 20 / 49 |
| (a) tau × fitted speed | 52 / 36 / 49 | 26 / 52 / 74 |
| (b) queue count | 71 / 24 / 39 | 52 / 18 / 49 |
| (c) LGBM travel-time only (5 feats) | 38 / 51 / 61 | 17 / 69 / 82 |
| (c) LGBM all | **25 / 65 / 67** | **7 / 76 / 80** |
| (c) shuffled-label control | 81 / 21 / 30 | 74 / 14 / 31 |

Seeds 3-5: 25 / 65 / 67. LGBM by distance: <= 100 ft 5 / 79 / 88 (203); > 100 ft
42 / 58 / 55 (369). Mid: tau × fitted speed best (20 / 57 / 79); Advance-trained LGBM < median.
**What LGBM learns:** gain = phase mean green 48 %, mean red 7 %, volume 6 %, tau 3 %; dropping
all partner features costs nothing (26 / 64 / 63). It recognises the design standard (short-green
minor approach → 75 ft; major → 180-320 ft) and will transfer only to agencies with the same
standard. The portable, physical part is travel time, and it needs a stop-bar partner.
**Data length** (LGBM all / travel-only medAE, all 572): 66 h 25 / 38; 24 h 25-26 / 39-42;
6 h 25 / 42-43; 2 h 26 / 42; 1 h 28 / 39 (±25 %: 65 → 62 %). Loop same-lane travel time: 6 h =
66 h (26 ft), 1 h 39 ft. **About 6 h suffices.**
**Failures:** speed from ON time no better than 40 mph (dur carries extension/zone length);
queue spill-back weak (held ONs in red are often not queue stops); > 100 ft stays poor.
**Verdict.** Feasible as a coarse estimate (medAE 25 ft, ~2/3 within ±25 %, from ~6 h). Production
needs: the function model's stop-bar zone on the same predicted phase as partner (same lane via
A3 grouping), per-agency calibration of speed / model, and an "unknown" output when no partner
peak exists (45 % of advance detectors here). Not pursued; model/ unchanged.
