# 39 — Setback distance, the user's physics rule (no ML first)
`trackA/sb3_pairs.py` (same-lane P from note 30's print pair model, OOF, per window), `sb3_physics.py`
(the rule), `sb4_eval.py` → `%DC_WORK%/trackA/setback/sb4_results_g4_4.csv` (+ `_coverage`). Truth as
note 31: printed single distance, Advance, folds_v3, all 186 old hold-out signals + locked_v2 excluded
(asserted): **568** Advance (note 31: 572), 32 Mid apart. Phase = timing phase; partner role = v3 label.
**Rule.** Speed = (20 ft car + zone: Count 6 / Presence 20 ft) / median ON time of isolated (3 s clear)
vehicles >= 15 s into green, low-flow hours; pulse zones give none; 15–60 mph. Source order: partner >
phase Count > phase Presence > advance's own ON (20 + 6 ft) > 40 mph. Travel time = advance ON (>= 4 s
into green, >= 4 s before yellow, isolated) → NEXT ON of the same-lane (P >= .5) stop-bar zone 0.2–15 s
later, Count > YR > Presence; tau = median within [0.7, 1.4] x histogram mode, >= 5 hits, mode >= 1.5x
background. d = v·tau + zone edge (Presence 20 ft); < 15 ft → 0. No accepted partner → unknown.
**66 h, data-derived pairing** — medAE ft / % within ±25 % / ±50 ft (n)
| subset | user rule | adv ON first | tau x 40 mph |
|---|---|---|---|
| all 568 (unknown → fold median) | 60 / 38 / 46 | 60 / 35 / 47 | 74 / 22 / 28 |
| covered (same-lane partner, 47 %) | **27 / 50 / 63** (269) | 29 / 44 / 64 | 68 / 15 / 24 |
| loop / radar / video covered | 25 / 55 / 70 (203) · 90 / 36 / 33 (33) · 59 / 30 / 48 (33) | | |
| same-lane Count / Presence partner | 60 / 48 / 48 (77) · 24 / 50 / 69 (192) | 44 / 49 / 53 · 27 / 42 / 68 | |
| <= 100 ft / > 100 ft · Mid | 19 / 50 / 79 (131) · 59 / 50 / 48 (138) · Mid 20 / 66 / 84 (32) | | |
Coverage 269 / 568: 202 have NO stop-bar zone on their timing phase, 75 no accepted peak, 22 only an
other-lane peak (52–98 ft, not used). Loop 203 / 385, video 33 / 131, radar 33 / 52 covered.
**Pairing is not the bottleneck:** pick right 87 % (252 with print lanes); print-lane oracle 27/50/64 (256). **Speed is.** log-corr of speed vs implied print/tau speed: partner ON time .54 (Presence) /
.42 (Count, only 12 non-pulse), phase Presence .52, advance's own ON .62–.73 (reads +26–35 % high with
26 ft: effective length ~20 ft). Count zones are ~all pulse → no speed, and Count partners sit at
long-setback / high-speed approaches, so "Count preferred" does NOT help (60 vs 24 ft). Per-vehicle
speed x tau (`phys_veh`) 29 / 43 / 60: no better. Green filter 10 s instead of 4 s (chosen on these
folds, disclosed): covered 37 / 48 / 56 (210); 2 s: 30 / 46 / 62 (278).
**vs note 31.** Loop same-lane: rule 25 ft (203, nothing fitted) = note-31 tau x fitted speed 26 ft (174).
All-568 60 ft vs note-31 LGBM 25 ft — that LGBM is the design standard (green time), not physics.
**Data length** (covered medAE / coverage): 66 h 27 / 47 %; 24 h 31–33 / 46 %; 6 h 34–50 / 41 %;
2 h 48 / 36 %; 1 h 57 / 30 % (peak and midday alike). **>= 24 h needed; 6 h is marginal.**
**Second step, reported apart (fitted on other folds, covered rows).** Per-technology speed x tau:
27 / 55 / 72 (all-568 50 / 40 / 51), loop 21, video 28, radar 58. LightGBM residual on physics-only
features: 22 / 62 / 67, but <= 100 ft 7 ft and > 100 ft 64 ft (worse than the rule): it snaps short tau
to 75 ft = learns the agency standard. Shuffled-label control 38. **Not recommended.**
**Verdict.** The rule works where it applies: loop advance with a same-lane stop-bar zone on the same
phase, >= 24 h: medAE ~25 ft, ~55 % within ±25 %, 70 % within 50 ft; short setbacks (<= 100 ft) 19 ft.
Long setbacks, radar and video stay poor (~60–90 ft): stop-bar ON-time speed is weak (pulse count
zones, unknown zone length); video/radar ON is not a line crossing. "Unknown" for the 53 % without a
same-lane partner. Cheapest gain: per-technology speed calibration (loop 25 → 21 ft), per-agency only.
