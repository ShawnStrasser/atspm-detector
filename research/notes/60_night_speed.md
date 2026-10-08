# 60 — Deterministic night-time free-flow speed per phase (validation plan step 7, side output) (2026-10-01)
Not trained. `trackA/sp1_night_speed.py` (numpy only, self-contained, could move into model/ unchanged): for each predicted
Advance with a predicted setback, its lane-mate stop-bar zone (D lanes, same predicted phase; Count > Yellow_Red > Presence,
zones of that role in the lane merged), match isolated night vehicles advance ON -> stop-bar ON; speed = (setback - zone edge,
Presence 20 ft else 0) / travel time. Vehicle rules: advance ON 00:00-05:00 local; no other advance ON within 10 s; stop-bar
zone empty, no ON in the 3 s before; phase green >= 3 s before the advance ON and >= 1 s after the stop-bar ON; exactly one
stop-bar ON within d / 15 mph, none before d / 80 mph; setback >= 50 ft. Chance matches: same matching against the stop-bar
zone shifted by -61 / +47 / +89 s = n_bg; answer needs n >= 5 and n >= 3 x n_bg. Estimate = median (p25, p75 = spread) of the
vehicles within +-25 % of the histogram mode of log speed. Per detector and per phase (vehicles pooled); NaN + reason
(no_night / no_partner / no_setback / too_few / weak_peak). Inputs: OOF lanes_D.func (function, phase, lanes; note 58) and
sb7 `all` P50 setback (note 58). Print distance / technology for evaluation only. CPU 4 threads; locked_v2 asserted absent.
Code `trackA/sp1_eval.py run|report`; work `%DC_WORK%/trackA/speed/sp1/` (det / phase.parquet, results.json); ~3 min a run.
Windows: note 58's nine + night-anchored n30 (Sat 01:00-01:30), n2h (01:00-03:00), n6 (Sun 00-06) that BORROW m30_c's
predictions (Sat 21:30, nearest 30-min sample; disclosed approximation).
## First version failed its control
v1 (10-90 mph window, plain median, no background test): a control with the stop-bar zone shifted +47 s still answered
12-17 % of phases (median 18 mph, n 17-26) -> chance matches pulled real answers low. Fix above (15-80 mph, background
test, mode-anchored median; bounds are physical, not fitted to any truth). v2 control: **2 answers of 2,741 phases (0.1 %)**
vs 623 real (n_bg median 10 vs n 163 at 66 h).
## Results (phases with >= 1 predicted Advance; v2)
| window | phases | answered | 2/6 · 4/8 · other | 2/6 where a lane-mate exists | median n |
|---|---|---|---|---|---|
| m30 a-d, h6 a/b (no night) | 2,639-2,735 | 0 (no_night) | - | - | - |
| n30 (30 min of night) | 2,639 | 11.3 % | 22.6 · 0.7 · 0.8 % | 52.5 % | 13 |
| n2h / n6 | 2,639 | 17.2 / 19.9 % | 33.7 / 36.1 · 2.6 / 7.6 · 1.1 / 1.7 % | 78.1 / 83.7 % | 31 / 65 |
| h24_a / h24_b | 2,743 | 19.0 / 19.7 % | 36.6 · 6.1 · 1.8 % (a) | 79.6 % | 76 / 69 |
| full66 | 2,741 | 22.7 % | 38.2 · 14.2 · 4.9 % | 82.8 % | 163 |
Why not answered (full66): too_few 1,353 (side streets / lefts: at night the vehicle arrives on red and calls the phase, so
the green rule rejects it), no_partner 741 (27 %: no predicted stop-bar zone in the advance's lane — main streets with
advance-only detection), weak_peak 24. Detector level full66: loop 12.9 %, radar 62 %, video 24.8 % answered.
**Distribution** (phase medians, mph, p10/25/50/75/90): full66 20.6 / 24.7 / 29.8 / 37.6 / 44.9; h24_a 22.6 / 26.3 / 31.7 /
37.6 / 43.0. Within 25-55 mph 68 % (full) / 74-75 % (h24); within 20-60 87 / 93 %; < 15 mph 0, > 65 mph 2 %. 2/6 median
31.6, 4/8 24.9, other 25.4. Within-phase spread (p75-p25)/median .13. The low tail is NOT setback error: with the PRINT
setback the same detectors give 20.9 / 24.6 / 29.0 / 38.2 / 43.5 (70 % in 25-55). It is the measure itself: mean speed over
the last 75-400 ft of the approach (deceleration, turners) and the assumed zone edges; read it as approach speed, not posted.
**Sensitivity to setback** (detectors with a printed single distance, pred vs print setback, same vehicles): full66 n 149:
|speed error| median 14.2 % (setback 13.7 %, corr .99), within 10 % 38 %, within 5 mph 56 %; h24_a / b 15.6 / 13.6 %, 52-53 %
within 5 mph. Loop 11 % (98), radar 21 % (38), video 29 % (13). Speed error = setback error passed through, one for one.
**Repeatability**: h24_a vs h24_b (two different nights) |diff| median 1.5 mph, p90 5.7 (n 478); h24_a vs full66 1.9 / 6.4.
Short night samples vs full66 4.4-4.9 / 11.5-11.9 mph (mostly the 30-min-sample setback and lanes they borrow).
Opposing through approaches (phase numbers, evaluation only): 2 vs 6 log-corr .50, |diff| median 3.0 mph (n 173); 4 vs 8
.61 / 4.6 (n 14). Per print technology (full66 answered, median, in 25-55): loop 30.2 / 68 %, radar 28.9 / 74 %, video 31.0 /
54 % (video p10-p90 19.5-58; video print-setback speeds 16.6-64.6: a video zone ON is not a line crossing).
## Ground truth
None available: the config export holds DeviceId / Phase / Function / Detector only; the print table and the timing
descriptions carry no posted speed (searched; one print mentions a radar queue zone's 0-40 mph filter, not a limit).
Plausibility only (above). A posted-speed or spot-speed table from the user would allow a real check.
## Verdict
Works where it applies: through phases with a stop-bar zone in the advance's lane and >= 2 h of night (78-84 % of those
answer; 2 h ~ 24 h coverage), repeatable night to night (1.5 mph), clean against the shifted control. Error is the setback
error one for one (~14 %; loop 11 %). Overall coverage is low (19-23 % of phases with an Advance, 0 without night) and
it reads approach speed (median 30 mph, 25 % below 25 mph), not posted speed. Side output only; nothing shipped.
Open for the orchestrator: offer to the user as a loop-mainline add-on, or park with the setback output.
