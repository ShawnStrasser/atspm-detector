# 38 — Detector health from the hi-res log: rules vs a self-supervised model (2026-09-24, fault events removed 09-28)
Code `research/code/health/` (`health_core` scorer; `hb_build/real/calib/data/synth/model/eval/nofault_eval/spotcheck`);
artefacts `%DC_WORK%/health/`. Training signals only (folds_v4, 780); locked never read. No print / technology / lane input.
## Packaged scorer — `health_core.health(events_df, start, end, detectors=None)` (numpy + pandas)
Events -> 5-min bins per channel (ONs, occupancy, re-triggers < 0.3 s, longest ON; 81/82 only) -> statistics against
the signal's OTHER detectors -> one score per rule (0 fine, .35 at the suspect limit, 1 at the bad limit, NaN = not run)
-> `health_score = prod(1 - s)`, status bad (< .25) / suspect (< .70) / ok / not_enough_data (< 20 ONs, or silent while
the signal was too quiet), one plain sentence (`reason`, with times) and `not_checked` (which rules could not run and why).
0.06-0.16 s per signal-window. `detectors` = the controller's channel list: without it a channel silent for the WHOLE
window has no row in the log and cannot be reported dead. (+ note 40's `rapid` rule.) Rules:
| rule | statistic | suspect / bad |
|---|---|---|
| dead | silent all window; expected = 1/4 of the median live share x signal ONs | 15 / 60 expected |
| sudden silence | best silent run; expected ONs in it = own share outside the run x siblings inside (onset reported) | 30 / 100 |
| stuck-on | longest ON (not over a > 120 s comms gap) | 15 / 60 min |
| chatter | share of ONs re-triggering < 0.3 s (>= 50 ONs) | 30 % / 60 % |
| volume; night > day | ONs per 5 min; own night/day rate / signal's, if own > 1 (2 h of each) | 150 / 250; 3x / 8x |
| level drop | best change point of the share, Poisson LLR > 50 (15-min bins) | to 15 % / 5 % |
| erratic | 15-min residual variance / Poisson (and robust z > 2 vs siblings) | 12 / 30 |
| correlation | expected (Poisson) minus observed 15-min corr with the signal; windows >= 12 h | .60 / .80 |
Limits = ~p99.5 of 8,666 "presumed healthy" detectors (print-labelled, checks pass); silence reports co-silent detectors.
## Self-supervised model (`hb_synth.py`, `hb_model.py`)
Base = detectors the rules call ok, no weak flag (Sept 2026, Dec 2024). Random 2 / 6 / 24 h windows; up to 3 detectors
per window get one of 10 faults (dropout, intermittent, stuck, chatter, random ONs, erratic, undercount, overcount,
correlation break, dead). 159k rows; LightGBM on 31 rule/shape statistics, six grouped folds. Seeds agree within 1 pt;
shuffle control recall .02-.05. No neural set model: the cheapest one already fails to transfer to real problems.
## Synthetic faults, held-out folds (rules strong on stuck/dead/dropout .81-1.00, weak on erratic/scale/random)
| window | rules FA / recall | model @ rules' FA | model @ 2 % train FA: FA / recall | rules OR model p > .8: FA / recall |
|---|---|---|---|---|
| 2 h | 2.0 % / .54 | .74 | 3.7 % / .80 | 2.2 % / .69 |
| 6 h | 1.3 % / .63 | .77 | 2.8 % / .84 | 1.4 % / .76 |
| 24 h | 1.4 % / .69 | .83 | 2.1 % / .85 | 1.5 % / .77 |
## Real known problems, Sept 2026, WITH the fault rule (superseded; % flagged 2h / 6h / 24h / 66h)
Rules: presumed healthy (FA) 1.1 / 1.1 / 2.0 / 4.3; dead on print 92-100; dq health fail 27 / 41 / 61 / 96; label-check
health fail 27 / 34 / 61 / 90; card erratic 18 / 26 / 63 / 88; share fell > 5x since Dec 2024 7-9. Model at 2 % thr: FA
3.3-3.7, no better on any real set. not_enough_data: 8 / 4 / 2 % of healthy at 2 / 6 / 24 h. A degradation BEFORE the
window is invisible to any within-window check; 66-h "false alarms" include real-looking group outages.
## No fault events (user, 2026-09-28) — `hb_nofault_eval.py`
Events 83-88 are set up per cabinet, guarantee nothing, can be wrong: removed from `health_core` (ALLOWED, bins, `n_fault`,
the fault rule; channels seen only via faults lose their row: 468 window-rows) and `hb_act_eval.max_ratio`; `hb_model`
still has `n_fault` (research, not packaged). Fault-defined truth sets: dq health fail, label-check health fail, card
erratic (= faults OR stuck-on OR chatter; dq also saturation). No-fault version keeps a detector only if its dq_core reason
names stuck-on / chatter / saturation; 349 fault-only members dropped (FA set unchanged). % flagged, before (note-40
scorer incl. `rapid`) -> after:
| group (n) | 2 h | 6 h | 24 h | 66 h |
|---|---|---|---|---|
| presumed healthy = FA (8,666) | 1.4 -> **1.4** | 1.4 -> **1.3** | 2.3 -> **2.3** | 4.6 -> **4.2** |
| dead on print (866) / dead since Dec (112) / card dead (544) | unchanged 92-100 | | | |
| dq health fail, old def (436) | 30.0 -> 14.7 | 43.2 -> 18.6 | 61.7 -> 31.5 | 95.9 -> 48.4 |
| dq health fail, no-fault def (87) | 24.5 | 27.6 | 62.6 | 96.6 |
| label-check health, no-fault def (74) | 21.6 | 25.7 | 62.2 | 95.9 |
| card erratic, no-fault def (19) | 14.0 | 13.2 | 73.7 | 100 |
| fault-only members, dropped (349) | 31.3 -> 12.2 | 47.1 -> 16.3 | 61.3 -> 23.8 | 95.7 -> 36.4 |
Synthetic unchanged (clean base had no fault flags). The fault rule only bought catches of fault-defined labels
(circular); on actuation-defined problems nothing is lost; 12-36 % of fault-only detectors still flagged from actuations.
`model/health.py` (production) still has a `controller_fault_events` flag — not touched.
## Decision
**Rules, no fault events.** The model is no better on real problems at ~3x the FA; research only (needs lightgbm).
Spot-check rebuilt for the no-fault scorer (`hb_spotcheck.py`): `review/spotcheck_health.xlsx`, 15 new training signals
with a print (8 bad, one per rule; 4 suspect; 3 ok), one chart per row in `review/spotcheck_health_charts/`.
