# 40 — Detector health from individual actuations (ON / OFF physics) (2026-09-28)
Code `research/code/health/`: `health_core.act_stats` (per-actuation statistics), `hb_act.py` (Sept 2026 windows +
Dec 2024 full), `hb_act_eval.py` (all tables below). Artefacts `%DC_WORK%/health/act_*.parquet`. Training signals only
(folds_v4 minus locked_v2, 761 Sept / 373 Dec signals); hi-res log only. Harness = note 38 (same windows, groups, FA set).
## What was built (the user's three ideas), per detector and window
* Per actuation: ON duration, OFF gap, ON->ON interval (anything spanning a > 120 s comms gap dropped). Statistics:
  share of ON->ON < 0.5 s / < 1 s, bursts (>= 5 ONs, every interval < 1 s), re-trigger < 0.3 s, repeated identical
  intervals / ON-OFF toggling (0.1-s clock, >= 4 in a row), ON time in 2-15 min ONs, ONs 0-0.25 s after a colour change
  vs chance (crosstalk).
* Flow-occupancy: in the uncongested branch occupancy = volume x mean ON, so the test is whether the mean ON per
  actuation holds: 5-min bins > 6x or < 1/6 of the detector's own median, Spearman(volume, occupancy), "sticky" bins
  (> 50 % occupied, <= 2 ONs); a green-gated copy using ONs >= 5 s into green of the detector's best-lift phase
  (phase-anonymous; stop-bar queues on red excluded), 15-min mean ON off by 4x.
* Sibling distance: Jensen-Shannon distance of the ON-duration / ON->ON histograms to the most similar sibling.
* "Normal per behaviour" learned from the log, not the mode: groups pulse (> 50 % one-tick ONs; pulse count zones show
  60-80 %, not 100 %) / median ON <= .5 / 1.5 / 4 / > 4 s. Limit = p99.8 of presumed-healthy in the group (>= 50 ONs).
## Screening: OR with the note-38 rules vs simply LOOSENING those rules to the same false-alarm rate
Positives pooled = dq / label-check health fail / card erratic (the sets with actuations). Catch new / loosened, and
90 % CI of the difference (bootstrap over signals). FA rises 0.1-1.4 pt per family.
| family | 2 h | 6 h | 24 h | 66 h |
|---|---|---|---|---|
| fast ON->ON < 0.5 / 1 s | 29.0 / 27.8 (+0.3..+2.3) | 42.9 / 42.1 (-0.1..+1.8) | 61.4 / 61.4 | 95.9 / 95.9 |
| bursts | 29.7 / 27.6 (+1.1..+3.3) | 43.1 / 41.9 (+0.4..+2.2) | 61.7 / 61.2 (-0.3..+1.3) | 95.9 / 95.9 |
| re-trigger < 0.3 s by group; long ONs 2-15 min | within +-0.7 of loosened | same | same | same |
| flow-occupancy (5 stats) | 28.5 / 31.3 (-4.2..-1.2) | 42.5 / 42.8 | 61.5 / 62.0 | 96.1 / 96.6 |
| sibling histogram; colour lock | 27.0 / 27.8 (-1.7..0); 26.8 / 27.3 | worse | worse | worse |
| all 13 at once | 32.6 / 36.2 (-5.7..-1.3) | 45.0 / 48.3 | 62.6 / 65.3 | 96.3 / 96.8 |
Rhythm / toggling fired on ~0 detectors in any set. Flow-occupancy / sibling distance: heavy healthy tails.
## Kept: `rapid` rule in health_core (ioi_lt05, ioi_lt1, burst_frac / group limit; suspect at 1x, bad at 2x; >= 50 ONs)
| group (share flagged, %) | 2 h | 6 h | 24 h | 66 h |
|---|---|---|---|---|
| presumed healthy = FA | 1.1 -> 1.4 | 1.1 -> 1.4 | 2.0 -> 2.3 | 4.3 -> 4.6 |
| dq health fail | 26.6 -> 30.0 | 41.4 -> 43.2 | 60.9 -> 61.7 | 95.6 -> 95.9 |
| label-check health fail | 27.0 -> 28.1 | 33.8 -> 34.3 | 61.1 -> 61.6 | 90.3 -> 90.8 |
| card erratic | 18.3 -> 20.0 | 25.8 -> 26.7 | 62.5 -> 63.3 | 88.3 -> 90.0 |
| share fell > 5x since Dec 2024 | 7.0 -> 7.8 | 5.8 -> 7.0 | 8.1 -> 8.1 | 9.3 -> 9.3 |
Dead / dead-since-Dec / card-dead unchanged (no actuations to look at, by construction). Pooled vs loosened at equal
FA: 2 h 30.0 / 28.0 (+1.1..+3.2), 6 h 43.2 / 42.4 (-0.1..+1.9), 24 h and 66 h nil. Controls: rapid ratio shuffled
across detectors -> -2.4..-5.4 pt vs loosened at 2 h; limits fitted on a random half of signals, scored on the other
half (3 seeds): FA +0.3-0.4 pt, catch +1.2..+3.5 pt at 2 h. Gain is real but confined to short windows.
**Non-circular check, Dec 2024 -> Sept 2026:** of 6,565 detectors the rules call ok in Dec 2024 (>= 50 ONs), `rapid`
flags 37; 10.8 % of them (4) are dead in Sept 2026 vs 2.8 % base and 5.4 % (2) for the rules' own near-misses of the
same count (fast family alone 4 / 28). Small n: a pre-failure hint, not a proof.
**Circularity:** dq / label-check / card-erratic labels come from 66-h faults 84-88, ON > 900 s and re-trigger < 0.3 s
> 30 % — the chatter family shares physics with `rapid`. Only 3 of the 68 new catches on positives had 66-h chatter
> 30 %; most carry fault events outside the short window. The 66-h column stays partly circular (note 38).
Runtime: `health()` 0.07 s per signal at 2 h, 0.48 s at 66 h (was 0.06-0.16; act_stats(light=True) adds it).
## Examples
* 08045 d23 (loop, stop-bar Presence; label-check health + saturation), Mon 07-09: ON 0.1 s / OFF 0.5 s repeating
  (07:01:32.2, 32.9, 33.5 ...), 46 % of ONs within 1 s of the previous, 29 % inside bursts, median ON 0.2 s where a
  presence zone holds seconds. Old rules ok at 2 h (re-trigger share 24 % < 30 %); `rapid` bad.
* 03031 d9 = d44 (one radar zone on two inputs, Count / Yellow_Red; dq health fail), Sat 13-15: 35 % of ONs within 1 s, 11-12 bursts;
  26 fault events over 66 h but 3 in the window -> old rules ok at 2 h, `rapid` bad.
* 2B024 d10 / d11 (loops, median ON 1.2 / 1.7 s), Dec 2024: 18-21 % ON->ON < 1 s, 10-16 % in bursts, rules ok; both
  dead on the Sept 2026 print. 06079 d19 (12.8k ONs, 19 % < 1 s in Dec) silent in Sept 2026.
* False alarms left: long radar "Other" zones (2B062 d51, 22 % < 1 s: two vehicles in one zone), low-count bike
  loops (20-55 ONs; hence the 50-ON minimum).
## Decision
Keep `rapid` (in `health_core.health`, numpy + pandas, log only); drop the rest (none beats loosening the existing
rules at equal FA). Headroom is small: the real-problem sets are mostly dead or fault-event defined.
