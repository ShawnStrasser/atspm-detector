# 118a — Health v118: stuck every ON, extension-time note, erratic counts re-done, count-drop evidence (2026-10-07)
User review of health_review_v3 (Oct 7), points 3-6 only (time-of-day model and too-fast-vs-green are other tasks).
Base = v110 (all F-fixes; h110_resolve, reproduced 115,518 / 115,518 with every new fix off). Same w40 data, 763 training
signals, locked_v2 asserted absent; hi-res log + classifier outputs only; rule-based, numpy, no fault events; CPU 4 procs.
Code `research/code/health118/h118_{events,erratic,resolve,report}.py`; work `%DC_WORK%/s118/`. Nothing adopted.
## Fixes (switchable G1-G4)
G1 stuck: every ON over the type limit is listed with time, length and label (counted / queue / shared). Advance, Count
   and Yellow_Red zones are never excused as 'shared' (whole-phase hold = fault on those zones). New config note C2
   'holds ON although set to pulse - not really pulse' for pulse-mode Count with >= 1 ON >= 60 s or >= 3 ONs >= 5 s
   (24-h pulse Count: any ON >= 5 s 8.5 %, >= 60 s 2.6 %); skipped when C1 already explains the holds.
G2 extension time: new event pass (h118_events, 197 s): too-fast (ON->ON < .5 / < 1 s, bursts) and max-in-5-min now use
   only ONs that START a continuous ON; an ON logged again with no OFF never counts (= h108 values when no re-logs:
   max5 97,169 / 97,169 equal). Limits refit (limits118.csv): Other 2+ max5 126 -> 94, ON->ON < 1 s .226 -> .132;
   Other 1 max5 120 -> 96; others ~unchanged. Config note C1 on Count zones: D1 rule, or >= 20 re-logged ONs and
   >= 5 % of ONs (24-h Count: normal-mode p99 re-logged share .35, pulse .005).
G3 erratic counts (h118_erratic): expected per 15 min = yardstick x its share in the 30 min either side (period left
   out); range +- 3 sd, sd^2 = s (R + 1)(1 + s) + (0.15 e)^2, and >= 5 counts; yardstick must have >= 10 counts in that
   hour. Statistic = share of its counts outside the range; finding needs >= 2 periods outside; limit p99.8 of healthy
   per type x volume band x length (bad 2x); no yardstick -> not scored (Y1 watch on the whole signal as before).
   The chart can draw exactly this: band + marked periods. C = .15: healthy off-period rate .6-1.3 % per volume band
   (C = 0: 1.0-1.8 %; first study run). Fix during the study: an empty yardstick period gave expected 0 -> +1 / >= 10 rule.
G4 count drops: decision = v110 F2; adds drop time, own counts per 15 min before / after and expected after (earlier
   share x rest of signal x type hourly pattern; series in drop_series118.parquet). Guard: expected after > 3x the
   detector's highest 15-min count -> not scored: 1 of 19,299 candidates (11021 d19, night-busy, still bad on others).
   Flagged drops: expected after / own max 15-min count median .27, max 2.35 -> the line stays on the chart.
## Rates per 100 detector-windows (rates118.csv; flagged = suspect + bad)
| window | flagged v110 -> v118 | bad | stuck | erratic | too fast | 5-min | drops | C1 | C2 |
|---|---|---|---|---|---|---|---|---|---|
| 30 min | .554 -> .563 | .104 -> .109 | .224 -> .226 | - | .199 -> .212 | .205 -> .212 | - | .026 | .026 |
| 3 h | 1.404 -> 1.319 | .598 -> .540 | .465 = | .403 -> .246 | .444 -> .454 | .287 -> .273 | .062 = | .062 | .126 |
| 24 h | 3.796 -> 3.789 | 1.648 -> 1.682 | 1.405 = | .567 -> .547 | .767 -> .773 | .317 -> .304 | .314 -> .311 | .088 | .591 |
Newly flagged / unflagged v118: 24 h +.118 / -.125, 3 h +.089 / -.174, 30 min +.053 / -.044. G1 moves stuck from
suspect to bad (+.020 bad at 24 h), no new flags.
## User rows whose outcome changes (user_rows118.csv)
* 34 10037 d22 Advance: suspect -> bad - 112 + 28 min held ON no longer excused because 3 other detectors were ON too.
* 6 01064 d42 Other: suspect -> ok - the 117 in 5 min were re-logged ONs (56 % of its ONs); 43 vehicle starts. Erratic:
  0 periods outside range (was 12.2 'erratic' from a drifting cross-phase share). User leaned ok.
* 17 11021 d2 Advance: bad -> suspect (chattering) - only 1 of 12 periods outside range (06:00, 128 vs ~49). AGAINST the
  user ('most of phase 2 erratic'): d2/d3/d4 jump TOGETHER, which a mates-relative check cannot see.
* 4 2B068 d19, 5 2B058 d46: + C1 config note (stays ok). 10 04016 d20: + C2 'not really pulse' (106-s ON; stays watch).
* Unchanged: 3 / 33 stuck rows now list every ON (08CM405 d37: 30 + 56 + 43 min); 23 08073 d7 erratic kept (8 periods,
  9 % outside, e.g. Mon 15:45 235 vs 67); 22 2B316 d3 'dropped Mon 20:15: 81 per 15 min before, 2 after, ~23 expected';
  7 2B502 d4 ok (choppy / night-high = time-of-day task); 16 10055 d5 suspect (chatter; erratic .12 < Bike-1 limit .38).
## Tested, not adopted
* Expected share = R-weighted median over +-2 h (robust to bursts): healthy limits rise 3-5x, 08073 d7 lost; dropped.
* Own-roughness (15-min count vs median of its own +-30 min, same band): catches some whole-phase jumpiness (11021 d3
  24 h Mon, d4 24 h Sun, 08073 d7), not row 17 d2; union with G3 would add ~+.3 / 100 fires. Orchestrator call.
## Open
Low-volume types (Bike-1) get lenient erratic limits (few counts -> large share outside); 10055 d5's car-like bursts
(Mon 08:30-10:00) are mostly after the 3-h sample end; its 24-h sample is bad on silent / drops / profile / volume.
