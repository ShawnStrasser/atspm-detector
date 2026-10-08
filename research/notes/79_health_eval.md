# 79 — Detector health: an evaluation set, the package's real performance, and relative checks (2026-10-03)
Why no progress on health since note 47: no labels that the health rules did not themselves produce. This note builds the
best set we have, scores the package scorer (health_core v5 + note-56 / 77 fixes; research copy = package copy, diffed),
and tests four hi-res-only ideas. Code `research/code/health/h79_{evalset,run,score,cand}.py`; out `%DC_WORK%/health79/`.
CPU 4 workers; training signals only (folds_v4 minus locked_v2, asserted); no fault events; `model/` and package untouched.
## Evaluation set (`evalset.parquet`, per detector; tier = independence from the hi-res rules)
| class | A independent | B partly | C circular (66-h rule on the stg log) | U user answer |
|---|---|---|---|---|
| dead | 112 print-dead, alive Dec 2024 | 755 print-dead never seen + 332 card-dead | - | 1 |
| stuck-on | - | - | 83 (dq stuck-on, card) | 7 |
| chatter | - | - | 22 (dq chatter / saturation) | 2 |
| intermittent | - | - | - | 6 |
| degraded (undercount) | 43 share fell > 5x Dec 2024 -> Sept 2026 | 70 advance < .45 of its print-lane stop-bar loop | 26 near-dead | 4 |
Negatives = 8,630 "presumed healthy" (print-labelled, all checks pass) - unverified. User 'ok' answers 5. 405 signals.
Scored on fixed windows: 4 x 30 min (day), 2 x 3 h, 2 x 24 h, in two periods: **w40 = 26-28 Sep (independent
re-observation: labels from 18-21 Sep)** and stg = 18-21 Sep (same period as tier C = optimistic). Inputs = stg OOF
phase / function / lanes of the matching length. prod = exactly the package call (`detectors=None`); listed = + channel list.
## Current package, w40, prod (recall % of class rows flagged bad/suspect; 30 min / 3 h / 24 h)
| class | recall | stg (same period) | note |
|---|---|---|---|
| false alarm (presumed healthy) | **0.33 / 1.16 / 3.38** | 0.33 / 1.09 / 2.65 | listed mode .92 / 1.43 / 3.40 |
| dead | **0 / 0 / 0.1** (listed **38.5 / 93.8 / 97.2**) | 0 / 0 / 0 (listed 43 / 96 / 99.7) | a channel silent all window has no row in prod; 97.7 % of labels still silent in w40 |
| chatter (23) | 42.4 / 67.4 / 82.6 | 53.3 / 65.2 / 87.0 | labels circular |
| stuck-on (90) | 2.2 / 9.0 / 25.8 | 4.8 / 11.2 / 60.1 | episodic: caught only if an episode falls in the sample |
| intermittent (6, user) | 4.2 / 16.7 / 66.7 | 4.2 / 8.3 / 83.3 | |
| degraded (143) | 1.2 / 2.1 / 8.1 | 0.2 / 1.4 / 6.3 | tier A 4.2 / 4.8 / 10.7; B 0 / 1.4 / 4.3 |
User rows on w40: his problems flagged 5 / 15 / 50 %, his ok rows 5 / 0 / 10 % (n 20 / 5; stg 6 / 18 / 70 %).
Precision inside the labelled universe (positives + presumed healthy; prevalence artificial, so a ranking, not a rate):
any problem 34 / 22 / 17 %; by the rule that fired, 24 h: stuck 13 % (252 flags), chatter 28 % (90), choppy / silence /
correlation 2 % (304), level / night drop 4 % (53). Most flags land on "presumed healthy" detectors: either false alarms
or unlabelled real problems - this set cannot tell which.
By technology (evaluation only, w40 24 h): FA loop 3.4 / radar 2.3 / video 8.2 %; recall on non-dead positives loop 16.8 /
radar 12.7 / video 44.1 % (video positives are mostly chatter / stuck: easy kinds). radar_or_video n = 20.
## Candidate checks (hi-res only; `h79_cand.py`): limit = p99.5 of presumed healthy on a calibration half of signals,
judged on the other half, both halves x 3 split seeds; controls = statistic shuffled among the signal-window's
detectors, and random flags at the same added FA. w40, new catches (detector-windows) at 24 h / 3 h / 30 min:
| check | +FA pt | new catches | shuffled | tier A new (base 9 / 4 / 7 of 83 / 82 / 150) |
|---|---|---|---|---|
| (a) green-miss asymmetry vs best same-phase partner (busy greens the partner had, this one silent) | .40 / .50 / .48 | 7 / 6 / 17 | 3 / 5 / 4 | 0 / 2 / 3 |
| (a) count ratio vs that partner, one pooled limit | .50 / .53 / .43 | 7 / 6 / 26 | 7 / 4 / 7 | 0 / 0 / 2 |
| (a) count ratio vs same-predicted-function phase peers | .22 / .22 / .20 | 20 / 15 / 16 | 2 / 3 / 3 | 3.3 / 3.3 / 6 |
| **(c) partner ratio, limit per (own, partner) predicted function** | **1.18 / 1.04 / .89** | **56 / 33 / 43** | 8 / 9 / 3 | **7 / 5.3 / 11** |
| (b) drift within sample (2nd-half share, Poisson z) | .37 / .40 / .38 | 1 / 1 / 8 | 1 / 2 / 5 | 0 / 0 / 1 |
Tier-A recall gain, signal bootstrap 95 % (26-27 signals): (c) +7.2 [1.2, 15.9] / +7.3 [1.5, 14.3] / +8.0 [2.4, 14.6] pt;
same-function peers +4.8 [0, 11.9] / +2.4 [0, 7.7] / +4.0 [0, 8.4]. stg repeats it ((c) tier A +8 / 6.3 / 11.7).
Caveats: most of (c)'s catches are tier B (the same ratio idea with print lanes - circular by construction) and the
peer ratio's are tier C near-dead (same). Only tier A is a fair test, and it is 42 detectors. (c) costs ~200 new flags
on 8.6k healthy detectors per 24-h window set (FA 3.4 -> 4.6 %; 3 h 1.2 -> 2.2 %): about 1 in 5 of its added flags is a
labelled problem. User's 2B422 d8 (advance at .40 of its partner) still missed (partner function unknown).
(c) presence-through-red / count-off-in-red as health rules: not built - the label checks showed they flag configuration
(46 presence zones set to pulse, 37 count zones not in pulse; field list) rather than hardware.
(d) learned scorer: not built - ~67 non-circular non-dead positives (A + U); note 38's synthetic model already failed to transfer.
## Verdict (orchestrator decides; nothing changed in the package)
* Health is good at what physics makes loud (chatter; dead given a channel list) and blind to undercounting (1-11 %).
* Candidate: (c) as an INFORMATION note ("counts x % of d_k, its partner") - real (CI > 0, beats shuffle 5-15x) but it
  roughly doubles flags at 3 h; not as a status change until labelled undercounts exist. Green-miss and drift: dropped.
* Dead in production needs the channel list (one input per signal) - without it a dead channel never appears.
* What would make health measurable: a short list of known-bad detectors on 26-28 Sep (type + rough dates) and
  ~30 known-good, plus per-signal channel lists. Then precision becomes a number, not a ranking.
