# 70 — Function error decomposition on the current candidate (note 51 redone) (2026-10-01)
Candidate = cand64 `function_rows.parquet` (229-feature trees x 3 seeds + fj TCN head w 0.6, six folds x 3 seeds, D-lane
decode, stack pick, cy twin decode; truth_v3s; Dec-role filter; ATSPM-only stack-aware score). Headline pool >= 30 min.
Code `research/code/evaluation/err70.py` (imports note 51's rules) -> `%DC_WORK%/err70/` (`attrib70.txt`,
`errors_{everything,realistic}.parquet`). locked_v2 asserted absent. Analysis only; nothing trained. CPU, 40 s.
Same hierarchical rules as note 51 (first match wins), with two additions: B5 = `stack_extra` (error type that did not
exist in note 51) and D = what note 51's catch-all C4 held besides PM<->PM (Bike / Mid <-> PM).
Three columns so the change can be split: OLD = function_v3e argmax on note-51 truth / sets, ATSPM-scored (no stack credit)
= note 51 restated on today's score; BRIDGE = same v3e predictions on today's truth / sets / stack credit (label + scoring
fixes only); NEW = the candidate. Note 51's own figures (7-class, all windows) are not on this scale: realistic 11.55 pt.
## Today, >= 30 min (pt of rows; share of errors)
| bucket | realistic 183,024 rows / 651 sig. | everything 187,980 / 652 |
|---|---|---|
| total error | **8.26** (acc .9174) | 9.52 (acc .9048 = headline) |
| (a) label likely wrong | **1.52 (18 %)** | 2.97 (31 %; incl. label-check fail 1.48) |
| (b) ambiguous by definition | **3.21 (39 %)** | 3.10 (33 %) |
| (c) model-fixable | **3.30 (40 %)** | 3.23 (34 %) |
| (d) other / unknown | **0.23 (3 %)** | 0.22 (2 %) |
Sub-causes, realistic (pt): (a) config-only label the model contradicts at p >= .8 .49 (loops / no-tech, Adv->Pres, Other->Adv),
unhealthy / no data .45, Dec-2024 stale role still in scoring .33 (90 % radar stop-bar Count; not caught by the Dec filter),
config-only unsure .26. (b) Other subtype acting like a PM class 1.44 (long_zone 832 rows, radar advance_presence 773,
presence_20_75 316, superseded loops 282; called Advance / Presence), Mid / series loop vs Advance / Presence .64 (77 % loop),
YR/Count twins with counts within 5 % .62 (86 % radar), radar long Advance called Other .44, stacked extra lane .07.
(c) systematic ->Other 1.12 (stop-bar Presence->Other 1,010 rows, Advance->Other 675; 61 % radar; p < .5 on 69 %; the
undecoded tree argmax was right on 38 % of these, after the lane decode on 9 % -> the per-lane decode / pick demotes them),
short-sample miss 1.09 (30 min / 1 h wrong, longest window right; Pres->Other, Adv->Other, YR->Count), systematic among PM
classes .93 (Adv<->Pres 643 rows, YR<->Count with differing counts 457, Count->Pres 205; 40 % of these detectors wrong in
every window), phase wrong .15. (d) Bike <-> Presence / Advance / Count .23.
## What the fixes since note 51 removed (realistic, >= 30 min, pt: OLD -> BRIDGE -> NEW)
| bucket | OLD | BRIDGE (labels + scoring) | NEW (model) | removed by labels / by model |
|---|---|---|---|---|
| total | 10.26 | 9.15 | 8.26 | -1.11 / -0.89 |
| (a) | 2.26 | 1.66 | 1.52 | -0.60 / -0.14 (Dec stale .90 -> .30 -> .33) |
| (b) | 4.46 | 4.02 | 3.21 | -0.44 / -0.81 (B1 2.52 -> 2.02 -> 1.44; Mid .85 -> .64; B5 0 -> .18 -> .07) |
| (c) | 3.34 | 3.27 | 3.30 | -0.07 / +0.03 (among PM 1.44 -> .93, but ->Other .65 -> 1.12, short 1.03 -> 1.09) |
| (d) | 0.20 | 0.20 | 0.23 | 0 / +0.03 |
Everything set: total 11.47 -> 10.38 -> 9.52; (a) 3.68 -> 3.08 -> 2.97; (b) 4.32 -> 3.90 -> 3.10; (c) 3.28 -> 3.20 -> 3.23.
Reading: the label work (Dec-role filter, v3s relabels, YR==Count exclusion, stack credit) took out mostly (a) and some
(b); the model work (D lanes, stack pick, fj, twin decode) took out mostly (b) (the model now calls more long / advance-
presence zones and Mid loops Other) and moved PM<->PM confusions into ->Other — the fixable bucket did not shrink in total.
Shares: realistic 18 / 39 / 40 / 3 % (note 51: 21 / 44 / 36 %, different scale).
## Caveats
Rules are an estimate, not a relabel; first match wins, so (a) takes precedence over (b) / (c). OLD has no stack credit
(stack groups were not defined then), so B5 is new and small. Health = note 51's (v5 Sept, v3 Dec). Bridge = v3e argmax
without any decode, so "removed by model" includes the decode / pick, not only the fj head.
## Ceiling (realistic, >= 30 min, today .917)
* Perfect labels alone: +1.5 pt -> ~.933 on the same model (measured score rises; real-world accuracy does not).
* Model side: (c) is 3.3 pt; recovering half (the usual yield of a targeted lever here) -> ~.95 with clean labels. Biggest
  lever = the ->Other demotions (1.1 pt, low confidence, decode-caused) and 30 min / 1 h misses (1.1 pt).
* (b) 3.2 pt moves only with a definition / scoring decision (score long zones / advance-presence as their nearest class;
  superseded loops and identical YR/Count twins are unidentifiable from the log). Clean labels + half of (c) + all of (b): ~.98.
## User summary (6 lines)
Today the function model gets about 92 of 100 detector-samples right (30 min or longer, clean-label set).
Of the 8 it misses: about 2 are wrong labels, 3 are detectors that behave like another class by design, 3 are real model errors.
Since the last review, label clean-up removed 1.1 points of error and model changes another 0.9.
Fixing the remaining labels would read about 93 %; fixing half the real model errors would reach about 95 %.
The last 3 points need a definition choice (e.g. call long radar zones by their nearest class), not a better model.
So ~95 % is a realistic ceiling for this data; ~98 % only if the definitions change too.
