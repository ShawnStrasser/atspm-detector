# 51 — Function error review: where function_v3e's errors sit (validation plan step 1) (2026-09-30)
function_v3e six-fold OOF (frame v6e, run `..._h3`, 3-seed mean), note-49 sets: everything 268,861 rows / 652 sig.
(acc7 .8724, 34,303 errors), realistic 261,622 / 651 (.8845, 30,215) — both match note 49. Code
`research/code/evaluation/err51_{build,analyse,examples,attrib}.py` -> `%DC_WORK%/trackA/err51/` (`rows.parquet`,
`breakdown.txt`, `examples.txt`, `attrib.txt`, `errors_*.parquet`). locked_v2 asserted absent. Analysis only; nothing trained.
Rows = detector-windows; shares are row-weighted. Figures below: realistic (everything in brackets where it differs).
## Confusion
* Recall: Adv .927, Pres .943, Count .910, YR .714, Mid .810, Bike .814, Other .663. Precision Other .757 (lowest).
* Unordered pairs, share of errors: Adv–Other .207, Other–Pres .181, Count–YR .147, Adv–Pres .135, Adv–Mid .064,
  Count–Pres .064 [.101]. Ordered top: Other->Adv .125, Other->Pres .118, YR->Count .109, Adv->Other .081.
## Breakdowns (acc; share of errors)
* Window: m5 .857, m10 .869, m30 .886, h1 .884, h3 .899, h6 .900, h24 .901, full .896. Length explains ~4 pt at most;
  long windows still miss 10 %. Pair rates are ~flat across windows (Other->Adv 17 vs 13 per 1000 rows, m5 vs full).
* Technology (prints): loop .930 (.26), radar .869 (.52), video .744 (.14), none .804. Loop-labelled Other is called
  Advance 72 % of the time (superseded loops); YR->Count is radar / video only.
* Health: ok .888 (.91), suspect .852 (.06), bad .739 (.03). Permissive phase .883 (no gap); right-turn lane .816 (.085).
* Label source: print-high .895, config-only .797, user ruling .779 (hard cases by selection). Tier: complete_high .903,
  mixed .882, incomplete .884, no tier .745. Unusual layout .787 (.15). Label check fail .448 / misconfigured .379 [everything].
* Phase: right .892, WRONG .656 (4,736 rows; .054 of errors; excess over right ~0.4 pt), no timing phase .703.
* Signals: worst 10 % by error count hold .45 of errors; worst 10 % by rate hold .30 of errors on 7 % of rows (acc .51);
  without them acc .913; median signal .94. Detectors: .30 of errors from detectors wrong in every window, .39 mostly
  wrong, .31 sometimes wrong -> errors are mostly per-detector systematic, not sampling noise.
* Confidence: p>.97 acc .973 (.09 of errors), .5–.7 .615, <.5 .431.
* NEW: period drift. 142 detectors are wrong in Dec 2024 (<.2) and right in Sept 2026 (>.8), p_med .95, 90 % radar,
  mostly Count->Presence / Presence->Advance with the config export saying the Dec role (e.g. 04091 d9 "Rad B Count",
  config Presence). Labels come from the current print, so the Dec rows carry a stale label (2,288 errors = .87 pt).
  Where config differs from truth: Dec acc .51 vs Sept .70. These rows are also in TRAINING.
## Examples (~10 per pair, `examples.txt`) — verdicts
* Other->Advance: superseded_by_radar loops (acc .035: behave exactly like advance loops; role not in the log),
  radar advance_presence zones (user-ruled Other; occ .18–.20, dur 2 s — the same kind of zone that prints elsewhere
  call Advance), dump / on-ramp loops. -> ambiguous by definition; ~1 in 10 label doubtful (config-only Other).
* Other->Presence: long_zone (acc .42) / presence_20_75 / rail presence at the stop bar — behave as Presence. Ambiguous.
* YR->Count: 83 % of YR zones are pulse; 30 % of YR errors have a same-phase Count with the same count (±5 %: acc .55 =
  coin toss, the zone is logged identically), 21 % have no Count on the phase. Ambiguous; the rest (count differs) is fixable.
* Advance->Other: radar advance zones NOT in pulse (dur 3–5 s, occ .2–.3, held in red) = mirror of advance_presence.
  Ambiguous (print wording). Some twins with identical counts (04026 d51/d52: data issue).
* Advance<->Presence / Count->Presence: many config-only labels contradicted by the description or a twin (2B091 d27,
  04034 d22 "RL Presence" labelled Advance) = label wrong; video advance zones held in red = genuinely presence-like;
  most radar Count->Presence / Presence->Advance with det acc ~.5 = the Dec-drift case above.
## Attribution (hierarchical rules, `err51_attrib.py`; an estimate, not a relabel) — pt of accuracy, share of errors
| group | realistic | everything |
|---|---|---|
| A label noise: Dec stale .87, unhealthy/no_data/bad .60, config-only confident .57 / unsure .34 (+ fail/misconf 1.51) | **2.39 (21 %)** | 3.85 (30 %) |
| B ambiguous: Other subtype acting as PM class 2.80, Mid/series vs Adv/Pres 1.01, YR/Count twin .78, radar long Adv .43 | **5.03 (44 %)** | 4.87 (38 %) |
| C fixable: short-sample miss 1.49, phase wrong .23, systematic PM confusions 1.67, systematic ->Other .74 | **4.13 (36 %)** | 4.04 (32 %) |
Cross-check (excess error over print-high/pass rate .104): config-only +1,536 errors, unhealthy/no_data +695 — same scale as A.
## What it means for the next experiments
* Ceiling is labels + definitions, not capacity: ~7.4 pt of 11.5 is A+B. Cheapest real gain: drop / relabel the Dec rows
  of drifted channels in training and scoring (use the Dec-era config where it differs from the 2026 print) — needs an
  orchestrator decision (it is a scoring-rule change); A3 config-only confident misses are a review list.
* Fixable part, by lever: (1) short samples 1.5 pt (m5–h1; Pres->Other, Adv<->Pres, YR<->Count) -> TCN / mixed-length
  training; (2) systematic PM confusions 1.7 pt + ->Other .7 pt, 70 % radar, per-detector consistent -> same-phase /
  same-lane context (lanes as inputs, joint role decoding: one Count per lane, count-differs YR) and radar-specific
  features; (3) phase .2–.4 pt -> not worth a function-side fix. Longer samples beyond 3 h: nothing to gain.
* B is only partly reducible: superseded loops and identical YR/Count twins are unidentifiable from the log; long / advance-
  presence zones vs Advance/Presence need a definition decision (or report them as their nearest class), not a model.
