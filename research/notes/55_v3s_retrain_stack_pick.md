# 55 — v3s labels retrained + stacked-group pick rule in the per-lane decode (2026-09-30)
User rules 2026-09-30 (AGENTS.md "ATSPM classes, one per lane"): stacked members all carry the real class; loser =
next free class; pick = (a) unhealthy loses, (b) a zone spanning all lanes is not a lane, lane-by-lane wins,
(c) the advance tracking the stop-bar Count total wins. CPU only, OOF six folds, locked_v2 asserted absent everywhere.
`model/` untouched. Outputs `%DC_WORK%/trackA/atspm54/v3s/score.*`, `%DC_WORK%/trackA/atspm55/v3s/pick.json` (+ `pick_globalkeys.json`).
## 1. v3s labels + function retrain (note 54's recipe, nothing new)
* `stack_labels_v3s.py`: 29 groups applied (all unreviewed = rule applies), 41 rows relabelled, 7 YR==Count out,
  33 radar-over-loop signals re-admitted (980 rows). Training 297,280 rows / 692 signals (= note 54 dry run).
* `v3_retrain.py fit --frame v6e --labels v3s ... --dec-role --dec-role-rule fp`, 3 seeds -> run
  `run_51ba131222_exclude_min5_clean_valnc_h3_drfp` (~23 min). `ln5_extend.py lanes --tag v3s` (456,033 det-windows, lane known .771).
* `atspm_score.py` (note-54 step 4 rows: v3s truth, Dec fp rule, stack credit, decode gated >= 30 min):
| run | n E | ATSPM E / R | acc7 E / R | stack_extra E | secondary E |
|---|---|---|---|---|---|
| v3e (note 54) | 261,528 | .8892 / .9015 | .8827 / .8952 | 412 | .957 |
| **v3s retrain** | 261,528 | **.8910 / .9036** | .8845 / .8972 | 426 | .956 |
  Paired signal bootstrap: ATSPM +0.18 [+0.08, +0.30] E, +0.21 [+0.11, +0.31] R; acc7 +0.18 [+0.07, +0.28]. Every
  window up (5 min .8590 -> .8612 ... full .9090 -> .9112). Below the 1-pt function bar: a label-rule change, not a model gain.
  Decode vs argmax on the new run: +0.19 [-0.05, +0.47]; full window +0.84 [+0.38, +1.33] (= note 54).
## 2. Pick rule (`lanes/atspm_decode.py` `pick=`, inputs `lanes/ln6_pick.py`, eval `evaluation/atspm_pick55.py`)
* Inputs, hi-res only, per window >= 30 min (17 min, 5 workers): health_core v5 run ON THE WINDOW (inputs = predicted
  phase / OOF probabilities / ln5 lanes): ok 248,071, n.e.d. 45,839, suspect 4,660, bad 2,322, not_run 3,540 (health_core
  KeyError 'pulse_frac' on 164 short windows - a health_core bug, left). Span: covers >= 2 mutually non-co-located
  detectors of its phase (chance-corrected ON coincidence >= .5 at best lag within +-2 s), union explains >= .15 more of
  its ONs than the best one, more actuations than each -> 4,869 flags (1.6 %). Track: corr of binned counts with the
  predicted-Count total (lane's, else phase's). Co-location peers (either way >= .5) define a hi-res "stack".
* Decode: first-choice ATSPM claims settled first; inside a stack, order = (healthy, not spanning, track [Advance], P);
  stacks taken by their best P; a spanning A/P/C zone is put on all its covered detectors' lanes. Losers -> next free class.
* First version applied the keys to ALL first-choice claims ("pick_global"): -0.17 pt [-0.27, -0.10] -> scoped to stacks.
| decode (step-4 rows) | ATSPM E | vs greedy, pt [CI] | ATSPM R | vs greedy R | stack_extra |
|---|---|---|---|---|---|
| greedy (note 54) | .8910 | - | .9036 | - | 426 |
| first-choice-first, P only (control) | .8910 | 0.00 | .9036 | 0.00 | 426 |
| **pick (a+b+c, stack-scoped)** | .8904 | -0.06 [-0.12, -0.01] | .9031 | -0.05 [-0.12, +0.01] | 435 |
| pick_a / pick_b / pick_c alone | .8910 / .8906 / .8909 | -0.00 / -0.04 / -0.01 | | | |
| pick_global | .8893 | -0.17 [-0.27, -0.10] | .9019 | -0.17 | 432 |
| pick, inputs shuffled in group | .8840 | -0.70 [-0.83, -0.60] | .8965 | -0.71 | 426 |
  Per seed (each seed's OOF decoded alone) pick - greedy: -0.07 / -0.05 / -0.07 E (seed spread of greedy itself .8897-.8904).
  Stacked members >= 30 min: greedy .806 -> pick .786 (stack_extra 206 -> 215).
* What the pick chooses (EVALUATION labels only: print lanes_spanned, print technology, dead / dq_suspect / validated
  'unhealthy'). Contests = >= 2 first-choice claimants of one class on overlapping lanes; "true stack" = >= 2 with that truth.
| contests | greedy | pick | shuffled |
|---|---|---|---|
| true stack, print span vs 1-lane (868): lane-by-lane only / spanning only / both (extra lane) | 682 / 116 / 70 | **768 / 65 / 35** | 706 / 101 / 59 |
| all, same (1,934) | 1,244 / 582 / 108 | 1,397 / 476 / 61 | 1,279 / 548 / 90 |
| true stack radar vs loop (262): loop / radar / both | 81 / 155 / 26 | **167 / 82 / 13** | 128 / 110 / 24 |
| healthy member only, health labels differ: true stack 332 / all 1,019 | .783 / .741 | .783 / .706 | .753 / .705 |
* (a) does nothing useful: the window health flag hits 175 of 1,443 label-unhealthy contest members and 392 label-healthy
  ones (labels are whole-period print / DQ facts; the flag is per window). (b) works as meant: lane-by-lane wins 79 -> 88 %
  of true-stack span contests, extra lanes halved; span flag on print-spanning 557 vs print-single-lane 345 members.
* Inferred lane count (single-lane ATSPM-decoded detectors) vs print n_lanes, phase-windows >= 30 min, exact / over / under:
  all 62,846: greedy .803 / .059 / .138 -> pick .802 / .059 / .139 (raw lane output .805); stack phases 567: .623 / .155 /
  .222 -> .624 / .150 / .226; span-flagged phases 4,255: .752 / .082 / .166 -> .742 / .076 / .182 (fewer extra lanes, but
  flags on real single-lane zones remove real lanes).
## Verdict / caveats
* v3s retrain: +0.18 / +0.21 pt ATSPM, CI above 0; adopt as the function_v3s baseline (nothing ships during the search).
* Pick rule: does the user's job on stacks (loop over spanning radar) at -0.06 pt ATSPM (CI just below 0, consistent over
  seeds - real but tiny). (a) health inert as built; (c) neutral. Keep `pick` (stack-scoped) as an option; default stays greedy until decided.
* Lane-count plausibility is not improved overall; the span flag's false positives (~38 % of flags on 1-lane zones) are the limit.
