# 56 — Pick rule as default; stack-relative health key; tighter span flag (2026-09-30)
Orchestrator decision on note 55: the stack-scoped pick rule is the DEFAULT decode (user wants the healthy /
lane-by-lane member; -0.06 pt ATSPM accepted). CPU only, OOF six folds, run `run_51ba131222_..._drfp` (v3s), lanes
`ln5_lanes_v3s`; locked_v2 asserted absent everywhere. `model/` untouched. Out: `%DC_WORK%/trackA/atspm56/v3s/pick56.json`.
## 1. health_core KeyError 'pulse_frac' (research copy only; model/health.py does not use health_core)
* Cause: windows where NO detector reaches min_on durations (night h1_b) -> act_stats frame has no pulse_frac /
  med_dur columns -> rapid_ratio() KeyError (287 windows in note 55's log). Fix in `health/health_core.py`: reindex the
  ACOLS columns before rapid_ratio. Re-tested on 002e2655 dec h1_b: 13 x not_enough_data, no error.
## 2. Stack-relative health key (`lanes/ln7_stackhealth.py`, 89,904 stack-member windows >= 30 min, ~7 min, 4 workers)
* Each hi-res stack (co-location / spanning cover, ln6's definitions) gets ONE independent reference R = counts of
  the phase's detectors outside the stack (predicted Count first, else any, else rest of signal); per member: chi
  (Poisson dispersion of binned counts around a_d*R), surge, drop, chat, lvl (vs same-lane Count). Flag `unh2` =
  chi >= 10 and >= 3 x (best partner's chi + 0.5). Rule chosen among 6 candidates by the labels below (mild selection).
| stack members >= 30 min (label-unhealthy 3,958 / 89,855) | flags | hit | false-flag | precision | flag on the unhealthy one, label-differing stacks |
|---|---|---|---|---|---|
| old: health_core window status bad/suspect | 1,794 | .086 | .0169 | .191 | .582 (378 flags; base .291) |
| **new unh2 (chi rel.)** | 1,262 | .083 | .0109 | .261 | .653 (498 flags) |
| surge / drop / chatter / lvl alone | | .057 / .034 / .016 / .004 | .009 / .016 / .005 / .009 | .23 / .09 / .13 / .02 | dropped |
  Labels (dead / dq_suspect / validated 'unhealthy') are whole-period; hit rates stay low for both keys.
* Signal 13025 P6 (det 39 radar, spikes up to 219 ONs / 5 min while loop 28 has 0; chi 132 vs 0.3 full window):
  old key flagged BOTH 28 and 39 in h24_a / full66 (cancels); unh2 flags only 39 (h3_a, h6_a/b, h24_a/b, full66).
  Det 28 holds Advance in 9 of 14 windows >= 30 min under pick (greedy: 1); in the other 5, 28's own first choice is
  Other / Yellow_Red (model, not the pick). The losing radar 39 goes to Yellow_Red (next free class) = an ATSPM error.
## 3. Span flag v2
* Side-by-side cue: excess (over chance) near-simultaneous (<= 1 s) ONs between the covered peers (`coinc_x`, per
  peer ON): two lanes see side-by-side vehicles, two pieces of ONE lane do not. span2 = span1 and coinc_x >= .02.
  The "merge" test (spanning zone shows ONE ON per side-by-side pair) and peak divergence r_hi < r_lo did not separate
  (radar zones count both vehicles); a single-covered-peer variant (radar over ONE loop) had precision .29 -> dropped.
| span flag vs print lanes_spanned (windows >= 30 min) | all: flags / prec | pred A/P/C: flags / prec | pred Advance |
|---|---|---|---|
| span1 (note 55) | 3,305 / .809 | 1,011 / .530 | 451 / .339 |
| **span2** | 2,031 / .923 | 378 / .643 | 94 / .457 |
  Most span1 false flags = advance loops "covered" by 2 unlabelled sub-zone channels splitting the same lane (e.g. 01069).
## 4. Re-score (step-4 rows, 261,528 E / 254,540 R; CI = paired signal bootstrap, pt)
| decode | ATSPM E | vs greedy | vs pick55 | ATSPM R | stack_extra | stacked members | lane-by-lane / spanning / both (868) | loop / radar / both (262) |
|---|---|---|---|---|---|---|---|---|
| greedy (note 54) | .8910 | - | | .9036 | 426 | .8055 | 682 / 116 / 70 | 81 / 155 / 26 |
| pick55 (h-core + span1) | .8904 | -0.06 [-.12,-.01] | - | .9031 | 435 | .7863 | 768 / 65 / 35 | 167 / 82 / 13 |
| **pick_h2 = new default** | .8904 | -0.06 [-.12,-.01] | 0.00 [-.00,+.01] | .9031 | 435 | .7869 | 767 / 66 / 35 | 166 / 83 / 13 |
| pick_h2s2 (span2 for both) | .8909 | -0.01 [-.05,+.02] | +0.05 [+.01,+.10] | .9036 | 441 | .7835 | 745 / 67 / 56 | 156 / 83 / 23 |
| pick_h2s12 (span2 lanes, span1 key) | .8909 | -0.01 [-.06,+.03] | +0.05 [+.02,+.08] | .9036 | 441 | .7835 | 750 / 64 / 54 | 156 / 83 / 23 |
| pick_h2s2, inputs shuffled | .8863 | -0.47 [-.57,-.38] | -0.41 | .8989 | 428 | .7959 | 709 / 94 / 63 | 129 / 108 / 25 |
  Per seed (E): greedy .8901/.8897/.8904; pick55 .8894/.8892/.8897; pick_h2 .8894/.8892/.8898; pick_h2s2 .8899/.8896/.8903.
  Healthy-member-only (true stacks, health labels differ, 332): greedy .783, pick55 .783, pick_h2 .780, h2s2 .777.
* Lane count exact / over / under vs print: all 62,846 phase-windows greedy .803/.059/.138, pick55 .802/.059/.139,
  h2s2 .803/.059/.138; stack phases (567) .623 / .624 / .623; span1-flagged phases (4,255) .752 / .742 / .753;
  span2-flagged (2,802) .762 / .749 / .765. span2 removes pick55's lane-count loss on flagged phases.
## Verdict
* Keep h2 (neutral on ATSPM, better flag by every label measure, fixes the 13025 double flag): default (a) = unh2.
* span2: helps ATSPM (+0.05 vs pick55, CI > 0) and lane counts, but HURTS the user's stack goal (lane-by-lane wins
  768 -> 745-750 of 868, loop-over-radar 167 -> 156, stacked members .7863 -> .7835). Not in the default; kept as
  inputs (`sp_*` in ln7) + `span_key` option in atspm_decode for the orchestrator.
* Default wired: `atspm_decode.stack_pick()` + DEFAULT_PICK; `atspm_score.py` makes `pick_stack` PRIMARY when the ln6 /
  ln7 inputs exist (`--no-pick` = note 54). Verified: .8904 / .9031, 1,262 health flags (= pick_h2).
* Finding: the pick's ATSPM cost is the LOSER rule, not the keys: stacked-member A->wrongA 53 (greedy) -> 72 (pick55) /
  70 (h2s2): the losing member falls to its next free ATSPM class (e.g. radar 39 -> Yellow_Red). Sending in-stack
  losers to their best non-ATSPM class is the obvious next test (user had agreed "next free class" - decision needed).
