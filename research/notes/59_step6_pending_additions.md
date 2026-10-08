# 59 — Pending function additions on the current set-up (validation plan step 6) (2026-10-01)
Baseline = current set-up: function note-57 arm (229 features, 3 seeds, blend phase input, frame v6e); lanes = joint model D
(note 58, `lanes/ln8/lanes_D.func`); decode = stack-scoped pick, stack_loser nonatspm_ap; pick inputs REBUILT on D lanes +
the 229 arm (`lanes/ln6_pick_D229`, `ln7_stackhealth_D229`). ATSPM-only stack-aware score on note-55 step-4 rows (261,528 E /
254,540 R det-windows), six folds OOF, paired signal bootstrap (pt). CPU, 6 threads. locked_v2 asserted absent; nothing shipped.
Code `trackA/s59_step6.py`; work `%DC_WORK%/s59/` (score_*.json, agg_pooled.json, fit/<cfg>/, inner/, nlanes/, lanefeat/).
## Baseline (exact)
* 3 seeds: ATSPM **.8914 E / .9042 R**; by window E m5 .8607, m10 .8732, m30 .8907, h1 .8908, h3 .9062, h6 .9080, h24 .9122,
  full .9135. Old (v3s-lane) pick inputs: -0.003 [-0.01, 0.00] -> note 57/58's approximation was harmless.
* Single seeds s0 / s1 / s2: .8905 / .8904 / .8903 E (-0.08..-0.11 vs the 3-seed mean). Screen = config seed 0 vs base seed 0.
## Additions (one fold-set = six folds x seed 0, each with its control; bar = +1 pt ATSPM)
| addition | what | ATSPM E (R) | delta E [CI] (R) | 5/10 min pooled | >= 30 min pooled |
|---|---|---|---|---|---|
| base @s0 | - | .8905 (.9034) | - | - | - |
| (a) lanes | 13 cols: phase n_lanes (+conf), lanes spanned, lane conf, lane rank, n det on phase, n lane-mates, volume rank and log ratio to busiest mate, mates' max P(A/P/C/YR); NESTED (below) | .8907 (.9034) | +0.02 [-0.09,+0.11] (+0.01) | -0.37 [-0.51,-0.24] | **+0.17 [+0.04,+0.29]** (R +0.15 [+0.03,+0.28]) |
| (a) shuffled control | same columns permuted across rows | .8905 (.9033) | -0.01 [-0.08,+0.07] | -0.05 | +0.01 [-0.07,+0.09] |
| (b) eta | print subtype `eta` (radar) -> 8th non-ATSPM class: 622 training rows / 29 detectors | .8895 (.9024) | -0.10 [-0.17,-0.03] | -0.21 [-0.33,-0.10] | -0.06 [-0.14,+0.02] |
| (b) etaadv | eta + radar advance_presence: 9,132 rows / 338 detectors | .8898 (.9025) | -0.08 [-0.19,+0.02] | -0.19 | -0.03 [-0.15,+0.07] |
| (c) health | 16 cols, FUNCTION-FREE: health_core v5 on the window (score, status, n_families, 7 rule scores; predicted phase only) + note-56 stack stats vs a function-free reference (chi, surge, drop, chatter, rel. chi, stack size) | .8900 (.9028) | -0.05 [-0.13,+0.02] | -0.14 [-0.25,-0.01] | -0.02 [-0.11,+0.07] |
| (c) shuffled control | | .8894 (.9022) | -0.12 [-0.20,-0.04] | | |
| (d) offpeak | 12 cols on the off-peak part only (outside weekday 06:30-09:00 / 15:30-18:30), NaN if < 15 min of it or < 5 ONs: rate, rate / whole-window rate, pulse share, ON dur median / p90, ON-in-red share, red / green occupancy, ONs per green, first-ON lag, ON at green start, last-OFF share of green | .8901 (.9029) | -0.04 [-0.12,+0.03] | -0.01 | -0.06 [-0.13,+0.02] |
| (d) shuffled control | | .8895 (.9023) | -0.10 [-0.17,-0.03] | | |
| (d) opw | training rows weighted 1 + off-peak share of their window (mean share .63) | .8897 (.9025) | -0.08 [-0.15,-0.01] | -0.19 [-0.32,-0.08] | -0.04 [-0.12,+0.05] |
By window (E, delta pt vs base@s0; full table in score_screen1/2.json): lanes m5 -0.33, m10 -0.41, m30 +0.09, h1 +0.23 [+0.05,
+0.40], h3 +0.15, h6 +0.14, h24 +0.29 [+0.10,+0.49], full +0.13; health / offpeak / eta / opw: every window within -0.27..+0.10.
Refit noise is visible: the shuffled controls land at -0.10..-0.12 overall (5/10 min -0.1..-0.3) - a single-seed refit of
this head moves short windows by ~0.2 pt on its own.
## Nesting for (a) (D uses function features + OOF function probabilities)
* Outer fold k's TRAINING rows get lanes built with no fold-k label: function (229 arm) refit on folds not in {k, j} (early stop
  on a third fold), 3 seeds -> probs for fold j (90 fits, 56 min); D pair model refit on labelled pairs of folds not in {k, j}
  with those probs (3 seeds) -> pairs of fold j; lane decode of fold j (note-58 lam for fold j, span prior from folds not in
  {k, j}). Fold-k TEST rows get the ordinary OOF D lanes + OOF probs. Lane known 0.88 per nested set (= D); nested vs ordinary
  lanes on the same rows: n_lanes equal 95-96 %, mates' P(Count) r .98. Lane features exist only >= 30 min (59 % of rows).
## Coverage
Off-peak features on 56 % of rows (m5/m10 0 %, m30 43 %, h1 73 %, h3 86 %, h6-full 97-99 %); the Sept sample is two-thirds
weekend, so for most Sept windows "off-peak" = the whole window (only m30_a/d, h3_a partly peak). Health features 67 % of rows
(windows >= 30 min; stack stats 20 %). ETA labels exist for only 29 detectors (print subtype `eta`).
## Verdict (keep rule: new additions must clear the 1-pt function screen; nothing clears it -> all dropped, not tuned)
* (a) lanes as inputs: DROPPED by the bar, but the only real signal here: +0.17 pt on >= 30-min windows (CI clear, control
  flat), paid for by -0.37 pt on 5/10-min windows where the columns are absent (the trees re-split on them). Worth recording:
  the per-lane decode already captures most lane value (note 58); the extra is ~0.2 pt, far under 1 pt.
* (b) radar-ETA class: dropped. 29 labelled detectors are too few; adding advance_presence zones does not help either. ATSPM-
  wise neutral to slightly worse (short windows).
* (c) health as input (function-free, so no circularity): dropped; indistinguishable from its control at >= 30 min.
* (d) off-peak-only features and off-peak weighting: dropped; no window gains. In this sample off-peak mostly equals the whole
  window, so the features largely duplicate existing ones - a weekday-heavy pull would be the fairer test.
## For the orchestrator
* Baseline numbers above (.8914 / .9042 with rebuilt pick inputs) are the exact step-6 reference for step 7.
* Possible follow-up (NOT done, it would be tuning): lane features gated to >= 30 min by a separate head / no short-window
  rows in that head would keep the +0.17 without the short-window cost; still ~0.2 pt, under the bar.
