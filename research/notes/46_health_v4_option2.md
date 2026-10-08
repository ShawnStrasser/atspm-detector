# 46 — Detector health v4 (the user's 2026-09-29 review ideas) and the "option 2" test (2026-09-29)
Code `research/code/health/` (`health_core` + `h4_*`); artefacts `%DC_WORK%/health4/`, deliverable
`%DC_WORK%/health/health_v4.parquet` (66/72 h, both periods). Training signals only (locked_v2 absent); no fault events.
Inputs, all out of fold: phase + `top_prob` (frame v6), function probabilities (function_v3d recipe OOF), lanes spanned
(note 42 `lane_output` on the OOF predictions; its pair model is in-sample, used only to pick a rapid limit).
## Ideas -> kept / dropped (flags `health_core.OPTS`; each gated by the data it needs, listed in `not_checked`)
* (a) **Recurring spikes** KEPT: 15-min spikes (z >= 3, >= 5 extra ONs vs the phase reference) that recur within +-30 min
  on other dates and whose actuations look like vehicles (ON->ON < 1 s share under the group limit, ON time 1/3-3x
  usual) cap `choppy` at suspect with 2-3 days of data; cleared (ok, noted) only from >= 4 days and 3 dates; implausible
  spikes say so. 24 h windows: not checked. Spike periods of a choppy call are listed as bad periods.
* (b) **Recovery vs the tracking partner** KEPT: recovery share judged against the same-phase sibling (twins included)
  with the best 15-min correlation outside the episode (>= .7), else the phase. 03033 d8 -> 96 % vs d10 (was 46 %).
* (c) **Group failure** KEPT (text only): a shared stuck-on (>= 3 held ON) where a same-phase sibling kept counting
  (>= 50 % of its share) is reported as sensors failing together, not a queue; still capped at suspect; days repeated.
* (d) **Rapid in the worst hour** DROPPED (`rapid_hour`): at equal FA vs loosening +2 / 0 catches (24 h / 66 h).
  2B044 d14 has none: worst hour 6-7 % ON->ON < 1 s (limit 16 %).
* (e) **Lane-aware rapid limits** KEPT: spanning 2+ lanes -> max(limit, p99.8 of healthy spanning detectors);
  rapid FA .42 -> .37 % of healthy windows, no positive lost; lanes-shuffled control loses it. 07035 d19 = 1 lane.
* (f) **Rolling consistency** DROPPED as a rule (below). (g) **Model confidence** DROPPED as an input (P(Bike)-weighted
  silence, phase reference only if P >= .5: -0.4 pt catches at 2 h, no FA gain); kept as a note in the reason.
## Result, % flagged (bad or suspect), Sept 2026 windows 2 h / 6 h / 24 h / 66 h (v3 = OPTS off, same inputs)
| group (n at 66 h) | v3 | **v4 kept** | v4 all options on |
|---|---|---|---|
| presumed healthy = FA (8,666) | 1.00 / 1.10 / 2.33 / 3.90 | **0.95 / 1.06 / 2.28 / 3.88** | 0.98 / 1.13 / 2.42 / 4.32 |
| % bad on healthy | .38 / .40 / .73 / .92 | **.37 / .39 / .71 / .70** | .37 / .40 / .76 / .78 |
| dq health (87) / label-check (74) / card erratic (19) | 24.1/27.0/63.2/97.7; 21.2/24.3/62.2/97.3; 14/13/74/100 | identical | +1 at 24 h |
| dead groups / share fell > 5x since Dec (43) | 92-100 / 7.0-9.3 | identical | identical |
1 detector = 1.1-1.4 pt at 87/74. User answers (19, later answer wins): v3 18/19 (03033 d8 bad, he said suspect),
**v4 19/19**; remarks 3/5 unchanged (2B422 d8, 11042 d22: lane-pairing issues, not health rules).
66 h v3 -> v4: bad -> suspect 39 (partner recovery, recurring spikes), suspect -> bad 3, suspect -> ok 5 (lanes).
**Stability, new independent period** (late Sept pull, Sat 26 Sep 16:15 - Mon 28 Sep, 56 h, not contiguous): of
16,873 detectors, bad stays bad 94.3 %, ok stays ok 96.9 %; suspect is episodic (40 % stay suspect, 53 % -> ok).
Function accuracy (set A OOF) by v4 status ok .875 / suspect .833 / bad .736 (realistic .887 / .848 / .739). Runtime
(loaded machine) 0.33 s per signal at 2 h, 1.7 s at 66 h (v3 on the same inputs 0.34 / 1.24 s).
## (f) Rolling function consistency, out of fold (`h4_rolling`: 6 saved fold models, max diff to saved OOF 0.0)
Function head re-run on 33 consecutive 2-h windows of every signal (phase fixed at the OOF site answer); windows
< 20 ONs skipped (22 %); per detector tv to its own median probabilities, flips, tv minus the signal's median.
* Real problems NOT already flagged: 66 h 0 of 31 caught at +0.5 / +1 / +2 pt FA; 24 h 0-1 of 47 / 62. AUC .58-.65.
* Prospective (non-circular): Sept 18-21 ok -> bad on Sept 26-28: top 5 % inconsistent 1.2 % vs .5 % (volume-matched),
  at the shuffled control's p97.5 (1.1 %). Earlier screen on the frame's fixed windows: Dec 2024 ok -> bad Sept 2026
  7.0-7.6 % vs 3.5 % (shuffled p97.5 5.5-5.8 %). A real but weak pre-failure hint (~2x), like note 40's `rapid`.
* 2B146 d16 (his spike case) is Count in all 33 windows: its behaviour does not change inside the spikes.
## PART 2 - option 2: classify -> health v4 -> remove bad PERIODS -> classify again
Periods: 5,506 (spike 3,454, stuck 1,226, silent 606, rapid-hour 220 - run on the all-options file); 2,921 signal-
windows of the frame touch one. Function: package path, phase fixed at the frame's OOF, fold models (pass1 = frame OOF
argmax on 98.1 %). 2a = the detector's own ONs in its periods removed; 2b = periods cut for every channel.
| set (rows) | base | 2a, 95 % CI | 2b, 95 % CI | own-period rows (n) 2a / 2b |
|---|---|---|---|---|
| A everything (268,861) | .8714 | **-0.017 [-0.031, -0.006]** | -0.035 [-0.058, -0.011] | 2,371: -1.5 / +0.55 pt |
| realistic: A minus label-check fail / misconfigured (261,622) | .8834 | -0.018 [-0.031, -0.005] | -0.038 [-0.063, -0.015] | 2,290: -1.5 / +0.6 pt |
| C clean-label (54,386) | .9079 | -0.029 [-0.072, +0.004] | -0.026 [-0.064, +0.005] | 279: -2.2 / 0.0 pt |
By window (A, 2b) 30 min -0.09, 6 h 0.00, full +0.10 pt. Upper bound before running: all errors on own-period rows =
0.11 pt (v3 periods) - far below the 1-pt bar. Phase (no fold phase models: 71 released signals, never trained on,
final_v1 trees + stage-13 GRU blend, 139 signal-windows): 2a changes no answer; 2b -0.16 pt at 30 min, +0.05 at
6 h, 0 at full. Bound over the whole frame (errors in touched signal-windows): 30 min 0.17 pt. **Dropped** - no
subgroup clearly helped (own-period rows 2b +0.55 pt is 13 rows net). Cost: a second full pass (package 0.55 s per
signal at 30 min, 2.1-2.5 s at 3-24 h) for every signal-window with a period, plus health.
**Spot-check** `review/spotcheck_health.xlsx` rebuilt: 8 rows (his 5 questions answered: 2B146 d16, 03033 d8, 10045 d4, 2B044 d14,
07035 d19; new: 10018 d2 partner recovery at 52 %, 10090 d22 daily stuck -> bad, 2B067 d42 lane-relaxed -> ok);
charts `v4_*.png`; all 25 earlier answers in sheet "earlier answers" (backup `health4/spotcheck_health_v3_backup.xlsx`).
