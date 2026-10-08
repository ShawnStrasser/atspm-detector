# 57 — Trees optimised on their own (validation plan step 2) (2026-09-30 / 10-01)
Phase = LightGBM pair ranker (+ joint decoder), no network; function = 7-class LightGBM head, v3s labels, note-55 recipe.
Six folds of `folds_v4` everywhere (phase pool: 22 signals move vs note 37's phase map). CPU, 6 threads, serial.
Code `lightgbm/t57_phase.py`, `trackA/t57_function.py`; work `%DC_WORK%/trees57/`. locked_v2 asserted absent; nothing shipped.
* Phase pool = note 37's (772 signals with rows, 1,670,485 labelled rows, 261 features, 22 windows), recipe unchanged.
  Two numbers per note 48: everything = timing label, >= 5 actuations; realistic = minus label-check fail / misconfigured
  and high-confidence print-phase disagreements, switch / additional-call phase credited (labels v3s). Per window family
  = mean of its windows; paired signal bootstrap (2,000). 30 min n = 47,258 E / 45,284 R det-windows.
* Function = note 55 step-4 rows (261,528 E / 254,540 R), ATSPM-only stack-aware score, per-lane decode with the
  stack-scoped pick rule, gated >= 30 min, **stack_loser = nonatspm_ap (note 56b default) for EVERY function number
  here** (all re-scored in one pass after the default changed). Lanes / pick inputs = the v3s baseline's for every config
  (approximation; recomputing them costs ~35 min a config). Baseline = the note-55 v3s run itself (reproduced .8904/.9031
  under the old default). Seeds: 1-seed variants compared with seed 0 of the baseline (paired).
## Phase (noise: single-seed decoded spread at 30 min .9683-.9688, ~0.03 pt sd)
| arm (E; R in brackets) | 5 min | 30 min | 6 h | full |
|---|---|---|---|---|
| full 261 feat, 3-seed bag + decoder | .9436 (.9470) | **.9696 (.9723)** | .9790 (.9818) | .9814 (.9836) |
| same, no decoder (ranker bag only) | -2.39 [-2.68,-2.11] | **-1.53 [-1.75,-1.32]** | -0.63 | -0.39 |
| same, single seed (s0 / s1 / s2) | -0.12 / -0.13 / -0.04 | -0.11 / -0.09 / -0.13 (CI < 0) | -0.02 / -0.01 / +0.05 | -0.01 / -0.02 / -0.09 |
Leave-one-group-out, seed 0, decoded, delta pt vs full seed 0 [95 % CI], everything (realistic within 0.04 of it):
call 43/44 (39 col) m5 -18.1, **m30 -12.2**, h6 -8.4, full -7.2 · queue/release (45) m30 **-0.20 [-0.33,-0.09]**, full -0.17 ·
duration (28) m30 -0.09 [-0.20,+0.01], h6 -0.12 · cross-candidate rank/z/gap (112) m30 -0.01, full -0.10 · exclusivity (52)
-0.01 · timing histograms (34) +0.04 · partner diff (20) -0.02 · signal context (13) +0.05 · condition-dependent coord/free
(10, NaN when absent: 44-70 % NaN at 30 min) -0.00 (full -0.10 [-0.21,-0.01]) · NOISE control (+40 shuffled copies of real
columns) m5 -0.15, m30 -0.04, full -0.09 => single-group moves under ~0.15 pt are noise-sized.
**Joint drop** of xcand + excl + timing_hist + pdiff + signal_ctx + cond (261 -> 68 features), 3-seed bag + decoder vs full:
m5 +0.16 [-0.03,+0.34], **m30 -0.12 [-0.24,-0.01]** (R -0.11 [-0.23,+0.01]), h6 -0.11, full +0.01; single seeds m30 -0.15..-0.19.
## Function (seed noise: s0 / s1 / s2 ATSPM E .8891 / .8888 / .8894)
| arm (3 seeds unless @s0) | ATSPM E | ATSPM R | delta E pt [CI] | 30 min / full delta |
|---|---|---|---|---|
| full 442 feat (v3s baseline) | **.8901** | **.9029** | - | - |
| single seed (s0) | .8891 | .9019 | -0.10 [-0.14,-0.06] | -0.19 / -0.01 |
| trees-only phase input (frame v6t) | .8888 | .9016 | -0.13 [-0.21,-0.06] | -0.15 / +0.04 (5 min -0.30) |
| ATSPM rows weight 2 (Other/Mid/Bike keep weight 1) | .8906 | .9032 | +0.05 [-0.06,+0.17] | +0.08 / +0.00 |
| decision rule: non-ATSPM classes pooled (no refit) | .8902 | .9030 | +0.01 [-0.02,+0.04] | - |
| **drop pp_xcand+pp_pdiff+pp_v2+yr+ratio+top_prob (229 feat)** | .8897 | .9027 | -0.04 [-0.14,+0.06] | -0.16 / +0.11 |
| same, single seed (s0 / s1 / s2) vs full 3-seed | .8890-.8889 | .9019-.9018 | -0.11..-0.12 | m30 -0.17..-0.31 |
| + drop pp_core (colour shares; 213 feat) | .8852 | .8982 | -0.49 [-0.68,-0.30] | -0.67 / -0.54 |
Leave-one-group-out, seed 0, vs full seed 0 (E; R same within 0.02): lag (advance->stop-bar lag, 28) **-1.85 [-2.19,-1.54]** ·
sibling (73) -0.44 [-0.61,-0.29] · expert A2 px (54) -0.16 [-0.24,-0.08], px SHUFFLED control -0.12 [-0.20,-0.04] (the px
signal is real but ~0.15 pt) · condition-dependent coord/free (5 + ranks) -0.10 [-0.18,-0.02] · pp_core -0.04 · pp_xcand +0.04 ·
pp_pdiff +0.08 · pp_v2 +0.01 · yr +0.03 · ratios +0.04 · top_prob +0.08. Decode for reference (3 seeds): greedy +0.09 vs pick,
argmax -0.10 (full window -0.72 [-1.22,-0.24]).
## Verdict (bars: phase 0.3 pt at 30 min, function 1 pt; "real" = CI clear of 0)
* Phase: keep the **joint decoder** (+1.5 pt at 30 min, +2.4 at 5 min: the only component far above the bar) and the call,
  queue/release, duration and colour-share features. The other six groups cost nothing measurable alone; dropped together they
  cost 0.12 pt at 30 min (real, under the bar). 3-seed bagging: +0.09-0.13 pt at 30 min (real, under the bar, 3x tree cost).
  **Simplest not beaten by the bar: 68-feature ranker, 1 seed, + decoder** (30 min .9677 E / .9705 R vs .9696 / .9723 for the
  261-feature 3-seed bag). Phase-call events are the trees' backbone: without them 30 min falls to .846 (portability risk for
  agencies that do not log 43/44).
* Function: lag and sibling families pay (lag above the 1-pt bar); px, coord/free real but small; six families add nothing.
  Trees-only phase input costs 0.13 pt (real; 5 min -0.3) - the blend input is a wiring nicety, under the bar. ATSPM
  weighting and pooled decision rule: nothing. 3-seed averaging +0.10 (real, under the bar).
  **Simplest not beaten by the bar: 229-feature head, 1 seed** (ATSPM .8890 E / .9019 R, -0.11 pt vs the 442-feature 3-seed).
* Both arms saved as OOF; if "beyond noise" rather than "the bar" is the keep rule, the full 3-seed arms are the trees side.
## OOF for the head-to-head (`%DC_WORK%/trees57/`) and caveats
* Phase `phase/<cfg>/oof.parquet` (KEY4, p0/p2 per seed + bag) for `full/` and `drop_xcand-excl-timing_hist-pdiff-signal_ctx-
  cond/`; function `function/v6e/<cfg>/P_first.all.wi_s{s}_f{k}.npy` (frame v6e fold order; full = note-55 run dir), `v6t/full/`.
* Lanes / pick inputs fixed at the baseline's; v6t = note-37 trees-only phase OOF (261 feat, old fold map). Phase rows now on
  folds_v4, so not row-identical to note 48.
