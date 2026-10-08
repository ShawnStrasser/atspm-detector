# 114 — Health fully out of classification: stacker and stack pick without pk_unhealthy (2026-10-07)
Brief (user): health must be a separate output only. v6b's function stacker read `pk_unhealthy` (note 56 stack-relative
chatter flag, pick.py) and the per-lane decode used it as the first stack-pick key (healthy first). Test removing both.
CPU 4 threads, saved OOF, fit111 / f109 recipe (1-seed trees OOF, Sept-2026 rows, v4q, folds_v4, PRM0 x 150), stacker
seeds 0-2, locked_v2 asserted absent, exam signals not opened. Code `research/code/final114/f114.py`; work `%DC_WORK%/s114/`.
## 1. What in the classification path reads health (v6b package, file:line)
- function_stage.py:232/237 `PK.unhealthy(PK.stack_health(...))` -> `pk_unhealthy` -> stacker column (stacker.py:107,
  stacker.json feature 23 of 47) AND decode key (function_stage.py:250 -> atspm_decode.py:302 stack_pick -> :149/:157).
- function_stage.py:239-244 `_health_context` (:85-119): health_core run already skipped (stacker.needs_health_core False,
  stacker.py:133), but `stack_health` still ran (:110) and its chi / surge / drop / chat / rel_chi were thrown away;
  only `hf_clus_n` (stack size = co-location structure, not health) was read. health_core imported (:23) for hc.ALLOWED.
- predict.py:427-445 gate_frame (health.py:19-28 no_events / near_zero_volume / stuck_on / chatter_storm) -> :825-855
  withholds answers = the classifiability gate (allowed).
- predict.py:886-888: health suspect/bad adds a status note + review_flag / review_reason; phase / function values
  unchanged. health_core itself (predict.py:742) runs after classification on the predicted phase / function (one way).
- Nothing else: phase features, decoders, similarity, networks, lanes, setback read no health output (grep).
pk_unhealthy is True on 0.27 % of rows (0.34 % of training rows).
## 2. OOF result (score114.json; function ATSPM E, seed-mean ok, paired signal bootstrap vs v6b recipe = base)
Base mean3 seed 0 reproduces v6's OOF exactly (P_m3_t0_s0, max |diff| 0). Arm nou = stacker without the column + decode
without the key; stk / dec = each half alone.
| variant | >= 30 min E | >= 30 R | 5 min E | per-seed >= 30 E |
|---|---|---|---|---|
| mean3 (default, base .9305) | -0.007 [-0.031,+0.015] | -0.011 [-0.033,+0.011] | -0.02 [-0.08,+0.03] | +0.002 / -0.029 / +0.006 |
| single (base .9284) | -0.003 [-0.021,+0.014] | -0.005 [-0.022,+0.011] | +0.006 [-0.036,+0.047] | +0.008 / -0.019 / +0.003 |
| nonet (base .9216) | +0.027 [+0.005,+0.050] | +0.016 [-0.007,+0.039] | +0.013 [-0.029,+0.056] | +0.033 / +0.027 / +0.020 |
| le2h fast (mean3 <= 2 h, nonet above) | +0.012 [-0.008,+0.033] | +0.004 [-0.017,+0.024] | = mean3 | |
Decode key alone (dec): 0.000 everywhere below 3 h, +0.003 at >= 30 min (24 h +0.01) - the key almost never decides.
10 min mean3 +0.06 [+0.01,+0.11]. Rows / signals: >= 30 E 126,505 / 642, 5 min 24,782 / 639. All within noise.
## 3. Decision: REMOVE (cost < 0.1 pt, CIs include 0, 5 min within noise)
Full-data seed-0 refits without the column: `%DC_WORK%/s114/weights_stacker/` (stacker_{mean3,single,nonet}_s0.onnx +
.txt, stacker.json 46 columns, `hf_clus_n` renamed `stack_n`; ONNX parity <= 7.2e-16, parity114.json).
Patch (`%DC_WORK%/s114/patch/note114.patch`, patched modules in `patch/new/`):
- pick.py: stack_health / _health_rel / rel_chi / unhealthy / CHI_* removed; new `stack_sizes(on, dets, M)` = cluster
  sizes only (equal to old clus_n on 300 synthetic samples, 1,853 members).
- atspm_decode.py: `unhealthy` dropped from stack_pick / decode_signal and from the _pick_order key.
- stacker.py: HF_COLS / HEALTH_CORE_COLS / needs_health_core removed; ALL_NAMES = 46 (stack_n, no pk_unhealthy).
- function_stage.py: no health_core import, no _health_context / _SkipCore / health SQL; one pair_mats per phase group
  shared by span_feats and stack_sizes (was 3 computations); _on_window reads ON times only, window filter in SQL.
- predict.py:57 docstring: drop "the stack-relative unhealthy key".
Scratch copy (v6b + patch + new weights) on the bundled sample: invariance checks 2 / 3 / 3b / 4 PASS; check 1 and the
no-network check FAIL only on probability drift (max 0.037 / 0.050 / 0.025 for 30 min / 10 min / trees) with 0 answer,
phase, lane or health changes -> production build must re-freeze references (check.py --freeze). Function stage
0.045 -> 0.039 s per call on the sample. Research parity harness (parity111) not re-run.
