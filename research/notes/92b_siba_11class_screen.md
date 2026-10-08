# 92b — siba with an 11-class function head (note 92's Other subclasses): 2-fold screen (2026-10-05) — not adopted
Brief (orchestrator): train siba on the 11-class labels (Other -> Upstream_presence / Long_zone / Setback_presence /
Departure, note 92 map), folds 0 and 3, seed 0, v4o; score inside note 92's K-class chain (kway and sum decodes) vs v4f on
the same folds, >= 30 min E and R, plus how many upstream zones are called Advance. Local A1000, CPU 4 threads, locked_v2 absent.
## Set-up
* Labels `tcn53/func_rows_v4o_sub11.parquet` (`final92/sub92b.py rows`: v4o y, Other -> subclass by print subtype = note 92
  census; 23,406 usable rows re-labelled; class sizes = note 92). `neural/tcn69_func.py --classes` (new knob, default 7
  classes unchanged; stored in cfg, restored by infer); snapshot `tcn53/snap92` = snap74b + that file.
* `x92_siba11` folds 0 / 3: note-69 siba recipe, --accum 2, filtered inference (tree p >= .01). 37 / 51 epochs (best 29 / 43),
  ~50 / ~75 min. Net = ONE seed (the 7-class arms use the x86_siba4l 3-seed mean: a seed-diversity handicap for net11).
* Only two folds of the new net exist, so every screen arm's stacker is re-fitted "cross-2-fold" (fold 0 scored by a stacker
  trained on fold 3 rows only and vice versa; 3 LightGBM seeds, note-92 inputs). Arms: v4f2 (v4f recipe), sub2 (note-92 sub:
  11-class trees + x86 net), net11 (11-class trees + the new net: its 7-class view in the 47 columns + its 4 subclass probs).
  One-fold stackers cost ~1 pt absolute (v4f2 .9067 vs full v4f .9170): read the PAIRED contrasts, not the levels.
## Result (folds 0 + 3, 185 signals, v4l truth, ATSPM-only stack-aware, paired signal bootstrap, pt [95 % CI])
| | >= 30 E (52,724) | >= 30 R (51,444) | 10 min E | 5 min E |
|---|---|---|---|---|
| v4f2 / sub2 kway / net11 kway | .9067 / .9034 / .9035 | .9153 / .9118 / .9117 | .8853 / .8805 / .8746 | .8746 / .8707 / .8720 |
| net11 kway - sub2 kway | +0.01 [-0.51,+0.54] | -0.01 [-0.57,+0.54] | -0.59 [-1.36,+0.19] | +0.13 [-0.48,+0.74] |
| net11 sum - sub2 sum | -0.03 [-0.54,+0.49] | -0.05 [-0.61,+0.50] | -0.47 [-1.18,+0.25] | +0.17 [-0.38,+0.74] |
| net11 kway - v4f2 | -0.33 [-0.81,+0.20] | -0.36 [-0.87,+0.16] | **-1.07 [-1.81,-0.33]** | -0.26 [-0.88,+0.38] |
By class >= 30 E vs v4f2 (net11 kway / sub2 kway): Advance +0.71 / -0.45, Presence -0.06 / +1.23*, Count -0.18 / -0.25, YR +0.13 /
+0.46, non-ATSPM **-3.23* / -3.42***. References on the same rows (full six-fold stackers): v4f .9170, note-92 sub kway .9168.
Upstream_presence truth rows (>= 30 min E, 2,333 rows / 114 det.), called Advance (called any ATSPM): v4f 377 (448), v4f2 321
(388), sub2 kway 396 (482), **net11 kway 419 (491)**, net11 sum 407 (477). Exact subclass, net11 kway / sub2 kway: Upstream .662 /
.643, Setback .655 / .621, Long_zone .019 / .017, Departure 0 / 0 (96 rows; note-92 full stacker .865).
## Verdict: NOT ADOPTED
The 11-class head adds nothing over note 92's trees-only subclasses (>= 30 min +0.01 / -0.03, CIs +-0.5 on two folds; 10 min
-0.5..-0.6 n.s.), calls MORE upstream zones Advance (419 vs 396 sub2, 321 v4f2), and the whole subclass chain stays below the
7-class recipe on the same footing (-0.33 n.s., 10 min -1.07*). No reason to promote to six folds (screening rule: >= 1 pt).
Code `research/code/final92/sub92b.py` (rows / stack --arm v4f2|sub2|net11 / score); work `%DC_WORK%/x92/oof2/` (sub92b.json).
