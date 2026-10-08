# 73b — Six-fold OOF check of note-73 speed idea 1: GRU only on candidates with tree probability >= thr (2026-10-01)
Code `research/code/evaluation/prof73b.py`; out `%DC_WORK%/prof73/oof73b.json`, `oof73b_q.parquet`, `oof73b.log`. CPU, 4 threads.
Saved OOF only, cand64 phase pipeline: note-57 trees (3-seed bag `p0_bag`) + p3 GRU OOF (K = 4 past 2 h), 0.5 / 0.5 BEFORE the
joint decoder; decoder re-fitted out of fold on folds_v4 exactly as `cand64.decode`. Filter = the GRU is used only on candidate
phases whose TREE probability is >= thr; it is renormalised over the kept candidates; the other candidates keep the tree
probability alone. This is what production's `gru_blend.mix` does for pairs the network does not score. Same rows, sets and
signal bootstrap as cand64; delta = filter - current candidate. Locked_v2 absent (cand64 asserts).
Checks: the reference blend reproduces cand64's p0_blend (max |diff| 2e-16). Re-decoding at 4 threads reproduces p2_cand
exactly: 0 scored rows differ.
## Phase, six folds OOF (accuracy [95 % CI]; delta pt [CI])
| filter | GRU pair rows kept | >= 30 min E | delta E | >= 30 min R | delta R | 5 min E delta | 10 min E delta |
|---|---|---|---|---|---|---|---|
| none (current) | 100 % | .9818 [.9787,.9849] | - | .9842 [.9813,.9870] | - | (.9646) | (.9726) |
| thr .003 | 42.2 % | .9817 | -0.007 [-0.027,+0.009] | .9842 | -0.004 [-0.024,+0.013] | -0.022 [-0.074,+0.028] | -0.047 [-0.101,0.000] |
| **thr .01** | **31.7 %** | **.9818** | **-0.001 [-0.024,+0.019]** | **.9843** | **+0.002 [-0.022,+0.023]** | +0.016 [-0.038,+0.066] | -0.006 [-0.058,+0.041] |
| thr .03 | 25.3 % | .9818 | +0.003 [-0.022,+0.028] | .9843 | +0.009 [-0.015,+0.033] | -0.025 [-0.091,+0.037] | -0.029 [-0.087,+0.027] |
Realistic 5 / 10 min at thr .01: +0.006 [-0.046,+0.057] / +0.006 [-0.048,+0.054]. Every CI contains 0 and is about +-0.02 pt
at >= 30 min, well inside the seed noise (0.07 pt trees). The filter is accuracy-neutral at all three thresholds. .01 is the only
one that is neutral at every length (.003 / .03 dip slightly at 10 min, n.s.).
## Function (via the phase input)
Not re-scored: the function frame's phase input (frame v6e) is not rebuilt from this pipeline. Upper bound instead: the
decoded top phase changes on 0.18 % of >= 30-min detector-windows at thr .01 (5 / 10 min 0.41 / 0.36 %; .003: 0.16 %; .03:
0.27 %). The function score cannot move by more than that share, and only a fraction of those rows would change class.
## Speed (from note 73)
Pooled over the OOF the filter keeps 32 % of GRU pairs at .01 (19-27 % on the two bench signals). GRU inference scales with
pairs: expect about -65 % of GRU time. Measured on the bench signals at 3 h: -4.2 s busiest, -2.1 s typical.
## Two pieces instead of four
Not testable from saved OOF: per-piece OOF exists only for the stage-13 GRU (`trackB/pieces/gru2_pieces_f*`), not for p3.
Note 33 (stage-13 GRU, frozen decoder): K = 2 vs all pieces -0.05 / -0.04 / -0.23 / -0.05 pt at 3 h / 6 h / 24 h / full
(not recommended there). Testing it on p3 needs a p3 per-piece re-inference (GPU / cloud). The candidate filter already
saves more time (~-65 % vs -50 %) at no measurable cost, so K = 2 is not needed.
## Verdict (for the orchestrator)
Adopt thr .01 at lock time: phase neutral (>= 30 min -0.001 E / +0.002 R, CI +-0.02 pt), function bound 0.18 %, GRU time
about -65 %. With ONNX tree ensembles (adopted): busiest 3 h about 9.1 -> 4.0 s, typical about 5.6 -> 2.6 s (note 73).
The fj half of the same filter (idea 2) still needs an fj re-inference to check.
