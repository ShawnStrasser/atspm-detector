# 115b — efficiency audit, phase features (package v6b), analysis only

Scope: features.py, features_partner.py, similarity.py, decode.py, gru_blend.py of
`dc_work/final_v3_candidate_v6b` (+ gru_input SQL that gru_blend.streams_for runs). No package edited.
Private copy, scripts, timings: `dc_work/s115/audit_phase_features/` (audit.py = per-function / per-SQL timers
inside predict(); variants.py = rewrites vs originals on the real chunk tables, 7 reps, outputs compared;
fast_patch.py = the rewrites as a monkeypatch; e2e.py = predict() orig vs patched, 1 cold + 5 warm).
Bench signals (bench71, none locked): busiest_ch (43 det) 3 h / 24 h, busiest_ev (31 det, 68k calls) 24 h, typical_r8 3 h. DuckDB 4 threads; BLAS 4 threads. Machine shared: compare paired numbers only.

## Where the time goes (t_*_full.json, profile full, warm median)
predict() 1.34 s (3 h) / 2.47-2.76 s (24 h). Phase-feature area = build_features 0.30 / 0.71-0.89 s +
streams_for 0.03 / 0.10-0.12 s + decode.assemble 0.024 s; i.e. ~25 % (3 h) and ~35 % (24 h).
Everything heavy is already DuckDB SQL; the Python loops (_mask_features, _partner_features) cost
0.04 s + 0.02 s and do not grow with sample length (they loop over mask groups x dets x phases).
Outside this area: siba net 0.29-0.43 s, post_outputs (lanes/setback/health) 0.22-0.91 s.

## Findings, ranked by measured saving at 24 h (variants v_*.json; all outputs equal)
1. Duplicate ON x candidate ASOF joins (features.py:73-83 SQL_STATE, features_partner.py:40-50 SQL_J,
   features.py:162-176 SQL_CALL_FWD, features_partner.py:104-109 SQL_CALLS2, plus long-ON release join
   twice, features.py:144-151 and features_partner.py:65-73). Rewrite: one table j with t_off, t43, t44
   and both time bases; one lo2 (dur > 1.5) with dur > 3 filter for FINE. 0.534->0.432 s and
   0.639->0.513 s (24 h), 0.141->0.126 s (3 h). Max rel diff 7e-17 (orig vs orig run-to-run 7e-18).
2. Lead neighbours in SQL (similarity.py:377-409: per-second unnest + 8-lag joins). Rewrite in numpy:
   per-second onset matrix, O = sum_l A[:, :-l] @ A[:, l:].T (float32 BLAS, exact for counts), strata
   via bincount, E = M @ rate.T. 0.146->0.065 s, 0.178->0.068 s (24 h), 0.034->0.016 s (3 h). w rel diff
   <= 7e-16. Memory: D x seconds float32 = 15 MB per day at 43 det (multi-day: chunk by day if needed).
3. Detector ON pairing recomputed for the network streams (gru_input.SQL_DET via SQL_DET_CH over the
   whole sample AND windowed = 2 window-function passes over ev; same definition as onev_all,
   predict.py:311-320). Rewrite: read onev_all JOIN devmap. 0.086->0.010 s, 0.111->0.012 s (24 h),
   0.023->0.007 s (3 h). Identical rows. Runs only when a network runs.
4. SQL_CALL_REV (features.py:178-193): calls x all detectors cross join + ASOF. numpy searchsorted
   per detector + bincount: 0.111->0.080 s, 0.213->0.135 s (24 h); 3 h ~0.003 s. Identical.
5. add_rank_features (features.py:386-405): 28 x 4 groupby calls on list-of-Series keys. One ngroup
   code + DataFrame groupby rank/mean/std/max: 0.037->0.012-0.015 s at any length. finalise
   (features.py:408-419) per-column cast loop -> one astype(dict): whole finalise 0.059-0.069 ->
   0.042-0.047 s. Identical values and dtypes.
6. Not rewritten (small): _mask_features 0.04 s, _partner_features 0.02 s (numpy-matrix version would
   be ~ms; tie logic tied_best must be kept); decode.assemble 0.024 s; gb.mix / softmax by detector
   string keys 1-2 ms; devmap read 5x, onmask/masktime/cand read twice (< 2 ms total).

## End to end (e2e_*; 2 paired rounds each, warm median s, orig -> patched with 1-5)
busiest_ch 3 h 2.68->2.57, 2.64->2.55; typical_r8 3 h 1.48->1.41, 1.73->1.56;
busiest_ch 24 h 4.82->4.27, 4.58->3.76; busiest_ev 24 h 4.16->3.19, 3.70->3.36 (machine loaded ~2x
vs the first audit). Saving ~0.1 s at 3 h (4-7 %), 0.35-0.8 s at 24 h (8-18 %).
Answers: every output column identical (float diff 0) on all 4 signals. check.py NOT run on the patched copy: run it (incl. 3b shuffle) when the rewrites are applied.

## Libraries
DuckDB + numpy are right; polars no gain (+30 MB); numba no (loops ~0.06 s, llvmlite ~100 MB, JIT
warm-up > saving). Event times stay float64; features float32; j (~0.9 M rows at 24 h) and j2 never
coexisted, so the merge does not raise the peak.
