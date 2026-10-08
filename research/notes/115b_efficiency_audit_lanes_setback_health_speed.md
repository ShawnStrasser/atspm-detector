# 115b - efficiency audit: lanes / setback / health / night speed (package v6b; analysis only)

Scope: lanes.py, setback.py, health.py, health_core.py, night_speed.py in dc_work/final_v3_candidate_v6b. No package
changed. Work + private instrumented copy: dc_work/s115/audit_lanes_setback_health_speed (prof.py = timers,
cmp_health.py / cmp_lanes.py = parity, fast/ = prototypes, out_*.txt / prof_*.json = numbers).
Bench signals bench71 typical_r8 (22 ch) 3 h / 24 h, plus busiest_ch (43 ch) 24 h. The machine was shared
(CPU 85-90 % from other agents in the second half), so compare old vs new in pairs only; absolute numbers ~2x idle.

Stage cost (warm, idle part of the run, prof_*.json): post stages (setback + health + night speed) 0.20 s of
0.95 s at 3 h, 0.49 of 1.71 s at 24 h, 0.63 of 2.43 s at 24 h / 43 ch. Lanes run inside the function stage:
0.02-0.06 s idle. Health is the largest: 0.08 / 0.30 / 0.43 s. Night speed 0.004 s. health.py gate ~0.
health_core is NOT run twice: the v6b stacker reads no health_core column (needs_health_core False).

Findings, ranked by measured saving (paired, loaded machine; 3 h / 24 h / 24 h 43 ch):
H1 health_core: the same event frame is filtered, de-duplicated, converted to seconds, gap-scanned and lexsorted
   FIVE times (events_to_bins 115, act_stats 630, on_arrays 1139, on_episodes 1033, _cont_on_starts 1551).
   Rewrite: build once (_Prep: t, eid, par, gaps, sorted detector arrays) in health() and pass it down.
   Prototype fast/health_core.py: -0.021 / -0.197 / -0.223 s (19 / 41 / 36 % of health). Output BIT-IDENTICAL
   (all columns, 3 signals).
L1 lanes Decoder coordinate ascent (_best_L 388, _det_conf 433): every candidate lane set is a full O(n^2)
   _score in Python (5,166 calls at 43 ch). Rewrite: score all alternative sets of one detector in one numpy
   batch, same summation order, same tie rule. Prototype fast/lanes_dec.py: -0.009 / ~0 / -0.093 s (-0.011 on
   busiest_ev 24 h). Whole predict output identical on 4 signals.
H2 post_outputs (predict 733) fetches the device's events into pandas, then health filters / de-duplicates again
   although `ev` is already DISTINCT and channel-filtered. Rewrite: one SQL with the window filter, epoch seconds
   computed in DuckDB, fetchnumpy (optionally ORDER BY channel, time, ON first). -0.010 / -0.054 / -0.059 s; the
   seconds array is bit-identical. Also removes the pandas Timestamp frame (no whole-day frame in pandas).
H3 events_to_bins: np.add.at for n_on / n_chat -> np.bincount(ci*nb+b) (exact, integer counts); ci by a Python
   dict per ON -> np.searchsorted(dets, ...). -0.014 s at 24 h / 43 ch, identical. occ stays np.add.at
   (float32 accumulation order; bincount would move the last bit). maximum.at is faster than a sort-based max.
H4 health: pd.to_timedelta / Timestamp arithmetic inside per-detector loops (night_drop hour 1440, spike_stats
   t15 1200, _hm, episode bounds): ~0.016 s at 24 h (cProfile). Use integer hours from the int64 microsecond
   clock; must be verified at hour edges before adopting (not prototyped).
S1 setback: pair cues for the setback pair model (SQL_CORR + SQL_MIN + cues, ~0.02-0.035 s at 24 h) repeat the
   lanes pair cues (lanes.signal_pair_cues) for advance -> stop-bar pairs. Reuse = saving, but DuckDB corr()
   vs numpy differ in the last bits -> needs a p_same parity check (1e-6) and re-freeze. Not bit-identical.
S2 setback small Python: _pick groupby + 2 sorts per target (0.007-0.009 s; one stable sort over all candidates
   + groupby.head(1) gives the same pick), HGBNumpy.raw tree-by-tree (0.010 s; traverse all trees at once, keep
   the sequential sum), pivot_table for phase make-up (groupby size). Together ~0.02 s, identical if done as said.
N1 night speed: the per-signal onev_all / cyc_all re-query (0.006-0.009 s) repeats function_stage._on_window;
   pass its on / off arrays through. vehicle_speeds re-sorts the same arrays 4x (background) - <0.003 s.
Minor (< 0.01 s each, identical): det_stats runs twice over all detectors though only drop_* is used from the
   phase pass; _partner / spike_stats re-aggregate 15-min counts on every call; twins / _refs computed twice;
   lanes signal_pair_cues computes the 6 symmetric correlations for both orientations.

Not worth it: numba / polars (no remaining event-level Python loop; the cost was repeated work, numpy covers the
rest - no new edge dependency); ONNX for the setback HGB (float32 thresholds, saves < 0.01 s).
Dead code in the package (size only): lanes.on_times / on_times_from_intervals, health_core.cycle_counts,
act_stats non-light branch, rapid_hour (OPTS off), cues' lead_lag columns.
Total if H1-H3 + L1 adopted: ~-0.04 s at 3 h, ~-0.26 s at 24 h (22 ch), ~-0.39 s at 24 h (43 ch) on the loaded
machine (about 8-9 % of a 24 h call), all bit-identical; check.py --freeze not needed if outputs unchanged.
