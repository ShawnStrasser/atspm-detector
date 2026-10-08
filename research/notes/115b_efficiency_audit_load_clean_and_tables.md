# 115b - efficiency audit of v6b: load, clean, chunk tables, assembly (analysis only)

Package dc_work/final_v3_candidate_v6b untouched. Private copies + scripts in dc_work/s115/audit_load_clean_and_tables
(pkg = v6b copy, pkg2 = finding 1 applied; audit.py per-function + per-SQL timer, bench_*.py, timings_*.json).
Bench signals (non-locked, bench71): typical_r8 3 h (31k events, 22 ch), busiest_ch 24 h (222k events, 43 ch);
also r8 24 h, busiest_ch 3 h. 4 threads, warm median. Machine shared (55-70 % busy during the A/B runs): only
paired / same-session comparisons are trusted; "unloaded" = the first audit run (timings_<sig>_<L>.json).

Scope cost (unloaded): predict.py + common.py own code and SQL = 128 ms of 1.10 s at 3 h (12 %), 208 ms of
2.49 s at 24 h busiest (8 %). Biggest items: chunk tables 21 / 69 ms, load 9 / 30, _function_frame 33 / 32,
post_outputs glue 17 / 27, _assemble 10 / 10. common.py = constants only.
No Python loop over events anywhere in these files; every loop is over detectors or pairs (<= 344 rows).

Findings, ranked by measured saving (3 h / 24 h busiest, unloaded-equivalent)
1. gru_input.build_streams re-derives onev_all twice (SQL_DET line 144 and SQL_DET_CH = DISTINCT over a second
   full LEAD pass, line 112/143): 15 / 61 ms. Rewrite: read onev_all (+ cand) when present. A/B paired, 3 rounds:
   streams_for 65 -> 49 ms (3 h), 159 -> 70 ms (24 h) under load; outputs bit-identical (max diff 0, all text equal).
   Saving ~12 / ~55 ms. (Cross-area: file belongs to network inputs; table is ours.)
2. VARCHAR DeviceId as the window-partition and join key in every chunk table (predict.py:276-402). Rewrite: build
   devmap from the source, store an integer dev in the event table (ev kept as a view for other modules), partition
   and join on dev. All 7 tables identical; load+tables 151-178 -> 128-136 ms (24 h), 41-57 -> 32-34 ms (3 h) under
   load; ~-6 / ~-18 ms unloaded.
3. Same signal intervals fetched three times (function_stage._on_window fs:71 9 ms, setback sb_iv 10 ms,
   post_outputs night speed predict.py:753 3 ms at 24 h): fetch once, pass arrays. Estimated -3 / -10 ms, identical.
4. _function_frame (predict.py:583) 33 ms in pipeline, 14 ms isolated; 65 % is function.add_sibling_features
   (35 groupby.transform on <= 43 rows). One groupby.agg + merge: est. -8 ms; function.py owner to verify.
5. Three extra full scans of ev for facts (load stats line 298, detector_universe 406, signal_facts 414): 8 / 16 ms.
   count(DISTINCT CASE) -> bit_count(bit_or(1<<p-1)) measured 15.5 -> 11.8 ms, identical; merging scans est. -3..-6.
6. devmap fetched to pandas 4x (predict.py:481, features.py:349, features_partner.py:261, features_expert.py:303),
   ~1.3 ms each: fetch once, -4 ms.
7. _assemble (808) 10 ms: 5 merges + itertuples status loop (3 ms); np.select would save ~3 ms. Leave.
8. _softmax/_normalise_by_detector build string keys: 1.8 ms total; group on the two columns, identical. Leave.

Cross-area pointer (not this area, not measured as a rewrite): health_core.health (413 ms at 24 h, largest non-network
CPU item) re-derives ON intervals from the raw per-signal event frame (fetched at predict.py:733) five times
(events_to_bins 102, act_stats 96, on_episodes 72, on_arrays 65, _cont_on_starts 45 ms, cProfile-inflated). Missing-OFF
handling may differ from onev_all, so any rewrite needs its own identity check.

Tested and rejected: DuckDB threads=1 (SQL_DET 60 -> 111 ms at 24 h; keep 4); ORDER BY at load (+50 ms, saves ~15);
DISTINCT before cast / one-level casts (within noise). numba / polars: no event-level Python loops here, so no
material gain; not worth an edge dependency. dtypes fine (ev USMALLINT; epoch-second times must stay float64;
dur FLOAT). Memory: no large frames in scope; the biggest is the per-signal event fetch for health (~5 MB at 24 h).

Achievable in scope with no output change: findings 1+2+3+5+6 ~ -25 ms at 3 h (~2 %), ~ -90 ms at 24 h (~3.6 %).
