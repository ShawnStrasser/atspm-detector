# 115b — efficiency audit, network + function (package v6b), analysis only

Scope: funcnet.py, gru_input.py, function.py, features_expert.py, function_stage.py, stacker.py, pick.py,
atspm_decode.py of `dc_work/final_v3_candidate_v6b`. No package edited. Work: `dc_work/s115/audit_network_and_function/`
(prof.py = per-function timers + cProfile inside a warm predict(); bench_net.py / bench_sql.py / bench_cyc.py =
micro-benches, 5-7 reps; pkg_rw/ = private copy with rewrites R1-R5; run_cmp.py + cmp.py = outputs of pkg vs pkg_rw;
rewrites_*.diff). Bench signals (bench71, not locked): typical_r8 (22 det) and busiest_ch (43 det), 3 h and 24 h,
profile full, 4 threads. Machine shared: paired numbers only (paired_summary.json, 3 alternating rounds).

## Where the time goes (warm, timings_pkg_*.json)
Area total ~0.45 s (r8 3 h) .. ~1.2 s (busiest 24 h) of 1.0-3.2 s. Network pass (funcnet.both, shared by phase and
function, memo works: 12 piece calls = 3 members x 4 pieces) 0.26-0.50 s, of which ONNX pair graphs 0.17-0.29 s
(the floor), assemble 0.07-0.18 s. Everything else in the area is <= 0.1 s per block. Out of scope but large:
lanes.lanes 0.03-0.17 s (called from function_stage), health_core 0.07-0.30 s.

## Findings, ranked by measured saving (all rewrites: outputs bit-identical, see below)
R1 funcnet.py:172-190, 217-223 — `assemble` builds the FULL [D, K, 15, T] raster (18-35 MB) per member per piece
   (3x the same array), then copies the kept rows (x[j]); kept pairs are only 19-27 % of D x K. Rewrite: build only
   the kept pairs (gather det[di] / ph[ki]) once per piece, share across members (loop piece-outer to hold one piece).
   both(): 0.325->0.214-0.232 s (r8 3 h), 0.258->0.160-0.193 (r8 24 h), 0.496->0.315 (busiest 3 h) (net_*.json);
   paired in predict 0.06-0.26 s. Raster per piece 18->3.4-4.8 MB / 35->6.5 MB. Pair batch 64/128/256/all: no gain.
R4 function_stage.py:224-243 + :110 / pick.py:193 — pair_mats (n^2 python match_frac) runs 3x per phase group
   (span_feats, stack_health for pick, stack_health again for the health context); the second stack_health only
   feeds hf_chi/surge/drop/chat/rel_chi, which the stacker no longer reads (47 cols: only hf_clus_n, and clusters do
   not depend on cls/lanes). Rewrite: pair_mats once per group, pass to both; take hf_clus_n from the pick rows.
   function_stage.run -0.009 (r8 3 h), -0.008..-0.025 (r8 24 h), -0.041 (busiest 3 h), -0.077 s (busiest 24 h).
R2 gru_input.py:380-382 — SQL_DET (LEAD over every 81/82) runs twice (SQL_DET_CH over the whole sample + windowed),
   both = onev_all already built in build_chunk_tables. Read onev_all JOIN devmap: 0.028->0.005 (r8 3 h),
   0.070->0.009 (r8 24 h), 0.104->0.010 s (busiest 24 h) (sql_*.json); identical rows. Same finding as the
   phase-features audit (#3) — one fix.
R3 gru_input.py:384 SQL_CYC and features_expert.py:357-384 cycw compute the same 1/8/10/11 cycle window twice.
   Shared temp table cyc5 (built once, both derive): ~0.008 (3 h) / ~0.02 s (24 h) (bench_cyc json); identical.
R5 function.py:100-101 — two python lambdas in groupby.agg; precomputed 0/1 columns + "mean": 0.001-0.008 s.

## Judged fine (no rewrite worth a change)
atspm_decode (python, 0.2 ms: groups of <= ~8), stacker.ctx_X / stack_X (7 ms), add_sibling_features (7 ms),
_tied_top (6 ms), features_expert SQL_DET / SQL_CYC / SQL_PAIR (15-35 ms each, all DuckDB, dense pair grid
D^2 x nb is 177k rows at 43 det / 96 bins), funcnet.render / gru_input.cover / onrate (vectorised numpy, 17-28 ms
for 4 pieces), _on_window (5-17 ms), build_streams python per-candidate loops (ms), pick.track_feats (3 ms),
twin decode. Models, ORT sessions, stacker bags cached; net pass memoised. dtypes: rasters float32, ONNX float32; stacker X float64 (as trained, 47 cols x D rows — negligible).
gru_input.render / assemble (9-ch GRU) are not on the v6 path (network_kind siba_phase_head).
Libraries: numba / polars NOT worth adding — the remaining python loop after R4 (pair_mats) costs <= 0.037 s at
busiest 24 h; the rest is ONNX or DuckDB. Untested option: fuse the 3 member pair graphs into one ONNX graph
(1 call per piece instead of 3) — weights change, needs parity.

## Verification
pkg vs pkg_rw (R1-R5 together) on 10 cases (sample; r8 30 min / 3 h / 24 h / 24 h le2h; busiest_ch 30 min / 3 h /
24 h; busiest_ev 3 h; typical_r11 24 h): every output column and phase table identical, max float diff 0.0
(cmp.py base1 rw1). check.py on pkg_rw: 6/6 pass, stored answers drift 0.0. Paired predict() wall: -0.28 s r8 3 h,
-0.21 s busiest 3 h, -0.32 s busiest 24 h, r8 24 h within noise (-0.03 .. +0.06 by stage sums).
