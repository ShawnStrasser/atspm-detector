# 151 Batched predict (many signals per pass), outputs unchanged

User go 2026-10-19: vectorise predict() across signals, prove outputs identical, measure. src changed, not committed.
Backup of the old src: %DC_WORK%/bak_pre151/src. Code research/code/perf151 (prep, run, cmp, prof*, bsize). Work
%DC_WORK%/s151 (h24b_events.parquet = h24_b day of the 755 note-140 signals, runs/).

## What changed (src/atspm_detector)
- pipeline: the per-signal loop (`_one_signal`) is gone. Signals are processed `batch_signals` at a time (new arg /
  `--batch-signals`, default 10). Every signal keeps its own windows: table W (w0/w1 data span, b0/b1 sample window,
  per-signal formulas unchanged). SQL gets them from small per-signal parameter tables (common.param_table: the
  literal text the old SQL inlined, CAST to DOUBLE, so bit-equal) or from bound floats (on_table, night-speed greens).
- trees_onnx.by_group: onnxruntime TreeEnsemble sums trees in one order for calls of <= 50 rows and another above
  (measured: last-bit differences). Each signal block is scored as it would be alone (small blocks one call each,
  all larger blocks in one call); verified 0 differing values on all 7 tree models with interleaved groups.
- features: apply_window / fx bins / gate / streams (piece filter) per signal; join blocks keep whole signals and
  read only their signals' cycle / call / candidate copies (was a full-table ASOF per block: quadratic in a batch).
  ORDER BY added where Python consumes DuckDB rows (sim graph, lag table, onmask / masktime).
- setback: one call per group of signals sharing [b0, b1) (one call with a common start / end), tree calls by signal.
- health: events fetched for 10 signals per query; assess() still per signal (not vectorised, see below).
- notes / failures are per signal; a failing batched feature / function stage re-runs the batch one signal at a time.
- network rasters dropped after the phase pass (the memoised pass is reused by the function stage).

## Identity (every output column incl. health_*, and the return_phases table; floats compared bit for bit)
- bundled sample, 1 / 10 / 100 h24_b signals (one call), 100 signals each alone, 4-5 day logs (3 signals alone and
  together): new == old, new alone == new batch, all IDENTICAL (0 differing values, NaN == NaN).
- old is deterministic run to run (two alone100 runs identical).
- atspm-detector-check 7/7 pass (no re-freeze), pytest 26 passed 4 skipped (new tests/test_batch.py: 3 signals of
  different windows, batch_signals None / 2 / 1 == each alone, exact).

## Speed (same machine, src run from local disk; health = fetch + assess)
| job (24 h unless said) | old wall | new wall | old class. | new class. | old health | new health | peak MB old/new |
| 1 signal | 1.3 s | 1.3 s | 1.2 | 1.2 | 0.1 | 0.1 | 305 / 305 |
| 10 signals, one call | 10.5 s | 5.9 s | 9.5 | 4.9 | 1.0 | 1.0 | 473 / 699 |
| 100 signals, one call | 128.8 s | 80.6 s | 116.8 | 68.1 | 12.0 | 12.5 | 1720 / 2104 |
| 100 signals, one call each | 139.3 s | 132.1 s | 126.6 | 119.4 | 12.7 | 12.7 | 950 / 915 |
| 4-5 day logs (3 alone + 3 together) | 29.5 s | 28.7 s | 25.5 | 24.5 | 4.0 | 4.2 | 1376 / 1388 |
| bundled 30-min sample | 0.6 s | 0.6 s | | | | | 218 / 221 |
Per signal-day in a 100-signal call: 1.29 s -> 0.81 s (1.6x). Old 755 signals (chunk 100): 1051 s, 1.98 GB; new 755
not run (user: subset enough). Batch-size sweep, same 50 signals: 1 -> 1.43 s/signal (1.17 GB), 5 -> .98, 10 -> .90
(1.51 GB), 25 -> .88 (2.23 GB), 50 -> .87 (2.72 GB): default 10. A multi-day log is one signal: no batch gain.

## Not done / open
- Health checks not vectorised across signals (assess ~115 ms/signal-day: two stage-1 passes + many small pandas
  steps, ~150 KB of per-signal code); only the event fetch is batched. Big rewrite, high identity risk, <= 15 % of time.
- Remaining time is per-signal work: the network (~85 ms/signal-day), function stage loop, lead graph, health.
  A process pool over batches would scale with cores (not built; docs already advise a process pool).
- README speed line (1.7 s per 24 h) not updated (README numbers must trace to notes; orchestrator's call).
- Machine was shared during some runs; the comparison table is from one clean sequential chain.
