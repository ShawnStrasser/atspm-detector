# 153 Detector health in DuckDB SQL: health(con, ...) / health_sql(...); predict keeps the numpy health

User go 2026-10-20 (orchestrator task, branch health-sql2 from main d187b2d): health entirely in DuckDB SQL,
vectorised across signals, rounding differences accepted; SQL text for an existing connection; time health ALONE.
Code src/atspm_detector/health_duckdb.py; research/code/perf153 (dev153 / out153 parity on the note-152 captures,
run153 / cmp153 predict-level, api153 health on a connection). Work %DC_WORK%/s153 (main_src = d187b2d, runs/).

## What was built
- 197 SQL steps (CTEs) in 9 blocks: inputs (windows, allowed events, gaps from per-minute min / max, covered 5-min
  bins), per-ON layer (lag / lead incl. 2-3 neighbours: continuous ONs, prior / next ON, prior start; rare OFF-OFF
  cases by ASOF), 5-min grid + running sums (one lookup per step: chained lookups made DuckDB pair them, 30x slower),
  stage 1 (templated twice: pass 2 without pass-1 bad yardsticks), occupancy context, Count held ON (+ green test),
  per-ON stats, too fast / too many (float32 casts as the research), hourly GEH, spiky (list windows), holds +
  queue context, limits / refs / tod models as VALUES tables, busy at night, verdict template (pre-pass + final),
  clean silence yardstick, text / categories / periods (JSON) / signal notes in SQL. No pandas step after the query.
- API: health_sql(events, detectors, start, end, probabilities) -> one WITH query (steps prefixed hq_, caller's table
  names put in after renaming). health(con, ...) -> DataFrame: same steps as temp tables (used-once steps inlined,
  each table dropped after its last use), signals grouped by max_events (4 M), distinct= flag. Detectors table:
  DeviceId, Detector, phase, function, lanes [+ p_* probabilities]. Exported from atspm_detector; module name
  health_duckdb (a module called health_sql would be shadowed by the function of that name).
- Partner tie-break by counts weighted by bin index (was compacted rank): only reachable with identical counts.

## Identity (vs numpy health; statuses, categories, reasons, periods, notes exact; scores within 1e-6)
- Captures (note 152, numpy outputs): sample, 40 x 30 min, 40 x 3 h, 100 x 24 h, 200 x 24 h (h24x), 3 multi-day logs:
  0 differences in every output column; internal measures equal except float32-level (c_occ_like, occ ~1e-6).
- predict on the SQL health (commit 90b979a) vs main d187b2d: sample, 1 / 10 / 100 x 24 h, multi, 100 alone: every
  output column equal (non-health bit-equal, phase tables equal); SQL batch == alone exact. check 7/7 (no freeze).
- health(con) on events in DuckDB with predict's answers as detectors: 1 / 10 / 100 / multi all equal, except 4 Bike
  texts on 100 x 24 h: "(covers 1 lane)" -- predict's lanes column is blank for bike zones (doc: give lanes = 1).
- Fixed on the way: greatest/least skip NULLs in DuckDB (NaN-propagating spots need CASE); identifiers are case
  insensitive (cs vs CS clash); regr_slope instead of the cancelling sum formula (0 vs 1e-13 flipped Q2).

## Health alone, same machine, back to back, 4 threads (predict --threads 4; s)
| job | main d187b2d | SQL in predict | per signal-day main / SQL | peak MB main / SQL |
| bundled sample | 0.04 | 0.30 | | 213 / 234 |
| 1 x 24 h | 0.12 | 0.41 | 120 / 410 ms | 300 / 315 |
| 10 x 24 h | 0.58 | 0.75 | 58 / 75 ms | 694 / 702 |
| 100 x 24 h (auto batch 25) | 7.98 | 7.37 | 80 / 74 ms | 2711 / 2778 |
| 4-5 day logs, 3 alone + 3 together | 3.42 | 4.27 | 122 / 152 ms | 1417 / 1676 |
| 100 x 24 h each alone | 11.40 | 39.58 | 114 / 396 ms | 900 / 933 |
health(con) with the events already in DuckDB (no copy): 1 x 24 h 0.51 s (peak 206 MB), 10: 0.95 s (505), 100: 7.4 s
(2.0 GB, groups of <= 4 M events), the three multi-day logs together 1.87 s (975 MB). One WITH query: + ~0.5 s planning.
Where it goes: one signal-day = ~0.35 s, nearly all DuckDB planning (EXPLAIN takes as long as the run; same with
fewer / more temp tables, BEGIN / COMMIT 10x slower); 25 signals 1.7 s (per-ON windows ~40 %, the rest bins / text).

## Decision (task rule: keep the SQL version in predict only if not slower)
- Slower for 1-10 signals, multi-day logs and single-signal calls; faster only in large batches (100: 7.4 vs 8.0 s).
  So predict keeps the numpy health: commit 90b979a switches it, 9e36911 reverts it (revert the revert to switch).
- The SQL is kept as the new API (health / health_sql): its point is running where the data is, no copy.

## Not done / open
- Two health implementations now (numpy in predict, SQL in the API); equal today, a change must go into both.
  Orchestrator / user: keep both, or switch predict to SQL (+0.3 s per call) and drop health_core / v4 / counts.
- Stage-1 fallback text (detectors without a known function) is a short form in SQL; never reached by predict.
- Update 2026-10-20: user chose SQL-only; predict() now runs the SQL health and the numpy health modules (health_core, health_v4, health_v4_stats, health_counts) are deleted.
