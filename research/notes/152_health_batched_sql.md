# 152 Health batched across signals (DuckDB fetch + stacked checks), memory-based batch size

User go 2026-10-19 (orchestrator task on branch health-sql). Code research/code/perf152 (cap152 / par152 / profg152 /
prof152 / run152 / cmp152); work %DC_WORK%/s152 (cap/ captured assess inputs + main outputs, runs/, main/ = e65d6ea src).

## What changed (src/atspm_detector)
- Batch keys: a health_core.Prep may hold several signals sharing one window; signal s's channel c = key s*128 + c,
  its phase p = key s*65536 + p. Every per-detector / per-phase computation then runs once for all signals; every
  signal-wide thing (rest-of-signal references, coverage, gaps, twins, holds at the same time, signal notes) is kept
  inside its signal (padded signals x rows layout, within-signal pairs). Signal 0 keeps the plain numbers, so one
  signal alone runs the same code. health_v4.assess = assess_group of one signal.
- pipeline._health_prep: one DuckDB pass per window group (10 signals): detector events sorted by (signal, channel,
  time, ON first), colour events sorted by (signal, phase, time, code), comms gaps (lead over all event times; ties are
  0 apart, so no DISTINCT), covered 5-min bins (integer ns // 300 s; 15-min = any of its three). No numpy sort left.
- Per-detector Python loops replaced: silent runs (best_runs), level change point, twins, references (padded matmul,
  integer counts only), median-excluding-self, Kahan sums per group (one step per rank: same order as _ksum), phase
  timelines per segment, p50 by partition, next OFF by a backward running minimum (= searchsorted), ON-count inside
  a hold from the tie group (= the old float composite searchsorted), colour events / on-view memos shared.
- Kept as loops (numpy's own arithmetic is the definition): np.corrcoef pairs (partner, like_corr; cached across the
  two stage-1 passes; c_occ_all only where c_occ_same is NaN), np.interp per phase timeline, per-signal W @ counts
  (BLAS) in hourly GEH, episodes, text. DuckDB SQL is used for the fetch, not for float sums (pandas Kahan / float32
  add.at / BLAS orders are not reproducible in SQL); stacked numpy does the rest.
- predict(batch_signals=None): auto = floor(0.5 x free physical MB / (260 MB per M loaded events x events per signal))
  in [1, 25], 10 if free memory unknown (Windows GlobalMemoryStatusEx, Linux MemAvailable, else sysconf). Explicit
  value wins. Health group size stays 10 (group 10 = 25 = 100 in speed, measured: 6.47 / 6.51 / 6.50 s per 100).

## Identity (floats bit for bit, NaN == NaN, text exact)
- assess level (par152, incl. every internal X column): sample, 40 x 30 min, 40 x 3 h, 3 multi-day logs, 300 x 24 h;
  each alone and all in one batch: 0 differences.
- predict level vs main e65d6ea (cmp152, every detector column incl. health_*, bad periods, notes, phase table):
  sample, 1 / 10 / 100 signals, 4-5 day logs, 100 alone == 100 batched, batch_signals 1 / 10 / 25 / 50 / 100 / auto:
  all IDENTICAL. atspm-detector-check 7/7 (no freeze); pytest 28 passed 4 skipped (+ health batch == alone,
  memory-based batch).

## Speed (same machine, local src copies, back to back; health = fetch + assess; 24 h unless said)
| job | main wall | new wall | main health | new health | peak MB main / new |
| 1 signal | 1.26 s | 1.25 s | 0.120 | 0.120 | 301 / 301 |
| 10 signals, one call | 5.79 | 5.37 | 1.00 | 0.58 | 700 / 699 |
| 100 signals, one call (batch 10) | 79.2 | 74.7 | 12.27 | 7.72 | 2111 / 2116 |
| 100 signals, auto batch (25 here) | | 72.3 | | 8.07 | / 2708 |
| 100 signals each alone | 130.2 | 129.8 | 12.5 | 11.2 | |
| 4-5 day logs (3 alone + 3 together) | 28.4 | 27.3 | 4.2 | 3.3 | 1499 / 1425 |
| bundled sample | 0.62 | 0.62 | | | 213 / 213 |
Health per signal-day in a batch: 123 -> 77 ms (1.6x); assess alone on captured inputs 102 -> ~61 ms (group of 10).
A single signal gains nothing (5 SQL statements cost what the old numpy sorts saved). Remaining health time is
event-level numpy (bins with float32 add.at, per-ON passes), memory-bound: stacking more signals does not help.

## Memory vs batch (100 signals, 17.3 M loaded events, one call): batch 1 1734 MB, 10 2120, 25 2728, 50 3853,
100 4459; wall 127 / 75 / 73 / 75 / 73 s -> ~40-45 MB per extra signal-day (~260 MB / M events), no speed above 10-25,
hence ceiling 25 (orchestrator may prefer 50 per the task text: one constant).

## Not done / open
- Single-signal health not faster; README speed line untouched (README numbers trace to notes; orchestrator's call).
- Failure handling: a failing health group is retried signal by signal (before: a failing fetch marked all 10).
