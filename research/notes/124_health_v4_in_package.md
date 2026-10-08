# 124 Health v4 adopted into the production package (`dc_work/final_v7_prod`, 7.0.0) (2026-10-07)
Research scorer v4d (score_v4c + score_v4d on h118 / h110 / h108 / h104, 117 / 118c fast + volume, 116 night, 118c day
band, 118c clean dropout) now produces the package's health output. Backup before: `dc_work/final_v7_prod_bak124`.
Scripts `research/code/final124/` (refs124, cmp124, par124, cmp_par124, edge124, bench124); outputs `dc_work/s124/`.

**Package changes.** `health_v4_stats.py` (log statistics: ports of 12 research passes, one signal window, numpy /
pandas / DuckDB, no event-level Python loop), `health_v4.py` (resolver v4 + v4c + severity v4d, plain outputs, API
`assess()`), `weights/health/health_v4_refs.json` (335 KB: every population table the research computes at run time,
captured by running the research code: cell / g3 / fast limits for v4 and v4c, occ-hi, tod weights, q95, share
medians, 116 night model, day band), `charts.py` (optional, matplotlib extra `[charts]`), pipeline `post_outputs`
calls `assess`; `health_core` stays as stage 1 (two passes, R9). New columns `health_categories` ("Stuck on (bad);
Erratic counts (suspect)"), `health_config` (C1 / C2 setup notes), `health_signal_note`; `health_status` can be
`watch`; `health_reason` = "det 22: P8 Advance (covers 1 lane) - Stuck on: held ON 3 times, 2 h 41 min in total".
Length rules: limits of the nearest research length (30 min / 3 h / 24 h; cuts 1.22 h, 8.49 h); whole-day checks
(busy at night, unusual day, signal note) per full calendar day, worst day kept (>= 24 h without midnight: clock-hour
day); weekend = Sunday tables, weekday = Monday tables. Shipped night / band references are all-data fits (research:
out-of-fold). README, pipeline docstring, model card updated. Version stays 7.0.0 (never shipped).

**Health equality** (cmp124; 24 signals of the 40 sheet rows, every detector, h3_a / h3_b / h24_a / h24_b = 2,126
detector-windows, research classifier inputs; 74 columns: status, findings, watch, config notes, rules, 13 scores,
~55 statistics and limits):
- research mode (research stage 1 + its out-of-fold refs): 0 differences; 40 / 40 sheet rows identical. Same under
  pandas 3.0.6 (text columns identical too). (tod_level NaN vs 'not scored' at 3 h is only the spelling.)
- package mode (v7 stage 1 + shipped refs): 2,123 / 2,126 statuses equal, 40 / 40 sheet rows. 3 differ, all 'busy at
  night' from the all-data night model: 2B068 h24_a det 7: P3 Count suspect -> ok; 2B069 h24_a det 48: P8 Yellow-red
  ok -> suspect; 08073 h24_b det 37: P6 Bike bad -> suspect. v7 stage 1 with the out-of-fold refs: 0 status
  differences (its 115d / 115dx fixes move 5 pass-1 statuses, absorbed).
**Parity** (note-115 set, 132 cases, 2,627 detectors, `s124/par/cmp_par124.json`): 29 non-health columns, 15,705
candidate rows, 688 phase rows: 0 differing cells (floats ==). Health status moved on 49 detectors (flagged 77 -> 64;
10 now 'watch'); `status` 43 / `review_flag` 27 / `review_reason` 41 rows follow the health text.
**Edge** (edge124: 1 / 3 / 5 min, no 43/44, no 7-10, no 131, no / one Begin Green, one / no detector, 2 signals, empty
window, 24 h from 06:00, 30 h, 2 days, 7 days, a channel held ON 3 h): no exception, non-health columns identical.
**Check** 7 / 7 PASS from src and from the wheel in a fresh venv (pandas 3.0.6, numpy 2.5.3, duckdb 1.5.6, ort 1.30):
old 6 + new 'health v4 smoke test' (refs load, reason starts with the label, categories parse, a detector held ON 12 min
-> Stuck on with the period). Checks 2-3b map "det N" / "P N" in health text back before comparing. --freeze done.
Wheel 5.50 -> 5.65 MB, package files 65 -> 69, uncompressed 8.97 -> 9.42 MB. No warnings (-W always) under pandas 3 at
30 min / 3 h / 24 h / 7 days.
**Bench** (bench124, 4 threads, fresh process, warm median, mean of the 4 bench71 signals, shared machine):
| sample | warm s old -> new | cold s | peak MB |
|---|---|---|---|
| 30 min | 0.58 -> 0.72 | 0.90 -> 1.06 | 252 -> 252 |
| 3 h | 0.89 -> 1.07 | 1.24 -> 1.43 | 270 -> 282 |
| 24 h | 1.42 -> 1.76 | 1.81 -> 2.15 | 384 -> 387 |
7 days, 1 thread (n08, 38 ch): 15.6 / 14.6 -> 17.2 / 16.5 s (cold / warm), peak 925 -> 843 MB. Made faster during the
port (same answers): episodes listed from 5 min (a 1-min list built a 100 MB pairwise matrix), colour timeline
vectorised, queue context per phase matrix. Health is now 0.24 s of a 3-h call (43 ch), 0.5 s at 24 h; the rest is
pandas overhead spread thin. 3 h is now 2.3x the beta's 0.47 s (was 1.9x).
**Charts** `health_chart(events, out, detector, path=)`: detector + phase mates, counts and % ON per 15 min, listed
periods shaded, reason under the chart (checked: 10037 det 22, `s124/chart_10037_d22.png`).
**Not ported:** the sheet-only whole-phase note (118d `phase_note`); the 4-h persistence rule (note 121, off).
