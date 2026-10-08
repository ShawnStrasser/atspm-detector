# 128 Far start / end clipped to the data + health stage 2.8x faster, same answers (2026-10-07)
`dc_work/final_v7_next` = copy of `final_v7_prod` (7.0.0, untouched; swap in after the exam). Scripts
`research/code/final128/` (run from `%DC_WORK%/s128`), outputs `dc_work/s128/`. CPU, 4 threads, locked_v2 absent.

**1. Window clip** (`pipeline._clip_window`): a start / end more than 30 min (`CLIP_TOL_S`; quiet nights leave
gaps up to ~18 min) outside a signal's own data is treated as not given for that signal; inside 30 min the requested
clock window is kept (training used clock windows). Per signal, so a multi-signal call clips each on its own.
- 63 cases (3 signals x 30 min / 3 h / 24 h x start 1 h / 7 d / 365 d early / 2000-01-01, end 2030, both, exact):
  every column identical to the OLD package called with the out-of-range bound left out. Two signals in one call
  (one starts 7 h late, one ends 10 h early) = each alone, bound left out. 30-min sample, start 365 d early:
  41.6 s -> 0.55 s; start 2000-01-01: > 10 min (note 127) -> 0.51 s.
- Behaviour change: only where a bound is > 30 min outside the data. Edge battery `h24_from_06` (start 10 h before
  the data): 8 function probabilities, 7 lane confidences, 1 setback (9 ft) move vs old-with-start; = old-without-start
  bit for bit. README + pipeline docstring say it.
- Status text: values printed against a limit now always read below it (`_below`): "phase probability 0.986 (need >=
  0.99)" (was 0.99 / 0.99); "only 9.9 min of data" (was "only 10 min" for 9.998 min, e.g. the bundled 10-min sample).

**2. Health speed** (health_core / health_v4 / health_v4_stats; health opens NO DuckDB connection now):
- once per window, shared by both stage-1 passes and the v4 layer (`Prep.cache`): 5 / 15-min bins and one per-ON view
  for both bin sizes, continuous ONs, episodes (the >= 15-min list = the >= 5-min list cut), per-actuation tables,
  partner-note counts and correlations, twins, clock hours;
- stage 1: phase-reference silent runs only for the detectors that use them; unused model-study statistics no longer
  computed (chop5/30, chop15_self/sig, corr_ph(_exp), beta15, *_ref, n_twin, zero15_exp, max_dev_1h, mean_on_s,
  occ_max_bin, disp(_z): internal columns, no rule or output reads them); row views instead of iterrows; columns
  batched; act_stats(light) vectorised (p90 / gap shares it never used dropped);
- v4 layer: long pandas tables -> detector x bin matrices; pandas compensated groupby sums reproduced (`_ksum`, same
  order); the h117/h118c DuckDB query ported to numpy with DuckDB's float32 / float64 types; resolver, severity and
  outputs on arrays; lookups precomputed; integer-nanosecond clock maths.
- Exactness: numpy medians = DuckDB window medians (792 columns); colour stats = old SQL (132 windows + 300 random);
  Kahan / median replicas = pandas 2.3.3 and 3.0.6; ns conversion = pandas (9.1 M values).

**3. Parity / checks**
- note-115 set (132 cases, 2,627 detectors): all 43 columns, 15,705 candidate rows, 688 phase rows identical (equals).
- health on the 132 captured inputs, pandas 2.3.3 and 3.0.6: output + every shared internal column identical.
- = research scorer (note-124 method, 24 signals x 8 windows incl. 4 x 30 min, 4,180 detector-windows): status
  4,180 / 4,180, 40 / 40 sheet rows; the 7 diagnostic columns that differ (30-min statistics the research left NaN,
  tod_level spelling) differ identically for prod. Package mode prod vs new: every shared column identical.
- edge battery (18 cases): all columns identical except the clipped case. check 7/7, no --freeze (status text is not
  a reference column). Wheel 5.66 MB, 69 files = src; fresh venv (pandas 3.0.6, duckdb 1.5.6, ort 1.30): 7/7, no warning.

**Bench** (bench124.py, fresh process, warm median, mean of 4 bench71 signals; quiet: load 0-19 %, exam finished):
| sample | warm s old -> new | cold s | peak MB | health share of a call |
|---|---|---|---|---|
| 30 min | 0.72 -> 0.61 | 1.06 -> 0.96 | 245 -> 245 | 0.20 -> 0.08 (r8) |
| 3 h | 1.08 -> 0.91 | 1.63 -> 1.37 | 285 -> 272 | 0.26 -> 0.09 (0.06-0.11) |
| 24 h | 1.79 -> 1.47 | 2.17 -> 1.86 | 389 -> 386 | 0.52 -> 0.22 (busiest) |
| 7 d, 1 thread (n08) | 16.5 -> 15.0 | 17.0 -> 15.6 | 944 -> 871 | 3.4 -> 1.6 |
3 h is back to the pre-health-v4 level (0.89 s, note 124). Target < 0.1 s at 3 h: mean 0.089, worst 0.105 (r8: two
stage-1 and two resolver passes). Left: stage 1 two passes (~40 % of health), np.add.at occupancy at 7 days.
