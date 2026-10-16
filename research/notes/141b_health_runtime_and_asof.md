# 141b Health runtime per check, the ASOF "143 s" claim, vectorisation plan (research only, nothing changed in src)

Code `research/code/perf141/`, work `%DC_WORK%/s141p/`; DuckDB 1.5.6, Python 3.13, package 1.0.1 from `src/`, 4 threads.

## 1. Health wall time inside predict() (30 signal-days, w40 window h24_b, none locked)
All health functions wrapped (exclusive time) + section marks in in-memory copies of hc.health / hv4.assess; overhead nil.
Health (assess + its event fetch) = median 161 ms, p90 291 ms per signal-day; predict median 1.31 s, p90 2.02 s;
health = 13.0 % of predict (p90 14.8 %). Median 16.5 detectors, 169k events.
ms median / p90 / % of health (s141p/prof/buckets.csv):
- Stuck on (S1 episodes + v4 episodes_ctx + _stuck_stats/_stuck_eps)  11.8 / 19.2 / 6.1
- Busy at night + Unusual daily pattern (_time_of_day, 24-h only)       8.7 / 13.7 / 4.9
- Too-fast: colour_stats 5.6 / 8.4 / 3.2 + rule F fast_even 3.2 / 3.6 / 1.7 (Too many = q5 from the same pass, ~0 own)
- Goes silent (S1 recovery + v4 pre-pass + dropout_clean)               4.6 / 19.1 / 4.0
- Misses vehicles at night (S1 night_drop, _partner)                    4.0 / 14.4 / 3.5
- Erratic counts 3.5 / 6.9 / 2.0; Count drops 1.2 / 2.3 / 0.7 (+ shared shape110 0.8 % + loop glue 0.9 %); Chattering ~0
- Config notes (act118) 3.1 / 4.3 / 1.7; ON longer + Count zone held ON (occ_hi) 1.6 / 2.2 / 1.1; Erratic time ON 1.7 / 2.3 / 0.9
- DEAD: partner_notes 'undercounts vs partner' 3.5 / 8.0 / 2.5 -- its text is appended to the stage-1 reason, which
  only reaches users via _fallback, and _fallback cuts everything after '; note: ' -> never shown.
Shared: events->Prep 18.5 / 36.1 / 10.7 (DuckDB fetch 4.3 + prep_arrays lexsort/unique 14.3); stage-1-only checks
(choppy, rapid, spikes; feed pass-1 status = yardsticks / healthy mates / R9) 15.8 / 33.2 / 10.0; resolver + severity
+ text 17.6 / 21.5 / 9.2; 5-min bins 11.2 / 27.5 / 7.2; 15-min bins + ref 10.6 / 18.8 / 6.2; colour bins 8.4 / 16.3 / 5.1;
stage-1 frame (refs, rules, pass 2 on 9/30 signals) 5.5 / 13.1 / 3.8; Q2 queue context (slopes, like_corr) 6.4 / 8.2 / 3.7;
per-ON stats (Count mode, longest ON) 6.5 / 11.3 / 3.7; S1 det_stats 5.6 / 11.6 / 3.3; assess glue 5.7 / 2.9 %.
Multi-day (typical_r8 / busiest_ev, 24 h -> 5 d, events x5.7): health 220 -> 822 / 250 -> 1112 ms (x3.7 / x4.4),
predict x2.8 / x3.4, health share 15 -> 20 % / 12 -> 16 %. Fastest growers: 5-min bins x8.4 / x7.4, events->Prep
x5.7, Stuck on x6.0 / x4.7, colour bins x6.3; resolver, Goes silent, rules ~x1.1-1.4. Nothing quadratic; n log n at most.

## 2. ASOF claim (SQL_ONC NESTED_LOOP_JOIN, 5 d 142 s -> 14.7 s with asof_loop_join_threshold = 0): NOT reproduced
13 runs (3 signals x 24 h / 48 h / 5 d from staging_2026_w40 + asclog 24 h / 48 h / 4.1 d + 3 signals x 5 d in one
file), fresh process each, default vs SET 0: SQL_ONC 0.017-0.165 s (multi 0.38 s), setback 0.16-0.61 s, predict
1.6-6.6 s (multi 15.1-15.4 s) both ways; outputs identical, all columns, 13/13. EXPLAIN on every ASOF statement in
predict: ASOF_JOIN only (features.py x7, setback x2). Isolated SQL_ONC on the dumped sb_* tables: ASOF_JOIN in DuckDB
1.5.6, 1.5.2 (system Python) and 1.1.3; left-side estimates 5.7k-297k rows (>> 64).
Mechanism is real if triggered: forcing the loop join (threshold 1e9) gives SQL_ONC 3.3 s @24 h, 14.2 s @48 h,
91 s @5 d (quadratic; matches the claimed magnitudes) and predict 1.5 s -> 147 s @24 h because ALL 9 ASOF joins
switch (features.py 'j' alone 18.7 s). Setting exists in 1.5.2 / 1.5.6 (default 64), NOT in 1.1.3 (SET raises
CatalogException); pyproject allows duckdb >= 1.0 -> any adoption must be try/except-guarded. 'g AS MATERIALIZED':
identical rows, same speed (0.12-0.17 s). Verdict: harmless insurance, trigger unknown (get the claimant's
log + DuckDB version). Note 137 (sub-linear) stands: 24 h -> 5 d predict x2.3-3.4.

## 3. Vectorisation plan (health, ranked by measured saving at 24 h; plan only)
ATSPM_Aggregation_package patterns: LAG/LEAD per (device, channel) to pair ON/OFF and flag repeats (split_failures
step2, timeline paired_event); running SUM(+1/-1) and SUM(start flag) OVER = gaps-and-islands ids (cycle number,
detector group), MAX/LAST_VALUE IGNORE NULLS OVER group = forward-filled state; UNION ALL of phase and detector events
+ window carry (or ASOF, phase_wait) to label each actuation with the signal state; synthetic barrier events;
time_bucket + generate_series/UNNEST to split intervals over bins (platoon_ratio); ROWS BETWEEN k PRECEDING/FOLLOWING
moving windows; GROUP BY .. HAVING count(CASE ..) validity; LAG(TimeStamp) over all events for comms gaps (event_gaps).
1. Per-ON layer once in DuckDB (~45 ms of ~65): prep_arrays sort 14.3 (ORDER BY in the fetch + LAG gaps), 5-min and
   15-min bins 21.9 (interval split per bin), _act_light 5.8 / hs.act_stats 6.5 / act118 3.1 / on_arrays 2.0 /
   on_episodes 2.4 / _cont_on_starts (per-channel loops -> LAG/LEAD + islands, GROUP BY channel), colour_bins 8.4
   (ASOF of ON starts on cyc_all, as features.py already does; FILTER counts per bin). Scales x6-8 with days.
2. Resolver/severity/lookups (~13 of 21): _resolve row loop 8.5, severity_v4d 2.7, Refs.lookup/_least_strict 4.5
   -> numpy masks + merge on key columns; keep only the reason text in Python (_outputs 2.6).
3. Per-detector bin loops -> 2-D numpy over all detectors (~15 of ~25): level/erratic 6.5, _det_stats/_silent 5.6,
   like_corr+slopes 6.4 (_ksum_rows per column), _median_excl per bin, _partner 2.0, night_drop 4.0. Islands of quiet
   bins (silent runs) could be SQL gaps-and-islands, but the matrices are small: numpy batch is enough.
4. Drop partner_notes (3.5 ms, 2.5 %): output-neutral on read; verify with atspm-detector-check.
5. Stage-1 pass-1 status (choppy/rapid/spikes 15.8 + pass 2) duplicates v4 stats; merging changes yardsticks -> a
   research decision, not a speed fix. ToD 8.7 and episode context 3.4 are small Python loops; last.
Caveat: note 128 moved health OUT of DuckDB (float32 / compensated sums for parity): any port must pass the check.
