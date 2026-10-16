# 140 Health on 12-hour samples: every criterion, every threshold (2026-10-14)
Orchestrator task. Research only, nothing in src/ touched, no fault events, no locked signal (asserted). Code
research/code/health140 (h140_run, h140_parity, h140_tableA, h140_tableB, h140_plots); work dc_work/s140; output
review/thresholds_all/ (tableA_criteria.csv/png, tableA_by_function.csv/png, tableB_thresholds.csv, crit*.png).
## Sample and rules
- Window: Monday 2026-09-28 06:00-18:00, one per signal, w40 events. 763 non-locked training signals; 755 scored,
  8 skipped (no classifier answers / no out-of-fold reference fold). 14,796 detectors with a predicted function
  + 130 channels without one (package fallback = old stage-1 checks only; reported apart, not in criterion rows).
- Classifier answers = health4/inputs.parquet, period stg, wgroup h24 (phase, function, probabilities, lane span).
- Rules = score_v4e (note 135) with too-fast replaced by score_v4i (note 139 v4i). Engine = package
  health_v4.assess in note-135 research mode (research health_core stage 1, out-of-fold refs), too-fast swapped in
  the v4c pass (zf_c := runs/1000 ONs / cut, limit 1, N >= 50), 'ON longer than its kind' minutes exposed by a
  one-column source patch of occ_hi.
- Parity: same engine on research window h24_b vs s139/v4i/resolved_v4i: 14,806 / 14,806 statuses equal; runs, ONs,
  runs/1000, too-fast score 0 differences.
## What a 12-h sample uses
- window_group: < 1.22 h -> 30 min, < 8.49 h -> 3 h, else 24 h. 12 h -> the 24-h limits (too-fast cut 7.90 stop-bar,
  7.42 Advance/Mid, 8.00 Bike/Other per 1,000 ONs; all per-type p99.8 cells at wg h24); 15-min bins.
- Not run on 06-18: busy at night and unusual daily pattern (need a whole 00-24 day), misses vehicles at night (needs
  >= 2 h of 21-05 bins). Rule P (>= 4 h) can therefore only fire via silent >= 4 h or erratic time ON >= 16 periods.
- Count-type limits (minutes, holds per sample) were fitted on 24-h windows: on 12 h they are about twice as lenient.
## Table A (12 h, % of 14,796; suspect / bad as reported per category)
ok 93.63, watch 1.53, suspect 1.78, bad 1.10, not enough data 1.95 (research 24-h h24_b: ok 92.87, bad 2.22,
suspect 2.13). Stuck on 1.21 / 0.48 (raw over limit 1.69, queue-cleared 0.24); goes silent .12 / .18 (+ .20 watch);
count drops .09 / .12; erratic counts .22 / .30; too-fast .06 / .09 (+ .03 watch); chattering .46 / .07; too many
.11 / .02; erratic time ON 0 (max 68 % of its limit); count zone held ON .01 / 0; ON longer than its kind watch 1.35
(raw 5.35); C1 note .13 (0.90 % of Count), C2 note .45 (3.18 % of Count); short-ON watch .04; no-mate watch .05.
By function flagged (suspect+bad): Advance 3.87, Presence 3.27, Count 0.85, YR 0.74, Mid 2.49, Bike 2.55, Other 2.94;
Bike not enough data 30.7 %. 130 no-function channels: bad 20 %, suspect 7.7 % (old stage-1 rules).
## Table B
411 rows from s124/refs/health_v4_refs.json (the references the research scorer uses), same fallback order. Found:
cell and erratic limits are p99.8 (h110 sets R8.Q = .998 at import; h108's own default .995 is not used); occ limit
of 'ON longer than its kind' p99.5; volume max(p99.8, 1800); erratic floor .02 with cells >= 300.
Severity detail now explicit: the code's 'bad' value is where the check score reaches 1.0; a finding is REPORTED bad
from score .75 = suspect + 0.615 x (bad - suspect) (count drops / night: on log ratio). Column bad_reported added,
plot bad lines drawn there (e.g. goes silent 73 not 100, count held ON 159 not 240 min, too-fast 1.615 x cut).
## Plots (13 PNGs, all opened)
Per function panels where limits differ; lines at the panel's most common sub-type; '% at or beyond' counted against
each detector's own applied limit (includes the model-unsure loosening), i.e. before queue / borderline rules.
## Open
- No 12-h-specific limits exist; if 12 h becomes a product length, count-type limits need their own 12-h fit.
