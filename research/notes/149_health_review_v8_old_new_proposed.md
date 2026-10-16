# 149 Health review v8: one Proposed suspect / bad / barely good per check, at Old / New / Proposed (2026-10-15)
Orchestrator task (user: "update health review 7 ... old, new and proposed; one suspect, bad and barely good per
type"). Research only: src/, v7, thresholds v8-v12, USER_INPUT untouched. No fault events; locked_v2 asserted absent.
Code research/code/health149 (h149_data, h149_pick, h149_charts, h149_sheet); work dc_work/s149.
## Data and limits
d143 + 1.1.0 display measures (out144_h24b) + ep143 = h146_sheet / h147c exactly; h24_b Mon 2026-09-28, 755 signals,
14,806 detectors. Old = h143_sim.current_limits, New = h144_sim.final_limits (bad points = package defaults),
Proposed = h147c_alloc.ModelC.run(knobs147c: Rs 1050, Rb 2) = v12 with Presence / Other stuck bad 60 min, Bike = 1
lane. Scores h147_lib.all_scores2, final status h144_sim.status. Reproduces flagged147c: ok / suspect / bad
Old 94.37 / 2.07 / 1.93, New 96.18 / .83 / 1.34, Proposed 93.31 / 2.52 / 2.53.
## Per-check result ("status" in the sheet)
= the package status with every other check's score removed (check alone), ok when no finding of the check is
left; the detector's final status added in brackets when different. Seven types: too fast and too many split.
Check-alone counts at Proposed (suspect / bad): stuck 255 / 87, counts fall 34 / 11, night 111 / 9, busy 0 / 243
(every busy finding is bad), too fast 14 / 7, too many 9 / 0, erratic 31 / 35.
## Picks (h149_pick, seed 149)
Suspect / bad = Proposed check result; barely good = Proposed check ok and final ok, measure < Proposed suspect
limit, among the 25 closest. Preference: flagged by its own measure (>= limit), result follows the measure in all
three sets (no cross-check suppression), new signal (not in v1-v7 / spotcheck sheets; asclog is one external
log, not in the 755), Old / New / Proposed disagree, only this check flags it, type (fn x lanes) not yet used in
the check. Closeness = measure / Proposed suspect limit (falling measures: limit % / measure %).
Rows 21 (20 with charts): stuck 3, counts fall 3 (all goes silent), night 3, busy 3, too fast 3, too many 2 + 1
'no Proposed bad' row (max busiest 5 min 3,360 veh/h/lane < bad 4,500), erratic 3. 18 of 20 on new signals;
2B055 (too fast bad, 5 candidates) and 10062 (too many suspect, 9 candidates) were in earlier sheets (marked *).
Busy at night: no suspect level exists, the 'suspect' slot shows a weaker finding (score .5, still bad by rule P).
Too fast / too many / erratic are fixed at New in v12, so New = Proposed there (Old differs on some).
## Charts review/health_review_v8_charts (20 PNG, all opened)
Evidence panel = note-145 style (h145_charts helpers): raw ON bars + phase band for stuck / too fast / too many;
24-h 15-min counts vs phase mates scaled, flagged period shaded, for counts fall / night / busy / erratic. Strip
below: measure (black) vs this detector's Old (grey dashed) / New (red) / Proposed (green) suspect / bad limits,
labelled with values, yellow = suspect zone, red = bad zone, each row's result on the right.
## Seen while building (for the orchestrator, not acted on)
- Night misses: 7 of the 9 Proposed-bad night findings are silent stretches at night (night count ~100 % of
  expected); row 8 shows one where the silent 22:10-24:00 makes Old suspect / Proposed bad.
- Counts fall silent >= 4 h: any finding is bad (rule P); row 6 is Old bad / Proposed ok at 52 expected in 7 h.
- Erratic row 19: Old final ok although the check alone is suspect (Old drops erratic next to another finding).
## Outputs
review/health_review_v8.xlsx (Examples: # / Signal / Det / Check / Measure / Old / New / Proposed / Chart / verdict /
comment; Totals: final status and per-check flagged % from flagged147c.csv), review/health_review_v8_charts/,
dc_work/s149 (all149.parquet, examples149.parquet, rows149.csv).
## 149b Proposed = note 147d (GEH), v8 rebuilt in place (backup review/_backup/health_review_v8_149.xlsx + charts dir)
h149_data d / h149_pick d / h149_charts d / h149_sheet d (+ h149_geh: hourly GEH = h147d_geh for one detector, matches
geh147d.parquet on all 9 GEH rows). Proposed = h147d_alloc.ModelD at knobs147d (final 94.14 / 2.15 / 2.07 =
flagged147d); Old / New unchanged (ratio measures). Rows kept when their 147d Proposed check result still fits the
slot (barely good: still among the 25 closest under Proposed): 10 kept, 11 re-picked (stuck suspect, counts fall
barely good (now a GEH count drop), night suspect + barely good, busy barely good, too many all three (a Proposed
bad now exists: 08097, 2,735 > 2,520), erratic all three). 21 rows, 21 charts (opened); 19 new signals
(2B055, 08097 earlier sheets). GEH rows: Measure shows GEH and the old % measure; Proposed limit in GEH (erratic in
hours with GEH >= 10); charts add hourly signed GEH bars with the Proposed lines and two strips (Old / New in %,
Proposed in GEH). Seen: night row 9 counts 5 vs ~114 at night (4.8 %, Old / New bad) yet night GEH 4.98 = ok
(low night volume keeps GEH small); erratic barely good row 21's two GEH hours are its only mate going silent
(yardstick); night bad row 10001 is bad via its silent stretch, its night GEH is 2.8.
149c: 'Day chart' column added after 'Chart' (h149_day.py; 21 PNGs <row>_day_<signal>_det<d>.png: 24-h 15-min counts + % ON of all detectors on the predicted phase, measured period shaded; all opened); user's 17 verdict / comment rows moved one column right unchanged (checked vs review/_backup/health_review_v8_before_daychart.xlsx); 08097 det 20 is alone on its predicted phase.
## 149d Proposed = note 147e (backup review/_backup/health_review_v8_before_147e.xlsx)
h149_data / pick / charts / day / sheet with argument e (+ h149_geh.under_series). Proposed = h147e_alloc.ModelE at
knobs147d R_s / R_b (final 94.02 / 2.13 / 2.19 = flagged147e); Old / New on the original hold table. Erratic rows
dropped; 10 rows kept, 8 re-picked (same-check prior detector preferred: 04022 moves to counts-fall bad), 4 added
(undercounts suspect / bad / barely good, stuck common-failure extra 11025 det 21 with dets 8, 10, 11) = 22 rows,
44 PNGs (too-many day charts in 5-min bins). User verdicts carried by (signal, det): 11 of 17; the 6 dropped
belonged to replaced detectors (2B351, 04050, 05100, 2B066, 10052, 08007). Seen: a 'bad' status can sit inside the
strip's yellow zone (bad = score > .75, 62 % of the way to the drawn bad limit: row 6 GEH 14.1 vs 10 / 15);
row 14 too fast 437 per 1,000 > bad 251 stays suspect (rule F, Presence); undercount rows 20-21 are 2-lane
detectors vs a 1-lane mate x2.
149e (147g): v8 rebuilt in place by h149_g.py (backup review/_backup/health_review_v8_before_147g.xlsx): 11 judged rows removed (unchanged or now matching the verdict; undercounts dropped), kept 04070 d36 + 10041 d48 (now bad with spiky; verdicts kept), added 8 Spiky counts (6 spiky_only147g, barely ok 10031 d2 sum 32.5, dramatic bad 13003 d4 z 33); 2B388 d2 / 2B358 d4 / 2B436 d4 skipped: their dramatic jumps come from low-count mates (40 / 41) scaled up - a yardstick artefact of the spiky check.
149f (147j): h149_j.py rebuilt v8 in place (backup _backup/health_review_v8_before_147j.xlsx + charts _147g): rows 1-2 kept with entries (04070 d36 / 10041 d48 still final bad via spiky 37.1 / 31.30), spiky re-picked: 6 spiky-only one per type (2B028 d9, 2B062 d23, 2B497 d3, 03087 d53, 03031 d28, 2B348 d64) + barely good 08048 d15 (31.2); no dramatic row; Proposed bad limits shown at the score-.75 point (as v12 from 147h). Seen: 2B062 d23 tracks its mates closely yet sums 58 from many z 2-4 jumps (z0 2 accumulates small jumps).
149g (147k): h149_k.py rebuilt v8 in place (backup _backup/health_review_v8_before_147k.xlsx + charts _147j): kept 10041 d48 (now suspect, 1 spike) and 2B062 d23 (now ok, 1 spike) with entries; added 2 clearly spiky (10068 d25 3 spikes, 10004 d23 2), the 4 mild of gallery k (03049 d56, 01072 d58, 05032 d2 = stuck 00-08:30 + spikes shared with det 50, 06033 d10 low volume), 1 one-spike 13005 d8. Spike chart: 10-min counts + mates scaled, red triangles with GEH (n asserted = spk_n2). 2B062 d23 has no mates series in bins147k although det 22 / 24 track it.
149h (147l): h149_l.py rebuilt v8 in place (backup _backup/health_review_v8_before_147l.xlsx + charts _147k): all 9 judged rows removed (147l matches; 10041 now suspect = his 'possibly suspect'); shown again with empty verdict: 2B497 d3 (0 bursts, ok; he suspect), 2B348 d64 (7 bursts, bad; he suspect); added 3 clear (08170 d15 4 bursts, 08052 d1 2, 03123 d58 2), 3 marginal low-volume (05026 d1, 03028 d20, 08117 d2; 2 each) and 10105 d8 (1 burst). Burst chart: 10-min counts, dashed expected, red triangle + peak ratio (count asserted = spk_ev). Seen: 08052 d1 follows its mate det 13 exactly (incl. the 10:20 peak) yet counts 2 bursts.
149i (147m): h149_m.py rebuilt v8 in place (backup _backup/health_review_v8_before_147m.xlsx + charts _147l): all judged rows removed; 06033 d10 shown again (1 burst, ok; he suspect); gallery m: 10023 d9, 09008 d11 Bike (5 bursts, 485/day, bad via GEH 11), 13006 d8, 10069 d1; marginal 08019 d8, 2B001 d15; 08019 d23 (5 bursts; mate det 22 has its own big night peaks 01-06 h, partly shared); 12055 d38 one burst. Burst chart: ratio + GEH per burst, day total.
