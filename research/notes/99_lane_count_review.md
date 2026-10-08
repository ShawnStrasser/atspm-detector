# 99 — Lane-count review sheet: OOF lanes D vs print lane counts (2026-10-05) — analysis only, nothing trained
Brief (orchestrator): sheet of phases where the lane model's out-of-fold lane count disagrees with the print, samples
>= 30 min, majority answer per phase, ~25 most confidently wrong, mix of over / under. CPU 4 threads, GPU not touched,
locked_v2 asserted absent. Code `research/code/evaluation/lane_review99.py` (rows / data / write); work `%DC_WORK%/rev99/`.
## Inputs
* Model = lanes D, six-fold OOF decode of note 77 (`final_v3_work/f77/ln8/lanes_D.func{,_ph}.parquet`; per-fold lam;
  phase grouping = OOF predicted phase). Sept-2026 log only (stg), 14 windows >= 30 min (m30 x4, h1 x3, h3 x2, h6 x2,
  h24 x2, full66). Phase answer = majority n_lanes over the samples where the predicted phase exists.
* Truth = print lanes per phase, ln1_cues.truth rules on the CURRENT label table (v4q): 2,529 phases / 517 signals
  (= note-90 truth + 10 phases, no n_lanes changed); 2,515 matched to a predicted phase.
* How sure = share of samples giving the majority answer x mean n_lanes_conf of those samples (rank key), then # samples.
## Result (phase level, majority over samples)
* Per sample: n_lanes exact .844 (m30 .813, h1 .786, h3 .871, h6 .880, h24 .883, full66 .884) — matches note 58/77 scope.
* Phase majority wrong 291 / 2,515 (11.6 %): over 76, under 215. Truth -> model: 2->1 140, 3->2 62, 2->3 35, 1->2 30,
  3->1 9, 3->4 8, other 7. Wrong by print lanes: 1 lane 2.3 %, 2 lanes 20.2 %, 3 lanes 34.5 %.
* 121 of the 215 under-counts are BLIND: the log cannot show the missing lane(s) — the active single-lane vehicle
  detectors (>= 10 ONs in 66 h) cover no more lanes than the model counted. 79 = only lane-spanning zones there (e.g.
  one Advance + one Mid both spanning lanes 1+2), 42 = the missing lane's loop is dead / near-silent. Not asked.
* Real disagreements 170 (6.8 % of phases): over 76, under 94. Rates (real errors / phases):
  technology loop 6.5 % (1,248), radar 6.5 % (842), video 5.7 % (298), mixed-technology phase 13.1 % (107);
  phase with a spanning zone 10.5 % vs 5.8 %; stacked group 13.0 % (23 phases); two same-role detectors in one print lane
  14.8 % (81; 10 of 12 are over-counts); a print detector predicted on another phase 18.4 % vs 6.1 %; model group holds a
  detector the label puts on another phase 21.9 % vs 5.8 % (28 of 76 over-counts = phase errors leaking in);
  <= 2 vehicle detectors 4.6 % vs 9.4 % (few detectors = mostly blind, not wrong).
* Seen on the sheet rows: low-volume lane loops (100-460 ONs / 66 h, e.g. a shared right-turn lane) called Bike by the
  function head -> no lane -> 3-lane phases counted as 2 (2B001, 2B029, 2B089); radar Count zones of two lanes merged
  (03057, 04097); one-lane phases split by an extra 'Other' loop or by detectors the label puts on a neighbour phase.
## Sheet `review/lane_count_review.xlsx` (+ `review/lane_count_review_charts/`, 25 PNG)
* 25 rows, 24 signals: 11 over-counts / 14 under-counts (proportional to the 76 / 94 real split), <= 3 per signal,
  blind phases left out; all rows how-sure >= .84 (11-14 of 14 samples on most).
* Columns: #, Signal, Phase, Print says (lanes), Model says (lanes), How sure ("13 of 14 samples, 97% sure"), Detectors on
  the phase (one per line, "det 15: Advance - model lane 1, print lane 2"; print-phase detectors + the model's group,
  "model puts it on P4" / "print puts it on P5" / "no actuations in the sample" / "model calls it Bike (no lane)"),
  What this row asks (plain, with the first merged / split pair named), Chart, Print link, Answer, Comment.
  Question on top as briefed. Rows sorted by signal.
* Chart (from saved full66 ONs, `rev99/review_data/`): 15-min counts per detector (legend det / print function / total),
  same-moment heatmap (% of row det's ONs with column det turning on within 0.5 s), lane table (print vs model lane).
* Spot check rows 1, 6, 12, 18, 25: print link opens the right cabinet print (title-block TSSU ID 01030 / 3044 / 04097 /
  2B001 / 2C039 read from the rendered page), chart link resolves, model counts re-read from the OOF file match.
## Caveats
* OOF lanes are note 77's (v3s/v4l-era function block); the shipped package's lanes D is refit on v4o (note 90) — the
  rows show the research OOF answer, not the package's.
* "Blind" uses print lanes + 66-h activity: a loop dead in Sept may work on another day.
