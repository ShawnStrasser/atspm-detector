# 81 — v4l adopted as truth + training labels; review sheet v1b; user's v1 answers folded in (2026-10-03)
Orchestrator brief: (1) adopt v4l everywhere, re-score the champion on it; (2) review sheet v1b (champion vs v4l, p >= .8,
nothing already on v1, definitional rows pending the radar / long-zone question left out); (3) the user's 4 answered
v1 rows as a correction list. CPU, <= 4 threads, no GPU, nothing trained. locked_v2 asserted absent everywhere.
## 1. v4l adopted
* `rpath.LABELS_CURRENT` = `research/labels/function_labels_v4l.parquet` (+ `LABELS_V3S`, `DEC_ROLE_CURRENT`). Pointed at
  it: `atspm_score.V3S`, `err70.meta`, `err78.stage_labels`, `review64.label_facts`, `review72.select`, `t57_phase`,
  `ln8_validate.V3S`, `sb7_validate.V3S`; `v3_retrain.LABEL_SETS["v3s"]` = (v4l, `cabinet_v4l/dec_role_changed_v4l`)
  so every trainer asking for "v3s" trains on v4l (old table = `"v3s_orig"`). `s80.py` keeps its explicit v3s / v4l
  paths (before / after). `t57_function.stage_fit`: the hard assert on BASE_RUN's fingerprint became a guard that refuses
  to write a non-v3s fit into the v3s trees folders (set OUT / BASE_RUN as `s80.stage_fit` does).
* Orchestrator rules folded into `cabinet/v4l_rules.py apply` (in place, idempotent; out of training AND scoring):
  R5 low-confidence print readings whose channel tie or class was decided by hi-res behaviour (hold-through-red share,
  ON lengths, count pairing / sums, lead / follow, liveness): regex `BEH_TIE` on the reader's confidence_reason, source
  print_low; 7 rows exempted by hand where behaviour is only a remark next to a print position / config code (2C050 d2,
  10063 d2/5/17/19, 10002 d50/52). 68 rows, 24 signals (video 44, loop 20, radar 4); 58 were training rows (Presence 17,
  Advance 15, Other 11, Mid 11, Bike 4). Includes the note-80 examples 2C052, 08003 d2/3, 06016 d16/17, 10002 d2/3/16/17.
  R6 unused inputs (whole-intersection Other, no config function) with < 15 ONs in the newest window = noise: 10 rows,
  9 signals (2B388 / 2B556 / 2C039 d41, 2C045 d40, 2C044 d40, 12067 d25/43, 04001 / 04009 / 04180 d31).
* Training rows (train_use_validated) 11,390 -> 11,322. Scoring truth unchanged (R5 rows are print_low, never truth;
  R6 rows never had truth; 2B091 already out by R2).
* Change list `research/labels/function_label_changes_v4l.csv` regenerated (`v4l_rules.py diff`): 390 rows vs v3s
  (new causes behaviour_tied_low 60, noise_input 10, user_correction 0 extra rows: 2B091 already R2).
## 2. Headline on v4l (champion = note-77 OOF, stacker 3 seeds, gate .9, pick, twin decode; signal bootstrap 95 % CI)
| window | E | R |
|---|---|---|
| >= 30 min | **.9190** [.9078,.9288] n 187,444 | **.9289** [.9186,.9384] n 182,828 |
| 10 min | .9030 [.8914,.9132] n 37,504 | .9135 [.9022,.9238] |
| 5 min | .8936 [.8827,.9046] n 35,809 | .9043 [.8937,.9146] |
Identical row-by-row to note 80's `ok_v4l_champ` (the new rules touch no scored truth); `s80.py score --truth v4l` and
`review81.py rows` (err78 decode) agree. Phase headline unchanged (.9818 E / .9842 R; phase truth = timing).
## 3. User's v1 answers (`research/labels/function_label_corrections_v4l.csv`, applied as R4)
2B091 d21 (row 1, M: loop cannot be a stop-bar count, no record) and d9 / 27 / 28 (row 2, M: throw the signal out) ->
excluded (already by R1 / R2; now recorded in `user_review`); 2C009 d19 (row 3) and d5 (row 4): L, label right ->
`phase_user_confirmed` = True (timing phase stays the truth; never asked again). The sheet v1 was only read
(md5 unchanged before / after: 002344690f13...).
## 4. Review sheet v1b
Code `evaluation/review81.py` (rows | review [--write]) on review64 / review72's builder; work `%DC_WORK%/rev81/`
(function_rows, errors_everything = champion note-70 attribution on v4l, review81_dryrun.json, review81_rows.parquet,
review81_same_pattern_as_v1.csv, chart data review_data81/). Phase rows = cand64's (phase champion unchanged).
* Start: 580 detector items (531 function, 49 phase). Dropped: definitional pending the radar / long-zone question 298
  (note-70 B1-B5 on every wrong window + radar Advance called Other, which B2 missed because channels 49-52 carry the
  plain 'advance' subtype: 4 items), not confident (< .8) 171, already on v1 54, < 3 samples 6, unhealthy 5.
* Kept 46 items -> 38 collapsed rows; 6 more left out because they repeat a v1 row's pattern at the same signal (ask once):
  2C027 d1, 05012 d50/52, 2B060 d22, 14016 d37, 2C009 d17 (= the answered row 3: label right), 14037 d39 -> listed in
  `review81_same_pattern_as_v1.csv` to carry the v1 answers over.
* **Sheet `review/function_label_review_v1b.xlsx` + `function_label_review_v1b_charts/` (32 PNG): 32 rows on 29 signals**
  (21 function, 11 phase; 31 confident contradiction, 1 config-only; 7 multi-detector; 27 with a print link; radar 14,
  video 9, loop 7, unknown 2; label source print_high 24, print_medium 4, config 3). How sure >= .9: 9, >= .95: 3.
  Top patterns: Advance -> Presence 4, Presence -> Other 4, Presence -> Advance 4, YR -> Count 2, Presence -> Count 2.
* Builder fix: review64 showed a phase item's most common predicted phase over ALL its windows = the label itself when
  wrong in exactly half (14037 d39 / d50 read "phase 2 -> phase 2"); v1b shows the phase of its errors (v1: no such row).
* Spot-check rows 1, 2, 4, 15, 27 (2B062 d21 Advance -> Other 12/12 wrong p .98; 2C023 d7 / d21 Advance -> Count 13/13
  p .98 / .96; 04030 d23 Presence -> Count 6/6 p .95; 10048 d7 Presence -> Advance 12/12 p .87; 13011 d53 phase 8 -> 5
  p .83): labels, phases and answers match function_rows / v4l / phase_rows; charts exist and match; print links open
  the right signal's PDF (10048, 13011 have none on file).
## For the orchestrator
* v1b answers -> add rows to the correction list, re-run `v4l_rules.py apply` (M on a function row = exclude / relabel).
* The 6 same-pattern detectors should inherit the user's v1 answer for their row (2C009 d17: row 3 says label right;
  not applied - the user answered per detector).
