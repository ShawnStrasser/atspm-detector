# 80 — Label pass v4l: released NEWTEST into the training pipeline, incomplete prints re-read (2026-10-03)
Goal: raise label quality inside the frame (note 78: label-caused error 0.4-1.5 pt). CPU, <= 4 threads, no pulls, share untouched.
Code `research/code/cabinet/` (v4l_store.py, v4l_expl_sweep.py, v4l_rules.py; store mode `DC_CAB_STORE` in cab_common / cab_final /
label_check / stack_labels_v3s), `research/code/evaluation/s80.py`. Work: `%DC_WORK%/cabinet_v4l/` (store copy), `%DC_WORK%/lab80/`.
Labels `research/labels/function_labels_v4l.parquet` (v3s schema; README updated); change list `function_label_changes_v4l.csv`.
## 1. Released NEWTEST (71) into the training pipeline
* Separate store copy `cabinet_v4l` = training store + ONLY the 71 released records (record, extract, PDF copy, renders, crops,
  activity rows) from `cabinet_locked`; batch_36 = 69 (08409 / 08421: no print). Store mode: hold-out = locked_v2 (asserted exactly
  115, disjoint from released); `v4l_store.py check` + every writer assert no locked_v2 DeviceId (records, renders, crops, activity,
  label table, dq, dec-role). The 72 + 43 locked signals are never read beyond the id lists (the locked activity file is read and
  filtered to the 71). review/ untouched (store lists go to cabinet_v4l/lists/).
* Now run on them like any training signal: fixes, DQ (incremental, 69 signals), whole-intersection rule, label check (released rows
  pass 788 -> 795), stacked / YR-twin step, Dec-role (they have no Dec config: no_signal), overrides / dead lists.
## 2. Incomplete prints re-read (130 of 195 incomplete frame signals; 3 opus agents, ~20 min wall)
* Ranked by function error on config-only / medium-low rows (err78 everything set): 34 signals carry all of it (2,105 error rows),
  but 50 incomplete signals have NO intersection diagram (2B099, 2B091, 2B142, 2B331, 2B143, 2B095 ... = 54 % of that error) and
  their .dgn twins on the share are the same drawing -> not completable from prints. Re-read: the 130 with a diagram and an unread
  or unexplained channel (not unusual_layout), error-ranked first.
* Result: unread active print detectors 238 -> 188, unexplained channels 351 -> 273, 36 fixes of old readings (14077 / 08081 / 08129 /
  08300 long-zone -> Presence per the user's "written extent only" rule; 2C045 / 2C037 coded radar zones that were "explained" away;
  10046 d49 A* -> Other/advance_presence), 06033 sheet 7 found (31 detectors). Most incomplete signals stay incomplete: stale prints
  (site re-equipped / renumbered after the print), unnumbered video zones, Wavetronix C11 channels 33-44 not drawn.
* Sweep: 78 explained_other entries at 16 signals explained possible performance-measure detectors away ("added after the print",
  "renumbered", "radar template channel"...) -> removed (`lab80/expl_removed.csv`); whole-intersection Other must never cover them.
* Tiers (frame + released): incomplete 201 -> 183, complete_mixed 420 -> 436, complete_high 136 -> 138.
## 3. Rules folded in (cleansing only)
R1 config Count / YR on a print loop (user 2026-10-01, review v1 row 1 + hard rule): print reading wins at any confidence (53), else
excluded (13). R2 2B091 excluded (user: "throw the whole thing out"; 24 rows). R3 re-read YR zones count-identical to Count (3).
## 4. What changed (v3s -> v4l; 16,759 -> 16,775 rows): 339 labels (training label or scoring truth)
| cause | rows | truth changed | training changed |
|---|---|---|---|
| print re-read | 203 | 36 | 177 |
| loop config Count (R1) | 66 | 14 | 27 |
| explained_other sweep | 25 | 0 | 25 |
| pipeline re-run (label check on new stats) | 15 | 0 | 15 |
| user excluded 2B091 (R2) | 14 | 14 | 3 |
| released NEWTEST pipeline | 13 | 0 | 13 |
| YR identical to Count (R3) | 3 | 0 | 0 |
Truth: 64 rows (61 ATSPM): 29 added (Presence 14, Count 9, YR 8 ...), 32 removed (Count 18, Advance 12), few relabelled. Training:
260 (208 added, 44 removed, 13 relabelled); train_use_validated 11,229 -> 11,390. 643 scored rows (>= 30 min).
## 5. Effect on the function champion (>= 30 min unless said; signal bootstrap 95 % CI, pt)
| | E | R | 5 min E | 10 min E |
|---|---|---|---|---|
| champion on v3s (reproduced) | .9171 | .9282 | .8916 | .9007 |
| A. same OOF rescored on v4l (own rows) | .9190, +0.20 [+0.06,+0.38] | .9289, +0.07 [-0.01,+0.18] | +0.20 [+0.08,+0.36] | +0.23 [+0.09,+0.40] |
|    common rows only | +0.02 [0.00,+0.06] | +0.02 [0.00,+0.06] | | |
| B. trees arm c + stacker refit on v4l, scored on v4l | .9194, +0.03 [-0.02,+0.09] | .9293, +0.04 [-0.01,+0.10] | +0.04 [-0.06,+0.13] | +0.01 [-0.10,+0.11] |
|    same refit scored on v3s truth | +0.03 [-0.02,+0.09] | +0.04 [-0.01,+0.10] | | |
| A + B (v3s champion -> v4l refit) | +0.23 [+0.08,+0.42] | +0.10 [+0.00,+0.23] | +0.24 [+0.08,+0.42] | +0.23 [+0.05,+0.44] |
* A is a measurement correction: 681 E rows leave the set (model right on 39.5 % of them: wrong / unknowable truth), 145 enter (73.8 %);
  not a better model. B (training labels) is inside the seed noise (stacker +-0.02, trees +-0.07): neutral, 0 of 3 >= 1-pt rule.
  Released-only rows B -0.02 [-0.17,+0.15]. Held fixed: siba TCN, lanes D, pick / health inputs, frame (all v3s-trained).
* Refit outputs: trees `final_v3_work/f76/function_c_v4l/`, stacker `f77/s74/f80v4l/`, ok arrays + s80.json in `lab80/`.
## 6. For the orchestrator
* Adopt v4l as the scoring truth (it removes known-wrong truth) and as training labels (neutral). Rebuild review v1b on v4l.
* The remaining label-error mass sits on 50 no-diagram prints and stale prints; only newer prints / zone maps (user / staff)
  or the user's review can fix it. Reader questions (agents): tying unnumbered zones by behaviour (low conf, 2C052 / 08003 / 06016
  / 10002) acceptable?; channels 40/41 with no phase / text at Region-2 sites = test channels?; upstream video boxes without a
  code -> Advance (11016)?; channels with < 15 ONs on unused inputs explained as noise (2B388 / 2B556 / 2C039).
* Page ranking still misses zone sheets (06033 sheet 7, 2B382 sheet 6): a full render sweep of incomplete prints could find more.
