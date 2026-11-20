# 156 - ped movement assignment model (which phase / ped overlap each ped input calls)

## Step 1: features (2026-10-29)
Code `research/code/ped156/ped_features.py` -> `features(con, events_table, start, end)` (DuckDB SQL only, temp tables
pf_*, no extra dependency; meant to move into the package). Runner `ped_features_run.py` (4 GB, 6 threads).
Output `dc_work/pedcfg/ped_features_<period>.parquet`: keys DeviceId, ped_detector, kind (phase/overlap), movement, period
+ 24 features (list in the module docstring). Candidates = every input with presses (90) x every phase 1-16 with a 21 or
45 and every ped overlap 1-32 with a 67 or 68 at that signal. Overlaps have no ped-call code -> call features NULL.
Sources: sep / oct = dc_work/ped2 (note 155); dec2024 = data/raw Train_Dec_{2,3,4}_2024 minus locked_v2. No other
ped-bearing raw source found in dc_work/data (staging_other / staging_ped_2026 are what sep / oct were cut from).
Overlap codes in the data: 67 walk and 68 clearance only (no 69); overlap numbers 9-17 (17 at 1 Sep signal, 11 rows).

| period | rows | signals | inputs | phase rows | overlap rows | run s |
|---|---|---|---|---|---|---|
| sep (Sep 18-21 2026, 2.75 d) | 9,748 | 816 | 2,724 | 9,615 | 133 (25 signals) | 4.8 |
| oct (Oct 4-7 2026, 2.3 d) | 9,019 | 758 | 2,548 | 9,019 | 0 (pull has no 67/68) | 3.2 |
| dec2024 (Dec 2-4 2024) | 4,501 | 371 | 1,258 | 4,480 | 21 (5 signals) | 3.8 |
- Inputs = note 155's ped_health inputs (Sep 2,724). ~7 % of inputs (9 % Dec) have < 10 presses.
- Checked: call1, walk120, cov1 equal a pandas brute force on one signal (a5c3076d, 36 rows) exactly.
- Descriptive (no labels yet): best candidate by "free-press call share" (1 - free_nocall) is >= 0.5 for 93.8 % of Sep
  inputs and has the same number as the input for 92.7 %; by raw call1 only 51 % reach 0.5 (most presses come while the
  call is already in, so the controller logs no new 45) -> call1 alone is weak, the free-press and coverage features carry it.
  Best by cov1 >= 0.5 for 95 %. same_number is a strong prior (~93-96 % agreement) - the model step must report how much
  it relies on it (train with and without it).
- Oct: the ~28 ped-overlap inputs have no overlap candidate (no 67 in the pull) -> exclude them from Oct validation.
## Next
Labels (configured ped input -> phase / overlap), LightGBM ranker like the vehicle phase ranker (pipeline.py), train on
one period, validate on the other.

## Step 2: labels from Kinetic Signals (2026-10-29)
Fetch: `dc_work/pedcfg/fetch_pedcfg.py` (+ `kinsig_client.py`, reads the bot login from the team notebook at run time;
both stay in dc_work, not here). Read-only `deviceUIMibs(databaseId=<latest upload_completed stored db>, names=[12 ped
plan MIBs])` returns all 10 plans x up to 72 ped detectors in one call (no live subscription needed). Raw per device:
`dc_work/pedcfg/raw/<id>_MaxViewPedestrianDetectorPlansGrid_all.json`. Build: `research/code/ped156/build_ped_labels.py`
-> `dc_work/pedcfg/ped_labels.parquet` (plan 1 always + any other plan with a non-default entry) and `ped_labels_summary.txt`.
- 1,169 devices tried, 1,150 OK (1,019 signals from stored db; 131 ramp meters from KinSig device copy, no ped config);
  19 failed (16 disabled for comms / test controllers, 3 ramp meters timed out). 1,130 of 1,133 event-data devices covered.
- Measured, plan 1: 4,077 assigned ped inputs on 1,019 signals. Input n -> phase n only: 92.9 %; call phase = input
  number 3,994 (98 %). 213 call 2+ movements (82 with 2+ phases, e.g. 4 -> 4 + [8]); 164 have a ped overlap (call overlap
  often the same number as the phase, e.g. 2 -> ph 2 + ovl 2; overlaps 9-17 also used); 33 overlap only. Additional
  overlaps and cancel phases unused. Inputs 12/14/16/18 -> 2/4/6/8 at a few signals.
- Decode: list fields are comma number lists in the API, not bitfields. Overlaps kept as numbers (17 seen).
- Stored db == KinSig device copy (4 devices, 0 diffs); 09040 matches the UI. Plans differing from plan 1: 15 devices;
  live active plan polled: 11 on plan 1, 13002-13005 on plan 3 (plan 3 only adds extra phases 4<->8).
- For the ranker: label = plan 1 all_phases / all_overlaps; check the overlap numbering against 67/68 event parameters.

## Step 3: model (2026-10-29)
Code `research/code/ped156/`: `ped_data.py` (labels -> active plan, join), `train_ped_assign.py` (CV, baselines),
`final_ped_assign.py` (final fit, Dec 2024, disagreements, ONNX export + parity), `unseen_check.py` (cross-period and
Dec 2024 tests on held-out signals). Outputs `dc_work/pedcfg/model/`
(results.json, oof_pooled.parquet, ped_assign_s0.txt, runtime.json, disagreements_all.csv), `dc_work/pedcfg/disagreements.csv`.
Package: `src/atspm_detector/ped_assign.py` + `weights/ped_assign/{ped_assign_s0.onnx, manifest.json}` +
`tests/test_ped_assign.py` (new files only; edits to existing files in `dc_work/pedcfg/integration_patch.md`).

**Labels.** Active plan = plan 1, except 13002-13005 = plan 3 (live poll 2026-10-29). Target per candidate row = movement in
all_phases (phase) / all_overlaps (overlap). Evaluation target per input = configured movements WITH ped activity in the
window. Measured: in Sep 92 of 124 inputs with a configured overlap have no 67/68 for it - almost all "phase n + overlap n"
(overlap n never times a walk); every overlap-only input's overlap (9-17) is among the signal's 67/68 numbers, so overlap
numbering in config = event Parameter (Sep 34 inputs, 94 % top-walk_cov overlap = configured). Oct: inputs with a configured
overlap are left out of training and evaluation (no 67/68 in the pull). Inputs with presses but no configured movement:
3 per period (2B357 input 7, 2B555 inputs 3 and 5) - reported, never trained on.

**Fix to step 1 features.** The idle-state ASOF lookup (`free_n`, `free_nocall`) matched arbitrarily between events logged at
the same instant (Sep: 788 instants, mostly a 45 logged with the 21) -> runs differed (free_n by up to 24). Now one state per
instant (walk > clearance > call > don't walk); features re-run, deterministic (repeat runs: decisions identical).

**Model.** LightGBM binary, 24 features, 300 rounds, 15 leaves, lr .05, min_data_in_leaf 40, one seed. Plus an
input-renumbered copy of the training rows (each signal's ped input numbers permuted; only same_number changes) at weight 0.1:
without it the model took input 6 -> phase 6 at p .93 when every press of input 6 placed a call on phase 8 and none on
phase 6 (crossed wiring is rare in the labels; same-number + no call evidence is mostly ped recall). With it: synthetic
crossed wiring right, pooled OOF up 99.24 -> 99.32 %, Dec 2024 97.52 -> 97.44 % (1 input). Threshold 0.5: pooled OOF exact is
flat over 0.2-0.6 (99.30-99.34 %); tuned per-period thresholds wander (0.1-0.6) with no material change.
Gain: cov1 55 %, call1 17 %, margin_walk 13 %, call_lift 4 %, free_nowalk 3 %, margin_call 3 %, rank_cov 1 %, same_number 1 %.

Exact-set accuracy per input (measured; threshold tuned on the train period's OOF).
Which tests score unseen signals: Sep CV, Oct CV and pooled OOF are device-grouped (every test signal held out of
training). The first cross-period / Dec 2024 runs trained on every signal of the training period(s), and 756 / 758 Oct
signals (2,523 / 2,548 Oct inputs) are in Sep, 369 / 371 Dec 2024 signals in Sep or Oct - so those are "new days, same
signals" (rows marked *same signals*). `unseen_check.py` repeats them with device-grouped folds (train on the other
period(s) of in-fold signals, score held-out signals only) - rows marked *unseen signals*.
| | Sep CV | Oct CV | Sep -> Oct | Oct -> Sep | pooled OOF (thr .5) | Dec 2024 (stale labels) |
|---|---|---|---|---|---|---|
| model, unseen signals | 99.37 % | 99.50 % | 99.38 % | 98.49 % | 99.32 % | 97.52 % |
| model, same signals | - | - | 99.55 % | 99.04 % | - | 97.44 % |
| model without same_number, unseen signals | 97.83 % | 98.56 % | 98.31 % | 97.39 % | 98.27 % | 97.60 % |
| model without same_number, same signals | - | - | 98.68 % | 97.79 % | - | 97.36 % |
| same number (2/4/6/8 -> n; no training) | 94.30 % | 95.38 % | 95.38 % | 94.30 % | ~94.8 % | 93.77 % |
| best single feature (cov1 >= .26 / .35; one network-wide threshold) | - | - | 97.69 % | 95.33 % | - | - |
Seeing the signal in training is worth ~0.2-0.5 points cross-period (non-standard inputs: Sep -> Oct 96.1 -> 92.2 %,
Oct -> Sep 86.7 -> 79.0 %; Oct -> Sep overlap inputs 56 -> 25 %, n = 32, because Oct has no overlap rows). Dec 2024 is
unchanged within noise (97.44 vs 97.52 %). The headline number stays the pooled device-grouped OOF (99.32 %).
Pooled OOF (5,143 input-periods, device-grouped folds over Sep + Oct): pair precision .997, recall .996, F1 .997; standard
(input n -> phase n only) 99.69 % (4,898); non-standard 91.8 % (245): 2+ movements 92.0 % (150), with an overlap 100 % (32),
other single (e.g. 12 -> 2, 4 -> 5) 87.3 % (63). Model says nothing while config assigns: 8; config nothing, model
something: 1 input (2B357 input 7: 648 presses, the same count as input 4 -> likely one button wired to two inputs; the model
reads ph4). Oct -> Sep overlap inputs 56 % (Oct has no overlap rows to learn from). Without same_number the model fails
mostly on ped-recall signals (no 45 calls: phases 4 and 8 look identical) - same_number is the right tie-breaker there.
ped_health's rule Phase column vs the configured phase (Sep, 2,610 single-phase inputs): 99.20 %; model OOF 99.54 %.

**Disagreements** (pooled OOF, mean over periods; 20 inputs in all, all in `disagreements.csv`; not judged as truth):
05037 / 05039 / 03123 inputs 4 and 8 - presses of either call both 4 and 8 (config: one each); 13002-13005 plan 3 adds
4 <-> 8 but idle presses never call the extra phase (plan 3 probably not running in Sep / Oct; they are on plan 3 now);
03028 inputs 12/14/16/18 -> 2/4/6/8: a 45 within 1 s of 95-99 % of presses but a walk within 2 min after only 4-13 % (model
says none, p .2-.3: probably a model miss); 2B411 input 6: calls ph4 (p 1.0) not ph6; 2B357 input 7 (above);
05055 inputs 2 / 8 (1-2 presses); 2B503 input 4 + overlap 13 (8 presses); 2C044 input 2 drops ph8.

**Runtime** (package `ped_assignments`, DuckDB 6 threads / 4 GB, onnxruntime 4 threads; another workflow's benchmark held the
heavy lock during the runs): Sep window 2.75 d, 816 signals, 519 M events: 5.6-5.9 s (2,724 inputs); Dec 2024 3 d, 421
signals, 339 M events: 4.7-5.1 s (1,392 inputs). ONNX = LightGBM within 2e-14 (raw), 0 decision differences on 23,268 rows;
package features = research features within 5e-15.

Limits: labels are the 2026-10-28 settings (Sep / Oct only ~1 month earlier; Dec 2024 differences may be real changes);
movements with no ped activity in the window can't be seen (configured "phase n + overlap n" overlaps, and every overlap
when the source drops 67 / 68 - the Oct pull did: those inputs come out no_movement); mirrored inputs (one button on two
inputs) inherit the other input's phase; < 10 presses is flagged low_support; no separate locked test set (the 115
locked signals are not in the ped data) - unseen-signal numbers come from device-grouped folds only.

## Step 4: the model no longer knows any number (2026-11-02)
User rulings: same rule as the vehicle model (nothing specific to the numbers may be learned); no 2024 data at all; plan 1 labels at
13002-13005 for Sep and Oct; a phase that walks regardless of the button is "ped in recall": no answer, no guess, not a configuration
issue, not scored. Code `research/code/ped156/v2/` (ped_features2, ped_data2, train2, final2, invariance2, build_candidate); outputs
`dc_work/pedcfg/model_v2/final/` (results.json, errors.csv, invariance.json, control_old.json, health_{old,new}.json, oof.parquet);
candidate `model_v2/candidate`, `model_v2/swap_patch.md`. Not applied to the package. Dec 2024 is in none of this.

**What the vehicle model does** (ledger, note 76, check.py): no phase or channel number is ever an input, no tie is broken by one,
and check.py renumbers the raw log and requires every probability to follow. The first ped model broke this (same_number, 1 % of gain,
carrying about 1 point of accuracy, mostly on recall signals).

**Number audit** (measured by the tests below unless marked).
| where | finding | now |
|---|---|---|
| same_number | direct leak | removed |
| renumbered training copy | only changed same_number | removed |
| call_base, walk_base | float sums depend on row order, which moves with the numbering; an ulp can flip a rank between two phases that time together (note 76's leak) | summed in whole ms |
| rank_*, margin_* | rank() keeps ties, max excluding self has no order | clean, unchanged |
| idle state, ASOF joins | one state per instant by priority; ties equal in value | clean, unchanged |
| candidate list | every number in range treated the same; 1-16 / 1-32 is a validity filter | clean, unchanged |
| is_overlap | a kind, allowed; overlap numbers are never compared with anything now | clean |
| top_movement, phase_guess, phase_2nd | an exact tie went to the lowest number | empty on a tie |
| confidence | product order moved with the numbering | sorted first |
| training row order | sorted by number, and bagging reads row order | sorted by content |
| scoring groups ("standard") | use numbers for reporting only | allowed |
| ped_health's rule Phase column | not the model: ties go to "the phase numbered like the input" | open, user's call |

**Invariance test.** `tests/test_ped_invariance.py` (64 tests with test_ped_assign, all pass): random one-to-one maps of phases onto 1-16,
overlaps onto 1-32, inputs onto 1-72 in the raw log; features, probabilities and answers compared bit for bit. Real signals (invariance2.py,
3 renumberings, 135 Sep + 97 Oct signals, all recall and overlap signals included): 2,286 input answers and 8,133 candidate probabilities, 0 differ,
largest gap 0. Same test on the shipped model (control): 77 of 762 answers and 729 of 2,711 probabilities change. Training: all of Sep + Oct
renumbered per signal, features recomputed from the renumbered raw events: 17,921 rows, trees identical to the unscrambled model (2 of 2).
Research SQL = package template on the same 232 signal-periods: 2,711 rows, largest gap exactly 0. ONNX = LightGBM within 2e-14, 0 decisions differ on 18,767 rows.

**Recall** (measured; first rule, replaced by Step 5 below). A movement is in recall when at least half of its walks (>= 10 walks) had no press on any input of the signal since its
previous walk. Clean split: 5,081 of 5,229 movement-windows are under .1, only 7 between .1 and .5 (highest .35-.39 at 04119 ph 6, where calls match walks).
A button keeps its link if its own presses place calls (call1 >= .05, cov1 >= .5): 37 of 37 such rows are configured links; without it 22 % are, which
is why nothing can be read there. Those rows are gated: not trained on, never predicted. 136 movement-windows at 40 signals are in recall.
**88 of 5,143 button-windows are not scored** (59 buttons, 27 signals; Sep 58, Oct 30; all ordinary in the settings). The shipped model "got" all
88 by the number hint; the same model without the hint got 20 and guessed 2+ movements on 27. The packaged model outputs 59 + 30 = 89: these 88 plus 2B411 input 6 in Sep, a scored button whose held-out answer is ph 4 (see "Held-out answers vs the packaged model").
The rule constants were set looking at these windows (inferred stable: the split is wide).
Saved no-hint errors (103): 68 now recall, 20 now right, 15 still wrong. Ordinary buttons predicted 2+ movements (49 with plan 1 labels, the 41 was on the old
labels): 27 recall, 9 right, 13 still predicted 2+, and those are the signals where the log itself shows both phases called (below).

**03028 inputs 12 / 14 / 16 / 18** (measured, Sep + Oct, 301 presses). 97-100 % of the presses of each input land inside a walk of its own phase, 5.2 s after the walk began
(p10-p90 5.1-5.2 s, all 8 input-periods). The controller logs the ped call within 1 s, but the walk is already on, so usually no new walk follows: a walk within 2 min after only
4-20 % of presses. No person presses that exactly, so these are not people (inferred: a logic setting or a device in the cabinet, worth asking the signal shop). Inputs
2 / 4 / 6 / 8 are the real buttons there. 11001 inputs 14 / 18 have the same machine timing, 0.1 s after the cross phase's walk begins. Fix (no number): `served`
counts a press during the walk as served; it replaces walk120 in the walk features. All 8 become right, nothing else moved beyond noise.

**Model.** LightGBM binary, 23 features, same parameters, 300 rounds, seed 0, threshold fixed at 0.5, rows in content order. Gain: margin_served 40 %, cov1 40 %,
served 16 %, the rest under 1 % each. Package run on Sep: 6.0 s (shipped 4.8 s; shared machine). `ped_health(assign=True)`: every status count unchanged on Sep and Oct (packaged model on windows it trained on; held-out answers give the same flags, see below).

**Results** (folds grouped by signal, threshold .5). OLD as shipped = saved OOF on the old labels, every button counted (ties out to the 99.32 % reported).
OLD same scoring = the same probabilities on plan 1 labels, recall not counted, identical scored set as NEW.
| | OLD as shipped | OLD same scoring | NEW |
|---|---|---|---|
| all buttons | 99.32 % of 5,143 | 99.53 % (5,031 / 5,055) | **99.68 % (5,039 / 5,055)** |
| ordinary | 99.69 % of 4,898 | 99.71 % (4,812 / 4,826) | 99.71 % (4,812 / 4,826) |
| several movements | 92.0 % of 150 | 98.5 % (132 / 134) | 98.5 % (132 / 134) |
| outputs 2+ for them | - | 132 of 134 | 132 of 134 |
| overlap | 100 % of 32 | 100 % | 100 % |
| other single | 87.3 % of 63 | 87.3 % (55 / 63) | **100 % (63 / 63)** |
| recall, not scored | - | 88 | 88 |
| links: of those predicted, in settings | .997 | .9973 | .9973 |
| links: of those in settings, found | .996 | .9981 | .9996 |
| train Sep, test Oct, held-out signals | 99.26 % | 99.42 % | 99.62 % (2,385 / 2,394) |
| train Oct, test Sep, held-out signals | 98.42 % | 98.80 % | 98.57 % (2,623 / 2,661) |
Oct -> Sep: Oct has no overlap rows, so overlaps come out 0 of 32 (OLD 6 of 32); everything else in that test is 99.8 %. The final model trains on Sep overlaps.
Seeds: 16, 16, 17 wrong. If every recall button counted as wrong: 97.98 %. Simple rule "button n calls phase n": 95.55 % of all 5,143.
Slightly optimistic: the feature set (`served`) and the recall constants were chosen on these same data; the threshold was not tuned.

**Remaining 16 errors** (errors.csv; model vs settings, not judged as truth).
05037 inputs 4, 8 (Sep, Oct), 05039 inputs 4, 8 (Sep, Oct), 03123 input 4 (Sep, Oct): presses call both phases 4 and 8, settings list one (10 errors, the model reports what the log shows).
05055 inputs 2, 8 (Oct, 1 press each). 2B503 input 4 (Sep, 8 presses, adds overlap 13). 2B553 input 8 (Oct, 8 presses, settings 4 + 8, model 8).
2B094 input 4 (Oct, settings 4 + 8, model 8: the two phases look almost the same, a real miss). 2B411 input 6 (Sep: settings ph 6 which is in recall, calls go to ph 4 at p 1.0; on the review sheet).

**Held-out answers vs the packaged model** (measured, `v2/insample_check.py`, `final/insample_check.json`). Every result above, and the 16 errors, come from models that did not see the signal.
The packaged model trained on Sep and Oct, settings included, so it has learned these windows' answers. Run on those windows it differs from the held-out answer on 19 of 5,272 buttons (Sep 8, Oct 11): all 16 errors, plus 04056 inputs 4 and 6 in Oct and input 4 in Sep (3-5 presses, not scored).
| the 16 errors | packaged model, same window | model trained on the other month |
|---|---|---|
| matches the settings | 14 | 2 (2B094 input 4, 2B503 input 4) |
| still disagrees | 1 (05055 input 2, 1 press) | 14 |
| no answer | 1 (2B411 input 6: `ped in recall`) | 0 |
So a Sep or Oct run after the swap shows 1 disagreement where the fair test shows 16. It is not an accuracy figure and it hides the disagreements worth reading; use errors.csv for those.
The memory does not carry to another month: each month scored by a model trained on the other month with the same signals in training gets 2,386 / 2,394 (Sep -> Oct, 99.67 %) and 2,623 / 2,661 (Oct -> Sep, 98.57 %, including the 32 overlaps it cannot know), against 2,385 and 2,623 with those signals held out. 05037, 05039 and 03123 still come out as 4 + 8 (05039 input 4 in Oct as 8 alone).
Inferred from that: a window after Oct is a fair test of the packaged model, at signals it knows as well; Sep and Oct are not. Not measured: no later window was run, and overlap buttons cannot be tested across months because Oct logs no overlap events.
Status counts: held-out answers leave the same inputs with >= 10 presses without an answer as the packaged model does (Sep 2, Oct 27), so the "every status count unchanged" result holds on fair answers too. Sep -> Oct adds none. Oct -> Sep adds 30 Sep buttons, all set to overlaps, which an Oct-only model cannot know; the final model does train on overlaps.

## Step 5: recall by ped calls, not presses (2026-11-02)
Bug (measured by the check, `model_v2/verify/x12-x14`): "no press on any button since the previous walk" depends on press traffic. On short windows real recall phases (08050, 08043) fell under the limit and their buttons became config issues; the user ruled recall gets the note, never a config issue.
**Rule now:** a phase is in recall when at least half of its walks (5 walks or more) had no ped call (45) on it since its previous walk (`walk_nocall`). Overlaps have no call code, so they stay out of it: the simplest option, and no overlap met the press-based rule on Sep + Oct, so nothing is lost (measured). An overlap button whose overlap walks by itself could still read as "calls nothing" (inferred; not seen).
**Too few walks:** a phase with 1-4 walks in the window, not shown to be called, and no calls from this button: a button with no answer next to it is `low_support` ("not enough data"), never a config issue. At 5 walks and up nothing else flags: on the Saturday windows 0 of 1,282-2,268 non-recall phases, 81 of 82 recall phases caught at 1 h (measured); one non-recall phase flagged at 3 walks in the 1 h window, hence 5 with a margin. Fired 0 times on real windows here; synthetic tests only. The limits were chosen on these windows.
**Effect on full windows:** none. Same 136 movement-windows (135 at >= .9), same 405 gated rows, so the refit gives the same model file (sha 1d10ec7b...), same 5,039 / 5,055 right (99.68 %), same groups, same 88 recall, same 16 errors, same "Held-out answers vs the packaged model".
**"Calls nothing" config issues, Saturday 9/19** (`verify/x13_short_health.py`): new vs shipped: full 2 vs 2; 1 day 3 vs 6; 8 h 5 vs 8; 4 h 5 vs 13; 1 h 0 vs 14; Monday 4 h 2 vs 7 (before this fix: 2, 9, 24, 24, 27). None is on a recall phase (`verify/x17_short_after.py`). What is left: 2B555 inputs 3, 5 (no setting, also on the full window); 06030 input 8, 08128 input 6 (set to an overlap with no walk in the window); 04089 input 4, 2B020 input 8, 04092 input 8 (set to 4 + 8, phases called, model under 0.5 on 4-8 h: a model miss on short windows; the model trained on full windows only). More buttons get the recall note on short windows (59 full, 60-72 short, 56 Monday 4 h); not checked one by one.
**Re-run:** renumbering 3 x 135 Sep + 97 Oct signals: 0 of 2,286 answers and 8,133 probabilities differ, trees identical 2 of 2; research SQL = package on 2,711 rows, gap exactly 0. New tests fail under each of: rule on presses, min walks 10, overlaps let in, too-few-walks rule off. Also removed the two Dec 2024 mentions from the candidate's README and ped_health docstring. Not re-audited beyond this. ped_health's own recall note (no ped call on 90 % of 50 walks or more) is a separate rule and unchanged.
**Tests** on a fresh copy of dev with the swap: DuckDB 1.5.2 260 passed, 2 skipped; DuckDB 2.0 258 passed, 4 skipped.
