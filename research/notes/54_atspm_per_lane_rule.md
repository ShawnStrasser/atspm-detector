# 54 — ATSPM classes one per lane: per-lane decode, stack-aware ATSPM scoring, v3s labels (no retrain) (2026-09-30)
User rule 2026-09-30 (AGENTS.md "ATSPM classes, one per lane, and scoring"). function_v3e OOF re-scored (frame v6e, run `..._h3`, 3 seeds); nothing trained, `model/` untouched. locked_v2 asserted absent everywhere.
Code: `lanes/atspm_decode.py` (numpy), `lanes/ln5_extend.py`, `evaluation/atspm_score.py`, `cabinet/stack_labels_v3s.py`, `trackA/v3_retrain.py --labels v3s --dec-role-rule fp`. Output `%DC_WORK%/trackA/atspm54/v3e/score.{txt,json}`.
## Decode
* Lanes: `ln5_extend.py` fits note 42's pair model per fold (six held-out-fold models reproduce ln2's OOF P(same) exactly,
  max diff 0.0) and runs the lane decode on EVERY frame signal, both periods, all 14 windows (456,033 detector-windows,
  lane known .77; < 10 actuations / Bike get none), inputs = v6e predicted phase + v3e OOF function. ~18 min, 5 workers.
* Per (signal, period, window, predicted phase): (detector, class) pairs taken in order of probability; an ATSPM class is
  taken only if free on every lane the detector covers, else the detector moves to its next pair (may be another free
  ATSPM class, e.g. Count -> Yellow_Red); Other / Mid / Bike repeat freely; no lane known = unconstrained.
* Spanning ("strict"): a multi-lane detector holds its class on every lane it spans (one YR bar across two lanes excludes
  a per-lane YR; a multi-lane radar advance zone excludes loops of that class beneath it). "single" (spanning exempt) scores lower.
* **Gated at >= 30 min.** Ungated, the decode costs 4.1 pt at 5 min [-4.6,-3.3] and 1.3 pt at 10 min (pair model is
  trained on >= 30-min windows only; lanes unreliable there); 5 / 10 min keep argmax. Chosen on OOF — a tuning choice.
* Variants (step-4 rows, ATSPM E): argmax .8874, greedy .8892, exact (max sum log p) .8897, loser -> best non-ATSPM
  (user's literal wording) .8837, spanning exempt .8883. Greedy kept (portable, ~= exact).
## Scoring (`atspm_score.py`)
ATSPM score = 1 - errors / rows; errors: ATSPM truth called non-ATSPM or another ATSPM class; non-ATSPM called ATSPM.
Stack credit: members of a stacked group (all with truth = role) — the highest-P(role) member called the role is right,
extra members called the role are `stack_extra` errors (reported apart), members called non-ATSPM are right if one got
the role, else the highest-P(role) one is a miss. Secondary = exact class among non-ATSPM rows (Other/Mid/Bike). acc7 kept.
| step (all rows; everything / realistic) | n E | acc7 E / R | **ATSPM E / R** | stack_extra E | secondary E |
|---|---|---|---|---|---|
| 0 argmax, v3 truth (= note 49) | 268,861 | .8724 / .8845 | .8768 / .8889 | 0 | .963 |
| 1 + per-lane decode (>= 30 min) | 268,861 | .8721 / .8843 | .8778 / .8899 | 0 | .955 |
| 2 + stack credit (v3 truth: 12 groups qualify) | 268,861 | .8721 / .8843 | .8778 / .8899 | 6 | .955 |
| 3 + v3s truth (stack relabel, 7 YR==Count out) | 269,566 | .8725 / .8847 | .8791 / .8912 | 428 | .955 |
| **4 + Dec-role, function/phase rule** | 261,528 | .8827 / .8952 | **.8892 / .9015** | 412 | .957 |
| 4b step 4 with note 52's full Dec rule | 253,305 | .8853 / .8973 | .8919 / .9037 | 351 | .957 |
* Decode effect vs argmax on the step-4 rows (.8874 / .8998, extra 653): +0.18 pt ATSPM [-0.07, +0.45] everything, +0.17 realistic — inside noise; acc7 -0.13 (losers go to Other). ln3 lanes = ln5 lanes (.9140 vs .9139). By window
  (step-4 set): 30 min +0.15 [-0.18,+0.48], 1 h +0.29, 3 h -0.03, 6 h +0.17, 24 h +0.43 [-0.03,+0.92], **full +0.84 [+0.36,+1.37]**.
* Stacked members, >= 30 min: ATSPM acc .634 (argmax) -> .786 (decode); stack_extra 445 -> 204. The 412 left: 208 on
  5 / 10 min (no decode) + members the lane model split into different lanes (~16 % of stacked group-windows).
* Changed rows (ln3 scope) 2,298, 1,101 now right: Count->YR (truth YR 382 / Count 161), Presence->Other (Pres 230 / Other 186). Errors by truth (step 4, E): Adv 7,386 (+320 stack_extra), Pres 5,589, Count 5,336, YR 2,670; Other->A 5,334, Mid 1,606, Bike 656.
## Label side (`stack_labels_v3s.py` -> `research/labels/function_labels_v3s.parquet`, git-ignored)
* 29 stacked groups (all unreviewed: blank answers apply; "no" skips, "?" excludes; `--only-yes` for yes only): 69 members,
  41 relabelled (`stack_relabel`; function / label_print_first / print_function / truth_v3s = role; old values in *_v3).
* 7 YR zones identical to a Count zone -> `yr_count_identical`, `exclude_train_score`.
* 33 radar/video-over-loops signals re-admitted (`readmit_radar_over_loops`, 980 rows; incl. 04073 / 13025 / 2C023 from
  v3_retrain's EXTRA_UNUSUAL); 22 stay unusual (rewired / stale print, two intersections, railroad, reused channels).
* Dec-role recomputed on v3s labels, function/phase statuses only -> `cabinet/dec_role_changed_v3s.parquet`. Training rows
  (dry run, note-49 recipe + fp rule): v3 283,894 (659 sig.) -> v3 fp 276,781 -> **v3s fp 297,280 (692 sig.)**; v3 + full rule 268,871 (= note 52).
## Retrain (after the user's review of review/stacked_detectors_review.xlsx); then the caveats
`cabinet/stack_labels_v3s.py`; `trackA/v3_retrain.py fit --frame v6e --labels v3s --min-on 5 --clean --require-validated
--allow-not-checkable-high --health3 --dec-role --dec-role-rule fp --variants first.all.wi --seeds 0,1,2 --threads 6`;
`lanes/ln5_extend.py lanes --run <new ..._h3_drfp run> --tag v3s`; `evaluation/atspm_score.py --run <new run> --tag v3s
--lanes ln5_lanes_v3s --base run_ad6ea6959f_exclude_min5_clean_valnc_h3 --base-lanes ln5_lanes_v3e` (paired CI vs v3e).
* Caveats: decode gain not above noise overall (pays on long samples and stacks); gate chosen on OOF. Relabelled rows keep
  their old `validated` status (label checks ran on the old class); v3s is provisional until the user's review.
