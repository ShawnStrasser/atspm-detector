# 27 — Cabinet-print consolidation: label table v3, tiers, DQ, user lists (2026-09-23)

One command, `research/code/cabinet/cab_final.py` (all 35 batches, 688 signals; idempotent — a second run changes
nothing). Rulings live in `%DC_WORK%/cabinet/final_rulings.csv`; every record change is logged to `sweep_changes.csv`;
records backed up first (`cabinet/backup_pre_final/`). Summary: `cabinet/final_summary.json`.

## Steps
1. Batches processed only when every signal is `visual_done`. Rulings: `unusual_layout` (2B060, side streets rewired
   after the print), `keep_print` (2C009 ch 4/11/18/25/35/40, config "Queue Dummy" but real zones: asserted to stay
   print detectors; the derived-by-description test never touches a print channel).
2. Fixes, now repo copies (`fix_long_presence.py`, `fix_video_yr.py`, new `fix_radar_over_loops.py`):
   * long presence with `--config-desc` (config description Presence/Pres = a Presence code; confidence not raised on
     that evidence alone): 21 detectors / 11 signals, 15 by description only. Its validator now allows YR on non-loops.
   * video YR: 0 left to change.
   * radar over loops refined (orchestrator): 175 `superseded_by_radar` loops, 78 dead (stay). Of 97 live: **25 restored**
     (Advance 15, Mid 10; medium; low if the role came from the config label) because the covering radar zone is Other;
     65 kept (the radar zone has the loop's role); 6 unresolved (zone not named), 1 role unknown. Covering zone found by
     number in the reason (67), neighbouring loops' zones (3), wording (10), radar zones on the phase (11).
     01030 d10/d11 are kept: the long-presence fix made their covering zones Presence first.
3. `cab_build.build` now judges activity in each signal's **newest** window (staging, else Dec 2024): print detector with
   0 ONs -> `dead` (confidence no longer demoted); channel not on the print with >= 5 ONs -> Other row, `explained`
   (reader or derived-by-description, 1,181) or `unexplained` (573). Complete = print read (no active print detector
   left unclassified) and no unexplained channel.
4. Open user questions are overlays (`pending_user` column; answer = one `relabel` line): Q1 upstream P radar zones
   (`advance_presence`, 350), Q2 A(ATSPM)-coded stop-bar zones (13: 12052, 12060, 12072 **and 12067**), Q3 2B338 d10.

## Tiers (688 signals)
complete_high **124** (0 unusual), complete_mixed 383 (31 unusual), incomplete 181 (12 unusual); unusual_layout 43.

## Print detectors (13,204) by function x confidence (high / medium / low)
Advance 2594/754/214 · Presence 2777/622/155 · Count 1539/118/123 · Yellow_Red 545/41/62 · Mid 343/222/32 ·
Bike 472/120/57 · Other 756/544/141 · unclassified 973 (mostly no-diagram prints).

## DQ with print locations (dq_core, newest window, 70 s)
13,204 scored; suspect 13.2 % (4.7 % without dead). By function: Count 6 %, Presence 8 %, Advance 11 %, Other 16 %,
Mid 18 %, YR 24 %, Bike 25 %. Attached to print_labels (`dq_score, dq_flags, dq_reasons, dq_suspect`).

## Label table `research/labels/function_labels_v3.parquet` (README there)
15,479 rows / 715 training-dev signals (688 print + 27 v2-only); locked signals asserted absent (two TEST signals missing
from the plans table were caught by the assert and are now filtered from the hold-out CSVs directly).
* `label_print_first`: print_high 9,026 · config 2,280 · print_medium 1,354 · whole_intersection_other 895 · print_low 468.
* `label_print_agree` (OOF = note 25 b7, six folds, 3-seed mean, full window; 9,300 rows have one): 283 rows differ.
* Classes: Presence 3,845 · Advance 3,835 · Other 2,447 · Count 1,937 · Bike 686 · YR 673 · Mid 600.
* `train_use` 12,057 (complete_high 2,332 on 124 signals; complete_mixed 7,446; incomplete 1,883; no print 396).

## User lists (review/, git-ignored)
* `print_label_overrides.xlsx`: **594** high-confidence print labels that change the v2 label; model agrees 311 (of 565
  with a prediction). Top pairs: Other->Advance 76, Advance->Presence 75, Presence->Count 52, Presence->Other 51,
  Advance->Other 51, Count->Presence 28, Other->Presence 26, Presence->Mid 26. Sorted by review priority (`--review-only`
  rebuilds both lists): x(1+2p) model sides with config, x0.5 model agrees with print; x2 ATSPM class either side;
  x(1+log10(1+ONs)/4); x0.6 print cites zone table/written code; x0.3 unusual_layout (75 rows); x0.1 + last for the 35
  Other/Mid/Bike-only rows. Top = 03104 d6/d1, 03102 d7 (config sides, ~90 %).
* `dead_detectors_from_prints.csv`: **882** (712 in timing + 170 `in_timing` = no, drawn but not programmed, kept per
  user). Steps: 1,181 dead -> 1,016 not stale/unusual signal (46 signals) -> 882 not abandoned (superseded /
  not-commissioned / moved / outdated) -> 882 Sep-2026 window -> 882 device has other live channels.

## Open
* 573 unexplained active channels sit on incomplete signals and stay unlabelled; the 1,181 dead print detectors are
  high for a 66-h window — part are stale prints on otherwise-live signals (a documentation, not maintenance, issue).
* Next: retrain function on complete_high (+ mixed) with 7 classes, both label variants, unusual_layout scored apart.
