# 29 — Label-behaviour validation v3: data cleansing of the training labels (2026-09-28)
**2026-09-29: superseded in part by note 44 (v4: behaviour-found permissive phases, right-turn lanes, `misconfigured`).**
Code `cabinet/label_check.py` (cab_final step 7), follow-up `label_spotcheck.py`; v2 backups `%DC_WORK%/cabinet/backup_label_check/`.
Rules may use anything (print, timing, FYA events): cleansing only, never model inputs. Thresholds frozen (v2 calibration).
## Rules kept from v2 (AUC on trusted rows; limit)
| rule | stat | AUC | limit |
|---|---|---|---|
| Presence holds the call through red at peak | hold_frac_peak | .940 | loop >= 6.8 %, radar >= 12.1 %, else 8.7 % |
| Count off during red / short ONs | red_occ / dur_med | .892 / .865 | radar <= 6.3 %, else 11.1 % / <= 2.0 s |
| Advance arrivals on red (free-running bins only) | red_arr_q | .858 | loop >= .81, else .89 |
| Advance fires 1.5-9 s before the same-lane stop-bar zone | lead - chance | swap .991 | loop >= 4.2 pt, radar 9.2 |
| Loop advance vs same-lane stop-bar loop counts | ratio | lane .864 | >= .44 (now a FIELD issue, see below) |
| Mid tracks advance sum / Bike far below / loop never Count | span_c_off / bike_rel / - | .969 / .979 | .76 / 49 % |
## What the engineer's spot-check (review/spotcheck_label_checks.xlsx, 2026-09-28) changed
1. **2C036 det 2 shown as "pass" — a review-sheet bug, not a data bug.** Det 2 and 3 have 0 actuations in both windows
   (checked on det_intervals); the label table always had them `no_data` (card slot I2 dead), train_use False. The
   spot-check helper added card-fault rows with `only={card_fault}`, but card_fault was never among the listed checks,
   so it fell back to the class checks (all "not checkable"); the condensed sheet then read "no FAIL" as "pass".
   Reach: 1 of 25 detector rows in that sheet; 0 rows in v3 (no `pass` row has 0 actuations). The new sheet takes
   Result straight from `validated`.
2. **Fault events 83-88 never used.** Health = actuations only (stuck > 15 min, chatter > 30 % gaps < 0.3 s,
   saturation, card with BOTH outputs stuck/chattering); new status `unhealthy` (not trained, not a label verdict).
   Card "erratic" from faults alone is dropped. 118 former fails were fault-driven only.
3. **Protected-permissive (FYA) phases.** Left turns leave on the flashing yellow, which the phase's own colour logs
   as red. PPLT = FYA begin-permissive events (32, parameter = left-turn phase, `data/staging_other`, 208 devices;
   Dec pull too) >= 5, or the timing: a detector of phase 1/3/5/7 also calls 2/4/6/8 (e.g. 08040 dets 1, 13).
   343 phases / 178 signals / 775 labelled rows (259 phases by events, 84 by timing only). There presence holds-red
   and count off-in-red are NOT applied, and no phase flag is raised. Evidence: trusted Presence fail that rule on 11.3 %
   of PPLT phases (median hold .30) vs 2.3 % elsewhere (.66). 34 former fails cleared by the exemption.
4. **Order + occupancy cross-check** (engineer's 2B349 50 -> 48 -> 44): of a stop-bar Presence / Count pair (same lane
   or lane unknown, neither pulse-set) the Presence is >= 1.2x more occupied and counts fewer. On trusted pairs it
   fires in the label direction on 53 % and in the swapped direction on 0 %: support = a passed check and may excuse a
   failed count / presence rule (-> field issue); the reverse = fail `role_order_occupancy` (2 dets, 2B143 9/24).
   Advance-fires-first-and-presence-more-occupied (08040 1 -> 13) is weaker (90 % of trusted Presence, 31 % of
   Count): a passed check only, never an excuse. Occupancy / pulse share: `cabinet/label_check_extra.parquet`.
5. **Label wrong vs field problem.** Only `fail` removes a label. Label kept, `field_issue` set: presence zone set to
   pulse (>= 80 % of ONs <= 0.3 s; 42 Presence rows, 3 trusted; e.g. 2B108 4/18, 2B066 11, 01080 16); count zone not in
   pulse where vehicles stop, with pair support (2B349 44, 04034 4); advance loop undercounting its same-lane loop
   (2B422 8: all 62 former loop-count fails; OOF model disagreed with only 3 % of them).
6. **review/field_issues_for_staff.xlsx** (new, cab_final step 8): 859 rows / 447 signals — dead channels per signal
   with dead card slots (285), label-check config / health issues (250 + 5 confirmed by the engineer), clear-print
   labels the detector does not behave like (180), print vs timing phase / numbering / outdated print (139).
7. Radar ETA zones stay Other (subtype eta); a dedicated ETA class is only a planned experiment.
## Results (v3 of the check, 14,023 labelled non-released rows; released NEWTEST rows stay `pending`)
pass 10,755 · **fail 504 (4.5 % of checked; v2 846, 7.5 %)** · unhealthy 78 · not_checkable 1,496 · no_data 1,190.
Of v2's 846 fails: 504 still fail, 244 pass, 60 not_checkable (mostly pulse-set presence), 78 unhealthy.
Fail rate by class: Advance 7.3 % · Count 7.6 · Mid 7.1 · Bike 6.0 · Presence 3.9 · YR 0 · Other 0.
Failures by check: presence 137, advance order 121, count red 72, count short 67, red arrivals 62, loop-Count 56,
bike 34, mid 28, swap 1 (+ card 303, health 75, saturation 7: no_data / unhealthy, not label verdicts).
`train_use_validated` (pass-only) 9,577 -> **9,806**. Whether not_checkable rows train is still the open decision.
## Follow-up spot-check (review/spotcheck_label_checks.xlsx; charts review/spotcheck_label_charts/)
10 rows: 2C036 d2 (corrected, no data), 08040 d13 + 11039 d15 (FYA exemption), 01080 d16 (pulse presence), 2B349 d44 +
04034 d4 (order / occupancy -> Count kept, field issue), 2B422 d8 (advance misses vehicles), 11042 d22 (fault-only
fail now passes), 2B143 d9 (new swap), 01070 d11 (control: still fails). Earlier answers: sheet 2. Chart data: `cabinet/spotcheck_chart_data/`.
## Open (not done now)
* **Model side: permissive phases from hi-res alone.** On FYA phases a presence zone does not hold through "red"; the
  model sees only 81/82 + colour states. It may need a phase-anonymous feature for "vehicles clear this detector
  while the phase is red but a partner phase is green" (FYA events are not allowed inputs). Research item.
* Decision for the orchestrator: config-issue rows are kept as `pass` when another check passes (pulse-set presence is
  mostly not_checkable, so excluded from pass-only training); a pulse-set Presence behaves like a Count to the model.
