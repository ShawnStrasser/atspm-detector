# 44 — Label check v4: the engineer's spot-check of v3 applied (2026-09-29)
Continues note 29 (full). Code `research/code/cabinet/label_check.py` (v4 block in the docstring), `cab_final.py`
(step 7), rebuilt with `cab_final.py --skip-dq` (log `%DC_WORK%/cabinet/cab_final_lc4b.log`). Backup of everything
before: `%DC_WORK%/cabinet/backup_lc4_20260929/`. Thresholds of note 29 unchanged. Cleansing only, never model inputs.
## His answers (review/spotcheck_label_checks.xlsx, 10 rows) -> changes
1. **FYA events are a clue, not the definition** (11039 d15: some FYA signals log no event 32). Permissive left
   turns now also from BEHAVIOUR (`leave_red_stats`): on the phase's stop-bar zone (label Presence / Count / YR, not
   a right-turn lane) with the most red waits (ON starting in red, >= 3 s; >= 20 of them), the share that leave
   before green (OFF > 2 s before green, no new ON within 3 s) >= .20. Left-turn phase = odd phase, or all its print
   lanes L. Separation, FYA-event phases vs left-turn phases without events at signals that DO log them: AUC .95
   (228 / 48 phases; recall .89, 8 % of those flagged). First try on the phase's busiest-wait detector of any label:
   AUC .82 - an upstream loop also "empties" as the queue moves up (2B349 P3 det 50 was flagged: fixed).
   Phase-call events 43/44 (call dropped on red) were tried first: AUC .66-.75, dropped.
   Permissive phases now 667 on 355 signals: events 331, timing 122, behaviour 464 (222 behaviour only).
2. **Right-turn lanes** (print lane_type R, right turn on red) exempt from presence-holds-red / count-off-in-red
   like FYA: 01070 d11 now passes. Evidence: Presence in R lanes leave before green median .50 vs .01-.16 elsewhere.
3. **Misconfigured = not trained** (01080 d16, "you keep it anyway?"). New status `misconfigured` (label kept,
   excluded): presence zone set to pulse (46), count zone not in pulse where vehicles stop / presence failing with
   order-occupancy support (34). 80 rows (2B349 d44 and 04034 d4 among them - label right, zone set up wrong).
4. **Poor health = not trained.** Advance loop missing vehicles (was a pass with a field issue) -> `unhealthy`
   (71, incl. 2B422 d8); user rulings `unhealthy` in final_rulings.csv (2B422 d8, 11042 d22). Health v3 (note 43)
   is applied at TRAINING-ROW level per period instead (note 45), not in this table; `HEALTH_STATUS` hook unused.
5. **2B143** user rulings (final_rulings.csv, "config labels wrong"): 22, 23 Advance; 24 Mid; 8, 9 Presence. Det 8
   was a data-only channel: a ruling that names signal + detector now reaches data-only rows, and a user ruling wins
   on any row with a function (cab_final `_match`, `build_v3`). field_issues_for_staff.xlsx gets a section
   "controller configuration labels det N as X; it is Y" (only where the config differs: 24, 8, 9).
## Status counts (15,118 labelled rows incl. released NEWTEST; before -> after)
| status | pass | not_checkable | no_data | fail | unhealthy | misconfigured |
|---|---|---|---|---|---|---|
| v3 (note 29) | 11,542 | 1,599 | 1,325 | 556 | 96 | - |
| **v4** | **11,534** | **1,534** | 1,325 | **478** | **167** | **80** |
Fail rate of checked 4.6 % -> 4.0 %. Moves: fail -> pass 58 (right-turn lane 42, behaviour-permissive 13, 2B143 3);
fail -> unhealthy 20; pass -> misconfigured 38, not_checkable -> misconfigured 42; pass -> unhealthy 46;
not_checkable -> pass 18. Exemptions now on 529 right-lane and 722 permissive-phase rows.
`train_use_validated` 10,500 -> 10,491. Released NEWTEST rows: pass 788, nc 98, no_data 135, fail 44, unhealthy 26,
misconfigured 4. Field issues for staff: 883 rows / 478 signals (was 859 / 447).
## Open
* Behaviour-permissive uses the label to pick the stop-bar zone: a wrong stop-bar label on a protected phase could
  exempt itself (the exemption only skips two red-time rules; other checks still run). Only 13 fails cleared by it.
* A dedicated ETA class is still only a planned experiment (note 29 item 7).
