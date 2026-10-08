# 87 — Label-discretion review sheet: every training / dev label we dropped or changed (2026-10-04)
Brief (orchestrator, for the user): list EVERY label on training / dev signals that a rule or an agent's judgment left
out of training / scoring or changed, so the user can say whether each decision was right (wrongly dropped labels starve
or skew training). CPU only (DuckDB 4 threads), nothing trained or re-scored, GPU untouched. Locked answer key never
opened (only its column names, to answer the "same rules on locked?" question from code); locked_v2 asserted absent.
Code `evaluation/review87.py` (items | data | write); work `%DC_WORK%/rev87/` (items, counts.json, rows, chart data). Sheet `review/label_discretion_review_v1.xlsx`
+ `review/label_discretion_review_v1_charts/`.
## Universe (v4l, 16,775 rows, 15,278 labelled; one item per labelled detector and decision)
| decision (primary order) | detectors | train / dev / released | ATSPM label | model agrees (ATSPM) |
|---|---|---|---|---|
| whole signal unusual_layout (21 signals) | 438 | 333 / 22 / 83 | 346 | - |
| R1 loop + config Count / YR, no print reading -> out | 13 | 13 / 0 / 0 | 13 | 0 |
| R1 same, print reading used | 53 | 44 / 9 / 0 | 53 | 5 |
| YR identical to a Count zone -> out | 10 | 8 / 2 / 0 | 10 | 2 |
| R5 low-conf print reading tied by behaviour -> out | 68 | 63 / 1 / 4 | 40 | 29 |
| R6 unused input < 15 ONs / explained-other sweep -> out | 10 / 25 | 10 / 0 / 0, 23 / 0 / 2 | 0 | 0 |
| print re-read flipped the class / removed truth | 11 / 4 | 11 / 0 / 0, 4 / 0 / 0 | 1 / 3 | 0 / 0 |
| stacked group relabel | 41 | 36 / 4 / 1 | 18 | 11 |
| misconfigured (label kept) -> out of training | 70 | 61 / 5 / 4 | 70 | 25 |
| behaviour check failed -> out of training | 405 | 343 / 21 / 41 | 339 | 128 |
| unhealthy by actuations -> out of training | 166 | 139 / 4 / 23 | 137 | 117 |
| DQ check (non-fault part) -> out of training (--clean) | 114 | 105 / 5 / 4 | 83 | 68 |
| Dec-2024 config disagrees -> Dec samples dropped | 621 | 569 / 52 / 0 | 432 | 238 |
dev = the 2025 validation split (38 signals); released = the released NEWTEST half. 1,611 detector-decision items,
1,535 detectors + 21 whole signals (438 detectors) = 1,973 detectors with a decision. Model = champion note-77 OOF on
v4l (rev81 function_rows), majority answer over Sept samples >= 30 min (Dec samples for the Dec-role rows).
Not listed (counted): user's own decisions (R2 2B091 / R4: 16 labelled; source user_ruling 396); no_data 1,268 and
dead 1,014 labelled rows (no evidence either way); not_checkable rows not re-admitted by the print-high rule: 657
(Advance 430, Mid 111, Presence 91; config 454, print_medium 171) -- out of training for lack of a check, not a judgment;
DQ rows whose only evidence is detector fault events 84-88: 233 (see finding 1).
## Sheet
Priority tiers: 1 = an ATSPM label dropped / changed where the model's majority answer IS the original label;
2 = other ATSPM decisions; 3 = non-ATSPM. Rows grouped by signal (signal order = best tier, then model confidence);
one row per detector, collapsed when the same signal + decision + label + action + sub-reason (check code / DQ flags /
Dec class) + tier; a detector with several decisions shows the first and names the others ("Also: ..."). Columns: #,
signal, detector(s), label we had, what we did, why (one plain sentence, numbers from the check text), model says (class,
mean p, samples or detectors), chart, print, answer. Question on top (Y / N / ?). Tab "counts" = the table above.
Chart per row (lead detector; Dec log for Dec rows): review72 layout, title = label, action, plain reason.
**Result: 1,001 rows on 485 signals** (tier 1: 427, tier 2: 352, tier 3: 222; 336 multi-detector; 980 with a print link;
21 whole-signal rows; 33 rows without a >= 30-min model answer). Rows by decision: check failed 292, Dec-role 280,
unhealthy 135, DQ 99, misconfigured 48, R5 39, R1 print 25, stack 23, unusual signal 21, R6 9, sweep 8, R1 out 8,
YR identical 6, re-read flip 5, re-read removed 3. Build 9 min (charts on the share), rewrite 6 s.
Spot-check rows 3, 57, 300, 640, 930 (2B349 d1-3/15-17 Presence, Dec export Advance, model Presence 14/14 .94 on d1;
2B320 d1 Mid vs Dec Advance, model Advance .97; 03108 d5 Count Dec, no Dec model answer; 11014 d9 Advance fail
advance_red_arrivals 24 % vs 91 %, model Advance .63; 2B559 d43 Advance fail advance_leads_lane): labels, checks, Dec
classes (dec_role_changed_v4l) and model answers match the source tables; charts exist and match; print links resolve.
## Findings for the orchestrator
1. RULE VIOLATION: `v3_retrain --clean` drops every `dq_suspect` row from training; `dq_core` health counts fault events
   84-88 (`FAULT_N` / `FAULT_FRAC`), which the user banned for health / validation (2026-09-28). 233 labelled rows
   that pass every other gate are out of training ONLY on fault events (Advance 70, YR 57, Mid 49, Presence 30, Count
   12, Other 12, Bike 3). Not put on the sheet (the user already ruled); fix = recompute dq without faults (or drop the
   health part of dq_suspect) and refit. label_check's own health is actuation-only (fine).
2. Locked answer key (`function_labels_locked_v1`, built by `cab_final --locked-consolidate --locked-labels-only`):
   same print-first rule, whole-intersection Other, user fix_* rules, unusual_layout / dead out. NOT applied there:
   behaviour checks (validated = pending), misconfigured / unhealthy / DQ, Dec-role, stack relabel, YR identical, R1,
   R3, R5, R6, explained-other sweep, note-80 re-reads (cabinet_v4l excludes locked). The label-only rules (R1, YR
   identical, R5, R6, stack relabel, sweep) should be applied to the locked key (labels only, no scoring) before the one
   locked confirmation, or exam truth and training truth follow different rules.
3. Dec-role is the biggest item group (621 detectors; model agrees with today's label in Dec samples on 238 ATSPM ones,
   e.g. 2B349 d1-3 / 15-17 Presence vs Dec export Advance, model Presence .92 in Dec): the Dec export looks wrong there.
