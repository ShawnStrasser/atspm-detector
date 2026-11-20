# 155 - ped push-button health in the package (2.0.0): first pass, parity with note 129, review sheet

Built `src/atspm_detector/ped_health_duckdb.py` -> `ped_health(con, events_table, start, end, periods, max_events)`,
exported from `atspm_detector`; DuckDB SQL temp tables (prefix pq_) per group of signals, findings assembled in pandas.
Output per input: DeviceId, Input, Phase, Type, presses, status (ok/suspect/bad/not_enough_data), reason, config;
periods = vehicle shape (Detector = input; in_closure, whole_window always false). Tests tests/test_ped_health.py (4).
Code research/code/health154/ped2_extract.py, ped2_run.py, ped2_parity.py, ped_sheet_v2.py; work dc_work/ped2.
## User verdicts found (review/ped_health_review_v1.xlsx rows 1-14 answered, 15-30 blank; none in s154/labels.csv)
- Inputs switch together (03097, 09040) = "config: calls two ped phases" -> ok + config note (was suspect).
- Never calls (2B496, 08404, 2B555 ped 3) = maybe ped overlap -> checked: 28 of the 31 Sep never-call inputs are
  followed by a ped OVERLAP walk (67) in 84-100 % of presses. New: overlap fallback (>= 80 %) -> ok + config note.
- Recall (08050) -> config note. 2B555 regular: "say true peds, not people". Day chart preferred, legend not on the
  right (charts now: day panel + zoom, legend on top). 03013 "?" (flash?): no flash, signal cycled normally.
## Rules = note 129 / 129b, plus
- Two-phase buttons (one press logs P4 + P8 45 at once): the mode tie is broken towards phase = input number (research
  took whichever pandas met first); second phase >= 30 % -> config "calls P4 and P8 at once".
- An ON with no OFF before the log ends counts to the end, only if >= 3 walks come in it (2B021 ped 6: 82 min "ON",
  no walk = lost OFF, not stuck). Stuck reason for presence inputs says 10 min.
- Recall note not given to stuck inputs (their held call makes walks look uncalled). not_enough_data = < 10 presses
  and no finding. Severity: 0.5 at the limit, 0.8 at stuck limit / 75 % regular / 100 presses / 20 greens / 40 waits.
## Parity (per input-day, findings on the day their period starts; dc_work/ped2/parity.parquet)
| check | 2026 research / package / both | 2024 research / package / both |
|---|---|---|
| stuck on | 17 / 17 / 17 | 6 / 6 / 6 |
| machine-regular | 0 / 0 / 0 | 6 / 6 / 6 |
| never calls | 153 / 70 / 70 | 9 / 0 / 0 |
| call dropped | 87 / 88 / 86 | 183 / 182 / 180 |
| dead | 1 / 1 / 1 | 0 / 0 / 0 |
- Never calls: every research-only day is a ped-overlap input (83 Sep days, 28 inputs; Dec 2024: 03026, 04035, 2B496).
  Oct pull has no 67 events, so the same overlap inputs stay suspect there (48 of the Oct never-call days).
- Call dropped 8 input-day differences (4 each way): all two-phase buttons (01061, 2B053, 2B025, 2B033, 2B051, a87d162a): same call,
  charged to the other input of the pair (2B053: research lost the P4 drop entirely).
- Phase: identical except 27 (Sep) / 36 (Oct) two-phase buttons. Type: identical 2,749 / 2,749 when run on both
  windows pooled as research did (per window 253 / 277 differ: fewer presses -> unknown, or a pooled type).
- Statuses: research suspects that were only "together" (20 input-days 2026) are now ok + config.
## Rates (push inputs x full days, per 100 input-days) and run time
- 2026 (10,510): bad 0.067, suspect 1.20 (research 0.07 / 1.86). Stuck .067, regular 0, never calls .495 (Sep .07,
  Oct .94 = missing overlap events), call dropped .695, dead .010. Sep alone (overlap events present): bad .11, suspect .55.
- Dec 2024 (3,753): bad .16, suspect 5.0 (call dropped 4.85 = note 129's 2024 figure).
- Run time (8 threads, 8 GB, local parquet sorted by DeviceId/Timestamp): Sep 816 signals 2.75 days 519 M events
  15.5 s; Oct 758 signals 2.3 days 9.4 s; Dec 2024 378 signals 3 days 14.7 s. pytest 33 pass 2 skip; check 7/7.
- ATSPM_Report raw_events (Oct 21-27 2026) holds no ped events (no 21/22/23/45/89/90): cannot run there.
## Sheet review/ped_health_review.xlsx (17 rows, charts review/ped_health_review_charts/, all 17 opened)
1 2B129 stuck, 2 2B021 lost OFF (new rule), 3 09040 ped 8 stuck by 3 walks (user called 09040 config), 4 2C037
presence stuck, 5 2B555 regular (Dec 24), 6 2B555 never calls, 7 08404 overlap, 8 2B496 overlap (Oct data gap), 9 06030
overlap at 84 %, 10 2B392 249 greens, 11 2B095 2-green drop, 12 01067 waited 3 greens then walked, 13 03047 ped 18
dead + copy-of-ped-8 note, 14 03097 together -> config, 15 03013 4 inputs 95 s (not flash), 16 08050 recall, 17 01061
two-phase tie + marginal drop.
## Open
- 09040 ped 8 Sep 19: bad stuck by the 3-walk rule on the evening event the user called config. Decide.
- Call dropped is controller timing, not the button; 2-green limit catches marginal cases (01061: first green 1 s
  after the call). (A proposal to ignore greens < ~5 s after the call was dropped in 155b-3: the controller can
  serve a ped call at once.)
- Overlap threshold 80 % and the together/recall/two-phase config notes are user-verdict applications: confirm on sheet.
- Oct ped pull has half-hour blocks without ped events while the signal logs (03013): dead check uses any event as
  "logging", so a ped-only ingest gap could fake a dead input (1 dead flag found, real). Severity scale is my choice.
## 155b - round 2 (user review 2026-10-29): rules applied on a copy, answers, sheet
Code research/code/health155b/ped3_run.py (off / on / on+orphan on sep, oct, dec), ped3_eval.py (per input-day),
ped3_charts.py, ped3_sheet.py. Package copy dc_work/ped2/src_v3 (real src untouched: other jobs editing it); patch for
the coordinator dc_work/ped2/ped_rules_patch.md / .diff (applies cleanly to the base; full suite 37 pass, 2 skip).
### Rules (approved, on; `options=` switches them, all off = the 155 first pass exactly)
- Machine-regular -> bad always. Never calls / call dropped -> new status `config_issue` (own status value, like the
  vehicle side's lane_closure / timing_plan; order bad > suspect > config_issue > ok); plain config notes unchanged.
- (Superseded by 155b-4: any preemption excuses the whole wait, to its logged end or 6 h.) Preemption explains a call wait ->
  note "call waited during preemption" (severity 0) + config text. Preemption = any
  number, 102 -> its 104 or 105 -> its 111 (unpaired = start alone), + 120 s recovery (atspm phase_wait.sql default);
  only greens outside every preemption count (before it, or after it + recovery; round 2b fix, see below). Sep pull: 102/104 pair 3218 of 3221 (median 20 s);
  111 often missing (1676 of 2425 105 have no 111), hence both pairs. Oct pull has NO preempt events (19 codes only):
  every Oct wait counts as call dropped; Dec 2024 has them (5,554).
- Proposed, OFF: `orphan_clearance` (a 22 without its 21 = a walk whose start was lost). Orphan 22s: 168 of 526k in
  Sep (74 signals), 177 in Oct. Measured on rows 10/11 (2B392, 2B095) and on every Sep preemption note except 2B422 /
  2B042 (2B044 a lost 21 too; 05018, 05028, 05032, wrongly called preemption notes by the first draft, are lost 21s).
### Evaluation (round 2b; superseded by the 155b-4 table below)
| window | off: bad / suspect | on: bad / suspect / config_issue / note | on + orphan: config_issue / note |
|---|---|---|---|
| 2026 Sep+Oct (10,510) | .067 / 1.20 | .067 / .010 / 1.15 / .038 | .60 / .029 |
| Sep only (5,430) | .11 / .55 | .11 / .018 / .46 / .074 | .092 / .055 |
| Dec 2024 (3,753) | .16 / 5.0 | .32 / 0 / 4.77 / .08 | 4.40 / .08 |
- Inputs changed (off -> on): Sep 29 suspect -> config_issue, 4 suspect -> ok (preemption); Oct 73 suspect ->
  config_issue (48 of the Oct days are overlap buttons: the pull lacks 67); Dec 2 suspect -> bad (2B555 ped 3, 5),
  156 -> config_issue, 2 -> ok. Remaining suspect: 03047 ped 18 dead. Call-dropped input-days with orphan on: Sep
  21 -> 1 (10051), Oct 48 -> 10, Dec 179 -> 165. Run time unchanged (Sep 12 s, 4 GB, 6 threads).
### Answers
- 09040 (row 3): ped 2 and 8 are NOT one button (daytime presses independent: ped 2 ~200-500 / day, ped 8 ~15-40).
  Every evening with ped data (Sep 18, 19, 20, Oct 4, 6; Oct 5 17-21 h has no ped events in the pull) 4-10 min after
  computed sunset (45.672 N, 120.835 W): peds 2, 6, 8 ON together ~16 s, then ped 2 (always 231 s) and/or ped 8
  (187-229 s) held 3-4 min, P2 / P8 walking every cycle. Every hold is < 300 s and holds 1-3 walks of its phase
  (Sep 18, 20, Oct 4: 2 + 2; Oct 6: ped 2 2, ped 8 1; Sep 19: ped 8 alone, 3), so only Sep 19 trips the 3-walk rule
  (cycle timing; corrected in 155b-2: the together exemption is not why the others pass). Inferred: an electrical event at dusk (photocell / button power
  unit). Recommendation to the user: one consistent answer (a field fault); decision theirs.
- 03047 (row 13): inputs pair 2+12, 4+14, 6+16, 8+18 (99 % of ped 18 ONs within 0.5 s of a ped 8 ON; ~75-85 % the
  other way). Ped 18 silent Fri 18:45:52 - Sat 14:17:37 and ped 16 Fri 18:43:27 - Sat 14:05:35 while ped 8 (93) and
  ped 6 (63) were pressed (corrected 155b-4: 14:05 is ped 16): dead is right;
  ped 16 not flagged (level too low for 6 h / 20). Shared-part outage (inferred).
- 01061 (row 17): call 10:33:54 for P4+P8; greens 0.8 s later (8 s) and P4 10:34:41 (10 s) without walk; walk 10:35:45.
- 2B021 (row 2): no preempt events at 2B021 in Sep: the 82-min ON without walks is a lost OFF.
- 01067 (row 12): a real wait, 3 short P4 greens (11-20 s), nothing lost; Oct = no preempt data.
### Sheet review/ped_health_review.xlsx (backup dc_work/ped2/ped_health_review_before_round2.xlsx)
19 rows: 3, 13, 17 explained; 5, 6, 10, 11, 12, 17 UPDATED; new 18 (2B422 preemption note), 19 (10051 config_issue
with no preemption and no lost event). Charts review/ped_health_review_charts/r2_*.png (greens as bands, walks, calls,
preemptions shaded).
### Open (user) - answered 2026-10-29, see 155b-4
- 09040 dusk event: candidate 'ON >= 180 s through >= 2 walks' stuck rule -> switched ON (155b-4).
- Turn on orphan_clearance (lost walk start)? -> ON (user: data loss is never a fault; 155b-4).
### 155b-2 - round 2b corrections (2026-10-29, after an independent check)
- Preemption rule fixed (src_v3, patch regenerated). The first draft counted only the greens before the first
  preemption in the wait, so a preemption at the start of a wait excused every later green: 05018 ped 4 (a 23:50 -
  06:18 wait, ~100 P4 greens after a 105/104 logged at its start), 05028 ped 2 (82 greens) and ped 8, 05032 ped 4 and
  ped 8 read "waited during a preemption (fine)". Now a green that starts inside a preemption (start to end + 120 s)
  is excused and only greens outside every preemption count; >= 2 outside = call dropped. Those 5 inputs are
  config_issue again (lost walk starts, as noted above). New test case (ped 12: preempt at the call, 5 greens after
  it -> call dropped); ped 6 case moved so both its greens fall inside the recovery. Synthetic edge cases (verifier's
  edge.py) all call dropped now. Sep notes left: 2B044 4, 2B422 2 and 6, 2B042 8 (x2), 05032 4 (another wait);
  Dec: 2B356 4 and 6, 2B501 8. Package: 7 ped tests pass, full suite 37 pass / 2 skip; run times Sep 18-21 s on this
  share (unchanged vs off). Rates in the table above are the rerun.
- Row 3 (09040) explanation corrected (sheet, chart, this note). The together exemption did not hide the event: each
  hold had too few walks (measured, walks of the input's own phase strictly inside the hold: Sep 18 ped 2 231 s / 2,
  ped 8 217 s / 2; Sep 19 ped 8 227 s / 3; Sep 20 2 / 2; Oct 4 2 / 2; Oct 6 ped 2 231 s / 2, ped 8 187 s / 1) and
  all are < 300 s. Narrowing the exemption alone would flag nothing new. Candidate (tested, OFF;
  code/health155b/ped3_candidate_hold.py, dc_work/ped2/r2b_candidate_hold.csv): stuck also when ON >= 180 s through >=
  2 walks, no together exemption. Sep: stuck input-days 13 -> 18 (full days 8 -> 11, presence inputs included); Oct 4 -> 8
  (full days 3 -> 5); all added ones are push buttons. Adds 09040 on Sep 18, 20, Oct 4, 6 (so all 5 evenings) and 2 others: 2B093 ped 6 (Sep 19, 185 s,
  2 walks) and 03102 ped 8 (Oct 6, 216 s, few presses). bad per 100 full input-days 2026: .067 -> .114. Not checked
  by hand: the 2 others.
- Chart r2_03 redrawn: each evening's panel title gives hold length / walks inside it and the package's answer.
## 155b-4 - rounds 3-4 (user rulings 2026-10-29 afternoon and evening): applied on the copy, patch regenerated
Code research/code/health155b/ped4_run.py (tags off / final / nos2 / noorph / nogaps / late / cap0 on sep, oct, dec),
ped4_eval.py (dc_work/ped2/r4_eval.json), ped3_charts.py (preemption rule updated), ped3_sheet_r4.py. Copy
dc_work/ped2/src_v3; patch dc_work/ped2/ped_rules_patch.md / .diff (applies cleanly to the 16:16 real files; full
suite on a patched copy 46 pass, 2 skip; ped tests 12 pass). All options off = the 155 first pass (Sep, Dec identical;
Oct 1 row = the tie-order fix). Run time unchanged (Sep 15.1 s vs 14.9 s off; 4 GB, 6 threads).
### Rules now (on unless marked)
- Machine-regular -> bad; never calls / call dropped -> config_issue; stuck also ON >= 180 s through >= 2 walks (push
  inputs, no together exemption; 09040 dusk = a fault, user).
- Any preemption during a call's wait excuses the whole wait: note "omitted during preemption" (severity 0). A
  preemption = 102 or 105 to the first 104 / 111 / 110 of its number after it; none logged = still on, cap 6 h (user);
  + 120 s recovery. Sep: 5,646 starts; ended by 104 5,083, 111 549, 110 8; no end 6 (3 signals, Sep 21 03:24); none
  > 6 h; median 19 s. The cap changes no wait in any window (cap0 run identical). No preempt code in the log (Oct pull):
  reason "preemption not checked".
- Data loss never a finding (user): orphan_clearance (22 or 23 with no 21 since the last 23 = a walk from the phase's
  green <= 120 s before; Sep 121 + 216 such events of 526k walks); log_gaps (>= 2 whole clock minutes without any event
  of the signal): a wait ends at the gap, an ON across a gap is judged on walks only, a gap hour is not a "logging"
  hour for dead; never-calls reason says when the log has no 67 at all (status kept, like the preemption wording).
  Remaining paths (not fixed, measured 0): an ON whose OFF is lost at the very end of the log still counts if >= 3 walks
  of its phase come (from other presses); a ped-only logging hole (other codes logged) is not detected.
- Proposed, OFF: late_call - the first green is not owed when the call came after the yellow leading into it began
  (last event 8 of the signal <= 15 s before that green; the log has no ring / sequence, the later yellow of two rings
  can only excuse less). Effect: only 01061 ped 4 (Oct, sheet row 17) in all three windows; row 12 (01067) unchanged
  (call 18 s before the P3 yellow).
### Evaluation (r4_eval.json)
| window | ped inputs | bad (stuck rule off -> on) | suspect | config_issue | notes (waits) |
|---|---|---|---|---|---|
| Sep 18-21 2026 | 2,724 | 6 -> 8 (0.22 -> 0.29 %) | 1 | 3 | 4 |
| Oct 4-6 2026 | 2,548 | 2 -> 5 (0.08 -> 0.20 %) | 0 | 33 | 0 (no preempt data) |
| Dec 2-4 2024 | 1,258 | 8 -> 8 (0.64 %) | 0 | 37 | 1 |
- Stuck rule adds: 09040 ped 2 (Sep, Oct), 09040 ped 8 (Oct), 2B093 ped 6 (Sep 19), 03102 ped 8 (Oct 6). All flags
  together: Sep 0.44 %, Oct 1.5 %, Dec 3.6 % of ped inputs (user guide: <= ~5 %).
- Per 100 push input-days, full days: 2026 bad .114, suspect .010, config_issue .59, note .029 (first pass: bad .067,
  suspect 1.20); 2024 bad .32, config_issue .99, note .027 (first pass bad .16, suspect 5.0).
- Data-loss rules, inputs config_issue -> ok: lost walk start Sep 21, Oct 38, Dec 24; log gaps Sep 0, Oct 2 (14002,
  14005 ped 4: 85- and 43-min gaps), Dec 88. Dec 2024 has 1,152 daytime log gaps (Sep 120 with twice the signals); at
  half of them <= 382 signals logged that minute vs a normal daytime 5th percentile of 391 (feed dropouts).
### Sheet review/ped_health_review.xlsx (backup dc_work/ped2/ped_health_review_before_round4.xlsx)
21 rows, user answers kept. Updated: A1; rows 3 (churn), 10 and 11 (now ok), 12 and 17 (late_call option, chart 17
shows the P2 yellow), 18 (new preemption rule; chart redrawn: active span solid, recovery hatched). Row 13 already
says ped 18 back Sat 14:17:37.

## 155b-5 - round 5 (2026-10-29 evening, after an independent check): data loss closed for Dec 2024 and Oct 2026
The check found Dec 2024 still had 37 config_issue inputs from short feed holes the 2-min gap rule missed (all 38
call-dropped waits started at :12-:14 or :42-:44 past the hour, the minutes of Dec's recurring 60-180 s whole-network
silences). Code research/code/health155b/ped5_run.py (tags as ped4_run + h120 = holes only from 120 s), ped5_eval.py
(dc_work/ped2/r5/r5_eval.json), ped3_sheet_r5.py. Copy and patch as 155b-4 (patch md5 of the result: src
2fc281a94a3a759a8e08a31e73b89242, tests 1bea1a86a4fe12db6e7597b9aefe26c4; applies cleanly to the 16:16 real files; full
suite on a patched copy 49 pass, 2 skip; ped tests 15 pass, 3 new).
### Rules added (under the existing data-loss options, on)
- Log hole: 30 s or more with no event of any kind from the signal (exact, 10-s buckets of all events). A call wait ends
  at the first hole after the call; an ON with a hole inside is judged on walks only. (The 2-min gap still marks the
  dead check's non-logging hours.)
- Lost green start: a 22 / 23 with no 21 is a walk from the phase's green <= 120 s before it, unless the phase's own
  yellow came > 1 s before the 22 / 23 after that green (that green had ended; the new green's start was lost with the
  walk): then the walk starts at the 22 / 23. Dec blocks keep only the don't walk logged with the yellow.
- A log with no ped overlap walk (67) at all (the Oct pull) does not rate "never calls a walk": config note "never calls
  a walk not checked". 23 of the 25 Oct inputs it clears are overlap buttons in Sep (measured).
### Evaluation (r5_eval.json; measured)
| window | ped inputs | bad (stuck rule off -> on) | suspect | config_issue | notes (waits) |
|---|---|---|---|---|---|
| Sep 18-21 2026 | 2,724 | 6 -> 8 (0.22 -> 0.29 %) | 1 (0.04 %) | 3 (0.11 %) | 4 |
| Oct 4-6 2026 | 2,548 | 2 -> 5 (0.08 -> 0.20 %) | 0 | 8 (0.31 %; was 33) | 0 (no preempt data) |
| Dec 2-4 2024 | 1,258 | 8 -> 8 (0.64 %) | 0 | 0 (was 37) | 1 |
- All flags together: Sep 0.44 %, Oct 0.51 %, Dec 0.64 % of ped inputs. Stuck rule adds the same 5 inputs as 155b-4.
- Per 100 push input-days: 2026 bad .114, suspect .010, config_issue .133 (was .59), note .029; 2024 bad .32, config 0.
- Hole limit: Sep and Oct identical at 20, 30, 45, 60, 120 s; Dec config_issue at 120 s 28 -> 0 at 30 s. A 30-s
  whole-signal silence is common at quiet signals (Sep daytime 15,932 over 805 signals in 4 days, night far more), so
  a wait there is judged only on greens before the silence: no measured cost in 2026.
- Left config_issue waits (Sep 1, Oct 9): no silence >= 30 s and a clean green / yellow sequence in every one (no lost
  event); real waits. Still open (measured 0): a lost OFF at the very end of the log with 3+ walks from other presses;
  a hole that drops only ped events.
### Sheet (backup dc_work/ped2/ped_health_review_before_round5.xlsx)
Updated A1, rows 3 (all-flags shares), 6 and 8 (Oct overlap not checked; row 8 Oct now ok), 10 (lost-event and hole
counts), 12 (remaining waits are not data loss). User answers kept.
