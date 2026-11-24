# 163 — preempt_holds on by default; incremental stores and live

2026-11-24. User (after note 162): turn it on by default; it must work in incremental runs too; borderline cases keep
the rule as is ("expected to move some from bad to suspect, that is fine").

## Changes (src, nothing committed)
- OPTIONS["preempt_holds"] = True. Goes into the unreleased 2.2.0 (CHANGELOG section + one README line).
- Store judge (incremental.judge / judge_rows): hold() reads the preempt events from the store's ped log
  ({store}health_ped_events; LOG_CODES already held 102/104/105/110/111 in 2.2.0, so no extractor change, no
  EXTRACTOR_VERSION bump, existing stores work). A store also reads preempt events from BEFORE its window (it carries an
  ON held from before the window, so it must know a preemption that started then); a batch run reads its window only.
  An end with no start before it now starts at -infinity (batch: same answers, its holds start in the window).
  workers > 1 copies health_ped_events into the job file.  ped=False store: no ped log, judges as before (documented).
- health(workers > 1, preempt_table=...): the job copy reads events_table UNION preempt_table (was: preempt ignored).
- live 15-min stuck-on rule (_live_step): held time less the time inside preemptions (same rule, on timestamps) against
  lim_stuck. Each run's store is fresh, so the preempt events of the horizon go in the state (code 9240: Parameter =
  preempt number, Value = code), trimmed at the signal's bins-through marker less the horizon; a step reads those of the
  horizon before it, so chained runs equal one big run. The whole-day check already had them (history health_ped_events).

## Tests (tests/test_preempt_holds.py, synthetic)
- batch on/off (note 162); no preempt events = no change.
- store fed in 37-min cuts + a cut inside a preemption, 2 signals, with a preemption still open at the window end:
  judge_rows == health(start=T-H, end=T, bin_origin="clock", periods=True); also with workers=2 (a signal each).
- window starting inside a preemption and a hold (01:30): store carries the ON from 01:00 and reads the 102 from before
  the window -> not stuck (option off: stuck). Batch differs here by design (knows no ON before its window), as before.
- ped=False store == batch with the option off.
- live: det held ON from BASE, preemption BASE..+40 min (+2 min): stuck rows [0, 0, 0, 1] per bin vs [1, 1, 1, 1].
- atspm-detector-check passes with the default on; stored answers unchanged, no re-freeze.

## Classifier (user asked whether preemption periods are filtered there "as we already did")
Never done: no note before 155/162 mentions preemption; the classifier never reads 102-111 and uses every period as is.
Scale, network day 2024-12-02: 194 of 420 signals logged a preemption; merged time (+120 s) median 4 min / day per such
signal, p90 16 min, max 1.4 h; 26.7 h of ~10,080 signal-hours (0.26 %). Masking at inference only would feed the model
inputs unlike its training data; doing it properly = mask in features for training too, retrain, screen on the folds
(AGENTS.md noise floor). Expected gain on day samples ~nil; possible on short samples containing a train. Asked the user.
