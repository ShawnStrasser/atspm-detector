# 162 — stuck on: time held during a preemption does not count (option preempt_holds)

2026-11-24. User report: signal 05018 (Lightning Troubleshoot upload, 19-20 Nov 2026), det 7 / 71 on P4 "failed
together, stuck on". Log: preempt 5 102/105 at 22:52:39, 104 at 23:15:28; det 7 / 71 ON 22:56:40 -> 23:15:35 / 23:16:15,
released at P4's first green 23:16:04. Same again 20 Nov 22:24-22:45 inside preempt 5 (22:23:22-22:45:21). A queue
behind a train, not a fault. The package flagged det 7 only ("stuck on, held ON 2 times, 40 min"); det 71 and the
"together ... to 23:28" wording come from the app's own grouping (not in this repo; not traced).

## Rule (src, option OFF by default; batch runs only)
- Preemption = ped health's rule (ped_health_duckdb.PRE, user review 2026-10-29): from 102 or 105 to the first
  104 / 110 / 111 of that preempt number, at most 6 h, + 120 s recovery; an end with no start in the window = on from
  the window start. Overlapping preemptions merged.
- hold.dur = held time OUTSIDE every preemption (stuck-on judging: ep, efl, limits, texts); new hold.span = the hold's
  length, used where the hold means "the detector was ON" (silence / count drop left out of held time, partner
  exclusion, hold_co shared count). So a cleared hold is not re-reported as a silence.
- Not "any overlap excuses the hold" (ped's wait rule): a detector stuck 12 h is not cleared by a 2-min preemption.
- Preempt codes never enter evd (classifier input, AGENTS.md); predict builds a separate table evpre when the option is
  on and passes it as health(preempt_table=) (@PREEMPT@ in adapters.sql). health()/health_sql() default to the
  caller's events table; workers copy the codes with the allowed ones.
- Store / live (judge_store) unchanged: no preemption state is stored (would need a carry for an open preemption).

## Results
- 05018 upload: det 7 bad -> ok, both stuck-on findings gone, no new finding; every other column identical.
- Network day 2024-12-02 (data/raw/Train_Dec_2_2024, the 194 signals with preempt events, 4,201 detectors):
  7 status changes, all suspect -> ok (flagged 156 -> 149); 11 stuck-on periods gone, nothing added; phase, function,
  lanes etc. identical. Checked against the log:
  - 6c6c3c5c (6 dets, 7 periods): holds inside preempt 5 + 6 (14:53-15:10, 20:52-21:17) - trains, clear.
  - 2e12e106 det 9 (Advance): hold 15:33:00-15:44:21 just after a 52-s preempt-6 input (102 only, no entry);
    the 120-s recovery takes it from 11 to 9.3 min, under the limit. Borderline.
  - 804df802 det 7/22/23 (5 dets held together 18:16-18:28): preempt 3 only 18:25:53-18:27:03; outside time 9.5 min,
    under the limit, so this period goes; their status stays suspect (06:13 common hold). Borderline.
- Option off: check passes (no re-freeze), pytest 282 pass; SQL text snapshots regenerated (new span column; answers
  unchanged); new tests/test_preempt_holds.py (synthetic).
- Runtime, 194 signals: off 238 s, on 224 s (noise). A first try reading evd UNION evpre as a view took 327 s.

## Open (user)
- Turn on by default? Then CHANGELOG + release. Borderline cases above: keep recovery 120 s, or count only a
  preemption with an entry (105), or require the hold to start inside the preemption.
Code: research-free; src/atspm_detector/hq/sql/adapters.sql (hold), health_duckdb.py (PREEMPT, option),
pipeline.py (evpre/evh). Scratch scripts deleted.
