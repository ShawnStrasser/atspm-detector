# 150 Health 147m built into atspm-detector 1.1.0, parity, speed (2026-10-16)
User go 2026-10-16 ("final speed verifications ... republish"). Orchestrator task; not committed. Backup of src/tests/docs:
dc_work/bak_pre147m. Code research/code/health150 (h150_ref, h150_limits, h150_run, h150_sim, h150_parity,
h150_flagged, h150_verdict, bench150, bench150_cmp, prof150); work dc_work/s150. No fault events; locked_v2 absent.
## src/atspm_detector
- health_counts.py (new, 2-D numpy, no per-detector loops): hourly GEH (yardstick = partner corr >= .7 / phase ref /
  signal, as h147d_geh; share 07-19; drop point by cumulative two-sample stat; occupancy own / mates), 10-min spiky
  bursts (h147l events2), common holds (blocked broadcast), one-sort nanmedian (= np.nanmedian, 2x faster).
- health_v4.py: checks = stuck (+ common hold at the limit -> 1.0, signal note), count_on, goes silent, count drops (GEH,
  occupancy-confirmed), night (GEH; low-volume night silence <= .5), busy (mates delta for any finding), too fast / too
  many, spiky; Bike no counts-fall / night; erratic + 15-min level loop + shape110 removed (and their helpers in
  health_v4_stats). score(x, s, b) = .35 at s, .75 at b: the stored bad limit IS where one finding turns bad.
  limits() -> (suspect, bad); 24 h = JSON tables loosened to the least strict plausible type; shorter = 1.0.1 limits with
  bad starts in LJ['short'] (same statuses as before). GEH checks + spiky need >= 3 h. too_fast gives pair_share
  ('counts each vehicle twice' >= .9). Texts: GEH drop / night lines, spiky line, 'held ON together with det ...'.
- weights/health/health_limits.json (h150_limits): [suspect, bad] per check x type read off the 147m simulation on
  sure-function detectors (identical within a cell); stuck bad = user minutes; too-fast 2+ = 1-lane, bad x1.168;
  too many [2100, 2358]; GEH [10, 13.08] / [8, 10.46]; count held ON [30, 159]; spiky knobs. model_card hash + text.
- check.py (limits shape), charts.py (spiky shade), pipeline.py docstring. Version 1.1.0 (= pyproject attr).
## Parity (755 h24_b signals, 14,806 detectors with a function; harness h150_run = h144_run + Bike 1 lane)
- New measures vs the research feature scripts (geh147d, feat147e, h147m bursts): geh_night / geh_drop / eligibility,
  occ own / mates, bursts / gmax / daily / level: 0 differences.
- B: the 147m research scorer (h147 chain + h144 status) on the package's own measures and limits vs package: status 0,
  findings 0.
- Package vs the original 147m simulation (ref147m): 36 status / 35 finding differences, all explained:
  26 = limit rule for unsure-function detectors (f_top < .7; package: least strict plausible type's reviewed limit; 147m
  sim: 1.1.0 loosened limit x own-type factor); 11 = measures (A: 147m limits on package measures vs ref): goes-silent
  pre-pass now drops mates flagged by the NEW checks (drop_lam moved, 8), Q2 c_occ_like d143 .82 vs package .79 (1),
  stage-1 recovered-hold caps s_dropout / s_stuck (2). None is a 147m verdict / review-row detector.
- Status ok / suspect / bad / n.e.d.: package 93.92 / 2.30 / 2.12 / 1.66 (147m sim 93.96 / 2.28 / 2.11 / 1.66).
## Flagged (final status, % of detectors; flagged150.csv)   after susp / bad      before (note 144/148 src)
  24 h  stuck 1.67 / .78; counts fall .15 / .50; night .14 / .19; busy 0 / .64; fast+many .03 / .18;
        spiky .32 / .61; ANY 2.30 / 2.12 = 4.42 %                                  .84 / 1.34 = 2.18 %
  3 h   (200 signals x 2) .45 / .44 = .89 % (spiky .12, counts fall .16)            1.13 %
  30 min .53 / .15 = .68 %                                                          .81 %
  (short-sample too-fast moves .20 -> .13 % at 3 h: 2+ = 1-lane limit and bad x1.17 apply at every length)
## README verdicts (note 145 set, h150_verdict): faulty 14 / 20 flagged (before 14; +08073 d7, 08154 d1 spiky; -2B069
d23 busy without mates excess, -2B316 d3), fine 3 / 12 (before 2; +2B531 d35, 2B530 d60 busy at night, -12032 d41).
## Non-health outputs: bench150 35 jobs (30 h24_b + 5 logs, 724 detectors): every non-health column identical;
status / review_flag change only where health_status changed (54 rows). Bundled sample: unchanged (check, no freeze).
## Speed (bench150: local copies of both packages, best of 2, 4 threads; same machine, back to back)
health (assess + fetch) median 115.7 -> 113.1 ms, p90 153.9 -> 151.9; predict median 1,264 -> 1,266 ms.
Multi-day: typical_r8 5 d 596 -> 608, busiest_ev 5 d 839 -> 863, asclog 4.1 d 495 -> 512 ms (24 h -> 5 d x4.7 for
events x5.7: sub-linear). Running the package from the network share adds ~50 ms per predict (weight files), not code.
Per check (prof150, cProfile ms per signal-day, cumulative): stuck 18.3; counts fall + night 2.2 (was 7.0: level /
erratic loop gone); spiky 1.7; busy 5.4; too fast / many 13.0; limits 1.0 (an array version cost 3.5: kept as dict
lookups); scores + status 5.5; text 2.3; shared: stage 1 44.5, bins 35.7, prep 15.6. New code: GEH 1.2, spiky 1.7,
common holds .03. No DuckDB setting changed.
## Checks: atspm-detector-check 7 / 7 PASS (no --freeze); pytest 22 passed, 4 skipped; tests/test_health.py rewritten
(score at the bad limit, reviewed cut-offs + loosening, too fast + pairs, common holds, hourly GEH drop / night,
spiky incl. mates and low volume, sample categories). Grep of tracked + new files: no server / user names, credentials
or absolute paths. README health section + verdict line, CHANGELOG 1.1.0, docs/embedding.md, CITATION (2026-10-16).
## Choices made (for the orchestrator)
- Unsure-function loosening: documented package rule (least strict plausible type) instead of the 147m shortcut (26
  detectors; +.02 suspect / +.01 bad pt). One-line switch if the 147m behaviour is wanted.
- Multi-day spiky: bursts per day and vehicles per day = totals / days; < 24 h: counts as they are, vehicles x 24 / h.
- g4_unscored guard of the old level check dropped (1 detector, no overlap with GEH drops).
- TWIN FIX (orchestrator): twins (health_core.twins, one zone on two inputs) never make a common failure against each
  other: common-hold bads 22 -> 17 (6 signals; 3 -> ok, 2 stay bad by normal stuck); 24 h 2.30 / 2.10 %; verdicts same.
