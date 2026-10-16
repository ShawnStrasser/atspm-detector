# 144 atspm-detector 1.1.0: the six agreed health checks built into the package, vectorised (2026-10-14)
User go 2026-10-14 ("continue working to have the code optimized and built into 1.1.0").  Orchestrator task.  Backup of
1.0.1 src: dc_work/bak_1_0_1_pre110.  Code research/code/health144 (h144_limits, h144_sim, h144_run, h144_parity,
h144_flagged, h144_out, h144_refs, bench144, prof144); work dc_work/s144.  No fault events; locked_v2 asserted absent;
sample h24_b of notes 140-143 (755 training signals, 14,806 detectors with a predicted function).  Not committed.
## What changed in src/atspm_detector
- health_v4.py rewritten: six checks (stuck on + Count zone held ON; counts fall = goes silent / count drops, ONE
  finding (the larger); misses at night; busy at night; too-fast single gaps + too many; erratic only when nothing else
  fires), status / severity vectorised (numpy masks; rules R6, R7, Q2, Y, Y1, A, P, C, F kept; R5, R8, N1 gone),
  limits row-wise over dicts.  Dropped: chattering, v4i runs, colour_stats zf / bursts, erratic time ON (n3), ON longer
  than its kind (occ_hi8), unusual daily pattern (band_*), undercounts vs partner, the v4 pre-pass tables.
- health_core.py (stage 1): rules dropout / stuck / night_drop / level only (chatter, rapid, 5-min volume, night/day,
  choppy + spikes, corr, short ONs, partner_notes, act_stats gone); pass-1 status from those four.
- health_v4_stats.py: too_fast (note 141 item 1), q5_stats + fast_even on one start view (rule F keeps the research
  quirk: starts in the first 2 s of red are left out), act_stats / act118 vectorised (bit-identical on 720 checks).
- weights/health/health_limits.json (new; h144_limits.py): 24 h = fmax(note-143 fitted, 1.0.1) per check x type, both
  loosened over plausible functions; fitted cells beyond the population's observed maximum -> 1.0.1: too-fast all,
  erratic all, stuck Advance 2+ / Yellow_Red 2+ / Bike 1 / Bike 2+ (> 39,521 s), silent Other 2+ (5,842 > 2,240).
  Too-fast per-1,000 cuts of note 141 at every length; too many 24 h = fmax(2,250, 1.0.1).  health_v4_refs.json pruned
  to the tables still read (335 -> 170 KB, values unchanged; outputs identical).  Version 1.1.0 (+ model card hashes).
## Parity (h144_parity; research mode = OOF night models, as notes 140-143)
- A simulation (h143 scores + h144 status) on the package's own measures: status 0 / findings 0 differences.
- L every limit (stuck, silent, drops, night, busy, too-fast, too many, erratic) = simulation: 0 differences.
- B vs simulation on the 1.0.1 measures (sim144): 14,795 / 14,806 equal; 11 differ, all from moved yardsticks:
  with the 1.0.1 stage 1 only 4 (goes-silent pre-pass now on the six checks: drop_lam moved on 67); the slimmed stage 1
  adds 7 (pass-1 'bad' set: drop_lam 332, night_ratio 48, exc_15 178, c_occ_like 250 moved).  Net: bad->ok 3,
  bad->suspect 2, ok->bad 1, ok->suspect 3, suspect->ok 2 (flagged -1).  Adopted (cost of keeping old stage 1 ~15 ms).
- Package (shipped night model) vs research mode: 6 busy-at-night statuses differ (all-data vs OOF, note 124).
- Phase / function / lanes / setback / night speed: identical, 30 h24_b signal-days + 5 logs (724 detectors) and the
  bundled sample (check passes without --freeze: the sample's health columns are unchanged).
## Flagged, % of detectors with a function (status level; flagged144.csv, flagged144_all.csv)
24 h (755 signals, 14,806):  check            1.1.0 susp / bad / total     1.0.1 total
  1 Stuck on                     .40 / .13 / .53               1.94
  2 Counts fall                  .02 / .20 / .22                .69
  3 Misses vehicles at night     .19 / .16 / .34                .62
  4 Busy at night                .00 / .81 / .81                .90
  5 Too fast / too many          .08 / .09 / .17                .63
  6 Erratic counts               .15 / .13 / .28                .47   (1.0.1 also: chattering .49, erratic
  any                            .84 / 1.34 / 2.18             4.52   time ON .08, unusual daily pattern .19)
  status ok (incl. watch) / suspect / bad / not enough data: 96.17 / .84 / 1.34 / 1.65 (1.0.1 93.86 / 2.13 / 2.38 / 1.62)
3 h (200 signals x 2 windows, 7,760): any .63 / .50 / 1.13 (1.0.1 1.39); stuck .55, counts fall .34, too fast .21,
  erratic .12 (1.0.1 .31), night / busy 0.
30 min (395 windows, 7,576): any .73 / .08 / .81 (1.0.1 .79); stuck .57, too fast .24 (1.0.1 .09: single gaps with the
  24-h per-1,000 cut, as instructed), others 0.
Flags from the 1.0.1 fall-back cells: stuck 7 of 74 (Adv / YR 2+, Bike), silent 3 of 7 (Other 2+).  Too-fast flags
with < 100 ONs (the dropped R5 watch guarded these): 2 of 21.
## Runtime (bench144: 30 h24_b signal-days, best of 2, threads 4, same machine, 1.0.1 run before and after)
health (assess + event fetch) median 191 -> 119 ms, p90 249 -> 164; assess 171 -> 98; fetch 19 -> 20; predict median
1,348 -> 1,347 ms (noise), health share 13.7 -> 8.9 %.  Multi-day: typical_r8 5 d 808 -> 604, busiest_ev 5 d
1,085 -> 873, asclog 4.1 d 698 -> 495 ms.  Savings: dropped checks (~60 ms), limits row-wise (~15 ms), act stats,
joins batched.  Tried and NOT adopted: event fetch sorted in DuckDB (ORDER BY + window gaps 26.7 ms, ORDER BY time +
numpy 19.3 ms, numpy prep 16.0 ms median) -- DuckDB's sort is slower here; no DuckDB setting changed.  Not done: the
per-ON layer in SQL (bins keep float32 add.at accumulation for parity), level / erratic loop as 2-D numpy.
## Checks: atspm-detector-check (editable install, s124/p3venv) 7 / 7 PASS incl. blocked imports, no --freeze; pytest 15
passed, 4 skipped (atspm absent); new tests/test_health.py (six categories, limits >= 1.0.1, too-fast counting, sample).
## Open (orchestrator)
- docs/embedding.md still lists the removed `what`s (chattering, erratic time ON, unusual daily pattern): not in the
  edit list.  README 'How well' line (19 of 20; 0.4 / 1.2 / 3.9 per 100) is pre-1.1.0.  CITATION.cff set to 1.1.0.
- 30-min too-fast flags rise .09 -> .24 % with the 24-h per-1,000 cut; R5 low-volume guard dropped per the simulation.
