# 115dv Adversarial re-verification of package v7 after the 115d fixes (2026-10-07)
Package `dc_work/final_v7_prod` (7.0.0, shipped `dist/` wheel), nothing changed. CPU, 4 threads. Work `%DC_WORK%/s115dv`,
scripts `research/code/final115dv/`. Locked_v2 asserted absent everywhere; the 115 / 115v / bench71 signals excluded.

**Set-up.** 10 NEW signals (seed 20261008, 2 per channel quintile, 7-38 channels), Sunday 2026-09-20, cut into
30 min (07:30) / 3 h (15:00) / 24 h = 30 cases, 606 detectors, 3,859 candidate pairs, x full / le2h.
Fresh venvs from the shipped wheel: newest (pandas 3.0.6, numpy 2.5.3, duckdb 1.5.6, ort 1.30.0 = latest on pip) and
ort 1.21.1. ref114 and v7 src in the dev venv.

| item | result | evidence |
|---|---|---|
| parity v7 src vs ref114 | PASS | both profiles: every column identical (decisions, probabilities 0.0, candidates, phase tables) |
| parity shipped wheel (newest) | PASS | = ref114 and = src in every column, both profiles |
| parity wheel + ort 1.21.1 | PASS | decisions + phase tables identical; function probs differ <= 5e-5 (float noise) |
| wheel smoke, newest / ort 1.21.1 | PASS | 71 files = src; CLI full / le2h; no warning (-X dev -W always); check 6/6 both |
| ort floor | PASS | pip refuses wheel + ort 1.20.1 (ResolutionImpossible) |
| check from src (dev venv) | PASS | 6/6 |
| channel + phase permutation, 3 signals (n03, n06, n09) x 30 min / 3 h / 24 h x profiles (15 runs) | PASS | every column identical; one text diff is the partner's own channel label in health_watch ("d3" = permuted "d40"), not a leak |
| 115v battery (empty, 1 det, no green, no 7-11, no 43/44, duplicates, >64, shuffled, mixed names df/csv/parquet, string ts, tz, missing column, 2 signals, 1 / 5 min, stuck ON, channel 0), on v08 AND new n05 | PASS | all identical / sensible; missing column -> clear ValueError; stuck ON -> "stuck ON"; channel 0 dropped |
| F2 tz (Indianapolis, UTC view, fixed offset, pandas parquet, no-zone parquet, CSV -04:00, start/end offset; new: UTC-aware df, 2 signals tz, tz pd.Timestamp start/end, ISO 'T' csv) | PASS | identical; no-zone parquet warns as documented |
| F3 bad values (df, csv) | PASS | identical, "5 / 3 rows dropped" warning |
| F4 apostrophes (temp dir, parquet, csv; new: DeviceId "o'hare #1 é", batched "x'1" / "y''2") | PASS | identical |
| F5 batching (v05, v12, v03 file; new: 4 signals, mixed windows, one without begin-green, one without detectors, chunk 2, le2h shuffled 3 signals, glob of 2 files) | PASS | joint = chunked = alone, every column |
| F6 v08 dets 16 / 17 | PASS | ok, "790 / 782 actuations, nothing unusual" |
| F6 scan, 8 output sets (4,848 rows) | PASS | largest ratio in any health text 1.5x |
| F6 stress: partners thinned to 1-3 ONs | PASS | v7 ok (ref114 "4.8e13x") |
| F7 channel 0 / never-off / stuck most of window | PASS | identical / "turned on and never off" / "most of the window" |
| F1 block path forced (1 detector per block) vs one block, 5 signals x 3 h / 24 h | PASS | every column + phase table identical |
| 7 days | PASS | n08 (38 ch, 2.0 M events, 5 blocks): = ref114 every column; peak full 891 vs 1,237 MB, le2h 856 vs 1,129 MB; v13: 957 / 942 MB (115d: 969 / 971) |
| disallowed codes (4-6, 13, 21-23, 32-33, 45, 61-66, 89-90) and faults 83-88 added | PASS | identical |
| 90-min hole, start > end, one event, 1 actuation | PASS | no crash; sensible statuses |
| pandas-3 warnings on 3 new signals x 24 h + 30 min x 2 profiles | PASS | 0 warnings |

**Observations (not regressions; ref114 does exactly the same):**
- A DataFrame without start / end uses first-to-last event as the window; a path with start / end uses the exact
  window. v08: lane_conf .979 vs .951, setback 82 vs 80 ft. With start / end passed, df = path (identical).
- A 90-min hole inside a 3-h sample still reports minutes_of_data 180.
- A healthy busy detector whose same-phase partners are nearly silent (6-100 ONs in 3 h / 24 h) is called "bad,
  jumps around 21-130x"; the partners themselves are called bad too. Same numbers in ref114.

**Verdict:** all 9 fixes hold; no new defect found. Ready for the ship decision.
