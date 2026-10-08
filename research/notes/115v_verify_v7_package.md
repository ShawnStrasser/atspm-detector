# 115v Adversarial verification of package v7 (`dc_work/final_v7_prod`, 7.0.0) (2026-10-07)
CPU, 4 threads, nothing in the package changed. Work `%DC_WORK%/s115v`, scripts `research/code/final115v/`.
Locked_v2, note-115's 40 parity signals and the bench71 signals were excluded and asserted absent.
**Set-up.** 15 NEW signals (seed 20261007, 3 per channel quintile, 4-42 channels), Sunday 2026-09-20 (the build
used 09-19). Each is cut into 5 min / 30 min / 3 h / 24 h = 60 cases x full / le2h.
v7 = the shipped dist/ wheel in a FRESH venv (pandas 3.0.6, duckdb 1.5.6, ort 1.30); its installed files = src/.
ref114 and v6b ran in the dev venv (pandas 2.3.3, duckdb 1.5.5).

**(a) PASS.** v7 vs ref114 (1,109 detectors, 6,538 pairs per profile): every decision is identical and the largest
probability difference is 0.0. v7 src = the same. On a 7-day sample, v7 = ref114.
v7 vs v6b: phase identical; function 7/1,109 (both profiles); setback 48/38; night speed 2/1; health_watch 4/5.
These equal v6b vs ref114 exactly, so every change comes from note 114. The stacker sha256 = s114. The other
weights equal v6b's, apart from note text.

**(b) PASS for decisions.** Random 64-channel + 16-phase bijections (seed 7: 6 signals x 4 lengths x profiles;
seed 11: 3 x 4) leave every decision identical. Probabilities exceed 1e-6 in 5 cases: v12 function up to 2.2e-3 (full),
v14 5-min phase 4.4e-5. v6b shows the same (plus a 1.6e-5 le2h tie that v7 fixed).
Cause (v12 5 min): features, ranker and keep set are identical; the network output differs by 1.3e-7 (float order);
the stacker's within-phase rank columns (stacker.py:91) flip at near-ties. Float noise, not a number leak.

**(c) PASS.** Imports are stdlib + numpy / pandas / duckdb / onnxruntime only. No absolute paths in src, tests,
pyproject or README, nor in the byte-scanned weights. CPU provider (trees_onnx.py:41).

**(d) PASS with one defect.** Wheel install, CLI, `python -m`, DC_FAST_PROFILE and check 6/6 all work; CLI CSV = API.
- d1: `onnxruntime>=1.18` in pyproject is wrong. ort 1.20.1 fails 6/6 ("NOT_IMPLEMENTED TreeEnsemble(5)"); 1.21.1 and
  1.22.1 pass. The floor must be >=1.21. Floors for numpy, pandas, duckdb and Python 3.10 cannot be tested (only
  Python 3.13 here).
- d2: under pandas 3, each predict() prints 2 PerformanceWarnings (pipeline.py:479-480; frame of 222 blocks).
  Unclosed `json.load(open())` at pipeline.py:471,493 and funcnet.py:147.

**(e) PASS.** No event-level Python loop and no apply or agg(lambda). Measured on v13 24 h (40 channels, warm 2.2 s):
lanes.py:558 80 ms, features.py:306 30 ms, health_core.py:1400 29 ms, setback.py:467 14 ms, features_partner.py:169
9 ms; all others <= 6 ms; about 8 % in all. Rough v7/ref114 time: full .89 / .90 / .87 / .78, le2h .92 / .92 / .87 / .79.

**(f) FAIL.** OK: empty inputs, 1 detector, no begin-green / 7-11 / 43-44, 1 and 5 min, 7 days (2.32 M events),
duplicates, channels > 64, shuffled rows, mixed column names (df / csv / parquet), string timestamps, NULLs.
Defects (F2-F7 also exist in ref114 / v6b; F1 is new in v7):
- F1: 7-day sample, 40 channels: peak 1,528 vs 1,150 MB (+33 %), both profiles. Source: features.shared_parts
  (table j = every ON x every candidate, plus the ASOF call pass).
- F2: tz-aware input is converted to the machine time zone (pipeline.py:238). A 24 h sample shifts by 3 h; night
  speed changes on 5 detectors, setback on 9, lanes on 1.
- F3: one Parameter out of range on an allowed code (131 = -1, 150 = 70000) crashes the whole call
  (USMALLINT cast, pipeline.py:244-245).
- F4: an apostrophe in a path is not escaped: the temp dir (pipeline.py:173) breaks EVERY call; the events path
  breaks at pipeline.py:203/205.
- F5: answers depend on the batch: w0 / w1 are global over all signals (pipeline.py:483). v08 3 h joint vs alone:
  lane_conf .979 vs .927, setback 82 vs 80 ft.
- F6: health_core.py:919-926 divides by 1e-9. v08 3 h dets 16 and 17 are healthy but called "bad",
  "46731000000000x".
- F7: channel 0 is kept as a detector; a stuck-ON channel's status says "no actuations".
