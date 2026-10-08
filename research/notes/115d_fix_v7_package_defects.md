# 115d Fixes to package v7 after the verification (note 115v) (2026-10-07)
Fixed in place in `dc_work/final_v7_prod` (backup before the fixes: `final_v7_prod_bak115`). Still 7.0.0, wheel rebuilt.
Work `%DC_WORK%/s115d`, scripts `research/code/final115d/`. Locked_v2 asserted absent in every parity run. CPU only.

| defect | fix | evidence |
|---|---|---|
| d1 ort floor | `onnxruntime>=1.21`; tested versions written in pyproject + README | fresh venv, ort 1.21.1: check 6/6 PASS |
| d2 warnings, files | missing ranker columns + p0 / p0_tree added with one concat; every `json.load(open())` -> `common.read_json` (with-block, utf-8); smoke test now installs the SHIPPED `dist/` wheel, checks it byte-equals `src/`, and fails if predict() prints any warning (`-X dev -W always`) | pandas 3.0.6: no warning |
| F1 7-day memory | `features.shared_parts` builds the ON x candidate join `j` / `lo2` per block of whole detectors (<= 1.5 M rows); every aggregate is grouped by detector, so the same rows meet; one block (old code path) when it fits | below |
| F2 time zones | tz-aware DataFrame column -> its own wall-clock time (`tz_localize(None)`); parquet TIMESTAMPTZ -> zone from the pandas metadata (`timezone(zone, ts)`, fixed offsets too), none recorded -> UTC wall clock + warning; CSV / text with offsets -> offset dropped; `start` / `end` offsets dropped; documented in README + docstring | Indianapolis df, fixed-offset df, pandas parquet, CSV with -04:00, start/end with offset: all identical to the naive input |
| F3 bad values | all four columns TRY_CAST; rows with a missing / unreadable value or Parameter outside 0..65535 dropped and counted in one warning; CSV columns read as text (sniffed timestamp format kept) | 131=-1, 150=70000, NaT, NaN, 'abc', 'not a time': no crash, answers identical, "5 / 3 rows dropped" |
| F4 apostrophes | `_q()` SQL quoting for temp dir, file paths, memory, ids; start/end rebuilt from pd.Timestamp; check.py paths too | temp dir "tmp o'neil" + "o'brien dir/ev's.parquet / .csv": identical |
| F5 batch | predict() loads once, then runs each signal alone (`_one_signal`: own evd / devmap / window, key 1, cyc5 dropped) | v08+v05, v08+v12 (df) and v08+v03 (one file, le2h): batched = chunked = alone, every column |
| F6 rel_disp | no finding when the reference expects < 1 count in the usable bins (`MIN_EXPECTED`); no 1e-9 floor left (v >= e >= 1). Other 1e-9 floors checked: shares bounded by 1 or comparisons only | v08 3 h dets 16 / 17: bad "46731000000000x" -> ok |
| F7 channels | 81 / 82 keep channels 1..64 (0 dropped); gate counts an ON that never ends for channels with NO completed ON (`open_on_s`): new reason `never_off` "stuck ON (turned on and never off)"; `stuck_on` now checked before `near_zero_volume` | channel 0 rows: identical; all 81s removed: "stuck ON (turned on and never off)"; one 82 at 15 min of 3 h: "stuck ON for most of the window" |

A first F7 version also added the open final ON to channels WITH completed ONs: v07 5 min det 22 (a normal Presence,
queue at the window end) became "stuck ON". Dropped; only channels with no completed ON use it.

**Parity vs ref114** (dev venv, src; same cases and reference outputs as notes 115 / 115v):
- note-115 set (40 signals x 30 min / 3 h / 24 h + bench71 = 132 cases, 2,627 detectors) x full / le2h: every decision
  identical, per-phase tables identical; probabilities max 1.4e-8 (full), 1.3e-4 (le2h: the note-115 tie, unchanged).
- note-115v set (15 signals x 5 min / 30 min / 3 h / 24 h = 60 cases, 1,109 detectors) x full / le2h: probabilities
  identical (0.0); decisions identical except the intended F6 change, both profiles: v08 3 h det 16: P6 Advance and
  det 17: P6 Mid, health bad ("counts jump around 4.7e13x") -> ok; review flag off. Nothing else moved.
- F2 / F5 / F7 changes do not occur in these sets (naive timestamps, one signal per call, no channel 0 / stuck channel);
  they are proven by the tests above (`s115d/rob/robust_*.json`).
- Shipped wheel in a fresh venv (pandas 3.0.6, duckdb 1.5.6) on the 115v set = src, every column (0.0).

**Checks.** References did not move (no `--freeze`): check 6/6 PASS (dev venv, src). `tests/smoke_install.py` on the
shipped wheel: 71 files = src, fresh venv (pandas 3.0.6, duckdb 1.5.6, numpy 2.5.3, ort 1.30.0) install, CLI full /
le2h, no warnings, check 6/6 PASS; again with ort 1.21.1: PASS.

**Memory and time** (`bench115d.py`, one process per run, 1 cold + 3 warm calls, 2 rounds; CPU load during the bench
mean 10 %, max 19 % = this bench only). Peak working set of one call; warm = best of 6.
| sample | profile | ref114 peak / warm s | v7 fixed peak / warm s |
|---|---|---|---|
| 3 h, 40 ch (v13) | full | 359 MB / 2.49 | 244 MB / 1.52 |
| 3 h, 40 ch (v13) | le2h | 247 / 1.03 | 230 / 0.89 |
| 3 h busiest (bench71) | full | 332 / 1.37 | 239 / 1.02 |
| 3 h busiest (bench71) | le2h | 242 / 0.88 | 226 / 0.77 |
| 7 days, 40 ch (v13 x7) | full | 1,159 / 23.9 | 969 / 15.3 |
| 7 days, 40 ch (v13 x7) | le2h | 1,153 / 21.1 | 971 / 14.6 |
7 days: v7 before the fix 1,527 MB; the feature join now peaks at 732 MB, the remaining peak is setback (~950 MB,
unchanged code). Four calls in one process: 1.1 GB (ref114 1.3-1.6 GB). Warm times vary 2x run to run (3 h medians
2.5-3 s full for both): treat time as "not slower".
