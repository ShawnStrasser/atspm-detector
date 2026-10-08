# 115 Production package v7 (`detector_classifier` 7.0.0)

**Built:** `dc_work/final_v7_prod` (src layout, `pyproject.toml`, wheel `dist/detector_classifier-7.0.0-py3-none-any.whl`
6.0 MB, `tests/smoke_install.py`). v6b and `model/` untouched. Content = v6b + note 114 (pk_unhealthy out of stacker and
stack pick; stackers = `s114/weights_stacker`, sha256-identical) + every output-identical efficiency fix of notes 115a/b.
API = v6b's `predict()` (superset of model/predict.py's): `from detector_classifier import predict`; CLI
`detector-classifier`, `python -m detector_classifier`, `detector-classifier-check`. Deps numpy>=1.24, pandas>=2.0,
duckdb>=1.0, onnxruntime>=1.18 (pyarrow dropped: parquet I/O via DuckDB). predict.py -> `pipeline.py` (no attribute
shadowing); gru_input -> `streams.py`, gru_blend -> `blend.py`, features_yellowred -> `features_lag.py`; GRU/TCN code,
gru_onnx, dead yr / cycle_counts / on_times code gone; model card rewritten (38 files, sha256, accuracy with sources).

**Applied (all output-identical).** Mean warm s ref114 -> v7 over 4 bench71 signals, `s115/v7/stages115_b.json`:
| fix | stage | 30 min | 3 h full | 24 h full |
|---|---|---|---|---|
| int `dev` key in evd / chunk tables; facts distinct -> bitmask; devmap read once | load+tables | .016->.015 | .028->.025 | .091->.083 |
| streams from onev_all / cand + shared `cyc5` (also read by expert features) | streams | .020->.012 | .035->.019 | .088->.031 |
| ONE narrow ON x candidate table + one release table + ONE call-aggregate pass (was 2 joins + 4 ASOF passes); reverse calls numpy searchsorted (no call x detector cross join); rank features one grouping; one astype | features | .175->.148 | .264->.221 | .597->.455 |
| lead neighbours in numpy (integer counts, deterministic) | (in features) | .012->.005 | .020->.011 | .075->.054 |
| kept-pair raster built once per piece, shared by 3 members; ORT arena off for the 6 network graphs | network | .087->.073 | .293->.220 | .275->.202 |
| pair_mats once per phase group (114 patch); sibling / lag aggregates one grouping, no lambdas; ON times fetched once (shared with night speed); lane decoder batched move scoring + vectorised violation count | function | .130->.119 | .151->.139 | .218->.193 |
| health: events from evd as numpy (exact ns seconds, no pandas frame), prepared once for the 5 checks; bincount counts; 15-min counts cached; one clock-hour conversion | health | .038->.028 | .084->.051 | .333->.125 |
| setback models lazy (one P50 group; band only with Advance/Mid) | setback | ~0 | ~0 | ~0 |
| **total warm** (v7/ref per signal) | | .644->.573 (.87-.94) | 1.073->.881 (.78-.87) | 1.912->1.412 (.72-.82) |
le2h: 3 h .718->.641, 24 h 1.534->1.183. Cold 24 h full 2.27->1.92 s. Peak working set (range of 4): 30 min 282-354 ->
217-267 MB; 3 h full 301-404 -> 242-290; 3 h le2h 241-275 -> 221-266; 24 h full 339-485 -> 287-438; 24 h le2h 277-418 ->
279-424 (equal: the shared join costs ~+10-25 MB at 24 h on the busiest signals; a 3-ASOF single table cost +80 MB and was
narrowed). Machine shared (other agents): rough only, the quiet-machine bench is a separate stage.
ORT: ENABLE_ALL, sequential, 4 intra / 1 inter, no spinning, sessions cached; DuckDB 4 threads (1-2 threads: slower).
Not applied: setback pair-cue reuse (not identical), dropping `win` / remaining pandas key overhead (<= .05 s),
numba / polars (no event-level Python loop left), thread changes (tree sums may reorder).

**Parity** (`s115/parity`: 40 non-locked pool signals, seed 115, stratified by channels, 24-h extract each; 30 min / 3 h
by start/end, 24 h whole; + bench71 4 x 3 = 132 cases x full / le2h; locked_v2 asserted absent) vs **ref114** = v6b +
note-114 patch in the original code: 2,627 detectors, 15,705 candidate pairs per profile. Decisions (phase, 2nd phase,
function, lanes, n_lanes, setback distance / confidence, night speed, health status / reason / periods / watch, status,
review) identical 100 % in both profiles; per-phase tables identical. Probabilities: full max 1.4e-8; le2h 2 detectors
up to 1.3e-4 = reference nondeterminism: SQL lead-neighbour sums split an exact tie at the top-8 cut at random (ref114
x6 on s39_h3: 0.998505 x4 / 0.998632 x2; v7 always keeps both tied neighbours = note-76 rule = 0.998632; same on
s35_h3, where v6b and ref114 already differ by 6e-5). Edge cases (DataFrame lowercase, 3 min, no 43/44, no 7-10, no 131,
one detector, no detector, empty window, two signals chunked): identical (`s115/v7/edge115.py`).
**Documented change (v7 vs v6b, = note 114):** phase identical; function 15 / 2,627 detectors (full), 7 (le2h); setback
distance 142 (median 2 ft: p_mid is a setback input), night speed 14, health_watch 7. OOF of the shipped stackers = note
114: function E >= 30 min .9304 (v6b .9305), 5 min .9022, le2h .9268 (`s114/score114.json`). In-sample sanity on the
parity signals vs v4q (no stack credit): fixes / breaks 7 / 5 (full), 0 / 5 (le2h): small, as note 114.

**Checks:** `check --freeze` then check 6/6 PASS (references bit-identical to ref114's; pyarrow added to the blocked
imports). Fresh venv (`tests/smoke_install.py`, pandas 3.0.6, duckdb 1.5.6, numpy 2.5.3, ort 1.30): wheel installs, CLI
full / le2h run, check 6/6 PASS. Scripts: `research/code/final115/`; logs / outputs `dc_work/s115/{parity,v7}`.
