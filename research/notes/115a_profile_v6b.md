# 115a — End-to-end profile of package v6b (2026-10-07) — analysis only, no package modified
Brief: profile v6b (time + memory per stage, top functions, Python loops, SQL / ORT settings), rank hotspots with fixes.
Full tables: `%DC_WORK%/s115/profile115.md`. Code: `%DC_WORK%/s115/prof115.py` (stage timers + DuckDB proxy + tracemalloc +
pyinstrument, on the instrumented copy `s115/prof_pkg`), `analyse115.py`, `mem115.py`. Raw per case: `s115/runs/`.
Cases: bench98 extracts (bench71 r8 / r11 / busiest-events / busiest-channels) x 30 min / 3 h / 24 h x full / le2h, fresh process
each, 4 threads; locked_v2 asserted absent. Machine 50-65 % busy (other agents + Defender): absolute times 1.2-2.4x note 111;
use the shares.
## Method notes
* cProfile is unusable with duckdb 1.5.5 / Python 3.13: every Python caller of a DuckDB call disappears from the stats
  (`def g(): duckdb.connect().sql('select 1').fetchall()` -> g missing). Function times from pyinstrument (private install
  `s115/pylib`, not in the venv); stage times are exclusive wall timers incl. each stage's SQL.
## Where the time goes (warm, % of call; full profile)
* 30 min (0.7-1.8 s): spread — pair-feature SQL 13-17 %, setback 13-16 %, network pair graph 7-11 %, function expert SQL
  5-7 %, health 5-7 %, pandas glue; bch lanes decoder 10 %.
* 3 h full: network largest (ONNX pair graph 14-18 % + input assemble 4-7 %), health 6-9 %, pair SQL 11-15 %.
* 24 h: health 13-23 % (le2h 16-23 %), pair-feature SQL (base + partner) 18-27 %, lead-neighbour SQL 3-6 %, streams 4-5 %.
* DuckDB = 21-33 % of a 30-min call, 37-54 % of a 24-h call. Ranker / decoder / function trees / stacker / blend / night
  speed each < 1-4 %.
## Settings found
* ORT (all 23 sessions; 17 under le2h > 2 h): intra 4, inter 1, ENABLE_ALL, sequential, arena ON, mem-pattern ON, spinning
  off; created once per process (0.16-0.34 s cold), reused. DuckDB: new connection per predict() (0.017 s), 4 GB, 4 threads,
  ~99 statements per call.
* Memory: peak WS 300-511 MB; Python heap only 14-55 MB. Imports 116 MB, tree sessions +53, DuckDB features transient
  +38..+150 MB, network arena +77..+104 MB peak. Network sessions with arena OFF: peak 478 -> 374 MB (bch 24 h),
  381 -> 304 MB (3 h), retained 316 -> 220 MB, no time change.
## Waste found
* raw `ev` scanned by 16 statements; ON intervals derived twice from ev (+ a third LEAD pass just to list channels), color
  cycles three times (cyc_raw, streams SQL_CYC, expert cycw), calls / coord / cand twice.
* onev x cand ASOF cyc materialised twice (j, j2); ON -> call-43 ASOF twice; long-ON release ASOF twice; SQL_CALL_REV
  cross-joins calls x detectors (0.26 s at bev 24 h, slowest statement).
* health re-parses the 222-321k-event frame 5x (drop_duplicates + datetime -> seconds + lexsort each): 0.19-0.26 s of
  0.33-0.41 s standalone at 24 h.
* network: input x [D*K,15,T] assembled for ALL pairs on every member and piece, only 18-27 % kept (bch 63 / 344).
* pick pair_mats computed 3x per phase group; lanes cues computed for both orientations; lanes decoder rescoring O(n^2)
  with a new triu_indices per candidate move (5,166 calls at bch).
* ON data pulled / rebuilt 4x per signal; DeviceId / constant `win` object keys on every merge; float64 SQL aggregates
  cast to float32 column by column, then back to float64 for the ONNX trees.
## Top hotspots (est. saving, answers unchanged unless noted) — full list of 15 with fixes in profile115.md section 7
1 health single parse (0.25-0.45 s at 24 h) · 2 pair-SQL dedupe + numpy CALL_REV (0.3-0.5 s at bev 24 h) · 3 streams
from chunk tables (0.1-0.15 s at 24 h; parity check on SQL_CYC) · 4 assemble kept pairs once per piece (0.06-0.2 s) · 5 ORT
network arena off (-100 MB) · 6 lead SQL without per-second table (0.1-0.18 s at 24 h) · 7 lanes decoder delta scoring
(0.14-0.16 s at bch) · 8 setback cues reuse lanes' (0.05-0.2 s; parity check) · 9 pair_mats once (0.03-0.12 s) · 10 pandas
keys / casts (0.03-0.06 s per call) · 11-15 ON pull once, mirror cues, pair-graph batching, expert SQL, cold start.
Combined (1-12): about -35-45 % at 24 h, -20-30 % at 3 h full, -15-20 % at 30 min; -100 MB peak.
## Verdict
No accuracy or package change. The fixes are engineering-only and bit-identical except #3 / #8 (parity check). Next step
(orchestrator): implement 1-12 on a v6c copy, prove parity with check.py + the stored references, re-bench with bench98.
