# Embedding atspm_detector in an application

For developers and agents building `atspm_detector` into a desktop app, web service or scheduled job. The [README](https://github.com/ShawnStrasser/atspm-detector/blob/main/README.md) covers what the package does, basic use, how much data to give it and how accurate it is.

## API

```python
from atspm_detector import predict, list_signals, OUT_COLS, EXTRA_COLS, PHASE_COLS, __version__

det = predict(events, start=None, end=None, device_ids=None, threads=4, memory="4GB",
              chunk_signals=None, batch_signals=None, min_actuations=5, min_prob=0.0)
det, phases = predict(events, ..., return_phases=True)
det, findings = predict(events, ..., return_findings=True)        # both: det, phases, findings
ids = list_signals(events, start=None, end=None)        # DeviceIds present, sorted
```

- `events`: pandas DataFrame, or a path / glob to parquet or CSV. Columns `DeviceId`, `Timestamp`, `EventId`, `Parameter` (lowercase and snake_case variants accepted). Extra columns and other event codes are ignored; duplicate rows are dropped.
- Return: a DataFrame with columns `OUT_COLS + EXTRA_COLS`, one row per detector channel seen in the window. `phases` has `PHASE_COLS`, one row per phase. Treat the column lists as the contract; read columns by name.
- `predict` never raises on thin or odd data. Signals or detectors it cannot judge come back with an empty `phase_pred` / `function_pred`, a `status` saying why and `review_flag = True`. An empty input returns empty frames with the right columns.
- `min_prob > 0` withholds low-confidence answers (status "not confident enough"); the raw opinion stays in `phase_guess` / `function_guess`.
- Signals are processed together (`batch_signals` per batched pass; default from the free memory, 1-25, 10 if unknown), but each keeps its own window and nothing crosses between signals: answers for a signal are bit for bit identical whether it is sent alone or in a batch of any size.

## Time

- Timestamps are read as the controller's local clock. Pass naive local times.
- Time-zone-aware timestamps keep their own clock time; they are never converted to the server's zone.
- A parquet column stored as UTC with no zone recorded is read as UTC clock time, with a `UserWarning`. Convert to local naive time before calling if your store keeps UTC.
- `start` / `end` are clock-time strings or timestamps; `end` is exclusive. Without them a signal's window runs from its first to its last event. Pass them for an exact window.
- `minutes_of_data` is the window length; gaps in the log are not subtracted.

## Threads, memory, speed

- `predict` blocks. Call it from a worker thread or a job queue, never from a UI or request thread.
- Concurrent calls in one process are not tested. Run one call at a time per process; for parallel work use a process pool (one signal or one group of signals per task).
- DuckDB threads and memory: `predict(threads=, memory=)`. onnxruntime threads: environment variables `DC_TREE_THREADS` and `DC_NET_THREADS` (default 4 each). They are read at import, so set them before `import atspm_detector`.
- Model sessions are loaded on first use and cached for the life of the process. Keep a worker process alive between calls instead of starting a new one for each signal.
- Large batches: send many signals in one call; they are processed `batch_signals` at a time (default: half the free physical memory at about 45 MB per typical signal-day of input, between 1 and 25 signals, 10 when the free memory cannot be read; on 24-hour samples a batch of 10 is about 1.7x faster per signal than one at a time, above 10-25 there is no further gain, only memory). Health runs once per batch (DuckDB SQL, about 0.3 s fixed plus 60-75 ms per signal-day). Pass `batch_signals` to fix it. `chunk_signals=N` additionally loads N signals at a time to bound peak memory. Answers are the same either way.
- Rough cost on one desktop CPU: well under 2 s and about 250-300 MB per signal for a 3-hour sample. Measure on your own hardware.

## Errors and logging

- `FileNotFoundError` when the model weights are missing (broken install or a wrong `DC_WEIGHTS` folder).
- `UserWarning` for rows with missing or unreadable values (dropped and counted) and for the UTC case above. Capture them with `warnings.catch_warnings(record=True)` if you want them in your own log.
- `predict(verbose=True)` prints progress to stdout; the default prints nothing.
- Run `atspm-detector-check` (or `python -m atspm_detector.check`, exit code 0 = pass) after installing or upgrading, for example in your app's health check or CI.

## Storing results

- Key rows by `(DeviceId, Detector)` plus your own run id and window. Store `__version__` with each run; answers can change between versions.
- `health_status` and the findings table are a separate output. They never change a phase or function answer.
- Health findings: `return_findings=True` returns the findings table of the README (one row per finding: `DeviceId`, `DetectorType`, `Detector`, `Start`, `End`, `Label`, `Confidence`, `Detail`), with the reasons, set-up notes and notes about a whole signal.
- The health limits are `[suspect, bad]` pairs in `weights/health/health_limits.json`; a single finding is bad from the second number on.

## Detector health in SQL, on your own DuckDB connection

The health checks are plain DuckDB SQL (`predict` runs the same). When the events already sit in a DuckDB database (an atspm pipeline, for
example), run the health where they are instead of copying them out:

```python
import duckdb
from atspm_detector import health, health_sql

con = duckdb.connect("signals.duckdb")
f = health(con, "events", "detectors", start=None, end=None)    # the findings table, vehicle and ped
f, s = health(con, "events", "detectors", summary=True)         # + the status table
sql = health_sql("events", "detectors")                         # the vehicle findings as ONE query: con.execute(sql).df()
sql_f, sql_s = health_sql("events", "detectors", summary=True)  # the two queries
```

- `events`: `DeviceId`, `Timestamp`, `EventId`, `Parameter` (the hi-res log, local clock time). Other columns and event
  codes are ignored; duplicate rows are dropped (`health(..., distinct=False)` skips that when there are none).
- `detectors`: `DeviceId`, `Detector`, `phase`, `function`, `lanes`, optionally `p_advance`, `p_presence`, `p_count`,
  `p_yellow_red`, `p_other`, `p_mid`, `p_bike`. From a `predict` output: `phase` = `phase_guess`, `function` =
  `function_guess`, `lanes` = the number of entries in `lanes` (1 for a Bike zone), the `p_*` columns as they are.
  Without the probabilities (`health_sql(..., probabilities=False)`; `health` looks for them itself) each detector is
  judged by its own function's limits only, never the least strict of the plausible ones. Optional
  `configured_function` (the agency's configured function: `Presence`, `Count`, `Advance`, `Yellow_Red`, ...; empty =
  not known): the pulse-mode check (a predicted Presence detector that only pulses) then never flags a zone configured
  as Count, Advance or Yellow_Red (`health` looks for the column itself; `health_sql(..., configured=True)`).
- `start` / `end`: the sample window (`end` exclusive); default each signal's first event to one second past its last.
- Returns the findings table of the README; `summary=True` also the status table, one row per detector channel with
  actuations in the window and per calendar day. `health_status` in `predict` comes from the same checks and limits.
- Store the events sorted by `DeviceId`, then `Timestamp`: `health` reads them once per group of signals, and an
  unsorted file makes every group scan all of it. `options={...}` switches the vehicle rules
  (`health_duckdb.OPTIONS` lists them with their defaults), `ped_options={...}` the ped rules (`ped_health_duckdb.OPT`).
- Nothing is copied out of the database. `health` runs the steps as temp tables named `hq_*` (each dropped as soon as it
  is no longer needed) and groups signals so a group holds at most `max_events` events (default 4 million, about 2 GB
  peak). `health_sql` is one `WITH` query whose steps are all named `hq_*`; DuckDB takes about half a second to plan it,
  so for repeated calls prefer `health`. Table names are inserted as given.
- Cost on one desktop CPU (4 threads, events already in DuckDB): about 0.5 s for one signal-day, 1 s for 10, 7.5 s
  for 100 (about 75 ms per signal-day in large groups; a fixed half second or so per call).
- Many signals: `health(..., workers=N)` runs the groups in N Python processes at once (one DuckDB process uses only
  2-3 cores on this work, however many threads it has). The connection's `threads` and `memory_limit` are shared out
  among them; `SET threads = N` gives each worker one thread, the most work per core. The window's events are first
  copied once into a DuckDB file under the connection's `temp_directory` (else the system temp folder), so a source on
  a network share is read once and its order no longer matters. Each worker runs `health_duckdb.py` with the same
  Python (`sys.executable`), without importing the rest of the package. The tables must be readable with `SELECT` from the
  connection (tables, views or files; a registered DataFrame works too, it is copied). Same answers as `workers=1`.
  A whole network over 3 days: `SET threads = 12; SET memory_limit = '12GB'` and `workers=12`.
- Health kept up to date from chunks of events, and health in time bins: `atspm_detector.incremental` and `health_bins()`. A store is a
  DuckDB file; a short-lived machine must keep it somewhere permanent between runs. For a run every 15 minutes with the state kept as
  tables, see the next section.

## Live health: a run every 15 minutes, each signal on its own

`atspm_detector.live.live_run` (what atspm's `detector_health` runs with a state) writes each 15-minute bin's rows from the checks that can
answer at once, and a whole-day check per signal and midnight. Detector health is tracked per signal, like the performance measures: what
a signal gets never depends on any other signal, on how runs are cut, or on which signals share a run.

```python
from atspm_detector import live
r = live.live_run(con, "raw_data", "detectors", state=last_state, history="health_rows/*.parquet", ped_history="ped/*.parquet",
                  horizon_days=3, now=None, complete_through=None, clock_ahead_minutes=60, ped_history_days=28)
# tables on con: hl_final (the rows to add), hl_state (the next run's state), hl_rows (add to history), hl_ped (add to ped_history)
# r.findings (each whole-day check's findings, TimeStamp = its midnight), r.checked, r.held_back
```

- **State**: `TimeStamp, DeviceId, EventId, Parameter, Value`, every row with a `DeviceId`. Per signal: the end of its last written bin
  (9602), the midnight of its last whole-day check (9603), the quarter-hour of its first event (9604), the runs whose rows it has in history
  (9605), its events past its last bin (9700), the version (9606), and the carry, limits and open ped ONs. A run over some signals reads
  and writes only their rows: replace a signal's rows with the run's, keep the others, and separate runs over separate signals equal one
  run over all of them.
- **Bins**: a signal's bin is written by the run its own events reach the end of (its last event within a second of it), or
  `complete_through` (a time for every signal, or a table `DeviceId, Through`: the caller knows the signal's events are all in up to then).
  Its events after its last bin wait in the state, so every bin is judged whole in one run, with the rows it gets on time, however late
  the signal reports. A bin with none of the signal's events gets no row.
- **Midnight**: a signal's whole-day check runs at the first run its own events pass the midnight, or when the caller says the day is over
  for it (`complete_through` at or past it, or `now` at least `clock_ahead_minutes` past it). It needs a whole day of the signal's data since
  its first event and some of its data in the check's window. A day the caller closes before the signal's events reach it stays closed:
  later events still go into the store and the later checks, but its bins before that midnight are not written.
- **Clocks**: with `now` (the time now, on the controllers' clock), events stamped at or after `now + clock_ahead_minutes` are left out and
  counted in `r.held_back`, so a clock running ahead moves nothing on. Without `now` each signal's stamps are believed; a wrong clock moves
  only its own signal. Pass `now` in production.
- **Trim**: each signal's rows are trimmed at its own watermark less the horizon, so a signal days behind the others keeps its rows.
- **Equal to one big run**: the same events cut into any runs, each in its own process carrying only these tables, give each signal the same
  rows, findings, ped history and state, whatever the other signals do (tested with one signal 3 hours behind, one in hourly bursts, one 26
  hours behind with a 1-day horizon, subsets with their own cuts, and a signal alone).
- **Ped history**: the ped sudden stop against the button's own weeks (option `stop_history`) needs 14 logged days of the 28 before a
  silence, longer than the usual 3-day horizon. Each run's `hl_ped` holds only what that check reads: per signal and 15 minutes, the presses
  of each ped input (`Input`, `Presses`) and a row with no `Input` for each 15 minutes the signal logged anything. Keep the last
  `horizon_days + ped_history_days` days of it and pass it as `ped_history`; a check reads the `ped_history_days` before its window. Rows are
  read as the largest count per key, so a retried run's rows count once. About 0.4 kB of parquet per signal-day (about 250 rows on a busy
  signal: 96 logged quarter-hours and the press counts), against about 176 kB for `hl_rows`.
- **From 2.1**: a state written by 2.1 (one set of markers with no `DeviceId`) is converted on reading: every signal in it takes those
  markers, which is where each one was, and the runs it lists are listed for each of its signals and checked by run, as before. A signal that
  was behind the others in 2.1 lost those bins then; it goes on from the shared markers.
- **No fall-back hour handling**: atspm drops 1:00-2:00 on the November change day; nothing here treats that hour specially.

## atspm detector configuration

The [atspm](https://pypi.org/project/atspm/) package computes performance measures from the same hi-res events but needs a detector table. `to_atspm_config` builds it from the `predict` output:

```python
from atspm_detector import predict, to_atspm_config, atspm_coverage
from atspm import SignalDataProcessor

det = predict(events)
config = to_atspm_config(det)                 # DeviceId, Phase, Parameter, Function, DistanceFromStopBar
atspm_coverage(config)                        # per phase: arrival_on_green / split_failures / yellow_red enabled?
with SignalDataProcessor(raw_data=events, detector_config=config, bin_size=15, aggregations=[
        {"name": "arrival_on_green", "params": {"latency_offset_seconds": 0}},
        {"name": "split_failures", "params": {"red_time": 5, "red_occupancy_threshold": 0.80,
                                              "green_occupancy_threshold": 0.80, "by_approach": True}},
        {"name": "yellow_red", "params": {"latency_offset_seconds": 0}}]) as p:
    p.load()
    p.aggregate()
    aog = p.conn.query("SELECT * FROM arrival_on_green").df()
```

Command line: `atspm-detector --events events.parquet --out detectors.csv --out-atspm detector_config.csv`.

- atspm is not a dependency. Install it yourself: `pip install "atspm>=3.0"` (older ones cannot load on DuckDB 2). `to_atspm_config` itself needs only numpy and pandas.
- Columns: `DeviceId`, `Phase` (int16), `Parameter` (detector channel, int16), `Function`. atspm selects these four by name; of any others it reads only `DistanceFromStopBar`, `MPH` and `LatencyCorrection`, for arrival on green and platoon ratio. `Function` is matched exactly and case-sensitively.
- Mapping: Advance -> `Advance` (arrival_on_green, platoon_ratio), Presence -> `Presence` (split_failures), Yellow_Red -> `Yellow_Red` (yellow_red), Count -> `Count` (read only by atspm's detector_health, as a Count zone). Mid, Bike and Other are left out; add them with `include=(...)`. Rename with `function_names={"Count": "Stop Bar Count"}`.
- Left out by default, each with a reason in `return_report=True`: detectors without an answer, `health_status` "bad" (`exclude_health=()` keeps them), and same-phase, same-function detectors on a lane that already has one (`one_per_lane=False` keeps them). Lane-by-lane detectors win over one spanning their lanes, then the higher `function_prob`. This keeps a spanning zone stacked over lane loops from counting the same vehicles twice in arrival on green. Exact duplicate rows are never written (atspm would double-count them).
- `min_phase_prob` / `min_function_prob` leave out low-confidence answers. `device_id_dtype="int64"` casts `DeviceId` to match integer ids in your events (`predict` returns text ids; atspm's joins work either way). `extra_columns=True` adds `Lanes`, `DistanceFromStopBar` (ft), `PhaseProb`, `FunctionProb` and `HealthStatus`; atspm 3.0 uses that distance in place of its defaults, the rest are for your own use.
- `results` may also be the CSV written by `--out` or a parquet copy of it.
- Not available from the log, so not in the config: approach, direction and movement (atspm does not need them), and overlap assignments (the model gives the phase only).

## Charts

Optional health charts need matplotlib (`pip install "atspm-detector[charts]"`): `from atspm_detector.charts import health_chart`, then `health_chart(events, det, detector=15, path="det15.png", findings=findings)` draws one detector with its phase mates and shades its findings. `predict` never imports matplotlib. Otherwise draw from the returned columns (and your own copy of the events) in your application.
