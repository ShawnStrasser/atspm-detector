# Embedding atspm_detector in an application

For developers and agents building `atspm_detector` into a desktop app, web service or scheduled job. The [README](https://github.com/ShawnStrasser/atspm-detector/blob/main/README.md) covers what the package does, basic use, how much data to give it and how accurate it is.

## API

```python
from atspm_detector import predict, list_signals, OUT_COLS, EXTRA_COLS, PHASE_COLS, __version__

det = predict(events, start=None, end=None, device_ids=None, threads=4, memory="4GB",
              chunk_signals=None, min_actuations=5, min_prob=0.0)
det, phases = predict(events, ..., return_phases=True)
ids = list_signals(events, start=None, end=None)        # DeviceIds present, sorted
```

- `events`: pandas DataFrame, or a path / glob to parquet or CSV. Columns `DeviceId`, `Timestamp`, `EventId`, `Parameter` (lowercase and snake_case variants accepted). Extra columns and other event codes are ignored; duplicate rows are dropped.
- Return: a DataFrame with columns `OUT_COLS + EXTRA_COLS`, one row per detector channel seen in the window. `phases` has `PHASE_COLS`, one row per phase. Treat the column lists as the contract; read columns by name.
- `predict` never raises on thin or odd data. Signals or detectors it cannot judge come back with an empty `phase_pred` / `function_pred`, a `status` saying why and `review_flag = True`. An empty input returns empty frames with the right columns.
- `min_prob > 0` withholds low-confidence answers (status "not confident enough"); the raw opinion stays in `phase_guess` / `function_guess`.
- Each signal is scored on its own. Answers for a signal are identical whether it is sent alone or in a batch.

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
- Large batches: `chunk_signals=N` loads N signals at a time to bound peak memory. Answers are the same either way.
- Rough cost on one desktop CPU: well under 2 s and about 250-300 MB per signal for a 3-hour sample. Measure on your own hardware.

## Errors and logging

- `FileNotFoundError` when the model weights are missing (broken install or a wrong `DC_WEIGHTS` folder).
- `UserWarning` for rows with missing or unreadable values (dropped and counted) and for the UTC case above. Capture them with `warnings.catch_warnings(record=True)` if you want them in your own log.
- `predict(verbose=True)` prints progress to stdout; the default prints nothing.
- Run `atspm-detector-check` (or `python -m atspm_detector.check`, exit code 0 = pass) after installing or upgrading, for example in your app's health check or CI.

## Storing results

- Key rows by `(DeviceId, Detector)` plus your own run id and window. Store `__version__` with each run; answers can change between versions.
- `health_*` columns are a separate output. They never change a phase or function answer.
- `health_bad_periods` is JSON text: a list of `{start, end, what, recovered}` periods, or empty text when there are none.

## atspm detector configuration

The [atspm](https://pypi.org/project/atspm/) package computes performance measures from the same hi-res events but needs a detector table. `to_atspm_config` builds it from the `predict` output:

```python
from atspm_detector import predict, to_atspm_config, atspm_coverage
from atspm import SignalDataProcessor

det = predict(events)
config = to_atspm_config(det)                 # DeviceId, Phase, Parameter, Function
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

- atspm is not a dependency (it pulls pyarrow and ibis-framework, which currently requires pandas < 3). Install it yourself: `pip install atspm`. `to_atspm_config` itself needs only numpy and pandas.
- Columns: `DeviceId`, `Phase` (int16), `Parameter` (detector channel, int16), `Function`. atspm selects these four by name and ignores any others. `Function` is matched exactly and case-sensitively.
- Mapping: Advance -> `Advance` (arrival_on_green, platoon_ratio), Presence -> `Presence` (split_failures), Yellow_Red -> `Yellow_Red` (yellow_red), Count -> `Stopbar Count` (exported for your own joins; no atspm measure reads it). Mid, Bike and Other are left out; add them with `include=(...)`. Rename with `function_names={"Count": "Stop Bar Count"}`.
- Left out by default, each with a reason in `return_report=True`: detectors without an answer, `health_status` "bad" (`exclude_health=()` keeps them), and same-phase, same-function detectors on a lane that already has one (`one_per_lane=False` keeps them). Lane-by-lane detectors win over one spanning their lanes, then the higher `function_prob`. This keeps a spanning zone stacked over lane loops from counting the same vehicles twice in arrival on green. Exact duplicate rows are never written (atspm would double-count them).
- `min_phase_prob` / `min_function_prob` leave out low-confidence answers. `device_id_dtype="int64"` casts `DeviceId` to match integer ids in your events (`predict` returns text ids; atspm's joins work either way). `extra_columns=True` adds `Lanes`, `DistanceFromStopBar` (ft), `PhaseProb`, `FunctionProb` and `HealthStatus` for your own use.
- `results` may also be the CSV written by `--out` or a parquet copy of it.
- Not available from the log, so not in the config: approach, direction and movement (atspm does not need them), and overlap assignments (the model gives the phase only).

## Charts

Optional health charts need matplotlib (`pip install "atspm-detector[charts]"`): `from atspm_detector.charts import health_chart`, then `health_chart(events, det, detector=15, path="det15.png")` draws one detector with its phase mates and the periods the health check lists. `predict` never imports matplotlib. Otherwise draw from the returned columns (and your own copy of the events) in your application.
