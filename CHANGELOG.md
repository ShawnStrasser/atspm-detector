# Changelog

## 2.0.0 - one findings table, ped health, health kept up to date from chunks, time bins

Phase, function, lanes, setback and night speed are unchanged.

**Breaking**

- `health()` returns one findings table for vehicle detectors and ped buttons together, one row per finding; no row means no finding:
  `DeviceId`, `DetectorType` (`Vehicle` or `Ped`), `Detector`, `Start`, `End`, `Label`, `Confidence` (0.5 at a check's suspect limit, 0.8 at its bad limit), `Detail`. A finding with no time of its own gets
  one row per calendar day of the window. Config notes, signal notes, borderline (watch) findings and `not_enough_data` are rows too.
- `health(..., summary=True)` returns `(findings, status)`: `DeviceId`, `DetectorType`, `Detector`, `Date`, `Status`, exactly one row per detector or button and calendar day, enforced.
- Ped runs when the log holds ped input events; `vehicle=False` or `ped=False` runs one side. `ped_health()` is a thin helper for `vehicle=False`.
- `health_sql()` gives the vehicle findings as one query (`summary=True`: two queries). Ped has no one-query form.
- `predict()`: the only health column left is `health_status`; `health_score`, `health_reason`, `health_watch`, `health_categories`, `health_config`, `health_signal_note` and `health_bad_periods` are gone,
  their content is in the findings table. New: `return_findings=True`, `--out-findings`, `health_chart(findings=)`.
- Needs DuckDB 2.0 or newer. Python 3.11 to 3.14 (3.10 dropped: DuckDB 2.0 has no wheel for it; 3.15 waits for an onnxruntime wheel); numpy 2, pandas 2.2 or 3, onnxruntime 1.31, matplotlib 3.11 for the charts. New dependency `jinja2`. With atspm: 3.0 or newer, as older ones cannot load on DuckDB 2.

**Incremental health** (`atspm_detector.incremental`)

- `attach_store`, `update` (a chunk of events into small stored rows, any cutting of the events, one transaction, late rows counted), `judge`, `rebuild` (a signal again from raw events), `trim`,
  `export_store` / `import_store` (across DuckDB versions), `judge_rows` (the per-detector rows underneath).
- `judge()` returns `HealthResult(findings, status, q15, active, bins)`. Judged at T with horizon H on events from T - H on, it equals `health(start=T - H, end=T, bin_origin="clock", summary=True)`, for any cutting of the events.
- `outputs=("tables", "15min")` adds `q15`; `"active"` adds `active`, not yet reviewed. A store written by an incompatible version of the package is refused.
- New `bin_origin` of `health()` / `health_sql()`: `"window"` (default) or `"clock"` (the window starts on the quarter-hour at or before its start).
- New `atspm_detector.live`: a run every 15 minutes writes its own bins, the first run past a midnight also runs the whole check and adds what it finds in the previous day, and state is two tables, not a store file. Midnight comes from event stamps only. `judge_rows(limits=True)` adds `r.limits`, which the per-run rules read.

**Health in time bins**

- `judge(outputs=("tables", "bins"), bin_minutes=15)` and `health_bins()`: one row per bin per problem, `TimeStamp`, `DeviceId`, `DetectorType`, `Detector`, `Label`, `Score`; no row = no problem found.
  `Score` is `Confidence` scaled per rule: a complete failure scores about 1. A bin partly covered is averaged; problems at one moment share its worst value. Borderline findings and notes have
  no rows; not enough data has one row a day with an empty `Score`.
- Past rows never change. A later judge releases any problem it is the first to find in a bin of its window as a new row with its past `TimeStamp`, unless that bin, detector and label has a row.
  New rows of a bin share what its existing rows leave, so a bin adds up to at most 1; a bin already full gets the row with `Score` 0. Not enough data is not a late find.
- A button or detector rated stuck ON past the window keeps its bins until its next event, through an OFF with an ON at most 60 s after it for a button. `health()` on a window is unchanged.
- Dashboard ranges of the plain average of `Score` (its sum over the period / the bins of the period): vehicle 3 days Good under 0.2 %, Bad from 2 %; 7 days Good under 0.1 %, Bad from 1 %; ped 3 days Good under 0.05 %, Bad from 0.5 %;
  7 days Good under 0.02 %, Bad from 0.2 %; Suspect between. An average dilutes a short failure.

**Vehicle health**

- Windows from 30 minutes to 92 days, so a report can give it 2-3 days instead of a history baseline. Silences name the day, merge a quiet run broken by one near-empty bin and need an hour of data outside them; a silence under 15 min is
  suspect at most; its expected count is capped by the detector's own count at those clock times on the other days. Count drops get an end (back to 60 % of the pre-drop share). A detector without phase mates can be rated on a count drop;
  a long zone whose counts fall while its occupancy holds is not a count drop.
- New findings: all inputs of a signal silent for an hour or more of daytime while it serves greens, also across a stretch with no log; a whole phase dropping together; detectors of a phase logging identical counts; **No daily pattern**; **Counts too high**.
  Hour-by-hour drops and rises against each phase mate, by the detector's share of it at that time of day on the window's days. Busy at night is bad only at 1.5 x its limit.
- **Lane closures and timing plans** (statuses `lane_closure`, `timing_plan`): a lane near zero for an hour or more while a detector of the same kind on its phase or NEMA-paired phase takes its vehicles is a closure, not a failure,
  unless the detector has faults of its own before and after it that day; a silence that starts and ends with a coordination pattern change follows the timing plan.
- **Pulse mode**: a presence detector whose calls nearly all last 0.2 s or less is suspect, never bad from this alone. An optional `configured_function` column: a zone configured as Count, Advance or Yellow_Red is never flagged.
- Time ON is no longer imputed: an ON then an ON is still ON unless the whole signal stopped logging between them, a second OFF is ignored, a leading OFF dropped. Detectors that do not look like traffic are nobody's yardstick for
  the night and count checks. No "counts too high" on a day when the detector nearly stopped counting on the window's other days.
- Plain words for a rise against almost-silent phase mates: from 100 x the Detail says the counts rose far above its phase mates. Health text no longer depends on how DuckDB splits the work.
- Every rule is an option of `health(..., options={...})` (`health_duckdb.OPTIONS`; `OPTIONS_OFF` = 1.2.0's health without its imputed time ON).
- The health SQL moved into Jinja templates. `health()` first turns each signal's events into small 5-minute rows and judges those; a value exactly on a limit now counts as exactly on it, where rounding noise used to put it on either side.

**Ped health** (new)

- `ped_health(con, "events")`, and the ped side of `health()`: DuckDB SQL on the ped events. Checks: stuck on, long hold, machine-regular presses, never calls a walk, walks without presses, sustained pressing, sudden stop. Statuses ok, config_issue, suspect, bad, not_enough_data.
  Rules in the README; `options=` (`ped_options=` in `health()`) switches them (`ped_health_duckdb.OPT`).
- **Walks without presses**: walks jump while presses stop for an hour of log or more; finds a stuck ON whose ON was lost. Suspect, bad from 2.5 logged hours of walks at nearly every green. It counts unique presses, made inside the judge from the
  window's own presses; all other checks stay on raw presses.
- **Sustained pressing**: suspect; a day with 1,000 or more presses, at least 5 times the median of the window's other logged days (itself at least 20), and 5 or more 15-minute periods with 30 or more presses and at least 10 per walk.
- **Sudden stop**: the button's longest silence against its own level at those times of day, allowing for clumps; suspect at a chance of 1 in 1,000 or at 24 logged hours from 30 presses a day, bad at 1 in 1,000 with bursts capped. A quiet button cannot flag.
- Stuck-on periods of a button at most 60 s apart are one period (`hold_join`); a brief press between them stays in every count. A very long ON of a presence input with a log hole inside it is stuck on when its longest stretch between holes reaches 600 s.
- A call that waited is a note, never a fault; a wait with any preemption reads "omitted during preemption". Lost data never makes a finding. A button on ped recall is never a `config_issue`; recall is read from the log alone.
- `ped_assignments(con, "events")`: the phases and ped overlaps each ped input calls (a 23-feature LightGBM model in onnxruntime, `weights/ped_assign`). The model knows no input, phase or overlap number; a test renumbers the raw log and every answer must
  follow. `in_recall`: no answer, no guess. Right for 99.7 % of 5,055 inputs, signal-grouped cross-validation (88 recall inputs not scored).
- `ped_detectors(con, "events")`: the same answers in `predict()`'s columns and types, to stack under the vehicle detectors. `ped_health(..., assign=True)` turns an input with 10 presses or more that calls nothing into a `config_issue`.

**Speed and other**

- `health(..., workers=N)` runs the signal groups in N worker processes, same tables as `workers=1`: a network of 889 signals over 3 days, 7.3 min in one process, 1.8 min with `workers=12` (about 9 GB of memory at peak).
- `health()` reads the events once per group of signals: store them sorted by `DeviceId`, then `Timestamp`.
- The bundled 30-minute demo log is re-stamped to 2026-10-06, the same weekday and clock times.

## 1.2.0 - batched prediction, detector health in DuckDB SQL

- Detector health is now DuckDB SQL (`health_duckdb.py`), run for all signals of a batch in one pass straight
  from the loaded events; the numpy health modules (`health_core`, `health_v4`, `health_v4_stats`, `health_counts`)
  are removed. Same checks, limits and order of decisions: on the bundled sample, 100 24-hour signals (batched and alone)
  and 4-5 day logs every status, category, reason, bad period and note equals 1.1.0's; scores within 1e-6. Health time
  per signal-day (4 threads): 1 signal 0.41 s (was 0.12), 10 signals 75 ms (58), 100-250 signals 68-70 ms (79-83),
  2-day windows of 100-250 signals 58-59 ms (70-72); a fixed ~0.3 s of query planning per batch.
- New: `health(con, events_table, detectors_table, start=None, end=None)` and `health_sql(...)`: the same health on
  the caller's own DuckDB connection where the events already are (nothing copied out), or as one SQL query to run
  yourself. `detectors_table`: `DeviceId`, `Detector`, `phase`, `function`, `lanes` (+ optional class probabilities
  `p_advance` ... `p_bike`). See docs/embedding.md.
- `predict()` now processes many signals together (`batch_signals`; CLI `--batch-signals`): one set of DuckDB queries, one tree-model call per stage and one setback call for the whole batch, every signal still on its own window. On 24-hour samples a 100-signal call takes 81 s instead of 129 s (1.6x; 0.81 s per signal-day); one signal alone is unchanged. Peak memory for 100 signals 2.1 GB (was 1.7 GB); `batch_signals=1` restores one signal at a time.
- Answers are bit for bit the same as 1.1.0 and the same alone or in any batch (checked on the bundled sample, 100 24-hour signals alone and batched, and 4-5 day logs). The tree models are scored per signal block, because onnxruntime sums trees in a different order for calls of more than 50 rows.
- `batch_signals=None` (the default) now follows the free physical memory: half of it at about 260 MB per million loaded events of an average signal (about 45 MB per typical signal-day), between 1 and 25 signals; 10 when the free memory cannot be read. Above 10-25 signals a batch gains no speed and only costs memory (100 signals in one call: peak 2.1 GB at 10, 2.7 GB at 25, 3.9 GB at 50). An explicit `batch_signals` still wins.

## 1.1.0 - reviewed health checks, bike lanes

Includes the fixes prepared as 1.0.1 (never released).

- Fixed a crash on samples that span more than 24.8 days (`ValueError: 'list' argument must have no negative elements`): the network's time offsets were stored as 32-bit milliseconds and overflowed. They are now 64-bit. Answers on shorter samples are unchanged.
- A detector whose final function is Bike now covers exactly one lane and never adds a lane to its phase. Before, its phase's `phase_n_lanes` could be one too many (21 of 4,083 phases on 755 signals). Phase, function, setback and night speed are unchanged.
- Health is six checks, with limits reviewed by a traffic engineer: stuck on, counts fall, misses vehicles at night, busy at night, too-fast actuations / too many for the traffic, and the new spiky counts. Removed: erratic counts, chattering, erratic time ON, ON longer than its kind, unusual daily pattern and the "undercounts vs partner" note.
- Limits are `[suspect, bad]` pairs per check and detector type in `weights/health/health_limits.json`: a single finding is bad from the second number on (no hidden rule). Samples under about 8 hours keep the 1.0 limits for stuck on, goes silent and too many.
- Stuck on, 24 hours: bad from 10 minutes (bike), 20 (Count, Yellow_Red), 30 (Advance), 35 (Mid), 60 (Presence, Other). Detectors held ON together that start and end within 5 seconds (not two inputs of the same zone) are one common failure: bad, with a `health_signal_note`. A hold the queue test excused for 1 hour or more is not excused when the same traffic did not hold it elsewhere in the sample.
- Counts fall: goes silent, or the hourly count drops against the phase mates (GEH 10 suspect, 13 bad) while its own occupancy drops too. Misses vehicles at night: GEH of the average night hour (8 suspect, 10.5 bad); a night silence with fewer than 100 vehicles expected is suspect at most. Bike zones get neither. Both need 3 hours of data.
- Busy at night needs the detector to be busier at night than its phase mates for any finding.
- Too fast: every ON-to-ON gap under 0.8 s per lane (Count, Presence, Yellow_Red) or 0.6 s per lane (others), per 1,000 ONs; zones over 2+ lanes keep the 1-lane limit; bad from 1.17 x the limit; "counts each vehicle twice" when the short gaps come in pairs. Too many on 24 hours: busiest 5 minutes over 2,100 vehicles per hour per lane (bad 2,360).
- Spiky counts (new): bursts in 10-minute counts at least 3 x the detector's own level of the surrounding 2 hours and GEH 5, beyond its phase mates' own rise; 2 a day suspect, 3 bad; not checked under 200 vehicles a day, suspect at most under 500 unless a burst reaches GEH 10. Needs 3 hours of data.
- `health_bad_periods` `what` is one of `stuck on`, `count zone held ON`, `silent`, `silent at night`, `counts dropped`, `too-fast actuations`, `too many for the traffic`, `busy at night`, `spiky counts`.
- On 24-hour samples of 755 training signals 4.4 % of detectors are flagged (2.3 % suspect, 2.1 % bad; 1.0: 4.5 %). Health takes about 40 % less time than 1.0 (median 113 ms per signal-day, was 191 ms).

## 1.0.0 - first public release

- `predict()` returns one row per detector from the controller's high-resolution event log alone: phase, function (Advance, Presence, Count, Yellow_Red, Mid, Bike, Other), lanes, setback distance, night speed and detector health. `return_phases=True` adds a per-phase lane table. No cabinet print needed.
- `to_atspm_config()` turns the output into the `detector_config` of the [atspm](https://pypi.org/project/atspm/) package (Advance, Presence, Yellow_Red, and Count as "Stopbar Count"; detectors with health "bad" left out by default; at most one detector per phase, function and lane). `atspm_coverage()` lists which atspm measures each phase gets. Command line: `--out-atspm config.csv`.
- Command line tools `atspm-detector` (predict to CSV) and `atspm-detector-check` (checks the install on the bundled sample).
- CPU only. Dependencies: numpy, pandas, duckdb, onnxruntime (1.21 or newer). Optional: matplotlib for health charts (`[charts]`); atspm is installed separately (`pip install atspm`).
