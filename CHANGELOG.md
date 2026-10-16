# Changelog

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
