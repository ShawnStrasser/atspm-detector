# Changelog

## 1.0.0 - first public release

- `predict()` returns one row per detector from the controller's high-resolution event log alone: phase, function (Advance, Presence, Count, Yellow_Red, Mid, Bike, Other), lanes, setback distance, night speed and detector health. `return_phases=True` adds a per-phase lane table. No cabinet print needed.
- `to_atspm_config()` turns the output into the `detector_config` of the [atspm](https://pypi.org/project/atspm/) package (Advance, Presence, Yellow_Red, and Count as "Stopbar Count"; detectors with health "bad" left out by default; at most one detector per phase, function and lane). `atspm_coverage()` lists which atspm measures each phase gets. Command line: `--out-atspm config.csv`.
- Command line tools `atspm-detector` (predict to CSV) and `atspm-detector-check` (checks the install on the bundled sample).
- CPU only. Dependencies: numpy, pandas, duckdb, onnxruntime (1.21 or newer). Optional: matplotlib for health charts (`[charts]`); atspm is installed separately (`pip install atspm`).
