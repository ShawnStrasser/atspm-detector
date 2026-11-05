"""atspm_detector: phase, function, lanes, setback, night speed and health of every detector of a traffic signal,
from the controller's high-resolution event log alone (CPU, onnxruntime; no channel-to-phase table).

    from atspm_detector import predict, to_atspm_config
    out = predict("events.parquet")                  # one row per detector
    out, phases = predict(df, return_phases=True)    # + the per-phase lane table
    config = to_atspm_config(out)                    # detector configuration for the atspm package
    f = health(con, "events", "detectors")           # vehicle and ped health on your own DuckDB connection (no copy):
                                                     # one findings table; no row = no finding
    f, s = health(con, "events", "detectors", summary=True)   # + the status table, one row per detector and day
    sql = health_sql("events", "detectors")          # ... the vehicle part as one SQL query to run yourself
    p = ped_health(con, "events")                    # ped buttons only: health(..., vehicle=False)
    a = ped_assignments(con, "events")               # which phases / ped overlaps each ped input calls (model)
    d = ped_detectors(con, "events")                 # the same, in predict()'s columns (one row per input, movement)

Check an install with `python -m atspm_detector.check` (or the `atspm-detector-check` command).
"""
from __future__ import annotations

__version__ = "2.0.0"

from .atspm_export import atspm_coverage, to_atspm_config  # noqa: E402
from .pipeline import (DEFAULT_MODEL_DIR, EXTRA_COLS, OUT_COLS, PHASE_COLS, PROB_COLS,  # noqa: E402
                      list_signals, main, predict, run)
# note: the name `health` is the function below; the gate module stays importable as `atspm_detector.health` via
# `from atspm_detector.health import ...`
from .findings import health, health_sql, ped_health  # noqa: E402
from .bins import health_bins  # noqa: E402
from .ped_assign import ped_assignments, ped_detectors  # noqa: E402

__all__ = ["predict", "list_signals", "run", "main", "to_atspm_config", "atspm_coverage", "health", "health_sql", "health_bins",
           "ped_health", "ped_assignments", "ped_detectors", "OUT_COLS", "EXTRA_COLS", "PHASE_COLS", "PROB_COLS", "DEFAULT_MODEL_DIR", "__version__"]
