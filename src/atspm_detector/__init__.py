"""atspm_detector: phase, function, lanes, setback, night speed and health of every detector of a traffic signal,
from the controller's high-resolution event log alone (CPU, onnxruntime; no channel-to-phase table).

    from atspm_detector import predict, to_atspm_config
    out = predict("events.parquet")                  # one row per detector
    out, phases = predict(df, return_phases=True)    # + the per-phase lane table
    config = to_atspm_config(out)                    # detector configuration for the atspm package

Check an install with `python -m atspm_detector.check` (or the `atspm-detector-check` command).
"""
from __future__ import annotations

__version__ = "1.0.0"

from .atspm_export import atspm_coverage, to_atspm_config  # noqa: E402
from .pipeline import (DEFAULT_MODEL_DIR, EXTRA_COLS, OUT_COLS, PHASE_COLS, PROB_COLS,  # noqa: E402
                      list_signals, main, predict, run)

__all__ = ["predict", "list_signals", "run", "main", "to_atspm_config", "atspm_coverage", "OUT_COLS", "EXTRA_COLS",
           "PHASE_COLS", "PROB_COLS", "DEFAULT_MODEL_DIR", "__version__"]
