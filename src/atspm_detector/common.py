"""Event-code constants shared by the inference modules.

The hi-res log follows the Indiana enumerations.  Only the codes listed here are ever
read; everything else in the log is discarded before any feature is built.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EV_BEGIN_GREEN = 1
EV_GREEN_TERM = 7
EV_BEGIN_YELLOW = 8
EV_END_YELLOW = 9
EV_BEGIN_RED_CLEAR = 10
EV_END_RED_CLEAR = 11
EV_CALL_ON = 43          # phase call registered   (Parameter = phase)
EV_CALL_OFF = 44         # phase call dropped
EV_DET_OFF = 81          # detector off            (Parameter = detector channel)
EV_DET_ON = 82           # detector on
EV_DET_FAULT = (83, 84, 85, 86, 87, 88)   # restored / other / watchdog / open / short /
#                                           excessive change
EV_COORD_PATTERN = 131   # Parameter = pattern (0 or 254 => free, 255 => flash)
EV_COORD_YIELD = 150     # Parameter = coordinated phase
EV_FLASH = 173

# Detector fault events (83-88) are deliberately NOT read (user decision 2026-09-28): nothing in the
# model, the health output or the gate uses them.
ALLOWED_EVENTS = (1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173)

# Parameter > 64 on 81/82 are dummy detectors and are dropped.
MAX_DETECTOR_CHANNEL = 64


def read_json(path):
    """A JSON file (UTF-8), closed after reading."""
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def timedelta(days=0, hours=0, minutes=0, seconds=0) -> pd.Timedelta:
    """pd.Timedelta(days=, hours=, minutes=, seconds=) without the NumPy 'generic' timedelta DeprecationWarning that
    pandas 2.x raises under NumPy >= 2.5: the same integer nanoseconds, by pandas' own keyword formula."""
    ns = int(((days * 24 + hours) * 3600 + minutes * 60 + seconds) * 1_000_000_000)
    return pd.Timedelta(np.timedelta64(ns, "ns"))
