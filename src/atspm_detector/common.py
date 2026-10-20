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


def param_table(con, name: str, key: str, keys, cols: dict, key_type: str = "SMALLINT") -> None:
    """TEMP TABLE `name` with one row per signal: `key` (`key_type`) and one DOUBLE column per entry of `cols`
    {column: [SQL text per key]}.  Each value is its SQL text CAST to DOUBLE -- bit for bit the value a query got
    when the same text was written into it inline for one signal (a literal like 1759017600.1 is a DECIMAL, `w + 10`
    is DECIMAL arithmetic, and both are cast to DOUBLE where they meet a DOUBLE column).  Lets one batched query apply
    every signal's own window."""
    keys = list(keys)
    names = list(cols)
    if not keys:
        defs = ", ".join([f"{key} {key_type}"] + [f"{c} DOUBLE" for c in names])
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} ({defs})")
        return
    rows = []
    for i, k in enumerate(keys):
        kk = ("'" + str(k).replace("'", "''") + "'") if key_type == "VARCHAR" else f"{int(k)}::{key_type}"
        rows.append("(" + ", ".join([kk] + [f"CAST({cols[c][i]} AS DOUBLE)" for c in names]) + ")")
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT * FROM (VALUES {', '.join(rows)}) "
                f"t({', '.join([key] + names)})")


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
