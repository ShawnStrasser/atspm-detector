"""predict() output -> the detector configuration of the `atspm` package (https://pypi.org/project/atspm/).

    from atspm_detector import predict, to_atspm_config
    det = predict(events)
    config = to_atspm_config(det)          # DeviceId, Phase, Parameter, Function
    # then: atspm.SignalDataProcessor(raw_data=events, detector_config=config, ...)

Command line: `atspm-detector --events events.parquet --out detectors.csv --out-atspm detector_config.csv`.

What atspm reads from `detector_config`: four columns selected by name -- `DeviceId`, `Phase`, `Parameter` (the
detector channel) and `Function` (text, matched exactly and case-sensitively). Extra columns are ignored. One row per
(detector, phase). The measures that use it:

    Function 'Advance'    -> arrival_on_green, platoon_ratio
    Function 'Presence'   -> split_failures
    Function 'Yellow_Red' -> yellow_red
    any other Function    -> carried, never read (we write Count as 'Stopbar Count'; atspm's own sample config
                             spells it 'stop bar count' -- neither is read)

atspm's other measures (actuations, has_data, timeline, detector_health, ...) read the raw events only.

This module needs numpy and pandas only; it never imports atspm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATSPM_COLUMNS = ["DeviceId", "Phase", "Parameter", "Function"]

# our function class -> atspm Function text.  atspm measures read only Advance, Presence and Yellow_Red.
DEFAULT_FUNCTION_NAMES = {
    "Advance": "Advance",
    "Presence": "Presence",
    "Yellow_Red": "Yellow_Red",
    "Count": "Stopbar Count",
    "Mid": "Mid",
    "Bike": "Bike",
    "Other": "Other",
}
ATSPM_USES = {
    "Advance": ("arrival_on_green", "platoon_ratio"),
    "Presence": ("split_failures",),
    "Yellow_Red": ("yellow_red",),
}
ATSPM_CLASSES = ("Advance", "Presence", "Count", "Yellow_Red")      # at most one of each per lane
DEFAULT_INCLUDE = ("Advance", "Presence", "Yellow_Red", "Count")

_NEEDED = ["DeviceId", "Detector", "phase_pred", "function_pred"]
EXTRA_OUT = ["Lanes", "DistanceFromStopBar", "PhaseProb", "FunctionProb", "HealthStatus"]
REPORT_COLS = ["DeviceId", "Detector", "phase_pred", "function_pred", "Function", "exported", "reason"]
COVERAGE_COLS = ["DeviceId", "Phase", "arrival_on_green", "split_failures", "yellow_red"]


def _lane_set(v) -> frozenset:
    """'1,2' / '1' / 1 / 1.0 -> {1, 2} / {1}; missing or unreadable -> empty (no lane answer)."""
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return frozenset()
    try:
        return frozenset(int(float(x)) for x in str(v).split(",") if x.strip())
    except ValueError:
        return frozenset()


def _read(results) -> pd.DataFrame:
    """the predict() frame, or a CSV / parquet file written from it (parquet read with DuckDB: no pyarrow needed)."""
    if isinstance(results, pd.DataFrame):
        return results
    if isinstance(results, (str, Path)):
        p = str(results)
        if p.lower().endswith(".csv"):
            return pd.read_csv(p, dtype={"DeviceId": str, "lanes": str})
        import duckdb
        con = duckdb.connect()
        try:
            return con.execute("SELECT * FROM read_parquet(?)", [p]).df()
        finally:
            con.close()
    raise TypeError("results must be the DataFrame returned by predict(), or a CSV / parquet path to it")


def to_atspm_config(results, *,
                    include=DEFAULT_INCLUDE,
                    function_names: dict | None = None,
                    exclude_health=("bad",),
                    min_phase_prob: float = 0.0,
                    min_function_prob: float = 0.0,
                    one_per_lane: bool = True,
                    device_id_dtype=None,
                    extra_columns: bool = False,
                    return_report: bool = False):
    """predict() output -> atspm `detector_config` (columns DeviceId, Phase, Parameter, Function).

    results            the DataFrame returned by predict(), or a CSV / parquet path to one.
    include            function classes to export.  Default: the four ATSPM classes.  atspm reads only Advance,
                       Presence and Yellow_Red; Count rows are exported as 'Stopbar Count' for your own joins (e.g.
                       stop-bar volume per phase from atspm's `actuations` table).  Add "Mid", "Bike", "Other" for a
                       full inventory (atspm ignores them).
    function_names     override the class -> atspm Function text map (DEFAULT_FUNCTION_NAMES), e.g.
                       {"Count": "Stop Bar Count"}.
    exclude_health     health_status values left out (default "bad": a stuck or dead detector would corrupt
                       occupancy and arrival measures).  () keeps every detector.  Health never changes a phase or
                       function answer, only whether the detector is exported.
    min_phase_prob, min_function_prob
                       leave out answers below these probabilities (default 0 = off).
    one_per_lane       at most one detector per (signal, phase, Function) per lane: lane-by-lane detectors win over a
                       detector spanning their lanes, then the higher function probability; detectors without a lane
                       answer are kept.  Stops a spanning zone stacked over lane-by-lane loops from counting the same
                       vehicles twice in arrival_on_green / yellow_red.
    device_id_dtype    cast DeviceId (e.g. "int64") to the type of the events you give atspm.  predict() returns
                       DeviceId as text; atspm's joins work either way, but its output tables then carry text ids.
                       None = keep as returned.
    extra_columns      also return Lanes, DistanceFromStopBar (ft), PhaseProb, FunctionProb, HealthStatus.  atspm
                       ignores them.
    return_report      also return one row per input detector: exported or not, and why (column `reason`).

    Returns the config (sorted by DeviceId, Phase, Function, Parameter; no duplicate rows), or (config, report).
    Detectors without an answer (fewer than `min_actuations` ON events, or withheld by `min_prob`) are never exported.
    """
    r = _read(results)
    miss = [c for c in _NEEDED if c not in r.columns]
    if miss:
        raise ValueError(f"results is missing {miss}: pass the DataFrame returned by predict()")
    names = dict(DEFAULT_FUNCTION_NAMES)
    if function_names:
        names.update(function_names)
    unknown = [c for c in include if c not in names]
    if unknown:
        raise ValueError(f"no atspm Function name for {unknown}: pass function_names")

    d = r.reset_index(drop=True).copy()
    for c, default in (("phase_prob", np.nan), ("function_prob", np.nan), ("health_status", None),
                       ("health_reason", None), ("lanes", None), ("distance_ft", np.nan), ("status", None)):
        if c not in d.columns:
            d[c] = default
    fn = d["function_pred"].astype(object).where(d["function_pred"].notna(), None)
    reason = pd.Series([""] * len(d), index=d.index, dtype=object)

    def mark(mask, text):
        m = (mask & reason.eq("")).to_numpy(bool)
        reason[m] = text if isinstance(text, str) else pd.Series(text, index=d.index)[m]

    def txt(col) -> pd.Series:
        return d[col].astype(object).where(d[col].notna(), "").astype(str)

    no_ans = d["phase_pred"].isna() | fn.isna() | fn.astype(str).eq("")
    mark(no_ans, "no answer: " + txt("status"))
    mark(~fn.isin(list(include)), "class not exported: " + fn.astype(str))
    if exclude_health:
        mark(d["health_status"].isin(list(exclude_health)),
             "health " + txt("health_status") + ": " + txt("health_reason"))
    fprob = pd.to_numeric(d["function_prob"], errors="coerce")
    if min_phase_prob > 0:
        mark(pd.to_numeric(d["phase_prob"], errors="coerce").fillna(0).lt(min_phase_prob),
             f"phase_prob < {min_phase_prob}")
    if min_function_prob > 0:
        mark(fprob.fillna(0).lt(min_function_prob), f"function_prob < {min_function_prob}")

    if one_per_lane:
        cand = d.loc[reason.eq("") & fn.isin(ATSPM_CLASSES), ["DeviceId", "Detector", "phase_pred", "lanes"]].copy()
        cand["fn"] = fn[cand.index]
        cand["fprob"] = fprob[cand.index]
        cand["ls"] = cand["lanes"].map(_lane_set)
        cand["nl"] = cand["ls"].map(len)
        cand["k"] = cand["nl"].where(cand["nl"] > 0, 99)          # lane-by-lane first, spanning next, no lane last
        cand["det"] = pd.to_numeric(cand["Detector"])
        for _, g in cand.groupby(["DeviceId", "phase_pred", "fn"], sort=False):
            if len(g) < 2:
                continue
            g = g.sort_values(["k", "fprob", "det"], ascending=[True, False, True], na_position="last")
            taken: dict[int, int] = {}
            for i, ls, f, det in zip(g.index, g["ls"], g["fn"], g["det"]):
                if not ls:
                    continue                                        # no lane answer: not constrained
                clash = sorted({taken[x] for x in ls if x in taken})
                if clash:
                    reason[i] = f"lane already has {f}: det " + ",".join(str(c) for c in clash)
                else:
                    for x in ls:
                        taken[x] = int(det)

    keep = reason.eq("").to_numpy(bool)
    function = fn.map(names)
    k = d[keep]
    cfg = pd.DataFrame({
        "DeviceId": k["DeviceId"],
        "Phase": pd.to_numeric(k["phase_pred"]).astype("int16"),
        "Parameter": pd.to_numeric(k["Detector"]).astype("int16"),
        "Function": function[keep].astype(str),
    })
    if extra_columns:
        cfg["Lanes"] = k["lanes"].astype(object).where(k["lanes"].notna(), None)
        cfg["DistanceFromStopBar"] = pd.to_numeric(k["distance_ft"], errors="coerce").astype(float)
        cfg["PhaseProb"] = pd.to_numeric(k["phase_prob"], errors="coerce").astype(float)
        cfg["FunctionProb"] = fprob[keep].astype(float)
        cfg["HealthStatus"] = k["health_status"]
    if device_id_dtype is not None:
        cfg["DeviceId"] = cfg["DeviceId"].astype(device_id_dtype)
    cfg = cfg.drop_duplicates(ATSPM_COLUMNS)
    cfg = cfg.sort_values(["DeviceId", "Phase", "Function", "Parameter"]).reset_index(drop=True)
    if not return_report:
        return cfg
    rep = pd.DataFrame({"DeviceId": d["DeviceId"], "Detector": d["Detector"], "phase_pred": d["phase_pred"],
                        "function_pred": d["function_pred"], "Function": function.where(keep, None),
                        "exported": keep, "reason": reason.where(~keep, "")})
    return cfg, rep[REPORT_COLS].sort_values(["DeviceId", "Detector"]).reset_index(drop=True)


def atspm_coverage(config: pd.DataFrame) -> pd.DataFrame:
    """Per signal and phase of a config from to_atspm_config(): which atspm measures it enables (True / False)."""
    rows = []
    for (dv, ph), g in config.groupby(["DeviceId", "Phase"], sort=True):
        f = set(g["Function"])
        rows.append({"DeviceId": dv, "Phase": int(ph), "arrival_on_green": "Advance" in f,
                     "split_failures": "Presence" in f, "yellow_red": "Yellow_Red" in f})
    return pd.DataFrame(rows, columns=COVERAGE_COLS)
