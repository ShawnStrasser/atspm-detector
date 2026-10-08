"""Lanes redo (note 17) -- shared helpers: paths, locked-signal filter, lane text.

Lane text: some agencies name the lane in the channel description ("Loop 1 - RL Advance",
"Rad C - CL Presence").  RL / CL / LL is a lane label the model never sees; it is used
here for VALIDATION only.  The same table layout (DeviceId, target, lane) is what a
future per-(signal, phase) n_lanes label table plugs into (`load_nlanes_labels`).
"""
from __future__ import annotations
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
# repository root: DC_REPO, else three levels up when run from research/code/trackA
REPO = Path(os.environ.get("DC_REPO") or Path(__file__).resolve().parents[3])
WORK = DCW / "trackA"
LABELS = REPO / "research" / "labels" / "function_labels_v2.parquet"
PLANS = REPO / "data" / "detector_plans.parquet"
OFFICIAL = DCW / "official" / "labels_official.parquet"
# optional, future: per-(signal, phase) lane counts read off cabinet prints.
# columns DeviceId, target ("P2" / "O1"), n_lanes.  Validation only, never an input.
NLANES_LABELS = WORK / "nlanes_labels.parquet"
APC = ["Advance", "Presence", "Count"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=6")
    con.execute("SET preserve_insertion_order=false")
    (DCW / "tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


def locked_ids() -> set[str]:
    a = pd.read_csv(DCW / "data" / "splits" / "test_config.csv").DeviceId
    b = pd.read_csv(DCW / "official" / "newtest_signals.csv").DeviceId
    return set(a.str.lower()) | set(b.str.lower())


def lane_token(desc: pd.Series) -> pd.Series:
    """RL / CL / LL as a standalone token; NA when absent or ambiguous (two tokens)."""
    d = desc.astype(str).str.upper().str.replace("%20", " ", regex=False)
    hits = pd.DataFrame({t: d.str.contains(rf"\b{t}\b", regex=True, na=False)
                         for t in ("RL", "CL", "LL")})
    out = pd.Series(pd.NA, index=desc.index, dtype="object")
    one = hits.sum(axis=1) == 1
    out[one] = hits[one].idxmax(axis=1)
    return out


def unit_token(desc: pd.Series) -> pd.Series:
    """The sensor unit the channel comes from ("Rad A", "Cam C", "Matrix Rad F") --
    one unit watches one APPROACH (all its lanes), not one lane."""
    d = desc.astype(str).str.upper().str.replace("%20", " ", regex=False)
    return d.str.extract(r"\b((?:RAD|CAM)\s*[A-I])\b")[0].str.replace(r"\s+", " ",
                                                                     regex=True)


def labelled_detectors() -> pd.DataFrame:
    """Unlocked, non-dropped function labels joined to the official timing target
    (phase truth: call_phase, else call_overlap) and the plan's channel text."""
    lab = pd.read_parquet(LABELS)
    lab["DeviceId"] = lab.DeviceId.str.lower()
    lab = lab[lab.func5.notna() & ~lab.DeviceId.isin(locked_ids())]
    off = pd.read_parquet(OFFICIAL)[["DeviceId", "Detector", "target", "real_dec2024",
                                     "real_staging"]]
    off["DeviceId"] = off.DeviceId.str.lower()
    pl = pd.read_parquet(PLANS)[["DeviceId", "Detector", "description"]].rename(
        columns={"description": "plan_desc"})
    pl["DeviceId"] = pl.DeviceId.str.lower()
    d = (lab.merge(off.drop_duplicates(["DeviceId", "Detector"]),
                   on=["DeviceId", "Detector"], how="left")
            .merge(pl.drop_duplicates(["DeviceId", "Detector"]),
                   on=["DeviceId", "Detector"], how="left"))
    d["lane"] = lane_token(d.plan_desc)
    d["unit"] = unit_token(d.plan_desc)
    d["real"] = d.real_dec2024.fillna(False) | d.real_staging.fillna(False)
    return d


def load_nlanes_labels() -> pd.DataFrame | None:
    if NLANES_LABELS.exists():
        t = pd.read_parquet(NLANES_LABELS)
        t["DeviceId"] = t.DeviceId.str.lower()
        return t[~t.DeviceId.isin(locked_ids())]
    return None
