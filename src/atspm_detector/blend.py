"""The phase network's settings and the blend maths (`weights/blend.json`).

The phase network is the function network's own pair phase head (`funcnet.py`, three members, one shared pass).  Its
per-detector probabilities over the kept candidates (ranker p >= KEEP_MIN_TREE_P) are averaged with the pair ranker's
before the joint decoder:

    p0 := weight_lightgbm * ranker + (1 - weight_lightgbm) * network      (renormalised per detector)

`blend.json` (frozen on out-of-fold data): weight_lightgbm 0.5; the network runs at every sample length (note 120: the
fast switch and blend.json's cutoff_minutes are gone); a sample is read in pieces of at most chunk_minutes (30), every
piece up to long_minutes (120), above it long_max_chunks (4) evenly spaced pieces of a long_piece_grid (32) grid -- the
placement evaluated in note 33.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from .common import read_json

CONFIG_FILE = "blend.json"
KEEP_MIN_TREE_P = 0.01              # the network scores only candidates whose ranker probability is >= .01


def config(model_dir) -> dict | None:
    """The frozen blend settings, or None when this model folder has no network."""
    d = Path(model_dir)
    if not (d / CONFIG_FILE).exists():
        return None
    cfg = read_json(d / CONFIG_FILE)
    w = d / cfg.get("weights_file", "funcnet/manifest.json")
    if not w.exists():
        return None
    cfg["weights_path"] = w
    return cfg


def piece_plan(cfg: dict, minutes: float) -> dict:
    """`split_range` arguments for a sample this long: every 30-minute piece up to `long_minutes`, the long-sample plan
    above it."""
    long_min = cfg.get("long_minutes")
    if long_min is not None and minutes > float(long_min):
        return {"max_chunks": int(cfg["long_max_chunks"]),
                "grid": int(cfg.get("long_piece_grid", 0) or 0)}
    return {"max_chunks": 48, "grid": 0}


def streams_for(con, t0_ms: int, t1_ms: int, max_chunks: int = 48, grid: int = 0):
    """The interval streams of the pieces the network reads, built once per call.
    -> (streams {DeviceId: z}, pieces relative to the streams' origin)."""
    from . import streams as st
    pieces = st.split_range(int(t0_ms), int(t1_ms), st.CHUNK_MS, max_chunks=max_chunks, grid=grid)
    s = st.build_streams(con, int(t0_ms), int(t1_ms), windows=pieces)
    rel = st.split_range(0, int(t1_ms) - int(t0_ms), st.CHUNK_MS, max_chunks=max_chunks, grid=grid)
    return s, rel


def mix(df: pd.DataFrame, col: str, net: pd.DataFrame, weight: float) -> np.ndarray:
    """`weight` * df[col] + (1 - weight) * the network probability, renormalised per detector.  Rows the network has no
    opinion about (a candidate it did not score) keep the ranker probability."""
    key = ["DeviceId", "Detector", "cand_phase"]
    g = net.astype({"Detector": np.int64, "cand_phase": np.int64})
    left = df[key].astype({"Detector": np.int64, "cand_phase": np.int64})
    p_nn = left.merge(g, on=key, how="left").p_gru.to_numpy()
    base = np.asarray(df[col], dtype=float)
    # renormalise the network side over the candidates this frame actually carries
    gk = [df.DeviceId.to_numpy(), df.Detector.to_numpy()]
    s = pd.Series(np.where(np.isnan(p_nn), 0.0, p_nn))
    tot = s.groupby(gk, sort=False).transform("sum").to_numpy()
    ok = ~np.isnan(p_nn) & (tot > 0)
    p_nn = np.where(ok, np.where(tot > 0, p_nn / np.where(tot > 0, tot, 1.0), 0.0), np.nan)
    out = pd.Series(np.clip(np.where(np.isnan(p_nn), base, weight * base + (1.0 - weight) * p_nn), 1e-12, None))
    return (out / out.groupby(gk, sort=False).transform("sum")).to_numpy()
