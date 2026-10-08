"""Note 93 A: the PACKAGE health output (final_v3_candidate_v4f/health_core.py, called exactly as predict.post_outputs
calls it) on the same >= 30 min windows the six-fold OOF was scored on.

Inputs per window = the v4f OOF answers of that window: phase = TCN-blend decode top (s93/phase_rows) where the phase pool
covers the row, else the frame's pred_phase; phase_conf = its probability; function = the v4f stacker probabilities;
lanes spanned = number of lanes in the frame's lane string.  Events = the frame caches (dec / stg), allowed codes only,
de-duplicated.  Training signals only (locked_v2 asserted absent).  CPU, --procs <= 4.

    python h93_health.py [--procs 4] [--limit N]   -> %DC_WORK%/s93/health_rows.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
PKG = DCW / "final_v3_candidate_v4f"
OUT = DCW / "s93"
EVR = {"stg": DCW / "official" / "stg" / "cache" / "events", "dec": DCW / "cache" / "events"}
CODE = Path(__file__).resolve().parents[1]
FUNC_CLASSES = ["Advance", "Presence", "Count", "Yellow_Red", "Other", "Mid", "Bike"]
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
HC = None


def _hc():
    global HC
    if HC is None:
        spec = importlib.util.spec_from_file_location("health_core_v4f", PKG / "health_core.py")
        HC = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(HC)
    return HC


def windows() -> dict:
    sys.path.insert(0, str(CODE))
    import rpath  # noqa: F401
    import a2_features as A2F
    return {p: {n: (t0, s) for n, t0, s in A2F.WINDOWS[p] if s >= 1800} for p in A2F.WINDOWS}


def n_lanes(s) -> float:
    if not isinstance(s, str) or not s.strip():
        return np.nan
    return float(len([x for x in s.replace(";", ",").replace("|", ",").split(",") if x.strip()]))


def work(job):
    dev, period, wins, g = job
    hc = _hc()
    import pyarrow.dataset as ds
    warnings.filterwarnings("ignore")
    p = EVR[period] / f"DeviceId={dev}"
    if not p.is_dir():
        return pd.DataFrame(), f"no events {dev} {period}"
    ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    ev = ev[["Timestamp", "EventId", "Parameter"]].astype({"EventId": int, "Parameter": int}).drop_duplicates()
    rows, msg = [], ""
    for name, w in g.groupby("win"):
        t0, secs = wins[name]
        start, end = pd.Timestamp(t0), pd.Timestamp(t0) + pd.Timedelta(seconds=secs)
        e = ev[(ev.Timestamp >= start) & (ev.Timestamp < end)]
        phase = {int(d): float(v) for d, v in zip(w.Detector, w.ph) if pd.notna(v)}
        pconf = {int(d): float(v) for d, v in zip(w.Detector, w.ph_p) if pd.notna(v)}
        func = {int(r.Detector): {c: float(getattr(r, f"p_{c}")) for c in FUNC_CLASSES}
                for r in w.itertuples() if pd.notna(getattr(r, "p_Advance"))}
        span = {int(d): float(v) for d, v in zip(w.Detector, w.nl) if pd.notna(v) and v > 0}
        try:
            h = hc.health(e, start, end, None, phase or None, func or None, span or None, pconf or None)
        except Exception as exc:  # noqa: BLE001
            msg += f"health failed {dev} {period} {name}: {type(exc).__name__} {exc}; "
            continue
        if not len(h):
            continue
        keep = ["detector", "health_score", "status", "reason", "n_families"] + [c for c in h.columns if c.startswith("s_")]
        h = h[[c for c in keep if c in h.columns]].copy()
        h.insert(0, "win", name)
        h.insert(0, "period", period)
        h.insert(0, "DeviceId", dev)
        rows.append(h)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    fr = pd.read_parquet(OUT / "func_rows.parquet")
    fr = fr[fr.wgroup.isin(GE30)].copy()
    ph = pd.read_parquet(OUT / "phase_rows.parquet", columns=["DeviceId", "Detector", "win", "pred_phase_v4f",
                                                             "p_phase_v4f"])
    ph["period"] = np.where(ph.DeviceId.str.endswith("@stg"), "stg", "dec")
    ph["dkey"] = ph.DeviceId.str.replace("@stg", "", regex=False).str.lower()
    fr["dkey"] = fr.DeviceId.str.lower()
    fr = fr.merge(ph.drop(columns="DeviceId").astype({"Detector": fr.Detector.dtype}),
                  on=["dkey", "period", "Detector", "win"], how="left")
    fr["ph"] = fr.pred_phase_v4f.where(fr.pred_phase_v4f.notna(), fr.pred_phase)
    fr["ph_p"] = fr.p_phase_v4f
    fr["nl"] = fr.lanes5g.map(n_lanes)
    lk = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not fr.dkey.isin(lk).any()
    W = windows()
    cols = ["Detector", "win", "ph", "ph_p", "nl"] + [f"p_{c}" for c in FUNC_CLASSES]
    jobs = [(dev, per, W[per], g[cols].copy()) for (dev, per), g in fr.groupby(["DeviceId", "period"])]
    jobs.sort(key=lambda j: -len(j[3]))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} signal-periods, {len(fr):,} detector-windows; package {PKG.name}", flush=True)
    out, t0 = [], time.time()
    with Pool(a.procs) as pool:
        for i, (h, msg) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            if len(h):
                out.append(h)
            if msg:
                print(msg, flush=True)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    H = pd.concat(out, ignore_index=True)
    H.to_parquet(OUT / ("health_rows.parquet" if not a.limit else "health_rows_smoke.parquet"), index=False)
    print(f"health rows {len(H):,} {H.status.value_counts().to_dict()} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
