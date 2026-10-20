"""Note 153 dev harness: load the note-152 captured assess inputs (per signal: cleaned events, window, the classifier's
answers, the numpy outputs and internal X / E tables) into DuckDB tables and compare a CTE of the SQL health with them.

    python dev153.py <set> [--n N] [--cte NAME] [--cols a,b,c]
sets: sample, m30, h3, h24, h24x, multi (the cap152 pickles in %DC_WORK%/s152/cap)."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
CAP = DCW / "s152" / "cap"
SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC))
from atspm_detector import health_duckdb as HQ  # noqa: E402

PC = ["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other", "p_mid", "p_bike"]
CL = ["Advance", "Presence", "Count", "Yellow_Red", "Other", "Mid", "Bike"]


def load(name, n=None):
    with open(CAP / f"{name}.pkl", "rb") as f:
        rec = pickle.load(f)
    return rec[:n] if n else rec


def tables(rec):
    """events / detectors / windows frames for the records (DeviceId S000, S001, ...)."""
    ev, de, wi = [], [], []
    for i, r in enumerate(rec):
        dev = f"S{i:03d}"
        s = int(r["start"].value)
        ts = pd.to_datetime(s + np.rint(np.asarray(r["t"]) * 1e9).astype(np.int64), unit="ns")
        ev.append(pd.DataFrame({"DeviceId": dev, "Timestamp": ts, "EventId": np.asarray(r["eid"], np.int32),
                                "Parameter": np.asarray(r["par"], np.int32)}))
        keys = set(r["phase"]) | set(r["flab"]) | set(r["func"] or {}) | set(r["span"] or {})
        rows = []
        for k in sorted(keys):
            pr = (r["func"] or {}).get(k) or {}
            rows.append(dict(DeviceId=dev, Detector=int(k), phase=r["phase"].get(k, np.nan),
                             function=r["flab"].get(k), lanes=(r["span"] or {}).get(k, np.nan),
                             **{p: pr.get(c, np.nan) for p, c in zip(PC, CL)}))
        de.append(pd.DataFrame(rows))
        wi.append((dev, s, int(r["end"].value)))
    return (pd.concat(ev, ignore_index=True), pd.concat(de, ignore_index=True),
            pd.DataFrame(wi, columns=["DeviceId", "s_ns", "e_ns"]))


def connect(rec, threads=8):
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='10GB'")
    ev, de, wi = tables(rec)
    con.register("ev_df", ev)
    con.register("de_df", de)
    con.register("wi_df", wi)
    con.execute("CREATE TABLE events AS SELECT * FROM ev_df")
    con.execute("CREATE TABLE detectors AS SELECT * FROM de_df")
    con.execute("CREATE TABLE windows AS SELECT * FROM wi_df")
    return con


def expected(rec):
    """the captured X tables with DeviceId (detector numbers as they are: one signal per capture)."""
    out = []
    for i, r in enumerate(rec):
        X = r["X"].copy()
        X.insert(0, "DeviceId", f"S{i:03d}")
        out.append(X)
    return pd.concat(out, ignore_index=True)


def cte(con, name, windows="windows"):
    q = HQ.build(events="events", detectors="detectors", windows=windows, upto=name)
    return con.sql(q.text(f"SELECT x.*, w.DeviceId, x.det AS Detector FROM {name} x JOIN win w USING (dev)")).df()


def compare(got, exp, cols, key=("DeviceId", "Detector"), rtol=1e-6):
    m = exp[list(key) + cols].merge(got[list(key) + cols], on=list(key), how="left", suffixes=("", "_sql"))
    bad = {}
    for c in cols:
        a, b = m[c], m[c + "_sql"]
        num = (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
               and not pd.api.types.is_bool_dtype(a) and not pd.api.types.is_bool_dtype(b))
        if not num:
            eq = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        else:
            a, b = a.astype(float), b.astype(float)
            eq = (np.isclose(a, b, rtol=rtol, atol=1e-9)) | (a.isna() & b.isna())
        nb = int((~eq).sum())
        if nb:
            bad[c] = m.loc[~eq, list(key) + [c, c + "_sql"]].head(8)
        print(f"{c:16s} differ {nb:5d} / {len(m)}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--cte", default="st1")
    ap.add_argument("--cols", default="")
    a = ap.parse_args()
    rec = load(a.set, a.n)
    con = connect(rec)
    t0 = time.perf_counter()
    got = cte(con, a.cte)
    print(f"{a.cte}: {len(got)} rows in {time.perf_counter() - t0:.2f}s")
    if a.cols:
        got = got.rename(columns={"status": "status2", "status1": "status", "hs": "health_score"})
        exp = expected(rec).rename(columns={"detector": "Detector"})
        bad = compare(got, exp, a.cols.split(","))
        for c, d in bad.items():
            print(d.to_string())


if __name__ == "__main__":
    main()
