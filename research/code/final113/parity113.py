"""Note 113: parity of the package's lead-neighbour graph (v6b similarity.build_lead_window, SQL in the package) with
the training graph (f113 'lagt' from pairs113.parquet) on N staging signals x all 22 windows.

    python parity113.py <package dir> [N=12]     -> %HOME%/dc_work/s113/parity113.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PKG = Path(sys.argv[1])
NSIG = int(sys.argv[2]) if len(sys.argv) > 2 else 12
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pairs113 as PB  # noqa: E402  (re-points DC_WORK at the staging cache)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("pkg_similarity", PKG / "similarity.py")
sys.path.insert(0, str(PKG))
SM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SM)
from common import connect  # noqa: E402


def main():
    devs = PB.signals()[::max(1, 761 // NSIG)][:NSIG]
    con = connect(threads=4)
    PB.load_lean(con, devs)
    got = []
    for w in PB.bf.WINDOW_SETS["stgall"]:
        w0 = PB.bf._epoch(w)
        w1 = w0 + w["secs"]
        con.execute(f"CREATE OR REPLACE TEMP TABLE onev AS SELECT * FROM onev_all WHERE t_on >= {w0} AND t_on < {w1}")
        con.execute(f"CREATE OR REPLACE TEMP TABLE gs AS SELECT dev, greatest(t0,{w0}) AS t0, least(t1,{w1}) AS t1, mask "
                    f"FROM gs_all WHERE t1 > {w0} AND t0 < {w1}")
        got.append(SM.build_lead_window(con, w["win"]))
    g = pd.concat(got, ignore_index=True)
    g["DeviceId"] = g.DeviceId + "@stg"
    # training graph (f113.graphs 'lagt' recipe, from the float32 pair table)
    pr = pd.read_parquet(PB.HOME_WORK / "s113" / "pairs113.parquet")
    pr = pr[pr.DeviceId.isin(set(g.DeviceId))].rename(columns={"a": "Detector", "c": "other"})
    rev = pr[["DeviceId", "win", "Detector", "other", "OL", "EL"]].rename(
        columns={"Detector": "other", "other": "Detector", "OL": "OLr", "EL": "ELr"})
    pr = pr.merge(rev, on=["DeviceId", "win", "Detector", "other"], how="left")
    t = pd.DataFrame({"DeviceId": pr.DeviceId, "win": pr.win, "Detector": pr.Detector, "other": pr.other,
                      "x_ab": pr.OL.astype(float) - pr.EL.astype(float),
                      "x_ba": pr.OLr.fillna(0).astype(float) - pr.ELr.fillna(0).astype(float),
                      "na": pr.na, "nb": pr.nb})
    t = SM.lead_from_counts(t)
    k = ["DeviceId", "win", "Detector", "other"]
    m = g.merge(t, on=k, how="outer", suffixes=("_pkg", "_train"), indicator=True)
    both = m[m._merge == "both"]
    rel = (both.w_pkg - both.w_train).abs() / both.w_train.abs().clip(lower=1e-12)
    res = {"signals": len(devs), "edges_pkg": int(len(g)), "edges_train": int(len(t)),
           "only_pkg": int((m._merge == "left_only").sum()), "only_train": int((m._merge == "right_only").sum()),
           "w_max_rel_diff": float(rel.max()) if len(rel) else None, "w_p99_rel_diff": float(rel.quantile(.99))}
    print(res)
    json.dump(res, open(PB.HOME_WORK / "s113" / "parity113.json", "w"), indent=1)


if __name__ == "__main__":
    main()
