"""Note 118a: per-detector flags for the review-sample signals (health_review_v3 rows) and the user's rows whose
outcome changed v110 -> v118.  Reads %DC_WORK%/s118/resolved118.parquet (h118_resolve).

    python h118_report.py   -> %DC_WORK%/s118/flags_review118.csv, user_rows118.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s118"
KEY = ["DeviceId", "window", "detector"]
FN = {"Yellow_Red": "Yellow-red"}
CHK = {"stuck": "stuck on", "dropout": "goes silent", "level": "count drops", "choppy": "erratic counts",
       "chatter": "chattering", "rapid": "too-fast actuations", "volume": "too many in 5 min",
       "occspk": "erratic time ON", "prof": "time-of-day profile", "night_drop": "misses vehicles at night"}
CFG = {"C1": "config: ONs logged again with no OFF (extension time) - not expected on a stop-bar count zone",
       "C2": "config: holds ON although set to pulse - not really pulse"}


def det_label(r):
    ph = f"P{int(r.phase)} " if np.isfinite(r.phase) else ""
    return f"det {int(r.detector)}: {ph}{FN.get(r.fn, r.fn)}"


def findings(left):
    return ", ".join(CHK.get(k, k) for k in str(left).split(",") if k)


def main():
    A = pd.read_parquet(OUT / "resolved118.parquet")
    rows = pd.read_csv(DCW / "health110" / "review" / "review_rows110.csv")
    sig = rows[["signal", "dev", "window"]].drop_duplicates().rename(columns={"dev": "DeviceId"})
    F = A.merge(sig, on=["DeviceId", "window"])
    F["det"] = [det_label(r) for r in F.itertuples()]
    F["before"] = [f"{s}" + (f" ({findings(l)})" if l else "") for s, l in zip(F.st_v110, F.left_v110)]
    F["after"] = [f"{s}" + (f" ({findings(l)})" if l else "") for s, l in zip(F.st8, F.left8)]
    F["notes"] = [", ".join(CFG[c] for c in str(c_).split(",") if c) for c_ in F.cfg.fillna("")]
    F["drop"] = [f"dropped {t:%a %H:%M}: {b:.0f} per 15 min before, {a:.0f} after, ~{e:.0f} expected after"
                 if s >= .35 and pd.notna(t) else "" for t, b, a, e, s in
                 zip(F.drop_at, F.own_before, F.own_after, F.exp_after, F.s8_level.fillna(0))]
    cols = ["signal", "window", "det", "before", "after", "watch8", "notes", "stuck_eps", "err_txt", "drop"]
    F = F.sort_values(["signal", "window", "detector"])
    F[cols].rename(columns={"watch8": "watch_reason", "stuck_eps": "stuck_ons", "err_txt": "erratic_evidence"}).to_csv(
        OUT / "flags_review118.csv", index=False)
    # user rows + the extra detectors the user named
    extra = [("11021", "h3_b", d) for d in (3, 4)] + [("04035", "h3_b", 52), ("12032", "h24_b", 6)]
    U = rows[["n", "signal", "window", "detector"]].copy()
    U = pd.concat([U, pd.DataFrame([dict(n=np.nan, signal=s, window=w, detector=d) for s, w, d in extra])])
    U = U.merge(F[["signal", "window", "detector", "det", "before", "after", "notes", "stuck_eps", "err_txt", "drop",
                   "st_v110", "st8", "left_v110", "left8", "cfg"]], on=["signal", "window", "detector"], how="left")
    U["changed"] = (U.st_v110 != U.st8) | (U.left_v110.fillna("") != U.left8.fillna("")) | U.cfg.fillna("").ne("")
    U.to_csv(OUT / "user_rows118.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(U[["n", "signal", "det", "before", "after", "notes", "changed"]].to_string(index=False))
    print(len(F), "detector-windows on", F.signal.nunique(), "signals ->", OUT / "flags_review118.csv")


if __name__ == "__main__":
    main()
