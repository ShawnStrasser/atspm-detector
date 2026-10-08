"""Note 32 -- do the final_v3 candidate package's function features equal the research frame's?

    python v3_candidate_verify.py [--n-stg 3 --n-dec 2 --wins m30_a,h6_a,full]

Two checks on a few non-locked training signals (research frame v5 = the training design matrix):

A  definition parity: the package's `features_expert` run on package tables built from the
   signal's WHOLE cached event log (so intervals / cycles / coordination state that cross the
   window edge are known, as they were to the research builder), window = the research window.
   Expected equal to float32 precision.
B  end to end: the package's real path on the window's events only (load_events with
   start/end, build_features, score, _function_frame, features_expert) -> all 442 features vs
   the frame, on detectors whose predicted phase matches the frame's. Differences here are the
   window-edge effect every production feature has (the research frame was cut from a
   multi-day cache); reported, not asserted. Also checks no feature is missing.
Locked signals are asserted absent. Output: %DC_WORK%/final_v3_work/verify_features.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import a2_features as AF
import v3_retrain as V

PKG = V.DCW / "final_v3_candidate"
sys.path.insert(0, str(PKG))
import features_expert as fx  # noqa: E402
import predict as P  # noqa: E402

CACHE_EV = {"stg": V.DCW / "official" / "stg" / "cache" / "events",
            "dec": V.DCW / "cache" / "events"}
OUT = V.DCW / "final_v3_work"


def win_of(period: str, name: str):
    for n, t0, secs in AF.WINDOWS[period]:
        if n == name or (name == "full" and n.startswith("full")):
            return n, t0, secs
    raise KeyError(name)


def ev_path(period: str, dev: str) -> str:
    d = [p for p in CACHE_EV[period].iterdir() if p.name.lower() == f"deviceid={dev}"]
    assert len(d) == 1, (period, dev)
    return (d[0] / "*.parquet").as_posix()


def load_signal(con, period, dev, start=None, end=None):
    """The signal's cached events -> DataFrame (one signal only, a few million rows at most)."""
    q = f"SELECT '{dev}' AS DeviceId, Timestamp, EventId, Parameter FROM read_parquet('{ev_path(period, dev)}', hive_partitioning=false)"
    if start is not None:
        q += f" WHERE Timestamp >= TIMESTAMP '{start}' AND Timestamp < TIMESTAMP '{end}'"
    return con.sql(q).df()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stg", type=int, default=3)
    ap.add_argument("--n-dec", type=int, default=2)
    ap.add_argument("--wins", default="m30_a,h6_a,full")
    a = ap.parse_args()
    V.set_frame("v5")
    fr, cols = V.load_feats()
    px = [c for c in cols if c.startswith("px_")]
    lock = V.locked_signals()
    assert not fr.DeviceId.isin(lock).any()
    # a few mid-sized signals present in every requested window of their period
    rng = np.random.default_rng(32)
    picks = []
    for per, n in (("stg", a.n_stg), ("dec", a.n_dec)):
        s = fr[fr.period == per].groupby("DeviceId").Detector.nunique()
        s = s[(s >= 10) & (s <= 24)]
        picks += [(per, d) for d in rng.choice(sorted(s.index), n, replace=False)]
    report = {"signals": [], "A": {}, "B": {}}
    worstA, allB = [], []
    for per, dev in picks:
        assert dev not in lock
        con = P._connect(8, "6GB")
        t0all = time.time()
        full_ev = load_signal(con, per, dev)
        P.load_events(con, full_ev)
        P.build_chunk_tables(con)
        for w in a.wins.split(","):
            name, t0, secs = win_of(per, w)
            e0 = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
            ref = fr[(fr.period == per) & (fr.DeviceId == dev) & (fr.win == name)]
            if not len(ref):
                continue
            # ---- A: package definitions on edge-complete tables
            tgt = ref[["DeviceId", "Detector", "pred_phase"]]
            fx.prepare_tables(con, e0, e0 + secs, tgt)
            got = fx.compute(con, e0, secs)
            got = got.rename(columns={"det": "Detector"}).drop(columns=["dev", "px_n_on"])
            got["DeviceId"] = dev
            got["Detector"] = got.Detector.astype(ref.Detector.dtype)
            m = ref[["DeviceId", "Detector"] + px].merge(got, on=["DeviceId", "Detector"],
                                                          how="left", suffixes=("_ref", ""))
            resA = {}
            for c in px:
                x, y = m[c].astype(float).to_numpy(), m[f"{c}_ref"].astype(float).to_numpy()
                nn = int((np.isnan(x) != np.isnan(y)).sum())
                both = ~np.isnan(x) & ~np.isnan(y)
                rel = np.abs(x[both] - y[both]) / np.maximum(1.0, np.abs(y[both]))
                resA[c] = (float(rel.max()) if both.any() else 0.0, nn)
            mx = max(v[0] for v in resA.values())
            nnan = sum(v[1] for v in resA.values())
            worst = max(resA, key=lambda c: resA[c][0])
            report["A"][f"{per}|{dev}|{name}"] = {"rows": len(ref), "max_rel": mx,
                                                   "worst_col": worst, "nan_mismatch": nnan}
            worstA.append(mx)
            V.log(f"A {per} {dev[:8]} {name}: {len(ref)} dets, max rel {mx:.2e} ({worst}), "
                  f"NaN mismatches {nnan}")
        con.close()
        # ---- B: the real inference path on the window's events only
        for w in a.wins.split(","):
            name, t0, secs = win_of(per, w)
            ref = fr[(fr.period == per) & (fr.DeviceId == dev) & (fr.win == name)]
            if not len(ref):
                continue
            con = P._connect(8, "6GB")
            evw = load_signal(con, per, dev, t0, t0 + pd.Timedelta(seconds=float(secs)))
            w0, w1, _ = P.load_events(con, evw)
            P.build_chunk_tables(con)
            df, sim = P.build_features(con, w0, w1)
            df = P.score(df, sim, P.DEFAULT_MODEL_DIR)
            top = P._function_frame(df)
            # bins on the requested window, as predict() does when given start / end
            ex = fx.build(con, *P._bin_window(t0, t0 + pd.Timedelta(seconds=float(secs)), w0, w1), top)
            ex["Detector"] = ex.Detector.astype(top.Detector.dtype)
            top = top.merge(ex, on=["DeviceId", "Detector"], how="left")
            con.close()
            miss = [c for c in cols if c not in top.columns]
            top["Detector"] = top.Detector.astype(ref.Detector.dtype)
            m = ref[["DeviceId", "Detector", "pred_phase"] + cols].merge(
                top[["DeviceId", "Detector", "pred_phase"] + [c for c in cols if c in top]],
                on=["DeviceId", "Detector"], suffixes=("_ref", ""))
            same = m.pred_phase == m.pred_phase_ref
            mm = m[same]
            fam = {}
            for c in cols:
                if c in miss:
                    continue
                x = mm[c].astype(float).to_numpy()
                y = mm[f"{c}_ref"].astype(float).to_numpy()
                both = ~np.isnan(x) & ~np.isnan(y)
                rel = np.abs(x[both] - y[both]) / np.maximum(1.0, np.abs(y[both]))
                f = "expert" if c.startswith("px_") else "production"
                fam.setdefault(f, []).append(((rel <= 1e-3).mean() if both.any() else 1.0,
                                              int((np.isnan(x) != np.isnan(y)).sum())))
            r = {"rows": len(ref), "matched": int(len(m)), "same_pred_phase": int(same.sum()),
                 "missing_features": miss}
            for f, v in fam.items():
                r[f"{f}_mean_share_within_1e-3"] = round(float(np.mean([x[0] for x in v])), 4)
                r[f"{f}_nan_mismatch"] = int(sum(x[1] for x in v))
            report["B"][f"{per}|{dev}|{name}"] = r
            allB.append(r)
            V.log(f"B {per} {dev[:8]} {name}: {r}")
        report["signals"].append({"period": per, "DeviceId": dev,
                                  "secs": round(time.time() - t0all, 1)})
    report["A_max_rel_overall"] = max(worstA) if worstA else None
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT / "verify_features.json", "w"), indent=1, default=str)
    V.log(f"A overall max rel diff {report['A_max_rel_overall']:.2e} -> {OUT / 'verify_features.json'}")


if __name__ == "__main__":
    main()
