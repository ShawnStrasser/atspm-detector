"""Note 49 addendum -- does final_v3_candidate_v2 (function head = function_v3e, fed the phase BLEND `prob`)
build its function frame's phase columns the way frame v6e (its training frame) did?

    python v3e_package_verify.py [--n-stg 3 --n-dec 2 --wins m5_a,m30_a,h6_a,h24_a]

On a few non-locked training signals, per window, on the window's events only (the package's real path:
load_events, the GRU pieces exactly as predict() plans them, build_features, score, _function_frame):
A  definition parity: the package's `prob` replaced by frame v6e's own phase input (the phase_v3 blend OOF,
   `function_v3e/phase/phase_pred_v3blend_{per}.parquet`) -> _function_frame's pred_phase / top_prob must
   equal frame v6e's exactly (same argmax rule, same column).
B  end to end: the package's own blend (full-fit phase_v3, not OOF): function-frame pred_phase = the phase the
   package reports (argmax prob), and its agreement with frame v6e vs frame v6t (trees-only) pred_phase.
Locked signals asserted absent. Output: %DC_WORK%/final_v3_work/function_v3e/package_verify.json
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

PKG = V.DCW / "final_v3_candidate_v2"
sys.path.insert(0, str(PKG))
import gru_blend as gb  # noqa: E402
import predict as P  # noqa: E402
assert Path(P.__file__).resolve().parent == PKG.resolve(), P.__file__   # not note 32's package

CACHE_EV = {"stg": V.DCW / "official" / "stg" / "cache" / "events", "dec": V.DCW / "cache" / "events"}


def win_of(period: str, name: str):
    for n, t0, secs in AF.WINDOWS[period]:
        if n == name or (name == "full" and n.startswith("full")):
            return n, t0, secs
    raise KeyError(name)


def load_signal(con, period, dev, start, end):
    """One signal's cached events in [start, end) -> DataFrame."""
    d = [p for p in CACHE_EV[period].iterdir() if p.name.lower() == f"deviceid={dev}"]
    assert len(d) == 1, (period, dev)
    return con.sql(f"SELECT '{dev}' AS DeviceId, Timestamp, EventId, Parameter FROM read_parquet("
                   f"'{(d[0] / '*.parquet').as_posix()}', hive_partitioning=false) WHERE Timestamp >= "
                   f"TIMESTAMP '{start}' AND Timestamp < TIMESTAMP '{end}'").df()

W = V.DCW / "final_v3_work" / "function_v3e"
FR = {t: V.FRAMES[t][0] for t in ("v6e", "v6t")}
COLS = ["DeviceId", "Detector", "win", "period", "pred_phase", "top_prob"]
KEY = ["DeviceId", "Detector"]


def package_df(per, dev, t0, secs):
    con = P._connect(8, "6GB")
    try:
        evw = load_signal(con, per, dev, t0, t0 + pd.Timedelta(seconds=float(secs)))
        w0, w1, _ = P.load_events(con, evw)
        P.build_chunk_tables(con)
        cfg, gru = gb.config(P.DEFAULT_MODEL_DIR), None
        minutes = round((w1 - w0) / 60.0)
        if gb.runs_on(cfg, minutes):
            gru = gb.phase_probs(con, cfg["weights_path"], int(round(w0 * 1000)), int(round(w1 * 1000)),
                                 **gb.piece_plan(cfg, minutes))
        df, sim = P.build_features(con, w0, w1)
        return P.score(df, sim, P.DEFAULT_MODEL_DIR, gru, cfg)
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stg", type=int, default=3)
    ap.add_argument("--n-dec", type=int, default=2)
    ap.add_argument("--wins", default="m5_a,m30_a,h6_a,h24_a")
    a = ap.parse_args()
    lock = V.locked_signals()
    fe = pd.read_parquet(FR["v6e"], columns=COLS)
    ft = pd.read_parquet(FR["v6t"], columns=COLS)
    assert not fe.DeviceId.str.lower().isin(lock).any()
    rng = np.random.default_rng(49)
    picks = []
    for per, n in (("stg", a.n_stg), ("dec", a.n_dec)):
        s = fe[fe.period == per].groupby("DeviceId").Detector.nunique()
        s = s[(s >= 10) & (s <= 24)]
        picks += [(per, d) for d in rng.choice(sorted(s.index), n, replace=False)]
    rep, tot = {"signals": [], "windows": {}}, {"A_rows": 0, "A_pred_diff": 0, "A_top_prob_maxdiff": 0.0,
                                                 "B_rows": 0, "B_frame_eq_reported": 0, "B_eq_v6e": 0, "B_eq_v6t": 0}
    for per, dev in picks:
        assert dev.lower() not in lock
        oof = pd.read_parquet(W / "phase" / f"phase_pred_v3blend_{per}.parquet", filters=[("DeviceId", "==", dev)])
        t_sig = time.time()
        for w in a.wins.split(","):
            name, t0, secs = win_of(per, w)
            re_ = fe[(fe.period == per) & (fe.DeviceId == dev) & (fe.win == name)]
            rt = ft[(ft.period == per) & (ft.DeviceId == dev) & (ft.win == name)]
            if not len(re_):
                continue
            df = package_df(per, dev, t0, secs)
            # ---- A: the frame's own phase input through the package's _function_frame
            o = oof[oof.win == name].drop(columns="win")
            o["Detector"] = o.Detector.astype(df.Detector.dtype)
            dA = df.drop(columns="prob").merge(o, on=KEY + ["cand_phase"], how="inner")
            topA = P._function_frame(dA)
            topA["Detector"] = topA.Detector.astype(re_.Detector.dtype)
            mA = re_.merge(topA[KEY + ["pred_phase", "top_prob"]], on=KEY, suffixes=("_ref", ""))
            nA = int((mA.pred_phase != mA.pred_phase_ref).sum())
            dpA = float((mA.top_prob - mA.top_prob_ref).abs().max()) if len(mA) else 0.0
            # ---- B: the package's own blend end to end
            topB = P._function_frame(df)
            rep_ph = df.loc[df.groupby(KEY)["prob"].idxmax(), KEY + ["cand_phase"]]
            topB = topB.merge(rep_ph, on=KEY)
            topB["Detector"] = topB.Detector.astype(re_.Detector.dtype)
            mB = topB[KEY + ["pred_phase", "cand_phase"]].merge(
                re_[KEY + ["pred_phase"]], on=KEY, suffixes=("", "_v6e")).merge(
                rt[KEY + ["pred_phase"]].rename(columns={"pred_phase": "pred_phase_v6t"}), on=KEY)
            r = {"frame_rows": len(re_), "A_matched": len(mA), "A_pred_phase_diff": nA, "A_top_prob_maxdiff": dpA,
                 "B_matched": len(mB), "B_frame_eq_reported": int((mB.pred_phase == mB.cand_phase).sum()),
                 "B_eq_v6e": int((mB.pred_phase == mB.pred_phase_v6e).sum()),
                 "B_eq_v6t": int((mB.pred_phase == mB.pred_phase_v6t).sum())}
            rep["windows"][f"{per}|{dev}|{name}"] = r
            tot["A_rows"] += len(mA)
            tot["A_pred_diff"] += nA
            tot["A_top_prob_maxdiff"] = max(tot["A_top_prob_maxdiff"], dpA)
            tot["B_rows"] += len(mB)
            for k in ("B_frame_eq_reported", "B_eq_v6e", "B_eq_v6t"):
                tot[k] += r[k]
            V.log(f"{per} {dev[:8]} {name}: {r}")
        rep["signals"].append({"period": per, "DeviceId": dev, "secs": round(time.time() - t_sig, 1)})
    rep["total"] = tot
    json.dump(rep, open(W / "package_verify.json", "w"), indent=1, default=str)
    V.log(f"total {tot} -> {W / 'package_verify.json'}")


if __name__ == "__main__":
    main()
