"""Note 118b: health v4 = one research scorer.

v4 = v110 rules (h110_resolve, every F-fix)
   + note-118a fixes G1-G4 (stuck every ON / extension-time note C1 / not-really-pulse note C2 / erratic counts as
     share outside the 15-min expected range / count-drop evidence)                       (h118_resolve.run)
   + note-117 traffic-aware checks replacing the old ones:
       'too-fast actuations' = fast ONs beyond random arrivals at the robust local rate per color state (zf) or
       too many 5-min burst bins;  'too many for the traffic' = busiest 5-min green flow per lane vs
       max(healthy p99.8, 1800 veh/h/lane) (Advance: all time)                            (s117/stats117.parquet)
   + note-116 time-of-day check 'busy at night' (night 01-05 level vs busiest 4 h, per type, phase-mate excuse)
     replacing the old time-of-day profile (24 h) AND its 3-h watch                        (s116/oof_final.parquet)
     'goes silent' wins over 'busy at night' (a detector silent from 04:00 has its busiest 4 h at night).

Scored on every w40 detector-window (763 training signals, locked_v2 absent - asserted in the upstream passes); the
limits are population quantiles, so the population is needed even when only the review signals are reported.

    python score_v4.py   -> %DC_WORK%/s118b/{resolved_v4.parquet, review_v4.parquet, rates_v4.csv, cats_v4.csv}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "health"))
sys.path.insert(0, str(HERE))
import h118_resolve as R18  # noqa: E402
import h108_resolve as R8  # noqa: E402

DCW = R18.DCW
OUT = DCW / "s118b"
KEY = R18.KEY
PROF_LEVEL = {"suspect": 0.50, "bad": 0.80}          # status from the note-116 rule itself (resolver: suspect >.30, bad >.75)

# the explicit list of health categories v4 reports: key -> (plain name, kind)
CATEGORIES = {
    "stuck": ("Stuck on", "finding"),
    "dropout": ("Goes silent", "finding"),
    "level": ("Count drops", "finding"),
    "night_drop": ("Misses vehicles at night", "finding"),
    "choppy": ("Erratic counts", "finding"),
    "rapid": ("Too-fast actuations", "finding"),
    "volume": ("Too many for the traffic", "finding"),
    "chatter": ("Chattering", "finding"),
    "occspk": ("Erratic time ON", "finding"),
    "prof": ("Busy at night", "finding"),
    "occ_hi": ("ON longer than its kind", "watch"),
    "C1": ("Extension time on a count zone", "config note"),
    "C2": ("Set to pulse but holds ON", "config note"),
}


def load_117():
    S = pd.read_parquet(DCW / "s117" / "stats117.parquet",
                        columns=KEY + ["n_on", "new_fast_x", "new_vol_x", "zf", "lim_zf", "n_spk", "lim_n_spk",
                                       "fo_all", "fem_all", "q5_gy", "q5_all", "lim_vol", "ln"])
    return S.rename(columns={"n_on": "n_on117"})


def load_116():
    T = pd.read_parquet(DCW / "s116" / "oof_final.parquet",
                        columns=["DeviceId", "window", "detector", "z", "z_cnt", "z_occ", "r", "exp_r", "night_ratio",
                                 "mate_r", "dph", "n_night", "flag", "level", "group"])
    return T.rename(columns={c: f"tod_{c}" for c in T.columns if c not in KEY})


def make_scores_v4(G, T3):
    base = R18._make_scores_orig(G, T3)

    def scores_v4(X, E, fixes):
        S = base(X, E, fixes)
        enough = X.n_on117.fillna(0).to_numpy() >= 50
        S["rapid"] = np.where(enough & np.isfinite(X.new_fast_x), R8.score(X.new_fast_x, 1.0, 2.0), np.nan)
        S["volume"] = np.where(np.isfinite(X.new_vol_x), R8.score(X.new_vol_x, 1.0, 2.0), np.nan)
        lv = X.tod_level.fillna("not scored")
        pv = np.where(lv.eq("not scored"), np.nan, lv.map(PROF_LEVEL).fillna(0.0))
        silent = pd.to_numeric(S.get("dropout"), errors="coerce").fillna(0).to_numpy() >= .35
        S["prof"] = np.where(silent & np.isfinite(pv), 0.0, pv)
        return S
    return scores_v4


def run_v4(X0, E):
    X = X0.copy()
    X["prof3_x"] = np.nan                      # the 3-h profile watch is replaced (the new check needs >= 10 h)
    R18._make_scores_orig = getattr(R18, "_make_scores_orig", R18.make_scores)
    R18.make_scores = make_scores_v4
    try:
        R = R18.run(X, E, set(R18.GALL))
    finally:
        R18.make_scores = R18._make_scores_orig
    return R


def categories_of(r):
    """categories of one resolved row: findings (left8), watch occ_hi, config notes."""
    out = [k for k in str(r.left8).split(",") if k and k in CATEGORIES]
    if "occ_hi" in str(r.watch8).split(","):
        out.append("occ_hi")
    out += [c for c in str(r.cfg).split(",") if c in CATEGORIES]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X0, E = R18.prepare()
    D = pd.read_csv(DCW / "s118" / "drops118.csv", usecols=KEY + ["g4_unscored0"])
    X0 = X0.merge(D, on=KEY, how="left")
    X0["g4_unscored0"] = X0.g4_unscored0.fillna(False).astype(bool)
    X0 = X0.merge(load_117(), on=KEY, how="left").merge(load_116(), on=KEY, how="left")
    R = run_v4(X0, E)
    R["cats"] = [",".join(categories_of(r)) for r in R.itertuples()]
    V = pd.read_parquet(DCW / "s118" / "resolved118.parquet", columns=KEY + ["st8", "left8", "watch8", "cfg",
                                                                            "stuck_eps", "n_eps", "err_txt",
                                                                            "drop_at", "own_before", "own_after",
                                                                            "exp_after"])
    R = R.merge(V.rename(columns={"st8": "st_v118", "left8": "left_v118", "watch8": "watch_v118", "cfg": "cfg_v118"}),
                on=KEY, how="left")
    keep = [c for c in R.columns if not c.startswith("lsrc_") and R[c].dtype != object or c in (
        "DeviceId", "window", "fn", "type", "span", "band", "cmode", "wg", "st8", "left8", "watch8", "rules8",
        "cleared8", "dq8", "cfg", "cats", "st_v118", "left_v118", "watch_v118", "cfg_v118", "stuck_eps",
        "err_txt", "alt_fns", "tod_level", "tod_group", "reason", "drop_at", "night_ref", "ref110", "ref110_dets")]
    R[keep].to_parquet(OUT / "resolved_v4.parquet")
    # rates per 100 detector-windows
    FL = ("suspect", "bad")
    rows = []
    for wg, x in R.groupby("wg"):
        d = dict(window=wg, n=len(x), flagged_v118=100 * x.st_v118.isin(FL).mean(), flagged_v4=100 * x.st8.isin(FL).mean(),
                 bad_v118=100 * x.st_v118.eq("bad").mean(), bad_v4=100 * x.st8.eq("bad").mean(),
                 watch_v4=100 * x.st8.eq("watch").mean(),
                 newly=100 * (x.st8.isin(FL) & ~x.st_v118.isin(FL)).mean(),
                 cleared=100 * (~x.st8.isin(FL) & x.st_v118.isin(FL)).mean())
        for k in CATEGORIES:
            d[f"cat_{k}"] = 100 * x.cats.str.split(",").apply(lambda l, k=k: k in l).mean()
        rows.append(d)
    T = pd.DataFrame(rows).round(3)
    T.to_csv(OUT / "rates_v4.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(T.T.to_string())
    # review signals
    rv = pd.read_csv(DCW / "health110" / "review" / "rows_v110.csv")[["signal", "dev"]].drop_duplicates()
    Rv = R[keep].merge(rv.rename(columns={"dev": "DeviceId"}), on="DeviceId")
    Rv.to_parquet(OUT / "review_v4.parquet")
    C = Rv[Rv.cats.ne("")].assign(cat=Rv.cats.str.split(",")).explode("cat")
    C = C.groupby(["cat", "wg"]).size().unstack(fill_value=0)
    C.to_csv(OUT / "cats_v4.csv")
    print(C.to_string())
    print(len(Rv), "review detector-windows on", Rv.signal.nunique(), "signals")


if __name__ == "__main__":
    main()
