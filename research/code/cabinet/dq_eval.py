"""Run the DQ checks (dq_core.py) on config-labelled detectors of ~20 unlocked training
signals (the 10 pilot signals + 10 drawn with bike / mid loops) and report how often each
check passes on presumably-good detectors.  Location is proxied from the config function:
Advance -> advance, Presence / Count / Yellow_Red -> stopbar, Other: bike / mid from the
config subtype or channel text, else other.  lanes_spanned of a mid loop = number of
Advance loops on its phase (>= 2), else 1.  lane_index unknown (NA) -> best-partner pairing.

    python dq_eval.py [--period auto|stg|dec] [--n-extra 10]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import dq_core as Q

REPO = Q.Path(__file__).resolve().parents[3]
PILOTS = ["01001", "01005", "01006", "01007", "01009", "01010", "01011", "01014", "01015", "01017"]


def locked() -> set[str]:
    a = pd.read_csv(Q.DCW / "data" / "splits" / "test_config.csv").DeviceId
    b = pd.read_csv(Q.DCW / "official" / "newtest_signals.csv").DeviceId
    return set(a.str.lower()) | set(b.str.lower())


def build_input(n_extra: int, seed: int = 0) -> pd.DataFrame:
    lab = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v2.parquet")
    lab["dev"] = lab.DeviceId.str.lower()
    lab = lab[~lab.dev.isin(locked()) & lab.func5.notna() & ~lab.drop_from_use.fillna(False)]
    cf = lab.config_function.fillna("").str.lower()
    ds = lab.description.fillna("").str.replace("%20", " ").str.lower()
    loc = lab.func5.map({"Advance": "advance", "Presence": "stopbar", "Count": "stopbar",
                         "Yellow_Red": "stopbar"}).fillna("other")
    loc[(lab.func5 == "Other") & (cf.str.contains("bike") | ds.str.contains("bike"))] = "bike"
    loc[(lab.func5 == "Other") & cf.str.contains("mid")] = "mid"
    lab["location"] = loc
    has = lab.groupby("DeviceName").location.agg(lambda s: (("bike" in set(s)), ("mid" in set(s))))
    rng = np.random.default_rng(seed)
    pool_b = sorted(n for n, (b, m) in has.items() if b and n not in PILOTS)
    pool_m = sorted(n for n, (b, m) in has.items() if m and not b and n not in PILOTS)
    extra = list(rng.choice(pool_b, n_extra // 2, replace=False)) + \
        list(rng.choice(pool_m, n_extra - n_extra // 2, replace=False))
    d = lab[lab.DeviceName.isin(PILOTS + extra)].copy()
    nadv = d[d.location == "advance"].groupby(["DeviceId", "cfg_phase"]).size()
    d["lanes_spanned"] = [max(1, nadv.get((r.DeviceId, r.cfg_phase), 1)) if r.location == "mid" else 1
                          for r in d.itertuples()]
    d["lane_index"] = pd.NA
    d["technology"] = "loop"
    d["function"] = d.func5
    return d.rename(columns={"Detector": "detector", "cfg_phase": "phase"})[
        ["DeviceId", "DeviceName", "detector", "phase", "location", "lane_index", "lanes_spanned",
         "technology", "function", "func5", "config_function", "description"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="auto")
    ap.add_argument("--n-extra", type=int, default=10)
    a = ap.parse_args()
    d = build_input(a.n_extra)
    t = time.time()
    out = Q.run(d.drop(columns=["DeviceName", "func5", "config_function", "description"]), a.period)
    pr = out.attrs["pairs"]
    out = out.merge(d[["DeviceId", "detector", "DeviceName", "func5", "config_function", "description"]],
                    on=["DeviceId", "detector"])
    wd = Q.DCW / "cabinet"
    wd.mkdir(exist_ok=True)
    out.to_parquet(wd / f"dq_eval_{a.period}.parquet", index=False)
    if len(pr):
        pr["a"] = pr.a.astype(str)
        pr.to_parquet(wd / f"dq_eval_{a.period}_pairs.parquet", index=False)
    Q.log(f"{out.DeviceId.nunique()} signals, {len(out)} detectors, {time.time() - t:.0f}s "
          f"(per signal max {out.groupby('DeviceId').secs.first().max()}s)")
    good = out[~out.config_function.fillna("").str.lower().str.contains("broken")]
    rows = []
    for c in ["pair_sb", "span", "order", "excl", "sat", "bike", "dead", "health"]:
        s = good[f"s_{c}"]
        m = s.notna() & ~((c == "order") & (good.location == "advance"))
        rows.append({"check": c, "n": int(m.sum()), "pass(>=.5)": (s[m] >= 0.5).mean().round(3),
                     "mean": s[m].mean().round(3)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("suspect rate by location:\n",
          good.groupby("location").suspect_config_or_health.agg(["size", "mean"]).round(3))
    print("dq_score quantiles:", good.dq_score.quantile([.05, .1, .25, .5]).round(3).to_dict())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(out[out["flags"] != ""][["DeviceName", "detector", "phase", "location", "func5", "dq_score",
                                "reasons"]].to_string(index=False))


if __name__ == "__main__":
    main()
