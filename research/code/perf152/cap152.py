"""Note 152: capture the inputs of every health_v4.assess call made by predict() (package at --src) plus its outputs
(keep_all: per-detector table, signal note, the internal X and E tables) for the parity harness.

    python cap152.py <set> --src <package parent>
sets: h24      the first 100 seed-151 h24_b signals, one predict call (start / end = the day)
      h24x     the next 200 seed-151 h24_b signals, one call
      h3       40 signals (seed order 300-339), 06:00-09:00
      m30      40 signals (seed order 340-379), 16:00-16:30
      multi    the three 4-5 day logs (alone)
      sample   the bundled sample
Writes %DC_WORK%/s152/cap/<set>.pkl: list of dict(DeviceId, t, eid, par, start, end, phase, flab, func, span, pconf,
out, note, X, E)."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s152" / "cap"
DAY = "2026-09-28"
MULTI = ["ev_typical_r8_d5", "ev_busiest_ev_d5", "ev_asclog_d4"]


def order_ids():
    import numpy as np
    import pandas as pd
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not {i.lower() for i in ids} & locked
    rng = np.random.default_rng(151)
    return [ids[i] for i in rng.permutation(len(ids))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import pandas as pd
    from atspm_detector import pipeline as PP, health_v4 as hv4
    rec = []
    orig = hv4.assess

    def spy(P, start, end, phase, flab, func, span, pconf=None, **kw):
        out, note, X, E = orig(P, start, end, phase, flab, func, span, pconf, keep_all=True)
        rec.append(dict(t=P.t.copy(), eid=P.eid.copy(), par=P.par.copy(), start=pd.Timestamp(start),
                        end=pd.Timestamp(end), phase=phase, flab=flab, func=func, span=span, pconf=pconf,
                        out=out, note=note, X=X, E=E))
        return out, note
    hv4.assess = spy
    day = DCW / "s151" / "h24b_events.parquet"
    ids = order_ids()
    if a.set == "h24":
        PP.predict(str(day), device_ids=ids[:100], start=f"{DAY} 00:00", end="2026-09-29 00:00")
    elif a.set == "h24x":
        PP.predict(str(day), device_ids=ids[100:300], start=f"{DAY} 00:00", end="2026-09-29 00:00")
    elif a.set == "h3":
        PP.predict(str(day), device_ids=ids[300:340], start=f"{DAY} 06:00", end=f"{DAY} 09:00")
    elif a.set == "m30":
        PP.predict(str(day), device_ids=ids[340:380], start=f"{DAY} 16:00", end=f"{DAY} 16:30")
    elif a.set == "multi":
        for m in MULTI:
            PP.predict(pd.read_parquet(DCW / "s141p" / "data" / f"{m}.parquet"))
    elif a.set == "sample":
        PP.predict(str(Path(PP.__file__).parent / "data" / "sample_events.parquet"))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{a.set}.pkl", "wb") as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(a.set, len(rec), "assess calls captured")


if __name__ == "__main__":
    main()
