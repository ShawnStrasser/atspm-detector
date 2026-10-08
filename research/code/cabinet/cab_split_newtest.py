"""User decision 2026-09-24: move a RANDOM HALF of the 143 NEWTEST signals into training / dev; the other half
stays locked with the 43 TEST signals (note 36).

    python cab_split_newtest.py            writes the two files below (refuses to overwrite a different split)

Seed-fixed (SEED) and stratified by the DeviceName region prefix (first two characters), so both halves hold the
same regions in the same proportions: inside each region the signals are sorted by DeviceId, shuffled with the seed,
and the first floor(n/2) are released; for a region with an odd count one extra coin (same generator) decides
whether the odd signal is released. No label, prediction or metric is read: the draw uses names and ids only.

Outputs (the old split files are never edited):
  %DC_WORK%/official/newtest_released.csv   DeviceId, DeviceName, region, set = NEWTEST_released
  %DC_WORK%/official/locked_v2.csv          DeviceId, DeviceName, region, set = TEST | NEWTEST (the new hold-out)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from cab_common import DC_WORK, signals

SEED = 20260924
OFF = DC_WORK / "official"
REL, LOCKED_V2 = OFF / "newtest_released.csv", OFF / "locked_v2.csv"


def draw() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = signals()[["DeviceId", "DeviceName"]].assign(DeviceId=lambda d: d.DeviceId.str.lower()).drop_duplicates()
    nt = pd.read_csv(OFF / "newtest_signals.csv")[["DeviceId"]].assign(DeviceId=lambda d: d.DeviceId.str.lower())
    te = pd.DataFrame({"DeviceId": sorted(pd.read_csv(DC_WORK / "data/splits/test_config.csv")
                                          .DeviceId.str.lower().unique())})
    nt, te = nt.merge(s, on="DeviceId", how="left"), te.merge(s, on="DeviceId", how="left")
    assert nt.DeviceName.notna().all() and te.DeviceName.notna().all()
    assert len(nt) == 143 and len(te) == 43 and not set(nt.DeviceId) & set(te.DeviceId)
    for d in (nt, te):
        d["region"] = d.DeviceName.str[:2]
    rng = np.random.default_rng(SEED)
    rel = []
    for reg, g in nt.sort_values("DeviceId").groupby("region", sort=True):
        ids = g.DeviceId.to_numpy()[rng.permutation(len(g))]
        k = len(g) // 2 + (int(rng.integers(0, 2)) if len(g) % 2 else 0)
        rel += list(ids[:k])
    nt["released"] = nt.DeviceId.isin(rel)
    released = nt[nt.released].assign(set="NEWTEST_released")[["DeviceId", "DeviceName", "region", "set"]]
    locked = pd.concat([te.assign(set="TEST"), nt[~nt.released].assign(set="NEWTEST")], ignore_index=True)
    locked = locked[["DeviceId", "DeviceName", "region", "set"]]
    return (released.sort_values("DeviceName").reset_index(drop=True),
            locked.sort_values(["set", "DeviceName"]).reset_index(drop=True))


def main():
    rel, lk = draw()
    for f, d in ((REL, rel), (LOCKED_V2, lk)):
        if f.exists():
            old = pd.read_csv(f)
            if not old[list(d.columns)].astype(str).equals(d.astype(str)):
                sys.exit(f"{f} exists with a different split: refusing to overwrite")
        d.to_csv(f, index=False)
    assert not set(rel.DeviceId) & set(lk.DeviceId) and len(rel) + len(lk) == 186
    tab = pd.crosstab(pd.concat([rel.region, lk[lk.set == "NEWTEST"].region]),
                      ["released"] * len(rel) + ["locked"] * int((lk.set == "NEWTEST").sum()))
    print(f"released {len(rel)} | locked_v2 {len(lk)} = TEST {int((lk.set == 'TEST').sum())} + "
          f"NEWTEST {int((lk.set == 'NEWTEST').sum())}")
    print(tab.T.to_string())


if __name__ == "__main__":
    main()
