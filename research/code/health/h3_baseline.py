"""Note 43: healthy baselines for the spot-check charts (sampled, saved once; charts never re-score).

Per predicted function (OOF, 66-h), 40 presumed-healthy detectors that the scorer calls ok on the
Sept-2026 66-h window: the histogram of ON->ON intervals (log bins) -> p10 / median / p90 across
the sample.  Training signals only.

    python h3_baseline.py -> %DC_WORK%/health3/ioi_baseline.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

OUT = H.DCW / "health3"
EV = H.DCW / "official" / "stg" / "cache" / "events"
BINS = np.logspace(-1, np.log10(600), 40)
N_PER = 40


def main():
    nf = pd.read_parquet(H.HB / "nofault_eval.parquet")
    f = nf[(nf.window == "full") & nf.presumed_healthy.eq(True) & nf.status.eq("ok") & (nf.n_on >= 500)]
    pf = pd.read_parquet(OUT / "oof_pf.parquet")
    pf = pf[(pf.period == "stg") & (pf.wgroup == "full")]
    f = f.merge(pf[["DeviceId", "detector", "pred_function"]], on=["DeviceId", "detector"])
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    f = f[~f.DeviceId.isin(locked)]
    rows = []
    for fn, g in f.groupby("pred_function"):
        g = g.sample(min(N_PER, len(g)), random_state=41)
        for r in g.itertuples():
            e = ds.dataset(EV / f"DeviceId={r.DeviceId}").to_table(
                filter=(ds.field("EventId") == 82) & (ds.field("Parameter") == int(r.detector))).to_pandas()
            t = np.sort(e.Timestamp.drop_duplicates().to_numpy())
            ioi = np.diff(t) / np.timedelta64(1, "s")
            h, _ = np.histogram(ioi, BINS)
            rows.append(dict(pred_function=fn, DeviceId=r.DeviceId, detector=r.detector, hist=h / max(len(ioi), 1)))
    df = pd.DataFrame(rows)
    out = []
    for fn, g in df.groupby("pred_function"):
        Hm = np.vstack(g["hist"].to_numpy())
        out.append(dict(pred_function=fn, n=len(g), lo=np.quantile(Hm, .1, axis=0), med=np.median(Hm, 0),
                        hi=np.quantile(Hm, .9, axis=0), edges=BINS))
    pd.DataFrame(out).to_parquet(OUT / "ioi_baseline.parquet")
    print(pd.DataFrame(out)[["pred_function", "n"]])


if __name__ == "__main__":
    main()
