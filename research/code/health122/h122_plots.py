"""Note 122: charts of red/green new catches (drawn from saved data only: s122 stats + hist, s118c bins117).

Left: ON starts vs seconds since begin green (pred phase), detector thick, healthy-or-not phase mates thin, density;
dashed = median green length, dotted = median cycle.  If the timing phase differs, its curve is drawn dashed red.
Right: per hour, ONs per green+yellow hour vs per red hour (pred phase) from bins117.

    python h122_plots.py   -> %DC_WORK%/s122/plots/*.png, picks122.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
S = DCW / "s122"
CH = ["RR", "INV", "E5", "DEP", "OCC_hi", "OCC_lo", "ADV_hi", "ADV_lo"]
N_PER = {"RR": 3, "INV": 3, "E5": 3, "DEP": 3, "OCC_hi": 2, "OCC_lo": 1, "ADV_hi": 1, "ADV_lo": 1}


def main():
    X = pd.read_parquet(S / "stats122.parquet")
    H = np.load(S / "hist122.npz")["H"].astype(float)
    K = pd.read_parquet(S / "histkey122.parquet")
    K["i"] = np.arange(len(K))
    hk = {(r.DeviceId, r.window, r.detector, r.src): r.i for r in K.itertuples()}
    picks = []
    for c in CH:
        f = X[(X["fp_" + c] == 1) & ~X.v4flag]
        real = f[f.ph_ok].sort_values("wg", key=lambda s: s.map({"h24": 0, "h3": 1, "m30": 2}))
        real = real.drop_duplicates("DeviceId")
        picks += [(c, "real", r) for r in real.head(N_PER[c]).itertuples()]
        wr = f[f.ph_wrong].drop_duplicates("DeviceId")
        picks += [(c, "phase_wrong", r) for r in wr.head(1).itertuples()]
    B = pd.read_parquet(DCW / "s118c" / "bins117.parquet",
                        filters=[("DeviceId", "in", list({p[2].DeviceId for p in picks}))],
                        columns=["DeviceId", "window", "detector", "b", "nG", "nY", "nR", "sGY", "sR"])
    (S / "plots").mkdir(exist_ok=True)
    rows = []
    for c, kind, r in picks:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        mates = X[(X.DeviceId == r.DeviceId) & (X.window == r.window) & (X.pred == r.pred) & (X.detector != r.detector)]
        for m in mates.head(8).itertuples():
            i = hk.get((m.DeviceId, m.window, m.detector, "p"))
            if i is not None and H[i].sum() > 0:
                ax[0].plot(H[i] / H[i].sum(), lw=.8, alpha=.6, label=f"det {m.detector}: {m.fn}")
        i = hk.get((r.DeviceId, r.window, r.detector, "p"))
        ax[0].plot(H[i] / max(H[i].sum(), 1), lw=2.5, color="k", label=f"det {r.detector}: P{int(r.pred)} {r.fn}")
        if r.ph_wrong and pd.notna(r.true):
            j = hk.get((r.DeviceId, r.window, r.detector, "t"))
            if j is not None:
                ax[0].plot(H[j] / max(H[j].sum(), 1), lw=2, ls="--", color="r", label=f"same det on timing P{int(r.true)}")
        ax[0].axvline(r.p_gmed, ls="--", color="g")
        ax[0].axvline(min(r.p_cmed, 150), ls=":", color="grey")
        ax[0].set_xlabel("s since begin green (green ends ~dashed, cycle dotted)")
        ax[0].set_ylabel("share of ONs")
        ax[0].legend(fontsize=7)
        b = B[(B.DeviceId == r.DeviceId) & (B.window == r.window) & (B.detector == r.detector)].copy()
        if len(b):
            hrs = b.b // 12 if r.wg != "m30" else b.b
            g = b.assign(h=hrs).groupby("h")[["nG", "nY", "nR", "sGY", "sR"]].sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                ax[1].plot(g.index, (g.nG + g.nY) / g.sGY * 3600, "g-o", ms=3, label="ONs per green h")
                ax[1].plot(g.index, g.nR / g.sR * 3600, "r-o", ms=3, label="ONs per red h")
            ax[1].set_xlabel("hour" if r.wg != "m30" else "5-min bin")
            ax[1].legend(fontsize=7)
        name = f"{c}_{kind}_{r.DeviceName}_d{r.detector}_{r.window}"
        val = {"RR": r.p_rr, "INV": r.p_inv, "E5": r.p_e5, "DEP": r.p_dep}.get(c.split("_")[0], r.p_occ if "OCC" in c else r.p_rr)
        fig.suptitle(f"{c}: det {r.detector}: P{int(r.pred)} {r.fn} ({r.span} lane) - {r.DeviceName} {r.window} - "
                     f"value {val:.2f} vs limit {getattr(r, 'lim_' + c):.2f} - timing P{r.true_raw} - v4 {r.st_v4}", fontsize=9)
        fig.tight_layout()
        fig.savefig(S / "plots" / f"{name}.png", dpi=90)
        plt.close(fig)
        rows.append(dict(check=c, kind=kind, DeviceName=r.DeviceName, DeviceId=r.DeviceId, detector=r.detector,
                         window=r.window, fn=r.fn, span=r.span, pred=r.pred, true=r.true_raw, label_fn=r.function,
                         value=val, lim=getattr(r, "lim_" + c), st_v4=r.st_v4, n_on=r.n_on, png=f"{name}.png"))
    pd.DataFrame(rows).to_csv(S / "picks122.csv", index=False)
    print(pd.DataFrame(rows).drop(columns=["DeviceId", "png"]).round(3).to_string())


if __name__ == "__main__":
    main()
