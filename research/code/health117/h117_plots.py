"""Note 117: visual check - per 5 min: vehicle ONs in green+yellow vs red, fast ONs (< 1 s) observed vs expected from
random arrivals, and green flow per lane against saturation.  User rows + a seeded sample of cleared / new flags.

    python h117_plots.py   -> %DC_WORK%/s117/plots/*.png, picks117.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
S = DCW / "s117"
P = S / "plots"


def main():
    P.mkdir(exist_ok=True)
    X = pd.read_parquet(S / "stats117.parquet")
    dev = pd.read_csv(DCW / "health110" / "review" / "rows_v110.csv").drop_duplicates("signal").set_index("signal").dev
    sig_of = {v: k for k, v in dev.items()}
    U = pd.read_csv(S / "user117.csv")
    picks = [(dev[r.signal], r.window, r.det, f"user_{r.user.rstrip('?')}") for r in U.itertuples() if r.window.endswith("_b")]
    cl = X[X.old_any & ~X.new_any & X.wg.ne("m30")].sample(8, random_state=117)
    nw = X[~X.old_any & X.new_any & X.wg.ne("m30")].sample(6, random_state=117)
    picks += [(r.DeviceId, r.window, r.detector, "cleared") for r in cl.itertuples()]
    picks += [(r.DeviceId, r.window, r.detector, "new") for r in nw.itertuples()]
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=4")
    out = []
    for d, w, k, tag in picks:
        b = con.execute(f"""SELECT * FROM read_parquet('{(S / "bins117.parquet").as_posix()}')
                            WHERE DeviceId='{d}' AND "window"='{w}' AND detector={int(k)} ORDER BY b""").df()
        x = X[(X.DeviceId == d) & (X.window == w) & (X.detector == k)].iloc[0]
        sU = np.clip(300 - b.sGY - b.sR - b.sRg, 0, None)
        nGY = b.nG + b.nY
        eGY = np.where(b.sGY > 0, nGY * (1 - np.exp(-nGY / b.sGY.clip(lower=1e-9))), 0)
        eR = np.where(b.sR > 0, b.nR * (1 - np.exp(-b.nR / b.sR.clip(lower=1e-9))), 0)
        eU = np.where(sU > 0, b.nU * (1 - np.exp(-np.minimum(b.nU / np.maximum(sU, 1e-9), 50))), b.nU)
        t = b.b * 5 / 60
        fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        ax[0].step(t, nGY, where="post", label="ONs in green+yellow", color="#2a9d3a")
        ax[0].step(t, b.nR, where="post", label="ONs in red", color="#c0392b")
        ax[0].set_ylabel("ONs / 5 min")
        ax[0].legend(loc="upper left", fontsize=8)
        ax[1].step(t, b.fGY + b.fR + b.fU, where="post", label="fast ONs (< 1 s) observed", color="k")
        ax[1].step(t, eGY + eR + eU, where="post", label="expected from random arrivals", color="#888", ls="--")
        ax[1].step(t, b.fR, where="post", label="observed in red", color="#c0392b", lw=.8)
        ax[1].set_ylabel("fast ONs / 5 min")
        ax[1].legend(loc="upper left", fontsize=8)
        q = np.where(b.sGY >= 60, nGY / b.sGY.clip(lower=1) * 3600 / x.ln, np.nan)
        qa = b.n / 300 * 3600 / x.ln
        ax[2].step(t, q, where="post", label="green flow per lane (veh / green h)", color="#2a9d3a")
        ax[2].step(t, qa, where="post", label="all-time flow per lane", color="#555", lw=.8)
        ax[2].axhline(x.lim_vol, color="#c0392b", ls="--", lw=1, label=f"limit {x.lim_vol:.0f}")
        ax[2].set_ylabel("veh / h / lane")
        ax[2].set_xlabel("hours from window start")
        ax[2].legend(loc="upper left", fontsize=8)
        name = f"{sig_of.get(d, d[:8])} det {k}"
        fig.suptitle(f"{tag}: {name} ({x.type}, {w}) old {'fast ' if x.old_fast else ''}{'vol' if x.old_vol else ''} -> "
                     f"new {x.lvl_new}; fast excess {x.zf:.1f} SD (lim {x.lim_zf:.1f}), burst bins {x.n_spk:.0f} (lim {x.lim_n_spk:.0f}); busiest 5 min "
                     f"{(x.q5_all if x.fn == 'Advance' else x.q5_gy):.0f} / lane (limit {x.lim_vol:.0f})", fontsize=9)
        fig.tight_layout()
        f = P / f"{tag}_{name.replace(' ', '_')}_{w}.png"
        fig.savefig(f, dpi=80)
        plt.close(fig)
        out.append(dict(tag=tag, DeviceId=d, window=w, detector=k, png=f.name))
    pd.DataFrame(out).to_csv(S / "picks117.csv", index=False)
    print(len(out), "plots")


if __name__ == "__main__":
    main()
