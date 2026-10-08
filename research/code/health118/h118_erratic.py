"""Note 118a: erratic counts judged on what the chart shows - 15-min periods visibly outside the expected range.

Per detector-window (3 h, 24 h) from the note-110 15-min bins (x own count, R = its v110 yardstick, S = rest of the
signal, ok = covered).  Expected count in period b: e_b = R_b x the detector's share of R in the 30 min either side
(b excluded) - a share that drifts slowly over the day (lane balance) is followed, a single period jumping is not.
Range: e_b +- 3 sd, sd^2 = s_b (R_b + 1)(1 + s_b) + (C e_b)^2 (Poisson noise on both + natural lane-share scatter C, measured on
healthy detectors).  A period is OFF when the count is outside the range AND at least MIN_ABS counts away.
Statistic: n_off = OFF periods (both directions); the chart draws the range as a band and marks every OFF period.

    python h118_erratic.py [--study]   -> %DC_WORK%/s118/erratic118.parquet (+ study table with --study)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s118"
H = 2              # +- 2 periods = 30 min either side
Z = 3.0
MIN_ABS = 5.0
C = 0.15
MODE = "sum"      # expected share = its share of the yardstick over +- H periods (30 min); "wmed" (R-weighted median over +- HM) tested, not adopted (note 118a)
HM = 8            # +- 2 h
MIN_REF = 10.0     # the yardstick must have counted >= 10 in the hour around the period


def movsum(a, h):
    c = np.r_[0.0, np.cumsum(a)]
    n = len(a)
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return c[hi] - c[lo]


def wmed_share(x, r, ok, h):
    """per period b: the R-weighted median of x_j / R_j over the periods j within +- h of b (b itself left out,
    covered periods with R_j > 0 only) = the detector's usual share of its yardstick around b.  Robust: a burst
    or a dip lasting up to about half the window does not move it, a steady drift is followed."""
    n = len(x)
    off = np.arange(-h, h + 1)
    off = off[off != 0]
    J = np.arange(n)[:, None] + off[None]
    valid = (J >= 0) & (J < n)
    Jc = np.clip(J, 0, n - 1)
    use = valid & ok[Jc] & (r[Jc] > 0)
    rat = np.where(use, x[Jc] / np.maximum(r[Jc], 1e-9), np.inf)
    wt = np.where(use, r[Jc], 0.0)
    o = np.argsort(rat, 1)
    rs, ws = np.take_along_axis(rat, o, 1), np.take_along_axis(wt, o, 1)
    cw = np.cumsum(ws, 1)
    tot = cw[:, -1:]
    k = np.argmax(cw >= 0.5 * tot, 1)
    s = rs[np.arange(n), k]
    Rr = tot[:, 0]
    return np.where(Rr > 0, s, np.nan), Rr


def off_bins(x, r, ok, c=C, h=H, z=Z, min_abs=MIN_ABS, mode=None):
    """returns (e, half-width, off mask, scored mask)."""
    mode = mode or MODE
    x = np.where(ok, x, 0.0).astype(float)
    r = np.where(ok & np.isfinite(r), r, 0.0).astype(float)
    okf = ok.astype(float)
    if mode == "wmed":
        s, Rr = wmed_share(x, r, ok, HM)
        s = np.nan_to_num(s)
        X = s * Rr
        N = movsum(okf, HM) - okf
    else:
        X = movsum(x, h) - x
        Rr = movsum(r, h) - r
        N = movsum(okf, h) - okf
        s = X / np.maximum(Rr, 1e-9)
    e = s * r
    sd = np.sqrt(s * (r + 1) * (1 + s) + (c * e) ** 2)        # +1: a yardstick period with 0 counts still has noise
    scored = ok & (N >= 2) & (Rr >= MIN_REF) & ((X + x) > 0)
    hw = np.maximum(z * sd, min_abs)
    off = scored & (np.abs(x - e) > hw)
    return e, hw, off, scored


def run(cs=(C,), ref="R", mode=None):
    b = pd.read_parquet(DCW / "health110" / "b15.parquet", columns=["DeviceId", "window", "detector", "b", "x", "S",
                                                                    "R", "ok"])
    b = b.sort_values(["DeviceId", "window", "detector", "b"], kind="stable")
    key = b.DeviceId + "|" + b.window + "|" + b.detector.astype(str)
    codes, first = np.unique(key.to_numpy(), return_index=True)
    first = np.sort(first)
    last = np.r_[first[1:], len(b)]
    x, R, S, ok = (b[c].to_numpy() for c in ("x", "R", "S", "ok"))
    ok = ok.astype(bool)
    rows = []
    kk = key.to_numpy()
    for a, z in zip(first, last):
        d, w, det = kk[a].split("|")
        rr = R[a:z] if ref == "R" else S[a:z]
        rec = dict(DeviceId=d, window=w, detector=int(det))
        if not np.isfinite(rr).any():
            rows.append(rec)
            continue
        for c in cs:
            e, hw, off, sc = off_bins(x[a:z], rr, ok[a:z], c=c, mode=mode)
            sfx = f"_{int(round(c * 100)):02d}"
            rec[f"n_off{sfx}"] = int(off.sum())
            rec[f"n_up{sfx}"] = int((off & (x[a:z] > e)).sum())
            rec[f"n_sc{sfx}"] = int(sc.sum())
            xx = np.where(ok[a:z], x[a:z], 0.0)
            rec[f"exc{sfx}"] = float(np.where(off, np.abs(xx - e) - hw, 0.0).sum() / max(xx[sc].sum(), 1.0))
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cs = (0.10, 0.15, 0.20) if a.study else (C,)
    E = run(cs)
    if a.study:
        Es30 = run((C,), mode="sum").rename(columns={"n_off_15": "n_off_s30", "n_up_15": "n_up_s30", "n_sc_15": "n_sc_s30",
                                                     "exc_15": "exc_s30"})
        E = E.merge(Es30, on=["DeviceId", "window", "detector"], how="left")
    Es = run((C,), ref="S").rename(columns={"n_off_15": "n_off_sig", "n_up_15": "n_up_sig", "n_sc_15": "n_sc_sig", "exc_15": "exc_sig"})
    E = E.merge(Es, on=["DeviceId", "window", "detector"], how="left")
    E.to_parquet(OUT / ("erratic118_study.parquet" if a.study else "erratic118.parquet"))
    print(len(E), "rows")


if __name__ == "__main__":
    main()
