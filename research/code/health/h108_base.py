"""Note 108: baseline study of NORMAL detector behaviour per type, before any threshold is set.

Types come from the MODEL only: predicted function (Advance, Presence, Count, Yellow_Red, Mid, Bike, Other) x predicted
lane span (1 vs 2+), plus a volume band measured from the log (actuations per hour in the sample: low < 20, medium
20-100, high > 100).  Inside Count only, the ON mode is a MEASURED property (pulse = median clean ON <= 0.25 s, else
normal).  Technology (records) is joined for analysis only and never used by a rule.  No "long zone" type anywhere:
the queue pattern (occupancy rising while counts fall at high traffic) is measured per detector, within its class.

Data = note-96 / 104 w40 windows (Sat 26 - Mon 28 Sep 2026; 763 training signals, locked_v2 absent - asserted in
h96_occ / h108_events), the package health statistics (health96/health2 via health104/resolved104), the note-108 event
statistics (health108/act.parquet) and the note-96 15-min bins (n, occ, phase traffic ref = predicted Advance + Count
counts on the same predicted phase, self excluded).

Presumed healthy for statistic k = no finding (score >= .35) from any check OUTSIDE k's family, >= 20 actuations, a
predicted function - leave-own-check-out, so a limit is not truncated at the old limit of the same check.

    python h108_base.py build     -> health108/base.parquet, prof.parquet, pct_table.csv, plots/*.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "4")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "health108"
PLOT = OUT / "plots"
FNS = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
CHK = ("dropout", "stuck", "chatter", "rapid", "volume", "level", "choppy", "night_drop", "night_day", "corr", "occspk")
FAM = {"chatter": "fast", "rapid": "fast", "dropout": "silent", "stuck": "stuck", "level": "level",
       "night_drop": "level", "choppy": "shape", "corr": "shape", "volume": "shape", "night_day": "shape",
       "occspk": "occ"}
PULSE_MED = 0.25
BANDS = (20.0, 100.0)


def band_of(rate):
    return np.where(rate < BANDS[0], "low", np.where(rate < BANDS[1], "medium", "high"))


def load_base():
    R = pd.read_parquet(DCW / "health104" / "resolved104.parquet")
    A = pd.read_parquet(OUT / "act.parquet").drop(columns=["n_on"]).rename(
        columns={c: c + "_a" for c in ("chat_frac", "ioi_lt05", "ioi_lt1", "burst_frac")})
    X = R.merge(A, on=["DeviceId", "window", "detector"], how="left")
    X = X[X.fn.isin(FNS)].copy()
    X["span"] = np.where(pd.to_numeric(X.lanes, errors="coerce") >= 2, "2+", "1")
    X["rate"] = X.n_on / X.hours
    X["band"] = band_of(X.rate)
    X["cmode"] = np.where(X.fn.ne("Count"), "", np.where(X.dur_p50.isna(), "unknown",
                                                         np.where(X.dur_p50 <= PULSE_MED, "pulse", "normal")))
    X["type"] = X.fn + " " + X.span
    sc = pd.read_parquet(DCW / "health79" / "scored.parquet", columns=["DeviceId", "detector", "technology"])
    sc["DeviceId"] = sc.DeviceId.str.lower()
    X = X.merge(sc.dropna().drop_duplicates(["DeviceId", "detector"]), on=["DeviceId", "detector"], how="left")
    for k in CHK:
        X[f"f_{k}"] = pd.to_numeric(X.get(f"s_{k}"), errors="coerce").fillna(0) >= .35
    for fam in set(FAM.values()):
        other = [f"f_{k}" for k in CHK if FAM[k] != fam]
        X[f"hl_{fam}"] = ~X[other].any(axis=1) & (X.n_on >= 20)
    X["nd_rel"] = pd.to_numeric(X.night_day, errors="coerce") / pd.to_numeric(X.sig_night_day, errors="coerce").clip(
        lower=.15)
    return X


def bin_stats(X):
    """per detector-window from 15-min bins (3-h and 24-h samples): count and occupancy elasticity vs phase traffic,
    over all bins and over the busier half (ref >= its median); the queue pattern; hourly profile (24 h)."""
    Bn = pd.read_parquet(DCW / "health96" / "bins.parquet", columns=["DeviceId", "window", "detector", "b", "n", "occ",
                                                                      "ref"])
    Bn = Bn[~Bn.window.str.startswith("m30")]
    Bn = Bn[Bn.n.notna() & Bn.ref.notna()]
    lx = np.log(Bn.ref.to_numpy() + 1.0)
    Bn["lr"] = lx
    Bn["ln"] = np.log(Bn.n.to_numpy() + 1.0)
    Bn["lo"] = np.log(Bn.occ.to_numpy() + 0.005)
    g = Bn.groupby(["DeviceId", "window", "detector"])
    Bn["rmed"] = g.ref.transform("median")
    Bn["hi"] = Bn.ref >= Bn.rmed

    def slopes(d, pre):
        d = d.assign(x2=d.lr ** 2, xn=d.lr * d.ln, xo=d.lr * d.lo, one=1.0)
        s = d.groupby(["DeviceId", "window", "detector"])[["one", "lr", "ln", "lo", "x2", "xn", "xo"]].sum()
        vx = s.x2 - s.lr ** 2 / s.one
        ok = (s.one >= 6) & (vx > 0.5)
        out = pd.DataFrame({f"{pre}cnt": (s.xn - s.lr * s.ln / s.one) / vx, f"{pre}occ": (s.xo - s.lr * s.lo / s.one) / vx})
        return out.where(ok)

    E = slopes(Bn, "el_").join(slopes(Bn[Bn.hi], "elhi_"))
    return E.reset_index(), Bn


def profiles(Bn):
    """hourly shares of the day's actuations / ON time (24-h samples); hour = b // 4 (windows start 00:00)."""
    b = Bn[Bn.window.str.startswith("h24")].assign(h=lambda d: d.b // 4)
    H = b.groupby(["DeviceId", "window", "detector", "h"])[["n", "occ"]].sum().reset_index()
    nb = b.groupby(["DeviceId", "window", "detector"]).b.count().rename("nbins")
    P = H.pivot_table(index=["DeviceId", "window", "detector"], columns="h", values=["n", "occ"])
    P.columns = [f"{a}{b}" for a, b in P.columns]
    return P.join(nb)


def build():
    PLOT.mkdir(parents=True, exist_ok=True)
    X = load_base()
    E, Bn = bin_stats(X)
    X = X.merge(E, on=["DeviceId", "window", "detector"], how="left")
    X["queue_pat"] = (X.elhi_occ > 0) & (X.elhi_cnt < 0)
    X.to_parquet(OUT / "base.parquet")
    P = profiles(Bn)
    P.reset_index().to_parquet(OUT / "prof_raw.parquet")
    # traffic-level curves per type (24 h, healthy shape): traffic level = ref / the detector's max ref in deciles
    b = Bn[Bn.window.str.startswith("h24")].merge(
        X.loc[X.wg.eq("h24") & X.hl_shape, ["DeviceId", "window", "detector", "type", "fn", "span", "band"]],
        on=["DeviceId", "window", "detector"])
    g = b.groupby(["DeviceId", "window", "detector"])
    b["tl"] = np.clip(np.floor(10 * b.ref / g.ref.transform("max").clip(lower=1)), 0, 9)
    b["nrel"] = b.n / g.n.transform("mean").clip(lower=1e-9)
    b.to_parquet(OUT / "curve_bins.parquet")
    print("built", len(X), "rows")


# ------------------------------------------------------------------ description
STATS = [  # (column, label, window groups, healthy family)
    ("rate", "actuations per hour", ("h24",), "silent"),
    ("share_ref", "own count / phase Advance+Count traffic", ("h24",), "shape"),
    ("c_cnt_ref", "corr 15-min counts vs phase traffic", ("h24",), "shape"),
    ("c_occ_ref", "corr 15-min time ON vs phase traffic", ("h24",), "shape"),
    ("el_cnt", "count elasticity vs traffic (all bins)", ("h24",), "shape"),
    ("elhi_cnt", "count elasticity, busier half", ("h24",), "shape"),
    ("elhi_occ", "time-ON elasticity, busier half", ("h24",), "shape"),
    ("occ", "mean time ON", ("h24",), "occ"),
    ("chop15", "erratic counts (dispersion)", ("h3", "h24"), "shape"),
    ("nd_rel", "night/day ratio vs signal's", ("h24",), "shape"),
    ("dur_p50", "median clean ON (s)", ("h24",), "fast"),
    ("dur_p90", "p90 clean ON (s)", ("h24",), "fast"),
    ("rep_frac", "ON logged again without OFF", ("h24",), "fast"),
    ("max_on_s", "longest continuous ON (s)", ("h3", "h24"), "stuck"),
    ("gap_p50", "median OFF->ON gap (s)", ("h24",), "fast"),
    ("chat_frac_a", "re-ON < 0.3 s after OFF", ("h3", "h24"), "fast"),
    ("ioi_lt1_a", "ON->ON < 1 s", ("h3", "h24"), "fast"),
    ("burst_frac_a", "ONs in bursts (5 within ~4 s)", ("h3", "h24"), "fast"),
    ("max5", "max actuations in 5 min", ("h3", "h24"), "shape"),
    ("n3_exc", "min of time ON its count does not explain", ("h3", "h24"), "occ"),
]


def pct_table(X):
    rows = []
    for col, lab, wgs, fam in STATS:
        for wg in wgs:
            x = X[(X.wg == wg) & X[f"hl_{fam}"]]
            if col in ("chat_frac_a", "ioi_lt1_a", "burst_frac_a"):
                x = x[x.n_on >= 50]
            for (t,), g in x.groupby(["type"]):
                v = pd.to_numeric(g[col], errors="coerce").dropna()
                if len(v) < 20:
                    continue
                q = v.quantile([.05, .5, .95, .99, .995])
                rows.append(dict(stat=col, label=lab, wg=wg, type=t, n=len(v), p05=q[.05], p50=q[.5], p95=q[.95],
                                 p99=q[.99], p995=q[.995]))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "pct_table.csv", index=False)
    return T


def describe():
    X = pd.read_parquet(OUT / "base.parquet")
    T = pct_table(X)
    for s in ("rate", "share_ref", "c_cnt_ref", "c_occ_ref", "elhi_cnt", "elhi_occ", "chop15", "nd_rel", "dur_p50",
              "rep_frac", "max_on_s", "chat_frac_a", "ioi_lt1_a", "max5", "n3_exc"):
        t = T[T.stat == s]
        print(f"\n== {s}: {t.label.iloc[0]}")
        print(t.pivot_table(index="type", columns="wg", values=["n", "p50", "p95", "p99", "p995"]).round(3).to_string())
    h = X[X.wg.eq("h24") & X.hl_shape & X.elhi_cnt.notna()]
    print("\nqueue pattern (busier half: time ON rises, counts fall), share of healthy-shape detectors, 24 h:")
    print(h.groupby(["type"]).queue_pat.agg(["mean", "size"]).round(3).to_string())
    print(h.groupby(["fn", "band"]).queue_pat.mean().unstack().round(3).to_string())
    print(h.groupby(["fn", "technology"]).queue_pat.mean().unstack().round(3).to_string())
    print("\nrep_frac (ON again without OFF) by function x technology, 24 h, median / p95:")
    hh = X[X.wg.eq("h24") & (X.n_on >= 20)]
    print(hh.groupby(["fn", "technology"]).rep_frac.quantile(.95).unstack().round(3).to_string())
    print("\nCount mode (measured) x technology, 24 h:")
    print(pd.crosstab(hh[hh.fn == "Count"].cmode, hh[hh.fn == "Count"].technology).to_string())
    print("\nfast stats 1 vs 2+ lanes (p99.5, 24 h, n >= 50):")
    f = X[X.wg.eq("h24") & X.hl_fast & (X.n_on >= 50)]
    print(f.groupby(["fn", "span"])[["chat_frac_a", "ioi_lt05_a", "ioi_lt1_a", "burst_frac_a"]].quantile(.995).round(3)
          .to_string())


def plots():
    X = pd.read_parquet(OUT / "base.parquet")
    C = pd.read_parquet(OUT / "curve_bins.parquet")
    P = pd.read_parquet(OUT / "prof.parquet") if (OUT / "prof.parquet").exists() else None
    for t in sorted(X.type.unique()):
        x = X[X.type == t]
        fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
        ax = ax.ravel()
        c = C[C.type == t]
        if len(c):
            q = c.groupby("tl").nrel.quantile([.1, .5, .9]).unstack()
            ax[0].fill_between(q.index * 10 + 5, q[.1], q[.9], color="#9ecae1", alpha=.6, label="p10-p90")
            ax[0].plot(q.index * 10 + 5, q[.5], color="#08519c", lw=2, label="median")
            ax[0].set(title="counts per 15 min (x its mean) vs traffic", xlabel="phase traffic, % of its daily max",
                      ylabel="count / its mean")
            ax[0].legend(frameon=False)
            q = c.groupby("tl").occ.quantile([.1, .5, .9]).unstack() * 100
            ax[1].fill_between(q.index * 10 + 5, q[.1], q[.9], color="#fdae6b", alpha=.6, label="p10-p90")
            ax[1].plot(q.index * 10 + 5, q[.5], color="#a63603", lw=2, label="median")
            ax[1].set(title="% of 15 min ON vs traffic", xlabel="phase traffic, % of its daily max", ylabel="% ON")
            ax[1].legend(frameon=False)
        for i, (col, lab, wg, fam, lg) in enumerate([
                ("chop15", "erratic counts (24 h)", "h24", "shape", True),
                ("dur_p50", "median clean ON, s (24 h)", "h24", "fast", True),
                ("chat_frac_a", "share re-ON < 0.3 s (24 h)", "h24", "fast", False),
                ("ioi_lt1_a", "share ON->ON < 1 s (24 h)", "h24", "fast", False),
                ("nd_rel", "night/day vs signal's (24 h)", "h24", "shape", True)]):
            a = ax[2 + i]
            v = pd.to_numeric(x[(x.wg == wg) & x[f"hl_{fam}"]][col], errors="coerce").dropna()
            if lg:
                v = v[v > 0]
                if len(v):
                    a.hist(np.log10(v), bins=40, color="#6baed6")
                    a.set_xlabel("log10 " + col)
            elif len(v):
                a.hist(v, bins=40, color="#6baed6")
                a.set_xlabel(col)
            if len(v):
                for p_, ls in ((.5, "-"), (.995, "--")):
                    z = v.quantile(p_)
                    a.axvline(np.log10(z) if lg else z, color="#333333", ls=ls, lw=1.2)
            a.set_title(lab + "  (line p50, dashed p99.5)")
        a = ax[7]
        if P is not None:
            for wnd, col in (("h24_a", "#2a9d8f"), ("h24_b", "#a03ca0")):
                p = P[(P.type == t) & (P.window == wnd) & P.hl_prof]
                if len(p):
                    m = p[[f"s{h}" for h in range(24)]]
                    a.fill_between(range(24), m.quantile(.025), m.quantile(.975), color=col, alpha=.18)
                    a.plot(range(24), m.median(), color=col, lw=2, label=("Sun 27" if wnd == "h24_a" else "Mon 28"))
            a.set(title="hourly share of the day's actuations (p2.5-p97.5)", xlabel="hour")
            a.legend(frameon=False)
        fig.suptitle(f"{t} lane(s): presumed-healthy detectors, w40 (n 24-h samples = {int((x.wg == 'h24').sum())})",
                     x=.01, ha="left", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOT / f"type_{t.replace(' ', '_').replace('+', 'plus')}.png", dpi=80)
        plt.close(fig)
    print("plots in", PLOT)


if __name__ == "__main__":
    {"build": build, "describe": describe, "plots": plots}[sys.argv[1]]()
