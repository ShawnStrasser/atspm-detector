"""Optional health charts (matplotlib, not a package dependency: `pip install matplotlib`).

    from atspm_detector import predict
    from atspm_detector.charts import health_chart
    out = predict("day.parquet")
    health_chart("day.parquet", out, detector=15, path="det15.png")

One chart per detector, built from the package's own outputs and the log: the detector (thick) and its phase mates
(thin) over the sample, counts per 15 min on top and % ON per 15 min below, real units; the periods the health check lists (stuck on, silent, counts dropped, unusual
daily pattern) are shaded.  Title = the health categories; subtitle = 'det 15: P5 Presence', signal, sample, status;
the reason line is printed under the chart.  Nothing is re-scored: the chart shows the answers as given.
"""
from __future__ import annotations

import json
import textwrap

import numpy as np
import pandas as pd

THIS, BAD, MUTED, GRID, INK = "#1f4e9c", "#d32f2f", "#8a8984", "#e6e5e0", "#1a1a1a"
SHADE = {"stuck on": "#f8c9c4", "silent": "#f3d9a4", "silent at night": "#f3d9a4", "counts dropped": "#f3d9a4",
         "unusual daily pattern": "#e3dcf3"}
MATES = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#7a7a75", "#008300", "#8c564b", "#17becf",
         "#bcbd22", "#9e9ac8", "#c49c94"]


def _events(events, device_id):
    """the detector events of one signal from a DataFrame or a parquet / csv path (column spellings as predict)."""
    from . import pipeline as P
    if isinstance(events, pd.DataFrame):
        ev = events.copy()
    elif str(events).lower().endswith(".csv"):
        ev = pd.read_csv(events)
    else:
        import duckdb
        con = duckdb.connect()
        try:
            ev = con.sql(f"SELECT * FROM read_parquet({P._q(str(events))})").df()
        finally:
            con.close()
    ev = ev.rename(columns={c: P.ALIASES.get(c.lower(), c) for c in ev.columns})
    if device_id is not None and "DeviceId" in ev:
        ev = ev[ev.DeviceId.astype(str) == str(device_id)]
    ev = ev[ev.EventId.isin([81, 82]) & (ev.Parameter >= 1) & (ev.Parameter <= 64)]
    ev = ev.drop_duplicates(["Timestamp", "EventId", "Parameter"])
    ev["Timestamp"] = pd.to_datetime(ev.Timestamp)
    return ev


def _bins(ev, t0, t1, bs=900):
    """counts and % ON per 15 min of every channel (health_core.events_to_bins: continuous ONs, the health view)."""
    from . import health_core as hc
    t = (ev.Timestamp - t0).dt.total_seconds().to_numpy()
    P_ = hc.prep_arrays(t, ev.EventId.to_numpy(), ev.Parameter.to_numpy())
    B = hc.events_to_bins(P_, t0, t1, bin_s=bs)
    x = t0 + pd.to_timedelta(np.arange(B["n_on"].shape[1]) * bs + bs / 2, unit="s")
    return {int(d): (B["n_on"][i].astype(float), 100 * B["occ"][i].astype(float) / bs) for i, d in
            enumerate(B["dets"])}, x


def health_chart(events, out: pd.DataFrame, detector: int, device_id=None, path=None, max_mates: int = 12):
    """Draw the health chart of one detector; returns the matplotlib Figure (saved to `path` when given)."""
    try:
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:                                     # pragma: no cover
        raise ImportError("atspm_detector.charts needs matplotlib: pip install matplotlib") from exc
    o = out if device_id is None else out[out.DeviceId.astype(str) == str(device_id)]
    r = o[o.Detector.astype(int) == int(detector)]
    if r.empty:
        raise ValueError(f"detector {detector} is not in the output")
    r = r.iloc[0]
    ev = _events(events, r.DeviceId)
    t0 = ev.Timestamp.min().floor("15min")
    t1 = ev.Timestamp.max().ceil("15min")
    ph = r.phase_pred if pd.notna(r.phase_pred) else r.get("phase_guess")
    mates = o[(o.DeviceId == r.DeviceId) & (o.Detector != r.Detector)]
    mates = mates[(mates.phase_pred == ph) | (mates.get("phase_guess") == ph)] if pd.notna(ph) else mates.iloc[:0]
    mates = mates.sort_values("n_actuations", ascending=False).head(max_mates)
    fig, ax = plt.subplots(2, 1, figsize=(14, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    S, x = _bins(ev, t0, t1)
    empty = (np.zeros(len(x)), np.zeros(len(x)))
    for k, m in enumerate(mates.itertuples()):
        c, p = S.get(int(m.Detector), empty)
        lab = f"det {int(m.Detector)}: {str(m.function_pred or m.function_guess or '').replace('Yellow_Red', 'Yellow-red')}"
        ax[0].plot(x, c, color=MATES[k % len(MATES)], lw=1.1, alpha=.85, label=lab)
        ax[1].plot(x, p, color=MATES[k % len(MATES)], lw=1.1, alpha=.85)
    c, p = S.get(int(r.Detector), empty)
    me = str(r.health_reason).split(" - ")[0] if isinstance(r.health_reason, str) else f"det {int(r.Detector)}"
    ax[0].plot(x, c, color=THIS, lw=2.8, label=me)
    ax[1].plot(x, p, color=THIS, lw=2.8)
    per = json.loads(r.health_bad_periods) if isinstance(r.health_bad_periods, str) and r.health_bad_periods else []
    seen = set()
    for q in per:
        a, b = pd.Timestamp(q["start"]), pd.Timestamp(q["end"])
        col = SHADE.get(q["what"], "#f8c9c4")
        for a_ in ax:
            a_.axvspan(a, b, color=col, alpha=.55, lw=0, label=q["what"] if (q["what"] not in seen and a_ is ax[0]) else None)
        seen.add(q["what"])
    ax[0].set_ylabel("actuations per 15 min")
    ax[1].set_ylabel("% ON per 15 min")
    ax[1].set_ylim(0, 100)
    for a_ in ax:
        a_.grid(True, color=GRID, lw=.8)
        a_.spines[["top", "right"]].set_visible(False)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax[0].legend(loc="upper left", fontsize=9, ncol=2, frameon=False)
    cats = r.health_categories if isinstance(r.health_categories, str) and r.health_categories else ""
    title = cats or ("no problem found" if r.health_status == "ok" else str(r.health_status))
    hrs = (t1 - t0).total_seconds() / 3600
    fig.suptitle(title, x=.01, ha="left", fontsize=14, fontweight="bold", color=BAD if r.health_status in
                 ("suspect", "bad") else INK)
    ax[0].set_title(f"{me} · signal {r.DeviceId} · {t0:%a %d %b %H:%M}, {hrs:g} h · status {r.health_status}",
                    loc="left", fontsize=11, color=MUTED)
    txt = "\n".join(textwrap.wrap(str(r.health_reason), 150))
    if isinstance(r.get("health_config"), str) and r.health_config:
        txt += "\nsetup: " + r.health_config
    fig.text(.01, .01, txt, fontsize=10, color=INK, va="bottom")
    fig.tight_layout(rect=(0, .07, 1, .95))
    if path is not None:
        fig.savefig(path, dpi=90)
        plt.close(fig)
    return fig
