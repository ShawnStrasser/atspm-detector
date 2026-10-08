"""Note 82d: second chart per health review row ("Day chart", user 2026-10-05): this detector's actuations per 15 min
over the whole calendar day on which the problem was flagged (00:00-24:00), blue line, partner orange line, flagged
period shaded.  Same rows / partners as h82_charts3; PNGs go to review/health_review_v1_day_charts/ (same base names).

    python h82_daychart.py [--only 1,2]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h82_charts3 as C3  # noqa: E402

C2, M = C3.C2, C3.M
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DAYDIR = M.H.REPO / "review" / "health_review_v1_day_charts"


def flagged(r, x, ev, t0, t1):
    """(start, end, label) of the period the check flagged."""
    key = r.check
    if key == "stuck":
        return pd.Timestamp(r.ep_t0), pd.Timestamp(r.ep_t1), "the long ON"
    if key == "dropout":
        return (t0 + pd.Timedelta(seconds=int(r.drop_b0) * 300), t0 + pd.Timedelta(seconds=int(r.drop_b1) * 300),
                "silent stretch")
    if key == "volume":
        k = int(np.argmax(M.counts(C3.ons_of(ev, int(r.detector), t0, t1), t0, t1, 300)[1]))
        a = t0 + pd.Timedelta(seconds=300 * k)
        return a, a + pd.Timedelta(minutes=5), "the busiest 5 minutes"
    if key == "level":
        return t0 + pd.Timedelta(seconds=int(r.level_b) * 300), t1, "after the drop (in the sample)"
    if key == "night_drop":
        return None, None, "night (21:00-05:00) in the sample"
    if key == "night_day":
        return None, None, "night (00:00-05:00)"
    return t0, t1, "the sample the check looked at"


def chart(r, x, g, ev, partner, pkind, path):
    d = int(r.detector)
    t0, t1 = r.t0, r.t1
    a, b, lab = flagged(r, x, ev, t0, t1)
    day = (a if a is not None else t0).normalize()
    z0, z1 = day, day + pd.Timedelta(days=1)
    if r.check == "night_drop":
        shade = [(max(z0, t0), min(z1, day + pd.Timedelta(hours=5)), lab),
                 (max(z0, day + pd.Timedelta(hours=21)), min(z1, t1), lab)]
    elif r.check == "night_day":
        shade = [(z0, day + pd.Timedelta(hours=5), lab)]
    else:
        shade = [(max(a, z0), min(b, z1), lab)]
    shade = [s for s in shade if s[1] > s[0]]
    pl = None
    if partner is not None:
        pl = f"det {partner} " + {"partner": "(partner)", "same": "(same phase)", "busy": "(busiest other detector)"}[pkind]
    fig, ax = plt.subplots(figsize=(13, 5.6))
    C3.count_lines(ax, ev, d, partner, pl, z0, z1, 900, shade=shade)
    ax.xaxis.set_major_locator(C3.mdates.HourLocator(byhour=range(0, 24, 2)))
    ax.xaxis.set_major_formatter(C3.mdates.DateFormatter("%H:%M"))
    cov1 = ev.Timestamp.max()
    title = f"Det {d} - actuations per 15 min, {day:%a %d %b}"
    sub = f"Signal {r.signal}   |   model: {r.model_says}   |   check: {M.CHECKS[r.check][0]}"
    if cov1 < z1 - pd.Timedelta(minutes=15):
        sub += f"\nThe saved data end at {cov1:%H:%M}; nothing after that is shown."
        ax.axvspan(cov1, z1, color="#eeeeee", lw=0, zorder=0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=16, fontweight="bold", color=C3.INK)
    fig.text(0.01, 0.905, sub, ha="left", va="top", fontsize=12, color=C3.MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.89 - 0.04 * sub.count("\n")))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return title


if __name__ == "__main__":
    C2.chart = chart
    if "--out" not in sys.argv:
        sys.argv += ["--out", str(DAYDIR)]
    C2.main()
    src = M.OUT / "titles_charts2.csv"
    if src.exists() and "--only" not in sys.argv:
        src.replace(M.OUT / "titles_daychart.csv")
