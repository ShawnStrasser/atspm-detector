"""Three detector-health example charts for README.md (docs/images/health_*.png).

Drawn ONLY from saved plot data `%DC_WORK%/s118c/plot` (rows.json / marks.json / ev.parquet / day15.parquet, notes
118c / 121); nothing is re-scored. Adapted from the s_linkedin slides. Public images: no signal name or id, no date,
only the detector label ("det 37: P4 Presence").

    python research/code/readme/make_readme_images.py   ->  docs/images/health_{stuck_on,busy_at_night,unusual_day}.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
PD = DCW / "s118c" / "plot"
OUT = Path(__file__).resolve().parents[3] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

for f in ("segoeui.ttf", "segoeuib.ttf"):
    p = Path(r"C:\Windows\Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "axes.unicode_minus": False})

BG, INK, INK2, MUTED, GRID = "#FFFFFF", "#1A1A1A", "#4F4D48", "#8A8984", "#E6E5E0"
NEW, MATE, BAND, NIGHT, SHADE_BAD, BAD = "#1F4E9C", "#EB6834", "#E3E3DF", "#ECEAF6", "#F8C9C4", "#C62828"
W, H, DPI = 1100, 490, 100

ROWS = {r["row"]: r for r in json.loads((PD / "rows.json").read_text())}
MARKS = json.loads((PD / "marks.json").read_text())
DAY = pd.read_parquet(PD / "day15.parquet")
EV = pd.read_parquet(PD / "ev.parquet")


def pt(px):
    return px * 72 / DPI


def figure(title, sub, ylab):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    fig.text(28 / W, 1 - 22 / H, title, fontsize=pt(30), fontweight="bold", color=INK, va="top")
    fig.text(28 / W, 1 - 66 / H, sub, fontsize=pt(21), color=INK2, va="top")
    fig.text(28 / W, 1 - 108 / H, ylab, fontsize=pt(18), color=MUTED, va="top")
    ax = fig.add_axes([90 / W, 42 / H, 740 / W, 318 / H])           # bottom 42 px, height 318 px
    ax.set_facecolor(BG)
    return fig, ax


def clock_axis(a, d0):
    a.set_xlim(d0, d0 + pd.Timedelta(days=1))
    a.set_xticks([d0 + pd.Timedelta(hours=h) for h in (0, 6, 12, 18, 24)])
    a.set_xticklabels(["12 am", "6 am", "noon", "6 pm", "12 am"])
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(MUTED)
    a.grid(axis="y", color=GRID, lw=1.0)
    a.set_axisbelow(True)
    a.tick_params(colors=INK2, labelsize=pt(17), length=4)


def legend(fig, items, x=852, y=150):
    """vertical legend in the right margin (pixel coordinates, y from the top of the plot area)."""
    yy = H - y
    for kind, col, lab in items:
        if kind == "line":
            fig.add_artist(plt.Line2D([x / W, (x + 34) / W], [yy / H] * 2, color=col, lw=4 if col == NEW else 2.5,
                                      solid_capstyle="round"))
        elif kind == "dash":
            fig.add_artist(plt.Line2D([x / W, (x + 34) / W], [yy / H] * 2, color=col, lw=2, ls=(0, (3, 2))))
        else:
            fig.add_artist(plt.Rectangle((x / W, (yy - 9) / H), 34 / W, 18 / H, color=col, lw=0))
        fig.text((x + 44) / W, yy / H, lab, fontsize=pt(16), color=INK2, va="center", linespacing=1.2)
        yy -= 62


def pct_axis(a, top):
    a.set_ylim(0, top)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_yticklabels([f"{v} %" for v in (0, 25, 50, 75, 100)])


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=DPI, facecolor=BG)
    plt.close(fig)
    im = Image.open(p).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(p, optimize=True)
    print(name, Image.open(p).size, f"{p.stat().st_size / 1024:.0f} KB")


def stuck_on():
    r = ROWS[1]
    d0 = pd.Timestamp(r["day_t0"])
    d = DAY[(DAY.row == 1) & (DAY.t >= d0) & (DAY.t < d0 + pd.Timedelta(days=1))]
    spans = [s for s in MARKS["1"]["spans"] if s["label"] == "counted"]
    tot = r["tot_s"] / 60
    mins = [round(s["dur_s"] / 60) for s in spans]
    fig, a = figure("Stuck on", f"{r['label']}: held ON {len(spans)} times, {int(tot // 60)} h {round(tot % 60)} min "
                    f"in total, in one day", "% of each 15 min ON")
    for s in spans:
        a.axvspan(pd.Timestamp(s["t0"]), pd.Timestamp(s["t1"]), color=SHADE_BAD, lw=0, zorder=1)
    me, mate = d[d["self"]], d[~d["self"]]
    a.plot(mate.t, mate.pct_on, color=MATE, lw=1.8, zorder=3)
    a.plot(me.t, me.pct_on, color=NEW, lw=2.8, zorder=4)
    clock_axis(a, d0)
    pct_axis(a, 118)
    s1, s2, s3 = spans
    kw = dict(va="bottom", fontsize=pt(17), fontweight="bold", color=BAD)
    a.text(pd.Timestamp(s1["t1"]), 103, f"{mins[0]} min ", ha="right", **kw)
    a.text(pd.Timestamp(s2["t0"]), 103, f" {mins[1]} min", ha="left", **kw)
    a.text(pd.Timestamp(s3["t0"]) + (pd.Timestamp(s3["t1"]) - pd.Timestamp(s3["t0"])) / 2, 103, f"{mins[2]} min",
           ha="center", **kw)
    legend(fig, [("line", NEW, "this detector"), ("line", MATE, f"{mate.label.iloc[0]}\n(same phase)"),
                 ("patch", SHADE_BAD, f"ON without a break\nfor over {r['lim_s'] / 60:.0f} min")])
    save(fig, "health_stuck_on.png")


def busy_at_night():
    r = ROWS[26]
    e = EV[EV.row == 26]
    d0 = pd.Timestamp(r["day_t0"])

    def s(nm):
        x = e[e.series == nm].sort_values("t")
        return x.t, x.value.to_numpy(float)
    typ = float(e[(e.series == "otype_med") & e.t.dt.hour.between(1, 4)].value.mean())
    fig, a = figure("Busy at night", f"{r['label']}: ON {r['night_on']:.0f} % of the time from 1 to 5 am; "
                    f"detectors of its type: about {typ:.0f} %", "% of each hour ON")
    a.axvspan(d0 + pd.Timedelta(hours=1), d0 + pd.Timedelta(hours=5), color=NIGHT, lw=0, zorder=0)
    t, lo = s("otype_lo")
    _, hi = s("otype_hi")
    _, med = s("otype_med")
    a.fill_between(t, lo, hi, color=BAND, lw=0, zorder=1)
    a.plot(t, med, color=INK2, lw=1.6, ls=(0, (3, 2)), zorder=2)
    for nm in sorted(x for x in e.series.unique() if x.startswith("mate_pct:")):
        tm, v = s(nm)
        a.plot(tm, v, color=MATE, lw=1.5, alpha=0.9, zorder=3)
    t, v = s("pct_h")
    a.plot(t, v, color=NEW, lw=2.8, marker="o", ms=4.5, zorder=5)
    clock_axis(a, d0)
    pct_axis(a, 104)
    a.text(d0 + pd.Timedelta(hours=3), 101, "night", ha="center", va="top", fontsize=pt(16), color="#5B4FA8",
           fontweight="bold")
    legend(fig, [("line", NEW, "this detector"), ("line", MATE, f"the other {len(r['mates'])} detectors\non its phase"),
                 ("patch", BAND, "normal range for\nits type"), ("dash", INK2, "typical for its type")])
    save(fig, "health_busy_at_night.png")


def unusual_day():
    r = ROWS[30]
    d0 = pd.Timestamp(r["day_t0"])
    d = DAY[(DAY.row == 30) & (DAY.t >= d0) & (DAY.t < d0 + pd.Timedelta(days=1))].copy()
    d["h"] = d.t.dt.floor("h")
    hh = d.groupby(["label", "h"])["count"].sum().unstack(0)
    me, pair = r["label"], "det 18: P6 Advance"           # the other Advance loop on the same phase
    assert pair in r["mates"]
    h0, h1 = r["ev_h0"], r["ev_h1"] + 1
    win = hh.loc[d0 + pd.Timedelta(hours=h0): d0 + pd.Timedelta(hours=h1 - 1)]
    own, other = float(win[me].sum()), float(win[pair].sum())
    assert own == r["ev_obs"]
    fig, a = figure("Unusual daily pattern", f"{me}: {own:.0f} actuations from 11 am to 3 pm, about "
                    f"{r['ev_exp']:.0f} expected", "actuations per hour")
    a.axvspan(d0 + pd.Timedelta(hours=h0), d0 + pd.Timedelta(hours=h1), color=SHADE_BAD, lw=0, zorder=0, alpha=0.75)
    x = hh.index + pd.Timedelta(minutes=30)
    a.plot(x, hh[pair], color=MATE, lw=2, marker="o", ms=3.5, zorder=3)
    a.plot(x, hh[me], color=NEW, lw=2.8, marker="o", ms=4.5, zorder=5)
    clock_axis(a, d0)
    top = float(np.nanmax(hh[[me, pair]].to_numpy()))
    a.set_ylim(0, top * 1.22)
    a.text(d0 + pd.Timedelta(hours=(h0 + h1) / 2), top * 1.20, f"{own:.0f}  vs  {other:.0f}", ha="center", va="top",
           fontsize=pt(17), fontweight="bold", color=BAD)
    legend(fig, [("line", NEW, "this detector"), ("line", MATE, f"{pair}\n(the other advance loop)"),
                 ("patch", SHADE_BAD, "the hours the\ncheck flagged")])
    save(fig, "health_unusual_day.png")


if __name__ == "__main__":
    stuck_on()
    busy_at_night()
    unusual_day()
