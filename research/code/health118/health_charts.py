"""Notes 118b / 118c: one plain chart per health category (matplotlib; draws ONLY from saved plot data, never re-scores).

Built on the 'day chart' the user knows (the detector with its phase mates over the day, real units) with exactly ONE
piece of evidence added for the category. Title = the problem in plain words with the numbers the check used;
subtitle = detector label ('det 15: P5 Presence'), signal, sample, status.

    draw_category(row, day, ev, marks, path)   one category chart ('prof' / 'shape24': the two-panel time-of-day
                                               picture - 24 h counts AND % ON against its type's normal band + mates)
    draw_day(row, day, path)                   plain day chart: counts + % ON of the phase, all saved data

Inputs (h118c_plotdata.py): row = dict from rows.json; day = day15 rows of that sheet row; ev = ev rows of that row
(long: series, t, value); marks = dict from marks.json. Usable later as an optional visualisation add-on.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK, INK2, MUTED, GRID = "#1a1a1a", "#52514e", "#8a8984", "#e6e5e0"
THIS = "#1f4e9c"                      # this detector
BAD = "#d32f2f"                       # the evidence
OKC = "#008300"                       # green time
SHADE_BAD, SHADE_OK, SHADE_NIGHT = "#f8c9c4", "#e3e3df", "#eceaf6"
MATES = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#7a7a75", "#008300", "#8c564b", "#17becf",
         "#bcbd22", "#9e9ac8", "#c49c94"]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12, "font.family": "DejaVu Sans"})
FIG = (14, 6.2)


# ------------------------------------------------------------------ text helpers
def dur_txt(s):
    s = float(s)
    if s < 300:
        return f"{s:.0f} s"
    m = int(round(s / 60))
    return f"{m} min" if m < 60 else (f"{m // 60} h" + (f" {m % 60} min" if m % 60 else ""))


def n_txt(x):
    return f"{x:,.0f}"


def hm(t):
    return pd.Timestamp(t).strftime("%H:%M")


def sample_txt(row):
    a, b = pd.Timestamp(row["sample_t0"]), pd.Timestamp(row["sample_t1"])
    if row["sample_h"] >= 24:
        return f"{a:%a %d %b}, whole day"
    return f"{a:%a %d %b} {a:%H:%M}-{b:%H:%M} ({row['sample_h']:g} h sample)"


def n_lanes(row):
    """lanes the zone covers (the model's lane span; note 121): int, or None when unknown."""
    v = row.get("lanes")
    if v is not None and v == v:
        return max(int(round(float(v))), 1)
    sp = str(row.get("span", ""))
    return 1 if sp == "1" else (2 if sp.startswith("2") else None)


def lane_txt(row):
    n = n_lanes(row)
    return "" if n is None else f"covers {n} lane{'s' if n > 1 else ''}"


def lane_kind(row):
    """'1-lane Advance' - the type whose limits the checks use (function x lane span)."""
    n = n_lanes(row)
    fn = str(row.get("fn", "")).replace("_", "-")
    return fn if n is None else f"{n}-lane {fn}"


def subtitle(row):
    st = row["status"].replace("not_enough_data", "too little data")
    pn = f"   ·   {row['phase_note']}" if row.get("phase_note") else ""     # the whole phase did the same
    ln = f" ({lane_txt(row)})" if lane_txt(row) else ""
    return f"{row['label']}{ln}   ·   signal {row['signal']}   ·   {sample_txt(row)}   ·   status: {st}{pn}"


# ------------------------------------------------------------------ frame helpers
DAY_LINES = ("stuck", "dropout", "level", "night_drop")       # charts that draw the day data (context outside the sample)


def xrange_of(row):
    a, b = pd.Timestamp(row["sample_t0"]), pd.Timestamp(row["sample_t1"])
    if row["sample_h"] >= 24 or row["category"] not in DAY_LINES:
        return a, b
    d0, d1 = pd.Timestamp(row["day_t0"]), pd.Timestamp(row["day_t1"])
    return max(d0, a - pd.Timedelta(hours=2)), min(d1, b + pd.Timedelta(hours=2))


def style(ax, x0, x1, ylab):
    ax.set_xlim(x0, x1)
    ax.set_ylabel(ylab, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    span_h = (x1 - x0).total_seconds() / 3600
    step = 3 if span_h > 12 else 1
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(colors=INK2)
    ax.set_ylim(bottom=0)


def sample_bounds(ax, row):
    if row["sample_h"] < 24:
        for t in (row["sample_t0"], row["sample_t1"]):
            ax.axvline(pd.Timestamp(t), color=MUTED, ls=":", lw=1.4)


def finish(fig, ax, path, handles=None, labels=None):
    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def phase_lines(ax, day, x0, x1, col, max_mates=12):
    """the detector (thick blue) and its phase mates (thin) over [x0, x1); returns legend handles / labels."""
    d = day[(day.t >= x0) & (day.t < x1)]
    hs, ls = [], []
    me = d[d["self"]]
    if len(me):
        h, = ax.plot(me.t, me[col], color=THIS, lw=3.0, zorder=5)
        hs.append(h)
        ls.append(me.label.iloc[0] + "  (this detector)")
    mates = [k for k in d[~d["self"]].detector.unique()][:max_mates]
    for j, k in enumerate(mates):
        m = d[d.detector == k]
        h, = ax.plot(m.t, m[col], color=MATES[j % len(MATES)], lw=1.2, alpha=0.9, zorder=5.5)
        hs.append(h)
        ls.append(m.label.iloc[0])
    return hs, ls


def shade(ax, t0, t1, color, label=None, alpha=0.75, y=0.97, zorder=1):
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0, zorder=zorder)
    if label:
        mid = t0 + (t1 - t0) / 2
        prev = getattr(ax, "_lab_prev", [])
        while any(abs((mid - q).total_seconds()) < 14400 and abs(yy - y) < 0.01 for q, yy in prev):
            y = y - 0.07
        prev.append((mid, y))
        ax._lab_prev = prev
        ha = "center"
        lo, hi = ax.get_xlim()
        pos = (mdates.date2num(mid) - lo) / max(hi - lo, 1e-9)
        if pos > 0.85:
            mid, ha = t1, "right"
        elif pos < 0.15:
            mid, ha = t0, "left"
        ax.text(mid, y, label, transform=ax.get_xaxis_transform(), ha=ha, va="top",
                fontsize=12, fontweight="bold", color=INK, zorder=6)


def series(ev, name):
    s = ev[ev.series == name].sort_values("t")
    return s.t, s.value.to_numpy(float)


# ------------------------------------------------------------------ categories
def c_stuck(row, day, ev, mk, ax):
    sp = mk.get("spans", [])
    counted = [s for s in sp if s["label"] == "counted"]
    other = [s for s in sp if s["label"] != "counted"]
    if row["flagged"]:
        n = len(counted)
        title = (f"Stuck on {n} times, {dur_txt(row['tot_s'])} in total" if n > 1 else
                 f"Stuck on for {dur_txt(row['tot_s'])}")
    else:
        title = f"Long ON{'s' if len(other) > 1 else ''} explained by a queue: not stuck"
    for s in counted:
        shade(ax, s["t0"], s["t1"], SHADE_BAD, dur_txt(s["dur_s"]))
    longest = max(other, key=lambda q: q["dur_s"]) if other else None
    for s in other:
        shade(ax, s["t0"], s["t1"], SHADE_OK, f"{s['label']} {dur_txt(s['dur_s'])}" if s is longest else None)
    hs, ls = phase_lines(ax, day, *xr(row), "pct_on")
    ex = []
    if counted:
        ex.append((Patch(color=SHADE_BAD), f"held ON longer than {lane_kind(row)} zones allow "
                   f"({dur_txt(row['lim_s'])})"))
    if other:
        ex.append((Patch(color=SHADE_OK), "long ON while the phase was queued (not counted)"))
    return title, "% of each 15 min ON", hs + [e[0] for e in ex], ls + [e[1] for e in ex], (0, 105)


def c_dropout(row, day, ev, mk, ax):
    s = mk["spans"][0]
    if row["flagged"]:
        title = f"Goes silent for {dur_txt(s['dur_s'])} (about {n_txt(row['expected'])} actuations expected)"
        shade(ax, s["t0"], s["t1"], SHADE_BAD, f"no actuations {hm(s['t0'])}-{hm(s['t1'])}")
        hs, ls = phase_lines(ax, day, *xr(row), "count")
        return title, "actuations per 15 min", hs + [Patch(color=SHADE_BAD)], ls + ["silent stretch"], None
    dets = " + ".join(f"det {x}" for x in row["dropped"].split("+") if x)
    title = (f"Not silent: quiet {hm(s['t0'])}-{hm(s['t1'])}, only ~{row['expected']:.0f} expected "
             f"without the faulty {dets}")
    shade(ax, s["t0"], s["t1"], SHADE_OK, f"no actuations {hm(s['t0'])}-{hm(s['t1'])}")
    hs, ls = phase_lines(ax, day, *xr(row), "count")
    return (title, "actuations per 15 min", hs + [Patch(color=SHADE_OK), Line2D([], [], lw=0)],
            ls + ["quiet stretch (not a fault)", f"({dets}: busy at night themselves; with them ~{row['expected_old']:.0f})"],
            None)


def c_level(row, day, ev, mk, ax):
    t = pd.Timestamp(row["drop_at"])
    title = (f"Counts drop at {t:%H:%M}: {row['own_before']:.0f} → {row['own_after']:.0f} per 15 min "
             f"(about {row['exp_after']:.0f} expected)")
    hs, ls = phase_lines(ax, day, *xr(row), "count")
    te, e = series(ev, "expected")
    h, = ax.plot(te, e, color=BAD, lw=2.2, ls="--", zorder=6)
    ax.axvline(t, color=BAD, lw=1.5)
    ax.text(t, 0.97, f"  drop {t:%H:%M}", transform=ax.get_xaxis_transform(), color=BAD, fontsize=12,
            fontweight="bold", va="top")
    return title, "actuations per 15 min", hs + [h], ls + ["expected (its earlier share of the signal's traffic)"], None


def c_night(row, day, ev, mk, ax):
    if row.get("silent_night"):
        s = mk["spans"][0]
        title = (f"Misses vehicles at night: no actuations {hm(s['t0'])}-{hm(s['t1'])}, "
                 f"about {row['expected']:.0f} expected")
        for a, b in mk.get("night", []):
            shade(ax, a, b, SHADE_NIGHT, None, alpha=1.0)
        shade(ax, s["t0"], s["t1"], SHADE_BAD, f"no actuations {hm(s['t0'])}-{hm(s['t1'])}")
        hs, ls = phase_lines(ax, day, *xr(row), "count")
        return (title, "actuations per 15 min", hs + [Patch(color=SHADE_BAD), Patch(color=SHADE_NIGHT)],
                ls + ["silent while its phase mates counted", "night 21:00-05:00"], None)
    title = (f"Misses vehicles at night: {row['night_n']:.0f} actuations 21:00-05:00, "
             f"about {row['night_exp']:.0f} expected")
    for a, b in mk.get("night", []):
        shade(ax, a, b, SHADE_NIGHT, None, alpha=1.0)
    hs, ls = phase_lines(ax, day, *xr(row), "count")
    te, e = series(ev, "expected")
    h, = ax.plot(te, e, color=BAD, lw=2.2, ls="--", zorder=6)
    # zoom on the night: daytime lines run off the top
    d = day[(day.t >= pd.Timestamp(row["sample_t0"])) & (day.t < pd.Timestamp(row["sample_t1"]))]
    hr = d.t.dt.hour
    nmax = np.nanmax(np.r_[d[(hr >= 21) | (hr < 5)]["count"].to_numpy(float), e[np.isfinite(e)], 5])
    row["_ylim"] = (0, 1.35 * nmax)
    return (title, "actuations per 15 min", hs + [h, Patch(color=SHADE_NIGHT), Line2D([], [], lw=0)],
            ls + [f"expected at night (its daytime share of {row['ref']})", "night 21:00-05:00",
                  "(zoomed on the night: daytime runs off the top)"], None)


def c_choppy(row, day, ev, mk, ax):
    if row["flagged"]:
        title = f"Erratic counts: {row['n_off']} of {row['n_sc']} 15-min periods outside the expected range"
    else:
        title = f"Counts stay inside the expected range ({row['n_off']} of {row['n_sc']} periods outside): not flagged"
    tl, lo = series(ev, "lo")
    _, hi = series(ev, "hi")
    _, e = series(ev, "expected")
    to, x = series(ev, "own")
    _, off = series(ev, "off")
    ax.fill_between(tl, lo, hi, color=SHADE_OK, step=None, zorder=1)
    h1, = ax.plot(tl, e, color=INK2, lw=1.5, ls="--", zorder=3)
    h2, = ax.plot(to, x, color=THIS, lw=3.0, zorder=5)
    h3 = ax.scatter(to, off, s=110, color=BAD, zorder=7, edgecolor="white", lw=1.5)
    return (title, "actuations per 15 min", [h2, h1, Patch(color=SHADE_OK), h3],
            [row["label"] + "  (this detector)", "expected (its share of the counts of " + {"phase": "its phase mates", "same_fn_sig":
             "similar detectors at this signal", "same_fn": "similar detectors"}.get(row.get("ref"), "the signal")
             + ")", "normal range",
             "outside the range"], None)


def c_rapid(row, day, ev, mk, ax):
    tg, g = series(ev, "fast_green")
    _, r = series(ev, "fast_red")
    _, e = series(ev, "expected")
    _, bu = series(ev, "burst")
    tb = tg[np.isfinite(bu)]
    per = "5 min"
    w = pd.Timedelta(minutes=4.2)
    if row["sample_h"] >= 12:                      # a whole day: draw 15-min sums (the check works on 5-min periods)
        k = len(g) // 3 * 3
        g, r, e = (np.nansum(v[:k].reshape(-1, 3), 1) for v in (g, r, e))
        tg = tg.iloc[1:k:3]
        per, w = "15 min", pd.Timedelta(minutes=12.5)
    ax.bar(tg, g, width=w, color=OKC, zorder=3)
    ax.bar(tg, r, bottom=g, width=w, color=BAD, zorder=3)
    h, = ax.plot(tg, e, color=INK, lw=1.8, ls="--", zorder=4)
    top = np.nanmax(np.r_[g + r, e, 1])
    tot = pd.Series(g + r, index=tg)
    yb = [tot.iloc[int(np.argmin(np.abs((tg - t).dt.total_seconds().to_numpy())))] + 0.06 * top for t in tb]
    normal = row["fem_all"] + row["lim_zf"] * math.sqrt(row["fem_all"] + 1)
    bursts = row["flagged"] and row["x_spk"] >= row["x_zf"]
    hs = [Patch(color=OKC), Patch(color=BAD), h]
    ls = ["back-to-back vehicle ONs (< 1 s apart) in green", "back-to-back in red", "expected from its traffic"]
    if bursts:
        title = (f"Too-fast actuations: {row['n_spk']} bursts, normal for {lane_kind(row)} zones is at most "
                 f"{math.floor(row['lim_n_spk'])}")
        hb = ax.scatter(tb, yb, marker="v", s=90, color=INK, zorder=6)
        hs.append(hb)
        ls.append(f"burst: a 5-min period far above expected ({row['n_spk']})")
    elif row["flagged"]:
        title = (f"Too-fast actuations: {n_txt(row['fo_all'])} back-to-back, normal for {lane_kind(row)} zones "
                 f"up to {n_txt(normal)}")
    else:
        title = (f"Not too fast: {n_txt(row['fo_all'])} back-to-back vehicle ONs, normal for {lane_kind(row)} zones "
                 f"up to {n_txt(normal)}")
    if not bursts and row.get("n_retrig", 0) > 0:
        hs.append(Line2D([], [], lw=0))
        ls.append(f"(not counted: {n_txt(row['n_retrig'])} turning on again < 0.3 s after an OFF = chatter check)")
    return title, f"back-to-back actuations per {per}", hs, ls, None


def c_volume(row, day, ev, mk, ax):
    t, q = series(ev, "flow")
    what = "of green" if row["green_only"] else ""
    if row["flagged"]:
        title = f"Too many for the traffic: {n_txt(row['q5'])} veh/h per lane in one 5 min (limit {n_txt(row['lim_vol'])})"
    else:
        title = (f"Heavy traffic, not a fault: busiest {n_txt(row['q5'])} veh/h per lane {what} "
                 f"(limit {n_txt(row['lim_vol'])})")
    h1, = ax.plot(t, q, color=THIS, lw=2.0, zorder=4)
    h2 = ax.axhline(row["lim_vol"], color=BAD, lw=2.0, ls="--", zorder=5)
    k = int(np.nanargmax(q))
    ax.scatter([t.iloc[k]], [q[k]], s=120, color=BAD, zorder=6, edgecolor="white", lw=1.5)
    ax.annotate(f"{n_txt(q[k])} at {t.iloc[k]:%H:%M}", (t.iloc[k], q[k]), xytext=(10, 6), textcoords="offset points",
                fontsize=12, fontweight="bold", color=INK)
    ylab = ("vehicles per hour of green, per lane (5-min periods)" if row["green_only"]
            else "vehicles per hour per lane (5-min periods)")
    return (title, ylab, [h1, h2], [row["label"] + f"  ({lane_txt(row)})",
                                    f"limit for {lane_kind(row)} zones (1 in 500 healthy, at least 1,800 = saturation)"],
            None)


def c_chatter(row, day, ev, mk, ax):
    t, n = series(ev, "count")
    _, c = series(ev, "chat")
    title = (f"Chattering: {100 * row['chat_frac']:.1f} % turn on again < 0.3 s after an OFF "
             f"(limit {100 * row['lim_chat']:.1f} % for {lane_kind(row)})")
    w = pd.Timedelta(minutes=12)
    ax.bar(t, n, width=w, color="#c9d6ea", zorder=2)
    ax.bar(t, c, width=w, color=BAD, zorder=3)
    return (title, "actuations per 15 min", [Patch(color="#c9d6ea"), Patch(color=BAD)],
            ["all actuations", "turning on again < 0.3 s after an OFF (chatter)"], None)


def c_occspk(row, day, ev, mk, ax):
    t, o = series(ev, "pct_on")
    _, e = series(ev, "explained")
    _, s = series(ev, "spike")
    title = (f"Erratic time ON: {row['n3_exc']:.0f} min ON that its counts don't explain "
             f"(limit {row['lim_n3']:.0f} min for {lane_kind(row)} zones)")
    h1, = ax.plot(t, o, color=THIS, lw=2.6, zorder=4)
    h2, = ax.plot(t, e, color=INK2, lw=1.6, ls="--", zorder=3)
    h3 = ax.scatter(t, s, s=100, color=BAD, zorder=6, edgecolor="white", lw=1.5)
    return (title, "% of each 15 min ON", [h1, h2, h3],
            [row["label"] + "  (this detector)", f"ON time its counts explain ({row['dbar']:.1f} s per actuation)",
             "ON far longer than its counts explain"], (0, 105))


def c_prof(row, day, ev, mk, ax):
    t, c = series(ev, "count_h")
    _, m = series(ev, "mates_h")
    nr, er = row["night_ratio"], row["exp_ratio"]
    d0 = pd.Timestamp(row["day_t0"])
    if row["flagged"]:
        title = f"Busy at night: 01-05 h at {100 * nr:.0f} % of its busiest 4 hours (normal for its type {100 * er:.0f} %)"
    else:
        title = (f"Not flagged, just under the limit: night at {100 * nr:.0f} % of its busiest 4 hours "
                 f"(normal {100 * er:.0f} %)")
    shade(ax, d0 + pd.Timedelta(hours=1), d0 + pd.Timedelta(hours=5), SHADE_NIGHT, None, alpha=1.0)
    h1, = ax.plot(t, c, color=THIS, lw=3.0, marker="o", ms=6, zorder=5)
    h2, = ax.plot(t, m, color=MATES[0], lw=1.6, zorder=4)
    b0, b1 = pd.Timestamp(row["b4_t0"]), pd.Timestamp(row["b4_t1"])
    h3, = ax.plot([b0, b1], [row["b4"]] * 2, color=INK, lw=3, solid_capstyle="butt", zorder=6)
    h4, = ax.plot([d0 + pd.Timedelta(hours=1), d0 + pd.Timedelta(hours=5)], [row["exp_night"]] * 2, color=BAD, lw=3,
                  ls="--", zorder=6)
    ax.text(d0 + pd.Timedelta(hours=3), row["exp_night"], f"normal {100 * er:.0f} %", color=BAD, fontsize=12,
            fontweight="bold", ha="center", va="bottom")
    hs = [h1, h2, h3, h4, Patch(color=SHADE_NIGHT)]
    ls = [row["label"] + "  (this detector)", "its phase mates (scaled to its busiest hours)",
          f"its busiest 4 hours: {row['b4']:.0f} per hour", "normal night level for its type", "night 01:00-05:00"]
    return title, "actuations per hour", hs, ls, None


def c_occhi(row, day, ev, mk, ax):
    t, o = series(ev, "pct_on")
    _, lim = series(ev, "limit")
    _, hi = series(ev, "hi")
    title = f"Watch: ON longer than similar zones for {row['hi_min']:.0f} min"
    h1, = ax.plot(t, o, color=THIS, lw=2.6, zorder=4)
    h2, = ax.step(t, lim, where="mid", color=INK2, lw=1.6, ls="--", zorder=3)
    h3 = ax.scatter(t, hi, s=100, color=BAD, zorder=6, edgecolor="white", lw=1.5)
    return (title, "% of each 15 min ON", [h1, h2, h3],
            [row["label"] + "  (this detector)", "most similar zones reach at that traffic (1 in 200)",
             "longer, while its phase was not busy"], (0, 105))


def c_count_on(row, day, ev, mk, ax):
    t, o = series(ev, "pct_on")
    _, lim = series(ev, "limit")
    _, hi = series(ev, "hi")
    if row["flagged"]:
        title = (f"Count zone held ON: {row['hi_min']:.0f} min in light traffic, up to {row['hi_occ_max']:.0f} % ON "
                 f"(limit 20 %)")
    elif row.get("ext_fault"):                     # note 121 rule C: held ON by extension still spoils the counts
        title = f"Count zone held ON {row['hi_min']:.0f} min in light traffic by extension: counts unreliable"
    else:
        title = (f"Held ON {row['hi_min']:.0f} min, but ONs logged again without OFF explain it: "
                 f"config note, not a fault")
    h1, = ax.plot(t, o, color=THIS, lw=2.6, zorder=4)
    h2, = ax.step(t, lim, where="mid", color=INK2, lw=1.6, ls="--", zorder=3)
    h3 = ax.scatter(t, hi, s=100, color=BAD, zorder=6, edgecolor="white", lw=1.5)
    ls = [row["label"] + "  (this detector)",
          "most a count zone should be ON: 20 %, or 4 s per vehicle when busier",
          "above it while the signal's traffic was light"]
    if not row["flagged"]:
        ls[2] = f"above it ({n_txt(row['n_rep'])} ONs logged again without OFF = extension)"
    hs = [h1, h2, h3]
    if ((o > lim) & ~np.isfinite(hi)).any():         # above the line but not counted by the check
        hs.append(Line2D([], [], lw=0))
        ls.append("(above it without a dot: not counted, as the\n signal was busy or its phase mates ON too)")
    return title, "% of each 15 min ON", hs, ls, (0, 105)


def c_c1(row, day, ev, mk, ax):
    t, a = series(ev, "starts")
    _, r = series(ev, "relogged")
    tot = row["n_start"] + row["n_relog"]
    title = (f"Config note: {n_txt(row['n_relog'])} of {n_txt(tot)} ONs logged again with no OFF "
             f"(extension time on a count zone?)")
    w = pd.Timedelta(minutes=12)
    ax.bar(t, a, width=w, color=THIS, zorder=3)
    ax.bar(t, r, bottom=a, width=w, color=MATES[0], zorder=3)
    return (title, "ONs per 15 min", [Patch(color=THIS), Patch(color=MATES[0])],
            ["ON after an OFF (a vehicle)", "ON logged again with no OFF between"], None)


def c_c2(row, day, ev, mk, ax):
    p = mk.get("points", [])
    title = (f"Config note: set to pulse but held ON {row['n_ge5']} times over 5 s "
             f"(longest {dur_txt(row['long_max'])})")
    if p:
        tt = pd.to_datetime([q["t"] for q in p], format="ISO8601")
        dd = [q["dur_s"] for q in p]
        ax.vlines(tt, 0, dd, color=BAD, lw=2, zorder=3)
        h = ax.scatter(tt, dd, s=110, color=BAD, zorder=4, edgecolor="white", lw=1.5)
        k = int(np.argmax(dd))
        for a, b in [(tt[k], dd[k])]:
            if b >= 30:
                ax.annotate(dur_txt(b), (a, b), xytext=(6, 4), textcoords="offset points", fontsize=12, color=INK)
    h0 = ax.axhline(5, color=INK2, ls="--", lw=1.4)
    return (title, "length of the ON (seconds)", [h, h0],
            [f"ON of 5 s or more (its usual ON: {row['med_dur']:.1f} s = pulse)", "5 s"], None)


DRAW = {"count_on": c_count_on, "stuck": c_stuck, "dropout": c_dropout, "level": c_level, "night_drop": c_night, "choppy": c_choppy,
        "rapid": c_rapid, "volume": c_volume, "chatter": c_chatter, "occspk": c_occspk, "prof": c_prof,
        "occ_hi": c_occhi, "C1": c_c1, "C2": c_c2}


def xr(row):
    return xrange_of(row)


def _tod_title(row):
    c, f = row["category"], row["flagged"]
    if c == "prof":
        nr, er = 100 * row["night_ratio"], 100 * row["exp_ratio"]
        if not f:
            return f"Not flagged, just under the limit: night at {nr:.0f} % of its busiest 4 hours (normal {er:.0f} %)"
        if row.get("occ_drove") and row.get("night_on") is not None:
            eo = max(row["exp_on"], row.get("typ_on_night") or 0)
            eot = "under 1 %" if eo < 1 else f"about {eo:.0f} %"
            return f"Busy at night: ON {row['night_on']:.0f} % of the time 01-05 h (normal for its type {eot})"
        if row.get("mate_excused"):
            return (f"Busy at night like its whole phase: 01-05 h at {nr:.0f} % of its busiest hours "
                    f"(normal {er:.0f} %)")
        return f"Busy at night: 01-05 h at {nr:.0f} % of its busiest 4 hours (normal for its type {er:.0f} %)"
    h0, h1 = row["ev_h0"], row["ev_h1"] + 1
    span = f"{h0:02d}:00-{h1:02d}:00"
    if not f and row.get("cong"):
        return f"Fewer counts {span} but ON {row['pct_run']:.0f} % of the time: a queue, not a fault"
    who = "its phase mates" if row["ev_from"] == "mates" else "its type"
    word = "quiet" if row["ev_dir"] < 0 else "busy"
    return (f"Unusual day: {n_txt(row['ev_obs'])} actuations {span}, "
            f"{who} would have ~{n_txt(row['ev_exp'])} ({word})")


def draw_tod(row, ev, marks, path):
    """two panels on one clock: actuations per hour and % ON per hour; the detector (thick), its type's normal band
    (grey = the limit the check uses, dashed = typical), its phase mates (thin; counts scaled to its daily total)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9.0), sharex=True)
    d0 = pd.Timestamp(row["day_t0"])
    x0, x1 = d0, d0 + pd.Timedelta(days=1)
    hs, ls = [], []
    for ax, pre, col, ylab in ((a1, "", "count_h", "actuations per hour"), (a2, "o", "pct_h", "% of each hour ON")):
        tl, lo = series(ev, f"{pre}type_lo")
        _, hi = series(ev, f"{pre}type_hi")
        _, me = series(ev, f"{pre}type_med")
        ax.fill_between(tl, np.maximum(lo, 0), hi, color=SHADE_OK, zorder=1)
        hm_, = ax.plot(tl, me, color=INK2, lw=1.4, ls="--", zorder=2)
        names = sorted({s_.split(":", 1)[1] for s_ in ev.series.unique()
                        if s_.startswith("mate_count:" if pre == "" else "mate_pct:")})
        mh = []
        for j, nm in enumerate(names[:12]):
            tm, v = series(ev, ("mate_count:" if pre == "" else "mate_pct:") + nm)
            hh, = ax.plot(tm, v, color=MATES[j % len(MATES)], lw=1.2, alpha=.9, zorder=3)
            mh.append((hh, nm))
        t, v = series(ev, col)
        h_, = ax.plot(t, v, color=THIS, lw=3.0, marker="o", ms=5, zorder=5)
        style(ax, x0, x1, ylab)
        if pre == "":
            hs = [h_, Patch(color=SHADE_OK), hm_] + [m[0] for m in mh]
            ls = [row["label"] + "  (this detector)", f"normal range for its type ({lane_kind(row)})",
                  "typical for its type"] + [m[1] + " (scaled to its day)" for m in mh]
    a2.set_ylim(0, 102)
    if row["category"] == "prof":
        for ax in (a1, a2):
            shade(ax, d0 + pd.Timedelta(hours=1), d0 + pd.Timedelta(hours=5), SHADE_NIGHT, None, alpha=1.0, zorder=0)
        hs.append(Patch(color=SHADE_NIGHT))
        ls.append("night 01:00-05:00")
    elif row["ev_h0"] >= 0:
        a = d0 + pd.Timedelta(hours=row["ev_h0"])
        b = d0 + pd.Timedelta(hours=row["ev_h1"] + 1)
        colr = SHADE_BAD if row["flagged"] else "#cfe6cf"            # look-alike: soft green, not the band grey
        lab = f"{n_txt(row['ev_obs'])} vs ~{n_txt(row['ev_exp'])}"
        shade(a1, a, b, colr, lab, alpha=.6)
        if not row["flagged"] and row.get("cong"):
            shade(a2, a, b, colr, f"ON {row['pct_run']:.0f} %", alpha=.6)
        else:
            shade(a2, a, b, colr, None, alpha=.6)
        hs.append(Patch(color=colr))
        ls.append("the hours the check looked at" if row["flagged"] else "fewer counts while queued")
    title = _tod_title(row)
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=18, fontweight="bold", color=INK)
    sub = subtitle(row)
    if row.get("sig_wide"):
        sub += f"   ·   {row['sig_odd_n']} of {row['sig_n']} detectors at this signal have an unusual day"
    fig.text(0.01, 0.945, sub, ha="left", fontsize=12.5, color=INK2)
    a1.legend(hs, ls, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return title


def draw_category(row, day, ev, marks, path):
    if row["category"] in ("prof", "shape24") and (ev.series == "type_lo").any():
        return draw_tod(row, ev, marks, path)
    fig, ax = plt.subplots(figsize=FIG)
    ax.set_xlim(*xr(row))
    title, ylab, hs, ls, ylim = DRAW[row["category"]](row, day, ev, marks or {}, ax)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, subtitle(row), ha="left", fontsize=13, color=INK2)
    x0, x1 = xr(row)
    style(ax, x0, x1, ylab)
    ylim = row.pop("_ylim", None) or ylim
    if ylim:
        ax.set_ylim(*ylim)
    sample_bounds(ax, row)
    if row["sample_h"] < 24:
        hs = hs + [Line2D([], [], color=MUTED, ls=":", lw=1.4)]
        ls = ls + ["start / end of the sample checked"]
    finish(fig, ax, path, hs, ls)
    return title


def draw_day(row, day, path):
    z0, z1 = pd.Timestamp(row["data_t0"]), pd.Timestamp(row["data_t1"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.6), sharex=True)
    fig.suptitle(f"{row['label']}" + (" and its phase mates" if row.get("mates") else "") + ", all saved data", x=0.01, y=0.99, ha="left", fontsize=17,
                 fontweight="bold", color=INK)
    ln = f"{lane_txt(row)}   ·   " if lane_txt(row) else ""
    fig.text(0.01, 0.945, f"{ln}signal {row['signal']}   ·   {z0:%a %d %b %H:%M} - {z1:%a %d %b %H:%M}   ·   "
                          f"sample checked: {sample_txt(row)}", ha="left", fontsize=13, color=INK2)
    for j, (ax, col, yl) in enumerate(zip(axes, ("count", "pct_on"), ("actuations per 15 min", "% of each 15 min ON"))):
        a, b = pd.Timestamp(row["sample_t0"]), pd.Timestamp(row["sample_t1"])
        ax.axvspan(a, b, color="#f4f3ee", zorder=0, lw=0)
        hs, ls = phase_lines(ax, day, z0, z1, col)
        ax.set_xlim(z0, z1)
        ax.set_ylabel(yl, color=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
        ax.set_ylim(0, 102 if j else None)
        if j == 0:
            ax.legend(hs + [Patch(color="#f4f3ee")], ls + ["sample checked"], loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=90)
    plt.close(fig)
