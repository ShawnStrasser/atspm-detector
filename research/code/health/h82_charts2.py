"""Note 82b: redraw the health review charts in a simpler form (user: "why a step function, give me something simpler").

Same 70 rows, same file names in review/health_review_v1_charts/ (so the workbook links stay valid), same inputs and
same partner choice as h82_review.py.  One panel per chart, a one-line title that states the finding in plain words:
  * counts (goes silent, too many in 5 min, count drops, erratic, partner, almost none, night checks): side-by-side
    bars of this detector vs its partner per 1 / 5 / 15 min or hour;
  * stuck on, chattering, ON almost all the time: horizontal timeline of ON periods (thick bars), green / red of the
    predicted phase as a light background band where the window is short enough to show it;
  * too-short ONs, too-fast actuations: plain histogram (this detector vs partner, % of its own) with the limit marked.
No step plots.  The workbook is not written here.

    python h82_charts2.py [--only 1,2,3]
"""
from __future__ import annotations

import argparse
import os
import textwrap
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h82_review as M  # noqa: E402

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

R, H, hc = M.R, M.H, M.hc
THIS, PART = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e6e6e6"
GBAND, RBAND, MARK = "#d9f2d9", "#fbe1e1", "#c62828"
SHADE = "#fff1c2"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12})


# ------------------------------------------------------------------ helpers
def hm(t):
    return pd.Timestamp(t).strftime("%a %H:%M")


def pct(v, lim=None):
    if np.isfinite(v) and 0 < v < 0.005:
        return f"{100 * v:.1f} %"
    return M.pct(v, lim)


def dur_txt(sec):
    m = sec / 60
    return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h".replace(".0 h", " h")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def time_axis(ax, t0, t1):
    span = t1 - t0
    if span <= pd.Timedelta(minutes=3):
        ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=range(0, 60, 10 if span <= pd.Timedelta(seconds=60)
                                                                         else 30)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    elif span <= pd.Timedelta(minutes=45):
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    elif span <= pd.Timedelta(hours=4):
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    elif span <= pd.Timedelta(hours=10):
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(t0, t1)
    d0, d1 = t0.strftime("%a %d %b"), (t1 - pd.Timedelta(seconds=1)).strftime("%a %d %b")
    ax.set_xlabel(f"time of day ({d0})" if d0 == d1 else f"time of day ({d0} - {d1})")


def bin_label(bin_s):
    return {60: "minute", 300: "5 minutes", 900: "15 minutes", 3600: "hour"}[bin_s]


def bars(ax, t0, t1, bin_s, series, shade=(), hline=None, ylabel=None, share=False):
    """side-by-side bars per time bin; series = [(label, ON times, colour)]. share: % of its own total."""
    for a, b, _ in shade:
        ax.axvspan(max(a, t0), min(b, t1), color=SHADE, lw=0, zorder=0)
    n = len(series)
    bd = bin_s / 86400
    w = bd * (0.8 / n)
    vals = []
    for j, (lab, ons, col) in enumerate(series):
        tt, c = M.counts(np.asarray(ons), t0, t1, bin_s)
        c = c.astype(float)
        if share:
            c = 100 * c / max(c.sum(), 1)
        x = mdates.date2num(pd.to_datetime(tt)) - bd / 2 + bd * 0.1 + w * (j + 0.5)
        ax.bar(x, c, width=w, color=col, label=lab, zorder=2, lw=0)
        vals.append(c)
    if hline is not None:
        ax.axhline(hline[0], color=MARK, ls="--", lw=1.8, zorder=3, label=hline[1])
    ax.set_ylabel(ylabel or f"actuations per {bin_label(bin_s)}")
    time_axis(ax, t0, t1)
    style(ax)
    h, lab = ax.get_legend_handles_labels()
    if shade:
        h.append(Patch(color=SHADE))
        lab.append(shade[0][2])
    ax.legend(h, lab, frameon=False, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    top = max([v.max() for v in vals] + [hline[0] if hline else 0, 1])
    ax.set_ylim(0, top * 1.1)
    return vals


def timeline(ax, ev, t0, t1, dets, p):
    """thick bars = ON periods; rows = detectors; background = green / not green of phase p (short windows only)."""
    show_g = p is not None and np.isfinite(p) and (t1 - t0) <= pd.Timedelta(hours=2)
    if show_g:
        ax.axvspan(t0, t1, color=RBAND, lw=0, zorder=0)
        for a, b in M.greens(ev, p, t0, t1):
            if b > t0 and a < t1:
                ax.axvspan(max(a, t0), min(b, t1), color=GBAND, lw=0, zorder=0)
    span_d = (t1 - t0).total_seconds() / 86400
    n = len(dets)
    for j, (k, lab, col) in enumerate(dets):
        on, end, *_ = M.intervals(ev, k, t0 - pd.Timedelta(hours=6), t1)
        m = (end > np.datetime64(t0)) & (on < np.datetime64(t1))
        s = np.maximum(on[m], np.datetime64(t0))
        e = np.minimum(end[m], np.datetime64(t1))
        st = mdates.date2num(pd.to_datetime(s))
        w = np.maximum(mdates.date2num(pd.to_datetime(e)) - st, span_d / 1500)
        y = n - 1 - j
        ax.broken_barh(list(zip(st, w)), (y - 0.3, 0.6), color=col, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab, _ in dets][::-1])
    ax.set_ylim(-0.7, n - 0.3)
    time_axis(ax, t0, t1)
    style(ax)
    ax.grid(False)
    h = [Patch(color="#777777")]
    lab = ["thick bar = detector ON"]
    if show_g:
        h += [Patch(color=GBAND), Patch(color=RBAND)]
        lab += [f"phase {int(p)} green", f"phase {int(p)} not green"]
    ax.legend(h, lab, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    return show_g


def hist(ax, edges, labels, series, cut, cut_label, xlabel, unit="actuations"):
    """grouped bars, % of each detector's own values per category; vertical limit line after category `cut`."""
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    for j, (lab, v, col) in enumerate(series):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        h, _ = np.histogram(np.clip(v, edges[0], edges[-1] - 1e-9), edges)
        ax.bar(x - 0.4 + w * (j + 0.5), 100 * h / max(len(v), 1), width=w, color=col, label=f"{lab} ({len(v):,} {unit})",
               zorder=2)
    ax.axvline(cut + 0.5, color=MARK, ls="--", lw=1.8, zorder=3)
    ax.text(cut + 0.55, 0.97, cut_label, color=MARK, transform=ax.get_xaxis_transform(), va="top", fontsize=12,
            fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of that detector's total")
    style(ax)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)


def ons_of(ev, k, t0, t1):
    return M.intervals(ev, k, t0, t1)[2]


def bin_for(span, default=900):
    return 60 if span <= pd.Timedelta(minutes=45) else default


# ------------------------------------------------------------------ one chart
def chart(r, x, g, ev, partner, pkind, path):
    key, d, p = r.check, int(r.detector), r.pred_phase
    t0, t1 = r.t0, r.t1
    span = t1 - t0
    me = (f"det {d}", THIS)
    pl = None
    if partner is not None:
        pl = f"det {partner} " + {"partner": "(partner)", "same": "(same phase)", "busy": "(busiest other detector)"}[pkind]
    f = (lambda k: float(x.get(k, np.nan))) if x is not None else (lambda k: np.nan)  # noqa: E731
    fig, ax = plt.subplots(figsize=(13, 5.6))
    note = ""
    ons = ons_of(ev, d, t0, t1)
    two = lambda a, b: [(me[0], ons_of(ev, d, a, b), THIS)] + (  # noqa: E731
        [(pl, ons_of(ev, partner, a, b), PART)] if partner is not None else [])
    if key in ("stuck", "g_stuck", "chatter"):
        dets = [(d, f"det {d}", THIS)] + ([(partner, pl.replace(" (", "\n("), PART)] if partner is not None else [])
        if key == "stuck":
            a, b = pd.Timestamp(r.ep_t0), pd.Timestamp(r.ep_t1)
            m = max(pd.Timedelta(minutes=15), (b - a) * 0.3)
            z0, z1 = max(t0, a - m), min(t1, b + m)
            title = f"Det {d} stayed ON {dur_txt(f('ep_dur'))} without a break ({hm(a)}-{b:%H:%M})"
            if (z1 - z0) > pd.Timedelta(hours=2) and partner is not None:
                note = (f"Meanwhile det {partner} {pl[len(str(partner)) + 5:]} switched ON "
                        f"{len(ons_of(ev, partner, a, b)):,} times (too many to draw at this zoom).")
                dets = dets[:1]
        elif key == "g_stuck":
            z0, z1 = t0, t1
            title = f"Det {d} was ON {pct(g.frac_time_on, .90)} of the {M.dur(g.hours)} sample (limit 90 %)"
        else:
            tg = ev[ev.EventId.isin([81, 82]) & (ev.Parameter == d) & (ev.Timestamp >= t0) & (ev.Timestamp < t1)]
            tg = tg.sort_values(["Timestamp", "EventId"], ascending=[True, False])
            e, tt = tg.EventId.to_numpy(), tg.Timestamp.to_numpy()
            fast = np.r_[False, (e[1:] == 82) & (e[:-1] == 81) & ((tt[1:] - tt[:-1]) / np.timedelta64(1, "s") < 0.3)]
            z0 = M.busiest(tt, t0, t1, 30, fast)
            z1 = z0 + pd.Timedelta(seconds=30)
            inz = (tt >= np.datetime64(z0)) & (tt < np.datetime64(z1))
            n_on = int(((e == 82) & inz).sum())
            n_fast = int((fast & inz).sum())
            title = (f"Det {d}: {pct(f('chat_frac'), .30)} of its ONs came back within 0.3 s of switching OFF "
                     f"(limit 30 %)")
            note = (f"Busiest 30 s shown: {n_on} ONs, {n_fast} of them less than 0.3 s after the previous OFF. "
                    f"Whole sample: {f('n_on'):,.0f} actuations.")
        shown = timeline(ax, ev, z0, z1, dets, p)
        if key != "chatter" and not shown and p is not None and np.isfinite(p):
            note = (note + " " if note else "") + "Green / red not shown (too many signal cycles at this zoom)."
        if key == "chatter":
            ft = tt[fast & inz]
            ax.scatter(pd.to_datetime(ft), np.full(len(ft), len(dets) - 1 + 0.42), marker="v", s=90, color=MARK,
                       zorder=5, label="came back ON within 0.3 s")
            h_, l_ = ax.get_legend().legend_handles, [t.get_text() for t in ax.get_legend().get_texts()]
            mk = plt.Line2D([], [], marker="v", ls="", color=MARK, markersize=10)
            ax.legend(list(h_) + [mk], l_ + ["came back ON within 0.3 s"], frameon=False, loc="lower left",
                      bbox_to_anchor=(0, 1.0), ncol=4)
        if key == "stuck":
            ax.axvspan(a, b, ymin=0, ymax=1, fill=False, edgecolor=MARK, lw=2, ls="--", zorder=4)
            note = (note + " " if note else "") + "Red dashed box = the long ON."
            on_, end_, *_ = M.intervals(ev, d, z0 - pd.Timedelta(hours=6), z1)
            s_ = np.maximum(on_, np.datetime64(z0))
            e_ = np.minimum(end_, np.datetime64(z1))
            tot = ((e_ - s_)[e_ > s_] / np.timedelta64(1, "s")).sum() - (b - a).total_seconds()
            out_s = (z1 - z0).total_seconds() - (b - a).total_seconds()
            if out_s > 0 and tot / out_s > 0.5:
                note += (f" Outside the box it was ON {100 * tot / out_s:.0f} % of the time shown, with breaks too short "
                         f"to see at this zoom.")
    elif key == "dropout":
        a = t0 + pd.Timedelta(seconds=int(r.drop_b0) * 300)
        b = t0 + pd.Timedelta(seconds=int(r.drop_b1) * 300)
        m = max(pd.Timedelta(hours=1), (b - a) * 1.5)
        z0, z1 = max(t0, a - m), min(t1, b + m)
        bs = 300 if (z1 - z0) <= pd.Timedelta(hours=6) else 900
        bs = 60 if (z1 - z0) <= pd.Timedelta(minutes=45) else bs
        bars(ax, z0, z1, bs, two(z0, z1), [(a, b, "silent stretch")])
        own = int(((ons >= np.datetime64(a)) & (ons < np.datetime64(b))).sum())
        title = f"Det {d} counted {'nothing' if own == 0 else own} for {M.dur((b - a).total_seconds() / 3600)} ({hm(a)}-{b:%H:%M})"
        po = len(ons_of(ev, partner, a, b)) if partner is not None else 0
        if po > 0:
            title += f" while det {partner} counted {po:,}"
        else:
            title += f", where about {f('drop_lam'):.0f} were expected"
            if partner is not None:
                note = f"Det {partner} is shown for comparison; the expected count comes from " + (
                    "the other detectors on its phase." if x.get("drop_ref") == "phase" else "the rest of the signal.")
    elif key == "volume":
        allons = np.sort(ons)
        z = M.busiest(allons, t0, t1, 300)
        if span <= pd.Timedelta(hours=3):
            z0, z1 = t0, t1
        else:
            z0, z1 = max(t0, z.floor("5min") - pd.Timedelta(hours=1.5)), min(t1, z.floor("5min") + pd.Timedelta(hours=1.5))
        # 5-min bins aligned to the check's own bins (from the sample start)
        k0 = int(((z0 - t0).total_seconds()) // 300)
        z0 = t0 + pd.Timedelta(seconds=300 * k0)
        bars(ax, z0, z1, 300, two(z0, z1), hline=(150, "limit 150"))
        title = f"Det {d} had {f('max5'):.0f} actuations in one 5-minute period (limit 150)"
    elif key in ("level", "choppy", "partner", "g_low"):
        bs = bin_for(span, 900 if span <= pd.Timedelta(hours=6) or key == "choppy" else 3600)
        sh = []
        if key == "level":
            a = t0 + pd.Timedelta(seconds=int(r.level_b) * 300)
            sh = [(a, t1, "after the drop")]
            title = f"From {hm(a)} det {d} counted only {pct(f('level_ratio'), .15)} of its earlier share of the traffic (limit 15 %)"
        elif key == "choppy":
            who = "the other detectors on its phase" if x.get("ref_kind") == "phase" else "the rest of the signal"
            title = f"Det {d}'s 15-min counts jump up and down far more than traffic explains"
            note = (f"The swings are {f('chop15'):.1f}x what {who} would explain; healthy detectors stay under "
                    f"{f('chop_lim'):.1f}x.")
        elif key == "partner":
            pn = int(round(g.c_n / max(g.c_ratio, 1e-9)))
            lim = float(np.exp(-g.p_lim))
            title = f"Det {d} counted {int(g.c_n):,}, its partner det {partner} counted {pn:,} ({pct(g.c_ratio, lim)})"
            note = (f"Healthy {M.FN.get(g.pred_function, '?')} / {M.FN.get(g.pfn, '?')} pairs go down to "
                    f"{pct(lim, lim)}. Proposed check, not in the package.")
        else:
            title = f"Det {d} had only {int(g.n_on)} actuation{'' if int(g.n_on) == 1 else 's'} in {M.dur(g.hours)}"
            if partner is not None:
                title += f"; det {partner} had {len(ons_of(ev, partner, t0, t1)):,}"
            note = "Under 20 a day: the classifier gives no answer for it."
        bars(ax, t0, t1, bs, two(t0, t1), sh)
    elif key == "night_drop":
        night = []
        for day in pd.date_range(t0.normalize() - pd.Timedelta(days=1), t1.normalize(), freq="D"):
            a, b = day + pd.Timedelta(hours=21), day + pd.Timedelta(days=1, hours=5)
            if min(b, t1) > max(a, t0):
                night.append((max(a, t0), min(b, t1), "night (21:00-05:00)"))
        bars(ax, t0, t1, 3600, two(t0, t1), night)
        ref = str(x.get("night_ref"))
        title = f"At night det {d} counted {f('night_n'):.0f} where about {f('night_exp'):.0f} were expected"
        if "tracks" in ref:
            note = (f"Expected from det {partner}, which it follows by day. ")
        else:
            note = f"Expected from {ref}; det {partner} shown for comparison. "
        note += f"It counted {pct(f('night_ratio'), .19)} of what was expected (limit 19 %)."
    elif key in ("night_day", "corr"):
        rest = ev[ev.EventId.eq(82) & ev.Parameter.ne(d) & ev.Parameter.le(64) & (ev.Timestamp >= t0)
                  & (ev.Timestamp < t1)].Timestamp.to_numpy()
        sh = []
        if key == "night_day":
            sh = [(t0.normalize(), t0.normalize() + pd.Timedelta(hours=5), "night (00:00-05:00)")]
        bars(ax, t0, t1, 3600, [(f"det {d}", ons, THIS), ("all other detectors at the signal (combined)", rest, "#8a8a85")],
             sh, ylabel="% of the day's actuations\nin each hour", share=True)
        if key == "night_day":
            title = (f"Det {d} counted {f('night_day'):.1f}x as many per hour at night as by day "
                     f"(rest of the signal: {f('sig_night_day'):.2f}x)")
        else:
            title = f"Det {d}'s hourly counts do not rise and fall with the rest of the signal"
            note = f"Correlation {f('corr'):.2f}; normally about {f('corr_exp'):.2f}."
        note = (note + " " if note else "") + f"Each detector's bars add up to 100 % (det {d}: {len(ons):,} actuations)."
    elif key == "rapid":
        ioi = np.diff(np.sort(ons)) / np.timedelta64(1, "s")
        ser = [(f"det {d}", ioi, THIS)]
        if partner is not None:
            ser.append((pl, np.diff(np.sort(ons_of(ev, partner, t0, t1))) / np.timedelta64(1, "s"), PART))
        edges = [0, .5, 1, 2, 3, 5, 10, 30, 1e9]
        labs = ["under 0.5", "0.5-1", "1-2", "2-3", "3-5", "5-10", "10-30", "over 30"]
        hist(ax, edges, labs, ser, 1, "1 s", "seconds from one actuation to the next")
        title = (f"Det {d}: {pct(f('ioi_lt1'))} of its actuations came less than 1 s after the previous one")
        note = f"That is {f('rapid'):.2f}x the most that healthy detectors like it reach (limit 1x)."
    elif key == "short_on":
        dur = M.intervals(ev, d, t0, t1)[4]
        ser = [(f"det {d}", dur, THIS)]
        if partner is not None:
            ser.append((pl, M.intervals(ev, partner, t0, t1)[4], PART))
        edges = [0, .15, .25, .55, 1.05, 2.05, 5.05, 10.05, 1e9]
        labs = ["0.1", "0.2", "0.3-0.5", "0.6-1", "1.1-2", "2.1-5", "5.1-10", "over 10"]
        hist(ax, edges, labs, ser, 1, "0.2 s", "how long each ON lasted (seconds)", "ONs")
        title = f"Det {d}: {pct(f('short2'), .44)} of its ONs lasted 0.2 s or less (healthy stay under 44 %)"
        note = f"It is not set to pulse: its median ON is {f('med_dur'):.1f} s."
    else:
        raise ValueError(key)
    fig.suptitle(title, x=0.01, ha="left", fontsize=16, fontweight="bold", color=INK)
    sub = f"Signal {r.signal}   |   model: {r.model_says}   |   check: {M.CHECKS[key][0]}"
    lines = textwrap.wrap(note, 150) if note else []
    fig.text(0.01, 0.905, "\n".join([sub] + lines), ha="left", va="top", fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.89 - 0.04 * len(lines)))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return title


# ------------------------------------------------------------------ main (same loop as h82_review.main, no workbook)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(M.CH))
    a = ap.parse_args()
    only = {int(v) for v in a.only.split(",") if v}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_parquet(M.OUT / "chosen.parquet")
    old = pd.read_csv(M.OUT / "rows.csv")
    S["ord"] = S.check.map({k: i for i, k in enumerate(M.ORDER)})
    S = S.sort_values(["ord", "wlen", "DeviceId"]).reset_index(drop=True)
    assert len(S) == len(old) == 70
    names = pd.concat([pd.read_parquet(H.DCW / "official" / "labels_official.parquet", columns=["DeviceId", "DeviceName"]),
                       pd.read_parquet(H.REPO / "research" / "labels" / "function_labels_v3.parquet",
                                       columns=["DeviceId", "DeviceName"])]).drop_duplicates()
    names = names.dropna(subset=["DeviceName"])
    nm = dict(zip(names.DeviceId.str.lower(), names.DeviceName.astype(str)))
    R.init()
    cache, titles = {}, []
    for n, g in enumerate(S.itertuples(index=False), start=1):
        if only and n not in only:
            continue
        g = pd.Series(g._asdict())
        dev, d, w = g.DeviceId, int(g.detector), g.window
        o = old.iloc[n - 1]
        assert o.dev == dev and int(o.det) == d and o.window == w and o.key == g.check, (n, o.png)
        s0, hrs = R.WIN["w40"][w]
        t0 = pd.Timestamp(s0)
        t1 = t0 + pd.Timedelta(hours=hrs)
        ev = M.events(dev)
        ph, fn, ln, pc = R.inputs(w.split("_")[0], dev)
        if (dev, w) not in cache:
            cache[(dev, w)] = hc.health(ev, t0, t1, None, ph or None, fn or None, ln or None, pc or None)
        h = cache[(dev, w)]
        hx = h[h.detector == d]
        x = hx.iloc[0].copy() if len(hx) else None
        if x is not None:
            x["t0"] = t0
        p = ph.get(d, np.nan)
        mates = [k for k in ph if k != d and ph.get(k) == p and len(M.intervals(ev, k, t0, t1)[2]) >= 20] \
            if np.isfinite(p) else []
        partner, pkind = None, "partner"
        if g.check == "partner":
            partner = int(g.c_partner)
        elif x is not None and g.check == "night_drop" and "tracks" in str(x.get("night_ref")):
            partner = int(str(x.night_ref).split(",")[0][1:])
        elif x is not None and g.check == "stuck" and np.isfinite(float(x.get("ep_partner", np.nan))):
            partner = int(x.ep_partner)
        elif x is not None and g.check == "dropout" and np.isfinite(float(x.get("drop_partner", np.nan))):
            partner = int(x.drop_partner)
        if partner is None:
            partner, pkind = M.best_partner(ev, d, mates, t0, t1), "same"
        if partner is None:
            busy = ev[ev.EventId.eq(82) & ev.Parameter.ne(d) & ev.Parameter.le(64) & (ev.Timestamp >= t0)
                      & (ev.Timestamp < t1)].Parameter.value_counts()
            partner, pkind = (int(busy.index[0]), "busy") if len(busy) else (None, "busy")
        if pkind == "partner" and partner is not None and partner not in mates and ph.get(partner) != p:
            pkind = "partner"
        r = pd.Series(dict(check=g.check, detector=d, pred_phase=p, t0=t0, t1=t1, signal=o.signal, model_says=o.model))
        if x is not None:
            for c in ("ep_t0", "ep_t1", "drop_b0", "drop_b1", "level_b"):
                r[c] = x.get(c, np.nan)
        t = chart(r, x, g, ev, partner, pkind, out / o.png)
        titles.append((n, o.png, t))
        print(n, o.png, "|", t, flush=True)
    if not only:
        pd.DataFrame(titles, columns=["n", "png", "title"]).to_csv(M.OUT / "titles_charts2.csv", index=False)


if __name__ == "__main__":
    main()
