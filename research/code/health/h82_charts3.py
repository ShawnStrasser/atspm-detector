"""Note 82c: health review charts, third form (user, 2026-10-05: "no bar charts - line charts; show enough context").

Same 70 rows, same file names in review/health_review_v1_charts/ (workbook links stay valid), same inputs and partner
choice as h82_review.py / h82_charts2.py.  Rules:
  * counts over time = LINE chart, this detector blue, partner orange; the window starts well before the flagged
    period and ends well after it (saved events cover Sat 26 16:15 - Mon 28 24:00), flagged period shaded, the sample
    the check looked at marked with dotted lines when the chart reaches past it;
  * stuck on: ON/OFF timeline (thick bars) with an hour or more of normal behaviour either side when that fits
    (<= 2.5 h shown), else a line of "% of each 15 min ON"; ON almost all the time: line of "% of each 5 min ON";
  * chattering: line of "% of ONs back within 0.3 s" per 5 / 15 min over the sample and its surroundings;
  * too-short ONs / too-fast actuations: distribution drawn as lines (this detector vs partner) with the limit marked.
No bars, no step plots.  The workbook is not written here.

    python h82_charts3.py [--only 1,2,3] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h82_charts2 as C2  # noqa: E402

M, R, H, hc = C2.M, C2.R, C2.H, C2.hc
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

THIS, PART, REST = C2.THIS, C2.PART, "#7a7a75"
INK, MUTED, MARK, SHADE = C2.INK, C2.MUTED, C2.MARK, C2.SHADE
GBAND, RBAND = C2.GBAND, C2.RBAND
SAMPLE_C = "#555555"
hm, pct, dur_txt, style, ons_of = C2.hm, C2.pct, C2.dur_txt, C2.style, C2.ons_of
H1 = pd.Timedelta(hours=1)


# ------------------------------------------------------------------ helpers
def coverage(ev):
    t = ev.Timestamp
    return t.min().floor("5min"), t.max().ceil("5min")


def clip(z0, z1, cov):
    return max(z0, cov[0]), min(z1, cov[1])


def auto_bin(span):
    return 300 if span <= pd.Timedelta(hours=4) else 900 if span <= pd.Timedelta(hours=16) else 3600


def time_axis(ax, z0, z1):
    span = z1 - z0
    if span <= pd.Timedelta(hours=1):
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
        fmt = "%H:%M"
    elif span <= pd.Timedelta(hours=4.5):
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        fmt = "%H:%M"
    elif span <= pd.Timedelta(hours=16):
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 1 if span <= pd.Timedelta(hours=8) else 2)))
        fmt = "%H:%M"
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        fmt = "%a %H:%M"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.set_xlim(z0, z1)
    d0, d1 = z0.strftime("%a %d %b"), (z1 - pd.Timedelta(seconds=1)).strftime("%a %d %b")
    ax.set_xlabel(f"time of day ({d0})" if d0 == d1 else f"time ({d0} - {d1})")


def bin_word(bin_s):
    return {300: "5 minutes", 900: "15 minutes", 3600: "hour"}[bin_s]


def per_bin_counts(ev, k, z0, z1, bin_s):
    return M.counts(ons_of(ev, k, z0, z1), z0, z1, bin_s)


def occupancy(ev, k, z0, z1, bin_s):
    """% of each bin the detector was ON."""
    on, end, *_ = M.intervals(ev, k, z0 - pd.Timedelta(hours=12), z1)
    nb = int(np.ceil((z1 - z0).total_seconds() / bin_s))
    occ = np.zeros(nb)
    s = (np.maximum(on, np.datetime64(z0)) - np.datetime64(z0)) / np.timedelta64(1, "s")
    e = (np.minimum(end, np.datetime64(z1)) - np.datetime64(z0)) / np.timedelta64(1, "s")
    for a, b in zip(s, e):
        if b <= a:
            continue
        i0, i1 = int(a // bin_s), min(int(np.ceil(b / bin_s)), nb)
        for i in range(i0, i1):
            occ[i] += min(b, (i + 1) * bin_s) - max(a, i * bin_s)
    tt = z0 + pd.to_timedelta(np.arange(nb) * bin_s + bin_s / 2, unit="s")
    return tt, 100 * occ / bin_s


def fast_share(ev, k, z0, z1, bin_s, thr=0.3, min_n=5):
    """per bin: % of ONs that came back within thr s of the previous OFF (bins with < min_n ONs = gap)."""
    x = ev[ev.EventId.isin([81, 82]) & (ev.Parameter == k) & (ev.Timestamp >= z0) & (ev.Timestamp < z1)]
    x = x.sort_values(["Timestamp", "EventId"], ascending=[True, False])
    e, t = x.EventId.to_numpy(), x.Timestamp.to_numpy()
    nb = int(np.ceil((z1 - z0).total_seconds() / bin_s))
    is_on = e == 82
    fast = np.r_[False, (e[1:] == 82) & (e[:-1] == 81) & ((t[1:] - t[:-1]) / np.timedelta64(1, "s") < thr)]
    b = np.clip(((t - np.datetime64(z0)) / np.timedelta64(1, "s") // bin_s).astype(int), 0, nb - 1)
    n = np.bincount(b[is_on], minlength=nb).astype(float)
    f = np.bincount(b[fast], minlength=nb).astype(float)
    y = np.where(n >= min_n, 100 * f / np.maximum(n, 1), np.nan)
    tt = z0 + pd.to_timedelta(np.arange(nb) * bin_s + bin_s / 2, unit="s")
    return tt, y


def nights(z0, z1, lo, hi, label):
    out = []
    for day in pd.date_range(z0.normalize() - pd.Timedelta(days=1), z1.normalize(), freq="D"):
        a = day + pd.Timedelta(hours=lo)
        b = day + pd.Timedelta(days=1 if lo > hi else 0, hours=hi)
        if min(b, z1) > max(a, z0):
            out.append((max(a, z0), min(b, z1), label))
    return out


def line_chart(ax, z0, z1, series, ylabel, shade=(), hline=None, sample=None, ylim=None):
    """series = [(label, tt, y, colour, width)]; shade = [(a, b, label)]; sample = (t0, t1) dotted when inside."""
    for a, b, _ in shade:
        ax.axvspan(max(a, z0), min(b, z1), color=SHADE, lw=0, zorder=0)
    npts = max(len(s[1]) for s in series)
    for lab, tt, y, col, lw in series:
        ax.plot(pd.to_datetime(tt), y, color=col, lw=lw, label=lab, zorder=3,
                marker="o" if npts <= 60 else None, ms=4)
    h, lab = ax.get_legend_handles_labels()
    if hline is not None:
        ax.axhline(hline[0], color=MARK, ls="--", lw=1.8, zorder=2)
        h.append(Line2D([], [], color=MARK, ls="--", lw=1.8))
        lab.append(hline[1])
    if shade:
        h.append(Patch(color=SHADE))
        lab.append(shade[0][2])
    if sample is not None:
        drew = False
        for t in sample:
            if z0 < t < z1:
                ax.axvline(t, color=SAMPLE_C, ls=":", lw=2, zorder=2)
                drew = True
        if drew:
            h.append(Line2D([], [], color=SAMPLE_C, ls=":", lw=2))
            lab.append("start / end of the sample the check looked at")
    ax.set_ylabel(ylabel)
    time_axis(ax, z0, z1)
    style(ax)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        top = np.nanmax([np.nanmax(s[2]) if np.isfinite(s[2]).any() else 0 for s in series] +
                        [hline[0] if hline else 0, 1])
        ax.set_ylim(0, top * 1.08)
    ax.legend(h, lab, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)


def count_lines(ax, ev, d, partner, pl, z0, z1, bin_s, **kw):
    tt, x = per_bin_counts(ev, d, z0, z1, bin_s)
    ser = [(f"det {d}", tt, x, THIS, 2.6)]
    if partner is not None:
        _, y = per_bin_counts(ev, partner, z0, z1, bin_s)
        ser.append((pl, tt, y, PART, 2.0))
    line_chart(ax, z0, z1, ser, f"actuations per {bin_word(bin_s)}", **kw)


def timeline(ax, ev, z0, z1, dets, p, sample):
    show_g = p is not None and np.isfinite(p) and (z1 - z0) <= pd.Timedelta(hours=3)
    if show_g:
        ax.axvspan(z0, z1, color=RBAND, lw=0, zorder=0)
        for a, b in M.greens(ev, p, z0, z1):
            if b > z0 and a < z1:
                ax.axvspan(max(a, z0), min(b, z1), color=GBAND, lw=0, zorder=0)
    span_d = (z1 - z0).total_seconds() / 86400
    n = len(dets)
    for j, (k, lab, col) in enumerate(dets):
        on, end, *_ = M.intervals(ev, k, z0 - pd.Timedelta(hours=12), z1)
        m = (end > np.datetime64(z0)) & (on < np.datetime64(z1))
        s = np.maximum(on[m], np.datetime64(z0))
        e = np.minimum(end[m], np.datetime64(z1))
        st = mdates.date2num(pd.to_datetime(s))
        w = np.maximum(mdates.date2num(pd.to_datetime(e)) - st, span_d / 2500)
        ax.broken_barh(list(zip(st, w)), (n - 1 - j - 0.3, 0.6), color=col, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab, _ in dets][::-1])
    ax.set_ylim(-0.7, n - 0.3)
    time_axis(ax, z0, z1)
    style(ax)
    ax.grid(False)
    h, lab = [Patch(color="#777777")], ["thick bar = detector ON"]
    if show_g:
        h += [Patch(color=GBAND), Patch(color=RBAND)]
        lab += [f"phase {int(p)} green", f"phase {int(p)} not green"]
    drew = False
    for t in sample:
        if z0 < t < z1:
            ax.axvline(t, color=SAMPLE_C, ls=":", lw=2, zorder=4)
            drew = True
    if drew:
        h.append(Line2D([], [], color=SAMPLE_C, ls=":", lw=2))
        lab.append("sample start / end")
    ax.legend(h, lab, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    return show_g


def dist_lines(ax, edges, labels, series, cut, cut_label, xlabel, unit):
    x = np.arange(len(labels))
    for lab, v, col, lw in series:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        h, _ = np.histogram(np.clip(v, edges[0], edges[-1] - 1e-9), edges)
        ax.plot(x, 100 * h / max(len(v), 1), color=col, lw=lw, marker="o", ms=7, label=f"{lab} ({len(v):,} {unit})",
                zorder=3)
    ax.axvline(cut + 0.5, color=MARK, ls="--", lw=1.8, zorder=2)
    ax.text(cut + 0.55, 0.97, cut_label, color=MARK, transform=ax.get_xaxis_transform(), va="top", fontsize=12,
            fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of that detector's total")
    style(ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)


# ------------------------------------------------------------------ one chart
def chart(r, x, g, ev, partner, pkind, path):
    key, d, p = r.check, int(r.detector), r.pred_phase
    t0, t1 = r.t0, r.t1
    L = t1 - t0
    cov = coverage(ev)
    smp = (t0, t1)
    ctx = max(1.5 * H1, L * 0.5)                         # context either side of a whole-sample check
    pl = None
    if partner is not None:
        pl = f"det {partner} " + {"partner": "(partner)", "same": "(same phase)", "busy": "(busiest other detector)"}[pkind]
    f = (lambda k: float(x.get(k, np.nan))) if x is not None else (lambda k: np.nan)  # noqa: E731
    fig, ax = plt.subplots(figsize=(13, 5.8))
    note = ""
    ons = ons_of(ev, d, t0, t1)
    whole = [(t0, t1, "the sample the check looked at")]
    if key == "stuck":
        a, b = pd.Timestamp(r.ep_t0), pd.Timestamp(r.ep_t1)
        m = max(H1, (b - a) * 0.5)
        z0, z1 = clip(a - m, b + m, cov)
        title = f"Det {d} stayed ON {dur_txt(f('ep_dur'))} without a break ({hm(a)}-{b:%H:%M})"
        if (z1 - z0) <= pd.Timedelta(hours=3):
            dets = [(d, f"det {d}", THIS)] + ([(partner, pl.replace(" (", "\n("), PART)] if partner is not None else [])
            timeline(ax, ev, z0, z1, dets, p, smp)
            ax.axvspan(a, b, fill=False, edgecolor=MARK, lw=2, ls="--", zorder=5)
            note = "Red dashed box = the long ON. Before and after it you can see its normal ON / OFF pattern."
        else:
            bs = 300 if (z1 - z0) <= pd.Timedelta(hours=6) else 900
            tt, y = occupancy(ev, d, z0, z1, bs)
            ser = [(f"det {d}", tt, y, THIS, 2.6)]
            if partner is not None:
                ser.append((pl, *occupancy(ev, partner, z0, z1, bs), PART, 2.0))
            line_chart(ax, z0, z1, ser, f"% of each {bin_word(bs)}\nthe detector was ON", [(a, b, "the long ON")],
                       sample=smp, ylim=(0, 105))
            note = "A detector that is stuck ON sits at 100 %; a working one goes up and down with traffic."
    elif key == "g_stuck":
        z0, z1 = clip(t0 - ctx, t1 + ctx, cov)
        bs = 300 if (z1 - z0) <= pd.Timedelta(hours=4) else 900
        tt, y = occupancy(ev, d, z0, z1, bs)
        ser = [(f"det {d}", tt, y, THIS, 2.6)]
        if partner is not None:
            ser.append((pl, *occupancy(ev, partner, z0, z1, bs), PART, 2.0))
        line_chart(ax, z0, z1, ser, f"% of each {bin_word(bs)}\nthe detector was ON", whole, hline=(90, "limit 90 %"),
                   ylim=(0, 105))
        title = f"Det {d} was ON {pct(g.frac_time_on, .90)} of the {M.dur(g.hours)} sample (limit 90 %)"
    elif key == "chatter":
        z0, z1 = clip(t0 - ctx, t1 + ctx, cov)
        bs = max(300, auto_bin(z1 - z0))
        tt, y = fast_share(ev, d, z0, z1, bs)
        ser = [(f"det {d}", tt, y, THIS, 2.6)]
        if partner is not None:
            ser.append((pl, *fast_share(ev, partner, z0, z1, bs), PART, 2.0))
        line_chart(ax, z0, z1, ser, "% of its ONs that came back\nwithin 0.3 s of switching OFF", whole,
                   hline=(30, "limit 30 %"), ylim=(0, 105))
        title = f"Det {d}: {pct(f('chat_frac'), .30)} of its ONs came back within 0.3 s of switching OFF (limit 30 %)"
        note = (f"One point per {bin_word(bs)}; periods with fewer than 5 ONs are left out. "
                f"{f('n_on'):,.0f} actuations in the sample.")
    elif key == "dropout":
        a = t0 + pd.Timedelta(seconds=int(r.drop_b0) * 300)
        b = t0 + pd.Timedelta(seconds=int(r.drop_b1) * 300)
        m = max(2 * H1, b - a)
        z0, z1 = clip(a - m, b + m, cov)
        bs = 300 if (z1 - z0) <= pd.Timedelta(hours=6) else 900
        count_lines(ax, ev, d, partner, pl, z0, z1, bs, shade=[(a, b, "silent stretch")], sample=smp)
        own = int(((ons >= np.datetime64(a)) & (ons < np.datetime64(b))).sum())
        title = (f"Det {d} counted {'nothing' if own == 0 else own} for {M.dur((b - a).total_seconds() / 3600)} "
                 f"({hm(a)}-{b:%H:%M})")
        po = len(ons_of(ev, partner, a, b)) if partner is not None else 0
        if po > 0:
            title += f" while det {partner} counted {po:,}"
        else:
            title += f", where about {f('drop_lam'):.0f} were expected"
            if partner is not None:
                note = (f"Det {partner} is shown for comparison; the expected count comes from " +
                        ("the other detectors on its phase." if x.get("drop_ref") == "phase" else "the rest of the signal."))
    elif key == "volume":
        k = int(np.argmax(M.counts(ons, t0, t1, 300)[1]))
        a = t0 + pd.Timedelta(seconds=300 * k)
        b = a + pd.Timedelta(minutes=5)
        z0, z1 = clip(a - 2 * H1, b + 2 * H1, cov)
        z0 = a - pd.Timedelta(seconds=300 * int((a - z0).total_seconds() // 300))      # bins aligned with the check's
        count_lines(ax, ev, d, partner, pl, z0, z1, 300, shade=[(a, b, "the busiest 5 minutes")],
                    hline=(150, "limit 150"), sample=smp)
        title = f"Det {d} had {f('max5'):.0f} actuations in one 5-minute period (limit 150)"
    elif key in ("level", "choppy", "partner", "g_low"):
        z0, z1 = clip(t0 - ctx, t1 + ctx, cov)
        bs = auto_bin(z1 - z0)
        if key == "choppy":
            bs = max(bs, 900)
        sh, sm = whole, None
        if key == "level":
            a = t0 + pd.Timedelta(seconds=int(r.level_b) * 300)
            sh, sm = [(a, t1, "after the drop (in the sample)")], smp
            title = (f"From {hm(a)} det {d} counted only {pct(f('level_ratio'), .15)} of its earlier share of the "
                     f"traffic (limit 15 %)")
        elif key == "choppy":
            who = "the other detectors on its phase" if x.get("ref_kind") == "phase" else "the rest of the signal"
            title = f"Det {d}'s 15-min counts jump up and down far more than traffic explains"
            note = (f"The swings are {f('chop15'):.1f}x what {who} would explain; healthy detectors stay under "
                    f"{f('chop_lim'):.1f}x.")
        elif key == "partner":
            pn = int(round(g.c_n / max(g.c_ratio, 1e-9)))
            lim = float(np.exp(-g.p_lim))
            title = f"Det {d} counted {int(g.c_n):,}, its partner det {partner} counted {pn:,} ({pct(g.c_ratio, lim)})"
            note = (f"Counts are for the shaded sample. Healthy {M.FN.get(g.pred_function, '?')} / "
                    f"{M.FN.get(g.pfn, '?')} pairs go down to {pct(lim, lim)}. Proposed check, not in the package.")
        else:
            title = f"Det {d} had only {int(g.n_on)} actuation{'' if int(g.n_on) == 1 else 's'} in {M.dur(g.hours)}"
            if partner is not None:
                title += f"; det {partner} had {len(ons_of(ev, partner, t0, t1)):,}"
            note = "Counts are for the shaded sample. Under 20 a day: the classifier gives no answer for it."
        count_lines(ax, ev, d, partner, pl, z0, z1, bs, shade=sh, sample=sm)
    elif key == "night_drop":
        z0, z1 = cov
        count_lines(ax, ev, d, partner, pl, z0, z1, 3600, shade=nights(z0, z1, 21, 5, "night (21:00-05:00)"),
                    sample=smp)
        ref = str(x.get("night_ref"))
        title = f"At night det {d} counted {f('night_n'):.0f} where about {f('night_exp'):.0f} were expected"
        note = (f"Expected from det {partner}, which it follows by day. " if "tracks" in ref else
                f"Expected from {ref}; det {partner} shown for comparison. ")
        note += (f"It counted {pct(f('night_ratio'), .19)} of what was expected (limit 19 %). Numbers are for the "
                 f"night inside the dotted sample; the other nights are shown for context.")
    elif key in ("night_day", "corr"):
        z0, z1 = cov
        rest = ev[ev.EventId.eq(82) & ev.Parameter.ne(d) & ev.Parameter.le(64) & (ev.Timestamp >= z0)
                  & (ev.Timestamp < z1)].Timestamp.to_numpy()
        tt, xx = per_bin_counts(ev, d, z0, z1, 3600)
        _, yy = M.counts(rest, z0, z1, 3600)
        ser = [(f"det {d}", tt, 100 * xx / max(xx.sum(), 1), THIS, 2.6),
               ("all other detectors at the signal (combined)", tt, 100 * yy / max(yy.sum(), 1), REST, 2.0)]
        sh = nights(z0, z1, 0, 5, "night (00:00-05:00)") if key == "night_day" else []
        line_chart(ax, z0, z1, ser, "% of its actuations\n(whole period shown) in each hour", sh, sample=smp)
        if key == "night_day":
            title = (f"Det {d} counted {f('night_day'):.1f}x as many per hour at night as by day "
                     f"(rest of the signal: {f('sig_night_day'):.2f}x)")
        else:
            title = f"Det {d}'s hourly counts do not rise and fall with the rest of the signal"
            note = f"Correlation {f('corr'):.2f}; normally about {f('corr_exp'):.2f}. "
        note += ("Both lines are scaled to their own total so their shapes can be compared. Numbers are for the day "
                 "between the dotted lines; the rest is context.")
    elif key == "rapid":
        ioi = np.diff(np.sort(ons)) / np.timedelta64(1, "s")
        ser = [(f"det {d}", ioi, THIS, 2.6)]
        if partner is not None:
            ser.append((pl, np.diff(np.sort(ons_of(ev, partner, t0, t1))) / np.timedelta64(1, "s"), PART, 2.0))
        dist_lines(ax, [0, .5, 1, 2, 3, 5, 10, 30, 1e9],
                   ["under 0.5", "0.5-1", "1-2", "2-3", "3-5", "5-10", "10-30", "over 30"], ser, 1, "1 s",
                   "seconds from one actuation to the next", "actuations")
        title = f"Det {d}: {pct(f('ioi_lt1'))} of its actuations came less than 1 s after the previous one"
        note = (f"That is {f('rapid'):.2f}x the most that healthy detectors like it reach (limit 1x). "
                f"Each line = the share of that detector's actuations in each gap range, over the sample.")
    elif key == "short_on":
        ser = [(f"det {d}", M.intervals(ev, d, t0, t1)[4], THIS, 2.6)]
        if partner is not None:
            ser.append((pl, M.intervals(ev, partner, t0, t1)[4], PART, 2.0))
        dist_lines(ax, [0, .15, .25, .55, 1.05, 2.05, 5.05, 10.05, 1e9],
                   ["0.1", "0.2", "0.3-0.5", "0.6-1", "1.1-2", "2.1-5", "5.1-10", "over 10"], ser, 1, "0.2 s",
                   "how long each ON lasted (seconds)", "ONs")
        title = f"Det {d}: {pct(f('short2'), .44)} of its ONs lasted 0.2 s or less (healthy stay under 44 %)"
        note = (f"It is not set to pulse: its median ON is {f('med_dur'):.1f} s. Each line = the share of that "
                f"detector's ONs in each length range, over the sample.")
    else:
        raise ValueError(key)
    fig.suptitle(title, x=0.01, ha="left", fontsize=16, fontweight="bold", color=INK)
    sub = f"Signal {r.signal}   |   model: {r.model_says}   |   check: {M.CHECKS[key][0]}"
    lines = textwrap.wrap(note, 135) if note else []
    fig.text(0.01, 0.905, "\n".join([sub] + lines), ha="left", va="top", fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.89 - 0.04 * len(lines)))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return title


def main():
    C2.chart = chart                                   # same row loop / partner choice as h82_charts2.main
    C2.M.OUT  # noqa: B018
    C2.main()
    src = M.OUT / "titles_charts2.csv"
    if src.exists() and "--only" not in sys.argv:
        src.replace(M.OUT / "titles_charts3.csv")


if __name__ == "__main__":
    main()
