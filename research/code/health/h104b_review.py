"""Note 104b: review/health_review_v3.xlsx rebuilt in a short format (same 25 rows, same order as note 104).

User feedback on the note-104 sheet: too verbose, the same explanation on every line, charts unclear.  Now:
  * shared explanations ONCE in a short header block; each row one short "why" line (rule + deciding number);
  * every chart shows EVERY detector on the flagged detector's predicted phase as its own line, labelled
    "det 12 P4 Advance L1" (function, phase and lane = classifier outputs; lane from research lanes/lane_output.py,
    lane 1 = the busiest lane of the phase), the flagged detector thick; top = real 15-min counts, bottom = real % of
    each 15 min ON; count-based checks add the check's expected line (dashed, legend says how it is computed); the
    flagged period is shaded; nothing scaled.  Rows about ON lengths (N1', N3) add a panel with every ON's length.
  * the user's entries in the existing sheet (Answer / Comment) are carried over by (signal, detector), and the old
    sheet is backed up first (review/_backup/).
Rows = h104_review.select (seed 104), so nothing is re-chosen.  Saved w40 events only; hi-res log + classifier outputs.

    python h104b_review.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lanes"))
import h104_review as V3  # noqa: E402
import lane_output as LO  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402

RV, C3, M, O = V3.RV, V3.C3, V3.M, V3.O
FN = {"Advance": "Advance", "Presence": "Presence", "Count": "Count", "Yellow_Red": "Yellow-red", "Mid": "Mid",
      "Bike": "Bike", "Other": "Other"}
CHK = dict(V3.CHK)
XL, CH, DC, OUT = V3.XL, V3.CH, V3.DC, V3.OUT
H1 = pd.Timedelta(hours=1)
THIS = "#1f4e9c"
PAL = ["#eb6834", "#2a9d8f", "#a03ca0", "#c49a00", "#7a7a75", "#d1495b", "#5c8001", "#8c564b", "#17becf", "#e377c2",
       "#3b3b98", "#9e9e2a"]
SHADE, INK, MUTED, MARK = "#fff1c2", "#1a1a1a", "#555555", "#c62828"
LANE_DIR = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work")) / "lanes" / "model"
PM = None


# ------------------------------------------------------------------ data
def lanes_of(r, ev, I, t0, t1):
    """classifier lanes for the sample window: {det: '1' | '1,2' | ''}."""
    global PM
    if PM is None:
        PM = LO.PairModel(LANE_DIR)
    e = ev[(ev.Timestamp >= t0) & (ev.Timestamp < t1)].assign(DeviceId=r.DeviceId)
    pr = pd.DataFrame({"DeviceId": r.DeviceId, "Detector": I.index.astype(int),
                       "phase_pred": I.phase.to_numpy(), "function_pred": I.fn.to_numpy()})
    _, de = LO.lanes(e, pr, model_dir=PM, start=t0, end=t1)
    return dict(zip(de.Detector.astype(int), de.lanes.fillna("").astype(str)))


def lane_txt(ln, short=True):
    if not ln:
        return ""
    k = ln.split(",")
    if short:
        return "L" + "+".join(k)
    return ("lane " if len(k) == 1 else "lanes ") + "+".join(k)


def dlabel(I, ln, k, bad=False):
    p = I.phase.get(k, np.nan)
    s = f"det {k} " + (f"P{int(p)} " if np.isfinite(p) else "") + FN.get(I.fn.get(k), "?")
    if ln.get(k):
        s += " " + lane_txt(ln[k])
    return s + (" (flagged bad)" if bad else "")


def det_cell(I, ln, d):
    p = I.phase.get(d, np.nan)
    s = f"det {d} · " + (f"P{int(p)} " if np.isfinite(p) else "no phase · ") + FN.get(I.fn.get(d), "?")
    return s + (f" · {lane_txt(ln[d], False)}" if ln.get(d) else "")


def phase_dets(I, d, ix):
    p = I.phase.get(d, np.nan)
    o = [k for k in I.index if k != d and k in ix and np.isfinite(p) and I.phase.get(k) == p]
    rank = {"Advance": 0, "Count": 1, "Presence": 2, "Yellow_Red": 3, "Mid": 4, "Other": 5, "Bike": 6}
    return [d] + sorted(o, key=lambda k: (rank.get(I.fn.get(k), 9), k))


def traffic_ref(I, d, dets):
    bad = set(I.index[I.status.eq("bad")])
    c = [k for k in dets if k != d and k not in bad and I.fn.get(k) in ("Advance", "Count")]
    return c or [k for k in dets if k != d and k not in bad]


def plus(ks):
    return "+".join(map(str, ks))


# ------------------------------------------------------------------ drawing
def draw(path, title, panels, z0, z1, shade, sample):
    """panels = list of dicts(kind='lines'|'dur', ylabel, series=[(label, tt, y, col, lw, ls)], extra=[(label, tt, y,
    col, lw, ls)], hlines=[(y, label)], ylim, dur=(st, du, mo, d))."""
    hr = [1.0 if p["kind"] == "lines" else 0.7 for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 3.6 * sum(hr) + 1.2), sharex=True,
                             gridspec_kw=dict(height_ratios=hr))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        for a, b in shade[0]:
            ax.axvspan(max(a, z0), min(b, z1), color=SHADE, lw=0, zorder=0)
        hs, ls_ = [], []
        if p["kind"] == "lines":
            first = ax is axes[0]
            for lab, tt, y, col, lw, ls in p["series"]:
                ln_, = ax.plot(pd.to_datetime(tt), y, color=col, lw=lw, ls=ls, zorder=4 if lw > 2 else 3)
                if first:
                    hs.append(ln_)
                    ls_.append(lab)
            if not first:
                hs.append(Line2D([], [], color=MUTED, lw=1.3))
                ls_.append("same detectors and colors as the top panel")
            def mx(y):
                y = np.asarray(y, float)
                return np.nanmax(y) if np.isfinite(y).any() else 0
            real = max([mx(s[2]) for s in p["series"]] + [h[0] for h in p.get("hlines", [])] + [1])
            for lab, tt, y, col, lw, ls in p.get("extra", []):
                ln_, = ax.plot(pd.to_datetime(tt), y, color=col, lw=lw, ls=ls, zorder=4)
                hs.append(ln_)
                if not p.get("ylim") and mx(y) > 1.5 * real:
                    lab += f" (runs off the top of the chart, up to {mx(y):.0f})"
                ls_.append(lab)
            for y, lab, col in p.get("hlines", []):
                ax.axhline(y, color=col, ls="--", lw=1.6, zorder=2)
                hs.append(Line2D([], [], color=col, ls="--", lw=1.6))
                ls_.append(lab)
            top = max([real] + [min(mx(s[2]), 1.5 * real) for s in p.get("extra", [])])
            ax.set_ylim(*(p.get("ylim") or (0, top * 1.08)))
        else:
            st, du, mo, d = p["dur"]
            q = mo == 0
            ax.scatter(st[q], np.maximum(du[q], 0.05), s=7, color=THIS, alpha=.5, zorder=3)
            hs.append(Line2D([], [], marker="o", ls="", color=THIS, ms=4))
            ls_.append(f"det {d}: one dot per ON")
            if (~q).any():
                ax.scatter(st[~q], np.maximum(du[~q], 0.05), s=24, marker="x", color=MARK, zorder=4)
                hs.append(Line2D([], [], marker="x", ls="", color=MARK, ms=6))
                ls_.append("ON logged again with no OFF between")
            ax.set_yscale("log")
            ax.set_ylim(0.05, max(10, np.nanmax(du) * 1.5) if len(du) else 10)
            ax.set_yticks([0.1, 0.2, 1, 10, 100, 1000])
            ax.set_yticklabels(["0.1", "0.2", "1", "10", "100", "1000"])
        if shade[0] and ax is axes[0]:
            hs.append(Patch(color=SHADE))
            ls_.append(shade[1])
        if sample is not None:
            drew = False
            for t in sample:
                if z0 < t < z1:
                    ax.axvline(t, color=MUTED, ls=":", lw=1.6, zorder=2)
                    drew = True
            if drew and ax is axes[0]:
                hs.append(Line2D([], [], color=MUTED, ls=":", lw=1.6))
                ls_.append("start / end of the sample checked")
        ax.set_ylabel(p["ylabel"], fontsize=11)
        C3.time_axis(ax, z0, z1)
        C3.style(ax)
        ax.legend(hs, ls_, frameon=False, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=9.5)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(path, dpi=95, bbox_inches="tight")
    plt.close(fig)


def series_for(I, ln, dets, d, tt, Y, ix):
    bad = set(I.index[I.status.eq("bad")]) - {d}
    out = []
    for j, k in enumerate(dets):
        if k == d:
            out.append((dlabel(I, ln, k), tt, Y[ix[k]], THIS, 3.0, "-"))
        else:
            out.append((dlabel(I, ln, k, k in bad), tt, Y[ix[k]], PAL[(j - 1) % len(PAL)], 1.3,
                        ":" if k in bad else "-"))
    return out


def bin_spans(times, bs):
    return [(pd.Timestamp(t), pd.Timestamp(t) + pd.Timedelta(seconds=bs)) for t in times]


# ------------------------------------------------------------------ one row
def build_row(i, r, sig):
    d = int(r.detector)
    ev = RV.events(r.DeviceId)
    t0, t1, h = V3.wwin(r)
    cov0, cov1 = ev.Timestamp.min().floor("15min"), ev.Timestamp.max().ceil("15min")
    I = V3.info(r)
    ln = lanes_of(r, ev, I, t0, t1)
    k, rule = r.kind, r.v2rule
    bs = 900
    z0, z1 = (t0, t1) if h >= 24 else (max(cov0, t0 - 3 * H1), min(cov1, t1 + 3 * H1))
    if k == "KEEP" and rule == "R4":
        bs = 300
    exp_cnt, exp_occ, hl, shade, slab, dur = [], [], [], [], "", None
    why, chk = "", ""
    smp = f"{'24 h' if h >= 24 else '3 h'} {t0:%a %d}" + ("" if h >= 24 else f" {t0:%H:%M}")
    ixw, ttw, nw, ow = RV.bins(ev, t0, t1, 900)
    dets_all = phase_dets(I, d, ixw)
    ref = traffic_ref(I, d, dets_all)
    sc = pd.to_numeric(r.s_choppy, errors="coerce")
    x96 = {}

    def share_exp(mask_src, tt_src, n_src, ix_src, refs, tt_dst, n_dst, ix_dst):
        own = np.nansum(n_src[ix_src[d]][mask_src])
        tot = np.nansum(n_src[[ix_src[q] for q in refs]][:, mask_src])
        s = own / max(tot, 1e-9)
        return s, s * np.nansum(n_dst[[ix_dst[q] for q in refs]], 0)

    if k == "KEEP" and rule == "R9":
        p1 = pd.read_parquet(M.IN96 / "health.parquet", filters=[("DeviceId", "==", r.DeviceId),
                                                                 ("window", "==", r.window)]).set_index("detector")
        q = p1.loc[d]
        a = t0 + pd.Timedelta(seconds=300 * int(q.drop_b0))
        b = t0 + pd.Timedelta(seconds=300 * int(q.drop_b1))
        mates = [m for m in p1.index if m != d and p1.pred_phase.get(m) == p1.pred_phase.get(d)]
        brok = [m for m in mates if p1.status.get(m) == "bad"]
        z0, z1, bs = a - 2 * H1, b + 2 * H1, 300
        shade, slab = [(a, b)], "where the old check said 'silent'"
        x96 = dict(a=a, b=b, mates=mates, brok=brok)
    if k == "KEEP" and rule == "R4":
        z0, z1 = max(cov0, t0 - 3 * H1), min(cov1, t1 + 3 * H1)
    ix, tt, n, occ = RV.bins(ev, z0, z1, bs)
    dets = [q for q in dets_all if q in ix]
    tts = pd.to_datetime(tt)
    bw = "5 min" if bs == 300 else "15 min"

    # ---- per-kind evidence
    if k == "KEEP" and rule == "R2" and np.isfinite(sc) and sc >= .35 or (k == "KEEP" and rule == "R7"):
        cb = V3.choppy_bins(ev, r, I)
        top = np.argsort(cb["con"])[::-1][:3]
        shade = [(pd.Timestamp(cb["tt"][q]) - pd.Timedelta(minutes=7.5), pd.Timestamp(cb["tt"][q]) +
                  pd.Timedelta(minutes=7.5)) for q in top]
        slab = "the 3 15-min periods the check objects to most"
        src = f"det {plus(cb['ref'])}" if cb["kind"] != "signal" else "the whole signal"
        exp_cnt = [(f"expected det {d} = traffic of {src} x det {d}'s share over the 2 h around", cb["tt"],
                    np.where(cb["e"] > 0, cb["e"], np.nan), THIS, 2.0, "--")]
        chk = "erratic counts"
        if rule == "R2":
            why = (f"R2: long zone whose time ON follows phase traffic (r={r.c_occ_ref:.2f}); erratic "
                   f"{cb['D']:.1f} vs limit {float(r.chop_lim):.1f}")
        else:
            why = f"R7: its only finding and borderline (erratic {cb['D']:.1f} vs limit {float(r.chop_lim):.1f}) -> watch"
    elif k == "KEEP" and rule == "R2":                       # doesn't follow traffic
        m = np.ones(len(ttw), bool)
        s, e = share_exp(m, ttw, nw, ixw, ref, tt, n, ix)
        exp_cnt = [(f"expected det {d} = traffic of det {plus(ref)} x det {d}'s share over the day", tt, e, THIS, 2.0,
                    "--")]
        day = t0.normalize()
        shade, slab = [(day, day + 5 * H1)], "night 00:00-05:00"
        chk = CHK["corr"]
        why = (f"R2: counts follow the signal r={float(r['corr']):.2f} (usual {float(r.corr_exp):.2f}), but time ON "
               f"follows phase traffic (r={r.c_occ_ref:.2f})")
    elif k == "KEEP" and rule == "R3":
        cp = t0 + pd.Timedelta(minutes=5 * int(float(r.level_b)))
        m = pd.to_datetime(ttw) < cp
        s, e = share_exp(m, ttw, nw, ixw, ref, tt, n, ix)
        exp_cnt = [(f"expected det {d} = traffic of det {plus(ref)} x det {d}'s share before {cp:%H:%M}", tt, e, THIS,
                    2.0, "--")]
        shade, slab = [(cp, t1)], "after the drop"
        bf, af = pd.to_datetime(ttw) < cp, pd.to_datetime(ttw) >= cp
        ob, oa = np.nanmean(ow[ixw[d]][bf]), np.nanmean(ow[ixw[d]][af])
        chk = CHK["level"]
        why = (f"R3: count share fell to {float(r.level_ratio):.0%} of before (limit 15 %), but its time ON rose "
               f"({ob:.0f} % -> {oa:.0f} % ON)")
    elif k == "KEEP" and rule == "R4":
        j = int(np.nanargmax(n[ix[d]] * ((tts >= t0) & (tts < t1))))
        a = pd.Timestamp(tt[j]) - pd.Timedelta(seconds=150)
        shade, slab = [(a, a + pd.Timedelta(minutes=5))], "the busiest 5 min"
        hl = [(150, "old limit: 150 per 5 min", MARK), (200, "limit for a detector spanning 2 lanes: 200", "#5c8001")]
        chk = CHK["volume"]
        why = (f"R4: {float(r.max5):.0f} in 5 min is within the 2-lane limit (200); 'too-fast actuations' is "
               f"borderline -> watch (R7)")
    elif k == "KEEP" and rule == "R9":
        mates = [q for q in x96["mates"] if q in ix and q in ixw]
        m = np.ones(len(ttw), bool)
        s, e = share_exp(m, ttw, nw, ixw, mates, tt, n, ix)
        exp_cnt = [(f"old expected det {d} = det {plus(mates)} (incl. bad ones) x det {d}'s share over the day", tt, e,
                    THIS, 2.0, "--")]
        chk = CHK["dropout"]
        mins = (x96["b"] - x96["a"]).total_seconds() / 60
        hm_ = [q for q in x96["mates"] if q not in x96["brok"]]
        why = (f"R9: the old yardstick was det {plus(x96['mates'])}, " + ("all of them" if not hm_ else
               f"of which {plus(x96['brok'])}") + f" flagged bad; without them the {mins:.0f}-min night lull at "
               f"{x96['a']:%H:%M} is not flagged")
    elif k in ("I1", "I4") or rule in ("R1", "V1"):
        a, b = pd.Timestamp(r.ep_t0), pd.Timestamp(r.ep_t1)
        shade, slab = [(a, b)], "the long ON"
        chk = CHK["stuck"]
        mn = float(r.ep_dur) / 60
        if k == "I4" or rule == "V1":
            why = (f"R1b: ON {mn:.0f} min in {r.ep_refx:.1f}x usual traffic, {r.pre_occ:.0%} ON the hour before, time "
                   f"ON follows traffic (r={r.c_occ_ref:.2f})")
            if rule == "V1":
                why += f"; counts r={float(r['corr']):.2f} cleared by R2"
        else:
            fail = []
            if r.n_hpeer > 0 and not (r.ep_phx_h >= 1.5):
                fail.append(f"healthy peers only {r.ep_phx_h:.1f}x their usual time ON (queue needs 1.5x)")
            if r.n_hpeer > 0 and not (r.c_occ_hpeer >= .7):
                fail.append(f"its time ON matches theirs r={r.c_occ_hpeer:.2f} (needs .70)")
            if r.n_hpeer == 0:
                fail.append("no healthy peer on its phase")
            if r.light_full > 0:
                fail.append(f">= 90 % ON in {int(r.light_full)} light-traffic 15 min")
            why = f"R1': not a queue - " + "; ".join(fail)
    elif k == "I2" or rule == "N1":
        st, du, mo = V3.ons_pkg(ev, d, z0, z1)
        dur = (st, du, mo, d)
        mm = (st >= t0) & (st < t1)
        md = float(np.median(du[mm]))
        chk = CHK["occ_hi"]
        if md <= .25:
            lng = mm & (du > 1)
            shade = [(pd.Timestamp(s).floor("15min"), pd.Timestamp(s).floor("15min") + pd.Timedelta(minutes=15))
                     for s in st[lng]]
            slab = "15 min with an ON over 1 s"
            why = (f"N1': pulse zone (median ON {md:.1f} s) with {int(lng.sum())} ONs over 1 s (longest "
                   f"{du[mm].max():.0f} s)")
        else:
            why = f"N1': normal-mode zone (median ON {md:.1f} s), now compared only with normal-mode zones"
        if r.fn in M.COUNT_T and r.n3_n >= 2:
            dbar = float(r.dbar)
            exp_occ = [(f"expected time ON det {d} = its count x its usual ON ({dbar:.1f} s)", tt,
                        np.minimum(n[ix[d]] * dbar / bs * 100, 100), THIS, 2.0, "--")]
    elif k in ("I3", "I3b"):
        tm, _, S = V3.spike_marks(r, tt, occ[ix[d]])
        sb = 300 if r.window.startswith("m30") else 900
        shade = [(pd.Timestamp(t) - pd.Timedelta(seconds=sb / 2), pd.Timestamp(t) + pd.Timedelta(seconds=sb / 2))
                 for t in tm]
        slab = "15 min ON far longer than its count explains"
        dbar = float(r.dbar)
        exp_occ = [(f"expected time ON det {d} = its count x its usual ON ({dbar:.1f} s)", tt,
                    np.minimum(n[ix[d]] * dbar / bs * 100, 100), THIS, 2.0, "--")]
        st, du, mo = V3.ons_pkg(ev, d, z0, z1)
        dur = (st, du, mo, d)
        chk = CHK["occspk"] + " (new)"
        why = f"N3: {float(r.n3_exc):.0f} min ON not explained by its count (limit {float(r.n3_lim):.0f})"
        Om = pd.read_parquet(OUT / "n3_missing_off.parquet")
        Om = Om[(Om.DeviceId == r.DeviceId) & (Om.window == r.window) & (Om.detector == d)]
        if len(Om) and Om.t_missoff.sum() / 60 >= 1:
            why += f"; {Om.t_missoff.sum() / 60:.0f} min of it = ON logged again with no OFF"
    elif k == "I5":
        hr_ = pd.to_datetime(ttw).hour
        m = (hr_ >= 7) & (hr_ < 19)
        s, e = share_exp(m, ttw, nw, ixw, ref, tt, n, ix)
        exp_cnt = [(f"expected det {d} = traffic of det {plus(ref)} x det {d}'s daytime (07-19) share", tt, e, THIS,
                    2.0, "--")]
        day = t0.normalize()
        shade, slab = [(day, day + 5 * H1)], "night 00:00-05:00"
        nt = hr_ < 5
        a1, b1 = np.nansum(nw[ixw[d]][nt]) / 5, np.nansum(nw[ixw[d]][m]) / 12
        chk = CHK["night_rel"] + " (new)"
        why = (f"N5: its night rate = {a1 / max(b1, 1e-9):.0%} of its day rate vs {float(r.sig_night_day):.0%} for "
               f"the signal (limit 3x)")

    # ---- charts
    fnn = FN.get(I.fn.get(d), "?")
    pp = I.phase.get(d, np.nan)
    who = f"{sig} det {d} (P{int(pp)} {fnn}" + (f" {lane_txt(ln[d])}" if ln.get(d) else "") + ")"
    title = f"{who}: {chk.replace(' (new)', '')}, {bw} counts and time ON"
    panels = [dict(kind="lines", ylabel=f"actuations per {bw}", series=series_for(I, ln, dets, d, tt, n, ix),
                   extra=exp_cnt, hlines=hl),
              dict(kind="lines", ylabel=f"% of each {bw} ON", series=series_for(I, ln, dets, d, tt, occ, ix),
                   extra=exp_occ, ylim=(0, 102))]
    if dur is not None:
        panels.append(dict(kind="dur", ylabel=f"length of each ON (s)", dur=dur))
    tag = r.kind if r.kind != "KEEP" else f"v2r{int(r.v2n)}"
    png = CH / f"{i:02d}_{tag}_{sig}_d{d}.png"
    draw(png, title, panels, z0, z1, (shade, slab), (t0, t1))
    # day chart: all saved data, 15 min, same detectors
    y0, y1 = cov0, cov1
    ix2, tt2, n2, o2 = RV.bins(ev, y0, y1, 900)
    dets2 = [q for q in dets_all if q in ix2]
    dp = [dict(kind="lines", ylabel="actuations per 15 min", series=series_for(I, ln, dets2, d, tt2, n2, ix2)),
          dict(kind="lines", ylabel="% of each 15 min ON", series=series_for(I, ln, dets2, d, tt2, o2, ix2),
               ylim=(0, 102))]
    dpng = DC / png.name
    draw(dpng, f"{who}: all saved data, 15 min counts and time ON", dp, y0, y1, (shade, slab), (t0, t1))
    old = V3.old_txt(r).split(" (")[0]
    new = V3.ST.get(r.new_status, r.new_status)
    return dict(n=i, signal=sig, det=det_cell(I, ln, d), check=f"{chk} · {smp}", oldnew=f"{old} -> {new}", why=why,
                png=png, dpng=dpng, dev=r.DeviceId, window=r.window, detector=d, lanes=ln.get(d, ""),
                n_dets_phase=len(dets), title=title)


# ------------------------------------------------------------------ sheet
HEADER = [
    "Health review v3 (short form). Is this a real problem? Answer Y / N / ? in the yellow column.",
    "Data: hi-res log Sat 26 Sep 16:15 - Mon 28 Sep 24:00, 2026, plus the classifier's phase / function / lane. "
    "Old = what the current package says. New = the proposed rules. Old -> New: ok / watch (a note only, no status) / "
    "suspect / bad.",
    "Charts: every detector on the same phase is its own line, labelled 'det 12 P4 Advance L1' (lane 1 = busiest lane "
    "of the phase; L1+2 = spans both). This row's detector = thick dark blue. Dotted = a detector itself flagged bad. "
    "Top = real actuations per 15 min; bottom = real % of each 15 min ON. Nothing is scaled. Yellow band = the period "
    "the check flagged. Dashed line = 'expected' (legend says how it is computed).",
    "'Expected' = what this detector would count if it kept its usual share of the traffic on its phase (the other "
    "Advance / Count detectors there). For time ON: its own count x its usual ON length.",
    "Chart = the sample the check looked at (3-h samples: +-3 h around it). Day chart = all saved data.",
    "Checks: erratic counts = 15-min counts jump more than chance around 'expected'. Doesn't follow traffic = counts "
    "do not rise and fall with the signal. Count drops = its share of the phase traffic falls and stays low. Too many in "
    "5 min = more than one lane could pass. Goes silent = no actuations while 'expected' says it should have some. Stuck "
    "on = ON for a long stretch. ON longer than its kind = ONs much longer than other zones of the same type and mode.",
    "New checks: erratic time ON (N3) = counts fine, but ON far longer than its count explains. Busy at night (N5) = "
    "its night / day ratio is 3x the signal's.",
    "Rules: R1' = a long ON is only a queue if the HEALTHY detectors on its phase are ON 1.5x their usual and its time "
    "ON tracks theirs (r >= .70). R1b = long ON in heavy traffic that grew out of its normal daily curve -> watch "
    "(possible congestion). R2 = a long zone whose time ON follows traffic is ok even if its counts look odd. R3 = "
    "count drop is ok if its time ON kept up. R4 = 2-lane detectors get a 200 per 5 min limit. R7 = a single borderline "
    "finding -> watch. R9 = detectors flagged bad are removed from their neighbours' yardstick. N1' = count zones "
    "judged by mode: pulse (median ON <= 0.25 s) vs normal (ON for the whole vehicle).",
]


def write_xl(rows, keep):
    wb = Workbook()
    ws = wb.active
    ws.title = "Cases"
    for j, t in enumerate(HEADER, start=1):
        c = ws.cell(row=j, column=1, value=t)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if j == 1:
            c.font = Font(bold=True, size=12)
        ws.merge_cells(start_row=j, start_column=1, end_row=j, end_column=11)
        ws.row_dimensions[j].height = 18 if j == 1 else (16 * max(1, -(-len(t) // 210)) + 2)
    h0 = len(HEADER) + 2
    head = ["#", "Signal", "Det", "Check", "Old -> New", "Why it changed", "Chart", "Day chart", "Your earlier comment",
            "Answer (Y / N / ?)", "Comment"]
    for j, h in enumerate(head, start=1):
        c = ws.cell(row=h0, column=j, value=h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="2A78D6")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for k, r in enumerate(rows, start=h0 + 1):
        a, cm, er = keep.get((r["signal"], r["detector"]), (None, None, None))
        vals = [r["n"], r["signal"], r["det"], r["check"], r["oldnew"], r["why"], "chart", "day chart", er, a, cm]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=k, column=j, value=v)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row=k, column=7).hyperlink = f"{CH.name}/{r['png'].name}"
        ws.cell(row=k, column=8).hyperlink = f"{DC.name}/{r['dpng'].name}"
        for j in (7, 8):
            ws.cell(row=k, column=j).font = Font(color="0563C1", underline="single")
        ws.cell(row=k, column=9).font = Font(italic=True, color="555555")
        ws.cell(row=k, column=10).fill = PatternFill("solid", fgColor="FFF8DC")
    for col, w in zip("ABCDEFGHIJK", (4, 9, 26, 26, 17, 60, 7, 9, 40, 11, 40)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = ws.cell(row=h0 + 1, column=1)
    wb.save(XL)
    return h0


def old_entries(path):
    """(signal, det) -> (answer, comment, earlier comment) from the existing v3 sheet (note-104 layout)."""
    wb = openpyxl.load_workbook(path)
    ws = wb["Cases"]
    out = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        if row[0] is None or not isinstance(row[0], int):
            continue
        det = int(str(row[3]).split(":")[0].replace("det", "").strip())
        out[(str(row[2]), det)] = (row[11], row[12], row[10])
    return out


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else XL
    keep = old_entries(src)
    R = V3.load()
    v2 = V3.v2_answers()
    rows = V3.select(R, v2)
    order = {"KEEP": 1, "I1": 2, "I4": 3, "I2": 4, "I3": 5, "I3b": 6, "I5": 7}
    rows = sorted(rows, key=lambda r: (order[r.kind], r.v2n if r.v2n else 99))
    nm = RV.names()
    for p_ in (CH, DC):
        p_.mkdir(parents=True, exist_ok=True)
        for f in p_.glob("*.png"):
            f.unlink()
    out = []
    for i, r in enumerate(rows, start=1):
        sig = nm.get(r.DeviceId, r.DeviceId[:8])
        row = build_row(i, r, sig)
        out.append(row)
        print(i, sig, row["det"], "|", row["check"], "|", row["oldnew"], "|", row["why"], "| dets", row["n_dets_phase"])
    assert {(o["signal"], o["detector"]) for o in out} >= set(keep), "a row of the old sheet is missing"
    pd.DataFrame(out).to_csv(OUT / "review_rows_v3b.csv", index=False)
    h0 = write_xl(out, keep)
    print("saved", XL, len(out), "rows; table header at row", h0)


if __name__ == "__main__":
    main()
