"""Note 108: review/health_review_v3.xlsx rebuilt as a SPOT-CHECK sheet of the v108 health rules (h104b format).

About 2-3 examples of every flag in the v108 rule set (per-type limits = p99.8 of presumed-healthy detectors of the
same model function x lane span; time-of-day profile at p99.5, replacing 'doesn't follow traffic', 'busier at night'
and N5), plus, for the profile check, cases it catches that the old checks missed and cases the old checks caught that
it misses; plus the five detectors named in the orchestrator's message (2B334 d13 = the user's row-1 question,
12032 d41, 08CM405 d37, 2B068 d19, 2B058 d46).  Header once, one line per row; every detector on the predicted
phase is its own line ('det 12 P4 Advance L1'), real 15-min counts + % ON panels; profile rows add the detector's
hourly profile against its type + volume-band normal range.  The old sheet is backed up first and the user's entries
(Answer / Comment / earlier comments) are carried by (signal, detector).  Saved w40 events only; hi-res log +
classifier outputs; training signals only; seed 108.

    python h108_review.py [sheet to read the user's entries from; default = the backup just made]
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h104b_review as V  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402

V3, RV, C3 = V.V3, V.RV, V.C3
DCW = V.LANE_DIR.parents[1]
H8 = DCW / "health108"
TAG = "998p995"
REPO = HERE.parents[2]
XL = REPO / "review" / "health_review_v3.xlsx"
CH = REPO / "review" / "health_review_v3_charts"
DC = REPO / "review" / "health_review_v3_day_charts"
H1 = pd.Timedelta(hours=1)
FN = V.FN
ST = {"ok": "ok", "suspect": "suspect", "bad": "bad", "not_enough_data": "too little data", "watch": "watch"}
NAME = {"stuck": "stuck on", "dropout": "goes silent", "chatter": "chattering", "rapid": "too-fast actuations",
        "volume": "too many in 5 min", "level": "count drops", "choppy": "erratic counts", "night_drop":
        "misses vehicles at night", "night_day": "busier at night", "corr": "doesn't follow traffic",
        "occspk": "erratic time ON", "prof": "time-of-day profile", "occ_hi": "ON longer than its kind",
        "no_yardstick": "no yardstick", "short_on": "too-short ONs"}
# the orchestrator's five + earlier-reviewed detectors that are good examples of a v108 flag (keeps his comments)
FORCED = [("2B334", 13, "h24_a", "Q2u"), ("12032", 41, "h24_b", "Y"), ("08CM405", 37, "h24_a", "stuckF"),
          ("2B068", 19, "h3_b", "D1"), ("2B058", 46, "h24_b", "D1"), ("01064", 42, "h3_b", "choppy"),
          ("2B502", 4, "h24_b", "level"), ("04035", 53, "h3_b", "rapid2"), ("2B530", 60, "h24_b", "stuck_long"),
          ("04016", 20, "h24_b", "N1"), ("2B531", 35, "h24_b", "old_only"), ("2B368", 27, "h24_b", "stuck_long")]
PLAN = [("stuck_short", 1), ("Q1", 1), ("dropout", 1), ("chatter", 2), ("rapid1", 1),
        ("rapid2", 1), ("volume", 2), ("level", 1), ("choppy", 1), ("Q2", 1), ("night_drop", 1), ("occspk", 1),
        ("prof_new", 2), ("prof_both", 1), ("old_only", 1)]


def has(s, k):
    return s.fillna("").astype(str).str.split(",").apply(lambda l: k in l)


def load():
    R = pd.read_parquet(H8 / f"resolved108_q{TAG}.parquet")
    nm = RV.names()
    R["signal"] = R.DeviceId.map(lambda d: nm.get(d, d[:8]))
    return R, nm


def select(R):
    rng = np.random.default_rng(108)
    R = R.assign(rnd=rng.random(len(R)))
    base = R[R.wg.isin(["h3", "h24"]) & (R.n_on >= 20)]
    flagged = base.st8.isin(["suspect", "bad"])
    rows, used = [], set()
    for sig, d, w, kind in FORCED:
        x = R[(R.signal == sig) & (R.detector == d) & (R.window == w)].iloc[0].copy()
        x["kind"] = kind
        rows.append(x)
        used.add(x.DeviceId)
    oldsh = base.f_night_day | base.f_corr | has(base.rules, "N5")
    pr = base.s8_prof >= .35
    oldany = base[[f"f_{k}" for k in ("dropout", "stuck", "chatter", "rapid", "volume", "level", "choppy",
                                      "night_drop", "night_day", "corr", "occspk")]].any(axis=1) | has(base.rules, "N5")
    main = lambda k: base[[c for c in base.columns if c.startswith("s8_")]].idxmax(axis=1) == f"s8_{k}"  # noqa: E731
    C = {
        "stuck_short": base[flagged & has(base.left8, "stuck") & (base.stuck_x < 900) & main("stuck")],
        "stuck_long": base[flagged & has(base.left8, "stuck") & (base.stuck_x >= 900) & main("stuck")],
        "Q1": base[has(base.rules8, "Q1") & base.st8.eq("ok")],
        "dropout": base[flagged & has(base.left8, "dropout") & main("dropout")],
        "chatter": base[flagged & has(base.left8, "chatter") & main("chatter")],
        "rapid1": base[flagged & has(base.left8, "rapid") & main("rapid") & base.span.eq("1")],
        "rapid2": base[flagged & has(base.left8, "rapid") & main("rapid") & base.span.eq("2+")],
        "volume": base[flagged & has(base.left8, "volume") & main("volume") & (base.max5 < 150)],
        "level": base[flagged & has(base.left8, "level") & main("level")],
        "choppy": base[flagged & has(base.left8, "choppy") & main("choppy")],
        "Q2": base[has(base.rules8, "Q2") & base.st8.eq("ok")],
        "night_drop": base[flagged & has(base.left8, "night_drop")],
        "occspk": base[flagged & has(base.left8, "occspk")],
        "prof_new": base[pr & ~oldany & flagged & main("prof")],
        "prof_both": base[pr & oldsh & flagged],
        "old_only": base[oldsh & ~pr & base.st8.isin(["ok", "watch"])],
        "N1": base[has(base.rules8, "N1") & base.st8.eq("watch")],
    }
    return rows, {k: C[k].sort_values("rnd") for k, _ in PLAN}, used


# ------------------------------------------------------------------ text
def tname(r):
    return f"{FN.get(r.fn, r.fn)} {r.span}-lane" if r.span == "1" else f"{FN.get(r.fn, r.fn)} {r.span}-lanes"


def old_txt(r):
    s = {k: float(r[f"s_{k}"]) for k in ("dropout", "stuck", "chatter", "rapid", "volume", "level", "choppy",
                                         "night_drop", "night_day", "corr") if pd.notna(r.get(f"s_{k}"))
         and float(r[f"s_{k}"]) >= .35}
    mc = max(s, key=s.get) if s else None
    return ST.get(r.status, r.status) + (f" ({NAME[mc]})" if mc and r.status in ("suspect", "bad") else "")


def new_txt(r):
    left = [w for w in str(r.left8).split(",") if w and w != "nan"]
    wat = [w for w in str(r.watch8).split(",") if w and w != "nan"]
    st = ST.get(r.st8, r.st8)
    if left and r.st8 in ("suspect", "bad"):
        st += " (" + ", ".join(NAME.get(k, k) for k in left) + ")"
    elif wat and r.st8 == "watch":
        st += " (" + ", ".join(NAME.get(k, k) for k in wat) + ")"
    if "D1" in str(r.dq8):
        st += " + data note"
    return st


def mins(s):
    return f"{s / 60:.0f} min" if s >= 60 else f"{s:.0f} s"


def why(r):
    k, t = r.kind, tname(r)
    if k in ("stuck_short", "stuck_long", "stuckF"):
        w = f"stuck: ON {mins(r.stuck_x)} in one go; limit for {t} {mins(r.stuck_lim8)} (bad at 60 min)"
        if k == "stuckF":
            w += "; v104 called it possible congestion - no longer: its healthy phase mates were not queued"
        return w
    if k == "Q1":
        return (f"Q1: ON {mins(r.stuck_x)} during a queue - its healthy phase mates ON {r.ep_phx_h:.1f}x their usual, "
                f"its time ON tracks theirs (r={r.c_occ_hpeer:.2f}) -> ok")
    if k == "dropout":
        return f"silent: about {float(r.drop_lam):.0f} actuations expected meanwhile (package check, unchanged)"
    if k == "chatter":
        return f"chatter: {r.chat_frac:.0%} of ONs re-trigger < 0.3 s after the OFF; limit for {t} {r.lim_chat_frac:.0%}"
    if k in ("rapid1", "rapid2"):
        return (f"too fast: {r.ioi_lt1:.0%} of ONs within 1 s of the previous one; limit for {t} {r.lim_ioi_lt1:.0%}"
                + (" (2+ lanes: side-by-side cars allowed for)" if r.span == "2+" else ""))
    if k == "volume":
        return f"{r.max5:.0f} in one 5 min; limit for {t} {r.lim_max5:.0f} (was 150 for every detector)"
    if k == "level":
        return f"count share fell to {float(r.level_ratio):.0%} of before (package check, unchanged)"
    if k == "choppy":
        return f"erratic: {r.chop15:.1f} vs limit for {t} {r.lim_chop15:.1f} (healthy median ~1)"
    if k == "Q2":
        cl = ", ".join(NAME.get(c, c) for c in str(r.cleared8).split(",") if c and c != "nan")
        return (f"Q2: '{cl}' cleared - in its busier half its time ON keeps rising while its counts flatten (queue "
                f"pattern) and its time ON tracks its healthy phase mates (r={r.c_occ_like:.2f})")
    if k == "Q2u":
        w = (f"erratic {r.chop15:.1f} (limit for {t} {r.lim_chop15:.1f}); time ON vs its healthy phase mates "
             f"r={r.c_occ_like:.2f}")
        return w + (" - inside the normal range for its type (1 in 20 healthy Presence zones is below .44)"
                    if r.st8 == "ok" else "")
    if k == "night_drop":
        return (f"night 21-05: {float(r.night_n):.0f} actuations where ~{float(r.night_exp):.0f} expected from "
                f"{r.night_ref} (package check, unchanged)")
    if k == "occspk":
        return f"erratic time ON: {r.n3_exc:.0f} min not explained by its count; limit for {t} {r.lim_n3_exc:.0f}"
    if k in ("prof_new", "prof_both", "old_only"):
        w = (f"profile: {r.d_cnt:.0%} of its actuations / {r.d_occ:.0%} of its time ON in other hours than "
             f"normal for {t}, {band(r)} volume (limits {r.d_cnt_lim:.0%} / {r.d_occ_lim:.0%})")
        if k == "prof_new":
            w += "; no old check fired"
        elif k == "old_only":
            w += "; old " + ("busier-at-night" if r.f_night_day else "doesn't-follow-traffic") + " fired, profile not"
        return w
    if k == "N1":
        return f"ON longer than other {FN.get(r.fn, r.fn)} zones at that traffic for >= 30 min -> watch"
    if k == "Y":
        return ("Y: all its phase mates are flagged bad, so nothing on its phase can confirm or clear the old "
                "'silent' finding -> watch")
    if k == "D1":
        sh = r.rep_time_s / max(r.occ * r.hours * 3600, 1)
        return (f"D1: {sh:.0%} of its time ON is ONs logged again with no OFF between ({r.rep_time_s / 60:.0f} min) - "
                f"a logging pattern: data note, no status")
    return ""


def band(r):
    from h108_base import band_of
    return str(band_of(np.array([r.n_on / r.hours]))[0])


# ------------------------------------------------------------------ charts
def draw8(path, title, panels, z0, z1, shade, sample):
    """V.draw with one more panel kind: 'prof' (hour of day on its own x axis)."""
    hr = [1.0 if p["kind"] == "lines" else 0.75 for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 3.6 * sum(hr) + 1.4), gridspec_kw=dict(height_ratios=hr))
    axes = np.atleast_1d(axes)
    tmp = [p for p in panels if p["kind"] != "prof"]
    for ax, p in zip(axes, panels):
        if p["kind"] == "prof":
            hh = np.arange(24)
            b = p["band"]
            hs, ls_ = [], []
            if b is not None and len(b):
                ax.fill_between(b.h, 100 * b.lo, 100 * b.hi, color="#c9c9c9", alpha=.6, lw=0)
                hs.append(Patch(color="#c9c9c9"))
                ls_.append(p["band_lab"])
                l_, = ax.plot(b.h, 100 * b.med, color="#555555", lw=1.6, ls="--")
                hs.append(l_)
                ls_.append("median of healthy detectors like it")
            l_, = ax.plot(hh, 100 * p["own"], color=V.THIS, lw=3)
            hs.append(l_)
            ls_.append(p["own_lab"])
            if p.get("own_o") is not None:
                if b is not None and len(b):
                    l_, = ax.plot(b.h, 100 * b.omed, color="#eb6834", lw=1.4, ls="--")
                    hs.append(l_)
                    ls_.append("median share of time ON, healthy detectors like it")
                l_, = ax.plot(hh, 100 * p["own_o"], color="#eb6834", lw=2.4)
                hs.append(l_)
                ls_.append(p["own_o_lab"])
            ax.set_xlim(0, 23)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xlabel("hour of the day")
            ax.set_ylabel("% of the day's total", fontsize=11)
            C3.style(ax)
            ax.legend(hs, ls_, frameon=False, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=9.5)
            continue
        _one_time_panel(ax, p, z0, z1, shade, sample, first=(p is tmp[0]))
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=V.INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(path, dpi=95, bbox_inches="tight")
    plt.close(fig)


def _one_time_panel(ax, p, z0, z1, shade, sample, first):
    for a, b in shade[0]:
        ax.axvspan(max(a, z0), min(b, z1), color=V.SHADE, lw=0, zorder=0)
    hs, ls_ = [], []
    if p["kind"] == "lines":
        for lab, tt, y, col, lw, ls in p["series"]:
            ln_, = ax.plot(pd.to_datetime(tt), y, color=col, lw=lw, ls=ls, zorder=4 if lw > 2 else 3)
            if first:
                hs.append(ln_)
                ls_.append(lab)
        if not first:
            hs.append(Line2D([], [], color=V.MUTED, lw=1.3))
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
        ax.scatter(st[q], np.maximum(du[q], 0.05), s=7, color=V.THIS, alpha=.5, zorder=3)
        hs.append(Line2D([], [], marker="o", ls="", color=V.THIS, ms=4))
        ls_.append(f"det {d}: one dot per ON")
        if (~q).any():
            ax.scatter(st[~q], np.maximum(du[~q], 0.05), s=24, marker="x", color=V.MARK, zorder=4)
            hs.append(Line2D([], [], marker="x", ls="", color=V.MARK, ms=6))
            ls_.append("ON logged again with no OFF between")
        ax.set_yscale("log")
        ax.set_ylim(0.05, max(10, np.nanmax(du) * 1.5) if len(du) else 10)
        ax.set_yticks([0.1, 0.2, 1, 10, 100, 1000])
        ax.set_yticklabels(["0.1", "0.2", "1", "10", "100", "1000"])
    if shade[0] and first:
        hs.append(Patch(color=V.SHADE))
        ls_.append(shade[1])
    if sample is not None:
        drew = False
        for t in sample:
            if z0 < t < z1:
                ax.axvline(t, color=V.MUTED, ls=":", lw=1.6, zorder=2)
                drew = True
        if drew and first:
            hs.append(Line2D([], [], color=V.MUTED, ls=":", lw=1.6))
            ls_.append("start / end of the sample checked")
    ax.set_ylabel(p["ylabel"], fontsize=11)
    C3.time_axis(ax, z0, z1)
    C3.style(ax)
    ax.legend(hs, ls_, frameon=False, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=9.5)


def info8(r, R):
    x = R[(R.DeviceId == r.DeviceId) & (R.window == r.window)].set_index("detector")
    I = pd.DataFrame({"phase": x.phase, "fn": x.fn, "status": x.st8})
    return I


def build_row(i, r, R):
    d = int(r.detector)
    sig = r.signal
    ev = RV.events(r.DeviceId)
    t0, t1, h = V3.wwin(r)
    cov0, cov1 = ev.Timestamp.min().floor("15min"), ev.Timestamp.max().ceil("15min")
    I = info8(r, R)
    ln = V.lanes_of(r, ev, I, t0, t1)
    k = r.kind
    z0, z1 = (t0, t1) if h >= 24 else (max(cov0, t0 - 3 * H1), min(cov1, t1 + 3 * H1))
    bs = 300 if k == "volume" else 900
    ix, tt, n, occ = RV.bins(ev, z0, z1, bs)
    ixw, ttw, nw, ow = RV.bins(ev, t0, t1, 900)
    dets_all = V.phase_dets(I, d, ixw)
    dets = [q for q in dets_all if q in ix]
    ref = V.traffic_ref(I, d, dets_all)
    exp_cnt, exp_occ, hl, shade, slab, dur, prof = [], [], [], [], "", None, None
    bw = "5 min" if bs == 300 else "15 min"

    def share_exp(mask, refs):
        own = np.nansum(nw[ixw[d]][mask])
        tot = np.nansum(nw[[ixw[q] for q in refs]][:, mask])
        s = own / max(tot, 1e-9)
        return s * np.nansum(n[[ix[q] for q in refs]], 0)

    if k in ("stuck_short", "stuck_long", "stuckF", "Q1"):
        a, b = pd.Timestamp(r.ep_t0) if pd.notna(pd.to_datetime(r.ep_t0, errors="coerce")) else None, None
        if a is not None and pd.notna(pd.to_datetime(r.ep_t1, errors="coerce")):
            b = pd.Timestamp(r.ep_t1)
        else:
            st, du, mo = V3.ons_pkg(ev, d, t0, t1)
            j = int(np.argmax(du))
            a, b = pd.Timestamp(st[j]), pd.Timestamp(st[j]) + pd.Timedelta(seconds=float(du[j]))
        shade, slab = [(a, b)], "the long ON"
    elif k == "dropout" or k == "Y":
        src = r
        if k == "Y":
            p1 = pd.read_parquet(DCW / "health96" / "health.parquet", filters=[("DeviceId", "==", r.DeviceId),
                                                                               ("window", "==", r.window)])
            src = p1[p1.detector == d].iloc[0]
        a = t0 + pd.Timedelta(seconds=300 * int(float(src.drop_b0)))
        b = t0 + pd.Timedelta(seconds=300 * int(float(src.drop_b1)))
        shade, slab = [(a, b)], "where it went silent"
    elif k in ("choppy", "Q2", "Q2u"):
        cb = V3.choppy_bins(ev, r, I)
        top = np.argsort(cb["con"])[::-1][:3]
        shade = [(pd.Timestamp(cb["tt"][q]) - pd.Timedelta(minutes=7.5), pd.Timestamp(cb["tt"][q]) +
                  pd.Timedelta(minutes=7.5)) for q in top]
        slab = "the 3 15-min periods the check objects to most"
        src = "det " + "+".join(map(str, cb["ref"])) if cb["kind"] != "signal" else "the whole signal"
        exp_cnt = [(f"expected det {d} = counts of {src} x det {d}'s share over the 2 h around", cb["tt"],
                    np.where(cb["e"] > 0, cb["e"], np.nan), V.THIS, 2.0, "--")]
    elif k == "level":
        cp = t0 + pd.Timedelta(minutes=5 * int(float(r.level_b)))
        m = pd.to_datetime(ttw) < cp
        exp_cnt = [(f"expected det {d} = traffic of det {'+'.join(map(str, ref))} x det {d}'s share before "
                    f"{cp:%H:%M}", tt, share_exp(m, ref), V.THIS, 2.0, "--")]
        shade, slab = [(cp, t1)], "after the drop"
    elif k == "volume":
        j = int(np.nanargmax(n[ix[d]] * ((pd.to_datetime(tt) >= t0) & (pd.to_datetime(tt) < t1))))
        a = pd.Timestamp(tt[j]) - pd.Timedelta(seconds=150)
        shade, slab = [(a, a + pd.Timedelta(minutes=5))], "the busiest 5 min"
        hl = [(150, "old limit: 150 per 5 min", V.MARK),
              (float(r.lim_max5), f"new limit for {tname(r)}: {r.lim_max5:.0f}", "#5c8001")]
    elif k == "night_drop":
        day = t0.normalize()
        shade, slab = [(day, day + 5 * H1), (day + 21 * H1, day + 24 * H1)], "night 21:00-05:00"
    elif k == "occspk":
        tm, _, S = V3.spike_marks(r, tt, occ[ix[d]])
        shade = [(pd.Timestamp(q) - pd.Timedelta(minutes=7.5), pd.Timestamp(q) + pd.Timedelta(minutes=7.5)) for q in tm]
        slab = "15 min ON far longer than its count explains"
        dbar = float(r.dbar)
        exp_occ = [(f"expected time ON det {d} = its count x its usual ON ({dbar:.1f} s)", tt,
                    np.minimum(n[ix[d]] * dbar / bs * 100, 100), V.THIS, 2.0, "--")]
    if k in ("chatter", "rapid1", "rapid2", "occspk", "N1", "D1"):
        st, du, mo = V3.ons_pkg(ev, d, z0, z1)
        dur = (st, du, mo, d)
    if k in ("prof_new", "prof_both", "old_only"):
        P = pd.read_parquet(H8 / "prof.parquet", filters=[("DeviceId", "==", r.DeviceId), ("window", "==", r.window)])
        p = P[P.detector == d].iloc[0]
        Bd = pd.read_parquet(H8 / "prof_band.parquet")
        bnd = p.band
        b = Bd[(Bd.type == p.type) & (Bd.band == bnd) & (Bd.window == r.window)]
        blab = f"normal range (p2.5-p97.5) of healthy {tname(r)} detectors, {bnd} volume, same day (n={int(b.n.iloc[0])})" \
            if len(b) else ""
        if not len(b):
            b = Bd[(Bd.type.str.startswith(r.fn)) & (Bd.band == bnd) & (Bd.window == r.window)].groupby("h").mean(
                numeric_only=True).reset_index()
            blab = f"normal range of healthy {FN.get(r.fn, r.fn)} detectors, {bnd} volume"
        prof = dict(kind="prof", own=np.array([p[f"s{h_}"] for h_ in range(24)]), band=b, band_lab=blab,
                    own_lab=f"det {d}: share of its {int(p.n_on):,} actuations by hour",
                    own_o=np.array([p[f"o{h_}"] for h_ in range(24)]), own_o_lab=f"det {d}: share of its time ON by hour")
        day = t0.normalize()
        shade, slab = [(day, day + 5 * H1)], "night 00:00-05:00"
    fnn = FN.get(r.fn, "?")
    pp = I.phase.get(d, np.nan)
    who = f"{sig} det {d} (" + (f"P{int(pp)} " if np.isfinite(pp) else "") + fnn + \
        (f" {V.lane_txt(ln[d])}" if ln.get(d) else "") + ")"
    chk = KIND[k]
    title = f"{who}: {chk}, {bw} counts and time ON"
    panels = [dict(kind="lines", ylabel=f"actuations per {bw}", series=V.series_for(I, ln, dets, d, tt, n, ix),
                   extra=exp_cnt, hlines=hl),
              dict(kind="lines", ylabel=f"% of each {bw} ON", series=V.series_for(I, ln, dets, d, tt, occ, ix),
                   extra=exp_occ, ylim=(0, 102))]
    if dur is not None:
        panels.append(dict(kind="dur", ylabel="length of each ON (s)", dur=dur))
    if prof is not None:
        panels.append(prof)
    png = CH / f"{i:02d}_{k}_{sig}_d{d}.png"
    draw8(png, title, panels, z0, z1, (shade, slab), (t0, t1))
    ix2, tt2, n2, o2 = RV.bins(ev, cov0, cov1, 900)
    dets2 = [q for q in dets_all if q in ix2]
    dp = [dict(kind="lines", ylabel="actuations per 15 min", series=V.series_for(I, ln, dets2, d, tt2, n2, ix2)),
          dict(kind="lines", ylabel="% of each 15 min ON", series=V.series_for(I, ln, dets2, d, tt2, o2, ix2),
               ylim=(0, 102))]
    dpng = DC / png.name
    draw8(dpng, f"{who}: all saved data, 15 min counts and time ON", dp, cov0, cov1, (shade, slab), (t0, t1))
    smp = f"{'24 h' if h >= 24 else '3 h'} {t0:%a %d}" + ("" if h >= 24 else f" {t0:%H:%M}")
    return dict(n=i, signal=sig, det=V.det_cell(I, ln, d), check=f"{chk} · {smp}",
                oldnew=f"{old_txt(r)} -> {new_txt(r)}", why=why(r), png=png, dpng=dpng, dev=r.DeviceId,
                window=r.window, detector=d, kind=k, title=title)


KIND = {"stuck_short": "stuck on", "stuck_long": "stuck on", "stuckF": "stuck on", "Q1": "stuck on, queue (Q1)",
        "dropout": "goes silent", "chatter": "chattering", "rapid1": "too-fast actuations",
        "rapid2": "too-fast actuations (2+ lanes)", "volume": "too many in 5 min", "level": "count drops",
        "choppy": "erratic counts", "Q2": "erratic counts, queue (Q2)", "Q2u": "erratic counts",
        "night_drop": "misses vehicles at night", "occspk": "erratic time ON", "prof_new": "time-of-day profile (new)",
        "prof_both": "time-of-day profile", "old_only": "time-of-day profile (old check only)",
        "N1": "ON longer than its kind", "Y": "goes silent, no yardstick", "D1": "data note: ON again without OFF"}

HEADER = [
    "Health spot-check (v108). Is the NEW result right? Answer Y / N / ? in the yellow column.",
    "Data: hi-res log Sat 26 - Mon 28 Sep 2026 + the classifier's phase / function / lanes. Old = the current package; "
    "New = rules with limits taken from what healthy detectors of the SAME TYPE do. Type = the classifier's function x "
    "lanes spanned (1 or 2+). Old -> New: ok / watch (a note only) / suspect / bad.",
    "Limits: for each type, the value only 1 in 500 healthy detectors of that type exceeds (1 in 200 for the profile). "
    "So a Count zone and a Presence zone, or a 1-lane and a 2-lane detector, are no longer held to the same limit.",
    "Charts: every detector on the same phase is its own line ('det 12 P4 Advance L1'; L1+2 = spans both lanes). This "
    "row's detector = thick dark blue; dotted = flagged bad. Top = actuations, bottom = % of time ON. Nothing scaled. "
    "Yellow band = what the check flagged. Dashed = expected. Day chart = all saved data.",
    "Time-of-day profile (new, 24 h samples): the share of its day's actuations in each hour vs healthy detectors of the "
    "same type and volume (low < 20 / medium 20-100 / high > 100 per hour), same day. It replaces 'doesn't follow "
    "traffic' and 'busier at night' (it catches 81 % and 74 % of what they caught).",
    "Rules: Q1 = a long ON is a queue only if its healthy phase mates are ON 1.5x their usual and it tracks them. Q2 = "
    "erratic counts are ok if it shows the queue pattern (as traffic rises its time ON keeps rising while its counts "
    "flatten) and its time ON tracks its healthy phase mates. Y = no healthy phase mate left to judge it -> watch. D1 = "
    "ONs logged again with no OFF: a data note only.",
]


def old_entries(path):
    wb = openpyxl.load_workbook(path)
    ws = wb["Cases"]
    out = {}
    h0 = next(i for i, row in enumerate(ws.iter_rows(values_only=True), start=1) if row[0] == "#")
    for row in ws.iter_rows(min_row=h0 + 1, values_only=True):
        if row[0] is None:
            continue
        det = int(str(row[2]).split("·")[0].replace("det", "").strip())
        out[(str(row[1]), det)] = (row[9], row[10], row[8])
    return out


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
    head = ["#", "Signal", "Det", "Check", "Old -> New", "Why", "Chart", "Day chart", "Your earlier comment",
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
    for col, w in zip("ABCDEFGHIJK", (4, 9, 26, 26, 30, 62, 7, 9, 40, 11, 40)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = ws.cell(row=h0 + 1, column=1)
    wb.save(XL)


def backup():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bk = REPO / "review" / "_backup"
    bk.mkdir(exist_ok=True)
    shutil.copy2(XL, bk / f"health_review_v3_{stamp}.xlsx")
    for p_ in (CH, DC):
        if p_.exists():
            shutil.copytree(p_, bk / f"{p_.name}_{stamp}")
    return bk / f"health_review_v3_{stamp}.xlsx"


def main():
    bk = backup()
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else bk    # the user's entries are read from this sheet
    keep = old_entries(src)
    R, _ = load()
    rows, cand, used = select(R)
    for p_ in (CH, DC):
        p_.mkdir(parents=True, exist_ok=True)
        for f in p_.glob("*.png"):
            f.unlink()
    out = []

    def emit(r):
        row = build_row(len(out) + 1, r, R)
        print(len(out) + 1, row["signal"], row["det"], "|", row["check"], "|", row["oldnew"], "|", row["why"])
        return row
    for r in rows:
        out.append(emit(r))
    for kind, n in PLAN:
        got = 0
        for _, r in cand[kind].iterrows():
            if got == n:
                break
            if r.DeviceId in used:
                continue
            r = r.copy()
            r["kind"] = kind
            row = emit(r)
            # the lane label drawn on the chart must agree with the lane span the limit was chosen for
            if (row["det"].count("+") > 0) != (r.span == "2+"):
                for f in (row["png"], row["dpng"]):
                    f.unlink(missing_ok=True)
                print("   skipped: chart lanes disagree with the lane span of its limit")
                continue
            out.append(row)
            used.add(r.DeviceId)
            got += 1
        if got < n:
            print("short of examples:", kind, got, "of", n)
    # every answered / commented row of the old sheet must be carried
    need = {k for k, v in keep.items() if any(x not in (None, "") for x in v)}
    have = {(o["signal"], o["detector"]) for o in out}
    lost = need - have
    print("old entries with content:", len(need), "carried:", len(need & have), "not in the new rows:", sorted(lost))
    pd.DataFrame(out).to_csv(H8 / "review_rows108.csv", index=False)
    write_xl(out, keep)
    print("saved", XL, len(out), "rows; backup", bk, "; entries read from", src)


if __name__ == "__main__":
    main()
