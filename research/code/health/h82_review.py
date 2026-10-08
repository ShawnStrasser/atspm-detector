"""Note 82: health review sheet for the user (review/health_review_v1.xlsx + review/health_review_v1_charts/).

One row per example of a health check firing, ~5 per check, from the note-79 w40 windows (Sat 26 - Mon 28 Sep 2026;
30 min / 3 h / 24 h), training signals only (locked_v2 asserted absent).  Checks = every rule the candidate package
(final_v3_candidate_v3) runs: health_core rules that set suspect / bad, the short-ON note, the classifiability gate
(health.py), plus the note-79 partner-ratio candidate (proposed, not in the package).
Inputs to health = the saved note-79 inputs (stg OOF phase / function / lanes of the matching length) = what the
package sees: hi-res log + the classifier's own outputs.  No prints, no config, no fault events.

Selection: from the saved note-79 scores (run.parquet) and gate metrics (h82_gate.py); 3 clear + 2 borderline per
check, spread over signals, sample lengths and sensor types (sensor type from the records: selection spread and a
column for the user only, never a health input).  For the chosen rows the package health_core is re-run on the saved
events with the same inputs to get the statistics behind the sentence; status and rule scores are asserted equal to
the saved run (a reproduction check).  Charts are drawn from the saved events.

    python h82_review.py [--select-only]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h79_run as R  # noqa: E402
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("pkg_hc", H.DCW / "final_v3_candidate_v3" / "health_core.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

OUT = H.DCW / "health82"
XL = H.REPO / "review" / "health_review_v1.xlsx"
CH = H.REPO / "review" / "health_review_v1_charts"
EVD = H.DCW / "health4" / "w40_events"
WG = {"m30": "30 min", "h3_": "3 h", "h24": "24 h"}
DETC, P1, P2, P3, MUTED, INK, SHADE, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#0b0b0b", \
    "#fde2d6", "#e3f4e1"
FN = {"Advance": "Advance", "Presence": "Presence", "Count": "Count", "Yellow_Red": "Yellow-red", "Mid": "Mid",
      "Bike": "Bike", "Other": "Other"}

# check key -> (plain name, what it looks for, effect)
CHECKS = {
    "dropout": ("Goes silent", "Stops counting for a stretch while the other detectors on its phase keep counting",
                "suspect / bad"),
    "stuck": ("Stuck on", "Stays ON without a break for 15 minutes or more", "suspect / bad"),
    "chatter": ("Chattering", "Switches back ON within 0.3 s of switching OFF, again and again", "suspect / bad"),
    "rapid": ("Too-fast actuations", "New actuations start within 1 s of each other far more often than separate "
              "vehicles can", "suspect / bad"),
    "volume": ("Too many in 5 min", "150 or more actuations in one 5-minute period", "suspect / bad"),
    "level": ("Count drops", "From some moment on it counts only a small part of its earlier share of the traffic",
              "suspect / bad"),
    "choppy": ("Erratic counts", "Its 15-minute counts jump up and down in ways the other detectors on its phase "
               "do not", "suspect / bad"),
    "night_day": ("Busier at night", "Counts more at night than by day, unlike the rest of the signal",
                  "suspect / bad"),
    "night_drop": ("Misses vehicles at night", "At night it counts far fewer than the detector it normally follows",
                   "suspect / bad"),
    "corr": ("Doesn't follow traffic", "Its counts do not rise and fall with the rest of the signal over the day",
             "suspect / bad"),
    "short_on": ("Too-short ONs", "Many ONs of 0.2 s or less on a detector that is not set to pulse",
                 "note only"),
    "partner": ("Undercounts vs partner", "Counts far fewer than the same-phase detector it follows best "
                "(PROPOSED - not in the package yet; judged only with 20+ actuations)", "note only (proposed)"),
    "g_low": ("Almost no actuations", "Fewer than 20 actuations a day", "no answer given"),
    "g_stuck": ("ON almost all the time", "ON for 90 % or more of the sample", "no answer given"),
    "g_storm": ("Chatter storm", "600 or more actuations in one minute", "no answer given"),
    "dead": ("Dead channel", "Silent for the whole sample - only possible with a list of channels that should "
             "exist; the package has no such list, so it never runs", "not run"),
}
ORDER = ["stuck", "dropout", "chatter", "rapid", "volume", "short_on", "level", "choppy", "night_drop", "night_day",
         "corr", "partner", "g_low", "g_stuck", "g_storm", "dead"]


# =========================================================================== selection
def partner_limits(sc):
    """note-79 c1: -log(count ratio vs best same-phase partner); limit = p99.5 of presumed-healthy rows per (own,
    partner) predicted function and window length (>= 30 rows, else pooled), w40 prod."""
    X = sc[(sc.period == "w40") & (sc["mode"] == "prod") & sc.seen].copy()
    k = X[["DeviceId", "window", "detector", "pred_function"]].drop_duplicates(["DeviceId", "window", "detector"])
    X = X.merge(k.rename(columns={"detector": "c_partner", "pred_function": "pfn"}), on=["DeviceId", "window",
                                                                                        "c_partner"], how="left")
    X["st"] = -np.log(X.c_ratio.clip(lower=1e-3)).where(X.c_partner.notna())
    X["grp"] = X.pred_function.fillna("?") + ">" + X.pfn.fillna("?")
    hl = X.cls.eq("healthy") & X.tier.eq("N")
    lim = {}
    for w, g in X[hl].groupby("wlen"):
        pooled = g.st.quantile(.995)
        q = g.groupby("grp").st.agg(["count", lambda s: s.quantile(.995)])
        lim[w] = (pooled, {k_: v for k_, (c, v) in q.iterrows() if c >= 30})
    return lim


def candidates():
    run = pd.read_parquet(H.DCW / "health79" / "run.parquet")
    run = run[(run.period == "w40") & (run["mode"] == "prod")].copy()
    sc = pd.read_parquet(H.DCW / "health79" / "scored.parquet")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not run.DeviceId.str.lower().isin(locked).any()
    tech = sc[["DeviceId", "detector", "technology"]].dropna().drop_duplicates(["DeviceId", "detector"])
    run = run.merge(tech, on=["DeviceId", "detector"], how="left")
    run["wg"] = run.window.str[:3]
    run["wlen"] = run.wg.map({"m30": 0.5, "h3_": 3.0, "h24": 24.0})
    # partner function
    k = run[["DeviceId", "window", "detector", "pred_function"]]
    run = run.merge(k.rename(columns={"detector": "c_partner", "pred_function": "pfn"}),
                    on=["DeviceId", "window", "c_partner"], how="left")
    lim = partner_limits(sc)
    st = -np.log(run.c_ratio.clip(lower=1e-3)).where(run.c_partner.notna())
    grp = run.pred_function.fillna("?") + ">" + run.pfn.fillna("?")
    wl = run.wg.map({"m30": "m30", "h3_": "h3", "h24": "h24"})
    L = [lim[w][1].get(g, lim[w][0]) for w, g in zip(wl, grp)]
    run["p_stat"], run["p_lim"] = st.where(run.c_n >= 20), np.array(L, float)   # judged only with >= 20 ONs, as health
    rows = []
    for key in ("dropout", "stuck", "chatter", "rapid", "volume", "level", "choppy", "night_day", "night_drop", "corr"):
        m = run[f"s_{key}"] >= 0.35
        rows.append(run[m].assign(check=key, sev=run.loc[m, f"s_{key}"]))
    m = run.reason.fillna("").str.contains("last 0.2 s or less")
    rows.append(run[m].assign(check="short_on", sev=np.nan))
    m = run.p_stat >= run.p_lim
    rows.append(run[m].assign(check="partner", sev=(run.p_stat - run.p_lim)[m]))
    C = pd.concat(rows, ignore_index=True)
    g = pd.read_parquet(OUT / "gate_w40.parquet")
    g = g[~g.DeviceId.isin(locked)]
    base = run[["DeviceId", "window", "detector", "pred_phase", "pred_function", "technology", "wg", "wlen",
                "status"]].assign(DeviceId=lambda d: d.DeviceId.str.lower())
    g = g.merge(base, on=["DeviceId", "window", "detector"], how="left")
    G = pd.concat([g[g.gate == "near_zero_volume"].assign(check="g_low", sev=lambda d: -d.n_on),
                   g[g.gate == "stuck_on"].assign(check="g_stuck", sev=lambda d: d.frac_time_on),
                   g[g.gate == "chatter_storm"].assign(check="g_storm", sev=lambda d: d.max_on_per_min)])
    C["DeviceId"] = C.DeviceId.str.lower()
    C = pd.concat([C, G], ignore_index=True)
    rates = rate_table(run, g)
    return C, rates, lim


def rate_table(run, g):
    out = {}
    for wg in ("h3_", "h24"):
        x = run[run.wg == wg]
        n = len(x)
        r = {k: 100 * (x[f"s_{k}"] >= .35).sum() / n for k in ("dropout", "stuck", "chatter", "rapid", "volume",
                                                                "level", "choppy", "night_day", "night_drop", "corr")}
        r["short_on"] = 100 * x.reason.fillna("").str.contains("last 0.2 s or less").sum() / n
        r["partner"] = 100 * (x.p_stat >= x.p_lim).sum() / n
        r["any_status"] = 100 * x.status.isin(["suspect", "bad"]).sum() / n
        y = g[g.window.str[:3] == wg]
        for k, gv in (("g_low", "near_zero_volume"), ("g_stuck", "stuck_on"), ("g_storm", "chatter_storm")):
            r[k] = 100 * (y.gate == gv).sum() / max(len(y), 1)
        r["dead"] = 0.0
        out[wg] = r
    return out


def pick(C, per_check=5, seed=81):
    rng = np.random.default_rng(seed)
    C = C.assign(rnd=rng.random(len(C)))
    used_sig, chosen = {}, []
    for key in ORDER:
        x = C[C.check == key]
        if not len(x):
            continue
        x = x[x.n_on.fillna(99).ge(0)]
        # one row per detector (its most severe window), prefer an unused signal
        x = x.sort_values(["sev", "rnd"], ascending=[False, True], na_position="last")
        sel = []

        def take(cands, n, borderline=False):
            cands = cands.sort_values(["sev", "rnd"], ascending=[borderline, True], na_position="last")
            for _ in range(n):
                best, bscore = None, None
                for j, r in cands.iterrows():
                    if any((r.DeviceId == s.DeviceId) for s in sel):
                        continue
                    pen = 3 * used_sig.get(r.DeviceId, 0)
                    pen += sum(s.wg == r.wg for s in sel) + sum(s.technology == r.technology for s in sel)
                    rank = cands.index.get_loc(j)
                    sc_ = pen + rank / max(len(cands), 1) * 4
                    if bscore is None or sc_ < bscore:
                        best, bscore = r, sc_
                if best is None:
                    return
                sel.append(best)
                cands = cands.drop(best.name)

        if key in ("short_on",):
            take(x, per_check)
        else:
            sv = x.sev.dropna()
            if len(x) <= per_check:
                take(x, len(x))
            else:
                hi = sv.quantile(.5) if sv.nunique() > 1 else sv.min()
                take(x[x.sev >= hi], 3)
                take(x[~x.index.isin([s.name for s in sel])], per_check - len(sel), borderline=True)
        for s in sel:
            used_sig[s.DeviceId] = used_sig.get(s.DeviceId, 0) + 1
            chosen.append(s)
    return pd.DataFrame(chosen)


# =========================================================================== data for one example
_EV = {}


def events(dev):
    if dev not in _EV:
        p = EVD / f"DeviceId={dev}"
        if not p.is_dir():
            p = next(q for q in EVD.iterdir() if q.name.lower() == f"deviceid={dev}")
        e = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
        _EV.clear()
        _EV[dev] = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
    return _EV[dev]


def intervals(ev, d, t0, t1):
    """continuous ONs of channel d (ON that starts it -> next OFF), plus every ON time and every OFF->ON gap."""
    x = ev[ev.EventId.isin([81, 82]) & (ev.Parameter == d) & (ev.Timestamp >= t0) & (ev.Timestamp < t1)]
    x = x.sort_values(["Timestamp", "EventId"], ascending=[True, False])
    t, e = x.Timestamp.to_numpy(), x.EventId.to_numpy()
    first = (e == 82) & np.r_[True, e[:-1] != 82]
    on = t[first]
    off = t[e == 81]
    k = np.searchsorted(off, on, "left")
    end = np.where(k < len(off), off[np.minimum(k, len(off) - 1)], np.datetime64(t1))
    ons = t[e == 82]
    gap = np.where((e[1:] == 82) & (e[:-1] == 81), (t[1:] - t[:-1]) / np.timedelta64(1, "s"), np.nan)
    gap = gap[np.isfinite(gap)]
    # raw ON -> next-event-OFF durations (the package's per-actuation durations)
    dur = np.where((e[:-1] == 82) & (e[1:] == 81), (t[1:] - t[:-1]) / np.timedelta64(1, "s"), np.nan)
    dur = dur[np.isfinite(dur)]
    return on, end, ons, gap, dur


def greens(ev, p, t0, t1):
    if p is None or not np.isfinite(p):
        return []
    x = ev[ev.EventId.isin([1, 8]) & (ev.Parameter == int(p)) & (ev.Timestamp >= t0 - pd.Timedelta(minutes=5))
           & (ev.Timestamp < t1)].sort_values("Timestamp")
    out, cur = [], None
    for t, e in zip(x.Timestamp, x.EventId):
        if e == 1:
            cur = t
        elif cur is not None:
            out.append((cur, t))
            cur = None
    return out


def counts(ons, t0, t1, bin_s):
    nb = int(np.ceil((t1 - t0).total_seconds() / bin_s))
    s = (ons - np.datetime64(t0)) / np.timedelta64(1, "s")
    c = np.bincount(np.clip((s // bin_s).astype(int), 0, nb - 1), minlength=nb) if len(s) else np.zeros(nb)
    return t0 + pd.to_timedelta(np.arange(nb) * bin_s + bin_s / 2, unit="s"), c


def best_partner(ev, d, mates, t0, t1):
    """same-phase detector whose 15-min counts follow d best (ties: larger count); short windows: the busiest."""
    if not mates:
        return None
    bin_s = 900 if (t1 - t0) >= pd.Timedelta(hours=2) else 60
    _, x = counts(intervals(ev, d, t0, t1)[2], t0, t1, bin_s)
    best, bk = None, None
    for k in mates:
        _, y = counts(intervals(ev, k, t0, t1)[2], t0, t1, bin_s)
        c = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else -1
        key = (round(float(c), 9), y.sum())
        if bk is None or key > bk:
            best, bk = k, key
    return best


# =========================================================================== charts
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#ececec", lw=0.6)


def tfmt(ax, t0, t1):
    span = t1 - t0
    if span <= pd.Timedelta(minutes=15):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    elif span <= pd.Timedelta(hours=4):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax.set_xlim(t0, t1)


def panel_counts(ax, ev, d, others, t0, t1, bin_s, shade=(), title="", hline=None, scale_to=False):
    """counts per bin of d (blue) and the comparison detectors."""
    for (a, b) in shade:
        ax.axvspan(a, b, color=SHADE, lw=0, zorder=0)
    tt, x = counts(intervals(ev, d, t0, t1)[2], t0, t1, bin_s)
    for k, c in zip(others, (P1, P2, P3)):
        _, y = counts(intervals(ev, k, t0, t1)[2], t0, t1, bin_s)
        ax.step(tt, y, where="mid", color=c, lw=1.2, label=f"d{k}")
    ax.step(tt, x, where="mid", color=DETC, lw=2.4, label=f"d{d} (this detector)")
    if hline is not None:
        ax.axhline(hline, color="#e34948", ls="--", lw=0.9)
    lab = {60: "1 min", 300: "5 min", 900: "15 min", 3600: "hour"}.get(bin_s, f"{bin_s}s")
    ax.set_ylabel(f"actuations per {lab}", fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=9, loc="left", color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="upper left")
    tfmt(ax, t0, t1)
    style(ax)
    return x


def panel_trace(ax, ev, d, others, t0, t1, p, title=""):
    """ON/OFF bars of d and the comparison detectors; green of the predicted phase as a band."""
    for a, b in greens(ev, p, t0, t1):
        ax.axvspan(max(a, t0), min(b, t1), color=GREEN, lw=0, zorder=0)
    dets = [d] + list(others)
    for j, k in enumerate(dets):
        on, end, *_ = intervals(ev, k, t0 - pd.Timedelta(hours=2), t1)
        m = (end > np.datetime64(t0)) & (on < np.datetime64(t1))
        s = np.maximum(on[m], np.datetime64(t0))
        e = np.minimum(end[m], np.datetime64(t1))
        st = mdates.date2num(pd.to_datetime(s))
        w = mdates.date2num(pd.to_datetime(e)) - st
        w = np.maximum(w, (t1 - t0).total_seconds() / 86400 / 1500)
        ax.broken_barh(list(zip(st, w)), (len(dets) - 1 - j - 0.35, 0.7), color=DETC if k == d else (P1, P2, P3)[j - 1])
    ax.set_yticks(range(len(dets)))
    ax.set_yticklabels([f"d{k}" + (" (this)" if k == d else "") for k in dets][::-1], fontsize=8)
    ax.set_ylim(-0.6, len(dets) - 0.4)
    ax.set_title(title + ("   green band = green of its predicted phase" if p is not None and np.isfinite(p) else ""),
                 fontsize=9, loc="left", color=MUTED)
    tfmt(ax, t0, t1)
    style(ax)
    ax.grid(False)


def panel_hist(ax, vals, edges, shade_to, xlabel, title):
    ax.axvspan(edges[0], shade_to, color=SHADE, lw=0, zorder=0)
    for (lab, v), c, lw in zip(vals, (DETC, P1, P2), (2.4, 1.3, 1.3)):
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        h, _ = np.histogram(np.clip(v, edges[0], edges[-1] * 0.999), edges)
        ax.step(edges[:-1], 100 * h / len(v), where="post", color=c, lw=lw, label=f"{lab} ({len(v):,})")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel("% of all", fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=9, loc="left", color=MUTED)
    ax.legend(fontsize=7.5, frameon=False)
    style(ax)


def busiest(ons, t0, t1, width_s, score=None):
    """start of the width_s window with the most ONs (or the highest `score` events)."""
    s = (ons - np.datetime64(t0)) / np.timedelta64(1, "s")
    if score is not None:
        s = s[score]
    if not len(s):
        return t0
    s = np.sort(s)
    k = np.searchsorted(s, s + width_s) - np.arange(len(s))
    a = s[int(np.argmax(k))]
    return t0 + pd.Timedelta(seconds=max(a - width_s * 0.1, 0))


def chart(r, ev, mates, partner, path):
    d, p = int(r.detector), r.pred_phase
    t0, t1 = r.t0, r.t1
    others = ([partner] if partner is not None else []) + [k for k in mates if k != partner][:1]
    hours = (t1 - t0).total_seconds() / 3600
    bin_s = 60 if hours <= 0.5 else 300 if hours <= 3 else 900
    key = r.check
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.6), gridspec_kw={"height_ratios": [1, 1]})
    on, end, ons, gap, dur = intervals(ev, d, t0, t1)
    if key == "stuck":
        sh = [(r.ep_t0, r.ep_t1)] if isinstance(r.get("ep_t0"), pd.Timestamp) else []
        panel_counts(a1, ev, d, others, t0, t1, bin_s, sh, "actuations over the sample; shaded = the long ON")
        for k, c in zip(others[:1], (P1,)):
            o, e_, *_ = intervals(ev, k, t0, t1)
            a2.scatter(o, np.maximum((e_ - o) / np.timedelta64(1, "s"), .1), s=5, color=c, alpha=.5, label=f"d{k}")
        a2.scatter(on, np.maximum((end - on) / np.timedelta64(1, "s"), .1), s=8, color=DETC, label=f"d{d} (this)")
        a2.set_yscale("log")
        a2.axhline(900, color="#e34948", ls="--", lw=0.9)
        a2.set_ylabel("how long each ON lasted (s, log)", fontsize=9, color=MUTED)
        a2.set_title("each dot = one ON; dashed line = 15 minutes", fontsize=9, loc="left", color=MUTED)
        a2.legend(fontsize=7.5, frameon=False)
        tfmt(a2, t0, t1)
        style(a2)
    elif key == "dropout":
        a0 = t0 + pd.Timedelta(seconds=int(r.drop_b0) * 300)
        b0 = t0 + pd.Timedelta(seconds=int(r.drop_b1) * 300)
        panel_counts(a1, ev, d, others, t0, t1, bin_s, [(a0, b0)], "actuations over the sample; shaded = silent stretch")
        for (a, b) in [(a0, b0)]:
            a2.axvspan(a, b, color=SHADE, lw=0, zorder=0)
        for k, c, lw in [(d, DETC, 2.4)] + list(zip(others, (P1, P2), (1.2, 1.2))):
            o = np.sort(intervals(ev, k, t0, t1)[2])
            if len(o):
                a2.plot(pd.to_datetime(o), 100 * np.arange(1, len(o) + 1) / len(o), color=c, lw=lw,
                        label=f"d{k}" + (" (this)" if k == d else ""))
        a2.set_ylabel("% of its actuations so far", fontsize=9, color=MUTED)
        a2.set_title("running total of each detector's actuations: a flat stretch = no actuations", fontsize=9,
                     loc="left", color=MUTED)
        a2.legend(fontsize=7.5, frameon=False)
        tfmt(a2, t0, t1)
        style(a2)
    elif key in ("chatter", "rapid", "short_on", "volume"):
        if key == "chatter":
            tg = ev[ev.EventId.isin([81, 82]) & (ev.Parameter == d) & (ev.Timestamp >= t0) & (ev.Timestamp < t1)]
            tg = tg.sort_values(["Timestamp", "EventId"], ascending=[True, False])
            e = tg.EventId.to_numpy()
            tt = tg.Timestamp.to_numpy()
            fast = np.r_[False, (e[1:] == 82) & (e[:-1] == 81) & ((tt[1:] - tt[:-1]) / np.timedelta64(1, "s") < 0.3)]
            z0 = busiest(tt, t0, t1, 120, fast)
            hv = [(f"d{d} (this)", gap)] + [(f"d{k}", intervals(ev, k, t0, t1)[3]) for k in others[:1]]
            panel_hist(a2, hv, np.logspace(-1, np.log10(600), 36), 0.3,
                       "time from switching OFF to the next ON (s, log); shaded = under 0.3 s",
                       "how fast it switches back ON")
        elif key == "rapid":
            ioi = np.diff(np.sort(ons)) / np.timedelta64(1, "s")
            fast = np.r_[False, ioi < 1.0]
            z0 = busiest(np.sort(ons), t0, t1, 120, fast)
            hv = [(f"d{d} (this)", ioi)] + [(f"d{k}", np.diff(np.sort(intervals(ev, k, t0, t1)[2])) / np.timedelta64(1, "s"))
                                             for k in others[:1]]
            panel_hist(a2, hv, np.logspace(-1, np.log10(600), 36), 1.0,
                       "time from one ON to the next ON (s, log); shaded = under 1 s",
                       "time between successive actuations")
        elif key == "short_on":
            z0 = busiest(ons, t0, t1, 120)
            hv = [(f"d{d} (this)", dur)] + [(f"d{k}", intervals(ev, k, t0, t1)[4]) for k in others[:1]]
            panel_hist(a2, hv, np.logspace(-1.05, np.log10(120), 32), 0.25,
                       "how long each ON lasted (s, log); shaded = 0.2 s or less", "length of each ON")
        else:
            z0 = busiest(ons, t0, t1, 120)
            panel_counts(a2, ev, d, others, t0, t1, 300, title="actuations per 5 min; dashed = 150", hline=150)
        panel_trace(a1, ev, d, others, z0, z0 + pd.Timedelta(minutes=2), p,
                    f"ON / OFF over its busiest 2 minutes ({z0:%a %d %b %H:%M})")
    elif key in ("level", "choppy", "partner", "g_low"):
        b = 3600 if hours >= 24 else 900 if hours >= 3 else 60
        cp = []
        if key == "level" and np.isfinite(r.get("level_b", np.nan)):
            cp = [(t0 + pd.Timedelta(seconds=int(r.level_b) * 300), t1)]
        panel_counts(a1, ev, d, others, t0, t1, b, cp,
                     "actuations over the sample" + ("; shaded = after the drop" if cp else ""))
        for k, c, lw in [(d, DETC, 2.4)] + list(zip(others, (P1, P2), (1.2, 1.2))):
            o = np.sort(intervals(ev, k, t0, t1)[2])
            a2.plot(pd.to_datetime(o), np.arange(1, len(o) + 1), color=c, lw=lw,
                    label=f"d{k}" + (" (this)" if k == d else ""))
        for a, bb in cp:
            a2.axvspan(a, bb, color=SHADE, lw=0, zorder=0)
        a2.set_ylabel("actuations so far", fontsize=9, color=MUTED)
        a2.set_title("running total: the slope is the count rate", fontsize=9, loc="left", color=MUTED)
        a2.legend(fontsize=7.5, frameon=False)
        tfmt(a2, t0, t1)
        style(a2)
    elif key in ("night_day", "night_drop", "corr"):
        night = []
        for day in pd.date_range(t0.normalize() - pd.Timedelta(days=1), t1.normalize(), freq="D"):
            lo, hi = (21, 5) if key == "night_drop" else (0, 5)
            a, b = (day + pd.Timedelta(hours=lo), day + pd.Timedelta(days=1, hours=hi)) if lo > hi else \
                (day + pd.Timedelta(hours=lo), day + pd.Timedelta(hours=hi))
            night.append((max(a, t0), min(b, t1)))
        night = [(a, b) for a, b in night if b > a]
        oth = others
        if key == "corr":
            oth = others
        panel_counts(a1, ev, d, oth, t0, t1, 3600, [(a, b) for a, b in night],
                     "actuations per hour; shaded = night")
        tt, x = counts(ons, t0, t1, 3600)
        if key == "night_drop" and others:
            for k, c in zip(others[:1], (P1,)):
                tt, y = counts(intervals(ev, k, t0, t1)[2], t0, t1, 3600)
                a2.step(tt, np.clip(100 * x / np.maximum(y, 1), 0, 300), where="mid", color=c, lw=1.6,
                        label=f"d{d} as % of d{k}")
            a2.set_ylabel("this detector as % of the other", fontsize=9, color=MUTED)
            a2.set_title("hour by hour, its count as a share of the comparison detector's (capped at 300 %)",
                         fontsize=9, loc="left", color=MUTED)
        else:                                   # night vs day / follows traffic: judged against the whole signal
            rest = ev[ev.EventId.eq(82) & ev.Parameter.ne(d) & ev.Parameter.le(64) & (ev.Timestamp >= t0)
                      & (ev.Timestamp < t1)].Timestamp.to_numpy()
            _, y = counts(rest, t0, t1, 3600)
            a2.step(tt, 100 * y / max(y.sum(), 1), where="mid", color="#8a8a85", lw=1.6, label="all other detectors")
            a2.step(tt, 100 * x / max(x.sum(), 1), where="mid", color=DETC, lw=2.4, label=f"d{d} (this)")
            a2.set_ylabel("% of the day's actuations", fontsize=9, color=MUTED)
            a2.set_title("shape of the day: each hour as a % of the day's total, this detector vs the rest of the "
                         "signal", fontsize=9, loc="left", color=MUTED)
        for a, b in night:
            a2.axvspan(a, b, color=SHADE, lw=0, zorder=0)
        a2.legend(fontsize=7.5, frameon=False)
        tfmt(a2, t0, t1)
        style(a2)
    elif key == "g_stuck":
        panel_trace(a1, ev, d, others, t0, t1, None, "ON / OFF over the whole sample")
        for k, c, lw in [(d, DETC, 2.4)] + list(zip(others, (P1, P2), (1.2, 1.2))):
            o, e_, *_ = intervals(ev, k, t0 - pd.Timedelta(hours=2), t1)
            grid = pd.date_range(t0, t1, freq="1min")
            g_ = grid.to_numpy()
            occ = np.zeros(len(grid) - 1)
            for a, b in zip(o, e_):
                a, b = max(a, g_[0]), min(b, g_[-1])
                if b <= a:
                    continue
                i0 = np.searchsorted(g_, a, "right") - 1
                i1 = np.searchsorted(g_, b, "left")
                for i in range(i0, i1):
                    occ[i] += ((min(b, g_[i + 1]) - max(a, g_[i])) / np.timedelta64(1, "s"))
            a2.step(grid[:-1], 100 * occ / 60, where="post", color=c, lw=lw, label=f"d{k}" + (" (this)" if k == d else ""))
        a2.set_ylabel("% of each minute ON", fontsize=9, color=MUTED)
        a2.set_ylim(0, 125)
        a2.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper left")
        tfmt(a2, t0, t1)
        style(a2)
    fig.suptitle(f"Signal {r.signal}  -  detector {d}  -  model: {r.model_says}  -  check: {CHECKS[key][0]}",
                 fontsize=11.5, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.935, f"{r.saw}", fontsize=8.6, color=MUTED, ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=100)
    plt.close(fig)


# =========================================================================== sentences
def hm(t):
    return pd.Timestamp(t).strftime("%a %H:%M")


def pct(v, lim=None):
    """percentage; one decimal when it sits within 2 points of its limit (so 'at the limit' is visible)."""
    if lim is not None and abs(v - lim) < 0.02:
        return f"{100 * v:.1f} %"
    return f"{100 * v:.0f} %"


def dur(h):
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h".replace(".0 h", " h")


def sentence(key, x, g=None, partner=None):
    f = lambda k: float(x.get(k, np.nan)) if x is not None else np.nan  # noqa: E731
    if key == "stuck":
        s = f"Stayed ON without a break for {f('ep_dur') / 60:.0f} min ({hm(x.ep_t0)}-{pd.Timestamp(x.ep_t1):%H:%M})"
        if np.isfinite(f("ep_lam")) and f("ep_lam") >= 1:
            s += f"; about {f('ep_lam'):.0f} actuations were expected meanwhile"
        if np.isfinite(f("ep_co")) and f("ep_co") > 0:
            s += f"; {int(f('ep_co'))} other detector(s) were held ON at the same time"
        return s + "."
    if key == "dropout":
        h = (f("drop_b1") - f("drop_b0")) * 300 / 3600
        who = "the other detectors on its phase" if x.get("drop_ref") == "phase" else "the rest of the signal"
        s = (f"Silent for {dur(h)} from {hm(x.t0 + pd.Timedelta(seconds=int(f('drop_b0')) * 300))} while about "
             f"{f('drop_lam'):.0f} actuations were expected from {who}")
        if f("co_silent") > 0:
            s += f"; {int(f('co_silent'))} other detector(s) went silent with it"
        return s + "."
    if key == "chatter":
        return (f"{pct(f('chat_frac'), .30)} of its ONs come back within 0.3 s of switching OFF (limit 30 %); "
                f"{f('n_on'):,.0f} actuations in the sample.")
    if key == "rapid":
        nb = int(f("n_burst"))
        rp = f("rapid")
        return (f"{pct(f('ioi_lt1'))} of its actuations start within 1 s of the previous one ({nb} burst"
                f"{'' if nb == 1 else 's'} of 5 or more); that is {rp:.2f}x the most that healthy detectors like it "
                f"reach (limit 1x).")
    if key == "volume":
        return f"{f('max5'):.0f} actuations in one 5-minute period (limit 150)."
    if key == "level":
        return (f"From {hm(x.t0 + pd.Timedelta(seconds=int(f('level_b')) * 300))} it counted only "
                f"{pct(f('level_ratio'), .15)} of its earlier share of the signal's traffic (limit 15 %).")
    if key == "choppy":
        who = "the other detectors on its phase" if x.get("ref_kind") == "phase" else "the rest of the signal"
        return (f"Its 15-min counts swing {f('chop15'):.1f}x more than {who} explain (healthy stay under "
                f"{f('chop_lim'):.1f}x).")
    if key == "night_day":
        return (f"At night (00-05) it counted {f('night_day'):.1f}x its daytime rate; the other detectors "
                f"{f('sig_night_day'):.2f}x.")
    if key == "night_drop":
        ref = str(x.get("night_ref")).replace(", which it tracks", " (which it follows by day)")
        return (f"At night (21-05) it counted {f('night_n'):.0f} where about {f('night_exp'):.0f} were expected from "
                f"{ref}: {pct(f('night_ratio'), .19)} (limit 19 %).")
    if key == "corr":
        return (f"Its counts do not follow the signal over the day (correlation {f('corr'):.2f}, expected about "
                f"{f('corr_exp'):.2f}).")
    if key == "short_on":
        return (f"{pct(f('short2'), .44)} of its ONs last 0.2 s or less although it is not set to pulse (median ON "
                f"{f('med_dur'):.1f} s); healthy detectors stay under 44 %.")
    if key == "partner":
        lim = float(np.exp(-g.p_lim))
        return (f"Counted {int(g.c_n):,} vs {int(round(g.c_n / max(g.c_ratio, 1e-9))):,} on d{int(g.c_partner)}, the "
                f"same-phase detector it follows best: {pct(g.c_ratio, lim)}; healthy {FN.get(g.pred_function, '?')} / "
                f"{FN.get(g.pfn, '?')} pairs go down to {pct(lim, lim)}.")
    if key == "g_low":
        return (f"Only {int(g.n_on)} actuation{'' if int(g.n_on) == 1 else 's'} in {dur(g.hours)} (under 20 a day), "
                f"so the classifier gives no answer for it.")
    if key == "g_stuck":
        return f"ON {pct(g.frac_time_on, .90)} of the {dur(g.hours)} sample, so the classifier gives no answer for it."
    if key == "g_storm":
        return f"{int(g.max_on_per_min)} actuations in one minute, so the classifier gives no answer for it."
    return ""


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    C, rates, lim = candidates()
    print(C.groupby("check").size())
    S = pick(C)
    S.to_parquet(OUT / "chosen.parquet")
    print(S.groupby("check").size(), len(S))
    pd.Series({f"{w}|{k}": v for w, r in rates.items() for k, v in r.items()}).to_csv(OUT / "rates.csv")
    if a.select_only:
        return
    names = pd.concat([pd.read_parquet(H.DCW / "official" / "labels_official.parquet", columns=["DeviceId", "DeviceName"]),
                       pd.read_parquet(H.REPO / "research" / "labels" / "function_labels_v3.parquet",
                                       columns=["DeviceId", "DeviceName"])]).drop_duplicates()
    names = names.dropna(subset=["DeviceName"])
    nm = dict(zip(names.DeviceId.str.lower(), names.DeviceName.astype(str)))
    R.init()
    CH.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    S = S.sort_values(["check", "DeviceId"], key=lambda s: s.map({k: i for i, k in enumerate(ORDER)})
                      if s.name == "check" else s).reset_index(drop=True)
    S["ord"] = S.check.map({k: i for i, k in enumerate(ORDER)})
    S = S.sort_values(["ord", "wlen", "DeviceId"]).reset_index(drop=True)
    mism = []
    for n, g in enumerate(S.itertuples(index=False), start=1):
        g = pd.Series(g._asdict())
        dev, d, w = g.DeviceId, int(g.detector), g.window
        s0, hrs = R.WIN["w40"][w]
        t0 = pd.Timestamp(s0)
        t1 = t0 + pd.Timedelta(hours=hrs)
        ev = events(dev)
        wg = w.split("_")[0]
        ph, fn, ln, pc = R.inputs(wg, dev)
        if (dev, w) not in cache:
            cache[(dev, w)] = hc.health(ev, t0, t1, None, ph or None, fn or None, ln or None, pc or None)
        h = cache[(dev, w)]
        hx = h[h.detector == d]
        x = hx.iloc[0].copy() if len(hx) else None
        if x is not None:
            x["t0"] = t0
            # reproduction check against the saved note-79 run
            if pd.notna(g.get("status")) and x.status != g.status:
                mism.append((dev, w, d, g.status, x.status))
            if g.check in ("dropout", "stuck", "chatter", "rapid", "volume", "level", "choppy", "night_day",
                           "night_drop", "corr"):
                if not np.isclose(float(x.get(f"s_{g.check}", np.nan)), float(g.sev)):
                    mism.append((dev, w, d, g.check, float(g.sev), float(x.get(f"s_{g.check}", np.nan))))
        p = ph.get(d, np.nan)
        fl = {int(k): max(v, key=v.get) for k, v in fn.items()}
        f_ = fl.get(d)
        model = (f"Ph {int(p)} " if np.isfinite(p) else "no phase ") + FN.get(f_, "?")
        mates = [k for k in ph if k != d and ph.get(k) == p and len(intervals(ev, k, t0, t1)[2]) >= 20] \
            if np.isfinite(p) else []
        partner = None
        if g.check == "partner":
            partner = int(g.c_partner)
        elif x is not None and g.check == "night_drop" and "tracks" in str(x.get("night_ref")):
            partner = int(str(x.night_ref).split(",")[0][1:])
        elif x is not None and g.check == "stuck" and np.isfinite(float(x.get("ep_partner", np.nan))):
            partner = int(x.ep_partner)
        elif x is not None and g.check == "dropout" and np.isfinite(float(x.get("drop_partner", np.nan))):
            partner = int(x.drop_partner)
        if partner is None:
            partner = best_partner(ev, d, mates, t0, t1)
        if partner is None:                                    # no same-phase mate: the busiest other detector
            busy = ev[ev.EventId.eq(82) & ev.Parameter.ne(d) & ev.Parameter.le(64) & (ev.Timestamp >= t0)
                      & (ev.Timestamp < t1)].Parameter.value_counts()
            partner = int(busy.index[0]) if len(busy) else None
        mates = [k for k in mates if k != partner]
        saw = sentence(g.check, x, g, partner)
        status = {"ok": "ok", "suspect": "suspect", "bad": "bad", "not_enough_data": "too little data"}.get(
            x.status if x is not None else "", "-")
        if g.check.startswith("g_"):
            status = "no answer"
        sig = nm.get(dev, dev[:8])
        r = pd.Series(dict(check=g.check, detector=d, pred_phase=p, t0=t0, t1=t1, signal=sig, model_says=model,
                           saw=saw))
        if x is not None:
            for c in ("ep_t0", "ep_t1", "drop_b0", "drop_b1", "level_b"):
                r[c] = x.get(c, np.nan)
        png = CH / f"{n:02d}_{g.check}_{sig}_d{d}.png"
        chart(r, ev, mates, partner, png)
        tech = g.technology if isinstance(g.technology, str) else "-"
        rows.append(dict(n=n, check=CHECKS[g.check][0], signal=sig, det=d, model=model,
                         sample=f"{WG[g.wg]}, {t0:%a %d %b %H:%M}", sensor=tech.replace("_", " "), saw=saw,
                         status=status, png=png.name, key=g.check, dev=dev, window=w, sev=g.sev))
        print(n, g.check, sig, d, w, status, "|", saw)
    print("reproduction mismatches:", mism)
    pd.DataFrame(rows).to_csv(OUT / "rows.csv", index=False)
    write_xl(rows, rates)


def write_xl(rows, rates):
    wb = Workbook()
    ws = wb.active
    ws.title = "Examples"
    q = ("Is this detector really faulty in the way the check says? Answer Y (yes, real problem), N (no, false alarm) "
         "or ? (can't tell).")
    ws.append([q])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=12)
    ws.cell(1, 1).font = Font(bold=True, size=12)
    ws.row_dimensions[1].height = 34
    ws.append(["Data: Sat 26 - Mon 28 Sep 2026. Each row = one detector in one sample. 'Model' = what the "
               "classifier thinks the detector is (phase, function). 'Result' = what the package reports for it. "
               "The chart shows the evidence; the other colored lines are detectors on the same phase for comparison."])
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=12)
    ws.row_dimensions[2].height = 30
    hdr = ["#", "Check", "Signal", "Det", "Model", "Sample", "Sensor", "What the check saw", "Result", "Chart",
           "Answer (Y / N / ?)", "Comment"]
    ws.append(hdr)
    link = Font(color="0563C1", underline="single")
    for r in rows:
        ws.append([r["n"], r["check"], r["signal"], r["det"], r["model"], r["sample"], r["sensor"], r["saw"],
                   r["status"], "chart", "", ""])
        c = ws.cell(ws.max_row, 10)
        c.hyperlink = f"{CH.name}/{r['png']}"
        c.font = link
    widths = (4, 16, 7, 5, 13, 16, 7, 62, 9, 7, 10, 30)
    for j, wdt in enumerate(widths):
        ws.column_dimensions[chr(65 + j)].width = wdt
    for c in ws[3]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for row in ws.iter_rows(min_row=1):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A4"
    w2 = wb.create_sheet("Checks")
    w2.append(["Check", "What it looks for", "What happens when it fires", "Per 100 detectors, 3 h sample",
               "Per 100 detectors, 24 h sample", "Examples"])
    nex = pd.Series([r["key"] for r in rows]).value_counts().to_dict()
    for k in ORDER:
        nm_, what, eff = CHECKS[k]
        r3, r24 = rates["h3_"].get(k, 0.0), rates["h24"].get(k, 0.0)
        w2.append([nm_, what, eff, round(r3, 2), round(r24, 2), nex.get(k, 0)])
    w2.append(["Any check (suspect or bad)", "", "", round(rates["h3_"]["any_status"], 2),
               round(rates["h24"]["any_status"], 2), ""])
    w2.append([])
    w2.append(["Counted on every detector with at least one actuation in two 3-h and two 24-h samples of 26-28 Sep "
               "2026 (training signals only). A check that needs a night or 12 h of data cannot fire in a 3-h sample."])
    for j, wdt in enumerate((24, 60, 18, 12, 12, 9)):
        w2.column_dimensions[chr(65 + j)].width = wdt
    for c in w2[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for row in w2.iter_rows(min_row=1):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    w2.freeze_panes = "A2"
    XL.parent.mkdir(exist_ok=True)
    wb.save(XL)
    print("saved", XL, len(rows), "rows")


if __name__ == "__main__":
    main()
