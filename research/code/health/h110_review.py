"""Note 110: review/health_review_v3.xlsx rebuilt for resolver v110 (h110_resolve) with the note-108b verified-chart
method (h108c_review: every number / time in a row's 'Why' is re-computed from the saved events and drawn on its chart;
QA log per row).  Same 30 rows (the user's 12 entries + 'Our reply' carried cell by cell) + one or two examples for
every check whose behaviour changed in v110.  Old = the package; New = v110; a row whose result changed since the
last sheet (v108) says so.

    python h110_review.py <backup xlsx to read the user's entries from> [rows]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h108c_review as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

hc, V3, V, RV = C.hc, C.V3, C.V, C.RV
DCW = C.DCW
H10 = DCW / "health110"
WK = H10 / "review"
FN, THIS, INK, MARK, MUTED = C.FN, C.THIS, C.INK, C.MARK, C.MUTED
H1, M15 = C.H1, C.M15
tname, fmt_t, span_txt, mins = C.tname, C.fmt_t, C.span_txt, C.mins
NAME = dict(C.NAME, no_yardstick_chop="no yardstick for erratic counts", prof_partial="time-of-day profile (3 h)")
C.NAME.update(NAME)
KIND = dict(C.KIND, Y1="erratic counts, no yardstick", F2="count drops (time of day)", F3="stuck on, several times",
            Q1b="stuck on, long queue test (Q1b)", F5="model unsure of the function", P3="time-of-day profile (3 h)")
NEW_ROWS = [  # (kind, DeviceId, window, detector) - chosen from the v110 changes (h110_resolve, seed 110 samples)
    ("Y1", "6795b5a6-59f2-46c8-a4b5-63622e9fdbc1", "h24_a", 1),
    ("F2", "be117825-4e01-4b23-8873-d170dc0e6812", "h24_b", 23),
    ("F3", "78cdd179-729a-4525-a8db-ad90ebcfccf4", "h24_a", 24),
    ("Q1b", "30feb116-22fc-4615-8b9e-f44991a10ecb", "h24_a", 22),
    ("F5", "f82912a9-a754-4ece-a61c-8a840392ea1b", "h24_b", 49),
    ("P3", "ebece845-b8fe-4738-8b2b-5e5eb8f9033d", "h3_a", 22),
]
EPS = None
B15 = None
TODW = None
PROF = None
DESC = {"ioi_lt1": "start < 1 s", "ioi_lt05": "start < 0.5 s", "burst_frac": "bursts", "chat_frac": "re-trigger < 0.3 s",
        "max5": "5-min count"}
FIXNAME = {"chop15": "erratic counts", "chat_frac": "chattering", "ioi_lt1": "too-fast", "ioi_lt05": "too-fast",
           "burst_frac": "too-fast", "max5": "too many in 5 min", "stuck_x": "stuck on", "n3_exc": "erratic time ON"}


def lim_type(r, col):
    """type whose limit was used for statistic col (F5: the least strict plausible type)."""
    src = r.get(f"lsrc_{col}")
    fn = src if isinstance(src, str) and src else r.fn
    return tname(fn, r.span), (isinstance(src, str) and bool(src))


def f5_text(r):
    if not (isinstance(r.alt_fns, str) and r.alt_fns):
        return ""
    ps = sorted(((float(r[f"p_{k}"]), k) for k in r.alt_fns.split(",")), reverse=True)
    return ("the model is unsure of its function (" + ", ".join(f"{FN.get(k, k)} {100 * p:.0f} %" for p, k in ps) +
            "), so each limit is the least strict of these types")


def relabel(c, cols, fixed=True):
    """F5: the evidence functions of h108c name the model's own type; when a less strict plausible type's limit was
    used, say so in the text, panel title and limit labels (with the own type's limit for comparison)."""
    r = c.r
    shown = [k for k, _ in c.claims if k in cols]
    if shown:
        cols = shown
    alts = {lim_type(r, k) for k in cols}
    if len(alts) > 1:
        c.why[-1] += " (" + "; ".join(f"limit '{DESC.get(k, k)}' for {lim_type(r, k)[0]}" for k in cols) + ")"
        return
    for col in cols:
        lt, alt = lim_type(r, col)
        if not alt:
            continue
        tn = tname(r.fn, r.span)
        own = r.get(f"lim8_{col}")
        o = (f"{100 * own:.1f} %" if col in ("chat_frac", "ioi_lt1", "ioi_lt05", "burst_frac") else f"{own:.0f}")             if own is not None and np.isfinite(own) else "?"
        rep = (f"limits for {lt} (model may be wrong about type; {tn} would be {o})"
               if col != "max5" else f"limit for {lt} (model may be wrong about type; {tn} would be {o})")
        src = f"limits for {tn}" if col != "max5" else f"limit for {tn}"
        c.why[-1] = c.why[-1].replace(src, rep, 1)
        if c.panels and isinstance(c.panels[-1].get("title"), str):
            c.panels[-1]["title"] = c.panels[-1]["title"].replace(src, f"limits for {lt}" if col != "max5" else
                                                                  f"limit for {lt}")
            c.panels[-1]["hl"] = [(y, l.replace(tn, lt), cc) for y, l, cc in c.panels[-1].get("hl", [])]
            if own is not None and np.isfinite(own) and col != "max5":
                c.panels[-1]["hl"].append((100 * own, f"limit for {tn} (the model's top type, {DESC.get(col, col)}): {o}", MUTED))
        c.claims.append(("limit type", lt))
        return


def fast110(c, f):
    C.ev_fast(c, f)
    relabel(c, ("chat_frac",) if f == "chatter" else ("ioi_lt1", "ioi_lt05", "burst_frac"))


_orig_dlp = C.draw_lines_panel


def draw_lines_panel(ax, c, p):
    """h108c lines panel + 'stems2' (every long ON labelled, labels staggered with a leader line) + segment label."""
    from matplotlib.lines import Line2D
    if p.get("stems2"):
        st, du, ins = p["st"], p["du"], p["ins"]
        for s_, d_, i_ in zip(st, du, ins):
            ax.plot([s_, s_], [0, d_], color=THIS if i_ else "#9aa9c9", lw=2.2 if i_ else 1.4)
        ax.scatter(st[ins], du[ins], s=22, color=THIS, zorder=4)
        ax.scatter(st[~ins], du[~ins], s=16, color="#9aa9c9", zorder=4)
        top = max(np.max(du) if len(du) else 1, p["badv"]) * 1.15
        ax.set_ylim(0, top)
        ax.axhline(p["lim"], color=MARK, ls="--", lw=1.6)
        ax.axhline(p["badv"], color="#7b1fa2", ls="--", lw=1.6)
        ax.text(c.z0 + (c.z1 - c.z0) * 0.005, p["lim"], f" limit for {p['tn']}: {p['lim']:.0f} min", color=MARK,
                fontsize=9.5, va="bottom")
        ax.text(c.z0 + (c.z1 - c.z0) * 0.005, p["badv"], f" bad: {p['badv']:.0f} min", color="#7b1fa2", fontsize=9.5,
                va="bottom")
        hs = [Line2D([], [], color=THIS, lw=2.2), Line2D([], [], color="#9aa9c9", lw=1.4),
              Line2D([], [], color=MARK, ls="--"), Line2D([], [], color="#7b1fa2", ls="--")]
        ls_ = [f"det {c.d}: one stem per continuous ON of 1 min or more (in the sample)",
               "same, outside the sample (context only)", "limit (suspect) for one ON / the total",
               "bad (total of the counted ONs)"]
        for q, (s_, d_, lab) in enumerate(sorted(p["labels"], key=lambda z: z[0]), start=1):
            ax.annotate(str(q), (s_, d_), xytext=(0, 5), textcoords="offset points", color=MARK, fontsize=11,
                        fontweight="bold", ha="center", va="bottom")
            hs.append(Line2D([], [], ls="", marker=f"${q}$", color=MARK, ms=9))
            ls_.append(lab.replace(chr(10), " ", 1).replace(chr(10), " - "))
        C.leg(ax, hs, ls_)
        return
    _orig_dlp(ax, c, p)
    if p.get("marks") and not p.get("ylim"):
        ym = max([float(y) for _, y, _ in p["marks"] if np.isfinite(y)] or [0])
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.15 * ym))
    if p.get("seglab"):
        leg = ax.get_legend()
        for t in leg.get_texts():
            if t.get_text() == "average share before / after the change point":
                t.set_text(p["seglab"])


C.draw_lines_panel = draw_lines_panel


# ------------------------------------------------------------------ F1 erratic counts with the v110 reference
def ev_choppy110(c, fired=True):
    r = c.r
    B = c.B5
    dets = [int(d) for d in B["dets"]]
    i = dets.index(c.d)
    a = hc._agg(B["n_on"].astype(float), 3)
    ok = hc._agg(B["cov"][None].astype(float), 3)[0] == 3
    tw = [t for t in c.twins]
    if r.ref110 == "none":
        live = B["n_on"][:, B["cov"]].sum(1) > 0
        refi = [k for k in range(len(dets)) if k != i and live[k] and dets[k] not in tw]
        D = hc.rel_disp(a[i], a[refi].sum(0), ok, 8)[0]
        c.auto.append(("erratic vs whole signal", D, float(r.chop_sig)))
        who = "all other detectors on the signal (mixed functions)"
    else:
        refd = [int(x) for x in str(r.ref110_dets).split("+")]
        refi = [dets.index(k) for k in refd]
        D = hc.rel_disp(a[i], a[refi].sum(0), ok, 8)[0]
        c.auto.append(("erratic chop110", D, float(r.chop110)))
        who = "det " + "+".join(map(str, refd))
    x = np.where(ok, a[i], 0.0)
    rr = np.where(ok, a[refi].sum(0), 0.0)
    h = 8
    X = hc._movsum(x, h) - x
    Rr = hc._movsum(rr, h) - rr
    s = X / np.maximum(Rr, 1e-9)
    e, v = s * rr, s * rr * (1 + s)
    m = ok & (Rr > 0) & (X + x > 0) & ((e + x) > 0)
    tt = c.t0 + pd.to_timedelta(np.arange(len(x)) * 900 + 450, unit="s")
    lim = float(r.lim_chop15)
    tn, alt = lim_type(r, "chop15")
    kind = {"phase": "its phase mates", "same_fn": f"its {FN.get(r.fn, r.fn)} phase mate",
            "same_fn_sig": f"the other {FN.get(r.fn, r.fn)} detectors on the signal (fewer than 2 usable phase mates)",
            "none": "no yardstick"}[r.ref110]
    twm = [t for t in tw if C.np.isfinite(c.I.phase.get(t, np.nan)) and c.I.phase.get(t) == r.phase]
    twtxt = (f"det {' and '.join(map(str, twm))} count the same as det {c.d} (same zone on another input) so they are "
             f"never its yardstick; " if twm else "")
    con = np.where(m, (x - e) ** 2, 0.0)
    top = np.argsort(con)[::-1][:3]
    sd = np.sqrt(np.maximum(v, 0))
    if r.ref110 == "none":
        title = (f"Erratic counts: NO yardstick (no usable phase mate, no other {FN.get(r.fn, r.fn)} on the signal). "
                 f"Against the whole signal it would be {D:.1f} vs limit {lim:.1f} - not scored")
    else:
        title = (f"Erratic counts = {D:.1f} vs {who} ({kind}); limit for {tn} {lim:.1f}"
                 + (" (bad from %.1f)" % (2 * lim) if fired else ""))
    pan = dict(kind="lines", ylabel="actuations per 15 min",
               series=[(f"det {c.d}: actual", tt, np.where(ok, x, np.nan), THIS, 2.6, "-")],
               extra=[(f"expected det {c.d} = its share of {who} over the 2 h around each 15 min", tt,
                       np.where(m, e, np.nan), THIS, 1.8, "--")],
               band=(tt, np.where(m, np.maximum(e - 2 * sd, 0), np.nan), np.where(m, e + 2 * sd, np.nan),
                     "chance range (expected +- 2 x the usual random variation)"),
               marks=[(tt[k], x[k], f"{x[k] - e[k]:+.0f}") for k in top if m[k]],
               mark_lab="the 3 periods that add most to the score (actual - expected)", title=title)
    c.panels.append(pan)
    if r.ref110 == "none":
        w = (f"erratic counts: no yardstick - {twtxt}no other usable phase mate and no other {FN.get(r.fn, r.fn)} "
             f"detector on the signal, so it is not scored (the earlier check silently used the whole signal: {D:.1f} vs limit "
             f"{lim:.1f}) -> watch ('no yardstick')")
    else:
        w = (f"erratic counts {D:.1f} vs limit for {tn} {lim:.1f}; expected = its share of {who} ({twtxt}{kind})")
        if not fired:
            w += " -> not flagged"
    c.why.append(w)
    c.claims += [("erratic", f"{D:.1f}"), ("limit", f"{lim:.1f}")]
    c.shade += [(tt[k] - M15 / 2, tt[k] + M15 / 2) for k in top if m[k]]
    c.slab.append("the 3 periods that add most")


# ------------------------------------------------------------------ F2 count drop, time-of-day aware
def ev_level110(c, fired=True, old=False):
    global B15, TODW
    r = c.r
    b = pd.read_parquet(H10 / "b15.parquet", filters=[("DeviceId", "==", c.dev), ("window", "==", c.w),
                                                      ("detector", "==", c.d)]).sort_values("b")
    if TODW is None:
        TODW = pd.read_parquet(H10 / "tod_w.parquet")
    day = "h24_a" if c.w in ("h24_a", "h3_a") else "h24_b"
    W = TODW[(TODW.type == r.type) & (TODW.band == r.band) & (TODW.window == day) & (TODW.n_w >= 30)].set_index(
        "hour").w
    w = b.hour.map(W).fillna(1.0).to_numpy(float)
    x, S, ok = b.x.to_numpy(float), b.S.to_numpy(float), b.ok.to_numpy(bool)
    tt = c.t0 + pd.to_timedelta(b.b.to_numpy() * 900 + 450, unit="s")
    import h110_resolve as HR
    ratio, llr, cb = HR.level110(x, S, ok, w)
    if np.isfinite(r.lv_ratio110):
        c.auto.append(("level110 ratio", ratio, float(r.lv_ratio110)))
    g = S * w
    if cb < 0:
        # no qualifying cut: show the v108 cut (whole-signal share) and what the detector's own counts did there
        r0, l0, c0 = HR.level110(x, S, ok, np.ones_like(w), min_h_bins=2, min_e=0.0, own_fall=False)
        cut = c0
        a1, a2 = x[:cut][ok[:cut]].mean(), x[cut:][ok[cut:]].mean()
        cp = c.t0 + pd.Timedelta(minutes=15 * cut)
        pan = dict(kind="lines", ylabel="actuations per 15 min",
                   series=[(f"det {c.d}: actual", tt, np.where(ok, x, np.nan), THIS, 2.6, "-")],
                   segs=[(c.t0, cp, a1, f"before {fmt_t(cp)}: {a1:.0f} per 15 min"),
                         (cp, c.t1, a2, f"after: {a2:.0f} per 15 min")],
                   seglab="its average count per 15 min before / after the earlier change point",
                   title=(f"Count drops: no drop - its own counts average {a1:.0f} per 15 min before "
                          f"{fmt_t(cp)} and {a2:.0f} after (a drop needs them to at least halve)"))
        c.panels.append(pan)
        c.why.append(f"count drops: not flagged any more - the earlier 'drop' at {fmt_t(cp, True)} was a fall in its share "
                     f"of the rest of the signal, but its own counts did not fall ({a1:.0f} per "
                     f"15 min before, {a2:.0f} after; a drop needs them to at least halve): the rest of the signal "
                     f"woke up while it kept counting")
        c.claims += [("before", f"{a1:.0f}"), ("after", f"{a2:.0f}")]
        return
    m1, m2 = ok & (np.arange(len(x)) < cb), ok & (np.arange(len(x)) >= cb)
    sh1 = (x[m1].sum() + .5) / max(g[m1].sum(), 1e-9)
    sh2 = (x[m2].sum() + .5) / max(g[m2].sum(), 1e-9)
    a1, a2 = x[m1].mean(), x[m2].mean()
    cp = c.t0 + pd.Timedelta(minutes=15 * cb)
    e = np.where(ok, np.where(np.arange(len(x)) < cb, sh1, sh1) * g, np.nan)
    pan = dict(kind="lines", ylabel="actuations per 15 min",
               series=[(f"det {c.d}: actual", tt, np.where(ok, x, np.nan), THIS, 2.6, "-")],
               extra=[(f"expected at its share before {fmt_t(cp)} = rest of the signal x the hourly pattern of healthy "
                       f"{tname(r.fn, r.span)} detectors", tt, e, THIS, 1.8, "--")],
               segs=[(c.t0, cp, a1, f"before {fmt_t(cp)}: {a1:.0f} per 15 min"),
                     (cp, c.t1, a2, f"after: {a2:.0f} per 15 min")],
               seglab="its average count per 15 min before / after the change point",
               title=(f"Count drops: after {fmt_t(cp)} it counts {ratio:.0%} of what its share before predicts "
                      f"(suspect below 15 %, bad below 5 %); own counts {a1:.0f} -> {a2:.0f} per 15 min"))
    c.panels.append(pan)
    w_ = (f"count drop: after {fmt_t(cp, True)} it counted {ratio:.0%} of what its earlier share predicts (rest of the "
          f"signal x the hourly pattern of healthy {tname(r.fn, r.span)} detectors; suspect below 15 %), own counts "
          f"{a1:.0f} -> {a2:.0f} per 15 min")
    if not fired:
        w_ += " -> not flagged"
    c.why.append(w_)
    c.shade.append((cp, c.t1))
    c.slab.append("after the drop")
    c.claims += [("ratio", f"{ratio:.0%}"), ("before", f"{a1:.0f}"), ("after", f"{a2:.0f}")]


# ------------------------------------------------------------------ F3 / F4 stuck: every episode, queue per episode
def eps_of(c):
    global EPS
    if EPS is None:
        EPS = pd.read_parquet(H10 / "eps_ctx.parquet").join(pd.read_parquet(H10 / "eps_traf.parquet"))
    r = c.r
    E = EPS[(EPS.DeviceId == c.dev) & (EPS.window == c.w) & (EPS.detector == c.d)].copy()
    L = float(r.stuck_lim8)
    E = E[E.dur_s >= L].sort_values("t0")
    ok = (E.n_hpeer_e.fillna(0) > 0) & (E.phx_h_e >= 1.5) & (E.corr_h_e >= 0.70) & (E.refx_e >= 0.5) & \
        (E.light_e.fillna(1) == 0) & (E.trafx_e >= 0.5)
    E["q1"] = ok & (E.dur_s < 3600)
    E["q1b"] = ok & (E.dur_s >= 3600) & (E.cover_e >= 0.75)
    E["shared"] = E.co5 >= 3
    E["count"] = ~E.q1 & ~E.q1b & ~E.shared
    E["what"] = np.where(E.q1, "queue (Q1)", np.where(E.q1b, "long queue (Q1b)", np.where(
        E.shared, np.where(E.trafx_e < 0.5, "shared, no traffic", "shared"), "counts")))
    return E, L


def ev_stuck110(c):
    r = c.r
    E, L = eps_of(c)
    st, du, mo = V3.ons_pkg(c.ev, c.d, c.z0, c.z1)
    en = st + pd.to_timedelta(du, unit="s")
    ins = np.asarray((st >= c.t0) & (st < c.t1))
    badv = max(C.STUCK_BAD, 2 * L)
    tot = float(E.dur_s[E["count"]].sum())
    n = int(E["count"].sum())
    c.auto.append(("stuck total counted (s)", tot, float(r.st_tot_s) if pd.notna(r.st_tot_s) else 0.0))
    c.auto.append(("stuck episodes counted", n, float(r.st_n_ep) if pd.notna(r.st_n_ep) else 0.0))
    tn, alt = lim_type(r, "stuck_x")
    nl = float(r.lim_n_ep) if pd.notna(r.lim_n_ep) else np.nan
    parts = [f"{e.dur_s / 60:.0f} min {span_txt(e.t0, e.t1)}" + ("" if e.what == "counts" else f" [{e.what}]")
             for e in E.itertuples()]
    w = (f"{len(E)} ON(s) over the limit for {tn} ({L / 60:.0f} min): " + "; ".join(parts[:5]) +
         ("; ..." if len(parts) > 5 else "") +
         f". Counted (not a queue, not shared): {n} = {tot / 60:.0f} min in total (suspect from {L / 60:.0f} min, bad "
         f"from {badv / 60:.0f} min")
    if np.isfinite(nl) and nl >= 1:
        w += f"; or more than {nl:.0f} such ONs, the most 1 in 500 healthy {tn} zones have)"
    else:
        w += ")"
    sc = float(r.s8_stuck) if pd.notna(r.s8_stuck) else 0.0
    if n == 0 and (E.shared & ~E.q1 & ~E.q1b).any() and sc >= 0.35:
        w += ("; held at suspect: its long ON(s) were shared with 3 or more other detectors and were not a queue - "
              "a shared event (cabinet, unit or communications) rather than this detector alone")
    if sc == 0.35 and n == 1 and pd.to_numeric(r.s_stuck, errors="coerce") == 0.35:
        w += "; held at suspect: one ON only, and afterwards it counted its usual share again"
    c.why.append(w)
    big = du >= 60
    lab = {pd.Timestamp(e.t0): f"{e.dur_s / 60:.0f} min\n{fmt_t(e.t0)}-{fmt_t(e.t1)}"
           + ("" if e.what == "counts" else f"\n{e.what}") for e in E.itertuples()}
    labels = []
    for s_, d_ in zip(st[big], du[big]):
        k = min(lab, key=lambda t: abs((t - s_).total_seconds())) if lab else None
        if k is not None and abs((k - s_).total_seconds()) < 2:
            labels.append((s_, d_ / 60, lab[k]))
    pan = dict(kind="lines", stems2=True, ylabel="length of each ON (minutes)", st=st[big], du=du[big] / 60,
               ins=ins[big],
               lim=L / 60, badv=badv / 60, tn=tn, labels=labels,
               title=(f"Stuck on: {len(E)} ON(s) over {L / 60:.0f} min; counted {n} = {tot / 60:.0f} min in total "
                      f"(suspect from {L / 60:.0f}, bad from {badv / 60:.0f} min in total)"))
    c.panels.append(pan)
    c.shade += [(e.t0, e.t1) for e in E.itertuples()]
    c.slab.append("the long ON(s)")
    c.claims += [("ONs over limit", f"{len(E)}"), ("counted total", f"{tot / 60:.0f} min"), ("limit", f"{L / 60:.0f} min")]
    return E


def ev_queue110(c, E):
    """queue evidence for the longest episode that was tested against the healthy mates (cleared or not)."""
    r = c.r
    if not len(E):
        return
    e = E.sort_values("dur_s").iloc[-1]          # the longest ON over the limit (cleared or not)
    q = C.queue_context(c, pd.Timestamp(e.t0), pd.Timestamp(e.t1))
    c.auto = [a for a in c.auto if not a[0].startswith("Q1:")]
    if q is None:
        c.why.append("queue test: no healthy phase mate")
        return
    c.auto.append(("Q1 mates x (episode)", q["x"], float(e.phx_h_e)))
    c.auto.append(("Q1 r (episode)", q["r"], float(e.corr_h_e)))
    ms = "+".join(map(str, q["mates"]))
    txt = (f"for its {e.dur_s / 60:.0f}-min ON at {fmt_t(e.t0)}: phase mate(s) not called bad by the first pass, det {ms}, {100 * q['during']:.0f} % "
           f"ON during it vs {100 * q['usual']:.0f} % over the sample ({q['x']:.1f}x; a queue needs 1.5x); outside it "
           f"its % ON vs theirs r = {q['r']:.2f} (needs 0.70)")
    if e.dur_s >= 3600:
        txt += (f"; over 60 min, so the mates must also be queued through it: a mate above the p95 % ON of healthy "
                f"detectors of its type in {100 * e.cover_e:.0f} % of its 15-min periods (needs 75 %)")
    # traffic evidence (orchestrator 2026-10-06): the phase's Advance / Count detectors still counting during the ON
    Bc = c.bctx
    bs = int(Bc.bs.iloc[0])
    tds = [int(k) for k in str(e.traf_dets).split("+") if k]
    tr = Bc[Bc.detector.isin(tds)].groupby("b").n.sum().sort_index()
    ia = int((pd.Timestamp(e.t0) - c.t0).total_seconds() // bs)
    iz = int(np.ceil((pd.Timestamp(e.t1) - c.t0).total_seconds() / bs))
    fa = int(np.ceil((pd.Timestamp(e.t0) - c.t0).total_seconds() / bs))
    fz = int((pd.Timestamp(e.t1) - c.t0).total_seconds() // bs)
    if fz > fa:
        ia, iz = fa, fz
    ins = tr.reindex(range(ia, iz)).dropna()
    dur_n, usu_n = float(ins.mean()), float(tr.mean())
    tx = dur_n / max(usu_n, 1e-9)
    c.auto.append(("Q1 traffic x (episode)", tx, float(e.trafx_e)))
    tlab = "+".join(map(str, tds))
    tsrc = ("its phase's Advance / Count" if any(c.I.fn.get(k) in ("Advance", "Count") for k in tds) else
            "no Advance / Count on its phase: its other phase mates")
    txt += (f"; traffic: det {tlab} ({tsrc}) counted {dur_n:.0f} per 15 min during it vs {usu_n:.0f} "
            f"over the sample ({tx:.2f}x; a queue needs 0.5x - a phase held ON with no counts is no congestion)")
    tt_ = c.t0 + pd.to_timedelta(tr.index.to_numpy() * bs + bs / 2, unit="s")
    c.cnt_extra.append(dict(lab=f"det {tlab} ({tsrc}): total per 15 min", tt=tt_, y=tr.to_numpy(float),
                            col=INK, ls="--", lw=2.0))
    c.cnt_title.append(f"{fmt_t(e.t0)} ON: det {tlab} counted {dur_n:.0f} per 15 min during it vs {usu_n:.0f} usual "
                       f"({tx:.2f}x, needs 0.5x)")
    c.claims += [("traffic during", f"{dur_n:.0f}"), ("traffic usual", f"{usu_n:.0f}")]
    verdict = "-> queue, cleared" if (e.q1 or e.q1b) else "-> not a queue"
    c.why.append(("Q1b " if e.dur_s >= 3600 else "Q1 ") + txt + " " + verdict)
    c.occ_extra.append(dict(lab=f"phase mate(s) not called bad by the first pass, det {ms}: average % ON", tt=q["tt"],
                            y=q["pm"], col=INK,
                            ls="--", lw=2.0))
    c.occ_hl.append((100 * q["usual"], f"mates' average over the sample: {100 * q['usual']:.0f} %", INK))
    c.occ_title.append(f"{fmt_t(e.t0)} ON: mates {100 * q['during']:.0f} % ON during it vs {100 * q['usual']:.0f} % "
                       f"usual ({q['x']:.1f}x, needs 1.5x); r = {q['r']:.2f} (needs 0.70)"
                       + (f"; mates queued in {100 * e.cover_e:.0f} % of it (needs 75 %)" if e.dur_s >= 3600 else ""))
    c.claims += [("mates ratio", f"{q['x']:.1f}x"), ("r", f"{q['r']:.2f}")]
    if e.dur_s >= 3600:
        c.claims.append(("queued cover", f"{100 * e.cover_e:.0f} %"))
        qt = pd.read_csv(H10 / "queue_q95_by_type.csv").set_index("type").q_type
        Bc = c.bctx
        bs = int(Bc.bs.iloc[0])
        ty = c.R_types
        mm = Bc[Bc.detector.isin(q["mates"])].copy()
        mm["q"] = mm.detector.map(lambda k: qt.get(ty.get(int(k)), np.nan))
        qb = set(mm[mm.occ >= mm.q].b.tolist())
        ia = int((pd.Timestamp(e.t0) - c.t0).total_seconds() // bs)
        iz = int(np.ceil((pd.Timestamp(e.t1) - c.t0).total_seconds() / bs))
        bb = np.arange(ia, iz)
        tt = c.t0 + pd.to_timedelta(bb * bs + bs / 2, unit="s")
        y = np.where([b in qb for b in bb], 101.0, np.nan)
        thr = ", ".join(f"det {int(k)} {100 * qt.get(ty.get(int(k)), np.nan):.0f} %" for k in q["mates"])
        c.occ_extra.append(dict(lab=f"15 min of the long ON with a healthy mate queued (% ON above what only 1 in 20 "
                                    f"healthy detectors of its type reach: {thr})", tt=tt, y=y, col=MARK, ls="-",
                                lw=7.0))
        nq = int(np.isfinite(y).sum())
        c.occ_title[-1] += f" [{nq} of {len(bb)} periods marked]"


# ------------------------------------------------------------------ F6 profile on a 3-h sample (watch)
def ev_prof3(c):
    global PROF
    if PROF is None:
        PROF = pd.read_parquet(C.H8 / "prof.parquet")
    r = c.r
    w24, h0 = {"h3_a": ("h24_a", 12), "h3_b": ("h24_b", 6)}[c.w]
    hs = list(range(h0, h0 + 3))
    P3 = pd.read_parquet(H10 / "prof3.parquet")
    me = P3[(P3.DeviceId == c.dev) & (P3.window == c.w) & (P3.detector == c.d)].iloc[0]
    p = PROF[(PROF.window == w24)]
    own = PROF[(PROF.DeviceId == c.dev) & (PROF.window == w24) & (PROF.detector == c.d)].iloc[0]
    n = p[[f"n{h}" for h in hs]].to_numpy(float)
    tot = n.sum(1)
    band = np.where(tot / 3 < 20, "low", np.where(tot / 3 < 100, "medium", "high"))
    g = p[p.hl_prof.to_numpy() & (tot >= 50) & (band == me.band) & (p.type == me.type).to_numpy()]
    if len(g) < 100:
        g = p[p.hl_prof.to_numpy() & (tot >= 50) & (band == me.band) & (p.fn == me.fn).to_numpy()]
    out = {}
    for nm, pre in (("cnt", "n"), ("occ", "o")):
        A = g[[f"{pre}{h}" for h in hs]].to_numpy(float)
        A = A / np.clip(A.sum(1, keepdims=True), 1e-9, None)
        o = own[[f"{pre}{h}" for h in hs]].to_numpy(float)
        o = o / max(o.sum(), 1e-9)
        med = np.median(A, 0)
        out[nm] = dict(own=o, med=med, lo=np.quantile(A, .025, 0), hi=np.quantile(A, .975, 0), d=0.5 * np.abs(o - med).sum())
    c.auto.append(("profile 3h d_cnt", out["cnt"]["d"], float(me.d3_cnt)))
    c.auto.append(("profile 3h d_occ", out["occ"]["d"], float(me.d3_occ)))
    lim = {"cnt": float(me.d3_cnt_lim), "occ": float(me.d3_occ_lim)}
    k = max(out, key=lambda q: out[q]["d"] / lim[q])
    o = out[k]
    what = "actuations" if k == "cnt" else "time ON"
    vb = {"low": "low volume", "medium": "medium volume", "high": "high volume"}[me.band]
    grp = f"{tname(r.fn, r.span)}, {vb}"
    c.panels.append(dict(kind="hour3", hs=hs, own=100 * o["own"], med=100 * o["med"], lo=100 * o["lo"], hi=100 * o["hi"],
                         n=len(g), grp=grp, what=what,
                         title=(f"Time-of-day profile on this 3-h sample ({what}): {100 * o['d']:.0f} % of it sits in "
                                f"other hours than normal for a {grp}; limit {100 * lim[k]:.0f} % - watch only (3 h)")))
    c.why.append(f"3-h profile: {100 * o['d']:.0f} % of its {what} in these 3 hours sit in other hours than healthy "
                 f"{grp} detectors (limit {100 * lim[k]:.0f} %) -> watch only: a profile from under 12 h finds about a "
                 f"quarter of real problems")
    c.claims += [("3h profile", f"{100 * o['d']:.0f} %"), ("limit", f"{100 * lim[k]:.0f} %")]


# ------------------------------------------------------------------ one row
def build(i, rr, R, keep):
    c = C.Row()
    c.r = r = R[(R.DeviceId == rr.dev) & (R.window == rr.window) & (R.detector == rr.detector)].iloc[0]
    c.dev, c.w, c.d, c.kind, c.sig = rr.dev, rr.window, int(rr.detector), rr.kind, rr.signal
    c.ev = RV.events(c.dev)
    c.t0, c.t1, c.h = V3.wwin(r)
    cov0, cov1 = c.ev.Timestamp.min().floor("15min"), c.ev.Timestamp.max().ceil("15min")
    c.z0, c.z1 = max(cov0, c.t0 - 3 * H1), min(cov1, c.t1 + 3 * H1)
    x = R[(R.DeviceId == c.dev) & (R.window == c.w)].set_index("detector")
    c.I = I = pd.DataFrame({"phase": x.phase, "fn": x.fn, "status": x.st8, "status1": x.status})
    c.R_types = x.type.to_dict()
    c.ln = V.lanes_of(r, c.ev, I, c.t0, c.t1)
    x1, x2, ph, ph2, bad1 = C.package(c.ev, c.w, c.dev)
    use2 = c.d not in bad1 and x2 is not x1
    xp = (x2 if use2 else x1).set_index("detector").loc[c.d]
    c.xp, c.x1 = xp, x1.set_index("detector").loc[c.d]
    c.B5 = hc.events_to_bins(c.ev, c.t0, c.t1)
    c.i5, c.refi, c.refk, c.twins = C.refs_for(c.B5, ph2 if use2 else ph, c.d)
    c.b15 = RV.bins(c.ev, c.z0, c.z1, 900)
    c.dets = V.phase_dets(I, c.d, c.b15[0])
    c.bctx = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet", filters=[("DeviceId", "==", c.dev),
                                                                            ("window", "==", c.w)])
    c.panels, c.why, c.claims, c.auto, c.shade, c.slab = [], [], [], [], [], []
    c.cnt_extra, c.cnt_hl, c.cnt_marks, c.cnt_title = [], [], [], []
    c.occ_extra, c.occ_hl, c.occ_title = [], [], []
    c.use5, c.window_only, c.ep, c.reply = False, False, None, ""
    sp = lambda s: [k for k in str(s).split(",") if k and k != "nan"]  # noqa: E731
    left, watch, rules = sp(r.left8), sp(r.watch8), sp(r.rules8)
    left8, rules8 = sp(r.left8_v108), sp(r.rules8_v108)
    s8 = {k: float(r[f"s8_{k}"]) if pd.notna(r.get(f"s8_{k}")) else 0.0 for k in
          ("stuck", "dropout", "chatter", "rapid", "volume", "level", "choppy", "night_drop", "occspk", "prof")}
    tn = tname(r.fn, r.span)
    k = c.kind
    if f5_text(r):
        c.why.append(f5_text(r))
    done = set()
    if k == "Y":
        i_, ref1, k1, _ = C.refs_for(c.B5, ph, c.d)
        C.ev_dropout(c, c.x1, ref1, k1, pass_lab="first pass: ")
        mates = [int(c.B5["dets"][q]) for q in ref1]
        c.cnt_extra[-1]["lab"] = "first pass: " + c.cnt_extra[-1]["lab"]
        C.ev_dropout(c, c.xp, c.refi, "signal", pass_lab="second pass, judged on the rest of the signal: ")
        c.cnt_extra[-1].update(ls=":", col="#7a7a75", lab="second pass: " + c.cnt_extra[-1]["lab"])
        w2 = c.why.pop()
        c.why.append(f"det {'+'.join(map(str, mates))} were all called bad by the first pass, so without them: "
                     + w2.split(": ", 1)[1].split(" (suspect")[0] + " - below 30, and no healthy detector is left "
                     "on its phase to confirm or clear it -> watch ('no yardstick')")
    if k in ("Q2u", "Q2"):
        if k == "Q2u":
            ev_choppy110(c, fired=False)
            done.add("choppy")
            rr_, lab, ms = C.like_context(c)
            c.why.append(f"its % ON follows the average of {lab} det {'+'.join(map(str, ms))}: r = {rr_:.2f}")
        else:
            ev_level110(c, fired=True)
            done.add("level")
            q = C.ev_queue_pattern(c)
            rr_, lab, ms = C.like_context(c)
            c.why[-1:] = [f"{c.why[-1]}; cleared by Q2: in its busier half its time ON rises with traffic "
                          f"(slope {q['occ']:.2f}) faster than its counts (slope {q['cnt']:.2f}) = queue pattern, and "
                          f"its % ON tracks {lab} det {'+'.join(map(str, ms))} (r = {rr_:.2f}, needs 0.80) -> ok"]
    order = sorted(left, key=lambda q: -s8.get(q, 0))
    E = None
    for f in order:
        if f in done:
            continue
        if f == "stuck":
            E = ev_stuck110(c)
            ev_queue110(c, E)
        elif f == "dropout":
            C.ev_dropout(c, c.xp, c.refi, c.refk)
        elif f == "level":
            ev_level110(c, True)
        elif f == "choppy":
            ev_choppy110(c, True)
        elif f == "volume":
            C.ev_volume(c)
            lt, alt = lim_type(r, "max5")
            if alt:
                o8 = r.get("lim8_max5")
                c.why[-1] = c.why[-1].replace(f"limit for {tn}", f"limit for {lt} (model may be wrong about type; "
                                                                  f"{tn} would be {o8:.0f})")
                c.cnt_hl[0] = (c.cnt_hl[0][0], c.cnt_hl[0][1].replace(tn, lt), c.cnt_hl[0][2])
                c.cnt_hl.append((o8, f"limit for {tn} (the model's top type): {o8:.0f}", "#7b1fa2"))
                c.cnt_title[-1] = c.cnt_title[-1].replace(f"limit for {tn}", f"limit for {lt}")
        elif f in ("chatter", "rapid"):
            fast110(c, f)
        elif f == "night_drop":
            C.ev_night(c, c.xp)
        elif f == "occspk":
            C.ev_n3(c)
        elif f == "prof":
            C.ev_prof(c, True)
        done.add(f)
    # cleared / changed findings that the row is about
    if "stuck" not in done and (k in ("Q1", "Q1b", "stuck_long", "stuck_short", "stuckF", "F3") or "stuck" in left8):
        E = ev_stuck110(c)
        ev_queue110(c, E)
        done.add("stuck")
    if "level" not in done and (k in ("level", "F2") or "level" in left8):
        ev_level110(c, fired=False)
        done.add("level")
    if "choppy" not in done and (k in ("choppy", "Y1") or "choppy" in left8 or "no_yardstick_chop" in watch):
        ev_choppy110(c, fired=False)
        done.add("choppy")
    if "rapid" in watch and "R7" in rules:
        fast110(c, "rapid")
        c.why[-1] += " - just over and its only finding -> watch"
    if "occ_hi" in watch:
        C.ev_n1(c)
    if "prof_partial" in watch:
        ev_prof3(c)
    if k in ("old_only",) or (k == "F5" and "prof" in left8 and "prof" not in left) or \
            (k == "prof_new" and "prof" not in left):
        C.ev_prof(c, False)
        if f5_text(r) and (r.d_cnt_lim > 0):
            c.why[-1] = c.why[-1].replace(" -> not flagged", " (limits = the least strict of its plausible types) -> "
                                                            "not flagged")
        if k == "old_only":
            c.why[-1] += " (the old 'doesn't follow traffic' / 'busier at night' check had fired)"
    dropped = (set(left8) | set(sp(r.watch8_v108))) - set(left) - set(watch)
    if k == "F5" or (f5_text(r) and dropped):
        for f in sorted(dropped - done):
            if f in ("chatter", "rapid"):
                fast110(c, f)
                c.why[-1] += " -> not flagged"
            elif f == "volume":
                C.ev_volume(c)
                c.why[-1] += " -> not flagged"
    if "D1" in str(r.dq8):
        C.ev_d1(c)
    if not c.why:
        c.why.append("nothing over a limit")
    # what changed since the last sheet (v108)
    st8v = str(r.st8_v108)
    if st8v != r.st8 or sorted(left8) != sorted(left):
        old8 = C.ST.get(st8v, st8v) + (f" ({', '.join(NAME.get(q, q) for q in left8)})" if left8 and
                                       st8v in ("suspect", "bad") else "")
        if c.kind in {k_ for k_, *_ in NEW_ROWS}:
            c.why.insert(0, f"NEW EXAMPLE of a fixed check (before the fix: {old8})")
        else:
            c.why.insert(0, f"CHANGED since the last sheet (was {old8})")
    who_ = f"{c.sig} det {c.d} (P{int(r.phase)} {FN.get(r.fn, r.fn)}" + (f" {C.lane_short(c.ln.get(c.d, ''))}"
                                                                           if c.ln.get(c.d) else "") + ")"
    new = C.ST.get(r.st8, r.st8)
    old = C.ST.get(r.status, r.status)
    title = f"{who_}: {old} -> {new}"
    png = C.CH / f"{i:02d}_{c.kind}_{c.sig}_d{c.d}.png"
    draw(c, png, title)
    dpng = C.DC / png.name
    C.day_chart(c, dpng, f"{who_}: all saved data (Sat 26 16:15 - Mon 28 24:00), 15-min counts and % ON")
    smp = (f"24 h {c.t0:%a %d}" if c.h >= 24 else f"3 h {c.t0:%a %d} {c.t0:%H:%M}-{c.t1:%H:%M}")
    det_txt = f"det {c.d} · P{int(r.phase)} {FN.get(r.fn, r.fn)}" + (f" · {C.lane_long(c.ln.get(c.d, ''))}"
                                                                     if c.ln.get(c.d) else "")
    return dict(n=i, signal=c.sig, detector=c.d, det=det_txt, check=f"{KIND[c.kind]} · {smp}",
                oldnew=f"{C.old_txt(r)} -> {new_txt(r)}", why="; ".join(c.why), png=png, dpng=dpng, dev=c.dev,
                window=c.w, kind=c.kind, claims=c.claims, auto=c.auto, title=title)


def new_txt(r):
    sp = lambda s: [k for k in str(s).split(",") if k and k != "nan"]  # noqa: E731
    left, wat = sp(r.left8), sp(r.watch8)
    st = C.ST.get(r.st8, r.st8)
    if left and r.st8 in ("suspect", "bad"):
        st += " (" + ", ".join(NAME.get(k, k) for k in left) + ")"
    elif wat and r.st8 == "watch":
        st += " (" + ", ".join(NAME.get(k, k) for k in wat) + ")"
    if "D1" in str(r.dq8):
        st += " + data note"
    return st


def draw(c, path, title):
    """h108c draw; a 3-hour profile panel is drawn by the 24-h hour-panel code (values placed at its clock hours) and
    then re-drawn on its own 3-hour axis just before saving."""
    hp = [p for p in c.panels if p["kind"] == "hour3"]
    if not hp:
        return C.draw(c, path, title)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    for p in hp:
        p["kind"] = "hour"
        for key in ("own", "med", "lo", "hi"):
            full = np.full(24, np.nan)
            full[p["hs"]] = p[key]
            p[key + "3"], p[key] = p[key], full
        p.update(up=[], dn=[], ylabel=f"% of the 3 h's {p['what']}\nin each hour")
    real_save = plt.Figure.savefig

    def save(self, *a, **kw):
        for ax in self.axes:
            if tuple(round(v, 1) for v in ax.get_xlim()) == (-0.5, 23.5):
                p = hp[0]
                ax.cla()
                h = np.array(p["hs"])
                ax.fill_between(h, p["lo3"], p["hi3"], color="#d9d9d9", lw=0)
                l1, = ax.plot(h, p["med3"], color=MUTED, ls="--", lw=1.6, marker="s", ms=5)
                l2, = ax.plot(h, p["own3"], color=THIS, lw=3, marker="o", ms=6)
                for x_, y_, m_ in zip(h, p["own3"], p["med3"]):
                    up = y_ >= m_
                    ax.annotate(f"{y_:.0f} %", (x_, y_), xytext=(8, 6 if up else -16), textcoords="offset points",
                                color=THIS, fontsize=10.5, fontweight="bold")
                    ax.annotate(f"{m_:.0f} %", (x_, m_), xytext=(8, -16 if up else 6), textcoords="offset points",
                                color=MUTED, fontsize=10)
                ax.set_xlim(h[0] - 0.5, h[-1] + 0.5)
                ax.set_xticks(h)
                ax.set_xticklabels([f"{x:02d}:00-{x + 1:02d}:00" for x in h])
                ax.set_ylim(0, 100)
                ax.set_ylabel(f"% of the 3 h's {p['what']}\nin each hour", fontsize=10.5)
                C.style(ax)
                C.ptitle(ax, p["title"])
                C.leg(ax, [Patch(color="#d9d9d9"), l1, l2],
                      [f"normal range: 95 % of {p['n']} healthy {p['grp']} detectors, same hours, same day",
                       "median of those healthy detectors", f"det {c.d}"])
        return real_save(self, *a, **kw)
    plt.Figure.savefig = save
    try:
        C.draw(c, path, title)
    finally:
        plt.Figure.savefig = real_save


HEADER = [
    "Health spot-check v110. Is the NEW result right? Answer Y / N / ? in the yellow column. Each row's 'Why' line uses "
    "only numbers you can read on its chart. Rows marked CHANGED came out differently from the last sheet because a "
    "check was fixed (listed below); rows 31-36 are new examples of the fixed checks.",
    C.HEADER[1],
    C.HEADER[2],
    C.HEADER[3],
    "THE CHECKS (each judged on the sample in the Check column):",
    "Stuck on = every continuous ON (ON to the next OFF) longer than the limit for its type (5-15 min) counts, not "
    "only the longest: suspect when they add up to the limit, bad at 60 min in total, or when there are more such ONs "
    "than 1 in 500 healthy zones of its type have. An ON is not counted when it was a queue: its HEALTHY phase mates (not "
    "called bad by a first pass of all checks) were ON at least 1.5x their usual during it AND its % ON tracks theirs (r >= 0.70) outside it AND its phase's Advance / Count detectors still count at least half their usual during it (Q1); an ON over "
    "60 min must also have a mate queued (above the % ON only 1 in 20 healthy detectors of that mate's type reach) in "
    "at least 75 % of its 15-min periods (Q1b). An ON shared with 3+ other detectors is a shared event: suspect at most.",
    C.HEADER[6],
    "Count drops = its counts fall at some point and stay low compared with what its earlier share predicts, where the "
    "prediction follows the rest of the signal AND the usual hour-by-hour pattern of healthy detectors of its type; its "
    "own counts must at least halve (a detector that keeps counting while the signal wakes up at dawn has not dropped). "
    "After / before below 15 % = suspect, below 5 % = bad. Misses vehicles at night = 21:00-05:00 count / count "
    "expected from the detector it tracks (at its daytime share) below 0.19 = suspect, below 0.063 = bad.",
    "Erratic counts = how far its 15-min counts stray from 'expected' (its share of its yardstick's counts over the 2 h "
    "around), in units of normal random variation (about 1 for a healthy detector); limit per type, bad at 2x. "
    "Yardstick = 2+ other phase mates, else a phase mate of the same function, else the other detectors of its function "
    "on the signal; a twin (same zone on another input) never counts. No yardstick = not scored, 'watch' if the old "
    "whole-signal comparison would have flagged it. Q2: cleared if it shows the queue pattern (in the busier half of "
    "the day its time ON grows faster with traffic than its counts) AND its % ON tracks its healthy same-function mates "
    "(r >= 0.80).",
    C.HEADER[9],
    C.HEADER[10],
    "Time-of-day profile = the share of its actuations (and time ON) in each clock hour vs healthy detectors of the "
    "same type and volume (low < 20 / medium 20-100 / high > 100 per hour), same day; score = the % that sits in other "
    "hours than normal; limit per type and volume. A finding from 12 h of data; on 3-12 h only 'watch' (a short sample "
    "finds about a quarter of real problems). It replaces 'doesn't follow traffic' and 'busier at night'.",
    "Model unsure of the function (its top class below 70 %): every limit used is the least strict of the types it "
    "may be (classes at 15 % or more), and the row says so. Data note (D1) = most of its ON time is ONs logged again "
    "with no OFF between: reported, no status change.",
]


def main():
    src = Path(sys.argv[1])
    keep = C.old_entries(src)
    WK.mkdir(parents=True, exist_ok=True)
    C.HH.init()
    C.HEADER[:] = HEADER
    R = pd.read_parquet(H10 / "resolved110.parquet")
    L8 = pd.read_parquet(C.H8 / "resolved108_q998p995.parquet", columns=["DeviceId", "window", "detector"] +
                         [f"lim_{k}" for k in ("chat_frac", "ioi_lt1", "ioi_lt05", "burst_frac", "max5", "chop15")])
    R = R.merge(L8.rename(columns={c_: c_.replace("lim_", "lim8_") for c_ in L8.columns if c_.startswith("lim_")}),
                on=["DeviceId", "window", "detector"], how="left")
    nm = RV.names()
    rows = pd.read_csv(C.H8 / "review_rows108.csv")[["n", "signal", "dev", "window", "detector", "kind"]]
    add = [dict(n=31 + j, signal=nm.get(d, d[:8]), dev=d, window=w, detector=det, kind=k)
           for j, (k, d, w, det) in enumerate(NEW_ROWS)]
    rows = pd.concat([rows, pd.DataFrame(add)], ignore_index=True)
    rows.to_csv(WK / "review_rows110.csv", index=False)
    only = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    for p_ in (C.CH, C.DC):
        p_.mkdir(parents=True, exist_ok=True)
        if only is None:
            for f in p_.glob("*.png"):
                f.unlink()
    out, auto, claims = [], [], []
    for _, rr in rows.iterrows():
        if only is not None and int(rr.n) not in only:
            continue
        o = build(int(rr.n), rr, R, keep)
        print(o["n"], o["signal"], o["det"], "|", o["oldnew"], "|", o["why"], flush=True)
        for nm_, a, b in o["auto"]:
            ok = (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= max(0.02 * abs(b), 0.011)) or \
                 (not np.isfinite(a) and not np.isfinite(b))
            auto.append(dict(row=o["n"], signal=o["signal"], det=o["detector"], stat=nm_, chart=a, resolver=b,
                             match=bool(ok)))
            if not ok:
                print(f"   AUTO MISMATCH {nm_}: chart {a} vs resolver {b}")
        for cl, val in o["claims"]:
            claims.append(dict(row=o["n"], signal=o["signal"], det=o["detector"], claim=cl, value=val))
        out.append(o)
    pd.DataFrame(auto).to_csv(WK / ("auto_check.csv" if only is None else "auto_check_part.csv"), index=False)
    pd.DataFrame(claims).to_csv(WK / ("claims.csv" if only is None else "claims_part.csv"), index=False)
    pd.DataFrame([{k: v for k, v in o.items() if k not in ("claims", "auto")} for o in out]).to_csv(
        WK / ("rows_v110.csv" if only is None else "rows_part.csv"), index=False)
    if only is not None:
        return
    replies = json.loads((WK / "replies.json").read_text(encoding="utf-8")) if (WK / "replies.json").exists() else {}
    replies = {int(k): v for k, v in replies.items()}
    need = {k for k, v in keep.items() if any(v.get(q) not in (None, "") for q in ("answer", "comment", "earlier"))}
    have = {(o["signal"], o["detector"]) for o in out}
    print("user entries:", len(need), "carried:", len(need & have), "lost:", sorted(need - have))
    C.write_xl(out, keep, replies)
    chk = C.old_entries(C.XL)
    bad = [(k, q) for k in need for q in ("answer", "comment", "earlier") if keep[k].get(q) != chk.get(k, {}).get(q)]
    print("entries verified cell by cell:", "ALL MATCH" if not bad else f"MISMATCH {bad}")


if __name__ == "__main__":
    main()
