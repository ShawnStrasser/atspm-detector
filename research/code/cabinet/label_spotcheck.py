"""Follow-up spot-check of the label-behaviour validation (note 29, v3 of the check, 2026-09-28).

    python label_spotcheck.py            sample data (cached) -> charts -> review/spotcheck_label_checks.xlsx

At most 10 rows, the fixes most worth confirming. Charts are drawn from saved sample data
(`%DC_WORK%/cabinet/spotcheck_chart_data/`), never re-scored on the fly. The user's answers to the previous round are
kept in the sheet "your earlier answers". Cleansing / review only: nothing here is a model input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import label_check as L
from cab_common import CAB, DC_WORK, REPO

OUT = REPO / "review/spotcheck_label_checks.xlsx"
CHARTS = REPO / "review/spotcheck_label_charts"
DATA = CAB / "spotcheck_chart_data"
PDF = CAB / "pdf"
STG = DC_WORK / "official/stg/cache"
# light-surface categorical slots 1-4 (dataviz reference palette, validated order), text inks
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# (signal, det, chart kind, partner dets, why) - kinds: counts | timeline | durations | occ
PICKS = [
    ("2C036", 2, "counts", [3, 5, 6, 35, 36], "your example: was shown as pass - corrected"),
    ("08040", 13, "timeline", [1], "your example: flashing yellow arrow phase"),
    ("11039", 15, "timeline", None, "flashing yellow arrow phase found from the FYA events"),
    ("01080", 16, "durations", None, "presence zone set to pulse (new case)"),
    ("2B349", 44, "quiet", [50, 48], "your example: order and occupancy decide count vs presence"),
    ("04034", 4, "occ", None, "count zone occupied on red, but order / occupancy say Count (new case)"),
    ("2B422", 8, "counts", [10, 9, 11], "your example: advance loop misses vehicles"),
    ("11042", 22, "counts", [23], "your example: failed only on fault codes before"),
    ("2B143", 9, "occ", None, "new check: presence and count look swapped"),
    ("01070", 11, "timeline", None, "control: still fails after the fixes"),
]


def con():
    c = duckdb.connect()
    c.execute("SET memory_limit='8GB'")
    c.execute("SET threads=8")
    return c


def phase_dets(d, sig, det):
    x = d[(d.DeviceName == sig) & (d.detector == det)].iloc[0]
    g = d[(d.DeviceName == sig) & (d.p == x.p) & d.function.notna() & (d.n_on.fillna(0) >= 0)]
    return x, g


def extract(d: pd.DataFrame):
    """Sample data for every pick, saved once (charts are drawn from these files)."""
    DATA.mkdir(parents=True, exist_ok=True)
    cn = con()
    ivf = (STG / "det_intervals.parquet").as_posix()
    cyf = (STG / "phase_cycles.parquet").as_posix()
    for sig, det, kind, partners, _ in PICKS:
        f = DATA / f"{sig}_d{det}_{kind}.parquet"
        if f.exists():
            continue
        x, g = phase_dets(d, sig, det)
        dev = x.sid
        dets = sorted({det, *(partners or []), *[int(v) for v in g.detector]})
        dl = ",".join(map(str, dets))
        if kind == "counts":
            q = cn.sql(f"""SELECT Detector AS det, time_bucket(INTERVAL 15 MINUTE, t_on) AS t, count(*) AS n
                           FROM '{ivf}' WHERE lower(DeviceId) = '{dev}' AND Detector IN ({dl}) GROUP BY ALL""").df()
            # make zero-count detectors explicit on the same time grid
            grid = cn.sql(f"""SELECT DISTINCT time_bucket(INTERVAL 15 MINUTE, t_on) AS t FROM '{ivf}'
                              WHERE lower(DeviceId) = '{dev}'""").df()
            full = grid.merge(pd.DataFrame({"det": dets}), how="cross").merge(q, on=["t", "det"], how="left")
            full["n"] = full.n.fillna(0)
            full.to_parquet(f, index=False)
        elif kind == "durations":
            cn.sql(f"""SELECT Detector AS det, dur FROM '{ivf}' WHERE lower(DeviceId) = '{dev}' AND Detector IN ({dl})
                       AND dur IS NOT NULL""").df().to_parquet(f, index=False)
        elif kind in ("timeline", "quiet"):
            ph = int(x.p)
            # timeline: busiest 20 minutes of the target detector; quiet: a light-traffic 10 minutes (order visible)
            b = cn.sql(f"""SELECT time_bucket(INTERVAL 10 MINUTE, t_on) AS t, count(*) n FROM '{ivf}'
                           WHERE lower(DeviceId) = '{dev}' AND Detector = {det} GROUP BY 1 ORDER BY 2 DESC""").df()
            t0 = pd.Timestamp(b.t.iloc[0] if kind == "timeline" else b.t.iloc[int(len(b) * 0.6)])
            t1 = t0 + pd.Timedelta(minutes=20 if kind == "timeline" else 10)
            iv = cn.sql(f"""SELECT Detector AS det, t_on, t_off FROM '{ivf}' WHERE lower(DeviceId) = '{dev}'
                            AND Detector IN ({dl}) AND t_off >= '{t0}' AND t_on <= '{t1}'""").df()
            thr = L.THROUGH.get(ph, -1)
            cy = cn.sql(f"""SELECT Phase AS p, green_start, coalesce(yellow_start, red_start, next_green) AS ge
                            FROM '{cyf}' WHERE lower(DeviceId) = '{dev}' AND Phase IN ({ph}, {thr})
                            AND next_green >= '{t0}' AND green_start <= '{t1}'""").df()
            ev = cn.sql(f"""SELECT Timestamp AS ts, EventId AS e FROM read_parquet('{(L.RAW_STG_OTHER / '*/*.parquet').as_posix()}')
                            WHERE lower(DeviceId) = '{dev}' AND EventId IN (32, 33) AND Parameter = {ph}
                            AND Timestamp BETWEEN '{t0}' AND '{t1}' ORDER BY 1""").df()
            iv.assign(kind="det").to_parquet(f, index=False)
            cy.to_parquet(DATA / f"{sig}_d{det}_cycles.parquet", index=False)
            ev.to_parquet(DATA / f"{sig}_d{det}_fya.parquet", index=False)
            pd.DataFrame({"t0": [t0], "t1": [t1]}).to_parquet(DATA / f"{sig}_d{det}_window.parquet", index=False)
        elif kind == "occ":
            g[["detector", "function", "occ", "n_on"]].to_parquet(f, index=False)


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def draw(d: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CHARTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for sig, det, kind, partners, _ in PICKS:
        x, g = phase_dets(d, sig, det)
        lab = dict(zip(g.detector.astype(int), g.function))
        name = lambda k: f"det {k} ({lab.get(k, 'other phase')})"
        f = DATA / f"{sig}_d{det}_{kind}.parquet"
        q = pd.read_parquet(f)
        fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURF)
        _style(ax)
        if kind == "counts":
            show = [det] + [k for k in (partners or []) if k in set(q.det)]
            for i, k in enumerate(show[:4]):
                s = q[q.det == k].sort_values("t")
                ax.plot(s.t, s.n, lw=2 if k == det else 1.4, color=C[i], label=name(k) + f": {int(s.n.sum()):,} total")
            ax.set_ylabel("actuations per 15 min", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
            ttl = f"{sig} {x.phase_target}: 15-minute actuation counts, Sept 2026 window"
        elif kind == "durations":
            show = [det] + [int(k) for k in g.detector if k != det and (q.det == k).sum() >= 100][:3]
            bins = np.r_[0, 0.15, 0.25, 0.35, 0.5, 1, 2, 4, 8, 16, 60]
            w = 0.8 / len(show)
            for i, k in enumerate(show):
                h, _ = np.histogram(q[q.det == k].dur.clip(upper=59), bins=bins)
                ax.bar(np.arange(len(h)) + i * w, h / max(h.sum(), 1), width=w, color=C[i], label=name(k))
            ax.set_xticks(np.arange(len(bins) - 1) + 0.4 - w / 2)
            ax.set_xticklabels([f"{a:g}-{b:g}" for a, b in zip(bins[:-1], bins[1:])], fontsize=8)
            ax.set_xlabel("ON duration (s)", color=INK2, fontsize=9)
            ax.set_ylabel("share of actuations", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8)
            ttl = f"{sig} {x.phase_target}: how long each actuation lasts"
        elif kind in ("timeline", "quiet"):
            wdw = pd.read_parquet(DATA / f"{sig}_d{det}_window.parquet").iloc[0]
            cy = pd.read_parquet(DATA / f"{sig}_d{det}_cycles.parquet")
            ev = pd.read_parquet(DATA / f"{sig}_d{det}_fya.parquet")
            ph = int(x.p)
            show = [det] + [k for k in (partners if partners is not None else
                                        [int(v) for v in g.detector if v != det]) if k in set(q.det)][:3]
            rows = [f"P{ph} green"] + ([f"P{L.THROUGH[ph]} green (through)"] if ph in L.THROUGH else [])
            rows += (["FYA flashing (events 32/33)"] if len(ev) else []) + [name(k) for k in show]
            y = {r: len(rows) - i for i, r in enumerate(rows)}
            for r in cy.itertuples(index=False):
                lbl = f"P{ph} green" if r.p == ph else f"P{L.THROUGH.get(ph)} green (through)"
                ax.barh(y[lbl], (r.ge - r.green_start).total_seconds() / 86400, left=r.green_start, height=0.6,
                        color="#1baf7a" if r.p == ph else "#86b6ef")
            if len(ev):
                on = None
                for e in ev.itertuples(index=False):
                    if e.e == 32:
                        on = e.ts
                    elif on is not None:
                        ax.barh(y["FYA flashing (events 32/33)"], (e.ts - on).total_seconds() / 86400, left=on,
                                height=0.6, color="#eda100")
                        on = None
            for i, k in enumerate(show):
                s = q[q.det == k]
                ax.barh([y[name(k)]] * len(s), (s.t_off - s.t_on).dt.total_seconds() / 86400, left=s.t_on,
                        height=0.6, color=C[0] if k == det else INK2)
            ax.set_yticks(list(y.values()))
            ax.set_yticklabels(list(y.keys()), fontsize=8)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.set_xlim(wdw.t0, wdw.t1)
            ttl = (f"{sig} {x.phase_target}: " + ("20 busiest minutes" if kind == "timeline" else "10 light-traffic minutes")
                   + f" of det {det} (bars = detector ON / phase green)")
        else:  # occ
            q = q[q.n_on >= 100].sort_values("occ", ascending=False)
            fig.clf()
            ax1, ax2 = fig.subplots(1, 2)
            for a, col, lbl in ((ax1, "occ", "share of time ON"), (ax2, "n_on", "actuations in the window")):
                _style(a)
                a.barh(range(len(q)), q[col], color=[C[0] if k == det else "#86b6ef" for k in q.detector], height=0.6)
                a.set_yticks(range(len(q)))
                a.set_yticklabels([f"det {int(k)} ({fn})" for k, fn in zip(q.detector, q.function)], fontsize=8)
                a.invert_yaxis()
                a.set_title(lbl, fontsize=9, color=INK2, loc="left")
                a.grid(axis="y", visible=False)
                a.grid(axis="x", color=GRID, lw=0.8)
                for i, v in enumerate(q[col]):
                    a.text(v, i, f" {v:.1%}" if col == "occ" else f" {int(v):,}", va="center", fontsize=7, color=INK2)
            ax = ax1
            ttl = f"{sig} {x.phase_target}: presence zones should be MORE occupied with FEWER counts than count zones"
        fig.suptitle(ttl, x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        p = CHARTS / f"{sig}_d{det}.png"
        fig.savefig(p, dpi=110, facecolor=SURF)
        plt.close(fig)
        out[(sig, det)] = p
    return out


RESULT = {"pass": "pass (label kept)", "fail": "FAIL (label looks wrong)", "not_checkable": "not checkable (label kept)",
          "no_data": "no data", "unhealthy": "detector health problem (not trained on, label kept)"}


NOTE = {("2C036", 2): "Correction: the earlier 'pass' was a bug in how the review sheet was built (no check could run on "
                     "a detector with zero actuations, and the sheet read 'no failed check' as 'pass'); the label table "
                     "itself always had it as no data, never trained on. Detector 2 and 3 have 0 actuations in both the "
                     "Sept 2026 and Dec 2024 windows."}


def found(r) -> str:
    txt = str(r.validation_reason or "")
    parts = [p for p in txt.split(" | ") if p]
    if any("confirm the" in p and "field issue" in p for p in parts):  # the field-issue text already says it
        parts = [p for p in parts if not p.startswith("order / occupancy confirm the label")]
    n = f"{int(r.n_on):,} actuations in the Sept 2026 window" if np.isfinite(L._num(r.n_on)) else "no hi-res data"
    return f"{n}. " + " ".join(p.rstrip(".") + "." for p in parts)


def main():
    v3 = pd.read_parquet(L.V3)
    v3 = v3[~v3.released_from_newtest.astype(bool)].drop(columns="released_from_newtest")
    v3 = v3.drop(columns=[c for c in L.OUT_COLS + ["train_use_validated"] if c in v3])
    _, d, _ = L.run(v3, write_review=False)
    extract(d)
    charts = draw(d)
    pdfs = {p.name.split("_")[0]: p for p in PDF.glob("*.pdf")}
    rows = []
    for i, (sig, det, kind, _, why) in enumerate(PICKS, 1):
        r = d[(d.DeviceName == sig) & (d.detector == det)].iloc[0]
        rows.append({"#": i, "Signal": sig, "Phase": r.phase_target, "Detector": det, "Label": r.function,
                     "Result": RESULT.get(r.validated, r.validated),
                     "What the check found": f"Why shown: {why}. " + found(r)
                                             + (" " + NOTE[(sig, det)] if (sig, det) in NOTE else ""),
                     "Chart": charts[(sig, det)], "Cabinet print": pdfs.get(sig), "Your answer (yes / no / ?)": "",
                     "Your comment": ""})
    old = pd.DataFrame()
    if OUT.exists():  # a re-run keeps the earlier answers sheet, never the new rows
        sh = pd.read_excel(OUT, sheet_name=None)
        old = sh.get("your earlier answers", sh.get("spot check", pd.DataFrame()))
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    wb = Workbook()
    ws = wb.active
    ws.title = "spot check"
    cols = list(rows[0])
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        link = {"Chart": "open chart", "Cabinet print": "open print"}
        ws.append([(link[k] if v else "") if k in link else v for k, v in r.items()])
        for k in ("Chart", "Cabinet print"):
            if r[k]:
                cell = ws.cell(ws.max_row, cols.index(k) + 1)
                # chart: relative to the workbook (review/ travels as a folder); print: the local PDF copy
                cell.hyperlink = (f"{CHARTS.name}/{Path(r[k]).name}" if k == "Chart" else Path(r[k]).resolve().as_uri())
                cell.font = Font(color="0563C1", underline="single")
    for col, w in zip("ABCDEFGHIJK", (4, 8, 7, 9, 10, 24, 90, 11, 11, 14, 40)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    if len(old):
        w2 = wb.create_sheet("your earlier answers")
        w2.append(list(old.columns))
        for c in w2[1]:
            c.font = Font(bold=True)
        for rec in old.itertuples(index=False):
            w2.append([None if (isinstance(v, float) and not np.isfinite(v)) else v for v in rec])
        for row in w2.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        for col, w in zip("ABCDEFGHIJK", (4, 8, 7, 9, 10, 10, 70, 30, 14, 60, 11)):
            w2.column_dimensions[col].width = w
    try:
        wb.save(OUT)
    except PermissionError:
        wb.save(OUT.with_name(OUT.stem + "_new.xlsx"))
        print("spot-check workbook open in Excel -> wrote _new", file=sys.stderr)
    print(pd.DataFrame(rows)[["#", "Signal", "Detector", "Label", "Result"]].to_string(index=False))


if __name__ == "__main__":
    main()
