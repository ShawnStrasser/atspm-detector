"""Note 93 B: small review sheet for the Yellow_Red-without-Count zones (and their would-be Count partners) that the data
could NOT settle.  Narrow format ("det 41: P1 Yellow-Red"), one detector (or a same-role pair) per row, chart + print.

Chart data are saved first (%DC_WORK%/s93/review_data/: the Sat 19 Sep 12:00-15:00 stg window, every detector on the
row's phase + that phase's color events), then drawn from the saved files only.

    python r93_review.py data     -> %DC_WORK%/s93/review_data/*.parquet
    python r93_review.py write    -> review/yr_without_count_review.xlsx + review/yr_without_count_review_charts/
"""
from __future__ import annotations

import os
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "evaluation"))
import rpath  # noqa: F401,E402

DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
REPO = CODE.parents[1]
S93 = DCW / "s93"
DATA = S93 / "review_data"
EVR = DCW / "official" / "stg" / "cache" / "events"
T0, T1 = pd.Timestamp("2026-09-19 12:00:00"), pd.Timestamp("2026-09-19 15:00:00")
OUTNAME = "yr_without_count_review"
QUESTION = ("Yellow-Red zones on a phase with no Count zone. The data could not settle these rows. For each row: is the "
            "label right? Answer Y (yes), N (no - write what it is) or ? (can't tell). Answering is optional: taking these "
            "zones out of training or scoring does not change the model's accuracy (see the chat summary).")
HEAD = ["#", "Signal", "Detector", "What the data shows", "Model says", "Question", "Chart", "Print", "Answer"]
WIDTH = (4, 9, 26, 58, 16, 44, 7, 7, 26)
NICE = {"Yellow_Red": "Yellow-Red", "Count": "Count", "Presence": "Presence", "Advance": "Advance", "Other": "Other",
        "Mid": "Mid", "Bike": "Bike"}

# (signal, detectors, phase, label, what the data shows, question) -- filled from b93 numbers (note 93)
ROWS = [
    ("08CM406", [41], "P1", "Yellow_Red",
     "83 % of its ONs come during red, 244 ON/h (6x the presence zone); a stop-bar Yellow-Red zone rarely turns ON in red (typical 4 %). "
     "Fits a zone spanning the through lanes too. Print reading is low confidence.",
     "Is det 41 a Yellow-Red zone for P1 only?"),
    ("08CM406", [42], "P2", "Yellow_Red",
     "55 % of its ONs come during red, 343 ON/h; not pulse (0.9 s ONs). Print reading is low confidence.",
     "Is det 42 a Yellow-Red zone for P2?"),
    ("08CM406", [44], "P4", "Yellow_Red",
     "77 % of its ONs come during red, 545 ON/h; not pulse. Print reading is low confidence.",
     "Is det 44 a Yellow-Red zone for P4?"),
    ("12053", [33, 39], "P2/P6", "Yellow_Red",
     "Pulse zones (0.1 s), ~370 ON/h, 30-36 % of ONs during red: behave like advance count zones, not stop-bar zones. "
     "The other Yellow-Red zones at this signal behave like 12052's (stop-bar, off in red).",
     "Are det 33 (P2) and det 39 (P6) really Yellow-Red zones?"),
    ("12053", [37, 38, 43, 45], "P4/P7/P8/P3", "Yellow_Red",
     "25-87 % of ONs during red; 1-2 s ONs. Unlike a speed-filtered stop-bar zone (typical Yellow-Red: 4 % of ONs in red).",
     "Are det 37, 38, 43 and 45 really Yellow-Red zones?"),
    ("2B382", [5], "P2", "Yellow_Red",
     "1.8 s ONs, 4 % of ONs during red, 329 ON/h: looks like a stop-bar presence/count zone. Two unlabelled pulse zones on P2 "
     "(det 31, 36) look like count or Yellow-Red zones. Label from config only.",
     "Is det 5 the P2 Yellow-Red zone, or is it det 31 or 36?"),
    ("2C023", [7, 21], "P3/P7", "Advance",
     "Labelled Advance (print), but pulse zones with almost no ONs during red (4-9 %; typical advance 48 %) and see "
     "the first car about 4 s into green, like a stop-bar Count zone. Model says Count (94 %).",
     "Are det 7 (P3) and det 21 (P7) advance zones, or stop-bar Count zones?"),
    ("2B048", [8, 9], "P4", "Advance",
     "Described 'CO'; labelled Advance. No ONs during red (0 %), 2.2 s ONs, 23-27 ON/h. An advance zone normally sees "
     "arrivals in red. Model split Count / Advance.",
     "Are det 8 and 9 advance zones, or stop-bar Count zones?"),
    ("2C026", [8], "P4", "Advance",
     "Described 'CO-DUMMY'; print says Count (low confidence), label Advance. Pulse, 56 % of ONs during red (advance-like). "
     "Model split Count / Advance.",
     "Is det 8 an advance zone or a stop-bar Count zone?"),
    ("2C040", [8], "P4", "Advance",
     "Described 'CO'; print says Count (low confidence), label Advance. Pulse, 22 % of ONs during red. Model split Count / "
     "Advance.",
     "Is det 8 an advance zone or a stop-bar Count zone?"),
]


def names() -> dict:
    L = pd.read_parquet(REPO / "research/labels/function_labels_v4o.parquet", columns=["DeviceId", "DeviceName"])
    return dict(zip(L.DeviceName.astype(str), L.DeviceId))


def stage_data():
    import pyarrow.dataset as ds
    DATA.mkdir(parents=True, exist_ok=True)
    nm = names()
    ph = pd.read_parquet(S93 / "b93_phases.parquet")
    lk = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    for sig, dets, phs, *_ in ROWS:
        dev = nm[sig]
        assert dev.lower() not in lk
        pset = phs.split("/")
        mates = ph[(ph.DeviceName == sig) & ph.phase.isin(pset)]
        ch = sorted(set(mates.Detector.astype(int)) | set(dets))
        p = EVR / f"DeviceId={dev}"
        if not p.is_dir():
            p = EVR / f"DeviceId={dev.lower()}"
        e = ds.dataset(p).to_table(filter=(ds.field("Timestamp") >= T0) & (ds.field("Timestamp") < T1)).to_pandas()
        e = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
        pn = [int(x[1:]) for x in pset]
        keep = (e.EventId.isin([81, 82]) & e.Parameter.isin(ch)) | (e.EventId.isin([1, 8, 9, 10, 11]) & e.Parameter.isin(pn))
        e[keep].to_parquet(DATA / f"{sig}_{'_'.join(map(str, dets))}.parquet", index=False)
        mates[["Detector", "phase", "truth", "function", "description", "pred_top", "on_per_h", "on_red", "pulse",
               "first_on"]].to_parquet(DATA / f"{sig}_{'_'.join(map(str, dets))}_mates.parquet", index=False)
    print("chart data saved", DATA)


def _intervals(e, d):
    x = e[e.EventId.isin([81, 82]) & (e.Parameter == d)].sort_values(["Timestamp", "EventId"], ascending=[True, False])
    on, out = None, []
    for t, k in zip(x.Timestamp, x.EventId):
        if k == 82 and on is None:
            on = t
        elif k == 81 and on is not None:
            out.append((on, t))
            on = None
    return out


def draw(row, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sig, dets, phs, lab = row[:4]
    stem = f"{sig}_{'_'.join(map(str, dets))}"
    e = pd.read_parquet(DATA / f"{stem}.parquet")
    M = pd.read_parquet(DATA / f"{stem}_mates.parquet").sort_values(["phase", "Detector"])
    pn = [int(x[1:]) for x in phs.split("/")][0]
    w0 = pd.Timestamp("2026-09-19 13:00:00")
    w1 = w0 + pd.Timedelta(minutes=8)
    ee = e[(e.Timestamp >= w0) & (e.Timestamp < w1)]
    M1 = M[M.phase == f"P{pn}"]
    chs = list(M1.Detector.astype(int))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [3, 2]})
    # green / yellow bands of the phase
    c = ee[ee.EventId.isin([1, 8, 10]) & (ee.Parameter == pn)].sort_values("Timestamp")
    st, t_prev = None, w0
    for t, k in zip(c.Timestamp, c.EventId):
        if st == 1:
            a1.axvspan(t_prev, t, color="#c8ecc8", lw=0)
        elif st == 8:
            a1.axvspan(t_prev, t, color="#fff2b3", lw=0)
        st, t_prev = k, t
    if st == 1:
        a1.axvspan(t_prev, w1, color="#c8ecc8", lw=0)
    for i, d in enumerate(chs):
        iv = _intervals(ee, d)
        col = "#c0392b" if d in dets else "#34495e"
        for a, b in iv:
            a1.plot([a, max(b, a + pd.Timedelta(seconds=0.3))], [i, i], color=col, lw=6, solid_capstyle="butt")
    r = M1.set_index("Detector")
    a1.set_yticks(range(len(chs)))
    a1.set_yticklabels([f"det {d}: {NICE.get(r.truth.get(d) or r.function.get(d), '?') if isinstance(r.truth.get(d) or r.function.get(d), str) else 'no label'}"
                        for d in chs], fontsize=8)
    a1.set_xlim(w0, w1)
    a1.set_title(f"{sig} P{pn}: ONs, Sat 19 Sep 13:00-13:08 (green = P{pn} green, yellow = P{pn} yellow, red bars = "
                 f"this row)", fontsize=9)
    a1.tick_params(axis="x", labelsize=8)
    # bars: share of ONs in red and ONs per hour, every detector on the phase (>= 30 min windows, model features)
    x = np.arange(len(M))
    a2.bar(x - 0.2, 100 * M.on_red.fillna(0), 0.4, color=["#c0392b" if d in dets else "#7f8c8d" for d in M.Detector],
           label="% of ONs during red")
    a2b = a2.twinx()
    a2b.bar(x + 0.2, M.on_per_h.fillna(0), 0.4, color="#2e86c1", alpha=.6, label="ONs per hour")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{p} d{d}" for p, d in zip(M.phase, M.Detector)], fontsize=8, rotation=0)
    a2.set_ylabel("% ONs in red", fontsize=8)
    a2b.set_ylabel("ONs / hour", fontsize=8)
    a2.set_title("All detectors on the phase(s): grey/red = % of ONs during red, blue = ONs per hour (median of the "
                 "30-min+ samples)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def stage_write():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    import review64 as R
    out = REPO / "review"
    xl = out / f"{OUTNAME}.xlsx"
    if xl.exists():
        raise RuntimeError(f"{xl} exists - not overwriting (carry answers first)")
    cdir = out / f"{OUTNAME}_charts"
    cdir.mkdir(parents=True, exist_ok=True)
    ph = pd.read_parquet(S93 / "b93_phases.parquet")
    wb = Workbook()
    ws = wb.active
    ws.title = "rows"
    ws.append([QUESTION])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(HEAD))
    ws.row_dimensions[1].height = 60
    ws.append(HEAD)
    link = Font(color="0563C1", underline="single")
    for i, row in enumerate(ROWS, 1):
        sig, dets, phs, lab, what, ask = row
        stem = f"{sig}_{'_'.join(map(str, dets))}"
        png = cdir / f"{stem}.png"
        draw(row, png)
        m = ph[(ph.DeviceName == sig) & ph.Detector.isin(dets)]
        mod = ", ".join(sorted({NICE.get(v, v) for v in m.pred_top}))
        dtxt = ("det " + " and ".join(map(str, dets)) if len(dets) <= 2 else "det " + ", ".join(map(str, dets))) + \
            f": {phs} {NICE[lab]}"
        p = R.pdf_for(sig)
        pdf = "file:///" + urllib.parse.quote(str(p).replace("\\", "/"), safe=":/()_-.,'") if p else None
        ws.append([i, sig, dtxt, what, mod, ask, "chart", "print" if pdf else "none", ""])
        n = ws.max_row
        c = ws.cell(n, 7)
        c.hyperlink = f"{OUTNAME}_charts/{png.name}"
        c.font = link
        if pdf:
            c = ws.cell(n, 8)
            c.hyperlink = pdf
            c.font = link
    for j, w in enumerate(WIDTH):
        ws.column_dimensions[chr(65 + j)].width = w
    for c in ws[2]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for r in ws.iter_rows(min_row=1):
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A3"
    wb.save(xl)
    print("saved", xl, len(ROWS), "rows")


if __name__ == "__main__":
    {"data": stage_data, "write": stage_write}[sys.argv[1]]()
