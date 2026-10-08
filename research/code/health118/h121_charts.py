"""Note 121: re-draw the 40 health v4 example charts from the SAVED plot data (dc_work/s118c/plot), never re-scored,
with the note-121 wording: lane span on every chart / reason, 'turning on again' (not 'restart'), v4d status
(score_v4d.py: rule A two independent findings -> bad, rule C Count zone held ON by extension -> suspect).
No new review sheet: charts go to %DC_WORK%/s121/charts and day_charts; reasons to s121/rows121.csv.

    python h121_charts.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import health_charts as HC  # noqa: E402
import h118c_sheet as SH  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PD = DCW / "s118c" / "plot"
OUT = DCW / "s121"
CH, DC = OUT / "charts", OUT / "day_charts"
KEY = ["DeviceId", "window", "detector"]
LANE_CATS = {"stuck", "choppy", "rapid", "volume", "chatter", "occspk", "prof", "shape24"}   # limits per fn x lane span


def reason121(r):
    if r["category"] == "count_on" and r.get("ext_fault"):
        rs = f"Held ON {r['hi_min']:.0f} min by extension in light traffic: counts unreliable"
    else:
        rs = SH.reason(r)
    if r["category"] in LANE_CATS and HC.lane_txt(r) and r["category"] != "volume":
        rs = f"{rs}; {HC.lane_txt(r)}"
    if r["category"] == "volume":                 # limit is per lane already: say how many
        rs = f"{rs} ({HC.lane_txt(r)})"
    return rs


def main():
    rows = json.loads((PD / "rows.json").read_text())
    marks = json.loads((PD / "marks.json").read_text())
    day = pd.read_parquet(PD / "day15.parquet")
    ev = pd.read_parquet(PD / "ev.parquet")
    ln = pd.read_parquet(DCW / "s117" / "stats117.parquet", columns=KEY + ["ln"])
    S = pd.read_csv(OUT / "rows121.csv")
    for p in (CH, DC):
        p.mkdir(parents=True, exist_ok=True)
        for f in p.glob("*.png"):
            f.unlink()
    out = []
    for r in rows:
        i = r["row"]
        s = S[S.row == i].iloc[0]
        q = ln[(ln.DeviceId == r["DeviceId"]) & (ln.window == r["window"]) & (ln.detector == r["detector"])]
        r = dict(r, lanes=float(q.ln.iloc[0]) if len(q) else r.get("lanes"), status=s.new_status,
                 ext_fault=s.sev_rule == "C")
        dd, ee = day[day.row == i], ev[ev.row == i]
        if r["category"] == "prof":
            o = ee[(ee.series == "otype_med") & ee.t.dt.hour.between(1, 4)]
            if len(o):
                r = dict(r, typ_on_night=float(o.value.mean()))
        pn = SH.phase_note(r, dd, marks.get(str(i)), ee)
        if pn:
            r = dict(r, phase_note=pn[0], phase_n=(pn[1], pn[2]))
        stem = f"{i:02d}_{r['category']}_{r['signal']}_d{r['detector']}.png"
        title = HC.draw_category(r, dd, ee, marks.get(str(i)), CH / stem)
        HC.draw_day(r, dd, DC / stem)
        rs = reason121(r)
        assert len(rs.split()) <= 16, (i, rs)
        assert "restart" not in (title + rs).lower(), i
        out.append(dict(row=i, lanes=HC.n_lanes(r), title=title, reason=rs, png=stem))
        print(i, r["status"], "|", title, "|", rs, flush=True)
    T = pd.DataFrame(out)
    S = S.drop(columns=[c for c in ("lanes", "title", "reason", "png") if c in S]).merge(T, on="row", how="left")
    S.to_csv(OUT / "rows121.csv", index=False)


if __name__ == "__main__":
    main()
