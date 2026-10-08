"""Note 107: re-check the print lane counts with the user's arrow rule (labels only, nothing trained).

The user (note 105): lanes per phase = movement arrows drawn on the approach(es) of the print (one LTR arrow = 1 lane,
two arrows = 2); a right-turn lane that only an overlap serves is not counted (counting it accepted).

    python ln107_arrowcheck.py worklist      phases to check (truth n_lanes >= 2, or 1 where the v5c OOF says 2+ confidently)
                                             + per-signal briefs and diagram renders -> %DC_WORK%/rev107/
    python ln107_arrowcheck.py collect       merge the per-signal readings (rev107/read/*.csv) -> rev107/readings.csv
    python ln107_arrowcheck.py build         truth v3 = note-105 truth + high-confidence corrections
                                             -> f107/ln8 + research/labels/lane_truth_corrections_arrow_v3.csv, lane_truth_changes_v3.csv
    python ln107_arrowcheck.py score         re-score the existing v5c lane OOF (f105 'after') on truth v2 vs v3 (no retrain)
locked_v2 asserted absent; prints only from cabinet/pdf and (released NEWTEST only) cabinet_locked/pdf.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DCW = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
F105 = DCW / "final_v3_work" / "f105" / "ln8"
OUT = DCW / "rev107"
F107 = DCW / "final_v3_work" / "f107" / "ln8"
LABD = REPO / "research" / "labels"
WINS = ['full66', 'h1_a', 'h1_b', 'h1_c', 'h24_a', 'h24_b', 'h3_a', 'h3_b', 'h6_a', 'h6_b',
        'm30_a', 'm30_b', 'm30_c', 'm30_d']


def locked() -> set:
    return set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())


def prints() -> pd.DataFrame:
    a = pd.read_parquet(DCW / "cabinet" / "print_labels.parquet").assign(store="cabinet")
    b = pd.read_parquet(DCW / "cabinet_locked" / "print_labels.parquet").assign(store="cabinet_locked")
    rel = set(pd.read_csv(DCW / "official" / "newtest_released.csv").DeviceId.str.lower())
    b = b[b.DeviceId.str.lower().isin(rel)]           # only the 71 released NEWTEST signals
    x = pd.concat([a, b], ignore_index=True)
    x["DeviceId"] = x.DeviceId.str.lower()
    assert not x.DeviceId.isin(locked()).any()
    return x


def model_majority() -> pd.DataFrame:
    P = pd.read_parquet(F105 / "lanes_base_ph.parquet")
    P = P[(P.period == "stg") & P.win.isin(WINS)]
    rows = []
    for (dev, ph), g in P.groupby(["DeviceId", "phase"]):
        vc = g.n_lanes.value_counts()
        maj = int(vc.index[0])
        conf = float(g[g.n_lanes == maj].n_lanes_conf.mean())
        rows.append(dict(DeviceId=dev, ph_num=int(ph), m_major=maj, m_share=int(vc.iloc[0]), m_n=len(g), m_conf=conf))
    return pd.DataFrame(rows)


def stage_worklist(a):
    import fitz
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "img").mkdir(exist_ok=True)
    (OUT / "brief").mkdir(exist_ok=True)
    (OUT / "read").mkdir(exist_ok=True)
    ph = pd.read_parquet(F105 / "truth_phase.parquet")
    ok = pd.read_parquet(F105 / "truth_phase_ok.parquet")
    det = pd.read_parquet(F105 / "truth_det.parquet")
    lk = locked()
    assert not ph.DeviceId.isin(lk).any() and not det.DeviceId.isin(lk).any()
    ph = ph.merge(ok, on=["DeviceId", "target"]).assign(ph_num=lambda d: d.target.str[1:].astype(int))
    mm = model_majority()
    ph = ph.merge(mm, on=["DeviceId", "ph_num"], how="left")
    ph["why"] = np.where(ph.n_lanes >= 2, "multi",
                         np.where((ph.m_major >= 2) & (ph.m_share >= 9) & (ph.m_conf >= 0.8), "model2+", ""))
    W = ph[ph.why != ""].copy()
    pr = prints()
    names = pr.drop_duplicates("DeviceId").set_index("DeviceId")
    W["signal"] = W.DeviceId.map(names.DeviceName)
    W["store"] = W.DeviceId.map(names.store)
    W["pdf"] = W.DeviceId.map(names.pdf)
    W = W[W.signal.notna()]
    W.to_csv(OUT / "worklist.csv", index=False)
    print(len(W), W.why.value_counts().to_dict(), W.signal.nunique(), "signals")
    # per signal: diagram pages, renders, brief
    for sig, g in W.groupby("signal"):
        dev = g.DeviceId.iloc[0]
        p = pr[pr.DeviceId == dev]
        store, pdf = p.store.iloc[0], p.pdf.iloc[0]
        pages = sorted({int(v) for v in p.diagram_page.dropna()})
        src = DCW / store / "pdf" / pdf
        doc = fitz.open(src)
        if not pages:
            pages = list(range(1, doc.page_count + 1))
        imgs = []
        for pg in pages:
            if pg < 1 or pg > doc.page_count:
                continue
            fn = OUT / "img" / f"{sig}_p{pg}.png"
            if not fn.exists():
                pix = doc[pg - 1].get_pixmap(dpi=a.dpi)
                pix.save(fn)
                # 2 x 2 overlapping tiles for zooming
                from PIL import Image
                im = Image.open(fn)
                w, h = im.size
                for i, (x0, y0) in enumerate([(0, 0), (w * 0.45, 0), (0, h * 0.45), (w * 0.45, h * 0.45)]):
                    im.crop((int(x0), int(y0), int(x0 + w * 0.55), int(y0 + h * 0.55))).save(
                        OUT / "img" / f"{sig}_p{pg}_q{i}.png")
            imgs.append(fn.name)
        L = [f"SIGNAL {sig}   print: {store}/pdf/{pdf}   pages in the PDF: {doc.page_count}",
             f"diagram page(s) as read before: {pages}  -> images: {', '.join(imgs)} (+ _q0.._q3 zoom tiles: "
             "q0 top-left, q1 top-right, q2 bottom-left, q3 bottom-right)", ""]
        dd = det[det.DeviceId == dev]
        for _, r in g.sort_values("ph_num").iterrows():
            mtxt = (f"model says {int(r.m_major)} lanes ({int(r.m_share)}/{int(r.m_n)} samples, conf {r.m_conf:.2f})"
                    if pd.notna(r.m_major) else "model: phase not predicted")
            L.append(f"PHASE {r.target}: current truth n_lanes = {int(r.n_lanes)} (accepted: {r.n_ok}); {mtxt}")
            x = dd[dd.target == r.target].sort_values(["lane_index", "det"])
            for _, q in x.iterrows():
                if pd.isna(q.lane_index) and (q.lane_type is None or pd.isna(q.lane_type)) and q.print_confidence == "low":
                    continue
                pl = p[p.detector == q.det]
                loops = pl.loops.iloc[0] if len(pl) else None
                tech = pl.technology.iloc[0] if len(pl) else None
                ln = "no lane" if pd.isna(q.lane_index) else (
                    f"lane {int(q.lane_index)}" + (f"-{int(q.end)}" if q.span > 1 else ""))
                L.append(f"   det {int(q.det):>3}  loops {loops!s:<10} {tech!s:<6} {q.print_function!s:<10} "
                         f"{ln:<10} lane_type {q.lane_type!s:<6} conf {q.print_confidence}")
            L.append("")
        (OUT / "brief" / f"{sig}.txt").write_text("\n".join(L))
    print("briefs", W.signal.nunique())


def stage_collect(a):
    fs = sorted((OUT / "read").glob("*.csv"))
    R = pd.concat([pd.read_csv(f, dtype={"signal": str}) for f in fs], ignore_index=True)
    R["signal"] = R.signal.str.zfill(5)
    W = pd.read_csv(OUT / "worklist.csv", dtype={"signal": str})
    x = W[["signal", "target", "DeviceId", "n_lanes", "n_ok", "why", "m_major", "m_share", "m_conf"]].merge(
        R.drop(columns=["truth"], errors="ignore"), on=["signal", "target"], how="left")
    x["confidence"] = x.confidence.fillna("missing").astype(str).str.strip().str.lower()
    x["arrows"] = pd.to_numeric(x.arrows, errors="coerce")
    for c in ("rt_only", "ol_rt"):
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(int)
    x["differs"] = x.arrows.notna() & (x.arrows != x.n_lanes)
    x.to_csv(OUT / "readings.csv", index=False)
    print(len(fs), "signals read;", len(x), "phases;", x.confidence.value_counts().to_dict())
    print("differs by conf:", x[x.differs].confidence.value_counts().to_dict())
    print(pd.crosstab(x[x.differs].n_lanes, x[x.differs].arrows))


def _user_phases() -> set:
    c = pd.read_csv(LABD / "lane_truth_corrections_user_v2.csv", dtype={"signal": str})
    return {(s, t) for s, t, r in zip(c.signal, c.target, c.rule) if r == "U"}


def stage_build(a):
    """truth v3: high-confidence arrow corrections on top of the note-105 truth (user answers never overridden).
    rt-only differences (arrows exceed the truth only by exclusive right-turn lanes) are not corrections: both
    counts accepted (user: right-turn lanes 'either is fine').  verify.csv = my own re-reads (override the reader)."""
    F107.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(OUT / "readings.csv", dtype={"signal": str})
    rv = OUT / "verify.csv"
    V = pd.read_csv(rv, dtype={"signal": str}) if rv.exists() else pd.DataFrame(
        columns=["signal", "target", "arrows", "confidence", "reason"])
    ph = pd.read_parquet(F105 / "truth_phase.parquet")
    ok = pd.read_parquet(F105 / "truth_phase_ok.parquet")
    det = pd.read_parquet(F105 / "truth_det.parquet")
    assert not ph.DeviceId.isin(locked()).any()
    up = _user_phases()
    okd = {(d, t): {int(v) for v in s.split(",")} for d, t, s in zip(ok.DeviceId, ok.target, ok.n_ok)}
    ch, med = [], []
    for _, r in x[x.differs].iterrows():
        key = (r.DeviceId, r.target)
        new, conf, why = int(r.arrows), r.confidence, r.reason
        v = V[(V.signal == r.signal) & (V.target == r.target)]
        if len(v):
            new, conf, why = int(v.arrows.iloc[0]), v.confidence.iloc[0], f"verified: {v.reason.iloc[0]}"
            if new == r.n_lanes:
                med.append(dict(signal=r.signal, target=r.target, old=int(r.n_lanes), new=int(r.arrows),
                                confidence=r.confidence, outcome="re-read: truth confirmed", reason=why))
                continue
        rt = int(r.rt_only)
        if (r.signal, r.target) in up:
            outcome = "user answer kept"
        elif new < 1:
            outcome = "listed (no arrow of this phase: every detector on an overlap lane; phase kept)"
        elif rt and new > r.n_lanes and new - r.n_lanes <= rt:
            outcome = "rt-only: arrow count accepted as alternative"
            okd[key].add(new)
        elif conf == "high" and int(r.ol_rt) and new < r.n_lanes and r.n_lanes - new <= int(r.ol_rt):
            outcome = "corrected"      # overlap-only right-turn lane not counted (rule O); old count accepted
            why = f"[overlap lane] {why}"
        elif conf == "high":
            outcome = "corrected"
        else:
            outcome = f"listed ({conf})"
        row = dict(signal=r.signal, target=r.target, old=int(r.n_lanes), new=new, confidence=conf, outcome=outcome,
                   reason=why)
        (ch if outcome == "corrected" else med).append(row)
    C = pd.DataFrame(ch, columns=["signal", "target", "old", "new", "confidence", "outcome", "reason"])
    for _, c in C.iterrows():
        dev = x[x.signal == c.signal].DeviceId.iloc[0]
        m = (ph.DeviceId == dev) & (ph.target == c.target)
        ph.loc[m, "n_lanes"] = c.new
        olc = str(c.reason).startswith("[overlap lane]")
        okd[(dev, c.target)] = {int(c.new)} | ({int(c.old)} if olc else set())
        md = (det.DeviceId == dev) & (det.target == c.target) & det.lane_index.notna()
        if c.new == 1 and not olc:      # one lane: every laned detector on lane 1
            det.loc[md, "lane_index"] = 1
            det.loc[md, "end"] = 1.0
            det.loc[md, "span"] = 1
        else:               # lane positions unknown now -> no high-confidence detector pairs from this phase
            det.loc[md, "high_veh"] = False
            det.loc[md & (det.end > c.new), "end"] = float(c.new)
            det.loc[md & (det.lane_index > c.new), "lane_index"] = c.new
            det.loc[md, "span"] = (det.loc[md, "end"] - det.loc[md, "lane_index"].astype(float) + 1).astype(int)
    OK = pd.DataFrame([dict(DeviceId=d, target=t, n_ok=",".join(str(v) for v in sorted(s)))
                       for (d, t), s in okd.items()])
    assert not ph.DeviceId.isin(locked()).any()
    ph.to_parquet(F107 / "truth_phase.parquet", index=False)
    OK.to_parquet(F107 / "truth_phase_ok.parquet", index=False)
    det.to_parquet(F107 / "truth_det.parquet", index=False)
    C.to_csv(LABD / "lane_truth_changes_v3.csv", index=False)
    M = pd.DataFrame(med, columns=C.columns)
    M.to_csv(LABD / "lane_truth_arrow_list_v3.csv", index=False)
    print("corrected", len(C), "| listed", M.outcome.value_counts().to_dict())
    if len(C):
        print(pd.crosstab(C.old, C.new))


def _rows(ph, ok):
    P = pd.read_parquet(F105 / "lanes_base_ph.parquet")
    P = P[(P.period == "stg") & P.win.isin(WINS)]
    x = ph.assign(ph_num=ph.target.str[1:].astype(int)).merge(ok, on=["DeviceId", "target"]).merge(
        P[["DeviceId", "phase", "win", "n_lanes"]], left_on=["DeviceId", "ph_num"], right_on=["DeviceId", "phase"],
        suffixes=("", "_p"))
    x["ok"] = (x.n_lanes == x.n_lanes_p).astype(float)
    x["ok_len"] = [float(int(p) in {int(v) for v in s.split(",")}) for p, s in zip(x.n_lanes_p, x.n_ok)]
    return x


def _boot(x2, x3, col, n=2000, seed=0):
    k = ["DeviceId", "target", "win"]
    j = x2[k + [col]].merge(x3[k + [col]], on=k, suffixes=("_2", "_3"))
    g = j.groupby("DeviceId").agg(s2=(f"{col}_2", "sum"), s3=(f"{col}_3", "sum"), c=(f"{col}_2", "count"))
    s2, s3, c = g.s2.values, g.s3.values, g.c.values
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(c), len(c))
        d.append(100 * (s3[i].sum() - s2[i].sum()) / c[i].sum())
    return [round(100 * (s3.sum() - s2.sum()) / c.sum(), 2), round(float(np.percentile(d, 2.5)), 2),
            round(float(np.percentile(d, 97.5)), 2)]


def stage_score(a):
    """the existing v5c lane OOF (note-105 'after', 14 Sept windows >= 30 min) scored on truth v2 and v3; no retrain."""
    ph2, ok2 = pd.read_parquet(F105 / "truth_phase.parquet"), pd.read_parquet(F105 / "truth_phase_ok.parquet")
    ph3, ok3 = pd.read_parquet(F107 / "truth_phase.parquet"), pd.read_parquet(F107 / "truth_phase_ok.parquet")
    x2, x3 = _rows(ph2, ok2), _rows(ph3, ok3)
    C = pd.read_csv(LABD / "lane_truth_changes_v3.csv", dtype={"signal": str})
    pr = prints().drop_duplicates("DeviceId")
    nm = dict(zip(pr.DeviceName, pr.DeviceId))
    chg = {(nm[s], t) for s, t in zip(C.signal, C.target)}
    res = {"phase_samples": len(x3), "phases_changed": len(chg)}
    for lab, x in (("v2", x2), ("v3", x3)):
        c = np.array([(d, t) in chg for d, t in zip(x.DeviceId, x.target)])
        res[lab] = {"nl_exact": round(float(x.ok.mean()), 4), "nl_lenient": round(float(x.ok_len.mean()), 4),
                    "changed_phases_exact": round(float(x[c].ok.mean()), 4) if c.any() else None,
                    "changed_samples": int(c.sum()),
                    "by_truth_n": {int(k): round(float(v), 4) for k, v in x.groupby("n_lanes").ok.mean().items()},
                    "by_wg": {w: round(float(x[x.win.str.startswith(w)].ok.mean()), 4)
                              for w in ["m30", "h1", "h3", "h6", "h24", "full"]}}
    res["v3_minus_v2"] = {"nl_exact": _boot(x2, x3, "ok"), "nl_lenient": _boot(x2, x3, "ok_len")}
    e2, e3 = 1 - x2.ok.mean(), 1 - x3.ok.mean()
    res["error_v2"], res["error_v3"] = round(100 * e2, 2), round(100 * e3, 2)
    res["share_of_v2_error_removed"] = round(float((e2 - e3) / e2), 3)
    json.dump(res, open(F107 / "score107.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
