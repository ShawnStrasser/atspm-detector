"""Note 105: the user's lane answers (lane review v1 + v2, TR + R sheet) turned into lane-truth corrections, the print
'arrow rule' applied to the other TR + R approaches of note 102b, a consistent rule for overlap lanes, and a refit of the
2026 lane model (note-102 'base' recipe: lanes D both orientations, six folds x 3 seeds, lam picked per held-out fold).

Rules (labels only; the model still sees only the hi-res log):
  U  user answers, exactly: n_lanes = his preferred answer, his 'X is also acceptable' kept as an accepted alternative.
     A detector on a right-turn lane the user does not count gets NO lane (stays on its phase, function unchanged).
  A  arrow rule (TR + R split): lanes on an approach = movement arrows drawn on the print.  It reproduces all 6 user
     answers of review/tr_r_lane_examples.xlsx; applied by eye to the 8 other TR + R phases of note 102b that are in the
     lane truth (09040 print has no layout sheet -> unchanged).
  O  overlap lanes: a lane whose detectors are all described as an overlap lane (OLA..OLD in the timing description),
     lane type R, is NOT counted on any phase (detectors get no lane); counting it is an accepted alternative.
     = the user's explicit 04064 answers (OLA / OLB lanes: P1 and P5 have 1 lane) applied to every such lane.
  Lenient scoring also accepts n_lanes minus the R-only (right-turn) lanes still counted (user: 'either is fine').

    python ln105_lanefix.py build                corrected truth -> f105/ln8 + research/labels/lane_truth_*_v2.csv
    python ln105_lanefix.py fit|decode|full      = ln102 stages on the corrected truth (var base)
    python ln105_lanefix.py eval                 before (note-102 base) vs after (refit), both on the corrected truth
    python ln105_lanefix.py atspm                ATSPM function through the lane step (ln102 recipe), before vs after
CPU <= 4 workers.  locked_v2 asserted absent.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ln102_wide as L102  # noqa: E402

L8 = L102.L8
DCW, REPO = L102.DCW, L102.REPO
F95, F77, F102 = L102.F95, L102.F77, L102.OUT
OUT = DCW / "final_v3_work" / "f105" / "ln8"
LABD = REPO / "research" / "labels"
CORR2 = LABD / "lane_truth_corrections_user_v2.csv"
CHANGES = LABD / "lane_truth_changes_v2.csv"
L102.OUT = OUT
log = L102.log

# (signal, target, n_lanes, alt n_lanes accepted, source)
PH = [
    ("01030", "P8", 1, "", "U lane review v1 row 1: print misread, det 24 = stop-bar loop to the right, same lane"),
    ("01072", "P4", 1, "", "U lane review v1 row 2: same as row 1 (one wide lane)"),
    ("03010", "P4", 2, "3", "U lane review v1 row 3: 2 better, 3 acceptable (det 7 = on-ramp lane)"),
    ("03021", "P2", 2, "3", "U lane review v1 row 4: 2 better, 3 acceptable (det 6 = right-turn lane, not counted)"),
    ("03024", "P8", 3, "4", "U lane review v1 row 5: 3 lanes; model's 4 (on-ramp det 21) not faulted"),
    ("04064", "P4", 1, "2", "U lane review v1 row 10: 1 or 2 (overlap lane goes with P4)"),
    ("03057", "P7", 2, "", "U lane review v2 row 1: 2 lanes; det 21 Presence and det 47 Yellow_Red span both lanes"),
    ("04064", "P1", 1, "", "U lane review v2 row 2: P1 has 1 lane (dets 22 / 26 = OLA right-turn lane)"),
    ("04064", "P5", 1, "", "U lane review v2 row 2: P5 has 1 lane (dets 8 / 12 = OLB right-turn lane)"),
    ("04064", "P8", 1, "2", "U lane review v2 row 2: P8 1 lane, 2 acceptable (OLA goes with P8)"),
    ("04097", "P4", 1, "", "U lane review v2 row 3: P4 has 1 lane (det 10 slip-lane count: right-turn lane not counted)"),
    ("04104", "P1", 1, "", "U lane review v2 row 4: P1 has 1 lane (det 14 right side: not a lane of its own)"),
    ("08022", "P2", 2, "3", "U lane review v2 row 7: 2 lanes, 3 acceptable (det 4 = right-turn lane)"),
    ("2B001", "P6", 2, "3", "U lane review v2 row 9: 2 lanes, 3 acceptable because of the right-turn lane (det 19)"),
    ("2B047", "P2", 2, "3", "U lane review v2 row 10: 2 or 3 fine (det 5 right-turn lane; consistent option: not counted)"),
    ("2B089", "P6", 2, "3", "U lane review v2 row 11: 2 or 3 fine (det 19 right-turn lane; consistent option)"),
    ("2B130", "P2", 1, "2", "U lane review v2 row 12: 1 or 2 fine (det 2 right lane; consistent option: not counted)"),
    ("2B130", "P4", 1, "2", "U lane review v2 row 12: 1 or 2 fine (dets 7 / 14 right lane; consistent option)"),
    ("2C009", "P7", 1, "", "U lane review v2 row 15: P7 has 1 lane (dets 17 / 19 / 39 right lane not counted)"),
    ("2C039", "P6", 1, "2", "U lane review v2 row 16: P6 has 1 lane, 2 also fine (det 16 same lane)"),
    ("2C070", "P8", 2, "", "U lane review v2 row 17: P8 has 2 lanes (det 7 right lane not counted)"),
    ("05025", "P8", 1, "2", "U lane review v2 row 18: 2 lanes, 1 acceptable; lane 2 = OLA lane (dets on P1) -> rule O keeps 1"),
    ("10093", "P6", 2, "1", "U lane review v2 row 19: 2 lanes, 1 acceptable"),
    ("01072", "P8", 1, "", "U TR+R sheet row 5: 1 lane (one LTR arrow on the print)"),
    ("2B337", "P8", 1, "", "U TR+R sheet row 8: 1 lane (one LTR arrow)"),
    ("04013", "P4", 1, "", "U TR+R sheet row 10: 1 lane (one LTR arrow)"),
    ("07047", "P4", 1, "", "A arrow rule: one LTR arrow drawn (print sheet 5)"),
    ("07047", "P8", 1, "", "A arrow rule: one LTR arrow drawn (print sheet 5)"),
    ("07050", "P8", 1, "", "A arrow rule: one LTR arrow drawn (print sheet 5)"),
    ("08040", "P8", 1, "", "A arrow rule: one LTR arrow drawn (print page 6)"),
    ("08080", "P4", 1, "", "A arrow rule: one LTR arrow drawn (print sheet 5)"),
]
# (signal, target, det, 'lane:span' or 'none', source)
DT = [
    ("01030", "P8", 24, "1:1", "U v1 row 1"),
    ("01072", "P4", 9, "1:1", "U v1 row 2"), ("01072", "P4", 12, "1:1", "U v1 row 2 (was lanes 1+2)"),
    ("01072", "P4", 21, "1:1", "U v1 row 2 (was lanes 1+2)"),
    ("03021", "P2", 6, "none", "U v1 row 4: right-turn lane not counted"),
    ("03057", "P7", 21, "1:2", "U v2 row 1: spans both lanes"), ("03057", "P7", 47, "1:2", "U v2 row 1: spans both lanes"),
    ("04064", "P1", 22, "none", "U v2 row 2: OLA lane"), ("04064", "P1", 26, "none", "U v2 row 2: OLA lane"),
    ("04064", "P5", 8, "none", "U v2 row 2: OLB lane"), ("04064", "P5", 12, "none", "U v2 row 2: OLB lane"),
    ("04097", "P4", 10, "none", "U v2 row 3: right-turn slip lane not counted"),
    ("04104", "P1", 14, "none", "U v2 row 4: not a lane of its own"),
    ("08022", "P2", 4, "none", "U v2 row 7: right-turn lane"), ("2B001", "P6", 19, "none", "U v2 row 9: right-turn lane"),
    ("2B047", "P2", 5, "none", "U v2 row 10: right-turn lane"), ("2B089", "P6", 19, "none", "U v2 row 11: right-turn lane"),
    ("2B130", "P2", 2, "none", "U v2 row 12: right lane"), ("2B130", "P4", 7, "none", "U v2 row 12: right lane"),
    ("2B130", "P4", 14, "none", "U v2 row 12: right lane"),
    ("2C009", "P7", 17, "none", "U v2 row 15: right lane"), ("2C009", "P7", 19, "none", "U v2 row 15: right lane"),
    ("2C009", "P7", 39, "none", "U v2 row 15: right lane"),
    ("2C039", "P6", 16, "1:1", "U v2 row 16: same lane"), ("2C070", "P8", 7, "none", "U v2 row 17: right lane"),
] + [(s, t, d, "1:1", src) for s, t, dets, src in [
    ("01072", "P8", (22, 23, 24), "U TR+R row 5"), ("2B337", "P8", (22, 23, 24), "U TR+R row 8"),
    ("04013", "P4", (8, 9, 10), "U TR+R row 10"), ("07047", "P4", (8, 9, 10), "A arrow rule"),
    ("07047", "P8", (22, 23, 24), "A arrow rule"), ("07050", "P8", (22, 23, 24), "A arrow rule"),
    ("08040", "P8", (22, 23, 24), "A arrow rule"), ("08080", "P4", (8, 9, 10), "A arrow rule")] for d in dets]
OL_RX = re.compile(r"\bOL[A-D]\b")


def names():
    L = pd.read_parquet(L102.rpath.LABELS_CURRENT, columns=["DeviceId", "DeviceName"]).drop_duplicates()
    return dict(zip(L.DeviceName, L.DeviceId.str.lower())), dict(zip(L.DeviceId.str.lower(), L.DeviceName))


def _set_lane(det, m, val):
    if val == "none":
        det.loc[m, ["lane_index", "end"]] = [pd.NA, np.nan]
        det.loc[m, "span"] = 1
        det.loc[m, "high_veh"] = False
    else:
        li, sp = (int(x) for x in val.split(":"))
        det.loc[m, "lane_index"] = li
        det.loc[m, "span"] = sp
        det.loc[m, "end"] = li + sp - 1
        det.loc[m, "high_veh"] = (det.loc[m, "print_confidence"] == "high") & ~det.loc[m, "lane_type"].isin(
            L8.L1.NONVEH)


def stage_build(a):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("cues.parquet", "dets.parquet", "truth_text_pairs.parquet"):
        shutil.copy(F95 / f, OUT / f)
    for f in ("cues_rev.parquet", "keys.parquet"):
        shutil.copy(F77 / f, OUT / f)
    det0 = pd.read_parquet(F95 / "truth_det.parquet")
    ph0 = pd.read_parquet(F95 / "truth_phase.parquet")
    det, ph = det0.copy(), ph0.copy()
    nm, rn = names()
    lk = L8.locked()
    alt = {}
    ch, corr = [], []
    # ---- U / A phase + detector rows
    for s, t, n, al, src in PH:
        dev = nm[s]
        assert dev not in lk
        m = (ph.DeviceId == dev) & (ph.target == t)
        assert m.sum() == 1, (s, t)
        old = int(ph.loc[m, "n_lanes"].iloc[0])
        ph.loc[m, "n_lanes"] = n
        alt.setdefault((dev, t), set()).update(int(x) for x in str(al).split(",") if x)
        corr.append(dict(signal=s, target=t, kind="phase", det="", value=str(n), alt=al, rule=src[0], source=src))
        ch.append(dict(signal=s, target=t, kind="phase", det="", old=str(old), new=str(n), alt=al, rule=src[0],
                       source=src))
    for s, t, d, v, src in DT:
        dev = nm[s]
        m = (det.DeviceId == dev) & (det.target == t) & (det.det == d)
        assert m.sum() == 1, (s, t, d)
        r = det[m].iloc[0]
        old = "none" if pd.isna(r.lane_index) else f"{int(r.lane_index)}:{int(r.span)}"
        _set_lane(det, m, v)
        corr.append(dict(signal=s, target=t, kind="det", det=d, value=v, alt="", rule=src[0], source=src))
        ch.append(dict(signal=s, target=t, kind="det", det=d, old=old, new=v, alt="", rule=src[0], source=src))
    user_ph = {(nm[s], t) for s, t, *_ in PH}
    # ---- O: overlap lanes (timing description OLA..OLD, lane type R, every detector of that lane)
    o = pd.read_parquet(DCW / "official" / "labels_official.parquet", columns=["DeviceId", "Detector", "description"])
    o["DeviceId"] = o.DeviceId.str.lower()
    desc = {(d, int(x)): str(s) for d, x, s in zip(o.DeviceId, o.Detector, o.description)}
    det["ol"] = [bool(OL_RX.search(desc.get((d, int(x)), "") or "")) for d, x in zip(det.DeviceId, det.det)]
    n_ol = 0
    for (dev, t), g in det[det.lane_index.notna() & det.phase_kept].groupby(["DeviceId", "target"]):
        if (dev, t) in user_ph:
            continue
        rm = []
        for k, gl in g[g.span == 1].groupby("lane_index"):
            on = g[(g.lane_index <= k) & (g.end >= k)]
            if gl.ol.all() and (gl.lane_type == "R").all() and len(on) == len(gl):
                rm.append(int(k))
        if not rm:
            continue
        m = (ph.DeviceId == dev) & (ph.target == t)
        old = int(ph.loc[m, "n_lanes"].iloc[0])
        if old - len(rm) < 1:
            continue
        for k in rm:
            for i in det.index[(det.DeviceId == dev) & (det.target == t) & (det.lane_index == k)]:
                ch.append(dict(signal=rn[dev], target=t, kind="det", det=int(det.at[i, "det"]),
                               old=f"{k}:1", new="none", alt="", rule="O", source=f"O overlap lane: {desc[(dev, int(det.at[i, 'det']))]}"))
                corr.append(dict(signal=rn[dev], target=t, kind="det", det=int(det.at[i, "det"]), value="none", alt="",
                                 rule="O", source="O overlap lane (timing description names an overlap)"))
                _set_lane(det, det.index == i, "none")
        # renumber the lanes above a removed one
        mm = (det.DeviceId == dev) & (det.target == t) & det.lane_index.notna()
        for i in det.index[mm]:
            sh = sum(1 for k in rm if k < det.at[i, "lane_index"])
            if sh:
                det.at[i, "lane_index"] = int(det.at[i, "lane_index"]) - sh
                det.at[i, "end"] = det.at[i, "end"] - sh
        ph.loc[m, "n_lanes"] = old - len(rm)
        alt.setdefault((dev, t), set()).add(old)
        ch.append(dict(signal=rn[dev], target=t, kind="phase", det="", old=str(old), new=str(old - len(rm)),
                       alt=str(old), rule="O", source=f"O {len(rm)} overlap lane(s) not counted"))
        corr.append(dict(signal=rn[dev], target=t, kind="phase", det="", value=str(old - len(rm)), alt=str(old),
                         rule="O", source="O overlap lane(s) not counted; counting them accepted"))
        n_ol += 1
    det = det.drop(columns="ol")
    # ---- consistency: no lane above n_lanes
    x = det[det.lane_index.notna() & det.phase_kept].merge(ph, on=["DeviceId", "target"])
    bad = x[x.end > x.n_lanes]
    assert bad.empty, bad[["DeviceId", "target", "det", "lane_index", "end", "n_lanes"]]
    # ---- lenient alternatives: user alts, removed lanes, and n minus the R-only lanes still counted
    kr = {}
    for (dev, t), g in det[det.lane_index.notna() & det.phase_kept & (det.span == 1)].groupby(["DeviceId", "target"]):
        kr[(dev, t)] = sum(1 for k, gl in g.groupby("lane_index") if (gl.lane_type == "R").all())
    rows = []
    for dev, t, n in zip(ph.DeviceId, ph.target, ph.n_lanes):
        ok = {int(n)} | alt.get((dev, t), set())
        if kr.get((dev, t), 0) and n - kr[(dev, t)] >= 1:
            ok.add(int(n - kr[(dev, t)]))
        rows.append(dict(DeviceId=dev, target=t, n_ok=",".join(str(v) for v in sorted(ok))))
    A = pd.DataFrame(rows)
    assert not det.DeviceId.isin(lk).any() and not ph.DeviceId.isin(lk).any()
    det.to_parquet(OUT / "truth_det.parquet", index=False)
    ph.to_parquet(OUT / "truth_phase.parquet", index=False)
    A.to_parquet(OUT / "truth_phase_ok.parquet", index=False)
    C = pd.DataFrame(ch)
    pd.DataFrame(corr).to_csv(CORR2, index=False)
    C.to_csv(CHANGES, index=False)
    j = ph0.merge(ph, on=["DeviceId", "target"], suffixes=("_0", "_1"))
    s = {"phases": len(ph), "signals": int(ph.DeviceId.nunique()),
         "phase_n_changed": int((j.n_lanes_0 != j.n_lanes_1).sum()),
         "phase_n_changed_by_rule": C[(C.kind == "phase") & (C.old != C.new)].rule.value_counts().to_dict(),
         "det_rows_changed_by_rule": C[C.kind == "det"].rule.value_counts().to_dict(),
         "overlap_phases": n_ol, "phases_with_alt": int((A.n_ok.str.contains(",")).sum())}
    import ln2_pairmodel as L2
    T0 = L2.truth_pairs(det0, ph0)[["DeviceId", "da", "db", "same_lane"]]
    T1 = L2.truth_pairs(det, ph)[["DeviceId", "da", "db", "same_lane"]]
    s["high_pairs_before"], s["high_pairs_after"] = len(T0), len(T1)
    jj = T0.merge(T1, on=["DeviceId", "da", "db"], how="outer", suffixes=("_0", "_1"))
    s["pairs_label_flipped"] = int((jj.same_lane_0.notna() & jj.same_lane_1.notna()
                                    & (jj.same_lane_0 != jj.same_lane_1)).sum())
    s["pairs_dropped"] = int(jj.same_lane_1.isna().sum())
    s["pairs_added"] = int(jj.same_lane_0.isna().sum())
    json.dump(s, open(OUT / "build105.json", "w"), indent=1)
    log(json.dumps(s))


# ---------------------------------------------------------------------------------------------------- refit (ln102)
def stage_fit(a):
    L102.stage_fit(a)


def stage_decode(a):
    L102.stage_decode(a)


def stage_full(a):
    L102.stage_full(a)


# ---------------------------------------------------------------------------------------------------- eval
def _rows(ph_file: Path, ok: pd.DataFrame, ph: pd.DataFrame) -> pd.DataFrame:
    P = pd.read_parquet(ph_file)
    P = P[(P.period == "stg") & P.win.isin(L102.REVIEW_WINS)]
    x = ph.assign(ph_num=ph.target.str[1:].astype(int)).merge(ok, on=["DeviceId", "target"]).merge(
        P[["DeviceId", "phase", "win", "n_lanes"]], left_on=["DeviceId", "ph_num"], right_on=["DeviceId", "phase"],
        suffixes=("", "_p"))
    x["ok"] = (x.n_lanes == x.n_lanes_p).astype(float)
    x["ok_len"] = [float(int(p) in {int(v) for v in s.split(",")}) for p, s in zip(x.n_lanes_p, x.n_ok)]
    return x


def _dx(det_file: Path, det: pd.DataFrame, ph: pd.DataFrame) -> pd.DataFrame:
    import ln3_decode as L3
    D = pd.read_parquet(det_file, columns=["DeviceId", "Detector", "period", "win", "phase", "lanes"])
    D = D[(D.period == "stg") & D.win.isin(L102.REVIEW_WINS)].rename(columns={"Detector": "det"})
    tv = det[det.high_veh & det.phase_kept].merge(ph, on=["DeviceId", "target"])
    x = tv.merge(D, on=["DeviceId", "det"])
    x["lanes"] = x.lanes.fillna("")
    x["covered"] = (x.lanes != "") & (x.phase == x.ph_num)
    x["exact"] = L3.lane_exact(x)
    x["ok"] = x.exact.astype(float)
    return x


def stage_eval(a):
    det = pd.read_parquet(OUT / "truth_det.parquet")
    ph = pd.read_parquet(OUT / "truth_phase.parquet")
    ok = pd.read_parquet(OUT / "truth_phase_ok.parquet")
    det0 = pd.read_parquet(F102 / "truth_det.parquet")
    ph0 = pd.read_parquet(F102 / "truth_phase.parquet")
    src = {"before": (F102 / "lanes_base_ph.parquet", F102 / "lanes_base.parquet"),
           "after": (OUT / "lanes_base_ph.parquet", OUT / "lanes_base.parquet")}
    key = ["DeviceId", "target", "win"]
    R = {t: _rows(p, ok, ph) for t, (p, _) in src.items()}
    X = {t: _dx(d, det, ph) for t, (_, d) in src.items()}
    common = R["before"][key].merge(R["after"][key], on=key)
    dk = ["DeviceId", "det", "win"]
    cd = X["before"][dk].merge(X["after"][dk], on=dk)
    chg = set(zip(*[pd.read_csv(CHANGES, dtype={"signal": str}).pipe(
        lambda c: c.assign(DeviceId=c.signal.map(names()[0])))[k] for k in ("DeviceId", "target")]))
    res = {"truth": json.load(open(OUT / "build105.json"))}
    for t in src:
        x = R[t].merge(common, on=key)
        x["chg"] = [(d, g) in chg for d, g in zip(x.DeviceId, x.target)]
        dx = X[t].merge(cd, on=dk)
        R[t], X[t] = x, dx
        res[t] = {"phase_samples": int(len(x)), "nl_exact": round(float(x.ok.mean()), 4),
                  "nl_exact_lenient": round(float(x.ok_len.mean()), 4),
                  "nl_exact_changed_phases": round(float(x[x.chg].ok.mean()), 4), "n_changed": int(x.chg.sum()),
                  "nl_exact_by_wg": {w: round(float(x[x.win.map(L8.wgroup) == w].ok.mean()), 4)
                                     for w in ["m30", "h1", "h3", "h6", "h24", "full"]},
                  "det_rows": int(len(dx)), "det_cov": round(float(dx.covered.mean()), 4),
                  "det_lane_exact_covered": round(float(dx[dx.covered].exact.mean()), 4),
                  "det_lane_exact_all": round(float(dx.exact.mean()), 4)}
        log(f"{t}: {res[t]}")
    # truth-only effect: the before model on the OLD (note-102) truth, same 14 windows
    ok0 = ph0[["DeviceId", "target"]].assign(n_ok=ph0.n_lanes.astype(str))
    x0 = _rows(src["before"][0], ok0, ph0)
    d0 = _dx(src["before"][1], det0, ph0)
    res["before_on_old_truth"] = {"nl_exact": round(float(x0.ok.mean()), 4),
                                  "det_lane_exact_covered": round(float(d0[d0.covered].exact.mean()), 4)}
    b, f = R["before"], R["after"]
    res["after_minus_before"] = {
        "nl_exact": [round(100 * (f.ok.mean() - b.ok.mean()), 2)] + L8.boot(b, f),
        "nl_exact_lenient": [round(100 * (f.ok_len.mean() - b.ok_len.mean()), 2)] + L8.boot(b, f, col="ok_len"),
        "nl_exact_changed": [round(100 * (f[f.chg].ok.mean() - b[b.chg].ok.mean()), 2)] + L8.boot(b[b.chg], f[f.chg]),
        "det_lane_exact_all": [round(100 * (X["after"].ok.mean() - X["before"].ok.mean()), 2)]
        + L8.boot(X["before"], X["after"])}
    log(json.dumps(res["after_minus_before"]))
    json.dump(res, open(OUT / "eval105.json", "w"), indent=1, default=str)


def stage_atspm(a):
    for suf in ("", "_ph"):
        shutil.copy(F102 / f"lanes_base{suf}.parquet", OUT / f"lanes_before{suf}.parquet")
    a.tags, a.out = "before,base", "main"
    L102.stage_atspm(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--var", default="base")
    ap.add_argument("--notau", action="store_true")
    ap.add_argument("--advs", default="none")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
