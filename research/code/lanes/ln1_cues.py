"""Lane output (note 42), step 1 -- truth tables and pair cues for every staging window.

Truth (research/labels/function_labels_v3.parquet, read only; lp1_labels rules): complete_high /
complete_mixed tiers, not unusual, timing target a phase, print diagram phase = timing phase.
  ln1_truth_det.parquet   one row per print row on a kept phase (lane_index, span, high, vehicle)
  ln1_truth_phase.parquet (DeviceId, target, n_lanes) -- kept only if every row agrees and no row
                          sits beyond n_lanes
Locked signals (official/locked_v2.csv) asserted absent; the 71 released NEWTEST signals are in.

Cues (lane_output.signal_pair_cues = numpy form of lr2_pairs.py) for the Sept-2026 windows
m30_a..d, h6_a/b, h24_a/b, full66, on every pair of detectors (>= MIN_ON actuations in the window)
that share the PREDICTED phase (frame v6 pred_phase = out-of-fold phase) or the TIMING phase.
Predicted function = frame v6 first.all.wi OOF (seed mean, argmax, 7 classes).
  ln1_dets.parquet   (DeviceId, win, det, n_on, pred_phase, func, fold)
  ln1_pairs.parquet  (DeviceId, win, da, db, cues, context, same_pred, same_true)

    python ln1_cues.py [--workers 4]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse
import itertools
import json
import os
import time
import numpy as np
import pandas as pd
import lane_output as LO

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
OUT = DCW / "lanes"
V3 = REPO / "research" / "labels" / "function_labels_v3.parquet"
FRAME = DCW / "trackA" / "v3" / "frame_v6"
RUN = FRAME / "run_d3f4e7ccbc_exclude_min5_clean_valnc"
DI = DCW / "official" / "stg" / "cache" / "det_intervals.parquet"
NONVEH = {"bike", "departure", "other"}
WINS = {"m30_a": ("2026-09-21 07:30:00", 1800), "m30_b": ("2026-09-19 12:00:00", 1800),
        "m30_c": ("2026-09-19 21:30:00", 1800), "m30_d": ("2026-09-18 17:00:00", 1800),
        "h6_a": ("2026-09-20 06:00:00", 6 * 3600), "h6_b": ("2026-09-19 12:00:00", 6 * 3600),
        "h24_a": ("2026-09-19 00:00:00", 24 * 3600), "h24_b": ("2026-09-20 00:00:00", 24 * 3600),
        "full66": ("2026-09-18 16:15:00", 66 * 3600)}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.astype(str).str.lower())


def truth():
    d = pd.read_parquet(V3)
    d["DeviceId"] = d.DeviceId.str.lower()
    assert not d.DeviceId.isin(locked()).any(), "locked signal in v3"
    d = d[d.tier.isin(["complete_high", "complete_mixed"]) & ~d.unusual_layout.astype(bool)
          & (d.phase_target_type == "phase") & d.phase_target.notna()].copy()
    d["span"] = d.lanes_spanned.fillna(1).astype(int)
    d["end"] = d.lane_index.astype(float) + d.span - 1
    d["ph_num"] = pd.to_numeric(d.phase_target.str[1:], errors="coerce")
    d["diag_bad"] = d.phase_diagram.notna() & (pd.to_numeric(d.phase_diagram, errors="coerce")
                                               != d.ph_num)
    g = d.groupby(["DeviceId", "phase_target"])
    ph = g.agg(n_lanes=("n_lanes_phase", "max"), n_min=("n_lanes_phase", "min"),
               max_end=("end", "max"), diag_bad=("diag_bad", "any"),
               n_high=("print_confidence", lambda s: int((s == "high").sum()))).reset_index()
    keep = (ph.n_lanes.notna() & (ph.n_lanes == ph.n_min) & (ph.n_lanes >= 1)
            & ~(ph.max_end > ph.n_lanes) & ~ph.diag_bad & (ph.n_high >= 1))
    ph = ph[keep].rename(columns={"phase_target": "target"})[["DeviceId", "target", "n_lanes"]]
    ph["n_lanes"] = ph.n_lanes.astype(int)
    det = d.rename(columns={"phase_target": "target", "detector": "det"})
    det["det"] = det.det.astype(int)
    det["high_veh"] = ((det.print_confidence == "high") & det.lane_index.notna()
                       & ~det.lane_type.isin(NONVEH))
    det = det.merge(ph[["DeviceId", "target"]], on=["DeviceId", "target"], how="left",
                    indicator=True)
    det["phase_kept"] = det._merge == "both"
    det = det[["DeviceId", "det", "target", "ph_num", "lane_index", "span", "end", "lane_type",
               "print_function", "print_confidence", "high_veh", "phase_kept"]]
    return ph, det


def frame_preds() -> pd.DataFrame:
    k = pd.read_parquet(FRAME / "feat_frame.parquet",
                        columns=["DeviceId", "Detector", "period", "win", "fold", "pred_phase"])
    P = np.zeros((len(k), len(LO.C7)), np.float32)
    fo = k.fold.to_numpy()
    for s in (0, 1, 2):
        for f in range(6):
            P[fo == f] += np.load(RUN / f"P_first.all.wi_s{s}_f{f}.npy")
    k["func"] = np.array(LO.C7, object)[P.argmax(1)]
    k = k[(k.period == "stg") & k.win.isin(WINS)].drop(columns="period")
    k["DeviceId"] = k.DeviceId.str.lower()
    assert not k.DeviceId.isin(locked()).any()
    return k.rename(columns={"Detector": "det"}).reset_index(drop=True)


def work(args):
    dev, fk, tt = args
    import pyarrow.dataset as ds
    tab = ds.dataset(str(DI)).to_table(filter=ds.field("DeviceId") == dev,
                                       columns=["Detector", "t_on"]).to_pandas()
    tab = tab.drop_duplicates()
    ton = tab.t_on.astype("datetime64[us]")
    tab["t"] = ton.astype("int64").to_numpy() / 1e6
    tab["h"] = ton.dt.hour.to_numpy()
    on = {}
    for d, g in tab.groupby("Detector"):
        o = np.argsort(g.t.to_numpy(), kind="stable")
        on[int(d)] = (g.t.to_numpy()[o], g.h.to_numpy()[o])
    true_ph = dict(zip(tt.det, tt.target))
    drows, prows = [], []
    for win, (ts, secs) in WINS.items():
        f = fk[fk.win == win]
        if f.empty:
            continue
        t0 = pd.Timestamp(ts).value / 1e9
        t1 = t0 + secs
        n_on = {}
        for d in f.det.astype(int):
            t = on.get(d, (np.zeros(0),))[0]
            n_on[d] = int(((t >= t0) & (t < t1)).sum())
        f = f.assign(n_on=f.det.astype(int).map(n_on))
        drows.append(f.assign(DeviceId=dev))
        act = f[f.n_on >= LO.MIN_ON]
        dets = sorted(act.det.astype(int))
        pp = dict(zip(act.det.astype(int), act.pred_phase))
        func = dict(zip(act.det.astype(int), act.func))
        pairs = [(a, b) for a, b in itertools.combinations(dets, 2)
                 if pp[a] == pp[b] or (a in true_ph and b in true_ph and true_ph[a] == true_ph[b])]
        if not pairs:
            continue
        P = LO.signal_pair_cues(on, dets, pairs, t0, t1)
        nd = pd.Series(pp).groupby(pd.Series(pp)).transform("size").to_dict()
        P = LO.add_context(P, func, nd, secs / 3600.0)
        P["same_pred"] = [pp[a] == pp[b] for a, b in pairs]
        P["same_true"] = [a in true_ph and b in true_ph and true_ph[a] == true_ph[b]
                          for a, b in pairs]
        P["DeviceId"] = dev
        P["win"] = win
        prows.append(P)
    return (pd.concat(drows, ignore_index=True) if drows else None,
            pd.concat(prows, ignore_index=True) if prows else None)


def main(workers: int):
    OUT.mkdir(parents=True, exist_ok=True)
    ph, det = truth()
    ph.to_parquet(OUT / "ln1_truth_phase.parquet", index=False)
    det.to_parquet(OUT / "ln1_truth_det.parquet", index=False)
    log(f"truth: {len(ph)} phases / {ph.DeviceId.nunique()} signals, dist "
        f"{ph.n_lanes.value_counts().sort_index().to_dict()}; det rows {len(det)}, high-veh on kept "
        f"{int((det.high_veh & det.phase_kept).sum())}")
    fk = frame_preds()
    sig = sorted(set(ph.DeviceId) & set(fk.DeviceId))
    log(f"{len(sig)} signals with truth and frame rows")
    jobs = [(s, fk[fk.DeviceId == s], det[det.DeviceId == s]) for s in sig]
    D, P = [], []
    t0 = time.time()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, (d, p) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
                if d is not None:
                    D.append(d)
                if p is not None:
                    P.append(p)
                if i % 50 == 0:
                    log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    else:
        for i, j in enumerate(jobs):
            d, p = work(j)
            D.append(d)
            P.append(p)
            if i % 20 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    D = pd.concat([x for x in D if x is not None], ignore_index=True)
    P = pd.concat([x for x in P if x is not None], ignore_index=True)
    D.to_parquet(OUT / "ln1_dets.parquet", index=False)
    P.to_parquet(OUT / "ln1_pairs.parquet", index=False)
    log(f"wrote {len(D)} detector-windows, {len(P)} pairs ({time.time()-t0:.0f}s)")
    json.dump({"phases": int(len(ph)), "signals": len(sig), "pairs": int(len(P)),
               "det_windows": int(len(D))}, open(OUT / "ln1_cues.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.workers)
