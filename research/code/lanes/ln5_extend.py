"""Lane output (note 42) extended for note 54: out-of-fold lanes for EVERY frame signal and window (both periods, all
14 windows each), so the per-lane ATSPM decode (atspm_decode.py) can run wherever the function head runs.

fold   six fold pair models: note-42 recipe (LightGBM binary, 300 trees, 3 seeds, lane_output.FEATURES) fit on the
       labelled pair-windows of the OTHER five folds (ln2_pairmodel.load); span prior from those folds' signals
       (ln3_decode.span_prior); lam 3 / beta 0 (picked on all six folds in note 42) -> %DC_WORK%/lanes/fold_models/f<k>/
       (text models + lane_model.json, readable by lane_output.PairModel, numpy inference).
lanes  per signal: ONs from the interval cache of the period; predictions = frame v6e predicted phase + the function
       head's OOF argmax (--run, default function_v3e); cues / context / decode exactly as lane_output.lanes, with the
       pair model of the signal's own fold (a signal is never scored by a model that saw its lane labels; signals
       without lane labels were never in any training set) -> %DC_WORK%/lanes/ln5_lanes_<tag>.parquet
       (DeviceId, period, win, Detector, phase, lanes, lane_conf, n_on).

    python ln5_extend.py fold
    python ln5_extend.py lanes [--workers 5] [--run ...] [--tag v3e]
locked_v2 asserted absent. CPU only.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lane_output as LO  # noqa: E402
import ln1_cues as L1  # noqa: E402

OUT = L1.OUT
FM = OUT / "fold_models"
CACHE = {"dec": L1.DCW / "cache" / "det_intervals.parquet",
         "stg": L1.DCW / "official" / "stg" / "cache" / "det_intervals.parquet"}
RUN = "run_ad6ea6959f_exclude_min5_clean_valnc_h3"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fold():
    import lightgbm as lgb
    import ln2_pairmodel as L2
    import ln3_decode as L3
    P, T, D = L2.load()
    lab = P.labelled.to_numpy()
    det = pd.read_parquet(OUT / "ln1_truth_det.parquet")
    ph = pd.read_parquet(OUT / "ln1_truth_phase.parquet")
    fsig = D.groupby("DeviceId").fold.first()
    prm = dict(L2.PARAMS, n_jobs=6)
    for f in range(6):
        d = FM / f"f{f}"
        d.mkdir(parents=True, exist_ok=True)
        tr = lab & (P.fold != f).to_numpy()
        files = []
        for s in (0, 1, 2):
            m = lgb.LGBMClassifier(random_state=s, **prm).fit(P.loc[tr, LO.FEATURES], P.loc[tr, "same_lane"].astype(int))
            fn = f"lane_pair_s{s}.txt"
            m.booster_.save_model(str(d / fn))
            files.append(fn)
        prior = L3.span_prior(det, ph, D, set(fsig.index[fsig != f]))
        meta = {"pair_models": files, "features": LO.FEATURES, "span_prior": prior, "lam": 3.0, "beta": 0.0,
                "role_pen": 4.0, "min_on": LO.MIN_ON, "held_out_fold": f, "train_pair_windows": int(tr.sum())}
        json.dump(meta, open(d / "lane_model.json", "w"), indent=1)
        log(f"fold {f}: {int(tr.sum())} pair-windows")


def frame_preds(run: str) -> pd.DataFrame:
    import v3_retrain as V
    V.set_frame("v6e")
    k = pd.read_parquet(V.FEATS, columns=V.KEY + ["fold", "pred_phase", "det_n_on"])
    k["DeviceId"] = k.DeviceId.str.lower()
    assert not k.DeviceId.isin(L1.locked()).any()
    P = np.zeros((len(k), len(LO.C7)), np.float32)
    for s in (0, 1, 2):
        Ps, hv, got = V.load_oof(k, V.OUT / run, "first.all.wi", s, range(6))
        assert len(got) == 6
        P += Ps
    k["func"] = np.array(LO.C7, object)[P.argmax(1)]
    return k


def work(args):
    dev, period, fk, f = args
    import a2_features as A2F
    import pyarrow.dataset as ds
    pm = LO.PairModel(FM / f"f{int(f)}")
    tab = ds.dataset(str(CACHE[period])).to_table(filter=ds.field("DeviceId") == dev,
                                                  columns=["Detector", "t_on"]).to_pandas().drop_duplicates()
    ton = tab.t_on.astype("datetime64[us]")
    tab["t"] = ton.astype("int64").to_numpy() / 1e6
    tab["h"] = ton.dt.hour.to_numpy()
    on = {}
    for d, g in tab.groupby("Detector"):
        o = np.argsort(g.t.to_numpy(), kind="stable")
        on[int(d)] = (g.t.to_numpy()[o], g.h.to_numpy()[o])
    rows = []
    for name, t0w, secs in A2F.WINDOWS[period]:
        g = fk[fk.win == name]
        if g.empty:
            continue
        t0 = pd.Timestamp(t0w).value / 1e9
        t1 = t0 + secs
        n_on = {}
        for d in g.Detector.astype(int):
            t = on.get(d, (np.zeros(0),))[0]
            n_on[d] = int(((t >= t0) & (t < t1)).sum())
        gg = g.rename(columns={"pred_phase": "phase_use", "func": "func_use"})
        grp, func = LO.eligible(gg, n_on, pm.min_on)
        probs = {}
        pairs = [(a, b) for a, b in itertools.combinations(sorted(grp), 2) if grp[a] == grp[b]]
        if pairs:
            loc = {d: on[d] for d, k in n_on.items() if k >= pm.min_on and d in on}
            P = LO.signal_pair_cues(loc, sorted(loc), pairs, t0, t1)
            P = LO.add_context(P, func, LO.group_sizes(grp), secs / 3600.0)
            for a, b, p in zip(P.da, P.db, pm.predict(P)):
                probs[(int(a), int(b))] = float(p)
        _, dt = LO.decode_groups(dev, gg, n_on, secs / 3600.0, probs, pm.decoder, pm.min_on)
        dt["period"], dt["win"] = period, name
        dt["n_on"] = dt.Detector.map(n_on)
        rows.append(dt)
    return pd.concat(rows, ignore_index=True) if rows else None


def lanes(workers: int, run: str, tag: str):
    k = frame_preds(run)
    jobs = []
    for (dev, period), g in k.groupby(["DeviceId", "period"]):
        jobs.append((dev, period, g[["Detector", "win", "pred_phase", "func"]].copy(), int(g.fold.iloc[0])))
    log(f"{len(jobs)} signal-periods")
    out, t0 = [], time.time()
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            if r is not None:
                out.append(r)
            if i % 100 == 0:
                log(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)")
    D = pd.concat(out, ignore_index=True)
    D.to_parquet(OUT / f"ln5_lanes_{tag}.parquet", index=False)
    log(f"wrote {len(D):,} detector-windows, lane known {float((D.lanes != '').mean()):.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fold", "lanes"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--tag", default="v3e")
    a = ap.parse_args()
    fold() if a.stage == "fold" else lanes(a.workers, a.run, a.tag)
