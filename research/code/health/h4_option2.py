"""Note 46 PART 2 - "option 2": classify -> health v4 -> remove the bad PERIODS from the classifier's input
window -> classify again.  Function side, six folds out of fold.

Only signal-windows of the function frame (v6, both periods, 22 windows) that overlap a listed bad period of
ANY detector of the signal (health_v4.parquet, the site's full-window health) can change; every other row is
identical in both passes by construction.  For those signal-windows the function head is run twice through
the package path (h4_pkg: phase fixed at the frame's out-of-fold `pred_phase` / `top_prob` of that window,
function = the fold model that never saw the signal):
    pass1   the raw window
    pass2a  each detector's own actuations inside its bad periods removed (a stuck ON disappears; a spike's
            or rapid hour's extra ONs disappear; a silent period has nothing to remove)
    pass2b  the union of the signal's bad periods cut out of the window for every channel (colour events too)
Locked never read.

    python h4_option2.py run  -> %DC_WORK%/health4/option2_rows.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackA"))
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health4"
EVR = {"stg": H.DCW / "official" / "stg" / "cache" / "events", "dec": H.DCW / "cache" / "events"}
ALLOWED = [1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 131, 150, 173]
FR = PER = FO = SC = None


def windows():
    import a2_features as A2F
    return {(per, n): (t0, t0 + pd.Timedelta(seconds=int(sec))) for per, ws in A2F.WINDOWS.items() for n, t0, sec in ws}


def jobs(health_file: Path) -> pd.DataFrame:
    """(period, DeviceId, win) signal-windows overlapping any listed bad period."""
    h = pd.read_parquet(health_file, columns=["period", "DeviceId", "detector", "bad_periods"])
    h = h[h.bad_periods.fillna("[]").ne("[]")]
    rows = []
    for r in h.itertuples():
        for b in json.loads(r.bad_periods):
            rows.append((r.period, r.DeviceId.lower(), int(r.detector), pd.Timestamp(b["start"]), pd.Timestamp(b["end"]),
                         b["what"]))
    bp = pd.DataFrame(rows, columns=["period", "DeviceId", "detector", "start", "end", "what"])
    W = windows()
    out = []
    for (per, n), (w0, w1) in W.items():
        x = bp[(bp.period == per) & (bp.start < w1) & (bp.end > w0)]
        for dev in x.DeviceId.unique():
            out.append((per, dev, n))
    return pd.DataFrame(out, columns=["period", "DeviceId", "win"]), bp


def init(health_file):
    global FR, PER, FO, SC
    import v3_retrain as V
    V.set_frame("v6")
    fr = pd.read_parquet(V.FEATS, columns=V.KEY + ["pred_phase", "top_prob"])
    fr["DeviceId"] = fr.DeviceId.str.lower()
    FR = {k: g for k, g in fr.groupby(["period", "DeviceId", "win"])}
    _, bp = jobs(health_file)
    PER = {k: g for k, g in bp.groupby(["period", "DeviceId"])}
    f4 = pd.read_csv(H.DCW / "folds_v4.csv")
    FO = dict(zip(f4.DeviceId.str.lower(), f4.fold.astype(int)))
    SC = {}


def one(job):
    import h4_pkg
    per, dev, items = job
    k = FO[dev]
    if k not in SC:
        SC[k] = h4_pkg.FunctionScorer(k)
    p = EVR[per] / f"DeviceId={dev}"
    if not p.is_dir():
        return None
    ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(ALLOWED)).to_pandas()
    ev["DeviceId"] = dev
    W = windows()
    bp = PER[(per, dev)]
    out = []
    for win in items:
        g = FR.get((per, dev, win))
        if g is None:
            continue
        w0, w1 = W[(per, win)]
        ph = dict(zip(g.Detector.astype(int), g.pred_phase.astype(int)))
        pp = dict(zip(g.Detector.astype(int), g.top_prob.astype(float)))
        e = ev[(ev.Timestamp >= w0) & (ev.Timestamp < w1)]
        b = bp[(bp.start < w1) & (bp.end > w0)]
        # pass 2a: each detector's own actuations inside its own bad periods removed
        drop = np.zeros(len(e), bool)
        ts, par, isdet = e.Timestamp.to_numpy(), e.Parameter.to_numpy(), e.EventId.isin([81, 82]).to_numpy()
        for r in b.itertuples():
            drop |= isdet & (par == r.detector) & (ts >= np.datetime64(r.start)) & (ts < np.datetime64(r.end))
        # pass 2b: the union of the signal's bad periods cut out for every channel
        cut = np.zeros(len(e), bool)
        for r in b.itertuples():
            cut |= (ts >= np.datetime64(r.start)) & (ts < np.datetime64(r.end))
        res = {}
        for name, ee in (("p1", e), ("p2a", e[~drop]), ("p2b", e[~cut])):
            try:
                res[name] = SC[k].score(ee, w0, w1, ph, pp)
            except Exception as exc:  # noqa: BLE001
                print(per, dev, win, name, type(exc).__name__, exc, flush=True)
                res[name] = None
        if res["p1"] is None:
            continue
        m = res["p1"].rename(columns=lambda c: c if c == "Detector" else f"{c}_p1")
        for name in ("p2a", "p2b"):
            if res[name] is not None:
                m = m.merge(res[name].rename(columns=lambda c: c if c == "Detector" else f"{c}_{name}"),
                            on="Detector", how="left")
        own = b.groupby("detector").what.agg(lambda s: ",".join(sorted(set(s))))
        m["own_period"] = m.Detector.map(own)
        m["n_dropped_a"], m["secs_cut_b"] = int(drop.sum()), float(
            sum((min(r.end, w1) - max(r.start, w0)).total_seconds() for r in b.itertuples()))
        out.append(m.assign(period=per, DeviceId=dev, win=win, fold=k))
    return pd.concat(out, ignore_index=True) if out else None


def run(a):
    hf = H.HB / a.health
    J, bp = jobs(hf)
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not J.DeviceId.isin(locked).any()
    print("bad periods", len(bp), bp.what.value_counts().to_dict(), "| signal-windows", len(J),
          J.groupby("period").size().to_dict(), flush=True)
    todo = [(per, dev, list(g.win)) for (per, dev), g in J.groupby(["period", "DeviceId"])]
    t = time.time()
    res = []
    with Pool(a.procs, initializer=init, initargs=(hf,)) as p:
        for n, r in enumerate(p.imap_unordered(one, todo, chunksize=1)):
            if r is not None:
                res.append(r)
            if n % 50 == 0:
                print(n, len(todo), f"{time.time() - t:.0f}s", flush=True)
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(OUT / a.out, index=False)
    print(df.shape, f"{time.time() - t:.0f}s")


def evaluate(a):
    """Second pass vs first on the evaluation sets.  Rows outside the re-run signal-windows are identical in
    both passes; the difference is (right after - right before) on the re-run rows / all rows of the set."""
    import v3_retrain as V
    V.set_frame("v6")
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH = 5, True, True, True
    fr, _ = V.load_feats()
    fr = fr[V.KEY + ["wgroup", "det_n_on", "fold", "health_flag"]].copy()
    lab = V.load_labels(fr, "exclude")
    es = V.eval_sets(fr, lab)
    v3 = pd.read_parquet(V.LABELS_V3)
    v3["DeviceId"] = v3.DeviceId.str.lower()
    pf, src = v3.print_function, v3.source.astype("string")
    user = src.eq("user_ruling").fillna(False)
    high = v3.print_source.eq("print") & v3.print_confidence.eq("high") & pf.isin(V.C7)
    noprint = ~v3.print_source.eq("print").fillna(False) | pf.isna()
    truth = np.where(user, v3.function, np.where(high, pf, np.where(noprint & v3.func7_v2.notna(), v3.func7_v2, None)))
    t = pd.DataFrame({"DeviceId": v3.DeviceId, "Detector": v3.detector.astype(fr.Detector.dtype), "truth": truth,
                      "validated": v3.validated.astype(str)})
    fr["DeviceId"] = fr.DeviceId.str.lower()
    T = fr[V.KEY].merge(t, on=["DeviceId", "Detector"], how="left")
    yA = T.truth.to_numpy(object)
    mA = pd.notna(yA) & np.isin(yA, V.C7) & (fr.det_n_on >= 5).to_numpy()
    mR = mA & ~T.validated.isin(["fail", "misconfigured"]).to_numpy()
    mC, yC = es["FIX"][0], lab.print_function.to_numpy(object)
    P = np.mean([V.load_oof(fr, V.OUT / "run_ad6ea6959f_exclude_min5_clean_valnc_h3", "first.all.wi", s, range(6))[0]
                 for s in range(3)], 0)
    base = np.array(V.C7, object)[P.argmax(1)]
    o = pd.read_parquet(OUT / a.out)
    o["Detector"] = o.Detector.astype(fr.Detector.dtype)
    k = fr[V.KEY + ["wgroup"]].reset_index().merge(o, on=["period", "DeviceId", "Detector", "win"], how="inner")
    C7 = np.array(V.C7, object)
    arg = lambda sfx: C7[k[[f"p_{c}_{sfx}" for c in V.C7]].fillna(-1).to_numpy().argmax(1)]  # noqa: E731
    k["y1"], k["y2a"], k["y2b"] = arg("p1"), arg("p2a"), arg("p2b")
    k.loc[k[f"p_Advance_p2b"].isna(), "y2b"] = k.y1        # nothing left after the cut: no answer change
    k["base"] = base[k["index"].to_numpy()]
    sig = fr.DeviceId.to_numpy()
    print(f"re-run rows {len(k):,} (frame rows), package pass1 argmax = frame OOF argmax on {np.mean(k.y1 == k.base):.3f}")
    for nm, m, y in (("A everything", mA, yA), ("realistic (A minus fail / misconfigured)", mR, yA),
                     ("C clean-label", mC, yC)):
        kk = k[m[k["index"]]]
        yt = y[kk["index"]]
        N = int(m.sum())
        print(f"\n{nm}: {N:,} rows, base acc {np.mean(base[m] == y[m]):.4f}; re-run rows in set {len(kk):,} "
              f"({kk.DeviceId.nunique()} signals), pass1 acc on them {np.mean(kk.y1 == yt):.4f}")
        for p2 in ("y2a", "y2b"):
            d = (kk[p2].to_numpy() == yt).astype(int) - (kk.y1.to_numpy() == yt).astype(int)
            # signal bootstrap over the whole set
            s_all = pd.Series(0.0, index=pd.unique(sig[m]))
            s_all = s_all.add(pd.Series(d, index=kk.DeviceId.to_numpy()).groupby(level=0).sum(), fill_value=0)
            n_sig = pd.Series(sig[m]).value_counts().reindex(s_all.index).to_numpy(float)
            rng = np.random.default_rng(0)
            ii = rng.integers(0, len(s_all), (2000, len(s_all)))
            bs = s_all.to_numpy()[ii].sum(1) / n_sig[ii].sum(1)
            print(f"   {p2}: changed {int((kk[p2] != kk.y1).sum())}, fixed {int((d > 0).sum())}, broken {int((d < 0).sum())}"
                  f" -> {100 * d.sum() / N:+.3f} pt [{100 * np.quantile(bs, .025):+.3f}, {100 * np.quantile(bs, .975):+.3f}];"
                  f" on the re-run rows {100 * d.mean():+.2f} pt; by window "
                  + ", ".join(f"{w} {100 * d[(kk.wgroup == w).to_numpy()].sum() / max((m & (fr.wgroup == w).to_numpy()).sum(), 1):+.3f}"
                              for w in ("m30", "h6", "full")))
        own = kk.own_period.notna().to_numpy()
        for p2 in ("y2a", "y2b"):
            d = (kk[p2].to_numpy() == yt).astype(int) - (kk.y1.to_numpy() == yt).astype(int)
            print(f"   subgroup {p2}: detectors with their OWN bad period in the window: n {own.sum()}, pass1 acc "
                  f"{np.mean((kk.y1.to_numpy() == yt)[own]):.3f}, delta {100 * d[own].mean():+.2f} pt; siblings only: "
                  f"n {(~own).sum()}, delta {100 * d[~own].mean():+.2f} pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "eval"])
    ap.add_argument("--health", default="health_v4.parquet")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default="option2_rows.parquet")
    a = ap.parse_args()
    run(a) if a.stage == "run" else evaluate(a)
