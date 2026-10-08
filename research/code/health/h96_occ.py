"""Note 96: context-aware health - occupancy and traffic-reference statistics per detector and window (data study).

For every training signal (folds_v4 minus locked_v2; asserted) and every note-79 w40 window (4 x 30 min, 2 x 3 h,
2 x 24 h) this computes, from the hi-res log only (81 / 82 via the package health_core.events_to_bins: de-duplicated,
dummy channels > 64 dropped, continuous-ON occupancy) plus the classifier's own outputs (stg OOF phase / function /
lanes of the matching window length, health4/inputs.parquet - the note-79 inputs):

  * 15-min (5-min for 30-min windows) counts and occupancy (fraction of time ON) per detector;
  * a TRAFFIC REFERENCE: counts of the predicted Advance / Count detectors on the same predicted phase (self excluded),
    else on the signal, else all other detectors;
  * a CONGESTION index per bin: median over the signal's other detectors of (bin occupancy / own window-mean occupancy);
  * correlations of own counts and own occupancy with the traffic reference and with the congestion index;
  * same-signal same-type peers (occupancy and share);
  * the longest continuous-ON stretch (5-min bins >= 99 % ON) with the phase peers' occupancy / counts during it.

Writes %DC_WORK%/health96/feat.parquet (per DeviceId x window x detector) and bins.parquet (15-min bins, 5-min for
30-min windows, for the per-type normal-pattern study).  CPU, --procs <= 4.  No fault events, no prints/config.

    python h96_occ.py [--procs 4] [--limit N]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

warnings.filterwarnings("ignore")
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "health96"
EVD = DCW / "health4" / "w40_events"
spec = importlib.util.spec_from_file_location("pkg_hc", DCW / "final_v3_candidate_v4f" / "health_core.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

WIN = {"m30_a": ("2026-09-27 08:00", .5), "m30_b": ("2026-09-27 12:00", .5), "m30_c": ("2026-09-27 17:00", .5),
       "m30_d": ("2026-09-28 07:30", .5), "h3_a": ("2026-09-27 12:00", 3), "h3_b": ("2026-09-28 06:00", 3),
       "h24_a": ("2026-09-27 00:00", 24), "h24_b": ("2026-09-28 00:00", 24)}
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
TRAF = ("Advance", "Count")
IN = None


def init():
    global IN
    x = pd.read_parquet(DCW / "health4" / "inputs.parquet")
    x = x[(x.period == "stg") & x.wgroup.isin(["m30", "h3", "h24"])]
    x["DeviceId"] = x.DeviceId.str.lower()
    IN = {k: g for k, g in x.groupby(["wgroup", "DeviceId"])}


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def one(dev_dir):
    dev = dev_dir.name.split("=", 1)[1].lower()
    e = ds.dataset(dev_dir).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    e = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
    feats, bins = [], []
    for w, (s, h) in WIN.items():
        t0 = pd.Timestamp(s)
        t1 = t0 + pd.Timedelta(hours=h)
        bs = 300 if h < 1 else 900
        ew = e[(e.Timestamp >= t0) & (e.Timestamp < t1)]
        if not ew.EventId.isin((81, 82)).any():
            continue
        B = hc.events_to_bins(ew, t0, t1, bin_s=bs)
        B5 = hc.events_to_bins(ew, t0, t1, bin_s=300) if bs != 300 else B
        dets = B["dets"]
        n = B["n_on"].astype(float)
        occ = B["occ"].astype(float) / bs
        cov = B["cov"]
        n[:, ~cov] = np.nan
        occ[:, ~cov] = np.nan
        occ5 = B5["occ"].astype(float) / 300
        occ5[:, ~B5["cov"]] = np.nan
        g = IN.get((w[:-2] if w.startswith("m30") else w.split("_")[0], dev))
        ph, fn, ln, tp = {}, {}, {}, {}
        if g is not None:
            ph = dict(zip(g.detector.astype(int), g.pred_phase.astype(float)))
            fn = dict(zip(g.detector.astype(int), g.pred_function))
            ln = dict(zip(g.detector.astype(int), g.n_lanes_spanned))
            tp = dict(zip(g.detector.astype(int), g.top_prob))
        F = np.array([fn.get(int(d), None) for d in dets], dtype=object)
        P = np.array([ph.get(int(d), np.nan) for d in dets], dtype=float)
        tot = np.nansum(n, 1)
        mocc = np.nanmean(occ, 1)
        # congestion index per bin: median over detectors of occ / own mean occ (detectors with mean occ >= 1 %)
        rel = occ / np.where(mocc[:, None] > 0.01, mocc[:, None], np.nan)
        istraf = np.isin(F, TRAF)
        for i, d in enumerate(dets):
            oth = np.arange(len(dets)) != i
            same = oth & (P == P[i]) if np.isfinite(P[i]) else np.zeros(len(dets), bool)
            if (same & istraf).any():
                rk, rm = "phase", same & istraf
            elif (oth & istraf).any():
                rk, rm = "signal", oth & istraf
            elif oth.any():
                rk, rm = "all", oth
            else:
                rk, rm = "none", oth
            ref = np.nansum(n[rm], 0) if rm.any() else np.full(n.shape[1], np.nan)
            ref[~cov] = np.nan
            cong = np.nanmedian(rel[oth], 0) if oth.sum() >= 3 else np.full(n.shape[1], np.nan)
            phocc = np.nanmean(occ[same], 0) if same.any() else np.full(n.shape[1], np.nan)
            # same-type peers on the signal
            pt = oth & (F == F[i]) if F[i] is not None else np.zeros(len(dets), bool)
            # longest run of 5-min bins >= 99 % ON
            on5 = np.nan_to_num(occ5[i]) >= 0.99
            a_, b_ = hc._runs(on5)
            if len(a_):
                k = int(np.argmax(b_ - a_))
                ra, rb = int(a_[k]), int(b_[k])
                ph5 = np.nanmean(occ5[np.where(same)[0]][:, ra:rb]) if same.any() else np.nan
                ph5_all = np.nanmean(occ5[np.where(same)[0]]) if same.any() else np.nan
                sg5 = np.nanmean(occ5[np.where(oth)[0]][:, ra:rb]) if oth.any() else np.nan
                sg5_all = np.nanmean(occ5[np.where(oth)[0]]) if oth.any() else np.nan
                n5 = B5["n_on"].astype(float)
                refn = (np.nansum(n5[np.where(rm)[0]][:, ra:rb]) / max(rb - ra, 1)) if rm.any() else np.nan
                refn_all = (np.nansum(n5[np.where(rm)[0]]) / n5.shape[1]) if rm.any() else np.nan
                nheld = int(((occ5[np.where(oth)[0]][:, ra:rb] >= 0.99).mean(1) >= 0.8).sum()) if oth.any() else 0
                run_min, run_t0 = 5.0 * (rb - ra), t0 + pd.Timedelta(minutes=5 * ra)
            else:
                run_min, run_t0, ph5, ph5_all, sg5, sg5_all, refn, refn_all, nheld = 0.0, pd.NaT, *[np.nan] * 6, 0
            feats.append(dict(
                DeviceId=dev, window=w, detector=int(d), fn=F[i], phase=P[i], lanes=ln.get(int(d), np.nan),
                top_prob=tp.get(int(d), np.nan), hours=h, bin_s=bs, n_on=tot[i], occ=mocc[i],
                occ_p90=np.nanquantile(occ[i], .9), occ_max=np.nanmax(occ[i]), mean_on_s=(mocc[i] * h * 3600 /
                                                                                       max(tot[i], 1)),
                max5=float(np.nanmax(B5["n_on"][i])), ref_kind=rk, ref_n=np.nansum(ref),
                share_ref=tot[i] / max(np.nansum(ref), 1),
                c_cnt_ref=_corr(n[i], ref), c_occ_ref=_corr(occ[i], ref), c_occ_cong=_corr(occ[i], cong),
                c_cnt_cong=_corr(n[i], cong), c_occ_phocc=_corr(occ[i], phocc),
                cong_max=np.nanmax(cong) if np.isfinite(cong).any() else np.nan,
                peer_n=int(pt.sum()), peer_occ_med=np.nanmedian(mocc[pt]) if pt.any() else np.nan,
                peer_cnt_med=np.nanmedian(tot[pt]) if pt.any() else np.nan,
                run_min=run_min, run_t0=run_t0, run_phocc=ph5, run_phocc_all=ph5_all, run_sgocc=sg5,
                run_sgocc_all=sg5_all, run_refn=refn, run_refn_all=refn_all, run_nheld=nheld))
            if True:
                bins.append(pd.DataFrame(dict(DeviceId=dev, window=w, detector=int(d), fn=F[i], b=np.arange(n.shape[1]),
                                              n=n[i], occ=occ[i], ref=ref, cong=cong)))
    return pd.DataFrame(feats), (pd.concat(bins) if bins else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    folds = set(pd.read_csv(DCW / "folds_v4.csv").DeviceId.str.lower())
    dirs = [p for p in sorted(EVD.iterdir()) if p.name.split("=", 1)[1].lower() in folds - locked]
    assert not any(p.name.split("=", 1)[1].lower() in locked for p in dirs)
    if a.limit:
        dirs = dirs[:a.limit]
    t = time.time()
    with Pool(a.procs, initializer=init) as pool:
        res = pool.map(one, dirs, chunksize=4)
    F = pd.concat([r[0] for r in res], ignore_index=True)
    Bn = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
    F.to_parquet(OUT / ("feat.parquet" if not a.limit else "feat_smoke.parquet"))
    Bn.to_parquet(OUT / ("bins.parquet" if not a.limit else "bins_smoke.parquet"))
    print(len(dirs), "signals", len(F), "rows", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
