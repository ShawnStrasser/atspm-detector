"""Note 47: per-detector statistics behind the health v5 changes, Sept 2026 66-h window, training signals only
(folds_v4 minus locked_v2; no fault events).

  (a) continuous ON: an ON followed by more ONs before the OFF is ONE continuous ON (extension: the next vehicle
      arrives before the extension ends) - its length runs from the first ON to the next OFF.  Compared with the
      v3/v4 "ON with no OFF, held to the next ON" episodes.
  (b) short ONs: share of ON->OFF durations <= 0.2 s (<= 2 ticks) overall, by 3-h clock block and night vs day;
      pulse mode learned from the data (share of one-tick ONs).
  (c) night share vs the same-phase reference (a detector that stops counting at night while its phase does not).

    python h5_explore.py -> %DC_WORK%/health5/explore.parquet
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health5"
EV = H.DCW / "official" / "stg" / "cache" / "events"
T0, T1 = pd.Timestamp("2026-09-18 16:15"), pd.Timestamp("2026-09-21 10:25")
PH = None


def init():
    global PH
    x = pd.read_parquet(H.DCW / "health4" / "inputs.parquet", columns=["period", "DeviceId", "detector", "wgroup",
                                                                         "pred_phase", "pred_function"])
    x = x[(x.period == "stg") & (x.wgroup == "full")]
    PH = {k: g for k, g in x.groupby("DeviceId")}


def one(dev):
    p = EV / f"DeviceId={dev}"
    if not p.is_dir():
        return None
    e = ds.dataset(p).to_table(filter=ds.field("EventId").isin([81, 82])).to_pandas()
    e = e[(e.Timestamp >= T0) & (e.Timestamp < T1) & (e.Parameter <= hc.MAXCH)].drop_duplicates()
    if e.empty:
        return None
    t = (e.Timestamp - T0).dt.total_seconds().to_numpy()
    eid, par = e.EventId.to_numpy().astype(int), e.Parameter.to_numpy().astype(int)
    o = np.lexsort((eid != 82, t, par))
    t, eid, par = t[o], eid[o], par[o]
    hr = ((T0 + pd.to_timedelta(t, unit="s")).hour).to_numpy()
    g = PH.get(dev)
    ph = dict(zip(g.detector, g.pred_phase)) if g is not None else {}
    fn = dict(zip(g.detector, g.pred_function)) if g is not None else {}
    rows = []
    cnt = {}
    for c in np.unique(par):
        m = par == c
        tt, ee, hh = t[m], eid[m], hr[m]
        on = np.where(ee == 82)[0]
        nxt_e = np.r_[ee[1:], 0]
        nxt_t = np.r_[tt[1:], np.nan]
        prv_e = np.r_[0, ee[:-1]]
        # single ON -> OFF durations
        sing = on[nxt_e[on] == 81]
        dur = nxt_t[sing] - tt[sing]
        dh = hh[sing]
        # ON -> ON (no OFF between): old episode = to the next ON; continuous ON = chain start to the next OFF
        oo = on[nxt_e[on] == 82]
        old_ep = (nxt_t[oo] - tt[oo]).max() if len(oo) else 0.0
        # chains: an ON whose previous event is not an ON starts a chain; it ends at the next OFF
        starts = on[prv_e[on] != 82]
        offs = np.where(ee == 81)[0]
        k = np.searchsorted(offs, starts)
        endt = np.where(k < len(offs), tt[offs[np.minimum(k, len(offs) - 1)]], (T1 - T0).total_seconds())
        clen = endt - tt[starts]
        nin = np.diff(np.r_[np.searchsorted(on, starts), len(on)])        # ONs in each chain
        r = dict(DeviceId=dev, detector=int(c), n_on=len(on), n_onon=len(oo), old_ep=old_ep,
                 cont_max=float(clen.max()) if len(clen) else 0.0,
                 cont_max_n=int(nin[clen.argmax()]) if len(clen) else 0,
                 single_max=float(dur.max()) if len(dur) else 0.0, n_dur=len(dur),
                 pred_phase=ph.get(int(c), np.nan), pred_function=fn.get(int(c)))
        if len(dur) >= 20:
            r["tick1"] = float((dur < 0.15).mean())
            r["short2"] = float((dur < 0.25).mean())
            r["med_dur"] = float(np.median(dur))
            blk = dh // 3
            bs = [(dur[blk == b] < 0.25).mean() for b in range(8) if (blk == b).sum() >= 20]
            bm = [np.median(dur[blk == b]) for b in range(8) if (blk == b).sum() >= 20]
            r["short_blk_max"] = float(max(bs)) if bs else np.nan
            r["short_blk_min"] = float(min(bs)) if bs else np.nan
            r["med_blk_min"] = float(min(bm)) if bm else np.nan
            r["med_blk_max"] = float(max(bm)) if bm else np.nan
            nt, dy = (dh >= 21) | (dh < 5), (dh >= 7) & (dh < 19)
            r["n_night_dur"] = int(nt.sum())
            r["short_night"] = float((dur[nt] < 0.25).mean()) if nt.sum() >= 10 else np.nan
            r["short_day"] = float((dur[dy] < 0.25).mean()) if dy.sum() >= 20 else np.nan
            r["med_night"] = float(np.median(dur[nt])) if nt.sum() >= 10 else np.nan
            r["med_day"] = float(np.median(dur[dy])) if dy.sum() >= 20 else np.nan
        hn = hh[ee == 82]
        cnt[int(c)] = (int(((hn >= 21) | (hn < 5)).sum()), int(((hn >= 7) & (hn < 19)).sum()))
        rows.append(r)
    df = pd.DataFrame(rows)
    # night share vs the same-phase reference (other detectors on the predicted phase)
    ns = []
    for r in df.itertuples():
        sib = [d for d in cnt if d != r.detector and ph.get(d) == r.pred_phase and np.isfinite(r.pred_phase)]
        if len(sib) < 1:
            ns.append((np.nan, np.nan, np.nan))
            continue
        sn, sd = sum(cnt[d][0] for d in sib), sum(cnt[d][1] for d in sib)
        xn, xd = cnt[r.detector]
        exp_n = xd / max(sd, 1) * sn                      # night ONs expected from the day share
        ns.append((xn, exp_n, (xn + 0.5) / (exp_n + 0.5)))
    df[["n_night", "exp_night", "night_share_ratio"]] = ns
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = sorted({d.lower() for d in folds.DeviceId} - locked)
    with Pool(6, initializer=init) as p:
        res = [r for r in p.imap_unordered(one, devs, chunksize=4) if r is not None]
    df = pd.concat(res, ignore_index=True)
    assert not df.DeviceId.isin(locked).any()
    df.to_parquet(OUT / "explore.parquet", index=False)
    print(df.shape)


if __name__ == "__main__":
    main()
