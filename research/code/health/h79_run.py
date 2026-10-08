"""Note 79: score the current health (health_core v5 + note-56/77 fixes = the package copy) on fixed 30 min / 3 h /
24 h windows of two periods, and compute the candidate relative / self-consistency statistics next to it.

Periods: stg = Fri 18 - Mon 21 Sep 2026 (the period most weak labels were DERIVED from -> same-period = circular upper
bound); w40 = Sat 26 - Mon 28 Sep 2026 (independent re-observation: labels from stg, scored here).
Modes per window: prod = detectors=None exactly as the package calls it (a channel silent all window has no row);
listed = with the controller channel list (what a channel list from the user would add).
Inputs (phase / function / lanes spanned) = health4/inputs.parquet stg OOF answers of the matching window length.
Training signals only (folds_v4 minus locked_v2).  CPU, --procs <= 4.

    python h79_run.py [--procs 4] [--limit N] -> %DC_WORK%/health79/run.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_build as HB  # noqa: E402
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health79"
EVR = {"stg": H.DCW / "official" / "stg" / "cache" / "events", "w40": H.DCW / "health4" / "w40_events"}
WIN = {
    "stg": {"m30_a": ("2026-09-19 08:00", .5), "m30_b": ("2026-09-19 12:00", .5), "m30_c": ("2026-09-20 17:00", .5),
            "m30_d": ("2026-09-21 07:30", .5), "h3_a": ("2026-09-19 12:00", 3), "h3_b": ("2026-09-21 06:00", 3),
            "h24_a": ("2026-09-19 00:00", 24), "h24_b": ("2026-09-20 00:00", 24)},
    "w40": {"m30_a": ("2026-09-27 08:00", .5), "m30_b": ("2026-09-27 12:00", .5), "m30_c": ("2026-09-27 17:00", .5),
            "m30_d": ("2026-09-28 07:30", .5), "h3_a": ("2026-09-27 12:00", 3), "h3_b": ("2026-09-28 06:00", 3),
            "h24_a": ("2026-09-27 00:00", 24), "h24_b": ("2026-09-28 00:00", 24)},
}
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
KEEP = ["detector", "health_score", "status", "n_on", "cov_h", "pred_function", "pred_phase", "reason"]
IN = EXP = None


def init():
    global IN, EXP
    x = pd.read_parquet(H.DCW / "health4" / "inputs.parquet")
    x = x[(x.period == "stg") & x.wgroup.isin(["m30", "h3", "h24"])]
    IN = {k: g for k, g in x.groupby(["wgroup", "DeviceId"])}
    EXP = HB.expected_channels()


def inputs(wg, dev):
    g = IN.get((wg, dev))
    if g is None:
        return {}, {}, {}, {}
    ph = dict(zip(g.detector.astype(int), g.pred_phase.astype(float)))
    fn = {int(d): {c: float(v) for c, v in zip(C7, row)} for d, row in zip(g.detector, g[[f"p_{c}" for c in C7]].to_numpy())}
    return ph, fn, dict(zip(g.detector.astype(int), g.n_lanes_spanned)), dict(zip(g.detector.astype(int), g.top_prob))


# ------------------------------------------------------------------ candidate statistics (hi-res only)
def intervals(ev, t0, t1):
    """ON intervals per channel inside [t0, t1): {det: (on_s, off_s)} in seconds from t0 (continuous ON = from an ON to
    the next OFF, as health_core)."""
    e = ev[(ev.Timestamp >= t0) & (ev.Timestamp < t1) & ev.EventId.isin((81, 82)) & (ev.Parameter <= hc.MAXCH)]
    e = e.drop_duplicates()
    T = (t1 - t0).total_seconds()
    out = {}
    for d, g in e.groupby("Parameter"):
        g = g.sort_values(["Timestamp", "EventId"], ascending=[True, False])     # ON (82) before OFF (81)
        t = (g.Timestamp - t0).dt.total_seconds().to_numpy()
        k = g.EventId.to_numpy()
        on, off = [], []
        cur = None
        for ti, ki in zip(t, k):
            if ki == 82:
                if cur is None:
                    cur = ti
            elif cur is not None:
                on.append(cur)
                off.append(ti)
                cur = None
        if cur is not None:
            on.append(cur)
            off.append(T)
        out[int(d)] = (np.array(on), np.array(off), int((k == 82).sum()))
    return out


def greens(ev, t0, t1):
    e = ev[(ev.Timestamp >= t0) & (ev.Timestamp < t1) & ev.EventId.isin((1, 8))].sort_values("Timestamp")
    out = {}
    for p, g in e.groupby("Parameter"):
        t = (g.Timestamp - t0).dt.total_seconds().to_numpy()
        k = g.EventId.to_numpy()
        gs = []
        cur = None
        for ti, ki in zip(t, k):
            if ki == 1:
                cur = ti
            elif cur is not None:
                gs.append((cur, ti))
                cur = None
        out[float(p)] = np.array(gs) if gs else np.zeros((0, 2))
    return out


def active_in(on, off, G, pad=0.0):
    """Per green [gs, ge]: number of ON starts in it, and whether occupied at any time in it."""
    if len(G) == 0:
        return np.zeros(0), np.zeros(0, bool)
    gs, ge = G[:, 0], G[:, 1] + pad
    n = np.searchsorted(on, ge, "left") - np.searchsorted(on, gs, "left")
    # occupied at gs: last ON before gs has off > gs
    j = np.searchsorted(on, gs, "right") - 1
    occ0 = (j >= 0) & (off[np.clip(j, 0, None)] > gs) if len(on) else np.zeros(len(gs), bool)
    return n, (n > 0) | occ0


def cand_stats(ev, t0, t1, phase: dict, func: dict):
    """Per detector: best same-phase partner (15-min count corr), green-miss asymmetry vs it, count ratio vs it and vs
    same-function phase peers, half-window ratio drift vs the phase reference."""
    iv = intervals(ev, t0, t1)
    if not iv:
        return pd.DataFrame()
    G = greens(ev, t0, t1)
    T = (t1 - t0).total_seconds()
    nb = max(int(T // 900), 1)
    dets = sorted(iv)
    cnt15 = {d: np.bincount(np.minimum((iv[d][0] // 900).astype(int), nb - 1), minlength=nb) for d in dets}
    n = {d: len(iv[d][0]) for d in dets}
    fl = {d: max(v, key=v.get) for d, v in func.items()} if func else {}
    rows = []
    for d in dets:
        p = phase.get(d)
        r = dict(detector=d, c_n=n[d])
        if p is None or not np.isfinite(p):
            rows.append(r)
            continue
        mates = [k for k in dets if k != d and phase.get(k) == p and n[k] >= 20]
        r["c_nmates"] = len(mates)
        if not mates:
            rows.append(r)
            continue
        # best partner by 15-min correlation (ties: larger count)
        best, bc = None, -2.0
        if nb >= 4:
            for k in mates:
                a, b = cnt15[d], cnt15[k]
                c = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
                if np.isfinite(c) and (c > bc or (c == bc and n[k] > n[best])):
                    best, bc = k, c
        if best is None:                                # short window: the busiest phase-mate
            best = max(mates, key=lambda k: (n[k], -k))
            bc = np.nan
        r.update(c_partner=best, c_pcorr=bc, c_ratio=n[d] / max(n[best], 1))
        # green-miss asymmetry (a): greens where one of the pair is active and the other is not
        Gp = G.get(float(p), np.zeros((0, 2)))
        if len(Gp):
            nd_, ad = active_in(*iv[d][:2], Gp, pad=2.0)
            nk_, ak = active_in(*iv[best][:2], Gp, pad=2.0)
            busy_k = nk_ >= 2
            busy_d = nd_ >= 2
            r["c_g"] = int(busy_k.sum())
            r["c_miss"] = float((~ad[busy_k]).mean()) if busy_k.any() else np.nan
            r["c_miss_rev"] = float((~ak[busy_d]).mean()) if busy_d.any() else np.nan
            r["c_g_rev"] = int(busy_d.sum())
            r["c_ngreens"] = len(Gp)
        # same-function peers on the phase (lane-by-lane units of one role)
        f = fl.get(d)
        peers = [k for k in mates if fl.get(k) == f]
        if peers:
            r["c_peer_ratio"] = n[d] / np.median([n[k] for k in peers])
            r["c_npeer"] = len(peers)
        # drift: share of the phase reference, first vs second half (Poisson-safe log ratio)
        ref = np.sum([cnt15[k] for k in mates], axis=0)
        h = nb // 2
        if h >= 1:
            a1, a2 = cnt15[d][:h].sum(), cnt15[d][h:].sum()
            r1, r2 = ref[:h].sum(), ref[h:].sum()
            if r1 > 0 and r2 > 0:
                s1, s2 = (a1 + .5) / r1, (a2 + .5) / r2
                r["c_drift"] = float(np.log(s2 / s1))
                r["c_drift_n"] = int(a1 + a2)
                # Poisson z of the second-half count given the first-half share
                e2 = s1 * r2
                r["c_drift_z"] = float((a2 - e2) / np.sqrt(max(e2, 1.0)))
        rows.append(r)
    return pd.DataFrame(rows)


def one(dev):
    out = []
    for per in ("stg", "w40"):
        p = EVR[per] / f"DeviceId={dev}"
        if not p.is_dir():
            continue
        ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
        ev = ev[["Timestamp", "EventId", "Parameter"]]
        for w, (s, hrs) in WIN[per].items():
            t0 = pd.Timestamp(s)
            t1 = t0 + pd.Timedelta(hours=hrs)
            wg = w.split("_")[0]
            ph, fn, ln, pc = inputs(wg, dev)
            parts = []
            for mode, dl in (("prod", None), ("listed", EXP.get(dev))):
                try:
                    h = hc.health(ev, t0, t1, dl, ph or None, fn or None, ln or None, pc or None)
                except Exception as exc:                     # noqa: BLE001
                    print("ERR", dev, per, w, mode, type(exc).__name__, exc, flush=True)
                    continue
                if not len(h):
                    continue
                x = h[[c for c in KEEP + [c for c in h if c.startswith("s_")] if c in h]].copy()
                x.insert(0, "mode", mode)
                parts.append(x)
            if not parts:
                continue
            x = pd.concat(parts, ignore_index=True)
            try:
                c = cand_stats(ev, t0, t1, ph, fn)
            except Exception as exc:                         # noqa: BLE001
                print("ERRc", dev, per, w, type(exc).__name__, exc, flush=True)
                c = pd.DataFrame()
            if len(c):
                x = x.merge(c, on="detector", how="left")
            x.insert(0, "window", w)
            x.insert(0, "period", per)
            x.insert(0, "DeviceId", dev)
            out.append(x)
    return pd.concat(out, ignore_index=True) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="run.parquet")
    a = ap.parse_args()
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = sorted({d.lower() for d in folds.DeviceId} - locked)
    if a.limit:
        devs = devs[:: max(1, len(devs) // a.limit)][: a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with Pool(a.procs, initializer=init) as p:
        res = []
        for k, r in enumerate(p.imap_unordered(one, devs, chunksize=1)):
            if r is not None:
                res.append(r)
            if k % 50 == 0:
                print(k, f"{time.time() - t:.0f}s", flush=True)
    df = pd.concat(res, ignore_index=True)
    assert not df.DeviceId.isin(locked).any()
    df.to_parquet(OUT / a.out)
    print(df.shape, f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
