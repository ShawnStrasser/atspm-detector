"""Note 83: chatter / rapid statistics in whole 0.1-s ticks vs float seconds, and the recalibrated limits.

health_core (candidate v3) compared OFF -> ON gaps (chatter, < 0.3 s) and ON -> ON intervals (rapid, < 0.5 / < 1 s)
in float seconds measured from the window start, so a gap of exactly 3 / 5 / 10 ticks was counted only sometimes
(note 82).  Candidate v4 compares whole ticks (strict <).  This script computes both versions of every statistic
behind the two rules on the note-79 windows (stg + w40, 4 x 30 min, 2 x 3 h, 2 x 24 h, prod mode = the log's own
channels) for every training detector, and then

  calib   new limits that keep the presumed-healthy fire rate of each limit: for each limit L_old of a statistic
          (chat (0.30, 0.60); rapid's per-group ioi_lt05 / ioi_lt1 / burst_frac limits, normal and lane-spanning),
          L_new = the smallest value v with  P_new(stat >= v) <= P_old(stat >= L_old)  on presumed-healthy rows
          (note-79 evalset cls healthy, tier N; >= 50 ONs, as the rules), pooled over windows and both periods;
          then the fire rates old vs new of the rules (chat score > 0, rapid >= 1, bad levels), healthy and all.

    python h83_ticks.py run [--procs 4]     -> %DC_WORK%/health83/ticks.parquet
    python h83_ticks.py calib               -> %DC_WORK%/health83/calib.json
Training signals only (folds_v4 minus locked_v2, asserted).  CPU, numpy / pandas, <= 4 processes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
OUT = DCW / "health83"
OLD = DCW / "final_v3_candidate_v3" / "health_core.py"
NEW = DCW / "final_v3_candidate_v4" / "health_core.py"
EVR = {"stg": DCW / "official" / "stg" / "cache" / "events", "w40": DCW / "health4" / "w40_events"}
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h79_run as R  # noqa: E402

WIN = R.WIN
STATS = ["chat_frac", "ioi_lt05", "ioi_lt1", "burst_frac"]
MODS = {}
SPAN = None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def init():
    global SPAN
    MODS["old"], MODS["new"] = _load("hc_old", OLD), _load("hc_new", NEW)
    x = pd.read_parquet(DCW / "health4" / "inputs.parquet",
                        columns=["period", "DeviceId", "detector", "wgroup", "n_lanes_spanned"])
    x = x[(x.period == "stg") & x.wgroup.isin(["m30", "h3", "h24"])]
    SPAN = {(w, d): dict(zip(g.detector.astype(int), g.n_lanes_spanned.fillna(0))) for (w, d), g in
            x.groupby(["wgroup", "DeviceId"])}


def stats_one(m, ev, t0, t1):
    B = m.events_to_bins(ev, t0, t1, None)
    if not len(B["dets"]):
        return pd.DataFrame()
    n = B["n_on"][:, B["cov"]].sum(1)
    st = pd.DataFrame({"detector": B["dets"].astype(int), "n_on": n,
                       "chat_frac": B["n_chat"].sum(1) / np.maximum(n, 1)})
    a = m.act_stats(ev, t0, t1, light=True)
    if a is not None and len(a):
        a = a.reindex(columns=list(a.columns) + [c for c in ("pulse_frac", "med_dur", "ioi_lt05", "ioi_lt1",
                                                             "burst_frac") if c not in a])
        st = st.merge(a[["detector", "a_n_on", "pulse_frac", "med_dur", "ioi_lt05", "ioi_lt1", "burst_frac"]],
                      on="detector", how="left")
    return st


def one(dev):
    out = []
    for per in ("stg", "w40"):
        p = EVR[per] / f"DeviceId={dev}"
        if not p.is_dir():
            continue
        ev = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(MODS["new"].ALLOWED))).to_pandas()
        ev = ev[["Timestamp", "EventId", "Parameter"]]
        for w, (s, hrs) in WIN[per].items():
            t0 = pd.Timestamp(s)
            t1 = t0 + pd.Timedelta(hours=hrs)
            o, nw = stats_one(MODS["old"], ev, t0, t1), stats_one(MODS["new"], ev, t0, t1)
            if not len(o):
                continue
            x = o.merge(nw, on="detector", suffixes=("_old", "_new"))
            x["span"] = x.detector.map(SPAN.get((w.split("_")[0], dev), {})).fillna(0)
            x.insert(0, "window", w)
            x.insert(0, "period", per)
            x.insert(0, "DeviceId", dev)
            out.append(x)
    return pd.concat(out, ignore_index=True) if out else None


def stage_run(a):
    folds = pd.read_csv(DCW / "folds_v4.csv")
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    devs = sorted({d.lower() for d in folds.DeviceId} - locked)
    if a.limit:
        devs = devs[:a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    res = []
    with Pool(a.procs, initializer=init) as p:
        for k, r in enumerate(p.imap_unordered(one, devs, chunksize=1)):
            if r is not None:
                res.append(r)
            if k % 100 == 0:
                print(k, f"{time.time() - t:.0f}s", flush=True)
    df = pd.concat(res, ignore_index=True)
    assert not df.DeviceId.str.lower().isin(locked).any()
    df.to_parquet(OUT / "ticks.parquet")
    print(df.shape, f"{time.time() - t:.0f}s", flush=True)


def _newlim(old, new, L):
    """smallest v (on the new statistic's support) with share(new >= v) <= share(old >= L)."""
    target = float((old >= L).mean())
    cand = np.unique(new[np.isfinite(new)])
    cand = cand[cand > 0]
    best = float(np.nanmax(new)) + 1e-6 if len(cand) else L
    for v in cand[::-1]:
        if (new >= v).mean() <= target + 1e-12:
            best = float(v)
        else:
            break
    return best, target, float((new >= best).mean())


def stage_calib(a):
    init()
    hc = MODS["old"]
    X = pd.read_parquet(OUT / "ticks.parquet")
    e = pd.read_parquet(DCW / "health79" / "evalset.parquet")
    hl = e[(e.cls == "healthy") & (e.tier == "N")][["DeviceId", "detector"]].assign(healthy=True)
    X = X.merge(hl, on=["DeviceId", "detector"], how="left")
    X["healthy"] = X.healthy.fillna(False).astype(bool)
    res = {"rows": int(len(X)), "healthy_rows": int(X.healthy.sum())}
    # ---- how much the statistics moved
    mv = {}
    for c in STATS:
        a_, b_ = X[f"{c}_old"].astype(float), X[f"{c}_new"].astype(float)
        m = a_.notna() & b_.notna()
        mv[c] = {"rows": int(m.sum()), "changed": int((np.abs(a_[m] - b_[m]) > 1e-12).sum()),
                 "mean_old": round(float(a_[m].mean()), 5), "mean_new": round(float(b_[m].mean()), 5),
                 "max_abs": round(float(np.abs(a_[m] - b_[m]).max()), 4)}
    res["moved"] = mv
    # ---- chatter limits (n >= 50)
    H = X[X.healthy & (X.n_on_new >= 50)]
    lo = _newlim(H.chat_frac_old.to_numpy(float), H.chat_frac_new.to_numpy(float), hc.LIM["chat"][0])
    hi = _newlim(H.chat_frac_old.to_numpy(float), H.chat_frac_new.to_numpy(float), hc.LIM["chat"][1])
    res["chat"] = {"old": list(hc.LIM["chat"]), "new": [round(lo[0], 4), round(hi[0], 4)],
                   "healthy_share_old_new": [[lo[1], lo[2]], [hi[1], hi[2]]], "n": int(len(H))}
    # ---- rapid limits per behaviour group (normal; spanning = same ratio, too few spanning rows)
    H = X[X.healthy & (X.a_n_on_new >= hc.RAPID_MIN_ON)].copy()
    H["grp"] = hc.behaviour_group(H.pulse_frac_new, H.med_dur_new)
    newlim, detail = {}, {}
    for g, lims in hc.RAPID_LIM.items():
        Hg = H[H.grp == g]
        nl = []
        for j, c in enumerate(("ioi_lt05", "ioi_lt1", "burst_frac")):
            v = _newlim(Hg[f"{c}_old"].to_numpy(float), Hg[f"{c}_new"].to_numpy(float), lims[j])
            nl.append(round(v[0], 4))
            detail[f"{g}.{c}"] = {"old": lims[j], "new": round(v[0], 4), "share_old": round(v[1], 5),
                                  "share_new": round(v[2], 5), "n": int(len(Hg))}
        newlim[g] = tuple(nl)
    span = {g: tuple(round(sp * (nn / o if o > 0 else 1.0), 4) for sp, nn, o in
                     zip(hc.RAPID_LIM_SPAN[g], newlim[g], hc.RAPID_LIM[g])) for g in hc.RAPID_LIM_SPAN}
    res["rapid_lim_new"], res["rapid_lim_span_new"], res["rapid_detail"] = newlim, span, detail
    # ---- fire rates old vs new (rule level), healthy and everyone, by period / window length
    def rapid_ratio(df, sfx, lim, lim_span):
        g = hc.behaviour_group(df[f"pulse_frac{sfx}"], df[f"med_dur{sfx}"])
        sp = df.span.to_numpy(float) >= 2
        L = np.array([(lim_span if s else lim).get(k, (np.nan,) * 3) for k, s in zip(g, sp)], float).reshape(-1, 3)
        v = df[[f"ioi_lt05{sfx}", f"ioi_lt1{sfx}", f"burst_frac{sfx}"]].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nanmax(np.where(np.isfinite(v), v / L, np.nan), axis=1)
        return np.where(df[f"a_n_on{sfx}"].to_numpy(float) >= hc.RAPID_MIN_ON, r, np.nan)
    X["rp_old"] = rapid_ratio(X, "_old", hc.RAPID_LIM, hc.RAPID_LIM_SPAN)
    X["rp_new"] = rapid_ratio(X, "_new", newlim, span)
    X["rp_new_oldlim"] = rapid_ratio(X, "_new", hc.RAPID_LIM, hc.RAPID_LIM_SPAN)
    cl = res["chat"]["new"]
    rates = {}
    X["wl"] = X.window.str.split("_").str[0]
    for pop, P in (("healthy", X[X.healthy]), ("all", X)):
        for (per, wl), g in P.groupby(["period", "wl"]):
            n50 = g.n_on_new >= 50
            rates[f"{pop}.{per}.{wl}"] = {
                "n": int(len(g)),
                "chat_suspect_old": round(100 * float(((g.chat_frac_old >= hc.LIM['chat'][0]) & (g.n_on_old >= 50)).mean()), 3),
                "chat_suspect_new": round(100 * float(((g.chat_frac_new >= cl[0]) & n50).mean()), 3),
                "chat_suspect_new_oldlim": round(100 * float(((g.chat_frac_new >= hc.LIM['chat'][0]) & n50).mean()), 3),
                "chat_bad_old": round(100 * float(((g.chat_frac_old >= hc.LIM['chat'][1]) & (g.n_on_old >= 50)).mean()), 3),
                "chat_bad_new": round(100 * float(((g.chat_frac_new >= cl[1]) & n50).mean()), 3),
                "rapid_suspect_old": round(100 * float((g.rp_old >= 1).mean()), 3),
                "rapid_suspect_new": round(100 * float((g.rp_new >= 1).mean()), 3),
                "rapid_suspect_new_oldlim": round(100 * float((g.rp_new_oldlim >= 1).mean()), 3),
                "rapid_bad_old": round(100 * float((g.rp_old >= 2).mean()), 3),
                "rapid_bad_new": round(100 * float((g.rp_new >= 2).mean()), 3)}
    res["fire_rates_pct"] = rates
    # flips: detector-windows whose suspect call changes
    res["flips_all"] = {
        "chat_on": int(((X.chat_frac_new >= cl[0]) & (X.n_on_new >= 50) & ~((X.chat_frac_old >= hc.LIM['chat'][0]) & (X.n_on_old >= 50))).sum()),
        "chat_off": int((~((X.chat_frac_new >= cl[0]) & (X.n_on_new >= 50)) & (X.chat_frac_old >= hc.LIM['chat'][0]) & (X.n_on_old >= 50)).sum()),
        "rapid_on": int(((X.rp_new >= 1) & ~(X.rp_old >= 1)).sum()),
        "rapid_off": int((~(X.rp_new >= 1) & (X.rp_old >= 1)).sum())}
    json.dump(res, open(OUT / "calib.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "fire_rates_pct"}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "calib"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"run": stage_run, "calib": stage_calib}[a.stage](a)
