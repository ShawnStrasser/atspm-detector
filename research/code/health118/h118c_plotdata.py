"""Note 118c: saved plot data for the health v4 example sheet, rebuilt on the v4c scorer (score_v4c.py).

Everything a chart needs is computed HERE from saved data (event pulls, the v4c resolver output, the 116 / 117 / 118a /
118c work files) and written to %DC_WORK%/s118c/plot/; health_charts.py only draws from these files.

  rows.json        one record per sheet row: ids, label, status, category, the numbers the title uses
  day15.parquet    15-min actuations + % ON of the detector and its phase mates, all saved data
  ev.parquet       per-row evidence series (long: row, series, t, value)
  marks.json       per-row spans / points (stuck ONs, silent stretch, drop time, long ONs ...)

    python h118c_plotdata.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "health"))
sys.path.insert(0, str(HERE))
import health_core as hc  # noqa: E402
import h118_erratic as ER  # noqa: E402
import h104_resolve as R4  # noqa: E402
import score_v4c as SC  # noqa: E402

sys.path.insert(0, str(HERE.parent / "health116"))
import tod116 as T6  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s118c" / "plot"
W = DCW / "s118c"
EVD = DCW / "health4" / "w40_events"
S_WIN = {"m30_a": ("2026-09-27 08:00", .5), "m30_b": ("2026-09-27 12:00", .5), "m30_c": ("2026-09-27 17:00", .5),
         "m30_d": ("2026-09-28 07:30", .5), "h3_a": ("2026-09-27 12:00", 3), "h3_b": ("2026-09-28 06:00", 3),
         "h24_a": ("2026-09-27 00:00", 24), "h24_b": ("2026-09-28 00:00", 24)}
KEY = ["DeviceId", "window", "detector"]
FN = {"Yellow_Red": "Yellow-red"}

# (signal, detector, window, category) - 2-4 per category; user-reviewed cases first, healthy look-alikes marked by
# the resolver's own result (a look-alike is a row whose category is NOT in its v4 categories)
PICKS = [
    ("08CM405", 37, "h24_a", "stuck"), ("10037", 22, "h24_a", "stuck"), ("2B530", 60, "h24_b", "stuck"),
    ("01074", 4, "h24_b", "dropout"), ("10055", 9, "h24_a", "dropout"), ("12032", 40, "h24_b", "dropout"),
    ("2B316", 3, "h24_b", "level"), ("08073", 1, "h3_b", "level"),
    ("01062", 57, "h24_a", "night_drop"), ("04083", 2, "h24_b", "night_drop"), ("08CM405", 38, "h24_a", "night_drop"),
    ("08073", 7, "h24_b", "choppy"), ("2B557", 24, "h3_a", "choppy"), ("01064", 42, "h3_b", "choppy"),
    ("11021", 3, "h24_b", "rapid"), ("04083", 2, "h24_a", "rapid"), ("2B039", 18, "h24_b", "rapid"),
    ("2B058", 46, "h3_b", "rapid"),
    ("12032", 39, "h24_b", "volume"), ("08073", 7, "h24_b", "volume"), ("2C028", 37, "h3_b", "volume"),
    ("10055", 5, "h3_b", "chatter"), ("11021", 3, "h24_b", "chatter"),
    ("08052", 10, "h24_b", "occspk"), ("11021", 18, "h24_b", "occspk"),
    ("12032", 6, "h24_b", "prof"), ("04035", 52, "h24_b", "prof"), ("2B069", 23, "h24_b", "prof"),
    ("2B531", 35, "h24_b", "prof"),
    ("10028", 16, "h24_a", "shape24"), ("11021", 19, "h24_b", "shape24"), ("2B531", 35, "h24_a", "shape24"),
    ("2B316", 6, "h24_b", "count_on"), ("08019", 3, "h24_a", "count_on"),
    ("2B039", 23, "h24_b", "occ_hi"), ("08019", 16, "h24_b", "occ_hi"),
    ("2B068", 19, "h24_b", "C1"), ("2B058", 46, "h24_b", "C1"),
    ("04016", 20, "h24_b", "C2"), ("2B530", 42, "h24_b", "C2"),
]


def wwin(w):
    a, h = S_WIN[w]
    t0 = pd.Timestamp(a)
    return t0, t0 + pd.Timedelta(hours=h)


def day_of(w):
    t0, t1 = wwin(w)
    d0 = t0.normalize()
    return d0, d0 + pd.Timedelta(days=1)


_EV = {}


def events(dev):
    if dev not in _EV:
        p = EVD / f"DeviceId={dev}"
        if not p.is_dir():
            p = next(q for q in EVD.iterdir() if q.name.lower() == f"deviceid={dev}")
        e = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
        e = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
        e = e[~(e.EventId.isin((81, 82)) & (e.Parameter > hc.MAXCH))]
        _EV.clear()
        _EV[dev] = e.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    return _EV[dev]


def det_events(ev, d):
    """one channel's 81/82 events in time order (ON before OFF at the same instant): times (s since epoch), ids."""
    x = ev[ev.EventId.isin((81, 82)) & (ev.Parameter == d)]
    x = x.sort_values(["Timestamp", "EventId"], ascending=[True, False], kind="stable")
    return x.Timestamp.to_numpy(), x.EventId.to_numpy()


def cont_ons(t, e, tend):
    """continuous ONs (start ON -> next OFF): start times, durations (s), and re-logged ON times (ON after ON)."""
    prev_on = np.r_[False, e[:-1] == 82]
    start = (e == 82) & ~prev_on
    relog = (e == 82) & prev_on
    offs = np.where(e == 81)[0]
    si = np.where(start)[0]
    k = np.searchsorted(offs, si)
    end = np.where(k < len(offs), t[offs[np.minimum(k, len(offs) - 1)]] if len(offs) else np.datetime64(tend),
                   np.datetime64(tend))
    dur = (end - t[si]) / np.timedelta64(1, "s")
    return t[si], dur, t[relog]


def det_label(d, ph, fn):
    p = f"P{int(ph)} " if ph == ph and ph is not None else ""
    return f"det {int(d)}: {p}{FN.get(fn, fn)}"


def tod(i, r, BF, REF6, labs, d0, evs, rec, mk):
    """the time-of-day picture: its 24 h counts AND % ON against its type's normal band (the 118c band = +-4 robust
    SD, the finding limit), its phase mates (counts scaled to its daily total; % ON as measured); night-check numbers."""
    f = BF[(BF.DeviceId == r.DeviceId) & (BF.window == r.window) & (BF.detector == r.detector)].iloc[0]
    ht = d0 + pd.to_timedelta(np.arange(24) * 3600 + 1800, unit="s")

    def add(nm, v):
        evs.append(pd.DataFrame({"row": i, "series": nm, "t": ht, "value": np.asarray(v, float)}))
    H = range(24)
    add("count_h", [f[f"n{h}"] for h in H])
    add("pct_h", [f[f"o{h}"] for h in H])
    for nm in ("lo", "med", "hi"):
        add(f"type_{nm}", [f[f"t{nm}{h}"] for h in H])
        add(f"otype_{nm}", [f[f"o{nm}{h}"] for h in H])
    M = BF[(BF.DeviceId == r.DeviceId) & (BF.window == r.window) & (BF.phase == f.phase) & (BF.detector != r.detector)
           & BF.ok & (BF.tot >= 200)].sort_values("detector")
    for m in M.itertuples():
        lab = labs.get(int(m.detector), "det %d" % int(m.detector))
        n = np.array([getattr(m, f"n{h}") for h in H], float)
        add("mate_count:" + lab, n / max(n.sum(), 1) * f.tot)
        add("mate_pct:" + lab, [getattr(m, f"o{h}") for h in H])
    pr = None
    if f.ev_h0 >= 0:
        pr = float(np.nanmean([f[f"o{h}"] for h in range(int(f.ev_h0), int(f.ev_h1) + 1)]))
    rec.update(tot=float(f.tot), group=str(f.group), ev_from=str(f.ev_from), ev_h=int(f.ev_h), ev_dir=int(f.ev_dir),
               ev_h0=int(f.ev_h0), ev_h1=int(f.ev_h1), ev_obs=float(f.ev_obs), ev_exp=float(f.ev_exp),
               band_flag=bool(f.flag), cong=bool(f.cong) or bool(f.mrun_cong), mate_excused=bool(f.mate_excused),
               run_h_type=int(f.run_h), run_h_mates=int(f.mrun_h), sig_wide=bool(f.sig_wide),
               sig_odd_n=int(f.sig_odd_n), sig_n=int(f.sig_n), pct_run=pr,
               run_dir_type=int(f.run_dir), run_h0_type=int(f.run_h0), run_h1_type=int(f.run_h1),
               run_obs_type=float(f.run_obs), run_exp_type=float(f.run_exp))
    # night check numbers, for the group that was judged (model unsure: the plausible type with the lowest z)
    N = np.array([[f[f"n{h}"] for h in H]], float)
    O = np.array([[f[f"o{h}"] / 100 for h in H]], float)
    alts = [a for a in str(r.alt_c).split(",") if a and a != "nan"] or [r.fn]
    best = None
    for fn in sorted(set(alts) | {r.fn}):
        o = T6.score(N, O, np.array([fn]), np.array([float(r.lanes) if r.lanes == r.lanes else 1.0]),
                     np.array([r.DeviceId]), np.array([np.nan]), REF6)
        if np.isfinite(o["z"][0]) and (best is None or o["z"][0] < best[1]["z"][0]):
            best = (fn, o)
    if best is None:
        return
    fn, o = best
    g = REF6[0][o["group"][0]]
    X, nn = T6.features(N, O)
    exp_r = float(o["exp_r"][0])
    b4 = float(T6._busiest4(N, np.ones(24, bool))[0])
    rec.update(night_fn=fn, night_ratio=float(X[0, 0] ** 2), exp_ratio=exp_r ** 2,
               exp_night=max(exp_r ** 2 * (b4 + .5) - .5, 0), b4=b4, night_mean=float(N[0, 1:5].mean()),
               z_cnt=float(o["z_cnt"][0]), z_occ=float(o["z_occ"][0]) if np.isfinite(o["z_occ"][0]) else None,
               z=float(r.bd_tod_z_c) if np.isfinite(r.bd_tod_z_c) else float(o["z"][0]), level=str(r.tod_level))
    k = int(np.argmax(np.convolve(np.r_[N[0], N[0, :3]], np.ones(4) / 4, "valid")[:24]))
    rec.update(b4_t0=str(d0 + pd.Timedelta(hours=k)), b4_t1=str(d0 + pd.Timedelta(hours=k + 4)))
    if bool(g["occ"]):
        exp_o = float(np.array([1.0, X[0, 2]]) @ g["bo"])
        rec.update(night_on=float(100 * O[0, 1:5].mean()), exp_on=float(100 * np.sin(np.clip(exp_o, 0, np.pi / 2)) ** 2))
    rec["occ_drove"] = bool(rec.get("z_occ") is not None and rec["z_occ"] > rec["z_cnt"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(W / "review_v4c.parquet")
    dev_of = R.drop_duplicates("signal").set_index("signal").DeviceId.to_dict()
    S117 = pd.read_parquet(W / "fast118c.parquet").merge(
        pd.read_parquet(DCW / "s117" / "stats117.parquet", columns=KEY + ["ln"]), on=KEY, how="left")
    BF = pd.read_parquet(W / "band_final.parquet")
    REF6 = T6.load_reference(DCW / "s116" / "tod116_ref.npz")
    EPS = pd.read_csv(W / "stuck_eps_v4c.csv", parse_dates=["t0", "t1"])
    Rall = pd.read_parquet(W / "resolved_v4c.parquet", columns=KEY + ["cmode"])
    _, BC = SC.occ_hi_c(Rall)
    rows, day, evs, marks = [], [], [], {}
    seen_day = set()
    bctx = None
    for i, (sig, d, w, cat) in enumerate(PICKS, start=1):
        dev = dev_of[sig]
        r = R[(R.DeviceId == dev) & (R.window == w) & (R.detector == d)].iloc[0]
        X = R[(R.DeviceId == dev) & (R.window == w)]
        ph = r.phase
        mates = X[(X.phase == ph) & (X.detector != d)].sort_values("detector") if ph == ph else X.iloc[:0]
        labs = {int(d): det_label(d, ph, r.fn)}
        labs.update({int(m.detector): det_label(m.detector, m.phase, m.fn) for m in mates.itertuples()})
        cats = [c for c in str(r.cats).split(",") if c]
        t0, t1 = wwin(w)
        d0, d1 = day_of(w)
        ev = events(dev)
        z0, z1 = ev.Timestamp.min().floor("15min"), ev.Timestamp.max().ceil("15min")
        # ---- day data (all saved data) for the detector and its mates
        dets = [int(d)] + [int(m) for m in mates.detector]
        B = hc.events_to_bins(ev, z0, z1, detectors=dets, bin_s=900)
        ix = {int(x): j for j, x in enumerate(B["dets"])}
        tt = z0 + pd.to_timedelta(np.arange(B["n_on"].shape[1]) * 900 + 450, unit="s")
        for dd in dets:
            if (i, dd) in seen_day or dd not in ix:
                continue
            j = ix[dd]
            n = B["n_on"][j].astype(float)
            o = B["occ"][j].astype(float) / 900 * 100
            n[~B["cov"]] = np.nan
            o[~B["cov"]] = np.nan
            day.append(pd.DataFrame({"row": i, "detector": dd, "label": labs[dd], "self": dd == d, "t": tt,
                                     "count": n, "pct_on": o, "chat": B["n_chat"][j].astype(float)}))
            seen_day.add((i, dd))
        rec = dict(row=i, signal=sig, DeviceId=dev, window=w, detector=int(d), category=cat, label=labs[int(d)],
                   fn=r.fn, span=r.span, type=r.type, status=r.st8, cats=cats, flagged=cat in cats,
                   sample_t0=str(t0), sample_t1=str(t1), day_t0=str(d0), day_t1=str(d1), data_t0=str(z0),
                   data_t1=str(z1), sample_h=float(S_WIN[w][1]), mates=[labs[k] for k in dets[1:]])
        mk = {}
        # ---- category evidence
        if cat == "stuck":
            e = EPS[(EPS.DeviceId == dev) & (EPS.window == w) & (EPS.detector == d)].sort_values("t0")
            mk["spans"] = [dict(t0=str(a), t1=str(b), dur_s=float(s), label=l) for a, b, s, l in
                           zip(e.t0, e.t1, e.dur_s, e.label)]
            rec.update(n_ep=int(np.nan_to_num(r.st_n_ep)), tot_s=float(np.nan_to_num(r.st_tot_s)),
                       lim_s=float(r.stuck_lim8))
        elif cat == "dropout":
            bs = 300.0                                  # package bins (health_core.BIN_S): drop_b0 / b1 in 5-min bins
            a = t0 + pd.Timedelta(seconds=bs * float(r.drop_b0))
            b = t0 + pd.Timedelta(seconds=bs * float(r.drop_b1))
            mk["spans"] = [dict(t0=str(a), t1=str(b), dur_s=(b - a).total_seconds(), label="silent")]
            rec.update(expected=float(r.drop_lam), silent_t0=str(a), silent_t1=str(b),
                       expected_old=float(r.drop_lam_pkg), dropped=str(r.dropped) if str(r.dropped) not in ("nan", "None") else "")
        elif cat == "level":
            s = pd.read_parquet(DCW / "s118" / "drop_series118.parquet",
                                filters=[("DeviceId", "==", dev), ("window", "==", w), ("detector", "==", int(d))])
            s = s.sort_values("b")
            t = t0 + pd.to_timedelta(s.b.astype("int64") * 900 + 450, unit="s")
            evs.append(pd.DataFrame({"row": i, "series": "own", "t": t, "value": s.x.to_numpy(float)}))
            evs.append(pd.DataFrame({"row": i, "series": "expected", "t": t, "value": s.expected.to_numpy(float)}))
            rec.update(drop_at=str(r.drop_at), own_before=float(r.own_before), own_after=float(r.own_after),
                       exp_after=float(r.exp_after))
        elif cat == "night_drop" and bool(r.silent_night):
            bs = 300.0
            a = t0 + pd.Timedelta(seconds=bs * float(r.drop_b0))
            b = t0 + pd.Timedelta(seconds=bs * float(r.drop_b1))
            mk["spans"] = [dict(t0=str(a), t1=str(b), dur_s=(b - a).total_seconds(), label="silent")]
            mk["night"] = [[str(t0), str(t0 + pd.Timedelta(hours=5))], [str(t0 + pd.Timedelta(hours=21)), str(t1)]]
            rec.update(silent_night=True, expected=float(r.drop_lam), silent_t0=str(a), silent_t1=str(b))
        elif cat == "night_drop":
            Bw = hc.events_to_bins(ev, t0, t1, bin_s=900)
            n = Bw["n_on"].astype(float)
            n[:, ~Bw["cov"]] = 0
            jx = {int(x): j for j, x in enumerate(Bw["dets"])}
            hr = Bw["hour"]
            ref = str(r.night_ref)
            if ref.startswith("d") and "tracks" in ref:
                rd = [int(ref[1:].split(",")[0])]
                rlab = f"det {rd[0]}"
            elif "phase" in ref:
                rd = [int(m) for m in mates.detector]
                rlab = "its phase mates"
            else:
                rd = [int(x) for x in Bw["dets"] if int(x) != d]
                rlab = "the rest of the signal"
            Rr = n[[jx[k] for k in rd if k in jx]].sum(0)
            own = n[jx[int(d)]]
            dy = Bw["cov"] & (hr >= hc.DAY[0]) & (hr < hc.DAY[1])
            nt = Bw["cov"] & ((hr >= hc.NIGHT_H[0]) | (hr < hc.NIGHT_H[1]))
            sh = own[dy].sum() / max(Rr[dy].sum(), 1)
            tt2 = t0 + pd.to_timedelta(np.arange(len(own)) * 900 + 450, unit="s")
            evs.append(pd.DataFrame({"row": i, "series": "expected", "t": tt2, "value": np.where(nt, sh * Rr, np.nan)}))
            rec.update(night_n=float(r.night_n), night_exp=float(r.night_exp), ref=rlab, night_n_chk=float(own[nt].sum()),
                       night_exp_chk=float(sh * Rr[nt].sum()))
            mk["night"] = [[str(t0), str(t0 + pd.Timedelta(hours=5))], [str(t0 + pd.Timedelta(hours=21)), str(t1)]]
        elif cat == "choppy":
            b = pd.read_parquet(DCW / "health110" / "b15.parquet",
                                filters=[("DeviceId", "==", dev), ("window", "==", w), ("detector", "==", int(d))])
            b = b.sort_values("b")
            e, hw, off, sc = ER.off_bins(b.x.to_numpy(float), b.R.to_numpy(float), b.ok.to_numpy(bool))
            t = t0 + pd.to_timedelta(b.b.astype("int64") * 900 + 450, unit="s")
            x = b.x.to_numpy(float)
            for nm, v in (("own", x), ("expected", np.where(sc, e, np.nan)), ("lo", np.where(sc, np.maximum(e - hw, 0), np.nan)),
                          ("hi", np.where(sc, e + hw, np.nan)), ("off", np.where(off, x, np.nan))):
                evs.append(pd.DataFrame({"row": i, "series": nm, "t": t, "value": v}))
            rec.update(n_off=int(off.sum()), n_sc=int(sc.sum()), exc=float(r.exc_15), lim_exc=float(r.lim_exc),
                       ref=str(r.ref110), ref_dets=str(r.ref110_dets) if "ref110_dets" in r else "")
        elif cat in ("rapid", "volume"):
            b = pd.read_parquet(W / "bins117.parquet",
                                filters=[("DeviceId", "==", dev), ("window", "==", w), ("detector", "==", int(d))])
            b = b.sort_values("b").reset_index(drop=True)
            s7 = S117[(S117.DeviceId == dev) & (S117.window == w) & (S117.detector == d)].iloc[0]
            t = t0 + pd.to_timedelta(b.b.astype("int64") * 300 + 150, unit="s")
            if cat == "rapid":
                # vehicle starts minus re-triggers (OFF -> ON < 0.3 s); fast = ON -> ON < 1 s that is not a re-trigger
                nGY = np.maximum(b.nG + b.nY - b.cGY, 0).to_numpy(float)
                nR = np.maximum(b.nR - b.cR, 0).to_numpy(float)
                nU = np.maximum(b.nU - b.cU, 0).to_numpy(float)
                sGY, sR = b.sGY.to_numpy(float), b.sR.to_numpy(float)
                sU = np.maximum(300 - b.sGY - b.sR - b.sRg, 0).to_numpy(float)

                def rate(nn, ss):
                    return np.where(ss >= 30, nn / np.maximum(ss, 1e-9), np.nan)

                def cmed(v):
                    out = np.full(len(v), np.nan)
                    for k in range(len(v)):
                        seg = v[max(0, k - 6):k + 7]
                        seg = seg[np.isfinite(seg)]
                        if len(seg):
                            out[k] = np.median(seg)
                    return out

                def ex(nn, ss, cap=None):
                    r_ = np.where(ss > 0, nn / np.maximum(ss, 1e-9), 0)
                    m_ = cmed(rate(nn, ss))
                    rr = np.minimum(r_, np.where(np.isfinite(m_), m_, r_))
                    if cap is not None:
                        rr = np.minimum(rr, cap)
                    return np.where(ss > 0, nn * (1 - np.exp(-rr)), 0.0)
                eg, er = ex(nGY, sGY), ex(nR, sR)
                eu = np.where(sU > 0, ex(nU, sU, 50), nU)
                ee = eg + er + eu
                fo = (b.xGY + b.xR + b.xU).to_numpy(float)
                zb = (fo - ee) / np.sqrt(ee + 1)
                burst = (zb >= 4) & (fo >= 5)
                for nm, v in (("fast_green", b.xGY.to_numpy(float)), ("fast_red", (b.xR + b.xU).to_numpy(float)),
                              ("expected", ee), ("burst", np.where(burst, fo, np.nan))):
                    evs.append(pd.DataFrame({"row": i, "series": nm, "t": t, "value": v}))
                rec.update(fo_all=float(s7.fo_c), fem_all=float(s7.fem_c), fo_chk=float(fo.sum()), fem_chk=float(ee.sum()),
                           n_spk=int(s7.n_spk_c), n_spk_chk=int(burst.sum()), lim_n_spk=float(s7.lim_spk_c),
                           zf=float(s7.zf_c), lim_zf=float(s7.lim_zf_c), x_zf=float(np.nan_to_num(s7.x_zf_c)),
                           x_spk=float(np.nan_to_num(s7.x_spk_c)), fast_x=float(np.nan_to_num(s7.fast_x_c)),
                           n_retrig=float((b.cGY + b.cR + b.cU).sum()), fo_old=float(s7.fo_all))
            else:
                ln = float(s7.ln)
                adv = r.fn == "Advance"
                if adv:
                    q = (b.n / 300 * 3600 / ln).to_numpy(float)
                else:
                    sGY = b.sGY.to_numpy(float)
                    q = np.where(sGY >= 60, (b.nG + b.nY).to_numpy(float) / np.maximum(sGY, 1e-9) * 3600 / ln, np.nan)
                evs.append(pd.DataFrame({"row": i, "series": "flow", "t": t, "value": q}))
                k = int(np.nanargmax(q))
                rec.update(q5=float(s7.q5_all if adv else s7.q5_gy), q5_chk=float(q[k]), lim_vol=float(s7.lim_vol_c),
                           lanes=ln, busiest_t=str(t[k]), green_only=not adv, vol_x=float(np.nan_to_num(s7.vol_x_c)))
        elif cat == "chatter":
            Bw = hc.events_to_bins(ev, t0, t1, detectors=[int(d)], bin_s=900)
            j = {int(x): q for q, x in enumerate(Bw["dets"])}[int(d)]
            n, c = Bw["n_on"][j].astype(float), Bw["n_chat"][j].astype(float)
            t = t0 + pd.to_timedelta(np.arange(len(n)) * 900 + 450, unit="s")
            evs.append(pd.DataFrame({"row": i, "series": "count", "t": t, "value": n}))
            evs.append(pd.DataFrame({"row": i, "series": "chat", "t": t, "value": c}))
            rec.update(chat_frac=float(r.chat_frac), lim_chat=float(r.lim_chat_frac), chat_chk=float(c.sum() / max(n.sum(), 1)),
                       n_chat=float(c.sum()), n_on=float(n.sum()))
        elif cat == "occspk":
            if bctx is None:
                bctx = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet")
            Bn = bctx[(bctx.DeviceId == dev) & (bctx.window == w)].copy()
            res, spk = R4.n3_stats(Bn)
            q = res[res.detector == d].iloc[0]
            o = Bn[Bn.detector == d].sort_values("b")
            e = o.n * q.dbar / o.bs
            t = t0 + pd.to_timedelta(o.b.astype("int64") * o.bs + o.bs / 2, unit="s")
            sp = spk[spk.detector == d]
            on_sp = o.b.isin(sp.b).to_numpy()
            evs.append(pd.DataFrame({"row": i, "series": "pct_on", "t": t, "value": 100 * o.occ.to_numpy(float)}))
            evs.append(pd.DataFrame({"row": i, "series": "explained", "t": t, "value": 100 * np.minimum(e, 1).to_numpy(float)}))
            evs.append(pd.DataFrame({"row": i, "series": "spike", "t": t, "value": np.where(on_sp, 100 * o.occ, np.nan)}))
            rec.update(n3_exc=float(r.n3_exc), n3_exc_chk=float(q.n3_exc), n3_n=int(q.n3_n), lim_n3=float(r.lim_n3_exc),
                       dbar=float(q.dbar))
        elif cat in ("prof", "shape24"):
            tod(i, r, BF, REF6, labs, d0, evs, rec, mk)
        elif cat == "count_on":
            o = BC[(BC.DeviceId == dev) & (BC.window == w) & (BC.detector == d)].sort_values("b")
            t = t0 + pd.to_timedelta(o.b.astype("int64") * o.bs + o.bs / 2, unit="s")
            evs.append(pd.DataFrame({"row": i, "series": "pct_on", "t": t, "value": 100 * o.occ.to_numpy(float)}))
            evs.append(pd.DataFrame({"row": i, "series": "limit", "t": t, "value": 100 * o.lim_c.to_numpy(float)}))
            evs.append(pd.DataFrame({"row": i, "series": "hi", "t": t, "value": np.where(o.hi_c, 100 * o.occ, np.nan)}))
            evs.append(pd.DataFrame({"row": i, "series": "count", "t": t, "value": o.n.to_numpy(float)}))
            rec.update(hi_min=float(o.hi_c.sum() * o.bs.iloc[0] / 60), hi_min_res=float(np.nan_to_num(r.hi_min_c)),
                       c1=bool(r.c1_pre), n_rep=float(r.n_rep), n_on_pkg=float(r.n_on),
                       hi_occ_max=float(100 * o.occ[o.hi_c].max()) if o.hi_c.any() else 0.0,
                       hi_cnt=float(o.n[o.hi_c].mean()) if o.hi_c.any() else 0.0,
                       raw_hi_min=float((o.occ > o.lim_c).sum() * o.bs.iloc[0] / 60))
        elif cat == "occ_hi":
            if bctx is None:
                bctx = pd.read_parquet(DCW / "health104" / "bins_ctx.parquet")
            if "lim_hi" not in globals():
                Bn = bctx[["DeviceId", "window", "detector", "fn", "b", "occ", "ref", "cong", "bs", "ph", "status"]].merge(
                    pd.read_parquet(DCW / "health108" / "base.parquet", columns=KEY + ["cmode"]), on=KEY)
                Bn = Bn[Bn.cmode.ne("unknown")]
                Bn["rmax"] = Bn.groupby(KEY).ref.transform("max")
                Bn["tl"] = np.clip(np.floor(5 * Bn.ref / Bn.rmax.where(Bn.rmax > 0)), 0, 4)
                globals()["lim_hi"] = Bn[Bn.status.eq("ok")].groupby(["fn", "cmode", "bs", "tl"]).occ.quantile(.995).rename(
                    "lim").reset_index()
                globals()["Bhi"] = Bn
            Bn = globals()["Bhi"]
            Bw = Bn[(Bn.DeviceId == dev) & (Bn.window == w)].merge(globals()["lim_hi"], on=["fn", "cmode", "bs", "tl"],
                                                                     how="left")
            g = Bw.groupby(["ph", "b"]).occ.agg(["sum", "count"]).rename(columns={"sum": "ps", "count": "pc"}).reset_index()
            Bw = Bw.merge(g, on=["ph", "b"], how="left")
            Bw["pocc"] = ((Bw.ps - Bw.occ.fillna(0)) / (Bw.pc - 1).where(Bw.pc > 1)).where(Bw.ph >= 0)
            Bw["hi"] = (Bw.occ > Bw.lim) & (Bw.cong < 1.5) & ~(Bw.pocc >= 0.40)
            o = Bw[Bw.detector == d].sort_values("b")
            t = t0 + pd.to_timedelta(o.b.astype("int64") * o.bs + o.bs / 2, unit="s")
            evs.append(pd.DataFrame({"row": i, "series": "pct_on", "t": t, "value": 100 * o.occ.to_numpy(float)}))
            evs.append(pd.DataFrame({"row": i, "series": "limit", "t": t, "value": 100 * o.lim.to_numpy(float)}))
            evs.append(pd.DataFrame({"row": i, "series": "hi", "t": t, "value": np.where(o.hi, 100 * o.occ, np.nan)}))
            rec.update(hi_min=float(o.hi.sum() * o.bs.iloc[0] / 60), occ_mean=float(100 * o.occ.mean()))
        elif cat in ("C1", "C2"):
            tt_, ee_ = det_events(ev, int(d))
            m = (tt_ >= np.datetime64(t0)) & (tt_ < np.datetime64(t1))
            st_, du_, rl_ = cont_ons(tt_[m], ee_[m], t1)
            if cat == "C1":
                edges = pd.date_range(t0, t1, freq="15min")
                a = np.histogram(st_.astype("datetime64[ns]").astype("int64"), edges.asi8)[0]
                c = np.histogram(rl_.astype("datetime64[ns]").astype("int64"), edges.asi8)[0]
                t = edges[:-1] + pd.Timedelta(minutes=7.5)
                evs.append(pd.DataFrame({"row": i, "series": "starts", "t": t, "value": a.astype(float)}))
                evs.append(pd.DataFrame({"row": i, "series": "relogged", "t": t, "value": c.astype(float)}))
                rec.update(n_start=int(len(st_)), n_relog=int(len(rl_)), n_rep=float(r.n_rep), n_on_pkg=float(r.n_on))
            else:
                k = du_ >= 5
                mk["points"] = [dict(t=str(pd.Timestamp(a)), dur_s=float(b)) for a, b in zip(st_[k], du_[k])]
                rec.update(n_ge5=int(k.sum()), n_ge60=int((du_ >= 60).sum()), long_max=float(du_.max()) if len(du_) else 0.0,
                           n_ge5_pkg=float(r.n_ge5), n_ge60_pkg=float(r.n_ge60), med_dur=float(np.median(du_)) if len(du_) else 0.0)
        rows.append(rec)
        marks[i] = mk
        print(i, sig, d, w, cat, "flagged" if rec["flagged"] else "LOOK-ALIKE", r.st8, flush=True)
    pd.concat(day, ignore_index=True).to_parquet(OUT / "day15.parquet")
    pd.concat(evs, ignore_index=True).to_parquet(OUT / "ev.parquet")
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    (OUT / "marks.json").write_text(json.dumps(marks, indent=1, default=str))


if __name__ == "__main__":
    main()
