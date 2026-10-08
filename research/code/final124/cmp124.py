"""Note 124: prove the production health v4 (detector_classifier.health_v4) = the research scorer v4d.

Signals: the 24 signals behind the 40 rows of review/health_review_v4.xlsx (s118c/plot/rows.json), every detector,
windows h3_a, h3_b, h24_a, h24_b (Sun 27 / Mon 28 Sep 2026; w40 events, locked_v2 absent - asserted).  The package is
called with the research's classifier inputs (health4/inputs.parquet, stg OOF) - the same inputs the research used.

Modes
  research : stage 1 = the research's health_core (= v4f, research/code/health/health_core.py), 24-h references =
             the out-of-fold fits the research used for that signal (s124/refs_oof).  Must equal resolved_v4d.
  package  : stage 1 = the package's own health_core (v7: 115d / 115dx fixes), shipped all-data 24-h references.
             Differences are listed with their cause.
  pkg_oof  : the package's stage 1 with the research's out-of-fold 24-h references (isolates the stage-1 fixes).

    python cmp124.py [research|package|both] [--sig N]   -> %DC_WORK%/s124/cmp_<mode>.parquet + summary print
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
PKG = Path(os.environ.get("DC_PKG") or (DCW / "final_v7_prod" / "src"))
sys.path.insert(0, str(PKG))
from detector_classifier import health_core as hc  # noqa: E402
from detector_classifier import health_v4 as H4  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hc_v4f", CODE / "health" / "health_core.py")
hc4f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc4f)

OUT = DCW / "s124"
EVD = DCW / "health4" / "w40_events"
WIN = {"h3_a": ("2026-09-27 12:00", 3), "h3_b": ("2026-09-28 06:00", 3),
       "h24_a": ("2026-09-27 00:00", 24), "h24_b": ("2026-09-28 00:00", 24),
       "m30_a": ("2026-09-27 08:00", .5), "m30_b": ("2026-09-27 12:00", .5), "m30_c": ("2026-09-27 17:00", .5),
       "m30_d": ("2026-09-28 07:30", .5)}
C7 = H4.C7
KEY = ["DeviceId", "window", "detector"]


def signals():
    rows = json.loads((DCW / "s118c" / "plot" / "rows.json").read_text())
    return sorted({r["DeviceId"] for r in rows})


def load_events(dev):
    p = EVD / f"DeviceId={dev}"
    if not p.is_dir():
        p = next(q for q in EVD.iterdir() if q.name.lower() == f"deviceid={dev}")
    e = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    return e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()


def prep(e, t0, t1):
    w = e[(e.Timestamp >= t0) & (e.Timestamp < t1)]
    w = w[~(w.EventId.isin((81, 82)) & (w.Parameter > 64))]
    t = (w.Timestamp - t0).dt.total_seconds().to_numpy()
    return hc.prep_arrays(t, w.EventId.to_numpy(), w.Parameter.to_numpy())


def inputs():
    x = pd.read_parquet(DCW / "health4" / "inputs.parquet")
    x = x[(x.period == "stg") & x.wgroup.isin(["m30", "h3", "h24"])]
    x["DeviceId"] = x.DeviceId.str.lower()
    return {k: g for k, g in x.groupby(["wgroup", "DeviceId"])}


def refs_for(dev, mode, folds, base):
    if mode == "package":
        return base                    # pkg_oof: the package's stage 1 with the research's out-of-fold 24-h refs
    f = folds[dev]
    tod_f = json.loads((OUT / "refs_oof" / f"tod_fold{f}.json").read_text())
    band_f = json.loads((OUT / "refs_oof" / f"band_fold{f}.json").read_text())
    return H4.Refs(base.d, tod_own=tod_f, tod_alt=base.d["tod"], band_own=band_f, band_alt=band_f)


def run(mode, sigs, wins):
    IN = inputs()
    folds = json.loads((OUT / "refs_oof" / "folds.json").read_text())
    base = H4._load_refs(str(OUT / "refs" / "health_v4_refs.json"))
    locked = set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not (set(sigs) & locked), "locked signal"
    res, sig_notes, times = [], [], []
    for dev in sigs:
        e = load_events(dev)
        R = refs_for(dev, mode, folds, base)
        for w in wins:
            s, h = WIN[w]
            t0 = pd.Timestamp(s)
            t1 = t0 + pd.Timedelta(hours=h)
            g = IN.get((w.split("_")[0], dev))
            if g is None:
                continue
            ph = dict(zip(g.detector.astype(int), g.pred_phase.astype(float)))
            fl = dict(zip(g.detector.astype(int), g.pred_function))
            fp = {int(d): {c: float(v) for c, v in zip(C7, row)} for d, row in
                  zip(g.detector, g[[f"p_{c}" for c in C7]].to_numpy())}
            ln = dict(zip(g.detector.astype(int), g.n_lanes_spanned.astype(float)))
            pc = dict(zip(g.detector.astype(int), g.top_prob.astype(float)))
            P = prep(e, t0, t1)
            if not np.isin(P.eid, (81, 82)).any():
                continue
            st1 = None
            if mode == "research":
                def st1(P_, a, b, d, phx, fpx, lnx, pcx, _e=e):
                    return hc4f.health(_e, a, b, None, phx, fpx, lnx, pcx)
            tt = time.perf_counter()
            out, note, Rx, Ex = H4.assess(P, t0, t1, ph, fl, fp, ln, pc, refs=R, stage1=st1, keep_all=True)
            times.append(time.perf_counter() - tt)
            Rx = Rx.assign(DeviceId=dev, window=w)
            Rx = Rx.merge(out, on="detector", how="left")
            res.append(Rx)
            sig_notes.append(dict(DeviceId=dev, window=w, note=note))
            print(dev[:8], w, len(Rx), f"{times[-1]:.2f}s", flush=True)
    A = pd.concat(res, ignore_index=True)
    return A, pd.DataFrame(sig_notes), times


CMP_STR = ["st_v4d", "st8", "left8", "watch8", "cfg", "rules8", "dq8", "sev_rule", "ref110", "tod_level", "cmode",
           "band", "span", "alt_fns"]
CMP_NUM = ["score8", "s8_dropout", "s8_level", "s8_night_drop", "s8_choppy", "s8_chatter", "s8_rapid", "s8_volume",
           "s8_stuck", "s8_occspk", "s8_prof", "s8_shape24", "s8_count_on", "n3_n", "n3_exc", "elhi_cnt", "elhi_occ",
           "c_occ_like", "occ", "share_ref", "share_x", "stuck_x", "dur_p50", "rep_time_s", "max_on_s", "n_hmates",
           "lv_ratio110", "lv_llr110", "lv_b110", "exc_15", "n_off_15", "n_sc_15", "exc_sig", "n_on117", "q5_gy",
           "q5_all", "fo_c", "fem_c", "zf_c", "n_spk_c", "fo_all", "fem_all", "zf", "n_spk", "drop_lam", "hi_min_c",
           "lim_chat_frac", "lim_stuck_x", "lim_n3_exc", "lim_n_ep", "lim_exc", "st_n_ep", "st_tot_s", "bd_ev_h",
           "bd_ev_obs", "bd_ev_exp", "n_rep", "n_ge5", "n_ge60", "n_ev"]


def compare(A, tag):
    Rr = pd.read_parquet(DCW / "s121" / "resolved_v4d.parquet")
    Rr = Rr[Rr.DeviceId.isin(A.DeviceId.unique()) & Rr.window.isin(A.window.unique())]
    for c in ("st_v4d", "occ_hi8"):
        pass
    M = A.merge(Rr, on=KEY, how="outer", suffixes=("", "_r"), indicator=True)
    print(tag, "rows: package", len(A), "research", len(Rr), M._merge.value_counts().to_dict())
    M = M[M._merge == "both"].copy()
    rep = []
    for c in CMP_STR:
        if c + "_r" not in M:
            continue
        a = M[c].fillna("").astype(str).replace({"nan": "", "None": ""})
        b = M[c + "_r"].fillna("").astype(str).replace({"nan": "", "None": ""})
        bad = a != b
        rep.append(dict(col=c, n=len(M), differ=int(bad.sum())))
        M[f"d_{c}"] = bad
    for c in CMP_NUM:
        if c + "_r" not in M:
            continue
        a = pd.to_numeric(M[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(M[c + "_r"], errors="coerce").to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        nanmis = np.isfinite(a) != np.isfinite(b)
        rel = np.where(both, np.abs(a - b) / np.maximum(np.abs(b), 1e-9), 0)
        bad = nanmis | (both & (rel > 1e-6))
        rep.append(dict(col=c, n=len(M), differ=int(bad.sum()), nan_mismatch=int(nanmis.sum()),
                        max_rel=float(rel.max()) if len(rel) else 0.0))
        M[f"d_{c}"] = bad
    T = pd.DataFrame(rep)
    print(T.to_string(index=False))
    for w, g in M.groupby("window"):
        print(tag, w, "final status equal:", f"{(g.st_v4d.astype(str) == g.st_v4d_r.astype(str)).mean():.4f}",
              len(g))
    return M, T


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "research"
    sigs = signals()
    if "--sig" in sys.argv:
        sigs = sigs[:int(sys.argv[sys.argv.index("--sig") + 1])]
    wins = ["h3_a", "h3_b", "h24_a", "h24_b"]
    if "--win" in sys.argv:
        wins = sys.argv[sys.argv.index("--win") + 1].split(",")
    modes = ["research", "package", "pkg_oof"] if mode == "both" else [mode]
    for md in modes:
        A, N, times = run(md, sigs, wins)
        keep = [c for c in A.columns if A[c].dtype != object or c in KEY + CMP_STR + H4.OUT_COLS]
        A2 = A.copy()
        for c in A2.columns:
            if A2[c].dtype == object:
                A2[c] = A2[c].astype(str)
        A2.to_parquet(OUT / f"cmp_{md}_raw.parquet")
        N.to_csv(OUT / f"signal_notes_{md}.csv", index=False)
        M, T = compare(A, md)
        for c in M.columns:
            if M[c].dtype == object:
                M[c] = M[c].astype(str)
        M.to_parquet(OUT / f"cmp_{md}.parquet")
        T.to_csv(OUT / f"cmp_{md}_summary.csv", index=False)
        print(md, "assess time per window: mean", f"{np.mean(times):.2f}s", "max", f"{np.max(times):.2f}s")


if __name__ == "__main__":
    main()
