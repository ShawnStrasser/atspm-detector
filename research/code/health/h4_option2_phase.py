"""Note 46 PART 2, phase side: option 2 (remove health-v4 bad periods, classify again) for the PHASE output.

Phase fold models were never saved, so the out-of-sample test uses the 71 released NEWTEST signals (Sept 2026):
the final_v3 candidate package's phase pipeline (final_v1 trees = ranker bag + joint decoder, stage-13 GRU blended
at every length, K = 4 pieces above 2 h) was fitted on the 701-signal DEV + NEWTRAIN pool and never saw them.
Every stg frame window of a released signal that overlaps a listed bad period (health_v4.parquet) is scored raw
(pass1) and with the periods removed (pass2a: each detector's own actuations in its own periods; pass2b: the
union of periods cut for every channel).  Truth = official timing (call phase; the timing's switch phase and
additional call phases also count as right, note 14 A1).  Locked never read.

    python h4_option2_phase.py -> %DC_WORK%/health4/option2_phase.parquet + printed table
"""
from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h4_option2 as O  # noqa: E402
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health4"
PER = None


def init():
    global PER
    _, bp = O.jobs(H.HB / "health_v4.parquet")
    PER = {k: g for k, g in bp.groupby("DeviceId")}


def one(job):
    import h4_pkg  # noqa: F401  (package on sys.path, logging off)
    P = h4_pkg.P
    dev, wins = job
    ev = ds.dataset(O.EVR["stg"] / f"DeviceId={dev}").to_table(filter=ds.field("EventId").isin(O.ALLOWED)).to_pandas()
    ev["DeviceId"] = dev
    W = O.windows()
    bp = PER[dev]
    bp = bp[bp.period == "stg"]
    out = []
    for win in wins:
        w0, w1 = W[("stg", win)]
        e = ev[(ev.Timestamp >= w0) & (ev.Timestamp < w1)]
        b = bp[(bp.start < w1) & (bp.end > w0)]
        ts, par, isdet = e.Timestamp.to_numpy(), e.Parameter.to_numpy(), e.EventId.isin([81, 82]).to_numpy()
        drop, cut = np.zeros(len(e), bool), np.zeros(len(e), bool)
        for r in b.itertuples():
            inn = (ts >= np.datetime64(r.start)) & (ts < np.datetime64(r.end))
            drop |= isdet & (par == r.detector) & inn
            cut |= inn
        res = None
        for name, ee in (("p1", e), ("p2a", e[~drop]), ("p2b", e[~cut])):
            try:
                x = P.predict(ee, str(w0), str(w1), threads=2, memory="3GB")
                x = x[["Detector", "phase_guess", "phase_guess_prob"]].rename(
                    columns={"phase_guess": f"ph_{name}", "phase_guess_prob": f"pp_{name}"})
            except Exception as exc:  # noqa: BLE001
                print(dev, win, name, type(exc).__name__, exc, flush=True)
                x = None
            if x is not None:
                res = x if res is None else res.merge(x, on="Detector", how="outer")
        if res is not None:
            res["own_period"] = res.Detector.isin(set(b.detector))
            out.append(res.assign(DeviceId=dev, win=win))
    return pd.concat(out, ignore_index=True) if out else None


def main():
    rel = set(pd.read_csv(H.DCW / "official" / "newtest_released.csv").DeviceId.str.lower())
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not rel & locked
    J, _ = O.jobs(H.HB / "health_v4.parquet")
    J = J[(J.period == "stg") & J.DeviceId.isin(rel)]
    todo = [(d, list(g.win)) for d, g in J.groupby("DeviceId")]
    print("released signals with a period:", len(todo), "signal-windows", len(J), flush=True)
    t = time.time()
    with Pool(6, initializer=init) as p:
        res = [r for r in p.imap_unordered(one, todo) if r is not None]
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(OUT / "option2_phase.parquet", index=False)
    print(df.shape, f"{time.time() - t:.0f}s")
    evaluate(df)


def evaluate(df=None):
    df = pd.read_parquet(OUT / "option2_phase.parquet") if df is None else df
    o = pd.read_parquet(H.DCW / "official" / "labels_official.parquet",
                        columns=["DeviceId", "Detector", "target_type", "target_num", "switch_phase",
                                 "additional_call_phases"])
    o = o[o.target_type == "phase"]
    o["DeviceId"] = o.DeviceId.str.lower()
    d = df.merge(o, on=["DeviceId", "Detector"], how="inner")

    def right(col):
        ok = d[col].to_numpy(float) == d.target_num.to_numpy(float)
        sw = d.switch_phase.fillna(0).to_numpy(float)
        ok |= (sw > 0) & (d[col].to_numpy(float) == sw)
        add = d.additional_call_phases.fillna("").astype(str)
        ok |= np.array([str(int(v)) in a.replace(" ", "").split(",") if np.isfinite(v) else False
                        for v, a in zip(d[col].to_numpy(float), add)])
        return ok
    d = d[d.ph_p1.notna()]
    d["fam"] = d.win.map(lambda w: "full" if w.startswith("full") else w.split("_")[0])
    # denominators: every labelled released-signal row of the family in the phase OOF frame (trees_oof_bywindow)
    t = pd.read_parquet(H.DCW / "final_v3_work" / "phase_v3" / "work" / "trees_oof_bywindow.parquet",
                        columns=["DeviceId", "Detector", "win"])
    t = t[t.DeviceId.str.endswith("@stg")].drop_duplicates()
    t["DeviceId"] = t.DeviceId.str.replace("@stg", "").str.lower()
    rel = set(pd.read_csv(H.DCW / "official" / "newtest_released.csv").DeviceId.str.lower())
    t = t[t.DeviceId.isin(rel)]
    t["fam"] = t.win.map(lambda w: "full" if w.startswith("full") else w.split("_")[0])
    N = t.groupby("fam").size()
    r1 = right("ph_p1")
    for p2 in ("ph_p2a", "ph_p2b"):
        r2 = right(p2) & d[p2].notna().to_numpy()
        r2 = np.where(d[p2].isna(), r1, r2)
        dd = r2.astype(int) - r1.astype(int)
        print(p2, "overall rows re-run", len(d), "fixed", int((dd > 0).sum()), "broken", int((dd < 0).sum()))
        for fam in ("m30", "h6", "full", "h24", "h3", "h1", "m10", "m5"):
            m = (d.fam == fam).to_numpy()
            if m.any():
                print(f"   {fam}: re-run rows {int(m.sum())} (of {int(N.get(fam, 0))} released rows), pass1 acc "
                      f"{r1[m].mean():.4f}, delta {dd[m].sum():+d} rows = {100 * dd[m].sum() / max(N.get(fam, 1), 1):+.3f} pt")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        main()
