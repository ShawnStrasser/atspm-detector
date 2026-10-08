"""Note 43 step 3: is a detector's data good again after a stuck-on / silent episode that recovered?

Events (Sept 2026, 66-h window, training signals): stuck-on episodes (ON >= 15 min; an ON with no
OFF logged counts only if >= 30 ONs were expected in it from the rest of the signal) and silent
runs (note-38 dropout run, >= 30 expected, ended before the window end).  Each event with >= 3 h of
data on both sides: the detector's share of the rest of the signal in the 6 h before vs the 6 h
after; the same for healthy detectors at random split points (null).  Long run: detectors with a
recovered event in Dec 2024 vs without, and their state in Sept 2026.

    python h3_recovery.py -> prints; %DC_WORK%/health3/recovery_events.parquet
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health3"
USE_PHASE = "--signal" not in sys.argv
_pf = pd.read_parquet(OUT / "oof_pf.parquet", columns=["period", "DeviceId", "detector", "wgroup", "pred_phase"])
PFP = {k: dict(zip(g.detector, g.pred_phase)) for k, g in _pf[_pf.wgroup == "full"].groupby(["period", "DeviceId"])}
W6, W3 = 72, 36          # 6 h / 3 h in 5-min bins


REF = {}          # (DeviceId, detector) -> reference rows (same predicted phase, >= 2 live, no twins)


def share(B, i, a, b):
    x = B["n_on"][i, a:b].astype(float)
    ref = REF.get(i)
    S = B["n_on"][:, a:b].sum(0) - x if ref is None else B["n_on"][ref, a:b].sum(0)
    c = B["cov"][a:b]
    return (x[c].sum() + 0.5) / max(S[c].sum(), 1.0), c.sum()


def events_for(B, ep, st, dev):
    """one row per recovered event of each detector in B's window."""
    dets = list(B["dets"])
    nb = B["n_on"].shape[1]
    rows = []
    for r in ep.itertuples():
        if r.open_end or r.detector not in dets:
            continue
        i = dets.index(r.detector)
        a = int((r.t0 - B["start"]).total_seconds() // B["bin_s"])
        b = int(np.ceil((r.t1 - B["start"]).total_seconds() / B["bin_s"]))
        x, S = B["n_on"][i].astype(float), B["n_on"].sum(0) - B["n_on"][i]
        out = np.r_[0:max(a, 0), min(b, nb):nb]
        lam = x[out].sum() / max(S[out].sum(), 1) * S[max(a, 0):min(b, nb)].sum()
        if r.no_off and lam < 30:
            continue
        rows.append(dict(DeviceId=dev, detector=r.detector, kind="stuck", b0=a, b1=b, lam=lam, co=r.co_stuck))
    for r in st.itertuples():
        if r.drop_lam >= 30 and not r.drop_to_end and r.drop_b1 > r.drop_b0:
            rows.append(dict(DeviceId=dev, detector=r.detector, kind="silent", b0=int(r.drop_b0), b1=int(r.drop_b1),
                             lam=r.drop_lam, co=r.co_silent))
    return rows


def measure(B, rows):
    dets = list(B["dets"])
    nb = B["n_on"].shape[1]
    for r in rows:
        i = dets.index(r["detector"])
        a, b = r["b0"], r["b1"]
        r["ok"] = a >= W3 and nb - b >= W3
        if not r["ok"]:
            continue
        s0, c0 = share(B, i, max(a - W6, 0), a)
        s1, c1 = share(B, i, b, min(b + W6, nb))
        r["ok"] = c0 >= W3 and c1 >= W3
        r["ratio"] = s1 / s0
        # counts in the 6 h after vs expected from the share before
        x = B["n_on"][i, b:b + W6].astype(float)
        ref = REF.get(i)
        S = (B["n_on"][:, b:b + W6].sum(0) - x if ref is None else B["n_on"][ref, b:b + W6].sum(0))[B["cov"][b:b + W6]]
        r["n_after"], r["exp_after"] = x[B["cov"][b:b + W6]].sum(), s0 * S.sum()
    return rows


def period(per, ev_root, healthy):
    import os
    rng = np.random.default_rng(41)
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    st_all = pd.read_parquet(H.HB / "real_stats.parquet")
    st_all = st_all[(st_all.period == per) & (st_all.window == "full")]
    out, null = [], []
    for f in sorted((H.BINS / per).glob("*.npz")):
        dev = f.stem
        if dev in locked or not (ev_root / f"DeviceId={dev}").is_dir():
            continue
        B = H.load_B(per, dev)
        REF.clear()
        ph = PFP.get((per, dev))
        if ph and USE_PHASE:
            n, cv = B["n_on"].astype(float), B["cov"]
            live = n[:, cv].sum(1) > 0
            ref, kind, _ = hc._refs(B["dets"], live, ph, hc.twins(n, cv))
            REF.update({i: ref[i] for i in range(len(ref)) if kind[i] == "phase" and len(ref[i]) >= 2})
        ev = ds.dataset(ev_root / f"DeviceId={dev}").to_table(
            filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
        t1 = B["start"] + pd.Timedelta(seconds=B["n_on"].shape[1] * B["bin_s"])
        ep = hc.on_episodes(ev, B["start"], t1)
        st = st_all[st_all.DeviceId == dev]
        rows = measure(B, events_for(B, ep, st, dev))
        out += rows
        # null: healthy detectors with no event, a random split with a 2-h gap
        ev_dets = {r["detector"] for r in rows}
        nb = B["n_on"].shape[1]
        for d in healthy.get(dev, []):
            if d in ev_dets or d not in list(B["dets"]):
                continue
            a = int(rng.integers(W3, nb - W3 - 24))
            null += measure(B, [dict(DeviceId=dev, detector=d, kind="null", b0=a, b1=a + 24, lam=np.nan, co=0)])
    return pd.DataFrame(out), pd.DataFrame(null)


def summarise(df, label):
    d = df[df.ok.eq(True)]
    lr = np.abs(np.log2(d.ratio))
    print(f"{label:34s} n {len(d):5d} | share after/before: median {d.ratio.median():.2f}, "
          f"within 1.5x {100 * (lr < np.log2(1.5)).mean():.0f} %, off by > 2x {100 * (lr > 1).mean():.0f} %, "
          f"fell below 1/2 {100 * (d.ratio < .5).mean():.0f} %")


def main():
    nf = pd.read_parquet(H.HB / "nofault_eval.parquet")
    f = nf[nf.window == "full"]
    healthy = f[f.presumed_healthy.eq(True) & (f.n_on >= 200)].groupby("DeviceId").detector.apply(list).to_dict()
    ev, null = period("stg", H.DCW / "official" / "stg" / "cache" / "events", healthy)
    ev["period"] = "stg"
    for k in ("stuck", "silent"):
        summarise(ev[ev.kind == k], f"Sept 2026 {k}, recovered")
        summarise(ev[(ev.kind == k) & (ev.co == 0)], f"  ... alone (no co-{k} detector)")
    summarise(null, "healthy, random split (null)")
    # several events on one detector in 66 h?
    g = ev.groupby(["DeviceId", "detector"]).size()
    print("detectors with a recovered event:", len(g), "| with 2+ events in 66 h:", int((g >= 2).sum()))
    # Dec 2024 -> Sept 2026
    evd, _ = period("dec", H.DCW / "cache" / "events", {})
    evd["period"] = "dec"
    rd = pd.read_parquet(H.HB / "real_stats.parquet")
    dec = rd[(rd.period == "dec") & (rd.window == "full") & (rd.n_on >= 50)][["DeviceId", "detector"]]
    sep = f[["DeviceId", "detector", "status", "n_on", "wl_dead_print"]]
    dec = dec.merge(sep, on=["DeviceId", "detector"], how="inner")
    hit = set(zip(evd.DeviceId, evd.detector))
    dec["dec_event"] = [(a, b) in hit for a, b in zip(dec.DeviceId, dec.detector)]
    dec["sept_dead"] = dec.n_on.eq(0)
    dec["sept_bad"] = dec.status.eq("bad")
    print(dec.groupby("dec_event")[["sept_dead", "sept_bad"]].mean().mul(100).round(1).assign(
        n=dec.groupby("dec_event").size()))
    pd.concat([ev, evd, null.assign(period="stg")], ignore_index=True).to_parquet(
        OUT / ("recovery_events.parquet" if USE_PHASE else "recovery_events_signal.parquet"))


if __name__ == "__main__":
    main()
