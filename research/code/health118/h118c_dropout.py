"""Note 118c item 3: 'goes silent' with a clean yardstick.

The package check expects, inside a silent run, the detector's share (outside the run) x its reference's counts
inside the run; reference = its other live phase mates (>= 2, twins excluded) else the rest of the signal.  A mate
that is itself flagged (busy at night, too-fast, too many, chattering, erratic, stuck ...) inflates or distorts that
expectation (12032 d40: 'silent 01:00-02:05, ~101 expected' because d6 / d39 are busy at night).  Here the
reference drops every mate with a v4 finding other than the silence family (goes silent / count drops / misses at
night - those only LOWER the expectation); < 2 clean phase mates -> the clean rest of the signal.

Recomputed exactly as the package (health_core.events_to_bins + det_stats silent runs) for every detector-window
with a candidate (drop_lam >= 30); validation = the same code with nothing dropped reproduces the stored drop_lam.

    python h118c_dropout.py   -> %DC_WORK%/s118c/dropout118c.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "health"))
import health_core as hc  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
OUT = DCW / "s118c"
EVD = DCW / "health4" / "w40_events"
KEY = ["DeviceId", "window", "detector"]
S_WIN = {"m30_a": ("2026-09-27 08:00", .5), "m30_b": ("2026-09-27 12:00", .5), "m30_c": ("2026-09-27 17:00", .5),
         "m30_d": ("2026-09-28 07:30", .5), "h3_a": ("2026-09-27 12:00", 3), "h3_b": ("2026-09-28 06:00", 3),
         "h24_a": ("2026-09-27 00:00", 24), "h24_b": ("2026-09-28 00:00", 24)}
SILENT_FAM = {"dropout", "level", "night_drop"}


def events(dev):
    p = EVD / f"DeviceId={dev}"
    if not p.is_dir():
        p = next(q for q in EVD.iterdir() if q.name.lower() == f"deviceid={dev}")
    e = ds.dataset(p).to_table(filter=ds.field("EventId").isin(list(hc.ALLOWED))).to_pandas()
    e = e[["Timestamp", "EventId", "Parameter"]].drop_duplicates()
    return e[~(e.EventId.isin((81, 82)) & (e.Parameter > hc.MAXCH))]


def best_run(n, occ, cov, i, S):
    """the package's silent-run search for detector row i against reference counts S (5-min bins)."""
    x = n[i]
    n_on = x[cov].sum()
    quiet = (x == 0) & (occ[i] < 1) & cov
    s0, s1 = hc._runs(quiet)
    best = (0.0, -1, -1)
    if len(s0) and n_on > 0:
        cS = np.r_[0, np.cumsum(np.where(cov, S, 0))]
        for a, b in zip(s0, s1):
            s_in = cS[b] - cS[a]
            s_out = cS[-1] - s_in
            lam = n_on / max(s_out, 1) * s_in
            if lam > best[0]:
                best = (lam, a, b)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(DCW / "s118b" / "resolved_v4.parquet",
                        columns=KEY + ["drop_lam", "drop_b0", "drop_b1", "n_on", "phase", "left8", "wg"])
    R["fl"] = R.left8.fillna("").str.split(",").apply(lambda l: bool(set(x for x in l if x) - SILENT_FAM))
    C = R[(R.drop_lam >= 30) & (R.n_on > 0)]
    out = []
    for (dev, w), g in C.groupby(["DeviceId", "window"]):
        ev = events(dev)
        a, h = S_WIN[w]
        t0 = pd.Timestamp(a)
        B = hc.events_to_bins(ev, t0, t0 + pd.Timedelta(hours=h), None)
        n, occ, cov = B["n_on"].astype(float), B["occ"].astype(float), B["cov"]
        dets = np.array([int(d) for d in B["dets"]])
        live = n[:, cov].sum(1) > 0
        Xw = R[(R.DeviceId == dev) & (R.window == w)].set_index("detector")
        phase = {int(d): Xw.phase.get(int(d), np.nan) for d in dets}
        ref, kind, _ = hc._refs(dets, live, phase, hc.twins(n, cov))
        flagged = np.array([bool(Xw.fl.get(int(d), False)) for d in dets])
        tot = n.sum(0)
        for r in g.itertuples():
            if int(r.detector) not in set(dets):
                continue
            i = int(np.flatnonzero(dets == int(r.detector))[0])
            # package reference (validation)
            if kind[i] == "phase":
                S0 = n[ref[i]].sum(0)
                cand = np.array(ref[i])
            else:
                S0 = tot - n[i]
                cand = np.flatnonzero(np.arange(len(dets)) != i)
            lam0 = best_run(n, occ, cov, i, S0)
            # clean reference: drop flagged mates; < 2 clean phase mates -> the clean rest of the signal
            dropped = [int(dets[j]) for j in cand if flagged[j]]
            if kind[i] == "phase":
                keep = [j for j in ref[i] if not flagged[j]]
                if len(keep) >= 2:
                    S1, k1 = n[keep].sum(0), "phase"
                else:
                    oth = [j for j in range(len(dets)) if j != i and not flagged[j]]
                    S1, k1 = n[oth].sum(0), "signal"
                    dropped = [int(dets[j]) for j in range(len(dets)) if j != i and flagged[j]]
            else:
                oth = [j for j in range(len(dets)) if j != i and not flagged[j]]
                S1, k1 = n[oth].sum(0), "signal"
            lam1 = best_run(n, occ, cov, i, S1)
            out.append(dict(DeviceId=dev, window=w, detector=int(r.detector), drop_lam_pkg=r.drop_lam,
                            drop_lam_chk=lam0[0], b0_chk=lam0[1], b1_chk=lam0[2], kind0=kind[i],
                            drop_lam_c=lam1[0], drop_b0_c=lam1[1], drop_b1_c=lam1[2], kind_c=k1,
                            dropped="+".join(map(str, dropped)), n_dropped=len(dropped)))
    D = pd.DataFrame(out)
    D.to_parquet(OUT / "dropout118c.parquet")
    ok = np.isclose(D.drop_lam_chk, D.drop_lam_pkg, rtol=1e-6)
    print("validation: recomputed = stored drop_lam:", f"{ok.mean():.4f}", f"({ok.sum()} / {len(D)})")
    print("with a flagged mate dropped:", int((D.n_dropped > 0).sum()), "; expected falls below 30 (no finding):",
          int(((D.drop_lam_c < 30) & (D.drop_lam_pkg >= 30)).sum()), "; rises:", int((D.drop_lam_c > D.drop_lam_pkg + 1e-6).sum()))


if __name__ == "__main__":
    main()
