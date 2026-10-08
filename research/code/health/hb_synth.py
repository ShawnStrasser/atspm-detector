"""Note 38: synthetic-fault data set (self-supervised).

Base = detectors that the rules call ok / not_enough_data on the whole period AND carry no
known-problem flag (weak_labels).  Windows of 2 h / 6 h / 24 h are cut at random (2 h and 6 h
start between 06:00 and 18:00); in each, up to 3 base detectors get one synthetic fault each
(hb_data.FAULTS); every detector of the signal is then re-scored with the SAME statistics the
rules use.  Rows: label = fault kind, 'none' (clean base detector) or 'unknown' (not in base).

    python hb_synth.py [--seed 0] [--reps 2]   -> %DC_WORK%/health/synth_s{seed}.parquet
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
LENS = (2, 6, 24)
_G = {}


def _init(clean):
    _G["clean"] = clean


def one(args):
    per, dev, seed, reps = args
    B = H.load_B(per, dev)
    if B is None:
        return None
    clean = _G["clean"].get((per, dev), set())
    rng = np.random.default_rng([seed, int(dev[:8], 16), 0 if per == "stg" else 1])
    nbf = B["n_on"].shape[1]
    out = []
    for L in LENS:
        nb = L * H.BPH
        if nb > nbf:
            continue
        for rep in range(reps):
            if L < 24:
                # start hour 06-18 local
                starts = [a for a in range(0, nbf - nb) if 6 <= B["hour"][a] < 18 and a % 3 == 0]
                a = int(rng.choice(starts))
            else:
                a = int(rng.integers(0, nbf - nb + 1))
            S = H.slice_B(B, a, a + nb)
            orig = S["n_on"].copy()
            cand = [i for i, d in enumerate(S["dets"]) if int(d) in clean and orig[i].sum() > 0]
            lab = np.array(["none" if int(d) in clean else "unknown" for d in S["dets"]], dtype=object)
            lost = np.zeros(len(lab))
            fbins = np.zeros(len(lab))
            ib0, ib1 = np.full(len(lab), -1), np.full(len(lab), -1)
            if cand:
                pick = rng.choice(cand, size=min(3, len(cand)), replace=False)
                for i in pick:
                    kind = str(rng.choice(H.FAULTS))
                    m = H.inject(S, i, kind, rng, B, a)
                    lab[i] = kind
                    lost[i] = np.abs(orig[i][m] - S["n_on"][i][m]).sum() + (
                        S["occ"][i][m].sum() / 60 if kind == "stuck" else 0)
                    fbins[i] = m.mean()
                    w = np.where(m)[0]
                    ib0[i], ib1[i] = w[0], w[-1] + 1
            st = H.stats_of(S)
            st["label"] = lab
            st["changed"] = lost      # actuations removed / added (+ stuck minutes)
            st["frac_bins"] = fbins
            st["orig_on"] = orig.sum(1)
            st["inj_b0"], st["inj_b1"] = ib0, ib1
            st.insert(0, "win_h", L)
            st.insert(0, "rep", rep)
            st.insert(0, "period", per)
            st.insert(0, "DeviceId", dev)
            out.append(st)
    return pd.concat(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    R = pd.read_parquet(H.HB / "real_stats.parquet")
    W = pd.read_parquet(H.HB / "weak_labels.parquet")
    bad = set(zip(W.DeviceId[W.wl_any], W.detector[W.wl_any].astype(int)))
    full = pd.concat([H.rescore(g).assign(period=per, DeviceId=g.DeviceId.to_numpy())
                      for per, g in R[R.window.eq("full")].groupby("period")])
    ok = full[full.status.isin(["ok", "not_enough_data"])]
    clean = {}
    for (per, dev), g in ok.groupby(["period", "DeviceId"]):
        clean[(per, dev)] = {int(d) for d in g.detector if (dev, int(d)) not in bad}
    jobs = [(per, dev, a.seed, a.reps) for (per, dev) in clean]
    t = time.time()
    with Pool(6, initializer=_init, initargs=(clean,)) as p:
        res = [r for r in p.imap_unordered(one, jobs, chunksize=4) if r is not None]
    df = pd.concat(res, ignore_index=True)
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    df["fold"] = df.DeviceId.map(dict(zip(folds.DeviceId.str.lower(), folds.fold)))
    df.to_parquet(H.HB / f"synth_s{a.seed}.parquet")
    print(len(df), df.label.value_counts().to_dict(), f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
