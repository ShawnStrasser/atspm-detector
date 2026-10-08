"""Note 79: candidate health checks (hi-res only) judged on the evaluation set with a held-out split by signal.

Candidates (statistics from h79_run.cand_stats; higher score = worse):
  a1 green-miss asymmetry: share of the partner's busy greens (>= 2 ONs) in which the detector was never active, minus
     the reverse (partner = same predicted phase, best 15-min count correlation; >= 8 busy greens)
  a2 count ratio vs that partner, -log(ratio) (partner >= 50 ONs)
  a3 count ratio vs the median same-predicted-function peers on the phase, -log(ratio)
  c1 = a2 with limits per (own predicted function, partner's predicted function) - class-specific expectations
  b1 within-window drift: Poisson z of the second-half count given the first-half share of the phase reference, -z
Each limit = quantile q of presumed-healthy rows of the CALIBRATION half of signals (per window length; c1 also per
function pair, >= 30 healthy rows else the pooled limit); judged on the other half, both halves, 3 split seeds.
Comparators: random flags at the same added false-alarm rate (rand_*; the status cannot be loosened - health_score is
exactly 1.0 for every ok row), and the candidate statistic shuffled among the detectors of each signal-window.

    python h79_cand.py [--period w40] -> prints; %DC_WORK%/health79/cand.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h79_score as S  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
POS = ["stuck", "chatter", "intermittent", "degraded"]


def stats(M):
    x = M
    a1 = (x.c_miss - x.c_miss_rev.fillna(0)).where(x.c_g >= 8)
    a2 = -np.log(x.c_ratio.clip(lower=1e-3)).where(x.c_partner.notna())
    a3 = -np.log(x.c_peer_ratio.clip(lower=1e-3))
    b1 = (-x.c_drift_z).where(x.c_drift_n.fillna(0) >= 0)
    return {"a1": a1, "a2": a2, "a3": a3, "c1": a2, "b1": b1}


def partner_fn(M):
    k = M[["DeviceId", "period", "window", "detector", "pred_function"]].drop_duplicates(
        ["DeviceId", "period", "window", "detector"]).rename(columns={"detector": "c_partner", "pred_function": "pfn"})
    out = M.merge(k, on=["DeviceId", "period", "window", "c_partner"], how="left")
    assert len(out) == len(M)
    return out


def limits(st, cal_h, grp=None, q=.995):
    if grp is None:
        return st[cal_h].quantile(q)
    pooled = st[cal_h].quantile(q)
    g = st[cal_h].groupby(grp[cal_h])
    lim = g.quantile(q).where(g.count() >= 30)
    return grp.map(lim).fillna(pooled)


def run(M, period, seeds=(0, 1, 2), qs=(.995, .99)):
    X = M[(M.period == period) & (M["mode"] == "prod") & M.seen].copy()
    X = partner_fn(X)
    X["pos"] = X.cls.isin(POS)
    X["hl"] = X.cls.eq("healthy") & X.tier.eq("N")
    X["fgrp"] = X.pred_function.fillna("?") + ">" + X.pfn.fillna("?")
    ST = stats(X)
    rng = np.random.default_rng(0)
    res = []
    for w in S.LEN:
        Wm = X.wlen.eq(w).to_numpy()
        for name, st0 in ST.items():
            for shuf in (False, True):
                st = st0.copy()
                if shuf:
                    v = st.to_numpy().copy()
                    for _, ix in X[Wm].groupby(["DeviceId", "window"]).indices.items():
                        idx = np.where(Wm)[0][ix]
                        v[idx] = v[idx][rng.permutation(len(idx))]
                    st = pd.Series(v, index=st.index)
                for q in qs:
                    for seed in seeds:
                        hv = X.DeviceId.map(lambda d: S.half(d, seed)).to_numpy()
                        for cal in (0, 1):
                            C = Wm & (hv == cal)
                            T = Wm & (hv != cal)
                            if name == "c1":
                                lim = limits(st, C & X.hl.to_numpy(), X.fgrp, q)
                            else:
                                lim = limits(st, C & X.hl.to_numpy(), None, q)
                            new = (st >= lim).fillna(False).to_numpy() if np.ndim(lim) else (st >= lim).fillna(False).to_numpy()
                            comb = X.flag.to_numpy() | new
                            r = dict(wlen=w, cand=name, shuf=shuf, q=q, seed=seed, cal=cal,
                                     fa_base=100 * X.flag.to_numpy()[T & X.hl.to_numpy()].mean(),
                                     fa_comb=100 * comb[T & X.hl.to_numpy()].mean(),
                                     n_pos=int((T & X.pos.to_numpy()).sum()))
                            r["fa_new_n"] = int((comb & ~X.flag.to_numpy())[T & X.hl.to_numpy()].sum())
                            for c in POS + ["all", "degA", "degB", "degCU"]:
                                if c == "all":
                                    pm = T & X.pos.to_numpy()
                                elif c in ("degA", "degB", "degCU"):
                                    pm = T & X.cls.eq("degraded").to_numpy() & X.tier.isin(list(c[3:])).to_numpy()
                                else:
                                    pm = T & X.cls.eq(c).to_numpy()
                                # random flags at the same added FA would catch this many of the unflagged positives
                                r[f"rand_{c}"] = (r["fa_comb"] - r["fa_base"]) / 100 * int((~X.flag.to_numpy())[pm].sum())
                                r[f"base_{c}"] = int(X.flag.to_numpy()[pm].sum())
                                r[f"comb_{c}"] = int(comb[pm].sum())
                                r[f"n_{c}"] = int(pm.sum())
                            res.append(r)
    return pd.DataFrame(res)


def summarise(D):
    g = D.groupby(["wlen", "cand", "shuf", "q", "seed"]).sum(numeric_only=True)   # both halves = every signal once
    out = pd.DataFrame({
        "dFA_pt": (g.fa_comb - g.fa_base) / 2,
        "new_FA_n": g.fa_new_n, "new_all": g.comb_all - g.base_all, "rand_all": g.rand_all,
        **{f"new_{c}": g[f"comb_{c}"] - g[f"base_{c}"] for c in POS + ["degA", "degB", "degCU"]},
        "n_all": g.n_all, "base_all": g.base_all, "n_degA": g.n_degA, "base_degA": g.base_degA})
    return out.groupby(["wlen", "cand", "shuf", "q"]).mean().round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="w40")
    a = ap.parse_args()
    M = pd.read_parquet(S.OUT / "scored.parquet")
    D = run(M, a.period)
    D.to_csv(S.OUT / f"cand_{a.period}.csv", index=False)
    t = summarise(D)
    print(f"period {a.period}: mean over 3 split seeds; counts are detector-windows on the test halves (every signal once)")
    print(t.to_string())


if __name__ == "__main__":
    main()
