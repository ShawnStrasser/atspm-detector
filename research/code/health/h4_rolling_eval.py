"""Note 46 (f): does rolling function consistency catch real detector problems at a low false-alarm rate?

Per detector, from h4_rolling.py (2-h windows, out of fold): windows with >= 20 actuations only (night / thin
windows skipped); reference = the detector's own median class-probability vector over its windows; per window
the total-variation distance to it (tv) and whether the argmax differs from its modal class (flip); `rel` = tv
minus the median tv of the signal's other detectors in that window (a window where every detector wobbles is
the window, not the detector).  Needs >= 3 usable windows (>= 6 h).
  A  66-h and 24-h windows: catches of real problems NOT already flagged by health v4 (final), at +0.5 / +1 pt
     false alarms on presumed-healthy detectors.
  B  prospective, non-circular: Sept 18-21 detectors health calls ok -> bad on Sept 26-28 (health_w40)?
     top 5 % inconsistent vs base rate, volume-decile matched, shuffled control.

    python h4_rolling_eval.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
OUT = H.DCW / "health4"
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
PC = [f"p_{c}" for c in C7]


def det_stats(r: pd.DataFrame) -> pd.DataFrame:
    r = r[r.det_n_on >= 20].copy()
    med = r.groupby(["DeviceId", "Detector"])[PC].transform("median")
    M = med.to_numpy()
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
    S = r[PC].to_numpy()
    r["tv"] = 0.5 * np.abs(S - M).sum(1)
    mode = r.assign(a=S.argmax(1)).groupby(["DeviceId", "Detector"]).a.agg(lambda s: s.mode().iat[0])
    r["flip"] = S.argmax(1) != mode.reindex(pd.MultiIndex.from_frame(r[["DeviceId", "Detector"]])).to_numpy()
    sm = r.groupby(["DeviceId", "k"]).tv.transform("median")
    r["rel"] = r.tv - sm
    return r.groupby(["DeviceId", "Detector"]).agg(n_win=("tv", "size"), tv_max=("tv", "max"),
                                                    tv_p90=("tv", lambda x: x.quantile(.9)),
                                                    flip=("flip", "mean"), rel_max=("rel", "max")).reset_index()


def incremental(m, stat, label):
    H_ = m.presumed_healthy & ~m.flag & m[stat].notna()
    P_ = m.pos & ~m.flag
    out = [f"{label} {stat}: healthy {int(H_.sum())}, positives missed by health {int(P_.sum())} "
           f"(with stat {int((P_ & m[stat].notna()).sum())})"]
    for add in (0.5, 1.0, 2.0):
        thr = m.loc[H_, stat].quantile(1 - add / 100)
        out.append(f"+{add} pt FA (>{thr:.2f}): {int((P_ & (m[stat] > thr)).sum())} caught")
    y = np.r_[np.ones(int((m.pos & m[stat].notna()).sum())), np.zeros(int((m.presumed_healthy & m[stat].notna()).sum()))]
    v = np.r_[m.loc[m.pos & m[stat].notna(), stat], m.loc[m.presumed_healthy & m[stat].notna(), stat]]
    rk = pd.Series(v).rank().to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    out.append(f"AUC {(rk[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1):.3f}")
    print(" | ".join(out))


def main():
    r = pd.read_parquet(OUT / "rolling_stg.parquet")
    print("rolling rows", len(r), "signals", r.DeviceId.nunique(), "usable windows share",
          round(float((r.det_n_on >= 20).mean()), 3))
    s = pd.read_parquet(H.HB / "nofault_eval.parquet")
    s["pos"] = ((s.wl_dq_health & s.nf_dq.eq(True)) | (s.wl_lc_health & s.nf_health.eq(True))
                | (s.wl_card_erratic & s.nf_card.eq(True)) | s.wl_degraded).fillna(False).astype(bool)
    s["presumed_healthy"] = s.presumed_healthy.fillna(False).astype(bool)
    f = pd.read_parquet(OUT / "final_eval_final.parquet", columns=["DeviceId", "window", "detector", "status"])
    for w, (lo, hi) in {"full": (None, None), "24h_a": ("2026-09-19", "2026-09-20"),
                        "24h_b": ("2026-09-20", "2026-09-21")}.items():
        rr = r if lo is None else r[(r.t0 >= lo) & (r.t1 <= hi)]
        d = det_stats(rr)
        d = d[d.n_win >= 3].rename(columns={"Detector": "detector"})
        m = s[s.window == w].drop(columns=["status"], errors="ignore").merge(f[f.window == w], on=["DeviceId", "window", "detector"]).merge(
            d, on=["DeviceId", "detector"], how="left")
        m["flag"] = m.status.isin(["bad", "suspect"])
        for st in ("tv_max", "tv_p90", "flip", "rel_max"):
            incremental(m, st, w)
    # B: prospective Sept 18-21 -> Sept 26-28
    d = det_stats(r).rename(columns={"Detector": "detector"})
    d = d[d.n_win >= 3]
    a = f[f.window == "full"][["DeviceId", "detector", "status"]]
    w40 = pd.read_parquet(OUT / "health_w40.parquet", columns=["DeviceId", "detector", "status", "n_on"])
    x = d.merge(a, on=["DeviceId", "detector"]).merge(w40.rename(columns={"status": "st40", "n_on": "n40"}),
                                                      on=["DeviceId", "detector"])
    nf = pd.read_parquet(OUT / "final_eval_final.parquet", columns=["DeviceId", "window", "detector", "n_on"])
    x = x.merge(nf[nf.window == "full"][["DeviceId", "detector", "n_on"]], on=["DeviceId", "detector"])
    x = x[x.status == "ok"].copy()
    x["bad40"] = x.st40.eq("bad")
    x["vq"] = pd.qcut(x.n_on.rank(method="first"), 10, labels=False)
    print(f"\nprospective: {len(x)} detectors ok on Sept 18-21; bad on Sept 26-28 {x.bad40.mean():.3f}")
    rng = np.random.default_rng(0)
    for st in ("tv_max", "tv_p90", "flip", "rel_max"):
        x["r"] = x.groupby("vq")[st].rank(pct=True)
        sel = x.r >= 0.95
        exp = x.groupby("vq").bad40.mean().reindex(x[sel].vq).mean()
        sh = []
        for _ in range(300):
            p = x.groupby("vq")[st].transform(lambda z: z.sample(frac=1, random_state=int(rng.integers(1e9))).to_numpy())
            sh.append(x[p.groupby(x.vq).rank(pct=True) >= 0.95].bad40.mean())
        print(f"  {st} top 5 % within volume decile: n {int(sel.sum())} bad later {x[sel].bad40.mean():.3f} "
              f"(volume-only {exp:.3f}; shuffled mean {np.mean(sh):.3f}, p97.5 {np.quantile(sh, .975):.3f})")


if __name__ == "__main__":
    main()
