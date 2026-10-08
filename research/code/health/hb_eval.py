"""Note 38: rules vs model vs rules+model on synthetic faults (held-out folds) and real weak labels.

    python hb_eval.py [--tag s0]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402
import hb_calib as C  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def synth_eval(tag, synth):
    S = pd.read_parquet(H.HB / synth)
    S = S[S.label.ne("unknown") & S.fold.notna()].reset_index(drop=True)
    M = pd.read_parquet(H.HB / f"model_oof_{tag}.parquet")
    S["p_model"], S["thr"] = M.p_model.to_numpy(), M.thr.to_numpy()   # same row order
    assert (M.detector.to_numpy() == S.detector.to_numpy()).all()
    S["rules"] = S.status.isin(["bad", "suspect"])
    S["model"] = S.p_model > S.thr
    # model threshold matched to the rules' false-alarm rate on clean rows, per window length
    S["model_m"] = False
    for L, g in S.groupby("win_h"):
        fa = g.rules[g.label.eq("none")].mean()
        th = np.quantile(g.p_model[g.label.eq("none")], 1 - fa)
        S.loc[g.index, "model_m"] = g.p_model > th
    S["both"] = S.rules | S.model
    S["visible"] = S.changed >= 20
    clean = S[S.label.eq("none")]
    fa = clean.groupby("win_h")[["rules", "model", "model_m", "both"]].mean()
    f = S[S.label.ne("none")]
    rec = f.groupby(["label", "win_h"])[["rules", "model", "model_m", "both"]].mean().unstack("win_h")
    recv = f[f.visible].groupby("win_h")[["rules", "model", "model_m", "both"]].mean()
    reca = f.groupby("win_h")[["rules", "model", "model_m", "both"]].mean()
    # precision at the synthetic prevalence
    prec = {}
    for c in ["rules", "model", "model_m", "both"]:
        prec[c] = S.groupby("win_h").apply(lambda g: g.label[g[c]].ne("none").mean())
    # bin localisation: dropout / intermittent found by the rule's silent run
    loc = f[f.label.isin(["dropout", "intermittent"]) & f.rules & (f.drop_b1 > 0)]
    inter = (np.minimum(loc.drop_b1, loc.inj_b1) - np.maximum(loc.drop_b0, loc.inj_b0)).clip(lower=0)
    union = np.maximum(loc.drop_b1, loc.inj_b1) - np.minimum(loc.drop_b0, loc.inj_b0)
    iou = (inter / union).groupby(loc.win_h).median()
    onset = (loc.drop_b0 - loc.inj_b0).abs().groupby(loc.win_h).median() * 5
    return dict(fa=fa, rec=rec, recv=recv, reca=reca, prec=pd.DataFrame(prec), iou=iou, onset_min=onset)


def real_eval(tag):
    s = pd.read_parquet(H.HB / "rules_real.parquet")
    M = pd.read_parquet(H.HB / f"model_real_{tag}.parquet")
    M = M[M.period.eq("stg")]
    s = s.merge(M[["DeviceId", "window", "detector", "p_model", "thr"]],
                on=["DeviceId", "window", "detector"], how="left")
    R = pd.read_parquet(H.HB / "real_stats.parquet")
    deg, _ = C.degraded(R)
    s["wl_degraded"] = [(d, k) in deg for d, k in zip(s.DeviceId, s.detector)]
    s["rules"] = s.status.isin(["bad", "suspect"])
    s["model"] = s.p_model > s.thr
    s["both"] = s.rules | s.model
    s["wlen"] = s.window.str.split("_").str[0]
    groups = {"presumed healthy (false alarm)": s.presumed_healthy.eq(True),
              "dead on print (listed)": s.wl_dead_print.eq(True),
              "live Dec 2024, dead Sept 2026": s.wl_dead_since_dec.eq(True),
              "card: both outputs dead": s.wl_card_dead.eq(True),
              "card: both outputs erratic": s.wl_card_erratic.eq(True),
              "dq_core health fail": s.wl_dq_health.eq(True),
              "label-check health fail": s.wl_lc_health.eq(True),
              "share fell >5x since Dec 2024": s.wl_degraded}
    rows = []
    for g, m in groups.items():
        for w, x in s[m].groupby("wlen"):
            rows.append(dict(group=g, wlen=w, n=x.drop_duplicates(["DeviceId", "detector"]).shape[0],
                             rules=x.rules.mean(), model=x.model.mean(), both=x.both.mean(),
                             ned=x.status.eq("not_enough_data").mean()))
    return pd.DataFrame(rows), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--synth", default="synth_s0.parquet")
    a = ap.parse_args()
    r = synth_eval(a.tag, a.synth)
    print("== SYNTHETIC (held-out folds)  false-alarm rate on clean detectors by window h");
    print(r["fa"].round(4))
    print("== recall, all injected"); print(r["reca"].round(3))
    print("== recall, visible (>= 20 actuations changed)"); print(r["recv"].round(3))
    print("== precision at synthetic prevalence"); print(r["prec"].round(3))
    print("== recall by fault type x window h"); print(r["rec"].round(2).to_string())
    print("== dropout localisation: median IoU", r["iou"].round(2).to_dict(), "onset err min",
          r["onset_min"].to_dict())
    t, s = real_eval(a.tag)
    order = ["2h", "6h", "24h", "full"]
    for c in ("rules", "model", "both", "ned"):
        print(f"== REAL weak labels: flagged share ({c})")
        print(t.pivot_table(index="group", columns="wlen", values=c)[order].round(3))
    print(t.pivot_table(index="group", columns="wlen", values="n")[["full"]])
    s.to_parquet(H.HB / f"eval_real_{a.tag}.parquet")


if __name__ == "__main__":
    main()
